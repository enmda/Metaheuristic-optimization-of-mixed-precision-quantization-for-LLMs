"""Model loading and evaluation: perplexity, memory, and inference speed."""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


_MAX_FP32_ELEMS = 2**24  # 64 MB of fp32 weight upcast at a time


def _linear_fp32_compute(self, x):
    x32 = x.float()
    w, b = self.weight, self.bias
    step = max(1, _MAX_FP32_ELEMS // w.shape[1])  # output rows per slice
    # Huge matrices (lm_head: 151936x1536 = 892 MB in fp32) are upcast in
    # row slices; each slice yields a block of output features.
    outs = [
        F.linear(x32, w[i : i + step].float(),
                 None if b is None else b[i : i + step].float()).to(x.dtype)
        for i in range(0, w.shape[0], step)
    ]
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)


def use_fp32_compute(model):
    """Keep weights stored in their dtype but run every Linear's matmul in fp32.

    On the GTX 1650 (TU117), fp16 GEMMs run ~4.5x slower than fp32
    (0.33 vs 1.5 TFLOPS measured), while fp32 weights don't fit in 4 GB.
    Upcasting one layer's weight at a time gives fp32 speed at fp16 memory.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = _linear_fp32_compute.__get__(module)
    return model


def load_model(name: str, dtype: str = "float16", device: str = "cuda",
               compute_dtype: str | None = None):
    """Load a causal LM and its tokenizer in eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=getattr(torch, dtype), device_map=device
    )
    model.eval()  # disables dropout etc.; we only do inference
    if compute_dtype == "float32":
        use_fp32_compute(model)
    return model, tokenizer


def load_wikitext_tokens(tokenizer, subset="wikitext-2-raw-v1", split="test"):
    """Return the whole split as one long 1-D tensor of token ids.

    Joining with blank lines is the convention used by GPTQ/AWQ papers,
    so our perplexity numbers are comparable with theirs.
    """
    data = load_dataset("Salesforce/wikitext", subset, split=split)
    text = "\n\n".join(data["text"])
    return tokenizer(text, return_tensors="pt").input_ids[0]


def spread_windows(n_total: int, k: int) -> list[int]:
    """`k` evenly spaced window indices out of `n_total` (fixed, deterministic)."""
    return torch.linspace(0, n_total - 1, k).round().long().tolist()


@torch.no_grad()  # no gradients needed -> saves a lot of memory
def perplexity(model, tokens, seq_len=512, max_samples=None, chunk=128,
               window_ids=None):
    """Perplexity over non-overlapping windows of `seq_len` tokens.

    `window_ids` selects a subset of windows (fast proxy evaluation);
    otherwise the first `max_samples` windows (all if None) are used.
    The vocabulary projection (lm_head) is applied in chunks of positions so
    the full [seq_len x vocab] logits tensor never exists in fp32 at once.
    """
    device = next(model.parameters()).device
    if window_ids is None:
        n_windows = tokens.numel() // seq_len
        if max_samples is not None:
            n_windows = min(n_windows, max_samples)
        window_ids = range(n_windows)

    total_nll, total_count = 0.0, 0
    for i in tqdm(window_ids, desc="ppl", leave=False):
        ids = tokens[i * seq_len : (i + 1) * seq_len].unsqueeze(0).to(device)
        hidden = model.model(input_ids=ids).last_hidden_state  # [1, T, H]

        # Position t predicts token t+1, so drop the last hidden state
        # and the first target token.
        hidden, targets = hidden[:, :-1], ids[:, 1:]
        for s in range(0, hidden.shape[1], chunk):
            logits = model.lm_head(hidden[:, s : s + chunk]).float()
            total_nll += F.cross_entropy(
                logits.flatten(0, 1), targets[:, s : s + chunk].flatten(),
                reduction="sum",
            ).item()
            total_count += targets[:, s : s + chunk].numel()

    return math.exp(total_nll / total_count)


def weight_bytes(model) -> int:
    """Bytes occupied by parameters (shared/tied tensors counted once)."""
    seen, total = set(), 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel() * p.element_size()
    return total


@torch.no_grad()
def measure_speed(model, tokenizer, prompt_len=256, new_tokens=64, repeats=3):
    """Prefill and decode throughput in tokens/second (median of repeats)."""
    device = next(model.parameters()).device
    ids = torch.randint(100, 10_000, (1, prompt_len), device=device)
    mask = torch.ones_like(ids)

    def timed(fn):
        torch.cuda.synchronize()  # GPU is async: wait before reading clock
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t

    model(input_ids=ids[:, :16])  # warm-up: first call pays one-off costs

    prefill = sorted(timed(lambda: model(input_ids=ids)) for _ in range(repeats))
    gen = lambda: model.generate(
        input_ids=ids, attention_mask=mask, max_new_tokens=new_tokens,
        min_new_tokens=new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    total = sorted(timed(gen) for _ in range(repeats))

    t_prefill, t_total = prefill[repeats // 2], total[repeats // 2]
    return {
        "prefill_tok_per_s": prompt_len / t_prefill,
        "decode_tok_per_s": new_tokens / (t_total - t_prefill),
    }
