"""Stage 1: full-precision (fp16) reference measurements.

Usage: python scripts/baseline_fp16.py [--config configs/default.yaml]
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml

from quantmh.eval import (
    load_model, load_wikitext_tokens, measure_speed, perplexity, weight_bytes,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--out", default="results/baseline_fp16")
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))
torch.manual_seed(cfg["seed"])
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
shutil.copy(args.config, out / "config.yaml")  # record what produced results

print(f"Loading {cfg['model']['name']} ...")
model, tok = load_model(**cfg["model"])
torch.cuda.reset_peak_memory_stats()
w_bytes = weight_bytes(model)
print(f"weights: {w_bytes / 2**30:.2f} GiB, "
      f"VRAM after load: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

tokens = load_wikitext_tokens(tok, cfg["eval"]["subset"], cfg["eval"]["split"])
print(f"eval tokens: {tokens.numel():,}")

ppl = perplexity(model, tokens, cfg["eval"]["seq_len"], cfg["eval"]["max_samples"])
print(f"perplexity: {ppl:.3f}")

speed = measure_speed(model, tok)
print(f"prefill: {speed['prefill_tok_per_s']:.0f} tok/s, "
      f"decode: {speed['decode_tok_per_s']:.1f} tok/s")

metrics = {
    "model": cfg["model"]["name"],
    "dtype": cfg["model"]["dtype"],
    "compute_dtype": cfg["model"].get("compute_dtype"),
    "n_params": sum(p.numel() for p in model.parameters()),
    "weight_gib": w_bytes / 2**30,
    "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    "eval_tokens": tokens.numel(),
    "seq_len": cfg["eval"]["seq_len"],
    "perplexity": ppl,
    **speed,
}
json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))
