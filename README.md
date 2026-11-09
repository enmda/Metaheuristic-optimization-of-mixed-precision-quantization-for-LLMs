# Metaheuristic Optimization of Mixed-Precision Quantization for LLMs

UFAZ research project (Elvin Mahmudzada). Search for per-layer weight bit-widths
for Qwen2.5-1.5B using uniform baselines, random/greedy search, and
metaheuristics (Genetic Algorithm, Simulated Annealing). Proposal: `docs/proposal.pdf`.

## Setup

```bash
conda activate research-ufaz-quantization-metaheuristics
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
python scripts/check_env.py
```

Hardware: GTX 1650 Mobile (4 GB VRAM, fp16), 15 GB RAM.

## Layout

```
configs/     experiment configs (YAML)
docs/        proposal and write-ups
notebooks/   exploration
results/     experiment outputs (git-ignored)
scripts/     entry points
src/quantmh/ library code (quantization, evaluation, search algorithms)
```

## Plan

1. Model setup + full-precision baseline (perplexity, memory, speed)
2. Uniform quantization baselines
3. Random and greedy search
4. GA and SA; comparative evaluation
