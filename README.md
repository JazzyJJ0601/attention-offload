# Attention Offloading

Selective attention score computation offloading from GPU to CPU for running large language models on consumer hardware. Targets the memory bottleneck of attention score matrices during long-context inference.

## Problem

Standard attention computes a score matrix of shape `[seq_len, seq_len]`. For a 32K context window this matrix alone needs ~4GB in FP32, exceeding consumer GPU capacity even when model weights fit. Full-weight offloading is too slow; this approach offloads only the score computation.

## Approach

- **Keeps on GPU**: KV cache, model weights, value tensors
- **Offloads to CPU**: Attention score computation (QK^T), softmax normalization, weighted sum
- **Threshold-gated**: Only triggers offloading when sequence length exceeds a configurable threshold

## Usage

```python
from attention_offload import AttentionOffloader
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
offloader = AttentionOffloader(model, offload_threshold=4096)
offloader.register_hooks()

# Inference proceeds normally; hooks handle offload
outputs = model(input_ids)
```

## Benchmark

See `benchmark/results.md` for perplexity and memory comparisons vs full GPU baseline.

## Project Structure

- `attention_offload.py` — Core `AttentionOffloader` class with hook registration
- `methodology.md` — Detailed explanation of memory bottleneck and offloading strategy
- `benchmark/` — Perplexity and memory benchmark scripts
- `research/` — Related work survey
- `tests/` — Unit tests

## Status

Prototype. Benchmarks run on Qwen3-8B with comparison of GPU memory usage and inference speed at various sequence lengths.