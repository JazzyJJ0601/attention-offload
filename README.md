# Attention Offload

Long-context decoding for Qwen3-8B on one 24 GB GPU, when the KV cache no longer fits next to the weights.
Attention stays **exact**: no tokens are dropped or compressed.

**Result (RTX 3090 Ti, Qwen3-8B bf16, 32 greedy tokens):** at a 49,152-token prompt the all-GPU baseline runs
out of memory. Split attention decodes at **215 ms/token**, against 294 ms/token for CPU attention and
2,201 ms/token for the usual "copy the cache to the GPU each step" offload. It produces the same 32 tokens as the
reference. Below 16k tokens nothing is offloaded, so it runs at full GPU speed.

## The idea

Qwen3-8B in bf16 takes about 16.4 GB, and its KV cache costs 144 KiB per token, so a 48k prompt needs about
6.75 GB more than a 24 GB card has left. The usual offloaded cache copies every layer's keys and values back to
the GPU on every decoding step, so you pay the PCIe cost of the whole cache per token.

Split attention instead keeps the first G tokens of every layer on the GPU and the rest in CPU memory. On each
step:

1. the GPU attends to its part (flash attention), while
2. the CPU attends to its part at the same time (only the query goes over; only a small output comes back);
3. the two partial results are merged exactly with their log-sum-exp.

The merge is exact maths, not an approximation, so the output is the same attention up to floating-point
rounding.

## Results

ms per generated token (lower is better). Same prompt (WikiText-2), same prefill, 32 greedy tokens.

| Prompt | GPU only (baseline) | Fetch to GPU | CPU attention | **Split (ours)** |
|---:|---:|---:|---:|---:|
| 8,192 | 25.9 | 371.3 | 74.7 | **25.9** (nothing offloaded) |
| 16,384 | 27.4 | 739.1 | 123.5 | **32.0** |
| 32,768 | 30.4 (21.8 GB peak) | 1,469.1 | 293.7 | **129.7** |
| 49,152 | **out of memory** | 2,200.5 | 294.2 | **215.3** |

- **Above the GPU's limit**, split is 1.4× faster than CPU attention and 10× faster than fetching the cache.
- **At 32k**, where the whole cache still just fits (21.8 GB peak), the GPU is faster (30.4 ms). Split is the
  option for when it doesn't fit, or when you want to keep about 2 GB free (19.5 GB peak).
- **Correctness:** every offloaded mode generated the same 32 tokens as its reference (the GPU run, or fetch at
  48k where the GPU can't run). The largest logit difference was 0.58 on a logit scale of about 28, from bf16
  vs fp32 arithmetic.

Full numbers, including peak memory and where each part of the cache lived, are in [RESULTS.md](RESULTS.md) and
`results/real.json`.

## Limits

- One GPU, one prompt at a time, greedy decoding. Prefill is done layer by layer and isn't optimised.
- The CPU part scales with the offloaded length; CPU attention (24 threads) timed almost the same at 32k and
  48k in this run, which I haven't explained yet.
- G (tokens kept on the GPU) is fixed at 16,384 here, not tuned to the free memory.

## Run it

```bash
python results/run_real.py          # needs Qwen3-8B locally (path at the top of the script), ~20 GB host RAM
python -m pytest -q tests           # merge maths and cache bookkeeping
```
