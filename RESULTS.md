# Results: split attention vs offload baselines on Qwen3-8B

- **Hardware:** RTX 3090 Ti (23.55 GB), 24 CPU threads, host memory capped at 20 GB.
- **Model:** Qwen3-8B bf16, run layer by layer with its own weights. Checked against the Hugging Face forward
  pass first: largest prefill logit difference 0.0625, largest decode-step difference 0.125, on a logit scale of 28.
- **Workload:** a WikiText-2 test prompt cut to each length, then 32 greedy tokens. KV cache is 144 KiB/token.
- **Split:** G = 16,384 tokens per layer on the GPU, the rest on the CPU.

| Prompt | Mode | ms/token | Peak GPU (GB) | KV on GPU (GB) | KV on CPU (GB) | Same tokens | Max logit diff |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8,192 | gpu | 25.9 | 16.90 | 1.13 | 0 | ref | |
| 8,192 | fetch | 371.3 | 15.78 | 0 | 1.13 | 32/32 | 0.00 |
| 8,192 | cpu | 74.7 | 15.77 | 0 | 2.26 | 32/32 | 0.31 |
| 8,192 | **split** | **25.9** | 16.90 | 1.13 | 0 | 32/32 | 0.00 |
| 16,384 | gpu | 27.4 | 18.53 | 2.25 | 0 | ref | |
| 16,384 | fetch | 739.1 | 16.28 | 0 | 2.25 | 32/32 | 0.00 |
| 16,384 | cpu | 123.5 | 16.27 | 0 | 4.51 | 32/32 | 0.25 |
| 16,384 | **split** | **32.0** | 18.52 | 2.25 | 0.01 | 32/32 | 0.25 |
| 32,768 | gpu | 30.4 | 21.78 | 4.50 | 0 | ref | |
| 32,768 | fetch | 1,469.1 | 17.28 | 0 | 4.50 | 32/32 | 0.00 |
| 32,768 | cpu | 293.7 | 17.28 | 0 | 9.01 | 32/32 | 0.31 |
| 32,768 | **split** | **129.7** | 19.53 | 2.25 | 4.51 | 32/32 | 0.56 |
| 49,152 | gpu | out of memory | | | | | |
| 49,152 | fetch | 2,200.5 | 18.29 | 0 | 6.75 | ref | |
| 49,152 | cpu | 294.2 | 18.28 | 0 | 13.51 | 32/32 | 0.56 |
| 49,152 | **split** | **215.3** | 20.53 | 2.25 | 9.01 | 32/32 | 0.58 |

The cpu mode keeps its cache in fp32, so it uses twice the host memory.

## Reading it

- **Above what fits on the GPU (48k):** split 215.3 ms/token vs CPU attention 294.2 (1.37× faster) and fetch
  2,200.5 (10.2× faster).
- **At 32k:** split 129.7 vs CPU 293.7 (2.3×) and fetch 1,469.1 (11.3×). The all-GPU run still fits here and is
  faster (30.4 ms), but uses 2.25 GB more GPU memory than split.
- **At or below G:** split is the GPU path (25.9 at 8k). The 16k row adds a few tokens on the CPU and costs
  4.6 ms/token more.
- **Not explained yet:** CPU attention took almost the same time at 32k and 48k.

## History

An earlier version of this repo hooked the attention layers but was only tested on 10–30 token prompts, below
its 4,096-token threshold, so it never offloaded anything. It was replaced by this layer-wise runner and
re-measured above the threshold. The first 32k/48k run was killed by the host-memory cap (pinned copies of the
cache). That was fixed and re-run; the 8k/16k rows come from the first run.

Reproduce: `python results/run_real.py` (lengths via `AOFF_LENGTHS=8192,16384,32768,49152`).
