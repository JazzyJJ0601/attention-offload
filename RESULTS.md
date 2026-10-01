# Attention Offload Results

**Status:** Measured on Qwen3-8B, but the test prompts were below the 4096-token offload threshold, so offloading never triggered. This run only shows the hooks add no measurable overhead; a long-context run is still to do.

Run command: `python3 repos/attention-offload/results/run_real.py`

## Comparison Table

| Prompt | Baseline Time (s) | Offloaded Time (s) |
|--------|------------------|-------------------|
| The sky is blue because... | 0.4793 | 0.4795 |
| Machine learning enables... | 0.4776 | 0.4697 |
| Artificial intelligence... | 0.4673 | 0.4686 |

## Summary

- Average baseline time: 0.4747s
- Average offloaded time: 0.4726s
- Speed difference: -0.0021s (-0.4%)
- Outputs match across all prompts: Yes

## Interpretation

The offloaded attention mechanism injects hooks into each attention layer that track sequence length
but fall through to standard PyTorch attention for sequences under the threshold (4096 tokens).
With our short prompts (10–30 tokens), both passes execute identical compute, so any timing
variation reflects hook overhead and CUDA kernel launch variability rather than actual offloading.
The small measured difference shows that hook injection adds negligible overhead — the offloader
is ready to offload attention scores to CPU for longer sequences where the quadratic memory
bottleneck kicks in.
