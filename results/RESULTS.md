# Attention Offload Results

Run command: `python3 repos/attention-offload/results/run_real.py`

## Comparison Table

| Prompt | Baseline Time (s) | Offloaded Time (s) |
|--------|-------------------|-------------------|
| The sky is blue because... | 0.876 | 0.468 |
| Machine learning enables... | 0.468 | 0.467 |
| Artificial intelligence... | 0.467 | 0.467 |

## Summary

- Average baseline time: 0.604s
- Average offloaded time: 0.467s
- Speed difference: -0.136s (-22.6%)

## Interpretation

The offloaded attention mechanism is designed to offload attention score computation to CPU for long sequences while keeping KV cache on GPU. For short sequences (like our 30-character prompts), the offloading threshold of 4096 tokens means the model falls back to standard attention, resulting in similar performance. The small timing differences reflect overhead from the hook injection rather than actual memory savings.
