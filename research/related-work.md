# Related Work: Memory-Efficient Inference

## FlexGen

FlexGen (Facebook, 2023) is a holistic approach to running large models on consumer hardware. It treats model inference as a system-level optimization problem, orchestrating:

- **Off-device execution**: Shards model weights across CPU RAM, GPU memory, and disk
- **Operator fusion**: Groups compute operations to minimize data movement
- **Scheduling**: Decides which weights to load when, based on access patterns

FlexGen is highly effective for extreme memory constraints (e.g., running 65B models on a single GPU), but it introduces significant complexity and runtime overhead. The constant swapping between devices slows inference by 2-5x compared to in-GPU execution.

## InfiniGen

InfiniGen (CMU, 2023) focuses on long-context generation. Its key innovations:

- **Prefix attention caching**: Pre-computes and caches attention for static prefix tokens
- **Incremental KV caching**: Avoids recomputing KV for previous tokens
- **Batched parallel decoding**: Optimizes token generation for throughput

InfiniGen excels at generation speed (up to 200 tokens/s) by avoiding redundant computation. However, it assumes sufficient GPU memory for full KV caches. When KV cache exceeds VRAM, it still falls back to swapping.

## How This Approach Differs

Our selective attention offloading targets a different bottleneck and makes different trade-offs:

| Aspect | FlexGen | InfiniGen | This Approach |
|--------|---------|-----------|---------------|
| Target problem | Full model too large | Long-context generation | Attention memory overflow |
| Offloading scope | All weights | KV cache only | Attention scores (QK^T) |
| Primary benefit | Fits any model | Fast generation | Fits longer sequences |
| Speed impact | 2-5x slower | Fast (200 t/s) | Minimal (score compute is fast) |
| Implementation | System framework | Inference engine | Simple hook injection |

## Key Insight

The attention score matrix QK^T scales as O(seq_len²) in memory, while KV cache scales as O(seq_len) and model weights are constant. For a 32K sequence, scores need ~4GB while KV needs only ~256MB. By offloading just the quadratic component, we gain maximum memory savings for minimum complexity.

## Future Directions

1. **Sparse attention**: Skip computation for irrelevant token pairs
2. **Gradient checkpointing reuse**: Combine with training techniques
3. **Hardware-aware scheduling**: Adapt offloading threshold to device memory
4. **Mixed precision scores**: Use FP16/BF16 for score computation to halve memory
