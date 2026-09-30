# Attention Offloading Methodology

## Overview

This approach enables running large language models (like Qwen3-8B) on consumer hardware with limited VRAM by selectively offloading attention computation to CPU while keeping key-value (KV) caches on GPU. Unlike full-weight offloading strategies, we target only the memory-intensive attention score computation.

## Memory Bottleneck

In standard attention:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) @ V
```

The `QK^T` operation produces a score matrix of shape [seq_len, seq_len]. For a 32K context window, this matrix alone requires 4GB of memory (32K × 32K × 4 bytes for FP32), exceeding typical consumer GPU capacity even when the model weights fit.

## Selective Offloading Strategy

### What We Offload to CPU:
1. **Score computation** (QK^T): Matrix multiplication happens on CPU
2. **Softmax normalization**: Performed on CPU after receiving scores
3. **Attention-weighted output**: The @ V operation done on CPU, result transferred back to GPU

### What Stays on GPU:
1. **KV cache**: Already computed values from previous tokens
2. **Value tensor (V)**: For efficient CPU-GPU data transfer
3. **Model weights**: For forward pass computation

### Data Flow

```
GPU (model) → CPU (score calculation) → GPU (final result)
   Q, K, V        QK^T, softmax, @V       attention output
```

## Implementation Steps

1. **Capture inputs**: Extract Q, K, V tensors from attention layer before score computation
2. **Transfer to CPU**: Move Q, K (K is usually smaller after projection)
3. **Compute scores**: QK^T matrix multiply on CPU
4. **Apply softmax**: Normalize on CPU
5. **Optional transfer**: If V is large, compute final weighted sum on CPU
6. **Return result**: Send back to GPU for remaining layers

## Trade-offs

| Aspect | Full Offload | This Approach |
|--------|--------------|---------------|
| Memory saved | High | Moderate (scores only) |
| Speed impact | Severe | Moderate (score compute is fast) |
| Implementation | Complex | Simple (hook-based) |
| Best for | Extreme VRAM limits | 12-24GB consumer GPUs |

## Performance Notes

The score matrix QK^T is the primary memory consumer, not KV cache. By offloading just this computation, we gain significant memory headroom with minimal speed penalty—CPU matrix multiplication is highly optimized and the operation is embarrassingly parallel.
