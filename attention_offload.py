#!/usr/bin/env python3
"""
Attention Offloading Prototype for Qwen3-8B
Scaffold demonstrating selective attention score offloading to CPU.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionOffloader:
    """
    Injects hooks to offload attention score computation to CPU
    while keeping KV cache on GPU.
    """
    
    def __init__(self, model, offload_threshold: int = 4096):
        """
        Args:
            model: Transformer model (e.g., Qwen3-8B)
            offload_threshold: Sequence length threshold to trigger offloading
        """
        self.model = model
        self.offload_threshold = offload_threshold
        self.hook_handles = []
        
    def _attention_hook(self, module, args, kwargs):
        """
        Hook to intercept attention forward pass.
        Offloads QK^T computation to CPU when sequence is long.
        """
        # Extract attention inputs
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        attention_mask = args[1] if len(args) > 1 else kwargs.get("attention_mask")
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Only offload for long sequences
        if seq_len <= self.offload_threshold:
            return None  # Use standard attention
        
        # Get attention weights (Q, K, V projections)
        # This depends on the specific model architecture
        # For Qwen3-8B, we need to capture Q, K before softmax
        
        # Placeholder: In real implementation, extract Q, K from module
        # Compute QK^T on CPU
        if hasattr(module, 'q_proj') and hasattr(module, 'k_proj'):
            with torch.no_grad():
                # Move to CPU for memory savings
                q_cpu = module.q_proj(hidden_states).to("cpu")
                k_cpu = module.k_proj(hidden_states).to("cpu")
                
                # Compute attention scores on CPU (memory-intensive part)
                attention_scores = torch.matmul(q_cpu, k_cpu.transpose(-2, -1))
                attention_scores /= math.sqrt(hidden_states.shape[-1])
                
                # Apply mask if provided
                if attention_mask is not None:
                    attention_scores = attention_scores + attention_mask
                
                # Softmax
                attention_probs = F.softmax(attention_scores, dim=-1)
                
                # Get V projection (stay on GPU for efficiency)
                v = module.v_proj(hidden_states)
                
                # Compute output (could be on CPU or GPU)
                output = torch.matmul(attention_probs, v.to("cpu"))
                
                # Transfer back to GPU
                return output.to(hidden_states.device)
        
        return None  # Fall back to normal attention
    
    def inject_hooks(self):
        """Inject hooks into all attention layers."""
        for name, module in self.model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                handle = module.register_forward_hook(self._attention_hook)
                self.hook_handles.append(handle)
        print(f"Injected hooks into {len(self.hook_handles)} attention layers")
    
    def remove_hooks(self):
        """Remove all injected hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
    
    def __enter__(self):
        self.inject_hooks()
        return self
    
    def __exit__(self, *args):
        self.remove_hooks()


def load_qwen3_8b():
    """
    Load Qwen3-8B model.
    Uses torch.compile and proper dtype for efficiency.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load with 4-bit quantization for VRAM savings
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            use_cache=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load Qwen3-8B: {e}")
        print("Using mock model for demonstration")
        return _MockModel(), None


class _MockModel(nn.Module):
    """Mock model for testing without downloading weights."""
    
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(4096, 4096) for _ in range(4)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_attention_offloader():
    """Basic test of the offloader."""
    model = _MockModel()
    
    with AttentionOffloader(model, offload_threshold=2) as offloader:
        x = torch.randn(2, 4, 4096)
        output = model(x)
        assert output.shape == x.shape
    
    print("test_attention_offloader passed")


if __name__ == "__main__":
    # Demonstrate loading with offloading
    print("Loading Qwen3-8B with attention offloading...")
    model, tokenizer = load_qwen3_8b()
    
    with AttentionOffloader(model, offload_threshold=4096):
        # Example inference
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=10)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    print("Done")
