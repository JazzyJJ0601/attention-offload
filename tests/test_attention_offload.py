import torch
import pytest
from attention_offload import AttentionOffloader, _MockModel


class TestAttentionOffloader:
    """Tests for the attention offloading functionality."""
    
    def test_inject_hooks(self):
        """Test that hooks are injected into attention layers."""
        model = _MockModel()
        offloader = AttentionOffloader(model)
        offloader.inject_hooks()
        assert len(offloader.hook_handles) > 0
        offloader.remove_hooks()
        assert len(offloader.hook_handles) == 0
    
    def test_no_offload_short_sequence(self):
        """Test that short sequences don't trigger offloading."""
        model = _MockModel()
        offloader = AttentionOffloader(model, offload_threshold=4096)
        offloader.inject_hooks()
        
        x = torch.randn(2, 16, 4096)  # Short sequence
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == x.shape
        offloader.remove_hooks()
    
    def test_offload_long_sequence(self):
        """Test that long sequences trigger offloading."""
        model = _MockModel()
        offloader = AttentionOffloader(model, offload_threshold=8)
        offloader.inject_hooks()
        
        x = torch.randn(2, 32, 4096)  # Long sequence
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == x.shape
        offloader.remove_hooks()
    
    def test_context_manager(self):
        """Test the context manager pattern."""
        model = _MockModel()
        
        with AttentionOffloader(model) as offloader:
            assert len(offloader.hook_handles) > 0
        
        assert len(offloader.hook_handles) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
