import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results"))
import run_real  # noqa: E402
from run_real import attend, merge  # noqa: E402


def test_merge_of_partials_is_exact():
    torch.manual_seed(0)
    q, k, v = torch.randn(8, 4, 64), torch.randn(8, 300, 64), torch.randn(8, 300, 64)
    full, _ = attend(q, k, v, 0.125)
    o1, l1 = attend(q, k[:, :170], v[:, :170], 0.125)
    o2, l2 = attend(q, k[:, 170:], v[:, 170:], 0.125)
    assert torch.allclose(merge(o1, l1, o2, l2), full, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("mode", ["gpu", "fetch", "cpu", "split"])
def test_every_mode_matches_hf_on_a_tiny_qwen3(mode):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=3,
                      num_attention_heads=8, num_key_value_heads=2, head_dim=64, max_position_embeddings=1024)
    model = Qwen3ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()
    r = run_real.Runner(model)
    ids = torch.randint(0, 512, (1, 200), device="cuda")
    run_real.NEW_TOKENS = 60
    cache = run_real.Cache(r, 210, mode, 120)              # split: 120 tokens on the GPU, the rest on the CPU
    with torch.no_grad():
        hf = model(ids).logits[0].float()
        r.prefill(ids[:, :150], cache.store)
        for t in range(150, 200):                          # teacher-forced decode steps
            out = r.step(ids[0, t], t, cache.attn)
            err = (out - hf[t]).abs().max().item()
            assert err < 0.05 * hf[t].abs().max().item(), (mode, t, err)
