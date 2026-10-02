"""Real Qwen3-8B run: long-context decoding when the KV cache does not fit next to the weights on the GPU.

Qwen3-8B in bf16 takes ~16.4 GB of a 24 GB card; its KV cache costs 144 KiB per token, so long prompts run the
GPU out of memory during decoding. Four ways to decode the same prompt, all EXACT attention (no tokens dropped):

  gpu    whole KV cache on the GPU (the baseline; runs out of memory on long prompts)
  fetch  KV cache in pinned CPU memory, each layer's KV copied to the GPU every step, with the next layer's copy
         overlapped on a side stream (what offloaded caches in common libraries do)
  cpu    KV cache in CPU memory (fp32); only the query goes to the CPU and only the attention output comes back
  split  ours: the first G tokens of every layer stay on the GPU, the rest live in CPU memory. Each step the GPU
         attends to its part while the CPU attends to its part at the same time; the two partial results are
         combined exactly with their log-sum-exp. Below G tokens nothing is offloaded at all.

The model is run layer by layer with its own weights (verified against the Hugging Face forward pass first).
Prefill is identical for every mode (layer-wise, flash attention, one layer's KV on the GPU at a time). For each
prompt length: decode N tokens greedily, report ms/token, peak GPU memory, and the agreement of tokens and logits
with the GPU-resident result. Writes results/real.json.
"""
import gc
import json
import os
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_PATH = "/home/jasper/eirene-projects/03-inference-lab/ai-lab/models/Qwen--Qwen3-8B"
LENGTHS = tuple(int(x) for x in os.environ.get("AOFF_LENGTHS", "8192,16384,32768,49152").split(","))
HOST_LIMIT_GB = 18.0   # the run is capped at 20 GB of host memory; skip a mode rather than be killed
NEW_TOKENS, G = 32, 16384
OUT = Path(__file__).resolve().parent / "real.json"


def attend(q, k, v, scale):
    """Exact attention of q (kvh, g, D) over k, v (kvh, T, D) in fp32. Returns (out (kvh, g, D), lse (kvh, g))."""
    s = (q.float() @ k.float().transpose(1, 2)) * scale
    lse = torch.logsumexp(s, -1)
    return torch.exp(s - lse[..., None]) @ v.float(), lse


def attend_gpu(q, k, v, scale):
    """Same on the GPU with flash attention (bf16, fp32 log-sum-exp): one KV head = one batch row."""
    o, lse = torch.ops.aten._scaled_dot_product_flash_attention(q[:, None], k[:, None], v[:, None], 0.0, False,
                                                                 False, scale=scale)[:2]
    return o[:, 0].float(), lse[:, 0]


def merge(o1, l1, o2, l2):
    """Combine two exact partial attentions over disjoint key sets."""
    m = torch.maximum(l1, l2)
    w1, w2 = torch.exp(l1 - m)[..., None], torch.exp(l2 - m)[..., None]
    return (o1 * w1 + o2 * w2) / (w1 + w2)


class Runner:
    def __init__(self, model):
        self.m = model.model
        self.head = model.lm_head
        c = model.config
        self.L, self.H, self.KVH = c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads
        self.D = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
        self.g = self.H // self.KVH
        self.scale = self.D ** -0.5

    def qkv(self, layer, x, pos):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        a = layer.self_attn
        h = layer.input_layernorm(x)
        shape = (*h.shape[:-1], -1, self.D)
        q = a.q_norm(a.q_proj(h).view(shape)).transpose(1, 2)
        k = a.k_norm(a.k_proj(h).view(shape)).transpose(1, 2)
        v = a.v_proj(h).view(shape).transpose(1, 2)
        cos, sin = self.m.rotary_emb(h, pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v                                                      # (1, heads, S, D)

    def finish(self, layer, x, att):
        x = x + layer.self_attn.o_proj(att)
        for s in range(0, x.shape[1], 4096):                                # MLP in chunks to bound memory
            c = x[:, s:s + 4096]
            x[:, s:s + 4096] = c + layer.mlp(layer.post_attention_layernorm(c))
        return x

    @torch.no_grad()
    def prefill(self, ids, store):
        """Layer-wise prefill; store(layer, k, v) receives (kvh, T, D) bf16 GPU tensors. Returns last logits."""
        x = self.m.embed_tokens(ids)
        pos = torch.arange(ids.shape[1], device=ids.device)[None]
        for i, layer in enumerate(self.m.layers):
            q, k, v = self.qkv(layer, x, pos)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale, enable_gqa=True)
            del q
            store(i, k[0], v[0])
            del k, v
            x = self.finish(layer, x, att.transpose(1, 2).reshape(1, ids.shape[1], -1))
        return self.head(self.m.norm(x[:, -1:])).float()[0, -1]

    @torch.no_grad()
    def step(self, tok, t, attn):
        """One decode step at position t; attn(layer_idx, q (kvh,g,D), k, v (kvh,1,D)) -> (kvh, g, D)."""
        x = self.m.embed_tokens(tok.view(1, 1))
        pos = torch.tensor([[t]], device=tok.device)
        for i, layer in enumerate(self.m.layers):
            q, k, v = self.qkv(layer, x, pos)
            o = attn(i, q[0].reshape(self.KVH, self.g, self.D), k[0], v[0])
            x = self.finish(layer, x, o.reshape(1, 1, -1).to(x.dtype))
        return self.head(self.m.norm(x)).float()[0, -1]


class Cache:
    """KV cache for one decode run. Positions [0, G) of every layer live on the GPU (bf16), the rest in CPU memory.
      gpu    G = everything
      fetch  prompt KV in pinned CPU memory as (L, T, kvh, D) so each layer is one contiguous copy; the next
             layer's copy runs on a side stream while this layer computes; newly decoded tokens stay on the GPU
      cpu    G = 0; CPU attention (fp32) over the whole cache
      split  G = min(G, T); GPU and CPU each attend to their part at the same time, merged by log-sum-exp
    """

    def __init__(self, r, T, mode, g_tokens):
        self.r, self.mode, self.n = r, mode, 0
        self.G = T if mode == "gpu" else (min(g_tokens, T) if mode == "split" else 0)
        self.gk = torch.empty(r.L, r.KVH, self.G, r.D, dtype=torch.bfloat16, device="cuda")
        self.gv = torch.empty_like(self.gk)
        if mode == "fetch":
            self.ck = torch.empty(r.L, T, r.KVH, r.D, dtype=torch.bfloat16, pin_memory=True)
            self.cv = torch.empty_like(self.ck, pin_memory=True)
            self.tk = torch.empty(r.L, r.KVH, NEW_TOKENS + 1, r.D, dtype=torch.bfloat16, device="cuda")
            self.tv = torch.empty_like(self.tk)
            self.side, self.buf, self.P = torch.cuda.Stream(), {}, 0
        else:
            self.ck = torch.empty(r.L, r.KVH, T - self.G, r.D, dtype=torch.float32)
            self.cv = torch.empty_like(self.ck)

    def gpu_bytes(self):
        return 2 * self.gk.numel() * 2

    def store(self, i, k, v):
        T = k.shape[1]
        if self.mode == "fetch":
            self.ck[i, :T].copy_(k.transpose(0, 1))
            self.cv[i, :T].copy_(v.transpose(0, 1))
            self.P = T
        else:
            g = min(self.G, T)
            self.gk[i, :, :g], self.gv[i, :, :g] = k[:, :g], v[:, :g]
            if T > g:
                self.ck[i, :, :T - g].copy_(k[:, g:])
                self.cv[i, :, :T - g].copy_(v[:, g:])
        self.n = T

    def _fetch(self, i):
        main = torch.cuda.current_stream()
        with torch.cuda.stream(self.side):
            kk = self.ck[i, :self.P].to("cuda", non_blocking=True)
            vv = self.cv[i, :self.P].to("cuda", non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.side)
        kk.record_stream(main)
        vv.record_stream(main)
        return kk, vv, ev

    def attn(self, i, q, k, v):
        """q (kvh, g, D), k/v (kvh, 1, D) for the token at position self.n. Returns (kvh, g, D) fp32."""
        r, n, last = self.r, self.n, i == self.r.L - 1
        if self.mode == "fetch":
            if i == 0:
                self.buf[0] = self._fetch(0)
            kk, vv, ev = self.buf.pop(i)
            if not last:
                self.buf[i + 1] = self._fetch(i + 1)
            d = n - self.P
            self.tk[i, :, d:d + 1], self.tv[i, :, d:d + 1] = k, v
            torch.cuda.current_stream().wait_event(ev)
            kk = torch.cat([kk.transpose(0, 1), self.tk[i, :, :d + 1]], 1)
            vv = torch.cat([vv.transpose(0, 1), self.tv[i, :, :d + 1]], 1)
            out = attend_gpu(q, kk, vv, r.scale)[0]
        elif n < self.G:                          # fits on the GPU (gpu mode, or split below G)
            self.gk[i, :, n:n + 1], self.gv[i, :, n:n + 1] = k, v
            out = attend_gpu(q, self.gk[i, :, :n + 1], self.gv[i, :, :n + 1], r.scale)[0]
        else:
            c = n - self.G
            qkv = torch.cat([q.reshape(-1, r.D), k.reshape(-1, r.D), v.reshape(-1, r.D)]).float().cpu()
            self.ck[i, :, c] = qkv[r.H:r.H + r.KVH]
            self.cv[i, :, c] = qkv[r.H + r.KVH:]
            if self.G:                            # GPU part is launched (async) before the CPU part runs
                og, lg = attend_gpu(q, self.gk[i], self.gv[i], r.scale)
            oc, lc = attend(qkv[:r.H].view(r.KVH, r.g, r.D), self.ck[i, :, :c + 1], self.cv[i, :, :c + 1], r.scale)
            oc, lc = oc.to("cuda"), lc.to("cuda")
            out = merge(og, lg, oc, lc) if self.G else oc
        if last:
            self.n += 1
        return out


def rss_gb():
    with open("/proc/self/status") as f:
        kb = {l.split(":")[0]: int(l.split()[1]) for l in f if l.startswith(("VmRSS", "RssShmem"))}
    return kb.get("VmRSS", 0) / 2**20


def run_mode(r, ids, mode, ref=None):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    T = ids.shape[1] + NEW_TOKENS
    row = {"mode": mode, "prompt": ids.shape[1]}
    cpu_gb = 2 * r.L * r.KVH * r.D * (T if mode == "fetch" else T - (T if mode == "gpu" else min(G, T) if mode == "split" else 0)) \
        * (2 if mode == "fetch" else 4) / 2**30
    if rss_gb() + cpu_gb > HOST_LIMIT_GB:
        row.update(skipped=f"needs {cpu_gb:.1f} GB host memory on top of {rss_gb():.1f} GB in use (limit {HOST_LIMIT_GB:.0f} GB)")
        return row
    try:
        cache = Cache(r, T, mode, G)
        t0 = time.time()
        logits = r.prefill(ids, cache.store)
        torch.cuda.synchronize()
        row["prefill_s"] = round(time.time() - t0, 2)
        toks, steps, times = [], [], []
        tok = logits.argmax()
        for s in range(NEW_TOKENS):
            toks.append(int(tok))
            steps.append(logits.cpu())
            t1 = time.time()
            logits = r.step(tok, ids.shape[1] + s, cache.attn)
            tok = logits.argmax()
            torch.cuda.synchronize()
            times.append(time.time() - t1)
        row.update(ms_per_token=round(1000 * sum(times[2:]) / len(times[2:]), 1),
                   peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   kv_on_gpu_gb=round(cache.gpu_bytes() / 2**30, 2),
                   kv_on_cpu_gb=round((cache.ck.numel() + cache.cv.numel()) * cache.ck.element_size() / 2**30, 2),
                   tokens=toks)
        if ref is not None:
            row["same_tokens_as_ref"] = sum(a == b for a, b in zip(toks, ref["tokens"]))
            n = min(len(steps), len(ref["steps"]))
            # teacher-forced comparison is only valid while the token streams agree
            agree = next((j for j in range(n) if toks[j] != ref["tokens"][j]), n)
            diffs = [(steps[j] - ref["steps"][j]).abs().max().item() for j in range(min(agree + 1, n))]
            row["max_logit_diff_vs_ref"] = round(max(diffs), 4)
        row["_steps"] = steps
    except torch.cuda.OutOfMemoryError:
        row["oom"] = True
    finally:
        cache = None
        gc.collect()
        torch.cuda.empty_cache()
        torch._C._host_emptyCache()               # give cached pinned host memory back (it outlives the tensors)
    return row


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, local_files_only=True,
                                                 device_map="cuda").eval()
    r = Runner(model)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    all_ids = tok(text, return_tensors="pt").input_ids.cuda()
    res = {"model": "Qwen3-8B bf16", "gpu": torch.cuda.get_device_name(), "gpu_total_gb":
           round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), "cpu_threads": torch.get_num_threads(),
           "new_tokens": NEW_TOKENS, "split_gpu_tokens": G, "kv_kib_per_token": 2 * r.L * r.KVH * r.D * 2 / 1024}

    with torch.no_grad():                                 # the layer loop must reproduce the HF forward pass
        short = all_ids[:, :256]
        hf = model(short).logits[0].float()
        mine = r.prefill(short, lambda *a: None)
        c = Cache(r, 300, "gpu", 0)
        r.prefill(short[:, :-1], c.store)
        nxt = r.step(short[0, -1], 255, c.attn)
        res["check_vs_hf"] = {"prefill_last_logit_max_diff": round((mine - hf[-1]).abs().max().item(), 4),
                              "decode_step_max_diff": round((nxt - hf[-1]).abs().max().item(), 4),
                              "hf_logit_scale": round(hf[-1].abs().max().item(), 2)}
        print(res["check_vs_hf"], flush=True)
        del c, hf

    old = json.loads(OUT.read_text())["rows"] if OUT.exists() else []
    res["rows"] = [x for x in old if x["prompt"] not in LENGTHS]   # keep finished lengths from an earlier run
    for T in LENGTHS:
        ids = all_ids[:, :T]
        ref = None
        for mode in ("gpu", "fetch", "cpu", "split"):
            row = run_mode(r, ids, mode, ref)
            if mode in ("gpu", "fetch") and ref is None and not row.get("oom") and not row.get("skipped"):
                ref = {"tokens": row["tokens"], "steps": row["_steps"], "mode": mode}
            elif ref is not None:
                row["ref"] = ref["mode"]
            row.pop("_steps", None)
            row.pop("tokens", None)
            res["rows"].append(row)
            print(row, flush=True)
            OUT.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
