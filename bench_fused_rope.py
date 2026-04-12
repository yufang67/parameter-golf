#!/usr/bin/env python3
"""Benchmark fused norm+rope+gain vs torch.compile baseline."""
import torch
import torch.nn.functional as F
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from train_gpt_improved import (
    fused_norm_rope_gain, fused_norm_rope, apply_rotary_emb, _HAS_TRITON,
)
assert _HAS_TRITON, "Triton not available"

B, T, H_q, H_kv, D = 48, 2048, 8, 4, 64
RH = 16  # rope_dims=32 -> rope_half=16
rope_dims = RH * 2
WARMUP, ITERS = 20, 100

torch.manual_seed(42)
q = torch.randn(B, T, H_q, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, T, H_kv, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
gain = torch.randn(H_q, device='cuda', dtype=torch.float32)

inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dims, 2, dtype=torch.float32, device='cuda') / rope_dims))
freqs = torch.outer(torch.arange(T, device='cuda', dtype=torch.float32), inv_freq)
cos = freqs.cos()[None, :, None, :]
sin = freqs.sin()[None, :, None, :]

# === Baseline (separate ops, compiled) ===
@torch.compile(fullgraph=True)
def baseline_q(x, cos, sin, gain):
    x = F.rms_norm(x, (x.size(-1),))
    x = apply_rotary_emb(x, cos, sin, rope_dims)
    return x * gain.to(dtype=x.dtype)[None, None, :, None]

@torch.compile(fullgraph=True)
def baseline_k(x, cos, sin):
    x = F.rms_norm(x, (x.size(-1),))
    return apply_rotary_emb(x, cos, sin, rope_dims)

# Warmup compiled
for _ in range(WARMUP):
    qo = baseline_q(q, cos, sin, gain)
    ko = baseline_k(k, cos, sin)
    (qo.sum() + ko.sum()).backward()
    q.grad = None; k.grad = None
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(ITERS):
    qo = baseline_q(q, cos, sin, gain)
    ko = baseline_k(k, cos, sin)
    (qo.sum() + ko.sum()).backward()
    q.grad = None; k.grad = None
torch.cuda.synchronize()
baseline_ms = (time.perf_counter() - t0) / ITERS * 1000
print(f"torch.compile baseline: {baseline_ms:.3f} ms/iter")

# === Fused Triton ===
for _ in range(WARMUP):
    qo = fused_norm_rope_gain(q, cos, sin, gain, RH)
    ko = fused_norm_rope(k, cos, sin, RH)
    (qo.sum() + ko.sum()).backward()
    q.grad = None; k.grad = None
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(ITERS):
    qo = fused_norm_rope_gain(q, cos, sin, gain, RH)
    ko = fused_norm_rope(k, cos, sin, RH)
    (qo.sum() + ko.sum()).backward()
    q.grad = None; k.grad = None
torch.cuda.synchronize()
fused_ms = (time.perf_counter() - t0) / ITERS * 1000
print(f"fused Triton kernel:    {fused_ms:.3f} ms/iter")

speedup = baseline_ms / fused_ms
print(f"speedup: {speedup:.2f}x")
