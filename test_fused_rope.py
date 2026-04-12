#!/usr/bin/env python3
"""Test fused norm+rope+gain kernel correctness."""
import torch
import torch.nn.functional as F
import sys

# Import the fused kernels from the training script
sys.path.insert(0, '.')
# We need _HAS_TRITON and the kernel functions
import triton, triton.language as tl

# Re-import just the kernel code by importing the module
# Actually, easier to just test via the module
from train_gpt_improved import (
    CausalSelfAttention, Rotary, apply_rotary_emb, Hyperparameters,
)

# Check triton is available
try:
    from train_gpt_improved import fused_norm_rope_gain, fused_norm_rope, _HAS_TRITON
    assert _HAS_TRITON, "Triton not available"
except ImportError:
    print("fused_norm_rope_gain not available, Triton not found")
    sys.exit(1)

def test_forward():
    B, T, H, D = 2, 16, 8, 64
    RH = 16  # rope_dims=32 -> rope_half=16
    rope_dims = RH * 2
    torch.manual_seed(42)
    x = torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16)
    gain = torch.randn(H, device='cuda', dtype=torch.float32)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dims, 2, dtype=torch.float32, device='cuda') / rope_dims))
    t_vals = torch.arange(T, device='cuda', dtype=torch.float32)
    freqs = torch.outer(t_vals, inv_freq)
    cos = freqs.cos()[None, :, None, :]
    sin = freqs.sin()[None, :, None, :]

    # Reference
    x_ref = x.clone()
    q_normed = F.rms_norm(x_ref, (x_ref.size(-1),))
    x_rope, x_pass = q_normed[..., :rope_dims], q_normed[..., rope_dims:]
    x1, x2 = x_rope[..., :RH], x_rope[..., RH:]
    x_rope_out = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    ref_out = torch.cat((x_rope_out, x_pass), dim=-1)
    ref_out = ref_out * gain.to(dtype=ref_out.dtype)[None, None, :, None]

    # Fused
    fused_out = fused_norm_rope_gain(x.clone(), cos, sin, gain, RH)

    max_diff = (ref_out.float() - fused_out.float()).abs().max().item()
    print(f'Forward Q (with gain) max diff: {max_diff:.6e}')
    assert max_diff < 0.05, f'Forward diff too large: {max_diff}'

    # Test K path (no gain)
    k_normed = F.rms_norm(x.clone(), (D,))
    k_rope, k_pass = k_normed[..., :rope_dims], k_normed[..., rope_dims:]
    k1, k2 = k_rope[..., :RH], k_rope[..., RH:]
    k_rope_out = torch.cat((k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos), dim=-1)
    k_ref_out = torch.cat((k_rope_out, k_pass), dim=-1)

    k_fused = fused_norm_rope(x.clone(), cos, sin, RH)
    k_diff = (k_ref_out.float() - k_fused.float()).abs().max().item()
    print(f'Forward K (no gain) max diff: {k_diff:.6e}')
    assert k_diff < 0.05, f'K forward diff too large: {k_diff}'

def test_backward():
    B, T, H, D = 2, 16, 8, 64
    RH = 16
    rope_dims = RH * 2
    torch.manual_seed(123)
    x = torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16)
    gain = torch.randn(H, device='cuda', dtype=torch.float32, requires_grad=True)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dims, 2, dtype=torch.float32, device='cuda') / rope_dims))
    t_vals = torch.arange(T, device='cuda', dtype=torch.float32)
    freqs = torch.outer(t_vals, inv_freq)
    cos = freqs.cos()[None, :, None, :]
    sin = freqs.sin()[None, :, None, :]
    grad_out = torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16)

    # Reference backward
    x_ref = x.clone().detach().requires_grad_(True)
    gain_ref = gain.clone().detach().requires_grad_(True)
    q_normed = F.rms_norm(x_ref, (x_ref.size(-1),))
    x_rope, x_pass = q_normed[..., :rope_dims], q_normed[..., rope_dims:]
    x1, x2 = x_rope[..., :RH], x_rope[..., RH:]
    x_rope_out = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    ref_out = torch.cat((x_rope_out, x_pass), dim=-1)
    ref_out = ref_out * gain_ref.to(dtype=ref_out.dtype)[None, None, :, None]
    ref_out.backward(grad_out)

    # Fused backward
    x_fused = x.clone().detach().requires_grad_(True)
    gain_fused = gain.clone().detach().requires_grad_(True)
    fused_out = fused_norm_rope_gain(x_fused, cos, sin, gain_fused, RH)
    fused_out.backward(grad_out)

    dx_diff = (x_ref.grad.float() - x_fused.grad.float()).abs().max().item()
    print(f'Backward dx max diff: {dx_diff:.6e}')
    assert dx_diff < 0.1, f'Backward dx diff too large: {dx_diff}'

    dgain_diff = (gain_ref.grad.float() - gain_fused.grad.float()).abs().max().item()
    print(f'Backward dgain max diff: {dgain_diff:.6e}')
    assert dgain_diff < 1.0, f'Backward dgain diff too large: {dgain_diff}'

if __name__ == '__main__':
    test_forward()
    test_backward()
    print('All fused rope kernel tests passed!')
