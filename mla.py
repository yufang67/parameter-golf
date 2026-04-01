"""Attention variants for parameter-golf.

- MultiHeadLatentAttention (MLA) — DeepSeek-V2 style low-rank KV compression.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def _apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), None)


class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))


class MultiHeadLatentAttention(nn.Module):
    """MLA: compress KV into a low-rank latent, then expand to per-head K/V.

    Uses decoupled RoPE (DeepSeek-V2 style): positional information flows
    through a separate small projection that bypasses the KV bottleneck,
    so each head gets independent position-aware keys without being
    rank-limited by kv_latent_dim.

    Q = [q_nope | q_rope]   — split into content and position parts
    K = [k_nope | k_rope]   — k_nope from latent (no RoPE), k_rope from x (with RoPE)
    score = (q_nope · k_nope + q_rope · k_rope) / sqrt(head_dim)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kv_latent_dim: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dim: int | None = None,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        # Decoupled RoPE: split head_dim into nope (content) and rope (position) parts
        # Default rope_dim = head_dim // 2, matching DeepSeek-V2's design
        self.rope_dim = rope_dim if rope_dim is not None else self.head_dim // 2
        if self.rope_dim % 2 != 0:
            raise ValueError("rope_dim must be even")
        self.nope_dim = self.head_dim - self.rope_dim

        # Q – full-rank, produces both q_nope and q_rope
        # Fused with k_rope_proj: single matmul from x -> [Q, k_rope]
        self._q_dim = dim
        self._k_rope_dim = self.rope_dim * num_heads
        self.c_q_krope = CastedLinear(dim, self._q_dim + self._k_rope_dim, bias=False)

        # KV compression for content (no RoPE): x -> latent -> [k_nope, V] (fused)
        self.kv_down = CastedLinear(dim, kv_latent_dim, bias=False)
        self.kv_norm = RMSNorm()
        self._k_nope_dim = self.nope_dim * num_heads
        self._v_dim = dim
        self.kv_up = CastedLinear(kv_latent_dim, self._k_nope_dim + self._v_dim, bias=False)

        # Output projection
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.rope_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape

        # Fused Q + k_rope projection: 1 matmul instead of 2
        q_krope = self.c_q_krope(x)
        q = q_krope[..., :self._q_dim].reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        q_nope = q[..., :self.nope_dim]
        q_rope = q[..., self.nope_dim:]
        k_rope = q_krope[..., self._q_dim:].reshape(bsz, seqlen, self.num_heads, self.rope_dim).transpose(1, 2)

        # Fused k_nope + V up-projection: 1 matmul instead of 2
        c_kv = self.kv_norm(self.kv_down(x))
        kv_up = self.kv_up(c_kv)
        k_nope = kv_up[..., :self._k_nope_dim].reshape(bsz, seqlen, self.num_heads, self.nope_dim).transpose(1, 2)
        v = kv_up[..., self._k_nope_dim:].reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        # Normalize content Q/K
        q_nope = F.rms_norm(q_nope, (q_nope.size(-1),))
        k_nope = F.rms_norm(k_nope, (k_nope.size(-1),))

        # Apply RoPE only to position parts
        q_rope = F.rms_norm(q_rope, (q_rope.size(-1),))
        k_rope = F.rms_norm(k_rope, (k_rope.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q_rope.dtype)
        q_rope = _apply_rotary_emb(q_rope, cos, sin)
        k_rope = _apply_rotary_emb(k_rope, cos, sin)

        # Reassemble full Q and K: [content | position]
        q_full = torch.cat([q_nope, q_rope], dim=-1)  # [B, H, T, head_dim]
        k_full = torch.cat([k_nope, k_rope], dim=-1)  # [B, H, T, head_dim]

        q_full = q_full * self.q_gain.to(dtype=q_full.dtype)[None, :, None, None]

        y = F.scaled_dot_product_attention(q_full, k_full, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)
