"""Custom ops + MPS kernel registration for natten-mps.

Exposes `natten_mps::na2d_qk` and `natten_mps::na2d_av` (split path).  The
fused `na2d` is added separately once the split path validates.

Tensor layout (matches natten 0.17.5 legacy split-path semantics):
  q, k:  [B, H, X, Y, D]   heads-second
  attn: [B, H, X, Y, K*K]
  v:    [B, H, X, Y, D]
  out:  [B, H, X, Y, D]
where K is kernel_size, X/Y is the token grid, D is head_dim.

Scaling (sqrt(d)) is NOT applied inside `na2d_qk` — caller multiplies and
softmaxes before calling `na2d_av`.
"""

from __future__ import annotations

from typing import Tuple
import torch
from torch import Tensor
from torch.library import custom_op, register_fake

from ._compile import get_lib


# -- na2d_qk -----------------------------------------------------------------

@custom_op("natten_mps::na2d_qk", mutates_args=())
def na2d_qk(q: Tensor, k: Tensor, kernel_size: int, dilation: int = 1) -> Tensor:
    """Windowed Q·K dot product. Implemented on MPS; raises on other devices."""
    raise NotImplementedError(
        "natten_mps.na2d_qk only implements the MPS kernel. "
        "Use natten.functional.na2d_qk for CPU/CUDA."
    )


@register_fake("natten_mps::na2d_qk")
def _na2d_qk_meta(q: Tensor, k: Tensor, kernel_size: int, dilation: int = 1):
    B, H, X, Y, _ = q.shape
    return q.new_empty(B, H, X, Y, kernel_size * kernel_size, dtype=torch.float32)


@na2d_qk.register_kernel("mps")
def _na2d_qk_mps(q: Tensor, k: Tensor, kernel_size: int, dilation: int = 1) -> Tensor:
    assert q.is_contiguous() and k.is_contiguous(), \
        "natten_mps.na2d_qk: inputs must be contiguous (call .contiguous() first)"
    assert q.shape == k.shape, f"na2d_qk: q/k shape mismatch {q.shape} vs {k.shape}"
    B, H, X, Y, D = q.shape
    K = int(kernel_size)
    dil = int(dilation)

    out = torch.empty(B, H, X, Y, K * K, device=q.device, dtype=torch.float32)
    shape = torch.tensor([B, H, X, Y, D, K, dil], dtype=torch.int32, device=q.device)

    lib = get_lib()
    lib.na2d_qk(
        q, k, out, shape,
        threads=(Y, X, B * H),
        group_size=(8, 8, 1),
    )
    return out


# -- na2d_av -----------------------------------------------------------------

@custom_op("natten_mps::na2d_av", mutates_args=())
def na2d_av(attn: Tensor, v: Tensor, kernel_size: int, dilation: int = 1) -> Tensor:
    """Windowed attention · V weighted gather. MPS-only."""
    raise NotImplementedError(
        "natten_mps.na2d_av only implements the MPS kernel. "
        "Use natten.functional.na2d_av for CPU/CUDA."
    )


@register_fake("natten_mps::na2d_av")
def _na2d_av_meta(attn: Tensor, v: Tensor, kernel_size: int, dilation: int = 1):
    B, H, X, Y, _ = attn.shape
    D = v.shape[-1]
    return attn.new_empty(B, H, X, Y, D, dtype=v.dtype)


@na2d_av.register_kernel("mps")
def _na2d_av_mps(attn: Tensor, v: Tensor, kernel_size: int, dilation: int = 1) -> Tensor:
    assert attn.is_contiguous() and v.is_contiguous(), \
        "natten_mps.na2d_av: inputs must be contiguous"
    B, H, X, Y, KK = attn.shape
    B2, H2, X2, Y2, D = v.shape
    assert (B, H, X, Y) == (B2, H2, X2, Y2), \
        f"na2d_av: attn/v batch-head-spatial shape mismatch"
    K = int(kernel_size)
    dil = int(dilation)
    assert K * K == KK, f"na2d_av: kernel_size² ({K*K}) != attn[-1] ({KK})"

    out = torch.empty(B, H, X, Y, D, device=v.device, dtype=v.dtype)
    shape = torch.tensor([B, H, X, Y, D, K, dil], dtype=torch.int32, device=v.device)

    lib = get_lib()
    lib.na2d_av(
        attn, v, out, shape,
        threads=(Y, X, B * H),
        group_size=(8, 8, 1),
    )
    return out
