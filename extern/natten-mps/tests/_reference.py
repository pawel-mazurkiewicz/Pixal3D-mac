"""Pure-torch reference implementations of na2d_qk / na2d_av.

Matches natten 0.17.5 CPU naive semantics (BSD-3 / Apache-2.0).  The
algorithm is faithful to the original:

  csrc/include/natten/cpu/naive/pointwise_neighborhood_2d.hpp
  csrc/include/natten/natten_cpu_commons.h  (get_window_start)

Used as algorithm anchor for parity tests at small shapes — slow Python
loops, but fully under our control (no dependency on installing
natten==0.17.5, which has no Mac wheel).

Boundary handling: "edge-shift-clamp"
  window is always exactly kernel_size tokens wide; corner queries
  don't center, they anchor to the nearest valid range.

  start = max(i - nbr_eff, 0)
  if (i + nbr_eff) >= L:
      start += L - i - nbr_eff - 1   # (negative shift)

where nbr_eff = (kernel_size // 2) * dilation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _window_start(i: int, L: int, nbr_eff: int) -> int:
    """natten edge-shift-clamp window anchor.

    Returns the start coordinate of the K-tap window that "covers" query i.
    """
    s = max(i - nbr_eff, 0)
    if i + nbr_eff >= L:
        s += L - i - nbr_eff - 1
    return s


def na2d_qk_reference(
    q: Tensor, k: Tensor, kernel_size: int, dilation: int = 1
) -> Tensor:
    """Windowed neighborhood Q·K dot product.

    Args:
        q: [B, H, X, Y, D]  (heads-second; matches natten 0.17.5 legacy layout)
        k: [B, H, X, Y, D]
        kernel_size: int (square kernel; asymmetric not supported here)
        dilation: int

    Returns:
        attn_scores: [B, H, X, Y, K*K], dtype=fp32 regardless of input dtype.
    """
    assert q.shape == k.shape, f"q/k shape mismatch: {q.shape} vs {k.shape}"
    B, H, X, Y, D = q.shape
    K = int(kernel_size)
    dil = int(dilation)
    nbr_eff = (K // 2) * dil
    # natten edge-shift-clamp requires the dilated kernel to fit at least
    # once along each axis: extent = (K - 1) * dilation + 1.
    min_extent = (K - 1) * dil + 1
    assert X >= min_extent and Y >= min_extent, (
        f"natten config invalid: X={X}, Y={Y} must be >= "
        f"(K-1)*dilation+1 = {min_extent} for K={K}, dilation={dil}"
    )

    # fp32 accumulator for parity with natten CPU naive
    out = torch.zeros(B, H, X, Y, K * K, device=q.device, dtype=torch.float32)
    q_f = q.float()
    k_f = k.float()

    for x in range(X):
        sx = _window_start(x, X, nbr_eff)
        for y in range(Y):
            sy = _window_start(y, Y, nbr_eff)
            for ki in range(K):
                kx = sx + ki * dilation
                for kj in range(K):
                    ky = sy + kj * dilation
                    # q[:, :, x, y, :] · k[:, :, kx, ky, :]  along D
                    out[:, :, x, y, ki * K + kj] = (
                        q_f[:, :, x, y, :] * k_f[:, :, kx, ky, :]
                    ).sum(dim=-1)
    return out


def na2d_av_reference(
    attn: Tensor, v: Tensor, kernel_size: int, dilation: int = 1
) -> Tensor:
    """Windowed attention · V weighted gather.

    Args:
        attn: [B, H, X, Y, K*K]   (post-softmax weights, fp32)
        v:    [B, H, X, Y, D]
        kernel_size: int (square)
        dilation: int

    Returns:
        out: [B, H, X, Y, D], same dtype as v.
    """
    B, H, X, Y, KK = attn.shape
    B2, H2, X2, Y2, D = v.shape
    assert (B, H, X, Y) == (B2, H2, X2, Y2), "attn/v spatial shape mismatch"
    K = int(kernel_size)
    assert K * K == KK, f"kernel_size² ({K*K}) != attn[-1] ({KK})"
    dil = int(dilation)
    nbr_eff = (K // 2) * dil
    min_extent = (K - 1) * dil + 1
    assert X >= min_extent and Y >= min_extent, (
        f"natten config invalid: X={X}, Y={Y} must be >= "
        f"(K-1)*dilation+1 = {min_extent} for K={K}, dilation={dil}"
    )

    out = torch.zeros(B, H, X, Y, D, device=v.device, dtype=torch.float32)
    attn_f = attn.float()
    v_f = v.float()

    for x in range(X):
        sx = _window_start(x, X, nbr_eff)
        for y in range(Y):
            sy = _window_start(y, Y, nbr_eff)
            for ki in range(K):
                kx = sx + ki * dilation
                for kj in range(K):
                    ky = sy + kj * dilation
                    w = attn_f[:, :, x, y, ki * K + kj].unsqueeze(-1)  # [B,H,1]
                    out[:, :, x, y, :] += w * v_f[:, :, kx, ky, :]

    return out.to(v.dtype)
