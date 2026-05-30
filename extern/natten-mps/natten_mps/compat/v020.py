"""Drop-in replacement for the community natten-mps `compat.v020` API.

Why this exists
---------------
The community `natten-mps==0.3.0` package exposes a *fused* `na2d(q, k, v, ...)`
that requires Q, K, and V to share a head dim. NAF's cross-attention is
asymmetric — Q/K have head_dim 64 while V has head_dim 256 — so the community
fused kernel raises `ValueError: Head dim must match: K has 64, V has 256` on
every call and the pipeline silently falls back to pure-PyTorch attention.

Our in-house kernels are a *split* path (`na2d_qk` over Q/K, then `na2d_av`
over attn/V), which handles asymmetric K/V natively. This module re-implements
the `compat.v020` surface that `generate_mps.py` imports, backed by those split
kernels via the shim's `_na2d_mps_impl`, so the existing pipeline wiring routes
to real Metal execution instead of the fallback — with no generate_mps.py
change required.

Numerically equivalent to the PyTorch fallback path (validated against CUDA
cutlass-fna golden at NAF shapes, S11) — this is a performance/cleanliness
change, not a quality change.
"""

from __future__ import annotations

import torch

from .._shim import _na2d_mps_impl
from .._ops import na2d_av, na2d_qk  # re-export split path (parity w/ community)


def has_cuda() -> bool:
    return False


def has_mps() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def has_fna() -> bool:
    return False


def na2d(query, key, value, kernel_size, *args, **kwargs):
    """Fused 2D neighborhood attention.

    On MPS, decomposes into our split kernels (na2d_qk -> softmax -> na2d_av),
    which handle asymmetric K/V. Off MPS, defers to upstream natten so the
    shim stays transparent on CPU/CUDA.

    Accepts and ignores the natten-only `backend` and `stride` kwargs (NAF
    always calls with stride=1; backend is a CUDA dispatch hint that does not
    apply to our Metal path).
    """
    kwargs.pop("backend", None)
    stride = kwargs.pop("stride", 1)
    if stride not in (1, (1, 1), [1, 1]):
        raise NotImplementedError(
            f"natten_mps.compat.v020.na2d only supports stride=1 (got {stride!r})"
        )

    if query.device.type == "mps":
        return _na2d_mps_impl(query, key, value, kernel_size, *args, **kwargs)

    # CPU / CUDA: defer to the genuine upstream natten implementation. Use the
    # original na2d the shim stashed (if it patched it) to avoid recursing back
    # into our dispatch.
    import natten

    upstream = getattr(natten, "_original_na2d", None) or natten.na2d
    return upstream(query, key, value, kernel_size, *args, **kwargs)


__all__ = [
    "na2d",
    "na2d_qk",
    "na2d_av",
    "has_cuda",
    "has_mps",
    "has_fna",
]
