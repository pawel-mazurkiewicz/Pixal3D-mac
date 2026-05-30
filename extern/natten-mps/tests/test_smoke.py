"""Smoke tests: package imports cleanly, shader compiles, ops execute.

Does NOT validate correctness — that's test_parity.py / test_cuda_golden.py.
"""

from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS unavailable",
)


def test_import_clean():
    import natten_mps  # noqa: F401


def test_shader_compiles():
    from natten_mps._compile import get_lib
    lib = get_lib()
    assert hasattr(lib, "na2d_qk"), "na2d_qk kernel not found in compiled lib"
    assert hasattr(lib, "na2d_av"), "na2d_av kernel not found in compiled lib"


def test_na2d_qk_runs():
    torch.manual_seed(0)
    B, H, X, Y, D, K = 1, 2, 16, 16, 32, 9
    q = torch.randn(B, H, X, Y, D, device="mps", dtype=torch.float32)
    k = torch.randn(B, H, X, Y, D, device="mps", dtype=torch.float32)
    out = torch.ops.natten_mps.na2d_qk(q, k, K, 1)
    assert out.shape == (B, H, X, Y, K * K)
    assert out.device.type == "mps"
    assert torch.isfinite(out).all()


def test_na2d_av_runs():
    torch.manual_seed(0)
    B, H, X, Y, D, K = 1, 2, 16, 16, 32, 9
    attn = torch.randn(B, H, X, Y, K * K, device="mps", dtype=torch.float32).softmax(-1)
    v = torch.randn(B, H, X, Y, D, device="mps", dtype=torch.float32)
    out = torch.ops.natten_mps.na2d_av(attn, v, K, 1)
    assert out.shape == (B, H, X, Y, D)
    assert out.device.type == "mps"
    assert torch.isfinite(out).all()


def test_shim_installs():
    import natten_mps
    # Already auto-installed on import; calling again should be idempotent.
    assert natten_mps.install_shim() is False  # already patched
    assert natten_mps.install_shim(force=True) is True
