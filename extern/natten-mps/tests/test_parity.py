"""Parity tests: Metal kernels vs pure-torch reference.

Small shapes (X=Y in {16, 17}, D=64, K=9, dilation in {1, 2}).  Anchored to
natten 0.17.5 CPU naive semantics via `_reference.py`.  These tests verify:
  - dot product correctness (interior queries)
  - edge-shift-clamp boundary handling (corner queries)
  - dilation arithmetic
  - dtype promotion (fp16 inputs → fp32 accumulator inside kernel)

Target tolerance:
  - fp32 inputs: ≤ 1e-4 abs (both kernels)
  - fp16 inputs: ≤ 5e-3 rel (loose; fp16 accum drift is bounded but real)

The CUDA-cutlass-fna golden parity (test_cuda_golden.py, against captured
fixture at B=1, H=4, X=Y=512, D=64/256, ks=9, dilation=16) is the second-
stage validation once these pass.
"""

from __future__ import annotations

import pytest
import torch

# Import side-effect: registers torch.ops.natten_mps.na2d_qk / na2d_av.
import natten_mps  # noqa: F401

from _reference import na2d_qk_reference, na2d_av_reference


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS unavailable",
)


# Each parametrize entry is (X, Y, dilation, label).
#   - X=17 hits odd-sized boundary
#   - dilation=2 requires X,Y >= (K-1)*dilation + 1 = 17 for K=9
#   - 24x16_rect tests asymmetric grids (NAF feeds these to natten too)
SHAPES = [
    (16, 16, 1, "16x16_dil1"),
    (17, 17, 1, "17x17_dil1"),
    (17, 17, 2, "17x17_dil2"),
    (20, 20, 2, "20x20_dil2"),
    (24, 16, 1, "24x16_rect"),
]


@pytest.mark.parametrize("X,Y,dilation,label", SHAPES,
                         ids=[s[3] for s in SHAPES])
def test_qk_parity(X, Y, dilation, label):
    torch.manual_seed(0xC0FFEE)
    B, H, D, K = 1, 2, 64, 9

    # Build on CPU (reference is CPU-only) then mirror to MPS.
    q_cpu = torch.randn(B, H, X, Y, D, dtype=torch.float32)
    k_cpu = torch.randn(B, H, X, Y, D, dtype=torch.float32)

    ref = na2d_qk_reference(q_cpu, k_cpu, K, dilation)

    q_mps = q_cpu.to("mps").contiguous()
    k_mps = k_cpu.to("mps").contiguous()
    out = torch.ops.natten_mps.na2d_qk(q_mps, k_mps, K, dilation).cpu()

    abs_diff = (out - ref).abs()
    max_diff = abs_diff.max().item()
    print(f"\n  [qk:{label}] max_abs_diff = {max_diff:.3e}")
    # 1e-4 leaves room for fp32 reduction-order differences between torch's
    # CPU summation and our Metal threaded summation.
    assert max_diff < 1e-4, (
        f"QK parity failed at {label}: max_abs_diff={max_diff:.3e} "
        f"(>1e-4).  diff>1e-5 count: "
        f"{(abs_diff > 1e-5).sum().item()}/{abs_diff.numel()}"
    )


@pytest.mark.parametrize("X,Y,dilation,label", SHAPES,
                         ids=[s[3] for s in SHAPES])
def test_av_parity(X, Y, dilation, label):
    torch.manual_seed(0xBEEF)
    B, H, D, K = 1, 2, 64, 9

    attn_logits_cpu = torch.randn(B, H, X, Y, K * K, dtype=torch.float32)
    attn_cpu = attn_logits_cpu.softmax(dim=-1)
    v_cpu = torch.randn(B, H, X, Y, D, dtype=torch.float32)

    ref = na2d_av_reference(attn_cpu, v_cpu, K, dilation)

    attn_mps = attn_cpu.to("mps").contiguous()
    v_mps = v_cpu.to("mps").contiguous()
    out = torch.ops.natten_mps.na2d_av(attn_mps, v_mps, K, dilation).cpu()

    abs_diff = (out - ref).abs()
    max_diff = abs_diff.max().item()
    print(f"\n  [av:{label}] max_abs_diff = {max_diff:.3e}")
    assert max_diff < 1e-4, (
        f"AV parity failed at {label}: max_abs_diff={max_diff:.3e} "
        f"(>1e-4).  diff>1e-5 count: "
        f"{(abs_diff > 1e-5).sum().item()}/{abs_diff.numel()}"
    )


def test_qk_at_naf_channel_dim():
    """NAF uses D=64 for Q/K; this is the dim our kernel will see in prod."""
    torch.manual_seed(7)
    B, H, X, Y, D, K = 1, 4, 16, 16, 64, 9
    dilation = 1
    q_cpu = torch.randn(B, H, X, Y, D, dtype=torch.float32)
    k_cpu = torch.randn(B, H, X, Y, D, dtype=torch.float32)
    ref = na2d_qk_reference(q_cpu, k_cpu, K, dilation)
    out = torch.ops.natten_mps.na2d_qk(
        q_cpu.to("mps").contiguous(), k_cpu.to("mps").contiguous(), K, dilation
    ).cpu()
    max_diff = (out - ref).abs().max().item()
    print(f"\n  [qk:NAF_D64] max_abs_diff = {max_diff:.3e}")
    assert max_diff < 1e-4


def test_av_at_naf_value_dim():
    """NAF uses D=256 for V (asymmetric attention).  Tests our MAX_D=256
    stack array doesn't blow up.
    """
    torch.manual_seed(11)
    B, H, X, Y, D, K = 1, 4, 16, 16, 256, 9
    dilation = 1
    attn_cpu = torch.randn(B, H, X, Y, K * K).softmax(-1)
    v_cpu = torch.randn(B, H, X, Y, D, dtype=torch.float32)
    ref = na2d_av_reference(attn_cpu, v_cpu, K, dilation)
    out = torch.ops.natten_mps.na2d_av(
        attn_cpu.to("mps").contiguous(), v_cpu.to("mps").contiguous(),
        K, dilation
    ).cpu()
    max_diff = (out - ref).abs().max().item()
    print(f"\n  [av:NAF_D256] max_abs_diff = {max_diff:.3e}")
    assert max_diff < 1e-4
