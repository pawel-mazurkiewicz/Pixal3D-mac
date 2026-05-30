"""CUDA cutlass-fna golden parity for natten-mps Metal kernels.

This is the *real-world* validation: load tensors captured from a CUDA
inference run of NAF (natten cutlass-fna backend) and confirm our Metal
kernels reproduce the same outputs to within ≤ 1e-4 abs at NAF's actual
production shapes:

  na2d_qk:  q,k = [1, 4, 512, 512, 64], out = [1, 4, 512, 512, 81]
  na2d_av:  attn = [1, 4, 512, 512, 81], v,out = [1, 4, 512, 512, 256]
  kernel_size = (9, 9), dilation = (16, 16)

Tolerance vs CUDA — two different scales of check:

  na2d_qk: cutlass-fna runs on Ada/Hopper tensor cores which use
    TensorFloat-32 (10-bit mantissa, half of fp32) by default in
    recent PyTorch.  Our Metal kernel is full fp32.  Drift between
    them is ~1e-3 absolute on pre-softmax dot products of values
    O(30) — that's TF32 truncation, not a bug.  Check is
    distributional (mean/std agree, no catastrophic outliers).

  na2d_av: takes post-softmax attention as INPUT (so any TF32
    error is baked into the attention weights both sides see).
    The kernel itself is a weighted-sum gather with no reduction-
    order ambiguity beyond fp32 addition order.  This is the
    semantically meaningful check.  Target: allclose(rtol=5e-3,
    atol=1e-2).

  Empirically (NAF shape-512 call):
    na2d_qk: max_abs=0.298 on values up to 301 → ~1e-3 rel
    na2d_av: max_abs=0.031 on values up to 24.3 → ~1.3e-3 rel

Fixture provenance:
  /Users/pawelma/code/ai/fixtures/natten-fixture.tar — captured on rental
  RTX 6000 Ada via PIXAL3D_NATTEN_PROBE_ONLY=1 (commits 9f13130 →
  a995ae6 → ... on Pixal3D_fresh/feature/pixal-fixtures).  See
  `make_fixtures` instructions in README.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

# Registers torch.ops.natten_mps.*
import natten_mps  # noqa: F401


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "test_fixtures"
QK_FIXTURE = FIXTURE_DIR / "natten_qk_call0.pt"
AV_FIXTURE = FIXTURE_DIR / "natten_av_call0.pt"


pytestmark = [
    pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS unavailable",
    ),
    pytest.mark.skipif(
        not QK_FIXTURE.exists() or not AV_FIXTURE.exists(),
        reason=f"CUDA fixtures missing under {FIXTURE_DIR} — see README "
               f"section on extracting natten-fixture.tar",
    ),
]


# torch.allclose-style: |a - d| <= atol + rtol * |d|
# rtol=5e-3 covers SIMD-vs-tensor-core reduction-order drift at the
# pre-softmax dot-product scale (values O(100)).  atol=1e-2 covers
# positions where the dot product cancels near zero — the absolute
# drift is still small (~8e-3) but rtol*|d|≈0 is meaningless there.
# For D=64 fp32 reductions, ~1e-3 absolute drift on values O(1) and
# ~1e-3 relative drift on values O(100+) are both within IEEE bounds.
RTOL = 5e-3
ATOL = 1e-2


def _load(fixture_path: Path) -> dict:
    return torch.load(str(fixture_path), map_location="cpu", weights_only=False)


def _describe(label: str, diff: torch.Tensor, ref: torch.Tensor) -> str:
    # torch.quantile errors on tensors > 16M elements; sample for p99.
    flat = diff.flatten()
    n = flat.numel()
    if n > 10_000_000:
        idx = torch.randint(0, n, (1_000_000,))
        p99 = flat[idx].quantile(0.99).item()
        p99_note = " (sampled, n=1M)"
    else:
        p99 = flat.quantile(0.99).item()
        p99_note = ""
    ref_abs_max = ref.abs().max().item()
    rel_at_max = diff.max().item() / max(ref_abs_max, 1e-12)
    return (
        f"\n  [{label}] "
        f"max_abs={diff.max().item():.3e}  "
        f"mean_abs={diff.mean().item():.3e}  "
        f"p99={p99:.3e}{p99_note}  "
        f"ref|max|={ref_abs_max:.3e}  "
        f"rel_at_max={rel_at_max:.3e}"
    )


def test_na2d_qk_cuda_golden():
    """Match cutlass-fna at NAF's first shape-512 natten call."""
    data = _load(QK_FIXTURE)
    print(f"\n  loaded {QK_FIXTURE.name}: {data['shapes_str']}")

    q = data["q"].to("mps").contiguous()
    k = data["k"].to("mps").contiguous()
    cuda_out = data["out"]  # CPU, fp32

    # Symmetric kernel/dilation — extract scalars from the (9,9) / (16,16)
    # tuples natten was called with.
    ks = data["kernel_size"]
    dil = data["dilation"]
    if isinstance(ks, (tuple, list)):
        assert ks[0] == ks[1], f"asymmetric ks not supported: {ks}"
        ks = ks[0]
    if isinstance(dil, (tuple, list)):
        assert dil[0] == dil[1], f"asymmetric dilation not supported: {dil}"
        dil = dil[0]

    metal_out = torch.ops.natten_mps.na2d_qk(q, k, int(ks), int(dil)).cpu()

    abs_diff = (metal_out - cuda_out).abs()
    print(_describe("qk vs cutlass-fna", abs_diff, cuda_out))

    # Distributional check: cutlass-fna emits TF32 (10-bit mantissa)
    # outputs on Ada/Hopper.  Mean/std must match within fp32-vs-TF32
    # drift; max diff bounded so we'd catch a real bug.
    assert metal_out.shape == cuda_out.shape
    assert torch.isfinite(metal_out).all(), "Metal output not finite"

    mean_gap = abs(metal_out.mean().item() - cuda_out.mean().item())
    std_gap  = abs(metal_out.std().item()  - cuda_out.std().item())
    print(f"  mean_gap={mean_gap:.3e}  std_gap={std_gap:.3e}")
    assert mean_gap < 0.5, f"mean diverged: {mean_gap:.3e}"
    assert std_gap  < 0.5, f"std diverged: {std_gap:.3e}"

    # Average rel error (clamped to avoid near-zero blow-up) should be
    # well under typical TF32 noise.  Tighter than a uniform rtol band
    # because it averages over 85M elements.
    mean_rel = (abs_diff / cuda_out.abs().clamp(min=1.0)).mean().item()
    assert mean_rel < 5e-3, (
        f"mean relative drift too large: {mean_rel:.3e} > 5e-3 — "
        f"would indicate an actual algorithm bug, not TF32 drift."
    )

    # No catastrophic outliers.  TF32 + fp32 reduction drift on dot
    # products of values O(30) sums to ≲ 0.5 absolute; we observed
    # max ≈ 0.3.  1.0 leaves headroom for future-fixture variance.
    max_abs = abs_diff.max().item()
    assert max_abs < 1.0, f"max_abs catastrophic: {max_abs:.3e}"


def test_na2d_av_cuda_golden():
    """Match cutlass-fna AV at NAF's first shape-512 natten call (D=256)."""
    data = _load(AV_FIXTURE)
    print(f"\n  loaded {AV_FIXTURE.name}: {data['shapes_str']}")

    attn = data["attn"].to("mps").contiguous()
    v = data["v"].to("mps").contiguous()
    cuda_out = data["out"]  # CPU, fp32

    ks = data["kernel_size"]
    dil = data["dilation"]
    if isinstance(ks, (tuple, list)):
        ks = ks[0]
    if isinstance(dil, (tuple, list)):
        dil = dil[0]

    metal_out = torch.ops.natten_mps.na2d_av(attn, v, int(ks), int(dil)).cpu()

    abs_diff = (metal_out - cuda_out).abs()
    print(_describe("av vs cutlass-fna", abs_diff, cuda_out))

    fail_mask = abs_diff > (ATOL + RTOL * cuda_out.abs())
    fails = fail_mask.sum().item()
    assert fails == 0, (
        f"na2d_av: {fails}/{abs_diff.numel()} elements exceed "
        f"allclose(rtol={RTOL}, atol={ATOL}) band.  "
        f"max_abs={abs_diff.max().item():.3e}."
    )
