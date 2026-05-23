# Session 10 Findings — MPS `grid_sample` exonerated; natten-mps porting kicked off

**Date:** 2026-05-22
**Branch (Mac):** `feat/apple-silicon-port`
**Branch (rental staging):** `feature/pixal-fixtures` (Pixal3D_fresh repo)
**New repo:** `/Users/pawelma/code/ai/natten-mps/` (greenfield, MIT)

---

## Executive summary

Session 9 left an open question: of the residual ~2.5e-2 RED on z_proj
(01b/c/d) after `PIXAL3D_NAF_DEVICE=cpu`, is it (a) MPS `F.grid_sample` HR on
the upsampled feature map, or (b) torch CPU↔CUDA precision drift inside NAF
itself? **Today's experiment answered (b).** Moving the HR `proj_grid` call
off MPS to CPU shaved <2% off the residual. The remaining ~2.5e-2 RED moves
with NAF, not with `grid_sample`.

That answer killed the cheap fix paths and the work pivoted to **porting
natten's neighborhood-attention kernels to Metal** as a science / community
contribution. Substantial research was done; the new `natten-mps` package
exists with passing smoke tests (kernels execute on MPS, produce finite
output, custom-op bindings work, natten dispatch shim auto-installs).
Correctness validation is the next session's first job.

**A second discovery dropped a bomb on the session-9 baseline:** natten 0.21
(currently installed on Mac) has **no MPS backend at all** — `na2d` hard-fails
on MPS, and our "NAF on MPS" baseline in `mps_fdg/` must have been from an
older natten with the legacy split-path that happened to limp along on MPS.
The 2.5e-2 Δ is the difference between natten's `flex-fna` (CPU) and
`cutlass-fna` (CUDA) — two unrelated implementations, both blessed by
upstream. Closing that gap requires writing a third implementation
(`metal-fna`) on Apple Silicon.

---

## Experiment results

### Experiment 5: HR proj_grid on CPU

Same setup as session 9 Experiment 3 (`--fov 0.6061`, `PIXAL3D_NAF_DEVICE=cpu`,
`PIXAL3D_STOP_AFTER=01d_image_cond_tex_1024`), now with an additional
`PIXAL3D_PROJ_GRID_DEVICE=cpu` knob (added to
`pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py` —
uncommitted, diagnostic).

| Stage | Pre-fix | Exp 3 (NAF=cpu) | Exp 5 (NAF=cpu + proj_grid=cpu) | Δ between Exp 3 and Exp 5 |
|---|---|---|---|---|
| `01a` (no NAF) | RED 1.4e-2 | YELLOW 2.8e-4 | YELLOW **2.83e-4** | <1% |
| `01b` (NAF) | RED 1.3e-1 | RED 2.8e-2 | RED **2.77e-2** | <2% |
| `01c` (NAF) | RED 9.3e-2 | RED 2.6e-2 | RED **2.56e-2** | <2% |
| `01d` (NAF) | RED 1.2e-1 | RED 2.4e-2 | RED **2.40e-2** | <2% |

**Verdict:** MPS `F.grid_sample` HR is exonerated. The residual moves with
NAF, not with `grid_sample`.

`01a` is unaffected as expected — that path has `use_naf_upsample=False`,
so neither knob touches it.

### Discovery: NAF on MPS doesn't actually work with current natten

Direct probe:

```python
import torch, natten
q = torch.randn(1, 16, 16, 4, 64, device='mps')
out = natten.na2d(q, q, q, kernel_size=9, dilation=1, stride=1)
# -> NotImplementedError: NATTEN could not find a suitable backend
```

But on CPU:
```python
qc = q.cpu()
out = natten.na2d(qc, qc, qc, kernel_size=9, dilation=1, stride=1)
# OK, dispatches to flex-fna (torch.nn.attention.flex_attention)
```

natten 0.21's available backends are `cutlass-fna` / `hopper-fna` /
`blackwell-fna` (all CUDA-only) and `flex-fna` (CPU-only). **There is no MPS
backend in upstream natten today.** NAF's CrossAttention unconditionally
passes `backend="cutlass-fna"` (`attentions.py:72`), which would hard-fail
on Mac, but the dispatcher falls back to `flex-fna` when `backend=None`.

This means the original session-9 baseline `mps_fdg/` (where we measured
z_proj Δ=1.3e-1 with "NAF on MPS") **could not have been NAF actually
running on MPS via natten 0.21**. Either:
- An earlier natten was installed at the time with the legacy
  `na2d_qk`/`na2d_av` split path, which torch's MPS happened to ship
  enough kernels to limp through (incorrectly), OR
- Something else in the path was masking the failure.

Either way, the 2.5e-2 residual we've been chasing is the
flex-fna(CPU) ↔ cutlass-fna(CUDA) algorithmic gap — not torch precision
drift. The two implementations differ in scaling location, boundary masking
strategy, and tile/reduction order. Bridging them requires writing a
faithful Metal implementation of cutlass-fna's algorithm.

---

## Research summary

Two parallel research agents produced ~3500 words of focused briefs. Key
findings:

### PyTorch MPS custom-kernel APIs (PyTorch 2.5+/2.12)

- **`torch.mps.compile_shader(inline_msl_source)`** is the official runtime
  JIT compile entry. Returns a `_Library` whose `kernel void <name>`
  declarations become Python attributes. Marshals torch tensors → MTLBuffer,
  CPU scalars → `constant T&`, automatically.
- **`torch.library.custom_op` + `register_kernel("mps")` + `register_fake`**
  is the right binding pattern for an out-of-tree package. Pure-Python wheel,
  no Xcode toolchain on the end user's machine.
- The C++/ObjC++ + `.metallib` pattern (mtlgemm/mtlmesh) works too but is
  heavier; not needed for two stateless kernels.
- **No community natten MPS port exists** — Issue
  [SHI-Labs/NATTEN#102](https://github.com/SHI-Labs/NATTEN/issues/102) (2024
  request for MPS) closed without implementation. We're writing this from
  scratch.

### natten algorithmic spec

- **CPU naive in natten 0.17.5** is the cleanest reference — 70 lines of
  plain C++ at
  `csrc/include/natten/cpu/naive/pointwise_neighborhood_2d.hpp:166-272`.
  The 0.21 mainline removed it. We anchor algorithm spec there; install
  natten==0.17.5 in a side venv for goldens.
- **Boundary handling = edge-shift-clamp** (closed-form `get_window_start`,
  no padding/mirroring/masking for non-causal). Window is always exactly
  `kernel_size` tokens wide; corner queries don't center.
- **Modern natten 0.21 fused `na2d`** uses Flash-style online softmax with
  fp32 accumulator inside the kernel, heads-last layout `[B, X, Y, H, D]`.
  Legacy split path used heads-second `[B, H, X, Y, D]` and applied scale +
  softmax in Python between QK and AV calls.

### M5 Neural Accelerators / Metal 4

- 32-wide × 4-way FP16 dot-product datapath, 128 FMA/cycle/partition.
- FP16 inputs with either FP16 or FP32 accumulator, same throughput.
- Optimal tile **32×32+**, M and N must be compile-time constants.
- Not exposed via `simdgroup_matrix` on M5 — only through Metal Performance
  Primitives (MPP) `tensor_ops::matmul2d` + MTLTensor APIs.
- **v1 target should be `metal::dot` naïve kernel that runs M1→M5.** v2
  cooperative threadgroup tiling. v3 MPP `matmul2d` M5 specialization.

### Bit-faithful CUDA↔Metal is impossible by IEEE rounding semantics

- `metal::fma` and `metal::dot` are IEEE-rounded but reduction order is
  implementation-defined and differs from CUTLASS tensor-core ordering.
- **Target: ≤ 1e-4 max-abs (fp32), ≤ 5e-3 rel-tol (fp16)** — well within
  visible-quality threshold.

---

## What this session shipped

### Mac (`Pixal3D`, `feat/apple-silicon-port`)

| Commit | Description |
|---|---|
| *(uncommitted)* | `PIXAL3D_PROJ_GRID_DEVICE` knob in `image_conditioned_proj.py` — diagnostic only, exonerated MPS `grid_sample`, can be reverted |

### Rental staging (`Pixal3D_fresh`, `feature/pixal-fixtures`)

| Edit | Description |
|---|---|
| `inference.py`: natten capture wrapper | Monkey-patches `natten.na2d` to dump `(q, k, v, ks, dilation, scale, out)` per call. 3 captures per run (01b/01c/01d). |
| `inference.py`: `_dump_fixture` filter | When `PIXAL3D_NATTEN_PROBE_ONLY=1`, all non-natten fixture names are skipped. Implicit `PIXAL3D_STOP_AFTER=01b_natten_shape_512` if not already set. **Total rental download ~hundreds of MB instead of ~7 GB.** |
| `scripts/run_cuda_capture.sh` | Doc comment for the new probe-only knob. |

### New repo: `/Users/pawelma/code/ai/natten-mps/` (greenfield)

```
natten-mps/
  pyproject.toml      # MIT, pure-Python wheel, deps: torch>=2.5
  README.md
  LICENSE
  natten_mps/
    __init__.py       # auto-installs shim on import
    _compile.py       # lazy torch.mps.compile_shader singleton
    _ops.py           # @custom_op + register_kernel("mps") + register_fake
    _shim.py          # monkey-patches natten.na2d for MPS tensors
    shaders/
      shared.metal    # na_window_start (edge-shift-clamp closed form)
      na2d_qk.metal   # naïve correct: 1 thread / (b·H, x, y); fp32 accum
      na2d_av.metal   # same parallelization; weighted-sum gather
  tests/
    test_smoke.py     # 5 tests, all passing
```

**Smoke tests pass on Mac MPS:** kernels compile, dispatch via PyTorch
custom ops, touch real MTLBuffers, produce correctly-shaped finite output,
shim installs idempotently.

**Correctness untested** — that's the next session's first job, validating
against natten 0.17.5 CPU naive on toy inputs.

---

## Decisions locked this session

| Question | Decision |
|---|---|
| Package layout | Standalone `natten-mps` pure-Python wheel (community contribution path) |
| v1 kernel scope | Split path (`na2d_qk` + `na2d_av`) matching 0.17.5 CPU naive — easier to validate. v2 fused `na2d` later. |
| Validation target | CUDA `cutlass-fna` goldens at NAF's actual shapes (B=1, H=4, X=Y∈{144,256,…}, D=64, ks=9, dilation∈{1,2}). Target ≤1e-4 abs. 0.17.5 CPU naive is the algorithm anchor for windowing/boundary correctness. |
| Upstream shape | Standalone first (fast iteration), then PR a `metal-fna` backend to SHI-Labs/NATTEN |
| Bandwidth | `PIXAL3D_NATTEN_PROBE_ONLY=1` mode — rental run dumps only natten captures (~hundreds of MB) |

---

## Next-session priorities (in order)

### 1. CPU naive parity (~half day)

- Spin up `.venv-natten-cpu` with `pip install natten==0.17.5`
- Write `natten-mps/tests/test_parity.py`: random `(q, k, v)` tensors at
  small shapes (B=1, H=2, X=Y∈{16,17}, D=64, ks=9, dilation∈{1,2}). Compute
  Mac MPS output via our kernels; CPU output via natten 0.17.5
  `na2d_qk`/`na2d_av`. Diff at fp32.
- Target: ≤ 1e-5 abs (CPU naive is the algorithmic ground truth)
- Fix any boundary / indexing / dilation bugs that surface.

### 2. CUDA cutlass-fna goldens (~rental cycle)

- Kick off `PIXAL3D_NATTEN_PROBE_ONLY=1 ./scripts/run_cuda_capture.sh` on
  rental
- Pull `01b_natten_shape_512.pt` back to Mac
- Write `tests/test_cuda_golden.py`: load captured CUDA tensors, compute Mac
  MPS output via our kernels, diff vs CUDA `out`
- Target: ≤ 1e-4 abs
- If gap > 1e-4: investigate algorithm differences (scale location, accumulator
  precision, layout)

### 3. Wire into Pixal3D end-to-end (~half day)

- Add `import natten_mps` (env-gated by `PIXAL3D_NATTEN_MPS_ENABLE=1`) at top
  of `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`
- Re-run isolation experiment with NAF on MPS native (no
  `PIXAL3D_NAF_DEVICE=cpu`)
- Target: z_proj on 01b/c/d drops from 2.5e-2 RED → ≤1e-4 GREEN
- Full Mac pipeline run; eyeball mesh visible-quality

### 4. v2 fused `na2d_fused.metal` (~1.5 days)

- Flash-style online softmax in MSL, single kernel
- Matches modern natten 0.21 `na2d` semantics — no materialized
  `[B,H,X,Y,81]` attention tensor
- Also fixes the `backend="cutlass-fna"` hard-fail in NAF for any
  Mac-deployed model

### 5. M5 specialization via MPP `tensor_ops::matmul2d` (~1 day, optional)

- Cooperative 32-query tile, leverage M5 Neural Accelerators
- M-tier detection gate (M1-M4 use the simdgroup path, M5+ uses MPP)

### 6. Upstream PR to SHI-Labs/NATTEN (~1 day)

- Add `metal-fna` to `choose_backend` dispatch
- Examples, docs, CI on a Mac runner

---

## Heuristics learned this session

- **Two implementations of the same op can differ by 100× more than IEEE
  rounding allows.** flex-fna and cutlass-fna both compute "fused
  neighborhood attention" but produce 2.5e-2 Δ on z_proj — not because of
  precision drift but because they made different algorithmic choices
  (scaling location, mask handling, tile order). Don't chase precision when
  the answer is "different algorithm."
- **A failing import in a try/except can mask a complete absence of a code
  path.** NAF's `try: from natten.functional import na2d_qk` fell through to
  the modern path silently, and the modern path hard-fails on MPS — but
  produced output once via an older natten install. Always verify which
  branch fires at runtime.
- **Pure-Python `torch.mps.compile_shader` is dramatically lower-friction
  than `.metallib` + pybind11** for stateless GPU kernels. Two `.metal`
  files + 60 lines of Python = compiling Metal shaders dispatched as
  PyTorch custom ops. No setup.py build_ext.
- **`PIXAL3D_STOP_AFTER` generalizes well** — once any new instrumentation
  uses `_dump_fixture`, it inherits the early-exit + filter machinery for
  free. The natten capture wrapper inherits both with zero extra code.

---

## Open questions for future sessions

1. What is NAF's `naf_target_size` for each image_cond stage exactly? The
   captured 01b/01c/01d sizes will tell us, but a priori we don't know if
   shape_1024 means a 1024×1024 feature map (huge: q/k/v/out ~4 GB total per
   call) or something smaller. May force us to capture only at 01b.
2. Does NAF need ks > 9 anywhere in its pipeline? Currently coded as
   `kernel_size=9` everywhere; the Metal kernel doesn't bake this in but
   should be parametric-validated.
3. Once we have a correct Mac Metal port matching cutlass-fna, what's the
   visible-quality gap to mesh produced from a CUDA-cutlass-fna run? Maybe
   ~0 (problem solved), maybe more sessions still ahead for downstream DiT
   numerics.
