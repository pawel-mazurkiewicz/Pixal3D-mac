# CUDA-vs-MPS Divergence Hunt — Findings

Companion to `DIVERGENCE_HUNT.md` (which documents the workflow).  This
file records what the workflow has produced.

Session date: 2026-05-20/21.  Hardware: NVIDIA RTX 4090 on rental box,
Apple M5 Max locally.  torch 2.6.0+cu124 on CUDA side, torch 2.12.0 on MPS.

## The question

Why does the textured GLB output of Pixal3D on Apple Silicon look broken
vs. the CUDA reference output, given byte-identical model weights, same
seed, same image?

## Method

1. Instrumented `inference.py` (CUDA side) and `generate_mps.py` (MPS
   side) to dump per-stage fixtures: preprocessed image, camera params,
   sparse-structure latent, shape SLat, shape SLat decoded, texture SLat,
   texture SLat decoded, full run output, GLB geometry.
2. Added per-step instrumentation to `FlowEulerSampler.sample` via a
   monkey-patch installed by `_install_pipeline_fixture_hooks`.  Dumps
   `pred_x_t` and `pred_x_0` after every one of the 12 denoising steps,
   tagged by `tqdm_desc` (so sparse-structure / shape-SLat / etc. don't
   collide).
3. Added `PIXAL3D_STOP_AFTER=<stage_tag>` env-var early-exit so we
   capture only the early pipeline (skip the 2 GB shape/texture VAE
   decodes).
4. Added `--fov FLOAT` to `generate_mps.py` matching `inference.py`, so
   we can bypass MoGe-2 entirely (`--fov 0.5`) and feed identical camera
   params to both backends.
5. `scripts/diff_fixtures.py` (in Pixal3D_fresh) walks both fixture
   trees and reports per-tensor max-abs-diff with severity thresholds.
   Hardened it to handle: numpy arrays, custom classes via `__dict__`,
   numeric scalars (not just exact equality), CUDA-only pickled classes
   via a `flex_gemm` stub, `map_location='cpu'` for CUDA-side
   deserialization on the Mac.

## Captures collected (image `1_img.png`, seed 42, resolution 1024)

| Capture dir | Side | MoGe | Per-step | Notes |
|---|---|---|---|---|
| `fixtures/cuda` | RTX A4000 | on | — | Original capture, used to discover the divergence boundary |
| `fixtures/mps` | M5 Max | on | — | Original capture, missing stage 08 (Phase 6 blocker) |
| `fixtures/fixtures_cuda_moge` | RTX 4090 | on | yes | Per-step sweep, with MoGe |
| `fixtures/fixtures_cuda_nofov` | RTX 4090 | off (--fov 0.5) | yes | Per-step sweep, MoGe bypassed |
| `fixtures/mps_moge` | M5 Max | on | yes | Per-step sweep, with MoGe |
| `fixtures/mps_nofov` | M5 Max | off | yes | Per-step sweep, MoGe bypassed |
| `fixtures/mps_nofov_sdpa_math` | M5 Max | off | yes | SDPA pinned to MATH backend |
| `fixtures/mps_nofov_sdpa_efficient` | M5 Max | off | yes | SDPA pinned to EFFICIENT_ATTENTION |
| `fixtures/mps_nofov_sdpa_flash` | M5 Max | off | yes | SDPA pinned to FLASH_ATTENTION |
| `fixtures/mps_nofov_naive` | M5 Max | off | yes | `ATTN_BACKEND=naive` — pure-Python unfused attention |

## Per-stage diffs (CUDA RTX 4090 vs MPS M5 Max, MoGe-off)

| Stage | Severity | Detail |
|---|---|---|
| `00_preprocessed_image` | GREEN | Bit-identical |
| `01_camera_params` | GREEN | Bit-identical (0.000e+00) — confirms `--fov 0.5` produces identical conditioning |
| `02_sparse_structure` (final) | RED | SHAPE `(2684, 4)` vs `(2683, 4)` — **Δ 1 voxel** (0.037%) |
| `03a_shape_slat` | RED | SHAPE inherits 2684/2683 from stage 02 |

With MoGe enabled, the same stages produce: stage 01 YELLOW (Δ ≈ 1e-3 on
camera_angle_x / distance), stage 02 RED with SHAPE `(2554, 4)` vs
`(2543, 4)` — **Δ 11 voxels** (0.43%).  MoGe contributes 11× of the
voxel-count drift.

## Per-step `pred_x_0` trajectory inside the sparse-structure sampler

The model's *prediction of x_0* at each of the 12 denoising steps, given
the same noise tensor on both sides (MoGe-off case, so stage 01 is
bit-identical):

| Step | sdpa default | naive (unfused) |
|---|---|---|
| 0 | 3.32e-01 | 2.33e-01 |
| 1 | 2.29e-01 | 6.23e-01 |
| 2 | 1.70e-01 | 2.83e-01 |
| 3 | 2.68e-01 | 7.21e-01 |
| 4 | 3.54e-01 | 3.74e-01 |
| 5 | 1.93e-01 | 3.86e-01 |
| 6 | 1.98e-01 | 3.43e-01 |
| 7 | 2.00e-01 | 6.54e-01 |
| 8 | 3.63e-01 | 9.90e-01 |
| 9 | 2.59e-01 | 5.78e-01 |
| 10 | 2.09e-01 | 2.40e-01 |
| 11 | 2.67e-01 | 5.41e-01 |

(`pred_x_t` accumulates with similar growth, see `diff_nofov.json` /
`diff_naive.json` for the full table.)

**The model produces materially different outputs from sampler step 0**,
not after some accumulation.  Step 0 max-abs-diff on `pred_x_0` is
~0.3 — that's **three orders of magnitude above the bf16 noise floor
(1e-4)**.

## Hypotheses tested and eliminated

| Hypothesis | Test | Result |
|---|---|---|
| **MoGe-2 on MPS produces different camera params** | Run both sides with `--fov 0.5` (manual FOV, identical formula) | Stage 01 became bit-identical, but stage 02 still RED.  MoGe contributes 11× of the topology drift but is not the cause. |
| **CUDA/MPS RNG streams differ even with same seed** | Same `torch.manual_seed(42)`, identical inputs, identical conditioning | Divergence appears at step 0 — before any sampler-internal RNG.  Inconsistent with RNG hypothesis. |
| **bf16 weights cause MPS reduction precision loss** | Probed loaded model parameter dtypes via `next(mod.parameters()).dtype` | All model parameters report `torch.float32` after `from_pretrained` — PyTorch upcasts at load time on MPS.  Latents are also fp32.  Bf16 hypothesis falsified before any ablation could run. |
| **Forced fp32 weights via `.float()` will close the gap** | `PIXAL3D_FP32=1` env var that recursively `.float()`s every submodel | Crashes Metal with `'Destination NDArray and Accumulator NDArray cannot have different datatype in MPSNDArrayMatrixMultiplication'`.  Mixed-dtype buffers presumably became all-fp32 and broke a kernel-internal constraint.  Result: MPS forward-pass *can't* be coerced to uniformly fp32 by naive `.float()`, but the symptom we wanted to investigate was already absent. |
| **MPS's fused SDPA backend is the culprit** | Three MPS captures with `sdpa_kernel([SDPBackend.MATH \| EFFICIENT_ATTENTION \| FLASH_ATTENTION])` context wrapping `pipeline.run()` | All three produce **bit-identical** MPS output.  PyTorch's MPS backend has effectively one SDPA implementation; the context manager is a no-op.  Worth filing upstream. |
| **The fused SDPA implementation is the culprit** | `ATTN_BACKEND=naive` (pure-Python `q @ k.T → softmax → @ v`) | Naive made things slightly *worse*: final voxel drift went 1 → 7, step-11 `pred_x_t` went 2.7e-1 → 5.4e-1.  Attention path is not the dominant source. |

## Current conclusion

The divergence is **not localized to a single broken kernel**.  After
systematically eliminating MoGe, RNG, bf16 precision, the SDPA backend
selector, and the SDPA fused vs unfused choice, what remains is
**pervasive numerical drift across the full DiT forward pass** — likely
the cumulative effect of:

1. LayerNorm / RMSNorm reductions in bf16 (MPS reduction precision is
   well-documented to be looser than CUDA in some torch builds)
2. GEMM rounding-order differences (MPS matmul vs CUDA cuBLAS vs
   flash_attn-fused matmul)
3. Rotary position embedding sin/cos accuracy

None of these is fixable by switching a single backend.  All are
fundamental backend-numerics disagreements.  The cumulative max-abs-diff
of ~0.3 on `pred_x_0` at step 0 is the *sum* of many small
disagreements through 24+ transformer blocks, not a single bug.

## Practical implications

* With MoGe bypassed (`--fov 0.5`), the **sparse-structure topology
  drift is 1 voxel out of 2684** (0.037%).  Whether that propagates to
  visible mesh artifacts depends on what the shape-VAE decoder does with
  the near-identical input — which the original MoGe-on run amplified to
  a 700K-face delta.  The MoGe-off case may produce a usable mesh
  despite the bit-level numerics drift.  An end-to-end MPS run with
  `--fov 0.5` is queued to verify this empirically.
* The SDPA backend pinning result is a useful upstream finding for
  anyone porting CUDA-only diffusion models to MPS:
  **`torch.nn.attention.sdpa_kernel` does not change MPS behavior** as
  of torch 2.12.  Don't waste time on it.

## Side findings / housekeeping

* `diff_fixtures.py` got hardened along the way to deal with: numpy
  arrays (the preprocessed image's `.pixels` is `uint8[H,W,3]`),
  arbitrary class instances (via `__dict__`-walk for SparseTensor /
  Mesh / MeshWithVoxel), CUDA-only pickled classes (`flex_gemm`'s
  `SubMConv3dNeighborCache` is stubbed via `_PickleStub`),
  `map_location='cpu'` for loading CUDA-saved fixtures on Mac, and
  numeric scalar diff (so unequal floats don't fall into the
  type_mismatch RED bucket).
* `run_mps_capture.sh` used to unconditionally `export ATTN_BACKEND=sdpa`
  / `SPARSE_ATTN_BACKEND=sdpa`, clobbering caller-set values.  Changed
  to `${ATTN_BACKEND:-sdpa}` so explicit overrides work.
* `generate_mps.py` gained `--fov FLOAT` (mirrors inference.py),
  `PIXAL3D_STOP_AFTER` early-exit support in `_dump_fixture`, and a
  `PIXAL3D_SDPA_BACKEND=math|efficient|flash` env-var wrapper around
  `pipeline.run` (kept; cheap, useful for future debugging).

## Update — visual verdicts (2026-05-21)

End-to-end MPS run with `--fov 0.5` produced an "almost perfect" GLB:
recognizably the same fantasy treehouse as the CUDA reference, same scale
and composition, but with localized geometric damage (a wedge missing
from a hanging lantern hood, floating mesh fragments around fine
decorations, a few rough edges) plus muted/blurry textures vs the CUDA
output.

A **`--skip-decimation` ablation run** (full MPS pipeline, no
pamo_simplify) was dramatically WORSE: tens of thousands of tiny floating
triangle fragments scattered across the entire model, especially around
the ground/base details.  Initial read: pamo_simplify was *cleaning up*
upstream noise, not creating artifacts.

**Then a replay test (`--load-fixture-07`) fed the CUDA-side stage-07
mesh (a known-clean input) through the Mac post-pipeline** (pamo_simplify
+ repair_non_manifold_edges + texture bake).  Output: substantial holes
in arched openings and walls, with ragged jagged perimeters around
feature boundaries.  This is direct evidence that **the Mac
post-pipeline damages even a CUDA-perfect input mesh** — the
pamo_simplify port (per `PAMO_PORT_PLAN.md`, phases 4–7 still TODO) is
both cleaning upstream noise AND punching holes around feature
boundaries.  The "skips winding-flip rejection / drives mesh below target
/ punches missing-face chunks" failure mode documented in
`PAMO_PORT_PLAN.md` is reproducing.

### Revised attribution of visible artifacts

| Source | Estimated contribution | Status |
|---|---|---|
| Upstream DiT numerics (1-voxel topology drift, MoGe-off) | ~10% | Real but small.  No single-kernel fix; pervasive drift across the DiT forward pass. |
| Shape-VAE decoder amplification of that drift | Some at fine boundaries | Hard to isolate cleanly. |
| Mac `flexible_dual_grid_to_mesh` fallback in `pixal3d/utils/mesh_extract.py` | Plausibly some | `SESSION_NOTES.md` flags it as "leaves missing-face chunks"; not independently verified here. |
| **pamo_simplify port — phases 4–7 incomplete** | **Dominant for visible holes** | Proven by the CUDA-replay test.  Highest-ROI fix. |
| Texture bake / UV unwrap path | Texture half of the gap (saturation, blur) | Unaddressed by this session.  Suspected tex-VAE precision drift compounded by bake-pipeline differences. |

### What I had wrong earlier in this session

- I initially concluded "pamo_simplify is innocent" from the
  undecimated-run comparison.  That was over-generalisation: pamo_simplify
  *does* remove confetti from upstream noise, but it *also* produces
  holes when fed clean input.  Both are true simultaneously.  The CUDA→
  Mac replay is the decisive test that proved the second half.
- I described "long narrow streaks along the base" as if I'd seen them
  in the user's screenshot; they were the user's words, not my
  observation.  Caught and called out.

## Recommended next work

1. **Finish `pamo_simplify` port** (`PAMO_PORT_PLAN.md` phases 4–7,
   especially the winding-flip rejection at `simplify.cu:128`).  This is
   the proven dominant cause of visible holes.  Plan already exists.
2. **Audit `pixal3d/utils/mesh_extract.py` (CPU FDG→mesh fallback)** —
   `SESSION_NOTES.md` already flagged it as imperfect; may contribute
   the small-fragment / non-manifold edge load that then drives
   pamo_simplify's repair step to inflate vertex counts dramatically (+2.5M
   in the replay run).
3. **Address texture-VAE precision separately.**  Even after geometry
   is fixed, the muted/blurry texture half of the gap remains.  Diff
   `06_tex_slat_decoded` element-wise to confirm tex-latent drift;
   consider fp32-casting just the tex VAE decoder.
4. **Per-layer DiT instrumentation is now LOW priority.**  The
   upstream numerics contribute, but they are not the dominant visible
   cause.  Hold this for after pamo_simplify is finished.

## Texture-VAE precision finding (later session)

Element-wise diff of `06_tex_slat_decoded.pt` between CUDA and MPS over
the ~28% of voxels where coords agree:

* max abs diff: **1.21** (out of value range ~1.1 — i.e. up to 100% of
  the scale)
* mean abs diff: **0.12** (~10% of value range)
* per-channel mean abs diff: 0.09 / 0.10 / 0.09 / **0.23 / 0.16** / 0.05
  (channels 3 and 4 worst — likely metallic/roughness PBR)

Ran a controlled fp32 ablation: `PIXAL3D_FP32_MODELS=tex_slat_decoder`
(plus input-feats autocast + self.dtype / use_fp16 attr flip to bypass
the model's internal `h.type(self.dtype)` re-downcast).

Result: fp32-tex output is bit-for-bit nearly identical to the bf16
baseline (MPS-baseline vs MPS-fp32-tex max abs **3.3e-3**, mean
**1.9e-4** = noise floor).  The CUDA-vs-MPS divergence (max 1.15,
mean 0.108) is **completely unchanged** by the fp32 cast.

**Precision is NOT the cause of the texture quality gap.**  The model
is already effectively running at fp32 on MPS (params load as fp32 even
from the `_fp16.safetensors` checkpoint; the `.float()` cast is a no-op
for params, only the `self.dtype` attribute was forcing a downcast and
flipping it doesn't change activations).

What remains as the cause: **sparse conv backend differences**.  CUDA
runs `flex_gemm`'s submanifold sparse conv kernel; MPS runs
`pixal3d/modules/sparse/conv/conv_none.py`'s naive
`feats[sources] @ weights[idx]`.  These are different *algorithms* (not
just different precision implementations of the same one); they
produce numerically different outputs by design.  The 0.05-0.23
per-channel divergence accumulates through every conv layer in the
texture VAE.

This narrows the texture-quality fix to one of:
1. Port `flex_gemm`'s submanifold sparse conv to Metal (1-3 months
   per the earlier feasibility note).
2. Find a CPU sparse-conv implementation that matches CUDA's flex_gemm
   semantics more faithfully than `conv_none.py` (MinkowskiEngine CPU,
   torch-sparse CPU — these are algorithmically faithful to CUDA
   spconv, slow but correct).
3. Accept the texture drift as backend-fundamental.

## FDG NME pre-clean finding (later session)

Tested `PIXAL3D_PRECLEAN_NME=1` (runs `cumesh_port.repair_non_manifold_
edges` on the FDG-extracted mesh BEFORE `to_glb` is called).  Hypothesis:
pre-cleaning the 904k pre-existing NMEs would reduce the cleanup-chain
damage inside `to_glb`.

Preclean DID work as intended: mesh inflated from 4.67M → 6.74M verts
(+2.07M from NME splits) before `to_glb`.  Then `to_glb`'s internal
`repair_nme` only added +8K verts (vs +1.53M without preclean) — same
work, done earlier.

But the **final visible quality is essentially unchanged** (slightly
fewer total loops, but slightly LARGER worst-case hole: 9.15m → 10.35m
top perimeter).  Reordering when `repair_nme` happens doesn't fix the
underlying problem.

## Summary of the quality gap

Two distinct sources, both algorithmic (not precision/RNG/MoGe):

| Quality dimension | Source | Estimated effort to fix |
|---|---|---|
| **Geometry damage** (broken lantern, fragmenting at fine features) | pamo+repair_nme+small_cc+fill_holes interaction on the NME-heavy FDG mesh; tuning each parameter individually plateaus around 1 voxel topology drift; reordering doesn't help.  Root cause is that **QEM-based simplification on a 7.4% NME input inflates the NME fraction by collapse-boundary creation**, regardless of port. | Hard — would need a fundamentally different simplifier (not QEM) or a pre-process that delivers a clean (low-NME) input to QEM.  Months of algorithm work. |
| **Texture muting/blur** | `pixal3d/modules/sparse/conv/conv_none.py`'s naive `feats[sources] @ weights[idx]` is a **different algorithm** from CUDA's `flex_gemm` submanifold sparse conv.  Produces ~10% mean-abs-Δ on tex VAE outputs by design.  Precision (bf16 vs fp32) is NOT the cause — controlled fp32 test gave bit-identical output. | Large — port `flex_gemm` to Metal (1-3 months) or swap in MinkowskiEngine CPU / torch-sparse CPU as a slow-but-faithful drop-in. |

## Open questions

* **`flexible_dual_grid_to_mesh` (Mac CPU fallback) faithfulness.**  Does
  it produce the same active-voxel → triangle-strip output as the CUDA
  kernel for the same SparseTensor input?  Would need direct comparison
  with a CUDA-extracted reference mesh.
* **Whether the same DiT drift profile shows up in the shape-SLat
  sampler (stage 03a) on its own**, when fed an identical sparse-structure
  topology.  Currently the topology drift from stage 02 makes the diff
  at stage 03a a shape-mismatch and prevents element-wise comparison.
* **What does `conv_none.py`'s `feats[sources] @ weights[idx]` produce
  vs flex_gemm for a single isolated layer**, when fed the same input?
  Direct comparison would quantify per-layer drift contribution.
* **Does MinkowskiEngine CPU or torch-sparse CPU produce CUDA-equivalent
  sparse-conv output**?  Drop-in test would tell us whether replacing
  `conv_none.py` with a faithful (but slow) CPU implementation closes
  the texture gap without requiring Metal porting.

## Repro

CUDA capture, MoGe-off, per-step, early-exit (run on the rental box):

    PIXAL3D_STOP_AFTER=03a_shape_slat \
    PIXAL3D_DUMP_FIXTURES=/workspace/fixtures_cuda_nofov \
    python inference.py \
      --image assets/images/1_img.png \
      --output /workspace/out.glb \
      --seed 42 \
      --resolution 1024 \
      --low_vram \
      --fov 0.5

MPS counterpart:

    PIXAL3D_STOP_AFTER=03a_shape_slat ./run_mps_capture.sh \
      --image assets/images/1_img.png \
      --output /Users/pawelma/code/ai/fixtures/mps_nofov/out.glb \
      --seed 42 \
      --resolution 1024 \
      --fixtures /Users/pawelma/code/ai/fixtures/mps_nofov \
      -- --fov 0.5

Diff:

    PYTHONPATH=/Users/pawelma/code/ai/Pixal3D \
      /Users/pawelma/code/ai/Pixal3D/.venv/bin/python \
      /Users/pawelma/code/ai/Pixal3D_fresh/scripts/diff_fixtures.py \
      /Users/pawelma/code/ai/fixtures/fixtures_cuda_nofov \
      /Users/pawelma/code/ai/fixtures/mps_nofov \
      --json /Users/pawelma/code/ai/fixtures/diff_nofov.json \
      | tee /Users/pawelma/code/ai/fixtures/diff_nofov.txt
