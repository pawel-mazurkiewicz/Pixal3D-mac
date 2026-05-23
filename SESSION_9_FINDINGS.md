# Session 9 Findings — DINOv3 z_proj Localized as MPS↔CUDA Divergence Source

**Date:** 2026-05-22
**Branch (Mac):** `feat/apple-silicon-port`
**Branch (rental staging):** `feature/pixal-fixtures` (Pixal3D_fresh repo)
**Image / seed used throughout:** `assets/images/1_img.png`, seed=42, resolution=1024

---

## Executive summary

The visible-quality gap that's been the parent problem since sessions 5/6 has been
narrowed to **one specific component**: `DinoV3ProjFeatureExtractor.z_proj` —
the per-token projected DINOv3 features that condition every DiT in the pipeline.

Two contributors are now empirically isolated:

1. **NAF feature upsampler on MPS** — *dominant*. Moving NAF onto CPU cuts the
   z_proj divergence by ~5× on the three NAF-bearing image-cond stages
   (01b/01c/01d).
2. **`proj_grid` + camera-coord drift from MoGe-2** — secondary but real.
   Pinning `--fov 0.6061` (matching CUDA's MoGe-2 output exactly) plus moving
   NAF off MPS drops the no-NAF stage (01a) from Δ=1.4e-2 RED to Δ=2.8e-4
   YELLOW (~50× drop).

A **residual ~2.5e-2 RED remains** in 01b/01c/01d even after both fixes —
likely a CPU-torch-vs-CUDA-torch NAF subtle difference, or MPS `F.grid_sample`
over the HR feature map. Next dig.

Everything else — DINOv3 backbone, FDG VAE decoder, FDG extractor, cleanup
chain — is **GREEN or empirically exonerated**.

---

## What this rules out (confirmed innocent)

| Suspect | Evidence | Session |
|---|---|---|
| Mac cleanup chain | stage-08 replay produces sealed shell from CUDA inputs | 7 |
| FDG mesh extractor (`mesh_extract.py`) | bit-identical to upstream port on Mac inputs (session 8); produces same vertex count as CUDA on CUDA's exact fdg_* inputs (session 9) | 8 + 9 |
| DINOv3 ViT backbone | `z_global` (slice of DINOv3 output `z`) is GREEN max Δ=2.5e-5 — backbone is essentially identical MPS↔CUDA | 9 |
| `metal-flash-attention` integration | MPS sdpa already at fp32 precision floor; attention isn't the bottleneck | 7 |
| MLX-native port | produces visibly worse output than PyTorch/MPS | 7 |
| Pure RNG drift | rng_state CPU snapshot Δ=0 at stage-02 entry (sampler entrypoint) | 9 |

---

## Session arc (compressed)

1. **Started** at session-8's next step: "audit & fix the FDG extractor."
2. **Opus audit** of upstream `flexible_dual_grid.cpp` vs our port:
   *no algorithmic gap found, no Metal port needed*.
3. **Vendored upstream** `flexible_dual_grid_to_mesh` into
   `pixal3d/utils/mesh_extract.py` with a dict hashmap shim replacing
   `_lookup_sparse_voxels`. (Commit `3cf8aeb`.)
4. **Decisive A/B** on Mac saved checkpoint with the OLD extractor's output:
   vertices max-abs-diff = 0.0, 0 face mismatches. **Extractor is innocent.**
5. **Instrumented rental staging repo** (`Pixal3D_fresh`) to capture
   `fdg_*`, `fdg_h_feats`, DINOv3 image_cond outputs, and per-sampler RNG
   state. Mirrored on Mac (`generate_mps.py`).
6. **Wrote `scripts/setup_rental.sh`** — one-shot env setup with pinned
   natten 0.17.5 + flash_attn 2.8.3 wheels for older-arch GPUs.
7. **First rental fixture** missing `fdg_*` (rental clone predated push).
8. **Second rental fixture** had `image_cond` + RNG but STILL missing `fdg_*`
   — bug: `MeshWithVoxel(m.vertices, m.faces, ...)` wrap discards extra
   attrs. Patched (`c538883` fresh, `b7912dc` Mac).
9. **Third rental fixture** complete with everything.
10. **Mac dump** with full instrumentation (`mps_fdg/`).
11. **First MPS↔CUDA diff via `diff_fixtures.py`**: localized first-RED stage
    to DINOv3 `z_proj` on every image_cond model.
12. **Audited `DinoV3ProjFeatureExtractor`**: traced divergence to `proj_grid`
    + NAF, NOT the DINOv3 backbone (z_global is GREEN, z_proj is RED, same
    underlying tensor `z`).
13. **Isolation experiment** (`PIXAL3D_NAF_DEVICE=cpu`, `--fov 0.6061`,
    `PIXAL3D_STOP_AFTER=01d_image_cond_tex_1024`) — both contributors
    confirmed; residual RED remains.

---

## Key experiments + results

### Experiment 1: Mac vendored extractor on CUDA's stage-07 `fdg_*` inputs

| | Verts | Faces |
|---|---|---|
| Mac extractor output (this session) | 4,185,441 | 9,288,212 |
| CUDA's stored mesh (post-fill_holes) | 4,194,633 | 9,344,284 |
| Δ | 9,192 | 56,072 |

The 9,192-vert / 56,072-face gap is entirely from `m.fill_holes()` running
*after* `flexible_dual_grid_to_mesh` in `pipeline.decode()`. The extractor
itself produces a vertex per voxel, matching CUDA exactly.

**Verdict:** Mac extractor is faithful to CUDA on identical inputs. Confirmed
from both sides (Mac inputs in session 8, CUDA inputs now).

### Experiment 2: First MPS↔CUDA stage-by-stage diff (Mac baseline)

`Pixal3D_fresh/scripts/diff_fixtures.py mps_fdg/ vs fixtures_cuda/` — first
RED stage is `01a_image_cond_ss.pt`:

| Stage | Worst | Notes |
|---|---|---|
| `00` preprocessed image | GREEN | Identical input ✓ |
| `01` camera (MoGe-2) | YELLOW | `camera_angle_x` Δ=4.4e-4, `distance` Δ=1.2e-3 |
| `01a` image_cond_ss (no NAF) | **RED** | `z_global` Δ=1.4e-5 GREEN; `z_proj` Δ=**1.4e-2** |
| `01b` image_cond_shape_512 | **RED** | `z_global` GREEN; `z_proj` Δ=**1.3e-1** |
| `01c` image_cond_shape_1024 | **RED** | `z_global` GREEN; `z_proj` Δ=**9.3e-2** |
| `01d` image_cond_tex_1024 | **RED** | `z_global` GREEN; `z_proj` Δ=**1.2e-1** |
| `02` sparse_structure | **RED** | step-0 pred_x_0 Δ=0.18; final SHAPE (2550,4) vs (2543,4) |
| `03a/b`, `04`, `05`, `07` | **RED** | All cascade — different voxel counts inherited from 02 |
| `02_..._rng_state.cpu` | **GREEN** Δ=0 | RNG agreement at sampler entry |

Mac voxel count at stage-07: 4,492,536. CUDA: 4,185,441. **MPS produces ~7%
*more* voxels** with this run (opposite of `1_img_fdg_cap.npz`'s ~10%
*fewer*) — confirming run-to-run nondeterminism on MPS DiT.

### Experiment 3: Isolation (`--fov 0.6061` + `PIXAL3D_NAF_DEVICE=cpu`)

Re-running Mac with both fixes, stopping after `01d_image_cond_tex_1024`:

| Stage | Original Δ z_proj | Isolation Δ z_proj | Drop |
|---|---|---|---|
| `01` camera | YELLOW (1.2e-3) | **GREEN** (2.5e-5) | 50× |
| **`01a` (no NAF)** | RED 1.4e-2 | **YELLOW 2.8e-4** | **~50×** |
| **`01b` (with NAF)** | RED 1.3e-1 | RED 2.8e-2 | ~5× |
| **`01c` (with NAF)** | RED 9.3e-2 | RED 2.6e-2 | ~4× |
| **`01d` (with NAF)** | RED 1.2e-1 | RED 2.4e-2 | ~5× |

**Interpretation:**

- **`--fov 0.6061` + neutralized proj_grid:** 01a (no-NAF path) drops from
  Δ=1.4e-2 to Δ=2.8e-4, i.e. ~50×. The pure proj_grid + camera-coord drift
  contribution is essentially solved.
- **`PIXAL3D_NAF_DEVICE=cpu` (NAF off MPS):** 01b/c/d drop from 1e-1 to
  2.5e-2, i.e. ~5×. NAF on MPS was the dominant share of NAF-bearing stages.
- **Residual ~2.5e-2 RED in 01b/c/d**: comes from somewhere new. Candidates:
  - CPU-torch NAF vs CUDA-torch NAF subtle precision difference
  - MPS `F.grid_sample` on the (now CPU-computed) HR feature map after
    ferrying back to MPS for `proj_grid` sampling

### Experiment 4: `fdg_h_feats` threshold-flip analysis

`fdg_intersected = h.feats[..., 3:6] > 0` — a boolean derived from the
logits in the mid-decoder features.

| Threshold | Entries within |%|
|---|---|---|
| `|logit| < 1e-3` | 1,599 | 0.013% |
| `|logit| < 1e-2` | 16,405 | 0.131% |
| `|logit| < 1e-1` | 163,750 | 1.304% |
| `|logit| < 0.5` | 757,602 | 6.034% |

Logit range: [-121.8, 40.3]. Mean abs: 12.7.

**Verdict:** threshold-flip from precision drift is a *small* contributor —
only 0.13% of entries are within ±1e-2 of zero, and the upstream `z_proj` Δ
(up to 0.13) is ~10× larger than this danger zone width. The dominant
divergence chain is `z_proj` → DiT outputs → different sparse structure →
different voxel set, *not* a boundary-flip in the FDG decoder threshold.

---

## Layer-by-layer responsibility breakdown

Inside `DinoV3ProjFeatureExtractor.forward()` at
`pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:464+`:

```
image  ──(GREEN, identical)──>  extract_features  ──>  z (DINOv3 output)
                                                       │
                                  ┌────────────────────┴────────────────────┐
                                  ▼                                         ▼
                             z[:, 0:5]                                  z[:, 5:]
                          (clstoken + 4 reg)                       (patch tokens)
                                  │                                         │
                                  ▼                                         ▼
                             z_global ─> GREEN Δ≤1.4e-5         z_patchtokens_spatial
                                                                            │
                                                          ┌─────────────────┴──────────────────┐
                                                          ▼                                    ▼
                                              proj_grid (LR)                       (if NAF) naf_model
                                                          │                                    │
                                                          │                                    ▼
                                                          │                              hr_features
                                                          │                                    │
                                                          │                                    ▼
                                                          │                            proj_grid (HR)
                                                          │                                    │
                                                          ▼                                    ▼
                                                     z_proj_lr                            z_proj_hr
                                                          │                                    │
                                                          └──── cat(dim=-1) ───────────────────┘
                                                                  │
                                                                  ▼
                                                              z_proj ──> RED Δ up to 1.3e-1
```

Contributions (after Experiment 3):

| Path | Pre-fix Δ | Post-fix Δ | Status |
|---|---|---|---|
| DINOv3 backbone | — | Δ ≤ 2.5e-5 | GREEN — innocent |
| `z[:, ...]` slicing | — | bit-faithful | GREEN — innocent |
| `proj_grid` LR + camera drift | 1.4e-2 | 2.8e-4 | YELLOW — solved by `--fov` |
| `naf_model` on MPS | dominant | (CPU now) | RED → solved by `PIXAL3D_NAF_DEVICE=cpu` |
| `naf_model` CPU vs CUDA precision | — | ~? | candidate for residual 2.5e-2 |
| `F.grid_sample` HR on MPS | — | ~? | candidate for residual 2.5e-2 |

---

## What we built this session

### Mac (`Pixal3D`, `feat/apple-silicon-port`)

| Commit | Description |
|---|---|
| `3cf8aeb` | Vendor upstream `flexible_dual_grid_to_mesh` with dict hashmap shim; extractor empirically exonerated |
| `9744002` | Mirror fresh-repo capture instrumentation (fdg_h_feats + DINOv3 image_cond hooks + per-sampler RNG state) into `generate_mps.py` |
| `b7912dc` | Extend MeshWithVoxel attr-preservation loop with `fdg_h_feats` |
| *(uncommitted)* | `DinoV3ProjFeatureExtractor` `PIXAL3D_NAF_DEVICE` override + cross-device tensor ferry — diagnostic only, may or may not land |

### Rental staging (`Pixal3D_fresh`, `feature/pixal-fixtures`)

| Commit | Description |
|---|---|
| `f3129fd` | Capture pre-extraction FDG tensors at stage-07 |
| `28c44c8` | `scripts/setup_rental.sh` — one-shot rental env setup with pinned older-arch wheels (natten 0.17.5+torch260cu124, flash_attn 2.8.3+cu124torch2.6) |
| `f1d05de` | Capture `h.feats` (mid-decoder), DINOv3 cond outputs, per-sampler RNG state |
| `c538883` | Preserve `fdg_*` + `fdg_h_feats` across `MeshWithVoxel` wrap in pipeline |

### New instrumentation surface (captured per run when `PIXAL3D_DUMP_FIXTURES` is set)

```
00_metadata.json
00_preprocessed_image.pt
01_camera_params.pt
01a_image_cond_ss.pt              ← NEW: z_global + z_proj per DiT stage
01b_image_cond_shape_512.pt       ← NEW
01c_image_cond_shape_1024.pt      ← NEW
01d_image_cond_tex_1024.pt        ← NEW
02_sparse_structure_rng_state.pt  ← NEW: CPU + CUDA + MPS RNG snapshots
02_sparse_structure_stepXX.pt     (12 steps; pre-existed)
02_sparse_structure.pt            (pre-existed)
03a_shape_slat_rng_state.pt       ← NEW
03a_shape_slat_stepXX.pt          (12 steps; pre-existed)
03a_shape_slat.pt
03b_shape_slat_cascade_rng_state.pt   ← NEW
03b_shape_slat_cascade_stepXX.pt  (12 steps)
04_shape_slat_decoded.pt          (the FDG VAE decoder INPUT SparseTensor)
05_tex_slat_rng_state.pt          ← NEW
05_tex_slat_stepXX.pt             (12 steps)
05_tex_slat.pt
06_tex_slat_decoded.pt
07_run_output.pt                  (now has fdg_coords/fdg_dual_vertices/
                                  fdg_intersected/fdg_split_weight/fdg_h_feats)
08_to_glb_geometry.pt
```

### Bonus utility

`PIXAL3D_STOP_AFTER=<fixture_name>` env var (pre-existing, used this session):
fires after the named dump completes and exits cleanly. Lets the divergence
hunt iterate in seconds instead of running the full ~25 min Mac pipeline.

---

## Fixture inventory (kept)

| Path | What | Size |
|---|---|---|
| `/Users/pawelma/code/ai/fixtures/fixtures_cuda/` | Gold-standard CUDA reference (full instrumented capture) | ~7 GB |
| `/Users/pawelma/code/ai/fixtures/fixtures_cuda.tar` | Tarball backup of above | 7.5 GB |
| `/Users/pawelma/code/ai/fixtures/mps_fdg/` | Mac baseline (default settings, full pipeline, has `fdg_*` but no `fdg_h_feats`) | 7.8 GB |
| `/Users/pawelma/code/ai/fixtures/mps_isolate_naf_cpu/` | Tonight's isolation experiment (`--fov 0.6061 PIXAL3D_NAF_DEVICE=cpu PIXAL3D_STOP_AFTER=01d_image_cond_tex_1024`) | ~5 GB |
| `/Users/pawelma/code/ai/fixtures/sieve_test_stage08/` | Session-7 cleanup-only test outputs (`cleanup_only.glb`, `out.glb`) — evidence that cleanup chain is innocent | ~40 MB |

(All older `mps_*`, `cuda_*`, `mlx_*`, `fixtures*_moge*`, `cuda_fdg.tar`,
`captures.tar*` directories were retired-hypothesis investigations; safely
deletable — see session-9 chat for the full list.)

---

## Next steps (in priority order)

### 1. Isolate the residual 2.5e-2 RED in 01b/c/d

Two candidates remain after Experiment 3:

- **(a)** CPU-torch NAF vs CUDA-torch NAF subtle precision difference.
- **(b)** `F.grid_sample` on MPS over the HR feature map (after NAF on CPU
  produces hr_features and we ferry back to MPS for `proj_grid` HR sampling).

Cheap distinguishing experiment: force `proj_grid` HR sampling onto CPU
(add a knob like `PIXAL3D_PROJ_GRID_DEVICE=cpu` that ferries the hr_features
+ sampling op to CPU). If the residual Δ drops near zero → MPS `grid_sample`
is the cause. If still ~2.5e-2 → NAF CPU↔CUDA is the cause.

ETA: ~10 min including code patch and a `STOP_AFTER=01d` run.

### 2. Decide on the NAF fix

If NAF-on-MPS is confirmed the dominant remaining problem, choices:

- **(A)** Always run NAF on CPU on Mac. Quality wins, perf cost is small
  (NAF is ~2.5 MB; ~few sec per stage on CPU).
- **(B)** Investigate which specific MPS op in NAF (Conv2d? GroupNorm? GELU?)
  is the source. Could be fp16 contamination, conv-on-MPS precision quirk,
  or activation-function approximation. Audit `valeoai/NAF` source.
- **(C)** Port NAF to a Metal kernel that matches CUDA bit-for-bit. Large
  effort, only worth it if (A) has unacceptable perf.

Recommendation: **(A) immediately**, then **(B) when convenient**. (A) gets
us a noticeable visible-quality improvement now; (B) tells us if we can also
fix it on MPS itself.

### 3. Per-step DiT diff with `--fov` neutralized

The original Mac baseline (`mps_fdg/`) was diff'd against CUDA without
matching `--fov`. The cascading RED on 02 / 03 might shrink substantially
if we re-run Mac with `--fov 0.6061` and compare per-step samples — that'd
tell us whether the DiT itself is faithful given matched conditioning, or
whether it adds its own MPS quirks on top.

ETA: ~25 min Mac run (full pipeline this time, no STOP_AFTER), then `diff_fixtures.py`.

### 4. Texture fuzz (separate hunt; geometry sieve is now nearly explained)

Per pixal3d-mlx PLAN.md Phase 7: bilinear sampling boundary handling, fp16
vs fp32 accumulators in DINOv3 + NAF, texture bake step. Some of this is
now subsumed by today's NAF + grid_sample findings; the rest is the tex
SLat decoder pipeline.

### Lower priority / nice-to-have

- Adopt `PIXAL3D_SIMPLIFY=metal PIXAL3D_REPAIR_NME=metal
  PIXAL3D_UNIFY_FACE_ORIENTATIONS=metal` as the default for Mac runs — 50%
  wall-clock win per session 7 with no quality cost. Currently we keep
  hitting the default pamo path and burning 13 minutes per run on cleanup.
- Promote the `PIXAL3D_NAF_DEVICE=cpu` diagnostic into a proper feature
  flag once we decide on the NAF fix path.
- Investigate the coord-convention smoking gun (CUDA bbox long-axis-Y vs
  Mac long-axis-Z). Cosmetic but worth a one-line fix once we find where.

---

## Heuristics learned this session

- **`z_global` vs `z_proj` GREEN/RED contrast is a powerful localization
  signal.** If two outputs come from the same upstream tensor and one is
  GREEN while the other is RED, the divergence is *not* in the upstream —
  it's in the path that diverges. Saved us from auditing the entire
  DINOv3 ViT.
- **`PIXAL3D_STOP_AFTER` is gold.** A 5-minute targeted experiment replaces
  a 25-minute full pipeline run. Should be the default mode for divergence
  hunts.
- **Don't trust rental clones to be fresh.** Two rentals in a row ran with
  stale code despite supposedly being on the latest branch. Bake a
  `git log -1 --oneline` printout into the run.log header or have the
  rental script `git pull` before launching.
- **`MeshWithVoxel(m.vertices, m.faces, ...)` discards extra attrs.** Any
  diagnostic that stashes attrs on a `Mesh` object inside the model and
  expects to read them from the pipeline output needs to also patch the
  wrap site. Cost us one full rental cycle to learn.
- **Per-fixture pickle issues are real.** `07_run_output.pt` pickles
  `flex_gemm.ops.spconv.submanifold_conv3d.SubMConv3dNeighborCache` (CUDA
  hashmap cache). Loading on Mac requires a stub. The pattern is established
  in `generate_mps.load_stage_07_fixture` and is reusable.

---

## Open questions for future sessions

1. Why does the same image + seed produce **different fdg_coords counts**
   on different Mac runs (4,492,536 vs 3,807,324)? MPS DiT nondeterminism
   even with `torch.manual_seed`? Or is there another source of entropy
   we're missing? (`02_sparse_structure_rng_state.cpu` Δ=0 at sampler
   entry, so the entropy *is* identical going in.)
2. The `mps_full_nofov_tex_fp32/` experiment (tex VAE forced to fp32) was
   retired in session 6 as "VAEs are at the fp32 floor." But the divergence
   path we now have shows z_proj → DiT → SLat → VAE. The tex-VAE finding
   was always conditional on its inputs being identical, which they weren't.
   Worth a sanity revisit once the z_proj chain is cleaned up.
3. NAF (`valeoai/NAF`) is loaded fresh from torch.hub on each fresh rental.
   Worth pinning to a specific commit hash to rule out the network model
   drift confound? (Unlikely to be the issue but cheap to lock down.)
