# Next Steps — pamo Mac/MPS Port

Captured directions to pick from for future sessions.

> **Session 6 update (2026-05-22):** items #1 (tex-VAE precision) and #2
> (FDG NME pre-clean) below are now **empirically RULED OUT** —
> apples-to-apples diffs with `PIXAL3D_CPU_MODELS` proved both VAEs are
> bit-identical CPU↔MPS, and downstream cleanup tuning is at its floor.
> All visible-quality divergence localizes to the DiT samplers, dominated
> by the `flash_attn` (CUDA reference) vs `sdpa` (Mac fallback) attention
> backend mismatch.  See `OVERNIGHT_FINDINGS.md` and commit `d3e2c5c`.
>
> The original session-2 ordering is preserved below for history; do not
> work items #1 or #2 without new evidence.

## In-flight / ordered next (post-session-7)

> **Session 7 closing (2026-05-22):** Six findings, ordered by leverage:
>
> 1. **`metal-flash-attention` integration is RETIRED.**  Two independent
>    signals: (a) synthetic-Q/K/V experiment showed MPS sdpa is at fp32
>    precision floor (~1e-6 relative drift vs fp64 truth) — attention
>    isn't the bottleneck; (b) a community MLX-native port of Pixal3D
>    (`/Users/pawelma/code/ai/pixal3d-mlx`) using stock MLX SDPA
>    produces visibly *worse* output than ours.  Framework switch
>    doesn't help.
> 2. **Pedro's `mtlmesh` port is FAITHFUL to CUDA cumesh.**  Opus-grade
>    audit of `simplify.metal` vs `simplify.cu` (session-7) refuted the
>    earlier "missing winding-flip-reject" diagnosis — the check IS
>    there at `simplify.metal:56`, verbatim.  The earlier patch logic
>    that swapped to pamo was based on a misdiagnosis.
> 3. **`PIXAL3D_SIMPLIFY=metal` is a 50% wall-clock win** (24:55 →
>    12:19) with equivalent quality on `1_img.png` seed=42.  Pamo's
>    13:25 of post-processing becomes ~30s native Metal.
>    Added `PIXAL3D_REPAIR_NME=metal` opt-out; setting both + the
>    already-existing `PIXAL3D_UNIFY_FACE_ORIENTATIONS=metal` produces
>    the same wall-clock and same quality as simplify=metal alone (the
>    speed all came from simplify; the other patches were near-zero
>    cost).
> 4. **The real geometry bug is a SIEVE, not a simplify-cost-function
>    failure.**  Inside-out Blender shots show CUDA produces a sealed
>    shell, all Mac variants (pamo, simplify=metal, full Metal chain)
>    produce a sieve with thousands of small surface holes.  The
>    "jagged edges / missing triangles / shards" we've been chasing
>    are the OUTSIDE view of those same holes.
> 5. **Two precise suspects** for the sieve, decisive test pending:
>    (a) `pixal3d/utils/mesh_extract.py` CPU fallback (`flexible_dual_grid_to_mesh`)
>    leaving missing-face chunks vs CUDA's o_voxel extraction;
>    (b) Mac cleanup chain destroying good input.  Both could be
>    contributors; precise test is to load CUDA's pre-extracted stage-08
>    mesh and run only the cleanup chain on Mac.
> 6. **Texture fuzz** is independent of the sieve.  Separate hunt:
>    bilinear sampling boundary handling, fp16/fp32 accumulators in
>    DINOv3+NAF, texture bake step.  Don't conflate with geometry.

1. **DONE — sieve localized to mesh extraction.**  Decisive test run
   end of session 7 via `scratch/sieve_test_cleanup_only.py`: loaded
   CUDA's stage-08 pre-extracted mesh, ran ONLY our Mac cleanup chain
   (Pedro's Metal: `fill_holes → simplify → dedup → repair_nme →
   small_cc → fill_holes_2 → simplify_target → unify`).  Result:
   **sealed shell with one tiny hole** — equivalent to CUDA's own
   cleaned mesh quality.  Cleanup chain runs in 1 second total.
   → **The sieve culprit is `pixal3d/utils/mesh_extract.py` CPU
   `flexible_dual_grid_to_mesh` fallback.**  Likely missing edge
   cases for partial quads / degenerate cells / boundary handling vs
   the CUDA o_voxel kernel.  Output saved at
   `fixtures/sieve_test_stage08/cleanup_only.glb`.
2. **Next session START HERE — audit & fix the FDG extractor.**

   Canonical reference source (found end of session 7):
   ```
   /Users/pawelma/code/ai/trellis-mac/TRELLIS.2/o-voxel/src/convert/
     flexible_dual_grid.cpp        # 32.9K — the real implementation
     api.h                         # function signatures
   /Users/pawelma/code/ai/trellis-mac/TRELLIS.2/o-voxel/o_voxel/convert/
     flexible_dual_grid.py         # 283-line Python wrapper around _C
   ```

   Our (buggy) Python port:
   ```
   /Users/pawelma/code/ai/Pixal3D/pixal3d/utils/mesh_extract.py
     (283 lines; has both directions: mesh_to_flexible_dual_grid and
      flexible_dual_grid_to_mesh; has partial-quad constants but
      apparently mis-handles them)
   ```

   Pedro's possibly-adjacent Metal kernel (NOT a direct FDG port —
   it's dual contouring, related family):
   ```
   /Users/pawelma/code/ai/mtlmesh/src/metal/remesh.metal
     simple_dual_contour_u32_kernel / simple_dual_contour_u64_kernel
   ```

   **Concrete first action**: dispatch a fresh opus subagent to audit
   `flexible_dual_grid.cpp` vs our `mesh_extract.py` (same pattern
   that worked for simplify in session 7).  Prompt should ask for:
   1. Algorithm structure of upstream — CUDA-required (atomics,
      `__global__` kernels) vs torch-tensor-portable.
   2. Specific gaps in our Python port — partial-quad handling,
      degenerate cells, NME-free output guarantees, boundary cases.
   3. Recommended path: patch the Python port (small gaps) vs Metal
      port (structural CUDA dependence) vs adopt-mtlmesh-dual-contour
      (different algorithm, different topology than CUDA — fallback).

   Once the gap shape is known, the implementation move is clear.
   Session 7's audit-then-integrate flow took ~half a session to
   produce the 50% speedup; FDG fix is potentially the same pattern
   for the geometry sieve.
3. **Investigate the back-of-tower DiT difference** (image 28 from
   session 7 user notes).  Mac and CUDA produce subtly different
   topology in specific regions for the same seed.  Smaller concern
   than the sieve; only worth chasing once geometry is fixed.
2. **Investigate the coord-convention smoking gun.**  CUDA bbox is
   long-axis-Y (~0.69 × 0.88 × 0.62), all our Mac outputs are
   long-axis-Z (~0.69 × 0.60 × 0.88).  Shared between two independent
   Mac ports → lives upstream of both.  Cosmetic per user (textures
   align correctly on the rotated mesh) but worth a one-line fix once
   we find where.
3. **Texture-quality investigation (separate from geometry).**  Per
   pixal3d-mlx PLAN.md's Phase 7: bilinear sampling differences,
   fp16/fp32 accumulators in DINOv3 + NAF upsampler, texture bake.
4. **Always-on `PIXAL3D_CPU_MODELS=sparse_structure_flow_model,sparse_structure_decoder`**
   as a free incremental quality win (+~1 min wall, +0.1pp stage-02
   coord agreement).  Marginal but free.  Promote once regression-tested.

## Suspect lists (split by symptom)

### Geometry sieve (LOCALIZED — bug is in mesh extraction)

- **`pixal3d/utils/mesh_extract.py` CPU `flexible_dual_grid_to_mesh`** —
  this is the bug, confirmed end of session 7 via the stage-08 replay
  test.  The cleanup chain is innocent.  Audit against the CUDA
  reference (look at `CuMesh/` and upstream Pixal3D's o_voxel module
  for the canonical kernel).  Likely missing logic: partial-quad
  handling, degenerate-cell handling, NME-free output guarantees.

### Retired suspects (confirmed innocent via stage-08 test)

- ~~Cleanup chain destroying good input~~ — session 7 stage-08 replay
  showed the chain produces a sealed shell when given CUDA's clean mesh.
- ~~`fill_holes` default perimeter too tight~~ — same test showed Pedro's
  Metal fill_holes at default 3e-2 handles CUDA's input cleanly.
- ~~`small_cc` over-culling~~ — drops 132K verts on CUDA's clean input,
  but the result still looks good; the legit-fragment-loss concern is
  real but minor compared to the sieve.

### Texture fuzz (separate hunt — not relevant until geometry is fixed)

Per pixal3d-mlx PLAN.md's Phase 7 expected culprits:

- **Bilinear sampling boundary handling** in DINOv2 back-projection /
  voxel-to-pixel sampling.  CUDA / MPS / MLX all differ subtly.
- **fp16 vs fp32 accumulators** in conditioning models (DINOv3, NAF
  upsampler).
- **Camera coord conventions** — also explains the cosmetic Z-vs-Y
  long-axis rotation (see ordered #2).

## Retired (do not work)

- ~~Texture-VAE precision focus~~ (session 6 — VAEs proven deterministic).
- ~~Pre-clean FDG NMEs before pamo~~ (session 6 — cleanup is at its floor).
- ~~`philipturner/metal-flash-attention` integration~~ (session 7 — attention
  is at fp32 floor; not the bottleneck).
- ~~Switch to MLX-native port~~ (session 7 — produces visibly worse output
  than our PyTorch/MPS port).
- ~~Replace pamo as a *quality* fix~~ (session 7 — pamo and Pedro's Metal
  port produce the same sieve.  `PIXAL3D_SIMPLIFY=metal` is a 50% wall-clock
  win but does NOT close the geometry gap.  Sieve is upstream).
- ~~Sparse conv / `flex_gemm` divergence~~ (session 7 — MLX port author
  tried `flex_gemm_sparse_attn` and got no quality change.  Not the cause).

## Parked (low priority)

- Finish the SparseTensor device-mismatch bug in the `PIXAL3D_CPU_MODELS`
  wrapper so it works for `ElasticSLatFlowModel` too.  Only needed if we
  want the all-DiTs-on-CPU experimental baseline finished.

## Historical ordering (pre-session-6, kept for context)

1. **Texture-VAE precision focus** — [RULED OUT — session 6]
2. **Pre-clean FDG NMEs before pamo runs** — [RULED OUT — session 6]

## All candidate directions

### 1. Texture-VAE precision focus  *(next)*  **[RULED OUT — session 6]**

> Session 6 proved both VAEs (`shape_slat_decoder`, `tex_slat_decoder`)
> are 100% deterministic — bit-identical output on CPU vs MPS for the
> same SparseTensor input.  The muted-texture symptom is therefore not
> a tex-VAE precision issue; it's the upstream DiT drift exposing
> different SparseTensor coords to the decoder.  Do NOT spend time
> here.

The "muted/blurry texture" half of the visible quality gap is currently
completely unaddressed.

* Element-wise diff `06_tex_slat_decoded.pt` between CUDA and MPS where
  the SparseTensor coords align.  Find out how much magnitude difference
  the tex VAE decoder accumulates.
* If precision-driven: try fp32-casting just the tex VAE decoder
  (`pipeline.models["tex_slat_decoder"].float()`) — narrow enough to
  potentially avoid the all-fp32 Metal NDArrayMatrixMultiplication
  assertion that the global fp32 cast hit.
* If algorithmic (e.g. attention path inside tex VAE): note that the
  fix may need wider work (the per-step DiT result said attention isn't
  the dominant cause for shape SLat — may differ for tex).

Effort: ~half day.  Payoff: potentially high — fixes the texture half.

### 2. Pre-clean FDG NMEs before pamo runs  **[RULED OUT — session 6]**

> Session 6 confirmed the downstream cleanup chain is at its empirical
> floor: the geometry damage comes from the MPS DiT samplers producing
> fundamentally different SparseTensor coords than CUDA, not from
> repair_nme / small_cc / fill_holes mishandling NME-heavy input.
> Pre-cleaning NMEs would only attack a symptom.  Do NOT spend time
> here.

Attack the root cause of the geometry damage chain.

* The FDG cap mesh has 904K pre-existing non-manifold edges (per
  PAMO_PORT_PLAN.md Phase 6).  pamo's QEM creates more NMEs at
  collapse boundaries.  Then repair_nme + small_cc + fill_holes interact
  badly with the inflated NME load.
* Idea: run `cumesh_port.repair.repair_non_manifold_edges` on the FDG
  output BEFORE `to_glb` is called.  Should produce a less-NME-heavy
  input for the downstream cleanup chain.
* Risk: repair_nme inflates vertex count proportional to NME count.
  Pre-cleaning might still inflate the mesh, but the inflation gets
  consumed by pamo's first simplify rather than triggering the cascading
  cleanup failure.
* Test: full Mac pipeline with this pre-clean; compare visible quality
  to current best (E_smcc_1e-7).

Effort: ~1 day.  Payoff: potentially high — attacks root cause.

### 3. `flexible_dual_grid_to_mesh` CPU-fallback audit

The CPU fallback in `pixal3d/utils/mesh_extract.py` is the FDG → mesh
extraction step.  SESSION_NOTES.md says it "leaves missing-face chunks"
that the cleanup chain then has to recover from.

* Audit the CPU-port logic vs. the CUDA reference behaviour (the CUDA
  `flexible_dual_grid_to_mesh` kernel in o_voxel is the gold standard).
* Look for missing edge cases — partial quads, degenerate cells, etc.
* Might be the actual source of the 904K NMEs in FDG output.

Effort: ~half day.  Payoff: medium — depends on whether the FDG
extraction is actually a separate issue or feeds into Phase-2 cleanup.

### 4. Phase 7 — pamo on MPS device for speed

Pure speed; no quality change.

* `pamo_simplify.py` currently runs on CPU (PyTorch ops, ~5 min per
  simplify pass × 2 passes per run).  PyTorch's MPS backend should JIT
  most ops onto Metal cleanly per the plan.
* Watch points: `_pad_vertex_face_table` (gather), chunked
  `get_edge_collapse_cost` (scatter + per-edge conditional).
* Expected speed-up: 3-5× → ~1-2 min per pass.
* Makes future experiments faster but doesn't address quality.

Effort: ~1 day.  Payoff: iteration speed only.

### 5. Per-layer DiT instrumentation

Definitive but academic at this point.

* The divergence-hunt session established that pred_x_0 drifts ~0.3
  magnitude from sampler step 0 on the sparse-structure DiT, regardless
  of which precision / SDPA backend / fp32-vs-bf16 / attention impl.
* A per-layer hook would tell us WHICH transformer block first drifts
  by >1e-3.  Definitive answer about whether it's uniform drift or a
  single broken kernel.
* But we already strongly suspect it's uniform — and the dominant
  visible-quality issue is the Mac post-pipeline anyway.
* Worth doing only if quality optimization plateaus and we want to
  understand the residual ~10% upstream drift.

Effort: ~1 day code + one CUDA rental cycle.  Payoff: low to medium.

### 6. Audit cumesh CUDA's `repair_non_manifold_edges` vs CPU port

* Our `cumesh_port.repair` faithfully mirrors the CUDA face-corner
  union-find algorithm.  But maybe CUDA cumesh has secondary cleanup
  passes we missed.
* Probe: capture CUDA cumesh's intermediate state during a real run on
  the rental box.  Compare vertex inflation count.
* Maybe CUDA has a "merge near-coincident split-fans" pass we don't.

Effort: ~half day + CUDA rental.  Payoff: medium.

## Not started (backlog)

### Investigate `--fdg-cap-partial-quads` heuristic

PAMO_PORT_PLAN.md Phase 6 listed this as a possible direction.  The FDG
decoder produces partial quads when a voxel cell has irregular
isosurface intersection — those generate non-manifold edges.  Maybe a
different threshold for "partial" handling produces fewer NMEs.

### Replace flex_gemm sparse-conv with MinkowskiEngine CPU

If sparse conv path turns out to be a major source of accumulated drift
(after texture-VAE work is done), MinkowskiEngine has a CPU
implementation of submanifold sparse conv that we could swap in.  Slow
but algorithmically faithful.

### Port flash_attn to Metal (philipturner/metal-flash-attention)

Discussed in session — medium effort (~1-2 weeks) for an attention
upgrade.  Not the dominant cause but would close some of the residual
DiT drift.

### Port flex_gemm sparse conv to Metal

Large effort (1-3 months).  Only worth it if texture-VAE work proves
the precision drift is in the sparse-conv path and the M5 tensor
primitives can be leveraged.

## Heuristics from session 2026-05-21

* **MPS SDPA backend selector is a no-op** as of torch 2.12 — don't
  waste time on `sdpa_kernel(MATH|EFFICIENT|FLASH)` ablations.
  See `DIVERGENCE_FINDINGS.md`.
* **Naive (pure-Python) attention is slightly WORSE than fused SDPA**
  on MPS.  Don't reach for it as an ablation.
* **Forcing global fp32 weights crashes Metal** (NDArrayMatrixMultiplication
  destination/accumulator dtype mismatch).  Targeted submodule
  `.float()` may work where global does not.
* **Tuning post-pipeline parameters has marginal effect** — all six
  variants (PIXAL3D_FILL_HOLES_PERIMETER, PIXAL3D_SMALL_CC, combos,
  --force-texture-fallback) land in the same metric range.  The
  structural cause is upstream of the parameters.
