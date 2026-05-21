# Next Steps — pamo Mac/MPS Port

Captured directions to pick from for future sessions.  Ordered by likely
ROI as of session 2026-05-21 (post divergence-hunt + downstream
investigation).

## In-flight / ordered next

1. **Texture-VAE precision focus** (next)
2. **Pre-clean FDG NMEs before pamo runs**

## All candidate directions

### 1. Texture-VAE precision focus  *(next)*

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

### 2. Pre-clean FDG NMEs before pamo runs

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
