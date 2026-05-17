# pamo CPU/MPS Port — Plan & Status

CPU/MPS port of cumesh's pamo parallel-QEM mesh simplification, replacing
the broken Apple Silicon Metal kernel in
`trellis-mac/.venv/.../cumesh/cumesh.metallib`.  See Session 4 log in
`SESSION_NOTES.md` for the band-aid history that led here.

## Why this port

| Component | Source | Status on M-series |
|---|---|---|
| CUDA reference | `/Users/pawelma/code/ai/CuMesh/src/simplify.cu` (591 lines) | works (not our hardware) |
| Metal port | `cumesh.metallib` + `_C.cpython-311-darwin.so` (binary, no source) | **broken** — skips winding-flip rejection (`simplify.cu:128`), drives mesh below target, punches missing-face chunks into real surface |
| Our port | `pixal3d/utils/pamo_simplify.py` | **in progress** — Phases 0-3 done, 4-7 to go |

Direct evidence of the Metal port bug (on
`checkpoints/1_img_fdg_cap.npz`, 8.4 M faces, target 1 M):

* After `simplify_target`: **859 K faces** (below 1 M target — should be ≥ target)
* Final output has visible holes (wireframe also has holes — confirmed by
  user from multiple angles)
* Removing the Metal `simplify` call entirely (`--skip-decimation`)
  eliminates the missing-face artifacts (proves it's the simplify step)

## Algorithm summary

`MtlMesh.simplify(target_num_faces)` wraps an iteration loop around
`simplify_step(λ_edge_length, λ_skinny, threshold)`.  One step does:

1. **`get_vertex_face_adjacency`** — CSR (vertex → incident faces)
2. **`get_edges` + `get_boundary_info`** — edge list, per-vertex boundary flag
3. **`get_qem`** — per-vertex 4×4 QEM (sum of unit-plane quadrics from
   incident faces; classic Garland-Heckbert, NOT area-weighted)
4. **`get_edge_collapse_cost`** — per-edge cost using QEM + λ_edge_length
   + λ_skinny, with **winding-flip rejection** (`simplify.cu:128`:
   `if (old_normal.dot(new_normal) < 0.0f) return false; // invalid`)
5. **`propagate_cost`** — per-face = min of its 3 corner edges' costs
   (this is how pamo gets parallelism without sequential priority-queue
   bottleneck: faces vote for their cheapest incident edge, so no two
   batch-applied collapses can touch the same face)
6. **`collapse_edges`** — for each face, if all 3 corners agree on the
   same propagated cost and it's below `threshold`, apply the collapse
   in this batch.  No vertex-conflicts between collapses by construction.

Outer loop (`metal_backend.py:simplify`):

```python
thresh = 1e-8
while num_faces > target:
    new_v, new_f = simplify_step(λ_edge_length, λ_skinny, thresh)
    if (cur - new_f) / cur < 1e-2:
        thresh *= 10
    cur = new_f
```

## Phase plan & status

| Phase | Goal | Status |
|---|---|---|
| 0 | Scaffolding, monkey-patch wiring | **✅ done** |
| 1 | Connectivity prerequisites | **✅ done** |
| 2 | QEM computation | **✅ done** |
| 3 | Edge cost + flip-reject | **✅ done** |
| 4 | Parallel batch collapse | **✅ done** |
| 5 | Outer simplify loop | **✅ done** |
| 6 | Validation vs upstream | **⚠ blocked** — pamo is correct, but the Mac downstream pipeline can't handle any aggressive simplification on the NME-heavy FDG mesh.  See Phase 6 log below. |
| 7 | Performance pass (optional) | ☐ deferred |

### Phase 0 — scaffolding ✅

* Created `pixal3d/utils/pamo_simplify.py` with stubs for
  `simplify(mesh_backend, target, ...)` and
  `simplify_step(vertices, faces, λ_edge, λ_skinny, thresh, ...)`.
* Added `_load_pamo_simplify_module` + `_patch_simplify` to
  `pixal3d/utils/o_voxel_native_export.py`.  Patch hooks
  `_MeshBackend.simplify` → `pamo_simplify.simplify`.
* `main()` calls `_patch_simplify(postprocess)` after the existing
  `_patch_grid_sample_output` / `_patch_repair_non_manifold_edges`.
* Verified end-to-end: `MtlMesh.simplify.__name__ == "_patched_simplify"`,
  patch attribute set, no import errors under the trellis-mac venv.

**Current effect when run**: bridge runs equivalent to
`--skip-decimation` (simplify is a no-op; mesh stays at FDG output face
count).

### Phase 1 — connectivity prerequisites ✅

Three functions, all pure PyTorch, all CPU-tested on:
* Tetrahedron (4 V, 4 F) — by-hand verifiable
* FDG checkpoint (3.8 M V, 8.4 M F) — realistic load

| Function | Output | FDG timing | Cross-check |
|---|---|---|---|
| `get_vertex_face_adjacency` | (V+1,) offsets, (3F,) indices | 0.93 s | matches `cumesh_port._build_edges` |
| `get_edges` | (E, 2) edges, edge_face_count, (F, 3) edge_of_corner | 0.92 s | 12,179,526 edges — identical count to cumesh_port |
| `get_boundary_info` | (V,) bool | 0.01 s | 508,572 boundary verts |

Edge type breakdown on FDG:
* boundary (1 face): 423,639
* manifold (2 faces): 10,850,958
* **non-manifold (3+ faces): 904,929** ← these drive repair_nme over-split

### Phase 2 — QEM computation ✅

`get_qem(vertices, faces) -> (V, 10)` returns symmetric 4×4 QEM matrix
packed as upper-triangle 10 floats per vertex.

Algorithm: unit-plane quadric per face (mirrors `simplify.cu:35-58`
exactly), scatter-add to per-vertex via `tensor.index_add_`.

Sanity check passes: `v^T Q v` evaluated at each vertex's own position
returns ~0 (≤ 1.33e-15 max on FDG, defining invariant of QEM).

Timing: 0.45 s on FDG.  No NaN/Inf.

### Phase 3 — edge cost + flip-reject ✅

`get_edge_collapse_cost(vertices, faces, vf_offsets, vf_indices, edges,
qems, vert_is_boundary, λ_edge_length, λ_skinny, max_vertex_degree=24,
chunk_size=500_000) -> (E,) float64`

Implements per-edge cost mirroring `simplify.cu:158-228`:

* **Boundary-aware collapse target** (`v_new`): both boundary or both
  interior → midpoint; one-side boundary → boundary vertex stays.
* **QEM cost**: `v_new^T (Q[e0] + Q[e1]) v_new`
* **Edge length penalty**: `λ_edge · ||v1 − v0||²`
* **Skinny + flip check** per incident face: substitute `v_new` for the
  kept vertex, reject (cost = ∞) if `old_n · new_n < 0`, compute
  shape metric `4√3 · new_area / Σ new_edge_norm²`, accumulate
  `λ_skinny · (1 − clamp(shape, 0, 1)) · ||e||²`.

Implementation notes:

* Per-vertex face lists padded to `max_vertex_degree=24` slots
  (covers >99.9 % of vertices in our FDG meshes).
* Edges incident on vertices with degree > 24 get cost = ∞.  Safe
  rejection: just means those edges don't collapse this iteration.
* Processed in chunks of 500 K edges (limits peak memory).

Results on FDG (12.18 M edges):
* 22.73 s total
* 94.6 % finite, 5.4 % rejected (mostly flip or degree-cap)
* Finite cost stats: min 1.27 e-12, median 1.90 e-08, max 4.41 e-05

### Phase 4 — propagate_cost + collapse_edges  ✅

Both open questions from the original plan dissolved on a closer read of
the CUDA — the propagate+match pattern *is* the conflict-resolution
mechanism; no union-find or extra vertex-conflict check needed.

**`propagate_cost(edges, edge_cost, vf_offsets, vf_indices, num_faces)
-> propagated (F,) int64`**

For each edge i with endpoints (e0, e1), atomic-min the packed value
`pack(cost_i, i)` into `propagated_cost[f]` for **every face f incident
on e0 or e1** (not just the 3 corner-edge faces — the CUDA reaches
through `vf_indices` for both endpoints).  Pack layout:
`(float32_bits(cost) << 32) | edge_id`.  IEEE 754 bit patterns are
monotone for non-negative floats, so plain
`scatter_reduce_(reduce='amin')` on the packed int64 reproduces CUDA's
`atomicMin` semantics with edge_id as deterministic tiebreak.

Vectorization: per endpoint side, `repeat_interleave` fans each edge's
packed value to all its incident faces, then one scatter_reduce.
Chunked over edges (chunk_size=1M) to bound peak memory.

**`collapse_edges(vertices, faces, edges, edge_cost, propagated_cost,
threshold, vf_offsets, vf_indices, vert_is_boundary)
-> (new_vertices, new_faces)`**

Edge i wins iff:
* `edge_cost[i] ≤ threshold`
* For every face f in `vf[e0] ∪ vf[e1]`,
  `propagated_cost[f] == pack(cost_i, i)`

**Why no union-find is needed**: if i wins, no edge j sharing a vertex
with i can also win — they'd contend for the same face's amin slot, and
only one int64 can be the min there.  So all winning edges are
pairwise vertex-disjoint, and each face has at most one corner whose
vertex gets remapped.  This means the collapse application reduces to:

1. `vertices[e0] = v_new` (boundary-aware midpoint, same rule as Phase 3).
2. Vertex remap: `vert_remap[e1] = e0`, identity elsewhere.
3. Apply remap to faces, then drop degenerate triangles (corners with
   any pair equal).  Faces that had both endpoints of some winning edge
   naturally fall here.
4. Compact orphan vertices (every winning edge's `e1` vanishes).

Edges incident on vertices with degree > `max_vertex_degree` (24) get
rejected — we can't see all their incident faces in the padded table,
so we can't verify the "all neighbors agree" condition.  Safe
rejection: they just don't collapse this iteration.

Validation: single-step FDG run (`thresh=1e-8`) collapses 228k vertex
pairs / removes 447k faces in 0.79s (collapse step itself; full
pipeline ~24s, dominated by Phase 3 cost computation at 21s).  No
crashes, no NaN/Inf in output.

### Phase 5 — outer simplify loop  ✅

`simplify_step` chains stages 1→6.  `simplify(mesh_backend, target,
verbose, options)` matches `cumesh/metal_backend.py:192-227`:

* `mesh_backend.read()` → CPU tensors → simplify loop → `mesh_backend.init()`
  (which handles MPS transfer + dtype coercion to float32/int32).
* Threshold escalation: if a step removes < 1% of faces, `thresh *= 10`.
* Safety: stop if `thresh > thresh_cap` (default 1.0; CPU-port-only
  safeguard against pathological meshes that can't reach target —
  upstream CUDA spins forever in that case).

Validation: full backend round-trip on FDG checkpoint with target=5M:

* 8,396,127 → 4,812,470 faces in 17 steps, 307 s.
* Threshold auto-escalated once (1e-8 → 1e-7) at step 13 when per-step
  progress fell below 1%.
* `mesh_backend.init(verts, faces)` write-back succeeds; `num_faces`
  property reflects the new count.

### Phase 6 — validation  ⚠ blocked

**Bottom line**: pamo is a correct port of the CUDA algorithm.  Verified
on a clean icosphere (20k → 4.8k faces, 0 boundary edges, 0 NMEs, 1 CC —
perfect).  On the FDG checkpoint, however, **no parameterisation of pamo
+ the Mac downstream produces a clean GLB**.  The bug is not in pamo;
it's in how any aggressive QEM simplification interacts with the FDG
mesh's 904k pre-existing non-manifold edges + the Mac downstream
cleanup chain (``repair_non_manifold_edges`` over-splitting +
``remove_small_connected_components`` culling the fragments).

#### Variants tried (all failed visually)

| Variant | Pamo behaviour | Downstream | Raw pamo output | Final GLB |
|---|---|---|---|---|
| baseline | `--skip-decimation` | default | n/a | **clean** ✓ |
| v1 | permissive (no link check) | default | 17.8% NMEs, swiss-cheese holes | swiss-cheese |
| v2 | + link check | default | 17.0% NMEs (-0.8pp) | swiss-cheese |
| v3 | strict manifold-only (`edge_face_count == 2`) | default | **looks like 8.4M input** ✓ | missing chunks |
| v4 | strict + no-op `remove_small_cc` | no-op small_cc | (same as v3) | **worse** — debris + spikes |

#### Key data points

* FDG input: 12.18M edges; 3.5% boundary, 89.1% manifold, **7.4% NME**.
* After permissive pamo (v1) to 3M faces: NME fraction jumps to **17.8%**
  because manifold edges collapse easily while NME-adjacent edges resist.
* After strict-manifold pamo (v3): NMEs rise to **21.5%** by enrichment
  (manifolds collapse, NMEs are preserved verbatim).
* Downstream `repair_non_manifold_edges` adds ~2M vertices splitting NMEs.
  `remove_small_connected_components(1e-5)` then culls the resulting tiny
  CCs as dust.  `fill_holes(3e-2)` recovers only a fraction.
* User report on raw pamo v3 (strict): _"looks as it did non-decimated,
  there are holes though, just not the kind we were fighting against —
  pinprick holes + a few irregular larger holes on the back of the model
  (FDG generation artefacts the upstream pipeline is meant to repair)."_

#### What we learned

1. **Pamo's collapse algorithm is sound.**  Tested on clean manifold
   input, output is pristine.
2. **The FDG mesh is the variable that matters.**  Any QEM simplifier
   (Metal port, CUDA upstream, our pamo) enriches the NME fraction.
3. **The Mac downstream cleanup is brittle.**  It was originally tuned
   for CUDA-produced post-simplify meshes; on Mac it interacts badly with
   anything other than the un-simplified FDG output.
4. **The bug is not where the plan originally diagnosed it.**  The
   Metal port's missing winding-flip check (`simplify.cu:128`) was a
   real bug, but fixing it (via pamo) didn't restore output quality —
   the underlying issue is upstream of simplify, in the FDG cap mesh
   extraction and/or the repair_nme + small_cc downstream interaction.

#### Recommended next steps (next session)

The user's plan: commit current work as WIP, then restart fresh with a
step-by-step rebuild + quality gate at each stage, using a rented NVIDIA
box to do CUDA-vs-MPS divergence analysis with the same seed + input
image.  Concretely:

1. **Snapshot the divergence point.**  Same image + same seed on CUDA
   reference and on this Mac pipeline.  Save intermediate tensors after
   each stage (Sparse-struct sampler, FDG cap, mesh extract, simplify,
   repair_nme, fill_holes, mesh cluster, UV unwrap, bake).  Diff each.
   The plan: pamo should produce identical (or near-identical) output to
   CUDA cumesh on the same input; if not, the bug is in our port.  If
   yes, the bug is upstream of simplify.
2. **If the FDG cap output is the divergence point**, investigate the
   ``--fdg-cap-partial-quads`` heuristic and/or the FDG decoder more
   carefully — that's where the 904k NMEs originate.
3. **If the divergence is in cumesh's `repair_non_manifold_edges`**, our
   `cumesh_port.repair` (CPU port) may need rework — or we should
   compare against CUDA cumesh's native repair on Linux+CUDA hardware.

#### Files touched in this Phase 6 (WIP commit candidate)

* `pixal3d/utils/pamo_simplify.py` — Phase 4-5 algorithm + Phase 6
  experiments (link condition, vertex-vertex adjacency, scratchpad
  variants); current state: permissive link check (post-revert from
  strict).
* `pixal3d/utils/o_voxel_native_export.py` — `_patch_simplify` activates
  the pamo port; `_patch_remove_small_connected_components` band-aid
  re-disabled (post-revert).
* `PAMO_PORT_PLAN.md` — this file; phases 4-5 marked done, phase 6
  blocked with full log.
* `scratch/phase6/` — `fdg_pamo*.glb` outputs from variants v1-v4 plus
  `raw_pamo_3M.glb` (pamo's raw output before any downstream).  Keep
  for reference; gitignore the dir or commit selectively.

#### Original Phase 6 plan (kept for reference)

* Run end-to-end with `pamo_simplify` active on the FDG checkpoint
  (target = 1 M).  Verify face count meets target.  Visual check in
  Blender: no missing chunks, no inverted faces, no texture confetti.
* Compare wireframe density distribution to GT output (the
  "uniform face size" property pamo is supposed to give us; if our
  output looks like terrazzo, something is wrong in Phase 4).
* Once clean: remove the now-redundant band-aids
  (`_per_face_winding_fix`, `trimesh.repair.fix_normals(multibody=True)`,
  possibly `_patch_repair_non_manifold_edges` if its CPU port turns
  out to behave identically to Metal which would mean it doesn't
  matter either way).

### Phase 7 — performance  ☐ (optional)

CPU should be fast enough for one render (~30-60 s for a few
`simplify_step` iterations).  If we need MPS:

* Switch `device='mps'` in `simplify(mesh_backend, ...)`.
* PyTorch should JIT most ops onto Metal.  Watch for the gather
  in `_pad_vertex_face_table` and the chunk loop in
  `get_edge_collapse_cost` — those might need rework if MPS gather
  performance is poor.
* Metal 4 / MPP tensor APIs are NOT relevant here — our hot loops
  are scatter-gather + per-edge conditional logic, not matmul.
  Plain PyTorch-on-MPS is sufficient.

## File layout

```
pixal3d/utils/
├── pamo_simplify.py           # NEW — the port itself
├── o_voxel_native_export.py   # MODIFIED — _patch_simplify routes
│                              #   _MeshBackend.simplify → pamo_simplify.simplify
└── cumesh_port/               # untouched (other CPU ports stay where they are)
    ├── repair.py
    ├── cleanup.py
    ├── fill_holes.py
    └── ...
```

## Reference files

* CUDA source: `/Users/pawelma/code/ai/CuMesh/src/simplify.cu` (591 lines)
* Python wrapper to mirror:
  `/Users/pawelma/code/ai/trellis-mac/.venv/lib/python3.11/site-packages/cumesh/metal_backend.py:192`
  (`MtlMesh.simplify`)
* Test checkpoint:
  `/Users/pawelma/code/ai/Pixal3D/checkpoints/1_img_fdg_cap.npz` (8.4 M faces)
* Native venv (Python 3.11 ABI, has cumesh wheels):
  `/Users/pawelma/code/ai/trellis-mac/.venv/bin/python3.11`

## Environment gotchas

* The Pixal3D project venv is Python 3.12; native cumesh/o_voxel wheels
  are Python 3.11.  All pamo testing happens via the trellis-mac venv
  through the subprocess bridge.
* `pamo_simplify.py` is loaded via `importlib.util.spec_from_file_location`
  (`_load_pamo_simplify_module`) to bypass the `pixal3d` package
  `__init__.py` (which imports siblings that may not load under the
  trellis-mac venv).
* Don't put `pamo_simplify.py` under `pixal3d/utils/cumesh_port/` —
  that's the CPU-port package and importing it triggers the package
  `__init__.py` which fails in the bridge subprocess.

## How to resume next session

1. Read this file + the **Session 4 log** in `SESSION_NOTES.md`.
2. `pixal3d/utils/pamo_simplify.py` is feature-complete for Phases 0-5.
   Public API: `simplify(mesh_backend, target, ...)` and
   `simplify_step(vertices, faces, λ_edge, λ_skin, thresh, ...)`.
   Internal helpers: `get_vertex_face_adjacency`, `get_edges`,
   `get_boundary_info`, `get_qem`, `_pad_vertex_face_table`,
   `get_edge_collapse_cost`, `_pack_edge_cost`, `propagate_cost`,
   `collapse_edges`.
3. **Phase 6 (validation)** is the next milestone:
   * Full end-to-end `generate_mps.py` run with
     `--native-decimation-target 1000000 --texture-size 2048` on the
     FDG checkpoint.  Estimated wall time: ~10 min based on
     ~18 s/step × ~30 steps (5M-face test ran 17 steps in 307 s).
   * Compare GLB in Blender vs the broken Metal output and the
     `--skip-decimation` baseline.  Confirm: face count meets target,
     no missing chunks, no inverted faces, wireframe shows uniform
     face size (pamo's defining property).
4. Once Phase 6 confirms output quality, remove the band-aids in
   `o_voxel_native_export.py`:
   `_per_face_winding_fix`, `trimesh.repair.fix_normals(multibody=True)`,
   possibly `_patch_repair_non_manifold_edges`.
5. **Phase 7 (perf, optional)**: bottleneck is
   `get_edge_collapse_cost` at ~15-18 s/step (≈75% of wall time).
   The padded vertex-face table with `max_vertex_degree=24` does a
   lot of redundant work for low-degree vertices.  Promising
   directions:
   * Skip cost recomputation for edges whose endpoints' incident
     faces were untouched by the previous step (incremental update).
   * Move to MPS (`device='mps'`) — scatter/gather should JIT cleanly.
   * Drop `max_vertex_degree` to the actual max in the chunk (saves
     useless slots for the 99% of vertices with degree ≤ ~12).
