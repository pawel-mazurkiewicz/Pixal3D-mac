# Apple Silicon Port — Session Notes

WIP port of TencentARC/Pixal3D to Apple Silicon (MPS). Source of truth for
"what works, what doesn't, why we made each choice". Update this file when
shipping behaviour changes.

## High-level state

| Component | Status | Notes |
|---|---|---|
| Image → mesh generation on MPS | ✅ works | ~8 min on M5 Max, identical seeds to CUDA |
| Mesh checkpointing | ✅ works | `--save-mesh` / `--load-mesh` skip 8-min gen |
| Geometry-only GLB export | ✅ works | Mesh upright in Blender |
| Textured GLB export | ⚠️ partial | Colors land, but coverage/quality not yet at HF demo level |
| `o_voxel.postprocess.to_glb` Metal path | ❌ not done | Would need porting cumesh + flex_gemm + voxel rasterizer to Metal |

## What's in the box

CLI entrypoint: `generate_mps.py` (run it with `--help` for the full set of
knobs).

New modules:
- `generate_mps.py` — top-level CLI, NATTEN monkey-patch, mesh checkpoint
- `pixal3d/utils/_blender_uv_unwrap.py` — runs inside Blender via subprocess
- `pixal3d/utils/texture_baker.py` — KDTree-based texture bake (xatlas/Blender unwrap dispatch)
- `pixal3d/utils/mesh_extract.py` — CPU fallback for `o_voxel.convert.flexible_dual_grid_to_mesh`
- `pixal3d/modules/sparse/conv/conv_none.py` — passthrough sparse-conv backend so the model loads

Modified upstream:
- `pixal3d/models/sc_vaes/fdg_vae.py` — device-aware buffers
- `pixal3d/modules/sparse/attention/full_attn.py` — SDPA fallback for sparse attn
- `pixal3d/modules/sparse/config.py` — recognise `SPARSE_CONV_BACKEND=none` and `SPARSE_ATTN_BACKEND=sdpa`
- `pixal3d/pipelines/pixal3d_image_to_3d.py` — skip CUDA-only cache/cumesh calls when not on CUDA
- `pixal3d/pipelines/rembg/BiRefNet.py` — use the model's own device for inputs
- `pixal3d/representations/mesh/base.py` — `o_voxel.cumesh` optional
- `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py` — DINOv3 / NAF on the chosen device

## The main fixes that landed (in order)

### 1. NATTEN neighborhood-attention port (NAF model)
The NAF upsampler (`torch.hub.load("valeoai/NAF")`) hardcodes
`na2d(..., backend="cutlass-fna")` which requires libnatten compiled with CUDA.
This wheel only exposes `blackwell-fna / hopper-fna / cutlass-fna / flex-fna`;
the first three are CUDA-gated and `flex-fna` rejects MPS tensors AND has a
catastrophic eager-mode memory blow-up on CPU.

**Fix:** patch `natten.na2d` in `generate_mps.py` `load_runtime_deps()` with a
pure-PyTorch index-gather neighbourhood-attention implementation. Tiled
processing (`_NA2D_QUERY_CHUNK = 2048`) keeps peak working-set per call
under a few hundred MB; per-MTLBuffer cap on Apple Silicon would reject the
non-tiled allocation. Validated against NATTEN's own `flex-fna` (on CPU) for
the asymmetric-Q/V case NAF uses — fp32 noise floor diff (~5e-7).

### 2. Mesh-extract CPU fallback
`o_voxel.convert.flexible_dual_grid_to_mesh` is CUDA-only. Codex thread had
already added a CPU fallback in `pixal3d/utils/mesh_extract.py`, but it leaves
pinprick holes wherever the iso-surface couldn't close a cell. Pixal3D's
official CUDA pipeline calls `cumesh.fill_holes(max_hole_perimeter=3e-2)` —
we use Blender's `bmesh.ops.holes_fill` instead (see #6 below).

### 3. xatlas slowdown
`xatlas.parametrize()` with default `ChartOptions.max_cost = 2.0` makes the
segmenter explore very tight chart boundaries; at 200k+ faces this takes
30–60 min on this hardware (was 40+ min mid-session, killed manually).
Setting `max_cost = 1.0` cuts wall-clock by ~half with imperceptible texture-
quality loss. Lives in `pixal3d/utils/texture_baker.py:_uv_unwrap_xatlas`.

### 4. Blender as the UV unwrap backend
xatlas is the wrong tool for Pixal3D-density meshes — even after the
`max_cost` fix it still produces hundreds of sub-pixel charts. The HF demo's
fast `o_voxel.postprocess.to_glb` is CUDA-only.

**Fix:** subprocess into a local Blender install (`--background`,
`--factory-startup`), do the entire decimate-and-unwrap pipeline in a
`bpy`/`bmesh` script: `pixal3d/utils/_blender_uv_unwrap.py`. CLI:
`--uv-unwrap {auto,blender,xatlas}` (default `auto`), `--blender-exe PATH`.
Auto-detect via `$PATH` and `/Applications/Blender.app/...`.

### 5. Mesh checkpoint
Generation is ~8 min; iterating on bake / UV / export shouldn't pay that
cost. `--save-mesh PATH` writes the raw mesh + voxel attrs to a `.npz` after
generation; `--load-mesh PATH` resumes from that npz and skips image
preprocessing, MoGe camera estimation, NAF, and the whole sampling pipeline.
Lives in `generate_mps.py:save_mesh_checkpoint / load_mesh_checkpoint`.

### 6. Hole filling
Pixal3D's CPU mesh extractor leaves ~3–8-edge pinprick holes (image #16 in
the session log was a close-up of these). Blender's GUI op
`bpy.ops.mesh.fill_holes` is 10–100× slower than the `bmesh` equivalent
because it routes through the undo stack — it hung indefinitely on a
2.7M-face mesh. We use `bmesh.ops.holes_fill(bm, edges=boundary, sides=N)`
directly. Capped at 5M faces to avoid pathological cases on `--skip-
decimation` paths.

CLI: `--fill-holes-sides N` (default 12). Set to 0 to skip.

### 7. Bake IDW radius
`bake_texture` (`pixal3d/utils/texture_baker.py`) does KDTree-IDW sampling
from voxel attrs to each rasterised texel. The previous `max_dist =
voxel_size * 2.0` cutoff was tight at trellis-mac's 512 voxel resolution
but at Pixal3D's 1024 it dropped to ~0.002 world units. After heavy
decimation, interior texels of simplified triangles sit 0.02–0.05 units
from any voxel centre, so 99% of texels had zero weight → fully black
texture.

**Fix:** scale the cutoff in *voxel units* via `search_voxels` param
(`--bake-search-voxels`, default 50). Gutter protection preserved.

### 8. Decimation
`fast_simplification` won't go below ~921k faces on a Pixal3D mesh because
it refuses to collapse non-manifold edges. Blender's Decimate-Collapse
hits the same wall at ~2.76M. Voxel remesh would work but **destroys
non-watertight meshes** (Blender T70925) and Pixal3D's mesh extractor
emits open shells. So voxel remesh is OFF by default
(`--voxel-remesh-size 0`); we use iterative Decimate-Collapse via the
Blender script and accept the ~2.76M floor.

CLI: `--bake-face-target N` (default 100000 — best-effort target),
`--skip-decimation` (bake on the full 8.4M-face mesh), `--voxel-remesh-size F`
(opt-in for watertight inputs).

### 9. Export rotation (revised twice)
The original `EXPORT_ROTATION_ROWS` matrix was a bogus 180° rotation around
the `(0, 1, -1)` axis: it negated X and swapped Y↔Z with sign flips,
putting the mesh upside-down AND mirrored.

Diagnosed by loading the GLB into Blender via the BlenderMCP addon
(`mcp__blender__execute_blender_code`) and checking the bbox + heavy-mass
distribution along the tall axis. The mesh's *pre-rotation* AABB has Y as
the tallest axis (extent 0.88), confirming **Pixal3D outputs Y-up
natively** (same as glTF). No rotation is needed.

**Initial fix:** set `EXPORT_ROTATION_ROWS` to identity.

**Round 2 (after diffing against the official HF demo's GLB):** Pixal3D's
CUDA `o_voxel.postprocess.to_glb` applies a Z-axis flip (in glTF coords)
that we weren't doing.  The HF demo's AABB had `y = [-0.37, 0.28]` in
Blender (heavy mass toward camera) while ours had `y = [-0.24, 0.36]`
(heavy mass away from camera) — exact mirror.  Fixed by setting
`EXPORT_ROTATION_ROWS` to negate glTF Z, **and** flipping face winding in
`rotated_vertices` and `_apply_export_rotation_to_mesh` (a reflection has
det = -1; without winding flip, face normals point inward and the model
goes invisible under backface culling).  The rotation is now applied
*early* in both export paths — to vertices AND to the voxel attribute
coordinate frame — so the cube projection, KDTree query, and final GLB
all see the same rotated coordinates.  Math for the voxel-frame rotation
is in `_apply_export_rotation_to_mesh`'s docstring.

### 10. Alpha = 0 made the mesh look x-ray
After fixing rotation, the rendered fairy house in Blender Material
Preview looked **transparent** — towers visible only as silhouettes, you
could see through to the back of the model.  Cause: the baked
`base_color_img` is RGB (`(H, W, 3)`), but trimesh / pygltflib expand it
to RGBA on GLB write, with alpha = 0 in unfilled gutter texels.
glTF viewers honoured that as transparency even with default
`alphaMode = OPAQUE`.

**Fix in `export_glb_with_texture`:** explicitly compose RGBA with
`alpha = 255` everywhere, hand to PIL in RGBA mode, set
`alphaMode = OPAQUE` and `doubleSided = True` on the material.  The
"x-ray" disappeared immediately and ~20% coverage texture now shows up
as grayscale fallback where no texture data exists, instead of holes.

### 11. cube_project UV overlap (FIXED but the approach is fundamentally wrong — see verdict below)
`bpy.ops.uv.cube_project` in Blender 5.x just does flat planar projection
per face — every face direction lands in the same `[-0.5, 0.5]²` region.
Confirmed by partitioning faces by normal direction in the loaded GLB:
all six cube buckets shared the same UV range. That's why the previous
bake showed roof content on the bottom of the model (the +Y and -Y
charts wrote to the same texels).

**Fix in progress:** manual 6-chart cube projection in
`_blender_uv_unwrap.py`. Each face is bucketed by dominant normal axis
(±X, ±Y, ±Z), projected onto the perpendicular plane, and placed in its
own atlas slot. Two-row layout: 3 "big" buckets on top, 3 "small"
buckets on the bottom, slot WIDTH proportional to face count within
each row (Pixal3D's image-to-3D produces a ~90:10 face-direction split,
so equal-area slots over-stuff three charts and waste the other three).

## Verdict on the current UV strategy: not viable

After comparing our output to the **official Pixal3D HF-demo GLB** for the
same input image, the conclusion is that **cube projection is the wrong
approach entirely** for this model class.  GT's atlas is a dense,
tightly-packed mosaic of ~thousand-plus small UV islands covering most
of a 4096² canvas (clearly `cumesh.uv_unwrap` output: angle-clustered
small charts with proper bin packing).  Ours is 6 axis-aligned
projection slabs occupying ~20% of a 2048² canvas with most of the
atlas empty.  The geometry is correct, the alpha is correct, the
orientation is correct — but the *surface look* is unusable: where the
GT shows a textured fairy house with blue roof shingles, brown trunk,
green foliage and PBR sheen, ours shows the right geometry painted in
the right colours but at sub-pixel density, so it reads as a grayscale
projection with sparse colour hints.

Visually compared side-by-side in Blender: the user verdict was
"they don't look ANYTHING alike … it's not usable in any shape or form
now".  Agreed.

So cube projection is dead.  Next session needs a different UV
unwrapper.  The path forward (best/worst case ordered by realism):

1. **Decimate to ~200k faces, then xatlas at our relaxed
   `max_cost=1.0`.**  xatlas produces dense small charts naturally; at
   200k faces with that setting it runs in ~3-5 min and packs into a
   4096² atlas at ~6 texels per face.  The blocker is **decimation**:
   `fast_simplification` floors at ~920k and Blender Decimate-Collapse
   floors at ~2.76M, both because they refuse to collapse non-manifold
   edges.  Real fix: switch decimator.  Candidates:
   - **`pymeshlab`** — bundled QEM is much more tolerant of bad
     topology.  Pure Python wheel exists for macOS arm64.  This is
     the cheapest experiment.
   - **`pyfqmr`** — also lighter on topology constraints.
   - **trimesh.repair.fill_holes() before decimation** — close the
     non-manifold edges first, then existing decimators can collapse.
   - Last resort: roll our own QEM that ignores non-manifold rules.
2. **Port `cumesh.uv_unwrap` to PyTorch/MPS.**  This is the "right"
   answer that would produce a GT-equivalent atlas.  Real engineering
   work — the algorithm is angle-cluster + region-grow + parameterise
   + pack, none of which is trivial.  Estimate: 1-2 weeks.  Worth it
   only if Pixal3D becomes the daily-driver pipeline on Mac.
3. **Vertex colours fallback** (`--vertex-colors`, not yet
   implemented).  Skip the UV/atlas problem entirely; bake voxel
   attrs to per-vertex colour.  At ~1.9M vertices that's roughly a
   1378² atlas equivalent.  Visual quality is "OK for preview" but
   not PBR-grade.  Cheap to implement (~half a day) — useful as an
   "always works" fallback and for fast iteration.

The current code-path is committed in this state mainly as a
checkpoint: it gets the mesh out, gets orientation right, gets the
alpha right, gets the bake working, but **the cube_project step is
slated for removal next session**.

## Texture quality — what's still off

Even with manual cube projection + proportional packing, the rendered
fairy-house has large black regions. At 2.7M faces and 2048² atlas:
- 3 big charts (negative cube directions) get ~30% of atlas each
- 3 small charts get ~3% each
- Per-face area in big charts ≈ 1.5 texels — still sub-pixel on average
- → rasteriser only fills 19–35% of the atlas → many faces appear black

Knobs that help:
- `--texture-size 4096` — 4× more texels per face (8192 GLB ≈ 250 MB)
- `--skip-decimation` — full 8.4M-face mesh, ~3 min bake but every face
  sits exactly on a voxel surface so the IDW samples cleanly
- `--bake-search-voxels 100` — wider IDW (smooths gaps)

What we *haven't* tried but should:
1. **Vertex colours** (`--vertex-colors` flag, not yet implemented). Bake
   voxel attrs to per-vertex colour and skip UV entirely. At 1.9M
   vertices that's ~1378×1378 equivalent texture resolution, no UV
   issues. Should be the simplest path to "looks acceptable" output.
2. **8192² atlas** — last bullet on the brute-force ladder.
3. **Real `o_voxel.postprocess.to_glb` Metal port** — proper fix but
   1–2 weeks of work (cumesh + flex_gemm + voxel rasterizer all need
   Metal Shading Language reimplementations).

## Useful one-liners

```bash
# First run: generate + save checkpoint for iteration
python3 generate_mps.py assets/images/1_img.png --output output_3d \
  --pipeline-type 1024_cascade --steps 8 \
  --save-mesh checkpoints/1_img.npz

# Iterate on UV/bake/export only (~2-3 min instead of 11)
python3 generate_mps.py --load-mesh checkpoints/1_img.npz --output output_3d

# Maximum fidelity (~6-10 min, ~250 MB GLB)
python3 generate_mps.py --load-mesh checkpoints/1_img.npz --output output_3d \
  --skip-decimation --texture-size 4096

# Verify orientation in Blender (BlenderMCP addon must be running)
# - see mcp__blender__execute_blender_code calls in session log for the pattern
```

## File map for the bake export pipeline

```
generate_mps.py
 ├─ main()
 │    ├─ load_runtime_deps()      # NATTEN monkey-patch, imports
 │    ├─ load_mesh_checkpoint()   # OR generate (8 min)
 │    ├─ export_geometry_only()   # if --no-texture
 │    └─ export_fallback_texture_glb()
 │         ├─ simplify_vertices_faces()  # fast_simplification (xatlas path only)
 │         ├─ uv_unwrap(mode="auto")     # → Blender or xatlas
 │         │    └─ _uv_unwrap_blender()
 │         │         └─ subprocess: blender --python _blender_uv_unwrap.py
 │         │              ├─ remove_doubles + dissolve_degenerate
 │         │              ├─ Decimate-Collapse (iterative, to target_count)
 │         │              ├─ bmesh.ops.holes_fill
 │         │              ├─ quads_convert_to_tris
 │         │              ├─ MANUAL 6-CHART CUBE PROJECTION  ← current
 │         │              └─ proportional atlas packing       ← current
 │         ├─ bake_texture()      # KDTree-IDW from voxel attrs
 │         └─ export_glb_with_texture()
 └─ rotated_vertices()   # currently identity (was the bug)
```

## Things to remember about this hardware/environment

- M-series Macs cap individual MTLBuffer allocations well below total
  unified memory; if a tensor allocation fails with "Invalid buffer size",
  the fix is almost always tiling/chunking, not freeing memory.
- `bpy.ops` is 10–100× slower than `bmesh.ops` on large meshes — always
  reach for bmesh first for bulk operations.
- Blender 5.x removed `--noaudio`. Don't pass it (it'll be interpreted
  as a positional filename argument and crash the run).
- The BlenderMCP addon unregisters itself when `bpy.ops.wm.read_factory_settings(use_empty=True)`
  runs. Use targeted `bpy.data.objects.remove(...)` to reset the scene
  without nuking the addon.
- `fast_simplification`, Blender Decimate-Collapse, and Blender voxel
  remesh ALL refuse to collapse non-manifold meshes the way you'd want
  them to. The only thing that bypasses the floor is voxel-remesh, which
  in turn destroys non-watertight inputs.

---

## Session 2 log — cumesh port attempt (FAILED at quality threshold)

This session attempted to bridge the texture-quality gap by porting cumesh
operations from CUDA to CPU/numpy.  **Outcome: even after implementing what
appears to be the full TRELLIS.2 to_glb pipeline, the resulting mesh and
texture look fundamentally worse than the GT GLB.**  User assessment after
side-by-side comparison: *"completely fucked up mesh and texture, looks
nothing like the GT, frankly we're at a worse spot than we were beginning
this session"*.

### What was discovered upstream

The official Pixal3D / TRELLIS.2 texture path is **not** just
"`xatlas` on the raw mesh".  Source-of-truth is
`/Users/pawelma/code/ai/TRELLIS.2/o-voxel/o_voxel/postprocess.py :: to_glb`.
Pipeline (decimation_target=1M, texture_size=2048 by default):

1. `cumesh.fill_holes(max_hole_perimeter=3e-2)`
2. `cumesh.cuBVH(orig_vertices, orig_faces)` — build BVH on PRE-decimation mesh
3. `mesh.simplify(decimation_target * 3)`  — 3M faces (CUDA QEM, NOT pymeshlab QEM)
4. cleanup loop: `remove_duplicate_faces` → `repair_non_manifold_edges`
   → `remove_small_connected_components(1e-5)` → `fill_holes`
5. `mesh.simplify(decimation_target)` → 1M faces
6. cleanup loop **again**
7. `mesh.unify_face_orientations`
8. `mesh.uv_unwrap(compute_charts_kwargs={threshold_cone_half_angle_rad=π/2,
   refine_iterations=0, global_iterations=1, smooth_strength=1})`
9. `nvdiffrast.dr.rasterize` UVs → 3D positions per texel
10. `bvh.unsigned_distance(positions, return_uvw=True)` → project to ORIG mesh
11. `flex_gemm.ops.grid_sample_3d(attr_volume, ..., corrected_pos, mode='trilinear')`
12. `cv2.inpaint(base_color, gap_mask, 3, cv2.INPAINT_TELEA)` for gutter fill
13. Trimesh with vertex_normals, alphaMode='OPAQUE', doubleSided

Compute_charts is the key bottleneck — its `refine_iterations=0,
global_iterations=1` setting still runs the **collapse phase to convergence**.

### Ports written this session

All under `pixal3d/utils/cumesh_port/`:

| Module | Purpose | Status |
|---|---|---|
| `fill_holes.py` | Triangulate small closed loops (fan-fill at centroid) | works; only filled 7,804 closed loops in raw mesh (vs 91K open chains) |
| `repair.py` | Vertex-splitting union-find for non-manifold edges | works; preserves face count (vs pymeshlab destructive repair); +2.5M vertices added |
| `cleanup.py` | `remove_small_connected_components(min_area)`, `unify_face_orientations` (BFS winding propagation) | works |
| `compute_charts.py` | Single-pass greedy cone-region growing | works but produces too many charts (~640K) |
| `compute_charts_lloyd.py` | METIS-seeded Lloyd refinement | works but converges to 5000-chart minimum |
| `compute_charts_iterative.py` | Iterative cone-collapse with **mutual-min matching** (the real cumesh algorithm) | works, converges to ~75K charts on a 1M-face mesh |
| `uv_unwrap.py` | METIS-partition → per-chunk `xatlas.add_mesh` → xatlas pack | not used after compute_charts_iterative came online |
| `bvh_bake.py` | cKDTree-on-vertices closest-point projection (BVH approximation) | works; median drift sub-voxel |
| `bake_corrected.py` | Full BVH-projected bake (cKDTree + KDTree-IDW voxel sample) | works |
| `normals.py` | `compute_vertex_normals` (area-weighted) | works |
| `voxel_trilinear.py` | `trilinear_sample_sparse` — sorted-key binary-search lookup of 8 cube corners, no dense alloc | works; 25 GB dense grid → ~MB sparse table |

`pixal3d/utils/texture_baker.py :: export_glb_with_texture` extended to
accept a `vertex_normals=` kwarg.

### Why pymeshlab can't substitute for `cumesh.simplify`

pymeshlab's `meshing_decimation_quadric_edge_collapse` with
`preservetopology=True` floors at ~1.9M faces because boundary edges block
collapses.  With `preservetopology=False` it goes below the floor but
**collapses through non-manifold edges**, merging disjoint shells onto the
same vertices — produces "spiky blob" output where the visible silhouette
is destroyed.

cumesh's `mesh.simplify` is a GPU-accelerated QEM (built on the pamo
algorithm per their README) that handles non-manifold edges *correctly*
by splitting vertices rather than collapsing through them.  We could not
port this in this session.

### Pipeline matched, results still bad

The full TRELLIS.2-equivalent pipeline we built ends with:

* 75K compute_charts chunks (vs GT's likely ~30K)
* 164K final xatlas charts (vs GT's 65K — xatlas inflates our chunks 2.2×, vs GT's ~2×)
* Atlas utilization 77.5% (vs GT's ~95%)
* Vertex normals written into the GLB → smoother geometry rendering
* Sparse trilinear voxel sampling → matches TRELLIS.2's grid_sample_3d behaviour
* cKDTree-on-vertices BVH projection → matches cumesh.cuBVH.unsigned_distance approximately
* cv2.inpaint TELEA gutter fill → matches TRELLIS.2 exactly
* Color stats *quantitatively* match GT (covered mean R/G/B 48,50,41 vs 46,48,37)

And the result is **still visually unacceptable**:
the geometry looks faceted/spiky (vs GT's smooth surfaces),
the texture looks like terrazzo (vs GT's coherent surface texture).

### Specific failure modes (not yet diagnosed)

1. **Geometry quality.**  Our decimated 1.02M-face mesh looks *much*
   blockier than GT's 0.99M-face mesh in side-by-side renders.  Same
   target face count.  Same surface roundtrip should yield similar
   quality.  Difference is the decimator — pymeshlab QEM produces a
   topologically-preserved but visually-noisier mesh than cumesh's QEM.
   Manifests as sharp dihedral angles between adjacent faces.

2. **xatlas chart inflation.**  Our chunks → xatlas → 2.2× chart count.
   GT's chunks → xatlas → ~2× chart count.  Suggests our chunks
   are topologically "noisier" (more open boundary chains within them?
   disc-genus violations?).  Did not isolate the cause.

3. **Texture coherence.**  Even with stats matching, the atlas LOOKS
   like uniform speckle rather than coherent surface regions.  Each
   chart is small enough that bilinear filtering at chart boundaries
   pulls colors from semantically-unrelated atlas neighbours.

### Probes that turned out to be dead-ends

* **METIS partitioning** for chunking (`scratch/probe_metis.py`).  Up to
  10 K balanced connected partitions; xatlas STILL produced ~580 K final
  charts regardless of npart.  Conclusion: chunk *count* doesn't matter;
  chunk *quality* (specifically: low-noise normals + clean topology)
  does.
* **Dihedral-CC chunking** for chunking (`probe_charts_dihedral.py`).
  Filter face-adjacency by dihedral threshold, then connected_components.
  Pixal3D meshes have ~50 % of face pairs above 30 ° dihedral, so the
  filtered graph fragmented into 700 K + singleton-heavy chunks.  Dead.
* **xatlas `use_input_mesh_uvs` hint** (`probe_uv_hints.py`).
  Pre-computed planar UVs per chunk + `add_mesh(..., uvs=)` +
  `use_input_mesh_uvs=True`.  Reduced chart count by 3 % vs no-hint.
  pip xatlas's hint path doesn't actually fix segmentation.
* **`pymeshlab.meshing_close_holes`** (`probe_pymeshlab_v3.py`).  Closes
  large holes too aggressively and introduces non-manifold geometry that
  defeats downstream QEM (post-close decimation does literally zero
  collapses).

### Things in the box (intermediate artefacts)

* `scratch/decimated/v2_D_repair_then_topo.ply` — the cleanest decimation
  we got from pymeshlab (1.95 M faces, post-repair + no-fill-holes).
  Floor that we kept hitting.
* `scratch/full_pipeline/clean_premesh.ply` — output of our full
  TRELLIS.2-mirror cleanup pipeline at 1.02 M faces.
* `scratch/full_pipeline/full_pipeline_uv.npz` — UV-unwrapped (vertices,
  faces, uvs) ready to feed any bake.
* `scratch/filled/raw_filled_with_attrs.npz` — raw 8.4 M-face mesh with
  small loops closed, voxel attrs attached.
* `scratch/bake_v4.glb` — the best result of the session (matching color
  stats, but still visually wrong).
* `checkpoints/1_img.npz` — original generation checkpoint (kept).

### What would actually fix this (next session priorities)

A faithful port of `cumesh.simplify`.  Everything else in the pipeline
worked in some form, but the mesh that goes into UV-unwrap and bake is
the foundation — if it's blocky/noisy, the rest cannot recover.

The cumesh GPU QEM is in `CuMesh/src/simplify.cu` — 591 lines.  It uses
the pamo parallel-edge-collapse algorithm
(https://github.com/SarahWeiii/pamo) which is *NOT* a standard QEM and
*does* handle non-manifold geometry correctly without destroying it.
A CPU/numpy port is non-trivial but is the only path I see to closing
the geometry-quality gap.

### Pragmatic alternative — give up on Apple Silicon for textured GLB

Run the textured-export step on a remote CUDA box (or via Google Colab)
and pipe the result back.  Mesh generation already works on MPS in 8
minutes; the to_glb call needs ~30 seconds on a real GPU.  This
sidesteps the entire CPU-port effort.

### Direct evidence of the decimator gap

User-supplied side-by-side wireframe render (session image #31):

* GT mesh: 0.99 M faces, but the wireframe at distance reads as a **solid
  black surface** — the triangulation is so fine and evenly distributed
  that individual edges aren't visible at normal viewing distance.  Faces
  are roughly uniform-size across the model.
* Our mesh: 1.02 M faces, but the wireframe shows **clearly-visible
  individual triangles** — large faces in the towers and walls, dense
  packing only on high-curvature features (mushrooms, vegetation, trim).

Same face count, **wildly different face-size distribution**.  pymeshlab
QEM with `preservetopology=True` is greedy on planar regions and produces
big stretched triangles wherever curvature is low — fine in synthetic
benchmarks, terrible for Pixal3D output where the "low curvature" regions
are actually load-bearing surface (the wall texture, the roof slope) and
NEED uniform face density to avoid texture stretching in those areas.

cumesh.simplify's pamo-based GPU QEM produces visually-uniform output —
this is what makes the GT atlas look "coherent" and ours look like
terrazzo.  No amount of bake / inpaint / chart-segmentation tuning can
recover from this mismatch upstream.

---

## Session 3 log — native o_voxel bridge started

Reassessment: the right next step is to stop pre-decimating / re-baking with
substitute pieces and let the closest available `o_voxel.postprocess.to_glb`
implementation run on the raw generated mesh.

Discovery:

* `/Users/pawelma/code/ai/trellis-mac/.venv/lib/python3.11/site-packages`
  contains a Metal-native stack:
  * `o_voxel/postprocess.py` with `mtldiffrast` + `cumesh.CuMesh`
  * `cumesh.metal_backend.MtlMesh`
  * `mtlbvh.MtlBVH`
  * `flex_gemm` Metal kernels
* The Pixal3D workspace venv is Python 3.12, while the native Metal
  extensions are CPython 3.11 wheels.  In-process importing fails with a
  Python ABI mismatch, so the practical integration is a subprocess bridge
  using the native Python 3.11 interpreter.

Code added:

* `pixal3d/utils/o_voxel_native_export.py` — standalone bridge that loads the
  raw mesh/voxel `.npz`, calls native `o_voxel.postprocess.to_glb`, forces
  opaque/double-sided material semantics to match upstream Pixal3D, then writes
  the GLB.
* `generate_mps.py` now tries native `o_voxel` first for any textured export,
  including `--load-mesh` checkpoints.  It no longer pre-simplifies before
  `o_voxel`; the raw mesh goes into the native pipeline so `fill_holes`,
  `cumesh.simplify`, cleanup, `uv_unwrap`, rasterization, BVH projection, and
  sparse grid sampling happen in the upstream order.
* Native defaults:
  * `--native-decimation-target 1000000` (matches TRELLIS/Pixal3D)
  * `--native-remesh` opt-in only
  * `--o-voxel-python` optional override; default search is
    `$O_VOXEL_PYTHON`, `../trellis-mac/.venv/bin/python`, then `python3.11`
* The KDTree/xatlas/Blender fallback remains available through
  `--force-texture-fallback` and is used automatically if native export fails.

Verification so far:

* `py_compile` passes for `generate_mps.py` and the bridge under both the
  Pixal3D venv and the trellis-mac Python 3.11 venv.
* `generate_mps.py --help` exposes the new native flags.
* Native imports work in the trellis-mac venv:
  `o_voxel.postprocess` reports `gpu_deps=True`, `backend=metal`.
* A tiny synthetic cube export reaches `cumesh.metal_backend.MtlMesh`, but the
  current Codex sandbox reports `[MtlMesh] No Metal device found`, so full
  native GLB export could not be validated from this tool environment.
* End-to-end `generate_mps.py --load-mesh` smoke on the same tiny cube falls
  back cleanly after the native Metal-device failure and writes a GLB through
  the xatlas/KDTree fallback path.

Follow-up after first successful native `o_voxel` render on real hardware:

* Native texture/atlas quality is much closer to GT, especially at
  `--texture-size 4096`.
* Remaining issues seen in Blender:
  1. Output was on its side.
  2. Solid preview showed many missing-face/hole artifacts.
  3. Colors differ slightly from GT.
* Root cause for orientation: native `o_voxel.postprocess.to_glb` applies a
  final `(x, y, z) -> (x, z, -y)` conversion internally.  The MPS fallback
  path had already been corrected to Pixal3D's export orientation
  `(x, y, -z)`.  Bridge now defaults to `--native-output-transform pixal3d`,
  which maps native output `(x, z, -y) -> (x, y, -z)` and flips face winding
  for the reflection.  Use `--native-output-transform o_voxel` to preserve raw
  upstream orientation for debugging.
* Geometry stats from current native outputs:
  * `output_3d_native.glb`: 934,693 verts / 991,529 faces / 711,537 boundary
    edges.
  * `output_3d_native_2.glb`: 1,060,293 verts / 938,282 faces / 935,194
    boundary edges.
  These boundary counts explain the solid-preview artifacts; this is still a
  geometry openness problem, not UV/texture.
* Initial perimeter tuning did **not** improve the holes because the Python
  `cumesh_port.fill_holes` port was too strict: it grouped raw boundary-edge
  components, so branched boundary graphs were marked "not loops" wholesale.
  CUDA cumesh first builds connected components through manifold boundary
  vertices only.  The port now mirrors that logic, and native prefill is back
  to the upstream `3e-2` default.
  * On `checkpoints/1_img_fdg_cap.npz`, corrected prefill now finds
    9,578 boundary loops, fills 9,569 of them, adds 58,466 faces, and drops
    raw boundary edges from 423,639 to 365,173 before native `to_glb`.
* Next change moved upstream to `flexible_dual_grid_to_mesh`:
  * `pixal3d/utils/mesh_extract.py` now uses a vectorized sparse-coordinate
    lookup and can cap "partial quads": if an intersected dual-grid edge has
    3 of the 4 neighboring dual voxels present, it emits the one valid
    triangle instead of dropping the whole face.  This is controlled by
    `PIXAL3D_FDG_CAP_PARTIAL_QUADS`.
  * `generate_mps.py` enables that behavior by default via
    `--fdg-cap-partial-quads`; use `--no-fdg-cap-partial-quads` to compare
    against the strict extractor.
  * `--fdg-verbose` prints counts:
    voxels / intersections / full quads / dropped quads / partial triangles /
    output faces.
  * New checkpoints include the raw FDG tensors (`fdg_coords`,
    `fdg_dual_vertices`, `fdg_intersected`, `fdg_split_weight`) when saved
    after generation.  Such checkpoints can be re-extracted without re-running
    the model via `--load-mesh ... --reextract-mesh-from-fdg`.
  * Pipeline wrappers now copy those FDG tensors onto `MeshWithVoxel`; earlier
    generated checkpoints named `*_fdg_cap.npz` may still not contain them.

---

## Session 4 log — band-aid loop on Metal cumesh, then committed to pamo CPU port

This session chased the puddle-shaped missing-face artifacts the user kept
reporting on textured GLB output.  Several band-aid patches were tried and
each peeled away one symptom while revealing the next.  Final conclusion:
the Metal port of `cumesh.simplify` (precompiled binary, no source) skips
or breaks the winding-flip-rejection check that CUDA `simplify.cu:128`
enforces, producing inverted faces and degenerate collapses that drive
the mesh below the target face count and punch real geometry out of the
surface.  The only real fix is a faithful CPU/MPS port of cumesh's pamo
QEM.  We started that port — Phases 0–3 of 7 are landed and verified;
the rest is queued for the next session.

### Band-aids tried (and what they each revealed)

1. **`OUTPUT_TRANSFORMS["pixal3d"]` matrix corrected** to `(−x, y, −z)`
   (180° rotation around world Y, det=+1) — the model was sideways AND
   mirrored vs GT.  Patched + verified upright in Blender.  No regression.
2. **`orient_ccs_outward`** — per-CC centroid-outward test as a final
   pass after `unify_face_orientations`.  Flipped 38% of CCs / 8.8% of
   faces on the first run.  Reduced visible holes substantially but
   couldn't reliably orient flat patches (centroid lies in the plane)
   or thin tendrils.  Eventually reverted.
3. **CPU-port `repair_non_manifold_edges` monkey-patched onto
   `_MeshBackend.repair_non_manifold_edges`** — hypothesised the Metal
   port was over-splitting vertices.  Probe disproved: CPU port produces
   *identical* +1.92 M vertex splits as Metal, because the algorithm
   itself splits aggressively when the input has many non-manifold
   edges (which the FDG output always has).  Patch kept active (no
   harm) but not the bug we thought.
4. **`trimesh.repair.fix_normals(multibody=True)`** added at the end of
   `_write_geometry_stage` and after `to_glb` in main.  Per-CC outward
   detection via signed volume / ray-cast.  Helped less than expected:
   only 6 face flips on the simplify=1M run because the inverted faces
   sit *inside* otherwise-correct CCs (`unify_face_orientations` had
   already aligned each CC consistently — to the wrong direction in
   the bad cases — and `fix_normals` operates at CC granularity).
5. **`_per_face_winding_fix`** — iterative neighbour-voting flip
   detector that operates at face granularity.  Only found 6 conflict
   edges on the test mesh; not the right tool either.  Drafted but
   ultimately wasn't the fix.
6. **`remove_small_connected_components` no-op'd** — probe showed
   cleanup_1's `remove_small_cc(1e-5)` drops ~1 M faces (35 % of
   post-simplify mesh!) because the over-split surface looks like dust
   to the area filter.  Skipping it produces clean geometry but
   `fill_holes` then fan-fills the new boundaries with ~2.5 M arbitrary
   triangles whose UVs explode into texture confetti.  Cleaner geometry,
   broken texture.
7. **Less-aggressive simplify target / skip-decimation** — produced
   clean geometry (8.4 M faces) but the 347 MB texture-confetti output
   the user rejected.

### Diagnostic probes built

`pixal3d/utils/o_voxel_native_export.py` got
`--debug-stages-dir / --debug-raw-stages / --debug-stages-only` so we
can write a GLB after each native cleanup substep
(`00_input → 01_after_fill_holes → 02_after_simplify_3x →
03_cleanup1a_after_dedup → 04_cleanup1b_after_repair_nme →
05_cleanup1c_after_remove_small_cc → 06_cleanup1d_after_fill_holes →
07_after_simplify_target → 08_after_cleanup_2 →
09_after_unify_orientations`).  This is what isolated the simplify
step as the root.

### The real diagnosis (confirmed against `simplify.cu`)

The CUDA reference at `CuMesh/src/simplify.cu:128` rejects edge
collapses whose new face normal points opposite to the old:

```cuda
if (old_normal.dot(new_normal) < 0.0f) {
    return false; // invalid (flipped)
}
```

Empirically the Apple Silicon Metal port doesn't enforce this — on the
production target (decimation_target = 1 M) the output mesh comes back
with 859 K faces (below target) and visible chunks of surface
collapsed into nothing.  Every post-simplify band-aid is fighting an
upstream invariant violation.

### Decision: port cumesh pamo to CPU/MPS

Path 4-then-1 from a multi-option deliberation: revert the band-aids
first (to baseline against upstream's algorithm) and then bypass the
Metal kernel for `simplify` with our own port.  Phases 4–7 remaining;
detailed plan + status lives in **`PAMO_PORT_PLAN.md`** (created this
session).

### Code state at session end

* `pixal3d/utils/pamo_simplify.py` — new (354 lines), Phases 0-3 done
  and verified.  Stubs for Phases 4-5.
* `pixal3d/utils/o_voxel_native_export.py` — `_patch_simplify` wired
  (routes `_MeshBackend.simplify` → `pamo_simplify.simplify`, currently
  a no-op stub).  `_patch_repair_non_manifold_edges` kept active.
  `_patch_remove_small_connected_components` defined but **disabled**
  in `main()` (we want pamo tested without compensating band-aids).
  `_per_face_winding_fix` + `trimesh.fix_normals(multibody=True)`
  still wired in `_write_geometry_stage` and main `to_glb`
  post-processing; these become no-ops once pamo lands correctly
  (no flipped faces to fix), can be removed in Phase 6 validation.
* `pixal3d/utils/cumesh_port/` — clean.  `orient_ccs_outward` was
  added and then reverted.
* `OUTPUT_TRANSFORMS["pixal3d"]` matrix change to `(−x, y, −z)` stays
  in.

### Phase 3 spike results (FDG checkpoint, 12.18 M edges)

* `get_vertex_face_adjacency`: 0.93 s.  Output matches
  `cumesh_port._build_edges` exactly (12,179,526 edges; 423,639
  boundary edges; 904,929 non-manifold).
* `get_qem`: 0.45 s.  v^T·Q·v at each vertex's own position = 0 to
  float64 epsilon (the defining invariant).
* `get_edge_collapse_cost` with flip-reject + λ_edge + λ_skinny:
  22.7 s.  94.6 % of edges finite; 5.4 % rejected (mostly flip or
  degree-cap on non-manifold vertices).

---

## Session 5 log — divergence hunt + downstream investigation + tex-VAE precision

(2026-05-20 → 2026-05-21.  Companion docs created this session:
`DIVERGENCE_FINDINGS.md` for the full divergence-hunt report,
`NEXT_STEPS.md` for the prioritised candidate-directions list,
`LIB_PORT_NOTES.md` for preserved chat context.  WIP commit `e74ba5d`.)

### Goal

Phase 6 of PAMO_PORT_PLAN.md called for CUDA-vs-MPS divergence analysis
to localize the source of the persistent quality gap.  This session
executed that analysis end-to-end (on a rented RTX 4090), instrumented
the pipeline, ran controlled ablations, then pivoted to the texture-VAE
precision investigation.

### Infrastructure added

CLI / env-var surface for the divergence-hunt and downstream tuning:

| Switch | Purpose |
|---|---|
| `--fov FLOAT` | Manual FOV, bypasses MoGe-2 entirely (mirrors `inference.py`) |
| `--load-fixture-07 PATH` | Load a `07_run_output.pt` fixture and replay only the post-pipeline (skip the slow ~8 min generation) — isolation test for downstream damage |
| `PIXAL3D_DUMP_FIXTURES=DIR` | Per-stage fixture dump (was already in place; added per-step sampler hooks: dumps `pred_x_t` + `pred_x_0` after every one of 12 denoising steps for sparse-structure + shape-SLat + HR-shape-SLat + texture-SLat samplers) |
| `PIXAL3D_STOP_AFTER=<stage>` | Clean early-exit after a named fixture is dumped (e.g. `=03a_shape_slat`) |
| `PIXAL3D_SDPA_BACKEND=math|efficient|flash` | Pin SDPA backend (turned out a no-op on MPS as of torch 2.12) |
| `PIXAL3D_FP32_MODELS=<comma-list>` | Selective fp32 weight upcast per pipeline submodel + matching input-feats autocast wrapper on `decode_tex_slat` / `decode_shape_slat` |
| `PIXAL3D_FILL_HOLES_PERIMETER=<float>` | Override `_MeshBackend.fill_holes` perimeter threshold via CPU port |
| `PIXAL3D_SMALL_CC=noop|<float>` | Override `_MeshBackend.remove_small_connected_components` (noop or custom min_area) via CPU port |
| `PIXAL3D_PRECLEAN_NME=1` | Run `cumesh_port.repair_non_manifold_edges` on the FDG mesh BEFORE `to_glb` is called (drafted; not yet tested as of this writing) |

`run_mps_capture.sh` patched to respect caller-set `ATTN_BACKEND` /
`SPARSE_ATTN_BACKEND` (was previously unconditional `export …=sdpa`).

### Captures collected

CUDA rentals + Mac runs at seed 42, image `assets/images/1_img.png`,
resolution 1024.  All directories under `/Users/pawelma/code/ai/fixtures/`:

| Dir | Side | MoGe | Per-step | Notes |
|---|---|---|---|---|
| `cuda` | A4000 | on | — | Original capture; only full set including stage 08 |
| `mps` | M5 | on | — | Original capture; missing stage 08 (Phase 6 blocker) |
| `fixtures_cuda_moge` | 4090 | on | yes | Early-exit at 03a, per-step sweep |
| `fixtures_cuda_nofov` | 4090 | off | yes | Early-exit at 03a, MoGe bypassed |
| `mps_moge` | M5 | on | yes | Early-exit at 03a |
| `mps_nofov` | M5 | off | yes | Early-exit at 03a |
| `mps_nofov_sdpa_{math,efficient,flash}` | M5 | off | yes | SDPA backend sweep |
| `mps_nofov_naive` | M5 | off | yes | `ATTN_BACKEND=naive` (unfused) |
| `mps_full_nofov` | M5 | off | full | End-to-end MoGe-off; produced visible-quality GLB |
| `mps_full_nofov_undecimated` | M5 | off | full | `--skip-decimation` (covered-in-confetti result) |
| `mps_replay_cuda07` | M5 | (CUDA-mesh) | n/a | CUDA mesh fed through Mac post-pipeline |
| `mps_replay_cuda07_{fh01,smcc1e7,combo,smccnoop,fallback}` | M5 | (CUDA-mesh) | n/a | Post-pipeline-parameter sweep |
| `mps_full_nofov_tex_fp32` | M5 | off | up to 06 | tex_slat_decoder upcast to fp32 (running at session end) |

### What got proven and ruled out

For the per-step DiT divergence (stage 02 sparse-structure sampler):

| Hypothesis | Result |
|---|---|
| MoGe-2 produces different camera params | YES (~5e-4 to 1.5e-3 Δ on `camera_angle_x` / `distance`), but bypassing MoGe (`--fov 0.5`) didn't close the stage-02 topology gap, only shrank it 11× (Δ 11 voxels → Δ 1 voxel) |
| Same-seed RNG diverges between CUDA / MPS | Inconsistent with data: divergence appears at step 0, before any sampler-internal RNG |
| bf16 weights cause MPS reduction precision loss | Provisionally falsified: probe showed all model parameters already report `torch.float32` after `from_pretrained` on MPS.  Force-all-fp32 crashed Metal at NDArrayMatrixMultiplication |
| MPS's fused SDPA backend is the culprit | All three `sdpa_kernel(MATH|EFFICIENT|FLASH)` variants produced **bit-identical** MPS output — the context manager is a no-op on MPS as of torch 2.12 |
| Fused vs unfused attention is the culprit | `ATTN_BACKEND=naive` made things *slightly worse*, not better |

What survived: **pervasive numerical drift across the DiT forward pass**
(LayerNorm/RMSNorm reductions + GEMM rounding order + RoPE sin/cos
accuracy on MPS).  Not a single broken kernel.  ~0.3 max-abs-Δ on
`pred_x_0` at step 0 with byte-identical input.

For the downstream cleanup:

The CUDA-stage-07 → Mac-post-pipeline replay produced a GLB with holes
in arched openings even though the input mesh was CUDA-clean.  Damage
chain: pamo collapses → `repair_nme` adds +2.5M verts → `small_cc(1e-5)`
culls 2.19M as dust → `fill_holes(3e-2)` only recovers ~370k pinpricks
→ ~1M faces of real surface permanently gone.

Six tuning variants (`PIXAL3D_FILL_HOLES_PERIMETER` ∈ {default, 0.1, 0.3},
`PIXAL3D_SMALL_CC` ∈ {default, 1e-7, noop}, combos, `--force-texture-
fallback`) all land within 2-3× of each other on hole metrics:

| Variant | Boundary edges | Loops | Big (>5cm) | Huge (>15cm) | Top perim |
|---|---|---|---|---|---|
| A (CUDA reference) | 626K | 51,707 | 7,960 | 1,880 | 5.41m |
| B (Mac full, MoGe-off) | 766K | 95,541 | 6,378 | 1,294 | 9.15m |
| C (CUDA→Mac post defaults) | 747K | 101,563 | 5,649 | 947 | 3.37m |
| D (PIXAL3D_FILL_HOLES_PERIMETER=0.1) | 943K | 149,358 | 6,317 | 952 | **2.20m** |
| E (PIXAL3D_SMALL_CC=1e-7) | 1.13M | 225,611 | 5,420 | **899** | 2.37m |
| F (E + D combo) | 1.11M | 211,856 | 6,113 | 922 | 2.44m |
| G (smcc=noop + fh=0.3) | 1.07M | 200,315 | 7,437 | 942 | 2.35m |
| H (--force-texture-fallback) | 2.70M | (different topo)| 7,231 | 2,243 | (different topo) |

Practical recommendation: **`PIXAL3D_SMALL_CC=1e-7`** preserves the most
fine detail (lantern hood, decorative spires) at the cost of more
small-perimeter noise; the absolute worst-hole metric stays in the same
range as default.  No single-parameter fix solves the structural
problem.

### Texture-VAE finding (motivation for the in-flight test)

Element-wise diff of `06_tex_slat_decoded.pt` (CUDA vs MPS, MoGe-on
captures) over the 28% of voxels where coords agree:

* max abs diff: **1.21** (out of value range ~1.1)
* mean abs diff: **0.12** (~10% of value range)
* per-channel mean abs diff: 0.09 / 0.10 / 0.09 / **0.23 / 0.16** / 0.05
  (channels 3 and 4 worst — likely metallic / roughness PBR)

This is the "muted/blurry texture" half of the user-reported quality
gap.  **NOT bf16 noise** — fundamental disagreement.  Hypothesis being
tested: precision drift inside the tex-VAE decoder (the disk weights
were `tex_dec_next_dc_f16c32_fp16.safetensors`, though `next(parameters()).
dtype` reports fp32 after load — possibly mixed-dtype state in buffers
or non-parameter tensors).

Test in flight: `PIXAL3D_FP32_MODELS=tex_slat_decoder` + input-feats
autocast wrapper, full pipeline up to stage 06.  Compare 06 fixture
against baseline.  If divergence shrinks, this is the precision fix
path; if not, the texture VAE has algorithm-level mismatch.

### What's next

Per `NEXT_STEPS.md` (this session's planning doc, ordered by ROI):

1. **Texture-VAE precision focus** (in flight) — `PIXAL3D_FP32_MODELS=
   tex_slat_decoder` test, compare 06_tex_slat_decoded against baseline.
2. **Pre-clean FDG NMEs before pamo runs** — `PIXAL3D_PRECLEAN_NME=1`
   patch drafted, not yet tested.  Run `cumesh_port.repair_nme` on the
   FDG mesh before `to_glb` is called.
3. `flexible_dual_grid_to_mesh` CPU-fallback audit
4. Phase 7 — pamo on MPS device for speed
5. Per-layer DiT instrumentation
6. CUDA cumesh `repair_nme` vs CPU port audit

### Key new heuristics for future sessions

* **MPS `torch.nn.attention.sdpa_kernel()` is effectively a no-op** as
  of torch 2.12 — all three backend pins produce bit-identical output.
  Don't waste time on it.
* **Naive (pure-Python) attention is slightly WORSE than fused SDPA**
  on MPS for our model.  Not a useful ablation either.
* **Forcing global fp32 weights crashes Metal**
  (`NDArrayMatrixMultiplication`-destination/accumulator dtype
  mismatch).  Targeted per-submodel `.float()` + input-autocast wrapper
  (`PIXAL3D_FP32_MODELS`) is the workaround.
* **Tuning post-pipeline parameters has marginal effect.**  All
  variants land in the same metric range.  The structural cause is
  upstream of the parameters.
* **Three checkpoint levels exist** for different tuning needs:
  per-stage fixtures (PIXAL3D_DUMP_FIXTURES) for model-decoder tuning,
  `--save-mesh` for bake/UV/cleanup tuning, `--load-fixture-07` for
  post-pipeline replay.  Future debug runs should ideally chain all
  three so we don't pay the ~30 min cost twice.


## Session 7 — 2026-05-22 — sieve diagnosis + mtlmesh integration

### Headline

The geometry-quality bug is a **SIEVE** (thousands of small surface
holes) — not a simplify cost-function failure as previously hypothesized.
Inside-out Blender shots show CUDA produces a sealed shell, all Mac
variants (pamo, mtlmesh-simplify, full Pedro Metal chain) produce a
sieve.  Two precise suspects remain, decisively testable.

### Chronology

1. **Picked up handoff from session 6** (commit `d3e2c5c`).  Prior
   diagnosis: divergence in DiT samplers, attributed to flash_attn vs
   sdpa attention reduction-order drift compounding through samplers.
   Recommended path: port `philipturner/metal-flash-attention`.

2. **Ruled out metal-flash-attention as the answer.**  Synthetic
   Q/K/V probe (`scratch/probe_mlx_vs_sdpa_attention.py`) measured
   MPS sdpa drift at ~1e-6 relative vs fp64 truth — the fp32
   precision floor.  Per-op attention drift can't credibly compound
   to 3.06 mean-abs feature diff from a 1e-6 floor.  Attention is
   not the bottleneck.

3. **Found three Pedro forks not on disk before**: `mtlmesh` (Metal
   cumesh), `mtlgemm` (Metal GEMM), `mtlbvh` (Metal BVH).  Also found
   `pixal3d-mlx` (community MLX-native port of Pixal3D, 8 commits ahead
   of upstream, cites our `feat/apple-silicon-port` branch as a
   precursor).

4. **Ran pixal3d-mlx end-to-end on `1_img.png` seed=42** for
   apples-to-apples comparison.  Result: MLX port produces visibly
   *worse* output than our PyTorch/MPS port (more artifacts, worse
   textures, regressed transparency-by-camera-angle bug we already
   fixed via session-6 unify port).  Framework switch is not the
   answer.

5. **User flagged the showcase-mesh vs our-CUDA-reference distinction.**
   The published Pixal3D website mesh has details (window muntins, etc)
   that our own CUDA A4000 run does not.  Quality-ceiling gap (likely
   `1536_cascade` vs our `1024_cascade`), separate from Mac-vs-CUDA
   divergence.  Re-anchored comparisons.

6. **Audited `mtlmesh/src/metal/simplify.metal` vs
   `CuMesh/src/simplify.cu`** (subagent, opus).  Verdict
   ADOPT-AS-IS: Pedro's port is a faithful mechanical translation of
   CUDA.  The previously-flagged "missing winding-flip-reject at
   `simplify.cu:128`" was a misdiagnosis — the check is present at
   `simplify.metal:56` verbatim.

7. **Integrated mtlmesh into Pixal3D venv.**
   - Installed Metal Toolchain (macOS 26 separated it from Xcode).
   - Cloned and `pip install --no-build-isolation` of mtlmesh,
     mtlbvh, mtlgemm.
   - Added `PIXAL3D_SIMPLIFY=metal` env opt-out to `_patch_simplify`
     in `o_voxel_native_export.py`.  Default remains `pamo` for safe
     rollback.

8. **Ran with `PIXAL3D_SIMPLIFY=metal`** on `1_img.png` seed=42.
   Result: wall time 24:55 → 12:19 (50% faster, pamo's 13:25
   post-pipeline became ~30s native Metal).  repair_nme vertex
   inflation dropped from +2.5M to +19K (Metal simplify produces a
   near-manifold mesh; repair_nme has almost nothing to do).
   Quality: visually equivalent — not better, not worse.  Same shards
   and artifact density as pamo.

9. **Added `PIXAL3D_REPAIR_NME=metal` env opt-out**, ran full Metal
   chain (`PIXAL3D_SIMPLIFY=metal PIXAL3D_REPAIR_NME=metal
   PIXAL3D_UNIFY_FACE_ORIENTATIONS=metal`).  Same wall time, same
   quality.  The other patches were near-zero cost — all the speedup
   came from simplify.

10. **User's inside-out Blender shots revealed the sieve.**  CUDA
    shell from inside: sealed, smooth walls.  All Mac variants from
    inside: thousands of small holes (orange Blender hole-boundary
    outlines).  This is the dominant geometry bug; the visible
    "jagged edges / missing triangles" we'd been chasing are the
    outside view of the sieve.

### Files added/changed (committable)

- `pixal3d/utils/o_voxel_native_export.py`:
  - `_patch_simplify` now respects `PIXAL3D_SIMPLIFY=metal` (early
    return, leaves native `cumesh.CuMesh.simplify` in place).
  - `_patch_repair_non_manifold_edges` now respects
    `PIXAL3D_REPAIR_NME=metal` (same pattern).
  - Both default to existing CPU-port behaviour; opt-in only.
- `NEXT_STEPS.md`:
  - Session-7 update block reflecting end-state findings.
  - Ordered next list reorganized around the sieve hypothesis.
  - Suspect list split into geometry/texture symptoms.
  - Retired list expanded.

### Off-tree assets used (not committable; for next-session context)

- `/Users/pawelma/code/ai/CuMesh/` — CUDA cumesh source (gold standard).
- `/Users/pawelma/code/ai/mtlmesh/` — Pedro's Metal port (audited
  faithful for simplify; covers full mesh chain in Metal).
- `/Users/pawelma/code/ai/mtlbvh/` — Pedro's Metal BVH (dep of cumesh).
- `/Users/pawelma/code/ai/mtlgemm/` — Pedro's Metal GEMM (dep of cumesh).
- `/Users/pawelma/code/ai/pixal3d-mlx/` — community MLX-native port,
  Phase 7 (parity/quality) not started.  Useful as a reference for
  alternative ports of specific components, not as a wholesale switch.
- `/Users/pawelma/code/ai/trellis-mac/` — pedronaugusto's TRELLIS-2
  Apple Silicon port; uses the same Metal kernel infra.

### Concrete next move (start of next session)

Run the **decisive sieve test**:

1. Write a one-shot script to convert `fixtures/cuda/08_to_glb_geometry.pt`
   to the `.npz` format expected by `generate_mps.py --load-mesh`
   (keys: vertices, faces, coords, attrs, origin, voxel_size,
   resolution, optionally fdg_*).
2. Invoke `generate_mps.py --load-mesh <converted.npz> ...` with the
   same seed/params.  This bypasses our mesh extraction entirely.
3. Inspect output:
   - **Sealed** → bug is in `pixal3d/utils/mesh_extract.py` CPU
     `flexible_dual_grid_to_mesh`.  Audit edge cases vs CUDA o_voxel.
   - **Sieve** → bug is in Mac cleanup chain regardless of input
     quality.  Bisect stages on the same pre-extracted CUDA mesh.

### Key new heuristics

- **`mx.fast.scaled_dot_product_attention` is already at the kernel-fusion
  ceiling on M3 Ultra.**  Pedro's `flex_gemm_sparse_attn` and
  `mx.compile` were tried by the MLX port author with negative results
  (+14% slower and +4% slower respectively, no quality change).  Don't
  retest.
- **PAMO_PORT_PLAN.md's `simplify.cu:128` "missing check" diagnosis
  was wrong.**  Pedro's port at `simplify.metal:56` has the check.
  Whenever historical session notes flag "Metal port skips X", verify
  against source before relying.
- **Pamo and Pedro's Metal simplify produce the same sieve.**  The
  geometry destruction is NOT a simplify-cost-function problem.  Any
  future "fix simplify to close holes" work should be considered
  retired.
- **Texture fuzz is independent of geometry sieve.**  Don't try to
  fix textures by fixing geometry or vice versa; they are separate
  hunts with separate suspects.
