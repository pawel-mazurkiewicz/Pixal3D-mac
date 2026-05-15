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

### 9. Export rotation
The original `EXPORT_ROTATION_ROWS` matrix was a bogus 180° rotation around
the `(0, 1, -1)` axis: it negated X and swapped Y↔Z with sign flips,
putting the mesh upside-down AND mirrored.

Diagnosed by loading the GLB into Blender via the BlenderMCP addon
(`mcp__blender__execute_blender_code`) and checking the bbox + heavy-mass
distribution along the tall axis. The mesh's *pre-rotation* AABB has Y as
the tallest axis (extent 0.88), confirming **Pixal3D outputs Y-up
natively** (same as glTF). No rotation is needed.

**Fix:** set `EXPORT_ROTATION_ROWS` to identity.

### 10. cube_project UV overlap (NOT YET FULLY SOLVED — see "Next" below)
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
