"""UV-unwrap a triangle mesh via Blender.

Invoked as a subprocess from ``pixal3d.utils.texture_baker.uv_unwrap_blender``::

    blender --background --factory-startup \\
        --python _blender_uv_unwrap.py -- IN.npz OUT.npz \\
        [--voxel-size 0] [--target-count 0] [--uv-method cube]

Design choices:

- **Geometry passthrough by default.** Voxel remesh (the most attractive
  way to get predictable face counts) destroys non-watertight inputs by
  rebuilding them as if they were closed volumes — Pixal3D's mesh
  extractor outputs open shells and non-manifold edges, so voxel remesh
  turns it into swiss cheese.  Confirmed by Blender devs at T70925 and
  the official Remesh modifier docs.  We therefore default to passing
  the input mesh through unchanged.  Users who want fewer faces can pass
  ``--voxel-size`` (acceptable for watertight inputs) or ``--target-count``
  (Decimate-Collapse, topology-preserving but limited reduction).
- **cube_project + UV renormalisation.** ``smart_project`` on Pixal3D
  meshes produces hundreds of sub-pixel charts and ``pack_islands(scale=True)``
  in Blender 5.x does not scale them up to fill the atlas (utilisation
  collapses to 0.1%).  ``cube_project`` writes 6 axis-aligned charts in one
  call with high utilisation regardless of face count.  Its native output
  tiles across a 4x3 UV region rather than [0,1], so we renormalise to
  [0,1] preserving aspect ratio before saving.  Minor seam distortion at
  chart boundaries is acceptable for texture *baking* (each texel still
  uniquely maps to a 3D point, which is all the KDTree sampler needs).

Input  (IN.npz) :  vertices: (V, 3) float32,  faces: (F, 3) int32
Output (OUT.npz):  vertices: (V', 3) float32  — post-remesh + UV-split
                   faces:    (F', 3) int32    — indices into vertices
                   uvs:      (V', 2) float32  — one UV per vertex, in [0,1]
"""

import argparse
import sys

import bpy
import numpy as np


def main() -> None:
    if "--" not in sys.argv:
        raise SystemExit("expected arguments after `--`")
    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    parser.add_argument(
        "--voxel-size", type=float, default=0.0,
        help=(
            "Voxel size for REMESH modifier (world units).  0 (default) "
            "disables — recommended for non-watertight meshes like Pixal3D "
            "output.  > 0 rebuilds the surface at this resolution; only "
            "useful for closed/manifold inputs."
        ),
    )
    parser.add_argument(
        "--target-count", type=int, default=0,
        help=(
            "Target face count for iterative Decimate-Collapse, applied "
            "before UV unwrap.  0 (default) disables.  Best-effort; on "
            "topologically-constrained meshes the actual count may stay well "
            "above the target (Pixal3D meshes bottom out at ~2.7M)."
        ),
    )
    parser.add_argument(
        "--fill-holes-sides", type=int, default=12,
        help=(
            "Close holes whose boundary has at most this many edges.  "
            "Pixal3D's flexible-dual-grid extractor leaves pinprick holes "
            "(typically 3-8 edges) wherever it couldn't close a cell — "
            "the official CUDA pipeline calls cumesh.fill_holes(); we run "
            "Blender's mesh.fill_holes(sides=N) instead.  Set to 0 to skip."
        ),
    )
    parser.add_argument(
        "--uv-method", choices=["cube", "smart"], default="cube",
        help=(
            "'cube' (default): cube_project, 6 axis-aligned charts, high "
            "atlas utilisation.  'smart': angle-based smart_project, "
            "smoother charts but worse utilisation in Blender 5.x."
        ),
    )
    parser.add_argument("--angle", type=float, default=89.0,
                        help="angle_limit for smart_project (degrees)")
    args = parser.parse_args(argv)

    data = np.load(args.in_path)
    vertices = np.ascontiguousarray(data["vertices"], dtype=np.float32)
    faces = np.ascontiguousarray(data["faces"], dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise SystemExit(f"expected triangle faces, got shape {faces.shape}")

    bpy.ops.wm.read_factory_settings(use_empty=True)

    mesh = bpy.data.meshes.new("m")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()

    obj = bpy.data.objects.new("o", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    print(f"[blender] input: {len(faces):,} faces", flush=True)

    # Cleanup: merge near-duplicate verts and dissolve degenerate faces.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-6)
    bpy.ops.mesh.dissolve_degenerate()
    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[blender] after cleanup: {len(obj.data.polygons):,} faces", flush=True)

    if args.voxel_size > 0:
        mod = obj.modifiers.new("Remesh", type="REMESH")
        mod.mode = "VOXEL"
        mod.voxel_size = args.voxel_size
        mod.adaptivity = 0
        bpy.ops.object.modifier_apply(modifier=mod.name)
        print(f"[blender] after voxel_remesh(vs={args.voxel_size}): "
              f"{len(obj.data.polygons):,} faces", flush=True)
    if args.target_count > 0 and len(obj.data.polygons) > args.target_count:
        prev = len(obj.data.polygons) + 1
        for it in range(6):
            cur = len(obj.data.polygons)
            if cur <= args.target_count or cur >= prev:
                break
            prev = cur
            ratio = max(args.target_count / cur, 0.05)
            mod = obj.modifiers.new("Dec", type="DECIMATE")
            mod.decimate_type = "COLLAPSE"
            mod.ratio = ratio
            mod.use_collapse_triangulate = True
            bpy.ops.object.modifier_apply(modifier=mod.name)
            print(f"[blender] decimate iter {it}: {cur:,} -> "
                  f"{len(obj.data.polygons):,} faces (ratio={ratio:.4f})",
                  flush=True)

    # Close pinprick holes left by Pixal3D's CPU mesh extractor.  Run AFTER
    # decimation so we only iterate boundary edges on the reduced mesh.
    # We use bmesh directly instead of bpy.ops.mesh.fill_holes — the GUI
    # operator goes through Blender's undo stack and event loop, making it
    # 10-100x slower on million-face meshes (hung indefinitely on 2.7M in
    # tests).  bmesh.ops.holes_fill is the same algorithm without that
    # overhead.  The face-count cap is a hedge in case someone still hits
    # a pathological case.
    FILL_HOLES_FACE_LIMIT = 5_000_000
    if args.fill_holes_sides > 0:
        n_now = len(obj.data.polygons)
        if n_now > FILL_HOLES_FACE_LIMIT:
            print(f"[blender] skipping fill_holes: {n_now:,} faces > "
                  f"{FILL_HOLES_FACE_LIMIT:,} budget (pass "
                  f"--fill-holes-sides 0 to silence)", flush=True)
        else:
            import bmesh
            import time
            t0 = time.time()
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            boundary = [e for e in bm.edges if e.is_boundary]
            n_boundary = len(boundary)
            if boundary:
                bmesh.ops.holes_fill(
                    bm, edges=boundary, sides=int(args.fill_holes_sides),
                )
            bm.to_mesh(obj.data)
            bm.free()
            print(f"[blender] fill_holes(sides={args.fill_holes_sides}): "
                  f"{n_boundary:,} boundary edges -> "
                  f"{len(obj.data.polygons):,} faces in "
                  f"{time.time() - t0:.1f}s", flush=True)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(
        quad_method="BEAUTY", ngon_method="BEAUTY",
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh = obj.data

    n_verts = len(mesh.vertices)
    flat_pos = np.empty(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", flat_pos)
    positions = flat_pos.reshape(n_verts, 3)

    n_loops = len(mesh.loops)
    loop_vert = np.empty(n_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vert)

    n_poly = len(mesh.polygons)
    poly_starts = np.empty(n_poly, dtype=np.int32)
    poly_totals = np.empty(n_poly, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", poly_starts)
    mesh.polygons.foreach_get("loop_total", poly_totals)
    if not np.all(poly_totals == 3):
        raise SystemExit(
            "Blender returned a non-triangle polygon after triangulate"
        )

    if args.uv_method == "cube":
        # Manual 6-chart cube projection.  bpy.ops.uv.cube_project in
        # Blender 5.x just does flat planar projection per face — all six
        # cube directions land in the same [-0.5, 0.5]^2 region, so every
        # texel ends up shared between faces of contradictory 3D positions.
        # That's why the previous bake showed roof content on the bottom
        # of the model: the +Y chart and -Y chart wrote to the same texels.
        # Here we project each face onto its dominant cube axis ourselves
        # and offset each direction into its own 1/3 x 1/2 atlas slot.
        poly_normals = np.empty(n_poly * 3, dtype=np.float32)
        mesh.polygons.foreach_get("normal", poly_normals)
        poly_normals = poly_normals.reshape(n_poly, 3)

        abs_n = np.abs(poly_normals)
        dom_axis = np.argmax(abs_n, axis=1)
        dom_sign = np.sign(
            np.take_along_axis(poly_normals, dom_axis[:, None], axis=1)
        ).squeeze().astype(np.int8)
        # Bucket: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
        bucket = (dom_axis * 2 + (dom_sign < 0).astype(np.int64))

        # Each polygon is a triangle with three sequential loops.
        loop_to_poly = np.arange(n_loops) // 3
        loop_dom = dom_axis[loop_to_poly]
        loop_bucket = bucket[loop_to_poly]

        # Project each loop's 3D position onto the perpendicular plane of
        # its face's dominant axis.
        loop_pos = positions[loop_vert]
        proj = np.empty((n_loops, 2), dtype=np.float32)
        # dom=0 (X): project (Y, Z); dom=1 (Y): project (X, Z); dom=2 (Z): project (X, Y)
        axes_for_dom = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        for d in range(3):
            mask = loop_dom == d
            if not mask.any():
                continue
            ax = axes_for_dom[d]
            proj[mask, 0] = loop_pos[mask, ax[0]]
            proj[mask, 1] = loop_pos[mask, ax[1]]

        # Proportional packing: each chart's atlas area scales with its loop
        # count, so heavily-populated buckets get more texels per face.
        # Pixal3D's image-to-3D produces extremely uneven distributions —
        # typically ~90% of faces are in 3 of the 6 cube directions (the
        # half facing away from the input image's camera).  A fixed 1/6
        # layout would give those big buckets <1 texel/face at 2048^2, so
        # most of the surface bakes black.  Here we give each bucket an
        # atlas slot whose area = atlas_area * loop_count / total_loops.
        loops_per_bucket = np.array(
            [int((loop_bucket == b).sum()) for b in range(6)],
            dtype=np.float64,
        )
        non_empty = loops_per_bucket > 0
        order = np.argsort(-loops_per_bucket)
        big_idx = [b for b in order[:3] if non_empty[b]]
        small_idx = [b for b in order[3:] if non_empty[b]]
        big_total = loops_per_bucket[big_idx].sum() if big_idx else 0.0
        small_total = loops_per_bucket[small_idx].sum() if small_idx else 0.0
        grand = big_total + small_total
        big_row_h = (big_total / grand) if grand > 0 else 1.0
        small_row_h = 1.0 - big_row_h
        # Layout: a "big" row across the top, a "small" row across the bottom.
        # Within each row, widths are proportional to bucket loop counts.
        slot_rect = [None] * 6  # (x, y, w, h)
        cx = 0.0
        for b in big_idx:
            w = loops_per_bucket[b] / big_total
            slot_rect[b] = (cx, 0.0, w, big_row_h)
            cx += w
        cx = 0.0
        for b in small_idx:
            w = loops_per_bucket[b] / small_total
            slot_rect[b] = (cx, big_row_h, w, small_row_h)
            cx += w

        MARGIN = 0.002
        new_uv = np.zeros((n_loops, 2), dtype=np.float32)
        chart_stats = []
        for b in range(6):
            if slot_rect[b] is None:
                continue
            mask = loop_bucket == b
            if not mask.any():
                continue
            sx, sy, sw, sh = slot_rect[b]
            pm = proj[mask]
            pmin = pm.min(0)
            pmax = pm.max(0)
            ext_x = max(pmax[0] - pmin[0], 1e-9)
            ext_y = max(pmax[1] - pmin[1], 1e-9)
            target_w = max(sw - 2 * MARGIN, 1e-6)
            target_h = max(sh - 2 * MARGIN, 1e-6)
            # Uniform scale preserves aspect (no texture distortion).
            scale = min(target_w / ext_x, target_h / ext_y)
            used_w = ext_x * scale
            used_h = ext_y * scale
            extra_x = (target_w - used_w) / 2.0
            extra_y = (target_h - used_h) / 2.0
            new_uv[mask, 0] = (pm[:, 0] - pmin[0]) * scale + sx + MARGIN + extra_x
            new_uv[mask, 1] = (pm[:, 1] - pmin[1]) * scale + sy + MARGIN + extra_y
            chart_stats.append(
                (b, int(mask.sum()), used_w, used_h, sx, sy, sw, sh)
            )

        loop_uv = new_uv
        names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
        for b, n, uw, uh, sx, sy, sw, sh in chart_stats:
            fill = (uw * uh) / max(sw * sh, 1e-9) * 100
            print(f"[blender] chart {names[b]}: {n:>9,} loops, "
                  f"slot ({sx:.3f},{sy:.3f}) {sw:.3f}x{sh:.3f} "
                  f"(used {uw:.3f}x{uh:.3f}, {fill:.0f}% fill)",
                  flush=True)
    else:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        try:
            bpy.ops.uv.smart_project(
                angle_limit=args.angle, island_margin=0.0,
                area_weight=0.0, correct_aspect=True,
                scale_to_bounds=True,
            )
        except TypeError:
            bpy.ops.uv.smart_project(angle_limit=args.angle, island_margin=0.0)
        bpy.ops.object.mode_set(mode="OBJECT")
        mesh = obj.data
        flat_uv = np.empty(n_loops * 2, dtype=np.float32)
        mesh.uv_layers.active.data.foreach_get("uv", flat_uv)
        loop_uv = flat_uv.reshape(n_loops, 2)
        # Smart-project may stray outside [0,1]; fit to unit square.
        uv_min = loop_uv.min(axis=0)
        uv_max = loop_uv.max(axis=0)
        ext = (uv_max - uv_min).max()
        if ext > 0:
            loop_uv = (loop_uv - uv_min) / ext

    print(f"[blender] final: {len(mesh.polygons):,} faces, "
          f"UV method={args.uv_method}, "
          f"UV range u=[{loop_uv[:,0].min():.3f},{loop_uv[:,0].max():.3f}] "
          f"v=[{loop_uv[:,1].min():.3f},{loop_uv[:,1].max():.3f}]",
          flush=True)

    # Vertex-split at UV seams: dedupe (orig_vert, quantised_uv).
    uv_q = np.round(loop_uv * 1e5).astype(np.int64)
    combined = np.stack(
        [loop_vert.astype(np.int64), uv_q[:, 0], uv_q[:, 1]], axis=1
    )
    _u, idx, inverse = np.unique(
        combined, axis=0, return_index=True, return_inverse=True
    )
    vmapping = loop_vert[idx]
    new_uvs = loop_uv[idx]
    loop_to_new = inverse

    new_faces = loop_to_new[
        poly_starts[:, None] + np.arange(3)[None, :]
    ]
    new_positions = positions[vmapping]

    np.savez(
        args.out_path,
        vertices=new_positions.astype(np.float32),
        faces=new_faces.astype(np.int32),
        uvs=new_uvs.astype(np.float32),
    )


if __name__ == "__main__":
    main()
