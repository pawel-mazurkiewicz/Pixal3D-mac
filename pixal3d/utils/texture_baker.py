"""
CPU UV unwrap and texture baking fallback for Pixal3D meshes.
"""

import time
import numpy as np


def uv_unwrap(vertices, faces, mode="auto", blender_exe=None,
              target_count=None, voxel_size=None, uv_method="cube",
              fill_holes_sides=12):
    """Dispatch UV unwrapping between Blender (fast) and xatlas (default).

    Parameters
    ----------
    mode
        ``"auto"``    use Blender if its executable is found, else xatlas.
        ``"blender"`` require Blender; raise if not found.
        ``"xatlas"``  always use xatlas.
    blender_exe
        Explicit path to the Blender executable.  If ``None``, auto-detect.
    target_count
        If set and the Blender path is taken AND ``voxel_size`` is 0/None,
        the Blender script applies iterative Decimate-Collapse aiming for
        this face count before unwrapping.
    voxel_size
        If set and > 0 and the Blender path is taken, the Blender script
        rebuilds the mesh via voxel remesh at this size (world units)
        before unwrapping.  This is the recommended path for Pixal3D
        meshes — the voxel remesher produces a clean connected manifold
        with predictable face count, sidestepping the topology constraints
        that defeat fast_simplification and Decimate-Collapse alike.
    uv_method
        ``"cube"`` (default) or ``"smart"``.  cube_project gives ~70%%
        atlas utilisation in one call; smart_project gives smoother charts
        but worse utilisation in Blender 5.x because pack_islands(scale=True)
        does not behave as documented.  Ignored on the xatlas path.

    Notes
    -----
    The xatlas path ignores ``target_count``, ``voxel_size`` and
    ``uv_method``; pre-decimation is expected to happen upstream.
    """
    if mode not in {"auto", "blender", "xatlas"}:
        raise ValueError(f"unknown uv_unwrap mode: {mode!r}")

    if mode in {"auto", "blender"}:
        exe = blender_exe or _find_blender_exe()
        if exe is not None:
            try:
                return _uv_unwrap_blender(
                    vertices, faces, exe,
                    target_count=target_count,
                    voxel_size=voxel_size,
                    uv_method=uv_method,
                    fill_holes_sides=fill_holes_sides,
                )
            except Exception as e:
                if mode == "blender":
                    raise
                print(f"  [UV] Blender unwrap failed ({e}); falling back to xatlas.")
        elif mode == "blender":
            raise RuntimeError(
                "Blender executable not found.  Pass --blender-exe or install "
                "Blender (https://www.blender.org/download/)."
            )
        else:
            print("  [UV] Blender not found; using xatlas (slower).  Install "
                  "Blender for ~20-50x faster UV unwrap.")

    return _uv_unwrap_xatlas(vertices, faces)


def _uv_unwrap_xatlas(vertices, faces):
    """Unwrap mesh UVs with xatlas using fast-but-good chart options.

    ``xatlas.parametrize``'s default ChartOptions.max_cost=2.0 makes the
    segmenter explore very tight chart boundaries; on 100-200k face meshes
    this can take 30-60 minutes.  ``max_cost=1.0`` cuts wall-clock by
    roughly half with barely-perceptible quality loss for baked textures.
    """
    import xatlas

    vertices = np.ascontiguousarray(vertices.astype(np.float32))
    faces = np.ascontiguousarray(faces.astype(np.uint32))

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)

    chart_options = xatlas.ChartOptions()
    chart_options.max_cost = 1.0
    chart_options.max_iterations = 1

    pack_options = xatlas.PackOptions()
    pack_options.padding = 2
    pack_options.bilinear = True

    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    vmapping, indices, uvs = atlas[0]
    return vertices[vmapping], indices.reshape(-1, 3), uvs, vmapping


def _find_blender_exe():
    """Return a usable Blender executable path or None."""
    import os
    import shutil

    exe = shutil.which("blender")
    if exe:
        return exe
    # Standard macOS install location
    mac_default = "/Applications/Blender.app/Contents/MacOS/Blender"
    if os.path.exists(mac_default):
        return mac_default
    return None


def _uv_unwrap_blender(vertices, faces, blender_exe,
                       target_count=None, voxel_size=None, uv_method="cube",
                       fill_holes_sides=12):
    """Run Blender headless to remesh and UV-unwrap a triangle mesh.

    See ``_blender_uv_unwrap.py`` for the rationale behind voxel remesh
    + cube_project on Pixal3D meshes.  Returns ``(vertices, faces, uvs, None)``
    where ``vertices`` is already the remeshed + UV-seam-split position
    array.  The trailing ``None`` keeps the four-tuple shape consistent
    with the xatlas path.
    """
    import os
    import subprocess
    import tempfile
    import time

    script = os.path.join(os.path.dirname(__file__), "_blender_uv_unwrap.py")
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.npz")
        out_path = os.path.join(td, "out.npz")
        np.savez(in_path, vertices=vertices, faces=faces)

        cmd = [
            # NOTE: do not pass `--noaudio` — Blender 5.x removed it and
            # treats it as a positional filename, failing the entire run.
            blender_exe, "--background", "--factory-startup",
            "--python", script, "--",
            in_path, out_path,
            "--uv-method", uv_method,
            "--fill-holes-sides", str(int(fill_holes_sides)),
        ]
        if voxel_size is not None:
            cmd += ["--voxel-size", str(float(voxel_size))]
        if target_count is not None and target_count > 0:
            cmd += ["--target-count", str(int(target_count))]

        result = subprocess.run(cmd, capture_output=True, text=True)
        # Surface any [blender] progress lines so the caller sees what the
        # subprocess actually did (face count after remesh / decimate, etc.).
        for ln in result.stdout.splitlines():
            if "[blender]" in ln:
                print(f"  {ln}")
        if result.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(
                "Blender UV unwrap failed (returncode="
                f"{result.returncode}).\n--- stderr ---\n{result.stderr}\n"
                f"--- stdout (tail) ---\n{result.stdout[-2000:]}"
            )

        out = np.load(out_path)
        new_vertices = out["vertices"].astype(np.float32)
        new_faces = out["faces"].astype(np.int64).reshape(-1, 3)
        new_uvs = out["uvs"].astype(np.float32)

    print(f"  [UV] Blender remesh+unwrap: {time.time() - t0:.1f}s, "
          f"{len(new_vertices):,} vertices, {len(new_faces):,} faces")
    return new_vertices, new_faces, new_uvs, None


def _rasterize_uv_triangles(vertices, faces, uvs, texture_size):
    height = width = texture_size
    positions = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    uv_scale = np.array([width - 1, height - 1], dtype=np.float32)

    for face_idx, face in enumerate(faces):
        if face_idx > 0 and face_idx % 100000 == 0:
            print(f"    Rasterizing: {face_idx:,}/{len(faces):,}")

        i0, i1, i2 = face
        uv0 = uvs[i0] * uv_scale
        uv1 = uvs[i1] * uv_scale
        uv2 = uvs[i2] * uv_scale
        p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]

        min_x = max(int(np.floor(min(uv0[0], uv1[0], uv2[0]))), 0)
        max_x = min(int(np.ceil(max(uv0[0], uv1[0], uv2[0]))), width - 1)
        min_y = max(int(np.floor(min(uv0[1], uv1[1], uv2[1]))), 0)
        max_y = min(int(np.ceil(max(uv0[1], uv1[1], uv2[1]))), height - 1)
        if max_x < min_x or max_y < min_y:
            continue

        d00 = uv1[0] - uv0[0]
        d01 = uv2[0] - uv0[0]
        d10 = uv1[1] - uv0[1]
        d11 = uv2[1] - uv0[1]
        denom = d00 * d11 - d01 * d10
        if abs(denom) < 1e-10:
            continue
        inv_denom = 1.0 / denom

        xs = np.arange(min_x, max_x + 1, dtype=np.float32)
        ys = np.arange(min_y, max_y + 1, dtype=np.float32)
        px_grid, py_grid = np.meshgrid(xs, ys)
        dx = px_grid - uv0[0]
        dy = py_grid - uv0[1]

        u = (dx * d11 - d01 * dy) * inv_denom
        v = (d00 * dy - dx * d10) * inv_denom
        w = 1.0 - u - v
        inside = (u >= -0.001) & (v >= -0.001) & (w >= -0.001)
        if not inside.any():
            continue

        pos_3d = w[..., None] * p0 + u[..., None] * p1 + v[..., None] * p2
        iy, ix = np.where(inside)
        positions[ys.astype(int)[iy], xs.astype(int)[ix]] = pos_3d[iy, ix]
        mask[ys.astype(int)[iy], xs.astype(int)[ix]] = True

    return positions, mask


def bake_texture(
    vertices,
    faces,
    uvs,
    voxel_coords,
    voxel_attrs,
    origin,
    voxel_size,
    texture_size=1024,
    k_neighbors=8,
    search_voxels=50,
):
    from scipy.ndimage import binary_dilation, uniform_filter
    from scipy.spatial import cKDTree

    height = width = texture_size
    t0 = time.time()
    coords_np = voxel_coords.numpy() if hasattr(voxel_coords, "numpy") else voxel_coords
    attrs_np = voxel_attrs.numpy() if hasattr(voxel_attrs, "numpy") else voxel_attrs
    origin_np = origin.numpy() if hasattr(origin, "numpy") else np.array(origin)

    channel_count = attrs_np.shape[1]
    voxel_world = coords_np.astype(np.float32) * voxel_size + origin_np + voxel_size * 0.5
    if len(voxel_world) == 0:
        raise RuntimeError("Texture bake received an empty voxel attribute set")
    print(f"  Voxels: {len(coords_np):,}, channels: {channel_count}")
    print("  Building KDTree...")
    tree = cKDTree(voxel_world)

    print(f"  Rasterizing {len(faces):,} triangles into {texture_size}x{texture_size}...")
    positions, mask = _rasterize_uv_triangles(vertices, faces, uvs, texture_size)
    coverage = mask.sum() / (height * width) * 100
    print(f"    Coverage: {coverage:.1f}%")

    query_points = positions[mask]
    if len(query_points) == 0:
        raise RuntimeError("UV rasterization produced no covered texels")
    k_neighbors = min(k_neighbors, len(voxel_world))
    print(f"  Querying {len(query_points):,} texels, k={k_neighbors}...")
    distances, indices = tree.query(query_points, k=k_neighbors, workers=-1)
    if k_neighbors == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # The IDW search radius is scaled in *voxel units* so it tracks the
    # resolution of the source voxel grid.  After heavy QEM decimation
    # (e.g. 8M -> 100k faces, 80x reduction), texels in the interior of a
    # simplified triangle can sit dozens of voxels away from the nearest
    # voxel centre.  The original ``voxel_size * 2.0`` cutoff was tight
    # enough at 512 resolution to mostly work, but at 1024+ it zeroes out
    # all but the chart corners, producing a near-black atlas.  50 voxel
    # widths comfortably covers the typical face-size-after-decimation
    # while still excluding texels in atlas gutters.
    max_dist = voxel_size * search_voxels
    eps = voxel_size * 0.1
    weights = 1.0 / (distances + eps)
    weights[distances > max_dist] = 0.0
    weights_sum = weights.sum(axis=1, keepdims=True)
    has_neighbor = (weights_sum > 0).squeeze()
    weights = np.where(weights_sum > 0, weights / np.maximum(weights_sum, 1e-10), 0.0)

    sampled = (attrs_np[indices] * weights[..., None]).sum(axis=1)
    base_color = np.zeros((height, width, 3), dtype=np.float32)
    metallic = np.zeros((height, width), dtype=np.float32)
    roughness = np.ones((height, width), dtype=np.float32)

    ys, xs = np.where(mask)
    valid = has_neighbor
    base_color[ys[valid], xs[valid]] = np.clip(sampled[valid, 0:3], 0, 1)
    if channel_count > 3:
        metallic[ys[valid], xs[valid]] = np.clip(sampled[valid, 3], 0, 1)
    if channel_count > 4:
        roughness[ys[valid], xs[valid]] = np.clip(sampled[valid, 4], 0, 1)

    valid_mask = np.zeros((height, width), dtype=bool)
    valid_mask[ys[valid], xs[valid]] = True
    current_mask = valid_mask.copy()
    for _ in range(8):
        dilated = binary_dilation(current_mask, iterations=1)
        unfilled = dilated & ~current_mask
        if not unfilled.any():
            break
        for channel_idx in range(3):
            channel = base_color[:, :, channel_idx]
            blurred = uniform_filter(channel, size=3)
            channel[unfilled] = blurred[unfilled]
        current_mask = dilated

    base_color = np.power(np.clip(base_color, 0, 1), 1.0 / 2.2)
    base_color_img = (base_color * 255).astype(np.uint8)

    mr_img = np.zeros((height, width, 3), dtype=np.uint8)
    mr_img[:, :, 1] = (roughness * 255).astype(np.uint8)
    mr_img[:, :, 2] = (metallic * 255).astype(np.uint8)

    final_coverage = current_mask.sum() / (height * width) * 100
    print(f"  Final coverage: {final_coverage:.1f}%, bake: {time.time() - t0:.1f}s")
    return base_color_img, mr_img, current_mask


def export_glb_with_texture(vertices, faces, uvs, base_color_img, mr_img=None, output_path="output.glb"):
    import trimesh
    from PIL import Image

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(base_color_img),
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )
    if mr_img is not None:
        material.metallicRoughnessTexture = Image.fromarray(mr_img)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    mesh.export(output_path)
    return output_path
