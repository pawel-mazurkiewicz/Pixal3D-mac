"""BVH-corrected texture bake — port of the TRELLIS.2 to_glb bake step.

Why this matters:

Our pre-existing baker (pixal3d/utils/texture_baker.bake_texture) rasterizes
each face of the *decimated* mesh into the atlas, then samples voxel attrs
at each rasterized texel's 3D position via KDTree-IDW.  After heavy
decimation (8.4M → 1M faces), each simplified triangle's 3D interior can
sit dozens of voxel-widths away from the original surface where the voxel
colour actually lives.  IDW then smears neighbouring voxels' colours
across the texel, dulling saturation and producing the grey-ish look we
see vs the GT bake.

TRELLIS.2's o_voxel.postprocess.to_glb does this differently:

    pos = interpolate(out_vertices, rast, out_faces)        # texel → 3D on decimated mesh
    _, face_id, uvw = bvh.unsigned_distance(pos, return_uvw=True)
                                                            # project to closest pt on ORIGINAL mesh
    orig_tri_verts = orig_vertices[orig_faces[face_id]]
    corrected_pos = (orig_tri_verts * uvw).sum(dim=1)
    attrs = grid_sample_3d(attr_volume, ..., corrected_pos)  # trilinear from volume

We do the same here, just with trimesh.proximity (CPU AABB tree) and
KDTree-IDW for sampling (we don't have the dense volume — only a sparse
voxel cloud).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _rasterize_uv_triangles(
    vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray, size: int,
):
    """Replicate texture_baker._rasterize_uv_triangles to keep this module
    self-contained.  Returns (positions (size, size, 3), mask (size, size))."""
    from pixal3d.utils.texture_baker import _rasterize_uv_triangles
    return _rasterize_uv_triangles(vertices, faces, uvs, size)


def bake_texture_bvh_corrected(
    decim_vertices: np.ndarray,
    decim_faces: np.ndarray,
    uvs: np.ndarray,
    orig_vertices: np.ndarray,
    orig_faces: np.ndarray,
    voxel_coords: np.ndarray,
    voxel_attrs: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    texture_size: int = 4096,
    search_voxels: int = 50,
    verbose: bool = False,
):
    """Bake voxel attributes onto a UV atlas, projecting through the
    original pre-decimation surface for correct sampling.

    Parameters
    ----------
    decim_vertices, decim_faces, uvs
        The post-UV-unwrap mesh (with seam-split vertices).
    orig_vertices, orig_faces
        The mesh BEFORE decimation.  Provides the ground-truth surface
        we project decimated texel positions back onto.
    voxel_coords, voxel_attrs
        Sparse voxel attribute cloud (positions = ``coords * voxel_size + origin``,
        attrs in [0, 1]^C where C=6 for Pixal3D).
    origin, voxel_size
        Voxel grid origin and isotropic voxel size.
    texture_size
        Atlas resolution (square).
    search_voxels
        IDW cutoff in voxel-widths (matches existing baker).

    Returns
    -------
    base_color_img : (texture_size, texture_size, 3) uint8 — gamma-encoded
    mr_img         : (texture_size, texture_size, 3) uint8 — (B=metallic, G=rough, R=0)
    valid_mask     : (texture_size, texture_size) bool
    """
    import time
    import trimesh
    from scipy.ndimage import binary_dilation, uniform_filter
    from scipy.spatial import cKDTree

    h = w = texture_size
    t0 = time.time()

    # Voxel world positions (centre of each voxel cell).
    voxel_world = voxel_coords.astype(np.float32) * voxel_size + origin.astype(np.float32) + voxel_size * 0.5
    if verbose:
        print(f"[bvh-bake] Voxels: {voxel_world.shape[0]:,}; "
              f"orig mesh: V={orig_vertices.shape[0]:,} F={orig_faces.shape[0]:,}",
              flush=True)

    # 1. Rasterize the DECIMATED mesh into the atlas to get (texel → 3D pos).
    positions, mask = _rasterize_uv_triangles(decim_vertices, decim_faces, uvs, texture_size)
    coverage = float(mask.sum()) / (h * w) * 100.0
    if verbose:
        print(f"[bvh-bake] Rasterized: coverage={coverage:.1f}%", flush=True)

    query_pts = positions[mask]                                  # (N, 3) on decimated mesh
    if query_pts.shape[0] == 0:
        raise RuntimeError("UV rasterization produced zero covered texels")

    # 2. Project query_pts → closest *original vertex* (approximation of
    #    closest point on triangle mesh — trimesh.proximity's exact AABB
    #    walk takes 30+ min on 8M-face inputs, whereas cKDTree on vertices
    #    runs in seconds and is accurate to the original-vertex density,
    #    which is well below voxel scale on Pixal3D output).
    if verbose:
        print(f"[bvh-bake] Building cKDTree on original-mesh vertices...",
              flush=True)
    orig_tree = cKDTree(orig_vertices.astype(np.float32))
    if verbose:
        print(f"[bvh-bake] Projecting {query_pts.shape[0]:,} texels...",
              flush=True)
    _, orig_v_idx = orig_tree.query(query_pts, k=1, workers=-1)
    corrected = orig_vertices[orig_v_idx].astype(np.float32)
    if verbose:
        drift = np.linalg.norm(corrected - query_pts, axis=1)
        print(f"[bvh-bake]   median drift = {np.median(drift):.5f}  "
              f"max = {drift.max():.5f}  (voxel size = {voxel_size:.5f})",
              flush=True)

    # 3. KDTree-IDW sample voxels at the CORRECTED positions.
    tree = cKDTree(voxel_world)
    k = min(8, voxel_world.shape[0])
    distances, indices = tree.query(corrected, k=k, workers=-1)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    max_dist = voxel_size * search_voxels
    eps = voxel_size * 0.1
    weights = 1.0 / (distances + eps)
    weights[distances > max_dist] = 0.0
    weights_sum = weights.sum(axis=1, keepdims=True)
    has_neighbor = (weights_sum > 0).squeeze()
    weights = np.where(weights_sum > 0, weights / np.maximum(weights_sum, 1e-10), 0.0)
    sampled = (voxel_attrs[indices] * weights[..., None]).sum(axis=1)

    channel_count = voxel_attrs.shape[1]
    base_color = np.zeros((h, w, 3), dtype=np.float32)
    metallic = np.zeros((h, w), dtype=np.float32)
    roughness = np.ones((h, w), dtype=np.float32)
    ys, xs = np.where(mask)
    valid = has_neighbor
    base_color[ys[valid], xs[valid]] = np.clip(sampled[valid, 0:3], 0, 1)
    if channel_count > 3:
        metallic[ys[valid], xs[valid]] = np.clip(sampled[valid, 3], 0, 1)
    if channel_count > 4:
        roughness[ys[valid], xs[valid]] = np.clip(sampled[valid, 4], 0, 1)

    # 4. Gutter dilation (same as existing baker).
    valid_mask = np.zeros((h, w), dtype=bool)
    valid_mask[ys[valid], xs[valid]] = True
    current_mask = valid_mask.copy()
    for _ in range(8):
        dilated = binary_dilation(current_mask, iterations=1)
        unfilled = dilated & ~current_mask
        if not unfilled.any():
            break
        for ci in range(3):
            ch = base_color[:, :, ci]
            blurred = uniform_filter(ch, size=3)
            ch[unfilled] = blurred[unfilled]
        current_mask = dilated

    base_color = np.power(np.clip(base_color, 0, 1), 1.0 / 2.2)
    base_color_img = (base_color * 255).astype(np.uint8)
    mr_img = np.zeros((h, w, 3), dtype=np.uint8)
    mr_img[:, :, 1] = (roughness * 255).astype(np.uint8)
    mr_img[:, :, 2] = (metallic * 255).astype(np.uint8)

    final_coverage = current_mask.sum() / (h * w) * 100.0
    if verbose:
        print(f"[bvh-bake] Final coverage: {final_coverage:.1f}%, "
              f"total: {time.time() - t0:.1f}s", flush=True)
    return base_color_img, mr_img, current_mask
