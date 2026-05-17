"""BVH-corrected texture bake.

TRELLIS.2's o_voxel.postprocess.to_glb does a subtle but very important
thing in the bake step: instead of sampling voxel attrs at the rasterized
texel position (which lies on the *decimated* mesh and is therefore
geometrically wrong vs the original surface), it:

  1. Rasterizes each texel and gets the 3D position on the decimated mesh.
  2. Looks up the *closest point on the original pre-decimation mesh* via BVH.
  3. Returns face_id + barycentric uvw on the original mesh.
  4. Reconstructs the correct surface position as
     ``orig_tri_verts[face_id] @ uvw``.
  5. Samples the voxel attribute volume at THIS corrected position
     (trilinearly via grid_sample_3d in the CUDA code).

Without this correction every texel sees a position drift of up to a
voxel-size (or more after aggressive decimation), and the IDW-from-voxels
sampling smears colors across surface features.

This file ports steps (2)-(4) to CPU using trimesh.proximity (which uses
its own AABB tree under the hood, plenty fast for our sizes).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def closest_points_on_mesh(
    query_points: np.ndarray,
    orig_vertices: np.ndarray,
    orig_faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each query point, return (closest_point, face_id) on the mesh.

    Uses trimesh.proximity.ProximityQuery — internally an AABB tree.  For
    Pixal3D's ~10M-face original mesh, this is the slowest step of the
    bake (a few seconds for ~2.5M texel queries).
    """
    import trimesh
    mesh = trimesh.Trimesh(
        vertices=orig_vertices.astype(np.float64),
        faces=orig_faces.astype(np.int32),
        process=False,
    )
    pq = trimesh.proximity.ProximityQuery(mesh)
    closest, distances, face_ids = pq.on_surface(query_points.astype(np.float64))
    return closest.astype(np.float32), face_ids.astype(np.int64)


def bvh_corrected_positions(
    query_points: np.ndarray,
    orig_vertices: np.ndarray,
    orig_faces: np.ndarray,
) -> np.ndarray:
    """Project each query point onto the original mesh surface.

    Matches the TRELLIS.2 to_glb correction:
        _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
        orig_tri_verts = vertices[faces[face_id.long()]]
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    """
    closest, _ = closest_points_on_mesh(query_points, orig_vertices, orig_faces)
    return closest
