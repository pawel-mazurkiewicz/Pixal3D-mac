"""Vertex normal computation — port of cumesh.compute_vertex_normals.

cumesh's CUDA computes per-face normals (area-weighted), then sums them
into adjacent vertices and normalizes.  We do the same in numpy.

Why it matters: without vertex normals, GLB exports rely on flat per-face
shading in Blender / web viewers, making the surface look heavily faceted
even when the underlying geometry is fine.  GT's TRELLIS.2 export always
includes them.
"""
from __future__ import annotations

import numpy as np


def compute_vertex_normals(
    vertices: np.ndarray, faces: np.ndarray,
) -> np.ndarray:
    """Area-weighted vertex normals.

    For each face, compute the geometric normal (cross of two edges, length
    equals 2× area).  Sum this signed-area vector into each of the face's
    three vertices.  Normalize at the end.

    Returns (V, 3) float32.
    """
    V = vertices.shape[0]
    tri = vertices[faces].astype(np.float64)
    # Cross product yields a vector pointing along the face normal with
    # magnitude = 2 * face_area, which is exactly what we want as the
    # area-weighting factor.
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)                               # (F, 3)
    # Scatter-add into vertex normals.
    vert_normals = np.zeros((V, 3), dtype=np.float64)
    for k in range(3):
        np.add.at(vert_normals, faces[:, k].astype(np.int64), cross)
    lens = np.linalg.norm(vert_normals, axis=1)
    safe = np.where(lens > 1e-30, lens, 1.0)
    return (vert_normals / safe[:, None]).astype(np.float32)
