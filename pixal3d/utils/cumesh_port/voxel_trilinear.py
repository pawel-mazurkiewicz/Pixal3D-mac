"""Trilinear sparse-voxel attribute sampling.

Replicates the colour-blending behaviour of TRELLIS.2's
``flex_gemm.ops.grid_sample_3d`` on CPU, without allocating a dense
volumetric grid (the natural-shape grid at Pixal3D's 1024^3 resolution
× 6 channels × 4 bytes would be 25 GB).

Instead we keep the voxel data sparse and look up the 8 cube corners
around each query position via a sorted-key binary search:

    key(cx, cy, cz) = cx * R² + cy * R + cz
    sorted_keys = sorted(voxel_keys)
    for each corner: pos = np.searchsorted(sorted_keys, corner_key);
                     if sorted_keys[pos] == corner_key:  hit
                                              else: miss (weight 0)

Same continuous blend behaviour as a true trilinear; same "skip the
missing corner" graceful degradation; ~38 M lookups in a few seconds
with vectorized numpy.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def trilinear_sample_sparse(
    coords: np.ndarray,
    attrs: np.ndarray,
    grid_resolution: int,
    query_positions: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Sample sparse voxel grid trilinearly at world-space query positions.

    Parameters
    ----------
    coords : (V, 3) int (voxel cell indices, 0 ≤ c < grid_resolution)
    attrs  : (V, C) float
    grid_resolution : int (one side of the grid; coords are bounded by this)
    query_positions : (N, 3) float (world space)
    origin : (3,) float — grid origin (lower corner of cell (0, 0, 0))
    voxel_size : float

    Returns
    -------
    sampled : (N, C) float
        Trilinear blend of the 8 surrounding voxel-cell corners; corners
        with no occupied voxel are masked out and the remaining weights
        are re-normalised.  Texels whose 8 corners are all empty get a
        zero return — the caller should handle that with gutter
        inpainting (this module does no inpaint).
    """
    R = int(grid_resolution)
    C = int(attrs.shape[1])
    N = int(query_positions.shape[0])

    # Sparse voxel lookup table — sorted by encoded (cx, cy, cz) key.
    R2 = np.int64(R) * np.int64(R)
    voxel_keys = (
        coords[:, 0].astype(np.int64) * R2
        + coords[:, 1].astype(np.int64) * R
        + coords[:, 2].astype(np.int64)
    )
    order = np.argsort(voxel_keys, kind="stable")
    sorted_keys = voxel_keys[order]
    sorted_attrs = attrs[order].astype(np.float32, copy=False)

    # Grid coords for each query.
    gp = (query_positions.astype(np.float64) - origin.astype(np.float64)) / voxel_size
    i0 = np.floor(gp).astype(np.int64)
    f = (gp - i0).astype(np.float32)                       # (N, 3) ∈ [0, 1)
    i0 = np.clip(i0, 0, R - 2)
    i1 = i0 + 1

    out = np.zeros((N, C), dtype=np.float32)
    weight_sum = np.zeros((N, 1), dtype=np.float32)

    # Eight corners.
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix = i1[:, 0] if dx else i0[:, 0]
                iy = i1[:, 1] if dy else i0[:, 1]
                iz = i1[:, 2] if dz else i0[:, 2]
                wx = f[:, 0] if dx else (1.0 - f[:, 0])
                wy = f[:, 1] if dy else (1.0 - f[:, 1])
                wz = f[:, 2] if dz else (1.0 - f[:, 2])
                weights = (wx * wy * wz).astype(np.float32)[:, None]

                # Encode query corner keys, binary-search into sparse table.
                qkey = ix * R2 + iy * R + iz
                pos = np.searchsorted(sorted_keys, qkey)
                # Mask: position in-range AND key matches.
                in_range = pos < sorted_keys.shape[0]
                hit_mask = np.zeros_like(in_range, dtype=bool)
                hit_mask[in_range] = sorted_keys[pos[in_range]] == qkey[in_range]
                # Pull attrs for hits.
                attr_at_corner = np.zeros((N, C), dtype=np.float32)
                attr_at_corner[hit_mask] = sorted_attrs[pos[hit_mask]]
                w = weights * hit_mask[:, None].astype(np.float32)
                out += w * attr_at_corner
                weight_sum += w

    safe = np.where(weight_sum > 0, weight_sum, 1.0)
    out = out / safe
    # Texels whose 8 corners are all empty stay at zero.
    return out


# Backward compat aliases (kept so older scripts can still import).
def build_dense_grid(*args, **kwargs):
    raise RuntimeError(
        "build_dense_grid is deprecated — use trilinear_sample_sparse "
        "instead.  At Pixal3D's 1024^3 × 6-channel resolution the dense "
        "grid would be 25 GB."
    )


def trilinear_sample(*args, **kwargs):
    raise RuntimeError(
        "trilinear_sample(grid, ...) is deprecated.  Use "
        "trilinear_sample_sparse(coords, attrs, grid_resolution, ...) "
        "instead."
    )
