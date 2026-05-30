"""
Pure PyTorch mesh extraction from sparse voxel dual-grid data.

This mirrors the TRELLIS.2 Apple Silicon fallback for environments where
o_voxel.convert is unavailable or tied to CUDA.
"""

from typing import Union
import os
import numpy as np
import torch


_EDGE_NEIGHBOR_VOXEL_OFFSET = None
_QUAD_SPLIT_1 = None
_QUAD_SPLIT_2 = None


def mesh_to_flexible_dual_grid(*args, **kwargs):
    raise RuntimeError("mesh_to_flexible_dual_grid requires native o_voxel")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _build_hashmap(coords: torch.Tensor) -> dict:
    """Build a sparse `(x, y, z) -> row_index` map from a voxel-coords tensor.

    Mirrors the role of `o_voxel._C.hashmap_insert_3d_idx_as_val_cuda`: each
    occupied voxel is keyed by its integer (x, y, z) coordinate and maps to
    its row index in `coords`. A Python dict is sufficient — coords are
    guaranteed unique by upstream invariant — and runs on any device.
    """
    coords_cpu = coords.detach().cpu().long().contiguous().tolist()
    return {(x, y, z): i for i, (x, y, z) in enumerate(coords_cpu)}


def _hashmap_lookup_3d(
    hashmap: dict,
    queries: torch.Tensor,
    sentinel: int = -1,
) -> torch.Tensor:
    """Vectorized dict lookup: returns row_index per query, or `sentinel` for missing.

    Mirrors `o_voxel._C.hashmap_lookup_3d_cuda` semantics (which returns
    `0xffffffff` for missing); we use `-1` so the result fits a torch.long and
    callers can write `present_mask = (result >= 0)`. The leading "zero" column
    of upstream's 4D hash key is a CUDA-hashmap detail and is dropped here.
    """
    device = queries.device
    flat = queries.detach().cpu().long().contiguous().reshape(-1, 3).tolist()
    out = [hashmap.get((x, y, z), sentinel) for (x, y, z) in flat]
    result = torch.tensor(out, dtype=torch.long)
    return result.reshape(queries.shape[:-1]).to(device=device)


def flexible_dual_grid_to_mesh(
    coords: torch.Tensor,
    dual_vertices: torch.Tensor,
    intersected_flag: torch.Tensor,
    split_weight: Union[torch.Tensor, None],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    train: bool = False,
):
    """Pure-PyTorch FDG -> mesh extractor for CPU/MPS.

    Algorithm vendored verbatim from upstream
    ``o_voxel/convert/flexible_dual_grid.py:142-265`` (non-train branch). The
    only deviation: the two CUDA-only ``_C.hashmap_*_3d_cuda`` calls are
    replaced by `_build_hashmap`/`_hashmap_lookup_3d`, a Python-dict shim with
    matching semantics. Everything else (neighbor-offset gating, diagonal-pick
    triangulation, world-space placement) is upstream-exact.

    Set ``PIXAL3D_FDG_VERBOSE=1`` to print per-call extraction stats.
    """
    global _EDGE_NEIGHBOR_VOXEL_OFFSET, _QUAD_SPLIT_1, _QUAD_SPLIT_2

    if train:
        raise RuntimeError("Training mode is not supported by the pure PyTorch mesh extractor")

    device = coords.device
    if _EDGE_NEIGHBOR_VOXEL_OFFSET is None or _EDGE_NEIGHBOR_VOXEL_OFFSET.device != device:
        _EDGE_NEIGHBOR_VOXEL_OFFSET = torch.tensor([
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        ], dtype=torch.int, device=device).unsqueeze(0)
        _QUAD_SPLIT_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=device)
        _QUAD_SPLIT_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=device)

    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=device)

    if voxel_size is not None:
        if isinstance(voxel_size, (int, float)):
            voxel_size = [voxel_size] * 3
        if isinstance(voxel_size, (list, tuple, np.ndarray)):
            voxel_size = torch.tensor(np.array(voxel_size), dtype=torch.float32, device=device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        if isinstance(grid_size, int):
            grid_size = [grid_size] * 3
        if isinstance(grid_size, (list, tuple, np.ndarray)):
            grid_size = torch.tensor(np.array(grid_size), dtype=torch.int32, device=device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size.float()

    num_voxels = dual_vertices.shape[0]

    edge_neighbor_voxel = coords.reshape(num_voxels, 1, 1, 3) + _EDGE_NEIGHBOR_VOXEL_OFFSET  # (N, 3, 4, 3)
    connected_voxel = edge_neighbor_voxel[intersected_flag]                                   # (M, 4, 3)
    num_connected = connected_voxel.shape[0]

    mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0].reshape(1, 3)

    if num_connected == 0:
        return mesh_vertices, torch.zeros(0, 3, dtype=torch.long, device=device)

    # Hashmap insert + lookup (upstream uses CUDA hashmap; we use a dict shim).
    hashmap = _build_hashmap(coords)
    connected_voxel_indices = _hashmap_lookup_3d(hashmap, connected_voxel)            # (M, 4) long, -1 missing
    connected_voxel_valid = (connected_voxel_indices >= 0).all(dim=1)
    quad_indices = connected_voxel_indices[connected_voxel_valid]                     # (L, 4)

    if quad_indices.shape[0] == 0:
        if _env_flag("PIXAL3D_FDG_VERBOSE", default=False):
            print(
                "[FDG] Extracted mesh: "
                f"voxels={num_voxels:,}, intersections={num_connected:,}, full_quads=0, "
                f"dropped={num_connected:,}, faces=0"
            )
        return mesh_vertices, torch.zeros(0, 3, dtype=torch.long, device=device)

    if split_weight is None:
        # Diagonal-pick triangulation: choose the split whose two tris share
        # the more coplanar normals (= smaller dihedral fold).
        split_1 = quad_indices[:, _QUAD_SPLIT_1]
        n0 = torch.cross(
            mesh_vertices[split_1[:, 1]] - mesh_vertices[split_1[:, 0]],
            mesh_vertices[split_1[:, 2]] - mesh_vertices[split_1[:, 0]],
            dim=1,
        )
        n1 = torch.cross(
            mesh_vertices[split_1[:, 2]] - mesh_vertices[split_1[:, 1]],
            mesh_vertices[split_1[:, 3]] - mesh_vertices[split_1[:, 1]],
            dim=1,
        )
        align_1 = (n0 * n1).sum(dim=1, keepdim=True).abs()

        split_2 = quad_indices[:, _QUAD_SPLIT_2]
        n0 = torch.cross(
            mesh_vertices[split_2[:, 1]] - mesh_vertices[split_2[:, 0]],
            mesh_vertices[split_2[:, 2]] - mesh_vertices[split_2[:, 0]],
            dim=1,
        )
        n1 = torch.cross(
            mesh_vertices[split_2[:, 2]] - mesh_vertices[split_2[:, 1]],
            mesh_vertices[split_2[:, 3]] - mesh_vertices[split_2[:, 1]],
            dim=1,
        )
        align_2 = (n0 * n1).sum(dim=1, keepdim=True).abs()
        mesh_triangles = torch.where(align_1 > align_2, split_1, split_2).reshape(-1, 3)
    else:
        sw = split_weight[quad_indices]
        sw_02 = (sw[:, 0] * sw[:, 2]).reshape(-1)
        sw_13 = (sw[:, 1] * sw[:, 3]).reshape(-1)
        cond = (sw_02 > sw_13).unsqueeze(1).expand(-1, 6)
        mesh_triangles = torch.where(
            cond,
            quad_indices[:, _QUAD_SPLIT_1],
            quad_indices[:, _QUAD_SPLIT_2],
        ).reshape(-1, 3)

    if _env_flag("PIXAL3D_FDG_VERBOSE", default=False):
        invalid = int((~connected_voxel_valid).sum().item())
        print(
            "[FDG] Extracted mesh: "
            f"voxels={num_voxels:,}, intersections={num_connected:,}, "
            f"full_quads={quad_indices.shape[0]:,}, dropped={invalid:,}, "
            f"faces={mesh_triangles.shape[0]:,}"
        )

    return mesh_vertices, mesh_triangles
