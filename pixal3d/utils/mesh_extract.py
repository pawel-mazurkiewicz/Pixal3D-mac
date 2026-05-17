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
_PARTIAL_TRI_BY_MISSING = None


def mesh_to_flexible_dual_grid(*args, **kwargs):
    raise RuntimeError("mesh_to_flexible_dual_grid requires native o_voxel")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _linear_keys(coords: torch.Tensor, grid_size: torch.Tensor) -> torch.Tensor:
    coords = coords.long()
    _, gy, gz = [int(v) for v in grid_size.detach().cpu().tolist()]
    return coords[..., 0] * (gy * gz) + coords[..., 1] * gz + coords[..., 2]


def _lookup_sparse_voxels(
    coords: torch.Tensor,
    queries: torch.Tensor,
    grid_size: torch.Tensor,
) -> torch.Tensor:
    """Return voxel row indices for query coords, or -1 for missing/outside."""
    device = queries.device
    coords_cpu = coords.detach().cpu().long()
    queries_cpu = queries.detach().cpu().long().reshape(-1, 3)
    grid_cpu = grid_size.detach().cpu().long()

    valid = (
        (queries_cpu[:, 0] >= 0) & (queries_cpu[:, 0] < grid_cpu[0])
        & (queries_cpu[:, 1] >= 0) & (queries_cpu[:, 1] < grid_cpu[1])
        & (queries_cpu[:, 2] >= 0) & (queries_cpu[:, 2] < grid_cpu[2])
    )

    coord_keys = _linear_keys(coords_cpu, grid_cpu)
    order = torch.argsort(coord_keys)
    sorted_keys = coord_keys[order]

    query_keys = _linear_keys(queries_cpu.clamp_min(0), grid_cpu)
    pos = torch.searchsorted(sorted_keys, query_keys)
    in_range = pos < sorted_keys.numel()
    found = torch.zeros_like(valid)
    found[in_range] = sorted_keys[pos[in_range]] == query_keys[in_range]
    found &= valid

    out = torch.full((queries_cpu.shape[0],), -1, dtype=torch.long)
    out[found] = order[pos[found]]
    return out.reshape(queries.shape[:-1]).to(device=device)


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
    global _EDGE_NEIGHBOR_VOXEL_OFFSET, _QUAD_SPLIT_1, _QUAD_SPLIT_2, _PARTIAL_TRI_BY_MISSING

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
        _PARTIAL_TRI_BY_MISSING = torch.tensor([
            [3, 1, 2],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2],
        ], dtype=torch.long, device=device)

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

    edge_neighbor_voxel = coords.reshape(num_voxels, 1, 1, 3) + _EDGE_NEIGHBOR_VOXEL_OFFSET
    connected_voxel = edge_neighbor_voxel[intersected_flag]
    num_connected = connected_voxel.shape[0]

    if num_connected == 0:
        return (
            torch.zeros(0, 3, device=device),
            torch.zeros(0, 3, dtype=torch.long, device=device),
        )

    connected_voxel_indices = _lookup_sparse_voxels(coords, connected_voxel, grid_size).long()
    present_mask = connected_voxel_indices >= 0
    connected_voxel_valid = present_mask.all(dim=1)
    quad_indices = connected_voxel_indices[connected_voxel_valid].long()

    mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0].reshape(1, 3)
    mesh_triangles = []

    if quad_indices.shape[0] > 0 and split_weight is None:
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
        mesh_triangles.append(torch.where(align_1 > align_2, split_1, split_2).reshape(-1, 3))
    elif quad_indices.shape[0] > 0:
        split_weight = split_weight[quad_indices]
        sw_02 = (split_weight[:, 0] * split_weight[:, 2]).reshape(-1)
        sw_13 = (split_weight[:, 1] * split_weight[:, 3]).reshape(-1)
        cond = (sw_02 > sw_13).unsqueeze(1).expand(-1, 6)
        mesh_triangles.append(
            torch.where(
                cond,
                quad_indices[:, _QUAD_SPLIT_1],
                quad_indices[:, _QUAD_SPLIT_2],
            ).reshape(-1, 3)
        )

    cap_partial_quads = _env_flag("PIXAL3D_FDG_CAP_PARTIAL_QUADS", default=False)
    partial_triangles = 0
    if cap_partial_quads:
        partial3 = present_mask.sum(dim=1) == 3
        if bool(partial3.any().item()):
            partial = connected_voxel_indices[partial3].clamp_min(0)
            missing = (~present_mask[partial3]).int().argmax(dim=1)
            gather = _PARTIAL_TRI_BY_MISSING[missing]
            mesh_triangles.append(partial.gather(1, gather))
            partial_triangles = int(partial3.sum().item())

    if not mesh_triangles:
        return (
            torch.zeros(0, 3, device=device),
            torch.zeros(0, 3, dtype=torch.long, device=device),
        )

    mesh_triangles = torch.cat(mesh_triangles, dim=0)

    if _env_flag("PIXAL3D_FDG_VERBOSE", default=False):
        invalid = int((~connected_voxel_valid).sum().item())
        print(
            "[FDG] Extracted mesh: "
            f"voxels={num_voxels:,}, intersections={num_connected:,}, "
            f"full_quads={quad_indices.shape[0]:,}, dropped={invalid:,}, "
            f"partial_tris={partial_triangles:,}, faces={mesh_triangles.shape[0]:,}"
        )

    return mesh_vertices, mesh_triangles
