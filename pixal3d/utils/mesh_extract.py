"""
Pure PyTorch mesh extraction from sparse voxel dual-grid data.

This mirrors the TRELLIS.2 Apple Silicon fallback for environments where
o_voxel.convert is unavailable or tied to CUDA.
"""

from typing import Union
import numpy as np
import torch


_EDGE_NEIGHBOR_VOXEL_OFFSET = None
_QUAD_SPLIT_1 = None
_QUAD_SPLIT_2 = None


def mesh_to_flexible_dual_grid(*args, **kwargs):
    raise RuntimeError("mesh_to_flexible_dual_grid requires native o_voxel")


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
    coords_cpu = coords.cpu()
    coord_to_idx = {
        tuple(coords_cpu[i].tolist()): i
        for i in range(num_voxels)
    }

    edge_neighbor_voxel = coords.reshape(num_voxels, 1, 1, 3) + _EDGE_NEIGHBOR_VOXEL_OFFSET
    connected_voxel = edge_neighbor_voxel[intersected_flag]
    num_connected = connected_voxel.shape[0]

    if num_connected == 0:
        return (
            torch.zeros(0, 3, device=device),
            torch.zeros(0, 3, dtype=torch.long, device=device),
        )

    connected_cpu = connected_voxel.cpu().reshape(-1, 3)
    indices = []
    for idx in range(connected_cpu.shape[0]):
        key = tuple(connected_cpu[idx].tolist())
        indices.append(coord_to_idx.get(key, 0xFFFFFFFF))

    connected_voxel_indices = torch.tensor(indices, dtype=torch.int64, device=device).reshape(num_connected, 4)
    connected_voxel_valid = (connected_voxel_indices != 0xFFFFFFFF).all(dim=1)
    quad_indices = connected_voxel_indices[connected_voxel_valid].long()

    if quad_indices.shape[0] == 0:
        return (
            torch.zeros(0, 3, device=device),
            torch.zeros(0, 3, dtype=torch.long, device=device),
        )

    mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0].reshape(1, 3)

    if split_weight is None:
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
        split_weight = split_weight[quad_indices]
        sw_02 = (split_weight[:, 0] * split_weight[:, 2]).squeeze()
        sw_13 = (split_weight[:, 1] * split_weight[:, 3]).squeeze()
        cond = (sw_02 > sw_13).unsqueeze(1).expand(-1, 6)
        mesh_triangles = torch.where(
            cond,
            quad_indices[:, _QUAD_SPLIT_1],
            quad_indices[:, _QUAD_SPLIT_2],
        ).reshape(-1, 3)

    return mesh_vertices, mesh_triangles
