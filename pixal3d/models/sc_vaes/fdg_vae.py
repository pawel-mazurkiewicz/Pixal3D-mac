from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from .sparse_unet_vae import (
    SparseResBlock3d,
    SparseConvNeXtBlock3d,
    
    SparseResBlockDownsample3d,
    SparseResBlockUpsample3d,
    SparseResBlockS2C3d,
    SparseResBlockC2S3d,
)
from .sparse_unet_vae import (
    SparseUnetVaeEncoder,
    SparseUnetVaeDecoder,
)
from ...representations import Mesh
try:
    from o_voxel.convert import flexible_dual_grid_to_mesh as _native_flexible_dual_grid_to_mesh
except (ImportError, RuntimeError, OSError):
    _native_flexible_dual_grid_to_mesh = None
from ...utils.mesh_extract import flexible_dual_grid_to_mesh as _torch_flexible_dual_grid_to_mesh


def flexible_dual_grid_to_mesh(*args, **kwargs):
    coords = args[0] if args else kwargs.get('coords')
    if (
        _native_flexible_dual_grid_to_mesh is not None
        and isinstance(coords, torch.Tensor)
        and coords.device.type == 'cuda'
    ):
        return _native_flexible_dual_grid_to_mesh(*args, **kwargs)
    return _torch_flexible_dual_grid_to_mesh(*args, **kwargs)


class FlexiDualGridVaeEncoder(SparseUnetVaeEncoder):
    def __init__(
        self,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
    ):
        super().__init__(
            6,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            down_block_type,
            block_args,
            use_fp16,
        )
        
    def forward(self, vertices: sp.SparseTensor, intersected: sp.SparseTensor, sample_posterior=False, return_raw=False):
        x = vertices.replace(torch.cat([
            vertices.feats - 0.5,
            intersected.feats.float() - 0.5,
        ], dim=1))
        return super().forward(x, sample_posterior, return_raw)
    
    
class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    def __init__(
        self,
        resolution: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        voxel_margin: float = 0.5,
        use_fp16: bool = False,
    ):
        self.resolution = resolution
        self.voxel_margin = voxel_margin
        
        super().__init__(
            7,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            up_block_type,
            block_args,
            use_fp16,
        )

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution
        
    def forward(self, x: sp.SparseTensor, gt_intersected: sp.SparseTensor = None, **kwargs):
        decoded = super().forward(x, **kwargs)
        if self.training:
            h, subs_gt, subs = decoded
            vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
            intersected_logits = h.replace(h.feats[..., 3:6])
            quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
            mesh = [Mesh(*flexible_dual_grid_to_mesh(
                v.coords[:, 1:], v.feats, i.feats, q.feats,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                grid_size=self.resolution,
                train=True
            )) for v, i, q in zip(vertices, gt_intersected, quad_lerp)]
            return mesh, vertices, intersected_logits, subs_gt, subs
        else:
            out_list = list(decoded) if isinstance(decoded, tuple) else [decoded]
            h = out_list[0]
            vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
            intersected = h.replace(h.feats[..., 3:6] > 0)
            quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
            mesh = []
            for h_item, v, i, q in zip(h, vertices, intersected, quad_lerp):
                fdg_coords = v.coords[:, 1:]
                fdg_dual_vertices = v.feats
                fdg_intersected = i.feats
                fdg_split_weight = q.feats
                mesh_item = Mesh(*flexible_dual_grid_to_mesh(
                    fdg_coords, fdg_dual_vertices, fdg_intersected, fdg_split_weight,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    grid_size=self.resolution,
                    train=False
                ))
                mesh_item.fdg_coords = fdg_coords
                mesh_item.fdg_dual_vertices = fdg_dual_vertices
                mesh_item.fdg_intersected = fdg_intersected
                mesh_item.fdg_split_weight = fdg_split_weight
                # Raw per-batch-item mid-decoder features (pre-sigmoid /
                # pre-`>0` / pre-softplus split).  Captured so divergence at
                # the FDG VAE decoder can be isolated from threshold-flip
                # divergence on the boolean `intersected = feats[..., 3:6] > 0`.
                mesh_item.fdg_h_feats = h_item.feats
                mesh.append(mesh_item)
            out_list[0] = mesh
            return out_list[0] if len(out_list) == 1 else tuple(out_list)
