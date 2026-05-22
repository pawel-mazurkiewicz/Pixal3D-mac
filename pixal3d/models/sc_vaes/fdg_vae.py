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
from o_voxel.convert import flexible_dual_grid_to_mesh


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
                # Stash the pre-extraction FDG tensors on the mesh so that
                # _fixture_to_cpu can dump them at stage-07.  This is the
                # input side of `flexible_dual_grid_to_mesh`; capturing it
                # lets a Mac box replay only the FDG -> mesh + cleanup steps
                # on CUDA's exact inputs and compare against CUDA's output.
                mesh_item.fdg_coords = fdg_coords
                mesh_item.fdg_dual_vertices = fdg_dual_vertices
                mesh_item.fdg_intersected = fdg_intersected
                mesh_item.fdg_split_weight = fdg_split_weight
                # Also stash the raw per-batch-item mid-decoder features
                # (before sigmoid / > 0 / softplus split).  Lets us isolate
                # FDG-VAE-decoder divergence from threshold-flip divergence.
                # feats[..., 0:3] are pre-sigmoid vertex offsets, [3:6] are
                # pre-`>0` intersected logits (boundary-flip risk!),
                # [6:7] is pre-softplus quad-lerp weight.
                mesh_item.fdg_h_feats = h_item.feats
                mesh.append(mesh_item)
            out_list[0] = mesh
            return out_list[0] if len(out_list) == 1 else tuple(out_list)
