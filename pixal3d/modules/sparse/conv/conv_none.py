"""
Pure PyTorch sparse 3D convolution backend.

This is a portable fallback for Apple Silicon and CPU-only environments. It is
slower than flex_gemm/spconv, but it avoids CUDA-only extension imports.
"""

import math
import torch
import torch.nn as nn
from .. import SparseTensor


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and padding is None, \
        "conv_none only supports submanifold sparse convolution (stride=1, padding=None)"

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3

    weight = torch.empty((out_channels, in_channels, *self.kernel_size))
    self.weight = nn.Parameter(weight)
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # Match flex_gemm's layout: (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci).
    self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    device = x.feats.device
    dtype = x.feats.dtype
    coords = x.coords
    feats = x.feats
    num_voxels = coords.shape[0]

    cache_key = f"SubMConv3d_none_neighbor_{Kw}x{Kh}x{Kd}_dilation{self.dilation}"
    neighbor_cache = x.get_spatial_cache(cache_key)

    if neighbor_cache is None:
        coords_cpu = coords.cpu()
        coord_to_idx = {
            tuple(coords_cpu[i].tolist()): i
            for i in range(num_voxels)
        }

        src_indices = []
        tgt_indices = []
        kernel_indices = []
        dz, dy, dx = self.dilation

        for kz in range(Kd):
            for ky in range(Kh):
                for kx in range(Kw):
                    oz = (kz - Kd // 2) * dz
                    oy = (ky - Kh // 2) * dy
                    ox = (kx - Kw // 2) * dx
                    kernel_idx = kz * Kh * Kw + ky * Kw + kx

                    for voxel_idx in range(num_voxels):
                        b, x_coord, y_coord, z_coord = coords_cpu[voxel_idx].tolist()
                        neighbor_key = (b, x_coord + oz, y_coord + oy, z_coord + ox)
                        neighbor_idx = coord_to_idx.get(neighbor_key)
                        if neighbor_idx is None:
                            continue
                        src_indices.append(neighbor_idx)
                        tgt_indices.append(voxel_idx)
                        kernel_indices.append(kernel_idx)

        neighbor_cache = (
            torch.tensor(src_indices, dtype=torch.long, device=device),
            torch.tensor(tgt_indices, dtype=torch.long, device=device),
            torch.tensor(kernel_indices, dtype=torch.long, device=device),
        )
        x.register_spatial_cache(cache_key, neighbor_cache)

    src_idx, tgt_idx, kernel_idx = neighbor_cache
    kernel_count = Kd * Kh * Kw
    weights = self.weight.reshape(Co, kernel_count, Ci).permute(1, 2, 0)
    out = torch.zeros(num_voxels, Co, device=device, dtype=dtype)

    if len(src_idx) > 0:
        for idx in range(kernel_count):
            mask = kernel_idx == idx
            if not mask.any():
                continue
            sources = src_idx[mask]
            targets = tgt_idx[mask]
            edge_out = feats[sources] @ weights[idx]
            out.scatter_add_(0, targets.unsqueeze(1).expand(-1, Co), edge_out)

    if self.bias is not None:
        out = out + self.bias

    return x.replace(out)


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError("SparseInverseConv3d is not implemented for conv_none")


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError("SparseInverseConv3d is not implemented for conv_none")
