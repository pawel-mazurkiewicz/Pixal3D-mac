# Pixal3D Apple Silicon Port — Distilled Knowledge Base

This document serves as the condensed source of truth for the porting of Pixal3D to Apple Silicon (MPS), distilling findings from multiple research sessions.

## 🎯 Current Mission & Immediate Next Steps (Verbatim)

> **Top-of-queue for next session: capture CUDA-side stage-07 FDG tensors on a one-time CUDA rental** so we can replay only Mac's extractor + cleanup chain on CUDA's inputs. Expected outcome based on this session's finding: sealed shell. That would *independently* confirm from the other side that the sieve is upstream of extraction, leaving the MPS DiT divergence as the sole remaining target.

**Next session START HERE — capture CUDA-side stage-07 FDG tensors on a CUDA rental box.**

`save_mesh_checkpoint` (`generate_mps.py:1431`) already saves `fdg_coords / fdg_dual_vertices / fdg_intersected / fdg_split_weight` alongside `vertices/faces`. Run `generate_mps.py --save-mesh ...` once on a CUDA rental with the same image+seed (`1_img.png`, seed=42) and capture the resulting `.npz`.

With that fixture in hand, the diagnostic on Mac is ~3 seconds:

```python
import numpy as np, torch
from pixal3d.utils.mesh_extract import flexible_dual_grid_to_mesh

d = np.load("cuda_stage07_fdg.npz")
v, f = flexible_dual_grid_to_mesh(
    torch.from_numpy(d["fdg_coords"]).int(),
    torch.from_numpy(d["fdg_dual_vertices"]).float(),
    torch.from_numpy(d["fdg_intersected"]).bool(),
    torch.from_numpy(d["fdg_split_weight"]).float(),
    aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
    grid_size=int(d["resolution"]),
)
# then run the Mac cleanup chain and export GLB; compare to CUDA's GLB.
```

Expected based on session 8's finding: **sealed shell**. That confirms from the input side that the sieve is upstream of FDG extraction. After that the only remaining target is the MPS DiT divergence itself, which has been the parent problem since sessions 5/6.

---

## 🛠 Core Technical Architecture (Current State)

The port is centered around `generate_mps.py` with the following key integrations:

- **NAF Model:** Patched `natten.na2d` with a pure-PyTorch index-gather neighborhood-attention implementation (tiled to avoid `MTLBuffer` size limits).
- **Mesh Extraction:** Uses a CPU fallback for `flexible_dual_grid_to_mesh` in `pixal3d/utils/mesh_extract.py`. Recently vendored upstream logic verbatim with a dict-based hashmap shim; proven bit-identical to CUDA on identical inputs.
- **UV Unwrapping:** 
    - **Blender Backend:** Uses a subprocess to a local Blender install (`_blender_uv_unwrap.py`) for decimation and unwrapping.
    - **Cube Projection (Retired):** Attempted manual 6-chart cube projection; proven non-viable compared to GT's dense mosaic atlas.
- **Texture Baking:** KDTree-IDW sampling from voxel attributes to texels. Search radius scaled in voxel units (`--bake-search-voxels`) to handle decimation.
- **Export Pipeline:**
    - **Rotation:** Fixed to negate glTF Z and flip face winding to match HF demo.
    - **Materials:** Forced `alphaMode = OPAQUE` and `doubleSided = True` to prevent "x-ray" transparency.
- **Checkpointing:** `--save-mesh` / `--load-mesh` allows iterating on bake/UV/export without re-running the 8-minute generation.

---

## 💡 Indispensable Knowledge & Findings

### 1. The Geometry "Sieve" Problem
- **Symptom:** Mac outputs exhibit thousands of small surface holes ("sieve"), whereas CUDA produces a sealed shell.
- **Finding:** The sieve is **upstream of extraction**. 
    - The cleanup chain was ruled out via stage-08 replay (CUDA mesh $\rightarrow$ Mac cleanup = sealed shell).
    - The FDG extractor was ruled out via bit-identical comparison on same inputs.
- **Root Cause:** The MPS DiT/VAE produces fundamentally different `(coords, dual_vertices, intersected, split_weight)` than CUDA for the same seed.

### 2. Model Divergence (CUDA vs. MPS)
- **Numerical Drift:** Pervasive drift observed in the DiT forward pass (LayerNorm/RMSNorm reductions, GEMM rounding, RoPE accuracy).
- **Ruled Out:** 
    - MoGe-2 camera parameters (shrank gap but didn't close it).
    - Same-seed RNG (divergence starts at step 0).
    - SDPA Backend: `torch.nn.attention.sdpa_kernel` is a no-op on MPS as of torch 2.12; all variants produce identical output.
    - Attention Implementation: Naive (unfused) attention is slightly worse than fused SDPA.

### 3. Simplification & Topology
- **The Decimator Gap:** Standard QEM (e.g., pymeshlab) floors at ~1.9M faces on Pixal3D meshes because it refuses to collapse non-manifold edges.
- **PAMO Algorithm:** The CUDA reference (`cumesh.simplify`) uses the PAMO parallel-edge-collapse algorithm, which handles non-manifold geometry correctly and produces visually uniform face density.
- **Metal Port Issues:** The precompiled Metal `simplify` kernel was found to be missing the winding-flip-rejection check, leading to inverted faces and "puddle-shaped" artifacts.

### 4. Hardware & Environment Heuristics
- **MTLBuffer Limits:** Individual buffer allocations are capped. Large tensor operations must be tiled/chunked.
- **Blender Performance:** `bpy.ops` is orders of magnitude slower than `bmesh.ops` for bulk operations on large meshes.
- **Metal Precision:** Global `fp32` casting crashes Metal due to `NDArrayMatrixMultiplication` dtype mismatches. Targeted submodule `.float()` is required.

---

## 🚫 Retired / Disproven Paths (Do Not Re-explore)

- **Texture-VAE Precision:** VAEs proven deterministic; "muted" textures are caused by upstream DiT drift.
- **Pre-cleaning FDG NMEs:** Cleanup chain is already at its empirical floor; the damage is in the input.
- **`metal-flash-attention`:** MPS SDPA is already at the fp32 precision floor; attention is not the bottleneck.
- **MLX-native Port:** Produces visibly worse output than the PyTorch/MPS port.
- **Cube Projection UVs:** Fundamentally wrong approach; the target is a dense, angle-clustered mosaic atlas.
- **Global FP32 Upcast:** Causes Metal kernel crashes.
