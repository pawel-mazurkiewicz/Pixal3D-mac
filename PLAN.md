# Pixal3D Apple Silicon MPS CLI Port

## Summary
Build a CLI-only Apple Silicon path for Pixal3D image-to-GLB generation by transplanting the working Trellis.2 MPS compatibility patterns from `/Users/pawelma/code/ai/trellis-mac` into Pixal3D’s copied Trellis-style runtime.

The first deliverable is a command like:

```bash
python generate_mps.py input.png --output output_3d --pipeline-type 1024_cascade --steps 8
```

It should load `TencentARC/Pixal3D`, run Pixal3D generation on `mps`, and export a GLB. The web app, preview renderer, Hugging Face Spaces integration, and full CUDA-equivalent texture/render feature parity are out of scope for v1.

## Key Changes
- Add a new CLI entrypoint, `generate_mps.py`, based on Pixal3D `inference.py` and `trellis-mac/generate.py`.
- Set MPS-compatible env vars before importing torch or Pixal3D:
  - `PYTORCH_ENABLE_MPS_FALLBACK=1`
  - `ATTN_BACKEND=sdpa`
  - `SPARSE_ATTN_BACKEND=sdpa`
  - `SPARSE_CONV_BACKEND=flex_gemm` if Metal `flex_gemm` imports successfully, otherwise `none`
- Use `torch.device("mps")` by default, with optional `--device mps|cpu`.

## Implementation Changes
- Port sparse backend support from `trellis-mac`:
  - add `sdpa` and `naive` as valid sparse attention backends
  - add the SDPA padded variable-length attention path to Pixal3D sparse attention
  - add/copy the pure PyTorch `conv_none.py` sparse convolution fallback
- Replace runtime-critical CUDA assumptions with active-device logic:
  - pipeline loading and model movement should use `.to(device)`, not `.cuda()`
  - MoGe camera estimation should run on the selected device
  - DINOv3 projection conditioning models should expose/use a `device` property and move image tensors to that device
  - guard `torch.cuda.empty_cache()` and `torch.cuda.synchronize()` behind `torch.cuda.is_available()`
- Patch mesh decode/export behavior:
  - skip CUDA `cumesh` hole filling on MPS for v1
  - use the `trellis-mac` mesh extraction override/fallback if native `o_voxel.convert.flexible_dual_grid_to_mesh` is unavailable or CUDA-bound
  - keep tensors device-consistent during decode, then move mesh/export tensors to CPU before GLB export if mixed backend code requires it
- Implement GLB export using the safest working path:
  - first try Metal `o_voxel.postprocess.to_glb` if available
  - pre-simplify large meshes before texture baking using the `trellis-mac` face-target strategy
  - fall back to the `trellis-mac` KDTree/xatlas texture baker if Metal baking is unavailable
  - support `--no-texture` for geometry-only export
- Add CLI flags:
  - `--pipeline-type`, default `1024_cascade`
  - `--steps`, plus `--ss-steps`, `--shape-steps`, `--tex-steps`
  - guidance/rescale overrides matching Pixal3D inference defaults
  - `--max-num-tokens`, default `49152`
  - `--texture-size`, default `1024`
  - `--no-texture`
  - `--seed`, default `42`
  - `--output`, default `output_3d`

## Test Plan
- Static/import checks:
  - import Pixal3D with MPS env vars set
  - verify sparse config prints `Conv backend: flex_gemm` or `none`, and `Attention backend: sdpa`
- Model loading checks:
  - load `Pixal3DImageTo3DPipeline.from_pretrained("TencentARC/Pixal3D")`
  - build and move all four `DinoV3ProjFeatureExtractor` models to MPS
  - load MoGe on MPS
- Runtime checks:
  - run preprocessing and camera estimation on a sample image
  - run a low-step generation, e.g. `--steps 1 --pipeline-type 1024_cascade`
  - run normal generation, e.g. `--steps 8`
  - verify non-empty mesh vertices/faces before export
- Export checks:
  - export geometry-only GLB with `--no-texture`
  - export textured GLB via Metal path if available
  - force fallback texture path and verify GLB is written
- Acceptance criteria:
  - CLI completes on Apple Silicon without CUDA installed
  - output `.glb` exists and is non-empty
  - failures for known MPS/Metal watchdog or empty-mesh cases print actionable diagnostics instead of crashing obscurely

## Assumptions
- Optimize for a working CLI first; do not modify `app.py` for v1.
- Prefer MPS with CPU fallback over exact CUDA performance parity.
- Hole filling can be disabled initially.
- Texture baking quality can be slightly reduced in fallback mode as long as a valid GLB is produced.
- Existing dirty changes in `/Users/pawelma/code/ai/trellis-mac` must not be reverted or relied on unless explicitly inspected.
