"""
CLI-only Pixal3D image-to-GLB generation for Apple Silicon MPS.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _configure_mps_environment():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(ROOT / "autotune_cache.json"))
    os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "0")

    if "SPARSE_CONV_BACKEND" not in os.environ:
        try:
            import flex_gemm  # noqa: F401
            os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"
        except (ImportError, RuntimeError, OSError):
            os.environ["SPARSE_CONV_BACKEND"] = "none"


_configure_mps_environment()


MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
MODEL_PATH = "TencentARC/Pixal3D"
np = None
torch = None
Image = None
Pixal3DImageTo3DPipeline = None

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

# Pixal3D's mesh extractor already outputs in Y-up convention (glTF-native),
# so no rotation is needed for export — the pre-rotation AABB has its
# tallest extent on Y.  The previous matrix here was a bogus 180-degree
# rotation around (0, 1, -1) that flipped X horizontally AND swapped Y/Z
# with sign flips, leaving the model lying on its side AND mirrored.
# Identity matrix below preserves Pixal3D's native orientation.
EXPORT_ROTATION_ROWS = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]
EXPORT_ROTATION = None


def load_runtime_deps():
    global np, torch, Image, Pixal3DImageTo3DPipeline, EXPORT_ROTATION

    import numpy as _np
    import torch as _torch
    from PIL import Image as _Image
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline as _Pixal3DImageTo3DPipeline

    np = _np
    torch = _torch
    Image = _Image
    Pixal3DImageTo3DPipeline = _Pixal3DImageTo3DPipeline
    EXPORT_ROTATION = np.array(EXPORT_ROTATION_ROWS, dtype=np.float64)

    # Patch NATTEN's na2d before torch.hub.load loads the NAF upsampler.
    # The hub-cached attentions.py hardcodes backend="cutlass-fna" which in
    # this NATTEN wheel requires CUDA + libnatten.  The only non-CUDA backend
    # exposed is flex-fna, which (a) refuses MPS tensors and (b) on CPU runs
    # uncompiled and materialises the full Sq x Sk scores matrix — at NAF's
    # 128x128 HR map that is ~4 GB per intermediate, easily ballooning into
    # hundreds of GB once flex_attention's eager path keeps several copies
    # plus NATTEN's BlockMask machinery alive.
    #
    # Instead we implement neighborhood attention directly in PyTorch using
    # an index-gather: every query gathers its kh*kw in-bounds neighbors with
    # NATTEN's shifted-window rule (window slides inward at borders so the
    # neighbour count stays constant), then a single masked SDPA per query.
    # Memory is O(B*N*Sq*K*D_v) — a few GB at NAF sizes — and it runs on
    # MPS directly without any CPU round-trip.  Numerical match vs.
    # NATTEN flex-fna is at fp32 noise (~5e-7).
    try:
        import natten as _natten
        import torch.nn.functional as _F

        # Process queries in tiles so peak working-set per call is small.
        # Apple Silicon caps individual MTLBuffer allocations well below total
        # unified memory; at NAF's 512x512 HR map a full v_neigh tensor would
        # be ~80 GB and the MPS allocator refuses to create the buffer even
        # with plenty of free RAM.  Tiling keeps each chunk under a few hundred
        # MB while preserving exact NA semantics (each query is independent).
        _NA2D_QUERY_CHUNK = 2048

        def _torch_na2d(query, key, value, kernel_size, dilation=1,
                        scale=None, stride=1, backend=None, **kw):
            if stride != 1:
                raise NotImplementedError(
                    f"_torch_na2d: stride={stride} not supported"
                )
            kh, kw_ = ((kernel_size, kernel_size)
                       if isinstance(kernel_size, int) else kernel_size)
            dh, dw = ((dilation, dilation)
                      if isinstance(dilation, int) else dilation)
            B, H, W, N, Dq = query.shape
            Dv = value.shape[-1]
            if scale is None:
                scale = Dq ** -0.5
            dev = query.device
            HW = H * W
            K = kh * kw_

            half_h = (kh // 2) * dh
            half_w = (kw_ // 2) * dw
            ci = _torch.arange(H, device=dev).clamp(min=half_h,
                                                    max=H - 1 - half_h)
            cj = _torch.arange(W, device=dev).clamp(min=half_w,
                                                    max=W - 1 - half_w)
            off_h = dh * (_torch.arange(kh, device=dev) - kh // 2)
            off_w = dw * (_torch.arange(kw_, device=dev) - kw_ // 2)
            row_idx = ci[:, None] + off_h[None, :]
            col_idx = cj[:, None] + off_w[None, :]
            abs_idx_2d = (row_idx[:, None, :, None] * W
                          + col_idx[None, :, None, :]).reshape(HW, K)

            k_flat = key.permute(0, 3, 1, 2, 4).reshape(B * N, HW, Dq)
            v_flat = value.permute(0, 3, 1, 2, 4).reshape(B * N, HW, Dv)
            q_flat = query.permute(0, 3, 1, 2, 4).reshape(B * N, HW, Dq)

            out = _torch.empty(B * N, HW, Dv,
                               device=dev, dtype=query.dtype)
            chunk = _NA2D_QUERY_CHUNK
            for start in range(0, HW, chunk):
                end = min(start + chunk, HW)
                idx_chunk = abs_idx_2d[start:end].reshape(-1)  # ((end-start)*K,)
                k_n = k_flat.index_select(1, idx_chunk).reshape(
                    B * N, end - start, K, Dq)
                v_n = v_flat.index_select(1, idx_chunk).reshape(
                    B * N, end - start, K, Dv)
                q_c = q_flat[:, start:end, :]
                scores = _torch.einsum('bsd,bskd->bsk', q_c, k_n) * scale
                weights = scores.softmax(dim=-1)
                out[:, start:end, :] = _torch.einsum(
                    'bsk,bskd->bsd', weights, v_n)
                del k_n, v_n, scores, weights

            return out.reshape(B, N, H, W, Dv).permute(0, 2, 3, 1, 4).contiguous()

        _natten.na2d = _torch_na2d
        import natten.functional as _nf
        if hasattr(_nf, "na2d"):
            _nf.na2d = _torch_na2d
    except (ImportError, AttributeError):
        pass  # natten not installed; NAF model will fail gracefully later


def resolve_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is not available. Re-run with --device cpu.")
        return torch.device("mps")
    return torch.device("cpu")


def maybe_empty_cache(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def build_image_cond_model(config: dict, device: torch.device):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor

    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    model.to(device)
    return model


def load_moge_model(device: torch.device, model_name: str = MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(model_name).to(device)
    model.eval()
    return model


def init_pipeline(model_path: str, device: torch.device):
    print(f"[Pipeline] Loading from {model_path}...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)
    pipeline.to(device)

    print("[ImageCond] Building DinoV3 projection models...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"], device)
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"], device)
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"], device)
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"], device)

    print("[NAF] Pre-loading NAF upsampler models where configured...")
    for attr in (
        "image_cond_model_ss",
        "image_cond_model_shape_512",
        "image_cond_model_shape_1024",
        "image_cond_model_tex_1024",
    ):
        model = getattr(pipeline, attr)
        if getattr(model, "use_naf_upsample", False):
            model._load_naf()

    return pipeline


def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    return float((focal_length * resolution / 32.0).item())


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])
    grid_point = grid_point.to(torch.float32) @ rotation_matrix.T
    grid_point = grid_point / mesh_scale / 2
    xw, yw = grid_point[0].item(), grid_point[1].item()
    xt = float(target_point[0].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def get_camera_params_wild_moge(
    image_path: Path,
    moge_model,
    device: torch.device,
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
):
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)

    with torch.no_grad():
        output = moge_model.infer(image_tensor)

    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx = intrinsics[0, 0] * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))
    distance = distance_from_fov(
        camera_angle_x,
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale,
        image_resolution,
    )["distance_from_x"]
    return {"camera_angle_x": camera_angle_x, "distance": distance, "mesh_scale": mesh_scale}


def sampler_overrides(args, stage: str) -> dict:
    params = {}
    steps = getattr(args, f"{stage}_steps")
    guidance_strength = getattr(args, f"{stage}_guidance_strength")
    guidance_rescale = getattr(args, f"{stage}_guidance_rescale")
    rescale_t = getattr(args, f"{stage}_rescale_t")

    if args.steps is not None:
        params["steps"] = args.steps
    if steps is not None:
        params["steps"] = steps
    if guidance_strength is not None:
        params["guidance_strength"] = guidance_strength
    if guidance_rescale is not None:
        params["guidance_rescale"] = guidance_rescale
    if rescale_t is not None:
        params["rescale_t"] = rescale_t
    return params


def output_glb_path(output: str) -> Path:
    path = Path(output)
    if path.suffix.lower() != ".glb":
        path = path.with_suffix(".glb")
    return path


def rotated_vertices(vertices: np.ndarray) -> np.ndarray:
    return vertices @ EXPORT_ROTATION[:3, :3].T + EXPORT_ROTATION[:3, 3]


def simplify_vertices_faces(vertices: np.ndarray, faces: np.ndarray, target_faces: int):
    if len(faces) <= target_faces:
        return vertices, faces
    try:
        import fast_simplification
    except ImportError:
        print("  Warning: fast_simplification is not installed; using full mesh.")
        return vertices, faces

    ratio = max(0.0, min(1.0, 1.0 - (target_faces / len(faces))))
    print(f"  Simplifying mesh: {len(faces):,} -> ~{target_faces:,} faces")
    simp_vertices, simp_faces = fast_simplification.simplify(vertices, faces, ratio)
    return simp_vertices.astype(np.float32), simp_faces.astype(np.int64)


def patch_o_voxel_grid_sample_if_needed(postprocess_module):
    if getattr(postprocess_module, "_HAS_FLEX_GEMM", True):
        return

    import torch.nn.functional as F

    def _grid_sample_3d_fix(feats, coords, shape, grid, mode="trilinear"):
        batch, channels = shape[0], shape[1]
        depth, height, width = shape[2], shape[3], shape[4]
        dense_vol = torch.zeros(batch, channels, depth, height, width, dtype=feats.dtype, device=feats.device)
        if coords.shape[-1] == 3:
            coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)
        batch_idx = coords[:, 0].long()
        cx = coords[:, 1].long()
        cy = coords[:, 2].long()
        cz = coords[:, 3].long()
        dense_vol[batch_idx, :, cx, cy, cz] = feats
        grid_norm = torch.stack([
            grid[..., 2] / (width - 1) * 2 - 1,
            grid[..., 1] / (height - 1) * 2 - 1,
            grid[..., 0] / (depth - 1) * 2 - 1,
        ], dim=-1).reshape(batch, 1, 1, -1, 3)
        sampled = F.grid_sample(dense_vol, grid_norm, mode="bilinear", align_corners=True, padding_mode="border")
        samples = grid.shape[1]
        return sampled.reshape(batch, channels, samples).permute(0, 2, 1).reshape(batch * samples, channels)

    postprocess_module._grid_sample_3d = _grid_sample_3d_fix


def export_geometry_only(vertices: np.ndarray, faces: np.ndarray, glb_path: Path):
    import trimesh

    mesh = trimesh.Trimesh(vertices=rotated_vertices(vertices), faces=faces, process=False)
    mesh.export(str(glb_path))
    print(f"[Export] Geometry-only GLB saved to {glb_path}")


def try_export_metal_glb(mesh, pipeline, resolution: int, vertices: np.ndarray, faces: np.ndarray, glb_path: Path, args) -> bool:
    if args.no_texture or args.force_texture_fallback:
        return False

    try:
        import o_voxel.postprocess as postprocess
    except (ImportError, RuntimeError, OSError, AttributeError) as exc:
        print(f"[Export] o_voxel.postprocess unavailable: {exc}")
        return False

    if not hasattr(postprocess, "to_glb"):
        print("[Export] o_voxel.postprocess has no to_glb; using fallback baker.")
        return False

    try:
        patch_o_voxel_grid_sample_if_needed(postprocess)
        target_faces = min(args.bake_face_target, len(faces))
        bake_vertices, bake_faces = simplify_vertices_faces(vertices, faces, target_faces)
        vertices_t = torch.from_numpy(bake_vertices).float()
        faces_t = torch.from_numpy(bake_faces.astype(np.int32)).int()
        print(f"[Export] Baking PBR textures via o_voxel ({args.texture_size}x{args.texture_size})...")
        glb = postprocess.to_glb(
            vertices=vertices_t,
            faces=faces_t,
            attr_volume=mesh.attrs.detach().cpu(),
            coords=mesh.coords.detach().cpu(),
            attr_layout=getattr(mesh, "layout", pipeline.pbr_attr_layout),
            grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=target_faces,
            texture_size=args.texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=True,
        )
        glb.apply_transform(EXPORT_ROTATION)
        try:
            glb.export(str(glb_path), extension_webp=True)
        except TypeError:
            glb.export(str(glb_path))
        print(f"[Export] Textured GLB saved to {glb_path}")
        return True
    except Exception as exc:
        print(f"[Export] o_voxel texture bake failed: {exc}")
        return False


def export_fallback_texture_glb(mesh, vertices: np.ndarray, faces: np.ndarray, glb_path: Path, args):
    from pixal3d.utils.texture_baker import (
        _find_blender_exe, bake_texture, export_glb_with_texture, uv_unwrap,
    )

    target_faces = min(args.bake_face_target, len(faces))

    # Decide whether Blender will handle decimation+unwrap end-to-end.  When
    # it does, we skip Python's fast_simplification pre-pass entirely because
    # Blender's iterative Decimate-Collapse handles topology constraints
    # (non-manifold edges, isolated components) that fast_simplification
    # refuses to touch.  The xatlas path doesn't have that option, so we
    # still pre-decimate in Python for it (best effort).
    will_use_blender = args.uv_unwrap == "blender" or (
        args.uv_unwrap == "auto"
        and (args.blender_exe or _find_blender_exe()) is not None
    )

    if args.skip_decimation:
        # Full-fidelity path: don't decimate at all.  Blender script still
        # runs cleanup (remove_doubles + dissolve_degenerate) but doesn't
        # apply Decimate or voxel remesh.
        bake_vertices, bake_faces = vertices, faces
        target_count_for_uv = None
        voxel_size_for_uv = None
        print(f"[Export] --skip-decimation: baking onto full {len(faces):,}-face mesh")
    elif will_use_blender:
        bake_vertices, bake_faces = vertices, faces
        target_count_for_uv = target_faces
        voxel_size_for_uv = args.voxel_remesh_size if args.voxel_remesh_size > 0 else None
    else:
        bake_vertices, bake_faces = simplify_vertices_faces(vertices, faces, target_faces)
        target_count_for_uv = None
        voxel_size_for_uv = None

    print(f"[Export] Baking PBR textures via KDTree/{args.uv_unwrap} ({args.texture_size}x{args.texture_size})...")
    print(f"  UV unwrapping (mode={args.uv_unwrap}, "
          f"voxel_size={voxel_size_for_uv}, "
          f"target_count={target_count_for_uv}, "
          f"uv_method={args.uv_method})...")
    new_vertices, new_faces, uvs, _ = uv_unwrap(
        bake_vertices, bake_faces,
        mode=args.uv_unwrap, blender_exe=args.blender_exe,
        target_count=target_count_for_uv,
        voxel_size=voxel_size_for_uv,
        uv_method=args.uv_method,
        fill_holes_sides=args.fill_holes_sides,
    )
    base_color_img, mr_img, _ = bake_texture(
        new_vertices,
        new_faces,
        uvs,
        mesh.coords.detach().cpu().float(),
        mesh.attrs.detach().cpu().float(),
        mesh.origin.detach().cpu().float(),
        mesh.voxel_size,
        texture_size=args.texture_size,
        search_voxels=args.bake_search_voxels,
    )
    export_glb_with_texture(
        rotated_vertices(new_vertices),
        new_faces,
        uvs,
        base_color_img,
        mr_img,
        str(glb_path),
    )
    print(f"[Export] Textured GLB saved to {glb_path}")


def watchdog_help_message() -> str:
    return (
        "\nERROR: The decoder produced an empty mesh.\n"
        "On Apple Silicon this is commonly caused by a long-running Metal kernel\n"
        "being interrupted by the macOS GPU watchdog. Look above for Metal errors\n"
        "such as kIOGPUCommandBufferCallbackErrorImpactingInteractivity.\n"
        "\n"
        "Workarounds:\n"
        "  1. Retry with --steps 1 to verify the model path and export path.\n"
        "  2. Reduce load by closing display-heavy apps or running headless over SSH.\n"
        "  3. Try SPARSE_CONV_BACKEND=none python generate_mps.py ... for the slower fallback.\n"
        "  4. Try MTL_CAPTURE_ENABLED=1 python generate_mps.py ... to extend Metal debug timeouts.\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Pixal3D GLB from one image on Apple Silicon MPS")
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help=(
            "Path to input image.  Required unless --load-mesh PATH is given."
        ),
    )
    parser.add_argument("--output", default="output_3d", help="Output GLB path or basename (default: output_3d)")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Pixal3D model path or Hugging Face repo")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps", help="Runtime device (default: mps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--pipeline-type", choices=["1024_cascade", "1536_cascade"], default="1024_cascade")
    parser.add_argument("--steps", type=int, default=None, help="Override sampler steps for all stages")
    parser.add_argument("--ss-steps", type=int, default=None, help="Sparse-structure sampler steps")
    parser.add_argument("--shape-steps", type=int, default=None, help="Shape SLat sampler steps")
    parser.add_argument("--tex-steps", type=int, default=None, help="Texture SLat sampler steps")
    parser.add_argument("--ss-guidance-strength", "--ss-guidance", dest="ss_guidance_strength", type=float, default=None)
    parser.add_argument("--shape-guidance-strength", "--shape-guidance", dest="shape_guidance_strength", type=float, default=None)
    parser.add_argument("--tex-guidance-strength", "--tex-guidance", dest="tex_guidance_strength", type=float, default=None)
    parser.add_argument("--ss-guidance-rescale", type=float, default=None)
    parser.add_argument("--shape-guidance-rescale", type=float, default=None)
    parser.add_argument("--tex-guidance-rescale", type=float, default=None)
    parser.add_argument("--ss-rescale-t", type=float, default=None)
    parser.add_argument("--shape-rescale-t", type=float, default=None)
    parser.add_argument("--tex-rescale-t", type=float, default=None)
    parser.add_argument("--max-num-tokens", type=int, default=49152)
    parser.add_argument("--texture-size", type=int, default=2048, choices=[512, 1024, 2048, 4096])
    parser.add_argument(
        "--voxel-remesh-size",
        type=float,
        default=0.0,
        help=(
            "Voxel size for the Blender REMESH modifier (world units).  "
            "Default 0 (DISABLED).  Enable with a positive value (e.g. 0.004) "
            "ONLY if your input mesh is watertight — Pixal3D's mesh extractor "
            "outputs open shells, and voxel remesh on a non-watertight mesh "
            "produces swiss-cheese output (Blender T70925).  When enabled "
            "and the mesh is watertight, ~0.004 gives 100-200k triangles, "
            "and 0.002 gives 300-600k."
        ),
    )
    parser.add_argument(
        "--fill-holes-sides",
        type=int,
        default=12,
        help=(
            "When the Blender UV path is used, close any hole whose boundary "
            "has at most this many edges (mesh.fill_holes(sides=N)).  "
            "Pixal3D's CPU mesh extractor leaves pinprick holes (~3-8 edges) "
            "wherever it couldn't close a cell.  Set to 0 to skip."
        ),
    )
    parser.add_argument(
        "--uv-method",
        choices=["cube", "smart"],
        default="cube",
        help=(
            "Blender UV projection method.  'cube' (default): 6 axis-"
            "aligned charts via cube_project (~70%% atlas utilisation, "
            "minor seam distortion, ideal for texture baking).  'smart': "
            "angle-based smart_project (smoother charts but ~30%% atlas "
            "utilisation because Blender 5.x pack_islands does not scale "
            "as advertised)."
        ),
    )
    parser.add_argument(
        "--bake-search-voxels",
        type=int,
        default=50,
        help=(
            "IDW search radius for the texture bake, in *voxel units*.  Texels "
            "further than this many voxels from any sparse voxel get zero "
            "weight (atlas gutter protection).  After heavy decimation a "
            "single simplified face can span dozens of voxels, so the radius "
            "must comfortably exceed the per-face footprint.  Defaults to 50; "
            "lower to ~20 for cleaner gutters on lightly-decimated meshes, "
            "raise to ~100 if textures still look black on extreme "
            "decimation."
        ),
    )
    parser.add_argument(
        "--bake-face-target",
        type=int,
        default=100000,
        help=(
            "Decimation target for the texture-bake fallback path (KDTree + "
            "xatlas).  xatlas wall-clock scales close to linearly with face "
            "count; the texture itself is bounded by --texture-size, so at "
            "1024^2 atlases the bake quality is essentially identical between "
            "100k and 200k faces but unwrap time roughly halves.  Raise to "
            "200000+ for very fine surface detail; lower to 50000 for "
            "fastest export."
        ),
    )
    parser.add_argument(
        "--skip-decimation",
        action="store_true",
        help=(
            "Skip all mesh decimation (fast_simplification + Blender Decimate"
            "-Collapse) and bake onto the full original mesh.  Highest "
            "fidelity at the cost of bake time: 8.4M faces × cube_project + "
            "2048^2 atlas ≈ 2-3 min vs ~50s for the decimated default.  "
            "Ignores --bake-face-target.  GLB file size grows roughly with "
            "vertex count (full mesh → ~150 MB GLB)."
        ),
    )
    parser.add_argument("--no-texture", action="store_true", help="Export geometry-only GLB")
    parser.add_argument("--force-texture-fallback", action="store_true", help="Skip o_voxel.to_glb and use KDTree/xatlas")
    parser.add_argument(
        "--uv-unwrap",
        choices=["auto", "blender", "xatlas"],
        default="auto",
        help=(
            "UV unwrapper for the texture-bake fallback path.  "
            "'auto' (default) uses Blender's Smart UV Project if a Blender "
            "executable is found (a few seconds), else falls back to xatlas "
            "(can take 20-60 minutes on Pixal3D-style HR meshes).  "
            "'blender' requires Blender; 'xatlas' forces the pure-Python "
            "path even when Blender is available."
        ),
    )
    parser.add_argument(
        "--blender-exe",
        default=None,
        help=(
            "Path to the Blender executable.  Auto-detected from $PATH and "
            "/Applications/Blender.app on macOS if omitted."
        ),
    )
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument(
        "--save-mesh",
        default=None,
        help=(
            "After generation, dump the raw mesh + voxel attributes to this "
            ".npz path so subsequent UV/bake/export iterations can skip the "
            "~8-10 min generation step.  Pair with --load-mesh to resume."
        ),
    )
    parser.add_argument(
        "--load-mesh",
        default=None,
        help=(
            "Resume from a previously written --save-mesh checkpoint.  Skips "
            "image preprocessing, MoGe camera estimation, and the full "
            "generation pipeline; jumps straight to UV unwrap + texture bake "
            "+ GLB export."
        ),
    )
    return parser.parse_args()


class _LoadedMesh:
    """Minimal mesh-like wrapper exposing the attribute surface that
    ``export_fallback_texture_glb`` needs (``vertices``, ``faces``, ``coords``,
    ``attrs``, ``origin``, ``voxel_size``).  Used when resuming from a
    ``--save-mesh`` checkpoint."""

    __slots__ = ("vertices", "faces", "coords", "attrs", "origin", "voxel_size")


def save_mesh_checkpoint(path, mesh, vertices, faces, resolution):
    """Persist the generation result so we can iterate on UV/bake/export
    without re-running the ~8-10 min sampling pipeline."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out),
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        coords=mesh.coords.detach().cpu().numpy(),
        attrs=mesh.attrs.detach().cpu().numpy(),
        origin=mesh.origin.detach().cpu().numpy().astype(np.float32),
        voxel_size=np.float32(float(mesh.voxel_size)),
        resolution=np.int32(int(resolution)),
    )
    size_mb = out.stat().st_size / 1e6
    print(f"[Checkpoint] Saved mesh to {out} ({size_mb:.1f} MB)")


def load_mesh_checkpoint(path):
    """Inverse of :func:`save_mesh_checkpoint`.  Returns
    ``(mesh_wrapper, vertices, faces, resolution)``."""
    data = np.load(str(path))
    vertices = data["vertices"]
    faces = data["faces"]
    mesh = _LoadedMesh()
    mesh.vertices = torch.from_numpy(vertices)
    mesh.faces = torch.from_numpy(faces)
    mesh.coords = torch.from_numpy(data["coords"])
    mesh.attrs = torch.from_numpy(data["attrs"])
    mesh.origin = torch.from_numpy(data["origin"])
    mesh.voxel_size = float(data["voxel_size"])
    resolution = int(data["resolution"])
    return mesh, vertices, faces, resolution


def main():
    args = parse_args()
    glb_path = output_glb_path(args.output)
    glb_path.parent.mkdir(parents=True, exist_ok=True)

    load_runtime_deps()
    device = resolve_device(args.device)

    print("=" * 60)
    print("Pixal3D Apple Silicon CLI")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Sparse conv backend: {os.environ.get('SPARSE_CONV_BACKEND')}")
    print(f"Sparse attention backend: {os.environ.get('SPARSE_ATTN_BACKEND')}")

    t0 = time.time()
    pipeline = None
    from_checkpoint = bool(args.load_mesh)

    if from_checkpoint:
        ckpt_path = Path(args.load_mesh)
        if not ckpt_path.exists():
            raise SystemExit(f"Mesh checkpoint not found: {ckpt_path}")
        print(f"[Checkpoint] Loading mesh from {ckpt_path}")
        mesh, vertices, faces, resolution = load_mesh_checkpoint(ckpt_path)
        print(f"[Mesh] {vertices.shape[0]:,} vertices, {faces.shape[0]:,} triangles "
              f"(resolution={resolution})")
    else:
        if not args.image:
            raise SystemExit(
                "An input image is required unless --load-mesh PATH is given"
            )
        input_path = Path(args.image)
        if not input_path.exists():
            raise SystemExit(f"Input image not found: {input_path}")

        pipeline = init_pipeline(args.model_path, device)

        print("[MoGe] Loading camera estimator...")
        moge_model = load_moge_model(device)

        print(f"[Input] {input_path}")
        image = Image.open(input_path)
        image_preprocessed = pipeline.preprocess_image(image)

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=str(glb_path.parent))
        tmp_path = Path(tmp_file.name)
        tmp_file.close()
        try:
            image_preprocessed.save(tmp_path)
            print("[MoGe] Estimating camera parameters...")
            camera_params = get_camera_params_wild_moge(
                tmp_path,
                moge_model,
                device=device,
                mesh_scale=args.mesh_scale,
                extend_pixel=args.extend_pixel,
                image_resolution=args.image_resolution,
            )
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

        del moge_model
        maybe_empty_cache(device)
        print(f"[MoGe] camera_angle_x={camera_params['camera_angle_x']:.4f}, distance={camera_params['distance']:.4f}")

        torch.manual_seed(args.seed)
        print(f"[Generate] pipeline={args.pipeline_type}, seed={args.seed}")
        t_gen = time.time()
        try:
            mesh_list, (shape_slat, tex_slat, resolution) = pipeline.run(
                image_preprocessed,
                camera_params=camera_params,
                seed=args.seed,
                sparse_structure_sampler_params=sampler_overrides(args, "ss"),
                shape_slat_sampler_params=sampler_overrides(args, "shape"),
                tex_slat_sampler_params=sampler_overrides(args, "tex"),
                preprocess_image=False,
                return_latent=True,
                pipeline_type=args.pipeline_type,
                max_num_tokens=args.max_num_tokens,
            )
        except (IndexError, AssertionError) as exc:
            if any(sig in str(exc) for sig in ("non-zero size", "BVH needs at least 8 triangles")):
                print(watchdog_help_message())
                sys.exit(2)
            raise

        mesh = mesh_list[0]
        vertices = mesh.vertices.detach().cpu().numpy()
        faces = mesh.faces.detach().cpu().numpy()
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            print(watchdog_help_message())
            sys.exit(2)

        print(f"[Mesh] {vertices.shape[0]:,} vertices, {faces.shape[0]:,} triangles")
        print(f"[Timing] Generation: {time.time() - t_gen:.1f}s")

        if args.save_mesh:
            save_mesh_checkpoint(args.save_mesh, mesh, vertices, faces, resolution)

    has_voxels = hasattr(mesh, "attrs") and mesh.attrs is not None and hasattr(mesh, "coords") and mesh.coords is not None
    if args.no_texture or not has_voxels:
        export_geometry_only(vertices, faces, glb_path)
    elif from_checkpoint:
        # o_voxel.postprocess.to_glb needs the live pipeline + CUDA; when
        # resuming from a checkpoint we always go through the texture-bake
        # fallback (the only one that doesn't need o_voxel anyway).
        export_fallback_texture_glb(mesh, vertices, faces, glb_path, args)
    elif not try_export_metal_glb(mesh, pipeline, resolution, vertices, faces, glb_path, args):
        export_fallback_texture_glb(mesh, vertices, faces, glb_path, args)

    if not glb_path.exists() or glb_path.stat().st_size == 0:
        raise SystemExit(f"Export failed: GLB was not written or is empty: {glb_path}")

    print(f"[Done] GLB saved to {glb_path} ({glb_path.stat().st_size:,} bytes)")
    print(f"[Timing] Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
