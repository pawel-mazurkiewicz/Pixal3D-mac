import os
import argparse
import math
import time
import torch
import numpy as np
import cv2
from PIL import Image

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'

from pixal3d.pipelines import Pixal3DImageTo3DPipeline
import o_voxel

# ============================================================================
# Constants & Defaults
# ============================================================================

MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
MODEL_PATH = "TencentARC/Pixal3D"

# ============================================================================
# Fixture capture (CUDA-vs-MPS divergence analysis)
# ============================================================================
#
# Set PIXAL3D_DUMP_FIXTURES=/path/to/dir to dump intermediate tensors at every
# pipeline stage boundary.  Same dump format on CUDA and MPS — diff stage-by-
# stage to localise where the two backends first numerically diverge.
#
# Fixtures written (when enabled):
#   00_metadata.json         host/device/seed/image-SHA + args, on success
#   00_preprocessed_image.pt PIL image after rembg / square-pad
#   01_camera_params.pt      MoGe (or manual) camera dict
#   02_sparse_structure.pt   sample_sparse_structure() output
#   03a_shape_slat.pt        sample_shape_slat() output (non-cascade path)
#   03b_shape_slat_cascade.pt sample_shape_slat_cascade() output (cascade path)
#   04_shape_slat_decoded.pt decode_shape_slat() output
#   05_tex_slat.pt           sample_tex_slat() output
#   06_tex_slat_decoded.pt   decode_tex_slat() output
#   07_run_output.pt         pipeline.run() outer return — mesh + latents
#   08_to_glb_geometry.pt    o_voxel.postprocess.to_glb() result (geometry only)
#
# All tensors are detached + moved to CPU before saving so the .pt files are
# portable across devices.  Multi-call methods get a `_call{N}` suffix.
#
PIXAL3D_DUMP_FIXTURES = os.environ.get("PIXAL3D_DUMP_FIXTURES", "").strip()


def _fixture_to_cpu(x):
    """Recursively detach + move tensors to CPU for portable fixture saves."""
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _fixture_to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_fixture_to_cpu(v) for v in x)
    # Common non-tensor objects we want to capture by introspection
    if hasattr(x, "vertices") and hasattr(x, "faces"):
        # mesh-like (MeshWithVoxel from pipeline, trimesh.Geometry, etc.)
        out = {}
        for attr in ("vertices", "faces", "attrs", "coords", "uv",
                     "vertex_normals", "face_normals", "visual"):
            if hasattr(x, attr):
                v = getattr(x, attr)
                if v is not None:
                    try:
                        out[attr] = _fixture_to_cpu(v)
                    except Exception:
                        pass
        out["_repr"] = repr(x)[:200]
        return out
    return x


def _dump_fixture(name: str, payload, fixture_dir: str | None = None):
    """Save a fixture under <fixture_dir>/<name>.pt.  No-op if disabled."""
    target = fixture_dir if fixture_dir is not None else PIXAL3D_DUMP_FIXTURES
    if not target:
        return
    os.makedirs(target, exist_ok=True)
    out = os.path.join(target, f"{name}.pt")
    try:
        torch.save(_fixture_to_cpu(payload), out)
        sz = os.path.getsize(out)
        print(f"[fixture] {name} -> {out} ({sz / 1e6:.2f} MB)", flush=True)
    except Exception as exc:
        print(f"[fixture] WARN failed to dump {name}: {exc}", flush=True)


def _install_pipeline_fixture_hooks(pipeline, fixture_dir: str) -> None:
    """Monkey-patch pipeline stage methods to dump outputs after each call.

    Each method may be invoked more than once per inference (e.g. the cascade
    path calls shape_slat twice).  We append `_call{N}` so fixtures from
    multiple invocations don't overwrite each other.
    """
    import functools

    stage_map = {
        "sample_sparse_structure":      "02_sparse_structure",
        "sample_shape_slat":            "03a_shape_slat",
        "sample_shape_slat_cascade":    "03b_shape_slat_cascade",
        "decode_shape_slat":            "04_shape_slat_decoded",
        "sample_tex_slat":              "05_tex_slat",
        "decode_tex_slat":              "06_tex_slat_decoded",
    }

    call_counter: dict[str, int] = {n: 0 for n in stage_map.values()}

    for method_name, fixture_name in stage_map.items():
        original = getattr(pipeline, method_name, None)
        if original is None or not callable(original):
            continue

        def make_wrapper(orig, fname):
            @functools.wraps(orig)
            def wrapped(*args, **kwargs):
                result = orig(*args, **kwargs)
                idx = call_counter[fname]
                call_counter[fname] += 1
                tag = fname if idx == 0 else f"{fname}_call{idx}"
                _dump_fixture(tag, result, fixture_dir=fixture_dir)
                return result
            return wrapped

        setattr(pipeline, method_name, make_wrapper(original, fixture_name))
    print(f"[fixture] pipeline stage hooks installed -> {fixture_dir}",
          flush=True)


def _dump_run_metadata(fixture_dir: str, *, image_path: str, args_dict: dict,
                       seed: int) -> None:
    """Write 00_metadata.json with everything needed to reproduce the run."""
    import json
    import hashlib
    import platform

    os.makedirs(fixture_dir, exist_ok=True)

    try:
        with open(image_path, "rb") as f:
            image_sha256 = hashlib.sha256(f.read()).hexdigest()
        image_size = os.path.getsize(image_path)
    except OSError:
        image_sha256 = None
        image_size = None

    # Device detection — capture which backend this run will use so we know
    # which side of the CUDA-vs-MPS diff we're looking at.
    if torch.cuda.is_available():
        device_kind = "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_kind = "mps"
        device_name = "Apple Silicon"
    else:
        device_kind = "cpu"
        device_name = platform.processor() or "cpu"

    meta = {
        "image_path": os.path.abspath(image_path),
        "image_sha256": image_sha256,
        "image_size_bytes": image_size,
        "seed": seed,
        "args": args_dict,
        "torch_version": torch.__version__,
        "device_kind": device_kind,
        "device_name": device_name,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "env": {k: os.environ.get(k) for k in [
            "ATTN_BACKEND", "SPARSE_CONV_BACKEND", "SPARSE_ATTN_BACKEND",
            "PYTORCH_CUDA_ALLOC_CONF", "PIXAL3D_DUMP_FIXTURES",
        ] if os.environ.get(k) is not None},
    }
    out = os.path.join(fixture_dir, "00_metadata.json")
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[fixture] metadata -> {out}", flush=True)


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

# ============================================================================
# Model Loading
# ============================================================================

def build_image_cond_model(config: dict):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model


def load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name)
    moge_model = moge_model.to(device)
    moge_model.eval()
    return moge_model


def init_pipeline(model_path=MODEL_PATH, device="cuda", low_vram=False):
    print(f"[Pipeline] Loading from {model_path}...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)

    print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])

    if low_vram:
        # Low-VRAM mode: models stay on CPU, loaded to GPU on-demand per stage.
        # Peak VRAM = one flow model + one DinoV3, not all ~18 GB at once.
        print("[NAF] Pre-downloading NAF upsampler weights (CPU only)...")
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
        print("[Pipeline] Low-VRAM mode enabled.")
    else:
        # Standard mode: all models loaded to GPU at once (faster, needs more VRAM).
        pipeline.low_vram = False
        pipeline.cuda()
        pipeline.image_cond_model_ss.cuda()
        pipeline.image_cond_model_shape_512.cuda()
        pipeline.image_cond_model_shape_1024.cuda()
        pipeline.image_cond_model_tex_1024.cuda()
        print("[NAF] Pre-loading NAF upsampler model...")
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        print("[Pipeline] Standard mode (all models on GPU).")

    return pipeline

# ============================================================================
# Camera Estimation
# ============================================================================

def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    y_ndc = -(yt - image_resolution / 2.0)
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def get_camera_params_wild_moge(image_path, moge_model, device="cuda", mesh_scale=1.0, extend_pixel=0, image_resolution=512):
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        camera_angle_x, grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale, image_resolution
    )["distance_from_x"]
    return {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}

# ============================================================================
# Main Inference
# ============================================================================

def run_inference(
    image_path: str,
    output_path: str,
    seed: int = 42,
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_sampling_steps: int = 12,
    ss_rescale_t: float = 5.0,
    shape_slat_guidance_strength: float = 7.5,
    shape_slat_guidance_rescale: float = 0.5,
    shape_slat_sampling_steps: int = 12,
    shape_slat_rescale_t: float = 3.0,
    tex_slat_guidance_strength: float = 1.0,
    tex_slat_guidance_rescale: float = 0.0,
    tex_slat_sampling_steps: int = 12,
    tex_slat_rescale_t: float = 3.0,
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
    max_num_tokens: int = 49152,
    model_path: str = MODEL_PATH,
    manual_fov: float = -1.0,
    low_vram: bool = False,
    resolution: int = -1,
):
    # Load models
    pipeline = init_pipeline(model_path, low_vram=low_vram)

    # Fixture-capture instrumentation (CUDA-vs-MPS divergence analysis).
    # Enabled only when PIXAL3D_DUMP_FIXTURES is set in the environment.
    if PIXAL3D_DUMP_FIXTURES:
        _dump_run_metadata(
            PIXAL3D_DUMP_FIXTURES,
            image_path=image_path,
            args_dict={
                "output_path": output_path, "seed": seed,
                "ss_guidance_strength": ss_guidance_strength,
                "ss_guidance_rescale": ss_guidance_rescale,
                "ss_sampling_steps": ss_sampling_steps,
                "ss_rescale_t": ss_rescale_t,
                "shape_slat_guidance_strength": shape_slat_guidance_strength,
                "shape_slat_guidance_rescale": shape_slat_guidance_rescale,
                "shape_slat_sampling_steps": shape_slat_sampling_steps,
                "shape_slat_rescale_t": shape_slat_rescale_t,
                "tex_slat_guidance_strength": tex_slat_guidance_strength,
                "tex_slat_guidance_rescale": tex_slat_guidance_rescale,
                "tex_slat_sampling_steps": tex_slat_sampling_steps,
                "tex_slat_rescale_t": tex_slat_rescale_t,
                "mesh_scale": mesh_scale, "extend_pixel": extend_pixel,
                "image_resolution": image_resolution,
                "max_num_tokens": max_num_tokens, "model_path": model_path,
                "manual_fov": manual_fov, "low_vram": low_vram,
                "resolution": resolution,
            },
            seed=seed,
        )
        _install_pipeline_fixture_hooks(pipeline, PIXAL3D_DUMP_FIXTURES)

    # Preprocess image first — rembg loads to GPU for this call, then offloads.
    # MoGe is loaded afterwards so both never occupy VRAM at the same time.
    print(f"[Inference] Processing image: {image_path}")
    img = Image.open(image_path)
    image_preprocessed = pipeline.preprocess_image(img)
    _dump_fixture("00_preprocessed_image",
                  {"mode": image_preprocessed.mode,
                   "size": image_preprocessed.size,
                   "pixels": np.array(image_preprocessed)})

    # Save preprocessed image for MoGe
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), f"_tmp_preprocessed_{int(time.time()*1000)}.png")
    image_preprocessed.save(tmp_path)

    # Camera estimation
    if manual_fov > 0:
        # Use manually specified FOV (in radians)
        camera_angle_x = float(manual_fov)
        grid_point = torch.tensor([-1.0, 0.0, 0.0])
        distance = distance_from_fov(
            camera_angle_x, grid_point,
            torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
            mesh_scale, image_resolution
        )["distance_from_x"]
        camera_params = {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}
        print(f"[Inference] Using manual FOV: {math.degrees(manual_fov):.2f}° ({manual_fov:.4f} rad), distance={distance:.4f}")
    else:
        print("[MoGe-2] Loading model for camera estimation...")
        moge_model = load_moge_model(device="cuda")
        print("[Inference] Estimating camera parameters...")
        camera_params = get_camera_params_wild_moge(
            tmp_path, moge_model, device="cuda",
            mesh_scale=mesh_scale, extend_pixel=extend_pixel,
            image_resolution=image_resolution,
        )
        print(f"  camera_angle_x={camera_params['camera_angle_x']:.4f}, distance={camera_params['distance']:.4f}")
        # MoGe is only needed for camera estimation; free its VRAM for inference.
        moge_model.cpu()
        del moge_model
        torch.cuda.empty_cache()
    os.remove(tmp_path)
    _dump_fixture("01_camera_params", camera_params)

    # Run pipeline
    print("[Inference] Running 3D generation pipeline...")
    torch.manual_seed(seed)

    ss_sampler_override = {
        "steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
        "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t,
    }
    shape_sampler_override = {
        "steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
        "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t,
    }
    tex_sampler_override = {
        "steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t,
    }

    pipeline_type = f"{resolution if resolution > 0 else (1024 if low_vram else 1536)}_cascade"
    print(f"[Inference] Using pipeline_type={pipeline_type}")
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        image_preprocessed,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params=ss_sampler_override,
        shape_slat_sampler_params=shape_sampler_override,
        tex_slat_sampler_params=tex_sampler_override,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )

    mesh = mesh_list[0]
    _dump_fixture("07_run_output", {
        "mesh": mesh,
        "shape_slat": shape_slat,
        "tex_slat": tex_slat,
        "res": res,
    })

    # Extract GLB
    print("[Inference] Extracting GLB...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000, texture_size=4096,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
    )
    # Capture to_glb output BEFORE the rotation step so the fixture matches
    # what came out of the upstream postprocess pipeline.  Geometry-only:
    # textures are dumped separately if we ever need them.
    _dump_fixture("08_to_glb_geometry", glb)

    # Apply rotation
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)

    # Export
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    glb.export(output_path, extension_webp=True)
    print(f"[Done] GLB saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixal3D Inference: Image to GLB")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="./output.glb", help="Output GLB file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fov", type=float, default=-1.0,
                        help="Manual camera FOV in radians (e.g. 0.2). "
                             "If not set, FOV is auto-estimated via MoGe-2. "
                             "Try 0.2 rad if you notice distortion.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Model path or HuggingFace repo")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode: models stay on CPU and are loaded to GPU on-demand per stage. "
                             "Reduces peak VRAM from ~18GB to ~10-12GB at the cost of slower inference.")
    parser.add_argument("--resolution", type=int, default=-1,
                        help="Pipeline resolution (1024 or 1536). Default: 1024 if --low_vram, else 1536.")

    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        output_path=args.output,
        seed=args.seed,
        manual_fov=args.fov,
        model_path=args.model_path,
        low_vram=args.low_vram,
        resolution=args.resolution,
    )
