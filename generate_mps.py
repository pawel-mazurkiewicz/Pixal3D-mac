"""
CLI-only Pixal3D image-to-GLB generation for Apple Silicon MPS.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_NATIVE_O_VOXEL_PYTHON = (
    ROOT.parent / "trellis-mac" / ".venv" / "bin" / "python"
)

# ============================================================================
# Fixture capture — set PIXAL3D_DUMP_FIXTURES=/path to enable.
# Dumps intermediate tensors at every pipeline stage boundary so the MPS run
# can be diffed stage-by-stage against a CUDA reference capture.
# ============================================================================
PIXAL3D_DUMP_FIXTURES = os.environ.get("PIXAL3D_DUMP_FIXTURES", "").strip()


def _fixture_to_cpu(x):
    """Recursively detach + move tensors to CPU for portable fixture saves."""
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _fixture_to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_fixture_to_cpu(v) for v in x)
    if hasattr(x, "vertices") and hasattr(x, "faces"):
        out = {}
        for attr in ("vertices", "faces", "attrs", "coords", "uv",
                     "vertex_normals", "face_normals", "visual",
                     # Pre-extraction FDG tensors (set by
                     # FlexiDualGridVaeDecoder.forward).  Captured at
                     # stage-07 so a Mac dump can be diff'd against a CUDA
                     # dump on identical FDG inputs.
                     "fdg_coords", "fdg_dual_vertices",
                     "fdg_intersected", "fdg_split_weight",
                     # Raw mid-decoder features (pre-sigmoid / pre-`>0` /
                     # pre-softplus) — for isolating FDG-VAE-decoder
                     # divergence from threshold-flip divergence.
                     "fdg_h_feats"):
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
    """Save a fixture under <fixture_dir>/<name>.pt.  No-op if disabled.

    If the env var PIXAL3D_STOP_AFTER is set and matches this fixture name,
    exits the process cleanly after the save — used to skip late stages
    (e.g. shape/texture VAE decode) when we only care about the early
    pipeline."""
    target = fixture_dir if fixture_dir is not None else PIXAL3D_DUMP_FIXTURES
    if not target or torch is None:
        return
    os.makedirs(target, exist_ok=True)
    out = os.path.join(target, f"{name}.pt")
    try:
        torch.save(_fixture_to_cpu(payload), out)
        sz = os.path.getsize(out)
        print(f"[fixture] {name} -> {out} ({sz / 1e6:.2f} MB)", flush=True)
    except Exception as exc:
        print(f"[fixture] WARN failed to dump {name}: {exc}", flush=True)
    stop_after = os.environ.get("PIXAL3D_STOP_AFTER", "").strip()
    if stop_after and name == stop_after:
        print(f"[fixture] PIXAL3D_STOP_AFTER={stop_after} matched; exiting.",
              flush=True)
        sys.exit(0)


def _install_pipeline_fixture_hooks(pipeline, fixture_dir: str) -> None:
    """Monkey-patch pipeline stage methods to dump outputs after each call."""
    import functools

    stage_map = {
        "sample_sparse_structure":   "02_sparse_structure",
        "sample_shape_slat":         "03a_shape_slat",
        "sample_shape_slat_cascade": "03b_shape_slat_cascade",
        "decode_shape_slat":         "04_shape_slat_decoded",
        "sample_tex_slat":           "05_tex_slat",
        "decode_tex_slat":           "06_tex_slat_decoded",
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
    print(f"[fixture] pipeline stage hooks installed -> {fixture_dir}", flush=True)

    # --- DINOv3 image-conditioning model dumps ----------------------------
    # The pipeline has four separate DinoV3ProjFeatureExtractor instances —
    # one per DiT stage (sparse-structure / shape-LR / shape-HR / texture).
    # Each is called with the preprocessed image and returns a
    # (z_global, z_proj) tuple that conditions the corresponding DiT.
    # Capture each instance's output via forward hook so we can A/B the
    # DiT conditioning between MPS and CUDA at the source.
    image_cond_models = (
        ("01a_image_cond_ss",          "image_cond_model_ss"),
        ("01b_image_cond_shape_512",   "image_cond_model_shape_512"),
        ("01c_image_cond_shape_1024",  "image_cond_model_shape_1024"),
        ("01d_image_cond_tex_1024",    "image_cond_model_tex_1024"),
    )
    image_cond_counter: dict[str, int] = {name: 0 for name, _ in image_cond_models}

    def make_image_cond_hook(name):
        def _hook(module, module_args, output):
            idx = image_cond_counter[name]
            image_cond_counter[name] += 1
            tag = name if idx == 0 else f"{name}_call{idx}"
            payload = output
            if isinstance(output, tuple):
                payload = {"z_global": output[0], "z_proj": output[1]} \
                    if len(output) == 2 else {f"out_{i}": v for i, v in enumerate(output)}
            _dump_fixture(tag, payload, fixture_dir=fixture_dir)
        return _hook

    image_cond_installed = 0
    for fname, attr in image_cond_models:
        model = getattr(pipeline, attr, None)
        if model is None or not callable(model):
            continue
        try:
            model.register_forward_hook(make_image_cond_hook(fname))
            image_cond_installed += 1
        except Exception as exc:
            print(f"[fixture] WARN: cannot hook {attr}: {exc}", flush=True)
    print(f"[fixture] image_cond_model hooks installed ({image_cond_installed}/4)",
          flush=True)

    # --- per-step sampler dumps -------------------------------------------
    # Monkey-patch FlowEulerSampler.sample (the base method that the CFG and
    # GuidanceInterval subclasses delegate to via super().sample(...)) so we
    # capture every intermediate pred_x_t / pred_x_0 of the 12 denoising
    # steps.  Fixture names are derived from tqdm_desc so dumps from
    # different samplers (sparse-structure, shape SLat, HR shape SLat,
    # texture SLat) land in distinct files.
    try:
        from pixal3d.pipelines.samplers.flow_euler import FlowEulerSampler
    except Exception as exc:
        print(f"[fixture] WARN: cannot import FlowEulerSampler ({exc}); "
              "per-step dumps disabled.", flush=True)
        return

    if getattr(FlowEulerSampler.sample, "_pixal3d_patched", False):
        return  # already patched (e.g. hooks installed twice)

    import re
    _desc_to_tag = [
        ("hr shape slat",     "03b_shape_slat_cascade"),
        ("sparse structure",  "02_sparse_structure"),
        ("shape slat",        "03a_shape_slat"),
        ("texture slat",      "05_tex_slat"),
    ]
    def _tag_from_desc(desc: str) -> str:
        d = (desc or "").lower()
        for needle, tag in _desc_to_tag:
            if needle in d:
                return tag
        return "xx_" + re.sub(r"[^a-z0-9]+", "_", d).strip("_") or "xx_unknown"

    sampler_call_counter: dict[str, int] = {}
    original_sample = FlowEulerSampler.sample

    @functools.wraps(original_sample)
    def patched_sample(self, *args, **kwargs):
        # tqdm_desc may arrive as a kwarg or as the 6th positional arg
        # (model, noise, cond, steps, rescale_t, verbose, tqdm_desc, ...).
        desc = kwargs.get("tqdm_desc", "Sampling")
        # Snapshot RNG state BEFORE original_sample runs.  Per-step pred_x_t
        # exposes RNG drift indirectly; this is the explicit version.  On
        # Mac we capture MPS RNG too (different attribute than CUDA's).
        rng_snapshot = {"cpu": torch.get_rng_state()}
        try:
            if torch.cuda.is_available():
                rng_snapshot["cuda_all"] = torch.cuda.get_rng_state_all()
        except Exception as exc:
            rng_snapshot["cuda_all_err"] = repr(exc)
        try:
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                rng_snapshot["mps"] = torch.mps.get_rng_state()
        except Exception as exc:
            rng_snapshot["mps_err"] = repr(exc)
        result = original_sample(self, *args, **kwargs)
        tag = _tag_from_desc(desc)
        idx = sampler_call_counter.get(tag, 0)
        sampler_call_counter[tag] = idx + 1
        call_suffix = "" if idx == 0 else f"_call{idx}"
        _dump_fixture(f"{tag}{call_suffix}_rng_state", rng_snapshot,
                      fixture_dir=fixture_dir)
        try:
            pred_x_t = list(result.pred_x_t) if hasattr(result, "pred_x_t") else []
            pred_x_0 = list(result.pred_x_0) if hasattr(result, "pred_x_0") else []
        except Exception:
            pred_x_t, pred_x_0 = [], []
        n = min(len(pred_x_t), len(pred_x_0))
        for i in range(n):
            step_name = f"{tag}{call_suffix}_step{i:02d}"
            _dump_fixture(step_name,
                          {"pred_x_t": pred_x_t[i], "pred_x_0": pred_x_0[i]},
                          fixture_dir=fixture_dir)
        return result

    patched_sample._pixal3d_patched = True  # idempotency marker
    FlowEulerSampler.sample = patched_sample
    print("[fixture] FlowEulerSampler.sample patched for per-step dumps.",
          flush=True)


def _dump_run_metadata(fixture_dir: str, *, image_path: str, args_dict: dict,
                       seed: int) -> None:
    """Write 00_metadata.json."""
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

    device_kind = "mps" if (torch is not None and getattr(torch.backends, "mps", None)
                            and torch.backends.mps.is_available()) else "cpu"
    meta = {
        "image_path": os.path.abspath(image_path),
        "image_sha256": image_sha256,
        "image_size_bytes": image_size,
        "seed": seed,
        "args": args_dict,
        "torch_version": torch.__version__ if torch is not None else "?",
        "device_kind": device_kind,
        "device_name": "Apple Silicon",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "env": {k: os.environ.get(k) for k in [
            "ATTN_BACKEND", "SPARSE_CONV_BACKEND", "SPARSE_ATTN_BACKEND",
            "PIXAL3D_DUMP_FIXTURES",
        ] if os.environ.get(k) is not None},
    }
    out = os.path.join(fixture_dir, "00_metadata.json")
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[fixture] metadata -> {out}", flush=True)


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

# Pixal3D's mesh extractor outputs Y-up (matches glTF), but the reference
# CUDA pipeline (o_voxel.postprocess.to_glb) Y-flips the mesh before
# writing — confirmed by diffing AABBs of the official HF-demo output
# vs ours.  Our exports therefore need a single-axis Y reflection to land
# in the same orientation.  Because a reflection has det = -1 (face
# winding would flip and normals would point inward), `rotated_vertices`
# also reverses the face winding so the mesh stays correctly oriented.
EXPORT_ROTATION_ROWS = [
    # Negate glTF Z (depth axis).  Blender's importer converts glTF Y-up
    # to Z-up via a 90° rotation around X, which maps glTF +Z to Blender
    # -Y — so negating glTF Z corresponds to flipping the "facing" of the
    # model toward/away from the default Blender front-view camera.
    # That's what Pixal3D's reference pipeline does and what we need to
    # match the HF demo's orientation.
    [1, 0,  0, 0],
    [0, 1,  0, 0],
    [0, 0, -1, 0],
    [0, 0,  0, 1],
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

    # PIXAL3D_FP32_MODELS=<comma-list> selectively upcasts named submodels to
    # fp32.  Use case: testing whether a specific model's MPS bf16/fp16
    # numerics are the source of visible artifacts (e.g. tex_slat_decoder
    # for the muted/blurry texture issue identified in
    # DIVERGENCE_FINDINGS.md).
    #
    # For each model in the list, ALSO wraps the corresponding pipeline
    # call site so the SparseTensor input has its feats upcast to fp32
    # before it enters the model.  Without that wrap, the model gets fp16
    # input + fp32 weights → "expected mat1 and mat2 to have the same
    # dtype" crash inside sparse_conv3d_forward.
    fp32_models_env = os.environ.get("PIXAL3D_FP32_MODELS", "").strip()
    if fp32_models_env:
        names = [n.strip() for n in fp32_models_env.split(",") if n.strip()]
        print(f"[FP32] PIXAL3D_FP32_MODELS={names} — upcasting selected submodels.")
        for name in names:
            sub = pipeline.models.get(name) if pipeline.models else None
            if sub is None:
                print(f"  WARN: {name!r} not in pipeline.models; skipping.")
                continue
            try:
                before = next(sub.parameters()).dtype
            except StopIteration:
                print(f"  {name}: no parameters; skipping.")
                continue
            sub.float()
            # The Pixal3D VAEs (SparseStructureLatentVAE, SparseStructureVAE,
            # etc.) carry a self.dtype attribute and a `convert_to_fp32` method
            # that controls explicit `h.type(self.dtype)` casts inside forward().
            # `.float()` alone only updates parameters/buffers — those internal
            # casts still pull `self.dtype=fp16` and re-downcast activations,
            # producing "expected mat1 and mat2 to have the same dtype" at the
            # first conv.  Flip both.
            if hasattr(sub, "convert_to_fp32") and callable(sub.convert_to_fp32):
                sub.convert_to_fp32()
                print(f"  {name}: convert_to_fp32() called")
            if hasattr(sub, "dtype"):
                sub.dtype = torch.float32
                print(f"  {name}: .dtype attr -> torch.float32")
            if hasattr(sub, "use_fp16"):
                sub.use_fp16 = False
                print(f"  {name}: .use_fp16 attr -> False")
            after = next(sub.parameters()).dtype
            print(f"  {name}: param dtype {before} -> {after}")

        # Map model name -> pipeline call site whose input(s) need fp32
        # casting at the entry.  Each entry: list of (pipeline_method_name,
        # arg_name_or_index_to_cast).
        # Method -> list of arg positions to upcast.  decode_tex_slat
        # takes (slat, subs) where subs is a List[SparseTensor]; both
        # need casting.
        call_site_map = {
            "tex_slat_decoder": [("decode_tex_slat", 0), ("decode_tex_slat", 1)],
            "shape_slat_decoder": [("decode_shape_slat", 0)],
        }
        import functools as _ft

        def _cast_sparse_to_fp32(obj):
            """Recursively upcast SparseTensor.data['feats'] (or torch.Tensor) to fp32."""
            if torch.is_tensor(obj):
                return obj.float() if obj.is_floating_point() else obj
            if isinstance(obj, list):
                return [_cast_sparse_to_fp32(x) for x in obj]
            if isinstance(obj, tuple):
                return tuple(_cast_sparse_to_fp32(x) for x in obj)
            # Custom class with .data dict (SparseTensor)
            if hasattr(obj, "data") and isinstance(obj.data, dict) and "feats" in obj.data:
                feats = obj.data["feats"]
                if torch.is_tensor(feats) and feats.is_floating_point() and feats.dtype != torch.float32:
                    obj.data = dict(obj.data)
                    obj.data["feats"] = feats.float()
            return obj

        for name in names:
            for method_name, arg_pos in call_site_map.get(name, []):
                original = getattr(pipeline, method_name, None)
                if original is None or not callable(original):
                    continue

                def _make_wrapper(orig, pos):
                    @_ft.wraps(orig)
                    def wrapped(*args, **kwargs):
                        if pos < len(args):
                            args = list(args)
                            args[pos] = _cast_sparse_to_fp32(args[pos])
                            args = tuple(args)
                        return orig(*args, **kwargs)
                    return wrapped
                setattr(pipeline, method_name, _make_wrapper(original, arg_pos))
                print(f"  -> wrapped pipeline.{method_name}(arg[{arg_pos}]) to upcast input feats to fp32")

    # PIXAL3D_CPU_MODELS=<comma-list> forces named submodels to run on CPU
    # while the rest of the pipeline stays on MPS.  Buys numerical
    # determinism for the targeted submodel at the cost of MPS<->CPU
    # tensor shuffling.  Use case: testing whether a model's MPS
    # numerical drift (separate from bf16 precision) is the source of
    # output divergence vs the CUDA reference.  Common targets:
    #   tex_slat_decoder   — session-5 finding showed channels 3/4 (PBR
    #                        roughness/metallic) max-abs Δ 1.21 vs CUDA;
    #                        much larger than fp32 alone explains.
    #   shape_slat_decoder — secondary suspect for geometry drift.
    cpu_models_env = os.environ.get("PIXAL3D_CPU_MODELS", "").strip()
    if cpu_models_env:
        names = [n.strip() for n in cpu_models_env.split(",") if n.strip()]
        print(f"[CPU] PIXAL3D_CPU_MODELS={names} — moving selected submodels to CPU.")
        import functools as _ft2

        def _to_device(obj, device):
            """Recursively move tensors and SparseTensors to ``device``.

            For pixal3d SparseTensor, also drains the ``_spatial_cache``
            so spatial-conv neighbor tables get rebuilt on the new device
            (otherwise sparse_conv3d_forward gets feats on cpu but cached
            ``src_idx``/``tgt_idx``/``kernel_idx`` still on mps).
            """
            if torch.is_tensor(obj):
                return obj.to(device)
            if isinstance(obj, list):
                return [_to_device(x, device) for x in obj]
            if isinstance(obj, tuple):
                return tuple(_to_device(x, device) for x in obj)
            if isinstance(obj, dict):
                return {k: _to_device(v, device) for k, v in obj.items()}
            # SparseTensor: has a .data dict with tensor entries.
            moved_sparse = False
            if hasattr(obj, "data") and isinstance(obj.data, dict):
                new_data = {}
                for k, v in obj.data.items():
                    if torch.is_tensor(v):
                        new_data[k] = v.to(device)
                    else:
                        new_data[k] = v
                obj.data = new_data
                moved_sparse = True
            # Also walk direct coords/feats attributes (some SparseTensor
            # versions don't store via .data dict).
            for attr in ("coords", "feats"):
                if hasattr(obj, attr):
                    cur = getattr(obj, attr)
                    if torch.is_tensor(cur) and cur.device != torch.device(device):
                        try:
                            setattr(obj, attr, cur.to(device))
                            moved_sparse = True
                        except Exception:
                            pass
            # Drain spatial cache: device-specific neighbor tables become
            # stale once we move the SparseTensor.  Cleanest fix is to
            # let them get recomputed lazily on the new device.
            if hasattr(obj, "_spatial_cache") and isinstance(obj._spatial_cache, dict):
                if obj._spatial_cache:
                    obj._spatial_cache = {}
            if moved_sparse:
                return obj
            # Last resort: try .to() if available.
            if hasattr(obj, "to") and callable(obj.to):
                try:
                    return obj.to(device)
                except Exception:
                    pass
            return obj

        cpu_call_sites = {
            "tex_slat_decoder": [("decode_tex_slat", [0, 1])],
            "shape_slat_decoder": [("decode_shape_slat", [0])],
        }

        # Strategy: patch the submodel's forward() directly.  Pipeline-level
        # wrapping kept losing the race vs pipeline.to(device); forward-level
        # patching is the inescapable hook.
        import types as _types
        target_device = device

        def _force_cpu(m):
            """Force every parameter and buffer in ``m`` onto CPU, in place."""
            n_moved = 0
            for n, p in m.named_parameters():
                if p.device.type != "cpu":
                    p.data = p.data.cpu()
                    n_moved += 1
            for n, b in m.named_buffers():
                if b.device.type != "cpu":
                    new = b.cpu()
                    # set in place if possible
                    if hasattr(b, "data"):
                        b.data = new
                    else:
                        # named_buffers doesn't tell us the parent; reassign via the dict
                        pass
                    n_moved += 1
            return n_moved

        for name in names:
            sub = pipeline.models.get(name) if pipeline.models else None
            if sub is None:
                print(f"  WARN: {name!r} not in pipeline.models; skipping.")
                continue
            try:
                before_dev = next(sub.parameters()).device
            except StopIteration:
                print(f"  {name}: no parameters; skipping.")
                continue
            n_moved = _force_cpu(sub)
            after_dev = next(sub.parameters()).device
            print(f"  {name}: device {before_dev} -> {after_dev} ({n_moved} tensors moved)")

            # Sanity probe: pick a deep parameter and confirm it's on CPU.
            deep = list(sub.named_parameters())
            if deep:
                last_name, last_p = deep[-1]
                print(f"    deep-param check: {last_name} -> {last_p.device}")
            if hasattr(sub, "from_latent") and hasattr(sub.from_latent, "weight"):
                print(f"    from_latent.weight.device: {sub.from_latent.weight.device}")

            # Patch forward.  Capture the ORIGINAL bound forward.
            original_forward = sub.forward
            sub_name = name

            def _make_forward_patch(orig_forward, target_out_dev, this_sub, this_name):
                def patched_forward(*args, **kwargs):
                    # Belt-and-suspenders: re-force CPU before each call in
                    # case pipeline.to(device) ran between calls.
                    moved = _force_cpu(this_sub)
                    if moved:
                        print(f"  [CPU forward] {this_name}: re-moved {moved} tensors to cpu before call",
                              flush=True)
                    # Move args/kwargs to CPU
                    new_args = tuple(_to_device(a, torch.device("cpu")) for a in args)
                    new_kwargs = {k: _to_device(v, torch.device("cpu")) for k, v in kwargs.items()}
                    out = orig_forward(*new_args, **new_kwargs)
                    return _to_device(out, target_out_dev)
                return patched_forward

            sub.forward = _make_forward_patch(original_forward, target_device, sub, sub_name)
            print(f"  -> patched {name}.forward (CPU compute, output to {target_device})")

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


def _apply_export_rotation_to_mesh(vertices: np.ndarray, faces: np.ndarray, mesh):
    """Apply EXPORT_ROTATION to the mesh AND to the voxel attribute coords
    in one shot.  When the rotation includes a reflection (det < 0), the
    face winding is reversed so face normals continue to point outward.

    This must be called *before* UV unwrap and texture bake so that the
    cube projection, KDTree query, and final GLB export all operate in
    the same coordinate frame.  Otherwise the bake would fill texels using
    pre-rotation 3D positions while the exported vertices sit at post-
    rotation positions — texel content would land on the wrong faces
    (e.g. roof voxel attrs showing on the floor of the rendered mesh).
    """
    R = EXPORT_ROTATION[:3, :3]
    det = np.linalg.det(R)

    new_vertices = (vertices @ R.T).astype(np.float32)
    if det < 0:
        # Reflection: reverse triangle winding so outward normals stay outward.
        new_faces = faces[:, [0, 2, 1]].astype(faces.dtype)
    else:
        new_faces = faces

    # Rotate voxel data so the KDTree query in bake_texture finds the right
    # voxel for each (post-rotation) face position.  voxel_world is computed
    # as coords * vs + origin + vs/2.  After rotation:
    #   new_world = R @ old_world
    #             = R @ (coords*vs + origin + vs/2 * 1)
    #             = (coords @ R.T) * vs + (origin @ R.T) + vs/2 * (1 @ R.T)
    # which we can reconstruct by setting:
    #   new_coords = coords @ R.T
    #   new_origin = origin @ R.T + vs/2 * (R @ 1 - 1)
    coords_np = mesh.coords.detach().cpu().numpy().astype(np.float32)
    origin_np = mesh.origin.detach().cpu().numpy().astype(np.float32)
    vs = float(mesh.voxel_size)

    rotated_coords = coords_np @ R.T
    ones = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    origin_correction = 0.5 * vs * (ones @ R.T - ones)
    rotated_origin = origin_np @ R.T + origin_correction

    rotated_mesh = _LoadedMesh()
    rotated_mesh.vertices = torch.from_numpy(new_vertices)
    rotated_mesh.faces = torch.from_numpy(new_faces)
    rotated_mesh.coords = torch.from_numpy(rotated_coords)
    rotated_mesh.attrs = mesh.attrs
    rotated_mesh.origin = torch.from_numpy(rotated_origin)
    rotated_mesh.voxel_size = vs

    return new_vertices, new_faces, rotated_mesh


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

    # Match the texture path's coordinate convention: rotate via
    # EXPORT_ROTATION and, for reflections (det < 0), reverse face winding
    # so outward normals remain outward.
    R = EXPORT_ROTATION[:3, :3]
    out_vertices = rotated_vertices(vertices)
    if np.linalg.det(R) < 0:
        out_faces = faces[:, [0, 2, 1]]
    else:
        out_faces = faces
    mesh = trimesh.Trimesh(vertices=out_vertices, faces=out_faces, process=False)
    mesh.export(str(glb_path))
    print(f"[Export] Geometry-only GLB saved to {glb_path}")


def _write_mesh_npz(path: Path, mesh, vertices: np.ndarray, faces: np.ndarray, resolution: int):
    payload = dict(
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        coords=mesh.coords.detach().cpu().numpy(),
        attrs=mesh.attrs.detach().cpu().numpy(),
        origin=mesh.origin.detach().cpu().numpy().astype(np.float32),
        voxel_size=np.float32(float(mesh.voxel_size)),
        resolution=np.int32(int(resolution)),
    )
    for attr in ("fdg_coords", "fdg_dual_vertices", "fdg_intersected", "fdg_split_weight"):
        value = getattr(mesh, attr, None)
        if value is not None:
            payload[attr] = value.detach().cpu().numpy()
    np.savez(str(path), **payload)


def _native_o_voxel_python_candidates(args):
    seen = set()
    for candidate in (
        args.o_voxel_python,
        os.environ.get("O_VOXEL_PYTHON"),
        DEFAULT_NATIVE_O_VOXEL_PYTHON,
        shutil.which("python3.11"),
    ):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        yield path


def _resolve_native_o_voxel_python(args) -> Path | None:
    for path in _native_o_voxel_python_candidates(args):
        if path.exists():
            return path
    return None


def _native_o_voxel_env(glb_path: Path) -> dict:
    env = os.environ.copy()
    env.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(ROOT / "autotune_cache.json"))
    env.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "0")
    env.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    # Keep the native exporter's transient caches in a writable project path.
    env.setdefault("TMPDIR", str(glb_path.parent.resolve()))
    return env


def _configure_fdg_environment(args):
    os.environ["PIXAL3D_FDG_CAP_PARTIAL_QUADS"] = "1" if args.fdg_cap_partial_quads else "0"
    os.environ["PIXAL3D_FDG_VERBOSE"] = "1" if args.fdg_verbose else "0"


def _prefill_holes_for_native(vertices: np.ndarray, faces: np.ndarray, args) -> tuple[np.ndarray, np.ndarray]:
    perimeter = float(args.native_prefill_holes_perimeter)
    if perimeter <= 0:
        return vertices, faces
    try:
        from pixal3d.utils.cumesh_port.fill_holes import fill_holes
    except Exception as exc:
        print(f"[Export] Native pre-fill requested but cumesh_port.fill_holes is unavailable: {exc}")
        return vertices, faces

    print(f"[Export] Pre-filling closed boundary loops before native o_voxel (perimeter < {perimeter:g})...")
    return fill_holes(
        vertices.astype(np.float32, copy=False),
        faces.astype(np.int32, copy=False),
        max_hole_perimeter=perimeter,
        verbose=args.native_o_voxel_verbose,
    )


def try_export_native_o_voxel_glb(mesh, resolution: int, vertices: np.ndarray, faces: np.ndarray, glb_path: Path, args) -> bool:
    if args.no_texture or args.force_texture_fallback:
        return False

    native_python = _resolve_native_o_voxel_python(args)
    if native_python is None:
        print("[Export] Native o_voxel Python not found; using fallback baker.")
        return False

    bridge = ROOT / "pixal3d" / "utils" / "o_voxel_native_export.py"
    if not bridge.exists():
        print(f"[Export] Native o_voxel bridge missing: {bridge}")
        return False

    native_vertices, native_faces = _prefill_holes_for_native(vertices, faces, args)
    if args.skip_decimation:
        # o_voxel.to_glb has two simplify calls.  Passing the input face count
        # is not enough because fill_holes/cleanup can add faces before the
        # second call, which then still decimates.  Use a deliberately high
        # target so simplify() is a no-op while preserving UV/bake/export.
        target_faces = max(len(native_faces) * 4, int(args.native_decimation_target))
    else:
        target_faces = max(1, int(args.native_decimation_target))

    tmp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".npz",
        prefix="o_voxel_native_",
        dir=str(glb_path.parent),
    )
    native_input = Path(tmp_file.name)
    tmp_file.close()

    try:
        _write_mesh_npz(native_input, mesh, native_vertices, native_faces, resolution)
        print(
            f"[Export] Baking PBR textures via native o_voxel "
            f"({args.texture_size}x{args.texture_size}, target={target_faces:,})..."
        )
        cmd = [
            str(native_python),
            str(bridge),
            "--input", str(native_input),
            "--output", str(glb_path),
            "--texture-size", str(args.texture_size),
            "--decimation-target", str(target_faces),
            "--remesh-band", str(args.native_remesh_band),
            "--remesh-project", str(args.native_remesh_project),
            "--mesh-cluster-threshold", str(args.native_mesh_cluster_threshold),
            "--mesh-cluster-refine-iterations", str(args.native_mesh_cluster_refine_iterations),
            "--mesh-cluster-global-iterations", str(args.native_mesh_cluster_global_iterations),
            "--mesh-cluster-smooth-strength", str(args.native_mesh_cluster_smooth_strength),
            "--output-transform", args.native_output_transform,
        ]
        if args.native_remesh:
            cmd.append("--remesh")
        if args.native_o_voxel_verbose:
            cmd.append("--verbose")
        if args.native_debug_stages_dir:
            debug_dir = Path(args.native_debug_stages_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--debug-stages-dir", str(debug_dir)])
            if args.native_debug_stages_only:
                cmd.append("--debug-stages-only")
            if args.native_debug_raw_stages:
                cmd.append("--debug-raw-stages")

        result = subprocess.run(cmd, env=_native_o_voxel_env(glb_path), check=False)
        if result.returncode != 0:
            print(f"[Export] Native o_voxel export failed with exit code {result.returncode}; using fallback baker.")
            return False
        if not glb_path.exists() or glb_path.stat().st_size == 0:
            print("[Export] Native o_voxel finished but did not write a non-empty GLB; using fallback baker.")
            return False
        print(f"[Export] Textured GLB saved to {glb_path}")
        return True
    except Exception as exc:
        print(f"[Export] Native o_voxel texture bake failed: {exc}")
        return False
    finally:
        if os.environ.get("PIXAL3D_KEEP_NATIVE_O_VOXEL_INPUT") != "1":
            try:
                native_input.unlink()
            except FileNotFoundError:
                pass


def export_fallback_texture_glb(mesh, vertices: np.ndarray, faces: np.ndarray, glb_path: Path, args):
    from pixal3d.utils.texture_baker import (
        _find_blender_exe, bake_texture, export_glb_with_texture, uv_unwrap,
    )

    # Apply EXPORT_ROTATION up front: rotate the mesh AND the voxel attribute
    # frame together so the UV unwrap, bake, and final GLB write all operate
    # in the same coordinate system.  See _apply_export_rotation_to_mesh's
    # docstring for the math behind the origin correction.
    vertices, faces, mesh = _apply_export_rotation_to_mesh(vertices, faces, mesh)

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
    # Vertices were already rotated above — write directly.  No second pass
    # through rotated_vertices().
    export_glb_with_texture(
        new_vertices,
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
            "Path to input image.  Required unless --load-mesh PATH or "
            "--load-fixture-07 PATH is given."
        ),
    )
    parser.add_argument(
        "--load-fixture-07", type=str, default=None,
        help=(
            "Load a stage-07 fixture (`07_run_output.pt`) and replay only the "
            "downstream post-pipeline (pamo_simplify + texture bake + GLB "
            "export).  Used to isolate whether artifacts come from upstream "
            "DiT/VAE numerics or from the Mac post-pipeline code."
        ),
    )
    parser.add_argument("--output", default="output_3d", help="Output GLB path or basename (default: output_3d)")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Pixal3D model path or Hugging Face repo")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps", help="Runtime device (default: mps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--fov", type=float, default=-1.0,
                        help="Manual camera FOV in radians.  >0 bypasses MoGe-2 entirely "
                             "(use e.g. 0.5).  Default -1.0 keeps MoGe.")
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
        "--fdg-cap-partial-quads",
        dest="fdg_cap_partial_quads",
        action="store_true",
        default=True,
        help=(
            "During MPS flexible-dual-grid mesh extraction, emit a triangle "
            "when an intersected grid edge has 3 of its 4 neighboring dual "
            "voxels present.  This closes holes caused by the strict fallback "
            "extractor dropping the whole quad.  Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-fdg-cap-partial-quads",
        dest="fdg_cap_partial_quads",
        action="store_false",
        help="Disable the partial-quad cap and use the strict original extractor.",
    )
    parser.add_argument(
        "--fdg-verbose",
        action="store_true",
        help="Print flexible-dual-grid extraction counts while decoding the mesh.",
    )
    parser.add_argument(
        "--reextract-mesh-from-fdg",
        action="store_true",
        help=(
            "When loading a checkpoint saved with FDG tensors, re-run "
            "flexible_dual_grid_to_mesh using the current FDG extraction "
            "settings before export."
        ),
    )
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
        "--o-voxel-python",
        default=None,
        help=(
            "Python interpreter that can import the native Metal o_voxel stack. "
            "Defaults to $O_VOXEL_PYTHON, then ../trellis-mac/.venv/bin/python, "
            "then python3.11 if present."
        ),
    )
    parser.add_argument(
        "--native-decimation-target",
        type=int,
        default=1000000,
        help=(
            "Face target for native o_voxel.postprocess.to_glb.  Defaults to "
            "1,000,000 to match the Pixal3D/TRELLIS export pipeline.  The "
            "fallback baker still uses --bake-face-target."
        ),
    )
    parser.add_argument(
        "--native-remesh",
        action="store_true",
        help="Enable o_voxel's narrow-band dual-contouring remesh branch before simplification.",
    )
    parser.add_argument("--native-remesh-band", type=float, default=1.0)
    parser.add_argument("--native-remesh-project", type=float, default=0.0)
    parser.add_argument(
        "--native-prefill-holes-perimeter",
        type=float,
        default=3e-2,
        help=(
            "Run the local cumesh_port.fill_holes implementation on the raw "
            "mesh before native o_voxel.to_glb.  Defaults to cumesh's 3e-2 "
            "threshold; set 0 to rely only on native o_voxel's fill_holes."
        ),
    )
    parser.add_argument(
        "--native-output-transform",
        choices=["pixal3d", "o_voxel", "y_up"],
        default="pixal3d",
        help=(
            "Post-transform for native o_voxel output.  'pixal3d' fixes the "
            "sideways native output and matches the earlier MPS export axis "
            "convention; 'o_voxel' preserves raw native orientation."
        ),
    )
    parser.add_argument("--native-mesh-cluster-threshold", type=float, default=math.radians(90.0))
    parser.add_argument("--native-mesh-cluster-refine-iterations", type=int, default=0)
    parser.add_argument("--native-mesh-cluster-global-iterations", type=int, default=1)
    parser.add_argument("--native-mesh-cluster-smooth-strength", type=float, default=1.0)
    parser.add_argument(
        "--native-o-voxel-verbose",
        action="store_true",
        help="Print verbose progress from the native o_voxel subprocess.",
    )
    parser.add_argument(
        "--native-debug-stages-dir",
        type=str,
        default=None,
        help=(
            "Dump the cumesh.to_glb internal substages (00_input ... "
            "09_after_unify_orientations) as individual GLBs into this "
            "directory.  Forwarded to o_voxel_native_export.py --debug-stages-dir."
        ),
    )
    parser.add_argument(
        "--native-debug-stages-only",
        action="store_true",
        help=(
            "When used with --native-debug-stages-dir, skip the textured "
            "GLB export and write the final geometry stage as the output."
        ),
    )
    parser.add_argument(
        "--native-debug-raw-stages",
        action="store_true",
        help=(
            "When used with --native-debug-stages-dir, also dump the raw "
            "00_input + 01_after_fill_holes stages before any simplification."
        ),
    )
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

    __slots__ = (
        "vertices", "faces", "coords", "attrs", "origin", "voxel_size",
        "fdg_coords", "fdg_dual_vertices", "fdg_intersected", "fdg_split_weight",
    )


def save_mesh_checkpoint(path, mesh, vertices, faces, resolution):
    """Persist the generation result so we can iterate on UV/bake/export
    without re-running the ~8-10 min sampling pipeline."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_mesh_npz(out, mesh, vertices, faces, resolution)
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
    for attr in ("fdg_coords", "fdg_dual_vertices", "fdg_intersected", "fdg_split_weight"):
        if attr in data.files:
            setattr(mesh, attr, torch.from_numpy(data[attr]))
        else:
            setattr(mesh, attr, None)
    resolution = int(data["resolution"])
    return mesh, vertices, faces, resolution



def load_stage_07_fixture(path, resolution: int = 1024):
    """Load a ``07_run_output.pt`` fixture and reconstruct a mesh wrapper
    suitable for the post-pipeline export path.

    Used by ``--load-fixture-07`` to replay only the downstream (mesh
    extraction + simplification + texture bake + GLB write) part of the
    pipeline against a pre-recorded run.  Lets us isolate whether visible
    artifacts come from the upstream DiT/VAE numerics or from the
    post-pipeline code on Mac (FDG→mesh fallback, pamo_simplify,
    texture bake).

    The fixture's ``mesh`` is saved as a dict because ``_fixture_to_cpu``
    walks ``__dict__`` of unknown classes; the returned ``_LoadedMesh``
    re-wraps those tensors with the attributes the export path expects.

    ``origin`` and ``voxel_size`` are not in the dump (likely class
    properties on the original MeshWithVoxel) so we synthesise the
    canonical AABB-[-0.5,0.5] grid values used throughout the pipeline.
    """
    # Inject a stub for the CUDA-only class that the cuda-side fixture
    # references in its pickle.  Without this, torch.load raises
    # ModuleNotFoundError on flex_gemm even though the actual data we
    # want (mesh tensors) doesn't depend on the cache.
    import types as _types
    for _modpath in ("flex_gemm", "flex_gemm.ops", "flex_gemm.ops.spconv",
                     "flex_gemm.ops.spconv.submanifold_conv3d"):
        if _modpath not in sys.modules:
            sys.modules[_modpath] = _types.ModuleType(_modpath)
    _stub_mod = sys.modules["flex_gemm.ops.spconv.submanifold_conv3d"]
    if not hasattr(_stub_mod, "SubMConv3dNeighborCache"):
        class _PickleStub:
            def __setstate__(self, state):
                if isinstance(state, dict):
                    self.__dict__.update(state)
                else:
                    self.__dict__["_state"] = state
            def __reduce__(self):
                return (self.__class__, ())
        _stub_mod.SubMConv3dNeighborCache = type(
            "SubMConv3dNeighborCache", (_PickleStub,),
            {"__module__": "flex_gemm.ops.spconv.submanifold_conv3d"})

    payload = torch.load(str(path), weights_only=False, map_location="cpu")
    mesh_dict = payload["mesh"]
    if not isinstance(mesh_dict, dict):
        # Real MeshWithVoxel survived the round-trip; fall back to
        # treating it as such.
        mesh_dict = {k: v for k, v in mesh_dict.__dict__.items()}

    res = int(payload.get("res", resolution))

    mesh = _LoadedMesh()
    mesh.vertices = mesh_dict["vertices"].float()
    mesh.faces = mesh_dict["faces"].to(torch.int32)
    mesh.coords = mesh_dict["coords"].to(torch.int32)
    mesh.attrs = mesh_dict["attrs"].float()
    # Canonical AABB-[-0.5, 0.5] sparse grid (matches what TRELLIS.2 /
    # Pixal3D use throughout the pipeline).  Always reconstructable from
    # the resolution; not stored in the fixture.
    mesh.origin = torch.tensor([-0.5, -0.5, -0.5], dtype=torch.float32)
    mesh.voxel_size = 1.0 / float(res)
    # FDG re-extraction tensors are not in this fixture; leave None so
    # the reextract path is gracefully unavailable.
    for attr in ("fdg_coords", "fdg_dual_vertices",
                 "fdg_intersected", "fdg_split_weight"):
        setattr(mesh, attr, None)

    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy().astype(np.int32)
    return mesh, vertices, faces, res


def reextract_mesh_from_fdg(mesh, resolution: int):
    if any(getattr(mesh, attr, None) is None for attr in (
        "fdg_coords", "fdg_dual_vertices", "fdg_intersected", "fdg_split_weight",
    )):
        raise SystemExit(
            "--reextract-mesh-from-fdg requires a checkpoint saved after FDG "
            "checkpointing was added. Regenerate once with --save-mesh."
        )
    from pixal3d.utils.mesh_extract import flexible_dual_grid_to_mesh

    print("[FDG] Re-extracting mesh from checkpointed flexible-dual-grid tensors...")
    vertices_t, faces_t = flexible_dual_grid_to_mesh(
        mesh.fdg_coords,
        mesh.fdg_dual_vertices,
        mesh.fdg_intersected,
        mesh.fdg_split_weight,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=resolution,
        train=False,
    )
    mesh.vertices = vertices_t
    mesh.faces = faces_t.int()
    vertices = vertices_t.detach().cpu().numpy()
    faces = faces_t.detach().cpu().numpy().astype(np.int32)
    print(f"[FDG] Re-extracted {vertices.shape[0]:,} vertices, {faces.shape[0]:,} triangles")
    return mesh, vertices, faces


def main():
    args = parse_args()
    _configure_fdg_environment(args)
    glb_path = output_glb_path(args.output)
    glb_path.parent.mkdir(parents=True, exist_ok=True)

    load_runtime_deps()
    if PIXAL3D_DUMP_FIXTURES:
        _dump_run_metadata(PIXAL3D_DUMP_FIXTURES, image_path=args.image or "",
                           args_dict=vars(args), seed=args.seed)
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
    from_fixture = bool(args.load_fixture_07)

    if from_fixture:
        fx_path = Path(args.load_fixture_07)
        if not fx_path.exists():
            raise SystemExit(f"Stage-07 fixture not found: {fx_path}")
        print(f"[Fixture-07] Loading mesh from {fx_path}")
        mesh, vertices, faces, resolution = load_stage_07_fixture(
            fx_path, resolution=1024)
        print(f"[Mesh] {vertices.shape[0]:,} vertices, {faces.shape[0]:,} triangles "
              f"(resolution={resolution}) — skipping upstream pipeline.")
    elif from_checkpoint:
        ckpt_path = Path(args.load_mesh)
        if not ckpt_path.exists():
            raise SystemExit(f"Mesh checkpoint not found: {ckpt_path}")
        print(f"[Checkpoint] Loading mesh from {ckpt_path}")
        mesh, vertices, faces, resolution = load_mesh_checkpoint(ckpt_path)
        if args.reextract_mesh_from_fdg:
            mesh, vertices, faces = reextract_mesh_from_fdg(mesh, resolution)
        print(f"[Mesh] {vertices.shape[0]:,} vertices, {faces.shape[0]:,} triangles "
              f"(resolution={resolution})")
    else:
        if not args.image:
            raise SystemExit(
                "An input image is required unless --load-mesh PATH or "
                "--load-fixture-07 PATH is given"
            )
        input_path = Path(args.image)
        if not input_path.exists():
            raise SystemExit(f"Input image not found: {input_path}")

        pipeline = init_pipeline(args.model_path, device)

        print(f"[Input] {input_path}")
        image = Image.open(input_path)
        image_preprocessed = pipeline.preprocess_image(image)
        _dump_fixture("00_preprocessed_image", {
            "mode": image_preprocessed.mode,
            "size": image_preprocessed.size,
            "pixels": np.array(image_preprocessed),
        })

        if args.fov > 0:
            # Manual FOV: bypass MoGe entirely.  Matches the formula used by
            # Pixal3D_fresh/inference.py so CUDA-vs-MPS captures stay aligned.
            camera_angle_x = float(args.fov)
            grid_point = torch.tensor([-1.0, 0.0, 0.0])
            target_point = torch.tensor(
                [0 - args.extend_pixel,
                 args.image_resolution - 1 + args.extend_pixel]
            )
            distance = distance_from_fov(
                camera_angle_x, grid_point, target_point,
                args.mesh_scale, args.image_resolution,
            )["distance_from_x"]
            camera_params = {
                "camera_angle_x": camera_angle_x,
                "distance": distance,
                "mesh_scale": args.mesh_scale,
            }
            print(f"[Manual FOV] {math.degrees(args.fov):.2f}° "
                  f"({args.fov:.4f} rad), distance={distance:.4f}")
        else:
            print("[MoGe] Loading camera estimator...")
            moge_model = load_moge_model(device)
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

        _dump_fixture("01_camera_params", camera_params)

        torch.manual_seed(args.seed)
        print(f"[Generate] pipeline={args.pipeline_type}, seed={args.seed}")
        if PIXAL3D_DUMP_FIXTURES:
            _install_pipeline_fixture_hooks(pipeline, PIXAL3D_DUMP_FIXTURES)

        # PIXAL3D_SDPA_BACKEND=math|efficient|flash pins torch's SDPA path
        # for this run.  Lets us isolate whether MPS's fused attention paths
        # are the source of CUDA-vs-MPS divergence.  Unset = backend default.
        _sdpa_choice = os.environ.get("PIXAL3D_SDPA_BACKEND", "").strip().lower()
        _sdpa_ctx = nullcontext()
        if _sdpa_choice:
            try:
                import torch.nn.attention as _ta
                _sdpa_map = {
                    "math":      _ta.SDPBackend.MATH,
                    "efficient": _ta.SDPBackend.EFFICIENT_ATTENTION,
                    "flash":     _ta.SDPBackend.FLASH_ATTENTION,
                }
                if _sdpa_choice in _sdpa_map:
                    _sdpa_ctx = _ta.sdpa_kernel([_sdpa_map[_sdpa_choice]])
                    print(f"[SDPA] pinning attention to '{_sdpa_choice}'", flush=True)
                else:
                    print(f"[SDPA] WARN unknown PIXAL3D_SDPA_BACKEND='{_sdpa_choice}'; using default", flush=True)
            except Exception as _exc:
                print(f"[SDPA] WARN sdpa_kernel unavailable ({_exc}); using default", flush=True)

        t_gen = time.time()
        try:
          with _sdpa_ctx:
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
        _dump_fixture("07_run_output", {
            "mesh": mesh,
            "shape_slat": shape_slat,
            "tex_slat": tex_slat,
            "res": resolution,
        })
        vertices = mesh.vertices.detach().cpu().numpy()
        faces = mesh.faces.detach().cpu().numpy()
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            print(watchdog_help_message())
            sys.exit(2)

        print(f"[Mesh] {vertices.shape[0]:,} vertices, {faces.shape[0]:,} triangles")
        print(f"[Timing] Generation: {time.time() - t_gen:.1f}s")

        if args.save_mesh:
            save_mesh_checkpoint(args.save_mesh, mesh, vertices, faces, resolution)

    # PIXAL3D_PRECLEAN_NME=1 runs repair_non_manifold_edges on the FDG-
    # extracted mesh BEFORE to_glb's cleanup chain.  The FDG output has
    # ~904k pre-existing NMEs that compound when pamo's QEM collapses add
    # more NMEs on top.  Pre-cleaning produces a less-NME-heavy input for
    # the to_glb pipeline, which should reduce the small_cc culling cascade.
    # See PAMO_PORT_PLAN.md Phase 6 / NEXT_STEPS.md direction 2.
    if os.environ.get("PIXAL3D_PRECLEAN_NME", "").strip() in ("1", "true", "yes"):
        print("[Pre-clean] Running repair_non_manifold_edges on FDG mesh...")
        from pixal3d.utils.cumesh_port.repair import repair_non_manifold_edges as _cpu_repair_nme
        before_v = vertices.shape[0]
        before_f = faces.shape[0]
        _pre_t = time.time()
        new_v, new_f = _cpu_repair_nme(vertices, faces.astype(np.int64), verbose=True)
        new_f = new_f.astype(faces.dtype)
        print(f"[Pre-clean] V {before_v:,} -> {new_v.shape[0]:,} "
              f"(+{new_v.shape[0]-before_v:,}), F {before_f:,} -> {new_f.shape[0]:,}"
              f" in {time.time()-_pre_t:.1f}s")
        vertices = new_v
        faces = new_f
        if hasattr(mesh, "vertices") and isinstance(mesh.vertices, torch.Tensor):
            mesh.vertices = torch.from_numpy(new_v).to(mesh.vertices.device).to(mesh.vertices.dtype)
            mesh.faces = torch.from_numpy(new_f).to(mesh.faces.device).to(mesh.faces.dtype)

    has_voxels = hasattr(mesh, "attrs") and mesh.attrs is not None and hasattr(mesh, "coords") and mesh.coords is not None
    if args.no_texture or not has_voxels:
        export_geometry_only(vertices, faces, glb_path)
    elif not try_export_native_o_voxel_glb(mesh, resolution, vertices, faces, glb_path, args):
        export_fallback_texture_glb(mesh, vertices, faces, glb_path, args)

    if not glb_path.exists() or glb_path.stat().st_size == 0:
        raise SystemExit(f"Export failed: GLB was not written or is empty: {glb_path}")

    print(f"[Done] GLB saved to {glb_path} ({glb_path.stat().st_size:,} bytes)")
    print(f"[Timing] Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
