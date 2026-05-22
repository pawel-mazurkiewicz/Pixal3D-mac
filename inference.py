import os
import sys
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
                     "vertex_normals", "face_normals", "visual",
                     # Pre-extraction FDG tensors (set by
                     # FlexiDualGridVaeDecoder.forward). Capturing them at
                     # stage-07 lets a Mac box replay only the FDG -> mesh
                     # + cleanup steps on CUDA's exact inputs.
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
    pipeline.

    PIXAL3D_NATTEN_PROBE_ONLY=1 — minimal-bandwidth mode for natten parity
    work.  Filters everything except fixtures whose name contains "_natten_"
    so the rental run produces only the natten capture (hundreds of MB
    instead of ~7 GB).  Combine with PIXAL3D_STOP_AFTER=01b_natten_shape_512
    to exit immediately after the first natten capture.
    """
    target = fixture_dir if fixture_dir is not None else PIXAL3D_DUMP_FIXTURES
    if not target:
        return
    if os.environ.get("PIXAL3D_NATTEN_PROBE_ONLY", "").strip() == "1" \
            and "_natten_" not in name:
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
    """Monkey-patch pipeline stage methods to dump outputs after each call.

    Each method may be invoked more than once per inference (e.g. the cascade
    path calls shape_slat twice).  We append `_call{N}` so fixtures from
    multiple invocations don't overwrite each other.

    ``PIXAL3D_NATTEN_PROBE_ONLY=1`` — minimal-bandwidth mode for natten
    parity work.  Skips all stage dumps, image_cond dumps, and per-step
    sampler dumps.  Only the first natten capture (01b_natten_shape_512)
    fires; pipeline exits immediately after via implicit
    ``PIXAL3D_STOP_AFTER``.  Total download ~hundreds of MB instead of ~7 GB.
    """
    import functools

    PROBE_ONLY = os.environ.get("PIXAL3D_NATTEN_PROBE_ONLY", "").strip() == "1"
    if PROBE_ONLY:
        # Exit is driven by the natten capture wrappers themselves (they call
        # os._exit(0) after the first complete attention is dumped), so we
        # don't set STOP_AFTER here.  Earlier revisions auto-set
        # STOP_AFTER="01b_natten_shape_512", but that label is never produced
        # by the new sequential naming and never matched the legacy path —
        # the pipeline ran to completion despite probe mode.
        print("[fixture] PIXAL3D_NATTEN_PROBE_ONLY=1: skipping stage/"
              "image_cond/sampler dumps; natten wrappers will self-exit "
              "after the first complete attention.", flush=True)

    stage_map = {
        "sample_sparse_structure":      "02_sparse_structure",
        "sample_shape_slat":            "03a_shape_slat",
        "sample_shape_slat_cascade":    "03b_shape_slat_cascade",
        "decode_shape_slat":            "04_shape_slat_decoded",
        "sample_tex_slat":              "05_tex_slat",
        "decode_tex_slat":              "06_tex_slat_decoded",
    }

    call_counter: dict[str, int] = {n: 0 for n in stage_map.values()}

    if not PROBE_ONLY:
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

    # --- natten.na2d input/output capture ---------------------------------
    # NAF (valeoai/NAF) calls natten.na2d(q, k, v, kernel_size=(9,9),
    # dilation=..., stride=1, backend="cutlass-fna") for its CrossAttention
    # upsampler.  natten 0.21 has NO MPS backend, and cutlass-fna is the
    # CUDA gold reference we want to match in a future Metal port.
    # Capture (q, k, v, kernel_size, dilation, scale, out) on every call so
    # the Mac side can compute its candidate kernel on the SAME inputs and
    # diff against the CUDA reference output.
    # Three calls per pipeline run (one per NAF-bearing image_cond stage:
    # 01b/01c/01d — 01a has use_naf_upsample=False).
    # NAF picks ONE of two natten entry points at module-import time:
    #
    #   try:   from natten.functional import na2d_av, na2d_qk   # legacy split
    #          NATTEN_RECENT = False
    #   except: from natten import na2d                          # modern fused
    #          NATTEN_RECENT = True
    #
    # Which branch wins depends on the installed natten version (rental
    # box may have natten<0.21 with the split symbols; Mac has 0.21 which
    # dropped them).  Previous rental run showed na2d wrapper installed
    # but never fired — confirming the rental box hit the LEGACY path.
    #
    # We now patch all three possible entry points and rebind any
    # module-local copies via sys.modules.  Whichever branch NAF took,
    # at least one wrapper will fire.
    try:
        import natten as _natten
        import sys as _sys
        import atexit as _atexit

        _na2d_n     = {"n": 0}
        _na2d_qk_n  = {"n": 0}
        _na2d_av_n  = {"n": 0}

        _PROBE_EXIT_AFTER_FIRST = PROBE_ONLY  # captured from outer scope

        def _maybe_exit(label):
            if _PROBE_EXIT_AFTER_FIRST:
                print(f"[fixture] PROBE_ONLY: exiting cleanly after {label}.",
                      flush=True)
                # os._exit bypasses atexit handlers — that's fine here, the
                # fire-check is only useful when we *don't* exit early.
                os._exit(0)

        # --- modern fused na2d ---
        _orig_na2d = getattr(_natten, "na2d", None)
        _na2d_capture = None
        if _orig_na2d is not None:
            def _na2d_capture(query, key, value, kernel_size, *args, **kwargs):
                n = _na2d_n["n"]
                _na2d_n["n"] = n + 1
                tag = f"natten_call{n}"
                out = _orig_na2d(query, key, value, kernel_size, *args, **kwargs)
                payload = {
                    "func": "na2d",
                    "q": query.detach().cpu(),
                    "k": key.detach().cpu(),
                    "v": value.detach().cpu(),
                    "out": out.detach().cpu(),
                    "kernel_size": kernel_size,
                    "dilation": kwargs.get("dilation",
                                           args[1] if len(args) > 1 else 1),
                    "stride": kwargs.get("stride",
                                         args[0] if len(args) > 0 else 1),
                    "scale": kwargs.get("scale", None),
                    "is_causal": kwargs.get("is_causal", False),
                    "backend": kwargs.get("backend", None),
                    "shapes_str": (
                        f"q={tuple(query.shape)} k={tuple(key.shape)} "
                        f"v={tuple(value.shape)} out={tuple(out.shape)} "
                        f"ks={kernel_size} dilation={kwargs.get('dilation', 1)}"
                    ),
                }
                _dump_fixture(tag, payload, fixture_dir=fixture_dir)
                _maybe_exit(tag)
                return out
            _natten.na2d = _na2d_capture

        # --- legacy split path: na2d_qk + na2d_av ---
        _functional = getattr(_natten, "functional", None)
        _orig_qk = getattr(_functional, "na2d_qk", None) if _functional else None
        _orig_av = getattr(_functional, "na2d_av", None) if _functional else None

        _qk_capture = None
        if _orig_qk is not None:
            def _qk_capture(q, k, *args, **kwargs):
                n = _na2d_qk_n["n"]
                _na2d_qk_n["n"] = n + 1
                tag = f"natten_qk_call{n}"
                out = _orig_qk(q, k, *args, **kwargs)
                payload = {
                    "func": "na2d_qk",
                    "q": q.detach().cpu(),
                    "k": k.detach().cpu(),
                    "out": out.detach().cpu(),
                    "kernel_size": kwargs.get("kernel_size"),
                    "dilation": kwargs.get("dilation", 1),
                    "shapes_str": (
                        f"q={tuple(q.shape)} k={tuple(k.shape)} "
                        f"out={tuple(out.shape)} "
                        f"ks={kwargs.get('kernel_size')} "
                        f"dilation={kwargs.get('dilation', 1)}"
                    ),
                }
                _dump_fixture(tag, payload, fixture_dir=fixture_dir)
                # Do NOT exit here — wait for the matching AV call so we
                # capture a paired (qk, av) fixture before bailing out.
                return out
            _natten.functional.na2d_qk = _qk_capture

        _av_capture = None
        if _orig_av is not None:
            def _av_capture(attn, value, *args, **kwargs):
                n = _na2d_av_n["n"]
                _na2d_av_n["n"] = n + 1
                tag = f"natten_av_call{n}"
                out = _orig_av(attn, value, *args, **kwargs)
                payload = {
                    "func": "na2d_av",
                    "attn": attn.detach().cpu(),
                    "v": value.detach().cpu(),
                    "out": out.detach().cpu(),
                    "kernel_size": kwargs.get("kernel_size"),
                    "dilation": kwargs.get("dilation", 1),
                    "shapes_str": (
                        f"attn={tuple(attn.shape)} v={tuple(value.shape)} "
                        f"out={tuple(out.shape)} "
                        f"ks={kwargs.get('kernel_size')} "
                        f"dilation={kwargs.get('dilation', 1)}"
                    ),
                }
                _dump_fixture(tag, payload, fixture_dir=fixture_dir)
                # AV completes the legacy split attention — safe to exit
                # in probe mode now that the paired qk/av is on disk.
                _maybe_exit(tag)
                return out
            _natten.functional.na2d_av = _av_capture

        # Rebind module-local copies in any importer that has already
        # done `from natten import na2d` or `from natten.functional
        # import na2d_qk, na2d_av`.  Match by object identity against
        # the originals we just replaced.
        _orig_to_wrapper = {}
        if _orig_na2d is not None and _na2d_capture is not None:
            _orig_to_wrapper[(id(_orig_na2d), "na2d")] = _na2d_capture
        if _orig_qk is not None and _qk_capture is not None:
            _orig_to_wrapper[(id(_orig_qk), "na2d_qk")] = _qk_capture
        if _orig_av is not None and _av_capture is not None:
            _orig_to_wrapper[(id(_orig_av), "na2d_av")] = _av_capture

        _patched_local = []
        for _modname, _mod in list(_sys.modules.items()):
            if _mod is None or _mod is _natten or _mod is _functional:
                continue
            for _attr in ("na2d", "na2d_qk", "na2d_av"):
                try:
                    _local = getattr(_mod, _attr, None)
                except Exception:
                    continue
                if _local is None:
                    continue
                wrapper = _orig_to_wrapper.get((id(_local), _attr))
                if wrapper is not None:
                    try:
                        setattr(_mod, _attr, wrapper)
                        _patched_local.append(f"{_modname}.{_attr}")
                    except Exception as _exc:
                        print(f"[fixture] WARN: failed to rebind {_attr} "
                              f"in {_modname}: {_exc}", flush=True)

        print(f"[fixture] natten capture installed: "
              f"na2d={_orig_na2d is not None}, "
              f"na2d_qk={_orig_qk is not None}, "
              f"na2d_av={_orig_av is not None}; "
              f"local rebinds: {_patched_local or '(none)'}", flush=True)

        def _natten_fire_check():
            total = _na2d_n["n"] + _na2d_qk_n["n"] + _na2d_av_n["n"]
            if total == 0:
                print("[fixture] !!! WARN: ALL natten capture wrappers "
                      "(na2d, na2d_qk, na2d_av) were installed but NEVER "
                      "FIRED.  NAF call path bypassed everything we know "
                      "about.  Inspect attentions.py / installed natten "
                      "version on this box.", flush=True)
            else:
                print(f"[fixture] natten wrappers fired — "
                      f"na2d={_na2d_n['n']}, "
                      f"na2d_qk={_na2d_qk_n['n']}, "
                      f"na2d_av={_na2d_av_n['n']}.", flush=True)
        _atexit.register(_natten_fire_check)
    except Exception as exc:
        print(f"[fixture] WARN: cannot install natten capture: {exc}",
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
        desc = kwargs.get("tqdm_desc", "Sampling")
        # Snapshot RNG state BEFORE original_sample runs so we can verify
        # that two boxes (CUDA, MPS) started each sampler call with the same
        # entropy.  Per-step pred_x_t already exposes RNG drift indirectly;
        # this just makes the smoking gun explicit.
        rng_snapshot = {
            "cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                rng_snapshot["cuda_all"] = torch.cuda.get_rng_state_all()
            except Exception as exc:
                rng_snapshot["cuda_all_err"] = repr(exc)
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

    patched_sample._pixal3d_patched = True
    FlowEulerSampler.sample = patched_sample
    print("[fixture] FlowEulerSampler.sample patched for per-step dumps.",
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
