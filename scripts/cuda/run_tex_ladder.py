#!/usr/bin/env python
"""CUDA reference: full stage-ladder stats + latent dumps for the S5 colour hunt.

Single inference run that, via a monkeypatched inference._dump_fixture, for each
stage of interest:
  - prints a one-line `STAGE_JSON: {...}` with distribution stats (no torch.save,
    which corrupts SparseTensor fixtures on this box's torch), and
  - for the latents needed offline on Mac (01d conditioning, 03b shape_slat,
    04 subs, 05 tex_slat) saves raw feats/coords to out/ladder/<stage>.npz so they
    can be pulled and cross-injected without re-renting.

Exits cleanly right after 06_tex_slat_decoded (skips the OOM-prone GLB extract).

Run on the box:
  /venv/pixal3d/bin/python /workspace/cuda_bisect/run_tex_ladder.py
Stdout: one STAGE_JSON line per captured stage; basecolor sat stats for 06.
"""
import os, sys, json, importlib.util

REPO = "/workspace/Pixal3D-mac"
RUNDIR = "/workspace/cuda_bisect/ladder_run"
OUTDIR = "/workspace/cuda_bisect/out/ladder"
os.makedirs(RUNDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)
os.environ["PIXAL3D_DUMP_FIXTURES"] = RUNDIR  # enables hook installation
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np

spec = importlib.util.spec_from_file_location("inference", os.path.join(REPO, "inference.py"))
inf = importlib.util.module_from_spec(spec)
sys.modules["inference"] = inf
spec.loader.exec_module(inf)


def _tstats(t):
    """Distribution stats for a dense tensor (any shape)."""
    a = t.detach().float().cpu().numpy().reshape(-1)
    return {
        "shape": list(t.shape),
        "mean": round(float(a.mean()), 5),
        "std": round(float(a.std()), 5),
        "min": round(float(a.min()), 5),
        "max": round(float(a.max()), 5),
        "abs_mean": round(float(np.abs(a).mean()), 5),
        "l2": round(float(np.linalg.norm(a)), 3),
    }


def _feats_coords(payload):
    feats = getattr(payload, "feats", None)
    coords = getattr(payload, "coords", None)
    return feats, coords


def _basecolor_sat(feats):
    a = feats.detach().float().cpu().numpy()
    bc = np.clip(a[:, 0:3], 0, 1)
    mx = bc.max(1); mn = bc.min(1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return {
        "n_voxels": int(a.shape[0]), "n_ch": int(a.shape[1]),
        "range": [round(float(a.min()), 4), round(float(a.max()), 4)],
        "base_color_mean_rgb": [round(float(x), 4) for x in bc.mean(0)],
        "mean_sat": round(float(sat.mean()), 4),
        "frac_sat_gt_0.2": round(float((sat > 0.2).mean()), 4),
        "frac_red_dominant": round(float(((bc[:, 0] - np.maximum(bc[:, 1], bc[:, 2])) > 0.1).mean()), 4),
        "metallic_mean": round(float(np.clip(a[:, 3], 0, 1).mean()), 4) if a.shape[1] > 3 else None,
        "roughness_mean": round(float(np.clip(a[:, 4], 0, 1).mean()), 4) if a.shape[1] > 4 else None,
    }


def _emit(stage, obj):
    print(f"STAGE_JSON: {json.dumps({'stage': stage, **obj})}", flush=True)


def _save_npz(stage, **arrays):
    path = os.path.join(OUTDIR, f"{stage}.npz")
    np.savez(path, **{k: v for k, v in arrays.items() if v is not None})
    print(f"[ladder] saved {path}", flush=True)


def _patched_dump(name, payload, fixture_dir=None):
    if "natten" in name or "rng_state" in name or "_step" in name:
        return  # skip the giant natten / per-step traces

    try:
        # --- image conditioning (dict of z_global / z_proj) ---------------
        if name.startswith("01") and isinstance(payload, dict):
            obj = {}
            saved = {}
            for k, v in payload.items():
                if hasattr(v, "detach"):
                    obj[k] = _tstats(v)
                    saved[k] = v.detach().float().cpu().numpy()
            _emit(name, obj)
            if name == "01d_image_cond_tex_1024":
                _save_npz(name, **saved)
            return

        # --- decode_shape_slat output: (meshes, subs) ---------------------
        if name == "04_shape_slat_decoded":
            subs = None
            if isinstance(payload, (tuple, list)) and len(payload) >= 2:
                subs = payload[1]
            if subs is not None and isinstance(subs, (tuple, list)):
                info = []
                for i, s in enumerate(subs):
                    f, c = _feats_coords(s)
                    if f is not None:
                        info.append({"i": i, **_tstats(f)})
                        _save_npz(f"04_subs{i}",
                                  feats=f.detach().float().cpu().numpy(),
                                  coords=(c.detach().cpu().numpy() if c is not None else None))
                _emit(name, {"n_subs": len(subs), "subs": info})
            else:
                _emit(name, {"note": f"no subs list (type={type(payload).__name__})"})
            return

        # --- SparseTensor latents (02 / 03a / 03b / 05) -------------------
        feats, coords = _feats_coords(payload)
        if feats is not None:
            obj = _tstats(feats)
            obj["n_voxels"] = int(feats.shape[0])
            _emit(name, obj)
            if name in ("03b_shape_slat_cascade", "05_tex_slat"):
                _save_npz(name,
                          feats=feats.detach().float().cpu().numpy(),
                          coords=(coords.detach().cpu().numpy() if coords is not None else None))
            return

        # --- decoded texture field (06) -----------------------------------
        if name == "06_tex_slat_decoded":
            feats, coords = _feats_coords(payload)
            if feats is None:
                _emit(name, {"note": "no .feats", "type": type(payload).__name__})
            else:
                _emit(name, _basecolor_sat(feats))
            print("[ladder] reached 06 — exiting before GLB extract", flush=True)
            os._exit(0)

        _emit(name, {"note": f"unhandled payload type={type(payload).__name__}"})
    except Exception as exc:
        _emit(name, {"error": repr(exc)})


inf._dump_fixture = _patched_dump  # direct calls + hooks resolve the global at call time

inf.run_inference(
    image_path="assets/images/9_img.png",
    output_path="/workspace/cuda_bisect/out/ladder.glb",
    seed=42,
    manual_fov=0.0,  # <=0 => auto-estimate via MoGe-2 (matches default)
    model_path=inf.MODEL_PATH,
    low_vram=False,
    resolution=1024,
)
print("WARN: pipeline finished without reaching 06_tex_slat_decoded", flush=True)
