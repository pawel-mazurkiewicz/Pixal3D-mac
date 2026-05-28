#!/usr/bin/env python
"""Mac/MPS counterpart to scripts/cuda/run_tex_ladder.py.

Drives generate_mps.main() with its built-in fixture hooks enabled, but
monkeypatches generate_mps._dump_fixture so each stage emits a one-line
`STAGE_JSON: {...}` of distribution stats (no .pt dumps -> no disk fill, no
torch.save corruption) and the run exits right after 06_tex_slat_decoded
(skips the slow Mac bake).  Directly comparable to the CUDA ladder.

Run:  .venv-py310/bin/python scripts/mac_tex_ladder.py
"""
import os, sys, json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR = "/tmp/mac_ladder_run"
os.makedirs(RUNDIR, exist_ok=True)
os.environ["PIXAL3D_DUMP_FIXTURES"] = RUNDIR  # enables hook installation
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np
import generate_mps as gm


def _tstats(t):
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
    path = os.path.join("/tmp/mac_ladder_out_npz", f"{stage}.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{k: v for k, v in arrays.items() if v is not None})
    print(f"[ladder] saved {path}", flush=True)


def _patched_dump(name, payload, fixture_dir=None):
    if "natten" in name or "rng_state" in name or "_step" in name:
        return
    try:
        # decoded texture field FIRST (it is a SparseTensor too) -----------
        if name == "06_tex_slat_decoded":
            feats = getattr(payload, "feats", None)
            if feats is None:
                _emit(name, {"note": "no .feats", "type": type(payload).__name__})
            else:
                _emit(name, _basecolor_sat(feats))
            print("[ladder] reached 06 — exiting before bake", flush=True)
            os._exit(0)

        # image conditioning (dict of z_global / z_proj) -------------------
        if name.startswith("01") and isinstance(payload, dict):
            obj = {k: _tstats(v) for k, v in payload.items() if hasattr(v, "detach")}
            _emit(name, obj)
            if name == "01d_image_cond_tex_1024":
                _save_npz(name, **{k: v.detach().float().cpu().numpy()
                                   for k, v in payload.items() if hasattr(v, "detach")})
            return

        # decode_shape_slat output: (meshes, subs) -------------------------
        if name == "04_shape_slat_decoded":
            subs = payload[1] if isinstance(payload, (tuple, list)) and len(payload) >= 2 else None
            if isinstance(subs, (tuple, list)):
                info = []
                for i, s in enumerate(subs):
                    f = getattr(s, "feats", None); c = getattr(s, "coords", None)
                    if f is not None:
                        info.append({"i": i, **_tstats(f)})
                        _save_npz(f"04_subs{i}", feats=f.detach().float().cpu().numpy(),
                                  coords=(c.detach().cpu().numpy() if c is not None else None))
                _emit(name, {"n_subs": len(subs), "subs": info})
            else:
                _emit(name, {"note": f"no subs list (type={type(payload).__name__})"})
            return

        # SparseTensor latents (03a / 03b / 05) ----------------------------
        feats = getattr(payload, "feats", None)
        coords = getattr(payload, "coords", None)
        if feats is not None:
            obj = _tstats(feats); obj["n_voxels"] = int(feats.shape[0])
            _emit(name, obj)
            if name in ("03a_shape_slat", "03b_shape_slat_cascade", "05_tex_slat"):
                _save_npz(name, feats=feats.detach().float().cpu().numpy(),
                          coords=(coords.detach().cpu().numpy() if coords is not None else None))
            return

        # dense tensor (02 sparse structure) -------------------------------
        if hasattr(payload, "detach"):
            obj = _tstats(payload)
            try:
                obj["n_active_gt0"] = int((payload.detach().float() > 0).sum().item())
            except Exception:
                pass
            _emit(name, obj)
            return

        _emit(name, {"note": f"unhandled payload type={type(payload).__name__}"})
    except Exception as exc:
        _emit(name, {"error": repr(exc)})


gm._dump_fixture = _patched_dump

sys.argv = [
    "generate_mps.py",
    "assets/images/9_img.png",
    "--output", "/tmp/mac_ladder_out",
    "--device", "mps",
    "--pipeline-type", "1024_cascade",
    "--seed", "42",
]
gm.main()
print("WARN: pipeline finished without reaching 06_tex_slat_decoded", flush=True)
