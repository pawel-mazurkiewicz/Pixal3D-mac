#!/usr/bin/env python
"""Variance-recalibration causation probe for the shape-decoder collapse.

Hypothesis: Mac's shape-slat decoder produces subdiv logits with ~half CUDA's
variance (progressive through the cascade), which (a) drops voxels at the
binarization threshold -> smaller mesh, and (b) starves the tex decoder's
guide_subs -> grey colour.  This probe rescales each C2S block's subdiv logits
to MATCH CUDA's per-level (mean, std) using fixed robocrab-derived constants,
then runs the full Mac pipeline and measures 06 saturation + voxel count.

If geometry AND colour recover -> variance collapse is the proven cause and
per-level recalibration is a Mac-only fix lever.

Toggle: PIXAL3D_RECALIB=1 enables the rescale; unset = identity (baseline).
Run:  PIXAL3D_RECALIB=1 .venv-py310/bin/python scripts/mac_recalib_probe.py
"""
import os, sys, json

# MUST precede any torch-pulling import: generate_mps sets this inside
# _configure_mps_environment() (called from main()), but this probe imports
# torch + sparse_unet_vae at module load, which would initialize the MPS
# backend before the fallback is registered -> aten::segment_reduce (used by
# SparseTensor.std() in CFG guidance) crashes. Set it first.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR = "/tmp/mac_recalib_run"
os.makedirs(RUNDIR, exist_ok=True)
os.environ["PIXAL3D_DUMP_FIXTURES"] = RUNDIR
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np
import torch
import generate_mps as gm
import pixal3d.models.sc_vaes.sparse_unet_vae as M

RECALIB = os.environ.get("PIXAL3D_RECALIB", "").strip() == "1"

# Per-level (mac_mean, mac_std, cuda_mean, cuda_std) for the subdiv logits,
# measured on robocrab 9_img.png seed 42 1024_cascade (subs0..subs3).
_COEFFS = [
    (38.24624, 115.35922, 41.88572, 129.72485),
    (4.03776,  88.61266,  13.42821, 101.24020),
    (-5.33891, 29.49385,  -5.29934, 49.16254),
    (-1.05075, 20.17925,  -1.09990, 37.73084),
]
_LEVEL = [0]

_orig_c2s_forward = M.SparseResBlockC2S3d._forward


def _recalib_forward(self, x, subdiv=None):
    import torch.nn.functional as F
    if self.pred_subdiv:
        subdiv = self.to_subdiv(x)
        if RECALIB:
            i = _LEVEL[0]; _LEVEL[0] += 1
            if i < len(_COEFFS):
                mm, ms, cm, cs = _COEFFS[i]
                f = subdiv.feats
                subdiv = subdiv.replace((f - mm) / ms * cs + cm)
                print(f"[recalib] level {i}: subdiv rescaled "
                      f"(mac {mm:.2f}/{ms:.2f} -> cuda {cm:.2f}/{cs:.2f}), "
                      f"new std={float(subdiv.feats.std()):.2f}", flush=True)
    h = x.replace(self.norm1(x.feats))
    h = h.replace(F.silu(h.feats))
    h = self.conv1(h)
    subdiv_binarized = subdiv.replace(subdiv.feats > -M._SUBDIV_BIAS) if subdiv is not None else None
    h = self.updown(h, subdiv_binarized)
    x = self.updown(x, subdiv_binarized)
    h = h.replace(self.norm2(h.feats))
    h = h.replace(F.silu(h.feats))
    h = self.conv2(h)
    h = h + self.skip_connection(x)
    if self.pred_subdiv:
        return h, subdiv
    else:
        return h


M.SparseResBlockC2S3d._forward = _recalib_forward

# The shape decoder runs multiple passes (upsample() during the HR cascade,
# then the final forward()).  Reset the per-level counter at each pass entry
# so EVERY pass rescales levels 0..3 correctly (otherwise the global counter
# exhausts during upsample() and the final decode's guide_subs go un-rescaled).
_orig_dec_forward = M.SparseUnetVaeDecoder.forward
_orig_dec_upsample = getattr(M.SparseUnetVaeDecoder, "upsample", None)


def _reset_forward(self, *a, **k):
    _LEVEL[0] = 0
    return _orig_dec_forward(self, *a, **k)


M.SparseUnetVaeDecoder.forward = _reset_forward
if _orig_dec_upsample is not None:
    def _reset_upsample(self, *a, **k):
        _LEVEL[0] = 0
        return _orig_dec_upsample(self, *a, **k)
    M.SparseUnetVaeDecoder.upsample = _reset_upsample
print(f"[recalib] patched SparseResBlockC2S3d._forward + decoder pass-resets (RECALIB={RECALIB})", flush=True)


# --- stats dump (mirror of mac_tex_ladder) -------------------------------
def _basecolor_sat(feats):
    a = feats.detach().float().cpu().numpy()
    bc = np.clip(a[:, 0:3], 0, 1)
    mx = bc.max(1); mn = bc.min(1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return {
        "n_voxels": int(a.shape[0]),
        "base_color_mean_rgb": [round(float(x), 4) for x in bc.mean(0)],
        "mean_sat": round(float(sat.mean()), 4),
        "frac_sat_gt_0.2": round(float((sat > 0.2).mean()), 4),
        "frac_red_dominant": round(float(((bc[:, 0] - np.maximum(bc[:, 1], bc[:, 2])) > 0.1).mean()), 4),
        "metallic_mean": round(float(np.clip(a[:, 3], 0, 1).mean()), 4) if a.shape[1] > 3 else None,
    }


def _patched_dump(name, payload, fixture_dir=None):
    if "natten" in name or "rng_state" in name or "_step" in name:
        return
    try:
        if name == "06_tex_slat_decoded":
            feats = getattr(payload, "feats", None)
            if feats is not None:
                print("STAGE_JSON: " + json.dumps({"stage": name, **_basecolor_sat(feats)}), flush=True)
            print("[ladder] reached 06 — exiting before bake", flush=True)
            os._exit(0)
        if name == "04_shape_slat_decoded":
            subs = payload[1] if isinstance(payload, (tuple, list)) and len(payload) >= 2 else None
            if isinstance(subs, (tuple, list)):
                info = [{"i": i, "n": int(s.feats.shape[0]), "std": round(float(s.feats.std()), 2)}
                        for i, s in enumerate(subs) if getattr(s, "feats", None) is not None]
                print("STAGE_JSON: " + json.dumps({"stage": name, "subs": info}), flush=True)
    except Exception as exc:
        print(f"STAGE_JSON: {json.dumps({'stage': name, 'error': repr(exc)})}", flush=True)


gm._dump_fixture = _patched_dump

sys.argv = [
    "generate_mps.py",
    "assets/images/9_img.png",
    "--output", "/tmp/mac_recalib_out",
    "--device", "mps",
    "--pipeline-type", "1024_cascade",
    "--seed", "42",
]
gm.main()
print("WARN: finished without reaching 06", flush=True)
