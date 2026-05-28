#!/usr/bin/env python
"""Decode-only replay harness — skip the ~11-min HR sampling on every iteration.

The slow part of the pipeline is sampling (esp. HR shape SLat ~54s/step).  The
decode (decode_shape_slat -> decode_tex_slat) is what we actually iterate on for
the recalibration/op-hunt work.  This harness:

  capture:  run the full pipeline once, torch.save the decode inputs
            (shape_slat = decode_shape_slat input, tex_slat = sample_tex_slat
            output), then os._exit before the slow decode+bake.
  replay:   init pipeline (models only), load the cached inputs, run
            decode_shape_slat (+ optional recalib) -> subs, decode_tex_slat ->
            06 field, print subs std + 06 saturation.  ~4 min vs ~18 min.

With shape_slat FIXED, recalibrating the final decode cleanly isolates its
contribution to geometry + colour (no Stage-3a upsample() confound).

Usage:
  .venv-py310/bin/python scripts/mac_decode_replay.py capture
  PIXAL3D_RECALIB=1 .venv-py310/bin/python scripts/mac_decode_replay.py replay
"""
import os, sys, json
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO); sys.path.insert(0, REPO)
CACHE = "/tmp/decode_cache"
os.makedirs(CACHE, exist_ok=True)

import numpy as np
import torch
import generate_mps as gm
import pixal3d.models.sc_vaes.sparse_unet_vae as M

MODE = sys.argv[1] if len(sys.argv) > 1 else "replay"
RECALIB = os.environ.get("PIXAL3D_RECALIB", "").strip() == "1"

# Per-level (mac_mean, mac_std, cuda_mean, cuda_std) subdiv-logit moments
# (robocrab 9_img.png seed 42 1024_cascade, subs0..subs3).
_COEFFS = [
    (38.24624, 115.35922, 41.88572, 129.72485),
    (4.03776,  88.61266,  13.42821, 101.24020),
    (-5.33891, 29.49385,  -5.29934, 49.16254),
    (-1.05075, 20.17925,  -1.09990, 37.73084),
]
_LEVEL = [0]


def _drain(st):
    if hasattr(st, "_spatial_cache") and isinstance(st._spatial_cache, dict):
        st._spatial_cache = {}
    return st


def _sat(feats):
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


# ----------------------------------------------------------------- capture ---
if MODE == "capture":
    os.environ["PIXAL3D_DUMP_FIXTURES"] = "/tmp/decode_capture_run"
    os.makedirs("/tmp/decode_capture_run", exist_ok=True)
    _state = {"tex_slat": None}

    _orig_sample_tex = None

    def _install(pipeline):
        global _orig_sample_tex
        _orig_sample_tex = pipeline.sample_tex_slat

        def _wrap_tex(*a, **k):
            out = _orig_sample_tex(*a, **k)
            torch.save(_drain(out), os.path.join(CACHE, "tex_slat.pt"))
            print(f"[capture] saved tex_slat feats={tuple(out.feats.shape)}", flush=True)
            return out
        pipeline.sample_tex_slat = _wrap_tex

        _orig_decode_shape = pipeline.decode_shape_slat

        def _wrap_decode_shape(slat, *a, **k):
            res = a[0] if a else k.get("resolution")
            torch.save(_drain(slat), os.path.join(CACHE, "shape_slat.pt"))
            with open(os.path.join(CACHE, "meta.json"), "w") as f:
                json.dump({"resolution": int(res), "feats": list(slat.feats.shape)}, f)
            print(f"[capture] saved shape_slat feats={tuple(slat.feats.shape)} res={res} — exiting",
                  flush=True)
            os._exit(0)
        pipeline.decode_shape_slat = _wrap_decode_shape

    _orig_init = gm.init_pipeline

    def _init_hook(*a, **k):
        p = _orig_init(*a, **k)
        _install(p)
        return p
    gm.init_pipeline = _init_hook

    sys.argv = ["generate_mps.py", "assets/images/9_img.png", "--output", "/tmp/cap_out",
                "--device", "mps", "--pipeline-type", "1024_cascade", "--seed", "42"]
    gm.main()
    sys.exit(0)


# ------------------------------------------------------------------ replay ---
_orig_c2s = M.SparseResBlockC2S3d._forward


def _recalib_forward(self, x, subdiv=None):
    import torch.nn.functional as F
    if self.pred_subdiv:
        subdiv = self.to_subdiv(x)
        if RECALIB:
            i = _LEVEL[0]; _LEVEL[0] += 1
            if i < len(_COEFFS):
                mm, ms, cm, cs = _COEFFS[i]
                subdiv = subdiv.replace((subdiv.feats - mm) / ms * cs + cm)
    h = x.replace(self.norm1(x.feats)); h = h.replace(F.silu(h.feats)); h = self.conv1(h)
    sb = subdiv.replace(subdiv.feats > -M._SUBDIV_BIAS) if subdiv is not None else None
    h = self.updown(h, sb); x = self.updown(x, sb)
    h = h.replace(self.norm2(h.feats)); h = h.replace(F.silu(h.feats)); h = self.conv2(h)
    h = h + self.skip_connection(x)
    return (h, subdiv) if self.pred_subdiv else h


_orig_dec_fwd = M.SparseUnetVaeDecoder.forward


def _reset_fwd(self, *a, **k):
    _LEVEL[0] = 0
    return _orig_dec_fwd(self, *a, **k)


if RECALIB:
    M.SparseResBlockC2S3d._forward = _recalib_forward
    M.SparseUnetVaeDecoder.forward = _reset_fwd
print(f"[replay] RECALIB={RECALIB}", flush=True)

gm.load_runtime_deps()  # populates generate_mps's torch/model-class globals + MPS env
device = gm.resolve_device("mps")
pipeline = gm.init_pipeline(gm.MODEL_PATH, device)
meta = json.load(open(os.path.join(CACHE, "meta.json")))
res = meta["resolution"]
shape_slat = _drain(torch.load(os.path.join(CACHE, "shape_slat.pt"), map_location="cpu", weights_only=False)).to(device)
tex_slat = _drain(torch.load(os.path.join(CACHE, "tex_slat.pt"), map_location="cpu", weights_only=False)).to(device)
_drain(shape_slat); _drain(tex_slat)
print(f"[replay] loaded shape_slat={tuple(shape_slat.feats.shape)} tex_slat={tuple(tex_slat.feats.shape)} res={res}", flush=True)

# Optional: rescale tex_slat per-channel to CUDA's distribution to test whether
# the tex-flow variance collapse (esp. ch8/ch22) is the colour/red cause.
if os.environ.get("PIXAL3D_TEX_RECALIB", "").strip() == "1":
    cu = np.load("/tmp/cuda_ladder/05_tex_slat.npz")["feats"]
    tf = tex_slat.feats
    macm = tf.mean(0, keepdim=True); macs = tf.std(0, keepdim=True).clamp_min(1e-6)
    cum = torch.tensor(cu.mean(0), dtype=tf.dtype, device=tf.device)[None]
    cus = torch.tensor(cu.std(0), dtype=tf.dtype, device=tf.device)[None]
    tex_slat = tex_slat.replace((tf - macm) / macs * cus + cum)
    print(f"[tex-recalib] rescaled tex_slat per-channel to CUDA "
          f"(std {float(tf.std()):.3f} -> {float(tex_slat.feats.std()):.3f})", flush=True)

with torch.no_grad():
    meshes, subs = pipeline.decode_shape_slat(shape_slat, res)
    info = [{"i": i, "n": int(s.feats.shape[0]), "std": round(float(s.feats.std()), 2)} for i, s in enumerate(subs)]
    print("SUBS_JSON: " + json.dumps(info), flush=True)
    tex_voxels = pipeline.decode_tex_slat(tex_slat, subs)
    print("TEX_JSON: " + json.dumps(_sat(tex_voxels.feats)), flush=True)
print("[replay] done", flush=True)
