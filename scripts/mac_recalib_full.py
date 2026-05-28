#!/usr/bin/env python
"""Full pipeline with BOTH recalib levers -> bakes a GLB for visual verification.

Levers (gated by PIXAL3D_RECALIB=1; default on here):
  - shape decoder subdiv logits  -> moment-matched per level to CUDA (geometry)
  - tex_slat (decode_tex_slat in) -> moment-matched per channel to CUDA (colour)

Runs generate_mps.main() to completion (sampling -> decode -> bake -> GLB).

  .venv-py310/bin/python scripts/mac_recalib_full.py <out.glb>
"""
import os, sys, json
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO); sys.path.insert(0, REPO)

import numpy as np
import torch
import generate_mps as gm
import pixal3d.models.sc_vaes.sparse_unet_vae as M

OUT = sys.argv[1] if len(sys.argv) > 1 else "robocrab_recalib.glb"
RECALIB = os.environ.get("PIXAL3D_RECALIB", "1").strip() == "1"

# subdiv per-level (mac_mean, mac_std, cuda_mean, cuda_std)
_COEFFS = [
    (38.24624, 115.35922, 41.88572, 129.72485),
    (4.03776,  88.61266,  13.42821, 101.24020),
    (-5.33891, 29.49385,  -5.29934, 49.16254),
    (-1.05075, 20.17925,  -1.09990, 37.73084),
]
_LEVEL = [0]
# CUDA tex_slat per-channel stats (loaded once)
_CU_TEX = np.load("/tmp/cuda_ladder/05_tex_slat.npz")["feats"]
_CU_TMEAN = torch.tensor(_CU_TEX.mean(0))
_CU_TSTD = torch.tensor(_CU_TEX.std(0))

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
_orig_dec_up = getattr(M.SparseUnetVaeDecoder, "upsample", None)


def _reset_fwd(self, *a, **k):
    _LEVEL[0] = 0
    return _orig_dec_fwd(self, *a, **k)


if RECALIB:
    M.SparseResBlockC2S3d._forward = _recalib_forward
    M.SparseUnetVaeDecoder.forward = _reset_fwd
    if _orig_dec_up is not None:
        def _reset_up(self, *a, **k):
            _LEVEL[0] = 0
            return _orig_dec_up(self, *a, **k)
        M.SparseUnetVaeDecoder.upsample = _reset_up

# Wrap decode_tex_slat to recalibrate tex_slat per-channel to CUDA.
_orig_init = gm.init_pipeline


def _init_hook(*a, **k):
    p = _orig_init(*a, **k)
    if RECALIB:
        _orig_dts = p.decode_tex_slat

        def _wrap_dts(slat, subs, *aa, **kk):
            tf = slat.feats
            macm = tf.mean(0, keepdim=True)
            macs = tf.std(0, keepdim=True).clamp_min(1e-6)
            cum = _CU_TMEAN.to(tf)[None]; cus = _CU_TSTD.to(tf)[None]
            slat = slat.replace((tf - macm) / macs * cus + cum)
            print(f"[tex-recalib] tex_slat std {float(tf.std()):.3f} -> {float(slat.feats.std()):.3f}", flush=True)
            return _orig_dts(slat, subs, *aa, **kk)
        p.decode_tex_slat = _wrap_dts
    return p


gm.init_pipeline = _init_hook
print(f"[full] RECALIB={RECALIB} -> {OUT}", flush=True)

sys.argv = ["generate_mps.py", "assets/images/9_img.png", "--output", OUT,
            "--device", "mps", "--pipeline-type", "1024_cascade", "--seed", "42"]
gm.main()
print(f"[full] done -> {OUT}", flush=True)
