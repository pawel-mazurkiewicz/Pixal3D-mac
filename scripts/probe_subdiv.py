#!/usr/bin/env python
"""PIXAL3D_SUBDIV_BIAS (S19) + PIXAL3D_DUMP_SUBDIV (S20) — runtime monkeypatches.

Extracted from sparse_unet_vae.py so the production model stays pristine.

- SUBDIV_BIAS relaxes the cascade subdiv threshold `subdiv.feats > 0` to
  `> -bias` at the three C2S/upsample sites. DOCUMENTED DEAD LEVER (§4B: the
  near-zero band is <2% of values; dropped cells have decisive negatives).
- DUMP_SUBDIV dumps per-level subdiv tensors from SparseResBlockC2S3d for
  Mac<->CUDA cascade-divergence probing (S20).

Install BEFORE the pipeline is built:

    import scripts.probe_subdiv as p
    p.install(bias=0.0, dump_dir="/tmp/subdiv")   # bias 0.0 == exact upstream

Reproduces the exact in-tree bodies that previously lived behind the env vars.
"""
import os
import torch
import torch.nn.functional as F
import pixal3d.models.sc_vaes.sparse_unet_vae as M

_state = {"bias": 0.0, "dump": "", "counter": [0], "orig": {}}


def install(bias=0.0, dump_dir=""):
    _state["bias"] = float(bias)
    _state["dump"] = dump_dir
    _state["counter"][0] = 0
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)

    def _bias():
        return _state["bias"]

    def _updown(self, x, subdiv=None):  # SparseResBlock3d._updown
        if self.downsample:
            x = self.updown(x)
        elif self.upsample:
            x = self.updown(x, subdiv.replace(subdiv.feats > -_bias()))
        return x

    def _up_forward(self, x, subdiv=None):  # SparseResBlockUpsample3d._forward
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        sb = subdiv.replace(subdiv.feats > -_bias()) if subdiv is not None else None
        h = self.updown(h, sb)
        x = self.updown(x, sb)
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return (h, subdiv) if self.pred_subdiv else h

    def _c2s_forward(self, x, subdiv=None):  # SparseResBlockC2S3d._forward (+dump)
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        if _state["dump"] and subdiv is not None:
            idx = _state["counter"][0]
            _state["counter"][0] += 1
            try:
                torch.save({
                    "feats": subdiv.feats.detach().to(torch.float32).cpu(),
                    "coords": subdiv.coords.detach().cpu(),
                    "x_feats": x.feats.detach().to(torch.float32).cpu(),
                    "x_coords": x.coords.detach().cpu(),
                    "x_feats_stats": {
                        "mean": float(x.feats.detach().to(torch.float32).mean().cpu()),
                        "std": float(x.feats.detach().to(torch.float32).std().cpu()),
                    },
                    "level_idx": idx,
                    "channels": self.channels,
                    "out_channels": self.out_channels,
                }, os.path.join(_state["dump"], f"level_{idx:02d}.pt"))
                print(f"[dump_subdiv] level_{idx:02d}.pt: V={subdiv.feats.shape[0]} "
                      f"channels=({self.channels}->{self.out_channels}) "
                      f"feats.shape={tuple(subdiv.feats.shape)} x.feats={tuple(x.feats.shape)}")
            except Exception as e:
                print(f"[dump_subdiv] level {idx} failed: {e}")
        sb = subdiv.replace(subdiv.feats > -_bias()) if subdiv is not None else None
        h = self.updown(h, sb)
        x = self.updown(x, sb)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return (h, subdiv) if self.pred_subdiv else h

    _state["orig"] = {
        "u": M.SparseResBlock3d._updown,
        "up": M.SparseResBlockUpsample3d._forward,
        "c2s": M.SparseResBlockC2S3d._forward,
    }
    M.SparseResBlock3d._updown = _updown
    M.SparseResBlockUpsample3d._forward = _up_forward
    M.SparseResBlockC2S3d._forward = _c2s_forward
    print(f"[probe_subdiv] installed bias={_state['bias']} dump={dump_dir or 'off'}")


def uninstall():
    if _state["orig"]:
        M.SparseResBlock3d._updown = _state["orig"]["u"]
        M.SparseResBlockUpsample3d._forward = _state["orig"]["up"]
        M.SparseResBlockC2S3d._forward = _state["orig"]["c2s"]
        _state["orig"] = {}
