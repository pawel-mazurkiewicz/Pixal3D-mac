#!/usr/bin/env python
"""PIXAL3D_FP32_ATTN diagnostic (S21) — extracted as a runtime monkeypatch.

Forces the DENSE image-cond attention (`pixal3d.modules.attention.full_attn`) to
run SDPA in fp32 on MPS, to test whether fp16 SDPA accumulators were a colour/
quality factor.  RULED OUT (INVESTIGATION_FACTS §4E: "changes nothing").  Kept as
a runnable probe so the production module stays pristine.

Install BEFORE the pipeline is built (so the wrapped function is the one used):

    import scripts.probe_fp32_attn as p; p.install()
    import generate_mps as gm; gm.load_runtime_deps(); ...

NB: monkeypatches the module attribute; if a caller did
`from ...full_attn import scaled_dot_product_attention` at import time it would
hold the original — install early, before such imports resolve.
"""
import torch
import pixal3d.modules.attention.full_attn as _FA

_orig = _FA.scaled_dot_product_attention
_installed = [False]


def install():
    if _installed[0]:
        return

    def _up(x):
        return x.float() if torch.is_tensor(x) and x.is_floating_point() else x

    def wrapped(*args, **kwargs):
        dt = next((a.dtype for a in args if torch.is_tensor(a) and a.is_floating_point()), None)
        out = _orig(*[_up(a) for a in args], **{k: _up(v) for k, v in kwargs.items()})
        if dt is not None and torch.is_tensor(out):
            out = out.to(dt)
        return out

    _FA.scaled_dot_product_attention = wrapped
    _installed[0] = True
    print("[probe_fp32_attn] dense SDPA forced to fp32 (S21 PIXAL3D_FP32_ATTN equivalent)")


def uninstall():
    _FA.scaled_dot_product_attention = _orig
    _installed[0] = False
