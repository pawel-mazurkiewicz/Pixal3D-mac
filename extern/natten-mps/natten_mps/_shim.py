"""Monkey-patch natten dispatch so MPS tensors route to our Metal kernels.

Falls through to upstream natten for CPU / CUDA tensors so the shim is
transparent on non-MPS hardware.

This is a *temporary* shape — eventually we want a proper `metal-fna`
backend registered into natten's dispatch table via PR upstream.
"""

from __future__ import annotations

import os
import warnings
import torch


_PATCHED = False


def install(force: bool = False) -> bool:
    """Patch `natten.na2d`.  Returns True if patched, False if skipped."""
    global _PATCHED
    if _PATCHED and not force:
        return False
    if os.environ.get("NATTEN_MPS_DISABLE", "").strip() == "1":
        return False

    try:
        import natten as _natten
    except ImportError:
        warnings.warn(
            "natten not installed; natten-mps shim not activated.", stacklevel=2
        )
        return False

    if not hasattr(_natten, "na2d"):
        warnings.warn(
            "natten.na2d not found (older natten?); shim not activated.",
            stacklevel=2,
        )
        return False

    _orig_na2d = _natten.na2d

    def _na2d_dispatch(query, key, value, kernel_size, *args, **kwargs):
        # Route MPS to our kernels; everything else to upstream.
        if query.device.type == "mps":
            return _na2d_mps_impl(query, key, value, kernel_size, *args, **kwargs)
        return _orig_na2d(query, key, value, kernel_size, *args, **kwargs)

    _natten.na2d = _na2d_dispatch
    _natten._original_na2d = _orig_na2d  # type: ignore[attr-defined]

    # If anything imported `from natten import na2d` before our shim ran,
    # it holds a stale reference to the original function.  Walk
    # sys.modules and swap any module-local `na2d` binding identical (by
    # object identity) to the original we just replaced.  This is the
    # same trick used by the rental capture wrapper — proven necessary
    # for NAF's `from natten import na2d` import pattern.
    import sys as _sys
    _patched_local = []
    for _modname, _mod in list(_sys.modules.items()):
        if _mod is None or _mod is _natten:
            continue
        try:
            _local = getattr(_mod, "na2d", None)
        except Exception:
            continue
        if _local is _orig_na2d:
            try:
                setattr(_mod, "na2d", _na2d_dispatch)
                _patched_local.append(_modname)
            except Exception:
                pass
    if _patched_local:
        # Visible-but-not-noisy: print once per process so it's clear
        # the dispatch ALSO covers the local binding.
        print(f"[natten-mps] dispatch shim installed "
              f"(natten.na2d + local: {_patched_local})", flush=True)

    # Print a final summary if VERBOSE.  When the wrapper never fires
    # AND the caller is in verbose mode, scream loudly — silent installs
    # hide bugs.  Under normal use we stay quiet so test runs that use
    # torch.ops.natten_mps.* directly (no natten.na2d call) don't get a
    # misleading WARN.
    import atexit as _atexit
    def _final_count():
        n = _NA2D_MPS_CALL_COUNT[0]
        verbose = os.environ.get("NATTEN_MPS_VERBOSE", "").strip() == "1"
        if n > 0:
            print(f"[natten-mps] dispatched {n} call(s) total.", flush=True)
        elif verbose:
            print("[natten-mps] !!! WARN: shim was installed but the MPS "
                  "dispatch wrapper NEVER FIRED.  Either no MPS tensors "
                  "reached natten, or NAF/caller bypassed the patched "
                  "binding (e.g. lazy-imported natten via a path we "
                  "didn't rebind).", flush=True)
    _atexit.register(_final_count)

    _PATCHED = True
    return True


# Per-process call counter for diagnostic.  Bumped on every MPS dispatch.
# Printed at process exit if NATTEN_MPS_VERBOSE=1.
_NA2D_MPS_CALL_COUNT = [0]


def _na2d_mps_impl(query, key, value, kernel_size, *args, **kwargs):
    """Translate natten's modern fused na2d call into our split path
    (until na2d_fused is implemented).

    Modern natten layout: [B, X, Y, H, D]  (heads-last)
    Our kernels expect:   [B, H, X, Y, D]  (heads-second)
    """
    from . import _ops  # noqa: F401  ensures ops are registered
    from einops import rearrange

    _NA2D_MPS_CALL_COUNT[0] += 1
    if os.environ.get("NATTEN_MPS_VERBOSE", "").strip() == "1":
        print(f"[natten-mps] dispatch #{_NA2D_MPS_CALL_COUNT[0]}: "
              f"q={tuple(query.shape)} ks={kernel_size} "
              f"dil={kwargs.get('dilation', 1)} "
              f"dtype={query.dtype} dev={query.device}",
              flush=True)

    # Optional one-shot IO capture for cross-platform diffing.  Set
    # NATTEN_MPS_CAPTURE_FIRST=/path/to/dir to dump the inputs (q, k,
    # v in modern heads-last layout) AND our split-path intermediates
    # (attn_scores pre-softmax, attn_weights post-softmax, out) for
    # the FIRST call.  Pairs with the rental capture format produced
    # by Pixal3D_fresh's inference.py natten wrapper, so a diff tool
    # can answer 'did the inputs to natten match between Mac and CUDA?'
    _capture_dir = os.environ.get("NATTEN_MPS_CAPTURE_FIRST", "").strip()
    _do_capture = _capture_dir and _NA2D_MPS_CALL_COUNT[0] == 1

    # Resolve scale (modern na2d default is head_dim ** -0.5)
    scale = kwargs.get("scale")
    if scale is None:
        scale = query.shape[-1] ** -0.5

    # Normalize kernel_size + dilation to ints
    ks = kernel_size if isinstance(kernel_size, int) else int(kernel_size[0])
    dilation = kwargs.get("dilation", 1)
    if not isinstance(dilation, int):
        dilation = int(dilation[0])

    # Modern layout -> legacy split layout
    q = rearrange(query, "b x y h d -> b h x y d").contiguous()
    k = rearrange(key,   "b x y h d -> b h x y d").contiguous()
    v = rearrange(value, "b x y h d -> b h x y d").contiguous()

    attn_scores = torch.ops.natten_mps.na2d_qk(q, k, ks, dilation) * scale
    attn = attn_scores.softmax(dim=-1).to(v.dtype)
    out = torch.ops.natten_mps.na2d_av(attn, v, ks, dilation)

    if _do_capture:
        try:
            os.makedirs(_capture_dir, exist_ok=True)
            payload = {
                # Heads-second layout (matches Pixal3D_fresh capture format):
                #   q, k, v: [B, H, X, Y, D]
                #   attn_scores: [B, H, X, Y, K*K]
                #   attn: same shape, post-softmax (post-scale)
                #   out: [B, H, X, Y, D]
                "q":  q.detach().cpu(),
                "k":  k.detach().cpu(),
                "v":  v.detach().cpu(),
                "attn_scores": attn_scores.detach().cpu(),
                "attn": attn.detach().cpu(),
                "out": out.detach().cpu(),
                "kernel_size": ks,
                "dilation": dilation,
                "scale": float(scale),
                "shapes_str": (
                    f"q={tuple(q.shape)} k={tuple(k.shape)} "
                    f"v={tuple(v.shape)} out={tuple(out.shape)} "
                    f"ks={ks} dilation={dilation}"
                ),
            }
            import torch as _torch
            _path = os.path.join(_capture_dir, "natten_mps_call0.pt")
            _torch.save(payload, _path)
            print(f"[natten-mps] captured first call -> {_path}", flush=True)
        except Exception as _exc:
            print(f"[natten-mps] WARN: capture failed: {_exc}", flush=True)

    return rearrange(out, "b h x y d -> b x y h d").contiguous()
