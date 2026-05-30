"""Lazy compilation of the Metal shader library.

`torch.mps.compile_shader` JIT-compiles inline MSL into a `_Library` object
whose kernel functions become Python attributes.  We pay the ~100-200 ms
compile cost once per process, cached as a module-level singleton.
"""

from __future__ import annotations

from importlib.resources import files
import torch


_LIB = None  # set on first `get_lib()` call
# shared.metal MUST come first — it defines helpers used by both kernels.
_KERNEL_SOURCES = ("shared.metal", "na2d_qk.metal", "na2d_av.metal")


def get_lib():
    """Return the compiled Metal shader library, compiling on first call."""
    global _LIB
    if _LIB is not None:
        return _LIB

    if not torch.backends.mps.is_available():
        raise RuntimeError("natten-mps requires PyTorch MPS to be available")
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError(
            "natten-mps requires torch.mps.compile_shader (PyTorch >= 2.5)"
        )

    sources = []
    shader_root = files("natten_mps.shaders")
    for fname in _KERNEL_SOURCES:
        sources.append(shader_root.joinpath(fname).read_text())
    source = "\n\n".join(sources)

    _LIB = torch.mps.compile_shader(source)
    return _LIB
