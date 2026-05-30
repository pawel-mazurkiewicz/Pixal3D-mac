"""natten-mps: Metal/MPS backend for NATTEN on Apple Silicon."""

from __future__ import annotations

from . import _ops  # registers custom ops
from . import _shim
from ._ops import na2d_qk, na2d_av

__version__ = "0.1.0a0"
__all__ = ["na2d_qk", "na2d_av", "install_shim", "__version__"]


def install_shim(force: bool = False) -> bool:
    """Monkey-patch `natten.na2d` to route MPS tensors to our Metal kernels.

    Returns True if patched, False if skipped (already patched / disabled /
    natten not importable).  Set ``NATTEN_MPS_DISABLE=1`` to no-op.
    """
    return _shim.install(force=force)


# Auto-install on import unless explicitly disabled.
import os as _os
if _os.environ.get("NATTEN_MPS_NO_AUTOINSTALL", "").strip() != "1":
    install_shim()
