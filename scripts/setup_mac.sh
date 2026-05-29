#!/usr/bin/env bash
#
# setup_mac.sh — one-shot environment setup for the Pixal3D Apple Silicon port.
#
# Creates .venv (Python 3.10), installs the pinned Mac dependency set, and
# editable-installs the six vendored native packages in extern/ (which compile
# their Metal shaders to .metallib at install time).
#
# Idempotent: safe to re-run. Pass --recreate to wipe and rebuild .venv.
#
# Requirements (checked below):
#   - Apple Silicon Mac
#   - Xcode Command Line Tools (xcrun metal) — the extern packages build .metallib
#     ad hoc during install, so the Metal toolchain MUST be present first.
#   - Python 3.10 (Homebrew: brew install python@3.10)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${VENV:-.venv}"
RECREATE=0
[[ "${1:-}" == "--recreate" ]] && RECREATE=1

say() { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- preflight ---------------------------------------------------------------

[[ "$(uname -s)" == "Darwin" ]] || die "This setup is macOS-only."
if [[ "$(uname -m)" != "arm64" ]]; then
    die "Apple Silicon (arm64) required. Detected: $(uname -m)."
fi

say "Checking Metal toolchain (xcrun metal)…"
if ! xcrun --find metal >/dev/null 2>&1; then
    die "Metal compiler not found. Install the Xcode Command Line Tools first:
       xcode-select --install
     then verify with:  xcrun --find metal
     (Some macOS versions require a full Xcode install for the Metal compiler.)"
fi

# Locate a Python 3.10 interpreter.
PY="${PYTHON_BIN:-}"
if [[ -z "$PY" ]]; then
    for cand in \
        /opt/homebrew/opt/python@3.10/bin/python3.10 \
        "$(command -v python3.10 2>/dev/null || true)"; do
        if [[ -n "$cand" && -x "$cand" ]]; then PY="$cand"; break; fi
    done
fi
[[ -n "$PY" && -x "$PY" ]] || die "Python 3.10 not found. Install it (brew install python@3.10) or set PYTHON_BIN=/path/to/python3.10."

PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[[ "$PYVER" == "3.10" ]] || die "Expected Python 3.10, got $PYVER ($PY). The pinned deps require 3.10."
say "Using Python: $PY ($PYVER)"

# --- venv --------------------------------------------------------------------

if [[ $RECREATE -eq 1 && -d "$VENV" ]]; then
    say "--recreate: removing existing $VENV"
    rm -rf "$VENV"
fi
if [[ ! -d "$VENV" ]]; then
    say "Creating venv at $VENV"
    "$PY" -m venv "$VENV"
fi
VPY="$VENV/bin/python"

say "Upgrading pip / wheel / setuptools"
"$VPY" -m pip install -U pip wheel setuptools

# --- dependencies ------------------------------------------------------------

say "Installing pinned Mac dependencies (requirements-mac.txt)…"
"$VPY" -m pip install -r requirements-mac.txt

# --- vendored native Metal packages -----------------------------------------

# Order matters only loosely; install all editable with no build isolation.
PKGS=(mtlmesh mtlgemm mtlbvh mtldiffrast o_voxel natten-mps)
for pkg in "${PKGS[@]}"; do
    [[ -d "extern/$pkg" ]] || die "extern/$pkg missing — is the repo checked out fully?"
    say "Installing extern/$pkg (compiles Metal shaders)…"
    "$VPY" -m pip install -e "extern/$pkg" --no-build-isolation
done

# --- smoke check -------------------------------------------------------------

say "Verifying MPS + native imports…"
"$VPY" - <<'PYCHECK'
import torch
assert torch.backends.mps.is_available(), "MPS backend not available"
import cumesh, flex_gemm, mtlbvh, mtldiffrast, o_voxel, natten_mps  # noqa: F401
print("[setup]   torch", torch.__version__, "| MPS available | native packages import OK")
PYCHECK

say "Done. Run a generation with:"
echo "    $VPY generate_mps.py assets/images/0_img.png --output output_3d.glb"
