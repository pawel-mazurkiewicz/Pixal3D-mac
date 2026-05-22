#!/usr/bin/env bash
# One-shot environment setup for a fresh CUDA rental box.
#
# Creates (or refreshes) a conda env, installs Pixal3D's hfdemo
# requirements with two pinned overrides that ship wheels for older
# NVIDIA arches, then runs `hf auth login` so model downloads work.
#
# Overrides vs requirements-hfdemo.txt:
#   - natten 0.21.0+torch2.6cu124  ->  0.17.5+torch260cu124  (whl.natten.org
#                                       has wheels for sm_70/sm_75/sm_80;
#                                       upstream wheel is sm_86+ only).
#   - flash_attn_3 3.0.0b1         ->  flash_attn 2.8.3+cu124torch2.6
#                                       (mjun0812's prebuild wheels work on
#                                       older arches; flash_attn_3 v3 doesn't).
#
# Usage:
#   ./scripts/setup_rental.sh                 # creates env if missing, idempotent
#   ./scripts/setup_rental.sh --recreate      # wipes env first
#   ./scripts/setup_rental.sh --no-hf-login   # skip the auth step
#   ./scripts/setup_rental.sh --env-name foo  # custom env name (default: trellis2)
#
# Designed to be re-runnable on rental images that already have Pixal3D
# cloned and miniconda preinstalled.

set -euo pipefail

# --- defaults ---
ENV_NAME="pixal3d"
PYTHON_VER="3.10"
RECREATE=0
RUN_HF_LOGIN=1

FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu124torch2.6-cp310-cp310-linux_x86_64.whl"
NATTEN_SPEC="natten==0.17.5+torch260cu124"
NATTEN_INDEX="https://whl.natten.org"

usage() {
    cat <<EOF
Usage: $0 [--env-name NAME] [--python VER] [--recreate] [--no-hf-login]

Sets up a conda env for running Pixal3D's CUDA inference (and the FDG-
tensor capture flow). Idempotent: re-run after a partial failure and
it picks up where it left off.

Options:
  --env-name NAME    conda env name (default: $ENV_NAME)
  --python VER       python version (default: $PYTHON_VER)
  --recreate         wipe the env first if it exists
  --no-hf-login      skip 'hf auth login' at the end
  -h, --help         this help

After this finishes, you can run:
  conda activate $ENV_NAME
  ./scripts/run_cuda_capture.sh --image assets/images/0_img.png \\
      --output /workspace/out.glb --fixtures /workspace/fixtures_cuda
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)    ENV_NAME="$2";    shift 2 ;;
        --python)      PYTHON_VER="$2";  shift 2 ;;
        --recreate)    RECREATE=1;       shift ;;
        --no-hf-login) RUN_HF_LOGIN=0;   shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             echo "Unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

# --- locate this repo so we install the right requirements file ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REQ_FILE="$REPO_ROOT/requirements-hfdemo.txt"
if [[ ! -f "$REQ_FILE" ]]; then
    echo "ERROR: $REQ_FILE not found. Are you in the Pixal3D repo?" >&2
    exit 1
fi

# --- conda must be available ---
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: 'conda' not on PATH. Install miniconda first, or" >&2
    echo "       source ~/miniconda3/etc/profile.d/conda.sh before re-running." >&2
    exit 1
fi
# Enable `conda activate` inside this script.
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"

# --- info banner ---
echo "============================================================"
echo "Pixal3D rental setup"
echo "============================================================"
echo "  env name    : $ENV_NAME"
echo "  python      : $PYTHON_VER"
echo "  recreate    : $RECREATE"
echo "  hf login    : $RUN_HF_LOGIN"
echo "  repo root   : $REPO_ROOT"
echo "  requirements: $REQ_FILE"
echo "============================================================"
echo

# --- step 1: create or refresh the conda env ---
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [[ $RECREATE -eq 1 ]]; then
        echo "[setup] removing existing env '$ENV_NAME' (--recreate)"
        conda env remove -y -n "$ENV_NAME"
    else
        echo "[setup] env '$ENV_NAME' exists, reusing (pass --recreate to wipe)"
    fi
fi
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] creating env '$ENV_NAME' (python $PYTHON_VER)"
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VER" pip
fi

conda activate "$ENV_NAME"
echo "[setup] active env: $CONDA_DEFAULT_ENV  ($(command -v python))"
python --version

# --- step 2: install most of requirements-hfdemo.txt, EXCEPT the natten
#             and flash_attn lines that we override below.
#             Using `grep -v` to filter; install via pip from the filtered
#             stream so pip never tries to fetch the wrong wheels.
echo
echo "[setup] installing requirements-hfdemo.txt (excluding natten + flash_attn) ..."
TMP_REQ="$(mktemp -t pixal3d-req.XXXXXX.txt)"
trap 'rm -f "$TMP_REQ"' EXIT
grep -viE '(natten|flash_attn)' "$REQ_FILE" > "$TMP_REQ"
echo "  filtered $(wc -l < "$REQ_FILE") -> $(wc -l < "$TMP_REQ") lines"
pip install --no-cache-dir -r "$TMP_REQ"

# --- step 3: clean out any pre-existing natten / flash_attn variants ---
echo
echo "[setup] removing any pre-existing natten / flash_attn packages ..."
pip uninstall -y natten 2>/dev/null || true
pip uninstall -y flash_attn 2>/dev/null || true
pip uninstall -y flash-attn 2>/dev/null || true
pip uninstall -y flash_attn_3 2>/dev/null || true

# --- step 4: install the pinned overrides (older-arch-friendly wheels) ---
echo
echo "[setup] installing flash_attn 2.8.3+cu124torch2.6 prebuilt wheel ..."
pip install --no-cache-dir "$FLASH_ATTN_WHEEL"

echo
echo "[setup] installing $NATTEN_SPEC from $NATTEN_INDEX ..."
pip install --no-cache-dir "$NATTEN_SPEC" -f "$NATTEN_INDEX"

# --- step 5: sanity checks ---
echo
echo "[setup] sanity checks ..."
python - <<'PY'
import importlib, sys
def chk(name, attr=None):
    try:
        m = importlib.import_module(name)
        ver = getattr(m, attr or "__version__", "?")
        print(f"  ok  {name:14s} {ver}")
    except Exception as e:
        print(f"  ERR {name:14s} -> {e}", file=sys.stderr)
        sys.exit(1)
chk("torch")
chk("torchvision")
chk("transformers")
chk("flash_attn")
chk("natten")
import torch
print(f"  ok  torch.cuda.is_available() = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  ok  device: {torch.cuda.get_device_name(0)}")
    print(f"  ok  capability: {torch.cuda.get_device_capability(0)}")
PY

# --- step 6: hf auth (interactive) ---
if [[ $RUN_HF_LOGIN -eq 1 ]]; then
    echo
    echo "[setup] running 'hf auth login' (interactive — paste your HF token) ..."
    # 'hf' CLI ships with huggingface_hub (pulled in by transformers).
    # Older versions only have 'huggingface-cli login'; try both.
    if command -v hf >/dev/null 2>&1; then
        hf auth login
    elif command -v huggingface-cli >/dev/null 2>&1; then
        echo "  (hf CLI not found; falling back to legacy huggingface-cli)"
        huggingface-cli login
    else
        echo "  ERROR: neither 'hf' nor 'huggingface-cli' on PATH" >&2
        exit 1
    fi
fi

echo
echo "============================================================"
echo "Setup complete."
echo "============================================================"
echo "  conda activate $ENV_NAME"
echo "  ./scripts/run_cuda_capture.sh --help"
