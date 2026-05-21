#!/usr/bin/env bash
# Run Pixal3D inference on Apple Silicon MPS with fixture capture enabled.
#
# Counterpart of run_cuda_capture.sh.  Same args, but defaults the env
# variables Mac needs (sdpa attention, low_vram).
#
# Assumes:
#   - You have a Python venv at .venv with TRELLIS.2 + Pixal3D deps installed.
#   - Pixal3D's inference.py carries the fixture instrumentation.
#   - You're cd'd to the Pixal3D_fresh repo root.
#
# Usage:
#   ./scripts/run_mps_capture.sh --image assets/images/0_img.png \
#                                --output /tmp/out_mps.glb \
#                                --seed 42 \
#                                --resolution 1024 \
#                                --fixtures /tmp/fixtures_mps

set -euo pipefail

# --- defaults ---
IMAGE=""
OUTPUT=""
SEED=42
RESOLUTION=1024
FIXTURES=""
PYTHON_BIN=".venv/bin/python"
EXTRA_ARGS=()
NO_LOW_VRAM=0

usage() {
    cat <<EOF
Usage: $0 --image PATH --output PATH --fixtures DIR [--seed N] [--resolution N] [--python PATH] [--no-low-vram] [-- ...extra inference.py args]

Required:
  --image PATH        input image (must match the one used on CUDA byte-for-byte)
  --output PATH       output GLB path
  --fixtures DIR      where to dump fixtures

Optional:
  --seed N            random seed (default: $SEED, MUST match CUDA-side seed)
  --resolution N      1024 or 1536 (default: $RESOLUTION, MUST match CUDA side)
  --python PATH       python interpreter to use (default: $PYTHON_BIN)
  --no-low-vram       run without --low_vram (validates that low_vram is numerics-equivalent)
  -- ...              everything after \`--\` is passed through to inference.py
EOF
}

# --- arg parse ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)        IMAGE="$2";       shift 2 ;;
        --output)       OUTPUT="$2";      shift 2 ;;
        --seed)         SEED="$2";        shift 2 ;;
        --resolution)   RESOLUTION="$2";  shift 2 ;;
        --fixtures)     FIXTURES="$2";    shift 2 ;;
        --python)       PYTHON_BIN="$2";  shift 2 ;;
        --no-low-vram)  NO_LOW_VRAM=1;    shift ;;
        --)             shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$IMAGE" || -z "$OUTPUT" || -z "$FIXTURES" ]]; then
    echo "Missing required arg(s)." >&2
    usage
    exit 1
fi
if [[ ! -f "$IMAGE" ]]; then
    echo "Image not found: $IMAGE" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
    echo "Pass --python /path/to/python to override." >&2
    exit 1
fi

mkdir -p "$FIXTURES"
mkdir -p "$(dirname "$OUTPUT")"

# --- env: Mac needs sdpa instead of flash_attn ---
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
# SPARSE_CONV_BACKEND: leave unset and let the pipeline pick.  If your venv
# doesn't have flex_gemm-on-Metal, set SPARSE_CONV_BACKEND=none before
# invoking this script.
export PIXAL3D_DUMP_FIXTURES="$FIXTURES"

# Default to low_vram unless caller opts out
LOW_VRAM_FLAG=(--low_vram)
if [[ $NO_LOW_VRAM -eq 1 ]]; then
    LOW_VRAM_FLAG=()
fi

# --- info banner ---
echo "============================================================"
echo "Pixal3D MPS capture"
echo "============================================================"
echo "  image       : $IMAGE"
echo "  output      : $OUTPUT"
echo "  seed        : $SEED"
echo "  resolution  : $RESOLUTION"
echo "  fixtures    : $FIXTURES"
echo "  low_vram    : $([ $NO_LOW_VRAM -eq 0 ] && echo on || echo off)"
echo "  ATTN_BACKEND: $ATTN_BACKEND"
echo "  extra args  : ${EXTRA_ARGS[*]:-(none)}"
echo "  python      : $PYTHON_BIN"
echo "  torch       : $($PYTHON_BIN -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '???')"
echo "  mps avail   : $($PYTHON_BIN -c 'import torch; print(torch.backends.mps.is_available())' 2>/dev/null || echo '???')"
echo "============================================================"
echo

# --- run ---
LOG="$FIXTURES/run.log"
START=$(date +%s)
set +e
"$PYTHON_BIN" generate_mps.py \
    "$IMAGE" \
    --output "$OUTPUT" \
    --seed "$SEED" \
    --pipeline-type "${RESOLUTION}_cascade" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
END=$(date +%s)
ELAPSED=$((END - START))

echo
echo "============================================================"
echo "Capture finished in ${ELAPSED}s (exit $RC)"
echo "============================================================"

if [[ $RC -ne 0 ]]; then
    echo "inference.py exited non-zero — fixtures may be incomplete."
fi

echo
echo "Fixtures written to $FIXTURES:"
ls -lh "$FIXTURES" | sed 's/^/  /'

echo
echo "Now diff against the CUDA capture:"
echo "  $PYTHON_BIN scripts/diff_fixtures.py /path/to/fixtures_cuda $FIXTURES"

exit $RC
