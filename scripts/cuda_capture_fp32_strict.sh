#!/usr/bin/env bash
#
# CUDA fp32-strict capture wrapper (Session 15 / §9-C-decider).
#
# Disables TF32 tensor cores in cuDNN + cuBLAS so the CUDA reference is true
# fp32 (23-bit mantissa) instead of TF32 (10-bit mantissa).  Also enables
# cuDNN algorithm logging so the algos cuDNN picked per conv are visible.
#
# Hypothesis (Session 15): the 6.5e-3 max-abs gap between our Mac fp32 conv
# (direct or Winograd F(4,3)) and the existing CUDA reference is entirely
# explained by TF32 — Ampere+ GPUs default to TF32 for fp32 convs.  If true,
# this capture with TF32 disabled should land within fp32 noise (≤ 1e-4) of
# the Mac output, AND the cuDNN log will show CUDNN_TENSOR_OP_MATH was the
# old default.
#
# Usage on the CUDA rental box (defaults shown):
#   bash scripts/cuda_capture_fp32_strict.sh
#
# Env overrides:
#   ENTRY=<path>       — entry-point script (auto-detected by default; tries
#                        inference.py, generate.py, generate_mps.py in cwd)
#   IMAGE=<path>       — input image (default: assets/images/1_img.png)
#   FIXTURES_DIR=<dir> — capture output dir (default: /tmp/cuda_naf_trace_01b_fp32strict)
#   CUDNN_LOG=<path>   — cuDNN log destination (default: /tmp/cudnn_algos.log)
#   SEED, FOV, OUTPUT  — passed through to the entry script

set -euo pipefail

# ---- defaults ----
FIXTURES_DIR="${FIXTURES_DIR:-/tmp/cuda_naf_trace_01b_fp32strict}"
CUDNN_LOG="${CUDNN_LOG:-/tmp/cudnn_algos.log}"
IMAGE="${IMAGE:-assets/images/1_img.png}"
SEED="${SEED:-42}"
FOV="${FOV:-0.6061}"
OUTPUT="${OUTPUT:-/tmp/throwaway.glb}"

# ---- auto-detect entry point ----
if [[ -z "${ENTRY:-}" ]]; then
    for cand in inference.py generate.py generate_mps.py app.py; do
        if [[ -f "${cand}" ]]; then ENTRY="${cand}"; break; fi
    done
fi
if [[ -z "${ENTRY:-}" || ! -f "${ENTRY}" ]]; then
    echo "ERROR: no entry-point script found in $(pwd)" >&2
    echo "       checked: inference.py generate.py generate_mps.py app.py" >&2
    echo "       set ENTRY=<path> to override" >&2
    exit 1
fi

mkdir -p "${FIXTURES_DIR}"
rm -f "${CUDNN_LOG}"

# ---- TF32 disable (env layer) ----
# See https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

# ---- cuDNN algorithm logging ----
export CUDNN_LOGINFO_DBG=1
export CUDNN_LOGDEST_DBG="${CUDNN_LOG}"
# export CUDNN_LOGLEVEL_DBG=3   # uncomment for verbose

# ---- Python preamble (also set in Python in case env vars miss) ----
PYTHON_PRELUDE=$(cat <<'PYEOF'
import torch, os
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
print(f"[fp32-strict] allow_tf32={torch.backends.cuda.matmul.allow_tf32}/{torch.backends.cudnn.allow_tf32} "
      f"benchmark={torch.backends.cudnn.benchmark} deterministic={torch.backends.cudnn.deterministic}",
      flush=True)
print(f"[fp32-strict] CUDA capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'no cuda'}",
      flush=True)
PYEOF
)

# ---- Argument form depends on the entry point ----
# inference.py     : --image PATH ...
# generate_mps.py  : <positional IMAGE> ...
# generate.py      : likely --image too; try inference.py form first
ENTRY_NAME="$(basename "${ENTRY}")"
case "${ENTRY_NAME}" in
    inference.py|app.py|generate.py)
        ARGS_PY="['${ENTRY_NAME}', '--image', '${IMAGE}', '--seed', '${SEED}', '--fov', '${FOV}', '--output', '${OUTPUT}']"
        ;;
    generate_mps.py)
        ARGS_PY="['${ENTRY_NAME}', '${IMAGE}', '--seed', '${SEED}', '--fov', '${FOV}', '--output', '${OUTPUT}']"
        ;;
    *)
        ARGS_PY="['${ENTRY_NAME}', '${IMAGE}', '--seed', '${SEED}', '--fov', '${FOV}', '--output', '${OUTPUT}']"
        ;;
esac

# ---- Build runner ----
TMP_RUNNER=$(mktemp /tmp/fp32_strict_runner.XXXX.py)
trap "rm -f ${TMP_RUNNER}" EXIT
cat > "${TMP_RUNNER}" <<EOF
${PYTHON_PRELUDE}
import runpy, sys
sys.argv = ${ARGS_PY}
runpy.run_path("${ENTRY}", run_name="__main__")
EOF

echo "[fp32-strict] ENTRY=${ENTRY}"
echo "[fp32-strict] IMAGE=${IMAGE}"
echo "[fp32-strict] FIXTURES_DIR=${FIXTURES_DIR}"
echo "[fp32-strict] CUDNN_LOG=${CUDNN_LOG}"
echo "[fp32-strict] NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE}"
echo "[fp32-strict] sys.argv = ${ARGS_PY}"

PIXAL3D_DUMP_FIXTURES="${FIXTURES_DIR}" \
PIXAL3D_STOP_AFTER=01b_image_cond_shape_512 \
python "${TMP_RUNNER}"

# ---- Post-run summary of cuDNN algorithm picks ----
echo
echo "[fp32-strict] cuDNN algorithm picks summary (${CUDNN_LOG}):"
if [[ -f "${CUDNN_LOG}" ]]; then
    grep -E "algo|mathType|MATH_TYPE|TENSOR_OP|IMPLICIT_GEMM|WINOGRAD|FFT" \
        "${CUDNN_LOG}" | sort -u | head -50 || \
        echo "  (no algo/mathType lines found — check ${CUDNN_LOG} manually)"
else
    echo "  WARNING: ${CUDNN_LOG} not created — cuDNN logging may have failed"
fi

echo
echo "[fp32-strict] capture complete."
echo "  diff vs Mac:    python scripts/diff_naf_trace.py ${FIXTURES_DIR} <mac-fixtures-dir>"
echo "  diff vs old:    python scripts/diff_naf_trace.py ${FIXTURES_DIR} <old-cuda-tf32-fixtures-dir>"
