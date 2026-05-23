#!/usr/bin/env bash
#
# CUDA fp32-strict capture wrapper (Session 15 / §9-C-decider).
# Lives in Pixal3D_fresh; runs on the CUDA rental box from a Pixal3D_fresh
# checkout against inference.py.
#
# Disables TF32 in cuDNN + cuBLAS so the CUDA reference is true fp32
# (23-bit mantissa) instead of TF32 (10-bit mantissa).  Enables cuDNN
# algorithm logging so we can see what algos cuDNN picked per conv.
#
# Hypothesis: the 6.5e-3 max-abs gap between Mac fp32 conv (direct or
# Winograd F(4,3)) and the existing CUDA reference is entirely explained
# by TF32 — Ampere+ GPUs default to TF32 for fp32 convs.  If true, this
# capture with TF32 disabled should land within fp32 noise (≤ 1e-4) of
# the Mac output, AND the cuDNN log will show CUDNN_TENSOR_OP_MATH was
# the old default.
#
# Usage on the rental:
#   cd /workspace/Pixal3D_fresh
#   bash scripts/cuda_capture_fp32_strict.sh
#
# Env overrides:
#   IMAGE        — input image (default: assets/images/1_img.png)
#   FIXTURES_DIR — capture output dir (default: /tmp/cuda_naf_trace_01b_fp32strict)
#   CUDNN_LOG    — cuDNN log destination (default: /tmp/cudnn_algos.log)
#   SEED, FOV, OUTPUT — passed through to inference.py

set -euo pipefail

IMAGE="${IMAGE:-assets/images/1_img.png}"
FIXTURES_DIR="${FIXTURES_DIR:-/tmp/cuda_naf_trace_01b_fp32strict}"
CUDNN_LOG="${CUDNN_LOG:-/tmp/cudnn_algos.log}"
SEED="${SEED:-42}"
FOV="${FOV:-0.6061}"
OUTPUT="${OUTPUT:-/tmp/throwaway.glb}"

if [[ ! -f inference.py ]]; then
    echo "ERROR: inference.py not found in $(pwd)" >&2
    echo "       Run from a Pixal3D_fresh checkout root." >&2
    exit 1
fi

mkdir -p "${FIXTURES_DIR}"
rm -f "${CUDNN_LOG}"

# TF32 disable — env layer reads before PyTorch creates handles.
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

# cuDNN algorithm logging.
export CUDNN_LOGINFO_DBG=1
export CUDNN_LOGDEST_DBG="${CUDNN_LOG}"

# Build runner with Python-side belt-and-suspenders.
TMP_RUNNER=$(mktemp /tmp/fp32_strict_runner.XXXX.py)
trap "rm -f ${TMP_RUNNER}" EXIT
cat > "${TMP_RUNNER}" <<EOF
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

import runpy, sys
sys.argv = ["inference.py",
            "--image", "${IMAGE}",
            "--seed", "${SEED}",
            "--fov", "${FOV}",
            "--output", "${OUTPUT}"]
runpy.run_path("inference.py", run_name="__main__")
EOF

echo "[fp32-strict] IMAGE=${IMAGE}"
echo "[fp32-strict] FIXTURES_DIR=${FIXTURES_DIR}"
echo "[fp32-strict] CUDNN_LOG=${CUDNN_LOG}"

PIXAL3D_DUMP_FIXTURES="${FIXTURES_DIR}" \
PIXAL3D_STOP_AFTER=01b_image_cond_shape_512 \
python "${TMP_RUNNER}"

echo
echo "[fp32-strict] cuDNN algorithm picks summary (${CUDNN_LOG}):"
if [[ -f "${CUDNN_LOG}" ]]; then
    grep -E "algo|mathType|MATH_TYPE|TENSOR_OP|IMPLICIT_GEMM|WINOGRAD|FFT" \
        "${CUDNN_LOG}" | sort -u | head -50 || \
        echo "  (no algo/mathType lines found — check ${CUDNN_LOG} manually)"
else
    echo "  WARNING: ${CUDNN_LOG} not created"
fi

echo
echo "[fp32-strict] capture complete."
echo "  diff vs Mac:    python scripts/diff_naf_trace.py ${FIXTURES_DIR} <mac-fixtures-dir>"
echo "  diff vs old:    python scripts/diff_naf_trace.py ${FIXTURES_DIR} <old-cuda-tf32-fixtures-dir>"
