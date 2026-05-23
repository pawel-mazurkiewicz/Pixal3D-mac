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
# Usage on the CUDA rental box:
#   bash scripts/cuda_capture_fp32_strict.sh
#
# Outputs:
#   $FIXTURES_DIR — captured full-tensor .pt files (same layout as previous
#                   CUDA captures so existing diff scripts just work)
#   $CUDNN_LOG    — per-conv algo / mathType lines from cuDNN runtime
#
# Notes:
#   - Disabling TF32 ~doubles fp32 conv runtime on Ampere+.  Capture will be
#     slower than the prior reference.  Single image; should still finish in
#     a few minutes.
#   - allow_fp16_reduced_precision_reduction is left on its default (True).
#     If the gap doesn't close with just allow_tf32=False, try setting it to
#     False as a second-line debug.

set -euo pipefail

# ---- output paths ----
FIXTURES_DIR="${FIXTURES_DIR:-/tmp/cuda_naf_trace_01b_fp32strict}"
CUDNN_LOG="${CUDNN_LOG:-/tmp/cudnn_algos.log}"

mkdir -p "${FIXTURES_DIR}"
rm -f "${CUDNN_LOG}"

# ---- TF32 disable ----
# These env vars are read by PyTorch _before_ the cuDNN/cuBLAS handles are
# created, so they win over the Python defaults.  See:
#   https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
export NVIDIA_TF32_OVERRIDE=0          # cuBLAS, hardware-level
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
# Belt-and-suspenders: explicit Python override below as well.

# ---- cuDNN algorithm logging ----
# CUDNN_LOGINFO_DBG=1 enables informational logging; CUDNN_LOGDEST_DBG
# selects the destination ("stdout", "stderr", or a file path).
export CUDNN_LOGINFO_DBG=1
export CUDNN_LOGDEST_DBG="${CUDNN_LOG}"
# Optional verbosity:
# export CUDNN_LOGLEVEL_DBG=3          # 0=err, 1=warn, 2=info, 3=verbose

# ---- Python preamble (also set in Python in case env vars miss) ----
PYTHON_PRELUDE=$(cat <<'PYEOF'
import torch, os
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False           # disable heuristic algo search
torch.backends.cudnn.deterministic = True        # force deterministic algos
# Quick sanity print so the capture log shows the actual state.
print(f"[fp32-strict] allow_tf32={torch.backends.cuda.matmul.allow_tf32}/{torch.backends.cudnn.allow_tf32} "
      f"benchmark={torch.backends.cudnn.benchmark} deterministic={torch.backends.cudnn.deterministic}",
      flush=True)
print(f"[fp32-strict] CUDA capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'no cuda'}",
      flush=True)
PYEOF
)

# Inject prelude into a tiny wrapper that imports + runs generate_mps.
TMP_RUNNER=$(mktemp /tmp/fp32_strict_runner.XXXX.py)
trap "rm -f ${TMP_RUNNER}" EXIT
cat > "${TMP_RUNNER}" <<EOF
${PYTHON_PRELUDE}
import runpy, sys
sys.argv = ["generate_mps.py", "assets/images/1_img.png",
            "--seed", "42", "--fov", "0.6061",
            "--output", "/tmp/throwaway.glb"]
runpy.run_path("generate_mps.py", run_name="__main__")
EOF

echo "[fp32-strict] FIXTURES_DIR=${FIXTURES_DIR}"
echo "[fp32-strict] CUDNN_LOG=${CUDNN_LOG}"
echo "[fp32-strict] NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE}"

PIXAL3D_DUMP_FIXTURES="${FIXTURES_DIR}" \
PIXAL3D_STOP_AFTER=01b_image_cond_shape_512 \
python "${TMP_RUNNER}"

# ---- Post-run summary of cuDNN algorithm picks ----
echo
echo "[fp32-strict] cuDNN algorithm picks summary (${CUDNN_LOG}):"
if [[ -f "${CUDNN_LOG}" ]]; then
    # cuDNN log lines look like:
    #   I! cuDNN (v9101 80) function cudnnConvolutionForward() called: ...
    # Grep for algo/mathType selections.
    grep -E "algo|mathType|MATH_TYPE|TENSOR_OP|IMPLICIT_GEMM|WINOGRAD|FFT" \
        "${CUDNN_LOG}" | sort -u | head -50 || \
        echo "  (no algo/mathType lines found — check ${CUDNN_LOG} manually)"
else
    echo "  WARNING: ${CUDNN_LOG} not created — cuDNN logging may have failed"
fi

echo
echo "[fp32-strict] capture complete."
echo "  diff vs Mac:    .venv/bin/python scripts/diff_naf_trace.py ${FIXTURES_DIR} <mac-fixtures-dir>"
echo "  diff vs old:    .venv/bin/python scripts/diff_naf_trace.py ${FIXTURES_DIR} <old-cuda-tf32-fixtures-dir>"
