#!/usr/bin/env bash
# Run Pixal3D inference on CUDA with fixture capture enabled.
#
# Assumes:
#   - You're inside the `trellis2` conda env (or any env with TRELLIS.2 +
#     Pixal3D deps installed).
#   - Pixal3D's inference.py carries the fixture instrumentation (see
#     FIXTURE_CAPTURE.md).
#   - You're cd'd to the Pixal3D repo root.
#
# Output:
#   - <fixtures>/                : all .pt fixtures + 00_metadata.json + run.log
#   - <fixtures>.tar             : single tarball you can rsync back to your Mac
#   - <output>                   : the final GLB
#
# Usage:
#   ./scripts/run_cuda_capture.sh --image assets/images/0_img.png \
#                                 --output /workspace/out_cuda.glb \
#                                 --seed 42 \
#                                 --resolution 1024 \
#                                 --fixtures /workspace/fixtures_cuda

set -euo pipefail

# --- defaults ---
IMAGE=""
OUTPUT=""
SEED=42
RESOLUTION=1024
FIXTURES=""
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $0 --image PATH --output PATH --fixtures DIR [--seed N] [--resolution N] [-- ...extra inference.py args]

Required:
  --image PATH        input image (must exist on the box; rsync from Mac if needed)
  --output PATH       output GLB path (absolute recommended)
  --fixtures DIR      where to dump fixtures (also gets run.log + tarball)

Optional:
  --seed N            random seed (default: $SEED)
  --resolution N      pipeline resolution: 1024 or 1536 (default: $RESOLUTION)
  -- ...              everything after \`--\` is passed through to inference.py
                      (e.g. --fov 0.5, --low_vram for consumer GPUs)
EOF
}

# --- arg parse ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)      IMAGE="$2";      shift 2 ;;
        --output)     OUTPUT="$2";     shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        --fixtures)   FIXTURES="$2";   shift 2 ;;
        --)           shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help)    usage; exit 0 ;;
        *)            echo "Unknown arg: $1" >&2; usage; exit 1 ;;
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

mkdir -p "$FIXTURES"
mkdir -p "$(dirname "$OUTPUT")"

# --- env hygiene: ensure CUDA defaults, not the Mac sdpa workaround ---
# These are the upstream defaults; clear any inherited overrides so we
# capture the reference path.
unset ATTN_BACKEND SPARSE_ATTN_BACKEND SPARSE_CONV_BACKEND 2>/dev/null || true

export PIXAL3D_DUMP_FIXTURES="$FIXTURES"

# --- info banner ---
echo "============================================================"
echo "Pixal3D CUDA capture"
echo "============================================================"
echo "  image       : $IMAGE"
echo "  output      : $OUTPUT"
echo "  seed        : $SEED"
echo "  resolution  : $RESOLUTION"
echo "  fixtures    : $FIXTURES"
echo "  extra args  : ${EXTRA_ARGS[*]:-(none)}"
echo "  python      : $(command -v python)"
echo "  torch       : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '???')"
echo "  cuda avail  : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo '???')"
echo "  gpu         : $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo '???')"
echo "============================================================"
echo

# --- run ---
LOG="$FIXTURES/run.log"
START=$(date +%s)
set +e
python inference.py \
    --image "$IMAGE" \
    --output "$OUTPUT" \
    --seed "$SEED" \
    --resolution "$RESOLUTION" \
    "${EXTRA_ARGS[@]}" \
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

# --- summarise what was captured ---
echo
echo "Fixtures written to $FIXTURES:"
ls -lh "$FIXTURES" | sed 's/^/  /'

# --- tar it up for easy rsync back ---
TAR="${FIXTURES%.tar}.tar"
echo
echo "Tarballing -> $TAR"
tar -cf "$TAR" -C "$(dirname "$FIXTURES")" "$(basename "$FIXTURES")"
echo "  $(du -h "$TAR" | awk '{print $1}')  $TAR"

# Also include the GLB in the tar for convenience.
if [[ -f "$OUTPUT" ]]; then
    tar -rf "$TAR" -C "$(dirname "$OUTPUT")" "$(basename "$OUTPUT")"
    echo "  added GLB to tarball"
fi

echo
echo "Pull to Mac:"
echo "  rsync -avhP --info=progress2 user@<host>:${TAR} /Users/pawelma/code/ai/fixtures/"

exit $RC
