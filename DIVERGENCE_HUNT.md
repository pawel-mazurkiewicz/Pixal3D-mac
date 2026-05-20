# CUDA-vs-MPS Divergence Hunt

End-to-end workflow for finding **exactly which pipeline stage** the Apple
Silicon (MPS) path first diverges from the CUDA reference, using the
fixture-capture instrumentation built into `inference.py`.

This file documents the workflow.  The instrumentation itself is documented
in [`FIXTURE_CAPTURE.md`](FIXTURE_CAPTURE.md).

## Why this exists

Multiple previous sessions chased visual artefacts in the textured GLB by
trying to fix downstream symptoms (broken simplify, NME explosion, missing
faces).  Each fix peeled off one layer and revealed another.  The conclusion
in `PAMO_PORT_PLAN.md` Phase 6: **the bug is not where we keep looking — we
need to find the actual divergence point, not chase symptoms.**

Plan: run the *same image, same seed, same args* on CUDA and on MPS, capture
intermediate tensors at every pipeline stage boundary, and diff them stage
by stage.  The first stage with non-noise-floor diff **is** the bug
boundary.  Everything before it reproduces CUDA correctly; everything after
it is downstream consequence.

## Prerequisites

* A rental Linux box with an NVIDIA GPU (24 GB+ VRAM; H100 or A100 verified;
  consumer 4090 with `--low_vram` works for 1024 resolution).
* `ssh` access to the box.
* Mac with Apple Silicon (M1+; 24 GB+ unified memory recommended).
* This repo (`Pixal3D_fresh`) checked out on **both** machines with the same
  commit SHA.
* An input image present in `assets/images/` on both sides — same file, same
  bytes (the metadata SHA256 will confirm).
* HuggingFace token logged in on the Linux box (gated TRELLIS.2-4B, DINOv3,
  RMBG-2.0 weights).

## One-time setup — Linux box

```bash
# SSH in, use tmux so a disconnect doesn't kill multi-minute runs
ssh user@<rental-host>
tmux new -s pixal3d

# Install TRELLIS.2 base environment (per Pixal3D README step 1)
git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
cd TRELLIS.2
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
conda activate trellis2
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Clone Pixal3D *next to* TRELLIS.2 (its pixal3d package needs trellis2 importable)
cd ..
git clone https://github.com/TencentARC/Pixal3D.git
cd Pixal3D
pip install -r requirements.txt
pip install https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl

# Auth + gated-model access
hf auth login
# Approve in browser:
#   https://huggingface.co/microsoft/TRELLIS.2-4B
#   https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
#   https://huggingface.co/briaai/RMBG-2.0

# Copy the instrumented inference.py + scripts from your Mac.  Run this on
# the Mac, not the box:
#
#   scp inference.py scripts/run_cuda_capture.sh user@<host>:/workspace/Pixal3D/
#   scp scripts/diff_fixtures.py FIXTURE_CAPTURE.md DIVERGENCE_HUNT.md \
#       user@<host>:/workspace/Pixal3D/
```

## One-time setup — Mac

Already done if you've been working in `Pixal3D_fresh`:

* `inference.py` carries the fixture instrumentation (see `FIXTURE_CAPTURE.md`).
* `scripts/run_mps_capture.sh` and `scripts/diff_fixtures.py` are committed.
* Dependencies installed in `.venv`.

## The workflow (per-run)

### 1. CUDA capture on the Linux box

```bash
# In the tmux session on the rental box, inside the trellis2 conda env
cd /path/to/Pixal3D
./scripts/run_cuda_capture.sh \
    --image assets/images/0_img.png \
    --output /workspace/out_cuda.glb \
    --seed 42 \
    --resolution 1024 \
    --fixtures /workspace/fixtures_cuda
```

What this does:
* Clears any stale `SPARSE_ATTN_BACKEND` / `ATTN_BACKEND` so the run uses
  the CUDA defaults (flash_attn, flex_gemm).
* Sets `PIXAL3D_DUMP_FIXTURES`.
* Runs `inference.py`, tee-ing stdout to `<fixtures>/run.log`.
* Tars the fixture dir + log + output GLB into `<fixtures>.tar` so you can
  rsync one file back.

Expected wall-clock: ~5 min on H100, ~10 min on A100.

### 2. Pull fixtures back to the Mac

```bash
# On the Mac
rsync -avhP --info=progress2 \
    user@<rental-host>:/workspace/fixtures_cuda.tar \
    /Users/pawelma/code/ai/fixtures/
cd /Users/pawelma/code/ai/fixtures
tar -xf fixtures_cuda.tar
```

### 3. MPS capture on the Mac

```bash
cd /Users/pawelma/code/ai/Pixal3D_fresh
./scripts/run_mps_capture.sh \
    --image assets/images/0_img.png \
    --output /Users/pawelma/code/ai/fixtures/out_mps.glb \
    --seed 42 \
    --resolution 1024 \
    --fixtures /Users/pawelma/code/ai/fixtures/fixtures_mps
```

What this does:
* Sets `ATTN_BACKEND=sdpa`, `SPARSE_ATTN_BACKEND=sdpa` (Mac doesn't have flash_attn).
* Adds `--low_vram` automatically (Mac memory pressure).
* Sets `PIXAL3D_DUMP_FIXTURES`.
* Otherwise identical args to the CUDA run.

Expected wall-clock: ~10–15 min on M4 Pro, ~6–10 min on M5 Max.

### 4. Diff fixtures stage-by-stage

```bash
.venv/bin/python scripts/diff_fixtures.py \
    /Users/pawelma/code/ai/fixtures/fixtures_cuda \
    /Users/pawelma/code/ai/fixtures/fixtures_mps
```

Output is per-fixture, per-tensor diff stats:

```
== sanity ==
  fixtures_cuda:  device=cuda  seed=42  sha=6959e517ee4b  torch=2.6.0
  fixtures_mps:   device=mps   seed=42  sha=6959e517ee4b  torch=2.12.0

== 00_preprocessed_image.pt ==
  .pixels                                                          max=0.000e+00  mean=0.000e+00     GREEN
  .size                                                            equal                              GREEN

== 02_sparse_structure.pt ==
  ._repr                                                           equal                              GREEN
  .coords                                                          max=0.000e+00  mean=0.000e+00     GREEN
  .feats                                                           max=2.341e-04  mean=4.118e-05     GREEN

== 03a_shape_slat.pt ==
  .feats                                                           max=1.873e-01  mean=2.234e-03     RED
  ...
```

The first stage flagged **RED** is the divergence boundary.

Default thresholds (overridable):
* `< 1e-4`: GREEN (noise floor)
* `1e-4 – 1e-2`: YELLOW (likely OK but check context)
* `> 1e-2` or shape mismatch or missing fixture: RED

The script exits 0 if everything is GREEN/YELLOW, exits 1 if any RED.

### 5. Once you've found divergence

The divergence stage tells you what subsystem to investigate.  Common
suspects:

| First RED stage | Likely subsystem | Where to look |
|---|---|---|
| `00_preprocessed_image` | rembg/RMBG-2.0 device handling | `pixal3d/pipelines/rembg/BiRefNet.py` |
| `01_camera_params` | MoGe-2 on MPS | `inference.py:load_moge_model`, MoGe-2 itself |
| `02_sparse_structure` | Flow-matching sampler or sparse attention | `pixal3d/pipelines/samplers/`, `pixal3d/modules/sparse/attention/` |
| `03a/b_shape_slat` | Cascade-resolution shape DiT | `pixal3d/models/structured_latent_flow.py`, sparse conv backend |
| `04_shape_slat_decoded` | Shape VAE decoder | `pixal3d/models/sc_vaes/`, FDG mesh extract |
| `05_tex_slat` | Texture DiT | Same as shape but tex path |
| `06_tex_slat_decoded` | Texture VAE decoder + voxel attributes | Texture VAE, attr layout |
| `07_run_output` | Mesh extraction layer (`flexible_dual_grid_to_mesh`) | `pixal3d/utils/mesh_extract.py` (Mac fallback) |
| `08_to_glb_geometry` | o_voxel.postprocess.to_glb (cumesh + Metal stack) | Phase 6 territory; this is what we already chased |

To bisect further inside a divergent stage:

1. Add finer-grained hooks to `_install_pipeline_fixture_hooks` in
   `inference.py` — e.g. hook each sampler step instead of each sampler
   call.  The pattern generalises trivially; see the `stage_map` dict.
2. Re-run capture on both sides (you may be able to skip earlier stages by
   loading their fixtures and replaying just the suspect stage in isolation
   — see "Isolation runs" below).

### Isolation runs (skipping earlier stages)

Once you know stage N is the first divergence point, you can debug it in
isolation by **loading the CUDA-side fixture for stage N-1 and feeding it
into the MPS pipeline at stage N**.  No need to re-run earlier stages each
iteration.

The fixture-loading code is straightforward — `torch.load(...,
weights_only=False)` gives you the same dict/tensor structure that
`_dump_fixture` saw at the original call site.  Wrap the stage-N method to
accept the loaded tensor instead of computing from scratch.

This is dramatically faster than full reruns and lets you focus iteration
on the specific buggy subsystem.

## Resolution choice

Run `--resolution 1024` first.  It's faster, fixtures are smaller (~1–2 GB
total vs ~5 GB at 1536), and divergence will surface at any resolution.
Once 1024 is clean — or at least *understood* — do a confirmation pass at
1536 to verify the fix scales.

## Cost / time budget

Single rental session at 1024 resolution:

| Phase | Time |
|---|---|
| Linux box conda + TRELLIS.2 install (cold) | ~15–20 min |
| HF weight downloads (first run only, ~15 GB) | ~5–10 min (network-bound) |
| One CUDA capture run | ~5–10 min |
| Fixture rsync back to Mac | ~1–3 min |
| Mac MPS capture run | ~10–15 min |
| Diff + analysis | ~minutes |
| **Total first session** | **~45–60 min**, half of which is one-time setup |
| **Total per subsequent capture cycle** | **~20–30 min** |

If the rental is hourly billed: budget one hour for the first capture.
Subsequent captures are 10–15 min of GPU time each, so a second hour covers
3–4 cycles.

## Caveats

* **Same model weights**: if the CUDA box auto-downloads a different
  TRELLIS.2-4B revision than the Mac has cached, fixtures will diverge for
  silly reasons.  The metadata.json captures the torch version but not the
  HF model commit SHA.  If divergence shows up immediately at stage 02,
  check `~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/` on both
  sides for matching snapshots.
* **`--low_vram` numerical equivalence**: per upstream commit `619db88`,
  low_vram is supposed to be numerics-equivalent to the eager path (just
  loads per-stage instead of all-at-once).  If you want to confirm, do one
  Mac run **without** `--low_vram` and diff against the `--low_vram` run.
  Expected diff: zero.
* **MoGe-2 on MPS**: MoGe-2 is itself a CUDA-first model.  If MoGe-2's
  intrinsics estimation differs between backends, **all downstream stages
  diverge** because camera params feed into the shape generator.  If you
  see divergence starting at stage 01 and propagating, pass `--fov 0.5` to
  bypass MoGe entirely on both sides — that isolates the rest of the
  pipeline from MoGe-induced noise.
* **Sampler determinism**: even with the same `seed`, CUDA and MPS RNG
  streams differ.  Stage 02 onwards will have tiny diffs from the sampler
  noise alone.  These should be at the bf16 noise floor (~1e-4).  Larger
  diffs that grow downstream are real.

## File layout

```
Pixal3D_fresh/
├── inference.py                  # contains fixture instrumentation
├── FIXTURE_CAPTURE.md            # documents the instrumentation API
├── DIVERGENCE_HUNT.md            # this file — the workflow
└── scripts/
    ├── run_cuda_capture.sh       # Linux/CUDA driver
    ├── run_mps_capture.sh        # macOS/MPS driver
    └── diff_fixtures.py          # comparison utility
```
