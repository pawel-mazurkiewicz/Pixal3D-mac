<div align="center">

# Pixal3D — Apple Silicon (MPS / Metal) Port

**Unofficial fork that runs [Pixal3D](https://ldyang694.github.io/projects/pixal3d/) image-to-3D generation on Apple Silicon Macs — no CUDA required.**

</div>

<div align="center">
  <a href="https://ldyang694.github.io/projects/pixal3d/"><img src=https://img.shields.io/badge/Upstream%20Project-333399.svg?logo=googlehome height=22px></a>
  <a href="https://huggingface.co/TencentARC/Pixal3D"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href="https://arxiv.org/abs/2605.10922"><img src=https://img.shields.io/badge/Arxiv-b5212f.svg?logo=arxiv height=22px></a>
  <img src=https://img.shields.io/badge/Apple%20Silicon-MPS%20%2B%20Metal-000000.svg?logo=apple height=22px>
</div>

---

## What this is

This is a community port of **Pixal3D** (Pixel-Aligned 3D Generation from Images,
SIGGRAPH 2026, by Tsinghua University / Tencent ARC Lab / Victoria University of
Wellington) to **Apple Silicon**. The upstream pipeline targets NVIDIA GPUs and
depends on CUDA-only building blocks — `flash_attn`, `nvdiffrast`, fused
neighborhood attention (`natten` cutlass kernels), `flex_gemm` sparse 3D
convolutions, and the CUDA `o_voxel`/`cumesh` mesh post-processing kernels.

This fork replaces every one of those with an Apple-native equivalent so the full
single-image → textured GLB pipeline runs end-to-end on an M-series Mac using
**MPS** (Metal Performance Shaders) and a set of hand-written / vendored **Metal**
kernels. It is **not affiliated with or endorsed by** the original authors,
Tencent, or Tsinghua University.

> **Looking for the original?** The upstream README, model weights, online demo,
> and paper are linked in the badges above. If you have an NVIDIA GPU, use upstream
> — it is faster and is the reference implementation.

### How it came about

The port grew out of a long divergence-hunting investigation: get the CUDA
pipeline running on Metal, then chase down every place where Apple's fp32 / MPS
numerics or a missing kernel caused the mesh or texture to diverge from the CUDA
reference. Several genuine bugs were found and fixed along the way (an MPS fused
SDPA accuracy cliff above ~18k tokens, a Metal BVH traversal-stack overflow that
broke the remesh on large meshes, a rasterizer depth-test mismatch, and more).

The full investigation history is preserved outside this repo in
`../Pixal3D_investigation_docs/`. The two reference documents that remain in-tree
are:

- **[`PIPELINE_MAP.md`](PIPELINE_MAP.md)** — ground-truth atlas of every stage,
  model, monkey-patch, device-routing decision, env var, and known CUDA-vs-Mac
  divergence. Open this first when you need to understand the runtime.
- **[`INVESTIGATION_FACTS.md`](INVESTIGATION_FACTS.md)** — the facts-only ledger of
  bugs found, fixed, and ruled out.

## Status

The pipeline runs end-to-end and produces watertight, textured GLB meshes whose
geometry and PBR colour closely match the CUDA reference. Known residuals are
small and documented in `INVESTIGATION_FACTS.md` / `PIPELINE_MAP.md §6`
(intrinsic fp32-vs-TF32 reduction-order drift, minor thin-feature simplify holes).
Expect a single 1024-cascade generation to take several minutes on an M-series Mac.

## Requirements

- **Apple Silicon Mac** (M1 or newer). Intel Macs are not supported.
- **macOS** recent enough to provide the Metal toolchain (tested on macOS 14+).
- **Xcode Command Line Tools** — required. The vendored Metal packages compile
  `.metal` shaders to `.metallib` **at install time** using `xcrun metal`, so the
  Metal toolchain must be present *before* you run setup:

  ```bash
  xcode-select --install        # if not already installed
  xcrun --find metal            # should print a path, not an error
  ```

  (A full Xcode install also works and is required on some macOS versions for the
  Metal compiler.)
- **Python 3.10**, recommended via Homebrew: `brew install python@3.10`.
  3.10 matches the upstream Pixal3D / CUDA reference environment; `requirements-mac.txt`
  is pinned against it (e.g. `numpy<2.3`, which dropped 3.10 in 2.4+).
- Disk space for the model weights, which are pulled from Hugging Face
  (`TencentARC/Pixal3D`) on first run.

## Setup

### Option A — one-shot script (recommended)

```bash
./scripts/setup_mac.sh
```

This creates a `.venv` (Python 3.10), installs the pinned Mac dependencies, and
editable-installs the six vendored native packages in `extern/` (compiling their
Metal kernels). Re-run it any time; it is idempotent.

### Option B — manual

```bash
# 1. Create and populate the venv
/opt/homebrew/opt/python@3.10/bin/python3.10 -m venv .venv
.venv/bin/pip install -U pip wheel setuptools
.venv/bin/pip install -r requirements-mac.txt        # or: uv pip install --python .venv/bin/python -r requirements-mac.txt

# 2. Build + install the vendored native Metal packages (compiles .metal -> .metallib)
for pkg in mtlmesh mtlgemm mtlbvh mtldiffrast o_voxel natten-mps; do
  .venv/bin/pip install -e extern/$pkg --no-build-isolation
done
```

`--no-build-isolation` is required: some packages (notably `mtlgemm`) have
link-time / rpath quirks that PEP 517's isolated build environment doesn't satisfy.

See [`extern/README.md`](extern/README.md) for what each vendored package provides
and how it maps to the CUDA dependency it replaces.

## Usage

Generate a textured GLB from a single image:

```bash
.venv/bin/python generate_mps.py assets/images/0_img.png --output output_3d.glb
```

Common options:

| Flag | Default | Purpose |
|---|---|---|
| `--output PATH` | `output_3d` | Output GLB path / basename |
| `--seed N` | `42` | Random seed |
| `--pipeline-type` | `1024_cascade` | `1024_cascade` or `1536_cascade` |
| `--texture-size` | `2048` | Baked texture resolution (512–4096) |
| `--native-remesh` / `--no-native-remesh` | on | Narrow-band dual-contour remesh (watertight, matches CUDA). Off falls back to the simplify-sieve path |
| `--no-texture` | off | Geometry-only GLB |
| `--fov F` | auto (MoGe-2) | Override estimated camera FOV |

`generate_mps.py --help` lists the full set. Runtime behaviour is also controllable
through a large set of `PIXAL3D_*` environment variables (fp32 model casts, attention
backend, device pinning, the native cleanup chain, diagnostics) — these are
catalogued exhaustively in [`PIPELINE_MAP.md §7`](PIPELINE_MAP.md).

The production numerics recipe (matches the CUDA reference most closely) is already
the default; you do **not** need to set any env vars for a normal run.

## How it works (architecture)

The Mac pipeline is the upstream Pixal3D pipeline with CUDA building blocks swapped
for Apple-native ones. The native replacements are vendored in `extern/` and
imported in-process:

| `extern/` package | Replaces (CUDA) | Provides |
|---|---|---|
| `mtlmesh` (`cumesh`) | CUDA `cumesh` | Metal mesh post-processing: simplify (QEM), hole-fill, NME repair, dedup, unify, dual-contour remesh |
| `mtlgemm` (`flex_gemm`) | CUDA `flex_gemm` | Metal sparse Conv3D + `grid_sample_3d` |
| `mtlbvh` | — | Metal BVH ray/closest-triangle queries (drives the remesh UDF) |
| `mtldiffrast` | `nvdiffrast` | Metal differentiable rasterizer (texture bake) |
| `o_voxel` | CUDA `o_voxel` | Mesh extraction + texture-bake post-process (`to_glb`) |
| `natten-mps` | `natten` cutlass-fna | Metal neighborhood attention for the NAF upsampler |

On top of that, `generate_mps.py` installs a set of monkey-patches at startup
(attention shims, device routing, a pure-Python replacement for the CUDA-only FDG
hashmap, fp32 upcasts, the `naive` chunked-fp32 sparse-attention backend that works
around the MPS SDPA cliff). All of this is documented stage-by-stage in
`PIPELINE_MAP.md §3–§5`.

## Contributing fixes upstream

Several of the fixes in this fork are genuine bugs in the underlying native
libraries and should be sent back to their upstreams (Pedro Augusto's `mtl*`
packages and our own `natten-mps`). The per-package change inventory and the PR
workflow are written up in **[`extern/UPSTREAMING.md`](extern/UPSTREAMING.md)**.

## 🤗 Acknowledgements

This port stands on:

- **[Pixal3D](https://ldyang694.github.io/projects/pixal3d/)** — the original work
  by Dong-Yang Li, Wang Zhao, Yuxin Chen, Wenbo Hu, Meng-Hao Guo, Fang-Lue Zhang,
  Ying Shan, and Shi-Min Hu (Tsinghua University / Tencent ARC Lab / Victoria
  University of Wellington), built on
  [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) and
  [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2).
- **[Pedro Augusto](https://github.com/pedronaugusto)'s Metal libraries**
  ([mtlmesh](https://github.com/pedronaugusto/mtlmesh),
  [mtlgemm](https://github.com/pedronaugusto/mtlgemm),
  [mtlbvh](https://github.com/pedronaugusto/mtlbvh),
  [mtldiffrast](https://github.com/pedronaugusto/mtldiffrast),
  [trellis2-apple](https://github.com/pedronaugusto/trellis2-apple)) — the
  Apple-native mesh / GEMM / BVH / rasterizer kernels this port depends on.
- [TRELLIS](https://github.com/microsoft/TRELLIS),
  [TRELLIS.2](https://github.com/microsoft/TRELLIS.2), and
  [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2).

## 📄 Citation

If you use the model, please cite the original Pixal3D paper:

```bibtex
@article{li2026pixal3d,
    title={Pixal3D: Pixel-Aligned 3D Generation from Images},
    author={Li, Dong-Yang and Zhao, Wang and Chen, Yuxin and Hu, Wenbo and Guo, Meng-Hao and Zhang, Fang-Lue and Shan, Ying and Hu, Shi-Min},
    journal={arXiv preprint arXiv:2605.10922},
    year={2026}
}
```

## License

This fork inherits the upstream Pixal3D license (see [`LICENSE`](LICENSE)). The
vendored `extern/` packages carry their own licenses in their respective
subdirectories.
