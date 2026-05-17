# Pipeline Fixture Capture

Instrumentation added to `inference.py` for CUDA-vs-MPS divergence analysis:
dump intermediate tensors at every pipeline stage boundary on both backends,
then diff the resulting `.pt` files stage-by-stage to localise where the two
backends first numerically diverge.

All behaviour is **gated on the `PIXAL3D_DUMP_FIXTURES` environment
variable** — set it to a directory path to enable capture, leave it unset
for a normal run.

## What gets dumped

Up to twelve fixtures per inference, written into the directory pointed to
by `PIXAL3D_DUMP_FIXTURES`:

| File | Source | Contents |
|---|---|---|
| `00_metadata.json` | run start | image SHA256 + size, all run args, seed, torch version, **device kind (`cuda` / `mps` / `cpu`)**, device name, OS platform, Python version, key env vars |
| `00_preprocessed_image.pt` | after `pipeline.preprocess_image` | PIL image as `{mode, size, pixels}` (numpy) |
| `01_camera_params.pt` | after MoGe or manual FOV | `{camera_angle_x, distance, mesh_scale}` |
| `02_sparse_structure.pt` | hook on `sample_sparse_structure` | sparse-structure sampler output |
| `03a_shape_slat[_callN].pt` | hook on `sample_shape_slat` | shape SLat sampler output; `_call1`, `_call2` etc. if called multiple times |
| `03b_shape_slat_cascade.pt` | hook on `sample_shape_slat_cascade` | cascade-variant shape SLat output |
| `04_shape_slat_decoded.pt` | hook on `decode_shape_slat` | decoded shape latent |
| `05_tex_slat.pt` | hook on `sample_tex_slat` | texture SLat sampler output |
| `06_tex_slat_decoded.pt` | hook on `decode_tex_slat` | decoded texture latent |
| `07_run_output.pt` | after `pipeline.run` | `{mesh, shape_slat, tex_slat, res}` — `mesh` flattened to `{vertices, faces, attrs, coords, ...}` |
| `08_to_glb_geometry.pt` | after `o_voxel.postprocess.to_glb` | trimesh geometry flattened |

All tensors are detached and moved to CPU before saving so the `.pt` files
are portable across devices.

## Usage

```bash
# CUDA box (when rented)
PIXAL3D_DUMP_FIXTURES=/tmp/fixtures_cuda python inference.py \
    --image assets/images/0_img.png \
    --output /tmp/out_cuda.glb \
    --seed 42

# Mac
PIXAL3D_DUMP_FIXTURES=/tmp/fixtures_mps python inference.py \
    --image assets/images/0_img.png \
    --output /tmp/out_mps.glb \
    --seed 42
```

Each run logs one line per fixture saved:

```
[fixture] metadata -> /tmp/fixtures_cuda/00_metadata.json
[fixture] 00_preprocessed_image -> /tmp/fixtures_cuda/00_preprocessed_image.pt (3.14 MB)
[fixture] 02_sparse_structure -> /tmp/fixtures_cuda/02_sparse_structure.pt (...)
...
```

## Diffing the two backends

Minimal comparison loop:

```python
import torch, glob, os, json

CUDA_DIR = '/tmp/fixtures_cuda'
MPS_DIR  = '/tmp/fixtures_mps'

# Sanity: same image + seed?
for d in (CUDA_DIR, MPS_DIR):
    with open(os.path.join(d, '00_metadata.json')) as f:
        m = json.load(f)
    print(f"{d}: seed={m['seed']}  image_sha={m['image_sha256'][:12]}  device={m['device_kind']}")

def max_abs_diff(a, b):
    """Compare two arbitrary fixture payloads, returning (path, max_abs_diff)."""
    out = []
    def walk(x, y, path):
        if torch.is_tensor(x) and torch.is_tensor(y):
            if x.shape != y.shape:
                out.append((path, f'shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}'))
            else:
                d = (x.float() - y.float()).abs().max().item()
                out.append((path, d))
        elif isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k), y.get(k), f'{path}.{k}')
        elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            for i, (xi, yi) in enumerate(zip(x, y)):
                walk(xi, yi, f'{path}[{i}]')
    walk(a, b, '')
    return out

for cuda_f in sorted(glob.glob(f'{CUDA_DIR}/*.pt')):
    name = os.path.basename(cuda_f)
    mps_f = os.path.join(MPS_DIR, name)
    if not os.path.exists(mps_f):
        print(f'{name}: MISSING on MPS')
        continue
    a = torch.load(cuda_f, weights_only=False)
    b = torch.load(mps_f, weights_only=False)
    diffs = max_abs_diff(a, b)
    print(f'\n=== {name} ===')
    for path, val in diffs:
        print(f'  {path:60s} {val}')
```

**The first stage that shows a non-trivial max-abs-diff is the bug
boundary.** Everything before it is reproducing CUDA correctly; everything
after it is consequence, not cause.

A "non-trivial" diff threshold depends on stage:
* Float-precision arithmetic in attention / convs: ~1e-4 to 1e-3 between
  bf16 CUDA and fp32 MPS is normal noise. Anything ≥ 1e-2 worth a look.
* Discrete outputs (mesh vertex counts, face counts): ANY diff is the
  divergence point.
* Sampler outputs after exhaustive identical RNG: noise-floor diffs are
  unavoidable but should be tiny.

## How the instrumentation works

The implementation lives in `inference.py` between the imports and the
`IMAGE_COND_CONFIGS` constant block, plus six call-sites inside
`run_inference()`. No upstream files modified — pipeline stage capture is
implemented by monkey-patching the pipeline instance after construction.

Three helpers do the work:

* `_fixture_to_cpu(x)` — recursively detaches and moves tensors to CPU.
  Also handles dicts, lists/tuples, and mesh-like objects (anything with
  `vertices` + `faces` attrs gets flattened to a `dict` so it can survive
  `torch.save`).
* `_dump_fixture(name, payload)` — writes one `.pt` per call, prints a
  `[fixture] ...` log line. No-op if `PIXAL3D_DUMP_FIXTURES` is unset.
* `_install_pipeline_fixture_hooks(pipeline, fixture_dir)` — wraps the
  six internal stage methods (`sample_sparse_structure`,
  `sample_shape_slat`, `sample_shape_slat_cascade`, `decode_shape_slat`,
  `sample_tex_slat`, `decode_tex_slat`) with a `functools.wraps` shim that
  dumps each method's return value. A per-method call counter appends
  `_callN` so duplicate calls (e.g. cascade re-running shape_slat) don't
  overwrite each other.

## Adding more fixtures

To capture a new stage:

* **Outer boundary (visible in `inference.py`):** add a
  `_dump_fixture("NN_name", payload)` line.
* **Inside the pipeline (a method you can't easily intercept from
  outside):** add the method name to the `stage_map` dict inside
  `_install_pipeline_fixture_hooks`. No other changes needed.
* **Inside the sampler loop (per-step latents during flow matching):**
  this would require monkey-patching the sampler's `step` method, not the
  outer sampler call. Not done currently — add when you need it.

## Notes

* **`weights_only=False`** is required when loading because the fixtures
  contain non-tensor objects (dicts, strings, etc.). For the same reason
  these fixtures should be treated as untrusted if shared across machines.
* **MPS-side caveat:** torch.save of MPS tensors works, but if you ever
  see a `RuntimeError: ...mps...` during dump, the fallback is to call
  `.cpu()` explicitly. The `_fixture_to_cpu` helper already does this.
* **File sizes:** the SLat fixtures are the biggest (per-voxel features at
  resolution³); expect 50–500 MB each at 1536 resolution. The metadata
  + image + camera fixtures are tiny.
* **Disk:** plan for ~5 GB of fixtures per CUDA run at 1536 resolution.
  Halve for 1024.

## Status

* Helpers + hooks: implemented, syntax-verified.
* Dry-run tested in isolation with stubbed `o_voxel` / `pixal3d.pipelines`
  imports — fixtures save correctly, `_call1` suffix works, metadata
  captures device kind / image SHA / args.
* End-to-end run: not yet executed (requires TRELLIS.2 deps installed in
  the fresh checkout).
