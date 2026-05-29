# Investigation probe patches

Diagnostics that previously lived as env-gated edits inside production / vendored
files have been extracted so those files stay pristine.

## Runtime monkeypatch scripts (preferred — clean seams)

| Probe | Script | Was in |
|---|---|---|
| `PIXAL3D_FP32_ATTN` (S21, ruled out §4E) | `scripts/probe_fp32_attn.py` → `install()` | `pixal3d/modules/attention/full_attn.py` |
| `PIXAL3D_SUBDIV_BIAS` (S19 dead lever §4B) + `PIXAL3D_DUMP_SUBDIV` (S20) | `scripts/probe_subdiv.py` → `install(bias, dump_dir)` | `pixal3d/models/sc_vaes/sparse_unet_vae.py` |

Install before the pipeline is built, e.g.:

```python
import scripts.probe_subdiv as p; p.install(dump_dir="/tmp/subdiv")
import generate_mps as gm; gm.load_runtime_deps(); ...
```

## Applyable patch (no clean monkeypatch seam)

`probe_bake.o_voxel.diff` — the `PIXAL3D_BAKE_PROBE` colour/bake probe (S24, §4D).
It reads ~8 locals deep inside `o_voxel.postprocess.to_glb` (`attr_volume`, `attrs`,
`mask`, `_bvh_dist`, `coords`, `aabb`, `voxel_size`, `grid_size`), so it can't be
monkeypatched without copying the whole function. `to_glb` lives in the **vendored
`extern/o_voxel/` subtree**, so the edit is kept out-of-tree as a patch to avoid
polluting the upstream-syncable copy.

Apply / revert when needed:

```bash
git apply scripts/patches/probe_bake.o_voxel.diff      # enable, then PIXAL3D_BAKE_PROBE=1
git apply -R scripts/patches/probe_bake.o_voxel.diff   # revert before a subtree pull
```
