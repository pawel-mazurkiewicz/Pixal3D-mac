# Upstreaming the native-library fixes

Several fixes carried in `extern/` are genuine bugs in the underlying native
libraries, not Pixal3D-specific glue. They should be contributed back to their
upstreams. This document is the inventory + the PR workflow.

Each fix below is a self-contained commit touching only files under one
`extern/<pkg>/`, so it can be cherry-picked / replayed onto a clean upstream
checkout cleanly.

> **Commit hashes are branch-local.** The SHAs in the tables and commands below
> are the current `feat/apple-silicon-port` hashes; a future rebase will rewrite
> them. If one no longer resolves, find the commit by its subject line (the
> summaries below are unique) and substitute the new hash.

---

## Inventory of fixes to upstream

### `mtlbvh` → [pedronaugusto/mtlbvh](https://github.com/pedronaugusto/mtlbvh)

| Fix | Commit | Files | Summary |
|---|---|---|---|
| BVH traversal stack 24 → 64 | `0b057cf2` | `src/metal/bvh.metal` | `closest_triangle` (used by `unsigned_distance_kernel` and all ray/closest kernels) used a 24-deep `FixedStack` whose `push()` silently dropped entries past 24. The BVH is 4-ary (~`3*depth+1` max occupancy); a multi-million-triangle mesh has depth ~11–15 → needs ~34–46 slots, so whole subtrees were skipped and `unsigned_distance` over-estimated (returned >0 even at a mesh vertex). That corrupts the narrow-band UDF driving the dual-contour remesh → holey/fragmented mesh. Small meshes never overflowed, which hid it. |

**This is the highest-value upstream fix** — it was the root cause of the
long-standing "Metal remesh is broken" symptom (Bug E). Verification in the commit
message: at-vertex distance `0.002 → 0.000001`; remesh on the real fairy mesh
`8.03% → 0.00%` welded boundary. `tests/test_bvh.py` 25 passed.

### `mtlmesh` → [pedronaugusto/mtlmesh](https://github.com/pedronaugusto/mtlmesh)

| Fix | Commit | Files | Summary |
|---|---|---|---|
| Bound hash linear-probing loops + correct weak-CAS inserts | `13aae6c1` | `src/metal/hash.metal`, `src/metal/remesh.metal` | (1) Every linear-probe loop was an unbounded `while(true)`; a non-terminating probe wedges the GPU and can crash WindowServer / thermally lock the machine. Bounded every probe to ≤ N iterations (a correct table always resolves within N → behaviour unchanged) and guarded `hash32/hash64` against N==0. (2) MSL only has `atomic_compare_exchange_weak_explicit`, which can fail *spuriously* on an empty slot; the insert then advanced the probe, leaving a gap → lookups miss keys (~24% drop, non-deterministic). Fixed by retrying the SAME slot on spurious failure and advancing only on genuine occupancy (CUDA uses strong CAS). |
| `simplify.metal` volatile qualifier (Bug A) | `b21ac995` | `src/metal/simplify.metal` | The same physical buffer is bound as `device atomic_ulong*` to `propagate_cost_kernel` and `device const ulong*` to `collapse_edges_kernel`. Per MSL spec the compiler may cache stale reads on the bare-load side → the mutual-agreement check sees stale values → simultaneous collapses on shared-vertex edges → 92.6% non-manifold edges. Adding `volatile` to the reader declaration fixes it (`NME 92.6% → 5.15%`, CUDA 5.18%). |

> **Note on the `simplify.metal` fix:** this was originally developed against an
> earlier `mtlmesh` and re-applied on top of the vendored upstream import
> (`212079e`). If upstream has since touched `collapse_edges_kernel`, review the
> merge carefully — the only required change is the `volatile` qualifier on the
> `propagated_costs` reader parameter.

### `mtldiffrast` → [pedronaugusto/mtldiffrast](https://github.com/pedronaugusto/mtldiffrast)

| Fix | Commit | Files | Summary |
|---|---|---|---|
| Match nvdiffrast depth test (nearest z/w wins, keep-first ties) | `7d89a30c` | `src/metal/rasterize.metal`, `src/metal_rasterize.mm`, `tests/*` | The Metal rasterizer kept the *farthest* triangle on overlap (`GreaterEqual` + `clearDepth=0`), opposite to nvdiffrast's cudaraster (keeps smallest z/w via `atomicMin`). At the z=0 UV bake this made overlapping/seam texels resolve last-drawn-wins instead of keep-first, and clipped valid negative-NDC-z geometry. Fix: GL→Metal clip-z remap `(z+w)*0.5` in the vertex shader, recover NDC `2*depth-1` in the fragment shader, depth state `GreaterEqual → Less`, `clearDepth 0 → 1`, and the matching compute-fallback sentinel flip. The 3 affected tests had encoded the buggy behaviour and were corrected to the nvdiffrast convention. 66/66 tests pass. |

### `natten-mps` → [pawel-mazurkiewicz/natten-mps](https://github.com/pawel-mazurkiewicz/natten-mps) (our own)

| Change | Commit | Files | Summary |
|---|---|---|---|
| In-house Metal kernels for NAF attention (`compat.v020`) | `81bd9751` | `natten_mps/compat/v020.py`, `compat/__init__.py` | The community `natten-mps==0.3.0` fused `na2d` rejects NAF's asymmetric cross-attention (Q/K head_dim 64, V head_dim 256) → silent pure-PyTorch fallback. Added a `compat.v020` shim mirroring the community API but backed by our split kernels (`na2d_qk` → softmax → `na2d_av`) which handle asymmetric K/V natively. Validated ~1e-3 vs CUDA cutlass-fna golden (S11). **Behaviour change:** mesh shifts ~0.4% vs the PyTorch fallback (fp32 reduction-order difference). |
| Shim walk probes `__dict__`, not `getattr` | `466a3e55` | `natten_mps/_shim.py` | The `sys.modules` walk used `getattr(mod, "na2d")`, tripping transformers' lazy `_LazyModule.__getattr__` → ~180 alias-warning lines per run. Probe `mod.__dict__.get("na2d")` instead (genuine `from natten import na2d` bindings live in `__dict__`; lazy stubs resolve via `__getattr__`, so they're skipped silently). |

---

## Workflow

The `extern/` packages are plain in-tree copies (no `git subtree` linkage), so
upstreaming is done by replaying the fix commit(s) onto a fresh upstream clone
with `format-patch --relative`. Each fix above is a single commit touching only
files under one `extern/<pkg>/`, which keeps the PR minimal and reviewable.

```bash
# 1. Clone upstream fresh
git clone https://github.com/pedronaugusto/mtlbvh.git /tmp/mtlbvh-pr
cd /tmp/mtlbvh-pr
git checkout -b fix/bvh-traversal-stack

# 2. Produce the patch from this repo, stripping the extern/<pkg>/ prefix
git -C /Users/pawelma/code/ai/Pixal3D format-patch -1 0b057cf2 \
  --relative=extern/mtlbvh --stdout > /tmp/bvh.patch

# 3. Apply it onto the clean upstream tree
git am /tmp/bvh.patch        # or: git apply /tmp/bvh.patch && git commit

# 4. Build + run the package's own tests, then push and open the PR
#    (rebuild the .metallib — see below — then `pytest tests/`)
git push origin fix/bvh-traversal-stack
```

The key flag is **`--relative=extern/<pkg>`**, which rewrites the diff paths so
they apply at the upstream repo root instead of under `extern/<pkg>/`.

For `mtlmesh`, the two fixes (`13aae6c1`, `b21ac995`) are **not contiguous** in
history, so emit them as separate single-commit patches and `git am` both in the
order you want (oldest first):

```bash
git -C /Users/pawelma/code/ai/Pixal3D format-patch -1 b21ac995 --relative=extern/mtlmesh --stdout > /tmp/mtlmesh-1-volatile.patch
git -C /Users/pawelma/code/ai/Pixal3D format-patch -1 13aae6c1 --relative=extern/mtlmesh --stdout > /tmp/mtlmesh-2-hash-cas.patch
git am /tmp/mtlmesh-1-volatile.patch /tmp/mtlmesh-2-hash-cas.patch
```

### `o_voxel` (upstream is a subdirectory)

`extern/o_voxel` is a plain-file copy of the `o-voxel/` **subdirectory** of
[pedronaugusto/trellis2-apple](https://github.com/pedronaugusto/trellis2-apple)
@ `6055b86`. Because upstream lives in a subdir of a larger repo (not at the repo
root like the `mtl*` packages), `format-patch --relative` doesn't line up — to
upstream anything here, diff `extern/o_voxel` against that subdir in a fresh
`trellis2-apple` clone and apply the delta by hand. (Most Pixal3D-side `o_voxel`
behaviour lives in `pixal3d/utils/o_voxel_native_export.py`, not in the vendored
package — those are runtime monkey-patches and stay in this repo.)

## Rebuilding the Metal libraries after editing a shader

The native packages compile `.metal` → `.metallib` at install time (see each
package's `setup.py` `MetalBuildExt`). After editing any `.metal` file, recompile
by reinstalling editable:

```bash
.venv/bin/pip install -e extern/<pkg> --no-build-isolation --force-reinstall --no-deps
```

The underlying compile is just `xcrun metal` → `xcrun metallib`, e.g. for one
package:

```bash
cd extern/mtlmesh/src/metal
for mf in *.metal; do
  xcrun -sdk macosx metal -c "$mf" -o "${mf%.metal}.air" -std=metal4.0 -O2 \
    -D__HAVE_ATOMIC_ULONG__=1 -D__HAVE_ATOMIC_ULONG_MIN_MAX__=1 -I .
done
xcrun -sdk macosx metallib *.air -o ../cumesh.metallib && rm *.air
```

The `.metallib` is an untracked build artifact — never commit it; it's regenerated
on install.
