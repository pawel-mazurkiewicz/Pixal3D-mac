# extern/ — Vendored Native Dependencies

Native Metal / MPS packages required by Pixal3D's Apple Silicon port, vendored
in-tree as plain files so they ship with the repo and are buildable against
the in-tree Python 3.10 venv (`.venv-py310/`).

> **No `git subtree` linkage.** These started as `git subtree --squash` imports,
> but the subtree merge commits don't survive a rebase (they corrupt by replaying
> the vendored content at the repo root), so the branch was flattened to plain
> in-tree copies. Upstream sync is therefore manual/diff-based — see *Pulling
> upstream fixes* and *Pushing local fixes upstream* below.

## Packages

| Path | Upstream | Notes |
|---|---|---|
| `mtlmesh/` | [pedronaugusto/mtlmesh](https://github.com/pedronaugusto/mtlmesh) | Provides `cumesh`. Carries a local patch on top of upstream: `src/metal/simplify.metal` adds `volatile` qualifier to `collapse_edges_kernel`'s `propagated_costs` parameter (Bug A fix, see `INVESTIGATION_FACTS.md` Section 2). |
| `mtlgemm/` | [pedronaugusto/mtlgemm](https://github.com/pedronaugusto/mtlgemm) | Provides `flex_gemm` — sparse Conv3D + grid_sample_3d Metal kernels. |
| `mtlbvh/` | [pedronaugusto/mtlbvh](https://github.com/pedronaugusto/mtlbvh) | Provides `mtlbvh` — BVH ray queries used by mesh export. |
| `mtldiffrast/` | [pedronaugusto/mtldiffrast](https://github.com/pedronaugusto/mtldiffrast) | Provides `mtldiffrast` — Metal differentiable rasterizer. |
| `natten-mps/` | [pawel-mazurkiewicz/natten-mps](https://github.com/pawel-mazurkiewicz/natten-mps) | Provides `natten_mps` — in-house Metal port of NATTEN. **Now the active NAF attention path** (installed editable from the extern loop; replaced the community `ssmall256/natten-mps==0.3.0`). Our `compat/v020.py` mirrors the community API that `generate_mps.py` imports, but its split kernels (`na2d_qk` + `na2d_av`) handle NAF's asymmetric K=64/V=256 — which the community fused `na2d` rejected, forcing a pure-PyTorch fallback. Validated ~1e-3 vs CUDA cutlass-fna golden (S11). Note: shifts the mesh ~0.4% vs the PyTorch-fallback baseline (fp32 reduction-order difference at the NAF→DiT boundary). |
| `o_voxel/` | [pedronaugusto/trellis2-apple](https://github.com/pedronaugusto/trellis2-apple) (subdir `o-voxel/`) @ commit 6055b86 | Provides `o_voxel` — post-process / mesh-extract / texture-bake pipeline (`o_voxel.postprocess.to_glb`). Pedro's Mac-adapted fork of `microsoft/TRELLIS.2`'s o-voxel subdir. Copied as plain files (it was never subtree-imported — git subtree cannot extract a single subdirectory of a remote repo cleanly). Manual diff-based sync from upstream. |

## Why these are in-tree (not pip-installed)

Before the `feat/unify-py310` consolidation, these packages lived in a
separate Python 3.11 venv (`/Users/pawelma/code/ai/trellis-mac/.venv/`) and
were called from Pixal3D's 3.12 venv via subprocess (`pixal3d/utils/
o_voxel_native_export.py`). The split existed only because Pedro's wheels
were built for 3.11 and Pixal3D had drifted to 3.12 — neither version was
required.

Consolidating to a single 3.10 venv (matching upstream Pixal3D + the CUDA
reference) lets us import these packages in-process, with proper stack
traces across the boundary. Keeping the source in-tree as plain files keeps
it self-contained; upstream fixes are synced manually (see below).

## Building

> **Requires the Xcode Metal toolchain.** `mtlmesh`, `mtlgemm`, `mtlbvh`, and
> `mtldiffrast` compile their `.metal` shaders to `.metallib` **at install time**
> via `xcrun metal` (see each package's `setup.py` `MetalBuildExt`). Install the
> Xcode Command Line Tools first (`xcode-select --install`) and confirm
> `xcrun --find metal` succeeds. The `.metallib` files are untracked build
> artifacts — never commit them; they regenerate on install.

The easiest path is the repo's `./scripts/setup_mac.sh`. To do it by hand, all
packages install editable into `.venv/`:

```bash
for pkg in mtlmesh mtlgemm mtlbvh mtldiffrast o_voxel natten-mps; do
  .venv/bin/pip install -e extern/$pkg --no-build-isolation
done
```

`natten-mps` is pure-Python (no Metal compile step), but installing it the
same way is harmless. It must come from `extern/`, not the community PyPI
`natten-mps==0.3.0` — see the table note.

`--no-build-isolation` is required because some packages (notably mtlgemm)
have rpath / link-time quirks that PEP 517's isolated build environment
doesn't satisfy cleanly.

## Pulling upstream fixes

There is no `git subtree` linkage anymore, so pull upstream changes with a
manual diff-based sync: clone the upstream repo, diff it against the in-tree
copy, and apply the wanted deltas by hand.

```bash
# example for mtlmesh — substitute path + URL from the table above
git clone --depth=1 https://github.com/pedronaugusto/mtlmesh.git /tmp/mtlmesh-upstream
diff -ru extern/mtlmesh /tmp/mtlmesh-upstream \
  ':!*.metallib' ':!build' ':!*.egg-info'   # review, then apply by hand
```

The mtlmesh `simplify.metal` local patch (and any other local patches noted in
the table) must be preserved if upstream touches the same code — review the
diff carefully before applying.

## Pushing local fixes upstream

Several fixes in these packages are genuine upstream bugs (the `mtlbvh`
traversal-stack overflow, the `mtlmesh` hash/CAS + `simplify.metal` volatile
fixes, the `mtldiffrast` depth-test fix) and should be sent back as PRs. The full
per-package change inventory and the PR workflow (replaying individual commits
onto a clean upstream clone with `format-patch --relative`) live in
**[`UPSTREAMING.md`](UPSTREAMING.md)**.

With no subtree linkage, `git subtree push` is no longer available — replay the
local fixes onto a clean upstream clone with `format-patch --relative`:

```bash
# export just the commits that touched extern/mtlmesh, stripping the prefix
git format-patch --relative=extern/mtlmesh -o /tmp/mtlmesh-patches \
  <first-fix>^..HEAD -- extern/mtlmesh
# then, in a clean clone of the upstream repo:
git -C /path/to/mtlmesh am /tmp/mtlmesh-patches/*.patch
```
