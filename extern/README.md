# extern/ — Vendored Native Dependencies

Native Metal / MPS packages required by Pixal3D's Apple Silicon port, vendored
in-tree via `git subtree` so they ship with the repo and are buildable against
the in-tree Python 3.10 venv (`.venv-py310/`).

## Packages

| Path | Upstream | Notes |
|---|---|---|
| `mtlmesh/` | [pedronaugusto/mtlmesh](https://github.com/pedronaugusto/mtlmesh) | Provides `cumesh`. Carries a local patch on top of upstream: `src/metal/simplify.metal` adds `volatile` qualifier to `collapse_edges_kernel`'s `propagated_costs` parameter (Bug A fix, see `INVESTIGATION_FACTS.md` Section 2). |
| `mtlgemm/` | [pedronaugusto/mtlgemm](https://github.com/pedronaugusto/mtlgemm) | Provides `flex_gemm` — sparse Conv3D + grid_sample_3d Metal kernels. |
| `mtlbvh/` | [pedronaugusto/mtlbvh](https://github.com/pedronaugusto/mtlbvh) | Provides `mtlbvh` — BVH ray queries used by mesh export. |
| `mtldiffrast/` | [pedronaugusto/mtldiffrast](https://github.com/pedronaugusto/mtldiffrast) | Provides `mtldiffrast` — Metal differentiable rasterizer. |
| `natten-mps/` | [pawel-mazurkiewicz/natten-mps](https://github.com/pawel-mazurkiewicz/natten-mps) | Provides `natten_mps` — Metal port of NATTEN. In-house package, not the unrelated `ssmall256/natten-mps` on PyPI. |

## Why these are in-tree (not pip-installed)

Before the `feat/unify-py310` consolidation, these packages lived in a
separate Python 3.11 venv (`/Users/pawelma/code/ai/trellis-mac/.venv/`) and
were called from Pixal3D's 3.12 venv via subprocess (`pixal3d/utils/
o_voxel_native_export.py`). The split existed only because Pedro's wheels
were built for 3.11 and Pixal3D had drifted to 3.12 — neither version was
required.

Consolidating to a single 3.10 venv (matching upstream Pixal3D + the CUDA
reference) lets us import these packages in-process, with proper stack
traces across the boundary. Vendoring via subtree keeps the source self-
contained while preserving the ability to pull upstream fixes.

## Building

All packages install editable into `.venv-py310/`:

```bash
for pkg in mtlmesh mtlgemm mtlbvh mtldiffrast natten-mps; do
  .venv-py310/bin/pip install -e extern/$pkg --no-build-isolation
done
```

`--no-build-isolation` is required because some packages (notably mtlgemm)
have rpath / link-time quirks that PEP 517's isolated build environment
doesn't satisfy cleanly.

## Pulling upstream fixes

Subtree is set up to pull from upstream directly without named remotes:

```bash
git subtree pull --prefix=extern/mtlmesh \
  https://github.com/pedronaugusto/mtlmesh.git main --squash
```

Substitute the package path + upstream URL from the table above. The
mtlmesh `simplify.metal` local patch may need to be reapplied if upstream
touches the same kernel — review the merge carefully.

## Pushing local fixes upstream

```bash
git subtree push --prefix=extern/mtlmesh \
  git@github.com:pedronaugusto/mtlmesh.git <local-fix-branch>
```

This is the long-term path for upstreaming the `simplify.metal` volatile
fix to Pedro.
