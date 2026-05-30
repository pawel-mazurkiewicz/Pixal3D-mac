# natten-mps

A Metal/MPS backend for [NATTEN](https://github.com/SHI-Labs/NATTEN) —
Neighborhood Attention kernels on Apple Silicon.

**Status:** v0.1 (pre-alpha but validated). Split-path kernels (`na2d_qk`,
`na2d_av`) are correct and integrated. Public goal is a `metal-fna` backend
PR'd upstream to SHI-Labs/NATTEN once the fused `na2d` lands.

| Test                                  | Status        | Tolerance vs reference            |
|---------------------------------------|---------------|-----------------------------------|
| Smoke (compile, dispatch, MTLBuffer)  | 5/5 pass      | —                                 |
| Parity vs natten 0.17.5 CPU naive     | 12/12 pass    | max abs ≤ 1.5e-5 (fp32)           |
| CUDA cutlass-fna golden               | 2/2 pass      | ~1e-3 rel at NAF shape-512 call   |

## Why

`natten >= 0.21` ships CUDA-only backends (`cutlass-fna`, `hopper-fna`,
`blackwell-fna`) plus `flex-fna` which routes to
`torch.nn.attention.flex_attention` on CPU. **There is no MPS backend in
upstream natten today** ([SHI-Labs/NATTEN#102](https://github.com/SHI-Labs/NATTEN/issues/102)
was closed without implementation). Models that depend on natten — e.g.
[valeoai/NAF](https://github.com/valeoai/NAF) — therefore either fail on
Apple Silicon (NAF's hard-coded `backend="cutlass-fna"` raises) or fall
back to a CPU round-trip that's both slow and numerically distinct from
CUDA.

`natten-mps` provides two hand-written Metal Shading Language kernels:

- `na2d_qk` — windowed neighborhood Q·K dot product
- `na2d_av` — windowed weighted gather (attn · V)

…plus a Python dispatch shim that monkey-patches `natten.na2d` so MPS
tensors route to these kernels, while CPU/CUDA tensors fall through to
upstream natten unchanged.

A fused `na2d_fused.metal` (Flash-style online softmax, matching natten's
modern `na2d` semantics) is planned for v0.2.

## Requirements

- **Apple Silicon Mac** (M1 or later). Validated on M-series; M5 Neural
  Accelerator specialization is a future v0.3.
- **PyTorch ≥ 2.5** (uses `torch.mps.compile_shader` for runtime MSL JIT
  and `torch.library.custom_op` for the op binding). Tested on 2.6 and
  2.12.
- **natten ≥ 0.21** (the shim only patches `natten.na2d`; older versions
  expose legacy `na2d_qk`/`na2d_av` split symbols which we don't
  currently shim).
- **einops** (used by the dispatch shim for heads-last ↔ heads-second
  layout translation).

## Install

No PyPI release yet. Clone and editable-install:

```bash
git clone https://github.com/<TBD>/natten-mps
cd natten-mps
pip install -e .
```

Verify:
```bash
python -c "import natten_mps; print(natten_mps.__version__)"
# 0.1.0a0
```

## Usage

The minimal pattern: `import natten_mps` somewhere early in your process,
*before* any module imports `natten`. The package auto-installs the
dispatch shim on import (set `NATTEN_MPS_NO_AUTOINSTALL=1` to opt out).

```python
import natten_mps   # auto-installs the shim
import natten

# Now natten.na2d dispatches MPS tensors to our Metal kernels, while
# CPU/CUDA tensors fall through to upstream natten unchanged.
q = torch.randn(B, X, Y, H, D, device="mps")   # heads-last (modern natten layout)
k = torch.randn(B, X, Y, H, D, device="mps")
v = torch.randn(B, X, Y, H, D, device="mps")
out = natten.na2d(q, k, v, kernel_size=9, dilation=1, stride=1)
```

### Calling the ops directly

For finer-grained control (e.g. capturing post-softmax attention weights):

```python
import natten_mps  # registers torch.ops.natten_mps.*

# Heads-second layout: [B, H, X, Y, D]
attn_scores = torch.ops.natten_mps.na2d_qk(q, k, kernel_size=9, dilation=1)
attn_weights = (attn_scores * scale).softmax(dim=-1).to(v.dtype)
features = torch.ops.natten_mps.na2d_av(attn_weights, v, kernel_size=9, dilation=1)
```

### Integrating with a model that imports `from natten import na2d`

If your model does the local-import pattern:
```python
# mymodule.py
from natten import na2d                   # bound at import time
out = na2d(q, k, v, kernel_size=9, ...)   # later in forward()
```

…then **make sure `natten_mps` is imported before `mymodule`**. The shim
walks `sys.modules` to rebind already-imported callers' local `na2d`
references, but the cleanest pattern is shim-first.

### Diagnostics

| Env var                       | Effect                                              |
|-------------------------------|-----------------------------------------------------|
| `NATTEN_MPS_NO_AUTOINSTALL=1` | Don't auto-install the shim on import.              |
| `NATTEN_MPS_DISABLE=1`        | Disable the shim entirely (CPU/CUDA passthrough).   |
| `NATTEN_MPS_VERBOSE=1`        | Print every MPS dispatch with shape / kernel info.  |

At process exit, the shim always prints a one-line summary of dispatch
counts — useful for confirming the kernel actually fired in a real run.

## Constraints and known limitations

- **Symmetric kernels only.** `kernel_size` must be a scalar or
  `(k, k)`; rectangular `(kh, kw)` is not yet supported.
- **Symmetric dilation only.** Same restriction as `kernel_size`.
- **Square or near-square grids preferred.** The boundary "edge-shift-
  clamp" rule (matching natten 0.17.5 CPU naive) requires
  `X >= (K - 1) * dilation + 1` and similarly for Y; the kernel doesn't
  validate this and will index out of bounds if violated.
- **fp32 internally.** Attention scores accumulate in fp32 regardless of
  input dtype; AV output matches `v.dtype`.
- **D ≤ 256.** The `na2d_av` kernel uses a `constexpr MAX_D = 256` stack
  array per thread. Larger head dims need a re-tile or recompile with
  bumped `MAX_D`.
- **No causal masking.** `is_causal=True` will silently produce
  bidirectional output.
- **Split path only in v0.1.** The modern fused natten `na2d` is
  emulated as `na2d_qk → scale → softmax → na2d_av`, which materializes
  the full `[B, H, X, Y, K²]` attention tensor (memory cost). Fused
  v0.2 will use Flash-style online softmax.

## Provenance and validation

**Algorithm anchor:** [natten 0.17.5 CPU naive](https://github.com/SHI-Labs/NATTEN/blob/v0.17.5/csrc/include/natten/cpu/naive/pointwise_neighborhood_2d.hpp)
— BSD-3 / Apache-2.0 mix in upstream. Our Metal kernels mirror its
boundary handling (`get_window_start` closed form), reduction order
(fp32 accumulator), and indexing (`ki * K + kj` slot ordering).

**Parity validation** (`tests/test_parity.py`): random `(q, k, v)` at
small shapes diffed against a pure-torch reference (`tests/_reference.py`)
that re-implements the 0.17.5 semantics. Max abs ≤ 1.5e-5 fp32 across
12 configurations covering even/odd grids, asymmetric grids,
dilation ∈ {1, 2}, head counts {2, 4}, and D ∈ {64, 256}.

**CUDA golden validation** (`tests/test_cuda_golden.py`): real captured
tensors from a CUDA cutlass-fna run of NAF on RTX 6000 Ada at shape-512
production resolution `(B=1, H=4, X=Y=512, D=64/256, ks=9, dilation=16)`.
Drift vs cutlass-fna is at TF32-matmul scale (~1e-3 relative) — TF32 is
NVidia's default for fp32 matmul on Ada/Hopper. The Metal kernel itself
uses full fp32.

The fixture files (~3.2 GB extracted) live under `test_fixtures/` and are
gitignored.

### Measured numbers

**Parity vs pure-torch natten-0.17.5 reference** (`tests/test_parity.py`,
12 configurations: even/odd grids, asymmetric, dilation 1 and 2):

| op        | max abs diff      | dtype |
|-----------|-------------------|-------|
| `na2d_qk` | ≤ 1.5e-5 fp32     | fp32  |
| `na2d_av` | ≤ 1.8e-7 fp32     | fp32  |

The AV path is essentially exact because, given identical post-softmax
weights and v, the only thing that can drift is fp32 addition order
over `K² = 81` terms. The QK path has D=64 reduction order drift on top.

**CUDA golden vs cutlass-fna** (`tests/test_cuda_golden.py`, NAF
shape-512 capture, single call covering 262,144 query positions × 4
heads × 81 K/V taps):

| op        | max abs | mean abs | rel at max | mean rel | distribution                            |
|-----------|---------|----------|------------|----------|-----------------------------------------|
| `na2d_qk` | 0.298   | 0.105    | 9.9e-4     | < 5e-3   | mean gap 0.105 / std gap 0.043 vs CUDA  |
| `na2d_av` | 0.031   | 2.76e-4  | 1.2e-3     | 8.9e-4   | 0 elements outside allclose(5e-3, 1e-2) |

The QK output is pre-softmax dot products of D=64 vectors with values
O(2) — output magnitudes range up to **|cuda|_max ≈ 301**, so the 0.298
absolute max is 1e-3 relative. cutlass-fna on Ada uses TF32 (10-bit
mantissa) for the matmul; our Metal kernel uses full fp32. The drift is
exactly what TF32 vs fp32 produces on 64-element reductions of values
O(2). Distributionally the two outputs are equivalent: mean(metal)
≈ mean(cuda) to 0.07%, std(metal) ≈ std(cuda) to 0.07%.

The AV output is a weighted gather where the attention weights are
already fp32 inputs to the kernel, so reduction-order drift only enters
through the `K² = 81` summation per `v` channel. The result is
`allclose(metal, cuda, rtol=5e-3, atol=1e-2)` with zero outliers across
all 268 million output elements.

## Layout

```
natten_mps/
  __init__.py        # registers ops + shim on import
  _compile.py        # lazy torch.mps.compile_shader singleton
  _ops.py            # torch.library.custom_op + register_kernel("mps")
  _shim.py           # monkey-patch natten dispatch for MPS tensors
  shaders/
    shared.metal     # na_window_start (edge-shift-clamp closed form)
    na2d_qk.metal    # split path: Q·K
    na2d_av.metal    # split path: attn·V
tests/
  test_smoke.py        # ops compile and dispatch
  test_parity.py       # vs pure-torch reference at small shapes
  test_cuda_golden.py  # vs CUDA cutlass-fna at NAF production shapes
  _reference.py        # pure-torch port of natten 0.17.5 CPU naive
```

## Running the tests

```bash
pytest tests/test_smoke.py    # always runs (no fixtures needed)
pytest tests/test_parity.py   # always runs

# Golden test needs the CUDA capture extracted to test_fixtures/.
# See "Provenance" above for how the fixture was produced.
pytest tests/test_cuda_golden.py
```

## Contributing

Bug reports and PRs welcome, especially:

- M-series-specific perf optimizations (cooperative threadgroup tiling,
  MPP `tensor_ops::matmul2d` for M5+ Neural Accelerators)
- Fused `na2d` (Flash-style online softmax) — see `shaders/na2d_av.metal`
  for the gather kernel; the fused version needs `[Q_TILE, D]`
  threadgroup buffers for streaming softmax accumulators
- Rectangular `kernel_size` / `dilation` support
- 1D and 3D neighborhood attention (`na1d`, `na3d`) — same idiom

## License

MIT. See `LICENSE`.

Boundary-handling algorithm + indexing convention derived from natten
0.17.5 CPU naive (BSD-3 / Apache-2.0). Test fixtures captured from a
CUDA run of [valeoai/NAF](https://github.com/valeoai/NAF) via natten 0.21
`cutlass-fna` backend.
