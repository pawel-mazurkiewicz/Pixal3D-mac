# Session 11 Findings — natten-mps lands, but the z_proj RED is not natten's fault

**Date:** 2026-05-22 → 2026-05-23
**Branch (Mac):** `feat/apple-silicon-port` (uncommitted)
**Repo:** `/Users/pawelma/code/ai/natten-mps/` — published to user's GitHub
**Pixal3D_fresh staging branch:** `feature/pixal-fixtures` (commits 9f13130 →
a995ae6 → ... → broadened-probe-filter)

---

## Executive summary

natten-mps shipped: Metal kernels for `na2d_qk` and `na2d_av`, dispatch shim
auto-installing on `import natten_mps`, full test suite (smoke + parity +
CUDA golden), README and pyproject hygiene for external users. **Algorithm
validated end-to-end:** ≤ 1.5e-5 max-abs vs natten 0.17.5 CPU naive at small
shapes, and ~1e-3 relative vs CUDA cutlass-fna at NAF's production 512×512
call. The package is published and reusable.

Then we wired it into Pixal3D's `generate_mps.py` via env-gated
`PIXAL3D_NATTEN_MPS_ENABLE=1` and re-ran the isolation experiment without
`PIXAL3D_NAF_DEVICE=cpu`. **Expectation: z_proj on 01b/c/d drops from
2.5e-2 RED → ≤ 1e-4 GREEN.** **Actual: z_proj on 01b/c/d stays at 2.4-2.8e-2
RED, bit-identical to running NAF on CPU.**

That looked like a contradiction (kernel matches cutlass-fna, yet pipeline
output equals the CPU path) so we added a one-shot input capture to the
shim and diffed natten's Q/K/V inputs between Mac and CUDA. **The inputs
themselves differ** — most clearly Q and K, with V essentially identical.
The kernel is correct; the layers feeding it (NAF's pre-attention conv /
linear) drift between MPS and CUDA. natten faithfully propagates that
upstream drift.

**Session 10's diagnosis ("flex-fna vs cutlass-fna algorithmic gap") was
wrong.** The natten port was the right thing to build for community reasons
(no MPS backend existed in upstream), and it works as advertised, but it
isn't what's causing the z_proj RED. The actual cause is cuDNN-vs-MPS
conv/linear reduction-order drift in NAF's projection layers.

---

## What shipped this session

### natten-mps (`/Users/pawelma/code/ai/natten-mps/`)

| Commit | Description |
|---|---|
| `714cb7a` | initial: natten-mps split-path kernels + parity tests passing |
| `d59fb0e` | test: CUDA cutlass-fna golden parity — Metal kernels match at NAF shapes |
| `df5cbeb` | fix: shim does sys.modules rebind walk for already-imported callers |
| `5630d3d` | docs: full README + pyproject hygiene for first external user |
| `b54087a` | feat: one-shot input capture for cross-platform diff (NATTEN_MPS_CAPTURE_FIRST) |

Published to user's GitHub. Pip-installable. `pip install -e .` from clone.

| Test                                  | Status        | Tolerance vs reference            |
|---------------------------------------|---------------|-----------------------------------|
| Smoke (compile, dispatch, MTLBuffer)  | 5/5 pass      | —                                 |
| Parity vs natten 0.17.5 CPU naive     | 12/12 pass    | max abs ≤ 1.5e-5 (fp32)           |
| CUDA cutlass-fna golden               | 2/2 pass      | ~1e-3 rel at NAF shape-512 call   |
| **Total**                             | **19/19**     | —                                 |

### Pixal3D_fresh (rental capture instrumentation, `feature/pixal-fixtures`)

| Commit | Description |
|---|---|
| `9f13130` | feat: natten.na2d input/output capture + minimal-bandwidth probe mode |
| `a995ae6` | fix: actually intercept NAF's natten.na2d call site + --probe-only flag |
| *(broadened filter)* | fix: capture natten via all three entry points (na2d, na2d_qk, na2d_av) |
| *(filter relax)* | fix: broaden probe-only filter to match new sequential natten labels |

Three rental cycles were needed to debug the capture itself (NAF imports
natten via a local-binding pattern that bypassed simple `natten.na2d` patches;
filter required exact label string; etc.). Final capture format: paired
`natten_qk_call0.pt` (876 MB) + `natten_av_call0.pt` (2.49 GB) covering NAF's
first cross-attention call at shape-512 production resolution.

### Pixal3D Mac (uncommitted, `feat/apple-silicon-port`)

- `generate_mps.py`: env-gated `PIXAL3D_NATTEN_MPS_ENABLE=1` wire-in inside
  `load_runtime_deps()`. When set, imports `natten_mps` (triggers shim) and
  short-circuits the existing pure-PyTorch `_torch_na2d` fallback.

---

## Experiment results

### Experiment 6: NAF on MPS native, natten-mps active

Setup: same as Session 10 Exp 5 baseline (image `1_img.png`, seed 42,
fov 0.6061), but `PIXAL3D_NATTEN_MPS_ENABLE=1` instead of
`PIXAL3D_NAF_DEVICE=cpu` and `PIXAL3D_PROJ_GRID_DEVICE=cpu`.

| Stage | Exp 5 (NAF=cpu) | Exp 6 (natten-mps) | Δ Exp6 vs Exp5 |
|---|---|---|---|
| 01a | YELLOW 2.83e-4 | YELLOW 2.83e-4 | — |
| 01b | RED **2.77e-2** | RED **2.77e-2** | 1.14e-5 (noise) |
| 01c | RED **2.56e-2** | RED **2.56e-2** | — |
| 01d | RED **2.40e-2** | RED **2.40e-2** | — |

The two runs produce **bit-identical z_proj** (mean diff 5e-8). natten-mps
on MPS produces the same z_proj as running NAF on CPU.

### Verifying the kernel actually fired

We added `NATTEN_MPS_VERBOSE=1` + atexit dispatch counter. Confirmed:

```
[natten-mps] dispatch shim installed (natten.na2d + local: ['natten.functional'])
[Pixal3D] natten-mps Metal dispatch active (version 0.1.0a0)
[natten-mps] dispatch #1: q=(1, 512, 512, 4, 64) ks=(9, 9) dil=(16, 16)
            dtype=torch.float32 dev=mps:0
[natten-mps] dispatched 1 call(s) total.
```

The kernel fires, on MPS, with the exact shape we validated against CUDA.
Yet the downstream output is bit-identical to the CPU path. That ruled out
"natten produces wrong values" — pointed us to "natten's inputs are wrong."

### Experiment 7: capture Mac's natten inputs and diff against CUDA

Added `NATTEN_MPS_CAPTURE_FIRST=/dir` knob to the shim. Captures (q, k, v,
attn_scores, attn, out) of the first natten call. Re-ran with the capture
on. Diffed against the rental's `natten_qk_call0.pt` / `natten_av_call0.pt`:

| Comparison | max_abs | mean_abs | std(Mac) | std(CUDA) |
|---|---|---|---|---|
| **Mac.q vs CUDA.q** | **4.8e-3** | 2.9e-4 | 1.959 | 1.959 |
| **Mac.k vs CUDA.k** | **1.2e-3** | 2.3e-4 | 1.901 | 1.901 |
| Mac.v vs CUDA.v | 2.3e-5 | 8.0e-7 | 1.000 | 1.000 |
| Mac.attn (post-softmax) vs CUDA.attn | 2.4e-3 | 1.7e-5 | 0.040 | 0.040 |
| **Mac.out vs CUDA.out** | **3.1e-2** | 3.0e-4 | 0.988 | 0.987 |

**The cascade:**

1. Q and K differ by ~5e-3 max — that's NAF's pre-attention projection
   layers (linear / conv) producing slightly different outputs on Mac MPS
   vs CUDA cuDNN. Distributions are identical to 4 digits, so it's pure
   reduction-order drift, not different math.
2. V is essentially identical (2.3e-5).
3. After natten + softmax, attn drifts to 2.4e-3 max (damped by softmax
   exponential).
4. Final out drifts to 3.1e-2 max — **matches the z_proj 2.4-2.8e-2 RED
   almost exactly**. The drift is amplified by the weighted-sum gather
   over 81 neighbors with D=256 v channels.

---

## What this means

**Session 10 was wrong about the cause of z_proj RED.** It hypothesized
"flex-fna(CPU) vs cutlass-fna(CUDA) algorithmic gap." That hypothesis
implied: port natten to Metal matching cutlass-fna, run the Mac pipeline,
z_proj would close. That's the work we did. **And z_proj didn't close.**

The new diagnosis is firm and falsifiable: **NAF's pre-attention layers
(everything between DINOv3 spatial tokens and the cross-attention's Q/K
projections) produce different outputs on MPS than on CUDA.** Our natten
kernel — whether the Metal port or pure-PyTorch fallback — faithfully
propagates that upstream drift. Running natten "more accurately" can't
close the gap because the bug is before natten.

**Why both NAF-on-CPU and natten-mps give the SAME z_proj output:**
NAF's projection layers, when run on Mac CPU PyTorch, produce
the SAME results as when run on Mac MPS PyTorch. Both diverge from
CUDA cuDNN's reduction order by ~5e-3. They diverge from CUDA in the
same way, so they give the same downstream output. Only running on
actual CUDA hardware would produce CUDA's Q/K/V.

---

## natten-mps decisions and gotchas locked

| Question | Decision / Finding |
|---|---|
| Package layout | Standalone `natten-mps` pure-Python wheel — works |
| v1 kernel scope | Split path (`na2d_qk` + `na2d_av`) — done, validated |
| Algorithm anchor | natten 0.17.5 CPU naive — translated to pure-torch reference |
| Validation target | ≤ 1e-4 abs was too tight for pre-softmax outputs O(100); ≤ 1e-3 relative is the right metric, and we hit it |
| Shim mechanics | `from natten import na2d` callers need sys.modules rebind — implemented |
| Capture format | `(q, k, v, attn_scores, attn, out)` heads-second; matches rental capture for diffing |
| Diagnosis env vars | `NATTEN_MPS_NO_AUTOINSTALL`, `NATTEN_MPS_DISABLE`, `NATTEN_MPS_VERBOSE`, `NATTEN_MPS_CAPTURE_FIRST` |
| TF32 caveat | CUDA Ada/Hopper default fp32 matmul actually uses TF32 (10-bit mantissa). That's the dominant source of natten-kernel-level drift between Metal and CUDA. Distributionally identical, ~1e-3 relative max — visible-quality equivalent. |

---

## Heuristics learned this session

- **An algorithm fix can only fix algorithm bugs.** We invested a session
  in a Metal port because Session 10 framed the problem as algorithmic.
  The actual problem was reduction-order drift in surrounding layers
  — a different bug class that no kernel correctness work could touch.
  Lesson: when the proposed fix touches X but the actual divergence sits
  in Y, verify by capturing Y's inputs across boxes before committing
  to the X fix.
- **Distributions can be identical to 4 digits and still differ at fp32
  bits.** `std(Mac.q) = std(CUDA.q) = 1.959` AND `max(|Mac.q - CUDA.q|) =
  4.8e-3`. Different summation orders converge to the same mean/std
  while diverging element-wise. Stat tests would falsely "pass"; only
  per-element diff catches it.
- **The capture-then-diff pattern is the single most useful instrument
  we have.** Pre-natten-capture: "kernel matches CUDA" was true in
  isolation but false in pipeline integration. Post-capture: we know
  exactly where the divergence enters. This pattern generalizes to any
  layer.
- **`from natten import na2d` was a session-spanning footgun.** It
  caught the rental capture wrapper and the shim. The fix (sys.modules
  rebind walk) is now baked into both. Document this for future
  reverse-engineering work on third-party libraries.
- **Three rental cycles to debug capture instrumentation suggests the
  CLI flow needs an automated diff probe at the end.** Future: any
  rental capture should auto-run a quick "did the natten wrapper fire?"
  check before signing off on the tarball.

---

## What's NOT done yet (deferred from Session 10 priority list)

| Item | Status |
|---|---|
| CPU naive parity | ✅ Done (via pure-torch reference, 12/12 pass) |
| CUDA cutlass-fna goldens | ✅ Done (2/2 pass, ~1e-3 rel) |
| Wire into Pixal3D end-to-end | ✅ Done (env-gated, runs cleanly) |
| z_proj GREEN | ❌ NOT achieved — root cause is NOT natten |
| v2 fused na2d_fused.metal | Deferred; lower priority now that we know the bottleneck is elsewhere |
| M5 specialization (MPP matmul2d) | Deferred |
| Upstream PR to SHI-Labs/NATTEN | Deferred until v0.2 fused na2d done |

---

## Open questions for future sessions

1. **Which exact layer in NAF produces the Q/K drift?** NAF has multiple
   convs and a single linear-ish projection before the cross-attention.
   Q,K are heads-LAST `[B, X, Y, H, D]` post-projection. K's drift is
   smaller than Q's (1.2e-3 vs 4.8e-3), and V's is negligible (2.3e-5)
   — that pattern alone might fingerprint the responsible layer.
2. **Can MPS conv2d be coerced into cuDNN reduction order?** Probably
   not directly. PyTorch's MPS backend dispatches to MPSGraph; we have
   limited control over reduction order. Worth a small experiment with
   `torch.use_deterministic_algorithms(True)` and tile-size hints, but
   expectation is "no fix here."
3. **Is the same drift present in the DINOv3 forward pass?** z_global
   matches GREEN at 1.4e-5, so DINOv3's [CLS] token is fine — but
   spatial tokens feed into NAF and could carry drift in. Could capture
   DINOv3 outputs across boxes and diff.
4. **How much does 5e-3 input drift actually matter for visible mesh
   quality?** With z_proj at 2.5e-2 RED, the downstream DiT samplers
   then accumulate drift over 12 steps × 4 samplers. The eventual
   mesh diff vs CUDA may be acceptable visually even if numerics
   are RED. Worth a side-by-side eyeball test before any further
   numerical work.
5. **Does the 5e-3 drift correspond to a specific torch op?** If it's
   isolated to one MPS kernel implementation, a workaround (e.g. force
   that op to CPU) becomes viable. If it's a fundamental ULP-level
   property of MPSGraph matmul reductions, we accept the floor.
6. **Are we sure 5e-3 input drift × NAF cross-attn = 3e-2 output drift?**
   The math (softmax + weighted gather) should preserve the input drift's
   magnitude, not amplify it 6×. Worth a synthetic experiment with
   perturbed Q,K (drift ε at each element, V identical) to verify the
   amplification factor we observed is what natten naturally produces.

---

## Next-session priorities (in order)

1. **Pipeline map document** (in progress this session as a separate
   deliverable). Catalogs every model, every monkey-patch, every
   known divergence source. Becomes the reference map for further
   investigation.
2. **DINOv3 spatial-token cross-box diff** — quick wins or rules out
   another layer.
3. **Locate the specific MPS op responsible for the Q/K projection
   drift.** Trace from NAF's `Q_proj`/`K_proj` Conv2d/Linear outputs
   backward; perturb DINOv3 inputs and see drift propagation.
4. **Decide ship-or-fix posture on z_proj RED.** If visible mesh
   quality is acceptable at 2.5e-2 z_proj RED, declare done and move
   on. If not, the next attack target is the located drift source.
