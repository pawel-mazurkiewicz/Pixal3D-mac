# Pixal3D Apple Silicon Port — Investigation Facts

**Covers:** Sessions 9–22 (2026-05-22 → 2026-05-28)
**Branch:** `feat/apple-silicon-port` (old) / `feat/apple-silicon-port-v2` in `Pixal3D_fresh/` (current)
**Purpose:** Single-page reference for facts, ruled-out hypotheses, and hard numbers. Narrative lives in the session files.

---

## §1 — Pipeline Stage Map (brief)

| Stage | Name | Mac device |
|---|---|---|
| S0 | preprocess_image (RMBG-2 + crop) | MPS |
| S1 | camera params (MoGe-2 ViT-L) | MPS |
| S1a/b/c/d | NAF image conditioning (DINOv3 × 4) | MPS |
| S2 | sparse_structure DiT sample | MPS |
| S3a | shape_slat LR DiT (`shape_slat_flow_model_512`) | MPS |
| S3b | shape_slat HR cascade DiT (`shape_slat_flow_model_1024`) | MPS |
| S4 | shape_slat decode — `FlexiDualGridVaeDecoder` (FDG VAE) | MPS |
| S7 | raw mesh — `flexible_dual_grid_to_mesh` | CPU |
| S8 | cleanup + export — `o_voxel.postprocess.to_glb` (Pedro's Metal cumesh) | Metal subprocess |

Fixture names (`PIXAL3D_DUMP_FIXTURES`) map 1:1 to stage tags. CUDA reference fixtures at `/Users/pawelma/code/ai/fixtures/fixtures_cuda/`.

---

## §2 — Bugs Found and Fixed

### Bug A — `simplify.metal` type-mismatch UB (FIXED — Session 19)

**Location:** `/Users/pawelma/code/ai/mtlmesh/src/metal/simplify.metal`

**Root cause:** `propagated_costs` declared `device atomic_ulong*` in `propagate_cost_kernel` (writer) but `device const ulong*` in `collapse_edges_kernel` (reader). Per MSL spec this is UB — compiler caches stale reads → mutual-agreement check sees stale values → simultaneous collapses on shared-vertex edges → 92.6% NME explosion.

**Fix:** Change reader declaration to `device const volatile ulong*`. Forces fresh loads per access. One-line change.

**Verification:**

| Metric | Pre-fix | Post-fix | CUDA reference |
|---|---|---|---|
| NME vertex % | 92.6% | **5.15%** | 5.18% |
| Excess (>2-face) edges | 55% | **1.16%** | 1.35% |
| Manifold edges | 41% | **98.09%** | 98.07% |
| compute_charts result | 687,069 charts @ 1.31 f/c | **2,330 charts @ 120 f/c** | 1,731 @ 166 f/c |

**Build command:**
```bash
cd /Users/pawelma/code/ai/mtlmesh/src/metal
for mf in *.metal; do xcrun -sdk macosx metal -c "$mf" -o "${mf%.metal}.air" \
  -std=metal4.0 -O2 -D__HAVE_ATOMIC_ULONG__=1 -D__HAVE_ATOMIC_ULONG_MIN__=1 -I. ; done
xcrun -sdk macosx metallib *.air -o /Users/pawelma/code/ai/mtlmesh/src/cumesh.metallib
rm *.air
```

---

### Bug B — DiT bf16 activation casts (FIXED — Session 19)

**Location:** `generate_mps.py` — `PIXAL3D_FP32_MODELS` handler

**Root cause:** DiT classes (`sparse_structure_flow.py`, `structured_latent_flow.py`) call `manual_cast(h, self.dtype)` in forward. `self.dtype` resolves to bf16 (from checkpoint suffix) even when params are fp32 on disk. The pre-existing `PIXAL3D_FP32_MODELS` only called `convert_to_fp32()` (VAE API); DiTs use `convert_to(dtype)` — so DiTs were never actually upcasted.

**Fix:** Add fallback branch in `_install_pipeline_fixture_hooks`: if model lacks `convert_to_fp32` but has `convert_to`, call `model.convert_to(torch.float32)`. This sets `self.dtype = float32` so all subsequent `manual_cast` calls become no-ops.

**Effect on fairy house mesh (S19):**

| Metric | Mac bf16 | Mac fp32 | CUDA |
|---|---|---|---|
| NME vertex % | 31.98% | **22.28%** | 25.66% |
| Excess edges | 10.86% | **6.47%** | 5.14% |
| Manifold edges | 85.91% | **90.54%** | 89.33% |
| compute_charts | many | 32,284 @ 30.5 f/c | 39,042 @ 25.5 f/c |
| Wall time | 580s | **538s** | — |

---

### Bug C — mtldiffrast inverted depth test + z=0 tie-break (FIXED — Session 24)

Metal rasterizer kept the **farthest** triangle (`GreaterEqual` + `clearDepth=0`); nvdiffrast's
cudaraster keeps **smallest z/w** (`atomicMin` + `depth>oldDepth→zkill`, verified in
`TriangleSetup.inl:145`/`FineRaster.inl:154,353`). At the z=0 UV bake this gave last-drawn-wins
instead of keep-first (UV-seam speckle contributor); also clipped negative NDC-z. Fix
(`extern/mtldiffrast`): vertex z-remap `(z+w)*0.5`, fragment recovers `2*depth−1`, depth
`GreaterEqual`→`Less`, `clearDepth 0→1`, compute-fallback flipped; 4 test CPU-references corrected
to nvdiffrast convention. 66/66 tests pass; rebuilt in `.venv-py310`. Audits also confirmed
`interpolate`, `texture` (dormant on Mac path), `mtlbvh unsigned_distance` faithful. Not the §4E
colour cause; a real correctness fix + partial speckle mitigation.

### Bug D — MPS fused `scaled_dot_product_attention` wrong above ~18k–20k tokens (FIXED — Session 26)

**Root cause:** PyTorch's MPS fused `F.scaled_dot_product_attention` returns numerically
**wrong** output once the (key) sequence length crosses ~18k–20k tokens (mean-abs err
0.05–0.10, max >10 on an O(1) signal; fp32 too). The flow DiTs run sparse self/cross
attention over ~21,500 tokens (robocrab) → just past the cliff. The per-call error
compounds across the DiT's 30 residual blocks into the **S25 "variance collapse"** of the
sampled latents (HR shape slat std 4.99 vs CUDA 5.63; tex slat likewise) → geometry
size-shrinkage + colour desaturation. **This is the single mechanism behind S25's §4F.**

**Proof it's the MPS kernel, not code/precision/reduction-order:** identical fp32 call,
CPU 1.24629 == CUDA 1.24632 (and CUDA flash 1.24633), **MPS 1.07553** (lone outlier).
Forcing CUDA onto the same `sdpa` code path → 1.24632 (code correct). Per-op CPU-vs-MPS
trace: divergence isolated to `Sparse{MultiHead,Project}Attention` outputs (mean |r−1|
0.21/0.24) vs ≤0.013 for norms/RoPE/MLP. Self-contained repro (`scripts/mps_sdpa_repro.py`,
no Pixal3D): random q/k/v fp32, bit-exact through N=18000, breaks at N≥20000; manual
matmul+softmax on MPS is exact at every N ⇒ MPS matmul/softmax fine, only the fused kernel
wrong. torch 2.12.0 (also 2.6.0).

**Fix:** `pixal3d/modules/sparse/attention/full_attn.py` — the `naive` backend computes
attention as **chunked (2048-query) fp32 matmul→softmax→matmul** (chunking mandatory; the
full [B,H,N,N] score matrix is ~22 GB at N=21500 and itself fails on MPS).
`generate_mps.py` default flipped `SPARSE_ATTN_BACKEND` `sdpa→naive` (dense image-cond
attention stays `sdpa`). **Verification:** DiT forward MPS 1.0755→**1.2468** (=CUDA); full
HR shape latent std 4.99→**5.67** (CUDA 5.63); robocrab baked sat 0.139→**0.341**.
Upstream ticket: `MPS_SDPA_BUG_REPORT.md`. Likely affects all TRELLIS-family Apple-Silicon
ports using long-sequence sparse SDPA.

**Explains the per-object mystery:** sequence-length cliff — simple meshes (turtle) stay
under ~18–20k tokens and always rendered fine; complex meshes (fairy/robocrab, ~21.5k)
cross it. A step function, not randomness; why below-cliff tests kept "exonerating" Mac.

**Does NOT fix** thin-feature fragmentation/perforation — but that is **NOT** the §9-F
"sieve wall / inherent QEM WONTFIX" as S25/S26 framed it. **S27 supersedes that:** the
residual shredding is the broken **remesh** step (Bug E below), and it is fixable.

### Bug E — Mac skips bake-time remesh; Metal dual-contour remesh is broken (DIAGNOSED S27 — ROOT-CAUSED & FIXED S28, see Bug F)

**S28 correction:** the defect is NOT in cumesh's DC/hash kernels. The S27 8.78%/18,636
result reproduces on **CPU** tensors (not an MPS artifact), but the actual cause is one
layer down in **`mtlbvh`** (Bug F). With Bug F fixed the Metal remesh is watertight and
`--native-remesh` is now the Mac default. Verified: cumesh hash is 100% at 2M-entry scale
(CPU); Python topology is byte-identical to CUDA; DC kernel faithful. Dormant cumesh bug
noted: `metal_hash.mm` `tensor_to_buffer`+memcpy is incoherent for **MPS** tensors (CPU
500/500 vs MPS 0-occupied/racy) — never fires because production `to_glb` is CPU-tensor
(`extern/o_voxel/o_voxel/postprocess.py:136`).

**Symptom:** post-S26 Mac meshes are still "shredded" in high-detail regions (fairy
pumpkins/mushrooms, sword inlay, orc hairline) with a terrible atlas; CUDA outputs are
smooth, **watertight**, ~1.7k-chart. Coordinate-**welded** metrics (the correct ones —
indexed counts are UV-seam-polluted): CUDA `out.glb` boundary **0.0%**; Mac
`fairy_house_naive.glb` boundary **10.7%**, excess 6.4%.

**Root cause:** the CUDA reference pipeline always bakes with `to_glb(..., remesh=True)`
(`TRELLIS.2/app.py:503`, `example.py:43`, `README.md:162`) — its narrow-band
dual-contouring remesh **rebuilds a watertight manifold**. The Mac pipeline runs
`remesh=False` (`o_voxel_native_export.py:1314` passes `remesh=args.remesh`;
`--native-remesh` is opt-in/off) and falls back to the **simplify-sieve** branch
(`postprocess.py:215 if not remesh:`) → holey mesh → xatlas explodes to 30k–100k charts.
The reason remesh is off: the **Metal port of the remesh is broken** —
`cumesh.remeshing.remesh_narrow_band_dc` (`mtlmesh/src/metal/simple_dual_contour.metal` +
`svox2vert.metal`) outputs boundary **8.78% / 18,636 components** (largest body 77%)
instead of a watertight ~1-component manifold; the chain after it → **99,859 charts**.

**Proof the Mac decode is fine, only the bake is broken** (cross-bake, S27):
- The Metal `to_glb` **geometry** chain is faithful to CUDA — identical input mesh through
  Mac vs CUDA chains gives near-identical per-step NME/excess/components (at `repair_nme`
  CUDA shatters *more*: 232,348 vs Mac 195,909). So simplify/repair_nme/small_cc are NOT
  the bug (refutes the "hidden to_glb kernel" lead).
- Same Mac mesh → **CUDA** `to_glb(remesh=True)` → **welded boundary 0.0%**, 235 comps,
  largest 97.8% (rendered: clean, crisp). → CUDA `to_glb(remesh=False)` sieve → 7.99%.
  Watertightness comes entirely from remesh.

**Fix (next session):** port/repair the Metal DC remesh (compare to
`CuMesh/src/remesh/simple_dual_contour.cu` + `svox2vert.cu`; suspect Bug-A-class
atomic/volatile/memory-order or u32/u64 hash-cell mismatch in cell→vertex / dual-quad
emission), acceptance = ~0% welded boundary + ~1.7k charts on fairy, then flip Mac default
to `remesh=True`. Interim: route only the remesh step through a CPU/CUDA path.

### Bug F — mtlbvh closest-triangle BVH traversal stack overflow on large meshes (FIXED — Session 28)

**This is the real Bug-E root cause.** `mtlbvh.unsigned_distance` (the UDF that drives
the narrow-band DC remesh, and the texture-bake attribute sampling) **over-estimates
distance on large meshes**: queried exactly *at* a mesh vertex it returns mean 0.002 /
max 0.011 / 89% >1e-4 (must be ~0), and exceeds the nearest-vertex distance for 53% of
near-surface queries (physically impossible for a correct mesh distance). On the
1,280-face sphere it is correct (err 1.3e-4, 0% over-estimate) — the bug is
size-dependent.

**Root cause:** `unsigned_distance_kernel` → `closest_triangle`
(`extern/mtlbvh/src/metal/bvh.metal:254`) uses `FixedStack24` (`bvh.metal:177`) whose
`push` **silently drops** entries past 24 (`if (count < 24)`). The BVH is 4-ary and
pushes up to 4 children per pop, so max stack occupancy ≈ 3·tree_depth+1. The 8.6M-face
decoded mesh has BVH4 depth ~11–15 → needs ~34–46 slots → overflow → whole subtrees
skipped → the true nearest triangle missed → distance over-estimated → corrupt
narrow-band UDF → holey/fragmented remesh (8.03% boundary, 15,150 components on CPU).

**Fix:** `FixedStack24`→`FixedStack64` (`bvh.metal`; covers BVH4 depth ~21). Rebuild
`mtlbvh.metallib`. **Verification:** at-vertex distance 0.002→**0.000001** (0.2% >1e-4),
over-estimate 53%→**0%**; remesh output on the real fairy mesh 8.03%→**0.00%** welded
boundary (15,150→**565** comps); final baked GLB **0.06%** boundary / 353 comp / largest
0.974 (CUDA `out.glb` 0.0% / 235 / 0.978). Confetti → CUDA-quality. Likely also reduces
§4D texture speckle (bake samples attrs via the same BVH). See `SESSION_28_FINDINGS.md`.

### Bug G — cumesh hash kernels: unbounded GPU probe loops (machine-lockup) + weak-CAS holes (FIXED — Session 28)

**Lockup:** every linear-probing loop in `extern/mtlmesh/src/metal/{hash,remesh}.metal`
was an unbounded `while(true)`. On a GPU (host does `waitUntilCompleted`) any
non-terminating probe (full chain, garbage coords, `N==0`) wedges the GPU → WindowServer
crash → thermal lockup (hit once this session on the MPS remesh path). **Fix:** bounded
every probe loop to ≤`N` (a correct table always resolves within `N`, so behaviour is
unchanged) + `N==0` guards. **Weak-CAS:** MSL only has
`atomic_compare_exchange_weak_explicit`, which fails **spuriously** on empty slots; the
original advanced the probe on spurious failure → gap in the chain → lookups (which stop
at the first empty slot) miss the key. **Fix:** retry the SAME slot on spurious failure;
advance only on genuine occupancy (CUDA uses strong `atomicCAS`). Rebuild
`cumesh.metallib`. Robustness fixes; immaterial on the watertight CPU path but they
remove the lockup hazard and a real correctness gap.

### Bug H — GLB export dropped vertex normals → faceted shading (FIXED — Session 28)

Mac GLBs had **no `NORMAL` accessor** (CUDA `out.glb` has 787,900) → glTF viewers
flat-shade per-face → "chunky" look. `o_voxel.postprocess.to_glb` *does* compute smooth
normals (`postprocess.py:304,466-471`), but trimesh's exporter drops them unless
`include_normals=True` (verified: `None`→POSITION only; `True`→adds NORMAL). **Fix:**
`o_voxel_native_export.py:1365-1367` pass `include_normals=True`. Result: GLB
31.4→**41.8MB** (CUDA 44.4MB; 13MB gap → 2.6MB), renders smooth. Remaining 2.6MB = lower
texture variance (the §4F desaturation compresses smaller).

## §3 — Production Recipe

```bash
PIXAL3D_FP32_MODELS=sparse_structure_flow_model,sparse_structure_decoder,\
shape_slat_flow_model_512,shape_slat_flow_model_1024,tex_slat_flow_model_1024 \
  .venv/bin/python generate_mps.py assets/images/IMAGE.png --seed 42 --output output.glb
```

**DO NOT add** `shape_slat_decoder` (makes mesh slightly worse — S20 verified).
**DO NOT use** `--pipeline-type 1536_cascade` (catastrophic shatter: 147k connected components on M5 Max).
**DO NOT rely on** `PIXAL3D_SUBDIV_BIAS` (no useful effect — near-zero band < 2% of values).

**S28 update:** `--native-remesh` is now **default-on** (`--no-native-remesh` to opt out)
with project-back 0.9 — produces a watertight, CUDA-comparable mesh now that Bug F
(mtlbvh stack) is fixed. GLBs ship smooth vertex normals (Bug H). Rebuild both native
metallibs after a fresh checkout (they're untracked build artifacts):
`scripts/mtlmesh_build_metallib.sh`-equivalent for `extern/mtlmesh`, and recompile
`extern/mtlbvh/src/metal/bvh.metal`→`mtlbvh.metallib` (see Bug F).

**S26 update:** the real flow-DiT fix is `SPARSE_ATTN_BACKEND=naive` (chunked fp32
attention), now the Mac default — see Bug D. The `PIXAL3D_FP32_MODELS=*flow_model*` in the
recipe above does **not** fix the MPS SDPA bug (fp32 MPS SDPA still collapses, 1.0755 vs
CUDA 1.2463); it was a partial mitigation. With `naive` default, the flow models no longer
need fp32 for *this* reason (keep for any other precision concerns). Canonical env is
`.venv-py310` (the `.venv` 3.12 path here is stale per session-end notes).

---

## §4 — Hypotheses Tested → Ruled Out

### 4A — NAF/z_proj drift chase (Sessions 9–15)

Goal: close the `01b z_proj` drift (max_abs 2.77e-2 vs CUDA).

**S9 isolation experiment results** (`--fov 0.6061` + `PIXAL3D_NAF_DEVICE=cpu`):

| Stage | Original Δ z_proj | After both fixes | Drop |
|---|---|---|---|
| `01a` (no NAF) | RED 1.4e-2 | **YELLOW 2.8e-4** | ~50× |
| `01b` (with NAF) | RED 1.3e-1 | RED 2.8e-2 | ~5× |
| `01c` (with NAF) | RED 9.3e-2 | RED 2.6e-2 | ~4× |
| `01d` (with NAF) | RED 1.2e-1 | RED 2.4e-2 | ~5× |

**S11 natten input capture** — Q/K/V diff Mac vs CUDA at natten boundary:

| Tensor | max_abs | mean_abs | Verdict |
|---|---:|---:|---|
| Q | 4.8e-3 | 2.9e-4 | ❌ Already diverged before natten |
| K | 1.2e-3 | 2.3e-4 | ❌ Already diverged before natten |
| **V** | **2.3e-5** | **8.0e-7** | ✅ GREEN |
| natten output | 3.1e-2 | 3.0e-4 | — (propagated Q/K drift) |

| Hypothesis | Test | Evidence | Verdict |
|---|---|---|---|
| Metal flash attention / SDPA precision is the DiT bottleneck | Synthetic Q/K/V probe: MPS sdpa drift ~1e-6 relative vs fp64 truth — fp32 precision floor; all three `sdpa_kernel` backends bit-identical on MPS | ❌ Not the cause (S5/S7) |
| Unfused (naive) attention is better than SDPA on MPS | `ATTN_BACKEND=naive` test | Slightly WORSE, not better | ❌ Refuted (S5) |
| pixal3d-mlx framework switch fixes quality | End-to-end comparison | MLX port produces visibly worse output (more artifacts, worse textures, regressed bugs) | ❌ Not viable (S7) |
| DINOv3 backbone is wrong | z_global (DINOv3 output) cross-box diff | max Δ=1.4e-5 — essentially bit-identical | ✅ Innocent (S9) |
| FDG mesh extractor is wrong | Mac extractor on CUDA's `fdg_*` inputs | 0 vertex mismatches | ✅ Innocent (S8/S9) |
| MPS `F.grid_sample` HR is wrong | `PIXAL3D_PROJ_GRID_DEVICE=cpu` | <2% effect on z_proj (2.77e-2 → 2.56e-2) | ❌ Not the cause (S10) |
| natten algorithm (flex-fna vs cutlass-fna) is the gap | Built natten-mps Metal kernels (19/19 tests pass, ≤1.5e-5 vs CPU naive, ~1e-3 rel vs CUDA); wired into pipeline | z_proj unchanged — bit-identical to NAF-on-CPU path | ❌ Refuted (S11) — Q/K already diverged before natten |
| natten-mps is wrong | Capture Q/K/V inputs at natten boundary | Q: 4.8e-3, K: 1.2e-3 already diverged; V GREEN 2.3e-5 | ❌ Not the cause — upstream NAF layers drift (S11/S12) |
| NAF V/value path is wrong | Full trace — v_raw max 1.14e-5 | GREEN at every measurement | ❌ Not the cause (S12) |
| CoreML/ANE fixes conv drift | Per-conv swap (2 layers, 8 layers, whole-module vanilla fp16, whole-module hybrid fp32-GN) | z_proj stays 2.770e-2 ± 0.1% across ALL strategies | ❌ Retired (S14) |
| TF32 is the dominant Mac↔CUDA gap | Rental CUDA capture with `TORCH_CUDNN_ALLOW_TF32=0`, confirmed via `cudnn_algos.log` (32× DEFAULT_MATH, 0× TENSOR_OP_MATH) | Gap stayed at 2.70e-2 (was 2.77e-2 with TF32). TF32 explains ~1.1e-2 *within CUDA*, not the 2.7e-2 Mac gap | ❌ Refuted (S15) |
| Winograd F(4,3) algorithm mismatch | Built Metal Winograd F(4,3) kernel; diffed vs CUDA | Metal-Winograd: 6.5498e-3, Metal-Naive: 6.5269e-3 — identical to noise | ❌ Refuted (S15) |
| Custom Metal NAF image_encoder fixes z_proj | Built & wired 4 Metal kernels (Conv2d 3×3 reflect, Conv2d 1×1, GroupNorm 8, SiLU); 66 modules swapped | z_proj max 1.92e-4 vs MPSGraph baseline — correct, but mesh quality unchanged | ❌ Doesn't fix mesh (S15) — port ships as `PIXAL3D_NAF_METAL=1` opt-in |

**Why Mac CPU and Mac MPS give the same z_proj (S11 key insight):** NAF's projection layers on Mac CPU PyTorch and Mac MPS PyTorch produce the SAME outputs — both differ from CUDA cuDNN by ~5e-3 (same reduction-order divergence). Swapping natten for any Mac-side implementation can't help because the bug is in the layers feeding natten.

**Final conclusion on z_proj:** The 2.7e-2 gap is intrinsic fp32 framework difference (cuDNN DEFAULT_MATH vs MPSGraph). No Mac-side numerical fix can close it. **And it doesn't matter** — bisection (Session 15 Mesh #11: inject CUDA image_cond, mesh unchanged) proved z_proj drift does not cause the visible mesh damage.

**Key z_proj table (Session 12 forward-order NAF drift):**

| Boundary | max_abs | mean_abs |
|---|---:|---:|
| `image_encoder.encoder.0` (input) | 0 | 0 |
| `image_encoder.sem_encoder.1.conv1` | 6.56e-3 | 2.37e-4 |
| `image_encoder.sem_encoder.1.conv2` | 1.16e-2 | 2.65e-4 |
| `upsampler.forward.q_raw` | 2.29e-3 | 2.82e-4 |
| `upsampler.forward.v_raw` | **1.14e-5** | **7.98e-7** (GREEN) |
| `upsampler.forward.output` | 2.20e-2 | 2.97e-4 |
| `01b z_proj` | **2.77e-2** | 1.51e-4 |

**ANE per-conv table (Session 13 `sem_encoder.1.conv1`):**

| Backend | max_abs vs CUDA | mean_abs |
|---|---:|---:|
| torch MPS fp32 | 6.56e-3 | 2.37e-4 |
| torch CPU fp32 | 6.53e-3 | 2.37e-4 |
| CoreML ALL fp32 | 6.56e-3 | 2.37e-4 |
| **CoreML ANE fp16** | **1.08e-3** | **1.02e-5** (23× better) |
| CoreML CPU+NE fp16 | 2.72e-2 | 4.23e-4 (WORSE) |

Result: ANE wins locally but doesn't propagate to z_proj (S14).

---

### 4B — Cleanup chain hypotheses (Sessions 15–22)

| Hypothesis | Test | Evidence | Verdict |
|---|---|---|---|
| Session 7 "cleanup exoneration" was valid | Fed CUDA stage-07 (PRE-cleanup, 4.19M V) into Mac cleanup | Same sieve output as Mac's own FDG input → S7 test was on stage-08 (already cleaned) | ❌ Anti-pattern — never test cleanup on already-cleaned input |
| Missing `merge_vertices()` step causes the sieve | Blender Merge-By-Distance @ 1e-5: V 993k→571k (-42.5%), NME edges -75.7% | Mesh still visually broken after weld | ❌ Necessary but not sufficient (S15) |
| Vertex weld (cKDTree, eps=1e-5/1e-4) in Python | Pre-cleanup weld on Mac fairy raw | Only 19 (eps 1e-5) / 8706 (eps 1e-4) pairs found in 3.94M verts | ❌ No effect (S22) |
| `PIXAL3D_REPAIR_NME=pamo` fixes splits | Compare stage-04 V after metal vs pamo | metal: 3,197,312 V; pamo: 3,196,862 V (Δ=450, 0.014%) | ❌ Functionally equivalent (S16) |
| `fill_holes` perimeter tuning fixes visible holes | Sweep 3e-2 → 1e-1 → 1e0 | perimeter=1e0: boundary 6.7%→1.1% (metric win) but visible fan-triangulation "sunbursts" at doorways | ❌ Metric win, visual disaster (S22) |
| `SMALL_CC=noop` + `FILL_HOLES=3e-1` combo | Best env-var matrix run | B_true=46,689 (-74% vs default), looks worse due to fan patches | ❌ No acceptable combo found (S16 8-variant sweep) |
| `fast_simplification` as simplify replacement | End-to-end test | 78.2% NME (vs Pedro's 92.6%), blocky/spiky geometry, 342k charts (still terrazzo) | ❌ Worse geometry; partial chart-count win not worth it (S18) |
| xatlas single-call bypass (whole mesh, no charts) | `xatlas.parametrize` on ~800k-face mesh | Stuck at 64%, 22 GB RAM / 335 GB reported; never finished | ❌ Doesn't scale (S15, re-confirmed S18) |
| pymeshlab as simplify replacement | Full pipeline test | Never finishes on 4M-vert input | ❌ Ruled out (S15, S17, S18) |
| NME-edge guard in `get_edge_collapse_cost_kernel` | Patch applied + measured | NME 41% → 61% after simplify_3x (worse) | ❌ Made it worse; reverted (S22) |
| `memory_order_seq_cst` on atomic_min | Patch attempted | MSL compile error: `atomic_min_explicit` on `ulong` only accepts `relaxed` | ❌ Impossible (S22) |
| Pedro's `compute_charts` port is broken | CUDA compute_charts on Mac-simplified mesh | Mac: 687,069 charts; CUDA on same input: 557,532 charts (23% drift, same blowup) | ❌ Input mesh is the cause, not the port (S18) |
| Raw mesh (no simplify) has the same problem | `--native-skip-simplify`: 10.8M-face mesh → compute_charts | 1,230,098 charts @ **8.78 f/c** (good local coherence) | ✅ Raw mesh is fine; simplify destroys chart-ability |
| Pedro's Metal simplify is algorithmically wrong vs CUDA | Line-by-line comparison of simplify.metal vs simplify.cu | Algorithmically faithful; 358-line gap is CUDA boilerplate in .mm, not missing kernels | ❌ Not algorithmically wrong — numerically diverges (S17, S22) |

**Per-stage B_true trace (CUDA stage-07 input → Mac cleanup, from S16):**

| Stage | B_true (real coord-merged boundary edges) |
|---|---:|
| Input (CUDA stage-07 raw FDG, V=4.19M) | 420,575 |
| fill_holes (initial) | 420,549 |
| **simplify_3x** | 240,796 (−180K — closes holes ✅) |
| dedup | 242,566 |
| **repair_nme** | 242,566 (ΔB=0 — paired splits are bit-exact ✅) |
| **small_cc(1e-5)** | 239,133 (−1.06M faces over-culled ❌) |
| fill_holes(3e-2) | 194,556 (poor: 177K new faces / only 44K holes closed) |
| **simplify_target** | 228,895 (+34K — simplify ADDS holes ❌) |
| cleanup_2 | 178,709 (final) |

**CUDA produces B_true = 0.** Mac best achievable: ~47K (env-var combo), visually worse.

**Stage-by-stage Mac fairy raw (S22 baseline):**

| Stage | V | F | boundary% | NME% |
|---|---:|---:|---:|---:|
| input | 3,940,419 | 8,586,890 | 2.99 | 22.27 |
| fill_holes | 3,949,705 | 8,644,932 | 2.52 | 20.80 |
| simplify_3x | 1,153,275 | 2,922,752 | 4.37 | **41.43** ← simplify creates NMEs |
| repair_nme | 2,909,153 | 2,903,959 | 40.48 | **75.98** ← 2.5× V split, B explodes |
| fill_holes 2 | 3,276,406 | 4,660,704 | 6.23 | 13.72 (fan-fills splits) |
| simplify_1x | 1,352,317 | 928,591 | 9.46 | 11.20 |
| final | 624,244 | 985,464 | 6.72 | 16.48 |

**Root cause per CuMesh #28 (WONTFIX):** parallel QEM edge-collapse overshots on dense thin-feature topology. Acknowledged by upstream author. CUDA exhibits milder version; Mac amplifies it. Not a Mac-specific bug.

---

### 4C — Upstream DiT / decoder divergence chase (Sessions 19–22)

| Hypothesis | Test | Evidence | Verdict |
|---|---|---|---|
| Mac FDG extractor (`flexible_dual_grid_to_mesh`) is broken | Run Mac extractor on CUDA FDG fields | excess 5.16% (CUDA: 5.14%) → bit-parity | ✅ Extract is correct (S21) |
| Mac cleanup chain order differs from CUDA | Read canonical `o_voxel.postprocess.to_glb` source | Identical to Mac's `_run_geometry_chain` | ✅ Correct (S21) |
| Metal cleanup kernels produce different output than CUDA | Replay Mac chain on Mac raw mesh on CUDA hardware | Same 21% boundary → kernels are bit-parity | ✅ Kernels correct (S21) |
| Mac LR DiT (`shape_slat_flow_model_512`) is wrong | `PIXAL3D_CPU_MODELS=shape_slat_flow_model_512`: run CPU fp32, compare to MPS fp32 | coord IoU **1.000000**, feat mean \|Δ\| **3.46e-5** (vs Mac↔CUDA \|Δ\| mean 0.5) | ✅ Mac DiT is fp32-correct (S22) |
| 03b feats diverge in magnitude Mac vs CUDA | Probe saved fixtures: \|feats\| distribution | Mac: mean 0.7746 max 4.402; CUDA: mean 0.7748 max 4.442 — **statistically identical** | ❌ S21's divergence metric was noise, not magnitude shift (S22) |
| `mac_fairy_with_cuda03b.glb` validated the inject path | MD5 check | MD5-identical to pure Mac baseline — three filenames for one file | ❌ Injection test was broken; inject conclusion was wrong (S22) |
| `SPARSE_CONV_BACKEND=none` (pure PyTorch) differs from flex_gemm on Mac | `torch.equal` comparison of 03a outputs | **`torch.equal == True`, max \|Δ\| = 0.0** — bit-identical | ❌ flex_gemm / Triton-Metal sparse Conv3D NOT the divergence source (S21) |
| `PIXAL3D_FP32_MODELS` including `shape_slat_decoder` helps | End-to-end test | 873k V vs 889k V (worse); decoder native fp16 is already optimal | ❌ Dead (S20) |
| `--pipeline-type 1536_cascade` improves quality | Mac 1536_cascade test | 147,925 connected components, top component 321 V — catastrophic shatter | ❌ Mac locked to 1024_cascade (S20) |
| `PIXAL3D_SUBDIV_BIAS` shifts borderline voxels | Cascade L0 subdiv logit distribution | Near-zero band (-0.5, 0] < 2% of values; dropped cells have Mac logits mean ≈ -10 to -90 (decisive negatives) | ❌ Dead lever (S20) |
| Mac MPS fp32 matmul ≠ CPU fp32 | Direct comparison | Bit-identical (S21, S22 confirmed) | ✅ Mac matmul is correct |
| 02_sparse_structure diverges | IoU measurement Mac vs CUDA | IoU **0.9883** — near-parity | ✅ Sparse structure ≈ correct |

**Mac vs CUDA at key bisection points (fairy house, `--resolution 1024`, seed 42):**

| Stage | Mac fp32 | CUDA | IoU / Δ |
|---|---|---|---|
| 02_sparse_structure coords | 2,543 voxels | 2,550 voxels | **0.9926** ✅ |
| 03a_shape_slat feats \|Δ\| | — | — | mean 3.27, p99 14.4 ❌ (later found to be noise-equivalent — S22) |
| 03b inject → fdg_coords IoU | 4,029,429 V | 4,039,030 V | **0.9985** (with CUDA 03b injected) |
| 04_shape_slat_decoded sparse latents (coord overlap) | — | — | 88.2% (L0), 40.6% (L3/8) — from S16 |
| 07_run_output raw mesh V | 3,940,419 | 4,039,030 | 97.6% |
| 07 excess edges | 6.47% | 5.14% | 1.3pp gap |

**Remaining unexplained gap:** With CUDA 03b injected, Mac produces fdg_coords IoU 0.9985 vs CUDA. The residual ~0.15% gap plus the pre-injection divergence is unexplained but appears to be in the distribution of latent values that produce borderline voxel decisions (thin features). No single operator identified as the bug — MPS reductions across all DiT ops collectively differ from CUDA fp32 in ways that compound at thin-feature decision boundaries.

---

### 4D — Texture quality: fuzzy/speckled textures (Session 23)

**Symptom:** Mac textures look fuzzy/painterly and speckled vs CUDA's crisp output, even at `--texture-size 4096`.

**Two distinct defects, isolated:**

**(1) Speckle (dominant, ~15× CUDA) — PROVEN to be Mac's `o_voxel` bake chain.**

| Texture | speckle (\|px−med3\|>40) | LapVar |
|---|---:|---:|
| CUDA out.glb (CUDA's own bake) | 0.0085% | 63 |
| Mac normal run (all-Mac) | 0.1283% | 327 |
| **CUDA-perfect inputs → Mac bake (substitute)** | **0.1262%** | 415 |

The substitute bake (CUDA `07_run_output` mesh + `06_tex_slat` field, built into the bridge npz with `origin=[-0.5]³`, `voxel_size=1/1024`, run through Mac `o_voxel.to_glb`) reproduces the full-Mac speckle from inputs CUDA bakes cleanly. **Upstream (Mac DiT / mesh / tex_slat) is exonerated for the speckle — it is introduced entirely by the Mac bake chain.**

Speckle = ~0.16% near-black **dropout** texels, 97.9% in chart interiors (edges 6.4× enriched but only 0.3% of area). Dropout = a texel whose *interpolated* 3D position lands off the thin occupied shell (0/8 corners) → samples empty field → black.

Ruled out by direct test:
- Dense `F.grid_sample` fallback — NOT used; Mac takes the sparse `_flex_grid_sample_3d` path (`_HAS_FLEX_GEMM=True`).
- mtlgemm Metal `grid_sample_3d` — renormalizes partial-occupancy corners correctly; matches torch reference exactly on a constant-color thin-shell partial-occupancy probe (the 6/6 unit test uses dense inputs and misses this regime, but the targeted probe confirms it).
- Source-field noise — Mac tex_slat is spatially smooth (adjacent-voxel RGB \|Δ\| mean 0.016, 0.0046% jumps>0.3).
- Surface-off-shell — mesh vertices 99.99% land on occupied voxels.

**Remaining sub-cause (not yet split):** cumesh **simplify** roughening the surface vs **mtldiffrast** rasterization producing off-surface interpolated texel positions. `--skip-simplify` isolation avoided (xatlas chokes on the 4M-vert mesh); use per-texel position dump instead.

**(2) Softness (painterly) — mostly inherent, minor Mac low-pass.**
CUDA tex_slat adjacent-voxel \|Δ\| 0.0189 vs Mac 0.0161 (ratio 1.17× — nearly equal; softness is inherent to 1024³ tex decode, not a Mac defect). The one real Mac effect: CUDA preserves ~6× more *sharp* color transitions (jumps>0.3: 0.027% vs 0.0046%) — a mild MPS-decode low-pass. Minor contributor. Chart fragmentation (~10k Mac vs ~1.7k CUDA charts) compounds perceived blur via low per-chart texel density.

**Mitigation (cheap, sub-cause-agnostic):** post-bake, detect near-black interior dropout texels and inpaint from neighbors, OR fall back to nearest-occupied voxel when trilinear returns ~0. Kills visible speckle directly. Fixtures for repro: `/private/tmp/fixtures_fairy_cuda_v3/` (CUDA v3 full stage dump: `06_tex_slat_decoded.pt`, `07_run_output.pt`).

The Session-24 mtldiffrast depth/tie-break fix (Bug C) addresses one speckle
contributor (UV-seam texels where the wrong triangle won the z=0 overlap); the
dominant contributor is still the off-shell trilinear sampling on the roughened mesh.

### 4E — Texture COLOR desaturation: Mac tex_slat-flow (S5) bug, size-scaling (Session 24)

Mac loses base_color saturation/hue, **scaling with decoded mesh size**: turtle (small) ≈ GT;
orc (3.0M v) correct hue but splotchy; robocrab `9_img.png` (6.3M voxels) → grey, ~0% red for a
red robot. Mac also invents spurious metallic. Distinct from §4D speckle.

CUDA-vs-Mac decoded `06_tex_slat_decoded` base_color (same image, seed 42, 1024_cascade):

| metric | Mac | CUDA |
|---|---:|---:|
| mean saturation | 0.14 | 0.49 |
| frac red-dominant | 0.0001 | 0.405 |
| metallic mean | 0.21 | 0.0002 |

**Mac-specific bug** (CUDA colors correctly → weights/arch fine). **Localized to S5 (tex_slat
flow / its conditioning), upstream of the decoder.** Ruled out: bake/mtldiffrast/mtlbvh (audited
faithful; field is grey *pre*-bake); attention precision (`PIXAL3D_FP32_ATTN=1` → no change);
decoder precision (`tex_slat_decoder` in `FP32_MODELS` → no change ⇒ decoder faithful, input
`tex_slat` already color-dead); token cap (no-op at 1024, see §5). Supersedes the §5
"tex_slat_decoder (06) drift" framing — the colour/metallic corruption originates in S5, not the
decoder.

**Next (needs rental):** stage-diff Mac vs CUDA `image_cond` + `05_tex_slat`; or cross-inject
CUDA `tex_slat`+`subs` → Mac `decode_tex_slat`.

**Three independent failure modes:** (1) brush-stroke/splotchy texture — universal, §4D bake
speckle; (2) colour desaturation — size-scaling, S5 (this §); (3) thin-feature mesh destruction
(hair/drips) — size-scaling, QEM sieve (upstream WONTFIX).

**→ Largely RESOLVED in §4F (S25):** the "S5 colour" and the size-scaling are the SAME
mechanism (generative-latent variance collapse), in two modules. Conditioning + coarse shape are
bit-identical; colour is the tex-flow collapse, not the decoder.

### 4F — RESOLVED: generative-latent VARIANCE COLLAPSE is the root of both size + colour (Session 25)

Full Mac↔CUDA stage-ladder diff (robocrab 9_img.png, seed 42, 1024_cascade). Conditioning
(01a–01d, z_global l2 71.554 both) and coarse shape (03a 4386 vs 4387, std 5.82 vs 5.79) are
**bit-identical**. Divergence begins in the **decoders** as a systematic **variance collapse**,
compounding with cascade depth/sampling steps. Two loci → the two symptoms:

| locus | tensor | Mac vs CUDA | symptom |
|---|---|---|---|
| shape decoder | subdiv logits (`subs`) | std ratio 0.89→0.88→0.60→0.54 by level; subs3 20.2 vs 37.7 | geometry/size (6.27M vs 7.86M voxels) |
| tex flow | `05_tex_slat` | every ch 0.49–0.98 (ch22 0.49, ch8 0.67); std 2.46 vs 2.70 | colour + spurious metallic |

**Ruled out (the collapse is intrinsic, not fixable by a flag):** fp16 (`FP32_MODELS=shape_slat_decoder`
→ unchanged; `LayerNorm32` already fp32, `norm.py:5`); MPS-operator bug (`CPU_MODELS=shape_slat_decoder`
**bit-identical** to MPS — subs3 1,606,610 vs 1,606,647); the input (03a identical); conditioning.
⇒ intrinsic Mac-PyTorch (CPU≡MPS) vs CUDA backend reduction-order difference, *amplified* by the
deep cascade (≫ the §4A ~5e-3 drift).

**Mac-only recalibration (moment-match to CUDA) recovers both — PARTIALLY:**
- subdiv → geometry: voxel count 6.27M→**7.33M** (subs std matched 130/101/49/35).
- tex_slat per-channel → colour: red 0.0003→**0.066**, metallic 0.222→**0.0013** (CUDA 0.0002,
  near-perfect), sat 0.14→0.18. Hue short of CUDA (0.18 vs 0.49) — linear final-latent match
  can't undo per-step compounding.

**Geometry destruction = mesh PERFORATION, and it is LATENT-driven (not the bake/simplify):**
the **raw** decoded mesh (`flexible_dual_grid_to_mesh` output, BEFORE any `to_glb`/simplify) is
already shredded — **7.94% of edges are open/boundary** (Euler −83,044, non-watertight). Recalib
reduces it to **7.52%** (Euler −63,323, ~20k fewer holes). Simplify is genuinely exonerated here
(the holes precede it); §4C already showed the FDG extractor is faithful, so a perforated *raw*
mesh ⇒ a *gappy field* ⇒ latents. **Size-scaling**: Mac *fairy* raw boundary 2.99% ≈ CUDA 3.16%
(§8) but robocrab 7.94% — the collapse compounds with model size. ⇒ supersedes the earlier
"residual geometry = §4B wall" framing: for robocrab the dominant mesh damage is upstream in the
field, same variance-collapse mechanism as size+colour. (Crude recalib only −5%; needs the
faithful op-hunt fix.) Tooling: `scripts/mac_decode_replay.py` (decode-only, bit-exact, 18→6 min;
emits subdiv std + raw-mesh boundary metric). Detail + tables: `SESSION_25_FINDINGS.md`.
**Open:** op-hunt for the per-op reduction divergence (NEEDS RENTAL) — the unified fix for
geometry perforation + colour; per-step tex_slat recalib for hue (Mac-only).

**S26 RESOLUTION + corrections (supersedes parts of the above).** The op-hunt found it:
the variance collapse is **Bug D — MPS fused SDPA wrong above ~18–20k tokens**, compounding
in the flow DiTs. Corrections to the S25 framing here:
- "collapse is intrinsic, not fixable by a flag" → **wrong**: fixed by `SPARSE_ATTN_BACKEND=naive`.
- "CPU≡MPS, not an MPS bug" → held for the **decoder** (S26 confirms decoder bit-faithful,
  259-op max ratio 1.077) but is **false for the sampler/attention**: CPU 1.246 = CUDA,
  MPS 1.0755. S25 measured CPU≡MPS on the decoder and over-generalized to the collapse.
- "perforation is latent-driven" → **split**: geometry **size** + **colour** are
  latent/variance-driven and now FIXED (HR slat std 4.99→5.67; sat 0.139→0.341). Residual
  thin-feature **fragmentation/perforation** is **orthogonal** — the §9-F / §4B sieve wall
  (post-bake Mac fairy ~3× CUDA components), NOT the latent. The S25 "supersedes §4B"
  call is itself partially walked back: §4B sieve is back in scope for the *fragmentation*
  half. Recalibration is no longer needed (the real fix is upstream). Detail: `SESSION_26_FINDINGS.md`.

**S29 — §4F COLOUR CLOSED (texture path proven faithful to CUDA).** Controlled CPU-vs-MPS
bisection of the tex path + one CUDA reference run settle the residual:
- **Bake** loses no saturation: fairy decoded **field** sat 0.32 ≈ baked 0.34 (refutes S26's
  texture-size guess).
- **Tex decoder** bit-faithful CPU≡MPS (Δ 0.0; re-confirms S6) — `SparseUnetVaeDecoder` is
  conv/ConvNeXt, no long-seq SDPA.
- **Tex latent variance** recovered: `05_tex_slat` std 2.69 ≈ CUDA 2.70.
- **Tex flow** has **no SDPA-cliff attention** ("proj" is a per-voxel `nn.Linear`, not
  attention; self-attn = fairy 10.8k voxels *under* the cliff, robocrab ~21.5k *over* → the §4E
  size-scaling). Single forward on **real captured inputs**, CPU vs MPS: **mean\|Δ\| 1e-6 / max
  2.5e-5** ⇒ no second Bug-D.
- **CUDA fairy reference** (1_img seed 42, `scripts/cuda/run_tex_ladder.py`): Mac vs CUDA 06 —
  sat **0.324 vs 0.305** (Mac ≥ CUDA; muted look is *inherent*), metallic **0.408 vs 0.313**,
  base_color RGB identical, field voxels 3.94M vs 4.03M. ⇒ saturation at parity; metallic a
  *modest* ~0.09 residual (CUDA fairy metallic is 0.31, **not** ~0 — the 0.0002 was
  robocrab-specific; metallic is object-dependent), inherited from the upstream shape-cascade
  ~0.15% drift (§4C), not the tex path. Same intrinsic reduction-order class as §4A — no clean
  Mac fix. **Supersedes §4E's "Mac invents spurious metallic" framing** (it's mild amplification,
  not spurious) and the §4F "tex-flow recalib needed" lead.
- `generate_mps.py --tex-recalib` (default **off**) added as a **cosmetic** PBR override only
  (metallic moment-match + chroma boost); not a fidelity fix — a fixed metallic target can't be
  object-agnostic. Detail: `SESSION_29_FINDINGS.md`.

---

## §5 — Key Architecture Facts

- **All Mac fp32 paths agree:** `torch_mps_fp32 ≡ torch_cpu_fp32 ≡ coreml_cpu_fp32 ≡ coreml_all_fp32` — Apple Silicon conv/matmul reduction order is different from NVIDIA but internally consistent.
- **Generative-latent variance collapse (S25):** the Mac↔CUDA reduction difference, while ~5e-3 per op (§4A) and *non-causal for image_cond*, **DOES** compound through deep decoder/flow cascades into a systematic ~50% latent-variance collapse that causes BOTH the size-shrinkage and colour death. CPU≡MPS (not an MPS bug), fp32-insensitive (not fp16). Per-channel/per-level recalibration to CUDA's distribution recovers it (partially). The decoder/flow are uniquely *amplifying* where the DiTs (§4C "fp32-correct") were not. See §4F.
- **(S26 — supersedes the two bullets above as the mechanism)** The variance collapse is **MPS fused
  `scaled_dot_product_attention` returning wrong results above ~18–20k tokens** (Bug D), not a
  generic reduction-order amplification. CPU fp32 attention **== CUDA** (1.24629 vs 1.24632); only
  MPS diverges. CPU≡MPS holds for the **decoder** but **not** for long-sequence attention. Fix =
  `SPARSE_ATTN_BACKEND=naive` (chunked fp32 matmul+softmax). Self-contained repro exists.
- **(S26)** The robocrab/fairy flow DiTs run sparse attention over **~21,500 tokens** — above the MPS
  SDPA cliff. Mesh complexity → token count → whether the bug fires (turtle stayed under; fairy/crab
  crossed). Token count is gated by `max_num_tokens` / `actual_hr_resolution` in
  `pixal3d_image_to_3d.py:723`.
- **CoreML fp32 ≡ MPSGraph:** CoreML routes fp32 ops to GPU (MPSGraph). Same math, same result. ANE only activates for fp16.
- **CPU_AND_NE is universally worse:** Engine partition boundaries introduce extra fp casts. Never use.
- **`compute_units=ALL`, `compute_precision=FLOAT32` ≡ MPSGraph** (re-confirmed S13, S14).
- **Bridge subprocess is NO LONGER mandatory (S23 — unify-py310):** Originally Pedro's Metal stack (mtldiffrast + cumesh + mtlbvh + flex_gemm + o_voxel) shipped as CPython 3.11 wheels while the Pixal3D venv used 3.12, so all native export ran through `pixal3d/utils/o_voxel_native_export.py` as a subprocess under the trellis-mac 3.11 venv. **The 3.12/3.11 split was an accident of venv-creation timing, not a requirement.** Session 23 consolidated everything into a single **Python 3.10** venv (`.venv-py310`, matching upstream Pixal3D + the CUDA reference). The native packages are now vendored in `extern/` and built from source against 3.10, so the bridge runs **in-process by default** (`o_voxel_native_export.main(argv)` called directly). `PIXAL3D_NATIVE_SUBPROCESS=1` restores the old subprocess path; the in-process path also auto-falls-back to subprocess (sentinel 127) if `o_voxel.postprocess` isn't importable in the running interpreter (e.g. the legacy 3.12 venv's stub `o_voxel`). Verified bit-parity: fairy raw mesh V=3,940,419 / F=8,586,890 identical to the 3.12 baseline; wall time unchanged (~9.5 min). **Python version was never in the quality causal chain** — consolidation is a maintainability/reproducibility win, not a quality one.
- **MPS `sdpa_kernel()` context manager is a no-op** (S5, torch 2.12): all three backends (`MATH`, `EFFICIENT`, `FLASH`) produce bit-identical MPS output. Don't waste time pinning SDPA backends on MPS.
- **Forcing global fp32 upcast crashes Metal** (`NDArrayMatrixMultiplication` dtype mismatch at runtime). Per-submodel `.float()` via `PIXAL3D_FP32_MODELS` is the safe workaround.
- **pixal3d-mlx (community MLX port) produces worse output** than PyTorch/MPS port — more artifacts, worse textures, regressed transparency bug we'd already fixed. Not a viable path (S7).
- **`fast_simplification` floors at ~921K faces** on non-manifold Pixal3D meshes (refuses to collapse non-manifold edges). Blender Decimate-Collapse floors at ~2.76M.
- **pymeshlab `preservetopology=True` floors at ~1.9M; `preservetopology=False` produces spiky blobs** (collapses through non-manifold edges, destroys silhouette). Both options ruled out (S2).
- **tex_slat_decoder (06) drift is real and substantial** (S5): over the 28% coord-overlapping voxels, max abs diff 1.21, mean abs diff 0.12 (~10% of value range). Per-channel worst: channels 3/4 (metallic/roughness PBR). `PIXAL3D_FP32_MODELS=tex_slat_decoder` is in the production recipe.
- **natten 0.21 has NO MPS backend:** `na2d` hard-fails on MPS (only `cutlass-fna`/CUDA and `flex-fna`/CPU backends exist upstream). The in-house `natten-mps` package (S10–S11) provides Metal kernels but is NOT a z_proj fix (S11 proved Q/K are already wrong before natten).
- **natten-mps community package exists:** `pip install natten-mps` (ssmall256/natten-mps v0.3.0). Installed S15. Asymmetric K/V dims not supported (K=64, V=256 fires once then falls back to `_torch_na2d`).
- **mtlgemm vendored in-tree (S23):** now `extern/mtlgemm/` (was `/Users/pawelma/code/ai/mtlgemm/`) — Pedro's Metal port of flex_gemm. Build: `pip install -e extern/mtlgemm --no-build-isolation` (avoid rpath issue). Provides `sparse_submanifold_conv3d`, `grid_sample_3d`. Requires `torch>=2.11` on Darwin (`at::mps::dispatch_sync_with_rethrow`, PR #167445).
- **All native Metal deps vendored in `extern/` (S23):** `mtlmesh`(→cumesh), `mtlgemm`(→flex_gemm), `mtlbvh`, `mtldiffrast`, `natten-mps` via `git subtree` (squashed, upstream-syncable); `o_voxel` copied from `pedronaugusto/trellis2-apple` subdir `o-voxel/` @ 6055b86 (subtree can't pull a remote subdir). The mtlmesh Bug A `simplify.metal` volatile patch is carried as a local commit on top of the upstream subtree. See `extern/README.md`. Reproducible install: `requirements-mac.txt` + the extern editable-install loop.
- **`fdg_h_feats` threshold-flip is a small contributor (S9):** `fdg_intersected = h.feats[..., 3:6] > 0`. Logit range [-121.8, 40.3], mean abs 12.7. Only 0.13% of entries within ±1e-2 of zero. Borderline-flip from precision drift is not the dominant divergence source — the dominant chain is z_proj → DiT → different sparse structure → different voxel set.
- **`PIXAL3D_REPAIR_NME=pamo` vs `=metal`:** functionally equivalent (S16 verified: 3,197,312 vs 3,196,862 V, 0.014% difference). pamo adds CPU latency for no benefit.
- **Mac is locked to `1024_cascade` pipeline type** (S20 — 1536_cascade causes catastrophic shatter on Mac's Metal simplify).
- **`--max-num-tokens` is a no-op at 1024 (S24):** `pixal3d_image_to_3d.py:455` and `:731` gate the cap as `if num_tokens < max_num_tokens or hr_resolution == 1024:` — the `or hr_resolution == 1024` short-circuits at 1024, so the cap never binds. There is **no knob to shrink the decoded field at 1024**; field/voxel count (e.g. robocrab 6.32M) is set by the FDG cascade decode, not the SLAT token cap. Size-causation of §4E can only be shown cross-object or via stage-diff.
- **`--no-texture` smoke test:** reliable baseline-establishing tool. Raw Mac pipeline mesh has only pinprick holes; all catastrophic sieve damage is created by `o_voxel.postprocess.to_glb` (S17 revelation).
- **Canonical `o_voxel.postprocess.to_glb` chain** (3 simplify passes, not 1):
  ```
  fill_holes → simplify_3x → repair_nme → small_cc → fill_holes → simplify_1x → repair_nme → small_cc → fill_holes → unify_faces → simplify_target → compute_charts
  ```
  This is the `remesh=False` (simplify-sieve) branch — the one Mac uses by default. **It is
  intrinsically holey** (≈8–11% welded boundary) on dense thin-feature meshes on *both* Mac
  and CUDA. The CUDA reference does NOT use it — it uses `remesh=True`. See Bug E.
- **(S27) `to_glb` has two branches; CUDA uses `remesh=True`, Mac uses `remesh=False`.** The
  `else:` branch (`postprocess.py:251`) does narrow-band **dual-contouring remesh** that
  rebuilds a watertight manifold (→ 0% welded boundary, ~1.7k charts). The CUDA reference
  pipeline always passes `remesh=True`. This single flag is the dominant Mac↔CUDA mesh-quality
  divergence (Bug E). The geometry-cleanup kernels (simplify/repair_nme/small_cc) are
  **faithful** Mac↔CUDA (cross-bake proven) — they were never the bug.
- **(S27) Use coordinate-WELDED metrics for watertightness** (`np.round(V/1e-6)` then count
  boundary/components). Indexed-vertex component counts on baked GLBs are inflated 3–10× by
  UV-seam duplicate verts and are misleading (CUDA `out.glb` = 40k indexed comps but **0.0%
  welded boundary**, i.e. truly watertight). This corrects S26's "post-bake Mac ~3× CUDA
  components" framing — that gap was mostly UV seams, not geometry.
- **B_true (coord-merged boundary edges) is the correct mesh-quality metric**, not indexed boundary edges (which are polluted by UV-seam splits from texture bake — CUDA golden also has 42% dup verts at 1e-5 from UV seams).
- **Session 7 cleanup exoneration was invalid:** fed stage-08 (already cleaned) into Mac cleanup. Correct test: stage-07 (raw FDG pre-cleanup).
- **Scale matters:** Session 7 reference was 849K verts; current runs hit 4.19M (5× denser). Whatever cleanup bug exists scales non-linearly with mesh density.

---

## §6 — Infrastructure Built

| Script / Tool | Purpose | Session |
|---|---|---|
| `extern/natten-mps/` (S23; was `/Users/pawelma/code/ai/natten-mps/`) | Standalone Metal natten port (pure-Python). na2d_qk + na2d_av Metal kernels. 19/19 tests pass. Vendored in-tree as source-of-truth but **NOT pip-installed** — production uses the community `natten-mps==0.3.0` (ssmall256) for its `natten_mps.compat.v020` API that `generate_mps.py` imports. See §4A — natten is NOT the z_proj fix. | S10–S11, S23 |
| `extern/` (S23) | All native Metal deps vendored: mtlmesh, mtlgemm, mtlbvh, mtldiffrast (git subtree), o_voxel (copy), natten-mps. `extern/README.md` documents subtree-pull/push workflow. | S23 |
| `requirements-mac.txt` (S23) | Pinned reproducible dep set for `.venv-py310`. Includes the 7 implicit runtime deps `requirements.txt` omits (kornia, timm, opencv, imageio, imageio-ffmpeg, easydict, zstandard) needed by the BiRefNet/HuggingFace dynamic-module path. | S23 |
| `scripts/diff_fixtures.py` (Pixal3D_fresh) | Cross-box stage-by-stage fixture diff (first used S9) | S9 |
| `scripts/diff_naf_trace.py` | Compare CUDA vs Mac NAF trace payloads | S12 |
| `scripts/coreml_probe_stage1-4.py` | CoreML/ANE per-conv probes | S13–S14 |
| `scripts/naf_ane_swap.py` | `CoreMLConv2dWrap` + `CoreMLWholeModuleWrap` + production env-var hooks | S14 |
| `scripts/metal_naf_kernels.py` | 4 Metal kernels for NAF image encoder | S15 |
| `scripts/naf_metal_swap.py` | `MetalConv2d/MetalGroupNorm/MetalSiLU` drop-ins + `install_metal_on_naf` | S15 |
| `scripts/metal_winograd_f43_prototype.py` | Winograd F(4,3) Metal kernel (ruled out) | S15 |
| `scripts/cuda_substitute.py` | Stage-level CUDA fixture injection via `PIXAL3D_CUDA_SUBSTITUTE=stage:path` | S15 |
| `scripts/cuda/remote.sh` + per-test scripts | CUDA rental harness (vast.ai SSH+rsync wrapper) | S19 |
| `scripts/cuda_compute_charts_test.py` | CUDA-side compute_charts bisection | S18 |
| `scripts/run_metal_postprocess_chain.py` | Per-stage cleanup chain replay with NME metrics | S19+ |
| `scripts/run_canonical_postprocess.py` | Faithful o_voxel canonical chain replay | S19 |
| `scripts/diff_03a_outputs.py` | Coord-keyed feats diff for 03a SparseTensor dumps | S22 |
| `scripts/cuda/run_tex_ladder.py` + `scripts/mac_tex_ladder.py` | Mac↔CUDA stage-ladder stats (cond/latents/decoded) without torch.save | S25 |
| `scripts/mac_decode_replay.py` | Decode-only replay: capture decode inputs once, re-decode ±recalib in ~6 min (bit-exact vs full run) | S25 |
| `scripts/mac_recalib_probe.py` / `mac_recalib_full.py` | Variance-recalib levers (subdiv→geometry, tex_slat→colour); robocrab-specific constants | S25 |
| `scripts/mtlmesh_build_metallib.sh` | Fast metallib-only rebuild (~6s) | S22 |
| `scripts/mtlmesh_build_full.sh` | Full mtlmesh rebuild with Obj-C++ extension (~60s) | S22 |
| `cuda-remote-gpu` skill | Auto-packaged CUDA rental skill at `~/.agent-skill-manager/vault/` | S19 |

**Key env vars (PIXAL3D_* namespace):**

| Var | Effect |
|---|---|
| `PIXAL3D_FP32_MODELS=<comma-list>` | Upcast named models to fp32 (Bug B fix for DiTs) |
| `PIXAL3D_CPU_MODELS=<comma-list>` | Run named models on CPU fp32 (diagnostic) |
| `PIXAL3D_CUDA_SUBSTITUTE=stage:path[,...]` | Inject CUDA fixture at named stage |
| `PIXAL3D_HR_SLAT_INJECT=<path>` | Override HR cascade DiT output (03b) |
| `PIXAL3D_NAF_METAL=1` | Use custom Metal kernels for NAF image encoder |
| `PIXAL3D_NAF_ANE_REPLACE=<comma-list>` | Per-conv CoreML/ANE swap (retired — no z_proj benefit) |
| `PIXAL3D_SUBDIV_BIAS=<float>` | Shift C2S3d threshold (dead lever — documented no-op) |
| `PIXAL3D_DUMP_FIXTURES=<dir>` | Dump intermediate tensors at every stage |
| `PIXAL3D_STOP_AFTER=<stage>` | Abort pipeline after named stage |
| `PIXAL3D_NAF_TRACE=1` | Enable NAF layer-by-layer trace |
| `NATTEN_MPS_CAPTURE_FIRST=<dir>` | Dump natten Q/K/V/out on first call (used for S11 cross-box diff) |
| `PIXAL3D_NATTEN_MPS=pytorch` | Opt out of natten-mps to pure PyTorch |
| `PIXAL3D_NATIVE_SUBPROCESS=1` | (S23) Force the legacy subprocess o_voxel bridge instead of the new in-process default. In-process also auto-falls-back to subprocess if `o_voxel.postprocess` isn't importable in the running interpreter. |

---

## §7 — Remaining Gaps and Options

### Thin-feature mesh quality (fairy / palm / elf-bust)

**Root:** parallel QEM overshoot on dense thin-feature topology. Not Mac-specific — CUDA has milder version. Upstream WONTFIX (CuMesh #28).

**Options (cost-ordered):**

| Option | Effort | Notes |
|---|---|---|
| Ship for chunky inputs, document thin-feature limit | 0 | Turtle-class meshes work well. Honest, defensible. |
| Expose `--native-texture-size 4096` + xatlas pack knobs | Hours | Better atlas utilization even with many charts |
| Python NME-tolerant chart growing (~150 lines) | ~1 day | Walk faces by edge adjacency ignoring face_count constraint; merge by normal-cone cost |
| PaMO-style post-collapse self-intersection + revert in `simplify.metal` | 2–3 weeks | The canonical fix per PaMO paper. Neither Pedro's port nor upstream CuMesh has it. Required: triangle-triangle intersection test kernel, integration, validation, perf tuning. Would be an upstream contribution to `pedronaugusto/mtlmesh`. |
| Custom Metal simplify with link-condition check (Hoppe) | 2–3 weeks | Replace parallel QEM with one that checks vertex-link intersection before collapse |

### Pre-cleanup mesh (thin-feature origin in decoder)

**Residual fact:** Even with CUDA 03b injected, Mac decoder produces slightly fewer voxels than CUDA (fdg_coords IoU 0.9985 — 0.15% gap). The underlying source (borderline voxel flip rate in MPS LayerNorm / sparse-conv reductions inside `SparseResBlockC2S3d`) has not been pinpointed to a single op. `SPARSE_CONV_BACKEND=none` produces bit-identical output to flex_gemm on Mac — sparse Conv3D is not the source. Remaining suspects: `F.layer_norm` (S21 priority 1), MPS SDPA internal intermediates (S21 priority 2). Not yet tested.

---

## §8 — Known Good Numbers (reference baselines)

**CUDA fairy (1_img.png, seed 42, --resolution 1024) stage-07 raw mesh:**
- V=4,039,030, F=8,605,208
- boundary=3.16%, excess=5.14%, NME-vertex=25.66%

**CUDA fairy canonical cleanup output (300k target):**
- 1,731 charts @ 166 avg faces/chart

**Mac fp32 fairy (production recipe, S19+) stage-07 raw mesh:**
- V=3,940,419, F=8,586,890
- boundary=2.99%, excess=6.47%, NME-vertex=22.27%

**Mac fp32 fairy canonical cleanup output:**
- 32,284 charts @ 30.5 avg faces/chart (vs CUDA's 1,731)

**CUDA turtle (0_img.png, seed 42) cleanup output:**
- 1,731 charts @ 166 faces/chart

**Mac fp32 turtle cleanup output (post-Bug-A):**
- 2,330 charts @ 120 faces/chart ✅ (acceptable quality)
- (S26 note: turtle was always acceptable because its token count is **under** the MPS-SDPA
  ~18–20k cliff — Bug D never fired for it.)

**S26 — MPS SDPA cliff + Bug-D fix (robocrab 9_img.png, seed 42, 1024_cascade):**
- MPS fused SDPA, fp32, random q/k/v: bit-exact (mae 1e-6) through N=18000; **N≥20000 breaks**
  (mae 0.05–0.10, max ~11). CPU/CUDA exact at all N.
- Single HR-DiT forward (identical noise+cond): CUDA 1.24633 / CUDA-sdpa 1.24632 / CPU-fp32
  1.24629 / **MPS-sdpa 1.07553** / **MPS-naive(fix) 1.24676**.
- Decoder fed identical HR slat: subs std 115.36/88.61/29.49/20.18 **identical** Mac & CUDA;
  259-op max ratio 1.077.
- HR decode-input shape-slat std: CUDA 5.626 / Mac-broken 4.988 / **Mac-naive(fix) 5.6704**.
- Robocrab baked-texture saturation: broken **0.139** → naive **0.341** (CUDA target 0.49);
  frac sat>0.2 0.22→0.71.

**S26 — post-fix thin-feature fragmentation (post-bake GLB connected components):**
- CUDA fairy (pristine): 42,848 components, tiny(<200f) 0.63, open-edge 0.33.
- Mac fairy (naive, post-fix): **122,729** components (~3×), tiny 0.82, open-edge 0.49.
- Mac orc (naive): 70,949 comp. ⇒ residual sieve wall (§9-F), orthogonal to Bug D.
  **(S27 correction: these are UV-seam-inflated *indexed* counts; the real geometry metric
  is welded boundary — see below. The residual is Bug E remesh, not the §9-F "sieve wall".)**

**S27 — welded watertightness + remesh cross-bake (fairy, 1_img.png, seed 42, 1M target):**
- Coordinate-welded (eps 1e-6) production GLBs: CUDA `out.glb` boundary **0.0%** / excess
  0.24% / largest body 0.96 (watertight); Mac `fairy_house_naive.glb` boundary **10.7%** /
  excess 6.4% / largest 0.98 (holey).
- Cross-bake, same input mesh, per-step (Mac-Metal chain vs CUDA chain): simplify_3x NME
  39.9/40.3, excess 11.0/11.0; repair_nme components 195,909 (Mac) / **232,348 (CUDA, worse)**;
  simplify_1x components 5,237/5,316. ⇒ geometry chain faithful.
- Mac mesh → CUDA `to_glb`: **remesh=True → welded boundary 0.0%** (235 comps, largest 0.978);
  remesh=False sieve → 7.99%. (env: rental CUDA, `scripts/cuda/run_to_glb_bake.py`.)
- Metal `remesh_narrow_band_dc` on Mac mesh (BROKEN): boundary 8.78%, 18,636 components,
  largest 0.77; downstream compute_charts 99,859 charts. (env: Mac `.venv-py310`.)
- CUDA `out.glb` is 934,304 faces (1M target) — corrects §8's implied "300k/1,731-chart"
  production; the 1,731-chart figure was a remeshed clean manifold, not a decimation-target diff.
