# Overnight Findings — TL;DR

## The headline

**The divergence is entirely in the DiT samplers, not the decoders.** And it's
most likely dominated by **flash_attn vs sdpa attention-backend mismatch** —
CUDA reference fixture was captured with `ATTN_BACKEND=flash_attn`, Mac runs with
`sdpa` fallback. Same softmax, different reduction order → ~1e-3 to 1e-4 drift per
attention op → compounds across thousands of ops to 3.06 mean-abs feature diff at
the shape SLat stage.

## What I proved (apples-to-apples, MoGe-on, seed 42)

### Decoders are 100% deterministic on their inputs

- Moved tex VAE (`tex_slat_decoder`) to CPU → output **bit-identical** to MPS baseline.
- Moved shape VAE (`shape_slat_decoder`) to CPU → output **bit-identical** to MPS baseline.
- → Both VAEs are FINE. The 89.8% / 28.2% overlap loss is upstream.

### Localized cascade of upstream drift

| Stage | What it is | CUDA-MPS coord overlap | Feature drift |
|---|---|---|---|
| 03a | After SS DiT sampler | 99.7% | mean-abs **3.06** |
| 04 | After shape decoder (final) | 89.8% | mean-abs **72.4** |
| 04 subs[3] | Finest guide_sub | 43.1% | mean-abs **21.8** |
| 06 | After tex VAE | 28.2% | (in different coord set) |

Note the feature drift jump 3.06 → 72.4: that's NOT the shape decoder adding drift
(it's deterministic). It's the decoder's threshold-based `subdiv > 0` flipping
on near-zero values, which exposes the propagated DiT drift.

### Partial fix: SS DiT on CPU

- `PIXAL3D_CPU_MODELS=sparse_structure_flow_model,sparse_structure_decoder` →
  stage 02 went from 99.7% → 99.8% (20 voxel-diff → 12 voxel-diff).
- Confirms putting DiT samplers on CPU does narrow the gap.
- But not bit-identical to CUDA — because CUDA used `flash_attn` not `sdpa`.

### Currently running (when user wakes): all-DiTs-on-CPU

`PIXAL3D_CPU_MODELS=sparse_structure_flow_model,sparse_structure_decoder,shape_slat_flow_model_512,shape_slat_flow_model_1024,tex_slat_flow_model_1024`
— estimated 40-60 min wall. Tests cumulative CPU effect. Background id `bdklvegf0`.

## Recommended priorities

**1. Port `philipturner/metal-flash-attention`** — most impactful long-term.
   Closes attention to ~1e-3 of CUDA's flash_attn. 1-2 weeks of work, MIT licensed.
   Session 5's `LIB_PORT_NOTES.md` already sketched the integration plan.

**2. Re-capture CUDA reference with `ATTN_BACKEND=sdpa`** — fastest sanity check.
   ~1 hr CUDA rental. If our all-DiTs-CPU run is close to CUDA-SDPA, that confirms
   flash_attn is the dominant remaining variable.

**3. Quick win available right now:** always-on `PIXAL3D_CPU_MODELS=sparse_structure_flow_model,sparse_structure_decoder`.
   Adds ~1 min wall time, narrows stage 02 drift. Cumulative benefits if extended
   to other DiTs (at proportional wall-time cost).

## What we should STOP doing

The downstream cleanup tuning (small_cc threshold, fill_holes perimeter, etc.) —
**we are at the floor of what those can do.** The user already discovered this
empirically; the divergence localization confirms it. The hole damage in the
final mesh comes from the MPS sampler producing fundamentally different voxel
coords than CUDA, not from the cleanup pipeline doing the wrong thing on the
right input.

The CPU-unify_face_orientations port we did earlier IS a real fix (Metal binary
was genuinely broken), but it can only fix winding errors among the voxels
that survive. It doesn't recover the chunks of surface that small_cc culls
because they don't exist on Mac in the first place.

## Files of note

- `scratch/DIVERGENCE_LOCALIZATION.md` — full chronological investigation.
- `scratch/diff_cpu_vs_mps_tex_vae.py` — tex VAE determinism diff.
- `scratch/diff_shape_vae_cpu.py` — shape VAE determinism diff.
- `scratch/diff_ss_dit_cpu.py` — SS DiT stage-02 diff.
- `fixtures/mps_all_dits_cpu/` — in-progress all-DiTs-CPU run.
- `fixtures/cuda/00_metadata.json` — confirms `ATTN_BACKEND=flash_attn`.
- `generate_mps.py` — `PIXAL3D_CPU_MODELS` env var (new), patches model.forward
  to run on CPU and shuffle inputs/outputs via MPS<->CPU per call.

## All-DiTs-on-CPU experiment failed

Attempted `PIXAL3D_CPU_MODELS=sparse_structure_flow_model,sparse_structure_decoder,shape_slat_flow_model_512,shape_slat_flow_model_1024,tex_slat_flow_model_1024`.

Failed during shape SLat sampling step 0 with:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two
devices, mps:0 and cpu! (in SparseTensor.__elemwise__ -> torch.sub)
```

The error path was inside the CFG mixin's `_pred_to_xstart` call where
`(A * x_t) - (B * pred)` mixed MPS and CPU operands.

Root cause (unfinished debugging): my forward wrapper mutates the output
SparseTensor's `.data` dict in place and clears `_spatial_cache` — but the
ElasticSLatFlowModel's `pred` SparseTensor somehow still has one or more
device-dependent attributes that retain the CPU original. Could be:
- Caches inside the data dict that I didn't recognize (e.g., `_layout`, internal
  CSR indices in some backends).
- The `.replace()` chain in the model copying _spatial_cache by reference and
  re-using stale CPU tensors.

Workable fix probably exists with deeper SparseTensor introspection but I stopped
chasing it because we already have the architecturally important findings.

For the SS DiT (which uses dense tensors, not SparseTensors), the patch works fine
— `bar7y0xs4` ran cleanly.  The bug is specific to SparseTensor-producing models.

## What this DOES NOT change

The conclusions about WHERE divergence lives don't depend on the all-DiTs experiment
finishing.  We already have:
- VAEs are deterministic (proven).
- DiT samplers introduce drift (proven by SS-DiT-on-CPU narrowing the gap by 40%).
- CUDA reference used `ATTN_BACKEND=flash_attn` (proven via metadata).
- Mac uses `ATTN_BACKEND=sdpa` (proven, default fallback).

Whether all-DiTs on CPU narrows the gap further is interesting but doesn't change
the recommended next moves (metal-flash-attention port, or CUDA-SDPA rental
re-capture).
