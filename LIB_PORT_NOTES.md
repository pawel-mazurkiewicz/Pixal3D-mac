Good catch — and the model weights ARE bf16/fp16 on disk (note the filename suffixes: _bf16.safetensors for DiTs, _fp16.safetensors for decoders). So the fp32 ablation isn't moot — we just need to upcast the weights before inference, not the latents (which are already fp32). I'll get to that.

First, your porting question, then the ablation plan.

Porting feasibility

flash_attn → Metal — medium effort, ~1–2 weeks

This one is genuinely tractable:

- Existing work: philipturner/metal-flash-attention is an MIT-licensed Metal port of FlashAttention-2 with bf16 + fp32, tested back to M1. It compiles its own Metal shaders at runtime via MSL strings, no Xcode needed.
- M5 specifically: macOS 26's MPSGraph exposes the new tensor primitives (Apple's matrix coprocessor) via MPSGraphTensorMode.tensor and the new MPSGraphMatrixMultiplicationDescriptor paths. metal-flash-attention doesn't yet wire those — it uses regular Metal compute shaders — but bolting them on as a fast path for matmul tiles is plausible. A few hundred lines.
- Integration into Pixal3D: write a pixal3d.modules.attention.metal_flash_attn backend that torch.utils.cpp_extension loads, register it alongside the existing flash_attn/sdpa paths. The ATTENTION_BACKEND env var hook is already there.
- Numerical fidelity risk: even a faithful Metal port of FA2 will not be bit-identical to NVIDIA's flash_attn because of softmax accumulation ordering — but it should land within 1e-3 of flash_attn, vs the current 0.3 we're seeing with sdpa. That's the win.

Realistic timeline:
- Day 1–3: get metal-flash-attention building and running standalone tests against CUDA's flash_attn output on the same inputs.
- Day 4–7: PyTorch custom op wrapper, register as Pixal3D attention backend.
- Day 8–14: profiling, debug any layout differences (heads-first vs sequence-first), fix edge cases.

flex_gemm submanifold sparse 3D conv → Metal — large effort, 1–3 months

This is much harder. Submanifold sparse conv is:
- A gather based on a neighbor-index table (which is itself precomputed per layer / per dilation),
- A batched matmul at active positions only,
- A scatter back into output sparse positions = input sparse positions.

The matmul is the easy 30% — that's where M5's tensor primitives help. The hard 70% is:
- Building and caching the neighbor index efficiently for 3D voxel grids (CUDA's flex_gemm uses hash-based kernels here).
- Coalesced memory access under indirect addressing — Metal's memory model is restrictive vs CUDA's, and gather/scatter at scale requires careful tile layout.
- Submanifold-specific optimization — the "active voxels only" property is what makes it fast; without it you'd just be doing dense 3D conv with a mask.

Existing alternatives:
- MinkowskiEngine has a CPU path — correct but ~50–100× slower than 4090.
- TorchSparse has a CPU path — similar story.
- SpConv (the most-used spconv lib) is CUDA-only.

For Pixal3D specifically, the sparse conv is in the shape SLat DiT and the shape/tex VAE decoders. The DiT calls them every transformer block. The decoder calls them through a U-Net. So performance matters.

If I had to give you a one-month estimate: with M5 tensor primitives + reusing MinkowskiEngine's CPU index-builder + writing Metal compute shaders for the matmul tile, you could get something that's ~10–20% as fast as a 4090 (which would still be usable). The numerical fidelity question is separate from speed — even a correct, slow port would tell us whether sparse-conv numerics are part of our current divergence.

Combined assessment

Build flash_attn first — it's the cheap win and is likely most of the current divergence (the DiT is attention-heavy, sparse conv is only a small fraction of FLOPs per block). The reduced kernel-set after that port might be tolerable with the existing sparse conv fallback.

Sparse conv port becomes worth it only if post-flash_attn-port we still see >1e-2 divergence — and even then, switching to MinkowskiEngine-CPU as a correctness baseline might be a smarter first move than writing Metal shaders.
