// na2d_av — windowed neighborhood attention · V weighted gather.
//
// Matches natten 0.17.5 CPU naive semantics.  Caller has already softmaxed
// the attention logits across the K*K neighborhood dimension.
//
// Tensor layout:
//   attn:  [B, H, X, Y, KS*KS]   fp32 (post-softmax weights)
//   v:     [B, H, X, Y, D]       fp32 or fp16
//   out:   [B, H, X, Y, D]       same dtype as v
//
// Boundary handling: same edge-shift-clamp as na2d_qk (matched indexing).
// One thread per (b*H, x, y); loops over 81 (or K*K) neighbors AND D
// channels.  For NAF (D=64, K=9): 64*81 = 5184 FMA per thread, fine.

// Helpers (na_window_start) defined in shared.metal — concatenated first
// by natten_mps._compile.

kernel void na2d_av(
    device const float* ATTN  [[buffer(0)]],   // [B, H, X, Y, KS*KS]
    device const float* V     [[buffer(1)]],   // [B, H, X, Y, D]
    device float*       OUT   [[buffer(2)]],   // [B, H, X, Y, D]
    device const int*   SHAPE [[buffer(3)]],   // [B, H, X, Y, D, KS, DIL]
    uint3 gid [[thread_position_in_grid]])
{
    const int B   = SHAPE[0];
    const int H   = SHAPE[1];
    const int X   = SHAPE[2];
    const int Y   = SHAPE[3];
    const int D   = SHAPE[4];
    const int KS  = SHAPE[5];
    const int DIL = SHAPE[6];

    const int y  = int(gid.x);
    const int x  = int(gid.y);
    const int bh = int(gid.z);
    if (x >= X || y >= Y || bh >= B * H) return;

    const int nbr = KS / 2;
    const int sx_per = na_window_start(x, X, nbr * DIL);
    const int sy_per = na_window_start(y, Y, nbr * DIL);

    const int KK = KS * KS;
    const int spatial_stride_v = X * Y * D;
    const int row_stride_v     = Y * D;

    const int v_bh_base   = bh * spatial_stride_v;
    const int out_base    = bh * spatial_stride_v + x * row_stride_v + y * D;
    const int attn_base   = bh * (X * Y * KK) + x * (Y * KK) + y * KK;

    // Initialize accumulator row in registers (D=64 typical → 64 fp32 = 256B).
    // For larger D we'd tile; for NAF leave it explicit.  Metal compiles this
    // to a stack/local-register array; correctness first, optimize later.
    // We materialize partial sums in a small fixed array.
    constexpr int MAX_D = 256;   // upper bound; assert at host side
    float acc[MAX_D];
    for (int d = 0; d < D; ++d) acc[d] = 0.0f;

    for (int ki = 0; ki < KS; ++ki) {
        const int kx = sx_per + ki * DIL;
        for (int kj = 0; kj < KS; ++kj) {
            const int ky = sy_per + kj * DIL;
            const float w = ATTN[attn_base + ki * KS + kj];
            const int v_off = v_bh_base + kx * row_stride_v + ky * D;
            for (int d = 0; d < D; ++d) {
                acc[d] += w * V[v_off + d];
            }
        }
    }

    for (int d = 0; d < D; ++d) {
        OUT[out_base + d] = acc[d];
    }
}
