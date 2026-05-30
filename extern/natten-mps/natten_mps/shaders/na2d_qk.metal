// na2d_qk — windowed neighborhood Q·K dot product.
//
// Matches natten 0.17.5 CPU naive semantics (BSD-3 / Apache-2.0 mix in
// upstream).  No scaling applied here; caller scales + softmaxes externally.
//
// Tensor layout:  [B, H, X, Y, D]   (heads-second)
//   q, k contiguous, fp32 or fp16.
// Output layout:  [B, H, X, Y, K*K]  fp32 attention logits.
//
// Boundary handling: natten "edge-shift-clamp" — window is always K tokens
// wide; corner queries don't center, they anchor.  Closed form from
// natten_cpu_commons.h:
//
//   nbr = K / 2                       (integer division: 9/2 = 4)
//   start = max(i - nbr, 0)
//         + (i + nbr >= L) * (L - i - nbr - 1)
//
// One thread per output (b*H, x, y) coordinate; loops over the 81 neighbors.
// 8×8 threadgroup = 64 threads = one SIMD-group on Apple Silicon.
//
// TODO(v2): coalesce K-tile loads through threadgroup memory; specialize
// for ks=9, D=64; add MPP matmul2d path for M5+.

// Helpers (na_window_start) defined in shared.metal — concatenated first
// by natten_mps._compile.

kernel void na2d_qk(
    device const float* Q     [[buffer(0)]],   // [B, H, X, Y, D]
    device const float* K     [[buffer(1)]],   // [B, H, X, Y, D]
    device float*       OUT   [[buffer(2)]],   // [B, H, X, Y, KS*KS]
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

    // Stride (in element units) for [B, H, X, Y, D] contiguous tensor:
    //   d_stride       = 1
    //   y_stride       = D
    //   x_stride       = Y * D
    //   h_stride       = X * Y * D
    //   b_stride       = H * X * Y * D
    // Combined batch*head index `bh` covers both, with stride (X*Y*D).
    const int spatial_stride = X * Y * D;
    const int row_stride     = Y * D;

    const int q_base   = bh * spatial_stride + x * row_stride + y * D;
    const int k_bh_base = bh * spatial_stride;

    // Output stride: [B, H, X, Y, KS*KS] -> last dim = KS*KS
    const int KK = KS * KS;
    const int out_base = bh * (X * Y * KK) + x * (Y * KK) + y * KK;

    for (int ki = 0; ki < KS; ++ki) {
        const int kx = sx_per + ki * DIL;
        for (int kj = 0; kj < KS; ++kj) {
            const int ky = sy_per + kj * DIL;
            const int k_off = k_bh_base + kx * row_stride + ky * D;

            // fp32 accumulator (matches natten CPU naive fp32 path)
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) {
                acc += Q[q_base + d] * K[k_off + d];
            }
            OUT[out_base + ki * KS + kj] = acc;
        }
    }
}
