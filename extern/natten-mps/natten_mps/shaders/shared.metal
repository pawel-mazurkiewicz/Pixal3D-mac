// Shared helpers for natten-mps kernels.  Concatenated into the JIT-compiled
// shader source before each kernel file.

#include <metal_stdlib>
using namespace metal;

// natten's "edge-shift-clamp" window start.  See natten_cpu_commons.h:71-92.
//
//   nbr_eff = (KS / 2) * DIL
//   start   = max(i - nbr_eff, 0)
//           + (i + nbr_eff >= L) * (L - i - nbr_eff - 1)
//
// The result is the first key index of an `KS`-wide (post-dilation) window
// centered on query `i` along an axis of length `L`, with edges anchored
// rather than clipped — so the window is ALWAYS exactly KS tokens wide.
//
// Caller passes `nbr_eff` (= nbr * dilation) so the function is dilation-
// agnostic.
inline int na_window_start(int i, int L, int nbr_eff) {
    int s = max(i - nbr_eff, 0);
    if (i + nbr_eff >= L) {
        s += (L - i - nbr_eff - 1);
    }
    return s;
}
