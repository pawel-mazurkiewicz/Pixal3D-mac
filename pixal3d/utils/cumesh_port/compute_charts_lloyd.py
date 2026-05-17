"""Lloyd-iteration chart refinement — final piece of the cumesh compute_charts port.

Source: refine_charts_kernel in CuMesh/src/atlas.cu:681.

Per-face reassignment rule (from the CUDA kernel):

    For each face f:
        For each candidate chart c in (current_chart(f), neighbor_charts(f)):
            geo_sim   = axis(c) · normal(f)        # cosine similarity, must be > 0
            smooth_sim = sum_over_edges_in_c(edge_length) * lambda_smooth
            score(c)   = geo_sim + smooth_sim
        Reassign f to argmax_c score(c) (tie-break: smaller chart id).

The chart axis is recomputed *from membership* between iterations as the
area-weighted mean of face normals.  This is the key to escaping the
greedy-lock-in problem of single-pass region growing.

We seed with METIS to give the iteration a strong starting point — without
that, starting from "each face = own chart" needs hundreds of iterations
to converge to anything useful at our face counts.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _face_normals_areas(v: np.ndarray, f: np.ndarray):
    tri = v[f].astype(np.float64)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    n = np.cross(e1, e2)
    area2 = np.linalg.norm(n, axis=1)
    safe = np.where(area2 > 1e-30, area2, 1.0)
    return (n / safe[:, None]).astype(np.float32), (area2 * 0.5).astype(np.float32)


def _manifold_face_pairs_and_edgelen(
    v: np.ndarray, f: np.ndarray,
):
    """Return (face_pairs (M,2), edge_lengths (M,))."""
    F = f.shape[0]
    f64 = f.astype(np.int64)
    e_lo = np.minimum(f64[:, [0, 1, 2]], f64[:, [1, 2, 0]])
    e_hi = np.maximum(f64[:, [0, 1, 2]], f64[:, [1, 2, 0]])
    keys = ((e_lo << 32) | e_hi).reshape(-1)
    face_corner = np.repeat(np.arange(F, dtype=np.int64), 3)
    edge_vlo_global = e_lo.reshape(-1)
    edge_vhi_global = e_hi.reshape(-1)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    sf = face_corner[order]
    diffs = np.diff(sk) == 0
    starts = np.where(diffs)[0]
    fa = sf[starts]
    fb = sf[starts + 1]

    # Decode endpoints for length.
    elo = (sk[starts] >> 32).astype(np.int64)
    ehi = (sk[starts] & 0xFFFFFFFF).astype(np.int64)
    edge_len = np.linalg.norm(
        v[elo].astype(np.float64) - v[ehi].astype(np.float64), axis=1,
    ).astype(np.float32)

    pairs = np.stack([fa.astype(np.int64), fb.astype(np.int64)], axis=1)
    return pairs, edge_len


def _build_face_graph_csr(faces):
    F = faces.shape[0]
    f64 = faces.astype(np.int64)
    e_lo = np.minimum(f64[:, [0, 1, 2]], f64[:, [1, 2, 0]])
    e_hi = np.maximum(f64[:, [0, 1, 2]], f64[:, [1, 2, 0]])
    keys = ((e_lo << 32) | e_hi).reshape(-1)
    face_corner = np.repeat(np.arange(F, dtype=np.int64), 3)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    sf = face_corner[order]
    diffs = np.diff(sk) == 0
    starts = np.where(diffs)[0]
    fa = sf[starts]
    fb = sf[starts + 1]
    rows = np.concatenate([fa, fb])
    cols = np.concatenate([fb, fa])
    sort_idx = np.argsort(rows, kind="stable")
    rows_s = rows[sort_idx]
    cols_s = cols[sort_idx]
    offsets = np.searchsorted(rows_s, np.arange(F + 1))
    return offsets.astype(np.int64), cols_s.astype(np.int64)


def _metis_seed(faces: np.ndarray, nparts: int) -> np.ndarray:
    """Initial chart assignment from balanced face-graph partition."""
    import pymetis
    offsets, adj = _build_face_graph_csr(faces)
    cuts, membership = pymetis.part_graph(
        nparts=nparts, xadj=offsets.tolist(), adjncy=adj.tolist(),
    )
    return np.asarray(membership, dtype=np.int32)


def _compute_chart_axes(
    normals: np.ndarray, areas: np.ndarray, chart_ids: np.ndarray, n_charts: int,
) -> np.ndarray:
    """Area-weighted mean normal per chart, normalized.  Returns (n_charts, 3)."""
    axis_sum = np.zeros((n_charts, 3), dtype=np.float64)
    weighted = normals.astype(np.float64) * areas.astype(np.float64)[:, None]
    np.add.at(axis_sum, chart_ids, weighted)
    norms = np.linalg.norm(axis_sum, axis=1)
    safe = np.where(norms > 1e-12, norms, 1.0)
    axes = (axis_sum / safe[:, None]).astype(np.float32)
    return axes


def _lloyd_step(
    face_normals: np.ndarray,
    chart_ids: np.ndarray,
    chart_axes: np.ndarray,
    face_pairs: np.ndarray,
    edge_lengths: np.ndarray,
    lambda_smooth: float,
) -> np.ndarray:
    """One Lloyd reassignment iteration.

    For each face, evaluate self + up-to-3 distinct neighbor charts.  Pick
    the candidate maximising (geo_sim + lambda_smooth * smooth_sim).

    Vectorised: we expand the face-pair list to (2M, 2) of (face_id,
    neighbor_chart_id) tuples (each undirected pair contributes both
    directions), aggregate smooth_sim by (face, chart) using a hash, and
    pick the best per face.
    """
    F = chart_ids.shape[0]
    M = face_pairs.shape[0]

    # Each face's three (neighbor_face, edge_length) tuples — derive from
    # the symmetric expansion of face_pairs.
    src = np.concatenate([face_pairs[:, 0], face_pairs[:, 1]])             # (2M,)
    dst = np.concatenate([face_pairs[:, 1], face_pairs[:, 0]])             # (2M,)
    elens = np.concatenate([edge_lengths, edge_lengths])                   # (2M,)

    # For each (src, dst) pair, the candidate chart from the neighbor's perspective.
    cand_chart = chart_ids[dst].astype(np.int64)
    cand_score_smooth = elens                              # to be summed per (src, chart)

    # Score self chart for each face (and its smooth contribution = 0).
    self_chart = chart_ids.astype(np.int64)

    # Compose (src, chart) key for accumulation.  We need a per-face winner;
    # to do this without a per-face hash, we sort by (src, chart) and pick
    # max within each src group.
    #
    # First, deduplicate (src, chart) and sum smooth scores.
    # Use combined int64 key: src * (max_chart + 2) + chart.
    chart_max = max(int(chart_ids.max()), int(cand_chart.max())) + 1
    keys_pair = src.astype(np.int64) * chart_max + cand_chart
    # Also add (face, self_chart) with smooth=0 as an extra row for every face.
    self_keys = np.arange(F, dtype=np.int64) * chart_max + self_chart
    keys_all = np.concatenate([keys_pair, self_keys])
    smooth_all = np.concatenate([cand_score_smooth, np.zeros(F, dtype=np.float32)])
    src_all = np.concatenate([src.astype(np.int64), np.arange(F, dtype=np.int64)])
    chart_all = np.concatenate([cand_chart, self_chart])

    # Sort by key, group identical keys, sum smooth, keep one (src, chart).
    sort_idx = np.argsort(keys_all, kind="stable")
    keys_s = keys_all[sort_idx]
    smooth_s = smooth_all[sort_idx]
    src_s = src_all[sort_idx]
    chart_s = chart_all[sort_idx]

    # Run starts for unique (src, chart) pairs.
    breaks = np.concatenate([[True], keys_s[1:] != keys_s[:-1]])
    starts = np.where(breaks)[0]
    ends = np.concatenate([starts[1:], [len(keys_s)]])
    smooth_per_pair = np.add.reduceat(smooth_s, starts)
    src_per_pair = src_s[starts]
    chart_per_pair = chart_s[starts]

    # Score = chart_axes[chart] · face_normals[src] + lambda * smooth.
    geo = (chart_axes[chart_per_pair] * face_normals[src_per_pair]).sum(axis=1)
    valid = geo > 0
    score = geo + lambda_smooth * smooth_per_pair

    # Within each src, pick argmax score (with chart-id tie-break for stability).
    score_invalid_floor = -1e9
    score_masked = np.where(valid, score, score_invalid_floor)

    # Group by src and find max.  src_per_pair is already sorted by src
    # because we sorted by keys_all = src * chart_max + chart_id with chart_max
    # giving src primacy.
    src_breaks = np.concatenate([[True], src_per_pair[1:] != src_per_pair[:-1]])
    src_starts = np.where(src_breaks)[0]
    src_ends = np.concatenate([src_starts[1:], [len(src_per_pair)]])
    new_chart_ids = chart_ids.copy()
    for i in range(len(src_starts)):
        s, e = src_starts[i], src_ends[i]
        best = np.argmax(score_masked[s:e])
        if score_masked[s + best] > score_invalid_floor:
            new_chart_ids[int(src_per_pair[s])] = int(chart_per_pair[s + best])
    return new_chart_ids


def compute_charts_lloyd(
    vertices: np.ndarray,
    faces: np.ndarray,
    initial_chunks: int = 5000,
    iterations: int = 20,
    smooth_strength: float = 1.0,
    verbose: bool = False,
) -> np.ndarray:
    """METIS-seeded Lloyd iteration for chart segmentation.

    Returns dense ``chart_ids[f]`` for each face.

    Notes
    -----
    * ``initial_chunks`` is the METIS seed count — actual chart count
      will likely converge to a *smaller* number as similar charts
      merge during refinement.
    * ``smooth_strength`` matches cumesh's smooth_strength (default 1.0
      in cumesh.compute_charts).  Higher = stronger edge-length bias
      toward keeping faces with their neighbours.
    """
    F = faces.shape[0]
    if F == 0:
        return np.empty(0, dtype=np.int32)
    if verbose:
        print(f"[lloyd] F={F:,}", flush=True)

    normals, areas = _face_normals_areas(vertices, faces)
    face_pairs, edge_lens = _manifold_face_pairs_and_edgelen(vertices, faces)
    if verbose:
        print(f"[lloyd] manifold-adj M={face_pairs.shape[0]:,}", flush=True)

    chart_ids = _metis_seed(faces, initial_chunks)
    if verbose:
        print(f"[lloyd] METIS seed: {len(np.unique(chart_ids))} initial chunks",
              flush=True)

    for it in range(iterations):
        # Compress chart ids each iteration (some may have emptied).
        _, chart_ids = np.unique(chart_ids, return_inverse=True)
        chart_ids = chart_ids.astype(np.int32)
        n_charts = int(chart_ids.max()) + 1
        chart_axes = _compute_chart_axes(normals, areas, chart_ids, n_charts)

        new_chart_ids = _lloyd_step(
            normals, chart_ids, chart_axes,
            face_pairs, edge_lens, smooth_strength,
        )
        n_changed = int((new_chart_ids != chart_ids).sum())
        if verbose:
            print(f"[lloyd] iter {it+1}: n_charts={n_charts}, "
                  f"reassigned={n_changed:,}", flush=True)
        if n_changed == 0:
            break
        chart_ids = new_chart_ids

    _, chart_ids = np.unique(chart_ids, return_inverse=True)
    chart_ids = chart_ids.astype(np.int32)
    if verbose:
        print(f"[lloyd] final n_charts={int(chart_ids.max())+1:,}", flush=True)
    return chart_ids
