"""Iterative cone-collapse chart segmentation — faithful port of cumesh.compute_charts.

Source: CuMesh/src/atlas.cu :: compute_charts (line 1071) with the
default global_iterations=1, refine_iterations=0 used by
o_voxel.postprocess.to_glb (TRELLIS.2).  That setting means: skip the
refine phase entirely and just loop the collapse phase until convergence.

Collapse-phase algorithm:

    Initial: chart[f] = f for all faces (F charts).
    For each chart, cone = (axis = face_normal, half_angle = 0).

    Loop until no merge happens:
        1. Build chart-pair adjacency: for each manifold edge (f0, f1),
           if chart[f0] != chart[f1], record (chart[f0], chart[f1]).
           Aggregate shared boundary length per pair.
        2. For each pair, compute merge cost:
             new_half = merged_cone_half_angle(cone[c0], cone[c1])
             area_pen = area_penalty_weight * (area[c0] + area[c1])
             perim_pen = perim_area_ratio_weight * (new_perim^2 / new_area)
             cost = new_half + area_pen + perim_pen
        3. For each chart, find min-cost incident edge.
        4. Collapse pair (c0, c1) IFF:
             cost(c0, c1) <= threshold_cone_half_angle_rad
             AND (c0, c1) is c0's min-cost edge
             AND (c0, c1) is c1's min-cost edge.
           (Mutual matching prevents simultaneous conflicting merges.)
        5. Compress chart IDs.

The "mutual min" matching is what makes flat regions (large patches with
consistent normals) consolidate aggressively while corners stay split —
without it the greedy version locks in early and produces hundreds of
thousands of micro-charts.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _build_edges_and_face_pairs(
    vertices: np.ndarray, faces: np.ndarray,
):
    """Returns (face_pairs (M,2), edge_lengths (M,))."""
    F = faces.shape[0]
    f64 = faces.astype(np.int64, copy=False)
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
    elo = (sk[starts] >> 32).astype(np.int64)
    ehi = (sk[starts] & 0xFFFFFFFF).astype(np.int64)
    edge_len = np.linalg.norm(
        vertices[elo].astype(np.float64) - vertices[ehi].astype(np.float64), axis=1,
    ).astype(np.float64)
    return np.stack([fa, fb], axis=1), edge_len


def _face_normals_areas(v: np.ndarray, f: np.ndarray):
    tri = v[f].astype(np.float64)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    n = np.cross(e1, e2)
    a2 = np.linalg.norm(n, axis=1)
    safe = np.where(a2 > 1e-30, a2, 1.0)
    return (n / safe[:, None]).astype(np.float64), (a2 * 0.5).astype(np.float64)


def _merged_cone_half_angles_vec(
    axis0: np.ndarray, half0: np.ndarray, axis1: np.ndarray, half1: np.ndarray,
) -> np.ndarray:
    """Vectorized cone-merge half-angle.  Matches collapse_edges_kernel math.

    Inputs are (N, 3) for axes and (N,) for half-angles in radians.
    Returns (N,) merged half-angles.
    """
    cos_a = np.clip((axis0 * axis1).sum(axis=1), -1.0, 1.0)
    axis_angle = np.arccos(cos_a)
    low = np.minimum(-half0, axis_angle - half1)
    high = np.maximum(half0, axis_angle + half1)
    return (high - low) * 0.5


def _merge_cone_axis(
    axis0: np.ndarray, half0: float, axis1: np.ndarray, half1: float,
) -> np.ndarray:
    """Compute merged cone axis for two-cone merge (per-pair, scalar).

    Match collapse_edges_kernel; used inside the merge inner loop.
    """
    cos_a = float(np.clip(np.dot(axis0, axis1), -1.0, 1.0))
    axis_angle = math.acos(cos_a)
    if axis_angle < 1e-3:
        return axis0.copy()
    low = min(-half0, axis_angle - half1)
    high = max(half0, axis_angle + half1)
    new_axis_angle = (high + low) * 0.5
    perp = axis1 - axis0 * cos_a
    n = np.linalg.norm(perp)
    if n < 1e-12:
        return axis0.copy()
    perp = perp / n
    new_axis = axis0 * math.cos(new_axis_angle) + perp * math.sin(new_axis_angle)
    nn = np.linalg.norm(new_axis)
    if nn < 1e-12:
        return axis0.copy()
    return new_axis / nn


def compute_charts_iterative(
    vertices: np.ndarray,
    faces: np.ndarray,
    threshold_cone_half_angle_rad: float = math.radians(90.0),
    area_penalty_weight: float = 0.1,
    perimeter_area_ratio_weight: float = 1e-4,
    max_iterations: int = 50,
    verbose: bool = False,
) -> np.ndarray:
    """Iterative cone-collapse chart segmentation.

    Returns dense ``chart_ids[f]`` for each face.  Defaults match cumesh's
    compute_charts (with refine_iterations=0, global_iterations=1, which
    o_voxel.postprocess.to_glb uses).

    The iterative collapse runs to convergence (no more merges happen), so
    ``max_iterations`` is just a safety cap.
    """
    F = faces.shape[0]
    if F == 0:
        return np.empty(0, dtype=np.int32)

    normals, areas = _face_normals_areas(vertices, faces)
    face_pairs, edge_lens = _build_edges_and_face_pairs(vertices, faces)
    M = face_pairs.shape[0]
    if verbose:
        print(f"[iterative] F={F:,} manifold-adj={M:,}", flush=True)

    # Initial: each face is own chart.
    chart_axis = normals.copy()                     # (F, 3) float64
    chart_half = np.zeros(F, dtype=np.float64)      # (F,) radians
    chart_area = areas.copy()                       # (F,)
    # Perimeter = sum of boundary edge lengths.  Initial perimeter for each
    # face = sum of all 3 edge lengths (the triangle is a degenerate "chart"
    # with 3-edge boundary).  We'll track this.
    chart_perim = np.zeros(F, dtype=np.float64)
    # Reduce by face_pairs (each shared edge contributes to BOTH face perims).
    np.add.at(chart_perim, face_pairs[:, 0], edge_lens)
    np.add.at(chart_perim, face_pairs[:, 1], edge_lens)
    # Plus boundary-only edges: total triangle perim = sum of 3 edges, but
    # the face_pairs only includes manifold edges.  Compute total triangle
    # perim from face geometry.
    tri = vertices[faces].astype(np.float64)
    a = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    b = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    c = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    chart_perim = a + b + c                         # Full triangle perimeter

    # parent[c] = c's parent in union-find (chart_id collapse map).
    parent = np.arange(F, dtype=np.int64)

    def find_arr(arr):
        """In-place path compression: arr[i] = root of i."""
        # Iterate until convergence.  Since we always set parent[child] = root
        # after find, two passes suffice in practice.
        for _ in range(40):
            new = parent[arr]
            if np.array_equal(new, arr):
                return arr
            arr = new
        return arr

    threshold = float(threshold_cone_half_angle_rad)

    for it in range(max_iterations):
        # Lookup current chart for each face pair.
        c0 = parent[face_pairs[:, 0]]
        c1 = parent[face_pairs[:, 1]]
        # Path-compress to roots.
        c0 = find_arr(c0)
        c1 = find_arr(c1)
        # Canonicalise: c0 < c1.
        swap = c0 > c1
        cc0 = np.where(swap, c1, c0)
        cc1 = np.where(swap, c0, c1)
        # Only keep cross-chart pairs.
        cross = cc0 != cc1
        if not cross.any():
            if verbose:
                print(f"[iterative] iter {it+1}: no cross-chart pairs", flush=True)
            break
        cc0 = cc0[cross]
        cc1 = cc1[cross]
        eln = edge_lens[cross]

        # Aggregate per unique chart-pair (sum boundary lengths between them).
        key = cc0 * (F + 1) + cc1
        uniq_key, inverse = np.unique(key, return_inverse=True)
        pair_a = (uniq_key // (F + 1)).astype(np.int64)
        pair_b = (uniq_key % (F + 1)).astype(np.int64)
        pair_blen = np.zeros(uniq_key.shape[0], dtype=np.float64)
        np.add.at(pair_blen, inverse, eln)
        E = uniq_key.shape[0]

        # Compute merged-cone half-angle, area, perim for each pair.
        new_half = _merged_cone_half_angles_vec(
            chart_axis[pair_a], chart_half[pair_a],
            chart_axis[pair_b], chart_half[pair_b],
        )
        new_area = chart_area[pair_a] + chart_area[pair_b]
        new_perim = chart_perim[pair_a] + chart_perim[pair_b] - 2.0 * pair_blen
        safe_area = np.where(new_area > 1e-30, new_area, 1.0)
        cost = (
            new_half
            + area_penalty_weight * new_area
            + perimeter_area_ratio_weight * (new_perim ** 2 / safe_area)
        )

        # Find min-cost edge per chart (pack key/value so np.minimum.reduceat works).
        # For each chart c, look up its min-cost edge.
        # Approach: build (chart, cost, edge_id) tuples for BOTH endpoints of
        # each edge, sort by chart then cost, take first per chart.
        chart_ids_all = np.concatenate([pair_a, pair_b])
        cost_all = np.concatenate([cost, cost])
        edge_ids_all = np.concatenate([np.arange(E), np.arange(E)])
        # Stable sort by (chart, cost, edge_id) — primary chart, secondary cost.
        order = np.lexsort((edge_ids_all, cost_all, chart_ids_all))
        sorted_charts = chart_ids_all[order]
        sorted_costs = cost_all[order]
        sorted_eids = edge_ids_all[order]
        # First entry per chart group = min-cost edge.
        first_in_group = np.concatenate(
            [[True], sorted_charts[1:] != sorted_charts[:-1]],
        )
        idx = np.where(first_in_group)[0]
        min_chart = sorted_charts[idx]
        min_eid = sorted_eids[idx]
        min_cost_per_chart = np.full(F, np.inf, dtype=np.float64)
        min_eid_per_chart = np.full(F, -1, dtype=np.int64)
        min_cost_per_chart[min_chart] = sorted_costs[idx]
        min_eid_per_chart[min_chart] = min_eid

        # Collapse: pair (a, b) is collapsed iff cost <= threshold AND it's
        # min-cost edge for BOTH a and b.
        eid = np.arange(E)
        mutual = (
            (min_eid_per_chart[pair_a] == eid) &
            (min_eid_per_chart[pair_b] == eid) &
            (cost <= threshold)
        )
        if not mutual.any():
            if verbose:
                print(f"[iterative] iter {it+1}: no mutual-min collapse "
                      f"(min cost = {cost.min():.3f}, threshold = {threshold:.3f})",
                      flush=True)
            break

        # Apply collapses.  Multiple pairs can match in one iter; each
        # only involves one (cA, cB) pair which is mutually unique by
        # construction (a chart has at most one min-cost edge).
        merges = 0
        for eid_idx in np.where(mutual)[0]:
            a = int(pair_a[eid_idx])
            b = int(pair_b[eid_idx])
            # Re-resolve via find (in case prior merges in this batch
            # already moved a or b).
            while parent[a] != a:
                a = int(parent[a])
            while parent[b] != b:
                b = int(parent[b])
            if a == b:
                continue
            # Merge b → a (smaller id wins, canonical).
            keep, drop = (a, b) if a < b else (b, a)
            # Compute merged cone axis for the kept slot.
            new_axis_v = _merge_cone_axis(
                chart_axis[keep], float(chart_half[keep]),
                chart_axis[drop], float(chart_half[drop]),
            )
            new_half_v = _merged_cone_half_angles_vec(
                chart_axis[keep:keep+1], chart_half[keep:keep+1],
                chart_axis[drop:drop+1], chart_half[drop:drop+1],
            )[0]
            chart_axis[keep] = new_axis_v
            chart_half[keep] = new_half_v
            chart_area[keep] = chart_area[keep] + chart_area[drop]
            chart_perim[keep] = chart_perim[keep] + chart_perim[drop] - 2.0 * pair_blen[eid_idx]
            parent[drop] = keep
            merges += 1

        if verbose:
            n_active = int((find_arr(parent.copy()) == np.arange(F)).sum())
            print(f"[iterative] iter {it+1}: merged {merges:,} pairs; "
                  f"~active charts: {n_active:,}", flush=True)

        if merges == 0:
            break

    # Final compression.
    final = find_arr(parent.copy())
    _, dense = np.unique(final, return_inverse=True)
    if verbose:
        print(f"[iterative] final n_charts = {int(dense.max()) + 1:,}",
              flush=True)
    return dense.astype(np.int32)
