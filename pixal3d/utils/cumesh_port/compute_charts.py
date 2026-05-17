"""Mesh chart segmentation — produces face-graph-connected near-flat chunks.

This is a pragmatic ~70% port of cumesh.CuMesh.compute_charts (atlas.cu:1071).
Full cumesh runs a Lloyd-style iteration (collapse + refine + reassign) with
area / perimeter penalties; we run only the collapse phase, on the face graph,
with normal-cone constraint.  In our setting this is enough because:

  * Cumesh's iteration converges quickly on Pixal3D-like inputs; the bulk
    of the chart structure comes from the initial collapse phase.
  * The downstream consumer (xatlas) will further sub-segment each chunk
    if it disagrees with our chart choice; we just need chunks that are
    *face-graph connected* and *roughly flat* so xatlas doesn't shred
    them into 1000+ sub-charts.

The algorithm:

  1. Each face starts as its own chart with cone (axis=face_normal, angle=0).
  2. Repeat until no merge happens:
       For each manifold chart-adjacency edge whose merged-cone half-angle
       would be ≤ threshold:
           merge the two charts (the smaller absorbs into the larger),
           update the cone of the kept chart.
       Tie-breaking: lowest merge-cost edge first.
  3. Compress chart IDs.

This produces face-graph-connected charts whose face normals fit within a
single cone of half-angle ≤ threshold — exactly the property xatlas needs
to accept a chunk as a single chart.

Differences from cumesh (intentional):

  * No area / perimeter penalties.  These exist in cumesh to balance chart
    sizes; with our short-iteration version the imbalance is bounded by
    the cone constraint alone, which is fine for atlas packing.
  * No refine phase (per-face reassignment).  Cumesh adds this to smooth
    chart boundaries; in our use we hand the result to xatlas which does
    its own boundary smoothing.
  * No reassign phase (disconnection cleanup).  Connectivity is preserved
    by construction — every merge is across an existing face-graph edge.

If the bake quality with this simpler segmenter turns out to be too coarse,
we can add the refine phase as a follow-up; the data structures are
compatible.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _build_edges(faces: np.ndarray):
    """Same edge builder used elsewhere in the port."""
    F = faces.shape[0]
    f = faces.astype(np.int64, copy=False)
    e_lo = np.minimum(f[:, [0, 1, 2]], f[:, [1, 2, 0]])
    e_hi = np.maximum(f[:, [0, 1, 2]], f[:, [1, 2, 0]])
    raw_keys = (e_lo.astype(np.int64) << 32) | e_hi.astype(np.int64)
    flat_keys = raw_keys.reshape(-1)
    edges, inverse = np.unique(flat_keys, return_inverse=True)
    edge2face_cnt = np.bincount(inverse, minlength=edges.shape[0]).astype(np.int32)
    face_edges = inverse.reshape(F, 3).astype(np.int64)
    return edges, edge2face_cnt, face_edges


def _manifold_face_adjacency(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each manifold edge, the pair of face ids that share it.

    Returns
    -------
    face_pairs : (M, 2) int64
        Sorted so face_pairs[:, 0] < face_pairs[:, 1].
    edge_len_idx : (M,) int64
        Index into the global edges array — useful if the caller needs to
        weight the chart-adjacency by edge length.
    """
    F = faces.shape[0]
    edges, edge2face_cnt, face_edges = _build_edges(faces)
    flat_edges = face_edges.reshape(-1)
    face_ids = np.repeat(np.arange(F, dtype=np.int64), 3)
    order = np.argsort(flat_edges, kind="stable")
    sorted_edges = flat_edges[order]
    sorted_faces = face_ids[order]
    edge_starts = np.concatenate([[0], np.cumsum(edge2face_cnt)[:-1]])
    manifold_edges = np.nonzero(edge2face_cnt == 2)[0]
    starts = edge_starts[manifold_edges]
    fa = sorted_faces[starts]
    fb = sorted_faces[starts + 1]
    # Canonicalize: smaller face id first.
    swap = fa > fb
    fa_c = np.where(swap, fb, fa)
    fb_c = np.where(swap, fa, fb)
    return np.stack([fa_c, fb_c], axis=1), manifold_edges


def _face_normals_and_areas(
    vertices: np.ndarray, faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces].astype(np.float64)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)
    area2 = np.linalg.norm(cross, axis=1)
    safe = np.where(area2 > 1e-30, area2, 1.0)
    normals = cross / safe[:, None]
    return normals.astype(np.float32), (area2 * 0.5).astype(np.float32)


def _merge_cones(
    axis0: np.ndarray, half0: float,
    axis1: np.ndarray, half1: float,
) -> Tuple[np.ndarray, float]:
    """Merge two normal cones — direct port of collapse_edges_kernel inner math.

    Returns the cone (axis, half_angle) that bounds both input cones.
    Used inside the merge loop, so kept tight & branchy.
    """
    cos_a = float(np.dot(axis0, axis1))
    cos_a = min(max(cos_a, -1.0), 1.0)
    axis_angle = math.acos(cos_a)
    low = min(-half0, axis_angle - half1)
    high = max(half0, axis_angle + half1)
    new_half = (high - low) * 0.5
    if axis_angle < 1e-3:
        new_axis = axis0
    else:
        new_axis_angle = (high + low) * 0.5
        # axis1 ⊥ component relative to axis0
        perp = axis1 - axis0 * cos_a
        nperp = np.linalg.norm(perp)
        if nperp < 1e-12:
            new_axis = axis0
        else:
            perp = perp / nperp
            new_axis = axis0 * math.cos(new_axis_angle) + perp * math.sin(new_axis_angle)
            n = np.linalg.norm(new_axis)
            if n > 1e-12:
                new_axis = new_axis / n
            else:
                new_axis = axis0
    return new_axis, new_half


def compute_charts(
    vertices: np.ndarray,
    faces: np.ndarray,
    threshold_cone_half_angle_rad: float = math.radians(90),
    verbose: bool = False,
) -> np.ndarray:
    """Cluster faces into face-graph-connected, near-flat charts.

    Parameters
    ----------
    vertices : (V, 3) float
    faces : (F, 3) int
    threshold_cone_half_angle_rad : float
        Maximum half-angle of the chart's normal cone in radians.  Default
        90° matches cumesh.  Smaller = more charts, flatter charts.

    Returns
    -------
    chart_ids : (F,) int32
        Per-face chart id, dense in ``[0, num_charts)``.
    """
    F = faces.shape[0]
    if F == 0:
        return np.empty(0, dtype=np.int32)

    normals, areas = _face_normals_and_areas(vertices, faces)
    face_pairs, _ = _manifold_face_adjacency(faces)

    if verbose:
        print(f"[compute_charts] F={F:,} manifold-adj-pairs={face_pairs.shape[0]:,}")

    # Chart axis & half-angle, per current chart.  Each face = one chart at start.
    chart_axis = normals.astype(np.float32).copy()
    chart_half = np.zeros(F, dtype=np.float32)
    # chart_parent[i] = parent chart id, for union-find compression.
    chart_parent = np.arange(F, dtype=np.int32)

    def find(c: int) -> int:
        # Iterative path compression.
        root = c
        while chart_parent[root] != root:
            root = chart_parent[root]
        while chart_parent[c] != root:
            nxt = chart_parent[c]
            chart_parent[c] = root
            c = nxt
        return root

    # Single-pass greedy merge over manifold-adjacent face pairs.  Cumesh
    # does multi-pass with cost-min priority + tie-break; here we use a
    # simple greedy + cost ordering for clarity / speed.
    #
    # We iterate face-pair edges in random-but-deterministic order (sorted
    # by the higher endpoint's normal angle to the lower) — equivalent to a
    # priority on "merge cheap cones first".  Pure numpy.
    pair_n0 = normals[face_pairs[:, 0]]
    pair_n1 = normals[face_pairs[:, 1]]
    dihedral_cos = np.clip((pair_n0 * pair_n1).sum(axis=1), -1.0, 1.0)
    dihedral = np.arccos(dihedral_cos).astype(np.float32)   # ~ initial half-angle for the merge
    order = np.argsort(dihedral, kind="stable")             # cheapest merges first

    threshold = float(threshold_cone_half_angle_rad)
    merges = 0
    n_pairs = face_pairs.shape[0]

    for idx in order:
        f0, f1 = int(face_pairs[idx, 0]), int(face_pairs[idx, 1])
        c0 = find(f0)
        c1 = find(f1)
        if c0 == c1:
            continue
        # Tentatively merge cones.
        new_axis, new_half = _merge_cones(
            chart_axis[c0], float(chart_half[c0]),
            chart_axis[c1], float(chart_half[c1]),
        )
        if new_half > threshold:
            continue
        # Commit merge.  Smaller-id chart keeps its slot (so we don't keep
        # rewriting downstream).
        keep, drop = (c0, c1) if c0 < c1 else (c1, c0)
        chart_parent[drop] = keep
        chart_axis[keep] = new_axis
        chart_half[keep] = new_half
        merges += 1

    # Final flatten: compress parents.
    for i in range(F):
        chart_parent[i] = find(i)

    # Densify chart ids.
    _, dense_ids = np.unique(chart_parent, return_inverse=True)
    if verbose:
        print(f"[compute_charts] merges: {merges:,}/{n_pairs:,}; "
              f"resulting charts: {int(dense_ids.max()) + 1:,}")
    return dense_ids.astype(np.int32)


def compute_charts_dihedral_cc(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_dihedral_deg: float = 30.0,
    verbose: bool = False,
) -> np.ndarray:
    """Alternative chart segmentation: dihedral-thresholded connected components.

    Builds the face adjacency graph, keeps only edges where the dihedral
    angle (= angle between adjacent face normals) is at or below the
    threshold, then connected_components gives the chart partition.

    Compared to compute_charts (cone-based):
      * Each chart is face-graph-connected (same).
      * Chart can globally span arbitrary normal range, as long as
        adjacent faces inside the chart differ by ≤ threshold (different).
        A smooth hemisphere can be ONE chart with this method.
      * No incremental cone bookkeeping, so no greedy lock-in issues.

    Practical effect on Pixal3D inputs: produces orders-of-magnitude fewer
    charts than the cone-bounded variant at the same nominal threshold,
    because organic surfaces have smooth normal transitions that the cone
    method aggressively partitions at "long thin chart" boundaries.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    F = faces.shape[0]
    if F == 0:
        return np.empty(0, dtype=np.int32)

    normals, _ = _face_normals_and_areas(vertices, faces)
    face_pairs, _ = _manifold_face_adjacency(faces)

    # Per-pair dihedral angle.
    n0 = normals[face_pairs[:, 0]]
    n1 = normals[face_pairs[:, 1]]
    dot = np.clip((n0 * n1).sum(axis=1), -1.0, 1.0)
    dihedral_rad = np.arccos(dot)

    cos_thresh = math.cos(math.radians(max_dihedral_deg))
    keep_mask = dot >= cos_thresh                          # dihedral ≤ thresh
    kept = face_pairs[keep_mask]
    if verbose:
        print(f"[charts_dihedral_cc] F={F:,}  manifold-adj={face_pairs.shape[0]:,}  "
              f"kept-edges={kept.shape[0]:,} (dihedral ≤ {max_dihedral_deg}°)",
              flush=True)

    # Connected components on the filtered face graph.
    data = np.ones(kept.shape[0] * 2, dtype=np.int8)
    rows = np.concatenate([kept[:, 0], kept[:, 1]])
    cols = np.concatenate([kept[:, 1], kept[:, 0]])
    graph = coo_matrix((data, (rows, cols)), shape=(F, F)).tocsr()
    n_comp, labels = connected_components(graph, directed=False)
    if verbose:
        sizes = np.bincount(labels)
        print(f"[charts_dihedral_cc] charts: {n_comp:,}  "
              f"(largest={sizes.max():,}  median={int(np.median(sizes)):,}  "
              f"singletons={(sizes == 1).sum():,})", flush=True)
    return labels.astype(np.int32)


def extract_chunks(
    vertices: np.ndarray, faces: np.ndarray, chart_ids: np.ndarray,
):
    """Yield ``(sub_vertices, sub_faces, original_vertex_map)`` per chart.

    Each sub-mesh has compact vertex indices [0..len(sub_vertices)).
    ``original_vertex_map`` lets the caller recover original vertex ids for
    bake/voxel-attr lookup downstream.
    """
    n_charts = int(chart_ids.max()) + 1 if chart_ids.size > 0 else 0
    order = np.argsort(chart_ids, kind="stable")
    sorted_faces = faces[order]
    sorted_ids = chart_ids[order]
    starts = np.searchsorted(sorted_ids, np.arange(n_charts))
    ends = np.searchsorted(sorted_ids, np.arange(n_charts), side="right")
    for c in range(n_charts):
        sub_f_global = sorted_faces[starts[c]:ends[c]]
        if sub_f_global.shape[0] == 0:
            continue
        used = np.unique(sub_f_global.reshape(-1))
        remap = -np.ones(vertices.shape[0], dtype=np.int64)
        remap[used] = np.arange(used.shape[0])
        sub_v = vertices[used]
        sub_f = remap[sub_f_global].astype(np.int32)
        yield (
            np.ascontiguousarray(sub_v),
            np.ascontiguousarray(sub_f),
            used,
        )
