"""Port of cumesh.CuMesh.fill_holes (CUDA → CPU numpy).

Source algorithm: CuMesh/src/clean_up.cu :: CuMesh::fill_holes(float
max_hole_perimeter) at line 450, plus the supporting plumbing in
connectivity.cu (get_edges → get_boundary_info → get_vertex_is_manifold
→ get_manifold_boundary_adjacency → get_boundary_connected_components
→ get_boundary_loops).

The CUDA version is built around CUB primitives (DeviceSelect, SegmentedReduce,
RunLengthEncode, ExclusiveSum) because each phase is a parallel scan over
millions of edges.  At Pixal3D mesh sizes (≤ 10M faces) those scans are
trivially fast on CPU as plain numpy operations:

  CUDA primitive                          numpy equivalent
  ──────────────────────────────────────  ────────────────────────────────
  cub::DeviceSelect::If(pred)             arr[pred(arr)]
  cub::DeviceSegmentedReduce::Sum         np.add.reduceat
  cub::DeviceRunLengthEncode::Encode      np.diff + np.where
  cub::DeviceScan::ExclusiveSum           np.cumsum (then shift)
  scipy.sparse.csgraph.connected_components for the boundary-edge graph.

Why we port this *instead* of using pymeshlab's `meshing_close_holes`:
pymeshlab closes every hole regardless of size (no perimeter threshold) and
leaves new non-manifold geometry around the fill triangles, which then
defeats QEM decimation downstream — see the v2/v3 probes in scratch/ for
direct evidence.  Cumesh's algorithm is intentionally conservative: it only
fills small loops (default max_perimeter = 3e-2 in world units, matching
Pixal3D's voxel-grid pinprick scale) and uses a single centroid vertex per
loop with fan triangulation, which keeps the topology clean.

The fill triangle winding follows the canonical edge ordering (e0, e1, centroid)
where the (e0, e1) ordering comes from the sorted edge storage; downstream
shaders are configured for ``doubleSided = True`` already (see the alpha-fix
note in SESSION_NOTES.md §10), so a possible front/back mismatch on the new
fill triangles is invisible.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _build_edges(
    faces: np.ndarray, num_vertices: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sorted unique edges and per-edge face counts.

    Matches CuMesh::get_edges (connectivity.cu).  Each face contributes three
    edges; each edge is stored as a 64-bit key ``(min_v << 32) | max_v`` so
    edges shared between faces collapse to the same key.

    Returns
    -------
    edges : (E,) int64
        Encoded edge keys.  ``edge2vertices(edges[i])`` recovers (v0, v1).
    edge2face_cnt : (E,) int32
        Number of faces incident to each edge.  1 = boundary, 2 = manifold,
        >2 = non-manifold.
    face_edges : (F, 3) int32
        For each face, the indices of its three edges in ``edges``.  Not used
        by fill_holes but cheap to return for callers that want to inspect
        the topology.
    """
    F = faces.shape[0]
    # 3 edges per face.  CuMesh uses (lo << 32) | hi; we use the same so any
    # future bit-fiddling reads identically.
    f = faces.astype(np.int64, copy=False)
    e_lo = np.minimum(f[:, [0, 1, 2]], f[:, [1, 2, 0]])
    e_hi = np.maximum(f[:, [0, 1, 2]], f[:, [1, 2, 0]])
    raw_keys = (e_lo.astype(np.int64) << 32) | e_hi.astype(np.int64)
    flat_keys = raw_keys.reshape(-1)                          # (3F,)

    # Unique edges; inverse maps each (face, local_edge) slot to its edge id.
    edges, inverse = np.unique(flat_keys, return_inverse=True)
    edge2face_cnt = np.bincount(inverse, minlength=edges.shape[0]).astype(np.int32)
    face_edges = inverse.reshape(F, 3).astype(np.int32)
    return edges, edge2face_cnt, face_edges


def _decode_edge(edge_keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of the (lo << 32) | hi packing in _build_edges."""
    lo = (edge_keys >> 32).astype(np.int64)
    hi = (edge_keys & 0xFFFFFFFF).astype(np.int64)
    return lo, hi


def _manifold_boundary_vertices(
    edges: np.ndarray,
    edge2face_cnt: np.ndarray,
    bound_edge_v: np.ndarray,
    num_vertices: int,
) -> np.ndarray:
    """Return vertices allowed to connect boundary edges in cumesh.

    CuMesh does not connect boundary edges through every shared vertex.  It
    first excludes vertices that touch non-manifold edges or more than two
    boundary edges, then builds boundary-edge components through the remaining
    manifold boundary vertices.
    """
    boundary_degree = np.bincount(bound_edge_v.reshape(-1), minlength=num_vertices)
    has_non_manifold_edge = np.zeros(num_vertices, dtype=bool)
    non_manifold_edges = edge2face_cnt > 2
    if non_manifold_edges.any():
        ev0, ev1 = _decode_edge(edges[non_manifold_edges])
        has_non_manifold_edge[ev0] = True
        has_non_manifold_edge[ev1] = True

    # Source get_vertex_is_manifold allows one boundary edge, but those
    # endpoint vertices cannot connect two boundary edges.  Requiring degree 2
    # here matches the intended get_manifold_boundary_adjacency output and
    # avoids connecting open-chain endpoints to unrelated adjacency slots.
    return (
        (boundary_degree == 2)
        & ~has_non_manifold_edge
    )


def _boundary_connected_components(
    bound_edge_v: np.ndarray,
    num_vertices: int,
    manifold_boundary_vertex: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """Connected components of the manifold boundary-edge graph.

    This mirrors CuMesh::get_boundary_connected_components: each boundary edge
    starts as its own component, and components are hooked only through
    manifold boundary vertices.  Raw graph components can be branched; this
    split is what lets fill_holes recover simple loops inside those regions.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    B = bound_edge_v.shape[0]
    if B == 0:
        return 0, np.empty(0, dtype=np.int32)

    flat_vertices = bound_edge_v.reshape(-1)
    flat_edges = np.repeat(np.arange(B, dtype=np.int32), 2)
    order = np.argsort(flat_vertices, kind="stable")
    v_sorted = flat_vertices[order]
    e_sorted = flat_edges[order]

    run_breaks = np.concatenate([[True], np.diff(v_sorted) != 0])
    run_starts = np.where(run_breaks)[0]
    run_ends = np.concatenate([run_starts[1:], [len(v_sorted)]])

    rows = []
    cols = []
    for start, end in zip(run_starts, run_ends):
        vertex = int(v_sorted[start])
        if not manifold_boundary_vertex[vertex]:
            continue
        incident = np.unique(e_sorted[start:end])
        if incident.shape[0] != 2:
            continue
        a, b = int(incident[0]), int(incident[1])
        rows.extend([a, b])
        cols.extend([b, a])

    if not rows:
        return B, np.arange(B, dtype=np.int32)

    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(B, B),
    ).tocsr()
    n_comp, comp_id = connected_components(graph, directed=False)
    return int(n_comp), comp_id.astype(np.int32)


def _is_loop_component(
    bound_edge_v: np.ndarray, comp_id: np.ndarray, num_components: int,
) -> np.ndarray:
    """For each boundary connected component, is it a closed loop?

    Matches cumesh's `is_bound_conn_comp_loop_kernel`: every boundary edge in
    the component must have another boundary edge from the same component at
    both endpoints.  Open chains fail at endpoints; manifold loops pass.

    Returns a (num_components,) bool array.
    """
    if num_components == 0 or bound_edge_v.shape[0] == 0:
        return np.zeros(num_components, dtype=bool)

    # Per-edge degree contribution: each boundary edge bumps the degree of
    # each of its two endpoints by 1, in its component.  But we want a
    # per-(component,vertex) tally, not a per-vertex tally — because the
    # same vertex can appear in different components (rare but possible).
    # Build a composite key (component_id, vertex_id) and count.
    e_comp = np.repeat(comp_id, 2)                         # (2B,)
    e_vert = bound_edge_v.reshape(-1)                      # (2B,)
    # Composite key for grouping.  Vertex ids fit in ~24 bits at our sizes;
    # use int64 to avoid any collision.
    key = e_comp.astype(np.int64) * (e_vert.max() + 2) + e_vert.astype(np.int64)
    sort_idx = np.argsort(key, kind="stable")
    key_s = key[sort_idx]
    comp_s = e_comp[sort_idx]

    # Find run starts (each run = same (component, vertex) pair).
    run_breaks = np.concatenate([[True], np.diff(key_s) != 0])
    run_starts = np.where(run_breaks)[0]
    run_ends = np.concatenate([run_starts[1:], [len(key_s)]])
    run_lengths = run_ends - run_starts                    # degree per (comp, vert)
    run_comp = comp_s[run_starts]

    # A component is a loop iff no endpoint is alone inside that component.
    # CuMesh checks "count of other same-component boundaries" at each
    # endpoint, so the failing condition is run_length < 2.
    bad_mask = run_lengths < 2
    bad_comp_ids = run_comp[bad_mask]
    is_loop = np.ones(num_components, dtype=bool)
    if bad_comp_ids.size > 0:
        is_loop[bad_comp_ids] = False
    return is_loop


def fill_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_hole_perimeter: float = 3e-2,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate small boundary loops with a centroid fan.

    Direct port of CuMesh::fill_holes (clean_up.cu:450).  Only loops with
    perimeter strictly less than ``max_hole_perimeter`` are filled; larger
    holes (real features of the model) are left alone.

    Parameters
    ----------
    vertices : (V, 3) float32
        Vertex positions.  Returned array preserves dtype.
    faces : (F, 3) int32
        Triangle indices into ``vertices``.
    max_hole_perimeter : float
        Cumesh default is 3e-2 (world units), which on Pixal3D's
        normalized-to-unit-cube outputs corresponds to roughly 2× the voxel
        size — picks off exactly the pinprick holes the flexible-dual-grid
        iso-surface extractor leaves behind.
    verbose : bool
        If True, print a few stats about how many holes were detected /
        filled.  Useful when wiring this into the pipeline for the first
        time.

    Returns
    -------
    new_vertices : (V', 3) float32
        Original vertices + one centroid per filled hole.
    new_faces : (F', 3) int32
        Original faces + new fan triangles.

    Notes
    -----
    Winding: the (e0, e1) tuples in cumesh's edge storage are sorted lo-to-hi.
    The fill triangles (e0, e1, centroid) therefore inherit a winding that
    is *not* guaranteed to agree with the surrounding manifold's winding —
    matching the cumesh CUDA reference, which has the same behaviour.  The
    Pixal3D export path already sets ``doubleSided = True`` on the GLB
    material (see SESSION_NOTES.md §10), so this is visually invisible.
    """
    V = int(vertices.shape[0])
    F = int(faces.shape[0])
    if F == 0:
        return vertices, faces

    edges, edge2face_cnt, _ = _build_edges(faces, V)
    boundary_mask = edge2face_cnt == 1
    boundaries = np.nonzero(boundary_mask)[0]              # edge ids
    if verbose:
        print(f"[fill_holes] V={V:,} F={F:,} edges={edges.shape[0]:,} "
              f"boundary_edges={boundaries.shape[0]:,}")
    if boundaries.size == 0:
        return vertices, faces

    # Boundary edges → connected components through manifold boundary
    # vertices.  This is intentionally not the raw boundary graph.
    bound_edge_keys = edges[boundaries]
    bv0, bv1 = _decode_edge(bound_edge_keys)
    bound_edge_v = np.stack([bv0, bv1], axis=1).astype(np.int64)
    manifold_boundary_vertex = _manifold_boundary_vertices(
        edges, edge2face_cnt, bound_edge_v, V,
    )
    n_comp, comp_id = _boundary_connected_components(
        bound_edge_v, V, manifold_boundary_vertex,
    )
    if verbose:
        print(f"[fill_holes] manifold boundary components: {n_comp:,}")

    is_loop_comp = _is_loop_component(bound_edge_v, comp_id, n_comp)
    if verbose:
        print(f"[fill_holes] of which closed loops: {int(is_loop_comp.sum()):,}")

    # Per-edge: which component am I in, AND is my component a loop?
    edge_is_in_loop = is_loop_comp[comp_id]
    loop_edge_mask = edge_is_in_loop                       # one bool per boundary edge

    # Compute perimeter per loop component (sum of edge lengths over edges
    # whose component is a loop).  CuMesh uses segmented sum over a sorted
    # loop-id; we use np.add.at for clarity.
    edge_lengths = np.linalg.norm(
        vertices[bound_edge_v[:, 0]].astype(np.float64)
        - vertices[bound_edge_v[:, 1]].astype(np.float64),
        axis=1,
    )
    comp_perim = np.zeros(n_comp, dtype=np.float64)
    np.add.at(comp_perim, comp_id, edge_lengths)
    is_small_loop = is_loop_comp & (comp_perim < max_hole_perimeter)
    if verbose:
        print(f"[fill_holes] loops with perimeter < {max_hole_perimeter}: "
              f"{int(is_small_loop.sum()):,}")

    # Edges that we'll actually fill.
    fill_edge_mask = is_small_loop[comp_id]
    if not fill_edge_mask.any():
        return vertices, faces

    # ──────────────────────────────────────────────────────────────────────
    # Compress.  CuMesh does this via cub::DeviceSelect::Flagged + cumsum to
    # produce per-loop edge offsets and a per-edge loop-id.  In numpy:
    #   filtered_edges          : edges that pass fill_edge_mask
    #   filtered_comp_id_raw    : their (compressed-from-old-numbering)
    #                             loop id, dense [0, num_filled_loops)
    # ──────────────────────────────────────────────────────────────────────
    filtered = np.nonzero(fill_edge_mask)[0]
    f_bv = bound_edge_v[filtered]
    f_comp_id = comp_id[filtered]
    # Compress comp ids to [0, n_kept)
    kept_comp_ids, dense_comp_id = np.unique(f_comp_id, return_inverse=True)
    num_kept = int(kept_comp_ids.shape[0])

    # Centroid per loop = mean of vertex positions appearing in that loop.
    # CuMesh computes (sum of edge midpoints) / num_edges; for a closed loop
    # where each vertex appears in exactly 2 edges this equals the mean of
    # the vertex positions, which is the form we use here.
    #
    # Each filtered edge contributes both endpoints to its loop's vertex sum.
    # But each vertex appears twice (once per incident edge in a closed
    # loop), so the centroid = (sum of endpoints) / (2 * num_edges).
    sum_per_loop = np.zeros((num_kept, 3), dtype=np.float64)
    cnt_per_loop = np.zeros(num_kept, dtype=np.int64)
    np.add.at(sum_per_loop, dense_comp_id, vertices[f_bv[:, 0]].astype(np.float64))
    np.add.at(sum_per_loop, dense_comp_id, vertices[f_bv[:, 1]].astype(np.float64))
    np.add.at(cnt_per_loop, dense_comp_id, 2)
    centroids = (sum_per_loop / cnt_per_loop[:, None]).astype(vertices.dtype)

    # New triangles: (e0, e1, V + loop_id) for each filled edge.
    new_v_ids = (dense_comp_id + V).astype(faces.dtype)
    new_faces = np.stack(
        [f_bv[:, 0].astype(faces.dtype),
         f_bv[:, 1].astype(faces.dtype),
         new_v_ids],
        axis=1,
    )

    out_vertices = np.concatenate([vertices, centroids], axis=0)
    out_faces = np.concatenate([faces, new_faces], axis=0)
    if verbose:
        print(f"[fill_holes] added {centroids.shape[0]:,} new vertices "
              f"and {new_faces.shape[0]:,} new faces "
              f"(boundary edges removed: {filtered.shape[0]:,})")
    return out_vertices, out_faces
