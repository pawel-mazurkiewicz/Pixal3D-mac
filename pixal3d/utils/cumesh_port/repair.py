"""Port of cumesh.CuMesh.repair_non_manifold_edges (CUDA → CPU numpy).

Source: CuMesh/src/clean_up.cu :: CuMesh::repair_non_manifold_edges() (line 787).

Why we port this *instead of* pymeshlab's ``meshing_repair_non_manifold_edges``:

pymeshlab's repair *removes* faces sitting on non-manifold edges (we observed
it dropping 8.4M → 7.3M, a 1.1M-face loss).  Every removed face creates new
boundary edges around the hole it leaves behind, manufacturing exactly the
boundary constraint that defeats downstream QEM decimation.  Cumesh's repair
takes the opposite approach: it splits vertices that sit on non-manifold edges
into multiple copies (one per manifold-connected fan around that vertex),
preserving every face.  Result: a cleanly-manifold mesh with the same face
count as input and *fewer* boundary edges.

Algorithm (face-corner union-find):

    1. Build edges + per-edge face counts (same as in fill_holes.py).
    2. Manifold edges are those with exactly 2 incident faces.  For each,
       identify the two faces and the two shared vertices, and emit two
       corner-pairs ``(3 * face_a + local_a, 3 * face_b + local_b)`` — one
       per shared vertex.
    3. Treat each face-corner (3·F total) as a node.  Union-find on the
       corner-pair edges from step 2: two corners are unified iff they
       lie on the same manifold-connected fan around a shared vertex.
    4. Connected-component label = new vertex id.  Each non-manifold
       vertex is naturally split into one new vertex per manifold-fan.
    5. New face = the three labels of its three corners.  New vertex
       position = the original vertex position of any corner in the group
       (they all sample the same original vertex by construction).

Implementation note:

The CUDA reference uses an iterative ``hook + path-compress`` union-find on
the GPU.  We use ``scipy.sparse.csgraph.connected_components`` which runs in
single-pass linear time on CPU for inputs of our size (~25M corners,
~24M corner-pairs).  Equivalent result, ~2–5 s on this hardware.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _build_edges(
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same edge-builder as fill_holes — kept private to avoid a circular import."""
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


def repair_non_manifold_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split vertices on non-manifold edges so the result is locally manifold.

    Preserves all input faces.  Output vertex count may exceed input vertex
    count (some vertices are duplicated; coincident positions are intended,
    not an error).

    Parameters
    ----------
    vertices : (V, 3) float
        Vertex positions.
    faces : (F, 3) int
        Triangle indices.
    verbose : bool
        If True, print pre/post statistics.

    Returns
    -------
    new_vertices : (V', 3) float
        ``V' >= V`` — equal when the input is already locally manifold,
        larger when vertices needed splitting.
    new_faces : (F, 3) int
        Same shape and triangle count as input; only vertex indices change.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    F = int(faces.shape[0])
    V = int(vertices.shape[0])
    if F == 0:
        return vertices, faces

    edges, edge2face_cnt, face_edges = _build_edges(faces)
    n_manifold_edges = int((edge2face_cnt == 2).sum())
    n_boundary = int((edge2face_cnt == 1).sum())
    n_nonman = int((edge2face_cnt > 2).sum())
    if verbose:
        print(f"[repair] V={V:,} F={F:,} edges={edges.shape[0]:,} "
              f"(boundary={n_boundary:,} manifold={n_manifold_edges:,} "
              f"non-manifold={n_nonman:,})")
        if n_nonman:
            nme_mult = edge2face_cnt[edge2face_cnt > 2]
            mult_max = int(nme_mult.max())
            buckets = []
            for k in range(3, min(mult_max, 8) + 1):
                c = int((nme_mult == k).sum())
                if c:
                    buckets.append(f"{k}f={c:,}")
            if mult_max > 8:
                c8p = int((nme_mult > 8).sum())
                buckets.append(f"9+f={c8p:,}")
            print(f"[repair] NME multiplicity: " + " ".join(buckets) +
                  f" (max={mult_max})")
        deg = np.bincount(faces.reshape(-1).astype(np.int64), minlength=V)
        deg_nonzero = deg[deg > 0]
        if deg_nonzero.size:
            print(f"[repair] input vertex face-degree: "
                  f"min={int(deg_nonzero.min())} "
                  f"median={int(np.median(deg_nonzero))} "
                  f"max={int(deg.max())} "
                  f"mean={float(deg.mean()):.2f}")

    # For each manifold edge, get the two faces that share it.  We invert
    # face_edges (F, 3) → edge_to_face_pairs.  Since edge2face_cnt[e] == 2,
    # exactly two (face, local_edge) entries land on each manifold edge.
    manifold_edges = np.nonzero(edge2face_cnt == 2)[0]
    face_ids_per_corner = np.repeat(np.arange(F, dtype=np.int64), 3)        # (3F,)
    local_edge_per_corner = np.tile(np.arange(3, dtype=np.int64), F)        # (3F,)
    flat_edge_ids = face_edges.reshape(-1)                                  # (3F,)

    # Sort by edge id so the two corners on each manifold edge become adjacent.
    order = np.argsort(flat_edge_ids, kind="stable")
    sorted_edge_ids = flat_edge_ids[order]
    sorted_face_ids = face_ids_per_corner[order]
    sorted_local_e = local_edge_per_corner[order]

    # Find positions where each edge id begins; the run length comes from
    # edge2face_cnt.  Since cnt==2 for manifold edges, each manifold edge
    # contributes exactly two consecutive positions in the sorted arrays.
    edge_starts = np.concatenate([[0], np.cumsum(edge2face_cnt)[:-1]])
    man_edge_starts = edge_starts[manifold_edges]

    # face_a / face_b: the two faces sharing each manifold edge.
    face_a = sorted_face_ids[man_edge_starts]
    face_b = sorted_face_ids[man_edge_starts + 1]
    local_e_a = sorted_local_e[man_edge_starts]
    local_e_b = sorted_local_e[man_edge_starts + 1]

    # The two shared vertices are the endpoints of the edge.  In face_a,
    # edge `local_e_a` is the (v[local_e_a], v[(local_e_a+1) % 3]) pair.
    # Similarly for face_b.  We need to match these endpoints by their
    # *vertex id*, because face_a and face_b may use different local
    # indices for the same vertex.
    faces64 = faces.astype(np.int64, copy=False)
    # local_e_a==0 → corners (0,1); local_e_a==1 → (1,2); local_e_a==2 → (2,0).
    a_corner_first  = local_e_a                                            # 0,1, or 2
    a_corner_second = (local_e_a + 1) % 3
    b_corner_first  = local_e_b
    b_corner_second = (local_e_b + 1) % 3

    M = manifold_edges.shape[0]
    rows = np.arange(M, dtype=np.int64)
    a_v_first  = faces64[face_a, a_corner_first]
    a_v_second = faces64[face_a, a_corner_second]
    b_v_first  = faces64[face_b, b_corner_first]
    b_v_second = faces64[face_b, b_corner_second]

    # Match a_first to whichever b endpoint shares its vertex id.  The
    # other endpoint is the second shared vertex.  Vectorized:
    match_first = a_v_first == b_v_first
    # When match_first: pair (a_first ↔ b_first, a_second ↔ b_second).
    # Otherwise: pair (a_first ↔ b_second, a_second ↔ b_first).
    pair1_a_local = a_corner_first
    pair1_b_local = np.where(match_first, b_corner_first, b_corner_second)
    pair2_a_local = a_corner_second
    pair2_b_local = np.where(match_first, b_corner_second, b_corner_first)

    # Corner-graph edges: each is a unification of a (face, local_corner)
    # pair across the manifold edge.  Encode corner ids as 3*face+local.
    corner_a1 = 3 * face_a + pair1_a_local
    corner_b1 = 3 * face_b + pair1_b_local
    corner_a2 = 3 * face_a + pair2_a_local
    corner_b2 = 3 * face_b + pair2_b_local

    # Build sparse adjacency on 3F nodes; run connected_components.
    n_corners = 3 * F
    rows_all = np.concatenate([corner_a1, corner_a2])
    cols_all = np.concatenate([corner_b1, corner_b2])
    data = np.ones(rows_all.shape[0], dtype=np.int8)
    graph = coo_matrix((data, (rows_all, cols_all)), shape=(n_corners, n_corners)).tocsr()
    # connected_components is symmetric — we don't need to add the transpose.
    n_new_v, corner_labels = connected_components(graph, directed=False)

    # Output vertex positions: for each label group, the position is the
    # *original* vertex position of any corner in the group (they all index
    # the same input vertex by construction of the corner graph).  Pick the
    # first occurrence per label.
    first_corner_per_label = np.full(n_new_v, -1, dtype=np.int64)
    # Walk corners in order; record first hit per label.
    # Faster vectorised form: use unique with return_index.
    _, first_idx = np.unique(corner_labels, return_index=True)
    first_corner_per_label[corner_labels[first_idx]] = first_idx

    # Each corner's original vertex id = faces[face_id, local_id].
    orig_vert_per_corner = faces64.reshape(-1)
    new_vertex_orig_id = orig_vert_per_corner[first_corner_per_label]
    new_vertices = vertices[new_vertex_orig_id]

    # New faces: replace each corner with its label.
    new_faces = corner_labels.reshape(F, 3).astype(faces.dtype)

    if verbose:
        delta = n_new_v - V
        print(f"[repair] new V: {n_new_v:,} (was {V:,}, +{delta:,}); "
              f"F preserved at {F:,}")
        # Vertex split-factor histogram: how many output vertices did each
        # input vertex expand into?  fan_size>1 means the input vertex was
        # incident to a non-manifold join and got split.
        fan_counts = np.bincount(new_vertex_orig_id.astype(np.int64),
                                 minlength=V)
        used_fans = fan_counts[fan_counts > 0]
        n_unchanged = int((used_fans == 1).sum())
        n_split = int((used_fans > 1).sum())
        max_fan = int(used_fans.max()) if used_fans.size else 0
        fan_buckets = []
        for k in range(2, min(max_fan, 8) + 1):
            c = int((used_fans == k).sum())
            if c:
                fan_buckets.append(f"x{k}={c:,}")
        if max_fan > 8:
            c8p = int((used_fans > 8).sum())
            fan_buckets.append(f"x9+={c8p:,}")
        print(f"[repair] split factors: unchanged={n_unchanged:,} "
              f"split={n_split:,} max_fan={max_fan} "
              + (("(" + " ".join(fan_buckets) + ")") if fan_buckets else ""))
        post_edges, post_e2fc, _ = _build_edges(new_faces)
        post_boundary = int((post_e2fc == 1).sum())
        post_manifold = int((post_e2fc == 2).sum())
        post_nme = int((post_e2fc > 2).sum())
        print(f"[repair] post edges={post_edges.shape[0]:,} "
              f"(boundary={post_boundary:,} "
              f"manifold={post_manifold:,} "
              f"non-manifold={post_nme:,})")
        d_boundary = post_boundary - n_boundary
        print(f"[repair] boundary edge delta: "
              f"{n_boundary:,} -> {post_boundary:,} ({d_boundary:+,})")

    return new_vertices.astype(vertices.dtype), new_faces
