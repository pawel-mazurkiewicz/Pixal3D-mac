"""Port of cumesh.remove_small_connected_components + unify_face_orientations.

Source: CuMesh/src/clean_up.cu :: ``remove_small_connected_components(float min_area)``
and ``unify_face_orientations()``.

These are the cleanup ops to_glb runs between simplification stages.  We
were missing them — running the cumesh-port pipeline without them leaves
~60K dust components in the mesh and inconsistent winding, both of which
defeat the downstream chart segmentation.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces].astype(np.float64)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return (np.linalg.norm(cross, axis=1) * 0.5).astype(np.float32)


def _face_graph_csr(faces: np.ndarray):
    """CSR for the manifold face-adjacency graph."""
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
    return rows, cols


def remove_small_connected_components(
    vertices: np.ndarray, faces: np.ndarray, min_area: float = 1e-5,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop face-graph connected components below ``min_area`` total area.

    Matches CuMesh::remove_small_connected_components(min_area).  The
    component area is the sum of its face areas (signed magnitudes — i.e.
    triangle areas, not signed by winding).

    Default ``min_area = 1e-5`` matches o_voxel.postprocess.to_glb.

    Parameters
    ----------
    vertices : (V, 3) float
    faces    : (F, 3) int
    min_area : float
        Area threshold in world units.  Components with total area below
        this are dropped.
    verbose  : bool

    Returns
    -------
    new_vertices, new_faces — compacted (vertex array may shrink, faces
    re-indexed).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    F = faces.shape[0]
    if F == 0:
        return vertices, faces

    rows, cols = _face_graph_csr(faces)
    data = np.ones(len(rows), dtype=np.int8)
    graph = coo_matrix((data, (rows, cols)), shape=(F, F)).tocsr()
    n_comp, labels = connected_components(graph, directed=False)

    face_areas = _face_areas(vertices, faces)
    comp_area = np.zeros(n_comp, dtype=np.float64)
    np.add.at(comp_area, labels, face_areas)

    keep_comp = comp_area >= min_area
    keep_face = keep_comp[labels]
    if verbose:
        print(f"[remove_small_cc] {n_comp:,} components → "
              f"kept {int(keep_comp.sum()):,} (area ≥ {min_area}); "
              f"faces: {F:,} → {int(keep_face.sum()):,}", flush=True)

    if not keep_face.any():
        return np.empty((0, 3), dtype=vertices.dtype), \
               np.empty((0, 3), dtype=faces.dtype)
    new_f = faces[keep_face]
    used = np.unique(new_f.reshape(-1))
    remap = -np.ones(vertices.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    new_v = vertices[used]
    new_f = remap[new_f].astype(faces.dtype)
    return new_v.astype(vertices.dtype), new_f


def unify_face_orientations(
    vertices: np.ndarray, faces: np.ndarray, verbose: bool = False,
) -> np.ndarray:
    """Flip faces to have consistent winding across each connected component.

    Matches CuMesh::unify_face_orientations.  For each connected component
    of the manifold face graph, do a BFS: keep the seed face's winding,
    then propagate.  An adjacent face has consistent winding with the seed
    iff the shared edge appears in *opposite* directions in the two faces.
    If not, flip the face.

    Returns ``new_faces`` with the same shape as input.  Vertex array
    unchanged.
    """
    F = faces.shape[0]
    if F == 0:
        return faces

    # Build directed edges per face, with face/local-edge attribution.
    f64 = faces.astype(np.int64)
    # Edge in face's order: (v0->v1, v1->v2, v2->v0).
    src = f64[:, [0, 1, 2]].reshape(-1)
    dst = f64[:, [1, 2, 0]].reshape(-1)
    face_of_corner = np.repeat(np.arange(F, dtype=np.int64), 3)

    # Manifold pair lookup: for each unordered edge, the two faces sharing it.
    key_lo = np.minimum(src, dst)
    key_hi = np.maximum(src, dst)
    key = (key_lo.astype(np.int64) << 32) | key_hi.astype(np.int64)
    order = np.argsort(key, kind="stable")
    skey = key[order]
    sface = face_of_corner[order]
    ssrc = src[order]
    sdst = dst[order]
    diffs = np.diff(skey) == 0
    starts = np.where(diffs)[0]
    fa = sface[starts]
    fb = sface[starts + 1]
    sa_src = ssrc[starts]
    sa_dst = sdst[starts]
    sb_src = ssrc[starts + 1]
    sb_dst = sdst[starts + 1]
    # Consistent iff fa traverses src→dst opposite to fb (e.g. fa: 1→2, fb: 2→1).
    consistent = (sa_src == sb_dst) & (sa_dst == sb_src)
    # The edge graph used for BFS.
    # We store for each edge whether it's "consistent" or "needs flip".

    # Build adjacency CSR.
    n_pairs = fa.shape[0]
    edge_consistent = consistent.astype(np.int8)
    # Symmetric: store both directions for BFS.
    adj_src = np.concatenate([fa, fb])
    adj_dst = np.concatenate([fb, fa])
    adj_consistent = np.concatenate([edge_consistent, edge_consistent])
    # Sort by src to build offsets.
    osort = np.argsort(adj_src, kind="stable")
    adj_src_s = adj_src[osort]
    adj_dst_s = adj_dst[osort]
    adj_consistent_s = adj_consistent[osort]
    offsets = np.searchsorted(adj_src_s, np.arange(F + 1))

    # BFS face flip propagation.
    # flip_sign[f] = +1 if face keeps winding, -1 if flipped.
    flip_sign = np.zeros(F, dtype=np.int8)
    # BFS each unvisited face as root with sign +1.
    from collections import deque
    n_flipped = 0
    for seed in range(F):
        if flip_sign[seed] != 0:
            continue
        flip_sign[seed] = 1
        q = deque([seed])
        while q:
            u = q.popleft()
            su = int(flip_sign[u])
            for j in range(offsets[u], offsets[u + 1]):
                v = int(adj_dst_s[j])
                if flip_sign[v] != 0:
                    continue
                # If edge u-v is consistent, v keeps su sign;
                # if not, v flips relative to su.
                cons = adj_consistent_s[j]
                sv = su if cons else -su
                flip_sign[v] = sv
                if sv == -1:
                    n_flipped += 1
                q.append(v)

    # Apply flips: for faces with sign -1, swap two vertex indices to invert winding.
    new_faces = faces.copy()
    flip_mask = flip_sign == -1
    new_faces[flip_mask] = new_faces[flip_mask][:, [0, 2, 1]]
    if verbose:
        print(f"[unify_orient] flipped {n_flipped:,}/{F:,} faces", flush=True)
    return new_faces

