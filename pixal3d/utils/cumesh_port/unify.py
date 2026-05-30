"""CPU port of cumesh::CuMesh::unify_face_orientations.

Faithful port of /Users/pawelma/code/ai/CuMesh/src/clean_up.cu:1191.

The Apple Silicon Metal binary that ships with ``trellis-mac`` leaves
thousands of inverted-winding edges after this call — verified
empirically by the bandaid ``_per_face_winding_fix`` in
``o_voxel_native_export.py``, which still finds ~9k conflict edges and
~1k flippable faces after 8 iterations on a clean post-pamo mesh.  These
inverted faces become "missing-surface" holes under backface culling and
account for a substantial fraction of the visible damage the user
reports on the textured GLB output.

Algorithm (CUDA reference) — three steps:

1.  ``get_flip_flags``: for each MANIFOLD edge (face_a, face_b), decide
    whether the two faces traverse the shared edge in the SAME direction
    (= ``is_flipped=1``, winding inconsistent) or OPPOSITE directions
    (``is_flipped=0``, winding consistent).
2.  Parallel hook-and-compress union-find with ORIENTATION PARITY:
    each face gets a (root, parity) pair.  Faces in the same CC are
    forced into a common winding via the parity bit; cycles in the
    manifold-edge graph that have odd total flip (Möbius-strip CCs)
    cannot all be reconciled — CUDA's atomicMin race deterministically
    picks one assignment.
3.  For each face with parity=1, swap face[1] and face[2] in place.

This port uses scipy ``connected_components`` plus per-CC BFS instead of
parallel hook-compress; the result is equivalent for non-Möbius CCs and
deterministic in BFS arrival order for Möbius cycles.

Public API
----------

unify_face_orientations(vertices, faces, verbose=False) -> faces

Input ``faces`` is NOT mutated; a freshly-flipped copy is returned.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def _build_manifold_edges(faces: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Group face corners by undirected edge key; return manifold-edge info.

    Returns
    -------
    face_a, face_b : (M,) int64
        Indices of the two faces sharing each manifold edge.
    local_a, local_b : (M,) int64
        Local corner index (0/1/2) of the shared edge in each face.
        For face_a, the shared edge runs (faces[a, local_a],
        faces[a, (local_a + 1) % 3]).  Same convention for face_b.
    edge_counts : (E,) int64
        Per-edge face count (for diagnostics / cross-checking).
    """
    faces64 = faces.astype(np.int64, copy=False)
    F = faces64.shape[0]

    # Per-corner directed edge endpoints
    corner_a = faces64                          # (F, 3) — start vertex of corner edge
    corner_b = faces64[:, [1, 2, 0]]            # (F, 3) — end vertex of corner edge
    # Undirected canonical edge key (lo, hi)
    lo = np.minimum(corner_a, corner_b)
    hi = np.maximum(corner_a, corner_b)

    # Pack as int64: lo*max+hi (use a large multiplier to avoid collisions)
    V_max = int(faces64.max()) + 1
    edge_key = lo.astype(np.int64) * np.int64(V_max) + hi.astype(np.int64)  # (F, 3)

    flat_keys = edge_key.reshape(-1)
    face_id_per_corner = np.repeat(np.arange(F, dtype=np.int64), 3)
    local_id_per_corner = np.tile(np.arange(3, dtype=np.int64), F)

    order = np.argsort(flat_keys, kind="stable")
    sorted_keys = flat_keys[order]
    sorted_face_ids = face_id_per_corner[order]
    sorted_local_ids = local_id_per_corner[order]

    # Find runs of equal key
    change = np.empty(sorted_keys.shape[0], dtype=bool)
    change[0] = True
    change[1:] = sorted_keys[1:] != sorted_keys[:-1]
    edge_starts = np.nonzero(change)[0]
    edge_counts = np.diff(np.concatenate([edge_starts, [sorted_keys.shape[0]]]))

    manifold_mask = edge_counts == 2
    man_starts = edge_starts[manifold_mask]

    face_a = sorted_face_ids[man_starts]
    face_b = sorted_face_ids[man_starts + 1]
    local_a = sorted_local_ids[man_starts]
    local_b = sorted_local_ids[man_starts + 1]

    return face_a, face_b, local_a, local_b, edge_counts


def unify_face_orientations(
    vertices: np.ndarray,
    faces: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """Apply CUDA-equivalent consistent-winding pass to a mesh.

    Parameters
    ----------
    vertices : (V, 3) float
        Unused except for shape; kept for API symmetry with the cumesh
        backend, which carries vertices alongside faces.
    faces : (F, 3) int
        Triangle indices.
    verbose : bool
        Print diagnostics: manifold edge count, conflict count, faces flipped.

    Returns
    -------
    new_faces : (F, 3) int
        Same shape and dtype as input; faces with parity=1 have local
        corners 1 and 2 swapped.
    """
    from scipy.sparse import csr_matrix

    F = int(faces.shape[0])
    if F == 0:
        return faces.copy()

    face_a, face_b, local_a, local_b, edge_counts = _build_manifold_edges(faces)
    M = face_a.shape[0]
    n_boundary = int((edge_counts == 1).sum())
    n_manifold = int((edge_counts == 2).sum())
    n_nme = int((edge_counts > 2).sum())
    if verbose:
        print(f"[unify] V={vertices.shape[0]:,} F={F:,} "
              f"edges={edge_counts.shape[0]:,} "
              f"(boundary={n_boundary:,} manifold={n_manifold:,} "
              f"non-manifold={n_nme:,})")

    if M == 0:
        if verbose:
            print("[unify] no manifold edges; nothing to unify.")
        return faces.copy()

    # is_flipped per manifold edge.
    # CUDA's direction-based check is equivalent to:
    # "do both faces START the shared edge at the same vertex?"  If yes,
    # they traverse it the same direction = inconsistent winding (=1).
    faces64 = faces.astype(np.int64, copy=False)
    a_start = faces64[face_a, local_a]
    b_start = faces64[face_b, local_b]
    is_flipped = (a_start == b_start).astype(np.int8)
    # Sanity: in the CUDA reference, the second shared vertex must agree
    # with the consistency check.  If both starts equal, both ends must
    # also equal (the shared edge is the same edge).
    if verbose:
        a_end = faces64[face_a, (local_a + 1) % 3]
        b_end = faces64[face_b, (local_b + 1) % 3]
        # In consistent winding: a_start == b_end AND a_end == b_start
        # In inconsistent:        a_start == b_start AND a_end == b_end
        good_consistent = ((a_start == b_end) & (a_end == b_start) &
                           (is_flipped == 0))
        good_inconsistent = ((a_start == b_start) & (a_end == b_end) &
                             (is_flipped == 1))
        good = good_consistent | good_inconsistent
        bad = int((~good).sum())
        if bad:
            print(f"[unify] WARNING: {bad:,} manifold edges have "
                  f"non-canonical vertex pairing (possible upstream bug)")

    # Build face-face adjacency in CSR.
    # Bidirectional: store edges (a, b) and (b, a) with the same flip bit.
    rows = np.concatenate([face_a, face_b])
    cols = np.concatenate([face_b, face_a])
    # Use flip + 1 so 0 doesn't conflict with "absent" in CSR sparsity
    edata = np.concatenate([is_flipped, is_flipped]).astype(np.int8) + 1
    graph = csr_matrix((edata, (rows, cols)), shape=(F, F))
    indptr = graph.indptr
    indices = graph.indices
    edata = graph.data  # contains is_flipped + 1

    # BFS per connected component, assigning parity.  -1 means unvisited.
    parity = np.full(F, -1, dtype=np.int8)
    n_conflicts = 0
    n_ccs = 0
    # Iterative BFS using a Python list-as-queue.  collections.deque is
    # safer for large queues but Python lists are fast for append/pop(0)
    # on small queues; we use deque to be safe on large CCs.
    from collections import deque

    for start in range(F):
        if parity[start] != -1:
            continue
        parity[start] = 0
        n_ccs += 1
        queue = deque([start])
        while queue:
            u = queue.popleft()
            pu = parity[u]
            row_start = indptr[u]
            row_end = indptr[u + 1]
            # Iterate u's manifold neighbors.
            for j in range(row_start, row_end):
                v = indices[j]
                flip_uv = int(edata[j]) - 1
                target = pu ^ flip_uv
                pv = parity[v]
                if pv == -1:
                    parity[v] = target
                    queue.append(v)
                elif pv != target:
                    n_conflicts += 1

    # n_conflicts double-counts: each conflict edge is visited from
    # both endpoints in a symmetric BFS — once from u finding v already
    # visited, and once when v tried to expand to u and saw a match.
    # Actually no — when v was enqueued u was already in the visited
    # set, so v's expansion would have CONFIRMED u's parity (no
    # conflict).  Only the "discovery-time conflict" is counted, but
    # because the adjacency is symmetric we'll see each conflict edge
    # twice (once when u sees v, once when v sees u from later BFS).
    # Halve to report edge count, not visit count.
    n_conflicts = n_conflicts // 2

    n_flip = int((parity == 1).sum())
    if verbose:
        print(f"[unify] CCs={n_ccs:,}, conflict edges={n_conflicts:,}, "
              f"faces to flip={n_flip:,} ({100.0 * n_flip / F:.1f}%)")

    if n_flip == 0:
        return faces.copy()

    new_faces = faces.copy()
    flip_mask = parity == 1
    # In-place swap of columns 1 and 2 for flipped faces
    new_faces[flip_mask, 1], new_faces[flip_mask, 2] = (
        faces[flip_mask, 2].copy(),
        faces[flip_mask, 1].copy(),
    )

    return new_faces
