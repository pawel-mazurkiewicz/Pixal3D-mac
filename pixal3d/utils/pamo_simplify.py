"""CPU/MPS port of cumesh's pamo parallel-QEM mesh simplification.

Replaces the Metal port's ``cumesh.metal_backend.MtlMesh.simplify_step``.
The Metal port (shipped as a precompiled .metallib + .so binary in the
trellis-mac venv) appears to skip or break one of the per-collapse
validity checks documented in ``CuMesh/src/simplify.cu`` — empirically it
drives the mesh BELOW the requested face count and punches missing-face
chunks into real surface (see SESSION_NOTES probe on the
``1_img_fdg_cap.npz`` checkpoint).

This module ports the upstream pamo algorithm to PyTorch (``device='cpu'``
by default; one-line switch to ``'mps'`` once the CPU path is validated).
Algorithm reference:

  * ``CuMesh/src/simplify.cu``  (591 lines of CUDA)
  * https://github.com/SarahWeiii/pamo  (paper + reference implementation)

The CuMesh public API we replicate is the Python wrapper at
``cumesh/metal_backend.py``:

    def simplify(self, target_num_faces, verbose=False, options={}):
        # while num_faces > target, call simplify_step with escalating
        # threshold; each step does the 6-stage pipeline below

Six-stage pipeline (one ``simplify_step``):

  1. ``get_vertex_face_adjacency`` — CSR (vertex → incident faces)
  2. ``get_edges`` + ``get_boundary_info`` — edge list, per-vertex boundary flag
  3. ``get_qem`` — per-vertex 4×4 QEM (sum of plane quadrics from incident faces)
  4. ``get_edge_collapse_cost`` — per-edge cost using QEM + λ_edge_length
     + λ_skinny, with **winding-flip rejection** (``simplify.cu:128``:
     ``if (old_normal.dot(new_normal) < 0.0f) return false; // invalid``)
  5. ``propagate_cost`` — per-face = min of its 3 corner edges' costs
     (this is how pamo gets parallelism: faces vote for their cheapest
     incident edge, so no two batch-applied collapses can touch the same
     face)
  6. ``collapse_edges`` — for each face, if all 3 corners agree on the
     same propagated cost and it's below ``threshold``, apply the collapse
     in this batch.  No vertex-conflicts between collapses by construction.

Outer loop (``simplify``):

    thresh = 1e-8
    while num_faces > target:
        new_v, new_f = simplify_step(λ_edge_length, λ_skinny, thresh)
        if (cur - new_f) / cur < 1e-2:
            thresh *= 10
        cur = new_f

PHASE 0 STATUS: scaffolding only, all functions are stubs returning
unchanged input.  Wired into the bridge via
``_patch_simplify`` in ``o_voxel_native_export.py`` so calling
``MtlMesh.simplify(target)`` routes here instead of the buggy Metal kernel.
With everything stubbed the bridge runs equivalent to ``--skip-decimation``
(mesh stays at input face count); this confirms hookup before we fill in
phases 1-6.
"""

from __future__ import annotations

import numpy as np
import torch


__all__ = ["simplify", "simplify_step"]


# Defaults match cumesh/metal_backend.py:MtlMesh.simplify so behaviour
# matches the upstream API even without options being passed in.
_DEFAULT_THRESH = 1e-8
_DEFAULT_LAMBDA_EDGE_LENGTH = 1e-2
_DEFAULT_LAMBDA_SKINNY = 1e-3


def simplify(mesh_backend, target_num_faces: int, verbose: bool = False,
             options: dict | None = None) -> None:
    """Drop-in replacement for ``MtlMesh.simplify(target_num_faces, ...)``.

    Reads the current mesh from the backend, runs the threshold-escalating
    loop of ``simplify_step``, and writes the result back via
    ``mesh_backend.init(verts, faces)``.

    Mirrors ``cumesh.metal_backend.MtlMesh.simplify``: keep iterating
    while face count > target; if a step removes < 1% of faces (slow
    progress), multiply ``thresh`` by 10 to admit more collapses.  Safety
    cap on ``thresh`` prevents pathological spin on meshes that simply
    can't reach target (e.g. all-boundary or all-non-manifold).
    """
    import time
    options = options or {}
    assert isinstance(target_num_faces, int) and target_num_faces > 0

    cur_faces = int(mesh_backend.num_faces)
    if cur_faces <= target_num_faces:
        if verbose:
            print(
                f"[pamo_simplify] already at/below target "
                f"({cur_faces:,} ≤ {target_num_faces:,}); skip",
                flush=True,
            )
        return

    thresh = float(options.get("thresh", _DEFAULT_THRESH))
    lambda_edge_length = float(options.get("lambda_edge_length",
                                           _DEFAULT_LAMBDA_EDGE_LENGTH))
    lambda_skinny = float(options.get("lambda_skinny", _DEFAULT_LAMBDA_SKINNY))
    thresh_cap = float(options.get("thresh_cap", 1.0))

    verts, faces = mesh_backend.read()
    # CPU path for now; Phase 7 can promote to MPS once validated.
    verts = verts.detach().cpu()
    faces = faces.detach().cpu()

    t_start = time.perf_counter()
    if verbose:
        print(
            f"[pamo_simplify] start: V={verts.shape[0]:,}, "
            f"F={faces.shape[0]:,}, target={target_num_faces:,}, "
            f"thresh={thresh:.1e} (cap {thresh_cap:.1e})",
            flush=True,
        )

    num_face = int(faces.shape[0])
    step = 0
    while True:
        step += 1
        new_v, new_f = simplify_step(
            verts, faces, lambda_edge_length, lambda_skinny, thresh,
            verbose=verbose,
        )
        new_num_face = int(new_f.shape[0])

        if new_num_face <= target_num_faces:
            verts, faces = new_v, new_f
            if verbose:
                print(
                    f"[pamo_simplify] target reached at step {step}: "
                    f"F={new_num_face:,}", flush=True,
                )
            break

        del_faces = num_face - new_num_face
        if del_faces <= 0 or del_faces / max(num_face, 1) < 1e-2:
            thresh *= 10.0
            if verbose:
                print(
                    f"[pamo_simplify] step {step}: slow progress "
                    f"({del_faces:,} faces removed); thresh ↑ {thresh:.1e}",
                    flush=True,
                )
            if thresh > thresh_cap:
                if verbose:
                    print(
                        f"[pamo_simplify] threshold cap reached "
                        f"({thresh:.1e} > {thresh_cap:.1e}); stopping at "
                        f"F={new_num_face:,}", flush=True,
                    )
                verts, faces = new_v, new_f
                break

        num_face = new_num_face
        verts, faces = new_v, new_f

    if verbose:
        print(
            f"[pamo_simplify] done in {time.perf_counter()-t_start:.2f}s "
            f"({step} steps): V={verts.shape[0]:,}, F={faces.shape[0]:,}",
            flush=True,
        )

    mesh_backend.init(verts.contiguous(), faces.contiguous())
    # Phase 0: no-op.  Subsequent phases fill in the loop.


def get_vertex_face_adjacency(faces: torch.Tensor, num_vertices: int
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """CSR mapping vertex → incident face IDs.

    Mirrors ``CuMesh::get_vertex_face_adjacency`` (``connectivity.cu``).

    Args:
        faces: (F, 3) int tensor of triangle vertex indices.
        num_vertices: V.

    Returns:
        offsets: (V+1,) int64.  ``offsets[v+1] - offsets[v]`` faces incident on v.
        indices: (3F,) int64.   Face IDs grouped by vertex, in offset order.
    """
    F = int(faces.shape[0])
    if F == 0:
        return (
            torch.zeros(num_vertices + 1, dtype=torch.int64, device=faces.device),
            torch.zeros(0, dtype=torch.int64, device=faces.device),
        )

    vert_flat = faces.reshape(-1).to(torch.int64)                          # (3F,)
    face_flat = torch.arange(F, dtype=torch.int64, device=faces.device)
    face_flat = face_flat.repeat_interleave(3)                             # (3F,)

    order = torch.argsort(vert_flat, stable=True)
    sorted_verts = vert_flat[order]
    sorted_faces = face_flat[order]

    counts = torch.bincount(sorted_verts, minlength=num_vertices)
    offsets = torch.zeros(num_vertices + 1, dtype=torch.int64, device=faces.device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return offsets, sorted_faces


def get_edges(faces: torch.Tensor) -> dict:
    """Undirected edge list with face counts and corner→edge mapping.

    Mirrors ``CuMesh::get_edges`` (``connectivity.cu``).

    Args:
        faces: (F, 3) int tensor.

    Returns:
        dict with keys:
          ``edges``           (E, 2) int64 — (lo, hi) endpoints per edge.
          ``edge_face_count`` (E,)   int64 — how many faces share each edge.
                                            1 = boundary, 2 = manifold, ≥3 = non-manifold.
          ``edge_of_corner``  (F, 3) int64 — for each face corner i ∈ {0,1,2},
                                            the edge id of the directed half-edge
                                            (faces[f, i] → faces[f, (i+1)%3]).
    """
    F = int(faces.shape[0])
    if F == 0:
        return {
            "edges":           torch.zeros((0, 2), dtype=torch.int64, device=faces.device),
            "edge_face_count": torch.zeros(0, dtype=torch.int64, device=faces.device),
            "edge_of_corner":  torch.zeros((0, 3), dtype=torch.int64, device=faces.device),
        }

    f64 = faces.to(torch.int64)
    src = f64[:, [0, 1, 2]].reshape(-1)                                    # (3F,)
    dst = f64[:, [1, 2, 0]].reshape(-1)                                    # (3F,)

    # Encode undirected edge as int64 key: (lo << 32) | hi.  Works for vertex
    # counts up to 2^32 which is well beyond anything we'll see.
    key_lo = torch.minimum(src, dst)
    key_hi = torch.maximum(src, dst)
    key = (key_lo << 32) | key_hi                                          # (3F,)

    unique_keys, inverse = torch.unique(key, return_inverse=True)
    E = int(unique_keys.shape[0])

    edge_face_count = torch.bincount(inverse, minlength=E)
    edge_lo = (unique_keys >> 32)
    edge_hi = unique_keys & ((1 << 32) - 1)
    edges = torch.stack([edge_lo, edge_hi], dim=1)                         # (E, 2)
    edge_of_corner = inverse.reshape(F, 3)                                 # (F, 3)
    return {
        "edges": edges,
        "edge_face_count": edge_face_count,
        "edge_of_corner": edge_of_corner,
    }


def get_boundary_info(edges: torch.Tensor, edge_face_count: torch.Tensor,
                      num_vertices: int) -> torch.Tensor:
    """Per-vertex boundary flag.

    Mirrors ``CuMesh::get_boundary_info`` (``connectivity.cu``).  A vertex
    is on the mesh boundary iff it's an endpoint of any boundary edge
    (an edge belonging to exactly one face).

    Args:
        edges:           (E, 2) int64.
        edge_face_count: (E,)  int64.
        num_vertices:    V.

    Returns:
        (V,) bool — True if vertex lies on the boundary.
    """
    if edges.shape[0] == 0:
        return torch.zeros(num_vertices, dtype=torch.bool, device=edges.device)
    bnd_mask = edge_face_count == 1
    bnd_verts = edges[bnd_mask].reshape(-1).to(torch.int64)
    vert_is_boundary = torch.zeros(num_vertices, dtype=torch.bool, device=edges.device)
    vert_is_boundary[bnd_verts] = True
    return vert_is_boundary


def get_qem(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Per-vertex QEM matrix from sum of unit-plane quadrics on incident faces.

    Mirrors ``CuMesh::get_qem_kernel`` (``simplify.cu:35-58``):

        for each face f incident on v:
            n = ((v1 - v0) × (v2 - v0)).normalized()
            d = -(n · v0)
            qem_v.add_plane({n.x, n.y, n.z, d})

    Note this is **unit-normalized, not area-weighted** — standard
    Garland-Heckbert.  Computing per-face then scatter-adding is
    algebraically identical to the per-vertex loop in the CUDA kernel
    (each face contributes the same plane quadric to each of its 3
    vertices), and is far more vectorizable than the per-vertex loop.

    Args:
        vertices: (V, 3) float tensor.
        faces:    (F, 3) int tensor.

    Returns:
        qems: (V, 10) float64 — symmetric 4x4 QEM stored as upper triangle
              in row-major order:
              ``[Q00, Q01, Q02, Q03, Q11, Q12, Q13, Q22, Q23, Q33]``
              where ``Q = sum(p ⊗ p^T)`` over incident planes p = (n, d).
    """
    V = int(vertices.shape[0])
    F = int(faces.shape[0])
    if F == 0:
        return torch.zeros((V, 10), dtype=torch.float64, device=vertices.device)

    v0 = vertices[faces[:, 0]].to(torch.float64)
    v1 = vertices[faces[:, 1]].to(torch.float64)
    v2 = vertices[faces[:, 2]].to(torch.float64)

    # Face normal (non-unit) then normalize.  EPS guards against degenerate
    # faces (which have ‖n‖ ≈ 0) — the resulting plane is meaningless but
    # contributes nothing significant to the sum since coefficients stay small.
    n = torch.linalg.cross(v1 - v0, v2 - v0, dim=1)                        # (F, 3)
    n_norm = torch.linalg.vector_norm(n, dim=1, keepdim=True).clamp_min(1e-30)
    n_unit = n / n_norm                                                    # (F, 3)
    d = -(n_unit * v0).sum(dim=1, keepdim=True)                            # (F, 1)
    plane = torch.cat([n_unit, d], dim=1)                                  # (F, 4)

    a, b, c, dd = plane[:, 0], plane[:, 1], plane[:, 2], plane[:, 3]
    # Upper-triangle packed plane quadric q = p p^T (10 entries).
    qem_face = torch.stack([
        a * a, a * b, a * c, a * dd,
        b * b, b * c, b * dd,
        c * c, c * dd,
        dd * dd,
    ], dim=1)                                                              # (F, 10)

    # Scatter-add: each face contributes its plane quadric to all 3 of its
    # vertices' QEMs.  index_add_ is the canonical scatter-add in PyTorch.
    qems = torch.zeros((V, 10), dtype=torch.float64, device=vertices.device)
    for corner in range(3):
        verts_at_corner = faces[:, corner].to(torch.int64)
        qems.index_add_(0, verts_at_corner, qem_face)
    return qems


def _pad_vertex_face_table(vf_offsets: torch.Tensor, vf_indices: torch.Tensor,
                           num_vertices: int, max_degree: int
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad per-vertex face list to a fixed length for vectorized gather.

    Returns:
        padded_vf:  (V, max_degree) int64 — face IDs, ``-1`` in unused slots.
        full_degree:(V,)             int64 — true degree of each vertex (un-clamped).
                                              Edges incident on vertices with
                                              degree > max_degree must be rejected
                                              (cost=inf) because we can't see
                                              all their incident faces here.
    """
    V = num_vertices
    device = vf_offsets.device
    degree = vf_offsets[1:] - vf_offsets[:-1]                              # (V,)
    capped = torch.minimum(degree, torch.tensor(max_degree, dtype=degree.dtype, device=device))
    row_idx = torch.arange(max_degree, dtype=torch.int64, device=device).unsqueeze(0)
    valid = row_idx < capped.unsqueeze(1)                                  # (V, max_degree)
    base = vf_offsets[:-1].unsqueeze(1) + row_idx                          # (V, max_degree)
    base_safe = torch.where(valid, base, torch.zeros_like(base))
    padded = torch.where(valid, vf_indices[base_safe],
                         torch.full((), -1, dtype=torch.int64, device=device))
    return padded, degree


def build_vertex_vertex_adjacency(faces: torch.Tensor, num_vertices: int
                                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """CSR mapping vertex → sorted unique neighbor vertex IDs.

    For each vertex v, the list of unique v' such that (v, v') is an edge
    in the mesh.  Used by the link-condition check in
    ``get_edge_collapse_cost`` to reject collapses that would create
    non-manifold edges (extra common neighbors of e0 and e1).

    Args:
        faces:        (F, 3) int tensor.
        num_vertices: V.

    Returns:
        nb_offsets: (V+1,) int64 — ``nb_offsets[v+1] - nb_offsets[v]`` is
                                    the unique-neighbor count of v.
        nb_indices: (sum_v deg(v),) int64 — neighbor IDs grouped by source
                                            vertex, sorted within each group.
    """
    F = int(faces.shape[0])
    device = faces.device
    if F == 0:
        return (
            torch.zeros(num_vertices + 1, dtype=torch.int64, device=device),
            torch.zeros(0, dtype=torch.int64, device=device),
        )
    f64 = faces.to(torch.int64)
    # All 6 directed corner-edges per face: (a,b), (b,c), (c,a), (b,a), (c,b), (a,c).
    src = torch.cat([f64[:, 0], f64[:, 1], f64[:, 2],
                     f64[:, 1], f64[:, 2], f64[:, 0]])                     # (6F,)
    dst = torch.cat([f64[:, 1], f64[:, 2], f64[:, 0],
                     f64[:, 0], f64[:, 1], f64[:, 2]])                     # (6F,)

    # Dedup via packed key.  Then unpack back to (src, dst) — torch.unique
    # returns sorted keys, so src is monotone non-decreasing and dst is
    # sorted within each src group, exactly the CSR layout we want.
    stride = num_vertices + 1
    keys = src * stride + dst
    keys = torch.unique(keys)
    src_u = keys // stride
    dst_u = keys %  stride

    counts = torch.bincount(src_u, minlength=num_vertices)
    nb_offsets = torch.zeros(num_vertices + 1, dtype=torch.int64, device=device)
    nb_offsets[1:] = torch.cumsum(counts, dim=0)
    return nb_offsets, dst_u


def get_edge_collapse_cost(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vf_offsets: torch.Tensor,
    vf_indices: torch.Tensor,
    edges: torch.Tensor,
    qems: torch.Tensor,
    vert_is_boundary: torch.Tensor,
    lambda_edge_length: float,
    lambda_skinny: float,
    edge_face_count: torch.Tensor | None = None,
    nb_offsets: torch.Tensor | None = None,
    nb_indices: torch.Tensor | None = None,
    max_vertex_degree: int = 24,
    max_neighbor_degree: int = 24,
    chunk_size: int = 500_000,
) -> torch.Tensor:
    """Per-edge collapse cost with winding-flip + link-condition rejection.

    Mirrors ``CuMesh::get_edge_collapse_cost_kernel`` (``simplify.cu:158-228``)
    plus an extra **link-condition** filter that CUDA upstream lacks but
    which is necessary on dense FDG-style meshes to prevent non-manifold
    edge explosion in the output (Phase 6 finding: without this check,
    pamo's NME fraction jumps from 7% to 18% after the first simplify
    call, which the downstream ``repair_non_manifold_edges`` then splits
    into visible boundary rips).

    Edge cost components:
      - **QEM cost**: ``(Q[e0] + Q[e1]).evaluate(v_new)``
      - **Edge length**: ``λ_edge · ||v1 − v0||²``
      - **Skinny**: ``λ_skinny · (mean shape metric defect) · ||e||²``

    Hard rejections (cost = ∞):
      - **Winding flip**: any incident face would invert normal after collapse.
      - **Over-degree**: either endpoint has degree > ``max_vertex_degree``.
      - **Link condition violation**: ``|N(e0) ∩ N(e1)| > edge_face_count[i]``.
        This means a and b share more common neighbors than the opposite
        corners of the shared triangle(s), so collapse would merge edges
        that were distinct into NMEs.  Only applied when
        ``edge_face_count`` and the neighbor CSR are provided.

    Boundary collapse target follows the CUDA convention:
      - both boundary or both interior:  v_new = 0.5·(v0 + v1)
      - only v0 boundary:                v_new = v0
      - only v1 boundary:                v_new = v1

    Args:
        vertices:           (V, 3) float
        faces:              (F, 3) int
        vf_offsets:         (V+1,) int (from get_vertex_face_adjacency)
        vf_indices:         (3F,)  int (from get_vertex_face_adjacency)
        edges:              (E, 2) int (from get_edges)
        qems:               (V, 10) float64 (from get_qem)
        vert_is_boundary:   (V,) bool
        lambda_edge_length: float
        lambda_skinny:      float
        edge_face_count:    (E,) int (from get_edges); enables link check
        nb_offsets:         (V+1,) int (from build_vertex_vertex_adjacency)
        nb_indices:         (E_directed,) int (from build_vertex_vertex_adjacency)
        max_vertex_degree:  cap on per-vertex face degree for padded gather
        max_neighbor_degree:cap on per-vertex vertex degree for link check
        chunk_size:         int — number of edges processed per chunk

    Returns:
        (E,) float64 collapse cost; ``+inf`` for rejected edges.
    """
    import math
    device = vertices.device
    V = int(vertices.shape[0])
    E = int(edges.shape[0])
    if E == 0:
        return torch.zeros(0, dtype=torch.float64, device=device)

    padded_vf, full_degree = _pad_vertex_face_table(
        vf_offsets, vf_indices, V, max_vertex_degree)

    # Padded vertex-vertex table for link-condition check (when enabled).
    link_check_enabled = (
        edge_face_count is not None and nb_offsets is not None
        and nb_indices is not None
    )
    if link_check_enabled:
        padded_vv, full_nb_degree = _pad_vertex_face_table(
            nb_offsets, nb_indices, V, max_neighbor_degree)

    SQRT3_TIMES_4 = 4.0 * math.sqrt(3.0)
    cost_out = torch.empty(E, dtype=torch.float64, device=device)
    verts64 = vertices.to(torch.float64)

    for start in range(0, E, chunk_size):
        end = min(start + chunk_size, E)
        chunk = slice(start, end)
        e0 = edges[chunk, 0].to(torch.int64)
        e1 = edges[chunk, 1].to(torch.int64)
        Ec = e0.shape[0]

        v0 = verts64[e0]
        v1 = verts64[e1]
        b0 = vert_is_boundary[e0]
        b1 = vert_is_boundary[e1]

        # Collapse target weight (w0 for v0)
        w0 = torch.where(
            b0 & ~b1, torch.tensor(1.0, dtype=torch.float64, device=device),
            torch.where(~b0 & b1, torch.tensor(0.0, dtype=torch.float64, device=device),
                        torch.tensor(0.5, dtype=torch.float64, device=device)),
        )
        v_new = v0 * w0.unsqueeze(1) + v1 * (1.0 - w0).unsqueeze(1)        # (Ec, 3)

        # QEM cost: v_h^T (Q[e0]+Q[e1]) v_h
        qc = qems[e0] + qems[e1]                                           # (Ec, 10)
        ph = torch.cat([v_new, torch.ones((Ec, 1), dtype=torch.float64, device=device)], dim=1)
        x, y, z, w = ph[:, 0], ph[:, 1], ph[:, 2], ph[:, 3]
        qem_cost = (
            qc[:, 0] * x * x + qc[:, 4] * y * y + qc[:, 7] * z * z + qc[:, 9] * w * w
            + 2.0 * (qc[:, 1] * x * y + qc[:, 2] * x * z + qc[:, 3] * x * w
                     + qc[:, 5] * y * z + qc[:, 6] * y * w + qc[:, 8] * z * w)
        )
        edge_length_sq = ((v1 - v0) ** 2).sum(dim=1)

        # Reject if either endpoint exceeds max_vertex_degree (we can't see all faces)
        over_degree = (full_degree[e0] > max_vertex_degree) | (full_degree[e1] > max_vertex_degree)

        # Link condition: |N(e0) ∩ N(e1)| must equal edge_face_count[i].
        # Extra common neighbors → collapse would create non-manifold edges.
        link_violation = torch.zeros(Ec, dtype=torch.bool, device=device)
        if link_check_enabled:
            over_nb = ((full_nb_degree[e0] > max_neighbor_degree)
                       | (full_nb_degree[e1] > max_neighbor_degree))
            nb0_v = padded_vv[e0]                                          # (Ec, Mn)
            nb1_v = padded_vv[e1]
            valid0 = nb0_v >= 0
            valid1 = nb1_v >= 0
            # All-pairs comparison: match[i,j,k] = (nb0_v[i,j] == nb1_v[i,k] and both valid)
            match = (nb0_v.unsqueeze(2) == nb1_v.unsqueeze(1)) \
                  & valid0.unsqueeze(2) & valid1.unsqueeze(1)
            # nb0_v entries are unique-per-row (CSR was deduplicated), so
            # number of distinct common neighbors = number of nb0 slots with any match.
            common_count = match.any(dim=2).sum(dim=1)
            link_violation = (common_count > edge_face_count[chunk]) | over_nb

        # Process incident faces — two sides: kept=e0/removed=e1, and kept=e1/removed=e0
        def process_side(kept_vid, removed_vid):
            face_ids = padded_vf[kept_vid]                                 # (Ec, M)
            slot_valid = face_ids >= 0                                     # (Ec, M)
            face_ids_safe = torch.where(slot_valid, face_ids, torch.zeros_like(face_ids))
            fv = faces[face_ids_safe].to(torch.int64)                      # (Ec, M, 3)

            # Skip faces containing the removed vertex (collapse removes them)
            removed_b = removed_vid.unsqueeze(1).unsqueeze(2)              # (Ec, 1, 1)
            contains_removed = (fv == removed_b).any(dim=2)                # (Ec, M)
            do_check = slot_valid & ~contains_removed                      # (Ec, M)

            # Old triangle positions
            ov0 = verts64[fv[..., 0]]
            ov1 = verts64[fv[..., 1]]
            ov2 = verts64[fv[..., 2]]

            # Identify which corner is the kept vertex
            kept_b = kept_vid.unsqueeze(1).unsqueeze(2)
            is_k0 = (fv[..., 0:1] == kept_b)
            is_k1 = (fv[..., 1:2] == kept_b)
            is_k2 = (fv[..., 2:3] == kept_b)

            vn = v_new.unsqueeze(1).expand(-1, padded_vf.shape[1], -1)     # (Ec, M, 3)
            nv0 = torch.where(is_k0, vn, ov0)
            nv1 = torch.where(is_k1, vn, ov1)
            nv2 = torch.where(is_k2, vn, ov2)

            old_n = torch.linalg.cross(ov1 - ov0, ov2 - ov0, dim=2)        # (Ec, M, 3)
            new_n = torch.linalg.cross(nv1 - nv0, nv2 - nv0, dim=2)
            dot = (old_n * new_n).sum(dim=2)                               # (Ec, M)
            flip = (dot < 0.0) & do_check                                  # (Ec, M)

            # Skinny: 4√3 · new_area / sum_edge_norm²
            ne0 = nv1 - nv0
            ne1 = nv2 - nv0
            ne2 = nv2 - nv1
            denom = ((ne0 ** 2).sum(dim=2) + (ne1 ** 2).sum(dim=2)
                     + (ne2 ** 2).sum(dim=2)).clamp_min(1e-30)
            new_area = 0.5 * torch.linalg.vector_norm(new_n, dim=2)
            shape_metric = SQRT3_TIMES_4 * new_area / denom
            shape_metric = shape_metric.clamp(0.0, 1.0)
            skinny_term = (1.0 - shape_metric)
            skinny_term = torch.where(do_check, skinny_term, torch.zeros_like(skinny_term))

            return flip.any(dim=1), skinny_term.sum(dim=1), do_check.sum(dim=1)

        flip0, sk0, n0 = process_side(e0, e1)
        flip1, sk1, n1 = process_side(e1, e0)
        flip_any = flip0 | flip1
        skinny_total = sk0 + sk1
        n_tri = (n0 + n1).clamp_min(1).to(torch.float64)
        skinny_avg = skinny_total / n_tri

        cost = qem_cost + lambda_edge_length * edge_length_sq \
             + lambda_skinny * skinny_avg * edge_length_sq
        inf = torch.full_like(cost, float("inf"))
        cost = torch.where(flip_any | over_degree | link_violation, inf, cost)
        cost_out[chunk] = cost

    return cost_out



def _pack_edge_cost(edge_cost: torch.Tensor) -> torch.Tensor:
    """Pack (cost_f32_bits, edge_id) into int64 so int64-amin == float-amin
    with edge_id as deterministic tiebreak.

    Mirrors ``simplify.cu:pack_key_value_positive``.  IEEE 754 bit patterns
    for non-negative float32 are monotonic (and inf > any finite), so a
    plain ``scatter_reduce_(reduce='amin')`` on the packed int64 reproduces
    CUDA's ``atomicMin`` semantics.  Non-finite costs are normalized to
    +inf before packing.
    """
    E = int(edge_cost.shape[0])
    device = edge_cost.device
    cost32 = edge_cost.to(torch.float32)
    cost32 = torch.where(torch.isfinite(cost32), cost32,
                         torch.full_like(cost32, float("inf"))).contiguous()
    cost_bits = cost32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    edge_id = torch.arange(E, dtype=torch.int64, device=device)
    return (cost_bits << 32) | edge_id


def propagate_cost(edges: torch.Tensor, edge_cost: torch.Tensor,
                   vf_offsets: torch.Tensor, vf_indices: torch.Tensor,
                   num_faces: int, chunk_size: int = 1_000_000) -> torch.Tensor:
    """Per-face min-cost vote across edges incident on its vertices.

    Mirrors ``CuMesh::propagate_cost_kernel`` (``simplify.cu:269-300``):
    for every edge i with endpoints (e0, e1), atomic-min the packed
    ``(cost_i, i)`` value into ``propagated_cost[f]`` for every face f
    incident on e0 OR e1.

    The packed int64 layout lets us use plain ``scatter_reduce_(amin)``;
    the high 32 bits are the float32 cost bits (IEEE-monotone for
    non-negative floats) and the low 32 bits are the edge id (deterministic
    tiebreak).

    Args:
        edges:         (E, 2) int.
        edge_cost:     (E,)  float64 from ``get_edge_collapse_cost``.
        vf_offsets:    (V+1,) int CSR offsets.
        vf_indices:    (3F,)  int CSR indices.
        num_faces:     F.
        chunk_size:    number of edges processed per scatter to bound memory.

    Returns:
        (F,) int64 — packed ``(cost, edge_id)``.  Slots with no incident
        edge stay at ``int64.max`` (will never match any real edge's pack
        in ``collapse_edges``).
    """
    E = int(edges.shape[0])
    device = edges.device
    if num_faces == 0:
        return torch.zeros(0, dtype=torch.int64, device=device)

    propagated = torch.full((num_faces,),
                            torch.iinfo(torch.int64).max,
                            dtype=torch.int64, device=device)
    if E == 0:
        return propagated

    packed = _pack_edge_cost(edge_cost)                                    # (E,)

    for side in (0, 1):
        for cs in range(0, E, chunk_size):
            ce = min(cs + chunk_size, E)
            v = edges[cs:ce, side].to(torch.int64)                         # (Ec,)
            deg = vf_offsets[v + 1] - vf_offsets[v]                        # (Ec,)
            total = int(deg.sum().item())
            if total == 0:
                continue
            starts = vf_offsets[v]                                         # (Ec,)
            rep_packed = torch.repeat_interleave(packed[cs:ce], deg)       # (total,)
            base = torch.repeat_interleave(starts, deg)                    # (total,)
            cum = torch.cumsum(deg, 0)
            local = (torch.arange(total, dtype=torch.int64, device=device)
                     - torch.repeat_interleave(cum - deg, deg))            # 0..deg-1 per group
            face_ids = vf_indices[base + local]                            # (total,)
            propagated.scatter_reduce_(0, face_ids, rep_packed,
                                       reduce="amin", include_self=True)
    return propagated


def collapse_edges(vertices: torch.Tensor, faces: torch.Tensor,
                   edges: torch.Tensor, edge_cost: torch.Tensor,
                   propagated_cost: torch.Tensor, threshold: float,
                   vf_offsets: torch.Tensor, vf_indices: torch.Tensor,
                   vert_is_boundary: torch.Tensor,
                   max_vertex_degree: int = 24,
                   chunk_size: int = 500_000,
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel batched edge collapse with conflict-free guarantee.

    Mirrors ``CuMesh::collapse_edges_kernel`` (``simplify.cu:341-528``).

    **Conflict resolution** (the crucial pamo property):

    An edge i "wins" iff its cost ≤ threshold AND every face incident on
    *either* endpoint of i has ``propagated_cost[f] == pack(cost_i, i)``.
    If edge i wins, then by uniqueness of the int64 amin no other edge j
    sharing a vertex with i can also win — their packs differ and only one
    can be the min on the faces they both touch.  Hence no union-find is
    needed: the winning collapses are pairwise non-conflicting by
    construction (no two collapses share a vertex or a face).

    **Application**: for each winning edge (e0, e1):

    1. Set ``vertices[e0] = v_new`` (boundary-aware midpoint, same rule as
       in ``get_edge_collapse_cost``).
    2. Remap all faces' references to e1 → e0.  Faces that contained both
       endpoints (the 2-3 shared faces) now have two equal corners and
       drop in the degenerate-face cull.
    3. Compact orphan vertices (every winning edge's e1 vanishes).

    Edges with endpoint degree > ``max_vertex_degree`` are skipped — we
    can't see all their incident faces in the padded table, so we can't
    verify the "all neighbors agree" condition.  Safe rejection: they
    just don't collapse this iteration.

    Args:
        vertices:           (V, 3) float.
        faces:              (F, 3) int.
        edges:              (E, 2) int.
        edge_cost:          (E,)  float64.
        propagated_cost:    (F,)  int64 from ``propagate_cost``.
        threshold:          float — cost cutoff.
        vf_offsets:         (V+1,) int.
        vf_indices:         (3F,)  int.
        vert_is_boundary:   (V,)  bool.
        max_vertex_degree:  vertex-degree cap for the padded neighborhood check.
        chunk_size:         number of edges processed per chunk.

    Returns:
        (new_vertices, new_faces) — face dtype preserved from input.
    """
    V = int(vertices.shape[0])
    F = int(faces.shape[0])
    E = int(edges.shape[0])
    device = vertices.device
    faces_dtype = faces.dtype

    if E == 0 or F == 0:
        return vertices.clone(), faces.clone()

    packed = _pack_edge_cost(edge_cost)                                    # (E,)
    padded_vf, full_degree = _pad_vertex_face_table(
        vf_offsets, vf_indices, V, max_vertex_degree)

    wins = torch.zeros(E, dtype=torch.bool, device=device)
    cheap = (edge_cost <= float(threshold)) & torch.isfinite(edge_cost)

    for cs in range(0, E, chunk_size):
        ce = min(cs + chunk_size, E)
        cheap_c = cheap[cs:ce]
        if not bool(cheap_c.any().item()):
            continue
        e0c = edges[cs:ce, 0].to(torch.int64)
        e1c = edges[cs:ce, 1].to(torch.int64)
        over_deg = ((full_degree[e0c] > max_vertex_degree)
                    | (full_degree[e1c] > max_vertex_degree))
        cand = cheap_c & ~over_deg
        if not bool(cand.any().item()):
            continue

        packed_c = packed[cs:ce].unsqueeze(1)                              # (Ec, 1)
        nb0 = padded_vf[e0c]                                               # (Ec, M)
        nb1 = padded_vf[e1c]
        valid0 = nb0 >= 0
        valid1 = nb1 >= 0
        nb0s = torch.where(valid0, nb0, torch.zeros_like(nb0))
        nb1s = torch.where(valid1, nb1, torch.zeros_like(nb1))
        prop0 = propagated_cost[nb0s]                                      # (Ec, M)
        prop1 = propagated_cost[nb1s]
        eq0 = (prop0 == packed_c) | ~valid0
        eq1 = (prop1 == packed_c) | ~valid1
        all_eq = eq0.all(dim=1) & eq1.all(dim=1)
        wins[cs:ce] = cand & all_eq

    if not bool(wins.any().item()):
        return vertices.clone(), faces.clone()

    # Compute v_new for winning edges (boundary-aware midpoint).
    win_idx = wins.nonzero(as_tuple=True)[0]
    e0w = edges[win_idx, 0].to(torch.int64)
    e1w = edges[win_idx, 1].to(torch.int64)
    b0 = vert_is_boundary[e0w]
    b1 = vert_is_boundary[e1w]
    one = torch.tensor(1.0, dtype=vertices.dtype, device=device)
    zero = torch.tensor(0.0, dtype=vertices.dtype, device=device)
    half = torch.tensor(0.5, dtype=vertices.dtype, device=device)
    w0 = torch.where(b0 & ~b1, one, torch.where(~b0 & b1, zero, half))
    v_new = (vertices[e0w] * w0.unsqueeze(1)
             + vertices[e1w] * (1.0 - w0).unsqueeze(1))

    new_vertices = vertices.clone()
    new_vertices[e0w] = v_new

    # Vertex remap: e1 → e0 elsewhere identity.  Non-conflict guarantee
    # ensures no duplicate destinations and no chained renames.
    vert_remap = torch.arange(V, dtype=torch.int64, device=device)
    vert_remap[e1w] = e0w
    remapped_faces = vert_remap[faces.to(torch.int64)]                     # (F, 3)

    # Drop degenerate triangles — faces with two equal corners after remap.
    # These are exactly the 2-3 faces that contained both endpoints of some
    # winning edge.
    degen = ((remapped_faces[:, 0] == remapped_faces[:, 1])
             | (remapped_faces[:, 1] == remapped_faces[:, 2])
             | (remapped_faces[:, 0] == remapped_faces[:, 2]))
    surviving = remapped_faces[~degen]

    # Compact vertices: orphan e1 endpoints disappear.
    keep_vert = torch.ones(V, dtype=torch.bool, device=device)
    keep_vert[e1w] = False
    n_keep = int(keep_vert.sum().item())
    compact = torch.empty(V, dtype=torch.int64, device=device)
    compact[keep_vert] = torch.arange(n_keep, dtype=torch.int64, device=device)
    # Dead slots are unreferenced post-remap (because their old refs went
    # through vert_remap[e1w]=e0w first), so leaving them uninitialized is
    # fine — they're never read by ``compact[surviving]``.

    new_faces = compact[surviving].to(faces_dtype)
    new_vertices = new_vertices[keep_vert]
    return new_vertices, new_faces


def simplify_step(vertices: torch.Tensor, faces: torch.Tensor,
                  lambda_edge_length: float, lambda_skinny: float,
                  threshold: float, verbose: bool = False
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """One iteration of the pamo parallel-batch QEM simplify pipeline.

    Runs all stages once:

        1. ``get_vertex_face_adjacency``
        2. ``get_edges`` + ``get_boundary_info``
        2b. ``build_vertex_vertex_adjacency`` (for link-condition check)
        3. ``get_qem``
        4. ``get_edge_collapse_cost``  (winding-flip + link-condition rejection)
        5. ``propagate_cost``           (face-vote scatter)
        6. ``collapse_edges``           (batched non-conflicting collapse)

    Args:
        vertices: (V, 3) float tensor.
        faces:    (F, 3) int tensor.
        lambda_edge_length: edge-length penalty weight.
        lambda_skinny:      skinny-triangle avoidance weight.
        threshold:          maximum allowed collapse cost this batch.
        verbose:            print per-stage timing/stats.

    Returns:
        (new_vertices, new_faces) — same dtypes/device as inputs.
    """
    import time
    V = int(vertices.shape[0])
    F = int(faces.shape[0])
    if F == 0:
        return vertices, faces

    t0 = time.perf_counter()
    vf_offsets, vf_indices = get_vertex_face_adjacency(faces, V)
    t1 = time.perf_counter()

    edge_info = get_edges(faces)
    edges = edge_info["edges"]
    edge_face_count = edge_info["edge_face_count"]
    vert_is_boundary = get_boundary_info(edges, edge_face_count, V)
    t2 = time.perf_counter()

    nb_offsets, nb_indices = build_vertex_vertex_adjacency(faces, V)
    t2b = time.perf_counter()

    qems = get_qem(vertices, faces)
    t3 = time.perf_counter()

    edge_cost = get_edge_collapse_cost(
        vertices, faces, vf_offsets, vf_indices, edges, qems,
        vert_is_boundary, lambda_edge_length, lambda_skinny,
        edge_face_count=edge_face_count,
        nb_offsets=nb_offsets, nb_indices=nb_indices,
    )
    t4 = time.perf_counter()

    propagated = propagate_cost(
        edges, edge_cost, vf_offsets, vf_indices, num_faces=F,
    )
    t5 = time.perf_counter()

    new_v, new_f = collapse_edges(
        vertices, faces, edges, edge_cost, propagated, threshold,
        vf_offsets, vf_indices, vert_is_boundary,
    )
    t6 = time.perf_counter()

    if verbose:
        E = int(edges.shape[0])
        n_finite = int(torch.isfinite(edge_cost).sum().item())
        n_cheap = int(((edge_cost <= float(threshold))
                       & torch.isfinite(edge_cost)).sum().item())
        print(
            f"[pamo_simplify] step: V {V:,}→{new_v.shape[0]:,}, "
            f"F {F:,}→{new_f.shape[0]:,}, E={E:,}, "
            f"finite={n_finite:,} cheap={n_cheap:,}, thresh={threshold:.2e}, "
            f"t=adj/{t1-t0:.2f} edges/{t2-t1:.2f} vv/{t2b-t2:.2f} qem/{t3-t2b:.2f} "
            f"cost/{t4-t3:.2f} prop/{t5-t4:.2f} collapse/{t6-t5:.2f}s",
            flush=True,
        )

    return new_v, new_f
