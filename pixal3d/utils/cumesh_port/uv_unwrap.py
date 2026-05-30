"""End-to-end UV unwrap matching cumesh's pipeline:
   pre-segment into chunks → feed each as a separate mesh to xatlas → pack.

This is the integration entry-point used by the textured-export path.  It
replaces the cube_project / smart_project paths that produce sub-pixel
chart density on Pixal3D-class meshes.

Algorithm:
    1. Partition the face graph into N balanced connected chunks (METIS).
       This avoids the noise sensitivity that defeats normal-cone clustering
       on Pixal3D's noisy iso-surface output.
    2. Add each chunk as a separate mesh to xatlas.Atlas.
    3. xatlas does its own per-chunk segmentation + global packing.
    4. Stitch the per-chunk results back into a single (vertices, faces, uvs)
       tuple that the rest of the pipeline expects.

The trade-off this entry point makes: METIS chunks are connectivity-balanced
but not necessarily flat, so xatlas does some sub-segmentation per chunk.
On our tests this lands at ~3-10K final charts at 4096², which is the
ballpark that produces visible texture density (~hundreds of texels per
chart, not single-digit like the cube_project / smart_project paths).

Chunk count sweet spot
    Below ~500 chunks: xatlas chokes on the larger chunk size (slow / hangs).
    Above ~10000 chunks: too many singletons-ish, packing overhead climbs.
    The default 2000 is a reasonable starting point; production callers
    should tune for their input.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _build_face_graph_csr(faces: np.ndarray):
    """CSR adjacency for the manifold face graph."""
    F = faces.shape[0]
    f64 = faces.astype(np.int64, copy=False)
    e_lo = np.minimum(f64[:, [0, 1, 2]], f64[:, [1, 2, 0]])
    e_hi = np.maximum(f64[:, [0, 1, 2]], f64[:, [1, 2, 0]])
    keys = ((e_lo.astype(np.int64) << 32) | e_hi.astype(np.int64)).reshape(-1)
    face_of_corner = np.repeat(np.arange(F, dtype=np.int64), 3)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    sf = face_of_corner[order]
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


def cumesh_style_uv_unwrap(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_chunks: int = 2000,
    atlas_resolution: int = 4096,
    xatlas_max_cost: float = 2.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition mesh + per-chunk xatlas → unified atlas.

    Parameters
    ----------
    vertices : (V, 3) float32
    faces    : (F, 3) int32
    n_chunks : int
        METIS partition count.  Higher = more (smaller) chunks; xatlas then
        sub-segments each.  Trade speed against atlas packing efficiency.
    atlas_resolution : int
        Target atlas size for xatlas.  xatlas auto-expands if it can't fit.
    xatlas_max_cost : float
        Per-chart segmentation cost ceiling.  2.0 is xatlas's own default.

    Returns
    -------
    out_vertices : (V', 3) float32
        Post-seam-split vertices (one per loop, in xatlas convention).
    out_faces    : (F, 3)   int32
        Triangle indices into out_vertices.
    out_uvs      : (V', 2)  float32
        UV per output vertex, normalized to [0, 1].
    vmap_global  : (V',)    int32
        Maps out_vertices[i] back to original vertices[j] for downstream
        per-vertex attribute lookup (bake input).  Cleaner than recomputing
        a KDTree from positions.
    """
    import pymetis
    import xatlas

    F = int(faces.shape[0])
    if F == 0:
        return (vertices, faces, np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=np.int32))

    if verbose:
        print(f"[cumesh_uv] partitioning face graph into {n_chunks} chunks…",
              flush=True)
    offsets, adj = _build_face_graph_csr(faces)
    # pymetis accepts python lists or numpy arrays; lists are faster to
    # marshal at our sizes.  Newer pymetis prefers a CSRAdjacency object;
    # falling back keeps compatibility with the 2025.2 release.
    try:
        from pymetis import CSRAdjacency
        csr = CSRAdjacency(xadj=offsets, adjncy=adj)
        cuts, membership = pymetis.part_graph(nparts=n_chunks, adjacency=csr)
    except ImportError:
        cuts, membership = pymetis.part_graph(
            nparts=n_chunks,
            xadj=offsets.tolist(),
            adjncy=adj.tolist(),
        )
    labels = np.asarray(membership, dtype=np.int32)
    if verbose:
        print(f"[cumesh_uv] METIS cuts={cuts:,}", flush=True)

    # Order faces by label so we can slice per-chunk in one shot.
    order = np.argsort(labels, kind="stable")
    sorted_faces = faces[order]
    sorted_labels = labels[order]
    chunk_starts = np.searchsorted(sorted_labels, np.arange(n_chunks))
    chunk_ends = np.searchsorted(sorted_labels, np.arange(n_chunks),
                                 side="right")

    atlas = xatlas.Atlas()
    chunk_origvmap = []                         # vmap[chunk_local_v] = orig vid
    for c in range(n_chunks):
        sub_f_global = sorted_faces[chunk_starts[c]:chunk_ends[c]]
        if sub_f_global.shape[0] == 0:
            continue
        used = np.unique(sub_f_global.reshape(-1))
        remap = -np.ones(vertices.shape[0], dtype=np.int64)
        remap[used] = np.arange(used.shape[0])
        sub_v = np.ascontiguousarray(vertices[used].astype(np.float32))
        sub_f = np.ascontiguousarray(remap[sub_f_global].astype(np.uint32))
        atlas.add_mesh(sub_v, sub_f)
        chunk_origvmap.append(used)

    chart_opts = xatlas.ChartOptions()
    chart_opts.max_cost = xatlas_max_cost
    chart_opts.max_iterations = 1

    pack_opts = xatlas.PackOptions()
    pack_opts.padding = 2
    pack_opts.bilinear = True
    if atlas_resolution > 0:
        pack_opts.resolution = atlas_resolution

    if verbose:
        print(f"[cumesh_uv] xatlas.generate (max_cost={xatlas_max_cost}, "
              f"resolution={atlas_resolution})...", flush=True)
    atlas.generate(chart_options=chart_opts, pack_options=pack_opts)
    if verbose:
        try:
            util = atlas.utilization * 100
        except Exception:
            util = float("nan")
        print(f"[cumesh_uv] charts={atlas.chart_count:,}, "
              f"atlas={atlas.width}x{atlas.height}, util={util:.1f}%",
              flush=True)

    # Stitch: produce one global (vertices, faces, uvs) tuple in xatlas
    # post-seam-split convention.  Track the orig-vertex mapping so the
    # bake step can KDTree/sample voxel attrs directly.
    all_v = []
    all_f = []
    all_uv = []
    all_vmap = []
    cnt = 0
    for i, vmap_orig in enumerate(chunk_origvmap):
        vmap_chunk, faces_i, uvs_i = atlas[i]
        # vmap_chunk indexes into chunk vertices; chunk vertex j was orig vertex vmap_orig[j].
        orig_vids = vmap_orig[vmap_chunk]
        all_v.append(vertices[orig_vids])
        all_vmap.append(orig_vids.astype(np.int32))
        all_f.append(faces_i.reshape(-1, 3).astype(np.int32) + cnt)
        all_uv.append(uvs_i.astype(np.float32))
        cnt += vmap_chunk.shape[0]

    out_v = np.concatenate(all_v, axis=0).astype(np.float32)
    out_f = np.concatenate(all_f, axis=0).astype(np.int32)
    out_uv = np.concatenate(all_uv, axis=0).astype(np.float32)
    out_vmap = np.concatenate(all_vmap, axis=0).astype(np.int32)
    return out_v, out_f, out_uv, out_vmap
