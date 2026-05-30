"""CPU/numpy ports of cumesh primitives that are needed off-CUDA.

Pixal3D's official pipeline calls cumesh for hole filling and UV unwrap on
the GPU (see TencentARC/Pixal3D and JeffreyXiang/CuMesh).  On Apple Silicon
we have no CUDA, so we re-implement the algorithmically-simple pieces here
in numpy/scipy.  These ports are NOT GPU-accelerated — they target the
specific face counts Pixal3D produces (a few-million-face mesh once, not a
realtime workload), so single-threaded numpy is fine.

Currently ported:
    fill_holes                 — triangulate small boundary loops (perimeter-
                                 thresholded fan triangulation, matching
                                 cumesh's algorithm rather than pymeshlab's
                                 larger-hole heuristics or bmesh.ops.holes_fill's
                                 sides-threshold).
    repair_non_manifold_edges  — split vertices to make the mesh locally
                                 manifold.  Crucially, preserves all faces
                                 (pymeshlab's equivalent *removes* faces,
                                 creating new boundaries that defeat
                                 downstream decimation).
"""
from .fill_holes import fill_holes  # noqa: F401
from .repair import repair_non_manifold_edges  # noqa: F401
from .compute_charts import (  # noqa: F401
    compute_charts,
    compute_charts_dihedral_cc,
    extract_chunks,
)
from .compute_charts_lloyd import compute_charts_lloyd  # noqa: F401
from .compute_charts_iterative import compute_charts_iterative  # noqa: F401
from .cleanup import (  # noqa: F401
    remove_small_connected_components,
    unify_face_orientations,
)
from .bvh_bake import (  # noqa: F401
    closest_points_on_mesh,
    bvh_corrected_positions,
)
from .bake_corrected import bake_texture_bvh_corrected  # noqa: F401
from .normals import compute_vertex_normals  # noqa: F401
from .voxel_trilinear import (  # noqa: F401
    trilinear_sample_sparse,
    build_dense_grid,
    trilinear_sample,
)
from .uv_unwrap import cumesh_style_uv_unwrap  # noqa: F401
