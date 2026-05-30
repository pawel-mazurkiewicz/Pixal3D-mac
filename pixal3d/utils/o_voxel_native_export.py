"""
Subprocess bridge for the native o_voxel GLB exporter.

The Pixal3D CLI may run under a different Python ABI than the Metal-enabled
o_voxel/cumesh wheels.  This script is intentionally standalone so
generate_mps.py can call it with the interpreter that owns those native wheels.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch


DEFAULT_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}


OUTPUT_TRANSFORMS = {
    "o_voxel": np.eye(4, dtype=np.float64),
    # Native o_voxel writes the mesh tipped onto its side: in glTF's Y-up
    # frame the model's true "up" axis comes out along +Z (a standing
    # character is Z-tall, not Y-tall), so every viewer shows it lying down.
    # The transform that stands it upright is a -90° rotation about X,
    # (x, y, z) -> (-x, -z, -y), composed with o_voxel's prior conversion.
    # This is the same correction users were applying by hand in Blender
    # (quaternion wxyz=(1/√2, -1/√2, 0, 0)); baking it here means the GLB is
    # correct in any Y-up glTF viewer without a manual post-rotation.
    # Pure rotation (det=+1), so the face-winding flip below is skipped.
    "pixal3d": np.array([
        [-1, 0,  0, 0],
        [ 0, 0, -1, 0],
        [ 0, -1, 0, 0],
        [ 0, 0,  0, 1],
    ], dtype=np.float64),
    # Pure inverse of o_voxel's final q=(x, z, -y) conversion, useful for
    # debugging axis convention without the Pixal3D Z reflection.
    "y_up": np.array([
        [1, 0,  0, 0],
        [0, 0, -1, 0],
        [0, 1,  0, 0],
        [0, 0,  0, 1],
    ], dtype=np.float64),
}


def _iter_geometries(mesh):
    return mesh.geometry.values() if hasattr(mesh, "geometry") else [mesh]


def _apply_output_transform(mesh, mode: str) -> None:
    matrix = OUTPUT_TRANSFORMS[mode]
    if np.allclose(matrix, np.eye(4)):
        return

    det = float(np.linalg.det(matrix[:3, :3]))
    for geom in _iter_geometries(mesh):
        geom.apply_transform(matrix)
        if det < 0:
            geom.faces = geom.faces[:, [0, 2, 1]]
            geom._cache.clear()


def _force_opaque_material(mesh) -> None:
    """Match upstream Pixal3D's opaque GLB export behavior."""
    for geom in _iter_geometries(mesh):
        visual = getattr(geom, "visual", None)
        material = getattr(visual, "material", None)
        if material is None:
            continue
        if hasattr(material, "alphaMode"):
            material.alphaMode = "OPAQUE"
        if hasattr(material, "doubleSided"):
            material.doubleSided = True


def _trimesh_from_tensors(vertices: torch.Tensor, faces: torch.Tensor):
    import trimesh

    return trimesh.Trimesh(
        vertices=vertices.detach().cpu().numpy(),
        faces=faces.detach().cpu().numpy(),
        process=False,
    )


def _write_geometry_stage(mesh_backend, output_dir: Path, index: int, name: str, transform: str) -> Path:
    vertices, faces = mesh_backend.read()
    mesh = _trimesh_from_tensors(vertices, faces)
    # On the final stage, clean up Metal simplify's residue of locally
    # inverted faces (CUDA cumesh's simplify_step rejects flipping
    # collapses; the Apple Silicon Metal port apparently doesn't, so QEM
    # scatters inverted faces inside otherwise-correct CCs).  Two-stage:
    #  (1) ``_per_face_winding_fix`` — iterative face-level flip of
    #      individual outliers (scattered within CCs, where CC-level
    #      algorithms can't reach).
    #  (2) ``trimesh.fix_normals(multibody=True)`` — per-CC outward
    #      direction check (catches any whole CC still inverted).
    if name == "after_unify_orientations":
        try:
            import time as _time
            _t0 = _time.time()
            new_faces_np = _per_face_winding_fix(
                np.asarray(mesh.vertices),
                np.asarray(mesh.faces),
                max_iter=8,
                verbose=True,
            )
            mesh.faces = new_faces_np
            print(
                f"[o_voxel-native] _per_face_winding_fix: "
                f"{_time.time() - _t0:.1f}s",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[o_voxel-native] _per_face_winding_fix failed ({exc}); "
                "skipping",
                flush=True,
            )
        try:
            import time as _time
            import trimesh.repair
            _t0 = _time.time()
            trimesh.repair.fix_normals(mesh, multibody=True)
            print(
                f"[o_voxel-native] trimesh.fix_normals(multibody): "
                f"{_time.time() - _t0:.1f}s",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[o_voxel-native] trimesh.fix_normals failed ({exc}); "
                "skipping per-CC outward orientation fix",
                flush=True,
            )
    _apply_output_transform(mesh, transform)
    path = output_dir / f"{index:02d}_{name}.glb"
    mesh.export(str(path))
    print(
        f"[o_voxel-native] stage {index:02d}_{name}: "
        f"{len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces -> {path}",
        flush=True,
    )
    return path


def _export_debug_stages(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    output_dir: Path,
    decimation_target: int,
    output_transform: str,
    verbose: bool,
    include_raw: bool,
) -> Path:
    """Export native cumesh geometry stages without UV unwrap / texture bake."""
    from o_voxel import postprocess

    if not getattr(postprocess, "_HAS_GPU_DEPS", False):
        raise RuntimeError("Native o_voxel GPU dependencies are unavailable")

    device = torch.device("cpu" if getattr(postprocess, "_BACKEND", None) == "metal" else "cuda")
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = postprocess._MeshBackend()
    mesh.init(vertices.to(device), faces.to(device))

    stage = 0
    last_path = None
    if include_raw:
        last_path = _write_geometry_stage(mesh, output_dir, stage, "input", output_transform)
        stage += 1

    mesh.fill_holes(max_hole_perimeter=3e-2)
    if verbose:
        print(f"[o_voxel-native] after fill_holes: {mesh.num_vertices:,} vertices, {mesh.num_faces:,} faces")
    if include_raw:
        last_path = _write_geometry_stage(mesh, output_dir, stage, "after_fill_holes", output_transform)
        stage += 1

    mesh.simplify(decimation_target * 3, verbose=verbose)
    last_path = _write_geometry_stage(mesh, output_dir, stage, "after_simplify_3x", output_transform)
    stage += 1

    # ── cleanup_1: drilled into substeps to pinpoint where faces are lost.
    # SESSION 4 probe (skip-simplify): cleanup_1 dropped ~1M faces while
    # simplify was no-op, so the culprit is one of these four native calls.
    mesh.remove_duplicate_faces()
    last_path = _write_geometry_stage(mesh, output_dir, stage, "cleanup1a_after_dedup", output_transform)
    stage += 1

    mesh.repair_non_manifold_edges()
    last_path = _write_geometry_stage(mesh, output_dir, stage, "cleanup1b_after_repair_nme", output_transform)
    stage += 1

    mesh.remove_small_connected_components(1e-5)
    last_path = _write_geometry_stage(mesh, output_dir, stage, "cleanup1c_after_remove_small_cc", output_transform)
    stage += 1

    mesh.fill_holes(max_hole_perimeter=3e-2)
    last_path = _write_geometry_stage(mesh, output_dir, stage, "cleanup1d_after_fill_holes", output_transform)
    stage += 1

    mesh.simplify(decimation_target, verbose=verbose)
    last_path = _write_geometry_stage(mesh, output_dir, stage, "after_simplify_target", output_transform)
    stage += 1

    mesh.remove_duplicate_faces()
    mesh.repair_non_manifold_edges()
    mesh.remove_small_connected_components(1e-5)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    last_path = _write_geometry_stage(mesh, output_dir, stage, "after_cleanup_2", output_transform)
    stage += 1

    mesh.unify_face_orientations()
    last_path = _write_geometry_stage(mesh, output_dir, stage, "after_unify_orientations", output_transform)
    return last_path


def _patch_grid_sample_output(postprocess_module) -> None:
    """Normalize flex_gemm/grid_sample output to the shape to_glb assigns."""
    grid_sample = getattr(postprocess_module, "_grid_sample_3d", None)
    if grid_sample is None:
        return

    def _grid_sample_flat(feats, coords, shape, grid, mode="trilinear"):
        out = grid_sample(feats, coords, shape, grid, mode=mode)
        channels = int(shape[1])
        samples = int(grid.shape[1])
        if out.ndim == 3 and out.shape[0] == 1 and out.shape[1] == samples:
            return out.reshape(samples, channels)
        if out.ndim == 2 and out.shape == (channels, samples):
            return out.T.contiguous()
        return out

    postprocess_module._grid_sample_3d = _grid_sample_flat





def _patch_cleanup_to_noop(postprocess,
                            *,
                            fill_holes: bool = False,
                            repair_nme: bool = False,
                            small_cc: bool = False,
                            unify: bool = False,
                            dedup: bool = False,
                            degen: bool = False) -> None:
    """Selectively no-op cleanup ops on `_MeshBackend`.  Back-ported from
    Pixal3D_fresh/feat/apple-silicon-port-v2 (S17).  Applied AFTER any
    pre-existing env-var patches (`_patch_repair_non_manifold_edges` etc.)
    so this wins when both are set."""
    if not any([fill_holes, repair_nme, small_cc, unify, dedup, degen]):
        return
    backend_cls = postprocess._MeshBackend
    patched = []
    def _log_skip(op_name, self):
        print(f"[o_voxel-native]   skip {op_name} (V={self.num_vertices:,} F={self.num_faces:,})", flush=True)
    if fill_holes:
        def _noop_fh(self, max_hole_perimeter=3e-2): _log_skip("fill_holes", self)
        backend_cls.fill_holes = _noop_fh
        patched.append("fill_holes")
    if repair_nme:
        def _noop_rn(self): _log_skip("repair_non_manifold_edges", self)
        backend_cls.repair_non_manifold_edges = _noop_rn
        patched.append("repair_nme")
    if small_cc:
        def _noop_sc(self, min_area): _log_skip("remove_small_connected_components", self)
        backend_cls.remove_small_connected_components = _noop_sc
        patched.append("small_cc")
    if unify:
        def _noop_uf(self): _log_skip("unify_face_orientations", self)
        backend_cls.unify_face_orientations = _noop_uf
        patched.append("unify")
    if dedup:
        def _noop_df(self): _log_skip("remove_duplicate_faces", self)
        backend_cls.remove_duplicate_faces = _noop_df
        patched.append("dedup")
    if degen:
        def _noop_dg(self, abs_thresh=1e-24, rel_thresh=1e-12): _log_skip("remove_degenerate_faces", self)
        backend_cls.remove_degenerate_faces = _noop_dg
        patched.append("degen")
    print(f"[o_voxel-native] PATCH: selective cleanup-to-noop on {backend_cls.__name__}: {', '.join(patched)}", flush=True)


def _patch_simplify_to_noop(postprocess) -> None:
    """No-op cumesh.simplify (back-ported from S17)."""
    backend_cls = postprocess._MeshBackend
    print(f"[o_voxel-native] PATCH: simplify-to-noop on {backend_cls.__name__}", flush=True)
    def _noop(self, target_num_faces, verbose=False, options=None):
        print(f"[o_voxel-native]   skip simplify (target={target_num_faces:,}, V={self.num_vertices:,} F={self.num_faces:,})", flush=True)
    backend_cls.simplify = _noop


def _patch_simplify_skip_3x(postprocess) -> None:
    """S21: Skip only the FIRST simplify call (the destructive `target*3` pass).

    The canonical o_voxel.to_glb chain calls simplify twice:
      1. simplify(target*3) — intermediate decimation 8M→3M faces
      2. simplify(target)   — final decimation to target

    On Mac, the first pass amplifies NMEs catastrophically when the raw mesh
    has even modestly more NMEs than CUDA (6.5% vs 5.1%). Skipping it keeps
    the topology much cleaner. The second simplify(target) still runs and
    handles the final decimation on a now-repaired (post-repair_nme) mesh,
    where it doesn't catastrophically amplify NMEs.

    S21 measurement on fairy fp32 raw: final boundary 21.0% → 7.8%
    (3.6x reduction in unmatched edges).
    """
    backend_cls = postprocess._MeshBackend
    original = backend_cls.simplify
    _state = {"call_count": 0}
    print(f"[o_voxel-native] PATCH: simplify-skip-3x on {backend_cls.__name__} (only the 1st simplify call becomes a no-op)", flush=True)
    def _wrapped(self, target_num_faces, verbose=False, options=None):
        n = _state["call_count"]
        _state["call_count"] += 1
        if n == 0:
            print(f"[o_voxel-native]   skip simplify_3x (target={target_num_faces:,}, V={self.num_vertices:,} F={self.num_faces:,}) [S21 swiss-cheese fix]", flush=True)
            return
        return original(self, target_num_faces, verbose=verbose, options=options)
    backend_cls.simplify = _wrapped


def _patch_small_cc_threshold(postprocess, threshold: float) -> None:
    """Override min_area for remove_small_connected_components (back-ported from S17).
    No effect if cleanup_to_noop already patched small_cc to no-op."""
    backend_cls = postprocess._MeshBackend
    original = backend_cls.remove_small_connected_components
    def _override(self, min_area):
        print(f"[o_voxel-native]   override small_cc threshold {min_area} -> {threshold} (V={self.num_vertices:,} F={self.num_faces:,})", flush=True)
        return original(self, threshold)
    backend_cls.remove_small_connected_components = _override
    print(f"[o_voxel-native] PATCH: small_cc threshold override -> {threshold} on {backend_cls.__name__}", flush=True)


def _patch_presmooth_for_compute_charts(postprocess, mode: str, iterations: int) -> None:
    """Smooth mesh vertices right before compute_charts (S17b).

    Diagnostic chain established:
    - Raw post-extract mesh: ~9 faces/chart (good local-normal coherence)
    - Post-QEM-simplify (any simplifier): ~1-3 faces/chart (QEM perturbs the
      normals of faces sharing the moved vertex)
    - Compute_charts can't merge a mesh where neighboring face normals
      differ by >cone-threshold

    This patch wraps ``_MeshBackend.compute_charts`` to first apply Taubin
    (or Laplacian/Humphrey) smoothing on the mesh vertices, healing the
    per-face normal perturbations.  Volume-preserving Taubin recommended;
    Laplacian shrinks the mesh, Humphrey is in between.
    """
    if mode == "none":
        return
    backend_cls = postprocess._MeshBackend
    original = backend_cls.compute_charts
    import trimesh as _tm
    import torch as _torch
    import numpy as _np

    def _wrapped(self, *args, **kwargs):
        verts, faces = self.read()
        verts_np = verts.detach().cpu().numpy().astype(_np.float64, copy=False)
        faces_np = faces.detach().cpu().numpy().astype(_np.int64, copy=False)
        mesh = _tm.Trimesh(vertices=verts_np, faces=faces_np, process=False)

        # Identify "safe" vertices: those touching only manifold edges (shared by exactly 2 faces).
        # NME vertices (touched by edges with face_count != 2) cause Taubin's Laplacian to
        # yank them into shrapnel spikes; we freeze them at their original positions.
        edges_unique = mesh.edges_unique
        edge_face_count = _np.bincount(mesh.edges_unique_inverse,
                                        minlength=len(edges_unique))
        nme_edge_mask = edge_face_count != 2
        if nme_edge_mask.any():
            nme_vertices = _np.unique(edges_unique[nme_edge_mask].flatten())
        else:
            nme_vertices = _np.array([], dtype=_np.int64)
        n_safe = len(verts_np) - len(nme_vertices)
        n_total = len(verts_np)
        print(
            f"[o_voxel-native]   precharts-smooth: {mode}, iters={iterations}, "
            f"V={n_total:,} F={faces_np.shape[0]:,}, "
            f"freezing {len(nme_vertices):,} NME vertices ({100.0*len(nme_vertices)/max(n_total,1):.1f}%)",
            flush=True,
        )

        # Run smoothing one iteration at a time, snapping NME vertices back each time
        # so they can't propagate displacement to neighbors via Laplacian.
        original_verts = verts_np.copy()
        for it in range(iterations):
            if mode == "taubin":
                _tm.smoothing.filter_taubin(mesh, iterations=1)
            elif mode == "laplacian":
                _tm.smoothing.filter_laplacian(mesh, iterations=1)
            elif mode == "humphrey":
                _tm.smoothing.filter_humphrey(mesh, iterations=1)
            else:
                raise ValueError(f"unknown smoothing mode: {mode}")
            if len(nme_vertices) > 0:
                mesh.vertices[nme_vertices] = original_verts[nme_vertices]

        # Sanity check: report max displacement of safe vertices
        displacement = _np.linalg.norm(mesh.vertices - original_verts, axis=1)
        bbox_diag = float(_np.linalg.norm(original_verts.max(axis=0) - original_verts.min(axis=0)))
        print(
            f"[o_voxel-native]   smoothed: max displacement {displacement.max():.4f} "
            f"({100.0*displacement.max()/max(bbox_diag,1e-9):.2f}% of bbox diag), "
            f"mean {displacement.mean():.4f}",
            flush=True,
        )

        new_verts_t = _torch.from_numpy(mesh.vertices.astype(_np.float32)).contiguous()
        new_faces_t = _torch.from_numpy(faces_np.astype(_np.int32)).contiguous()
        self.init(new_verts_t, new_faces_t)
        return original(self, *args, **kwargs)

    backend_cls.compute_charts = _wrapped
    print(
        f"[o_voxel-native] PATCH: precharts-smooth on {backend_cls.__name__} "
        f"(mode={mode}, iters={iterations})",
        flush=True,
    )


def _patch_fill_holes_before_uv(postprocess, perimeter: float) -> None:
    """OPTIONAL watertight enhancement (off by default; not faithful to CUDA).

    The to_glb chain's last geometry op before uv_unwrap is the final
    ``simplify``, which reopens thin-feature holes; the built-in
    ``fill_holes(max_hole_perimeter=3e-2)`` passes run *earlier* and miss them,
    so the holes survive into the atlas (xatlas counts charts on a holey mesh).

    This wraps ``_MeshBackend.compute_charts`` to run one more
    ``fill_holes(max_hole_perimeter=perimeter)`` in-place just before chart
    computation.  Because it precedes UV unwrap + bake, the patch faces become
    part of the atlas (they get UVs + sampled texture) and close boundary loops
    so xatlas emits fewer charts.  Uses the backend's own (Metal) fill_holes —
    the same op already used four times in the chain — with a larger perimeter.
    """
    if perimeter <= 0.0:
        return
    backend_cls = postprocess._MeshBackend
    original = backend_cls.compute_charts

    def _wrapped(self, *args, **kwargs):
        try:
            before_v, before_f = self.num_vertices, self.num_faces
            self.fill_holes(max_hole_perimeter=perimeter)
            print(
                f"[o_voxel-native]   fill-holes-before-uv: perimeter={perimeter}, "
                f"V {before_v:,}->{self.num_vertices:,} F {before_f:,}->{self.num_faces:,}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[o_voxel-native]   fill-holes-before-uv failed ({exc}); "
                "proceeding without the extra fill",
                flush=True,
            )
        return original(self, *args, **kwargs)

    backend_cls.compute_charts = _wrapped
    print(
        f"[o_voxel-native] PATCH: fill-holes-before-uv on {backend_cls.__name__} "
        f"(max_hole_perimeter={perimeter})",
        flush=True,
    )


def _patch_simplify_fast(postprocess) -> None:
    """Replace ``_MeshBackend.simplify`` with ``fast_simplification`` (S17b).

    Bisection result: Pedro's Metal simplify produces a mesh whose
    per-triangle normal noise defeats compute_charts.  Same mesh handed to
    CUDA compute_charts → 557k charts (vs CUDA-simplified-mesh → ~hundreds).
    fast_simplification is a mature QEM C++ library, already on disk for the
    macOS CPU fallback path (postprocess_cpu.py).  This swap lets us test
    whether replacing the simplifier alone fixes chart-count blowup.
    """
    try:
        import fast_simplification
    except ImportError as exc:
        raise RuntimeError(
            "fast_simplification not installed in the bridge venv; "
            "pip install fast-simplification"
        ) from exc
    backend_cls = postprocess._MeshBackend
    import numpy as _np
    import torch as _torch

    def _simplify_fast(self, target_num_faces, verbose=False, options=None):
        verts, faces = self.read()
        verts_np = verts.detach().cpu().numpy().astype(_np.float64, copy=False)
        faces_np = faces.detach().cpu().numpy().astype(_np.int32, copy=False)
        num_faces = faces_np.shape[0]
        if num_faces <= target_num_faces:
            if verbose:
                print(f"[o_voxel-native]   simplify(fast): F={num_faces:,} already <= target={target_num_faces:,}, no-op", flush=True)
            return
        target_reduction = 1.0 - (target_num_faces / num_faces)
        if verbose:
            print(f"[o_voxel-native]   simplify(fast): F={num_faces:,} -> target={target_num_faces:,} (reduction={target_reduction:.3f})", flush=True)
        new_verts, new_faces = fast_simplification.simplify(
            verts_np, faces_np,
            target_reduction=target_reduction,
        )
        new_verts_t = _torch.from_numpy(new_verts.astype(_np.float32)).contiguous()
        new_faces_t = _torch.from_numpy(new_faces.astype(_np.int32)).contiguous()
        self.init(new_verts_t, new_faces_t)
        if verbose:
            print(f"[o_voxel-native]   simplify(fast): result V={new_verts.shape[0]:,} F={new_faces.shape[0]:,}", flush=True)

    backend_cls.simplify = _simplify_fast
    print(f"[o_voxel-native] PATCH: simplify swapped to fast_simplification on {backend_cls.__name__}", flush=True)


def _patch_compute_charts(postprocess,
                           *,
                           save_input_path: str = None,
                           area_weight: float = None,
                           perim_weight: float = None) -> None:
    """Wrap ``_MeshBackend.compute_charts`` for S17b bisection of the
    "Mac gets 687k charts vs CUDA's hundreds" problem.

    - ``save_input_path``: dump the exact mesh state + kwargs handed to
      compute_charts as a .npz, so it can be shipped to a CUDA box and
      run through ``cumesh.compute_charts`` there.  Lets us bisect whether
      the chart blowup is in Pedro's Metal port of compute_charts or in
      the upstream Metal simplify producing a noisier-normal mesh.
    - ``area_weight`` / ``perim_weight``: override the area_penalty_weight
      (CUDA default 0.1) and perimeter_area_ratio_weight (CUDA default
      0.0001) kwargs that o_voxel never plumbs through.  Setting to 0
      disables the area/perim penalties so the cone-cost is the only
      term, which is the most aggressive merge configuration the
      algorithm allows.
    """
    if save_input_path is None and area_weight is None and perim_weight is None:
        return
    backend_cls = postprocess._MeshBackend
    original = backend_cls.compute_charts
    import numpy as _np
    import torch as _torch

    def _wrapped(self, *args, **kwargs):
        if save_input_path is not None:
            verts, faces = self.read()
            verts_np = verts.detach().cpu().numpy()
            faces_np = faces.detach().cpu().numpy()
            saved_kwargs = dict(kwargs)
            # positional → named (compute_charts signature):
            # (threshold_cone_half_angle_rad, refine_iterations, global_iterations,
            #  smooth_strength, area_penalty_weight, perimeter_area_ratio_weight)
            arg_names = [
                "threshold_cone_half_angle_rad", "refine_iterations",
                "global_iterations", "smooth_strength",
                "area_penalty_weight", "perimeter_area_ratio_weight",
            ]
            for i, val in enumerate(args):
                if i < len(arg_names):
                    saved_kwargs[arg_names[i]] = val
            _np.savez_compressed(
                save_input_path,
                vertices=verts_np.astype(_np.float32),
                faces=faces_np.astype(_np.int32),
                **{f"kw_{k}": _np.asarray(v) for k, v in saved_kwargs.items()},
            )
            print(
                f"[o_voxel-native] saved compute_charts input -> {save_input_path} "
                f"(V={verts_np.shape[0]:,} F={faces_np.shape[0]:,}, kwargs={saved_kwargs})",
                flush=True,
            )
        if area_weight is not None:
            kwargs["area_penalty_weight"] = area_weight
        if perim_weight is not None:
            kwargs["perimeter_area_ratio_weight"] = perim_weight
        if area_weight is not None or perim_weight is not None:
            print(
                f"[o_voxel-native]   compute_charts kwargs override: "
                f"area_w={kwargs.get('area_penalty_weight', '<default>')} "
                f"perim_w={kwargs.get('perimeter_area_ratio_weight', '<default>')}",
                flush=True,
            )
        return original(self, *args, **kwargs)

    backend_cls.compute_charts = _wrapped
    flags = []
    if save_input_path is not None:
        flags.append(f"save_input={save_input_path}")
    if area_weight is not None:
        flags.append(f"area_w={area_weight}")
    if perim_weight is not None:
        flags.append(f"perim_w={perim_weight}")
    print(f"[o_voxel-native] PATCH: compute_charts wrapped on {backend_cls.__name__} ({', '.join(flags)})", flush=True)


def _load_cumesh_port_module(filename: str):
    """Load a single ``cumesh_port`` submodule by path, bypassing __init__.py.

    The bridge runs under the trellis-mac native venv where the Pixal3D
    project package is not on ``sys.path``, and even if we add it,
    ``cumesh_port/__init__.py`` imports siblings that may not be importable
    under that venv (different torch / numpy combinations).  Using
    ``importlib.util.spec_from_file_location`` loads just the one .py we
    want with no package-side effects.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent / "cumesh_port" / filename
    spec = importlib.util.spec_from_file_location(f"cumesh_port_{filename[:-3]}", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_repair_non_manifold_edges(postprocess_module) -> None:
    """Bypass the buggy Metal ``repair_non_manifold_edges`` kernel.

    The cumesh Metal port that ships with ``trellis-mac`` over-splits
    vertices ~3× when repairing non-manifold edges (probe data:
    1.1M → 3.0M vertices on the 8.4M-face Pixal3D input mesh, ~170%
    increase).  The CUDA reference (and our CPU port at
    ``cumesh_port.repair.repair_non_manifold_edges``) splits an order of
    magnitude less aggressively.

    The over-split mesh then survives the next steps with massive
    artefacts: ``remove_small_connected_components(1e-5)`` culls 35% of
    post-simplify faces (the freshly-fragmented surface looks like dust
    to the area filter), ``fill_holes`` patches the resulting boundaries
    with fan triangles of arbitrary winding, and ``unify_face_orientations``
    can't recover because the mesh is shattered into thousands of CCs.

    Net effect on the final GLB: solid-shaded view has large gaps where
    inverted-winding faces hide (the "missing triangles" pattern), while
    wireframe is fine because all the faces are nominally still present.

    Fix: at module-import time, replace the Metal wrapper with a Python
    function that reads tensors back to CPU, runs the faithful numpy port,
    and re-inits the backend with the repaired (vertices, faces) pair.
    Covers both ``_export_debug_stages`` (explicit call) and
    ``postprocess.to_glb``'s internal cleanup (which calls
    ``self.repair_non_manifold_edges()`` via the Python wrapper).
    """
    MeshBackend = getattr(postprocess_module, "_MeshBackend", None)
    if MeshBackend is None or getattr(MeshBackend, "_repair_nme_patched", False):
        return

    # Session 7: PIXAL3D_REPAIR_NME=metal opts into Pedro's mtlmesh
    # repair_non_manifold_edges.  Metal simplify produces far fewer NMEs
    # (session-7: +19K inflation instead of +2.5M).
    # Session 15: Metal is now the default (Metal cleanup chain validated
    # end-to-end via CUDA-mesh-through-Mac-cleanup exoneration test).
    # Session 16: noop diagnostic added — proved Mac's split breaks
    # edge-pair invariants vs CUDA's split.  PIXAL3D_REPAIR_NME=pamo opts
    # out to the CPU port for safe rollback; PIXAL3D_REPAIR_NME=noop skips
    # the call entirely (diagnostic).
    setting = os.environ.get("PIXAL3D_REPAIR_NME", "metal").strip().lower()

    if setting == "noop":
        def _noop_repair_nme(self):
            print(
                f"[o_voxel-native] skipping repair_non_manifold_edges "
                f"[V={self.num_vertices:,} F={self.num_faces:,}]",
                flush=True,
            )
        MeshBackend.repair_non_manifold_edges = _noop_repair_nme
        MeshBackend._repair_nme_patched = True
        print("[o_voxel-native] _MeshBackend.repair_non_manifold_edges "
              "patched (PIXAL3D_REPAIR_NME=noop)", flush=True)
        return

    if setting != "pamo":
        print(
            "[o_voxel-native] PIXAL3D_REPAIR_NME default (metal) — keeping "
            "native cumesh.CuMesh.repair_non_manifold_edges (mtlmesh)",
            flush=True,
        )
        MeshBackend._repair_nme_patched = True  # idempotent across calls
        return

    try:
        repair_mod = _load_cumesh_port_module("repair.py")
    except Exception as exc:
        print(
            f"[o_voxel-native] CPU cumesh_port.repair unavailable ({exc}); "
            "falling back to native Metal repair (will over-split, expect artifacts)",
            flush=True,
        )
        return

    cpu_repair = repair_mod.repair_non_manifold_edges

    def _patched_repair_nme(self):
        import torch

        verts_t, faces_t = self.read()
        verts_np = verts_t.detach().cpu().numpy()
        faces_np = faces_t.detach().cpu().numpy().astype(np.int64)

        before_v = verts_np.shape[0]
        before_f = faces_np.shape[0]
        repair_verbose = os.environ.get("PIXAL3D_REPAIR_VERBOSE", "0") not in ("0", "", "false", "False")
        new_verts_np, new_faces_np = cpu_repair(verts_np, faces_np, verbose=repair_verbose)
        after_v = new_verts_np.shape[0]
        after_f = new_faces_np.shape[0]
        print(
            f"[o_voxel-native] cpu repair_non_manifold_edges: "
            f"V {before_v:,} -> {after_v:,} (+{after_v - before_v:,}), "
            f"F {before_f:,} -> {after_f:,}",
            flush=True,
        )

        verts_out = torch.from_numpy(new_verts_np.astype(verts_t.detach().cpu().numpy().dtype)).to(verts_t.device)
        faces_dtype = faces_t.detach().cpu().numpy().dtype
        faces_out = torch.from_numpy(new_faces_np.astype(faces_dtype)).to(faces_t.device)
        self.init(verts_out, faces_out)

    MeshBackend.repair_non_manifold_edges = _patched_repair_nme
    MeshBackend._repair_nme_patched = True



def _patch_remove_small_connected_components(postprocess_module) -> None:
    """Tunable replacement for ``_MeshBackend.remove_small_connected_components``.

    Behaviour depends on ``PIXAL3D_SMALL_CC``:

      unset    -> patch is a no-op; the native Metal small_cc runs at its
                  upstream default ``min_area=1e-5``.  This is the legacy
                  behaviour (commented out by default in main).
      "noop"   -> swap small_cc for a print-only stub.  Per the plan's
                  variant v4 audit this leaves visible debris+spikes; keep
                  for diagnostic runs only.
      <float>  -> use the CPU port (``cumesh_port.cleanup.
                  remove_small_connected_components``) with the given
                  min_area override.  Lets us soften culling without
                  disabling it.  Example: ``PIXAL3D_SMALL_CC=1e-7`` keeps
                  fragments down to 0.1 mm² (noise floor) while still
                  removing literal dust.

    The plan's diagnosis (PAMO_PORT_PLAN.md, Phase 6 log) says the
    upstream 1e-5 threshold was calibrated for CUDA cumesh's repair output
    (which doesn't over-split this badly).  Our CPU repair port mirrors
    CUDA but pamo's collapses still introduce fresh NMEs that repair_nme
    then splits, so the mesh ends up fragmented; small_cc(1e-5) culls
    those fragments as dust, fill_holes(3e-2) recovers only the pinprick
    subset, and ~1M faces of real surface end up permanently gone.

    Tuning this threshold (or replacing small_cc with a perimeter-based
    criterion) is the most direct way to reduce that loss without
    introducing the v4 debris/spikes problem.
    """
    MeshBackend = getattr(postprocess_module, "_MeshBackend", None)
    if MeshBackend is None or getattr(MeshBackend, "_remove_small_cc_patched", False):
        return

    env_setting = os.environ.get("PIXAL3D_SMALL_CC", "").strip().lower()
    if not env_setting:
        return  # opt-in only; default keeps Metal native small_cc behaviour

    if env_setting == "noop":
        def _noop_remove_small_cc(self, min_area: float):
            print(
                f"[o_voxel-native] skipping remove_small_connected_components"
                f"(min_area={min_area:.1e}) "
                f"[V={self.num_vertices:,} F={self.num_faces:,}]",
                flush=True,
            )
        MeshBackend.remove_small_connected_components = _noop_remove_small_cc
        MeshBackend._remove_small_cc_patched = True
        print("[o_voxel-native] _MeshBackend.remove_small_connected_components "
              "patched (PIXAL3D_SMALL_CC=noop)", flush=True)
        return

    # Numeric override -> CPU port with custom min_area.
    try:
        custom_min_area = float(env_setting)
    except ValueError:
        print(f"[o_voxel-native] PIXAL3D_SMALL_CC={env_setting!r} "
              "is neither 'noop' nor a float; ignoring.", flush=True)
        return

    try:
        cleanup_mod = _load_cumesh_port_module("cleanup.py")
    except Exception as exc:
        print(f"[o_voxel-native] cumesh_port.cleanup unavailable ({exc}); "
              "falling back to native small_cc.", flush=True)
        return

    cpu_small_cc = cleanup_mod.remove_small_connected_components

    def _patched_small_cc(self, min_area: float):
        import torch

        verts_t, faces_t = self.read()
        verts_np = verts_t.detach().cpu().numpy()
        faces_np = faces_t.detach().cpu().numpy().astype(np.int64)
        before_v = verts_np.shape[0]
        before_f = faces_np.shape[0]
        new_verts_np, new_faces_np = cpu_small_cc(
            verts_np, faces_np, min_area=custom_min_area, verbose=False,
        )
        after_v = new_verts_np.shape[0]
        after_f = new_faces_np.shape[0]
        print(
            f"[o_voxel-native] cpu remove_small_cc(min_area={custom_min_area:.1e}, "
            f"caller_arg={min_area:.1e} OVERRIDDEN): "
            f"V {before_v:,} -> {after_v:,} ({after_v - before_v:+,}), "
            f"F {before_f:,} -> {after_f:,} ({after_f - before_f:+,})",
            flush=True,
        )
        verts_out = torch.from_numpy(
            new_verts_np.astype(verts_t.detach().cpu().numpy().dtype)
        ).to(verts_t.device)
        faces_dtype = faces_t.detach().cpu().numpy().dtype
        faces_out = torch.from_numpy(new_faces_np.astype(faces_dtype)).to(faces_t.device)
        self.init(verts_out, faces_out)

    MeshBackend.remove_small_connected_components = _patched_small_cc
    MeshBackend._remove_small_cc_patched = True
    print(f"[o_voxel-native] _MeshBackend.remove_small_connected_components "
          f"patched (PIXAL3D_SMALL_CC={custom_min_area:.1e})", flush=True)



def _patch_unify_face_orientations(postprocess_module) -> None:
    """Replace the Metal ``unify_face_orientations`` with the CUDA-faithful CPU port.

    Direct evidence the Metal implementation is broken: after it runs as
    stage 09 of the to_glb cleanup chain, the bandaid
    ``_per_face_winding_fix`` (in this file) still finds ~9k conflicting
    manifold edges and flips ~1k more faces in its 8 iterations.  A
    correct ``unify_face_orientations`` should leave ZERO conflicts on a
    consistently-orientable mesh (and just one undecided parity per
    Möbius-strip CC, of which there are usually none).

    Inverted faces caused by this Metal bug present as "missing surface"
    holes under backface culling, which we observe as chunks ripped out
    of the model's front in the textured GLB output even when the input
    mesh from CUDA stage 07 is geometrically intact.

    The CPU port at ``cumesh_port.unify.unify_face_orientations`` is a
    faithful translation of ``CuMesh::unify_face_orientations`` in
    ``CuMesh/src/clean_up.cu:1191``: manifold-edge graph + per-edge
    flip-flag + per-CC parity via BFS.

    Enabled by default (always patches when imported); opt out by setting
    ``PIXAL3D_UNIFY_FACE_ORIENTATIONS=metal`` to keep the buggy Metal
    binary in place (useful for A/B comparisons).
    """
    MeshBackend = getattr(postprocess_module, "_MeshBackend", None)
    if MeshBackend is None or getattr(MeshBackend, "_unify_patched", False):
        return

    # Session 15: Metal is now the default; PIXAL3D_UNIFY_FACE_ORIENTATIONS=pamo
    # opts out to the CPU port.
    if os.environ.get("PIXAL3D_UNIFY_FACE_ORIENTATIONS", "metal").strip().lower() != "pamo":
        print("[o_voxel-native] PIXAL3D_UNIFY_FACE_ORIENTATIONS default (metal); "
              "keeping native Metal unify_face_orientations",
              flush=True)
        return

    try:
        unify_mod = _load_cumesh_port_module("unify.py")
    except Exception as exc:
        print(
            f"[o_voxel-native] CPU cumesh_port.unify unavailable ({exc}); "
            "falling back to native Metal unify_face_orientations (will "
            "leave inverted faces, expect missing-surface artifacts)",
            flush=True,
        )
        return

    cpu_unify = unify_mod.unify_face_orientations

    def _patched_unify(self):
        import torch

        verts_t, faces_t = self.read()
        verts_np = verts_t.detach().cpu().numpy()
        faces_np = faces_t.detach().cpu().numpy().astype(np.int64)
        unify_verbose = os.environ.get(
            "PIXAL3D_UNIFY_VERBOSE", "0"
        ) not in ("0", "", "false", "False")
        new_faces_np = cpu_unify(verts_np, faces_np, verbose=unify_verbose)
        n_flipped = int((new_faces_np != faces_np).any(axis=1).sum())
        print(
            f"[o_voxel-native] cpu unify_face_orientations: "
            f"V={verts_np.shape[0]:,} F={faces_np.shape[0]:,} "
            f"flipped {n_flipped:,} faces "
            f"({100.0 * n_flipped / max(1, faces_np.shape[0]):.2f}%)",
            flush=True,
        )
        faces_dtype = faces_t.detach().cpu().numpy().dtype
        faces_out = torch.from_numpy(new_faces_np.astype(faces_dtype)).to(faces_t.device)
        # Vertices unchanged; only faces.  init() expects both, so feed
        # the original verts tensor straight through.
        self.init(verts_t, faces_out)

    MeshBackend.unify_face_orientations = _patched_unify
    MeshBackend._unify_patched = True
    print("[o_voxel-native] _MeshBackend.unify_face_orientations patched to "
          "cumesh_port.unify.unify_face_orientations",
          flush=True)



def _patch_fill_holes(postprocess_module) -> None:
    """Replace the Metal ``fill_holes`` with the CPU port, tunable via env.

    The Metal/native ``fill_holes`` runs with cumesh's default
    ``max_hole_perimeter=3e-2`` (3cm on Pixal3D's unit-cube scale).  That
    only fills pinprick holes — the FDG-iso-surface noise it was designed
    for.  Empirical Blender audit of CUDA→Mac-post-pipeline output: ~50k
    boundary loops in the 5-30cm perimeter range survive, contributing the
    visible localized damage (broken lantern hood, shattered decorative
    elements) the user reported.

    Setting ``PIXAL3D_FILL_HOLES_PERIMETER`` to a larger value (e.g. 0.1
    for 10cm or 0.3 for 30cm) lets us close more of those medium holes.
    The risk: filling legitimate features like small window openings.

    The CPU port (``cumesh_port.fill_holes``) is the same algorithm as
    cumesh (centroid fan over CLOSED LOOP boundary components, perimeter <
    threshold), so the perimeter semantics match exactly.
    """
    MeshBackend = getattr(postprocess_module, "_MeshBackend", None)
    if MeshBackend is None or getattr(MeshBackend, "_fill_holes_patched", False):
        return

    env_perim = os.environ.get("PIXAL3D_FILL_HOLES_PERIMETER", "").strip()
    if not env_perim:
        return  # opt-in only; absent env var keeps Metal default
    try:
        max_perim = float(env_perim)
    except ValueError:
        print(f"[o_voxel-native] PIXAL3D_FILL_HOLES_PERIMETER={env_perim!r} "
              "is not a float; ignoring.", flush=True)
        return

    try:
        fh_mod = _load_cumesh_port_module("fill_holes.py")
    except Exception as exc:
        print(f"[o_voxel-native] cumesh_port.fill_holes unavailable ({exc}); "
              "falling back to native fill_holes.", flush=True)
        return

    cpu_fill_holes = fh_mod.fill_holes

    def _patched_fill_holes(self, max_hole_perimeter: float = 3e-2):
        import torch

        # Honor the env override; ignore the caller-supplied value.
        perim = max_perim
        verts_t, faces_t = self.read()
        verts_np = verts_t.detach().cpu().numpy()
        faces_np = faces_t.detach().cpu().numpy().astype(np.int64)

        before_v = verts_np.shape[0]
        before_f = faces_np.shape[0]
        new_verts_np, new_faces_np = cpu_fill_holes(
            verts_np, faces_np,
            max_hole_perimeter=perim,
            verbose=False,
        )
        after_v = new_verts_np.shape[0]
        after_f = new_faces_np.shape[0]
        print(
            f"[o_voxel-native] cpu fill_holes(perim<{perim}): "
            f"V {before_v:,} -> {after_v:,} (+{after_v - before_v:,}), "
            f"F {before_f:,} -> {after_f:,} (+{after_f - before_f:,})",
            flush=True,
        )

        verts_out = torch.from_numpy(
            new_verts_np.astype(verts_t.detach().cpu().numpy().dtype)
        ).to(verts_t.device)
        faces_dtype = faces_t.detach().cpu().numpy().dtype
        faces_out = torch.from_numpy(new_faces_np.astype(faces_dtype)).to(faces_t.device)
        self.init(verts_out, faces_out)

    MeshBackend.fill_holes = _patched_fill_holes
    MeshBackend._fill_holes_patched = True
    print(f"[o_voxel-native] _MeshBackend.fill_holes patched "
          f"(PIXAL3D_FILL_HOLES_PERIMETER={max_perim})", flush=True)



def _load_pamo_simplify_module():
    """Load ``pamo_simplify.py`` via importlib (bypasses package __init__).

    Same pattern as ``_load_cumesh_port_module`` — the bridge runs under
    the trellis-mac native venv where the Pixal3D project package isn't
    on ``sys.path``.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent / "pamo_simplify.py"
    spec = importlib.util.spec_from_file_location("pamo_simplify", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_simplify(postprocess_module) -> None:
    """Replace ``_MeshBackend.simplify`` with our pamo CPU/MPS port.

    The Metal port's ``simplify_step`` (precompiled in
    ``cumesh.metallib``) drives the mesh BELOW the requested face count
    and punches missing-face chunks into real surface.  Direct evidence
    from the ``1_img_fdg_cap.npz`` probe: target=1M, output=859k faces
    with visibly missing surface in wireframe view.  CUDA upstream's
    ``simplify.cu:128`` rejects winding-flipping collapses
    (``if (old_normal.dot(new_normal) < 0.0f) return false; // invalid``)
    and is likely the check the Metal port skips.

    We route through ``pamo_simplify.simplify`` which ports the upstream
    pamo algorithm to PyTorch (Phases 0-5 complete; Phase 6 = end-to-end
    validation, this patch).
    """
    MeshBackend = getattr(postprocess_module, "_MeshBackend", None)
    if MeshBackend is None or getattr(MeshBackend, "_simplify_patched", False):
        return

    # Session 7: PIXAL3D_SIMPLIFY=metal opts into Pedro's mtlmesh Metal port
    # (JIT-compiled from src/metal/simplify.metal).  The session-7 audit
    # confirmed simplify.metal:56 carries the winding-flip-reject check
    # faithfully from CUDA simplify.cu:127-129, refuting the earlier
    # "Metal port skips the check" diagnosis that motivated this pamo patch.
    # Session 15: Metal is now the default (validated end-to-end via CUDA-
    # mesh-through-Mac-cleanup exoneration test).  PIXAL3D_SIMPLIFY=pamo
    # opts out to the CPU pamo port for safe rollback / debugging.
    backend = os.environ.get("PIXAL3D_SIMPLIFY", "metal").lower()
    if backend == "metal":
        print(
            "[o_voxel-native] PIXAL3D_SIMPLIFY=metal — keeping native "
            "cumesh.CuMesh.simplify (mtlmesh JIT-compiled from simplify.metal; "
            "session-7 audit confirmed faithful to CUDA simplify.cu)",
            flush=True,
        )
        MeshBackend._simplify_patched = True  # idempotent across multiple calls
        return
    if backend != "pamo":
        print(
            f"[o_voxel-native] PIXAL3D_SIMPLIFY={backend!r} unknown; "
            "defaulting to pamo",
            flush=True,
        )

    try:
        pamo_mod = _load_pamo_simplify_module()
    except Exception as exc:
        print(
            f"[o_voxel-native] pamo_simplify unavailable ({exc}); "
            "falling back to native Metal simplify_step (expect artifacts)",
            flush=True,
        )
        return

    def _patched_simplify(self, target_num_faces: int, verbose: bool = False,
                          options: dict | None = None):
        # Unconditional activation print: confirms pamo (not the buggy Metal
        # port) is running, regardless of the verbose flag set by the caller.
        # Phase 6 validation depends on this trace.  Cheap (one print per
        # to_glb call, typically 2 calls).
        print(
            f"[pamo_simplify] ACTIVE — target={target_num_faces:,}, "
            f"input V={self.num_vertices:,} F={self.num_faces:,}",
            flush=True,
        )
        return pamo_mod.simplify(
            self, target_num_faces,
            verbose=verbose, options=options or {},
        )

    MeshBackend.simplify = _patched_simplify
    MeshBackend._simplify_patched = True
    print("[o_voxel-native] _MeshBackend.simplify patched to pamo_simplify",
          flush=True)



def _per_face_winding_fix(vertices: np.ndarray, faces: np.ndarray,
                          max_iter: int = 8, verbose: bool = False) -> np.ndarray:
    """Iteratively flip individual faces whose winding disagrees with neighbours.

    Background
    ----------
    CC-level orientation algorithms (CuMesh's BFS-based
    ``unify_face_orientations``, trimesh's ``fix_normals(multibody=True)``,
    Blender's ``recalc_outside``) treat each connected component as a unit:
    they either flip the whole CC or leave it alone.  That works on CUDA
    cumesh because its ``simplify_step`` rejects edge collapses that would
    invert face winding (``simplify.cu:128``), so the post-simplify mesh's
    inversions are coherent (whole CCs).  The Apple Silicon Metal port
    appears to skip that reject-flip check, so QEM scatters inverted faces
    *inside* otherwise-correctly-oriented CCs.  No CC-level algorithm can
    fix those — they sit there as transparent gaps in solid view (and as
    broken texture patches in textured renders).

    Algorithm
    ---------
    For each face F, build its 3 directed edges (v0→v1, v1→v2, v2→v0).
    Group all directed edges across the mesh by undirected key
    (min(a,b), max(a,b)).  Each manifold edge contributes a pair of faces;
    those two faces' windings are *consistent* iff they traverse the
    shared edge in **opposite** directions.

    Per face: ``n_consistent`` and ``n_inconsistent`` count agreement votes
    from manifold neighbours.  Faces with strict majority disagreement
    (``n_inconsistent > n_consistent``) get flipped.  Apply all flips
    simultaneously, then iterate.  ``> ``  (strict) avoids the
    1-cons-vs-1-incons oscillation between two paired faces.

    Limitations
    -----------
    On genuinely non-orientable patches (e.g. a Möbius-strip-like region
    introduced by simplify_step's worst collapses) no consistent assignment
    exists, so the algorithm leaves residual conflicts.  In practice the
    fraction is small.  Non-manifold edges (3+ faces) currently contribute
    one pair only (the first two in sorted order); a more complete handling
    would loop through all pairs but doesn't help for our usage where
    non-manifold edges are split before this runs anyway.
    """
    F = int(faces.shape[0])
    if F == 0:
        return faces
    new_faces = faces.copy().astype(np.int64)
    last_n_flip = -1
    for iteration in range(max_iter):
        src = new_faces[:, [0, 1, 2]].reshape(-1)
        dst = new_faces[:, [1, 2, 0]].reshape(-1)
        face_id = np.repeat(np.arange(F, dtype=np.int64), 3)

        key_lo = np.minimum(src, dst).astype(np.int64)
        key_hi = np.maximum(src, dst).astype(np.int64)
        key = (key_lo << 32) | key_hi

        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        sorted_face = face_id[order]
        sorted_src = src[order]
        sorted_dst = dst[order]

        diffs = np.diff(sorted_key) == 0
        pair_starts = np.where(diffs)[0]
        fa = sorted_face[pair_starts]
        fb = sorted_face[pair_starts + 1]
        sa_src = sorted_src[pair_starts]
        sa_dst = sorted_dst[pair_starts]
        sb_src = sorted_src[pair_starts + 1]
        sb_dst = sorted_dst[pair_starts + 1]

        consistent = (sa_src == sb_dst) & (sa_dst == sb_src)

        n_cons = np.zeros(F, dtype=np.int32)
        n_incon = np.zeros(F, dtype=np.int32)
        np.add.at(n_cons, fa, consistent.astype(np.int32))
        np.add.at(n_cons, fb, consistent.astype(np.int32))
        np.add.at(n_incon, fa, (~consistent).astype(np.int32))
        np.add.at(n_incon, fb, (~consistent).astype(np.int32))

        flip_mask = n_incon > n_cons
        n_flip = int(flip_mask.sum())
        if verbose:
            n_conflict_edges = int((~consistent).sum())
            print(
                f"[winding_fix] iter {iteration}: "
                f"conflict_edges={n_conflict_edges:,}, "
                f"flip_faces={n_flip:,}",
                flush=True,
            )
        if n_flip == 0 or n_flip == last_n_flip:
            break
        last_n_flip = n_flip
        new_faces[flip_mask] = new_faces[flip_mask][:, [0, 2, 1]]
    return new_faces.astype(faces.dtype)





def _load_npz(path: Path):
    data = np.load(str(path))
    vertices = torch.from_numpy(data["vertices"].astype(np.float32, copy=False))
    faces = torch.from_numpy(data["faces"].astype(np.int32, copy=False)).int()
    attrs = torch.from_numpy(data["attrs"].astype(np.float32, copy=False))
    coords = torch.from_numpy(data["coords"].astype(np.int32, copy=False)).int()
    origin = data["origin"].astype(np.float32, copy=False)
    voxel_size = float(data["voxel_size"])
    resolution = int(data["resolution"])
    grid_size = np.array([resolution, resolution, resolution], dtype=np.int32)
    aabb = np.stack([origin, origin + voxel_size * grid_size.astype(np.float32)], axis=0)
    return vertices, faces, attrs, coords, aabb, resolution


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export a Pixal3D mesh NPZ with native o_voxel")
    parser.add_argument("--input", required=True, help="Raw mesh/voxel NPZ")
    parser.add_argument("--output", required=True, help="Output GLB path")
    parser.add_argument("--debug-stages-dir", default=None, help="Write native cumesh geometry stages to this directory")
    parser.add_argument("--debug-stages-only", action="store_true", help="Only write debug stages; skip textured GLB export")
    parser.add_argument("--debug-raw-stages", action="store_true", help="Also export raw/full-face stages before simplification")
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    parser.add_argument("--remesh", action="store_true")
    parser.add_argument("--remesh-band", type=float, default=1.0)
    parser.add_argument("--remesh-project", type=float, default=0.0)
    parser.add_argument("--mesh-cluster-threshold", type=float, default=float(np.radians(90.0)))
    parser.add_argument("--mesh-cluster-refine-iterations", type=int, default=0)
    parser.add_argument("--mesh-cluster-global-iterations", type=int, default=1)
    parser.add_argument("--mesh-cluster-smooth-strength", type=float, default=1.0)
    parser.add_argument(
        "--output-transform",
        choices=sorted(OUTPUT_TRANSFORMS),
        default="pixal3d",
        help=(
            "Coordinate correction after native o_voxel.to_glb.  'pixal3d' "
            "matches the earlier MPS export orientation; 'o_voxel' preserves "
            "native upstream output; 'y_up' only undoes o_voxel's Y/Z swap."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    # ---- S17 back-port: selective cleanup skip + thresholds ----
    parser.add_argument("--skip-cleanup", action="store_true",
                        help="No-op all 6 cleanup ops (fill_holes/repair_nme/small_cc/unify/dedup/degen).")
    parser.add_argument("--skip-fill-holes", action="store_true")
    parser.add_argument("--skip-repair-nme", action="store_true")
    parser.add_argument("--skip-small-cc", action="store_true")
    parser.add_argument("--skip-unify", action="store_true")
    parser.add_argument("--skip-dedup", action="store_true")
    parser.add_argument("--skip-degen", action="store_true")
    parser.add_argument("--skip-simplify", action="store_true",
                        help="No-op cumesh.simplify (risky on multi-M-face meshes).")
    parser.add_argument("--skip-simplify-3x", action="store_true",
                        help="S21: skip only the destructive intermediate simplify(target*3) pass; "
                             "keep the final simplify(target). Cuts swiss-cheese boundary edges ~3x "
                             "on Mac MPS raw meshes by avoiding NME amplification on dense thin walls.")
    parser.add_argument("--small-cc-threshold", type=float, default=None,
                        help="Override min_area for remove_small_connected_components.  Try 1e-7.")
    parser.add_argument("--fill-holes-perimeter", type=float, default=None,
                        help="Override max_hole_perimeter for all fill_holes calls in o_voxel.to_glb.")
    parser.add_argument("--simplify-impl", choices=["metal", "fast"], default="metal",
                        help="Simplifier: 'metal' = Pedro's port (cumesh.simplify, default); "
                             "'fast' = fast_simplification (CPU QEM) — S17b test for compute_charts "
                             "chart-count blowup root cause.")
    # ---- S17b: compute_charts bisection + knob overrides ----
    parser.add_argument("--save-compute-charts-input", type=str, default=None,
                        help="Dump (vertices, faces, kwargs) handed to compute_charts to this .npz, "
                             "for cross-platform bisection (run on CUDA via scripts/cuda_compute_charts_test.py).")
    parser.add_argument("--compute-charts-area-weight", type=float, default=None,
                        help="Override area_penalty_weight kwarg to compute_charts (cumesh default 0.1; set 0 to disable).")
    parser.add_argument("--compute-charts-perim-weight", type=float, default=None,
                        help="Override perimeter_area_ratio_weight kwarg to compute_charts (cumesh default 0.0001; set 0 to disable).")
    parser.add_argument("--precharts-smooth", choices=["none", "laplacian", "taubin", "humphrey"], default="none",
                        help="Smooth mesh vertices right before compute_charts to heal QEM-induced "
                             "per-face normal perturbations (S17b root-cause fix).")
    parser.add_argument("--precharts-smooth-iterations", type=int, default=5,
                        help="Smoothing iterations (default 5).")
    parser.add_argument("--fill-holes-before-uv-perimeter", type=float, default=0.0,
                        help="OPTIONAL watertight enhancement (off when <=0; NOT faithful to the "
                             "CUDA reference). Run an extra fill_holes on the final mesh right "
                             "before compute_charts/uv_unwrap, closing thin-feature holes that the "
                             "post-simplify chain's built-in 3e-2 fill misses. Because it runs "
                             "pre-UV, the patches get UVs + texture and the chart count drops. "
                             "Value is max_hole_perimeter (e.g. 0.1); larger closes bigger holes "
                             "but may seal legitimate openings (mouths, cup interiors).")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault(
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
        str(Path(args.output).resolve().parent / "autotune_cache.json"),
    )
    os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "0")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

    from o_voxel import postprocess
    _patch_grid_sample_output(postprocess)
    _patch_repair_non_manifold_edges(postprocess)
    _patch_remove_small_connected_components(postprocess)  # opt-in via PIXAL3D_SMALL_CC
    _patch_simplify(postprocess)
    if args.fill_holes_perimeter is not None:
        os.environ["PIXAL3D_FILL_HOLES_PERIMETER"] = str(args.fill_holes_perimeter)
    _patch_fill_holes(postprocess)  # opt-in via PIXAL3D_FILL_HOLES_PERIMETER
    _patch_unify_face_orientations(postprocess)  # default on; opt out via PIXAL3D_UNIFY_FACE_ORIENTATIONS=metal
    # ---- S17 back-port: selective cleanup-to-noop overrides ----
    _patch_cleanup_to_noop(
        postprocess,
        fill_holes=args.skip_cleanup or args.skip_fill_holes,
        repair_nme=args.skip_cleanup or args.skip_repair_nme,
        small_cc=args.skip_cleanup or args.skip_small_cc,
        unify=args.skip_cleanup or args.skip_unify,
        dedup=args.skip_cleanup or args.skip_dedup,
        degen=args.skip_cleanup or args.skip_degen,
    )
    if args.skip_simplify:
        _patch_simplify_to_noop(postprocess)
    elif getattr(args, "skip_simplify_3x", False):
        _patch_simplify_skip_3x(postprocess)
    elif args.simplify_impl == "fast":
        _patch_simplify_fast(postprocess)
    if args.small_cc_threshold is not None and not (args.skip_cleanup or args.skip_small_cc):
        _patch_small_cc_threshold(postprocess, args.small_cc_threshold)
    # ---- S17b: compute_charts bisection + knob overrides ----
    _patch_compute_charts(
        postprocess,
        save_input_path=args.save_compute_charts_input,
        area_weight=args.compute_charts_area_weight,
        perim_weight=args.compute_charts_perim_weight,
    )
    # Apply smoothing patch AFTER compute_charts patch so smoothing happens
    # FIRST in call order (mesh is smoothed, then saved, then compute_charts runs).
    _patch_presmooth_for_compute_charts(
        postprocess,
        mode=args.precharts_smooth,
        iterations=args.precharts_smooth_iterations,
    )
    _patch_fill_holes_before_uv(postprocess, perimeter=args.fill_holes_before_uv_perimeter)

    vertices, faces, attrs, coords, aabb, resolution = _load_npz(Path(args.input))
    if args.verbose:
        print(
            f"[o_voxel-native] input mesh: {vertices.shape[0]:,} vertices, "
            f"{faces.shape[0]:,} faces, resolution={resolution}"
        )

    if args.debug_stages_dir is not None:
        last_stage = _export_debug_stages(
            vertices=vertices,
            faces=faces,
            output_dir=Path(args.debug_stages_dir),
            decimation_target=args.decimation_target,
            output_transform=args.output_transform,
            verbose=args.verbose,
            include_raw=args.debug_raw_stages,
        )
        if args.debug_stages_only:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(last_stage, output)
            if args.verbose:
                print(f"[o_voxel-native] copied final debug stage to {output}")
            return 0

    textured_mesh = postprocess.to_glb(
        vertices=vertices,
        faces=faces,
        attr_volume=attrs,
        coords=coords,
        attr_layout=DEFAULT_LAYOUT,
        aabb=aabb,
        grid_size=resolution,
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=args.remesh,
        remesh_band=args.remesh_band,
        remesh_project=args.remesh_project,
        mesh_cluster_threshold_cone_half_angle_rad=args.mesh_cluster_threshold,
        mesh_cluster_refine_iterations=args.mesh_cluster_refine_iterations,
        mesh_cluster_global_iterations=args.mesh_cluster_global_iterations,
        mesh_cluster_smooth_strength=args.mesh_cluster_smooth_strength,
        verbose=args.verbose,
        use_tqdm=not args.no_tqdm,
    )
    _force_opaque_material(textured_mesh)
    # Mirror the winding-repair pass from _write_geometry_stage onto the
    # textured pipeline output.  Stage 1: per-face flip of scattered
    # inverted faces (Metal simplify residue).  Stage 2: per-CC outward
    # direction check (catches any still-inverted whole CC).
    try:
        for geom in _iter_geometries(textured_mesh):
            if hasattr(geom, "faces"):
                new_faces_np = _per_face_winding_fix(
                    np.asarray(geom.vertices),
                    np.asarray(geom.faces),
                    max_iter=8,
                    verbose=args.verbose,
                )
                geom.faces = new_faces_np
        if args.verbose:
            print("[o_voxel-native] _per_face_winding_fix applied to textured mesh", flush=True)
    except Exception as exc:
        print(
            f"[o_voxel-native] _per_face_winding_fix (textured) failed ({exc}); "
            "skipping per-face winding repair",
            flush=True,
        )
    try:
        import trimesh.repair
        for geom in _iter_geometries(textured_mesh):
            if hasattr(geom, "faces"):
                trimesh.repair.fix_normals(geom, multibody=True)
        if args.verbose:
            print("[o_voxel-native] trimesh.fix_normals applied to textured mesh", flush=True)
    except Exception as exc:
        print(
            f"[o_voxel-native] trimesh.fix_normals (textured) failed ({exc}); "
            "skipping per-CC outward orientation fix",
            flush=True,
        )
    _apply_output_transform(textured_mesh, args.output_transform)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # include_normals=True: emit the NORMAL accessor. to_glb already computes
    # smooth vertex normals (postprocess.py compute_vertex_normals), but trimesh's
    # glb exporter drops them by default (include_normals=None) -> viewers fall
    # back to FLAT per-face normals -> faceted/"chunky" look. The CUDA reference
    # GLB ships normals; matching it makes Mac render smooth.
    try:
        textured_mesh.export(str(output), extension_webp=True, include_normals=True)
    except TypeError:
        textured_mesh.export(str(output), include_normals=True)

    if args.verbose:
        print(f"[o_voxel-native] wrote {output} ({output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
