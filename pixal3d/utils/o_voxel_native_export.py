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
    # Native o_voxel writes q=(x, z, -y).  Empirical Blender check vs the
    # ground-truth Pixal3D upstream output shows we need an additional
    # scale_Y(-1) and a 180° rotation around the axis (0, 1/√2, -1/√2)
    # (quaternion wxyz=(0, 0, 1/√2, -1/√2)) applied on top of the previous
    # (x, -z, -y) conversion.  Composing them collapses to a pure 180°
    # rotation around world Y: (x, y, z) -> (-x, y, -z).  Pure rotation
    # (det=+1) so the face-winding flip below is skipped automatically.
    "pixal3d": np.array([
        [-1, 0,  0, 0],
        [ 0, 1,  0, 0],
        [ 0, 0, -1, 0],
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
        new_verts_np, new_faces_np = cpu_repair(verts_np, faces_np, verbose=False)
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


def parse_args():
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    _patch_fill_holes(postprocess)  # opt-in via PIXAL3D_FILL_HOLES_PERIMETER

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
    try:
        textured_mesh.export(str(output), extension_webp=True)
    except TypeError:
        textured_mesh.export(str(output))

    if args.verbose:
        print(f"[o_voxel-native] wrote {output} ({output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
