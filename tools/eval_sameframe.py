"""Same-frame RGB-D reconstruction evaluation.

The one hard rule of this tool: **the mesh and the depth observation of a group
must come from the same frame**. A mesh exported for state ``closed`` was fitted
in the reference camera of ``reference_geometry.json -> references.closed.frame_id``
(currently ``closed_front_4``, *not* ``closed_front_0``). Scoring it against any
other frame mixes head pose and facial expression into the residual, which is
what made the earlier ``runtime/loss_analysis`` numbers unusable.

What is measured, per (group, state, frame):

* a fixed-seed sample of full-face depth pixels inside ``face_mask & depth > 0``
  (not the 68 landmarks, not ``xyz/valid``);
* a camera ray through every sampled pixel, pushed along the optical axis;
* the depth difference in millimetres along the optical axis (-Z) at the first
  intersection with the mesh, kept only if the hit passes the same gates the
  fitter uses (front facing, inside the face mask, within ``--max-depth-mm``);
* misses are counted in ``n_missing`` and lower ``coverage``; a miss is never
  scored as a 0 mm residual. Zero usable samples yields ``null`` metrics plus
  ``status="no_valid_samples"``.

Units and conventions (checked, never assumed silently):
depth ``uint16`` millimetres, mesh metres, camera ``X_right_Y_up_Z_backward``,
intrinsics ``(fx, fy, cx, cy)``, depth along ``-Z``.

Usage:
  runtime\\env\\python.exe tools\\eval_sameframe.py ^
      --run runtime\\outputs\\<run> --output runtime\\eval\\<label>
  runtime\\env\\python.exe tools\\eval_sameframe.py ^
      --run runtime\\outputs\\<new> --baseline-metrics runtime\\eval\\<old>\\metrics.json ^
      --output runtime\\eval\\compare_<label>
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

SIDES_ORDER = ("front", "left", "right")
SIDES_DEFAULT = ("front", "left", "right")
STATES_ORDER = ("closed", "open")
DEFAULT_MAX_DEPTH_MM = 30.0
DEFAULT_SAMPLES = 8000
DEFAULT_SEED = 20261101
DEPTH_UNIT = "mm"
MESH_UNIT = "m"
COORDINATE_CONVENTION = "X_right_Y_up_Z_backward"
# FaceCap-style back-face test: the triangle normal must point at the camera.
# Coplanar triangles (|cos| below this) are kept, so an exact 2 mm plane shift is
# still measurable instead of silently dropped.
MIN_FRONT_FACING_COS = 0.0
# Subset of the sample that is independently re-integrated to confirm that the
# vectorised ray/triangle path and the per-sample path agree.
CROSS_CHECK_SAMPLES = 256
# Rays are cast against triangles in blocks so an (R, T) matrix never explodes.
RAY_TRIANGLE_BLOCK = 512


# --------------------------------------------------------------------------- #
# input contracts
# --------------------------------------------------------------------------- #
def load_sequence(run_dir: Path) -> dict:
    path = run_dir / "prepared" / "sequence.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing prepared sequence manifest: {path}")
    sequence = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in (("depth_unit", DEPTH_UNIT), ("mesh_unit", MESH_UNIT),
                          ("coordinate_convention", COORDINATE_CONVENTION)):
        if sequence.get(key) != expected:
            raise ValueError(f"{path}: {key}={sequence.get(key)!r}, expected {expected!r}")
    if not sequence.get("frames"):
        raise ValueError(f"{path}: no frames declared")
    ids = [frame["id"] for frame in sequence["frames"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate frame ids")
    return sequence


def load_reference_geometry(run_dir: Path) -> tuple[Path, dict]:
    """Return the dense stage's reference geometry (the stage the delivery exports).

    The frame ids are read, never guessed: they are the contract this tool enforces.
    """
    path = run_dir / "reconstruction" / "dense" / "reference_geometry.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing reference geometry: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    references = payload.get("references")
    if not isinstance(references, dict):
        raise ValueError(f"{path}: no 'references' block")
    for state in STATES_ORDER:
        entry = references.get(state)
        if not isinstance(entry, dict) or not entry.get("frame_id"):
            raise ValueError(f"{path}: references.{state}.frame_id missing")
        if entry.get("mesh_unit") not in (None, MESH_UNIT):
            raise ValueError(f"{path}: references.{state}.mesh_unit != {MESH_UNIT!r}")
    return path, references


def resolve_frames(run_dir: Path, sequence: dict, references: dict) -> dict:
    """Map each state to its reference frame, enforcing same-frame evaluation.

    ``global_index`` is the frame's position in the ``sequence.json`` frame list
    and is the row to use in every per-frame array of ``parameters.npz``. It is
    distinct from the manifest's ``index`` field, which is the index *inside the
    capture segment*: ``open_front_4`` has ``index == 4`` but ``global_index == 19``.
    """
    by_id = {frame["id"]: frame for frame in sequence["frames"]}
    order = [frame["id"] for frame in sequence["frames"]]
    resolved = {}
    for state in STATES_ORDER:
        frame_id = references[state]["frame_id"]
        frame = by_id.get(frame_id)
        if frame is None:
            raise ValueError(
                f"reference frame {frame_id!r} for state {state!r} is not in sequence.json "
                f"({len(by_id)} frames declared)"
            )
        if frame["state"] != state:
            raise ValueError(
                f"reference frame {frame_id!r} belongs to state {frame['state']!r}, "
                f"but reference_geometry.json maps it to {state!r}"
            )
        npz = run_dir / "prepared" / frame["file"]
        if not npz.is_file():
            raise FileNotFoundError(f"missing prepared frame arrays: {npz}")
        resolved[state] = {
            "frame_id": frame_id,
            "view": frame["view"],
            "segment": frame["segment"],
            "global_index": int(order.index(frame_id)),
            "segment_index": int(frame["index"]),
            "file": frame["file"],
            "npz": str(npz),
        }
    for state in STATES_ORDER:
        declared = references[state].get("source")
        candidate = by_id[resolved[state]["frame_id"]].get("source")
        if declared is not None and candidate is not None and Path(declared) != Path(candidate):
            raise ValueError(
                f"references.{state}.source does not match sequence.json for "
                f"{resolved[state]['frame_id']!r}: {declared} vs {candidate}"
            )
    return resolved


def load_observation(npz_path: Path) -> dict:
    with np.load(npz_path) as data:
        observation = {
            "depth_mm": np.array(data["depth"], dtype=np.float64),
            "intrinsics": np.array(data["intrinsics"], dtype=np.float64),
            "face_mask": np.array(data["face_mask"], dtype=bool),
        }
    depth = observation["depth_mm"]
    intrinsics = observation["intrinsics"]
    face_mask = observation["face_mask"]
    if depth.ndim != 2:
        raise ValueError(f"{npz_path}: depth must be 2-D, got {depth.shape}")
    if face_mask.shape != depth.shape:
        raise ValueError(f"{npz_path}: face_mask {face_mask.shape} != depth {depth.shape}")
    if intrinsics.shape != (4,):
        raise ValueError(f"{npz_path}: intrinsics must be (fx, fy, cx, cy), got {intrinsics.shape}")
    if not np.isfinite(intrinsics).all() or (intrinsics[:2] <= 0).any():
        raise ValueError(f"{npz_path}: invalid intrinsics {intrinsics}")
    if not np.isfinite(depth).all() or depth.min() < 0:
        raise ValueError(f"{npz_path}: depth is not a finite non-negative millimetre map")
    return observation


def load_mesh(run_dir: Path, state: str,
              global_index: int | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """Return ``(vertices, faces, source)`` for one state's mesh.

    When ``global_index`` is given the geometry is taken from
    ``reconstruction/dense/parameters.npz -> dense_vertices[global_index]``, i.e.
    the per-frame dense result for *that exact frame*. This makes the same-frame
    guarantee structural instead of relying on the export happening to be right.

    ``delivery/head_<state>_m.obj`` is only a fallback for runs that predate the
    per-frame arrays; it is a single reference-frame export and is the thing the
    old loss analysis got wrong, so its use is recorded in ``source`` and checked
    against the per-frame array whenever both exist.
    """
    faces, faces_source = _load_faces(run_dir, state)
    if global_index is not None:
        parameters = run_dir / "reconstruction" / "dense" / "parameters.npz"
        if not parameters.is_file():
            raise FileNotFoundError(
                f"--require-per-frame geometry needs {parameters}; it is missing for "
                f"state {state!r}"
            )
        with np.load(parameters) as data:
            if "dense_vertices" not in data.files:
                raise ValueError(f"{parameters}: no 'dense_vertices' per-frame array")
            stacked = np.array(data["dense_vertices"], dtype=np.float64)
        if stacked.ndim != 3 or stacked.shape[2] != 3:
            raise ValueError(f"{parameters}: dense_vertices must be (n_frames, V, 3)")
        if not 0 <= global_index < stacked.shape[0]:
            raise IndexError(
                f"global_index {global_index} out of range for "
                f"dense_vertices with {stacked.shape[0]} frames"
            )
        vertices = stacked[global_index]
        if len(faces) and faces.max() >= len(vertices):
            raise ValueError(
                f"dense_faces index {faces.max()} exceeds {len(vertices)} vertices"
            )
        source = f"parameters.npz:dense_vertices[{global_index}]"
        _verify_against_delivery(run_dir, state, vertices)
        return vertices, faces, source

    vertices, faces_obj = _load_delivery_obj(run_dir, state)
    return vertices, faces_obj, "delivery/obj"


def _load_faces(run_dir: Path, state: str) -> tuple[np.ndarray, str]:
    parameters = run_dir / "reconstruction" / "dense" / "parameters.npz"
    if parameters.is_file():
        with np.load(parameters) as data:
            if "dense_faces" in data.files:
                faces = np.array(data["dense_faces"], dtype=np.int64)
                _check_faces(faces, parameters)
                return faces, "parameters.npz:dense_faces"
    vertices, faces = _load_delivery_obj(run_dir, state)
    del vertices
    return faces, "delivery/obj"


def _check_faces(faces: np.ndarray, origin) -> None:
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{origin}: faces must be (F, 3), got {faces.shape}")


def _load_delivery_obj(run_dir: Path, state: str) -> tuple[np.ndarray, np.ndarray]:
    manifest_path = run_dir / "delivery" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing delivery manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mesh_info = manifest.get("mesh")
    if not isinstance(mesh_info, dict):
        raise ValueError(f"{manifest_path}: no 'mesh' block")
    if mesh_info.get("unit") != MESH_UNIT:
        raise ValueError(f"{manifest_path}: mesh.unit={mesh_info.get('unit')!r}, expected {MESH_UNIT!r}")
    entry = (mesh_info.get("states") or {}).get(state)
    if not isinstance(entry, dict) or not entry.get("path"):
        raise ValueError(f"{manifest_path}: mesh.states.{state}.path missing")
    mesh_path = run_dir / "delivery" / entry["path"]
    if not mesh_path.is_file():
        raise FileNotFoundError(f"missing delivery mesh for state {state!r}: {mesh_path}")
    mesh = trimesh.load(str(mesh_path), process=False, maintain_order=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{mesh_path}: vertices must be (V, 3), got {vertices.shape}")
    _check_faces(faces, mesh_path)
    return vertices, faces


DELIVERY_MATCH_TOLERANCE_M = 1e-6


def _verify_against_delivery(run_dir: Path, state: str, vertices: np.ndarray) -> None:
    """Cross-check the per-frame array against the exported mesh, if present.

    They must agree for the reference frame the export claims. A disagreement
    means ``manifest.json`` and ``parameters.npz`` describe different frames --
    precisely the mixing this tool exists to prevent.
    """
    try:
        obj_vertices, obj_faces = _load_delivery_obj(run_dir, state)
    except FileNotFoundError:
        return
    if obj_vertices.shape != vertices.shape:
        raise ValueError(
            f"{state}: dense_vertices row has {vertices.shape[0]} vertices but the "
            f"delivery mesh has {obj_vertices.shape[0]}; topologies differ"
        )
    del obj_faces
    gap = float(np.linalg.norm(vertices - obj_vertices, axis=1).max())
    if gap > DELIVERY_MATCH_TOLERANCE_M:
        raise ValueError(
            f"{state}: dense_vertices[row] differs from the delivery mesh by {gap:.6g} m "
            f"(tolerance {DELIVERY_MATCH_TOLERANCE_M} m); the exported mesh is not that "
            "frame's geometry"
        )


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def sample_pixels(face_mask: np.ndarray, depth_mm: np.ndarray, samples: int, seed: int) -> np.ndarray:
    """Fixed-seed sample of ``(x, y)`` pixels inside ``face_mask & depth > 0``."""
    if samples <= 0:
        raise ValueError("--samples must be positive")
    ys, xs = np.nonzero(face_mask & (depth_mm > 0))
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    rng = np.random.default_rng(seed)
    count = min(samples, len(xs))
    take = rng.choice(len(xs), size=count, replace=False)
    return np.column_stack((xs[take], ys[take])).astype(np.int64)


def pixel_rays(pixels: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Unit direction of the camera ray through each pixel (X right, Y up, Z back)."""
    fx, fy, cx, cy = intrinsics
    x = (pixels[:, 0].astype(np.float64) - cx) / fx
    y = -(pixels[:, 1].astype(np.float64) - cy) / fy
    directions = np.column_stack((x, y, -np.ones(len(pixels), dtype=np.float64)))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# ray casting
# --------------------------------------------------------------------------- #
def _moller_trumbore(directions: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First-forward-hit ray/triangle test using the reported winding.

    ``directions`` is ``(R, 3)`` (unit, from the camera origin) and ``triangles``
    is ``(T, 3, 3)``; the result is ``(R, T)``. Returns ``(t, front_facing)`` with
    ``t = inf`` where nothing was hit. Triangles are not rewound: a back-facing
    wall stays behind the front surface, exactly as a physical measurement sees it.
    """
    direction = directions[:, np.newaxis, :]                            # (R, 1, 3)
    v0 = triangles[np.newaxis, :, 0]                                    # (1, T, 3)
    edge1 = triangles[np.newaxis, :, 1] - v0
    edge2 = triangles[np.newaxis, :, 2] - v0
    pvec = np.cross(direction, edge2)                                   # (R, T, 3)
    det = np.einsum("rtk,rtk->rt", pvec, edge1)
    admissible = np.abs(det) > 1e-14
    inv_det = np.zeros_like(det)
    inv_det[admissible] = 1.0 / det[admissible]
    tvec = -v0                                                          # camera at the origin
    u = np.einsum("rtk,rtk->rt", tvec, pvec) * inv_det
    qvec = np.cross(tvec, edge1)                                        # (R, T, 3)
    v = np.einsum("rtk,rtk->rt", direction, qvec) * inv_det
    t = np.einsum("rtk,rtk->rt", edge2, qvec) * inv_det
    inside = admissible & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 0.0)
    t = np.where(inside, t, np.inf)
    normal = np.cross(edge1, edge2)[0]                                  # (T, 3)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-15)
    front_facing = np.einsum("rtk,tk->rt", direction, -normal) > MIN_FRONT_FACING_COS
    return t, front_facing


def cast_rays(vertices: np.ndarray, faces: np.ndarray,
              pixels: np.ndarray, directions: np.ndarray) -> dict:
    """Intersect every ray with the whole mesh and return the per-ray depth.

    A ray that only meets back-facing triangles is a miss: the front surface is
    not there, and treating it as a measurement would flatter the reconstruction.
    """
    triangles = vertices[faces]
    n_rays = len(pixels)
    n_tri = len(faces)
    if n_tri == 0 or n_rays == 0:
        return {"hit": np.zeros(n_rays, dtype=bool),
                "accepted": np.zeros(n_rays, dtype=bool),
                "depth_mm": np.full(n_rays, np.nan),
                "ray_depth_mm": np.full(n_rays, np.nan),
                "front_facing": np.zeros(n_rays, dtype=bool),
                "t_m": np.full(n_rays, np.inf)}
    best_t = np.full(n_rays, np.inf)
    best_front = np.zeros(n_rays, dtype=bool)
    for start in range(0, n_tri, RAY_TRIANGLE_BLOCK):
        block = triangles[start:start + RAY_TRIANGLE_BLOCK]
        t, front = _moller_trumbore(directions, block)
        nearest = np.argmin(t, axis=1)              # (R,)
        rows = np.arange(n_rays)
        block_best = t[rows, nearest]
        better = block_best < best_t
        best_t = np.where(better, block_best, best_t)
        best_front = np.where(better, front[rows, nearest], best_front)
    hit = np.isfinite(best_t)
    # ``best_t`` is the distance travelled along the *unit* ray. The metric is the
    # optical-axis (Z) depth, so convert with |d_z| before differencing against a
    # uint16 depth pixel; using the path length instead inflates residuals by up
    # to ~0.5 mm at the frame corners and looks like real surface error.
    axis_scale = np.abs(directions[:, 2])
    ray_depth_mm = np.where(hit, best_t * axis_scale * 1000.0, np.nan)
    return {
        "hit": hit,
        "depth_mm": ray_depth_mm,
        "ray_depth_mm": ray_depth_mm,
        "front_facing": best_front,
        "t_m": best_t,
    }


def evaluate_group(vertices: np.ndarray, faces: np.ndarray, observation: dict,
                   pixels: np.ndarray, max_depth_mm: float) -> dict:
    """Score one sampled ray set against one mesh, keeping misses as misses."""
    n_sampled = len(pixels)
    if n_sampled == 0:
        return _empty_metrics(n_sampled, status="no_samples")
    directions = pixel_rays(pixels, observation["intrinsics"])
    result = cast_rays(vertices, faces, pixels, directions)
    u = np.rint(pixels[:, 0]).astype(np.int64)
    v = np.rint(pixels[:, 1]).astype(np.int64)
    observed = observation["depth_mm"][v, u]
    ray_depth = result["ray_depth_mm"]
    residual = ray_depth - observed
    # ``--max-depth-mm`` is the *residual* gate, the same quantity the fitter
    # thresholds on: the mesh must agree with the measurement to within that
    # distance. It is not the absolute camera range.
    accepted = (
        result["hit"]
        & result["front_facing"]
        & observation["face_mask"][v, u]
        & (observed > 0)
        & np.isfinite(residual)
        & (np.abs(residual) <= max_depth_mm)
    )
    n_used = int(accepted.sum())
    n_missing = int(n_sampled - n_used)
    if n_used == 0:
        metrics = _empty_metrics(n_sampled, status="no_valid_samples")
        metrics["n_missing"] = n_missing
        metrics["coverage"] = 0.0
        metrics["n_hit"] = int(result["hit"].sum())
        metrics["n_front_facing"] = int((result["hit"] & result["front_facing"]).sum())
        return metrics
    residuals = residual[accepted]
    coverage = n_used / n_sampled
    metrics = {
        "status": "ok",
        "n_sampled": int(n_sampled),
        "n_used": n_used,
        "n_missing": n_missing,
        "coverage": float(coverage),
        "mae_mm": float(np.mean(np.abs(residuals))),
        "rmse_mm": float(np.sqrt(np.mean(residuals ** 2))),
        "p95_mm": float(np.percentile(np.abs(residuals), 95)),
        "signed_mean_mm": float(np.mean(residuals)),
        "max_abs_mm": float(np.max(np.abs(residuals))),
        "n_hit": int(result["hit"].sum()),
        "n_front_facing": int((result["hit"] & result["front_facing"]).sum()),
        "n_rejected_by_residual_gate": int(
            (result["hit"] & result["front_facing"] & (np.abs(residual) > max_depth_mm)).sum()
        ),
    }
    cross = _cross_check(vertices, faces, pixels[accepted], directions[accepted],
                         ray_depth[accepted]) if n_used >= 2 else {}
    metrics["cross_check"] = cross
    return metrics


def _cross_check(vertices: np.ndarray, faces: np.ndarray, pixels: np.ndarray,
                 directions: np.ndarray, depth_mm: np.ndarray) -> dict:
    """Re-integrate a subset one ray at a time. Catches block-wise indexing errors."""
    count = min(CROSS_CHECK_SAMPLES, len(pixels))
    step = max(1, len(pixels) // count)
    ids = np.arange(0, len(pixels), step)[:count]
    triangles = vertices[faces]
    errors = []
    for i in ids:
        t, _ = _moller_trumbore(directions[i:i + 1], triangles)
        # Same axis conversion as the vectorised path, applied per ray.
        axis = abs(float(directions[i, 2]))
        errors.append(float(np.min(t) * axis * 1000.0 - depth_mm[i]))
    return {
        "n": int(len(ids)),
        "max_abs_difference_mm": float(np.max(np.abs(errors))) if errors else 0.0,
    }


def _empty_metrics(n_sampled: int, status: str) -> dict:
    return {
        "status": status,
        "n_sampled": int(n_sampled),
        "n_used": 0,
        "n_missing": int(n_sampled),
        "coverage": 0.0 if n_sampled else None,
        "mae_mm": None,
        "rmse_mm": None,
        "p95_mm": None,
        "signed_mean_mm": None,
        "max_abs_mm": None,
        "cross_check": {},
    }


# --------------------------------------------------------------------------- #
# geometry checks
# --------------------------------------------------------------------------- #
def load_flame_base(run_dir: Path, sequence: dict, frames: dict) -> dict:
    """FLAME base vertices in the reference camera of each state's own frame.

    ``posed_vertices`` in ``reconstruction/dense/parameters.npz`` is
    ``(n_frames, V, 3)`` and already lives in that frame's reference camera: its
    row for a state equals the first ``V`` vertices of that state's delivery mesh
    (identical centroid). The row index is the frame's global position in
    ``sequence.json`` -- *not* ``source_frame``, which is only the index inside
    its own capture segment (``open_front_4`` is also ``source_frame == 4``).
    The row is verified against ``frame_id`` rather than assumed. This is the
    undeformed base used for "max displacement" only.

    Note: ``front_anchor_vertices`` is stored in the *animated model* frame and
    its ``animated_model_to_reference_camera_4x4`` sidecar does not reproduce the
    delivery mesh when applied directly, so it is deliberately not used here.
    """
    parameters = run_dir / "reconstruction" / "dense" / "parameters.npz"
    if not parameters.is_file():
        return {}
    with np.load(parameters) as data:
        if "posed_vertices" not in data.files:
            return {}
        stacked = np.array(data["posed_vertices"], dtype=np.float64)
    if stacked.ndim != 3:
        raise ValueError(f"{parameters}: posed_vertices must be (n_frames, V, 3)")
    order = [frame["id"] for frame in sequence["frames"]]
    if len(order) != stacked.shape[0]:
        raise ValueError(
            f"{parameters}: {stacked.shape[0]} frames of vertices but sequence.json "
            f"declares {len(order)} frames"
        )
    base = {}
    for state in STATES_ORDER:
        frame_id = frames[state]["frame_id"]
        row = frames[state]["global_index"]
        if order[row] != frame_id:
            raise ValueError(
                f"global_index {row} does not correspond to reference frame {frame_id!r} "
                f"(sequence order gives {order[row]!r})"
            )
        base[state] = stacked[row]
    return base


def geometry_metrics(vertices: np.ndarray, faces: np.ndarray,
                     base_vertices: np.ndarray | None) -> dict:
    """Topology and orientation checks.

    The flip test compares each triangle normal with the area-weighted average
    normal of its own neighbourhood. A face whose normal disagrees with its
    neighbours is wound backwards *relative to the surface it belongs to*.

    A centroid-based test is deliberately NOT used: this delivery is an open
    head shell, not a closed body (``is_watertight == False``), so normals on the
    nose legitimately point away from the centroid and a centroid test would
    report ~20% false flips on a perfectly valid mesh.
    """
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    triangles = vertices[faces] if len(faces) else np.empty((0, 3, 3))
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) / 2.0
    if len(faces):
        normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-15)
        vertex_normals = np.zeros_like(vertices)
        for corner in range(3):
            np.add.at(vertex_normals, faces[:, corner], normals * areas[:, None])
        neighbourhood = vertex_normals[faces].mean(axis=1)
        agreement = np.einsum("ij,ij->i", normals, neighbourhood)
        flipped = int(np.sum(agreement < 0))
    else:
        flipped = 0
    ratio = float(flipped / len(faces)) if len(faces) else 0.0
    metrics = {
        "n_vertices": int(len(vertices)),
        "n_faces": int(len(faces)),
        "vertices_finite": bool(np.isfinite(vertices).all()),
        "face_indices_valid": bool(len(faces) == 0 or (faces.min() >= 0 and faces.max() < len(vertices))),
        "extent_m": [float(x) for x in extents],
        "metre_scale": bool(0.05 < float(extents.max()) < 0.80),
        "flipped_faces": flipped,
        "flipped_face_ratio": ratio,
        "flip_criterion": "face normal disagrees with its neighbourhood's area-weighted normal",
        "min_signed_area_ratio_ok": bool(ratio < 0.05),
        "total_area_mm2": float(areas.sum() * 1e6),
    }
    if base_vertices is not None:
        if len(base_vertices) > len(vertices):
            raise ValueError(
                f"FLAME base has {len(base_vertices)} vertices, more than the delivery mesh "
                f"({len(vertices)}); the dense subdivision never discards vertices"
            )
        # The delivery topology is the FLAME 5023 vertices subdivided in place, so the
        # original FLAME vertices are its first rows; compare only those.
        base = base_vertices[:len(vertices)]
        metrics["flame_base_vertex_count"] = int(len(base_vertices))
        metrics["max_displacement_mm_from_flame_base"] = float(
            np.linalg.norm(vertices[:len(base)] - base, axis=1).max() * 1000.0
        )
        metrics["mean_displacement_mm_from_flame_base"] = float(
            np.linalg.norm(vertices[:len(base)] - base, axis=1).mean() * 1000.0
        )
    else:
        metrics["flame_base_vertex_count"] = 0
        metrics["max_displacement_mm_from_flame_base"] = None
        metrics["mean_displacement_mm_from_flame_base"] = None
    return metrics


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def evaluate(run_dir: Path, max_depth_mm: float = DEFAULT_MAX_DEPTH_MM,
             samples: int = DEFAULT_SAMPLES, seed: int = DEFAULT_SEED) -> dict:
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    sequence = load_sequence(run_dir)
    reference_path, references = load_reference_geometry(run_dir)
    frames = resolve_frames(run_dir, sequence, references)

    by_key = {(frame["state"], frame["view"]): frame["id"] for frame in sequence["frames"]}
    groups = [
        (state, view)
        for state in STATES_ORDER
        for view in SIDES_DEFAULT
        if (state, view) in by_key
    ]

    flame_base = load_flame_base(run_dir, sequence, frames)

    wanted = {by_key[key] for key in groups} | {item["frame_id"] for item in frames.values()}
    frame_cache: dict[str, dict] = {}
    pixel_cache: dict[str, np.ndarray] = {}
    for frame_id in sorted(wanted):
        frame = next(item for item in sequence["frames"] if item["id"] == frame_id)
        observation = load_observation(run_dir / "prepared" / frame["file"])
        frame_cache[frame_id] = observation
        pixel_cache[frame_id] = sample_pixels(observation["face_mask"], observation["depth_mm"],
                                             samples, seed)

    meshes = {
        state: load_mesh(run_dir, state, frames[state]["global_index"])
        for state in STATES_ORDER
    }

    results = []
    geometry = {}
    for state in STATES_ORDER:
        vertices, mesh_faces, _ = meshes[state]
        geometry[state] = geometry_metrics(vertices, mesh_faces, flame_base.get(state))
    geometry["topology_shared"] = bool(len(meshes["closed"][1]) and len(meshes["open"][1])
                                       and np.array_equal(meshes["closed"][1], meshes["open"][1]))

    for state in STATES_ORDER:
        vertices, mesh_faces, mesh_source = meshes[state]
        identity = frames[state]
        metrics = evaluate_group(vertices, mesh_faces, frame_cache[identity["frame_id"]],
                                 pixel_cache[identity["frame_id"]], max_depth_mm)
        results.append({
            "group": identity["view"],
            "state": state,
            "frame_id": identity["frame_id"],
            "global_index": identity["global_index"],
            "mesh_source": mesh_source,
            "frame_role": "identity",
            "reference_frame_id": identity["frame_id"],
            **metrics,
        })
    return {
        "run": str(run_dir),
        "sampling": {
            "seed": int(seed),
            "samples_per_frame": int(samples),
            "region": "face_mask & depth > 0",
            "point_source": "full-face depth pixels (not the 68 landmarks)",
            "ray": "camera origin through pixel, unit direction",
            "residual": "absolute optical-axis depth difference at the first front hit (mm)",
            "max_depth_mm": float(max_depth_mm),
            "max_depth_meaning": "maximum |residual| in mm for a ray to be used; not a camera range",
            "min_front_facing_cos": MIN_FRONT_FACING_COS,
            "missing_policy": "counted in n_missing/coverage, never scored as 0 mm",
        },
        "units": {"depth": DEPTH_UNIT, "mesh": MESH_UNIT,
                  "coordinate_convention": COORDINATE_CONVENTION},
        "reference_geometry": str(reference_path),
        "reference_frames": {state: frames[state]["frame_id"] for state in STATES_ORDER},
        "frames_used": {state: frames[state] for state in STATES_ORDER},
        "sequence_frames": {f"{frame['state']}/{frame['view']}": frame["id"]
                            for frame in sequence["frames"]},
        "geometric_validity": geometry,
        "metrics": results,
    }


def write_outputs(payload: dict, output_dir: Path, baseline: dict | None) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists, refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)
    comparison = compare_with_baseline(payload, baseline) if baseline is not None else None
    document = dict(payload)
    if comparison is not None:
        document["comparison"] = comparison
    (output_dir / "metrics.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(payload["metrics"], output_dir / "metrics.csv")
    (output_dir / "report.md").write_text(render_report(document), encoding="utf-8")


def _write_csv(rows: list[dict], path: Path) -> None:
    columns = ["group", "state", "frame_id", "global_index", "mesh_source", "frame_role",
               "status", "n_sampled", "n_used", "n_missing", "coverage", "mae_mm",
               "rmse_mm", "p95_mm", "signed_mean_mm", "max_abs_mm"]
    lines = [",".join(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            cells.append("" if value is None else (f"{value:.6g}" if isinstance(value, float) else str(value)))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# baseline comparison
# --------------------------------------------------------------------------- #
def _key(row: dict) -> tuple[str, str, str]:
    return (row["group"], row["state"], row["frame_id"])


def _difference(new: dict, old: dict, field: str) -> float | None:
    a, b = new.get(field), old.get(field)
    if a is None or b is None:
        return None
    return float(a) - float(b)


def compare_with_baseline(payload: dict, baseline: dict) -> dict:
    old_rows = {_key(row): row for row in baseline.get("metrics", [])}
    new_rows = {_key(row): row for row in payload["metrics"]}
    rows = []
    for key in sorted(new_rows):
        new = new_rows[key]
        old = old_rows.get(key)
        entry = {
            "group": new["group"],
            "state": new["state"],
            "frame_id": new["frame_id"],
            "status": new["status"],
            "baseline_status": None if old is None else old["status"],
            "new": {field: new.get(field) for field in
                    ("mae_mm", "rmse_mm", "p95_mm", "coverage", "n_used", "n_missing", "n_sampled")},
            "baseline": None if old is None else {field: old.get(field) for field in
                                                 ("mae_mm", "rmse_mm", "p95_mm", "coverage",
                                                  "n_used", "n_missing", "n_sampled")},
            "delta": None if old is None else {
                field: _difference(new, old, field) for field in
                ("mae_mm", "rmse_mm", "p95_mm", "coverage", "n_used", "n_missing")
            },
            "coverage_dropped": None if old is None else bool(
                (new.get("coverage") or 0.0) < (old.get("coverage") or 0.0) - 1e-12
            ),
            "missing_increased": None if old is None else bool(
                (new.get("n_missing") or 0) > (old.get("n_missing") or 0)
            ),
            "comparable": None if old is None else True,
        }
        if old is not None and old.get("n_sampled") != new.get("n_sampled"):
            entry["comparable"] = False
            entry["warning"] = (
                f"n_sampled differs ({new.get('n_sampled')} vs {old.get('n_sampled')}): "
                "the two runs were sampled from different masks, the delta is not like-for-like"
            )
        rows.append(entry)
    return {
        "baseline_run": baseline.get("run"),
        "baseline_sampling": baseline.get("sampling"),
        "sampling_matches": baseline.get("sampling") == payload.get("sampling"),
        "reference_frames_new": payload["reference_frames"],
        "reference_frames_baseline": baseline.get("reference_frames"),
        "reference_frames_match": payload["reference_frames"] == baseline.get("reference_frames"),
        "rows": rows,
        "missing_groups": sorted(set(old_rows) - set(new_rows)),
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(document: dict) -> str:
    sampling = document["sampling"]
    lines = [
        "# 同帧重建评估报告",
        "",
        f"- 运行: `{document['run']}`",
        f"- 参考几何: `{document['reference_geometry']}`",
        f"- 深度单位/网格单位: `{document['units']['depth']}` / `{document['units']['mesh']}`，"
        f"坐标约定 `{document['units']['coordinate_convention']}`",
        f"- 采样: `{sampling['region']}`，每帧 {sampling['samples_per_frame']} 点，"
        f"seed={sampling['seed']}",
        f"- 射线: {sampling['ray']}",
        f"- 残差: {sampling['residual']}；命中门限 `|Δ| <= {sampling['max_depth_mm']} mm`"
        f"（{sampling['max_depth_meaning']}），正面朝向 `cos > {sampling['min_front_facing_cos']}`",
        f"- 缺失语义: {sampling['missing_policy']}",
        "",
        "## 同帧约束",
        "",
        "每个 state 的网格只与该 state 自己声明的 reference frame 观测比对；"
        "frame_id 直接读取 `reference_geometry.json`，不做 `_0` / `_4` 假设。",
        "",
        "| state | reference frame_id | view | prepared file |",
        "| --- | --- | --- | --- |",
    ]
    for state in STATES_ORDER:
        frame = document["frames_used"][state]
        lines.append(f"| {state} | `{frame['frame_id']}` | {frame['view']} | `{frame['file']}` |")
    lines += [
        "",
        "## 深度指标",
        "",
        "| group | state | frame_id | status | MAE (mm) | RMSE (mm) | P95 (mm) | coverage | n_used | n_missing | n_sampled |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in document["metrics"]:
        lines.append(
            "| {group} | {state} | `{frame}` | {status} | {mae} | {rmse} | {p95} | {cover} | "
            "{used} | {missing} | {sampled} |".format(
                group=row["group"], state=row["state"], frame=row["frame_id"],
                status=row["status"], mae=_fmt(row["mae_mm"]), rmse=_fmt(row["rmse_mm"]),
                p95=_fmt(row["p95_mm"]),
                cover="null" if row["coverage"] is None else f"{row['coverage'] * 100:.1f}%",
                used=row["n_used"], missing=row["n_missing"], sampled=row["n_sampled"],
            )
        )
    lines += ["", "命中分解（用于解释 coverage）:", "",
              "| group | state | n_hit | n_front_facing | n_rejected_by_residual_gate | n_used |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in document["metrics"]:
        lines.append(
            f"| {row['group']} | {row['state']} | {row.get('n_hit')} | "
            f"{row.get('n_front_facing')} | {row.get('n_rejected_by_residual_gate')} | "
            f"{row['n_used']} |"
        )
    lines += ["", "## 几何有效性", "",
              "| state | vertices | faces | finite | indices | extent (m) | metre scale | "
              "flipped faces | flipped ratio | max 位移 vs FLAME base (mm) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for state in STATES_ORDER:
        item = document["geometric_validity"][state]
        extent = " x ".join(f"{value:.4f}" for value in item["extent_m"])
        lines.append(
            f"| {state} | {item['n_vertices']} | {item['n_faces']} | {item['vertices_finite']} | "
            f"{item['face_indices_valid']} | {extent} | {item['metre_scale']} | "
            f"{item['flipped_faces']} | {item['flipped_face_ratio'] * 100:.2f}% | "
            f"{_fmt(item.get('max_displacement_mm_from_flame_base'))} |"
        )
    lines += ["", "法向翻转：比较每个三角面法向与其邻域面积加权平均法向；"
              "不一致即视为绕序反了。翻转比例须 < 5%。"
              "这里不用“相对质心朝外”判据——本交付是开放的头壳（非水密），"
              "鼻尖等区域法向本来就不朝质心，用质心判据会把约 20% 的合法面误判为翻转。", ""]
    comparison = document.get("comparison")
    if comparison:
        lines += ["## 与 baseline 对比", "",
                  f"- baseline: `{comparison['baseline_run']}`",
                  f"- 采样参数一致: {comparison['sampling_matches']}",
                  f"- 参考帧一致: {comparison['reference_frames_match']}",
                  "",
                  "| group | state | 指标 | baseline | 新值 | 差值 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for entry in comparison["rows"]:
            if entry["baseline"] is None:
                lines.append(f"| {entry['group']} | {entry['state']} | (baseline 无此项) | - | - | - |")
                continue
            for field, label in (("mae_mm", "MAE (mm)"), ("rmse_mm", "RMSE (mm)"),
                                 ("p95_mm", "P95 (mm)"), ("coverage", "coverage"),
                                 ("n_missing", "n_missing")):
                lines.append(
                    f"| {entry['group']} | {entry['state']} | {label} | "
                    f"{_fmt(entry['baseline'][field])} | {_fmt(entry['new'][field])} | "
                    f"{_fmt(entry['delta'][field])} |"
                )
            flags = []
            if entry["coverage_dropped"]:
                flags.append("覆盖率下降")
            if entry["missing_increased"]:
                flags.append("缺失数增加")
            if entry.get("comparable") is False:
                flags.append(entry.get("warning", "不可比"))
            if flags:
                lines.append(f"| {entry['group']} | {entry['state']} | 标记 |  | {', '.join(flags)} |  |")
        if comparison["missing_groups"]:
            lines += ["", "baseline 中存在但新 run 缺失的 (group, state, frame): "
                          + ", ".join(str(item) for item in comparison["missing_groups"])]
        lines.append("")
    lines += ["## 交叉校验", "",
              "每个分组额外用逐条射线的方式重积分子集，确认向量化路径与逐条路径一致"
              "（`max_abs_difference_mm` 应为 0）。", ""]
    for row in document["metrics"]:
        check = row.get("cross_check") or {}
        if check:
            lines.append(f"- {row['group']}/{row['state']}: n={check['n']}, "
                         f"max|Δ|={check['max_abs_difference_mm']:.6g} mm")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="eval_sameframe",
        description="Same-frame depth evaluation of a reconstruction run.",
    )
    parser.add_argument("--run", required=True, type=Path,
                        help="run directory containing prepared/ and delivery/")
    parser.add_argument("--output", required=True, type=Path,
                        help="output directory (must not exist)")
    parser.add_argument("--baseline-metrics", type=Path, default=None,
                        help="previous metrics.json to compare against")
    parser.add_argument("--max-depth-mm", type=float, default=DEFAULT_MAX_DEPTH_MM,
                        help="maximum accepted hit distance along the ray, mm")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="depth pixels sampled per frame")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="sampling seed (recorded in metrics.json)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    payload = evaluate(args.run, max_depth_mm=args.max_depth_mm,
                       samples=args.samples, seed=args.seed)
    baseline = None
    if args.baseline_metrics is not None:
        if not args.baseline_metrics.is_file():
            raise FileNotFoundError(f"baseline metrics not found: {args.baseline_metrics}")
        baseline = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
        old_sampling = baseline.get("sampling", {})
        for field in ("samples_per_frame", "seed", "max_depth_mm"):
            if old_sampling.get(field) != payload["sampling"][field]:
                raise ValueError(
                    f"--{field.replace('_', '-')} differs from the baseline "
                    f"({payload['sampling'][field]} vs {old_sampling.get(field)}); "
                    "pass matching flags so the comparison is like-for-like"
                )
    write_outputs(payload, args.output, baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
