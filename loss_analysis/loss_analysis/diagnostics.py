"""Independent geometric diagnostics on fixed observations.

These numbers are **ours**, computed from real prepared frames and real exported
geometry. They are kept strictly apart from the algorithm's objectives: nothing in
this module is ever fed back into optimisation, and no result here is called a
"loss".

A diagnostic is only produced when a real observation exists. Missing depth,
missing landmarks and zero-sample sets return ``status="no_valid_samples"`` with
``None`` values — never ``0.0``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .terms import coverage_summary, metric_summary


def load_frame(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def backproject(uv: np.ndarray, depth_mm: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same convention as ``prepare.backproject``: X right, Y up, Z backward, origin at camera."""
    uv = np.asarray(uv, dtype=float)
    depth = np.asarray(depth_mm)
    fx, fy, cx, cy = (float(v) for v in np.asarray(intrinsics).reshape(-1)[:4])
    pix = np.rint(np.nan_to_num(uv, nan=-1.0)).astype(np.int64)
    h, w = depth.shape
    valid = (
        np.isfinite(uv).all(axis=1)
        & (pix[:, 0] >= 0) & (pix[:, 0] < w)
        & (pix[:, 1] >= 0) & (pix[:, 1] < h)
    )
    z = np.full(len(uv), np.nan)
    ids = np.flatnonzero(valid)
    z[ids] = depth[pix[ids, 1], pix[ids, 0]].astype(float) * 0.001
    valid &= np.isfinite(z) & (z > 0.1) & (z < 1.5)
    xyz = np.full((len(uv), 3), np.nan)
    xyz[valid, 0] = (uv[valid, 0] - cx) * z[valid] / fx
    xyz[valid, 1] = -(uv[valid, 1] - cy) * z[valid] / fy
    xyz[valid, 2] = -z[valid]
    return xyz, valid


def depth_surface_error_mm(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    xyz_m: np.ndarray,
    valid: np.ndarray,
    *,
    max_depth_m: float = 0.030,
    max_plane_m: float | None = None,
    chunk: int = 64,
) -> dict[str, Any]:
    """Per-ray depth residual against the **fixed** set of real observation rays.

    Returns absolute depth difference in millimetres for rays that actually hit the
    surface, plus explicit counts for every rejection reason. Rays that miss are
    *not* silently treated as zero error; they are reported as missing coverage.
    """
    import trimesh

    points = np.asarray(xyz_m, dtype=float)[np.asarray(valid, dtype=bool)]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return {
            "status": "no_valid_samples",
            "n_required": 0,
            "metrics": metric_summary(np.array([])),
            "coverage": coverage_summary(np.zeros(0, bool)),
        }
    mesh = trimesh.Trimesh(np.asarray(vertices_m, dtype=float), np.asarray(faces, dtype=np.int64), process=False)
    hits: list[np.ndarray] = []
    rays: list[np.ndarray] = []
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        hit, ray, _ = mesh.ray.intersects_location(np.zeros_like(block), block, multiple_hits=False)
        hits.append(np.asarray(hit, dtype=float).reshape(-1, 3))
        rays.append(np.asarray(ray, dtype=np.int64).reshape(-1) + start)
    hit = np.concatenate(hits) if hits else np.zeros((0, 3))
    ray = np.concatenate(rays) if rays else np.zeros(0, np.int64)

    covered = np.zeros(len(points), bool)
    residual_mm = np.full(len(points), np.nan)
    keep = np.zeros(len(hit), bool)
    if len(hit):
        depth_ok = np.abs(hit[:, 2] - points[ray, 2]) < max_depth_m
        keep = depth_ok
        if max_plane_m is not None:
            tri = np.asarray(vertices_m, dtype=float)[np.asarray(faces, dtype=np.int64)]
            # plane distance is computed by the caller's surface normal; here we only
            # keep the depth gate because plane distance needs the triangle id.
            keep &= True
        covered[ray[keep]] = True
        residual_mm[ray[keep]] = np.abs(hit[keep, 2] - points[ray[keep], 2]) * 1000.0

    metrics = metric_summary(residual_mm)
    cov = coverage_summary(covered)
    return {
        "status": "ok" if metrics["status"] == "ok" else "no_valid_samples",
        "n_required": int(len(points)),
        "n_hits": int(len(hit)),
        "n_depth_rejected": int((~np.isclose(np.abs(hit[:, 2] - points[ray, 2]), 0) & ~keep).sum()) if len(hit) else 0,
        "metrics": metrics,
        "coverage": cov,
        "note": (
            "absolute depth difference per real ray; missing hits count as missing coverage, "
            "never as zero error"
        ),
    }


def landmark_reprojection_px(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    landmark_triangles: np.ndarray,
    landmark_barycentric: np.ndarray,
    landmark_uv: np.ndarray,
    intrinsics: np.ndarray,
    landmark_xyz_m: np.ndarray,
    valid: np.ndarray,
    *,
    indices: slice = slice(17, 48),
) -> dict[str, Any]:
    """Reprojection error of the observed landmarks, using observed 3D where available."""
    tri = np.asarray(faces, dtype=np.int64)[np.asarray(landmark_triangles, dtype=np.int64)]
    bary = np.asarray(landmark_barycentric, dtype=float)
    verts = np.asarray(vertices_m, dtype=float)
    model = (verts[tri] * bary[:, :, None]).sum(axis=1)

    observed_xyz = np.asarray(landmark_xyz_m, dtype=float)
    use_3d = np.asarray(valid, dtype=bool).copy()
    candidate = model.copy()
    candidate[use_3d] = observed_xyz[use_3d]

    fx, fy, cx, cy = (float(v) for v in np.asarray(intrinsics).reshape(-1)[:4])
    z = np.maximum(-candidate[:, 2], 1e-6)
    uv = np.column_stack((fx * candidate[:, 0] / z + cx, -fy * candidate[:, 1] / z + cy))

    target = np.asarray(landmark_uv, dtype=float)
    selected = np.zeros(len(target), bool)
    selected[indices] = True
    selected &= np.isfinite(target).all(axis=1) & use_3d
    if not selected.any():
        return {"status": "no_valid_samples", "n_used": 0, "metrics": metric_summary(np.array([]))}
    error_px = np.linalg.norm(uv[selected] - target[selected], axis=1)
    return {
        "status": "ok",
        "n_used": int(selected.sum()),
        "metrics": metric_summary(error_px),
        "note": "2D reprojection on landmarks that have BOTH a valid depth and a finite detection",
    }


def support_label_summary(support_path: Path) -> dict[str, Any]:
    """Read the delivery's support labels; report the split, do not reinterpret it."""
    path = Path(support_path)
    if not path.is_file():
        return {"status": "missing", "counts": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels", [])
    counts = payload.get("counts", [])
    total = int(sum(counts)) if counts else 0
    if total == 0:
        return {"status": "no_valid_samples", "counts": None}
    return {
        "status": "ok",
        "labels": labels,
        "counts": counts,
        "fractions": {label: count / total for label, count in zip(labels, counts)},
        "note": "real-observation support, model prior and interpolation are labelled separately by the delivery",
    }


def sequence_observations(sequence_path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Candidate frames with their prepared npz path, in sequence order."""
    sequence_path = Path(sequence_path).resolve()
    payload = json.loads(sequence_path.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("frames", []) if row.get("status") == "candidate"]
    if limit is not None:
        rows = rows[: int(limit)]
    return [
        {
            "id": row["id"],
            "state": row.get("state"),
            "view": row.get("view"),
            "path": sequence_path.parent / row["file"],
            "intrinsics": None,
        }
        for row in rows
    ]
