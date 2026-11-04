"""Shared serialization rules for observed reference-frame evidence."""
from __future__ import annotations

import numpy as np


INNER_LIP_PAIR = (62, 66)


def observed_landmarks_m(
    xyz_m: np.ndarray, valid: np.ndarray
) -> list[list[float] | None]:
    points = np.asarray(xyz_m, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if points.shape != (68, 3) or mask.shape != (68,):
        raise ValueError("landmark evidence must have shapes (68, 3) and (68,)")
    return [
        points[index].tolist() if mask[index] and np.isfinite(points[index]).all() else None
        for index in range(68)
    ]


def inner_lip_gap_mm(
    points_m: np.ndarray,
    valid: np.ndarray | None = None,
    pair: tuple[int, int] = INNER_LIP_PAIR,
) -> float | None:
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or max(pair) >= len(points):
        raise ValueError("points must be an Nx3 landmark array containing the lip pair")
    left, right = pair
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != (len(points),) or not mask[[left, right]].all():
            return None
    if not np.isfinite(points[[left, right]]).all():
        return None
    return float(np.linalg.norm(points[left] - points[right]) * 1000.0)
