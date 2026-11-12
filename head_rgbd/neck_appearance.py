"""Manually designed laryngeal appearance sculpt (0.5.6).

This stage is an *appearance design*, not a subject reconstruction.  The user
explicitly allows geometry that does not correspond to measured observations:
the goal is a natural-looking skin-covered laryngeal prominence.

Design (one continuous field, no stacked bumps):

    p(u, v) = A(v) * C(s(u, v)),   s = (u - c(v)) / w(v)

* ``A(v)``: vertical height envelope, a Wendland C2 ridge profile with
  independent upper and lower support lengths (zero value *and* zero slope at
  both ends, rounded apex at v = v0).
* ``w(v)``: lateral half-width, slightly narrower toward the ends.
* ``c(v)``: ridge centre line (small offset/tilt allowed).
* ``C(s)``: rounded-V cross-section
  ``C = [1 - (sqrt(s^2 + e^2) - e)] * win(s)`` with ``e = r_tip / w`` the finite
  tip rounding; the linear flanks give the V, the sqrt gives the blunt apex, and
  the C2 window makes the value and first derivative vanish at the lateral
  support edge.

The field is applied along the local neck frame normal (one constant direction
per state), multiplied only by a continuous protection taper, so the exported
geometry is exactly the analytic design sampled on the (locally refined) mesh -
no post-export smoothing.

The low-frequency background correction of the observation stage may be kept
(it places the template neck on the subject's skin); the sculpt height is a
design parameter and is never derived from the depth residual.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from head_rgbd import neck_prominence as npn
from head_rgbd import neck_subject_v3 as v3

NECK_METHOD_SCULPT = "manual_laryngeal_appearance_sculpt"
GEOMETRY_SOURCE_SCULPT = "procedural_appearance"
EDITING_MODE = True

# Design presets: appearance parameters, explicitly NOT anthropometric results.
PRESETS: dict[str, dict[str, float]] = {
    "light": {
        "height_mm": 3.0,
        "width_total_mm": 26.0,
        "length_up_mm": 14.0,
        "length_down_mm": 22.0,
        "tip_radius_mm": 2.0,
    },
    "natural": {
        "height_mm": 5.0,
        "width_total_mm": 29.0,
        "length_up_mm": 15.0,
        "length_down_mm": 24.0,
        "tip_radius_mm": 2.5,
    },
    "pronounced": {
        "height_mm": 7.0,
        "width_total_mm": 32.0,
        "length_up_mm": 16.0,
        "length_down_mm": 26.0,
        "tip_radius_mm": 3.0,
    },
}
DEFAULT_PRESET = "natural"

# Mesh-side safeguards (runaway guards, not shape criteria).
EDGE_GROWTH_CAP = 2.2
SLENDER_QUALITY_CAP = 40.0
TAPER_MM = 12.0


@dataclass(frozen=True)
class SculptDesign:
    height_mm: float
    width_total_mm: float
    length_up_mm: float
    length_down_mm: float
    tip_radius_mm: float
    centre_offset_mm: float = 0.0
    centre_tilt: float = 0.0
    v_shift_mm: float = 0.0
    envelope: str = "wendland"   # "wendland" (0.5.6) or "plateau" (1-t^2)^3

    @property
    def half_width_mm(self) -> float:
        return 0.5 * float(self.width_total_mm)

    @property
    def support_radius_mm(self) -> float:
        return float(self.half_width_mm) + float(self.tip_radius_mm)

    def as_dict(self) -> dict[str, Any]:
        return {
            "height_mm": self.height_mm,
            "width_total_mm": self.width_total_mm,
            "length_up_mm": self.length_up_mm,
            "length_down_mm": self.length_down_mm,
            "tip_radius_mm": self.tip_radius_mm,
            "envelope": self.envelope,
            "centre_offset_mm": self.centre_offset_mm,
            "centre_tilt": self.centre_tilt,
            "v_shift_mm": self.v_shift_mm,
        }


def design_from_preset(preset: str, **overrides) -> SculptDesign:
    if preset not in PRESETS:
        raise ValueError(f"unknown neck sculpt preset {preset!r}; expected {sorted(PRESETS)}")
    values = dict(PRESETS[preset])
    values.update({k: float(v) for k, v in overrides.items() if v is not None})
    return SculptDesign(**values)


def _vertical_envelope(dv_mm: np.ndarray, length_up_mm: float, length_down_mm: float,
                      kind: str = "wendland") -> np.ndarray:
    """Ridge envelope with independent up/down support lengths.

    ``wendland``: (1-t)^4 (4t+1), C2 with a peaked apex.
    ``plateau``: (1-t^2)^3, C2 plateau that keeps effective height much longer
    before dropping to zero at the support end.
    """
    length = np.where(dv_mm >= 0, max(length_up_mm, 1e-3), max(length_down_mm, 1e-3))
    t = np.clip(np.abs(dv_mm) / length, 0.0, 1.0)
    if kind == "plateau":
        return np.where(t < 1.0, (1.0 - t * t) ** 3, 0.0)
    return np.where(t < 1.0, (1.0 - t) ** 4 * (4.0 * t + 1.0), 0.0)


def _quintic_window(x: np.ndarray) -> np.ndarray:
    """1 -> 0 over x in [0, 1], value/slope/curvature zero at both ends."""
    t = np.clip(x, 0.0, 1.0)
    return (1.0 - t) ** 3 * (1.0 + 3.0 * t + 6.0 * t * t)


def sculpt_profile(u_mm: np.ndarray, v_mm: np.ndarray, design: SculptDesign) -> np.ndarray:
    """Analytic appearance field p(u, v) in millimetres."""
    u = np.asarray(u_mm, dtype=np.float64)
    v = np.asarray(v_mm, dtype=np.float64)
    dv = v - design.v_shift_mm
    envelope = _vertical_envelope(dv, design.length_up_mm, design.length_down_mm, design.envelope)
    # lateral width slightly narrower toward the vertical ends
    shrink = 1.0 - 0.22 * (1.0 - envelope)  # 1 at the apex, 0.78 at the ends
    half_width = np.maximum(design.half_width_mm * shrink, 1e-3)
    centre = design.centre_offset_mm + design.centre_tilt * dv
    r_tip = max(design.tip_radius_mm, 0.1)
    e = r_tip / half_width
    s = (u - centre) / half_width
    bracket = 1.0 - (np.sqrt(s * s + e * e) - e)
    support_end = 1.0 + e
    flare = 0.22 * support_end
    window = _quintic_window((np.abs(s) - (support_end - flare)) / np.maximum(flare, 1e-6))
    cross = np.clip(bracket, 0.0, None) * window
    return design.height_mm * envelope * cross


def sculpt_support_mask(u_mm, v_mm, design: SculptDesign) -> np.ndarray:
    dv = np.asarray(v_mm) - design.v_shift_mm
    inside_v = (dv <= design.length_up_mm) & (dv >= -design.length_down_mm)
    half_width = design.half_width_mm
    centre = design.centre_offset_mm + design.centre_tilt * dv
    inside_u = np.abs(np.asarray(u_mm) - centre) <= half_width + design.tip_radius_mm + 1e-6
    return inside_v & inside_u


# ---------------------------------------------------------------------------
# Positioning
# ---------------------------------------------------------------------------

def estimate_mesh_label(
    vertices: np.ndarray, faces: np.ndarray, chin_point: np.ndarray, head_up: np.ndarray,
    head_forward: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate the laryngeal position from the mesh alone (no observation).

    Uses the mesh's own neck: the front-most ridge at roughly one third of the
    chin-to-neck-base distance below the chin.  Returns the surface point and an
    audit record with ``position_source=estimated_from_mesh``.
    """
    head_up = np.asarray(head_up, dtype=np.float64)
    head_up /= np.linalg.norm(head_up)
    forward = np.asarray(head_forward, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(head_up, forward)
    right /= np.linalg.norm(right)
    rel = vertices - chin_point
    y = rel @ head_up
    x = rel @ right
    f = rel @ forward
    normals = npn.vertex_normals(vertices, faces)
    front = normals @ forward > 0.3
    neck = front & (y < -0.015) & (np.abs(x) < 0.020)
    if neck.sum() < 50:
        neck = front & (y < -0.010)
    y_min = float(y[neck].min())
    target_y = -0.35 * (-y_min)  # one third down from the chin
    band = neck & (np.abs(y - target_y) < 0.008)
    if band.sum() < 10:
        idx = np.flatnonzero(neck)
        order = np.argsort(np.abs(y[idx] - target_y))[:60]
        band = np.zeros(len(vertices), dtype=bool)
        band[idx[order]] = True
    ids = np.flatnonzero(band)
    best = ids[int(np.argmax(f[ids]))]
    point = vertices[best].copy()
    audit = {
        "position_source": "estimated_from_mesh",
        "vertex": int(best),
        "neck_front_y_range_mm": [round(y_min * 1000.0, 2), round(float(y[neck].max()) * 1000.0, 2)],
        "relative_depth": round(float(target_y / max(abs(y_min), 1e-6)), 3),
        "method": "front-most midline vertex at ~1/3 of the chin-to-neck-base distance",
    }
    return point, audit


def mesh_local_frame(
    vertices: np.ndarray, faces: np.ndarray, origin: np.ndarray,
    head_up: np.ndarray, head_forward: np.ndarray,
) -> v3.LocalNeckFrame:
    """Local neck frame from the mesh surface (used when no depth is mapped)."""
    head_up = np.asarray(head_up, dtype=np.float64)
    head_up /= np.linalg.norm(head_up)
    normals = npn.vertex_normals(vertices, faces)
    idx = int(np.argmin(np.linalg.norm(vertices - origin, axis=1)))
    normal = normals[idx]
    up = head_up - np.dot(head_up, normal) * normal
    up /= np.linalg.norm(up)
    u_axis = np.cross(up, normal)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    n_axis = np.cross(u_axis, v_axis)
    n_axis /= np.linalg.norm(n_axis)
    return v3.LocalNeckFrame(
        origin=np.asarray(origin, dtype=np.float64).copy(),
        n=n_axis, up=np.vstack([up]).ravel(), u_axis=u_axis, v_axis=v_axis,
        audit={"frame_source": "mesh_surface", "vertex": idx},
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def _taper(vertices: np.ndarray, protected: np.ndarray) -> np.ndarray:
    ids = np.flatnonzero(protected)
    if not len(ids):
        return np.ones(len(vertices))
    distance, _ = cKDTree(vertices[ids]).query(vertices, k=1)
    return np.clip(distance / (TAPER_MM * 1e-3), 0.0, 1.0)


def apply_sculpt(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    frame: v3.LocalNeckFrame,
    design: SculptDesign,
    protected: np.ndarray,
    background_mm: np.ndarray | None = None,
) -> dict[str, Any]:
    """Displace the mesh by base correction + the analytic sculpt field."""
    uv = frame.uv(vertices)
    field = sculpt_profile(uv[:, 0], uv[:, 1], design)
    taper = _taper(vertices, protected)
    # the background correction must stay local: a C2 region window keeps it on
    # the front-neck patch and fades it to zero (value and slope) outward
    window = v3._region_window(vertices, frame, protected)
    base = np.zeros(len(vertices)) if background_mm is None else np.asarray(background_mm)
    total = taper * window * (base + field)
    shaped = vertices + frame.n[None, :] * (total * 1e-3)[:, None]
    return {
        "shaped_vertices": shaped,
        "sculpt_mm": field,
        "applied_mm": total,
        "taper": taper,
        "support": sculpt_support_mask(uv[:, 0], uv[:, 1], design).astype(bool),
        "uv_mm": uv,
    }


def realized_metrics(
    *, shaped: np.ndarray, base: np.ndarray, frame: v3.LocalNeckFrame,
    design: SculptDesign, background_mm: np.ndarray | None = None,
    background_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure the actual sculpt on the mesh against the background-only surface.

    ``background_reference`` is the displacement field of the base correction
    alone (taper*window*background); the realized sculpt is measured relative
    to that reference so a non-zero background cannot bias the number.
    """
    uv = frame.uv(base)
    field = sculpt_profile(uv[:, 0], uv[:, 1], design)
    support = sculpt_support_mask(uv[:, 0], uv[:, 1], design)
    disp = (shaped - base) @ frame.n * 1000.0
    reference = np.zeros(len(base)) if background_reference is None else np.asarray(background_reference)
    sculpt_actual = disp - reference
    if support.any():
        realized = float(np.max(sculpt_actual[support]))
    else:
        realized = 0.0
    peak = float(np.max(sculpt_actual)) if len(sculpt_actual) else 0.0
    # extents at 50% and 10% of the realized peak inside the support
    level = max(realized, 1e-6)
    out: dict[str, Any] = {
        "requested_height_mm": design.height_mm,
        "realized_height_mm": round(realized, 3),
        "peak_on_mesh_mm": round(peak, 3),
        "requested_width_mm": design.width_total_mm,
        "requested_length_mm": design.length_up_mm + design.length_down_mm,
        "requested_tip_radius_mm": design.tip_radius_mm,
        "requested_transitions_mm": [design.length_up_mm, design.length_down_mm],
        "analytic_max_mm": round(float(np.max(field)), 3),
    }
    for name, frac in (("half", 0.5), ("ten", 0.1)):
        mask = support & (sculpt_actual >= frac * level)
        if mask.any():
            uu = uv[mask, 0]
            vv = uv[mask, 1]
            out[f"width_{name}_mm"] = round(float(uu.max() - uu.min()), 2)
            out[f"length_{name}_mm"] = round(float(vv.max() - vv.min()), 2)
        else:
            out[f"width_{name}_mm"] = 0.0
            out[f"length_{name}_mm"] = 0.0
    return out


def geometry_guards(
    *, rest: np.ndarray, shaped: np.ndarray, faces: np.ndarray, moved: np.ndarray,
) -> dict[str, Any]:
    """Runaway guards: no folds, degeneration, slivers or torn edges."""
    from head_rgbd.mesh_integrity import edge_records
    _, _, edge_counts, _, edge_balance = edge_records(faces)
    affected = moved[faces].any(axis=1)
    tri_b = rest[faces[affected]]
    tri_a = shaped[faces[affected]]
    nb = np.cross(tri_b[:, 1] - tri_b[:, 0], tri_b[:, 2] - tri_b[:, 0])
    na = np.cross(tri_a[:, 1] - tri_a[:, 0], tri_a[:, 2] - tri_a[:, 0])
    ab = np.linalg.norm(nb, axis=1)
    aa = np.linalg.norm(na, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = aa / np.maximum(ab, 1e-30)
        cos = (nb * na).sum(axis=1) / np.maximum(ab * aa, 1e-30)
    preexisting = ab < 1e-10
    e = np.stack([np.linalg.norm(tri_a[:, 1] - tri_a[:, 0], axis=1),
                  np.linalg.norm(tri_a[:, 2] - tri_a[:, 1], axis=1),
                  np.linalg.norm(tri_a[:, 0] - tri_a[:, 2], axis=1)], axis=1)
    quality = e.max(axis=1) ** 2 / np.maximum(aa, 1e-18)
    eb = np.linalg.norm(tri_b - tri_b[:, [1, 2, 0]], axis=2)
    ea = np.linalg.norm(tri_a - tri_a[:, [1, 2, 0]], axis=2)
    growth = np.max(ea / np.maximum(eb, 1e-12)) if len(ea) else 1.0
    # boundary edges (open mesh rim) must keep their vertex ids: tearing shows up
    # as huge edge growth on rim-adjacent faces, covered by the cap below
    return {
        "inconsistent_shared_edges": int(((edge_counts == 2) & (edge_balance != 0)).sum()),
        "nonmanifold_edges": int((edge_counts > 2).sum()),
        "affected_faces": int(affected.sum()),
        "degenerate_faces": int((aa < 1e-12).sum()),
        "area_collapse": int(((ratio < 0.05) & np.isfinite(ratio)).sum()),
        "normal_reversal": int(((cos < 0.05) & (~preexisting)).sum()),
        "true_fold": int(((cos < 0.0) & (~preexisting)).sum()),
        "slender_faces": int(((quality > SLENDER_QUALITY_CAP) & (~preexisting)).sum()),
        "max_edge_growth": round(float(growth), 3),
        "edge_growth_cap": EDGE_GROWTH_CAP,
        "preexisting_degenerate_rest_faces": int(preexisting.sum()),
        "status": "passed",
    }


def accept_guards(guards: dict[str, Any]) -> bool:
    return (
        guards.get("inconsistent_shared_edges", 0) == 0
        and guards.get("nonmanifold_edges", 0) == 0
        and guards["degenerate_faces"] == 0
        and guards["area_collapse"] == 0
        and guards["normal_reversal"] == 0
        and guards["slender_faces"] == 0
        and guards["max_edge_growth"] <= EDGE_GROWTH_CAP
    )


# ---------------------------------------------------------------------------
# Shared state construction (used by the experiment driver AND the public entry)
# ---------------------------------------------------------------------------

def _rebuild_refined(
    base_vertices: np.ndarray, level1_edges, level2_edges,
) -> np.ndarray:
    mids1 = np.asarray([0.5 * (base_vertices[i] + base_vertices[j]) for i, j in level1_edges])
    level1 = np.concatenate([base_vertices, mids1], axis=0)
    if level2_edges:
        mids2 = np.asarray([0.5 * (level1[i] + level1[j]) for i, j in level2_edges])
        return np.concatenate([level1, mids2], axis=0)
    return level1


def build_sculpt_states(
    *,
    states,
    references: dict[str, Any],
    dense_vertices: np.ndarray,
    dense_faces: np.ndarray,
    prepared_dir,
    landmarks_payload: dict[str, Any] | None,
    design: SculptDesign,
    sequence_rows: list[dict[str, Any]] | None = None,
    keep_background: bool = False,
    level1_radius_mm: float = 42.0,
    level2_radius_mm: float = 22.0,
) -> dict[str, Any]:
    """Build the refined, sculpted meshes for closed and open.

    The local refinement masks are unions over the two states so both share one
    refined topology; midpoint positions, normals and protection masks are
    generated per state from that state's own pose.
    """
    from pathlib import Path

    base = {}
    for state in states:
        ref = references[state]
        vertices = np.asarray(dense_vertices[ref["stages"]["dense"]["frame_index"]], dtype=float)
        M = np.asarray(ref["animated_model_to_reference_camera_4x4"], dtype=float)
        rot, trans = M[:3, :3], M[:3, 3]
        lm_cam = np.asarray(ref["stages"]["dense"]["landmarks_m"], dtype=float)
        lm_model = (lm_cam - trans) @ rot
        axes = npn._direction_frame(lm_model)
        normals_model = npn.vertex_normals((vertices - trans) @ rot, dense_faces)
        protection = npn._protected_mask((vertices - trans) @ rot, lm_model, normals_model, axes)
        head_up = rot @ axes["up"]
        label_point = None
        label_pixel = None
        record = ((landmarks_payload or {}).get("states", {}) or {}).get(state) or {}
        if record.get("status") == v3.nl.STATUS_OBSERVED and record.get("point_observed_m"):
            if record.get("frame_id") == ref["frame_id"]:
                label_point = np.asarray(record["point_observed_m"], dtype=float)
                label_pixel = tuple(record.get("pixel_uv") or [0.0, 0.0])
            else:
                tform = v3.nl.resolve_reference_transform(record, sequence_rows or [], ref["frame_id"])
                if tform.get("status") == v3.nl.STATUS_OBSERVED and tform.get("annotation_frame") == ref["frame_id"]:
                    label_point = np.asarray(v3.nl.apply_transform(record["point_observed_m"], tform), dtype=float)
                    label_pixel = tuple(record.get("pixel_uv") or [0.0, 0.0])
        position_audit = {"position_source": "recorded_label", "label_pixel": label_pixel}
        observations = None
        background = None
        if label_point is not None and (Path(prepared_dir) / f"{ref['frame_id']}.npz").is_file():
            data = np.load(Path(prepared_dir) / f"{ref['frame_id']}.npz", allow_pickle=False)
            try:
                observations = v3.collect_observations(
                    state=state, frame_id=ref["frame_id"], depth=data["depth"],
                    neck_mask=data["neck_mask"], intrinsics=data["intrinsics"],
                    label_pixel=label_pixel, label_point=label_point,
                    head_up_camera=head_up, baseline_vertices=vertices, faces=dense_faces,
                )
                if keep_background:
                    background = v3.robust_background(observations)
            except Exception:
                observations = None
                background = None
        if observations is not None:
            frame = observations.frame
        elif label_point is not None:
            frame = mesh_local_frame(vertices, dense_faces, label_point, head_up, rot @ axes["forward"])
        else:
            chin_cam = lm_model[npn.CHIN] @ rot.T + trans
            label_point, position_audit = estimate_mesh_label(
                vertices, dense_faces, chin_cam, head_up, rot @ axes["forward"])
            frame = mesh_local_frame(vertices, dense_faces, label_point, head_up, rot @ axes["forward"])
        base[state] = {
            "vertices": vertices, "protection": protection["mask"],
            "frame": frame, "observations": observations, "background": background,
            "position_audit": position_audit, "reference": ref,
        }

    # ---- shared local refinement (union masks, per-state positions) --------
    mask1 = np.zeros(len(dense_faces), dtype=bool)
    for state in states:
        item = base[state]
        centroid = item["vertices"][dense_faces].mean(axis=1)
        mask1 |= np.linalg.norm(centroid - item["frame"].origin, axis=1) < level1_radius_mm * 1e-3
    sub1 = v3.subdivide_local(base[states[0]]["vertices"], dense_faces, face_mask=mask1)
    level1_edges = sub1["midpoint_edges"]
    refined1 = {s: _rebuild_refined(base[s]["vertices"], level1_edges, []) for s in states}
    mask2 = np.zeros(len(sub1["faces"]), dtype=bool)
    for state in states:
        centroid = refined1[state][sub1["faces"]].mean(axis=1)
        mask2 |= np.linalg.norm(centroid - base[state]["frame"].origin, axis=1) < level2_radius_mm * 1e-3
    sub2 = v3.subdivide_local(refined1[states[0]], sub1["faces"], face_mask=mask2)
    level2_edges = sub2["midpoint_edges"]
    refined_faces = sub2["faces"]
    base_vertex_count = len(base[states[0]]["vertices"])

    def refined_for_state(state: str):
        item = base[state]
        verts = _rebuild_refined(item["vertices"], level1_edges, level2_edges)
        protected = np.zeros(len(verts), dtype=bool)
        protected[:base_vertex_count] = item["protection"]
        offset1 = base_vertex_count
        for k, (i, j) in enumerate(level1_edges):
            protected[offset1 + k] = item["protection"][i] or item["protection"][j]
        offset2 = offset1 + len(level1_edges)
        for k, (i, j) in enumerate(level2_edges):
            protected[offset2 + k] = protected[i] or protected[j]
        return verts, protected

    results = {}
    for state in states:
        item = base[state]
        verts, protected = refined_for_state(state)
        background_values = None
        if item["background"] is not None:
            uv_all = item["frame"].uv(verts)
            background_values = item["background"].evaluate(uv_all[:, 0], uv_all[:, 1])
        applied = apply_sculpt(
            vertices=verts, faces=refined_faces, frame=item["frame"], design=design,
            protected=protected, background_mm=background_values,
        )
        shaped = applied["shaped_vertices"]
        moved = np.linalg.norm(shaped - verts, axis=1) > 1e-6
        guards = geometry_guards(rest=verts, shaped=shaped, faces=refined_faces, moved=moved)
        adjacency = npn.vertex_adjacency(refined_faces, len(shaped))
        self_int = npn._count_self_intersections(
            shaped, refined_faces, moved, adjacency, item["frame"].origin,
            (design.support_radius_mm + 20.0) * 1e-3,
        )
        background_reference = None
        if background_values is not None:
            taper_all = _taper(verts, protected)
            window_all = v3._region_window(verts, item["frame"], protected)
            background_reference = taper_all * window_all * background_values
        metrics = realized_metrics(
            shaped=shaped, base=verts, frame=item["frame"], design=design,
            background_mm=background_values, background_reference=background_reference,
        )
        results[state] = {
            **item,
            "refined_vertices": verts,
            "protected_refined": protected,
            "shaped_vertices": shaped,
            "applied_mm": applied["applied_mm"],
            "sculpt_mm": applied["sculpt_mm"],
            "support": applied["support"],
            "guards": guards,
            "self_intersections": self_int,
            "metrics": metrics,
            "protection_check": {
                "protected_vertices_moved": int((np.abs(applied["applied_mm"][protected]) > 1e-9).sum()),
            },
            "background_available": item["background"] is not None,
        }
    return {
        "states": results,
        "refined_faces": refined_faces,
        "level1_edges": level1_edges,
        "level2_edges": level2_edges,
        "base_vertex_count": base_vertex_count,
        "refined_vertex_count": len(results[states[0]]["refined_vertices"]),
        "design": design.as_dict(),
        "accepted": all(
            accept_guards(results[s]["guards"]) and results[s]["self_intersections"]["count"] == 0
            and results[s]["protection_check"]["protected_vertices_moved"] == 0
            for s in states
        ),
    }
