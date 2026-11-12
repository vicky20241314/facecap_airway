"""Broad, low-amplitude laryngeal appearance design (0.5.8 delivery round).

A manual appearance design in the style of the reference mesh: the reference's
neck front is a broad, low, blunt, long-transition volume rather than a narrow
peak.  The field is a single compactly-supported smooth bump:

    p(u, v) = H * Su(u/half_u) * Sv(v/half_v)

* ``Sx(x) = (1 - x^2)^3`` on |x| <= 1 — a C2 plateau profile with a broad flat
  top and zero value/slope/curvature at the support edge, in both directions.
  No V-ridge, no sharp apex, no ball.
* sizes are coordinated by design (see CANDIDATES): width, length, height and
  the (implicit) tip bluntness scale together; if the neck space is short the
  WHOLE design shrinks by one factor — never the base alone.
* placement reuses the 0.5.7 measured scale (W_head/W_neck/L_neck, robust neck
  axis, normalised s) and the two-state protection-clearance centre scan; the
  centre is a shared base-topology vertex binding.

This remains a manual appearance design: editing_mode=true,
subject_observation_matched=false; with a reference-inspired (not transferred)
style, geometry_source=reference_inspired_appearance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from head_rgbd import neck_appearance as na
from head_rgbd import neck_prominence as npn
from head_rgbd import neck_subject_v3 as v3

NECK_METHOD_BROAD = "manual_broad_volume_appearance_sculpt"
GEOMETRY_SOURCE_BROAD = "reference_inspired_appearance"

# Design starting points for this template: appearance parameters only.
CANDIDATES: dict[str, dict[str, float]] = {
    "broad_l2": {"height_mm": 2.0, "width10_mm": 34.0, "length10_mm": 30.0},
    "broad_m2": {"height_mm": 2.5, "width10_mm": 36.0, "length10_mm": 32.0},
    "broad_h3": {"height_mm": 3.0, "width10_mm": 40.0, "length10_mm": 34.0},
}


@dataclass(frozen=True)
class BroadDesign:
    height_mm: float
    width10_mm: float      # visible width at 10% height
    length10_mm: float     # visible length at 10% height

    @property
    def half_u_mm(self) -> float:
        # for (1-x^2)^3 the 10% level sits at x ≈ 0.93 -> support ≈ width10/0.93
        return 0.5 * self.width10_mm / 0.93

    @property
    def half_v_mm(self) -> float:
        return 0.5 * self.length10_mm / 0.93

    def as_dict(self) -> dict[str, Any]:
        return {
            "height_mm": self.height_mm,
            "width10_mm": self.width10_mm,
            "length10_mm": self.length10_mm,
            "half_u_mm": round(self.half_u_mm, 2),
            "half_v_mm": round(self.half_v_mm, 2),
        }


def broad_profile(u_mm: np.ndarray, v_mm: np.ndarray, design: BroadDesign) -> np.ndarray:
    """Smooth broad bump: H * (1-(u/a)^2)^3 * (1-(v/b)^2)^3, zero outside."""
    u = np.asarray(u_mm, dtype=np.float64)
    v = np.asarray(v_mm, dtype=np.float64)
    a = max(design.half_u_mm, 1e-3)
    b = max(design.half_v_mm, 1e-3)
    su = np.where(np.abs(u) < a, (1.0 - (u / a) ** 2) ** 3, 0.0)
    sv = np.where(np.abs(v) < b, (1.0 - (v / b) ** 2) ** 3, 0.0)
    return design.height_mm * su * sv


def broad_support_mask(u_mm, v_mm, design: BroadDesign) -> np.ndarray:
    return (np.abs(np.asarray(u_mm)) <= design.half_u_mm) & \
           (np.abs(np.asarray(v_mm)) <= design.half_v_mm)


def coordinated_design(scale, height_mm: float, width10_mm: float,
                       length10_mm: float, *, min_gap_mm: float = 1.0) -> tuple[BroadDesign, dict[str, Any]]:
    """Scale width/length/height together to fit the measured neck space."""
    L = scale.L_neck_mm
    space_up = scale.centre_s * L - min_gap_mm
    space_dn = (1.0 - scale.centre_s) * L - min_gap_mm
    half_v = 0.5 * length10_mm / 0.93
    fit = min(1.0, space_up / half_v if half_v > 0 else 1.0,
              space_dn / half_v if half_v > 0 else 1.0)
    design = BroadDesign(height_mm=height_mm * fit,
                         width10_mm=width10_mm * fit,
                         length10_mm=length10_mm * fit)
    audit = {
        "height_mm": round(design.height_mm, 3),
        "width10_mm": round(design.width10_mm, 2),
        "length10_mm": round(design.length10_mm, 2),
        "half_u_mm": round(design.half_u_mm, 2),
        "half_v_mm": round(design.half_v_mm, 2),
        "space_up_mm": round(space_up, 2),
        "space_down_mm": round(space_dn, 2),
        "overall_fit": round(fit, 3),
        "H_over_Wneck": round(design.height_mm / max(scale.W_neck_mm, 1e-9), 4),
        "W10_over_Wneck": round(design.width10_mm / max(scale.W_neck_mm, 1e-9), 4),
        "L10_over_Lneck": round(design.length10_mm / max(L, 1e-9), 4),
        "note": "coordinated scaling: the whole design shrinks by one factor; "
                "never the base alone with the height kept",
    }
    return design, audit


def apply_broad(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    frame: v3.LocalNeckFrame,
    design: BroadDesign,
    protected: np.ndarray,
) -> dict[str, Any]:
    uv = frame.uv(vertices)
    field = broad_profile(uv[:, 0], uv[:, 1], design)
    taper = na._taper(vertices, protected)
    window = v3._region_window(vertices, frame, protected)
    total = taper * window * field
    shaped = vertices + frame.n[None, :] * (total * 1e-3)[:, None]
    return {
        "shaped_vertices": shaped,
        "applied_mm": total,
        "field_mm": field,
        "support": broad_support_mask(uv[:, 0], uv[:, 1], design).astype(bool),
        "taper": taper,
        "window": window,
    }


def realized_broad_metrics(
    *, shaped: np.ndarray, base: np.ndarray, frame: v3.LocalNeckFrame,
    design: BroadDesign,
) -> dict[str, Any]:
    uv = frame.uv(base)
    support = broad_support_mask(uv[:, 0], uv[:, 1], design)
    disp = (shaped - base) @ frame.n * 1000.0
    sculpt = disp  # no background term in this design
    realized = float(np.max(sculpt[support])) if support.any() else 0.0
    out: dict[str, Any] = {
        "requested_height_mm": design.height_mm,
        "realized_height_mm": round(realized, 3),
        "requested_width10_mm": design.width10_mm,
        "requested_length10_mm": design.length10_mm,
    }
    level = max(realized, 1e-6)
    for name, frac in (("w10", 0.10), ("w50", 0.50), ("l10", 0.10), ("l50", 0.50)):
        mask = support & (sculpt >= frac * level)
        if mask.any():
            if name.startswith("w"):
                out[name + "_mm"] = round(float(uv[mask, 0].max() - uv[mask, 0].min()), 2)
            else:
                out[name + "_mm"] = round(float(uv[mask, 1].max() - uv[mask, 1].min()), 2)
        else:
            out[name + "_mm"] = 0.0
    return out


def build_broad_states(
    *,
    states,
    references: dict[str, Any],
    dense_vertices: np.ndarray,
    dense_faces: np.ndarray,
    prepared_dir,
    landmarks_payload: dict[str, Any] | None,
    height_mm: float,
    width10_mm: float,
    length10_mm: float,
    sequence_rows: list[dict[str, Any]] | None = None,
    label_state: str = "closed",
) -> dict[str, Any]:
    """Measure the scale once, build the coordinated broad design for both states."""
    from head_rgbd import neck_proportional as npd

    base = {}
    chin_camera = head_up = head_forward = head_right = None
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
        base[state] = {
            "vertices": vertices, "protection": protection["mask"],
            "up": rot @ axes["up"], "forward": rot @ axes["forward"],
            "right": rot @ axes["right"],
            "chin": lm_model[npn.CHIN] @ rot.T + trans,
            "reference": ref, "normals": normals_model @ rot.T,
        }
        if state == label_state:
            chin_camera = base[state]["chin"]
            head_up, head_forward, head_right = base[state]["up"], base[state]["forward"], base[state]["right"]

    record = ((landmarks_payload or {}).get("states", {}) or {}).get(label_state) or {}
    label_point = None
    if record.get("status") == v3.nl.STATUS_OBSERVED and record.get("point_observed_m"):
        if record.get("frame_id") == base[label_state]["reference"]["frame_id"]:
            label_point = np.asarray(record["point_observed_m"], dtype=float)
        elif sequence_rows:
            tform = v3.nl.resolve_reference_transform(record, sequence_rows,
                                                      base[label_state]["reference"]["frame_id"])
            if tform.get("status") == v3.nl.STATUS_OBSERVED:
                label_point = np.asarray(v3.nl.apply_transform(record["point_observed_m"], tform), dtype=float)

    scale = npd.measure_neck_scale(
        {s: {"vertices": base[s]["vertices"], "faces": dense_faces,
             "protection": base[s]["protection"]} for s in states},
        chin_camera=chin_camera, head_up=head_up, head_forward=head_forward,
        head_right=head_right, label_point=label_point, label_state=label_state,
    )
    design, audit = coordinated_design(scale, height_mm, width10_mm, length10_mm)

    # shared centre binding on the base topology
    centre_vertex = scale.centre_base_vertex
    for state in states:
        item = base[state]
        centre_cam = item["vertices"][centre_vertex]
        normal = item["normals"][centre_vertex]
        up = item["up"] - np.dot(item["up"], normal) * normal
        up /= np.linalg.norm(up)
        u_axis = np.cross(up, normal); u_axis /= np.linalg.norm(u_axis)
        v_axis = np.cross(normal, u_axis); v_axis /= np.linalg.norm(v_axis)
        item["frame"] = v3.LocalNeckFrame(
            origin=centre_cam.copy(),
            n=np.cross(u_axis, v_axis) / np.linalg.norm(np.cross(u_axis, v_axis)),
            up=up, u_axis=u_axis, v_axis=v_axis,
            audit={"frame_source": "shared_centre_vertex", "vertex": int(centre_vertex),
                   "position_source": "normalised_midline_binding",
                   "label_suggestion_s": scale.label_s, "label_reason": scale.label_reason},
        )

    # shared two-level refinement, per-state midpoints
    centroid = {s: base[s]["vertices"][dense_faces].mean(axis=1) for s in states}
    mask1 = np.zeros(len(dense_faces), dtype=bool)
    for s in states:
        mask1 |= np.linalg.norm(centroid[s] - base[s]["frame"].origin, axis=1) < 0.042
    sub1 = v3.subdivide_local(base[states[0]]["vertices"], dense_faces, face_mask=mask1)
    l1 = sub1["midpoint_edges"]
    refined1 = {s: na._rebuild_refined(base[s]["vertices"], l1, []) for s in states}
    mask2 = np.zeros(len(sub1["faces"]), dtype=bool)
    for s in states:
        c2 = refined1[s][sub1["faces"]].mean(axis=1)
        mask2 |= np.linalg.norm(c2 - base[s]["frame"].origin, axis=1) < 0.026
    sub2 = v3.subdivide_local(refined1[states[0]], sub1["faces"], face_mask=mask2)
    l2 = sub2["midpoint_edges"]
    refined_faces = sub2["faces"]
    base_count = len(base[states[0]]["vertices"])

    results = {}
    for state in states:
        item = base[state]
        verts = na._rebuild_refined(item["vertices"], l1, l2)
        protected = np.zeros(len(verts), dtype=bool)
        protected[:base_count] = item["protection"]
        off1 = base_count
        for k, (i, j) in enumerate(l1):
            protected[off1 + k] = item["protection"][i] or item["protection"][j]
        off2 = off1 + len(l1)
        for k, (i, j) in enumerate(l2):
            protected[off2 + k] = protected[i] or protected[j]
        applied = apply_broad(vertices=verts, faces=refined_faces, frame=item["frame"],
                              design=design, protected=protected)
        shaped = applied["shaped_vertices"]
        moved = np.linalg.norm(shaped - verts, axis=1) > 1e-6
        guards = na.geometry_guards(rest=verts, shaped=shaped, faces=refined_faces, moved=moved)
        adjacency = npn.vertex_adjacency(refined_faces, len(shaped))
        self_int = npn._count_self_intersections(
            shaped, refined_faces, moved, adjacency, item["frame"].origin,
            (design.half_u_mm + 20.0) * 1e-3)
        metrics = realized_broad_metrics(shaped=shaped, base=verts, frame=item["frame"],
                                         design=design)
        results[state] = {
            **item,
            "refined_vertices": verts,
            "protected_refined": protected,
            "shaped_vertices": shaped,
            "applied_mm": applied["applied_mm"],
            "field_mm": applied["field_mm"],
            "support": applied["support"],
            "guards": guards,
            "self_intersections": self_int,
            "metrics": metrics,
            "protection_check": {"protected_vertices_moved": int((np.abs(applied["applied_mm"][protected]) > 1e-9).sum())},
        }
    accepted = all(
        na.accept_guards(results[s]["guards"]) and results[s]["self_intersections"]["count"] == 0
        and results[s]["protection_check"]["protected_vertices_moved"] == 0 for s in states)
    return {
        "states": results,
        "refined_faces": refined_faces,
        "level1_edges": l1,
        "level2_edges": l2,
        "base_vertex_count": base_count,
        "refined_vertex_count": len(results[states[0]]["refined_vertices"]),
        "design": design.as_dict(),
        "design_audit": audit,
        "scale": scale.as_dict(),
        "centre_binding": {"base_vertex": int(centre_vertex), "states": list(states)},
        "accepted": accepted,
    }
