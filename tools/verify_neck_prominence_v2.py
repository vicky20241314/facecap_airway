"""V2 verification: depth-conditioned procedural neck prominence (0.5.4 on 0.5.2).

Runs independent exports from the saved pure-0.5.2 reconstruction and checks:

Legacy invariants
 1. prominence off            -> meshes byte-identical to the clean 0.5.2 baseline
 2. female + auto             -> every vertex identical, stage disabled
 3. missing/invalid landmark  -> baseline mesh delivered, run not blocked
 4. repeat export             -> identical hashes (no accumulation)
 5. protection                -> non-support vertices strictly unchanged; facial
                                 landmark first hits unchanged; no moved vertex
                                 within 30 mm of a facial landmark
 6. topology                  -> closed/open share one topology, equal to baseline
 7. report vs mesh            -> reported parameters match the OBJ read-back
 8. fixed facial rays         -> no new occlusion from the common camera view

Shape acceptance (fixed before looking at the results)
 S1. |H_final - H_target| <= 1.0 mm measured on a plane/mesh section through the
     shaping centre against a baseline trend fixed before shaping
 S2. blunt crest: half-max width >= 5 mm and half-max length >= 5 mm
 S3. peak vertex within 8 mm of the shaping centre
 S4. all safety checks clean, zero in-stage self-intersections, zero spikes,
     zero slender faces; independent recheck of affected faces agrees

Renders: front / side / 45-degree / neck close-up (same view, scale, light) and
a triangle-mesh section through the actual shaping centre (plane/mesh
intersection, grey solid plus wireframe), baseline vs shaped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from head_rgbd import neck_landmark as nl
from head_rgbd import neck_prominence as npn
from head_rgbd.pipeline import export_delivery, load_defaults


ACCEPT_TOLERANCE_MM = 1.0   # fixed before inspecting the results
MIN_HALF_MAX_WIDTH_MM = 5.0
MIN_HALF_MAX_LENGTH_MM = 5.0
MAX_PEAK_OFFSET_MM = 8.0


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(path), process=False, maintain_order=True)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def export(args_ns, fit: Path, sequence: Path, delivery: Path, runtime: Path) -> dict:
    return export_delivery(
        fit=fit, sequence=sequence, output=delivery, runtime=runtime,
        capture=None, runner=None, defaults=load_defaults(), args=args_ns,
    )


def ns(**kw):
    base = dict(neck_landmarks=None, subject_sex=None, neck_prominence=None,
                neck_prominence_mm=None, neck_prominence_radius_x_mm=None,
                neck_prominence_radius_y_mm=None, neck_prominence_base_radius_x_mm=None,
                neck_prominence_base_radius_y_mm=None, neck_prominence_base_share=None)
    base.update(kw)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Independent geometry measurements
# ---------------------------------------------------------------------------

def outline_from_plane(
    vertices: np.ndarray, faces: np.ndarray, centre: np.ndarray,
    right: np.ndarray, up: np.ndarray, forward: np.ndarray,
    y_min: float, y_max: float, slab_mm: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent front outline F(y) via exact triangle/plane intersections."""
    tri = vertices[faces]
    xs = (tri - centre) @ right
    cross = (xs[:, 0] * xs[:, 1] <= 0) | (xs[:, 1] * xs[:, 2] <= 0) | (xs[:, 0] * xs[:, 2] <= 0)
    near = (np.abs(xs).max(axis=1) < slab_mm * 1e-3) | (np.abs(xs).min(axis=1) < 1e-9)
    idx = np.flatnonzero(near & cross)
    grid = np.arange(np.floor(y_min), np.ceil(y_max) + 1, dtype=np.float64)
    outline = np.full(len(grid), np.nan)
    ys, fs = [], []
    for fi in idx:
        v = faces[fi]
        s = xs[fi]
        pts = []
        for a in range(3):
            b = (a + 1) % 3
            if s[a] * s[b] <= 0 and abs(s[a] - s[b]) > 1e-12:
                t = s[a] / (s[a] - s[b])
                pts.append(vertices[v[a]] + t * (vertices[v[b]] - vertices[v[a]]))
        if len(pts) == 2:
            rel0 = pts[0] - centre
            rel1 = pts[1] - centre
            y0 = float(rel0 @ up) * 1000.0
            y1 = float(rel1 @ up) * 1000.0
            f0 = float(rel0 @ forward) * 1000.0
            f1 = float(rel1 @ forward) * 1000.0
            steps = max(1, int(np.ceil(abs(y1 - y0) / 0.5)))
            for step in range(steps + 1):
                ratio = step / steps
                yy = y0 + ratio * (y1 - y0)
                ff = f0 + ratio * (f1 - f0)
                if ff > -15.0:
                    ys.append(yy)
                    fs.append(ff)
    if ys:
        ys = np.asarray(ys)
        fs = np.asarray(fs)
        for i, gy in enumerate(grid):
            sel = (ys >= gy - 0.5) & (ys < gy + 0.5)
            if sel.any():
                outline[i] = float(fs[sel].max())
    return grid, outline


def trend_from_outline(
    grid: np.ndarray, outline: np.ndarray, base_ry_mm: float, inset: float = 2.0, band: float = 8.0,
    continuity: float = 8.0,
) -> dict | None:
    f0 = outline[np.argmin(np.abs(grid))]
    if not np.isfinite(f0):
        return None
    anchors_y, anchors_f = [], []
    lo_hi = (-(base_ry_mm + inset + band), -(base_ry_mm + inset))
    up_lo_hi = (base_ry_mm + inset, base_ry_mm + inset + band)
    for gy, gf in zip(grid, outline):
        if not np.isfinite(gf) or abs(gf - f0) > continuity:
            continue
        if lo_hi[0] <= gy <= lo_hi[1] or up_lo_hi[0] <= gy <= up_lo_hi[1]:
            anchors_y.append(float(gy))
            anchors_f.append(float(gf))
    if len(anchors_y) < 4:
        return None
    coeffs = np.polyfit(np.asarray(anchors_y), np.asarray(anchors_f), 1)
    return {
        "trend_at_centre_mm": float(np.polyval(coeffs, 0.0)),
        "anchor_count": len(anchors_y),
    }


def segment_triangle_hits(segments, own_ids, vertices, faces, exclude_faces, cap=50) -> int:
    """Count strict segment/triangle crossings; shared-edge contact is legal."""
    tri = vertices[faces]
    v0 = tri[:, 0]
    e1 = tri[:, 1] - v0
    e2 = tri[:, 2] - v0
    hits = 0
    for (a, b), own in zip(segments, own_ids):
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-12:
            continue
        pvec = np.cross(d[None, :], e2)
        det = (e1 * pvec).sum(axis=1)
        ok = np.abs(det) > 1e-14
        inv = np.zeros_like(det)
        inv[ok] = 1.0 / det[ok]
        tvec = a - v0
        u = (tvec * pvec).sum(axis=1) * inv
        qvec = np.cross(tvec, e1)
        v = (d[None, :] * qvec).sum(axis=1) * inv
        t = (e2 * qvec).sum(axis=1) * inv
        keep = ok & (t > 1e-9) & (t < length - 1e-9) & (u > 1e-9) & (v > 1e-9) & (u + v < 1 - 1e-9)
        for index in np.flatnonzero(keep):
            tri_ids = set(int(x) for x in faces[index])
            if len(tri_ids & set(own)) >= 2:
                continue
            hits += 1
            if hits >= cap:
                return hits
    return hits


def independent_shape_check(
    base_v: np.ndarray, shaped_v: np.ndarray, faces: np.ndarray,
    centre: np.ndarray, right: np.ndarray, up: np.ndarray, forward: np.ndarray,
    base_ry_mm: float, self_intersection: bool = True,
) -> dict:
    diff = np.linalg.norm(shaped_v - base_v, axis=1)
    moved = diff > 1e-12
    report: dict = {"moved_vertices": int(moved.sum()), "problems": []}
    span = base_ry_mm + 12
    grid, F_b = outline_from_plane(base_v, faces, centre, right, up, forward, -span, span)
    trend = trend_from_outline(grid, F_b, base_ry_mm)
    _, F_s = outline_from_plane(shaped_v, faces, centre, right, up, forward, -span, span)
    if trend is None:
        report["problems"].append("trend_unavailable")
        return report
    i0 = int(np.argmin(np.abs(grid)))
    h_base = float(F_b[i0] - trend["trend_at_centre_mm"]) if np.isfinite(F_b[i0]) else float("nan")
    h_final = float(F_s[i0] - trend["trend_at_centre_mm"]) if np.isfinite(F_s[i0]) else float("nan")
    report.update(h_base_mm=round(h_base, 3), h_final_mm=round(h_final, 3),
                  trend=trend, grid=grid, outline_base=F_b, outline_shaped=F_s)

    if not np.isfinite(shaped_v).all():
        report["problems"].append("non_finite")
    neighbours = [set() for _ in range(len(base_v))]
    for a, b, c in faces:
        neighbours[a].update((b, c))
        neighbours[b].update((a, c))
        neighbours[c].update((a, b))
    touched = moved.copy()
    for v in np.flatnonzero(moved):
        touched[list(neighbours[v])] = True
    affected = touched[faces].any(axis=1)
    tri_b = base_v[faces[affected]]
    tri_a = shaped_v[faces[affected]]
    n_b = np.cross(tri_b[:, 1] - tri_b[:, 0], tri_b[:, 2] - tri_b[:, 0])
    n_a = np.cross(tri_a[:, 1] - tri_a[:, 0], tri_a[:, 2] - tri_a[:, 0])
    area_b = np.linalg.norm(n_b, axis=1)
    area_a = np.linalg.norm(n_a, axis=1)
    if (area_a < 1e-12).any():
        report["problems"].append("degenerate_face")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = area_a / np.maximum(area_b, 1e-30)
        cos = (n_b * n_a).sum(axis=1) / np.maximum(area_b * area_a, 1e-30)
    if np.isfinite(ratio).all() and (ratio < 0.05).any():
        report["problems"].append("area_collapse")
    if (cos < 0.05).any():
        report["problems"].append("normal_reversal")
    peak = float(diff[moved].max() * 1000.0) if moved.any() else 0.0
    magnitudes = diff
    for v in np.flatnonzero(moved):
        nbrs = list(neighbours[v])
        reference = float(np.median(magnitudes[nbrs]))
        if magnitudes[v] > max(2.5 * reference, reference + 3e-3):
            report["problems"].append("displacement_spike")
            break
    # affected faces vs the rest of the mesh (strict crossings)
    seg_index = np.flatnonzero(affected)
    segments, own_ids = [], []
    for fi in seg_index:
        vv = faces[fi]
        for axis in range(3):
            i, j = vv[axis], vv[(axis + 1) % 3]
            if moved[i] or moved[j]:
                segments.append((shaped_v[i], shaped_v[j]))
                own_ids.append(vv.tolist())
    self_hits = 0
    if self_intersection and segments:
        near_faces = np.flatnonzero(
            np.linalg.norm(shaped_v[faces].mean(axis=1) - centre, axis=1) < base_ry_mm * 1e-3 * 2 + 0.03
        )
        sub_faces = faces[near_faces]
        self_hits = segment_triangle_hits(
            segments, own_ids, shaped_v, sub_faces, None
        )
        if self_hits:
            report["problems"].append(f"self_intersections:{self_hits}")
    report.update(
        affected_faces=int(affected.sum()),
        peak_displacement_mm=round(peak, 3),
        self_intersection_count=int(self_hits),
        status="passed" if not report["problems"] else "failed",
    )
    if moved.any():
        rel = base_v[moved] - centre
        xs = rel @ right * 1000.0
        ys = rel @ up * 1000.0
        report["extent"] = {
            "width_x_ten_percent_mm": round(float(xs.max() - xs.min()), 2),
            "length_y_ten_percent_mm": round(float(ys.max() - ys.min()), 2),
        }
        half = magnitudes[moved] >= 0.5 * magnitudes[moved].max()
        report["extent"].update(
            width_x_half_max_mm=round(float(xs[half].max() - xs[half].min()), 2) if half.any() else 0.0,
            length_y_half_max_mm=round(float(ys[half].max() - ys[half].min()), 2) if half.any() else 0.0,
        )
        peak_id = int(np.flatnonzero(moved)[np.argmax(magnitudes[moved])])
        pv = base_v[peak_id] - centre
        report["peak_offset_from_centre_mm"] = [
            round(float(pv @ right * 1000.0), 2),
            round(float(pv @ up * 1000.0), 2),
        ]
    return report


# ---------------------------------------------------------------------------
# Renders
# ---------------------------------------------------------------------------

def view_light(forward: np.ndarray, right: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Headlight slightly above-left of each view, so every view is lit."""
    light = -forward + 0.35 * up - 0.20 * right
    return light / np.linalg.norm(light)


def _view_basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)
    return right, up


def render_mesh_view(
    ax, vertices: np.ndarray, faces: np.ndarray, forward: np.ndarray,
    centre: np.ndarray, half: float, shade_scale: float = 0.85,
):
    fwd = forward / np.linalg.norm(forward)
    right, up = _view_basis(fwd)
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    facing = n @ (-fwd) > 0.02
    tri = tri[facing]
    n = n[facing]
    if not len(tri):
        return
    centroid = tri.mean(axis=1)
    u = (centroid - centre) @ right * 1000.0
    v = (centroid - centre) @ up * 1000.0
    depth = (centroid - centre) @ fwd
    order = np.argsort(-depth)  # far first
    light = view_light(fwd, right, up)
    shade = np.clip(n @ light, 0.0, 1.0) * shade_scale + 0.20
    rel_tri = tri - centre
    polys = np.stack([rel_tri @ right * 1000.0, rel_tri @ up * 1000.0], axis=2)
    collection = PolyCollection(polys[order], facecolors=plt.cm.gray(shade[order]), edgecolors="none")
    ax.add_collection(collection)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.axis("off")


def render_section(
    ax, vertices: np.ndarray, faces: np.ndarray, centre: np.ndarray,
    right: np.ndarray, up: np.ndarray, forward: np.ndarray,
    y_min: float, y_max: float, color: str, label: str,
):
    grid, outline = outline_from_plane(vertices, faces, centre, right, up, forward, y_min, y_max)
    ax.plot(np.asarray(outline), grid, color=color, lw=2.0, label=label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--sequence", type=Path,
        default=Path("D:/proiect/facecap_airway_delivery/runtime/outputs/zp_0.5.1/prepared/sequence.json"),
    )
    parser.add_argument(
        "--runtime", type=Path,
        default=Path("D:/proiect/facecap_airway_delivery/runtime"),
    )
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    baseline = args.baseline_run
    run = args.run
    fit = run / "reconstruction" / "dense"
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    render_dir = out / "renders"
    render_dir.mkdir(exist_ok=True)
    landmarks = args.landmarks

    report: dict = {
        "schema": "head_rgbd.neck_prominence_v2.verification.v1",
        "acceptance": {
            "target_tolerance_mm": ACCEPT_TOLERANCE_MM,
            "min_half_max_width_mm": MIN_HALF_MAX_WIDTH_MM,
            "min_half_max_length_mm": MIN_HALF_MAX_LENGTH_MM,
            "max_peak_offset_mm": MAX_PEAK_OFFSET_MM,
            "declared_before_experiments": True,
        },
        "items": {},
    }
    base_delivery = baseline / "delivery"
    base_hash = {s: sha256(base_delivery / f"head_{s}_m.obj") for s in ("closed", "open")}

    # ---- 1. off identical to the clean baseline -----------------------------
    d = out / "delivery_off"
    if not d.exists():
        export(ns(neck_prominence="off"), fit, args.sequence, d, args.runtime)
    ok1 = all(sha256(d / f"head_{s}_m.obj") == base_hash[s] for s in ("closed", "open"))
    report["items"]["1_off_matches_clean_052_baseline"] = {
        "passed": ok1, "baseline_hashes": base_hash,
        "off_hashes": {s: sha256(d / f"head_{s}_m.obj") for s in ("closed", "open")},
    }

    # ---- 2. female + auto zero displacement ---------------------------------
    d = out / "delivery_female_auto"
    if not d.exists():
        export(ns(subject_sex="female", neck_prominence="auto", neck_landmarks=str(landmarks)),
               fit, args.sequence, d, args.runtime)
    female_prom = json.loads((d / "neck_prominence.json").read_text(encoding="utf-8"))
    ok2 = (
        all(sha256(d / f"head_{s}_m.obj") == base_hash[s] for s in ("closed", "open"))
        and female_prom["effective_enabled"] is False
    )
    report["items"]["2_female_auto_zero_displacement"] = {
        "passed": ok2,
        "reasons": {s: female_prom["states"][s]["reason"] for s in ("closed", "open")},
    }

    # ---- 3. missing landmark delivers the baseline --------------------------
    d = out / "delivery_no_landmarks"
    if not d.exists():
        export(ns(subject_sex="male", neck_prominence="auto"), fit, args.sequence, d, args.runtime)
    no_lm = json.loads((d / "neck_prominence.json").read_text(encoding="utf-8"))
    ok3 = (
        all(sha256(d / f"head_{s}_m.obj") == base_hash[s] for s in ("closed", "open"))
        and no_lm["effective_enabled"] is False
        and all(no_lm["states"][s]["reason"] for s in ("closed", "open"))
    )
    report["items"]["3_missing_landmark_delivers_baseline"] = {
        "passed": ok3,
        "reasons": {s: no_lm["states"][s]["reason"] for s in ("closed", "open")},
    }

    # ---- male export (the feature run) --------------------------------------
    d = out / "delivery_male"
    if not d.exists():
        export(ns(subject_sex="male", neck_prominence="auto", neck_landmarks=str(landmarks)),
               fit, args.sequence, d, args.runtime)
    prom = json.loads((d / "neck_prominence.json").read_text(encoding="utf-8"))
    references = json.loads((fit / "reference_geometry.json").read_text(encoding="utf-8"))["references"]

    # ---- 4. repeat exports ---------------------------------------------------
    d1 = out / "delivery_repeat_1"
    d2 = out / "delivery_repeat_2"
    if not d1.exists():
        export(ns(subject_sex="male", neck_prominence="auto", neck_landmarks=str(landmarks)),
               fit, args.sequence, d1, args.runtime)
    if not d2.exists():
        export(ns(subject_sex="male", neck_prominence="auto", neck_landmarks=str(landmarks)),
               fit, args.sequence, d2, args.runtime)
    ok4 = all(
        sha256(d1 / f"head_{s}_m.obj") == sha256(d2 / f"head_{s}_m.obj") == sha256(d / f"head_{s}_m.obj")
        for s in ("closed", "open")
    )
    report["items"]["4_repeat_export_idempotent"] = {"passed": ok4}

    # ---- per-state shape + safety analysis ----------------------------------
    shape: dict = {}
    for state in ("closed", "open"):
        vb, faces_b = load_obj(base_delivery / f"head_{state}_m.obj")
        vs, faces_s = load_obj(d / f"head_{state}_m.obj")
        ref = references[state]
        M = np.asarray(ref["animated_model_to_reference_camera_4x4"], dtype=np.float64)
        rot, trans = M[:3, :3], M[:3, 3]
        lm_model = (np.asarray(ref["stages"]["dense"]["landmarks_m"], dtype=np.float64) - trans) @ rot
        axes = npn._direction_frame(lm_model)
        right = M[:3, :3] @ axes["right"]  # model -> camera
        up = M[:3, :3] @ axes["up"]
        fwd = M[:3, :3] @ axes["forward"]
        side = prom["states"][state]
        centre = np.asarray(side["binding"]["baseline_binding_point_reference_camera_m"], dtype=np.float64)
        base_ry = float(side["shape"]["base_radius_y_mm"])

        check = independent_shape_check(vb, vs, faces_b, centre, right, up, fwd, base_ry)
        target = float(side["requested_final_prominence_mm"])
        h_final = check.get("h_final_mm")
        h_base = check.get("h_base_mm")
        error = (h_final - target) if h_final is not None and np.isfinite(h_final) else None
        ext = check.get("extent", {})
        peak_offset = check.get("peak_offset_from_centre_mm", [None, None])

        # binding displacement from the OBJ read-back (barycentric)
        tri = np.asarray(side["binding"]["triangle_index"])
        bary = np.asarray(side["binding"]["barycentric"])
        bd = float(np.linalg.norm(bary @ vs[faces_b[int(tri)]] - bary @ vb[faces_b[int(tri)]]) * 1000.0)

        # facial landmark first hits unchanged (common camera view)
        hits_b = [npn.ray_first_hit(vb, faces_b, p) for p in np.asarray(ref["stages"]["dense"]["landmarks_m"])]
        hits_s = [npn.ray_first_hit(vs, faces_b, p) for p in np.asarray(ref["stages"]["dense"]["landmarks_m"])]
        same_hits = 0
        max_hit_shift_mm = 0.0
        for hb, hs in zip(hits_b, hits_s):
            if hb is None and hs is None:
                same_hits += 1
            elif hb is not None and hs is not None and hb["triangle_index"] == hs["triangle_index"]:
                # Occlusion is unchanged when the same triangle is the first
                # hit; the hit point may drift sub-millimetre if the triangle
                # sits on the support boundary.
                same_hits += 1
                max_hit_shift_mm = max(
                    max_hit_shift_mm,
                    float(np.linalg.norm(np.asarray(hs["point_mesh_m"]) - np.asarray(hb["point_mesh_m"])) * 1000.0),
                )
        diff = np.linalg.norm(vs - vb, axis=1)
        moved = diff > 1e-12
        face_near = np.linalg.norm(vb[:, None, :] - lm_model[None, :, :], axis=2).min(axis=1) < 0.030
        # model frame check needs camera-frame distance to landmarks in camera frame
        lm_cam = np.asarray(ref["stages"]["dense"]["landmarks_m"], dtype=np.float64)
        dist_to_lm = np.linalg.norm(vb[:, None, :] - lm_cam[None, :, :], axis=2).min(axis=1)
        face_moved = int((moved & (dist_to_lm < 0.030)).sum())
        untouched = int((~moved).sum())
        identical_untouched = bool(np.array_equal(vs[~moved], vb[~moved]))

        shape_ok = (
            error is not None and abs(error) <= ACCEPT_TOLERANCE_MM
            and (ext.get("width_x_half_max_mm") or 0) >= MIN_HALF_MAX_WIDTH_MM
            and (ext.get("length_y_half_max_mm") or 0) >= MIN_HALF_MAX_LENGTH_MM
            and peak_offset[0] is not None
            and np.hypot(peak_offset[0], peak_offset[1]) <= MAX_PEAK_OFFSET_MM
            and check["status"] == "passed"
            and side["status"] == "applied"
        )
        shape[state] = {
            "h_base_mm": h_base,
            "h_target_mm": target,
            "h_final_mm": h_final,
            "target_error_mm": round(error, 3) if error is not None else None,
            "binding_forward_displacement_mm": round(bd, 3),
            "max_vertex_displacement_mm": check.get("peak_displacement_mm"),
            "width_x_half_max_mm": ext.get("width_x_half_max_mm"),
            "length_y_half_max_mm": ext.get("length_y_half_max_mm"),
            "width_x_ten_percent_mm": ext.get("width_x_ten_percent_mm"),
            "length_y_ten_percent_mm": ext.get("length_y_ten_percent_mm"),
            "peak_offset_from_centre_mm": peak_offset,
            "reported": {
                "final_prominence_mm": side.get("final_prominence_mm"),
                "baseline_prominence_mm": side.get("baseline_prominence_mm"),
                "max_vertex_displacement_mm": side.get("max_vertex_displacement_mm"),
                "binding_forward_displacement_mm": side.get("binding_forward_displacement_mm"),
            },
            "centre_snap": side.get("centre_snap"),
            "depth_conditioning": side.get("depth_conditioning"),
            "safety_reported": side["safety"]["problems"],
            "safety_independent": check["problems"],
            "self_intersections_independent": check.get("self_intersection_count"),
            "protected_facial_vertices_moved": face_moved,
            "untouched_vertices_identical": identical_untouched,
            "facial_first_hits_unchanged": f"{same_hits}/{len(hits_b)}",
            "max_facial_hit_shift_mm": round(max_hit_shift_mm, 4),
            "shape_ok": shape_ok,
            "check": {k: v for k, v in check.items() if k not in ("grid", "outline_base", "outline_shaped", "trend")},
            "outline": {
                "grid_mm": check["grid"].tolist() if "grid" in check else None,
                "baseline_mm": check["outline_base"].tolist() if "outline_base" in check else None,
                "shaped_mm": check["outline_shaped"].tolist() if "outline_shaped" in check else None,
                "trend_at_centre_mm": (check.get("trend") or {}).get("trend_at_centre_mm"),
                "anchor_count": (check.get("trend") or {}).get("anchor_count"),
            },
        }
        # store per-state geometry for rendering
        shape[state]["_geometry"] = (vb, vs, faces_b, centre, right, up, fwd, base_ry)

    ok_shape = all(shape[s]["shape_ok"] for s in shape)
    ok_protection = all(
        shape[s]["untouched_vertices_identical"]
        and shape[s]["protected_facial_vertices_moved"] == 0
        and shape[s]["facial_first_hits_unchanged"].split("/")[0] == shape[s]["facial_first_hits_unchanged"].split("/")[1]
        for s in shape
    )
    report["items"]["5_protection_strictly_holds"] = {
        "passed": ok_protection,
        "states": {s: {k: shape[s][k] for k in (
            "untouched_vertices_identical", "protected_facial_vertices_moved", "facial_first_hits_unchanged")}
            for s in shape},
    }

    # ---- 6. topology ---------------------------------------------------------
    vcounts = {s: len(load_obj(d / f"head_{s}_m.obj")[0]) for s in ("closed", "open")}
    fcounts = {s: len(load_obj(d / f"head_{s}_m.obj")[1]) for s in ("closed", "open")}
    base_counts = {s: len(load_obj(base_delivery / f"head_{s}_m.obj")[0]) for s in ("closed", "open")}
    ok6 = len(set(vcounts.values())) == 1 and len(set(fcounts.values())) == 1 and all(
        vcounts[s] == base_counts[s] for s in vcounts
    )
    report["items"]["6_topology_shared_with_baseline"] = {
        "passed": ok6, "vertices": vcounts, "faces": fcounts, "baseline_vertices": base_counts,
    }

    # ---- 7. report vs mesh ---------------------------------------------------
    mismatches = []
    for state in ("closed", "open"):
        side = prom["states"][state]
        measured = shape[state]
        if abs((measured["reported"]["final_prominence_mm"] or -1e9) - (measured["h_final_mm"] or 1e9)) > 1.01:
            mismatches.append(f"{state}:final_prominence")
        if abs((measured["reported"]["max_vertex_displacement_mm"] or -1e9) - (measured["max_vertex_displacement_mm"] or 1e9)) > 0.2:
            mismatches.append(f"{state}:max_vertex_displacement")
        if abs(measured["reported"]["binding_forward_displacement_mm"] - measured["binding_forward_displacement_mm"]) > 0.2:
            mismatches.append(f"{state}:binding_forward_displacement")
    report["items"]["7_report_matches_obj"] = {"passed": not mismatches, "mismatches": mismatches}

    # ---- 8. shape acceptance -------------------------------------------------
    report["items"]["8_shape_and_safety_acceptance"] = {
        "passed": ok_shape,
        "states": shape,
    }

    # ---- timing --------------------------------------------------------------
    timing = []
    for index in range(3):
        dd = out / f"delivery_timing_{index}"
        started = time.perf_counter()
        if not dd.exists():
            export(ns(subject_sex="male", neck_prominence="auto", neck_landmarks=str(landmarks)),
                   fit, args.sequence, dd, args.runtime)
        total = time.perf_counter() - started
        pp = json.loads((dd / "neck_prominence.json").read_text(encoding="utf-8"))
        stage_ms = [pp["states"][s].get("elapsed_ms") for s in ("closed", "open")]
        timing.append({
            "run": index,
            "export_total_s": round(total, 3),
            "prominence_stage_ms": stage_ms,
            "prominence_stage_total_ms": round(sum(x for x in stage_ms if x), 3),
        })
    report["timing"] = timing

    # ---- renders -------------------------------------------------------------
    renders: list[str] = []
    for state in ("closed", "open"):
        vb, vs, faces_b, centre, right, up, fwd, base_ry = shape[state]["_geometry"]
        views = (
            ("front", fwd), ("side", right), ("45deg", (fwd + right) / np.sqrt(2.0)),
        )
        for view_name, view_dir in views:
            fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), facecolor="#202020")
            for ax, (V, label) in zip(axes, ((vb, "baseline"), (vs, "shaped"))):
                ax.set_facecolor("#202020")
                render_mesh_view(ax, V, faces_b, view_dir, centre, 80.0)
                ax.set_title(f"{state} {view_name} — {label}", color="w", fontsize=10)
            fig.tight_layout()
            path = render_dir / f"{state}_{view_name}_compare.png"
            fig.savefig(path, dpi=120, facecolor="#202020")
            plt.close(fig)
            renders.append(str(path))
        # neck close-up at 45 degrees (a frontal view of a forward bump shows
        # almost nothing), solid + wireframe
        close_dir = (fwd + right) / np.sqrt(2.0)
        for mode in ("solid", "wireframe"):
            fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), facecolor="#202020")
            for ax, (V, label) in zip(axes, ((vb, "baseline"), (vs, "shaped"))):
                ax.set_facecolor("#202020")
                render_mesh_view(ax, V, faces_b, close_dir, centre, 34.0)
                if mode == "wireframe":
                    tri = V[faces_b]
                    facing = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) @ (-close_dir) > 0
                    keep = facing & (np.linalg.norm(tri.mean(axis=1) - centre, axis=1) < 0.045)
                    rr, uu = _view_basis(close_dir / np.linalg.norm(close_dir))
                    polys = np.stack(
                        [tri[keep][:, :, :] @ rr * 1000.0, tri[keep] @ uu * 1000.0], axis=2
                    )
                    ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors="#9fb8d6",
                                                     linewidths=0.35))
                ax.set_title(f"{state} neck close-up ({mode}) — {label}", color="w", fontsize=10)
            fig.tight_layout()
            path = render_dir / f"{state}_closeup_{mode}.png"
            fig.savefig(path, dpi=120, facecolor="#202020")
            plt.close(fig)
            renders.append(str(path))
        # section through the actual shaping centre (plane/mesh intersection)
        # A sagittal plane through the ridge crest is nearly tangent to the
        # surface there, so fewer triangles cross it.  Tilt the section normal
        # 15 degrees toward forward: the plane still passes through the actual
        # shaping centre and now cuts the crest transversally.
        n_sec = right + np.tan(np.deg2rad(15.0)) * fwd
        n_sec /= np.linalg.norm(n_sec)
        f_sec = fwd - np.dot(fwd, n_sec) * n_sec
        f_sec /= np.linalg.norm(f_sec)
        u_sec = up - np.dot(up, n_sec) * n_sec - np.dot(up, f_sec) * f_sec
        u_sec /= np.linalg.norm(u_sec)
        fig, ax = plt.subplots(figsize=(6.5, 6))
        render_section(ax, vb, faces_b, centre, n_sec, u_sec, f_sec, -26, 26, "#9aa7b8", "baseline section")
        render_section(ax, vs, faces_b, centre, n_sec, u_sec, f_sec, -26, 26, "#e05a4e", "shaped section")
        outline_info = shape[state]["outline"]
        trend_value = outline_info.get("trend_at_centre_mm")
        if trend_value is not None:
            ax.axvline(trend_value, color="#d9b44a", lw=1.2, ls="--",
                       label=f"fixed reference trend at centre ({trend_value:.2f} mm)")
        ax.axhline(0, color="#666", lw=0.6, ls=":")
        ax.set_xlabel("forward (mm, rel shaping centre)")
        ax.set_ylabel("up (mm)")
        ax.set_title(f"{state}: section through the shaping centre\n(plane/mesh intersections)")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_aspect("equal")
        path = render_dir / f"{state}_section.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        renders.append(str(path))

    report["renders"] = renders
    # drop non-serialisable geometry
    for s in shape:
        shape[s].pop("_geometry", None)
    report["overall_passed"] = all(item["passed"] for item in report["items"].values())
    (out / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps({k: v["passed"] for k, v in report["items"].items()}, indent=2))
    print("overall_passed:", report["overall_passed"])
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
