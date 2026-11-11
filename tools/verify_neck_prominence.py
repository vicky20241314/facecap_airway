"""Eight-item verification of the marker-guided neck prominence (0.5.3 on 0.5.2).

Runs against a completed pure-0.5.2 baseline run and the same reconstruction
exported with different prominence configurations.  Produces
verification/verification_report.json and the comparison renders.

Items:
 1. prominence off            -> meshes identical to the clean 0.5.2 baseline
 2. female + auto             -> all vertices identical to the baseline
 3. male + valid landmark     -> continuous local bump in both states
 4. missing/invalid landmark  -> baseline mesh delivered, run not blocked
 5. protection                -> non-support vertices strictly unchanged; first
                                 camera-ray hits on facial landmarks unchanged
 6. repeat export             -> no displacement accumulation (identical hashes)
 7. units                     -> mm/m conversions in the report and the field
 8. geometry safety           -> independent recheck of every affected face
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

from head_rgbd import neck_landmark as nl
from head_rgbd import neck_prominence as npn
from head_rgbd.pipeline import export_delivery, load_defaults


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def obj_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(path), process=False, maintain_order=True)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def export(args_ns, fit: Path, sequence: Path, delivery: Path, runtime: Path) -> dict:
    return export_delivery(
        fit=fit,
        sequence=sequence,
        output=delivery,
        runtime=runtime,
        capture=None,
        runner=None,
        defaults=load_defaults(),
        args=args_ns,
    )


def face_first_hits(vertices: np.ndarray, faces: np.ndarray, targets: np.ndarray) -> list:
    """First visible hit from the camera origin towards each target point."""
    return [npn.ray_first_hit(vertices, faces, target) for target in targets]


def segment_triangle_hits(
    segments: np.ndarray, seg_faces: np.ndarray, vertices: np.ndarray, faces: np.ndarray
) -> int:
    """Count segment/triangle intersections, excluding faces sharing a segment endpoint."""
    hits = 0
    for (a, b), own in zip(segments, seg_faces):
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-12:
            continue
        tri = vertices[faces]
        v0 = tri[:, 0]
        e1 = tri[:, 1] - v0
        e2 = tri[:, 2] - v0
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
        keep = (
            ok & (t > 1e-9) & (t < length - 1e-9)
            & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        )
        for index in np.flatnonzero(keep):
            if set(faces[index].tolist()) & set(own):
                continue
            hits += 1
    return hits


def independent_geometry_check(
    base_v: np.ndarray, shaped_v: np.ndarray, faces: np.ndarray
) -> dict:
    moved = np.linalg.norm(shaped_v - base_v, axis=1) > 1e-12
    report: dict = {"moved_vertices": int(moved.sum())}
    problems: list[str] = []
    if not np.isfinite(shaped_v).all():
        problems.append("non_finite")
    # faces touching moved vertices or their neighbours
    touched = moved.copy()
    neighbours = [set() for _ in range(len(base_v))]
    for a, b, c in faces:
        neighbours[a].update((b, c))
        neighbours[b].update((a, c))
        neighbours[c].update((a, b))
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
        problems.append("degenerate_face")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = area_a / area_b
        cos = (n_b * n_a).sum(axis=1) / np.maximum(area_b * area_a, 1e-30)
    if np.isfinite(ratio).all() and (ratio < 0.05).any():
        problems.append("area_collapse")
    if (cos < 0.05).any():
        problems.append("normal_reversal")
    edges_b, edges_a = [], []
    for axis in range(3):
        edges_b.append(np.linalg.norm(tri_b[:, (axis + 1) % 3] - tri_b[:, axis], axis=1))
        edges_a.append(np.linalg.norm(tri_a[:, (axis + 1) % 3] - tri_a[:, axis], axis=1))
    edge_b = np.stack(edges_b)
    edge_a = np.stack(edges_a)
    if (edge_a > np.maximum(edge_b * 1.5, edge_b + 2e-3)).any():
        problems.append("abnormal_long_edge")
    # slope continuity across the support boundary
    moved_ids = np.flatnonzero(moved)
    magnitudes = np.linalg.norm(shaped_v - base_v, axis=1)
    for v in moved_ids:
        for n in neighbours[v]:
            span = np.linalg.norm(base_v[n] - base_v[v])
            if span > 0 and abs(magnitudes[n] - magnitudes[v]) > 1.5 * span + 1e-4:
                problems.append("slope_discontinuity")
                break
    # self-intersection: every affected edge of the shaped mesh vs nearby faces
    seg_index = np.flatnonzero(affected)
    segments, seg_faces = [], []
    for f in seg_index:
        for axis in range(3):
            i, j = faces[f, axis], faces[f, (axis + 1) % 3]
            if moved[i] or moved[j]:
                segments.append((shaped_v[i], shaped_v[j]))
                seg_faces.append(faces[f].tolist())
    intersections = segment_triangle_hits(
        np.asarray(segments), seg_faces, shaped_v, faces
    ) if segments else 0
    if intersections:
        problems.append(f"self_intersections:{intersections}")
    report.update(
        problems=problems,
        affected_faces=int(affected.sum()),
        self_intersection_count=int(intersections),
        peak_displacement_mm=float(magnitudes.max() * 1000.0) if moved.any() else 0.0,
        unprotected_vertices_moved=bool((~np.isin(np.arange(len(base_v)), moved_ids)).sum() == 0),
    )
    report["status"] = "passed" if not problems else "failed"
    return report


def render_comparisons(
    base_dir: Path, new_dir: Path, out_dir: Path, prom: dict
) -> list[str]:
    fit = None  # reference_geometry is inside the run's reconstruction
    files = []
    for state in ("closed", "open"):
        vb, F = obj_vertices_faces(base_dir / f"head_{state}_m.obj")
        vs, _ = obj_vertices_faces(new_dir / f"head_{state}_m.obj")
        diff = np.linalg.norm(vs - vb, axis=1)
        moved = diff > 1e-12
        r = prom["states"][state]
        q = np.asarray(r["binding"]["baseline_binding_point_reference_camera_m"])
        fwd = np.asarray(r["direction"]["forward_reference_camera"])
        light = np.array([0.3, 0.35, -0.89])
        light /= np.linalg.norm(light)

        def shaded(V):
            tri = V[F]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            return np.clip(n @ light, 0, 1) * 0.85 + 0.15

        fig = plt.figure(figsize=(13, 7), facecolor="#202020")
        for ax, V, title in (
            (fig.add_subplot(1, 2, 1), vb, f"{state} baseline (0.5.2)"),
            (fig.add_subplot(1, 2, 2), vs, f"{state} + marker-guided prominence"),
        ):
            ax.set_facecolor("#202020")
            ax.tripcolor(V[:, 0], V[:, 1], F, facecolors=shaded(V), cmap="gray", edgecolors="none")
            ax.set_xlim(q[0] - 0.085, q[0] + 0.085)
            ax.set_ylim(q[1] - 0.105, q[1] + 0.065)
            ax.set_title(title, color="w", fontsize=10)
            ax.set_aspect("equal")
            ax.axis("off")
        path = out_dir / f"{state}_front_compare.png"
        fig.savefig(path, dpi=110, facecolor="#202020")
        plt.close(fig)
        files.append(str(path))
    return files


def render_profiles(base_dir: Path, new_dir: Path, out_dir: Path, prom: dict) -> list[str]:
    files = []
    for state in ("closed", "open"):
        vb, _ = obj_vertices_faces(base_dir / f"head_{state}_m.obj")
        vs, _ = obj_vertices_faces(new_dir / f"head_{state}_m.obj")
        diff = np.linalg.norm(vs - vb, axis=1)
        r = prom["states"][state]
        b = r["binding"]
        q = np.asarray(b["baseline_binding_point_reference_camera_m"])
        fit = Path(prom["_reference_geometry_path"])
        rg = json.loads(fit.read_text(encoding="utf-8"))["references"][state]
        R = np.asarray(rg["animated_model_to_reference_camera_4x4"])[:3, :3]
        right = R @ np.asarray(r["direction"]["right_model"])
        up = R @ np.asarray(r["direction"]["up_model"])
        fwd = np.asarray(r["direction"]["forward_reference_camera"])

        # sagittal
        fig, ax = plt.subplots(figsize=(7, 6))
        for V, name, color in ((vb, "baseline", "#666"), (vs, "shaped", "#e04")):
            rel = V - q
            sel = np.abs(rel @ right) < 0.008
            ax.scatter((rel[sel] @ fwd) * 1000, (rel[sel] @ up) * 1000, s=12, c=color, label=name)
        marker = np.asarray(b["marker_point_reference_camera_m"]) - q
        ax.plot((marker @ fwd) * 1000, (marker @ up) * 1000, "b+", ms=14, mew=2, label="marker point")
        ax.plot(0, 0, "g+", ms=14, mew=2, label="baseline binding")
        sb = b.get("shaped_binding_point_reference_camera_m")
        if sb:
            s = np.asarray(sb) - q
            ax.plot((s @ fwd) * 1000, (s @ up) * 1000, "yx", ms=14, mew=2, label="shaped binding")
        ax.set_xlim(-40, 45)
        ax.set_ylim(-50, 55)
        ax.set_xlabel("forward (mm)")
        ax.set_ylabel("up (mm)")
        ax.set_title(f"{state}: neck front sagittal outline (midline ±8 mm)")
        ax.legend(loc="upper left")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        path = out_dir / f"{state}_sagittal_profile.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        files.append(str(path))

        # transverse local cross-section
        fig, ax = plt.subplots(figsize=(7, 5))
        for V, name, color in ((vb, "baseline", "#666"), (vs, "shaped", "#e04")):
            rel = V - q
            sel = np.abs(rel @ up) < 0.004
            ax.scatter((rel[sel] @ fwd) * 1000, (rel[sel] @ right) * 1000, s=12, c=color, label=name)
        ax.set_xlabel("forward (mm)")
        ax.set_ylabel("right (mm)")
        ax.set_title(f"{state}: transverse section at the binding height (±4 mm)")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        path = out_dir / f"{state}_transverse_section.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        files.append(str(path))

        # displacement field
        fig, ax = plt.subplots(figsize=(7, 4.5))
        rel = vb - q
        xx = (rel @ right) * 1000
        yy = (rel @ up) * 1000
        inw = np.sqrt((xx / 18.0) ** 2 + (yy / 27.0) ** 2) < 1.6
        sc = ax.scatter(xx[inw], yy[inw], c=diff[inw] * 1000, s=18, cmap="viridis")
        plt.colorbar(sc, label="displacement mm")
        ax.plot(0, 0, "r+", ms=10)
        ax.set_xlabel("right (mm)")
        ax.set_ylabel("up (mm)")
        ax.set_title(f"{state}: displacement field (max {diff.max()*1000:.2f} mm)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        path = out_dir / f"{state}_displacement_field.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        files.append(str(path))
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True, help="pure 0.5.2 run")
    parser.add_argument("--run", type=Path, required=True, help="run with reconstruction + neck_landmarks.json")
    parser.add_argument(
        "--sequence",
        type=Path,
        default=Path("D:/proiect/facecap_airway_delivery/runtime/outputs/zp_0.5.1/prepared/sequence.json"),
        help="prepared sequence.json of the run",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("D:/proiect/facecap_airway_delivery/runtime"),
        help="runtime directory holding models and env",
    )
    parser.add_argument("--landmarks", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="verification directory")
    args = parser.parse_args(argv)

    baseline = args.baseline_run
    run = args.run
    fit = run / "reconstruction" / "dense"
    sequence = args.sequence
    runtime = args.runtime
    landmarks = args.landmarks or (run / nl.LANDMARKS_FILE)
    out_dir = args.output or (run / "verification")
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"schema": "head_rgbd.neck_prominence.verification.v1", "items": {}}
    base_delivery = baseline / "delivery"
    base_hash = {
        state: sha256(base_delivery / f"head_{state}_m.obj") for state in ("closed", "open")
    }
    fit_report = json.loads((fit / "report.json").read_text(encoding="utf-8"))
    rg_path = fit / "reference_geometry.json"

    def ns(**kw):
        return SimpleNamespace(
            neck_landmarks=str(landmarks) if kw.pop("use_landmarks", True) else None, **kw
        )

    # ---- item 1: prominence off identical to clean baseline -----------------
    d = out_dir / "delivery_off"
    export(ns(subject_sex="male", neck_prominence="off"), fit, sequence, d, runtime)
    ok1 = all(sha256(d / f"head_{s}_m.obj") == base_hash[s] for s in ("closed", "open"))
    report["items"]["1_off_matches_clean_052_baseline"] = {
        "passed": ok1,
        "delivery": str(d),
        "baseline_hashes": base_hash,
        "off_hashes": {s: sha256(d / f"head_{s}_m.obj") for s in ("closed", "open")},
    }

    # ---- item 2: female + auto keeps every vertex ---------------------------
    d = out_dir / "delivery_female_auto"
    export(ns(subject_sex="female", neck_prominence="auto"), fit, sequence, d, runtime)
    ok2 = all(sha256(d / f"head_{s}_m.obj") == base_hash[s] for s in ("closed", "open"))
    female_prom = json.loads((d / "neck_prominence.json").read_text(encoding="utf-8"))
    report["items"]["2_female_auto_zero_displacement"] = {
        "passed": ok2 and female_prom["effective_enabled"] is False,
        "delivery": str(d),
        "reasons": {s: female_prom["states"][s]["reason"] for s in ("closed", "open")},
    }

    # ---- item 3: male + valid landmark shapes both states -------------------
    d = out_dir / "delivery_male"
    export(ns(subject_sex="male", neck_prominence="auto"), fit, sequence, d, runtime)
    prom = json.loads((d / "neck_prominence.json").read_text(encoding="utf-8"))
    bump: dict = {}
    for state in ("closed", "open"):
        vb, _ = obj_vertices_faces(base_delivery / f"head_{state}_m.obj")
        vs, _ = obj_vertices_faces(d / f"head_{state}_m.obj")
        diff = np.linalg.norm(vs - vb, axis=1)
        moved = diff > 1e-12
        # connectivity of the moved set
        adjacency = [set() for _ in range(len(vb))]
        faces = np.asarray(trimesh.load(str(base_delivery / f"head_{state}_m.obj"), process=False, maintain_order=True).faces)
        for a, b, c in faces:
            adjacency[a].update((b, c))
            adjacency[b].update((a, c))
            adjacency[c].update((a, b))
        ids = set(np.flatnonzero(moved).tolist())
        components, seen = 0, set()
        for v in list(ids):
            if v in seen:
                continue
            components += 1
            stack = [v]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                stack.extend(adjacency[x] & ids - seen)
        entry = prom["states"][state]
        bump[state] = {
            "moved_vertices": int(moved.sum()),
            "max_displacement_mm": float(diff.max() * 1000),
            "min_displacement_mm": float(diff[moved].min() * 1000) if moved.any() else 0.0,
            "connected_components": components,
            "status": entry["status"],
            "actual_amplitude_mm": entry.get("actual_amplitude_mm"),
            "selected_amplitude_mm": entry.get("selected_amplitude_mm"),
        }
    ok3 = all(
        bump[s]["status"] == "applied"
        and 0.5 <= bump[s]["max_displacement_mm"] <= 5.0
        and bump[s]["connected_components"] == 1
        and bump[s]["moved_vertices"] >= 5
        for s in ("closed", "open")
    )
    report["items"]["3_male_valid_landmark_continuous_bump"] = {
        "passed": ok3, "delivery": str(d), "states": bump,
    }

    # ---- item 4: missing or invalid landmark delivers the baseline ----------
    d = out_dir / "delivery_male_no_landmarks"
    export(
        ns(subject_sex="male", neck_prominence="auto", use_landmarks=False),
        fit, sequence, d, runtime,
    )
    no_lm = json.loads((d / "neck_prominence.json").read_text(encoding="utf-8"))
    ok4 = (
        all(sha256(d / f"head_{s}_m.obj") == base_hash[s] for s in ("closed", "open"))
        and no_lm["effective_enabled"] is False
        and all(no_lm["states"][s]["reason"] for s in ("closed", "open"))
    )
    report["items"]["4_missing_landmark_delivers_baseline"] = {
        "passed": ok4,
        "delivery": str(d),
        "reasons": {s: no_lm["states"][s]["reason"] for s in ("closed", "open")},
    }

    # ---- item 5: protection and unchanged facial first hits ----------------
    ok5, detail5 = True, {}
    for state in ("closed", "open"):
        vb, faces = obj_vertices_faces(base_delivery / f"head_{state}_m.obj")
        vs, _ = obj_vertices_faces(out_dir / "delivery_male" / f"head_{state}_m.obj")
        diff = np.linalg.norm(vs - vb, axis=1)
        moved = diff > 1e-12
        rg = json.loads(rg_path.read_text(encoding="utf-8"))["references"][state]
        lm = np.asarray(rg["stages"]["dense"]["landmarks_m"])
        hits_b = face_first_hits(vb, faces, lm)
        hits_s = face_first_hits(vs, faces, lm)
        same_hits = 0
        valid_hits = 0
        for hb, hs in zip(hits_b, hits_s):
            if hb is None and hs is None:
                same_hits += 1  # both meshes give no hit: unchanged
                continue
            if hb is None or hs is None:
                continue
            valid_hits += 1
            if hb["triangle_index"] == hs["triangle_index"] and np.allclose(
                hb["point_mesh_m"], hs["point_mesh_m"], atol=1e-12
            ):
                same_hits += 1
        untouched = np.array_equal(vs[~moved], vb[~moved])
        # strictly no movement near facial landmarks
        face_near = np.linalg.norm(vb[:, None, :] - lm[None, :, :], axis=2).min(axis=1) < 0.030
        face_moved = int((moved & face_near).sum())
        detail5[state] = {
            "untouched_vertices_identical": bool(untouched),
            "facial_landmark_first_hits_unchanged": f"{same_hits}/{len(hits_b)}",
            "facial_landmark_ray_hits": valid_hits,
            "facial_landmark_no_hit_both_meshes": int(sum(1 for hb in hits_b if hb is None)),
            "vertices_moved_within_30mm_of_face_landmarks": face_moved,
            "moved_vertices": int(moved.sum()),
        }
        ok5 = ok5 and untouched and same_hits == len(hits_b) and face_moved == 0
    report["items"]["5_protection_strictly_holds"] = {"passed": ok5, "states": detail5}

    # ---- item 6: repeated export does not accumulate ------------------------
    d1 = out_dir / "delivery_repeat_1"
    d2 = out_dir / "delivery_repeat_2"
    export(ns(subject_sex="male", neck_prominence="auto"), fit, sequence, d1, runtime)
    export(ns(subject_sex="male", neck_prominence="auto"), fit, sequence, d2, runtime)
    ok6 = all(
        sha256(d1 / f"head_{s}_m.obj") == sha256(d2 / f"head_{s}_m.obj") == sha256(
            out_dir / "delivery_male" / f"head_{s}_m.obj"
        )
        for s in ("closed", "open")
    )
    report["items"]["6_repeat_export_idempotent"] = {"passed": ok6}

    # ---- item 7: units and reported values ---------------------------------
    entry = prom["states"]["closed"]
    ok7 = (
        entry["requested_amplitude_mm"] == 3.0
        and entry.get("selected_amplitude_mm") in (2.0, 3.0, 4.0)
        and entry["support"]["radius_x_mm"] == 12.0
        and entry["support"]["radius_y_mm"] == 18.0
        and bump["closed"]["max_displacement_mm"] <= entry.get("selected_amplitude_mm", 0) * 1.001
        and all(
            abs((np.asarray(entry["binding"][k]) * 1000.0)).max() < 1000.0
            for k in ("marker_point_reference_camera_m", "baseline_binding_point_reference_camera_m")
        )
    )
    report["items"]["7_units_and_reported_values"] = {
        "passed": ok7,
        "requested_amplitude_mm": entry["requested_amplitude_mm"],
        "selected_amplitude_mm": entry.get("selected_amplitude_mm"),
        "radii_mm": [entry["support"]["radius_x_mm"], entry["support"]["radius_y_mm"]],
        "max_vertex_displacement_mm": bump["closed"]["max_displacement_mm"],
    }

    # ---- item 8: independent geometry safety recheck -----------------------
    safety: dict = {}
    for state in ("closed", "open"):
        vb, faces = obj_vertices_faces(base_delivery / f"head_{state}_m.obj")
        vs, _ = obj_vertices_faces(out_dir / "delivery_male" / f"head_{state}_m.obj")
        safety[state] = independent_geometry_check(vb, vs, faces)
    ok8 = all(v["status"] == "passed" for v in safety.values())
    report["items"]["8_geometry_safety_recheck"] = {"passed": ok8, "states": safety}

    # ---- timing: 3+ samples of the prominence stage -------------------------
    timing = []
    for index in range(3):
        d = out_dir / f"delivery_timing_{index}"
        started = time.perf_counter()
        export(ns(subject_sex="male", neck_prominence="auto"), fit, sequence, d, runtime)
        total = time.perf_counter() - started
        p = json.loads((d / "neck_prominence.json").read_text(encoding="utf-8"))
        stage_ms = [p["states"][s].get("elapsed_ms") for s in ("closed", "open")]
        timing.append({
            "run": index,
            "export_total_s": round(total, 3),
            "prominence_stage_ms": stage_ms,
            "prominence_stage_total_ms": round(sum(x for x in stage_ms if x), 3),
        })
    report["prominence_stage_timing"] = timing

    # ---- renders -----------------------------------------------------------
    prom["_reference_geometry_path"] = str(rg_path)
    report["renders"] = render_comparisons(base_delivery, out_dir / "delivery_male", out_dir, prom)
    report["renders"] += render_profiles(base_delivery, out_dir / "delivery_male", out_dir, prom)

    report["overall_passed"] = all(
        item["passed"] for item in report["items"].values()
    )
    (out_dir / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v["passed"] for k, v in report["items"].items()}, indent=2))
    print("overall_passed:", report["overall_passed"])
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
