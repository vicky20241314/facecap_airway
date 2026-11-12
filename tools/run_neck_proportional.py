"""Proportional sculpt experiment: presets, A/B envelope, annotated renders, report."""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from head_rgbd import neck_appearance as na
from head_rgbd import neck_proportional as npd
from head_rgbd import neck_subject_v3 as v3
from tools.run_neck_subject_v3 import load_obj, render_view, render_wire, sha256
from tools.run_neck_sculpt import plane_section

PROJECT = Path("D:/proiect/facecap_airway_delivery")
FIT_DIR = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/reconstruction/dense"
PREPARED = PROJECT / "runtime/outputs/zp_0.5.1/prepared"
LANDMARKS = PROJECT / "runtime/outputs/zp_052_neck_bump_v2/neck_landmarks_v2.json"
BASELINE_DELIVERY = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/delivery"
CANDIDATES = (
    ("light", "plateau"),
    ("natural", "plateau"),
    ("natural_wendland", "wendland"),
    ("pronounced", "plateau"),
)


def annotated_view(ax, V, F, view_dir, centre, half, frame, scale_info, design, support_rect):
    render_view(ax, V, F, view_dir, centre, half)
    fwd = np.asarray(view_dir, float); fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    rel = centre - frame.origin
    cu, cv = float(rel @ right) * 1000, float(rel @ up) * 1000
    ax.plot([cu], [cv], "r+", ms=12, mew=2)
    ax.add_patch(plt.Rectangle((cu - support_rect[0] / 2, cv - support_rect[2]),
                               support_rect[0], support_rect[2] + support_rect[1],
                               fill=False, ec="#e04", lw=1.2, ls="--"))
    ax.text(cu, cv + 2, f"design centre s={scale_info['centre_s']:.2f}", color="#ff9", fontsize=8,
            ha="center")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or (PROJECT / "runtime/outputs" / f"zp_057_neck_proportional_{stamp}")
    (out / "render").mkdir(parents=True, exist_ok=True)
    status, error = "failed", None
    t_all = time.perf_counter()
    timing, results = [], {}
    try:
        references = json.loads((FIT_DIR / "reference_geometry.json").read_text(encoding="utf-8"))["references"]
        with np.load(FIT_DIR / "parameters.npz", allow_pickle=False) as stored:
            dense_vertices, dense_faces = stored["dense_vertices"], stored["dense_faces"]
        landmarks_payload = json.loads(LANDMARKS.read_text(encoding="utf-8"))
        sequence_rows = json.loads((PREPARED / "sequence.json").read_text(encoding="utf-8"))["frames"]
        base_meshes = {s: load_obj(BASELINE_DELIVERY / f"head_{s}_m.obj") for s in ("closed", "open")}

        for name, envelope in CANDIDATES:
            preset = name.replace("_wendland", "")
            t0 = time.perf_counter()
            plan = npd.build_proportional_states(
                states=["closed", "open"], references=references, dense_vertices=dense_vertices,
                dense_faces=dense_faces, prepared_dir=PREPARED, landmarks_payload=landmarks_payload,
                preset=preset, sequence_rows=sequence_rows, envelope=envelope)
            timing.append({"stage": f"{name}:build", "seconds": round(time.perf_counter() - t0, 3)})
            delivery = out / "presets" / name
            delivery.mkdir(parents=True, exist_ok=True)
            disk = {}
            for state in ("closed", "open"):
                shaped = plan["states"][state]["shaped_vertices"]
                path = delivery / f"head_{state}_m.obj"
                trimesh.Trimesh(shaped, plan["refined_faces"], process=False).export(path, include_normals=True)
                v_read, f_read = load_obj(path)
                disk[state] = {
                    "path": str(path), "sha256": sha256(path), "vertices": int(len(v_read)),
                    "max_abs_diff_m": float(np.max(np.abs(v_read - shaped))) if len(v_read) == len(shaped) else None,
                    "faces_match": bool(np.array_equal(f_read, plan["refined_faces"])),
                }
            facial = {}
            for state in ("closed", "open"):
                item = plan["states"][state]
                lm_cam = np.asarray(references[state]["stages"]["dense"]["landmarks_m"], float)
                base_v, base_f = base_meshes[state]
                hb = v3.first_hit_batch(base_v, base_f, lm_cam)
                hs = v3.first_hit_batch(item["shaped_vertices"], plan["refined_faces"], lm_cam)
                ok = hb["hit"] & hs["hit"]
                shift = np.linalg.norm(hb["point_m"][ok] - hs["point_m"][ok], axis=1) * 1000.0
                tris = hb["triangle"][ok]
                fac = np.array([item["protected_refined"][plan["refined_faces"][t]].all() for t in tris]) if len(tris) else np.zeros(0, bool)
                facial[state] = {"facial_hits": int(fac.sum()),
                                 "max_facial_shift_mm": round(float(shift[fac].max()) if fac.any() else 0.0, 4)}
            metrics = {}
            for state in ("closed", "open"):
                r = plan["states"][state]
                metrics[state] = {**r["metrics"],
                                  "tip_radius_measured_mm": r["tip_radius_measured_mm"],
                                  "centre_s": r["centre_s"], "protection_distance_mm": r["protection_distance_mm"],
                                  "guards": r["guards"], "self_intersections": r["self_intersections"]["count"],
                                  "protected_moved": r["protection_check"]["protected_vertices_moved"]}
            checks = {
                "safe_geometry": all(na.accept_guards(plan["states"][s]["guards"]) for s in ("closed", "open")),
                "no_self_intersections": all(plan["states"][s]["self_intersections"]["count"] == 0 for s in ("closed", "open")),
                "protected_unchanged": all(plan["states"][s]["protection_check"]["protected_vertices_moved"] == 0 for s in ("closed", "open")),
                "facial_hits_within_tolerance": all(facial[s]["max_facial_shift_mm"] <= 0.05 for s in facial),
                "disk_obj_verified": all(d["max_abs_diff_m"] is not None and d["max_abs_diff_m"] <= 1e-6 and d["faces_match"] for d in disk.values()),
                "H_ratio_in_range": all(0.045 <= plan["design_audit"]["H_over_Wneck"] <= 0.065 for _ in [0]),
                "L10_ratio_in_range": all(0.35 <= plan["design_audit"]["L10_over_Lneck"] <= 0.50 for _ in [0]),
                "W10_ratio_in_range": 0.26 <= plan["design_audit"]["W10_over_Wneck"] <= 0.34,
                "L50_W50_preferred": all(
                    1.1 <= (metrics[s].get("length_half_mm") or 0) / max(metrics[s].get("width_half_mm") or 1e-9, 1e-9) <= 1.5
                    for s in ("closed", "open")),
                "open_close_consistency": abs(metrics["open"]["realized_height_mm"] - metrics["closed"]["realized_height_mm"])
                                          <= 0.15 * metrics["closed"]["realized_height_mm"],
            }
            results[name] = {"plan": plan, "disk": disk, "facial": facial, "metrics": metrics,
                             "checks": checks, "envelope": envelope, "preset": preset,
                             "accepted_core": all(v for k, v in checks.items()
                                                  if k in ("safe_geometry", "no_self_intersections", "protected_unchanged",
                                                           "facial_hits_within_tolerance", "disk_obj_verified"))}

        # renders for the three presets (natural = plateau)
        t0 = time.perf_counter()
        for name in ("light", "natural", "natural_wendland", "pronounced"):
            plan = results[name]["plan"]
            for state in ("closed", "open"):
                item = plan["states"][state]
                frame = item["frame"]
                base_v, base_f = base_meshes[state]
                shaped = item["shaped_vertices"]
                rf = plan["refined_faces"]
                for view_name, view_dir, half in (("front", -frame.n, 55.0), ("side", frame.u_axis, 55.0),
                                                  ("45deg", (frame.u_axis - frame.n) / np.sqrt(2), 55.0),
                                                  ("side_close", frame.u_axis, 30.0)):
                    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), facecolor="#202020")
                    for ax, (V, F, label) in zip(axes, ((base_v, base_f, "baseline 0.5.2"),
                                                        (shaped, rf, f"{name}"))):
                        ax.set_facecolor("#202020")
                        render_view(ax, V, F, view_dir, frame.origin, half)
                        ax.set_title(f"{name} / {state} — {label}", color="w", fontsize=9)
                    fig.tight_layout()
                    fig.savefig(out / "render" / f"{name}_{state}_{view_name}.png", dpi=110, facecolor="#202020")
                    plt.close(fig)
                # annotated side view with the support rectangle
                d = plan["design"]
                fig, ax = plt.subplots(figsize=(6.6, 5.6), facecolor="#202020")
                ax.set_facecolor("#202020")
                render_view(ax, shaped, rf, frame.u_axis, frame.origin, 45.0)
                annotated_view(ax, shaped, rf, frame.u_axis, frame.origin, 45.0, frame, plan["scale"], d,
                               (d["width_total_mm"], d["length_up_mm"], d["length_down_mm"]))
                ax.set_title(f"{name} / {state}: design centre, support, W_neck {plan['scale']['W_neck_mm']:.0f} mm, "
                             f"L_neck {plan['scale']['L_neck_mm']:.0f} mm", color="w", fontsize=8)
                fig.tight_layout()
                fig.savefig(out / "render" / f"{name}_{state}_annotated.png", dpi=110, facecolor="#202020")
                plt.close(fig)
                # sections
                for sname, up_axis, limit in (("long", frame.v_axis, 30.0), ("cross", frame.u_axis, 22.0)):
                    fig, ax = plt.subplots(figsize=(6.2, 5.2))
                    for V, F, label, color in ((base_v, base_f, "baseline", "#8a97a8"),
                                               (shaped, rf, name, "#e05a4e")):
                        x, y = plane_section(V, F, frame.origin, frame.u_axis if sname == "long" else frame.v_axis,
                                             up_axis, frame.n, -limit, limit)
                        ax.plot(y, x, color=color, lw=1.8, label=label)
                    ax.set_xlabel("outward n (mm)")
                    ax.set_ylabel("v (mm)" if sname == "long" else "u (mm)")
                    ax.set_title(f"{name} / {state}: {'longitudinal' if sname == 'long' else 'transverse'} section")
                    ax.grid(alpha=0.3); ax.legend()
                    fig.savefig(out / "render" / f"{name}_{state}_section_{sname}.png", dpi=110)
                    plt.close(fig)
        timing.append({"stage": "renders", "seconds": round(time.perf_counter() - t0, 3)})

        rows = []
        for name in results:
            for state in ("closed", "open"):
                m = results[name]["metrics"][state]
                a = results[name]["plan"]["design_audit"]
                rows.append({
                    "candidate": name, "state": state,
                    **{k: m.get(k) for k in ("requested_height_mm", "realized_height_mm", "width_ten_mm", "width_half_mm",
                                              "length_ten_mm", "length_half_mm", "tip_radius_measured_mm", "centre_s",
                                              "protection_distance_mm")},
                    "H_over_Wneck": a["H_over_Wneck"], "W10_over_Wneck": a["W10_over_Wneck"],
                    "L10_over_Lneck": a["L10_over_Lneck"],
                    "L50_over_W50": round((m.get("length_half_mm") or 0) / max(m.get("width_half_mm") or 1e-9, 1e-9), 3),
                    "true_fold": m["guards"]["true_fold"], "max_edge_growth": m["guards"]["max_edge_growth"],
                    "self_intersections": m["self_intersections"], "protected_moved": m["protected_moved"],
                })
        with (out / "proportional_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
        summary = {
            "schema": "head_rgbd.neck_proportional.experiment.v1",
            "version": "0.5.7",
            "editing_mode": True, "geometry_source": "procedural_appearance", "subject_observation_matched": False,
            "candidates": {name: {"preset": results[name]["preset"], "envelope": results[name]["envelope"],
                                  "scale": results[name]["plan"]["scale"],
                                  "design": results[name]["plan"]["design_audit"],
                                  "metrics": results[name]["metrics"],
                                  "facial": results[name]["facial"],
                                  "disk": results[name]["disk"],
                                  "checks": results[name]["checks"],
                                  "accepted_core": results[name]["accepted_core"]} for name in results},
            "default_candidate": "natural",
            "public_default_unchanged": True,
            "blockers": [],
        }
        blockers = []
        nat = results["natural"]
        if not nat["checks"]["W10_ratio_in_range"]:
            blockers.append("W10/W_neck outside 0.26-0.34: coordinated width chosen for a short neck "
                            "(nominal band recorded in the design audit)")
        if not nat["checks"]["L50_W50_preferred"]:
            blockers.append("L50/W50 outside the preferred 1.1-1.5 on the delivered mesh")
        if not nat["checks"]["open_close_consistency"]:
            blockers.append("open realized height deviates >15% from closed")
        summary["blockers"] = blockers
        (out / "experiment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        timing.append({"stage": "total", "seconds": round(time.perf_counter() - t_all, 3)})
        with (out / "timing.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stage", "seconds"])
            writer.writeheader(); writer.writerows(timing)

        from tools.neck_proportional_report import write_report
        report = write_report(out, summary=summary, timing=timing)
        (PROJECT / "docs" / "NECK_PROPORTIONAL_SCULPT_REPORT.html").write_bytes(report.read_bytes())
        core_ok = all(results[name]["accepted_core"] for name in results)
        status = "ok" if core_ok else "verification_failed"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        (out / "run_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        summary = {"status": status, "error": error, "exit_code": 0 if status == "ok" else 1,
                   "total_seconds": round(time.perf_counter() - t_all, 3), "output": str(out)}
        (out / "run_status.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
