"""Formal broad-volume delivery: three candidates, disk verification, final meshes, report."""
from __future__ import annotations

import argparse
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

from head_rgbd import neck_broad_volume as nb
from head_rgbd import neck_subject_v3 as v3
from tools.run_neck_subject_v3 import load_obj, render_view, render_wire, sha256
from tools.run_neck_sculpt import plane_section

PROJECT = Path("D:/proiect/facecap_airway_delivery")
FIT_DIR = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/reconstruction/dense"
PREPARED = PROJECT / "runtime/outputs/zp_0.5.1/prepared"
LANDMARKS = PROJECT / "runtime/outputs/zp_052_neck_bump_v2/neck_landmarks_v2.json"
BASELINE_DELIVERY = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/delivery"
REF_MESH = Path("D:/project_neck/line1/legacy/runtime/outputs/hack_default_verified/final/identity_rest.obj")
V056 = PROJECT / "runtime/outputs/zp_056_neck_sculpt_20261112_115219/presets/natural"
V057 = PROJECT / "runtime/outputs/zp_057_neck_proportional_20261112_124238/presets/natural"
CANDIDATES = ("broad_l2", "broad_m2", "broad_h3")
DEFAULT_CHOICE = "broad_m2"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or (PROJECT / "runtime/outputs" / f"neck_broad_volume_{stamp}")
    (out / "render").mkdir(parents=True, exist_ok=True)
    (out / "candidates").mkdir(exist_ok=True)
    (out / "final").mkdir(exist_ok=True)
    status, error = "failed", None
    t_all = time.perf_counter()
    results = {}
    timing = []
    try:
        references = json.loads((FIT_DIR / "reference_geometry.json").read_text(encoding="utf-8"))["references"]
        with np.load(FIT_DIR / "parameters.npz", allow_pickle=False) as stored:
            dense_vertices, dense_faces = stored["dense_vertices"], stored["dense_faces"]
        landmarks_payload = json.loads(LANDMARKS.read_text(encoding="utf-8"))
        sequence_rows = json.loads((PREPARED / "sequence.json").read_text(encoding="utf-8"))["frames"]
        base_meshes = {s: load_obj(BASELINE_DELIVERY / f"head_{s}_m.obj") for s in ("closed", "open")}
        ref_mesh = trimesh.load(str(REF_MESH), process=False, maintain_order=True)
        ref_V = np.asarray(ref_mesh.vertices)
        ref_F = np.asarray(ref_mesh.faces)
        ref_centre = ref_V.mean(axis=0)

        for name in CANDIDATES:
            p = nb.CANDIDATES[name]
            t0 = time.perf_counter()
            plan = nb.build_broad_states(
                states=["closed", "open"], references=references, dense_vertices=dense_vertices,
                dense_faces=dense_faces, prepared_dir=PREPARED, landmarks_payload=landmarks_payload,
                height_mm=p["height_mm"], width10_mm=p["width10_mm"], length10_mm=p["length10_mm"],
                sequence_rows=sequence_rows)
            timing.append({"stage": f"{name}:build", "seconds": round(time.perf_counter() - t0, 3)})
            delivery = out / "candidates" / name
            delivery.mkdir(parents=True, exist_ok=True)
            disk = {}
            for state in ("closed", "open"):
                shaped = plan["states"][state]["shaped_vertices"]
                path = delivery / f"head_{state}_m.obj"
                trimesh.Trimesh(shaped, plan["refined_faces"], process=False).export(path, include_normals=True)
                v_read, f_read = load_obj(path)
                disk[state] = {
                    "path": str(path), "sha256": sha256(path), "vertices": int(len(v_read)),
                    "faces": int(len(f_read)),
                    "max_abs_diff_m": float(np.max(np.abs(v_read - shaped))) if len(v_read) == len(shaped) else None,
                    "faces_match": bool(np.array_equal(f_read, plan["refined_faces"])),
                }
            # facial first-hit protection
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
                fac = np.array([item["protected_refined"][plan["refined_faces"][t]].all()
                                for t in tris]) if len(tris) else np.zeros(0, bool)
                facial[state] = {"facial_hits": int(fac.sum()),
                                 "max_facial_shift_mm": round(float(shift[fac].max()) if fac.any() else 0.0, 4)}
            checks = {
                "geometry_safe": all(nb.na.accept_guards(plan["states"][s]["guards"]) for s in ("closed", "open")),
                "no_self_intersections": all(plan["states"][s]["self_intersections"]["count"] == 0 for s in ("closed", "open")),
                "protected_unchanged": all(plan["states"][s]["protection_check"]["protected_vertices_moved"] == 0 for s in ("closed", "open")),
                "facial_hits_within_tolerance": all(facial[s]["max_facial_shift_mm"] <= 0.05 for s in facial),
                "disk_obj_verified": all(d["max_abs_diff_m"] is not None and d["max_abs_diff_m"] <= 1e-6 and d["faces_match"] for d in disk.values()),
                "broad_not_point": all(
                    (plan["states"][s]["metrics"].get("w50_mm") or 0) >= 8.0
                    and (plan["states"][s]["metrics"].get("realized_height_mm") or 9) <= 3.0
                    for s in ("closed", "open")),
            }
            results[name] = {"plan": plan, "disk": disk, "facial": facial, "checks": checks,
                             "accepted": all(checks.values())}
            print(name, "accepted", results[name]["accepted"],
                  {k: v for k, v in checks.items() if not v})

        # ---------- four-way comparison renders ----------
        t0 = time.perf_counter()
        chosen = next((n for n in (DEFAULT_CHOICE,) if n in results and results[n]["accepted"]), None)
        if chosen is None:
            chosen = next(n for n in results if results[n]["accepted"])
        plan = results[chosen]["plan"]
        compare = {
            "reference": (ref_V, ref_F, ref_centre),
        }
        for label, folder in (("056", V056), ("057", V057)):
            mesh = {}
            for s in ("closed", "open"):
                mesh[s] = load_obj(folder / f"head_{s}_m.obj")
            compare[label] = mesh
        for state in ("closed", "open"):
            frame = plan["states"][state]["frame"]
            centre = frame.origin
            fwd, right = frame.n, frame.u_axis
            shaped = plan["states"][state]["shaped_vertices"]
            rf = plan["refined_faces"]
            base_v, base_f = base_meshes[state]
            for view_name, view_dir, half in (("front", -fwd, 55.0), ("side", right, 55.0),
                                              ("45deg", (right - fwd) / np.sqrt(2.0), 55.0),
                                              ("neck_close", right, 32.0)):
                fig, axes = plt.subplots(1, 4, figsize=(21, 5.4), facecolor="#202020")
                # reference (rest frame) uses its own view basis
                ref_dir = np.array([1.0, 0, 0]) if view_name in ("side", "neck_close") else (
                    np.array([0, 0, -1.0]) if view_name == "front" else np.array([0.7, 0, -0.7]))
                for ax, (V, F, c, label) in zip(
                    axes,
                    ((ref_V, ref_F, ref_centre, "reference"),
                     *[(compare["056"][state][0], compare["056"][state][1], centre, "0.5.6 default"),
                       (compare["057"][state][0], compare["057"][state][1], centre, "0.5.7 candidate"),
                       (shaped, rf, centre, f"new {chosen}")])):
                    ax.set_facecolor("#202020")
                    render_view(ax, V, F, ref_dir if label == "reference" else view_dir, c, half)
                    ax.set_title(label, color="w", fontsize=9)
                fig.suptitle(f"{state} / {view_name} (same scale)", color="w")
                fig.tight_layout()
                fig.savefig(out / "render" / f"compare_{state}_{view_name}.png", dpi=100, facecolor="#202020")
                plt.close(fig)
            # wireframe + sections for the chosen candidate
            fig, ax = plt.subplots(figsize=(6, 5.2), facecolor="#202020")
            ax.set_facecolor("#202020")
            render_view(ax, shaped, rf, right, centre, 32.0)
            render_wire(ax, shaped, rf, right, centre, 32.0)
            ax.set_title(f"{chosen} / {state} wireframe", color="w", fontsize=9)
            fig.savefig(out / "render" / f"{chosen}_{state}_wireframe.png", dpi=110, facecolor="#202020")
            plt.close(fig)
            for sname, up_axis, limit in (("long", frame.v_axis, 26.0), ("cross", frame.u_axis, 24.0)):
                fig, ax = plt.subplots(figsize=(6.2, 5.2))
                for V, F, label, color in ((base_v, base_f, "baseline", "#8a97a8"),
                                           (shaped, rf, chosen, "#e05a4e")):
                    x, y = plane_section(V, F, centre, frame.u_axis if sname == "long" else up_axis,
                                         up_axis, frame.n, -limit, limit)
                    ax.plot(y, x, color=color, lw=1.8, label=label)
                ax.set_xlabel("outward n (mm)")
                ax.set_ylabel("v (mm)" if sname == "long" else "u (mm)")
                ax.set_title(f"{chosen} / {state}: {'longitudinal' if sname == 'long' else 'transverse'} section")
                ax.grid(alpha=0.3); ax.legend()
                fig.savefig(out / "render" / f"{chosen}_{state}_section_{sname}.png", dpi=110)
                plt.close(fig)
        timing.append({"stage": "renders", "seconds": round(time.perf_counter() - t0, 3)})

        # ---------- final delivery ----------
        plan = results[chosen]["plan"]
        for state in ("closed", "open"):
            shaped = plan["states"][state]["shaped_vertices"]
            trimesh.Trimesh(shaped, plan["refined_faces"], process=False).export(
                out / "final" / f"head_{state}_m.obj", include_normals=True)
        final_disk = {}
        for state in ("closed", "open"):
            v_read, f_read = load_obj(out / "final" / f"head_{state}_m.obj")
            final_disk[state] = {
                "path": str(out / "final" / f"head_{state}_m.obj"),
                "sha256": sha256(out / "final" / f"head_{state}_m.obj"),
                "vertices": int(len(v_read)), "faces": int(len(f_read)),
                "max_abs_diff_m": float(np.max(np.abs(v_read - plan["states"][state]["shaped_vertices"]))),
            }
        (out / "final" / "manifest.json").write_text(json.dumps({
            "schema": "head_rgbd.delivery.broad_volume.v1",
            "software": {"name": "facecap_airway_delivery", "version": "0.5.8-candidate"},
            "created_at": datetime.now().astimezone().isoformat(),
            "meshes": {"closed": "head_closed_m.obj", "open": "head_open_m.obj"},
            "mesh_unit": "m",
            "topology": "local red-green refined front-neck patch, shared closed/open",
            "editing_mode": True,
            "geometry_source": nb.GEOMETRY_SOURCE_BROAD,
            "subject_observation_matched": False,
            "candidate": chosen,
            "provenance": {"baseline_run": str(PROJECT / "runtime/outputs/zp_052_neck_bump_baseline"),
                            "reference_inspiration": str(REF_MESH),
                            "reference_used_for_transfer": False},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "final" / "shape_parameters.json").write_text(json.dumps({
            "schema": "head_rgbd.neck_broad_volume.shape.v1",
            "candidate": chosen,
            "design": plan["design"],
            "design_audit": plan["design_audit"],
            "scale": plan["scale"],
            "centre_binding": plan["centre_binding"],
            "states": {s: plan["states"][s]["metrics"] for s in ("closed", "open")},
            "profile": "p(u,v) = H * (1-(u/a)^2)^3 * (1-(v/b)^2)^3, C2 plateau in both directions",
            "note": "appearance design parameters inspired by the reference style; not measurements",
        }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        (out / "final" / "geometry_verification.json").write_text(json.dumps({
            "schema": "head_rgbd.neck_broad_volume.verification.v1",
            "chosen": chosen,
            "checks": results[chosen]["checks"],
            "facial": results[chosen]["facial"],
            "guards": {s: results[chosen]["plan"]["states"][s]["guards"] for s in ("closed", "open")},
            "self_intersections": {s: results[chosen]["plan"]["states"][s]["self_intersections"] for s in ("closed", "open")},
            "disk": final_disk,
            "all_candidates": {n: {"accepted": results[n]["accepted"], "checks": results[n]["checks"]} for n in results},
        }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

        summary = {
            "schema": "head_rgbd.neck_broad_volume.experiment.v1",
            "version": "0.5.8-candidate",
            "editing_mode": True,
            "geometry_source": nb.GEOMETRY_SOURCE_BROAD,
            "subject_observation_matched": False,
            "chosen": chosen,
            "candidates": {n: {
                "design_audit": results[n]["plan"]["design_audit"],
                "metrics": {s: results[n]["plan"]["states"][s]["metrics"] for s in ("closed", "open")},
                "checks": results[n]["checks"], "accepted": results[n]["accepted"],
                "disk": results[n]["disk"], "facial": results[n]["facial"],
                "scale": results[n]["plan"]["scale"],
            } for n in results},
            "final": final_disk,
            "default_entry_updated": False,
            "statement": "manual broad-volume appearance design (reference-inspired, not transferred); "
                         "two final meshes delivered under final/",
        }
        (out / "experiment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        timing.append({"stage": "total", "seconds": round(time.perf_counter() - t_all, 3)})
        with (out / "timing.csv").open("w", newline="", encoding="utf-8") as handle:
            import csv
            writer = csv.DictWriter(handle, fieldnames=["stage", "seconds"])
            writer.writeheader(); writer.writerows(timing)

        from tools.make_broad_volume_report import write_report
        write_report(out, summary=summary, timing=timing)
        (PROJECT / "docs" / "NECK_BROAD_VOLUME_DELIVERY_REPORT.html").write_bytes(
            (out / "NECK_BROAD_VOLUME_DELIVERY_REPORT.html").read_bytes())
        status = "ok" if results[chosen]["accepted"] else "verification_failed"
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
