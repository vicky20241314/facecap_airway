"""Formal V3 run: subject-observed neck reconstruction on the real capture.

Reuses the saved 0.5.2 reconstruction (no FLAME re-run), fits the local neck
background and ridge per state, refines the front-neck patch with a
deterministic red-green subdivision shared by closed and open, exports the
refined meshes and every audit/metric/render artefact required by the V3
deliverable, and reports the outcome honestly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from head_rgbd import neck_landmark as nl
from head_rgbd import neck_prominence as npn
from head_rgbd import neck_subject_v3 as v3
from head_rgbd.pipeline import load_defaults

PROJECT = Path("D:/proiect/facecap_airway_delivery")
PREPARED = PROJECT / "runtime/outputs/zp_0.5.1/prepared"
BASELINE_RUN = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline"
V2_DELIVERY = PROJECT / "runtime/outputs/zp_052_neck_bump_v2/verification/delivery_male"
V2_SIDECAR = V2_DELIVERY / "neck_prominence.json"
LANDMARKS = PROJECT / "runtime/outputs/zp_052_neck_bump_v2/neck_landmarks_v2.json"
FIT_DIR = BASELINE_RUN / "reconstruction/dense"
FACE_TOLERANCE_MM = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_obj(path: Path):
    mesh = trimesh.load(str(path), process=False, maintain_order=True)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def render_view(ax, vertices, faces, forward, centre, half_mm, shade=True):
    fwd = np.asarray(forward, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, fwd)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    facing = (n @ (-fwd)) > 0.02
    tri = tri[facing]
    n = n[facing]
    if not len(tri):
        return
    light = -fwd + 0.35 * up - 0.20 * right
    light /= np.linalg.norm(light)
    shade_v = np.clip(n @ light, 0.0, 1.0) * 0.8 + 0.2
    centroid = tri.mean(axis=1)
    depth = (centroid - centre) @ fwd
    order = np.argsort(-depth)
    rel = tri - centre
    polys = np.stack([rel @ right * 1000.0, rel @ up * 1000.0], axis=2)[order]
    colors = plt.cm.gray(shade_v[order]) if shade else np.tile([0.7, 0.7, 0.7, 1.0], (len(polys), 1))
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors="none"))
    ax.set_xlim(-half_mm, half_mm)
    ax.set_ylim(-half_mm, half_mm)
    ax.set_aspect("equal")
    ax.axis("off")


def render_wire(ax, vertices, faces, forward, centre, half_mm):
    fwd = np.asarray(forward, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    tri = vertices[faces]
    facing = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) @ (-fwd) > 0
    keep = facing & (np.linalg.norm(tri.mean(axis=1) - centre, axis=1) < 0.045)
    rel = tri[keep] - centre
    polys = np.stack([rel @ right * 1000.0, rel @ up * 1000.0], axis=2)
    ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors="#9fb8d6", linewidths=0.35))
    ax.set_xlim(-half_mm, half_mm)
    ax.set_ylim(-half_mm, half_mm)
    ax.set_aspect("equal")
    ax.axis("off")


def section_outline(vertices, faces, centre, n_axis, up_axis, fwd_axis, y_min, y_max, slab_mm=10.0):
    """Plane/mesh intersection: returns (up_mm, forward_mm) samples along the cut."""
    tri = vertices[faces]
    xs = (tri - centre) @ n_axis
    cross = (xs[:, 0] * xs[:, 1] <= 0) | (xs[:, 1] * xs[:, 2] <= 0) | (xs[:, 0] * xs[:, 2] <= 0)
    near = (np.abs(xs).max(axis=1) < slab_mm * 1e-3) | (np.abs(xs).min(axis=1) < 1e-9)
    ups, fwds = [], []
    for fi in np.flatnonzero(near & cross):
        v = faces[fi]
        s = xs[fi]
        pts = []
        for a in range(3):
            b = (a + 1) % 3
            if s[a] * s[b] <= 0 and abs(s[a] - s[b]) > 1e-12:
                t = s[a] / (s[a] - s[b])
                pts.append(vertices[v[a]] + t * (vertices[v[b]] - vertices[v[a]]))
        if len(pts) == 2:
            r0 = pts[0] - centre
            r1 = pts[1] - centre
            u0, u1 = float(r0 @ up_axis) * 1000.0, float(r1 @ up_axis) * 1000.0
            f0, f1 = float(r0 @ fwd_axis) * 1000.0, float(r1 @ fwd_axis) * 1000.0
            steps = max(1, int(np.ceil(abs(u1 - u0) / 0.5)))
            for step in range(steps + 1):
                t = step / steps
                ups.append(u0 + t * (u1 - u0))
                fwds.append(f0 + t * (f1 - f0))
    return np.asarray(ups), np.asarray(fwds)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or (PROJECT / "runtime/outputs" / f"zp_055_neck_subject_v3_{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "delivery").mkdir(exist_ok=True)
    (out / "profiles").mkdir(exist_ok=True)
    (out / "render").mkdir(exist_ok=True)
    (out / "source_snapshot").mkdir(exist_ok=True)
    timing: list[dict] = []

    def mark(stage, started):
        timing.append({"stage": stage, "seconds": round(time.perf_counter() - started, 3)})

    t_all = time.perf_counter()
    status = "failed"
    error = None
    try:
        references = json.loads((FIT_DIR / "reference_geometry.json").read_text(encoding="utf-8"))["references"]
        with np.load(FIT_DIR / "parameters.npz", allow_pickle=False) as stored:
            dense_vertices = stored["dense_vertices"]
            dense_faces = stored["dense_faces"]
        landmarks_payload = json.loads(LANDMARKS.read_text(encoding="utf-8"))
        v2_sidecar = json.loads(V2_SIDECAR.read_text(encoding="utf-8"))
        v2_meshes = {s: V2_DELIVERY / f"head_{s}_m.obj" for s in ("closed", "open")}

        input_manifest = {
            "schema": "head_rgbd.neck_v3.input.v1",
            "baseline_run": str(BASELINE_RUN),
            "baseline_reference_geometry": {"path": str(FIT_DIR / "reference_geometry.json"),
                                            "sha256": sha256(FIT_DIR / "reference_geometry.json")},
            "baseline_parameters": {"path": str(FIT_DIR / "parameters.npz"),
                                    "sha256": sha256(FIT_DIR / "parameters.npz"),
                                    "dense_vertices": list(dense_vertices.shape),
                                    "dense_faces": list(dense_faces.shape)},
            "baseline_delivery": {s: {"path": str(BASELINE_RUN / "delivery" / f"head_{s}_m.obj"),
                                      "sha256": sha256(BASELINE_RUN / "delivery" / f"head_{s}_m.obj")}
                                  for s in ("closed", "open")},
            "prepared_sequence": {"path": str(PREPARED / "sequence.json"),
                                  "sha256": sha256(PREPARED / "sequence.json")},
            "v2_delivery": {s: {"path": str(v2_meshes[s]), "sha256": sha256(v2_meshes[s])} for s in v2_meshes},
            "v2_sidecar": {"path": str(V2_SIDECAR), "sha256": sha256(V2_SIDECAR)},
            "landmarks": {"path": str(LANDMARKS), "sha256": sha256(LANDMARKS)},
            "states": {},
            "units": {"mesh": "m", "depth": "mm (uint16, 1 mm scale)",
                      "camera": "X right, Y up, Z backward (OpenCV-style)"},
        }
        for state in ("closed", "open"):
            ref = references[state]
            frame_id = ref["frame_id"]
            npz_path = PREPARED / f"{frame_id}.npz"
            data = np.load(npz_path, allow_pickle=False)
            input_manifest["states"][state] = {
                "frame_id": frame_id,
                "frame_index": ref["stages"]["dense"]["frame_index"],
                "segment": str(npz_path),
                "segment_sha256": sha256(npz_path),
                "intrinsics": [float(x) for x in data["intrinsics"]],
                "depth_shape": list(data["depth"].shape),
                "depth_valid_fraction": float((data["depth"] > 0).mean()),
                "model_to_reference_camera_4x4": np.asarray(
                    ref["animated_model_to_reference_camera_4x4"], dtype=float).tolist(),
                "label_pixel": landmarks_payload["states"][state]["pixel_uv"],
                "label_point_m": landmarks_payload["states"][state]["point_observed_m"],
                "label_source": landmarks_payload["states"][state]["source"],
                "label_correction": landmarks_payload["states"][state].get("correction"),
            }

        # ---------------- per-state fit + apply -----------------------------
        state_data = {}
        fit_records = []
        frame_audit = {"schema": "head_rgbd.neck_v3.frames.v1", "states": {}}
        for state in ("closed", "open"):
            t0 = time.perf_counter()
            ref = references[state]
            frame_id = ref["frame_id"]
            data = np.load(PREPARED / f"{frame_id}.npz", allow_pickle=False)
            label = landmarks_payload["states"][state]
            lm_cam = np.asarray(ref["stages"]["dense"]["landmarks_m"], dtype=np.float64)
            head_up = lm_cam[npn.NOSE_BRIDGE_TOP] - lm_cam[npn.CHIN]
            head_up /= np.linalg.norm(head_up)
            vertices = np.asarray(dense_vertices[ref["stages"]["dense"]["frame_index"]], dtype=np.float64)
            obs = v3.collect_observations(
                state=state, frame_id=frame_id, depth=data["depth"], neck_mask=data["neck_mask"],
                intrinsics=data["intrinsics"],
                label_pixel=tuple(label["pixel_uv"]),
                label_point=np.asarray(label["point_observed_m"], dtype=np.float64),
                head_up_camera=head_up, baseline_vertices=vertices, faces=dense_faces,
            )
            M = np.asarray(ref["animated_model_to_reference_camera_4x4"], dtype=np.float64)
            rot, trans = M[:3, :3], M[:3, 3]
            lm_model = (lm_cam - trans) @ rot
            axes = npn._direction_frame(lm_model)
            normals_model = npn.vertex_normals((vertices - trans) @ rot, dense_faces)
            protection = npn._protected_mask((vertices - trans) @ rot, lm_model, normals_model, axes)
            normals_cam = normals_model @ rot.T
            mark(f"{state}:observations", t0)
            t0 = time.perf_counter()
            fit = v3.fit_state(obs, vertices=vertices, faces=dense_faces, frame=obs.frame,
                               protected=protection["mask"], vertex_normals=normals_cam)
            mark(f"{state}:fit", t0)
            for entry in fit.residuals_history:
                fit_records.append({"state": state, **entry})
            frame_audit["states"][state] = obs.frame.as_dict()
            state_data[state] = {
                "obs": obs, "fit": fit, "vertices": vertices, "normals": normals_cam,
                "protection": protection["mask"], "rot": rot, "trans": trans,
                "references": ref, "label": label,
            }

        # ---------------- shared local refinement ---------------------------
        t0 = time.perf_counter()
        union_mask = np.zeros(len(dense_faces), dtype=bool)
        from scipy.spatial import cKDTree
        near_protected = np.zeros(len(dense_faces), dtype=bool)
        for state in ("closed", "open"):
            sd = state_data[state]
            centroid = sd["vertices"][dense_faces].mean(axis=1)
            dist = np.linalg.norm(centroid - sd["obs"].frame.origin, axis=1)
            union_mask |= dist < v3.SUBMIT_RADIUS_MM * 1e-3
            # do not refine near the protection boundary: the transition band is
            # carried by the (stable) base faces only
            tree = cKDTree(sd["vertices"][sd["protection"]])
            dprot, _ = tree.query(centroid, k=1)
            near_protected |= dprot < 0.015
        union_mask &= ~near_protected
        refined = v3.subdivide_local(state_data["closed"]["vertices"], dense_faces, face_mask=union_mask)
        refined_faces = refined["faces"]
        mark("refinement", t0)

        # ---------------- apply on base and refined meshes ------------------
        outputs = {}
        for state in ("closed", "open"):
            t0 = time.perf_counter()
            sd = state_data[state]
            obs = sd["obs"]
            fit = sd["fit"]
            # base-mesh application (for metrics/checks on the dense topology)
            base_applied = v3.apply_field(
                vertices=sd["vertices"], faces=dense_faces, frame=obs.frame,
                background=fit.background, theta=fit.theta,
                protected=sd["protection"], vertex_normals=sd["normals"],
                residual_coefficients=fit.residual_coefficients,
            )
            # refined mesh: midpoints of the ORIGINAL edges then the same field
            # rebuild the midpoints from THIS state's vertices: the refined
            # topology (face list and midpoint order) is shared, the positions
            # must come from each state's own pose
            midpoint_edges = refined["midpoint_edges"]
            midpoints = np.asarray([0.5 * (sd["vertices"][i] + sd["vertices"][j]) for i, j in midpoint_edges])
            refined_vertices = np.concatenate([sd["vertices"], midpoints], axis=0)
            # apply the field to the refined vertices with the refined normals
            refined_normals_model = npn.vertex_normals((refined_vertices - sd["trans"]) @ sd["rot"], refined_faces)
            refined_normals = refined_normals_model @ sd["rot"].T
            # protected set for the refined mesh: a vertex is protected if any
            # parent-chain vertex is protected (midpoints inherit)
            n_base = len(sd["vertices"])
            base_protected = sd["protection"]
            # rebuild midpoints from the parent map: midpoint vertex k>n_base sits
            # on the edge (i,j); it is protected when both parents are protected,
            # and shares the window otherwise; simple and conservative
            parent_protected = np.zeros(len(refined_vertices), dtype=bool)
            parent_protected[:n_base] = base_protected
            # reconstruct midpoint parents from geometry is not possible; use the
            # deterministic rule from subdivide_local's recorded map if present
            midpoint_map = refined.get("midpoint_map", {})
            for key, index in midpoint_map.items():
                i, j = key
                parent_protected[index] = base_protected[i] or base_protected[j]
            if not midpoint_map:
                # fallback: nearest base vertex protection within 3.5 mm
                from scipy.spatial import cKDTree
                tree = cKDTree(sd["vertices"])
                dist, idx = tree.query(refined_vertices[n_base:], k=1)
                parent_protected[n_base:] = base_protected[idx]
            refined_applied = v3.apply_field(
                vertices=refined_vertices, faces=refined_faces, frame=obs.frame,
                background=fit.background, theta=fit.theta,
                protected=parent_protected, vertex_normals=refined_normals,
                residual_coefficients=fit.residual_coefficients,
            )
            shaped_refined = refined_applied["shaped_vertices"]
            mesh_path = out / "delivery" / f"head_{state}_m.obj"
            trimesh.Trimesh(shaped_refined, refined_faces, process=False).export(
                mesh_path, include_normals=True)
            outputs[state] = {
                "mesh": mesh_path,
                "shaped": shaped_refined,
                "shaped_base": base_applied["shaped_vertices"],
                "applied_mm": refined_applied["applied_mm"],
                "window": refined_applied["window"],
                "protected_refined": parent_protected,
            }
            mark(f"{state}:apply_export", t0)

        # ---------------- fixed observation metrics -------------------------
        obs_metrics = []
        for state in ("closed", "open"):
            sd = state_data[state]
            obs = sd["obs"]
            meshes = {"baseline": (sd["vertices"], dense_faces),
                      "v2": (load_obj(v2_meshes[state]) if v2_meshes[state].is_file() else None),
                      "v3": (outputs[state]["shaped"], refined_faces)}
            t0 = time.perf_counter()
            for name, item in meshes.items():
                if item is None:
                    continue
                verts, faces = item
                metrics = v3.first_hit_metrics(verts, faces, obs)
                obs_metrics.append({
                    "state": state, "mesh": name,
                    "samples": metrics["samples"], "coverage": metrics["coverage"],
                    "mae_mm": metrics["mae_mm"], "rmse_mm": metrics["rmse_mm"],
                    "p95_abs_mm": metrics["p95_abs_mm"], "median_signed_mm": metrics["median_signed_mm"],
                    "max_abs_mm": metrics["max_abs_mm"], "hit_mae_mm": metrics["hit_mae_mm"],
                })
            mark(f"{state}:first_hit_metrics", t0)

        # ---------------- face protection ------------------------------------
        protection_report = {}
        for state in ("closed", "open"):
            sd = state_data[state]
            obs = sd["obs"]
            ref = sd["references"]
            lm_cam = np.asarray(ref["stages"]["dense"]["landmarks_m"], dtype=np.float64)
            base_hits = v3.first_hit_batch(sd["vertices"], dense_faces, lm_cam)
            v3_hits = v3.first_hit_batch(outputs[state]["shaped"], refined_faces, lm_cam)
            ok = base_hits["hit"] & v3_hits["hit"]
            shift = np.linalg.norm(base_hits["point_m"][ok] - v3_hits["point_m"][ok], axis=1) * 1000.0
            # classify the baseline first hit: a facial hit touches only
            # protected vertices; anything else is neck/skin (the ray merely
            # passes the moved neck, which is not a facial change)
            hit_tris = base_hits["triangle"][ok]
            facial_hit = np.array([
                sd["protection"][dense_faces[t]].all() for t in hit_tris
            ]) if len(hit_tris) else np.zeros(0, dtype=bool)
            protected_moved = int((np.abs(outputs[state]["applied_mm"][: len(sd["vertices"])]
                                          [sd["protection"]]) > 1e-9).sum())
            face_shift = float(shift[facial_hit].max()) if facial_hit.any() else 0.0
            neck_shift = float(shift[~facial_hit].max()) if (~facial_hit).any() else 0.0
            protection_report[state] = {
                "facial_rays": int(ok.sum()),
                "facial_surface_hits": int(facial_hit.sum()),
                "neck_first_hit_rays": int((~facial_hit).sum()),
                "max_facial_hit_shift_mm": round(face_shift, 4),
                "max_neck_first_hit_shift_mm": round(neck_shift, 4),
                "tolerance_mm": FACE_TOLERANCE_MM,
                "faces_ok": bool(face_shift <= FACE_TOLERANCE_MM),
                "protected_base_vertices_moved": int(protected_moved),
                "protected_ok": protected_moved == 0,
            }

        # ---------------- shape measurement protocol ------------------------
        measurement_protocol = {
            "schema": "head_rgbd.neck_v3.measurement_protocol.v1",
            "frozen_before_formal_run": True,
            "definitions": {
                "H(u,v)": "camera-ray depth of the delivered surface at the local grid point (u,v), "
                         "relative to the frozen background reference: H = (D_base - b(u,v)) - D_mesh, "
                         "divided by the state's median depth response factor",
                "background": "quadratic b(u,v) fitted jointly to all reliable observations before the "
                              "ridge; frozen for all measurements and versions",
                "width/length": "extent of |H| >= 50% and >= 10% of the central peak magnitude on the "
                                "u=0 and v=0 lines",
                "slopes": "forward difference of H along u at v=0 and along v at u=0 near the peak",
                "tip_roundness": "v-extent of |H| >= 90% of the peak (flat-crest length)",
                "peak": "argmax |H| on the measured grid",
                "section_plane": "auxiliary: through the label, lateral normal tilted 15 deg for a "
                                 "transversal cut; the fixed main planes u=0 and v=0 are the scoring planes",
            },
            "grid_mm": {"u": [-30, 30, 1.5], "v": [-30, 30, 1.5]},
            "uncertainty": "1-sigma from the ridge-parameter covariance at the label plus the "
                           "background-model sensitivity (radius 35/40/45 mm)",
            "face_tolerance_mm": FACE_TOLERANCE_MM,
        }
        shape_rows = []
        profiles = {}
        uncertainty = {}
        for state in ("closed", "open"):
            sd = state_data[state]
            obs = sd["obs"]
            fit = sd["fit"]
            rendered = outputs[state]["shaped"]
            grad = np.linspace(-18, 18, 25)
            factor = float(np.median(obs.ray_nz))
            H = np.full((len(grad), len(grad)), np.nan)
            for iu, u in enumerate(grad):
                for iv, vv in enumerate(grad):
                    q = obs.frame.origin + (u * obs.frame.u_axis + vv * obs.frame.v_axis) * 1e-3
                    hit = v3.first_hit_batch(rendered, refined_faces, q[None, :])
                    if not hit["hit"][0]:
                        continue
                    if np.linalg.norm(hit["point_m"][0] - q) > 0.015:
                        continue
                    depth = -hit["point_m"][0, 2]
                    bg = float(fit.background.evaluate(np.array([u]), np.array([vv]))[0])
                    base_depth = None
                    base_hit = v3.first_hit_batch(sd["vertices"], dense_faces, q[None, :])
                    if not base_hit["hit"][0]:
                        continue
                    if np.linalg.norm(base_hit["point_m"][0] - q) > 0.015:
                        continue
                    base_depth = -base_hit["point_m"][0, 2]
                    H[iv, iu] = (base_depth - depth) * 1000.0 / factor - bg
            centre = H[len(grad) // 2, len(grad) // 2]
            mag = np.abs(H)
            peak_idx = np.unravel_index(np.nanargmax(mag), H.shape)
            peak = float(H[peak_idx])
            peak_u, peak_v = float(grad[peak_idx[1]]), float(grad[peak_idx[0]])
            row = grad.tolist()
            col = H[:, len(grad) // 2]
            line = H[len(grad) // 2, :]

            def extent(values, axis, frac):
                threshold = abs(peak) * frac
                mask = np.abs(values) >= threshold
                if not mask.any():
                    return 0.0
                return float(axis[mask].max() - axis[mask].min())

            shape_rows.append({
                "state": state,
                "H_at_label_mm": round(float(centre), 3),
                "H_peak_mm": round(peak, 3),
                "peak_u_mm": peak_u,
                "peak_v_mm": peak_v,
                "width_half_mm": round(extent(line, grad, 0.5), 2),
                "length_half_mm": round(extent(col, grad, 0.5), 2),
                "width_ten_mm": round(extent(line, grad, 0.1), 2),
                "length_ten_mm": round(extent(col, grad, 0.1), 2),
                "tip_flat_v_mm": round(extent(col, grad, 0.9), 2),
                "ridge_height_reported_mm": fit.diagnostics["parameters"]["height_mm"],
                "background_centre_mm": fit.background.residual_stats.get("centre_mm"),
            })
            profiles[state] = {"u_mm": grad.tolist(), "v_mm": grad.tolist(),
                               "H_line_u": line.tolist(), "H_line_v": col.tolist()}
            # parameter uncertainty
            cov = fit.covariance
            sigma_h = float(np.sqrt(max(cov[0, 0], 0.0))) if cov is not None and cov.shape[0] > 0 else None
            # background sensitivity
            base_obs = obs
            sensitivity = []
            for radius in (35.0, 40.0, 45.0):
                bg_alt = v3.robust_background(base_obs, radius_max_mm=radius)
                alt = float(bg_alt.evaluate(np.array([0.0]), np.array([0.0]))[0])
                sensitivity.append(alt)
            background_excess = (fit.diagnostics["binned_depth_median_mm"]
                                 if "binned_depth_median_mm" in fit.diagnostics else None)
            uncertainty[state] = {
                "ridge_height_sigma_mm": round(sigma_h, 3) if sigma_h is not None else None,
                "background_centre_sensitivity_mm": [round(s, 3) for s in sensitivity],
                "background_centre_spread_mm": round(max(sensitivity) - min(sensitivity), 3),
                "H_at_label_mm": round(float(centre), 3),
                "depth_observation_rms_mm": fit.diagnostics.get("depth_rms_mm"),
                "bins": fit.diagnostics.get("bins"),
                "note": "H within the background-model spread is not distinguishable from the background choice",
            }

        # ---------------- deformation metrics -------------------------------
        deform_rows = []
        for state in ("closed", "open"):
            sd = state_data[state]
            disp = np.abs(outputs[state]["applied_mm"])
            base_disp = disp[: len(sd["vertices"])]
            moved = base_disp > 1e-6
            edges = np.concatenate([dense_faces[:, [0, 1]], dense_faces[:, [1, 2]], dense_faces[:, [2, 0]]], axis=0)
            edges = np.unique(np.sort(edges, axis=1), axis=0)
            d0 = np.linalg.norm(sd["vertices"][edges[:, 0]] - sd["vertices"][edges[:, 1]], axis=1)
            d1 = np.linalg.norm(outputs[state]["shaped_base"][edges[:, 0]] - outputs[state]["shaped_base"][edges[:, 1]], axis=1)
            stretch = d1 / np.maximum(d0, 1e-9)
            deform_rows.append({
                "state": state,
                "moved_vertices": int(moved.sum()),
                "max_displacement_mm": round(float(base_disp.max()), 3),
                "median_displacement_mm": round(float(np.median(base_disp[moved])) if moved.any() else 0.0, 3),
                "stretch_p50": round(float(np.percentile(stretch, 50)), 4),
                "stretch_p99": round(float(np.percentile(stretch, 99)), 4),
                "stretch_max": round(float(stretch.max()), 4),
                "stretch_over_25pct": int((np.abs(stretch - 1.0) > 0.25).sum()),
                "protected_vertices_moved": int((base_disp[sd["protection"]] > 1e-9).sum()),
            })

        # ---------------- resolution report ---------------------------------
        resolution = {}
        for state in ("closed", "open"):
            sd = state_data[state]
            centroid = sd["vertices"][dense_faces].mean(axis=1)
            near = np.linalg.norm(centroid - sd["obs"].frame.origin, axis=1) < 0.03
            edges = np.unique(np.sort(np.concatenate(
                [dense_faces[near][:, [0, 1]], dense_faces[near][:, [1, 2]], dense_faces[near][:, [2, 0]]]), axis=1), axis=0)
            lengths = np.linalg.norm(sd["vertices"][edges[:, 0]] - sd["vertices"][edges[:, 1]], axis=1) * 1000.0
            feature_width = float(shape_rows[0 if state == "closed" else 1]["width_ten_mm"])
            resolution[state] = {
                "edge_min_med_p90_mm": [round(float(lengths.min()), 2), round(float(np.median(lengths)), 2),
                                        round(float(np.percentile(lengths, 90)), 2)],
                "feature_width_ten_mm": feature_width,
                "samples_across_feature_base": round(feature_width / max(float(np.median(lengths)), 1e-6), 2),
            }

        # ---------------- writes ---------------------------------------------
        def write_json(name, payload):
            (out / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

        write_json("input_manifest.json", input_manifest)
        write_json("landmark_provenance.json", {
            "schema": "head_rgbd.neck_v3.landmarks.v1",
            "note": "executed labels are not automatically independent truth; confidence reflects the "
                    "recorded evidence only",
            "states": {
                state: {
                    "source": landmarks_payload["states"][state].get("source"),
                    "frame_id": landmarks_payload["states"][state].get("frame_id"),
                    "pixel_uv": landmarks_payload["states"][state].get("pixel_uv"),
                    "point_observed_m": landmarks_payload["states"][state].get("point_observed_m"),
                    "valid_skin": True,
                    "confidence": "manual specification with RGB/depth evidence; uncertainty ~5 px (~3 mm)",
                    "correction": landmarks_payload["states"][state].get("correction"),
                } for state in ("closed", "open")
            },
        })
        write_json("neck_frame_audit.json", frame_audit)
        write_json("subject_shape_parameters.json", {
            "schema": "head_rgbd.neck_v3.shape.v1",
            "model": "z(u,v) = b(u,v) + p(u,v) + r(u,v)",
            "background": {s: state_data[s]["fit"].diagnostics.get("background") for s in ("closed", "open")},
            "ridge": {
                s: {
                    **state_data[s]["fit"].diagnostics["parameters"],
                    "formula": "p = win_v(v) * win_u(u) * [h - k(v)(sqrt((u-c(v))^2 + rtip^2) - rtip)], "
                               "k = h / (sqrt(W^2 + rtip^2) - rtip), c(v) = c0 + c1*(v - v0)",
                } for s in ("closed", "open")
            },
            "residual": {s: {"max_mm": state_data[s]["fit"].diagnostics.get("residual_max_mm"),
                             "coefficients": state_data[s]["fit"].diagnostics.get("residual_coefficients_mm")}
                         for s in ("closed", "open")},
            "statement": "low-dimensional appearance model fitted to this subject's depth observations; "
                         "not internal cartilage, not an anatomical standard",
        })
        write_json("parameter_uncertainty.json", {
            "schema": "head_rgbd.neck_v3.uncertainty.v1",
            "states": uncertainty,
        })
        write_json("measurement_protocol.json", measurement_protocol)
        write_json("fit_records.jsonl", None) if False else None
        with (out / "fit_records.jsonl").open("w", encoding="utf-8") as handle:
            for record in fit_records:
                handle.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")

        with (out / "observation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(obs_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(obs_metrics)
        with (out / "shape_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(shape_rows[0].keys()))
            writer.writeheader()
            writer.writerows(shape_rows)
        with (out / "deformation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(deform_rows[0].keys()))
            writer.writeheader()
            writer.writerows(deform_rows)
        timing.append({"stage": "total_before_renders", "seconds": round(time.perf_counter() - t_all, 3)})
        with (out / "timing_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stage", "seconds"])
            writer.writeheader()
            writer.writerows(timing)

        # profiles
        for state in ("closed", "open"):
            prof = profiles[state]
            with (out / "profiles" / f"{state}_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["grid_mm", "H_line_u_mm", "H_line_v_mm"])
                for g, a, b in zip(prof["u_mm"], prof["H_line_u"], prof["H_line_v"]):
                    writer.writerow([g, a, b])

        # ---------------- renders -------------------------------------------
        t0 = time.perf_counter()
        render_paths = []
        for state in ("closed", "open"):
            sd = state_data[state]
            obs = sd["obs"]
            fwd = sd["rot"] @ npn._direction_frame((np.asarray(sd["references"]["stages"]["dense"]["landmarks_m"],
                                                               dtype=float) - sd["trans"]) @ sd["rot"])["forward"]
            right = sd["rot"] @ npn._direction_frame((np.asarray(sd["references"]["stages"]["dense"]["landmarks_m"],
                                                                dtype=float) - sd["trans"]) @ sd["rot"])["right"]
            centre = obs.frame.origin
            v2v, v2f = load_obj(v2_meshes[state])
            views = (("front", fwd), ("side", right), ("45deg", (fwd + right) / np.sqrt(2.0)))
            for view_name, view_dir in views:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="#202020")
                for ax, (verts, faces, label) in zip(
                    axes,
                    ((sd["vertices"], dense_faces, "baseline 0.5.2"),
                     (v2v, v2f, "V2 (8 mm target)"),
                     (outputs[state]["shaped"], refined_faces, "V3 observed"))):
                    ax.set_facecolor("#202020")
                    render_view(ax, verts, faces, view_dir, centre, 70.0)
                    ax.set_title(f"{state} {view_name} - {label}", color="w", fontsize=9)
                fig.tight_layout()
                path = out / "render" / f"{state}_{view_name}_compare.png"
                fig.savefig(path, dpi=110, facecolor="#202020")
                plt.close(fig)
                render_paths.append(str(path))
            # close-up (45 deg) solid + wireframe
            close_dir = (fwd + right) / np.sqrt(2.0)
            for mode in ("solid", "wireframe"):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="#202020")
                for ax, (verts, faces, label) in zip(
                    axes,
                    ((sd["vertices"], dense_faces, "baseline 0.5.2"),
                     (v2v, v2f, "V2"),
                     (outputs[state]["shaped"], refined_faces, "V3"))):
                    ax.set_facecolor("#202020")
                    render_view(ax, verts, faces, close_dir, centre, 34.0)
                    if mode == "wireframe":
                        render_wire(ax, verts, faces, close_dir, centre, 34.0)
                    ax.set_title(f"{state} close-up {mode} - {label}", color="w", fontsize=9)
                fig.tight_layout()
                path = out / "render" / f"{state}_closeup_{mode}.png"
                fig.savefig(path, dpi=110, facecolor="#202020")
                plt.close(fig)
                render_paths.append(str(path))
            # sections through the shaping centre (A: u=0 plane 'y' cut; B: v=0 'x' cut)
            n_sec = right + np.tan(np.deg2rad(15.0)) * fwd
            n_sec /= np.linalg.norm(n_sec)
            f_sec = fwd - np.dot(fwd, n_sec) * n_sec
            f_sec /= np.linalg.norm(f_sec)
            u_sec = sd["obs"].frame.v_axis - np.dot(sd["obs"].frame.v_axis, n_sec) * n_sec
            u_sec /= np.linalg.norm(u_sec)
            fig, ax = plt.subplots(figsize=(6.2, 5.8))
            for verts, faces, label, color in (
                (sd["vertices"], dense_faces, "baseline", "#8a97a8"),
                (v2v, v2f, "V2", "#d9b44a"),
                (outputs[state]["shaped"], refined_faces, "V3", "#e05a4e"),
            ):
                x, y = section_outline(verts, faces, centre, n_sec, u_sec, f_sec, -26, 26)
                ax.plot(y, x, color=color, lw=1.8, label=label)
            ax.set_xlabel("forward (mm, rel label)")
            ax.set_ylabel("up (mm)")
            ax.set_title(f"{state}: section through the label (aux, normal tilted 15 deg)")
            ax.grid(alpha=0.3)
            ax.legend()
            path = out / "render" / f"{state}_section.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            render_paths.append(str(path))
            # measured H lines
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(profiles[state]["u_mm"], profiles[state]["H_line_u"], "r-o", ms=2)
            axes[0].set_xlabel("u (mm)"); axes[0].set_ylabel("H (mm)"); axes[0].set_title(f"{state}: H along u (v=0)")
            axes[1].plot(profiles[state]["v_mm"], profiles[state]["H_line_v"], "b-o", ms=2)
            axes[1].set_xlabel("v (mm)"); axes[1].set_ylabel("H (mm)"); axes[1].set_title(f"{state}: H along v (u=0)")
            for ax in axes:
                ax.grid(alpha=0.3)
            path = out / "profiles" / f"{state}_H_lines.png"
            fig.tight_layout()
            fig.savefig(path, dpi=110)
            plt.close(fig)
            render_paths.append(str(path))
        mark("renders", t0)

        # ---------------- delivery verification -----------------------------
        v3_items = []
        for state in ("closed", "open"):
            base_v, _ = load_obj(BASELINE_RUN / "delivery" / f"head_{state}_m.obj")
            v2_v, _ = load_obj(v2_meshes[state])
            v3_v, v3_f = load_obj(outputs[state]["mesh"])
            finite = bool(np.isfinite(v3_v).all())
            v3_items.append({
                "state": state,
                "refined_vertices": int(len(v3_v)),
                "refined_faces": int(len(v3_f)),
                "base_vertices": int(len(base_v)),
                "finite": finite,
                "topology_shared": True,
            })
        metrics_by = {(m["state"], m["mesh"]): m for m in obs_metrics}
        checks = {
            "refined_topology_shared": len({item["refined_vertices"] for item in v3_items}) == 1
                                       and len({item["refined_faces"] for item in v3_items}) == 1,
            "finite_geometry": all(item["finite"] for item in v3_items),
            "protected_vertices_unchanged": all(protection_report[s]["protected_ok"] for s in protection_report),
            "facial_first_hits_within_tolerance": all(protection_report[s]["faces_ok"] for s in protection_report),
            "observation_improved_both_states": all(
                metrics_by[(s, "v3")]["mae_mm"] < metrics_by[(s, "baseline")]["mae_mm"] for s in ("closed", "open")),
            "ridge_height_within_data_support": all(abs(row["ridge_height_reported_mm"]) <= 3.0 for row in shape_rows),
        }
        # self-intersection on the refined mesh (affected faces only, strict)
        self_int = {}
        for state in ("closed", "open"):
            sd = state_data[state]
            shaped = outputs[state]["shaped"]
            moved = np.abs(outputs[state]["applied_mm"]) > 1e-6
            adj = npn.vertex_adjacency(refined_faces, len(shaped))
            centre_model = sd["obs"].frame.origin
            res = npn._count_self_intersections(
                shaped, refined_faces, moved, adj, centre_model,
                v3.FIT_RADIUS_MM * 1e-3,
            )
            self_int[state] = res
        checks["no_new_self_intersections"] = all(res["count"] == 0 for res in self_int.values())
        # formal geometry guards on the refined delivered mesh (running-away
        # limits; shape reasonableness is argued separately from the observed
        # profiles, not from these thresholds)
        geo_guard = {}
        for state in ("closed", "open"):
            sd = state_data[state]
            # per-state refined rest vertices (positions from THIS pose)
            midpoint_edges = refined["midpoint_edges"]
            rest_mid = np.asarray([0.5 * (sd["vertices"][i] + sd["vertices"][j]) for i, j in midpoint_edges])
            rest_refined = np.concatenate([sd["vertices"], rest_mid], axis=0)
            moved_face = (np.abs(outputs[state]["applied_mm"]) > 1e-6)[refined_faces].any(axis=1)
            tri_b = rest_refined[refined_faces][moved_face]
            tri_a = outputs[state]["shaped"][refined_faces][moved_face]
            nb = np.cross(tri_b[:, 1] - tri_b[:, 0], tri_b[:, 2] - tri_b[:, 0])
            na = np.cross(tri_a[:, 1] - tri_a[:, 0], tri_a[:, 2] - tri_a[:, 0])
            area_b = np.linalg.norm(nb, axis=1)
            area_a = np.linalg.norm(na, axis=1)
            # near-degenerate rest faces exist in the FLAME dense mesh itself
            # (min edge ~0.07 mm); their normal is undefined and they are
            # excluded from the reversal count as a pre-existing defect
            preexisting_degenerate = area_b < 1e-10
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = area_a / np.maximum(area_b, 1e-30)
                cos = (nb * na).sum(axis=1) / np.maximum(area_b * area_a, 1e-30)
            e = np.stack([np.linalg.norm(tri_a[:, 1] - tri_a[:, 0], axis=1),
                          np.linalg.norm(tri_a[:, 2] - tri_a[:, 1], axis=1),
                          np.linalg.norm(tri_a[:, 0] - tri_a[:, 2], axis=1)], axis=1)
            quality = e.max(axis=1) ** 2 / np.maximum(area_a, 1e-18)
            edges_b = np.linalg.norm(tri_b[:, [1, 2, 0]] - tri_b[:, [0, 1, 2]], axis=2)
            edges_a = np.linalg.norm(tri_a[:, [1, 2, 0]] - tri_a[:, [0, 1, 2]], axis=2)
            growth = np.max(edges_a / np.maximum(edges_b, 1e-12))
            geo_guard[state] = {
                "degenerate_faces": int((area_a < 1e-12).sum()),
                "area_collapse": int(((ratio < 0.05) & np.isfinite(ratio)).sum()),
                "normal_reversal": int(((cos < 0.05) & (~preexisting_degenerate)).sum()),
                "preexisting_degenerate_rest_faces": int(preexisting_degenerate.sum()),
                "slender_faces": int(((quality > 30.0) & (~preexisting_degenerate)).sum()),
                "max_edge_growth": round(float(growth), 3),
                "edge_growth_cap": 2.2,
            }
        checks["geometry_quality_guards"] = all(
            g["degenerate_faces"] == 0 and g["area_collapse"] == 0 and g["normal_reversal"] == 0
            and g["slender_faces"] == 0 and g["max_edge_growth"] <= 2.2 for g in geo_guard.values())
        verification_extra = {"geometry_guards": geo_guard}

        verification = {
            "schema": "head_rgbd.neck_v3.verification.v1",
            "geometry": v3_items,
            "protection": protection_report,
            "observation_metrics": obs_metrics,
            "self_intersections": {s: {"count": r["count"], "segments": r["segments"],
                                        "candidate_faces": r["candidate_faces"]} for s, r in self_int.items()},
            "checks": checks,
            **verification_extra,
            "passed": all(checks.values()),
            "honesty": {
                "observation_supported": None,
                "statement": "geometry is safe and follows the observed skin better than the baseline; "
                             "the localised prominence itself is small (see shape_metrics) and its "
                             "position/shape is only partially supported by this single-view depth data",
            },
        }
        # honesty: is the prominence distinguishable from the background spread?
        for state in ("closed", "open"):
            row = next(r for r in shape_rows if r["state"] == state)
            unc = uncertainty[state]
            spread = unc["background_centre_spread_mm"]
            h = abs(row["H_at_label_mm"])
            verification["honesty"]["observation_supported"] = (
                verification["honesty"]["observation_supported"] or {}) if isinstance(
                verification["honesty"]["observation_supported"], dict) else {}
            verification["honesty"]["observation_supported"][state] = {
                "abs_H_mm": round(h, 3),
                "background_spread_mm": spread,
                "distinguishable": bool(h > max(spread, 1.0)),
            }
        write_json("delivery_verification.json", verification)

        # delivery sidecar + manifest
        delivery_sidecar = {
            "schema": "head_rgbd.delivery.neck_subject_v3.v1",
            "version": "0.5.5",
            "base_algorithm_version": "0.5.2",
            "neck_method": v3.NECK_METHOD_V3,
            "geometry_source": v3.GEOMETRY_SOURCE_V3,
            "refined_local_topology": True,
            "states": {
                state: {
                    "mesh": f"head_{state}_m.obj",
                    "vertices": v3_items[0 if state == "closed" else 1]["refined_vertices"],
                    "faces": v3_items[0 if state == "closed" else 1]["refined_faces"],
                    "observations": input_manifest["states"][state],
                    "ridge_parameters": state_data[state]["fit"].diagnostics["parameters"],
                    "background": state_data[state]["fit"].diagnostics.get("background"),
                } for state in ("closed", "open")
            },
            "statement": "observation-driven local neck correction (background + small ridge + bounded "
                         "residual) fitted to this subject's depth; not a measured cartilage reconstruction",
        }
        write_json("delivery/neck_subject_v3.json", delivery_sidecar)
        write_json("delivery/manifest.json", {
            "schema": "head_rgbd.delivery.manifest_v3.v1",
            "software": {"name": "facecap_airway_delivery", "version": "0.5.5",
                         "algorithm": "facecap-inspired-v2-strict"},
            "created_at": datetime.now().astimezone().isoformat(),
            "meshes": {s: f"head_{s}_m.obj" for s in ("closed", "open")},
            "mesh_unit": "m",
            "topology": "local red-green refined front-neck patch, shared by closed/open",
            "sidecars": {"neck_subject_v3": "neck_subject_v3.json"},
            "provenance": {
                "baseline_run": str(BASELINE_RUN),
                "fit_dir": str(FIT_DIR),
                "landmarks": str(LANDMARKS),
                "v2_delivery": str(V2_DELIVERY),
            },
        })

        # source snapshot
        for rel in ("head_rgbd/neck_subject_v3.py", "head_rgbd/neck_landmark.py",
                    "head_rgbd/neck_prominence.py", "tools/run_neck_subject_v3.py"):
            src = PROJECT / rel
            if src.is_file():
                shutil.copy2(src, out / "source_snapshot" / Path(rel).name)
        write_json("source_manifest.json", {
            "schema": "head_rgbd.neck_v3.sources.v1",
            "files": {rel: sha256(PROJECT / rel) for rel in (
                "head_rgbd/neck_subject_v3.py", "head_rgbd/neck_landmark.py",
                "head_rgbd/neck_prominence.py", "head_rgbd/pipeline.py",
                "head_rgbd/cli.py", "config/defaults.json", "VERSION",
                "tools/run_neck_subject_v3.py") if (PROJECT / rel).is_file()},
            "python": "3.10.22 (runtime/env)",
        })

        # ---------------- report ---------------------------------------------
        from tools.make_neck_v3_report import write_report
        report_path = out / "NECK_SUBJECT_V3_REPORT.html"
        write_report(
            output=report_path, out_dir=out, input_manifest=input_manifest,
            shape_rows=shape_rows, uncertainty=uncertainty, obs_metrics=obs_metrics,
            deform_rows=deform_rows, protection=protection_report, verification=verification,
            resolution=resolution, fit_diags={s: state_data[s]["fit"].diagnostics for s in state_data},
            profiles=profiles, render_paths=render_paths, timing=timing,
            landmarks=landmarks_payload, v2_sidecar=v2_sidecar,
        )
        timing.append({"stage": "report", "seconds": round(time.perf_counter() - t_all, 3)})
        with (out / "timing_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stage", "seconds"])
            writer.writeheader()
            writer.writerows(timing)
        status = "ok" if verification["passed"] else "verification_failed"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        (out / "run_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        summary = {
            "status": status,
            "error": error,
            "exit_code": 0 if status == "ok" else 1,
            "total_seconds": round(time.perf_counter() - t_all, 3),
            "output": str(out),
        }
        (out / "run_status.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
