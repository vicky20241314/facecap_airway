"""One shared neutral identity fitted to every selected view at once.

What this stage does, and what it deliberately does not do:

* one FLAME shape vector is shared by the frontal anchor and every oblique
  keyframe, so the silhouette of a 40-degree view can correct the frontal width;
* each keyframe keeps its own rigid pose, expression offset and jaw offset, and
  those offsets are bounded around the values the frontal anchor produced;
* every observation enters with its own reliability weight (incidence, local
  depth-plane consistency, measured first-hit visibility) and every angle group
  gets the same total vote, so adding frames at one angle cannot buy influence;
* there is no per-view vertex ownership. A vertex is updated by whichever views
  really see it, whatever their label, and it is only removed when it is
  genuinely unobserved, self-occluded or anatomically excluded;
* nothing here is non-rigid in the per-vertex sense. That is the next stage, and
  it happens once.

The output ``parameters.npz`` and ``report.json`` are written so that
:mod:`head_rgbd.motion_v2` can consume them unchanged as the single pose
correction step.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch
from smplx.lbs import batch_rodrigues

from head_rgbd.audit import sha
from head_rgbd.experiment.turn_tracking import template_landmarks
from head_rgbd.fit_c import landmark_embedding
from head_rgbd.fit_d import load_npz
from head_rgbd.geometry_c import landmark_validity, ray_surface, sample_depth
from head_rgbd.prepare import rigid
from head_rgbd.reference_contract import inner_lip_gap_mm, observed_landmarks_m
from head_rgbd.reporting import write_json
from head_rgbd.surface_v2 import local_depth_planes
from core.fit_options import FitOptions
from core.integrated_fitting import make_flame


def run(run_dir: Path, config, device: str = "cuda") -> dict:
    run_dir = Path(run_dir).resolve()
    prepared = run_dir / "prepared"
    sequence_path = prepared / "sequence.json"
    anchor_sequence = prepared / "anchor_sequence.json"
    anchor_fit = run_dir / "front_anchor"
    output = run_dir / "joint_shape"
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    info = json.loads(sequence_path.read_text(encoding="utf-8"))
    _check_units(info, sequence_path)
    rows = [dict(row) for row in info["frames"] if row["status"] == "candidate"]
    for row in rows:
        row.setdefault("degraded", [])
    if not rows:
        raise ValueError(f"{sequence_path} has no candidate keyframe")
    if any(row.get("state") != "closed" for row in rows):
        raise ValueError("the shared identity fit accepts closed-mouth keyframes only")

    anchor_report = json.loads((anchor_fit / "report.json").read_text(encoding="utf-8"))
    if anchor_report["sequence_sha256"] != sha(anchor_sequence):
        raise ValueError("the frontal anchor fit does not bind prepared/anchor_sequence.json")
    saved = load_npz(anchor_fit / "parameters.npz")
    shape_components = int(saved["shape"].shape[1])
    expression_components = int(saved["expression"].shape[1])
    if shape_components != config.flame.shape_components:
        raise ValueError(f"anchor shape has {shape_components} components; config says {config.flame.shape_components}")
    if expression_components != config.flame.expression_components:
        raise ValueError(f"anchor expression has {expression_components} components; "
                         f"config says {config.flame.expression_components}")

    data = [load_npz(prepared / row["file"]) for row in rows]
    faces = np.asarray(saved["faces"])
    vertex_count = int(saved["posed_vertices"].shape[1])
    n = len(rows)
    joint = config.joint_shape
    keyframes = config.keyframes

    torch.set_num_threads(4)
    torch.manual_seed(int(config.experiment.seed))
    rng = np.random.default_rng(int(config.experiment.seed))
    dev = torch.device(device)
    tensor = lambda value: torch.as_tensor(value, device=dev, dtype=torch.float32)

    flame = make_flame(FitOptions(shape_components=shape_components,
                                  expression_components=expression_components), device, batch=n)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    anchor_index = {row["id"]: i for i, row in enumerate(anchor_report["frames"])}
    anchored = [(i, anchor_index[row["id"]]) for i, row in enumerate(rows) if row["id"] in anchor_index]
    if not anchored:
        raise ValueError("no joint keyframe is present in the frontal anchor fit")

    template = template_landmarks(config.flame)
    initial = np.tile(np.eye(3), (n, 1, 1))
    translations = np.zeros((n, 3))
    pose_bootstrap = []
    for i, frame in enumerate(data):
        valid = np.asarray(frame["valid"]).copy()
        valid[:27] = False
        valid[48:] = False
        entry = {"frame_id": rows[i]["id"], "valid_upper_landmarks": int(valid.sum())}
        if int(valid.sum()) < 8:
            translations[i] = np.array([0.0, 0.0, -0.5])
            entry["pose_source"] = "fallback_no_upper_face_depth"
            pose_bootstrap.append(entry)
            continue
        try:
            rotation, translation = rigid(template[valid], np.asarray(frame["xyz"])[valid])
            initial[i] = rotation
            translations[i] = translation
            entry["pose_source"] = "landmark_rigid_alignment"
            entry["median_residual_mm"] = float(np.median(np.linalg.norm(
                template[valid] @ rotation.T + translation - np.asarray(frame["xyz"])[valid], axis=1)) * 1000.)
        except (ValueError, np.linalg.LinAlgError):
            translations[i] = np.array([0.0, 0.0, -0.5])
            entry["pose_source"] = "fallback_degenerate_alignment"
        pose_bootstrap.append(entry)
    for i, j in anchored:
        initial[i] = np.asarray(saved["R"][j], dtype=np.float64)
        translations[i] = np.asarray(saved["t"][j], dtype=np.float64)
        pose_bootstrap[i]["pose_source"] = "frontal_anchor"

    anchor_rows = [j for _, j in anchored]
    anchor_expression = np.asarray(saved["expression"])[anchor_rows]
    anchor_jaw = np.asarray(saved["jaw"])[anchor_rows]
    expression_reference = anchor_expression.mean(axis=0)
    jaw_reference = anchor_jaw.mean(axis=0)
    expression_init = np.tile(expression_reference, (n, 1))
    jaw_init = np.tile(jaw_reference, (n, 1))
    for i, j in anchored:
        expression_init[i] = np.asarray(saved["expression"][j])
        jaw_init[i] = np.asarray(saved["jaw"][j])

    shape = torch.nn.Parameter(tensor(np.asarray(saved["shape"], dtype=np.float64)))
    expression = torch.nn.Parameter(tensor(expression_init))
    jaw = torch.nn.Parameter(tensor(jaw_init))
    rotation_delta = torch.nn.Parameter(torch.zeros(n, 3, device=dev))
    translation_delta = torch.nn.Parameter(torch.zeros(n, 3, device=dev))
    base_rotation = tensor(initial)
    base_translation = tensor(translations)
    anchor_shape = tensor(np.asarray(saved["shape"], dtype=np.float64))
    jaw_floor = tensor(jaw_reference - joint.jaw_delta_bound_rad)
    jaw_ceiling = tensor(jaw_reference + joint.jaw_delta_bound_rad)

    target3, target2 = [], []
    base_mask3, base_mask2, base_inner_lip_mask = [], [], []
    identity_points = []
    identity_uv = []
    identity_planes = []
    for i, frame in enumerate(data):
        valid3d, rejected = landmark_validity(frame)
        uv2d = np.asarray(frame["uv"])
        valid2d = np.isfinite(uv2d).all(axis=1)
        if "landmark2d_valid" in frame:
            valid2d = valid2d & np.asarray(frame["landmark2d_valid"], dtype=bool)
        stable, inner, mask2d = _landmark_masks(valid3d, valid2d, dev)
        target3.append(tensor(np.nan_to_num(np.asarray(frame["xyz"]))))
        target2.append(tensor(np.nan_to_num(uv2d)))
        base_mask3.append(stable)
        base_inner_lip_mask.append(inner)
        base_mask2.append(mask2d)
        rows[i]["landmark_depth_invalid"] = np.flatnonzero(~np.asarray(frame["valid"])).tolist()
        rows[i]["landmark_depth_outlier_rejected"] = rejected
        points, uv, _ = sample_depth(frame, joint.points_per_frame, rng,
                                      exclude_mouth=True, stride=3)
        identity_points.append(points)
        identity_uv.append(uv)
        identity_planes.append(local_depth_planes(frame, uv))

    # Visibility is recomputed from immutable per-frame validity at every
    # correspondence refresh. Repeated refresh calls must not accumulate a
    # permanent intersection from transient self-occlusions.
    mask3 = [value.clone() for value in base_mask3]
    inner_lip_mask = [value.clone() for value in base_inner_lip_mask]
    mask2 = [value.clone() for value in base_mask2]

    frame_weight, group_table = _frame_weights(rows)
    for i, row in enumerate(rows):
        row["frame_weight"] = round(float(frame_weight[i]), 6)
    weights = tensor(np.asarray(frame_weight, dtype=np.float64))

    depth_gate = float(joint.max_correspondence_depth_mm) * 0.001
    plane_gate = float(joint.max_correspondence_plane_mm) * 0.001

    def forward():
        rotation = batch_rodrigues(rotation_delta) @ base_rotation
        raw, landmarks = flame(
            shape_params=shape.expand(n, -1), expression_params=expression,
            pose_params=torch.cat((torch.zeros(n, 3, device=dev), jaw), 1),
            transl=torch.zeros(n, 3, device=dev), view_rotation=rotation)
        vertices = torch.einsum("bij,bvj->bvi", rotation, raw) + base_translation[:, None] + translation_delta[:, None]
        points = torch.einsum("bij,bkj->bki", rotation, landmarks) + base_translation[:, None] + translation_delta[:, None]
        return raw, vertices, points, rotation

    def robust(value):
        return torch.nn.functional.smooth_l1_loss(value, torch.zeros_like(value), beta=0.3)

    matches: dict[int, dict] = {}
    correspondence_log: list[dict] = []

    def refresh(step: int, tag: str):
        matches.clear()
        with torch.no_grad():
            _, vertices, landmarks, _ = forward()
            v_np = vertices.detach().cpu().numpy()
            l_np = landmarks.detach().cpu().numpy()
        for i in range(n):
            points = identity_points[i]
            if not len(points):
                correspondence_log.append({"step": step, "tag": tag, "frame_id": rows[i]["id"],
                                           "accepted": 0, "reason": "no_identity_depth_samples"})
                continue
            correspondence, stats = ray_surface(v_np[i], faces, points, depth_gate, plane_gate)
            self_hit, self_stats = ray_surface(v_np[i], faces, l_np[i], 0.002)
            reliable = np.zeros(68, dtype=bool)
            if self_hit is not None:
                reliable[self_hit["ray"]] = True
            discrepancy = np.linalg.norm(l_np[i] - np.nan_to_num(np.asarray(data[i]["xyz"])), axis=1)
            reliable &= discrepancy < float(joint.landmark_visibility_mm)
            rows[i]["landmark_self_visibility_rejected"] = np.flatnonzero(~reliable).tolist()
            # Refresh landmark masks on every correspondence cycle.  A cycle
            # with too few depth matches must not keep the previous cycle's
            # visibility state and silently credit a now-invalid track.
            visible = torch.as_tensor(reliable, device=dev, dtype=torch.bool)
            mask3[i] = _refresh_visibility(base_mask3[i], visible)
            inner_lip_mask[i] = _refresh_visibility(base_inner_lip_mask[i], visible)
            mask2[i] = _refresh_visibility(base_mask2[i], visible)
            if correspondence is None or stats["accepted"] < joint.min_correspondences:
                correspondence_log.append({"step": step, "tag": tag, "frame_id": rows[i]["id"],
                                           "reason": "insufficient_reliable_correspondences", **stats})
                continue
            observed_normal, scatter_mm, plane_good = identity_planes[i]
            rays = correspondence["ray"]
            agreement = (correspondence["normal"] * observed_normal[rays]).sum(1)
            local_keep = plane_good[rays] & (agreement > 0.5)
            if not local_keep.any():
                correspondence_log.append({
                    "step": step, "tag": tag, "frame_id": rows[i]["id"],
                    "reason": "insufficient_local_plane_correspondences",
                    "accepted": 0, "local_plane_rejected": int(len(rays)),
                })
                continue
            correspondence = {key: np.asarray(value)[local_keep] for key, value in correspondence.items()}
            agreement = agreement[local_keep]
            scatter_mm = scatter_mm[rays[local_keep]]
            stats = {**stats, "accepted": int(local_keep.sum()),
                     "local_plane_rejected": int((~local_keep).sum())}
            weight = _reliability_weights(correspondence["incidence"], scatter_mm, agreement)
            positive = weight > 0
            if int(positive.sum()) < joint.min_correspondences:
                correspondence_log.append({"step": step, "tag": tag, "frame_id": rows[i]["id"],
                                           "reason": "insufficient_positive_reliability_weights", **stats})
                continue
            # Keep only measured correspondences with a positive reliability.
            correspondence = {key: np.asarray(value)[positive] for key, value in correspondence.items()}
            stats = {**stats, "accepted": int(positive.sum()),
                     "zero_weight_rejected": int((~positive).sum())}
            weight = weight[positive]
            matches[i] = {
                "face": torch.as_tensor(faces[correspondence["face"]], device=dev),
                "bary": tensor(correspondence["bary"]),
                "normal": tensor(correspondence["normal"]),
                "target": tensor(correspondence["target"]),
                "weight": tensor(weight),
            }
            residual = np.abs(correspondence["plane"]) * 1000.
            rows[i]["depth_correspondences"] = int(stats["accepted"])
            rows[i]["depth_residual_median_mm"] = float(np.median(residual))
            rows[i]["depth_residual_p90_mm"] = float(np.percentile(residual, 90))
            rows[i]["reliability_weight_median"] = float(np.median(weight))
            rows[i]["reliability_weight_p90"] = float(np.percentile(weight, 90))
            correspondence_log.append({"step": step, "tag": tag, "frame_id": rows[i]["id"], **stats})

    def landmark_loss(i: int, points, vertices):
        valid3d = mask3[i]
        value = vertices.sum() * 0
        if bool(valid3d.any()):
            value = value + robust((points[valid3d] - target3[i][valid3d]) / 0.01)
        inner = inner_lip_mask[i]
        if bool(inner.any()):
            value = value + 0.25 * robust((points[inner] - target3[i][inner]) / 0.01)
        fx, fy, cx, cy = np.asarray(data[i]["intrinsics"], dtype=np.float64)
        z = (-points[:, 2]).clamp(min=0.05)
        projected = torch.stack((fx * points[:, 0] / z + cx, -fy * points[:, 1] / z + cy), 1)
        valid2d = mask2[i]
        if bool(valid2d.any()):
            value = value + 0.15 * robust((projected[valid2d] - target2[i][valid2d]) / 10.)
        if rows[i]["view"] == "front":
            mouth = valid2d.clone()
            mouth[:48] = False
            if bool(mouth.any()):
                value = value + 0.35 * robust((projected[mouth] - target2[i][mouth]) / 10.)
        if rows[i]["state"] == "closed":
            gaps = torch.linalg.norm(points[[61, 62, 63]] - points[[67, 66, 65]], dim=1)
            value = value + 0.15 * (torch.relu(gaps - 0.002) / 0.003).square().mean()
        return value

    history: list[dict] = []
    total_steps = int(joint.pose_steps + joint.joint_steps)
    step_counter = 0

    # Stage A: pose only. The identity stays exactly at the frontal anchor value.
    if joint.pose_steps:
        optimiser = torch.optim.Adam([{"params": [rotation_delta], "lr": 0.002},
                                      {"params": [translation_delta], "lr": 0.0005}])
        for step in range(int(joint.pose_steps)):
            optimiser.zero_grad()
            _, vertices, points, _ = forward()
            loss = vertices.sum() * 0
            for i in range(n):
                loss = loss + weights[i] * landmark_loss(i, points[i], vertices[i])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite objective in the pose stage")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([rotation_delta, translation_delta], 10.)
            optimiser.step()
            step_counter += 1
            if step % 20 == 0 or step == int(joint.pose_steps) - 1:
                record = {"stage": "pose", "step": step, "objective": float(loss.detach())}
                history.append(record)
                print(json.dumps(record), flush=True)

    # Stage B: shared identity plus bounded per-frame expression/jaw offsets.
    optimiser = torch.optim.Adam([
        {"params": [shape], "lr": 0.01},
        {"params": [expression], "lr": 0.02},
        {"params": [jaw], "lr": 0.002},
        {"params": [rotation_delta], "lr": 0.001},
        {"params": [translation_delta], "lr": 0.0003},
    ])
    jaw_center = tensor(jaw_reference)
    for step in range(int(joint.joint_steps)):
        if step % int(joint.correspondence_interval) == 0:
            refresh(step_counter, "joint")
        optimiser.zero_grad()
        _, vertices, points, _ = forward()
        loss = vertices.sum() * 0
        for i in range(n):
            frame = weights[i] * landmark_loss(i, points[i], vertices[i])
            match = matches.get(i)
            if match is not None:
                surface = (vertices[i][match["face"]] * match["bary"][:, :, None]).sum(1)
                residual = ((surface - match["target"]) * match["normal"]).sum(1)
                frame = frame + weights[i] * _weighted_robust(residual / 0.01, match["weight"])
            loss = loss + frame
        loss = loss + 0.002 * shape.square().mean()
        loss = loss + float(joint.expression_prior_weight) * expression.square().mean()
        loss = loss + float(joint.jaw_delta_prior_weight) * (jaw - jaw_center).square().mean()
        loss = loss + float(joint.anchor_shape_prior) * (shape - anchor_shape).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite objective in the joint stage")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([shape, expression, jaw, rotation_delta, translation_delta], 10.)
        optimiser.step()
        with torch.no_grad():
            bound = float(joint.expression_bound)
            expression.clamp_(-bound, bound)
            jaw.copy_(torch.maximum(torch.minimum(jaw, jaw_ceiling), jaw_floor))
            shape.clamp_(-4., 4.)
        step_counter += 1
        if step % 20 == 0 or step == int(joint.joint_steps) - 1:
            record = {"stage": "joint", "step": step, "objective": float(loss.detach()),
                      "correspondence_frames": len(matches),
                      "shape_absmax": float(shape.detach().abs().max()),
                      "expression_absmax": float(expression.detach().abs().max())}
            history.append(record)
            print(json.dumps(record), flush=True)

    with torch.no_grad():
        raw, posed, posed_landmarks, rotation = forward()
        landmark_faces, landmark_bary = landmark_embedding(flame, raw, jaw, rotation)
    raw_np = raw.cpu().numpy()
    vertices_np = posed.cpu().numpy()
    landmarks_np = posed_landmarks.cpu().numpy()
    rotation_np = rotation.cpu().numpy()
    landmark_faces = landmark_faces.cpu().numpy()
    landmark_bary = landmark_bary.cpu().numpy()
    shape_np = shape.detach().cpu().numpy()
    expression_np = expression.detach().cpu().numpy()
    jaw_np = jaw.detach().cpu().numpy()
    translation_np = (base_translation + translation_delta).detach().cpu().numpy()

    reference_i = anchored[0][0]
    mouth_excluded = _mouth_region(raw_np[reference_i], faces, landmark_faces[reference_i],
                                   landmark_bary[reference_i])
    deviation = np.zeros(vertex_count)
    for i, row in enumerate(rows):
        row["fitted_yaw_deg"] = _yaw(rotation_np[i])
        row["angle_group"] = row.get("angle_group") or f"yaw{int(round(row.get('guided_yaw_deg') or 0.)):+d}"
        row["expression_inner_lip_gap_mm"] = _model_gap(vertices_np[i], faces,
                                                        landmark_faces[i], landmark_bary[i])
        row["observed_inner_lip_gap_mm"] = inner_lip_gap_mm(data[i]["xyz"], data[i]["valid"])

    np.savez_compressed(
        output / "parameters.npz",
        shape=shape_np, expression=expression_np, jaw=jaw_np, R=rotation_np, t=translation_np,
        raw_vertices=raw_np, posed_vertices=vertices_np, refined_vertices=vertices_np,
        before_side_detail_vertices=vertices_np, faces=faces, landmarks=landmarks_np,
        deviation_m=deviation, effective_deviation_m=np.tile(deviation, (n, 1)),
        support_frames=np.zeros(vertex_count, dtype=np.int64),
        selected=np.arange(n), identity_selected=np.arange(n),
        landmark_faces=landmark_faces, landmark_barycentric=landmark_bary,
        mouth_identity_excluded=mouth_excluded,
        front_protected_vertices=np.zeros(vertex_count, dtype=bool),
        anchor_frame_indices=np.asarray([i for i, _ in anchored], dtype=np.int64),
        angle_group_weights=np.asarray([record["total_weight"] for record in group_table]),
    )

    reference = {
        "frame_id": rows[reference_i]["id"],
        "source": rows[reference_i].get("source"),
        "source_frame": rows[reference_i].get("source_frame"),
        "timestamp_s": rows[reference_i].get("timestamp_s"),
        "coordinate_system": "reference camera X_right_Y_up_backward; metres",
        "expression": expression_np[reference_i],
        "jaw_axis_angle_rad": jaw_np[reference_i],
        "observed_landmarks_m": observed_landmarks_m(data[reference_i]["xyz"], data[reference_i]["valid"]),
        "observed_valid": np.asarray(data[reference_i]["valid"]),
        "observed_inner_lip_gap_mm": inner_lip_gap_mm(data[reference_i]["xyz"], data[reference_i]["valid"]),
        "landmark_stage": "joint_shared_identity",
        "landmark_indices": "FLAME dynamic contour + static 68; zero based",
        "observation_is_manual_truth": False,
    }
    write_json(output / "reference_geometry.json",
               {"schema": "head_rgbd.turn.reference.v1", "references": {"closed": reference}})

    shape_delta = float(np.abs(shape_np - np.asarray(saved["shape"])).max())
    pose_delta = _pose_delta(rotation_np, translation_np, initial, translations, anchored)
    elapsed = time.perf_counter() - started
    report = {
        "schema": "head_rgbd.turn.shape.v1",
        "algorithm": "one shared FLAME identity over all closed keyframes, per-frame rigid pose, bounded closed expression",
        "research_only": True,
        "device": device,
        "sequence": str(sequence_path),
        "sequence_sha256": sha(sequence_path),
        "anchor": {
            "path": str(anchor_fit),
            "sequence": str(anchor_sequence),
            "sequence_sha256": anchor_report["sequence_sha256"],
            "parameters_sha256": sha(anchor_fit / "parameters.npz"),
            "role": "initialisation of shape, scale, frontal form and the frontal pose only; not a fixed surface",
        },
        "positive_landmark_policy": "real 3D depth at the detected pixels; missing depth is absent, never zero",
        "steps": {"pose": int(joint.pose_steps), "joint": int(joint.joint_steps), "total": total_steps},
        "angle_weighting": {
            "rule": ("each angle group gets 1/G of the total data weight; inside a group its bounded "
                     "keyframe/support observations are split by tracking quality"),
            "groups": group_table,
            "frame_count": n,
            "keyframe_count": sum(row.get("observation_role", "keyframe") == "keyframe" for row in rows),
            "support_count": sum(row.get("observation_role") == "support" for row in rows),
            "note": ("a group with several frames does not gain total influence, and a low-quality view "
                     "is never amplified by normalisation"),
        },
        "multi_frame_aggregation": {
            "policy": ("same-angle frames are registered independently and enter jointly as robust "
                       "point-to-plane constraints; raw depth images are never averaged"),
            "reliability": ("incidence squared, local depth-plane agreement/scatter, finite residuals, "
                            "and a finite positive-weight denominator with zero-weight fallback"),
            "visibility_refresh": ("self-visibility is recomputed from immutable base masks; transient "
                                   "occlusion no longer permanently removes a landmark"),
        },
        "shape_delta_from_anchor_absmax": shape_delta,
        "expression_bound": {"limit": float(joint.expression_bound),
                             "frames_at_bound": int((np.abs(expression_np) >= float(joint.expression_bound) - 1e-6).sum())},
        "jaw_delta_bound_rad": float(joint.jaw_delta_bound_rad),
        "pose_bootstrap": pose_bootstrap,
        "pose_change_from_initial": pose_delta,
        "topology": {"vertices": vertex_count, "faces": int(len(faces))},
        "frames": rows,
        "references": {"closed": {"frame_id": reference["frame_id"]}},
        "reference_geometry": reference,
        "correspondence_log": correspondence_log,
        "history": history,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else 0.,
        "elapsed_s": elapsed,
        "statuses": {"implementation": "implemented", "geometry_quality": "not_evaluated",
                     "independent_accuracy": "unverified"},
        "limitations": [
            "the shared identity is fitted to a sparse keyframe set, not to every frame of the turn",
            "yaw comes from landmark rigid pose against the neutral template, not from a calibrated goniometer",
            "expression is bounded around the frontal anchor to keep the mouth closed; it is not an expression model",
            "no independent scan, manual contour or calibrated external pose is used",
        ],
    }
    write_json(output / "report.json", report)
    print(json.dumps({"output": str(output), "elapsed_s": elapsed, "frames": n,
                      "shape_delta_from_anchor_absmax": shape_delta,
                      "correspondence_frames": len(matches)}), flush=True)
    return {"joint_shape": str(output), "elapsed_s": elapsed, "keyframes": n,
            "angle_groups": len(group_table), "shape_delta_from_anchor_absmax": shape_delta}


# ------------------------------------------------------------------------ detail
def _weighted_robust(residual, weight, *, beta: float = 0.3):
    """Pointwise robust loss followed by normalization by the valid weight mass."""
    residual = torch.as_tensor(residual)
    weight = torch.as_tensor(weight, device=residual.device, dtype=residual.dtype)
    if residual.shape != weight.shape:
        raise ValueError(f"residual/weight shape mismatch: {tuple(residual.shape)} != {tuple(weight.shape)}")
    valid = torch.isfinite(residual) & torch.isfinite(weight) & (weight > 0)
    if not bool(valid.any()):
        return residual.sum() * 0
    pointwise = torch.nn.functional.smooth_l1_loss(
        residual, torch.zeros_like(residual), beta=float(beta), reduction="none")
    selected = pointwise[valid] * weight[valid]
    return selected.sum() / weight[valid].sum()


def _refresh_visibility(base_mask, visible):
    """Rebuild the current mask from immutable validity and current visibility."""
    return torch.as_tensor(base_mask, dtype=torch.bool, device=visible.device) & visible


def _reliability_weights(incidence, scatter_mm, normal_agreement,
                         *, noise_floor_mm: float = 0.28867513459481287):
    """Per-ray weights that preserve measured differences between rays."""
    incidence = np.asarray(incidence, dtype=np.float64)
    scatter_mm = np.asarray(scatter_mm, dtype=np.float64)
    agreement = np.asarray(normal_agreement, dtype=np.float64)
    if not (incidence.shape == scatter_mm.shape == agreement.shape):
        raise ValueError("incidence, scatter and normal agreement must have equal shapes")
    if not len(incidence):
        return np.zeros(0, dtype=np.float64)
    finite = (np.isfinite(incidence) & np.isfinite(scatter_mm) & np.isfinite(agreement)
              & (incidence > 0) & (scatter_mm >= 0) & (agreement > 0.5))
    raw = np.zeros_like(incidence)
    plane_confidence = np.clip((agreement[finite] - 0.5) / 0.5, 0.0, 1.0)
    noise_confidence = (float(noise_floor_mm) ** 2
                        / np.maximum(scatter_mm[finite], float(noise_floor_mm)) ** 2)
    raw[finite] = incidence[finite] ** 2 * plane_confidence * noise_confidence
    positive = raw > 0
    if not positive.any():
        return raw
    mean = float(raw[positive].mean())
    if not np.isfinite(mean) or mean <= 1e-12:
        return np.zeros_like(raw)
    return raw / mean


def _landmark_masks(valid3d, valid2d, device):
    stable = np.asarray(valid3d, dtype=bool).copy()
    stable[:17] = False
    stable[60:] = False
    inner = np.zeros(68, dtype=bool)
    inner[60:] = np.asarray(valid3d, dtype=bool)[60:]
    projected = np.asarray(valid2d, dtype=bool).copy()
    projected[:17] = False
    return tuple(torch.as_tensor(mask, device=device, dtype=torch.bool)
                 for mask in (stable, inner, projected))


def _check_units(info: dict, path: Path) -> None:
    if (info.get("schema"), info.get("depth_unit"), info.get("mesh_unit"),
            info.get("coordinate_convention")) != ("head_rgbd.sequence.v1", "mm", "m",
                                                   "X_right_Y_up_Z_backward"):
        raise ValueError(f"{path} does not declare the prepared sequence contract")


def _frame_weights(rows) -> tuple[np.ndarray, list[dict]]:
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(str(row.get("angle_group")), []).append(i)
    count = len(groups)
    weight = np.zeros(len(rows))
    table = []
    for label, ids in groups.items():
        quality = np.asarray([max(float(rows[i].get("tracking_quality") or 0.0), 1e-6) for i in ids])
        share = quality / quality.sum()
        for local, i in enumerate(ids):
            weight[i] = share[local] / count
        table.append({"angle_group": label, "frames": [rows[i]["id"] for i in ids],
                      "total_weight": round(1.0 / count, 6),
                      "frame_weights": [round(float(weight[i]), 6) for i in ids],
                      "quality": [round(float(rows[i].get("tracking_quality") or 0.0), 4) for i in ids]})
    return weight, table


def _mouth_region(raw_vertices, faces, landmark_faces, landmark_bary) -> np.ndarray:
    """Generous canonical mouth box from the closed frontal lip landmarks."""
    triangles = faces[np.asarray(landmark_faces)[48:68]]
    lips = (raw_vertices[triangles] * np.asarray(landmark_bary)[48:68, :, None]).sum(1)
    return ((raw_vertices[:, 0] > lips[:, 0].min() - 0.008)
            & (raw_vertices[:, 0] < lips[:, 0].max() + 0.008)
            & (raw_vertices[:, 1] > lips[:, 1].min() - 0.008)
            & (raw_vertices[:, 1] < lips[:, 1].max() + 0.008)
            & (raw_vertices[:, 2] > lips[:, 2].min() - 0.02))


def _model_gap(vertices, faces, landmark_faces, landmark_bary) -> float:
    triangles = faces[np.asarray(landmark_faces)]
    landmarks = (vertices[triangles] * np.asarray(landmark_bary)[:, :, None]).sum(1)
    return float(np.linalg.norm(landmarks[62] - landmarks[66]) * 1000.)


def _yaw(rotation) -> float:
    from scipy.spatial.transform import Rotation

    return float(Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)[1])


def _pose_delta(rotation, translation, initial, translations, anchored) -> dict:
    from scipy.spatial.transform import Rotation

    angular, linear = [], []
    for i, _ in anchored:
        relative = rotation[i] @ np.asarray(initial[i]).T
        angular.append(float(np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec()))))
        linear.append(float(np.linalg.norm(translation[i] - np.asarray(translations[i])) * 1000.))
    return {"anchor_frames": len(angular),
            "max_rotation_deg": max(angular) if angular else 0.,
            "max_translation_mm": max(linear) if linear else 0.,
            "scope": "change of the frontal keyframes during the shared fit; the shape prior holds them loosely"}
