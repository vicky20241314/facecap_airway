"""Materialise keyframes and their bounded same-angle support observations.

One detection pass per retained observation, using the same detectors and the
same per-frame array contract as :mod:`head_rgbd.prepare`. Keyframes remain the
pose/coverage anchors; support observations let nearby reliable repeats constrain
the same surface after their own registration. Raw depth images are never
averaged.

Two sequence manifests are written and they reference the same frame files:

``prepared/sequence.json``
    Every selected keyframe. This drives the shared-identity fit, the pose
    correction and the displacement field.
``prepared/anchor_sequence.json``
    Only the frontal anchor keyframes. ``head_rgbd.fit_c --front-only`` runs on this
    file, so the frontal initialisation cannot be pulled sideways by an oblique
    frame that happens to sit near zero yaw.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np

from head_rgbd.audit import sha
from head_rgbd.contracts import COORDINATE_CONVENTION, DEPTH_UNIT, MESH_UNIT, SEQUENCE_SCHEMA
from head_rgbd.experiment.turn_inputs import ROLE_ANCHOR, ROLE_TURN, open_source
from head_rgbd.experiment.turn_tracking import detect_landmarks
from head_rgbd.prepare import backproject


def materialise(run_dir: Path, config, experiment: dict) -> dict:
    run_dir = Path(run_dir)
    selection = json.loads((run_dir / "selection" / "keyframes.json").read_text(encoding="utf-8"))
    keyframes = selection["keyframes"]
    supports = selection.get("support_frames", [])
    observations = [*keyframes, *supports]
    if not observations:
        raise ValueError("keyframe selection is empty; nothing to prepare")

    started = time.perf_counter()
    prepared_dir = run_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=False)

    sequences = {
        role: open_source(experiment["inputs"][role]["source"]["path"], role, config)
        for role in (ROLE_ANCHOR, ROLE_TURN)
    }
    assets = _prepare_parser()
    rows: list[dict] = []
    written: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for keyframe in observations:
        role = keyframe["role"]
        sequence = sequences[role]
        source_index = int(keyframe["source_index"])
        frame = sequence.read(source_index)
        source_key = str(frame["source"])
        if source_key not in hashes:
            hashes[source_key] = sha(Path(source_key))
        row, arrays = _observation(frame, keyframe, config, assets,
                                   calibration_status=sequence.metadata.calibration_status,
                                   source_sha256=hashes[source_key])
        if row is None:
            continue
        target = prepared_dir / f"{keyframe['id']}.npz"
        np.savez_compressed(target, **arrays)
        row["file"] = target.name
        rows.append(row)
        written[keyframe["id"]] = target.name
        print(json.dumps({"prepared": keyframe["id"], "yaw_deg": keyframe["yaw_deg"],
                          "sharpness": round(row["sharpness"], 2)}), flush=True)

    if not rows:
        raise ValueError("no keyframe could be prepared; every selected frame failed the RGB-D contract")
    anchor_rows = [row for row in rows if row["role"] == "anchor"
                   and row.get("observation_role") == "keyframe"]
    if not anchor_rows:
        raise ValueError("the anchor keyframe failed preparation; cannot initialise a frontal fit")
    rows.sort(key=lambda row: (0 if row["role"] == "anchor" else 1, row["angle_bucket_center_deg"]))
    for order, row in enumerate(rows):
        row["index"] = order

    metadata = _manifest_metadata(sequences, experiment, config)
    full = _sequence_payload(rows, metadata, experiment, anchor_only=False)
    anchor_frames = []
    for order, row in enumerate(anchor_rows):
        copy = dict(row)
        copy["index"] = order
        anchor_frames.append(copy)
    anchor = _sequence_payload(anchor_frames, metadata, experiment, anchor_only=True)
    _write(prepared_dir / "sequence.json", full)
    _write(prepared_dir / "anchor_sequence.json", anchor)

    summary = {
        "prepared_frames": len(rows),
        "keyframe_frames": sum(row.get("observation_role") == "keyframe" for row in rows),
        "support_frames": sum(row.get("observation_role") == "support" for row in rows),
        "anchor_frames": len(anchor_rows),
        "turn_observations": sum(row["role"] == "turn" for row in rows),
        "sequence": str(prepared_dir / "sequence.json"),
        "anchor_sequence": str(prepared_dir / "anchor_sequence.json"),
        "sequence_sha256": sha(prepared_dir / "sequence.json"),
        "anchor_sequence_sha256": sha(prepared_dir / "anchor_sequence.json"),
        "elapsed_s": time.perf_counter() - started,
        "detection_policy": ("one YuNet + TUFA pass per retained keyframe/support observation; "
                             "the remaining video frames are never prepared"),
    }
    _write(run_dir / "prepared" / "prepare_summary.json", summary)
    return summary


# ------------------------------------------------------------------------ detail
def _prepare_parser():
    from core import mask_detection

    mask_detection.require_local_parser()
    mask_detection._ONNX_PROVIDERS = ["CPUExecutionProvider"]
    return mask_detection


def _observation(frame: dict, keyframe: dict, config, mask_detection, *,
                 calibration_status: str, source_sha256: str):
    rgb, depth = frame["rgb"], frame["depth"]
    intrinsics = np.asarray(frame["intrinsics"], dtype=np.float64)
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all() or min(intrinsics[:2]) <= 0:
        raise ValueError(f"invalid intrinsics for keyframe {keyframe['id']}: {intrinsics}")
    uv, bbox = detect_landmarks(rgb)
    if uv is None:
        print(json.dumps({"prepared": keyframe["id"], "status": "rejected",
                          "reason": "landmark_detection_failed"}), flush=True)
        return None, None
    uv_valid = np.isfinite(uv).all(axis=1)
    xyz, depth_valid = backproject(uv, depth, intrinsics)
    valid = uv_valid & depth_valid
    xyz[~valid] = np.nan
    if int(valid[27:48].sum()) < 6:
        print(json.dumps({"prepared": keyframe["id"], "status": "rejected",
                          "reason": "insufficient_upper_face_depth"}), flush=True)
        return None, None

    from core.camera import Camera

    segmentation = mask_detection.get_face_data_single(
        Camera(*intrinsics.tolist()), rgb, depth, bbox_xywh=bbox)["parsing"]
    face = np.isin(segmentation, mask_detection.FACE_LABELS)
    face = cv2.erode(face.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    neck = segmentation == 14
    ys, xs = np.where(face & (depth > 100) & (depth < 1500))
    if len(xs) < 200:
        print(json.dumps({"prepared": keyframe["id"], "status": "rejected",
                          "reason": "insufficient_face_depth"}), flush=True)
        return None, None
    y0, y1 = max(0, int(ys.min())), min(depth.shape[0], int(ys.max()) + 1)
    x0, x1 = max(0, int(xs.min())), min(depth.shape[1], int(xs.max()) + 1)
    crop = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(cv2.resize(crop, (128, 128)), cv2.CV_64F).var())

    arrays = {
        "rgb": rgb,
        "depth": depth,
        "intrinsics": intrinsics,
        "uv": uv,
        "xyz": xyz.astype(np.float32),
        "valid": valid,
        "landmark2d_valid": uv_valid,
        "face_mask": face,
        "neck_mask": neck,
        "timestamp_s": np.asarray(float(frame["timestamp_s"]), dtype=np.float64),
        "source_frame": np.asarray(int(frame["source_frame"]), dtype=np.int64),
    }
    row = {
        "id": keyframe["id"],
        "segment": keyframe["tracking_group"],
        "state": "closed",
        "view": keyframe["view"],
        "index": int(keyframe["order"]),
        "timestamp_s": float(frame["timestamp_s"]),
        "source": str(frame["source"]),
        "source_sha256": source_sha256,
        "source_frame": int(frame["source_frame"]),
        "status": "candidate",
        "reason": None,
        "independent_extrinsics": False,
        "role": keyframe["role"],
        "angle_group": keyframe["angle_group"],
        "angle_bucket_center_deg": keyframe["angle_bucket_center_deg"],
        "tracking_quality": keyframe["quality"],
        "selection_reason": keyframe["selection_reason"],
        "observation_role": keyframe.get("observation_role", "keyframe"),
        "canonical_keyframe_id": keyframe.get("canonical_keyframe_id", keyframe["id"]),
        "guided_yaw_deg": keyframe["yaw_deg"],
        "sharpness": sharpness,
        "face_depth_pixels": int(len(xs)),
        "neck_depth_pixels": int(np.count_nonzero(neck & (depth > 100) & (depth < 1500))),
        "landmark_source": "YUNET_TUFA_SINGLE_FRAME",
        "face_bbox_xywh": None if bbox is None else np.asarray(bbox).tolist(),
        "overlap": {"status": "keyframe_sparse_sampling",
                    "adjacent_rgbd_overlap_not_measured": True},
        "tracking_barrier": "sparse_keyframe_selection",
        "degraded": [],
        "landmark_depth_invalid": np.flatnonzero(~valid).tolist(),
        "acquisition": {"calibration_status": calibration_status},
    }
    if row["neck_depth_pixels"] < 30:
        row["degraded"].append("neck_depth_unavailable; head reconstruction continues")
    return row, arrays


def _manifest_metadata(sequences, experiment: dict, config) -> dict:
    notes: list[str] = []
    for role, sequence in sequences.items():
        notes.extend(f"{role}: {note}" for note in sequence.metadata.notes)
    return {
        "schema": SEQUENCE_SCHEMA,
        "status": "prepared",
        "created_at": datetime.now().astimezone().isoformat(),
        "depth_unit": DEPTH_UNIT,
        "mesh_unit": MESH_UNIT,
        "coordinate_convention": COORDINATE_CONVENTION,
        "continuous_full_turn": True,
        "independent_extrinsics": False,
        "calibration_status": "declared_by_input_not_independently_calibrated",
        "human_silhouettes": False,
        "camera": None,
        "camera_model": "declared by the recorded session; hardware not probed here",
        "runtime_dir": str(experiment.get("runtime")),
        "inputs": {role: sequences[role].metadata.as_dict() for role in sequences},
        "notes": list(dict.fromkeys(notes)),
        "keyframe_policy": {
            "max_keyframes": config.keyframes.max_keyframes,
            "angle_bin_deg": config.keyframes.angle_bin_deg,
            "repeat_observations_per_angle": config.keyframes.repeat_observations_per_angle,
            "repeat_max_angle_delta_deg": config.keyframes.repeat_max_angle_delta_deg,
            "anchor_sequence": "anchor_sequence.json contains only role='anchor' frontal frames",
        },
        "limitations": [
            "keyframes plus bounded same-angle supports were prepared; the full turn is not a per-frame sequence",
            "yaw labels come from landmark rigid pose, not from a calibrated goniometer",
            "session alignment is declared by the input, not independently calibrated",
            "invalid landmark depth remains absent and is never a zero constraint",
        ],
    }


def _sequence_payload(rows, metadata: dict, experiment: dict, *, anchor_only: bool) -> dict:
    payload = dict(metadata)
    payload["frames"] = rows
    payload["candidate_frames"] = len(rows)
    payload["excluded_frames"] = 0
    payload["anchor_only"] = anchor_only
    payload["sources"] = [
        {"role": role, "path": str(item["source"]["path"]),
         "sha256": sha(Path(item["source"]["path"]))}
        for role, item in experiment["inputs"].items()
    ]
    return payload


def _write(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
