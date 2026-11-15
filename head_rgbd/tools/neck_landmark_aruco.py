"""Scan prepared frames for the ArUco laryngeal marker and record one landmark.

For each requested state the front-view frames are evaluated with
``aruco_marker.evaluate_frame``; the best observed record (sharpest accepted
marker) is written into neck_landmarks.json for that state.  Frames without a
valid marker produce a failed record with the reason; nothing is invented and
no template point is ever substituted.

Usage:
    python -m head_rgbd.tools.neck_landmark_aruco --prepared <dir> \
        --states closed open --output neck_landmarks.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from head_rgbd import aruco_marker as am
from head_rgbd import neck_landmark as nl
from head_rgbd import neck_stage as ns


def _load_dense(path: Path) -> tuple[dict, np.ndarray, dict]:
    """Dense reference frames for the subject (current schema only)."""
    with np.load(path / "parameters.npz", allow_pickle=False) as data:
        faces = np.asarray(data["dense_faces"], np.int64)
        vertices = data["dense_vertices"]
        states = {}
        for state in nl.STATES:
            refs = json.loads((path / "reference_geometry.json").read_text(encoding="utf-8"))
            stage = refs["references"][state]["stages"]["dense"]
            if "frame_index" not in stage:
                raise ValueError(
                    f"{path} dense stage for {state} has no frame_index; "
                    "this is an old 0.5.1-style archive, rebuild it from the capture")
            states[state] = {
                "V": np.asarray(vertices[stage["frame_index"]], float),
                "landmarks": np.asarray(stage["landmarks_m"], float),
            }
    references = json.loads((path / "reference_geometry.json").read_text(encoding="utf-8"))["references"]
    return states, faces, references


def _evaluate_state(prepared: Path, state: str, rows: list[dict], spec: am.ArucoSpec,
                    subject: dict | None = None,
                    reference_frame_id: str | None = None,
                    position_params: dict | None = None) -> dict:
    front = [
        row for row in rows
        if row.get("state") == state and row.get("view") == "front" and row.get("status") == "candidate"
    ]
    attempts = []
    best = None
    for row in front:
        frame_id = row["id"]
        path = prepared / f"{frame_id}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            rgb = data["rgb"]
            depth = data["depth"]
            intr = np.asarray(data["intrinsics"], dtype=np.float64).reshape(-1)
            timestamp = {"host_monotonic_s": float(data["timestamp_s"])}
        record = am.evaluate_frame(
            rgb, depth, intr, spec, state=state, frame_id=frame_id,
            timestamp=timestamp,
        )
        marker = record.get("marker") or {}
        size = marker.get("size") or {}
        attempts.append({
            "frame_id": frame_id,
            "status": record.get("status"),
            "reason": record.get("reason"),
            "blur_score": record.get("blur_score"),
            "detected_id": record.get("detected_id"),
            "detected_ids": marker.get("detected_ids"),
            "black_square_mm": size.get("black_square_mm"),
            "size_source": size.get("size_source"),
            "size_uncertainty_mm": size.get("size_uncertainty_mm"),
        })
        if record.get("status") == nl.STATUS_OBSERVED:
            score = record.get("blur_score") or 0.0
            if best is None or score > best[0]:
                best = (score, record)
    if best is None:
        failure = next(
            (a for a in attempts if a["status"] == nl.STATUS_FAILED), None
        )
        return {
            "name": nl.NAME,
            "definition": nl.DEFINITION,
            "definition_scope": nl.DEFINITION_SCOPE,
            "source": nl.SOURCE_ARUCO,
            "state": state,
            "status": nl.STATUS_FAILED,
            "reason": (failure or {}).get("reason") or "no_front_frame_with_valid_marker",
            "frame_id": None,
            "point_observed_m": None,
            "point_reference_m": None,
            "skin_estimate_reference_m": None,
            "reference_frame": reference_frame_id,
            "transform": {"status": nl.STATUS_FAILED,
                          "reason": "no_valid_marker_observation"},
            "larynx_center": None,
            "unit": "m",
            "attempts": attempts,
        }
    record = best[1]
    surface = record.get("point_observed_m")
    skin = record.get("skin_estimate_m")
    if subject is not None and reference_frame_id is not None:
        transform = nl.resolve_reference_transform(record, rows, reference_frame_id)
    else:
        transform = {
            "status": nl.STATUS_FAILED,
            "reason": ("reference_frame_not_supplied" if reference_frame_id is None
                       else "dense_subject_not_supplied"),
            "direction": "p_reference_camera = R @ p_annotation_camera + t",
            "unit": "m",
            "annotation_frame": record.get("frame_id"),
            "reference_frame": reference_frame_id,
        }
    point_reference = None
    skin_reference = None
    position = None
    if transform.get("status") == nl.STATUS_OBSERVED:
        point_reference = nl.apply_transform(surface, transform) if surface else None
        skin_reference = nl.apply_transform(skin, transform) if skin else None
        if point_reference is None:
            transform = {**transform, "status": nl.STATUS_FAILED,
                         "reason": "transform_apply_failed"}
    if point_reference is not None and subject is not None:
        position = nl.validate_marker_position(
            point_reference, state=state, V=subject["V"], F=subject["F"],
            ax=subject["ax"], curve=subject["curve"], h_chin=subject["h_chin"],
            p=position_params)
    observed = bool(transform.get("status") == nl.STATUS_OBSERVED
                    and position is not None
                    and position.get("status") == nl.STATUS_OBSERVED)
    reason = None
    if not observed:
        reason = (position or {}).get("reason") or transform.get("reason") or \
            "reference_transform_failed"
    larynx_center = None
    if position is not None:
        larynx_center = {
            "t_marker": position.get("t_marker"),
            "theta_marker_deg": position.get("theta_marker_deg"),
            "t_center_used": position.get("t_center_used") if observed else None,
            "theta_used_deg": 0.0,
            "t_gate": position.get("t_gate"),
            "source": nl.SOURCE_ARUCO,
        }
    marker = record.get("marker") or {}
    return {
        "name": nl.NAME,
        "definition": nl.DEFINITION,
        "definition_scope": nl.DEFINITION_SCOPE,
        "source": nl.SOURCE_ARUCO,
        "state": state,
        "status": nl.STATUS_OBSERVED if observed else nl.STATUS_FAILED,
        "reason": reason,
        "frame_id": record["frame_id"],
        "segment_file": f"{record['frame_id']}.npz",
        "image_size": record.get("image_size"),
        "pixel_uv": record.get("centre_px"),
        "timestamp": record.get("timestamp"),
        "quality": {"blur_score": record.get("blur_score"),
                    "centre_method": record.get("centre_method"),
                    "geometry": record.get("geometry")},
        "marker_plane": record.get("marker_plane"),
        "point_observed_m": surface,
        "marker_surface_point_m": surface,
        "skin_estimate_m": skin,
        "coordinate_system": nl.OBSERVATION_FRAME_COORDINATES,
        "unit": "m",
        "reference_frame": reference_frame_id,
        "reference_coordinate_system": "reference camera X_right_Y_up_Z_backward",
        "transform": transform,
        "point_reference_m": point_reference,
        "skin_estimate_reference_m": skin_reference,
        "larynx_center": larynx_center,
        "position_validation": position,
        "depth_check": record.get("depth_check"),
        "marker": marker,
        "attempts": attempts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--dense", type=Path, default=None,
                        help="reconstruction/dense directory; required for the "
                             "reference-frame transform and front-neck position gate")
    parser.add_argument("--states", nargs="+", choices=nl.STATES, default=list(nl.STATES))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--black-square-mm", type=float, default=am.DEFAULT_BLACK_SQUARE_MM,
                        help="nominal print size only; detection measures the actual size")
    parser.add_argument("--white-margin-mm", type=float, default=am.DEFAULT_WHITE_MARGIN_MM)
    args = parser.parse_args(argv)

    sequence_path = args.prepared / "sequence.json"
    rows = json.loads(sequence_path.read_text(encoding="utf-8"))["frames"]
    spec = am.ArucoSpec(black_square_mm=args.black_square_mm, white_margin_mm=args.white_margin_mm)
    subjects: dict[str, dict] = {}
    reference_ids: dict[str, str | None] = {}
    if args.dense is not None:
        states, faces, references = _load_dense(args.dense)
        for state in nl.STATES:
            V = states[state]["V"]
            _frame, ax, h_chin = ns.subject_frame(V, faces, states[state]["landmarks"])
            curve, _spine = ns.spine_curve(V, faces, ax, h_chin)
            subjects[state] = {"V": V, "F": faces, "ax": ax, "curve": curve,
                               "h_chin": h_chin}
            reference_ids[state] = references[state].get("frame_id")
    position_params = {
        "larynx_prior_gate_dn": -0.30,
        "larynx_prior_gate_fallback": 0.78,
        "marker_max_surface_distance_mm": 30.0,
    }

    output = args.output or (args.prepared / nl.LANDMARKS_FILE)
    payload = (
        json.loads(output.read_text(encoding="utf-8"))
        if output.is_file() and (output.read_text(encoding="utf-8").strip().startswith("{"))
        else nl.empty_landmarks_payload()
    )
    if payload.get("schema") != nl.LANDMARKS_SCHEMA:
        payload = nl.empty_landmarks_payload()
    summary = {}
    for state in args.states:
        record = _evaluate_state(
            args.prepared, state, rows, spec,
            subject=subjects.get(state),
            reference_frame_id=reference_ids.get(state),
            position_params=position_params)
        payload["states"][state] = record
        summary[state] = {
            "status": record["status"],
            "reason": record["reason"],
            "frame_id": record.get("frame_id"),
            "point_observed_m": record.get("point_observed_m"),
            "point_reference_m": record.get("point_reference_m"),
            "skin_estimate_reference_m": record.get("skin_estimate_reference_m"),
            "reference_frame": record.get("reference_frame"),
            "transform_status": (record.get("transform") or {}).get("status"),
            "larynx_center": record.get("larynx_center"),
            "detected_id": (record.get("marker") or {}).get("target_id"),
            "black_square_mm": ((record.get("marker") or {}).get("size") or {}).get(
                "black_square_mm"),
        }
    payload["sources"] = {
        state: (row.get("source") or nl.SOURCE_ARUCO) for state, row in payload["states"].items()
    }
    nl.write_landmarks_payload(output, payload)
    print(json.dumps({"output": str(output), "states": summary}, ensure_ascii=False, indent=2))
    ok = all(row["status"] == nl.STATUS_OBSERVED for row in summary.values())
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
