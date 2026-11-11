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


def _evaluate_state(prepared: Path, state: str, rows: list[dict], spec: am.ArucoSpec) -> dict:
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
        attempts.append({
            "frame_id": frame_id,
            "status": record.get("status"),
            "reason": record.get("reason"),
            "blur_score": record.get("blur_score"),
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
            "unit": "m",
            "attempts": attempts,
        }
    record = best[1]
    skin = record.get("skin_estimate_m") or record.get("point_observed_m")
    return {
        "name": nl.NAME,
        "definition": nl.DEFINITION,
        "definition_scope": nl.DEFINITION_SCOPE,
        "source": nl.SOURCE_ARUCO,
        "state": state,
        "status": nl.STATUS_OBSERVED,
        "reason": None,
        "frame_id": record["frame_id"],
        "segment_file": f"{record['frame_id']}.npz",
        "image_size": record.get("image_size"),
        "pixel_uv": record.get("centre_px"),
        "point_observed_m": skin,
        "marker_surface_point_m": record.get("point_observed_m"),
        "coordinate_system": nl.OBSERVATION_FRAME_COORDINATES,
        "unit": "m",
        "depth_check": record.get("depth_check"),
        "marker": record.get("marker"),
        "attempts": attempts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--states", nargs="+", choices=nl.STATES, default=list(nl.STATES))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--black-square-mm", type=float, default=am.DEFAULT_BLACK_SQUARE_MM)
    parser.add_argument("--white-margin-mm", type=float, default=am.DEFAULT_WHITE_MARGIN_MM)
    args = parser.parse_args(argv)

    sequence_path = args.prepared / "sequence.json"
    rows = json.loads(sequence_path.read_text(encoding="utf-8"))["frames"]
    spec = am.ArucoSpec(black_square_mm=args.black_square_mm, white_margin_mm=args.white_margin_mm)

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
        record = _evaluate_state(args.prepared, state, rows, spec)
        payload["states"][state] = record
        summary[state] = {
            "status": record["status"],
            "reason": record["reason"],
            "frame_id": record.get("frame_id"),
            "point_observed_m": record.get("point_observed_m"),
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
