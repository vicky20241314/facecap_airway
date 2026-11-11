"""Manually specify the laryngeal prominence point on one prepared frame.

Usage (interactive click):
    python -m head_rgbd.tools.neck_landmark_manual --prepared <dir> \
        --state closed --frame-id closed_front_0 --output neck_landmarks.json

Usage (recorded pixel, e.g. chosen by inspecting the saved frame image):
    ... --pixel 640 468

The click is depth-checked by ``neck_landmark.query_observed_point``; a hole,
depth jump or occlusion edge is recorded as a failure, never substituted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from head_rgbd import neck_landmark as nl


def _load_frame(prepared: Path, frame_id: str) -> dict[str, np.ndarray]:
    path = prepared / f"{frame_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"prepared frame not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _pick_pixel(rgb: np.ndarray) -> tuple[float, float] | None:
    import cv2

    window = "laryngeal prominence click (ESC cancels)"
    clicked: list[tuple[float, float]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, rgb[:, :, ::-1])
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        if clicked:
            break
    cv2.destroyWindow(window)
    return clicked[0] if clicked else None


def _record(
    *,
    state: str,
    frame_id: str,
    arrays: dict[str, np.ndarray],
    uv: tuple[float, float] | None,
    query: dict,
) -> dict:
    rgb = arrays["rgb"]
    depth = arrays["depth"]
    intr = np.asarray(arrays["intrinsics"], dtype=np.float64).reshape(-1)
    height, width = depth.shape
    return {
        "name": nl.NAME,
        "definition": nl.DEFINITION,
        "definition_scope": nl.DEFINITION_SCOPE,
        "source": nl.SOURCE_MANUAL,
        "state": state,
        "status": query.get("status", nl.STATUS_FAILED),
        "reason": query.get("reason"),
        "frame_id": frame_id,
        "segment_file": f"{frame_id}.npz",
        "source_frame": int(arrays["source_frame"]) if "source_frame" in arrays else None,
        "image_size": [int(width), int(height)],
        "pixel_uv": [float(uv[0]), float(uv[1])] if uv is not None else None,
        "point_observed_m": query.get("point_observed_m"),
        "coordinate_system": nl.OBSERVATION_FRAME_COORDINATES if query.get("status") == nl.STATUS_OBSERVED else None,
        "unit": "m",
        "depth": query.get("depth"),
        "observation_is_manual_truth": False,
        "note": "人工指定的喉结体表位置；不是内部甲状软骨的直接测量",
        "frame_binding": {
            "rgb_sha256": nl.array_sha256(rgb),
            "depth_sha256": nl.array_sha256(depth),
            "intrinsics_sha256": nl.array_sha256(intr),
            "intrinsics_fx_fy_cx_cy": [float(v) for v in intr],
            "image_size": [int(width), int(height)],
            "source_frame": int(arrays["source_frame"]) if "source_frame" in arrays else None,
            "statement": "displayed RGB, depth and intrinsics come from this exact saved frame",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True, help="prepared sequence directory")
    parser.add_argument("--state", choices=nl.STATES, required=True)
    parser.add_argument("--frame-id", required=True, help="e.g. closed_front_0")
    parser.add_argument("--output", type=Path, default=None, help="neck_landmarks.json to create/update")
    parser.add_argument("--pixel", type=float, nargs=2, metavar=("U", "V"), default=None,
                        help="recorded pixel choice (skips the interactive window)")
    args = parser.parse_args(argv)

    arrays = _load_frame(args.prepared, args.frame_id)
    uv = tuple(args.pixel) if args.pixel else _pick_pixel(arrays["rgb"])
    if uv is None:
        print(json.dumps({"status": "cancelled"}, ensure_ascii=False))
        return 1
    query = nl.query_observed_point(uv, arrays["depth"], arrays["intrinsics"])
    record = _record(state=args.state, frame_id=args.frame_id, arrays=arrays, uv=uv, query=query)

    output = args.output or (args.prepared / nl.LANDMARKS_FILE)
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("schema") != nl.LANDMARKS_SCHEMA:
            raise ValueError(f"unsupported schema in {output}")
    else:
        payload = nl.empty_landmarks_payload()
    payload["states"][args.state] = record
    payload["sources"] = {
        state: (row.get("source") or nl.SOURCE_MANUAL) for state, row in payload["states"].items()
    }
    nl.write_landmarks_payload(output, payload)

    print(json.dumps({
        "output": str(output),
        "state": args.state,
        "frame_id": args.frame_id,
        "status": record["status"],
        "reason": record["reason"],
        "pixel_uv": record["pixel_uv"],
        "point_observed_m": record["point_observed_m"],
    }, ensure_ascii=False, indent=2))
    return 0 if record["status"] == nl.STATUS_OBSERVED else 2


if __name__ == "__main__":
    raise SystemExit(main())
