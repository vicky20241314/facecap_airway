"""Convert an airway3d multi-view capture into a head_rgbd capture directory.

The airway3d capture stores one aligned reference frame per view (rgb, depth,
intrinsics) plus temporally aligned landmark stacks. This pipeline fits from
RGB-D, so each observation becomes one single-frame segment; the landmark stacks
are not copied because the prepare stage runs its own detector.

Only the reference frame per view is carried over, so a four-view capture
becomes a four-segment capture with one frame each. That is recorded in the
segment quality so downstream reports do not imply repeated observations.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "head_rgbd.capture.v1"


def _frame_record(depth, uv, valid, timestamp, intrinsics):
    ys, xs = np.where(depth > 0)
    fraction = float(len(xs) / depth.size)
    if valid.any():
        points = uv[valid]
        x0, y0 = points.min(0)
        x1, y1 = points.max(0)
        bbox = [int(max(x0 - 20, 0)), int(max(y0 - 20, 0)),
                int(x1 - x0 + 40), int(y1 - y0 + 40)]
        status = "detected"
    else:
        bbox = []
        status = "not_detected"
    return dict(
        host_monotonic_s=float(timestamp),
        hardware_timestamp_us=int(round(timestamp * 1e6)),
        global_timestamp_us=0,
        system_timestamp_us=int(round(timestamp * 1e6)),
        color_index=0,
        depth_index=0,
        color_depth_sync_delta_ms=0.0,
        depth_scale_mm=1.0,
        width=int(depth.shape[1]),
        height=int(depth.shape[0]),
        valid_depth_fraction=fraction,
        depth_min_mm=int(depth[depth > 0].min()) if len(xs) else None,
        depth_max_mm=int(depth.max()),
        face_bbox_xywh=bbox,
        face_detection_status=status,
        intrinsics=[float(v) for v in intrinsics],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="directory holding the airway3d session.json and observation_*.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-name", default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    session = json.loads((args.source / "session.json").read_text(encoding="utf-8"))
    if session.get("schema") != "airway3d.rgbd.multiview.v1":
        raise ValueError(f"unsupported source schema: {session.get('schema')!r}")
    if session.get("coordinate_convention") != "X_right_Y_up_Z_backward":
        raise ValueError("source coordinate convention is not X_right_Y_up_Z_backward")

    segments = []
    for index, observation in enumerate(session["observations"]):
        with np.load(args.source / observation["file"], allow_pickle=False) as stored:
            rgb = stored["rgb"]
            depth = stored["depth"]
            intrinsics = stored["intrinsics"]
            uv = stored["uv"]
            valid = stored["valid"]
            timestamps = stored["temporal_timestamps_s"] if "temporal_timestamps_s" in stored.files else np.zeros(1)
        view_id = observation["view_id"]
        name = f"segment_{index:02d}_{view_id}.npz"
        timestamp = float(timestamps[0]) if len(timestamps) else 0.0
        np.savez_compressed(
            args.output / name,
            rgb=rgb[None],
            depth=depth[None],
            intrinsics=intrinsics[None],
            timestamps_s=np.array([timestamp], np.float64),
            hardware_timestamp_us=np.array([int(round(timestamp * 1e6))], np.int64),
            global_timestamp_us=np.zeros(1, np.int64),
            system_timestamp_us=np.array([int(round(timestamp * 1e6))], np.int64),
            color_index=np.zeros(1, np.int64),
            depth_index=np.zeros(1, np.int64),
            color_depth_sync_delta_ms=np.zeros(1, np.float64),
            depth_scale_mm=np.ones(1, np.float64),
        )
        record = _frame_record(depth, uv, valid, timestamp, intrinsics)
        segment = dict(
            id=view_id,
            state=observation["state"],
            view=observation["view"],
            file=name,
            frame_records=[record],
            quality=dict(
                frames=1,
                source_frames=int(observation.get("quality", {}).get("frames", 1)),
                duration_s=0.0,
                timestamp_monotonic=True,
                timestamp_median_gap_s=0.0,
                timestamp_max_gap_s=0.0,
                color_depth_sync_median_ms=0.0,
                color_depth_sync_p95_ms=0.0,
                valid_depth_fraction_median=record["valid_depth_fraction"],
                valid_depth_fraction_min=record["valid_depth_fraction"],
                face_detection_frames=1 if record["face_detection_status"] == "detected" else 0,
                tracking_status="single_reference_frame_from_multiview_capture",
            ),
        )
        segments.append(segment)
        (args.output / f"segment_{index:02d}.json").write_text(
            json.dumps(segment, indent=1), encoding="utf-8")
        print(f"  {view_id:14s} {observation['state']:6s} {observation['view']:5s} "
              f"depth {depth.shape} intrinsics {np.round(intrinsics, 1)}")

    now = datetime.now(timezone.utc).astimezone().isoformat()
    manifest = dict(
        schema=SCHEMA,
        status="complete",
        created_at=now,
        completed_at=now,
        session_name=args.session_name or args.output.name,
        depth_unit="mm",
        mesh_unit="m",
        coordinate_convention="X_right_Y_up_Z_backward",
        rgb_depth_aligned=True,
        alignment=dict(
            mode="INHERITED_D2C",
            evidence="RGB and depth are the same size and share one intrinsics in the source observation",
            independent_physical_verification=False,
            rgb_depth_aligned=True,
            observed_color_depth_sync_median_ms=0.0,
        ),
        sdk_version="inherited",
        depth_work_mode="unknown",
        segments=segments,
        limitations=[
            "converted from an airway3d.rgbd.multiview.v1 capture; one reference frame per view",
            "RGB-D alignment is inherited from the source capture and not re-verified here",
            "temporal landmark stacks are not copied; the prepare stage re-detects landmarks",
        ],
        provenance=dict(
            imported_at=now,
            source=str(args.source),
            source_schema=session.get("schema"),
        ),
    )
    (args.output / "session.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {args.output} with {len(segments)} segments")


if __name__ == "__main__":
    main()
