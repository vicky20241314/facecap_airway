"""Position review: overlay landmark / binding / shaping centre on RGB + depth.

For each state the tool draws, on both the raw RGB and the depth image:
  * the original manual (or ArUco) landmark pixel;
  * the baseline-mesh binding point projected into the image;
  * the proposed shaping centre (binding point projected on the surface);
  * the head midline and the local neck midline (model-frame x=0 line);
  * the elliptical support footprint around the binding point;
  * the facial midplane (x=0 plane) trace for the offset diagnosis.

Output: <output>/<state>_rgb_overlay.png and <state>_depth_overlay.png plus
position_review.json with measured offsets (binding centre vs face midplane,
binding vs marker, neck midline offset).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def project(points_m: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """Reference-camera 3D (x, y, -z depth) -> pixel (u, v)."""
    fx, fy, cx, cy = intr
    p = np.asarray(points_m, dtype=np.float64)
    z = -p[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = cx + fx * p[:, 0] / safe
    v = cy - fy * p[:, 1] / safe
    return np.stack([u, v], axis=1)


def load_frame(prepared: Path, frame_id: str) -> dict:
    data = np.load(prepared / f"{frame_id}.npz", allow_pickle=False)
    return {k: data[k] for k in data.files}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="run with reconstruction + delivery")
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    fit = args.run / "reconstruction" / "dense"
    references = json.loads((fit / "reference_geometry.json").read_text(encoding="utf-8"))["references"]
    landmarks_payload = json.loads(args.landmarks.read_text(encoding="utf-8"))
    prom_path = args.run / "verification" / "delivery_male" / "neck_prominence.json"
    prom = (
        json.loads(prom_path.read_text(encoding="utf-8"))
        if prom_path.is_file()
        else json.loads((args.run / "delivery" / "neck_prominence.json").read_text(encoding="utf-8"))
    )

    review: dict = {"schema": "head_rgbd.neck_position_review.v1", "states": {}}
    for state in ("closed", "open"):
        record = landmarks_payload["states"][state]
        frame_id = record["frame_id"]
        frame = load_frame(args.prepared, frame_id)
        rgb = frame["rgb"]
        depth = frame["depth"].astype(np.float64)  # uint16 mm, 0 = invalid
        intr = frame["intrinsics"]
        reference = references[state]
        rg = reference["stages"]["dense"]
        model_to_camera = np.asarray(reference["animated_model_to_reference_camera_4x4"], dtype=np.float64)
        rot = model_to_camera[:3, :3]
        tr = model_to_camera[:3, 3]
        to_camera = lambda p: np.asarray(p) @ rot.T + tr
        landmarks_cam = np.asarray(rg["landmarks_m"], dtype=np.float64) @ rot.T + tr

        marker_pixel = np.asarray(record["pixel_uv"], dtype=np.float64)
        entry = prom["states"][state]
        b = entry["binding"]
        marker_cam = np.asarray(b["marker_point_reference_camera_m"], dtype=np.float64)
        binding_cam = np.asarray(b["baseline_binding_point_reference_camera_m"], dtype=np.float64)

        # Head axes in camera frame from the baseline model landmarks.
        pts = landmarks_cam
        right = pts[45] - pts[36]
        right /= np.linalg.norm(right)
        up = pts[27] - pts[8]
        up /= np.linalg.norm(up)
        right = right - np.dot(right, up) * up
        right /= np.linalg.norm(right)
        forward = np.cross(up, right)
        forward /= np.linalg.norm(forward)
        # Nose must be in front: fix the sign from the eye centre to nose tip.
        ref = pts[30] - 0.5 * (pts[36] + pts[45])
        if np.dot(ref, forward) < 0:
            forward = -forward

        # Face midplane: midsagittal plane through the eye midpoint.
        eye_mid = 0.5 * (pts[36] + pts[45])

        # Neck midline: midsagittal-plane points below the chin, at the eye
        # midpoint's forward depth extended down through the neck.
        plane_point = eye_mid - np.dot(eye_mid - binding_cam, right) * right
        midline_pts = [
            plane_point + (y_mm * 1e-3) * up for y_mm in range(-60, 61, 5)
        ]
        midline_uv = project(np.asarray(midline_pts), intr)

        binding_lateral_mm = float(np.dot(binding_cam - eye_mid, right) * 1000.0)
        marker_lateral_mm = float(np.dot(marker_cam - eye_mid, right) * 1000.0)

        # Support footprint ellipse around the binding (rx=12, ry=18 default).
        from math import cos, sin
        ellipse_pts = []
        for deg in range(0, 360, 6):
            t = np.deg2rad(deg)
            ellipse_pts.append(
                binding_cam
                + 12e-3 * cos(t) * right
                + 18e-3 * sin(t) * up
            )
        ellipse_uv = project(np.asarray(ellipse_pts), intr)

        marker_uv = project(marker_cam[None, :], intr)[0]
        binding_uv = project(binding_cam[None, :], intr)[0]

        # Depth local profile: along up axis through the binding, ±40 mm,
        # measured as the front-most valid depth in a 5-pixel lateral window.
        profile = []
        for y_mm in range(-40, 41, 2):
            p = binding_cam + y_mm * 1e-3 * up
            uv = project(p[None, :], intr)[0]
            u0, v0 = int(round(uv[0])), int(round(uv[1]))
            best = None
            for du in range(-2, 3):
                u = u0 + du
                if not (0 <= u < depth.shape[1] and 0 <= v0 < depth.shape[0]):
                    continue
                d = depth[v0, u]
                if d > 0:
                    best = d if best is None else min(best, d)
            profile.append({"y_mm": y_mm, "depth_mm": best})

        for image, kind, name in ((rgb, "rgb", f"{state}_rgb_overlay.png"),
                                  (depth, "depth", f"{state}_depth_overlay.png")):
            fig, ax = plt.subplots(figsize=(12.8, 7.2))
            if kind == "rgb":
                ax.imshow(image)
            else:
                show = np.where(depth > 0, depth, np.nan)
                im = ax.imshow(show, cmap="viridis_r")
                plt.colorbar(im, ax=ax, label="depth mm")
            ax.plot(*marker_pixel, "b+", ms=16, mew=2.5, label="original manual point")
            ax.plot(*marker_uv, "cs", ms=9, mfc="none", mew=1.6, label="marker 3D (backprojected)")
            ax.plot(*binding_uv, "g+", ms=14, mew=2.2, label="baseline mesh binding")
            ax.plot(midline_uv[:, 0], midline_uv[:, 1], "y-", lw=1.4, alpha=0.9, label="neck midline (midsagittal)")
            ax.plot(ellipse_uv[:, 0], ellipse_uv[:, 1], "r-", lw=1.2, alpha=0.85, label="support footprint 12x18 mm")
            ax.set_xlim(400, 900)
            ax.set_ylim(720, 300)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f"{state} / {frame_id}: position review")
            fig.savefig(args.output / name, dpi=110)
            plt.close(fig)

        # Lateral-offset diagnosis via mesh projection: how far is the binding
        # from the mesh's own midsagittal plane (face midplane)?
        review["states"][state] = {
            "frame_id": frame_id,
            "manual_pixel_uv": record["pixel_uv"] if isinstance(record["pixel_uv"], list) else marker_pixel.tolist(),
            "marker_reference_camera_m": marker_cam.tolist(),
            "binding_reference_camera_m": binding_cam.tolist(),
            "binding_to_marker_mm": float(np.linalg.norm(binding_cam - marker_cam) * 1000),
            "binding_lateral_offset_from_midsagittal_mm": binding_lateral_mm,
            "marker_lateral_offset_from_midsagittal_mm": marker_lateral_mm,
            "binding_depth_mm": float(-binding_cam[2] * 1000),
            "marker_depth_mm": float(-marker_cam[2] * 1000),
            "depth_profile_along_up": profile,
        }
    (args.output / "position_review.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for state, item in review["states"].items():
        print(state, json.dumps({k: v for k, v in item.items() if k != "depth_profile_along_up"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
