"""Per-region same-frame accuracy: lips, jaw/contour and the rest of the face.

The whole-face mean hides local regressions, so this reports the same residuals
split by the nearest of the 68 observed landmarks. Both runs are scored on the
identical rays and gates, so only the meshes differ.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from head_rgbd.prepare import backproject          # noqa: E402
from head_rgbd.ray_index import RayIndex           # noqa: E402

MAX_RESIDUAL_MM = 30.
MIN_FRONT_COS = 0.


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _regions(uv, landmarks):
    """Nearest observed landmark decides the region of a sampled pixel."""
    distance = np.linalg.norm(uv[:, None, :] - landmarks[None, :, :], axis=2)
    nearest = distance.argmin(1)
    region = np.full(len(uv), "face_other", dtype=object)
    region[nearest <= 16] = "jaw_contour"
    region[(nearest >= 48) & (nearest <= 67)] = "lips"
    return region, distance.min(1)


def _residuals(run, row, depth, intrinsics, uv, points):
    parameters = run / "reconstruction" / "dense" / "parameters.npz"
    with np.load(parameters, allow_pickle=False) as z:
        vertices = np.array(z["dense_vertices"])[row]
        faces = np.array(z["dense_faces"])
    hit, ray, face = RayIndex(vertices, faces).first_hits(points)
    residual = np.full(len(points), np.nan)
    covered = np.zeros(len(points), bool)
    if len(hit):
        tri = vertices[faces[face]]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        direction = -hit / np.maximum(np.linalg.norm(hit, axis=1, keepdims=True), 1e-12)
        front = (normal * direction).sum(1) > MIN_FRONT_COS
        # Camera Z points backwards, the depth map stores positive millimetres.
        axis_depth_mm = -hit[:, 2] * 1000.
        measured = np.abs(axis_depth_mm - depth[uv[ray, 1], uv[ray, 0]].astype(float))
        keep = front & (measured <= MAX_RESIDUAL_MM)
        residual[ray[keep]] = measured[keep]
        covered[ray[keep]] = True
    return residual, covered


def _summary(residual, covered, region):
    rows = []
    for name in ("lips", "jaw_contour", "face_other"):
        mask = region == name
        used = mask & covered & np.isfinite(residual)
        if not used.any():
            rows.append(dict(region=name, status="no_valid_samples", n_sampled=int(mask.sum()),
                             n_used=0, mae_mm=None, rmse_mm=None, p95_mm=None))
            continue
        r = residual[used]
        rows.append(dict(region=name, status="ok", n_sampled=int(mask.sum()), n_used=int(used.sum()),
                         mae_mm=float(np.abs(r).mean()), rmse_mm=float(np.sqrt((r ** 2).mean())),
                         p95_mm=float(np.percentile(np.abs(r), 95))))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    references = json.loads((args.run / "reconstruction" / "dense" /
                             "reference_geometry.json").read_text(encoding="utf-8"))["references"]
    sequence = json.loads((args.run / "prepared" / "sequence.json").read_text(encoding="utf-8"))
    order = [f["id"] for f in sequence["frames"]]
    rng = np.random.default_rng(args.seed)

    report = {"sampling": dict(samples=args.samples, seed=args.seed,
                               max_residual_mm=MAX_RESIDUAL_MM,
                               region_rule="nearest observed landmark; 0-16 jaw/contour, 48-67 lips"),
              "frames": []}
    for state, entry in references.items():
        frame_id = entry["frame_id"]
        row = order.index(frame_id)
        frame = _load(args.run / "prepared" / sequence["frames"][row]["file"])
        depth = frame["depth"]
        mask = frame["face_mask"] & (depth > 0)
        ys, xs = np.where(mask)
        take = rng.choice(len(xs), min(args.samples, len(xs)), replace=False)
        uv = np.column_stack((xs[take], ys[take]))
        points, ok = backproject(uv, depth, frame["intrinsics"])
        uv, points = uv[ok], points[ok]
        region, distance = _regions(uv, frame["uv"])

        row_report = {"state": state, "frame_id": frame_id, "global_index": row,
                      "n_sampled": int(len(points)), "regions": {}}
        for label, run in (("baseline", args.baseline), ("fast", args.run)):
            residual, covered = _residuals(run, row, depth, frame["intrinsics"], uv, points)
            row_report["regions"][label] = _summary(residual, covered, region)
        for base_row, fast_row in zip(row_report["regions"]["baseline"], row_report["regions"]["fast"]):
            if base_row["status"] == "ok" and fast_row["status"] == "ok":
                fast_row["delta"] = {k: fast_row[k] - base_row[k] for k in ("mae_mm", "rmse_mm", "p95_mm")}
        report["frames"].append(row_report)

    (args.output / "regions.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    lines = ["# Per-region same-frame accuracy", "",
             "Same rays and gates for both runs; only the mesh differs.", ""]
    for frame in report["frames"]:
        lines.append(f"## {frame['state']} (`{frame['frame_id']}`, row {frame['global_index']}), "
                     f"{frame['n_sampled']} rays")
        lines.append("")
        lines.append("| region | baseline MAE | fast MAE | dMAE | baseline RMSE | fast RMSE | base P95 | fast P95 | used/sampled |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for base_row, fast_row in zip(frame["regions"]["baseline"], frame["regions"]["fast"]):
            if base_row["status"] != "ok" or fast_row["status"] != "ok":
                lines.append(f"| {base_row['region']} | null | null | null | null | null | null | null | "
                             f"{fast_row['n_used']}/{fast_row['n_sampled']} |")
                continue
            d = fast_row["delta"]
            lines.append(f"| {base_row['region']} | {base_row['mae_mm']:.4f} | {fast_row['mae_mm']:.4f} | "
                         f"{d['mae_mm']:+.4f} | {base_row['rmse_mm']:.4f} | {fast_row['rmse_mm']:.4f} | "
                         f"{base_row['p95_mm']:.4f} | {fast_row['p95_mm']:.4f} | "
                         f"{fast_row['n_used']}/{fast_row['n_sampled']} |")
        lines.append("")
    (args.output / "regions.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
