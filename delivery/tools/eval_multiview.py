"""Per-view same-frame accuracy for several runs, split by shape-training role.

Each frame is scored against its own observation using that frame's own posed
mesh, so no cross-frame pose or expression difference leaks into the residual.
Frames are grouped by view (front/left/right), mouth state, and whether that
frame contributed to the shared shape, which makes a held-out side evaluation
possible without a separate export.
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

# Nearest observed landmark decides the region, so nose, jaw contour and the
# rest of the face are reported separately instead of as one whole-face mean.
REGION_LANDMARKS = {"nose": range(27, 36), "jaw_contour": range(0, 17),
                    "lips": range(48, 68)}

ROLE = {
    (True, "closed"): "shape_training",
    (False, "closed"): "held_out_closed",
    (False, "open"): "open",
    (True, "open"): "open",
}


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _regions(uv, landmarks):
    distance = np.linalg.norm(uv[:, None, :] - landmarks[None, :, :], axis=2)
    nearest = distance.argmin(1)
    region = np.full(len(uv), "face_other", dtype=object)
    for name, indices in REGION_LANDMARKS.items():
        region[np.isin(nearest, list(indices))] = name
    return region


def _frame_residuals(vertices, faces, points, uv, depth):
    hit, ray, face = RayIndex(vertices, faces).first_hits(points)
    residual = np.full(len(points), np.nan)
    covered = np.zeros(len(points), bool)
    if len(hit):
        tri = vertices[faces[face]]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        direction = -hit / np.maximum(np.linalg.norm(hit, axis=1, keepdims=True), 1e-12)
        front = (normal * direction).sum(1) > 0.
        measured = np.abs(-hit[:, 2] * 1000. - depth[uv[ray, 1], uv[ray, 0]].astype(float))
        keep = front & (measured <= MAX_RESIDUAL_MM)
        residual[ray[keep]] = measured[keep]
        covered[ray[keep]] = True
    return residual, covered


def evaluate(run, reference_split, samples, seed):
    sequence = json.loads((run / "prepared" / "sequence.json").read_text(encoding="utf-8"))
    frames = [f for f in sequence["frames"] if f.get("status") == "candidate"]
    parameters = run / "reconstruction" / "dense" / "parameters.npz"
    with np.load(parameters, allow_pickle=False) as z:
        vertices = np.array(z["dense_vertices"])
        faces = np.array(z["dense_faces"])
        selected = {int(i) for i in z["identity_selected"]}
    if len(vertices) != len(frames):
        raise ValueError(f"{run}: dense_vertices has {len(vertices)} rows for {len(frames)} frames")
    rng = np.random.default_rng(seed)
    rows = []
    for index, entry in enumerate(frames):
        frame = _load(run / "prepared" / entry["file"])
        depth = frame["depth"]
        mask = frame["face_mask"] & (depth > 0)
        ys, xs = np.where(mask)
        if not len(xs):
            continue
        take = rng.choice(len(xs), min(samples, len(xs)), replace=False)
        uv = np.column_stack((xs[take], ys[take]))
        points, ok = backproject(uv, depth, frame["intrinsics"])
        uv, points = uv[ok], points[ok]
        residual, covered = _frame_residuals(vertices[index], faces, points, uv, depth)
        region = _regions(uv, frame["uv"])
        rows.append(dict(
            region=region,
            frame_id=entry["id"], view=entry["view"], state=entry["state"], index=index,
            used_for_shape=index in reference_split,
            role=ROLE[(index in reference_split, entry["state"])],
            n_sampled=len(points), n_used=int(covered.sum()),
            n_missing=int(len(points) - covered.sum()),
            residual=residual, covered=covered,
        ))
    return rows


def summarize_by_region(rows):
    groups = {}
    for row in rows:
        for name in ("nose", "jaw_contour", "lips", "face_other"):
            mask = row["region"] == name
            if not mask.any():
                continue
            groups.setdefault((row["view"], row["state"], row["role"], name), []).append(
                dict(n_sampled=int(mask.sum()), n_used=int((mask & row["covered"]).sum()),
                     residual=row["residual"][mask], covered=row["covered"][mask]))
    out = []
    for (view, state, role, name), members in sorted(groups.items()):
        residual = np.concatenate([m["residual"] for m in members])
        covered = np.concatenate([m["covered"] for m in members])
        used = covered & np.isfinite(residual)
        entry = dict(view=view, state=state, role=role, region=name, n_frames=len(members),
                     n_sampled=int(len(residual)), n_used=int(used.sum()),
                     n_missing=int(len(residual) - used.sum()),
                     coverage=float(used.sum() / max(len(residual), 1)))
        if used.any():
            value = residual[used]
            entry.update(mae_mm=float(np.abs(value).mean()),
                         rmse_mm=float(np.sqrt((value ** 2).mean())),
                         p95_mm=float(np.percentile(np.abs(value), 95)))
        else:
            entry.update(mae_mm=None, rmse_mm=None, p95_mm=None)
        out.append(entry)
    return out


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["view"], row["state"], row["role"]), []).append(row)
    out = []
    for (view, state, role), members in sorted(groups.items()):
        residual = np.concatenate([m["residual"] for m in members])
        covered = np.concatenate([m["covered"] for m in members])
        used = covered & np.isfinite(residual)
        entry = dict(view=view, state=state, role=role, n_frames=len(members),
                     n_sampled=int(len(residual)), n_used=int(used.sum()),
                     n_missing=int(len(residual) - used.sum()),
                     coverage=float(used.sum() / max(len(residual), 1)))
        if used.any():
            value = residual[used]
            entry.update(mae_mm=float(np.abs(value).mean()),
                         rmse_mm=float(np.sqrt((value ** 2).mean())),
                         p95_mm=float(np.percentile(np.abs(value), 95)))
        else:
            entry.update(mae_mm=None, rmse_mm=None, p95_mm=None)
        out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="run directory; the first one defines the shape-training split")
    parser.add_argument("--label", action="append", default=None,
                        help="label per run, in the same order")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    labels = args.label or [f"run{i}" for i in range(len(args.run))]

    with np.load(args.run[0] / "reconstruction" / "dense" / "parameters.npz",
                 allow_pickle=False) as z:
        reference_split = {int(i) for i in z["identity_selected"]}

    payload = {"sampling": dict(samples_per_frame=args.samples, seed=args.seed,
                                max_residual_mm=MAX_RESIDUAL_MM,
                                residual="absolute optical-axis depth difference at the first front hit",
                                split_source=str(args.run[0]),
                                shape_training_frames=sorted(reference_split)),
               "runs": []}
    for label, run in zip(labels, args.run):
        rows = evaluate(run, reference_split, args.samples, args.seed)
        payload["runs"].append(dict(label=label, run=str(run),
                                    groups=summarize(rows),
                                    regions=summarize_by_region(rows),
                                    frames=[{k: v for k, v in r.items()
                                             if k not in ("residual", "covered", "region")} for r in rows]))
        print(f"== {label} ==")
        for g in payload["runs"][-1]["groups"]:
            if g["mae_mm"] is None:
                print(f"   {g['view']:5s} {g['state']:6s} {g['role']:17s} null ({g['n_used']}/{g['n_sampled']})")
                continue
            print(f"   {g['view']:5s} {g['state']:6s} {g['role']:17s} "
                  f"MAE {g['mae_mm']:6.3f}  RMSE {g['rmse_mm']:6.3f}  P95 {g['p95_mm']:6.3f}  "
                  f"cov {g['coverage']*100:5.2f}%  ({g['n_used']}/{g['n_sampled']}, miss {g['n_missing']})")

    # Deltas against the first run, matched on (view, state, role).
    base = {(g["view"], g["state"], g["role"]): g for g in payload["runs"][0]["groups"]}
    for entry in payload["runs"][1:]:
        for g in entry["groups"]:
            b = base.get((g["view"], g["state"], g["role"]))
            if b and b["mae_mm"] is not None and g["mae_mm"] is not None:
                g["delta_vs_first"] = dict(mae_mm=g["mae_mm"] - b["mae_mm"],
                                           rmse_mm=g["rmse_mm"] - b["rmse_mm"],
                                           p95_mm=g["p95_mm"] - b["p95_mm"],
                                           coverage=g["coverage"] - b["coverage"])

    # Region table for closed frames only: this is the lateral-geometry question.
    lines = ["# Per-region per-view accuracy (closed frames)", ""]
    for entry in payload["runs"]:
        lines.append(f"## {entry['label']}")
        lines.append("")
        lines.append("| view | role | region | frames | MAE | RMSE | P95 | coverage | used/sampled |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for g in entry["regions"]:
            if g["state"] != "closed" or g["mae_mm"] is None:
                continue
            lines.append(f"| {g['view']} | {g['role']} | {g['region']} | {g['n_frames']} | {g['mae_mm']:.3f} | "
                         f"{g['rmse_mm']:.3f} | {g['p95_mm']:.3f} | {g['coverage']*100:.2f}% | "
                         f"{g['n_used']}/{g['n_sampled']} |")
        lines.append("")
    (args.output / "regions_by_view.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (args.output / "multiview.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    lines = ["# Per-view same-frame accuracy", "",
             f"采样：每帧 {args.samples} 点，seed={args.seed}，门限 |残差| ≤ {MAX_RESIDUAL_MM:g} mm。", "",
             "同一帧用自己的姿态网格对自己的深度求残差，没有跨帧姿态/表情差。", ""]
    for entry in payload["runs"]:
        lines.append(f"## {entry['label']}")
        lines.append("")
        lines.append("| view | state | role | frames | MAE | RMSE | P95 | coverage | used/sampled | missing |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for g in entry["groups"]:
            if g["mae_mm"] is None:
                lines.append(f"| {g['view']} | {g['state']} | {g['role']} | {g['n_frames']} | null | null | null | "
                             f"{g['coverage']*100:.2f}% | {g['n_used']}/{g['n_sampled']} | {g['n_missing']} |")
                continue
            lines.append(f"| {g['view']} | {g['state']} | {g['role']} | {g['n_frames']} | {g['mae_mm']:.3f} | "
                         f"{g['rmse_mm']:.3f} | {g['p95_mm']:.3f} | {g['coverage']*100:.2f}% | "
                         f"{g['n_used']}/{g['n_sampled']} | {g['n_missing']} |")
        lines.append("")
    (args.output / "multiview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
