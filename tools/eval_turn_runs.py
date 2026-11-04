"""Compare static-keyframe and repeated-observation turn runs.

The comparison is a same-capture depth-consistency diagnostic. It is not an
independent geometric-accuracy measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from head_rgbd.experiment.turn_config import TurnConfig  # noqa: E402
from head_rgbd.experiment.turn_inputs import open_source  # noqa: E402
from head_rgbd.experiment.turn_keyframes import angle_bucket  # noqa: E402
from head_rgbd.geometry_c import ray_surface, sample_depth  # noqa: E402
from head_rgbd.prepare import rigid  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class _RunFrames:
    def __init__(self, run: Path, samples: int, seed: int):
        self.run = run
        self.samples = int(samples)
        self.seed = int(seed)
        self.report = _read(run / "field" / "report.json")
        self.sequence = _read(run / "prepared" / "sequence.json")
        self.experiment = _read(run / "experiment.json")
        self.config = TurnConfig.from_dict(self.experiment["config"],
                                           source_path=self.experiment.get("config_path"))
        self.index = {row["id"]: index for index, row in enumerate(self.report["frames"])}
        self.files = {row["id"]: row["file"] for row in self.sequence["frames"]
                      if row.get("status") == "candidate"}
        with np.load(run / "field" / "final_vertices.npz", allow_pickle=False) as stored:
            self.vertices = stored["vertices"].copy()
            self.faces = stored["faces"].copy()
        with np.load(run / "pose_correction" / "parameters.npz", allow_pickle=False) as stored:
            self.pose_rotation = stored["R"].copy()
            self.pose_translation = stored["t"].copy()
        self.sources = {}
        self.tracking = {}
        self.tracking_json = {}
        for role in ("anchor", "turn"):
            source_path = Path(self.experiment["inputs"][role]["source"]["path"])
            self.sources[role] = open_source(source_path, role, self.config)
            self.tracking_json[role] = _read(run / "tracking" / f"{role}.json")
            with np.load(run / "tracking" / f"{role}.npz", allow_pickle=False) as stored:
                self.tracking[role] = {key: stored[key].copy() for key in stored.files}

    def evaluate(self, frame_id: str) -> dict:
        index = self.index[frame_id]
        sequence_row = next(row for row in self.sequence["frames"] if row["id"] == frame_id)
        with np.load(self.run / "prepared" / self.files[frame_id], allow_pickle=False) as stored:
            frame = {key: stored[key] for key in stored.files}
        stable_seed = self.seed + sum((position + 1) * ord(character)
                                      for position, character in enumerate(frame_id))
        points = sample_depth(frame, self.samples, np.random.default_rng(stable_seed),
                              exclude_mouth=True, stride=3)[0]
        residual = np.full(len(points), np.nan)
        accepted = 0
        if len(points):
            correspondence, stats = ray_surface(
                self.vertices[index], self.faces, points, .030, None)
            accepted = int(stats.get("accepted", 0))
            if correspondence is not None:
                residual[correspondence["ray"]] = np.abs(
                    correspondence["hit"][:, 2] - correspondence["target"][:, 2]) * 1000.
        measured = residual[np.isfinite(residual)]
        return {
            "frame_id": frame_id,
            "view": sequence_row.get("view"),
            "median_mm": None if not len(measured) else float(np.median(measured)),
            "p90_mm": None if not len(measured) else float(np.percentile(measured, 90)),
            "coverage": accepted / max(len(points), 1),
        }

    def evaluate_holdout(self, canonical_id: str, role: str, source_index: int) -> dict:
        keyframe_index = self.index[canonical_id]
        mesh = self.vertices[keyframe_index].copy()
        key_rotation = self.pose_rotation[keyframe_index]
        key_translation = self.pose_translation[keyframe_index]
        relative_rotation = self.tracking[role]["R"][source_index] @ key_rotation.T
        relative_translation = (self.tracking[role]["t"][source_index]
                                - relative_rotation @ key_translation)
        mesh = mesh @ relative_rotation.T + relative_translation

        frame = self.sources[role].read(source_index)
        uv = self.tracking[role]["uv"][source_index]
        finite = np.isfinite(uv).all(axis=1)
        if int(finite.sum()) < 17:
            raise ValueError(f"held-out frame {role}/{source_index} has too few tracked landmarks")
        face_mask = np.zeros(frame["depth"].shape, dtype=np.uint8)
        hull = cv2.convexHull(np.rint(uv[finite]).astype(np.int32))
        cv2.fillConvexPoly(face_mask, hull, 1)
        face_mask = cv2.erode(face_mask, np.ones((5, 5), np.uint8)).astype(bool)
        for eye in (slice(36, 42), slice(42, 48)):
            if np.isfinite(uv[eye]).all():
                eye_mask = np.zeros(frame["depth"].shape, dtype=np.uint8)
                cv2.fillConvexPoly(eye_mask, cv2.convexHull(np.rint(uv[eye]).astype(np.int32)), 1)
                face_mask &= ~cv2.dilate(eye_mask, np.ones((5, 5), np.uint8)).astype(bool)
        frame["face_mask"] = face_mask
        frame["uv"] = uv
        rng = np.random.default_rng(self.seed + source_index + sum(map(ord, role)))
        points = sample_depth(frame, self.samples, rng, exclude_mouth=True, stride=3)[0]
        mesh = _icp_mesh_to_points(mesh, self.faces, points)
        return self._score_heldout(points, mesh, canonical_id, role, source_index)

    def _score_heldout(self, points, mesh, canonical_id, role, source_index):
        correspondence, stats = ray_surface(mesh, self.faces, points, .030, None)
        residual = np.full(len(points), np.nan)
        if correspondence is not None:
            residual[correspondence["ray"]] = np.abs(
                correspondence["hit"][:, 2] - correspondence["target"][:, 2]) * 1000.
        measured = residual[np.isfinite(residual)]
        return {
            "frame_id": f"holdout_{role}_{source_index:03d}",
            "canonical_keyframe_id": canonical_id,
            "view": self.tracking_json[role]["frames"][source_index].get("view_label"),
            "median_mm": None if not len(measured) else float(np.median(measured)),
            "p90_mm": None if not len(measured) else float(np.percentile(measured, 90)),
            "coverage": int(stats.get("accepted", 0)) / max(len(points), 1),
        }


def _icp_mesh_to_points(mesh, faces, points, iterations=5):
    current = np.asarray(mesh, dtype=np.float64).copy()
    for _ in range(iterations):
        correspondence, stats = ray_surface(current, faces, points, .050, None)
        if correspondence is None or int(stats.get("accepted", 0)) < 30:
            break
        try:
            rotation, translation = rigid(correspondence["hit"], correspondence["target"])
        except (ValueError, np.linalg.LinAlgError):
            break
        current = current @ rotation.T + translation
    return current


def _physical_id(row: dict):
    return str(row.get("source")), int(row.get("source_frame", row.get("source_index")))


def _select_holdouts(run: Path, selection: dict) -> list[dict]:
    experiment = _read(run / "experiment.json")
    config = TurnConfig.from_dict(experiment["config"], source_path=experiment.get("config_path"))
    tracking = {
        role: _read(run / "tracking" / f"{role}.json")["frames"]
        for role in ("anchor", "turn")
    }
    used = {_physical_id(row) for row in
            [*selection["keyframes"], *selection.get("support_frames", [])]}
    chosen = []
    for canonical in selection["keyframes"]:
        bucket = angle_bucket(canonical["yaw_deg"], config.keyframes.angle_bin_deg)
        candidates = []
        for role, rows in tracking.items():
            for row in rows:
                if (not row.get("usable") or row.get("state_label") not in (None, "closed")
                        or _physical_id(row) in used
                        or angle_bucket(row["yaw_deg"], config.keyframes.angle_bin_deg) != bucket):
                    continue
                delta = abs(float(row["yaw_deg"]) - float(canonical["yaw_deg"]))
                if delta > config.keyframes.repeat_max_angle_delta_deg:
                    continue
                candidates.append((float(row["quality"]), -delta, role, int(row["source_index"])))
        if candidates:
            _, _, role, source_index = max(candidates)
            used.add(_physical_id(tracking[role][source_index]))
            chosen.append({"canonical_keyframe_id": canonical["id"], "role": role,
                           "source_index": source_index, "yaw_deg": canonical["yaw_deg"]})
    return chosen


def _summary(rows) -> dict:
    rows = [row for row in rows if row.get("median_mm") is not None and row.get("p90_mm") is not None]
    medians = [float(row["median_mm"]) for row in rows]
    p90 = [float(row["p90_mm"]) for row in rows]
    coverage = [float(row["coverage"]) for row in rows]
    return {
        "frames": len(rows),
        "median_of_frame_medians_mm": None if not medians else round(float(np.median(medians)), 3),
        "mean_p90_mm": None if not p90 else round(float(np.mean(p90)), 3),
        "worst_frame_median_mm": None if not medians else round(float(max(medians)), 3),
        "mean_coverage": None if not coverage else round(float(np.mean(coverage)), 4),
    }


def _delta(before: dict, after: dict) -> dict:
    def value(item, key):
        return item.get(key)

    result = {}
    for key in ("median_of_frame_medians_mm", "mean_p90_mm", "worst_frame_median_mm", "mean_coverage"):
        a, b = value(before, key), value(after, key)
        result[key] = None if a is None or b is None else round(b - a, 4)
    return result


def _geometry(rows, component: str) -> dict:
    values = [row["geometry_change"][component] for row in rows
              if row.get("geometry_change", {}).get(component, {}).get("median_mm") is not None]
    if not values:
        return {"frames": 0, "median_of_frame_medians_mm": None, "mean_p90_mm": None}
    return {
        "frames": len(values),
        "median_of_frame_medians_mm": round(float(np.median([value["median_mm"] for value in values])), 3),
        "mean_p90_mm": round(float(np.mean([value["p90_mm"] for value in values])), 3),
    }


def compare(static_run: Path, repeated_run: Path, *, samples: int = 6000, seed: int = 20261103) -> dict:
    static_frames = _RunFrames(static_run, samples, seed)
    repeated_frames = _RunFrames(repeated_run, samples, seed)
    static_field = static_frames.report
    repeated_field = repeated_frames.report
    static_joint = _read(static_run / "joint_shape" / "report.json")
    repeated_joint = _read(repeated_run / "joint_shape" / "report.json")
    static_selection = _read(static_run / "selection" / "keyframes.json")
    repeated_selection = _read(repeated_run / "selection" / "keyframes.json")

    static_keyframes = [row["id"] for row in static_selection["keyframes"]]
    repeated_keyframes = [row["id"] for row in repeated_selection["keyframes"]]
    common = [frame_id for frame_id in static_keyframes if frame_id in repeated_keyframes]
    if not common:
        raise ValueError("the two runs have no common keyframes")

    common_entries = {
        "static": [static_frames.evaluate(frame_id) for frame_id in common],
        "repeated": [repeated_frames.evaluate(frame_id) for frame_id in common],
    }
    support_ids = {row["id"] for row in repeated_selection.get("support_frames", [])}
    support_entries = [repeated_frames.evaluate(frame_id) for frame_id in sorted(support_ids)]
    static_metric_rows = {row["frame_id"]: row
                          for row in static_field["metrics"]["per_view_depth_consistency"]}
    repeated_metric_rows = {row["frame_id"]: row
                            for row in repeated_field["metrics"]["per_view_depth_consistency"]}

    static_summary = _summary(common_entries["static"])
    repeated_summary = _summary(common_entries["repeated"])
    views = {}
    for view in sorted({row.get("view") for row in repeated_selection["keyframes"]}):
        static_view = [row for row in common_entries["static"] if row.get("view") == view]
        repeated_view = [row for row in common_entries["repeated"] if row.get("view") == view]
        views[view] = {
            "static": _summary(static_view),
            "repeated": _summary(repeated_view),
            "delta_repeated_minus_static": _delta(_summary(static_view), _summary(repeated_view)),
        }

    geometry = {}
    static_geometry_entries = [static_metric_rows[frame_id] for frame_id in common]
    repeated_geometry_entries = [repeated_metric_rows[frame_id] for frame_id in common]
    support_geometry_entries = [repeated_metric_rows[frame_id] for frame_id in sorted(support_ids)]
    for component in ("shared_shape_to_control", "control_to_loop", "shared_shape_to_final_total"):
        geometry[component] = {
            "static_common_keyframes": _geometry(static_geometry_entries, component),
            "repeated_common_keyframes": _geometry(repeated_geometry_entries, component),
            "repeated_support_frames": _geometry(support_geometry_entries, component),
        }

    holdouts = _select_holdouts(repeated_run, repeated_selection)
    heldout_rows = {
        "static": [static_frames.evaluate_holdout(item["canonical_keyframe_id"],
                                                  item["role"], item["source_index"])
                   for item in holdouts],
        "repeated": [repeated_frames.evaluate_holdout(item["canonical_keyframe_id"],
                                                      item["role"], item["source_index"])
                     for item in holdouts],
    }

    weights = {}
    for label, report in (("static", static_joint), ("repeated", repeated_joint)):
        group_rows = report["angle_weighting"]["groups"]
        weights[label] = {
            "groups": len(group_rows),
            "group_total_min": min((row["total_weight"] for row in group_rows), default=None),
            "group_total_max": max((row["total_weight"] for row in group_rows), default=None),
            "observation_count": sum(len(row["frames"]) for row in group_rows),
        }

    return {
        "schema": "head_rgbd.turn.evaluation.v1",
        "static_run": str(static_run.resolve()),
        "repeated_run": str(repeated_run.resolve()),
        "common_keyframes": common,
        "depth_consistency": {
            "definition": ("absolute optical-axis depth residual on the same-capture depth that drove "
                           "the fit; lower is better only as a training-data diagnostic"),
            "sampling": {"points_per_frame": samples, "frame_id_seed": seed,
                         "matched_pixels": "same frame ID, source arrays and deterministic seed"},
            "static": static_summary,
            "repeated": repeated_summary,
            "delta_repeated_minus_static": _delta(static_summary, repeated_summary),
            "by_view": views,
            "repeated_support_frames": _summary(support_entries),
            "held_out_same_angle_repeats": {
                "policy": ("one unused reliable repeat per angle group, evaluated after relative-pose "
                           "initialization and identical five-step ICP"),
                "static": _summary(heldout_rows["static"]),
                "repeated": _summary(heldout_rows["repeated"]),
                "delta_repeated_minus_static": _delta(
                    _summary(heldout_rows["static"]), _summary(heldout_rows["repeated"])),
                "frames": [{"static": static, "repeated": repeated}
                           for static, repeated in zip(heldout_rows["static"], heldout_rows["repeated"])],
            },
        },
        "geometry_change": {
            "definition": ("matched-vertex movement in each frame; this quantifies change and is not "
                           "an independent accuracy score"),
            "components": geometry,
        },
        "field": {
            "static_maximum_absolute_displacement_mm":
                static_field["field"]["maximum_absolute_displacement_mm"],
            "repeated_maximum_absolute_displacement_mm":
                repeated_field["field"]["maximum_absolute_displacement_mm"],
            "static_loop_blend": static_field["subdivision"]["loop_blend"]["blend"],
            "repeated_loop_blend": repeated_field["subdivision"]["loop_blend"]["blend"],
            "static_loop_fallback_reason": static_field["subdivision"]["loop_blend"]["fallback_reason"],
            "repeated_loop_fallback_reason": repeated_field["subdivision"]["loop_blend"]["fallback_reason"],
        },
        "angle_weights": weights,
        "elapsed_s": {
            "static": static_field["elapsed_s"],
            "repeated": repeated_field["elapsed_s"],
        },
        "limits": [
            "no independent scan or manual truth is available",
            "same-capture residuals can reward overfitting",
            "a later held-out-frame benchmark remains required before claiming general accuracy improvement",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--repeated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = compare(args.static, args.repeated, samples=args.samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "common_keyframes": len(payload["common_keyframes"]),
        "static": payload["depth_consistency"]["static"],
        "repeated": payload["depth_consistency"]["repeated"],
        "delta": payload["depth_consistency"]["delta_repeated_minus_static"],
        "field": payload["field"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
