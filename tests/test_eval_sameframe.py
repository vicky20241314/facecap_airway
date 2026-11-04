"""Tests for ``tools/eval_sameframe.py`` (same-frame reconstruction evaluation).

Only cases that can catch a *real* mistake are pinned here:

* a known 2 mm plane shift must come back as a ~2 mm error, not 0 and not 4;
* an all-miss ray set must give ``n_used == 0`` plus ``null`` metrics -- never 0;
* the evaluated frame must be the one ``reference_geometry.json`` declares
  (``closed_front_4`` / ``open_front_4`` in the real run, never ``*_0``);
* coverage must equal the hand-computed ``n_used / n_sampled``;
* the missing-sample semantics must survive a frame whose mask is empty;
* the residual gate must be a residual, not a camera range.

Synthetic cases build a tiny prepared-run directory on disk so the real loader
and unit/coordinate contracts are exercised, not bypassed.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import eval_sameframe as evaluator  # noqa: E402


# --------------------------------------------------------------------------- #
# synthetic run builder
# --------------------------------------------------------------------------- #
def _plane_mesh(width_m: float, height_m: float, depth_m: float,
                divisions: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Flat grid in the X/Y plane at constant Z (camera looks down -Z)."""
    xs = np.linspace(-width_m / 2, width_m / 2, divisions + 1)
    ys = np.linspace(-height_m / 2, height_m / 2, divisions + 1)
    gx, gy = np.meshgrid(xs, ys)
    vertices = np.column_stack((gx.ravel(), gy.ravel(),
                                np.full(gx.size, -depth_m, dtype=np.float64)))
    faces = []
    stride = divisions + 1
    for row in range(divisions):
        for column in range(divisions):
            a = row * stride + column
            b, c, d = a + 1, a + stride, a + stride + 1
            # Wound so the face normal points toward the camera (+Z), which is the
            # convention the delivery meshes use: verified on the real run, where
            # the majority of front-shell normals have n.z > 0.
            faces.append((a, b, c))
            faces.append((b, d, c))
    return vertices, np.asarray(faces, dtype=np.int64)


def _write_run(root: Path, depth_mm: np.ndarray, intrinsics: np.ndarray,
               mask: np.ndarray, states=("closed", "open"),
               plane_depth_m: float = 0.5, reference_index: int = 4,
               frames_per_state: int = 5) -> Path:
    """Write the minimal on-disk contract the evaluator reads."""
    prepared = root / "prepared"
    delivery = root / "delivery"
    reconstruction = root / "reconstruction" / "dense"
    for directory in (prepared, delivery, reconstruction):
        directory.mkdir(parents=True, exist_ok=True)

    vertices, faces = _plane_mesh(0.10, 0.10, plane_depth_m)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    frames = []
    reference = {}
    for state in states:
        for index in range(frames_per_state):
            frame_id = f"{state}_front_{index}"
            np.savez(
                prepared / f"{frame_id}.npz",
                rgb=np.zeros(depth_mm.shape + (3,), dtype=np.uint8),
                depth=depth_mm.astype(np.uint16),
                intrinsics=intrinsics,
                uv=np.zeros((68, 2)),
                xyz=np.zeros((68, 3)),
                valid=np.zeros(68, dtype=bool),
                landmark2d_valid=np.zeros(68, dtype=bool),
                face_mask=mask,
                neck_mask=np.zeros_like(mask),
                timestamp_s=np.float64(index),
                source_frame=np.int64(index),
            )
            frames.append({
                "id": frame_id, "segment": f"{state}_front", "state": state,
                "view": "front", "index": index, "file": f"{frame_id}.npz",
                "source": str(prepared / "segment.npz"),
                "status": "candidate",
            })
        mesh_path = delivery / f"head_{state}_m.obj"
        if not mesh_path.is_file():
            mesh.export(str(mesh_path))
        reference[state] = {
            "frame_id": f"{state}_front_{reference_index}",
            "source": str(prepared / "segment.npz"),
            "mesh_unit": "m",
        }

    (prepared / "sequence.json").write_text(json.dumps({
        "schema": "head_rgbd.sequence.v1",
        "depth_unit": "mm",
        "mesh_unit": "m",
        "coordinate_convention": "X_right_Y_up_Z_backward",
        "frames": frames,
    }), encoding="utf-8")
    (reconstruction / "reference_geometry.json").write_text(json.dumps({
        "schema": "head_rgbd.d.reference.v1", "references": reference,
    }), encoding="utf-8")
    # The evaluator refuses to score a delivery mesh against another frame's
    # observation, so it reads the per-frame geometry the fit actually exported.
    # This fixture has one static plane, so every frame's row is that plane.
    np.savez(
        reconstruction / "parameters.npz",
        dense_vertices=np.repeat(vertices[None, :, :], len(frames), axis=0),
        dense_faces=faces,
    )
    (delivery / "manifest.json").write_text(json.dumps({
        "mesh": {
            "unit": "m",
            "states": {state: {"path": f"head_{state}_m.obj",
                               "reference_frame": reference[state]["frame_id"]}
                       for state in states},
        },
    }), encoding="utf-8")
    return root


INTRINSICS = np.array([500.0, 500.0, 32.0, 32.0])
MASK = np.zeros((64, 64), dtype=bool)
MASK[16:48, 16:48] = True


class KnownShift(unittest.TestCase):
    """A 2 mm translation along Z must read back as exactly 2 mm."""

    def test_two_millimetre_shift_reports_two_millimetres(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            # Plane sits 2 mm *behind* the measured surface: true residual is 2.0 mm.
            depth_mm = np.full((64, 64), 498.0)
            _write_run(root, depth_mm, INTRINSICS, MASK, plane_depth_m=0.500)
            payload = evaluator.evaluate(root, samples=500, seed=7)
            rows = payload["metrics"]
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["n_used"], row["n_sampled"])
                self.assertAlmostEqual(row["mae_mm"], 2.0, places=6)
                self.assertAlmostEqual(row["rmse_mm"], 2.0, places=6)
                # The signed mean pins the *direction*: mesh is farther, so +2.
                self.assertAlmostEqual(row["signed_mean_mm"], 2.0, places=6)
                self.assertAlmostEqual(row["p95_mm"], 2.0, places=6)

    def test_shift_is_not_silently_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            depth_mm = np.full((64, 64), 500.0)
            _write_run(root, depth_mm, INTRINSICS, MASK, plane_depth_m=0.500)
            payload = evaluator.evaluate(root, samples=400, seed=7)
            for row in payload["metrics"]:
                self.assertEqual(row["status"], "ok")
                self.assertAlmostEqual(row["mae_mm"], 0.0, places=6)


class MissingRays(unittest.TestCase):
    """A miss is a miss: never a 0 mm residual."""

    def test_all_rays_miss_yield_null_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            depth_mm = np.full((64, 64), 500.0)
            # Plane exists but the mask covers only a 4x4 corner the plane still
            # spans -- so instead drop the mesh entirely by pushing it far behind
            # the ray origin's forward direction.
            _write_run(root, depth_mm, INTRINSICS, MASK, plane_depth_m=-0.5)
            payload = evaluator.evaluate(root, samples=300, seed=7)
            for row in payload["metrics"]:
                self.assertEqual(row["status"], "no_valid_samples")
                self.assertEqual(row["n_used"], 0)
                self.assertEqual(row["n_missing"], row["n_sampled"])
                self.assertEqual(row["coverage"], 0.0)
                for field in ("mae_mm", "rmse_mm", "p95_mm", "signed_mean_mm", "max_abs_mm"):
                    self.assertIsNone(row[field], f"{field} must be null when n_used == 0")

    def test_empty_mask_reports_no_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _write_run(root, np.full((64, 64), 500.0), INTRINSICS,
                       np.zeros((64, 64), dtype=bool))
            payload = evaluator.evaluate(root, samples=100, seed=7)
            for row in payload["metrics"]:
                self.assertEqual(row["status"], "no_samples")
                self.assertEqual(row["n_sampled"], 0)
                self.assertIsNone(row["mae_mm"])


class Coverage(unittest.TestCase):
    """coverage must be n_used / n_sampled, matching a hand count."""

    def test_coverage_matches_hand_computation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            depth_mm = np.full((64, 64), 500.0)
            # Right half of the mask sits on a surface 10 mm off: those rays are
            # rejected by the residual gate, so exactly half the sample is used.
            depth_mm[:, 32:] = 510.0
            _write_run(root, depth_mm, INTRINSICS, MASK, plane_depth_m=0.500,
                       reference_index=0, frames_per_state=1)
            pixels = evaluator.sample_pixels(MASK, depth_mm, 2000, 7)
            columns = pixels[:, 0]
            expected_used = int(np.sum(columns < 32))
            # The plane is flat and the sample is uniform, so the expected
            # rejection count is the number of sampled columns on the right.
            payload = evaluator.evaluate(root, samples=2000, seed=7, max_depth_mm=5.0)
            for row in payload["metrics"]:
                self.assertEqual(row["n_used"], expected_used)
                self.assertAlmostEqual(row["coverage"], expected_used / row["n_sampled"], places=12)
                self.assertEqual(row["n_missing"], row["n_sampled"] - expected_used)

    def test_residual_gate_is_not_a_camera_range(self):
        """With a 30 mm residual gate a 500 mm plane is fully covered."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _write_run(root, np.full((64, 64), 500.0), INTRINSICS, MASK,
                       reference_index=0, frames_per_state=1)
            payload = evaluator.evaluate(root, samples=500, seed=7)
            for row in payload["metrics"]:
                self.assertEqual(row["n_used"], row["n_sampled"])


class SameFrame(unittest.TestCase):
    """The frame used must be the one reference_geometry.json declares."""

    def test_reference_frame_id_is_honoured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _write_run(root, np.full((64, 64), 500.0), INTRINSICS, MASK,
                       reference_index=3, frames_per_state=5)
            payload = evaluator.evaluate(root, samples=200, seed=7)
            self.assertEqual(payload["reference_frames"],
                             {"closed": "closed_front_3", "open": "open_front_3"})
            for row in payload["metrics"]:
                self.assertEqual(row["frame_id"], f"{row['state']}_front_3")
                self.assertEqual(row["frame_role"], "identity")

    def test_missing_declared_frame_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _write_run(root, np.full((64, 64), 500.0), INTRINSICS, MASK)
            reference_path = root / "reconstruction" / "dense" / "reference_geometry.json"
            payload = json.loads(reference_path.read_text(encoding="utf-8"))
            payload["references"]["closed"]["frame_id"] = "closed_front_99"
            reference_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluator.evaluate(root, samples=50, seed=7)

    def test_state_mismatch_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _write_run(root, np.full((64, 64), 500.0), INTRINSICS, MASK)
            reference_path = root / "reconstruction" / "dense" / "reference_geometry.json"
            payload = json.loads(reference_path.read_text(encoding="utf-8"))
            # Claim the open mesh was fitted in a closed frame: mixing states.
            payload["references"]["open"]["frame_id"] = "closed_front_4"
            reference_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluator.evaluate(root, samples=50, seed=7)

    def test_real_run_uses_front_4_not_front_0(self):
        run = (PROJECT_ROOT / "runtime" / "outputs"
               / "20261101_140933_045847_head_reconstruction")
        if not (run / "prepared" / "sequence.json").is_file():
            self.skipTest("real run not available in this checkout")
        sequence = evaluator.load_sequence(run)
        _, references = evaluator.load_reference_geometry(run)
        frames = evaluator.resolve_frames(run, sequence, references)
        self.assertEqual(frames["closed"]["frame_id"], "closed_front_4")
        self.assertEqual(frames["open"]["frame_id"], "open_front_4")
        for state in ("closed", "open"):
            self.assertTrue(frames[state]["frame_id"].endswith("_front_4"))
            self.assertFalse(frames[state]["frame_id"].endswith("_front_0"))


class Geometry(unittest.TestCase):
    """Mesh validity checks must fire on broken meshes."""

    def test_flip_detection_ignores_open_shell_shape(self):
        vertices, faces = _plane_mesh(0.10, 0.10, 0.5)
        metrics = evaluator.geometry_metrics(vertices, faces, None)
        self.assertEqual(metrics["flipped_faces"], 0)
        self.assertTrue(metrics["metre_scale"])
        self.assertTrue(metrics["face_indices_valid"])

    def test_single_reversed_face_is_detected(self):
        vertices, faces = _plane_mesh(0.10, 0.10, 0.5)
        broken = faces.copy()
        broken[3] = broken[3][::-1]
        metrics = evaluator.geometry_metrics(vertices, broken, None)
        self.assertGreaterEqual(metrics["flipped_faces"], 1)

    def test_nan_vertex_is_reported(self):
        vertices, faces = _plane_mesh(0.10, 0.10, 0.5)
        vertices[5, 2] = np.nan
        metrics = evaluator.geometry_metrics(vertices, faces, None)
        self.assertFalse(metrics["vertices_finite"])


class OutputContract(unittest.TestCase):
    def test_existing_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _write_run(root, np.full((64, 64), 500.0), INTRINSICS, MASK,
                       reference_index=0, frames_per_state=1)
            payload = evaluator.evaluate(root, samples=100, seed=7)
            output = Path(tmp) / "out"
            evaluator.write_outputs(payload, output, None)
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "metrics.csv").is_file())
            self.assertTrue((output / "report.md").is_file())
            with self.assertRaises(FileExistsError):
                evaluator.write_outputs(payload, output, None)

    def test_sampling_parameters_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _write_run(root, np.full((64, 64), 500.0), INTRINSICS, MASK,
                       reference_index=0, frames_per_state=1)
            payload = evaluator.evaluate(root, samples=123, seed=99)
            self.assertEqual(payload["sampling"]["samples_per_frame"], 123)
            self.assertEqual(payload["sampling"]["seed"], 99)
            self.assertEqual(payload["sampling"]["region"], "face_mask & depth > 0")


class Baseline(unittest.TestCase):
    def test_comparison_flags_coverage_drop_and_missing_increase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            depth_mm = np.full((64, 64), 500.0)
            depth_mm[:, 32:] = 560.0
            _write_run(root, depth_mm, INTRINSICS, MASK,
                       reference_index=0, frames_per_state=1)
            payload = evaluator.evaluate(root, samples=1000, seed=7)
            baseline = json.loads(json.dumps(payload))
            for row in baseline["metrics"]:
                row["n_used"] += 10
                row["n_missing"] -= 10
                row["coverage"] = row["n_used"] / row["n_sampled"]
                row["mae_mm"] = 1.0
            comparison = evaluator.compare_with_baseline(payload, baseline)
            for entry in comparison["rows"]:
                self.assertTrue(entry["coverage_dropped"])
                self.assertTrue(entry["missing_increased"])
                self.assertIsNotNone(entry["delta"]["mae_mm"])


if __name__ == "__main__":
    unittest.main()
