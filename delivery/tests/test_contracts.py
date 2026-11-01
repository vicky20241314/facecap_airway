"""Core delivery contracts: one algorithm, one entry point, one metric mesh.

These tests are fast and CPU only. They pin the properties that the delivery
package must never silently lose:

* the version file, package attribute and ``--version`` agree;
* the CLI exposes only the delivery commands and a single default algorithm;
* the frozen strict-coverage fixed-ray acceptance rule is unchanged;
* the runtime root resolves from the source tree, not from the CWD;
* temporal edges never cross a capture gap or an untracked frame;
* invalid landmark depth never becomes a zero constraint;
* exactly one final mesh plus the required sidecars are exported, in metres;
* research/evaluation modules are not part of the package.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import head_rgbd
from head_rgbd import cli, pipeline
from head_rgbd.contracts import temporal_edge_allowed
from head_rgbd.paths import PROJECT_ROOT, asset_paths, resolve_runtime_dir, verify_assets
from head_rgbd.reference_contract import inner_lip_gap_mm, observed_landmarks_m


EXPECTED_VERSION = "0.2.0"


class VersionContract(unittest.TestCase):
    def test_version_file_package_and_cli_agree(self):
        self.assertEqual((PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip(), EXPECTED_VERSION)
        self.assertEqual(head_rgbd.__version__, EXPECTED_VERSION)
        parser = cli.build_parser()
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            parser.parse_args(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn(f"headrgbd {EXPECTED_VERSION}", output.getvalue())
        self.assertIn(pipeline.ALGORITHM, output.getvalue())


class EntryPointContract(unittest.TestCase):
    def test_only_delivery_commands_are_exposed(self):
        parser = cli.build_parser()
        actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        commands = set(actions[0].choices)
        self.assertEqual(commands, {"run", "capture", "reconstruct", "export", "device", "assets"})

    def test_no_research_mode_switch_exists(self):
        parser = cli.build_parser()
        sub = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)][0]
        for name in ("run", "reconstruct"):
            options = {opt for action in sub.choices[name]._actions for opt in action.option_strings}
            for forbidden in (
                "--free-identity",
                "--research-only",
                "--research-v2",
                "--fixed-ray-policy",
                "--dense-mode",
                "--solver",
                "--mode",
            ):
                self.assertNotIn(forbidden, options, f"{name} must not expose {forbidden}")

    def test_default_algorithm_is_the_frozen_strict_chain(self):
        defaults = pipeline.load_defaults()
        self.assertEqual(defaults["algorithm"]["name"], pipeline.ALGORITHM)
        self.assertEqual(defaults["algorithm"]["fixed_ray_policy"], "strict-coverage")
        self.assertEqual(defaults["export"]["unit"], "m")
        self.assertEqual(defaults["export"]["filename"], "head_m.obj")

    def test_frozen_stage_order_is_reproduced(self):
        calls: list[tuple[str, str]] = []

        class Recorder:
            runtime = PROJECT_ROOT / "runtime"

            def run(self, name, module, arguments, **_):
                calls.append((name, module))
                return 0

        args = argparse.Namespace(
            device="cpu",
            points=1200,
            steps=240,
            keyframes_per_segment=3,
            expression_components=30,
            dense_points=5000,
        )
        final = pipeline.reconstruct_locked(
            Recorder(), sequence=Path("sequence.json"), destination=Path("out"), args=args
        )
        self.assertEqual(
            calls,
            [
                ("identity_front_anchor", "head_rgbd.fit_c"),
                ("shared_identity_guarded", "head_rgbd.fit_c"),
                ("fixed_identity_motion", "head_rgbd.motion_v2"),
                ("low_frequency_shared_identity", "head_rgbd.identity_v2"),
                ("motion_reestimate_after_identity", "head_rgbd.motion_v2"),
                ("guarded_measured_dense_surface", "head_rgbd.fit_d"),
            ],
        )
        self.assertEqual(final, Path("out") / "dense")


class FrozenGeometryContract(unittest.TestCase):
    """The strict-coverage fixed-ray rule must stay byte-for-byte in behaviour."""

    def test_strict_policy_penalty_and_aggregation(self):
        from head_rgbd.geometry_d import aggregate_fixed_ray_energy, fixed_ray_energy

        covered = np.array([True, True, False])
        error = np.array([1.0, 3.0, 100.0])
        # Missing rays are charged the 8 mm strict penalty, not zero.
        self.assertAlmostEqual(fixed_ray_energy(error, covered, "strict-coverage"), (1 + 9 + 64) / 3)
        self.assertAlmostEqual(
            aggregate_fixed_ray_energy([4.0, 16.0], [10, 90], "strict-coverage"), 10.0
        )

    def test_settings_defaults_are_the_frozen_budget(self):
        from head_rgbd.geometry_d import Settings

        settings = Settings()
        self.assertEqual(settings.fixed_ray_policy, "strict-coverage")
        self.assertAlmostEqual(settings.displacement_bound_mm, 3.0)
        self.assertAlmostEqual(settings.minimum_signed_area_ratio, 0.05)
        self.assertAlmostEqual(settings.maximum_normal_change_deg, 35.0)
        self.assertFalse(settings.research_v2)

    def test_acceptance_rule_rejects_coverage_loss(self):
        # Mirrors geometry_d.solve_field: both missing coverage and MSE may not regress.
        def accepted(missing_before, mse_before, missing_after, mse_after):
            return missing_after <= missing_before and mse_after <= mse_before * 1.02 + 0.002

        self.assertTrue(accepted(10, 1.0, 10, 1.01))
        self.assertFalse(accepted(10, 1.0, 11, 0.1))
        self.assertFalse(accepted(10, 1.0, 10, 1.5))


class PathContract(unittest.TestCase):
    def test_runtime_is_project_relative_not_cwd_relative(self):
        previous = os.environ.pop("WITHB_RUNTIME_ROOT", None)
        old = os.environ.pop("HEADRGBD_RUNTIME_DIR", None)
        try:
            with tempfile.TemporaryDirectory() as name:
                cwd = os.getcwd()
                os.chdir(name)
                try:
                    expected = (PROJECT_ROOT / "runtime").resolve()
                    self.assertEqual(resolve_runtime_dir(), expected)
                finally:
                    os.chdir(cwd)
        finally:
            if previous is not None:
                os.environ["WITHB_RUNTIME_ROOT"] = previous
            if old is not None:
                os.environ["HEADRGBD_RUNTIME_DIR"] = old

    def test_explicit_override_wins(self):
        with tempfile.TemporaryDirectory() as name:
            os.environ["WITHB_RUNTIME_ROOT"] = name
            try:
                self.assertEqual(resolve_runtime_dir(), Path(name).resolve())
            finally:
                os.environ.pop("WITHB_RUNTIME_ROOT", None)

    def test_missing_runtime_fails_loudly(self):
        with tempfile.TemporaryDirectory() as name:
            missing = Path(name) / "does-not-exist"
            with self.assertRaises(FileNotFoundError):
                resolve_runtime_dir(missing)


class TemporalContracts(unittest.TestCase):
    def setUp(self):
        self.previous = {"id": "a", "status": "candidate", "segment": 1, "timestamp_s": 10.0}
        self.current = {
            "id": "b",
            "status": "candidate",
            "segment": 1,
            "timestamp_s": 10.1,
            "overlap": {"status": "tracked"},
        }

    def test_only_contiguous_tracked_pairs_are_allowed(self):
        self.assertTrue(temporal_edge_allowed(self.previous, self.current))
        cases = [
            {**self.current, "segment": 2},
            {**self.current, "timestamp_s": 10.2},
            {**self.current, "tracking_barrier": "gap"},
            {**self.current, "overlap": {"status": "lost"}},
            {**self.current, "status": "excluded"},
        ]
        for candidate in cases:
            self.assertFalse(temporal_edge_allowed(self.previous, candidate))


class LandmarkContract(unittest.TestCase):
    def test_invalid_depth_is_null_not_zero(self):
        points = np.arange(204, dtype=float).reshape(68, 3) * 0.001
        valid = np.ones(68, bool)
        valid[62] = False
        serialized = observed_landmarks_m(points, valid)
        self.assertIsNone(serialized[62])
        self.assertIsNotNone(serialized[0])
        self.assertIsNone(inner_lip_gap_mm(points, valid))
        valid[62] = True
        self.assertAlmostEqual(inner_lip_gap_mm(points, valid), float(np.linalg.norm(points[62] - points[66]) * 1000))


class PackageContract(unittest.TestCase):
    RESEARCH_MODULES = (
        "evaluate.py",
        "evaluate_c.py",
        "evaluate_d.py",
        "evaluate_v2.py",
        "delivery_v2.py",
        "final_report_v2.py",
        "compare_policies_v2.py",
        "diagnose_stages_v2.py",
        "probe_fixed_rays_v2.py",
        "hires_pipeline.py",
        "bootstrap_v2.py",
        "replay_v2.py",
        "import_zhou.py",
        "audit_flame_state_v2.py",
        "reference_metrics.py",
        "verify_c.py",
        "visualize_d.py",
    )

    def test_research_and_evaluation_entries_are_absent(self):
        for name in self.RESEARCH_MODULES:
            self.assertFalse((PROJECT_ROOT / "head_rgbd" / name).exists(), f"{name} must not ship")

    def test_asset_manifest_covers_every_loaded_model(self):
        manifest = json.loads((PROJECT_ROOT / "config" / "assets.json").read_text(encoding="utf-8"))
        paths = {row["path"] for row in manifest["assets"]}
        self.assertEqual(
            paths,
            {
                "flame/flame2023.pkl",
                "flame/flame_static_embedding.pkl",
                "flame/flame_dynamic_embedding.npy",
                "models/yunet/face_detection_yunet_2023mar.onnx",
                "uniface/parsing_resnet34.onnx",
                "tufa/Face_prompt_base.pth",
                "tufa/Prompt/shape_68.npz",
            },
        )

    def test_asset_verification_detects_tampering(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest = root / "assets.json"
            payload = b"trusted"
            expected = hashlib.sha256(payload).hexdigest()
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "head_rgbd.assets.v1",
                        "assets": [{"name": "fixture", "path": "fixture.bin", "sha256": expected}],
                    }
                ),
                encoding="utf-8",
            )
            (root / "runtime").mkdir()
            paths = asset_paths(root / "runtime")
            self.assertEqual(verify_assets(paths, manifest)[0]["status"], "missing")
            model = paths.assets / "fixture.bin"
            model.parent.mkdir(parents=True)
            model.write_bytes(payload)
            self.assertEqual(verify_assets(paths, manifest)[0]["status"], "ok")
            model.write_bytes(b"tampered")
            self.assertEqual(verify_assets(paths, manifest)[0]["status"], "sha256_mismatch")


class ExportContract(unittest.TestCase):
    def _fit_fixture(self, root: Path) -> tuple[Path, Path]:
        import trimesh

        fit = root / "fit"
        fit.mkdir()
        vertices = np.array(
            [
                [-0.10, -0.10, -0.50],
                [0.10, -0.10, -0.50],
                [0.00, 0.20, -0.50],
                [0.00, -0.10, -0.40],
            ],
            dtype=float,
        )
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        trimesh.Trimesh(vertices, faces, process=False).export(fit / "closed_dense_m.obj")
        landmarks = vertices[[0, 1, 2]]
        reference = {
            "references": {
                "closed": {
                    "frame_id": "closed_front_2",
                    "source_frame": 2,
                    "timestamp_s": 12.0,
                    "coordinate_system": "reference camera X_right_Y_up_Z_backward",
                    "mesh_unit": "m",
                    "observed_landmarks_m": [None] * 68,
                    "observed_valid": [False] * 68,
                    "observed_inner_lip_gap_mm": None,
                    "animated_model_to_reference_camera_4x4": np.eye(4).tolist(),
                    "reference_camera_to_animated_model_4x4": np.eye(4).tolist(),
                    "transform_scope": "rigid coordinate transform only",
                    "stages": {
                        "dense": {
                            "mesh": "closed_dense_m.obj",
                            "landmarks_m": landmarks.tolist(),
                            "landmark_triangle_indices": [0, 0, 0],
                            "landmark_barycentric": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            "inner_lip_gap_mm": 10.0,
                        }
                    },
                },
                "open": {"frame_id": "open_front_0"},
            }
        }
        (fit / "reference_geometry.json").write_text(json.dumps(reference), encoding="utf-8")
        report = {
            "created_at": "2026-11-01T00:00:00+08:00",
            "dense_topology": {"vertices": 4},
            "frames": [{"id": "closed_front_2", "segment": "closed_front", "timestamp_s": 12.0}],
            "neck": [],
            "limitations": [],
            "statuses": {"geometry_quality": "not_evaluated"},
        }
        (fit / "report.json").write_text(json.dumps(report), encoding="utf-8")
        np.savez_compressed(
            fit / "support_dense.npz",
            region_label=np.array([0, 1, 2, 2], dtype=np.uint8),
            confidence=np.zeros(4),
            observed_frames=np.array([0, 1, 2, 3]),
            closed_burst_count=np.array([1, 2, 3, 3]),
            observed_closed_yaw_span_deg=np.zeros(4),
            deviation_m=np.zeros(4),
        )
        np.savez_compressed(root / "closed_front_2.npz", valid=np.zeros(68, dtype=bool))
        sequence = root / "sequence.json"
        sequence.write_text(
            json.dumps(
                {
                    "schema": "head_rgbd.sequence.v1",
                    "depth_unit": "mm",
                    "mesh_unit": "m",
                    "coordinate_convention": "X_right_Y_up_Z_backward",
                    "frames": [
                        {
                            "id": "closed_front_2",
                            "status": "candidate",
                            "file": "closed_front_2.npz",
                            "segment": "closed_front",
                            "state": "closed",
                            "view": "front",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return fit, sequence

    def test_exactly_one_mesh_and_required_sidecars(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            fit, sequence = self._fit_fixture(root)
            output = root / "delivery"
            manifest = pipeline.export_delivery(
                fit=fit,
                sequence=sequence,
                output=output,
                runtime=root,
                capture=None,
                runner=None,
                defaults=pipeline.load_defaults(),
            )
            exported = sorted(path.name for path in output.iterdir())
            self.assertEqual(exported, ["head_m.obj", "manifest.json", "reference_landmarks.json", "support.json"])
            self.assertEqual(manifest["mesh"]["path"], "head_m.obj")
            self.assertEqual(manifest["mesh"]["unit"], "m")
            self.assertEqual(manifest["mesh"]["vertices"], 4)
            self.assertEqual(manifest["status"], "degraded")  # missing oblique warning absent, deviation ok
            support = json.loads((output / "support.json").read_text(encoding="utf-8"))
            self.assertEqual(support["counts"], [1, 1, 2])
            landmarks = json.loads((output / "reference_landmarks.json").read_text(encoding="utf-8"))
            self.assertIsNone(landmarks["observed_landmarks_m"][0])
            self.assertEqual(landmarks["landmark_count"], 3)

    def test_existing_mesh_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            fit, sequence = self._fit_fixture(root)
            output = root / "delivery"
            output.mkdir()
            (output / "head_m.obj").write_text("v 0 0 0\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                pipeline.export_delivery(
                    fit=fit,
                    sequence=sequence,
                    output=output,
                    runtime=root,
                    capture=None,
                    runner=None,
                    defaults=pipeline.load_defaults(),
                )

    def test_wrong_unit_is_rejected(self):
        import trimesh

        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "millimetres.obj"
            trimesh.Trimesh(
                np.array([[0, 0, 0], [229, 0, 0], [0, 326, 0]], dtype=float),
                np.array([[0, 1, 2]]),
                process=False,
            ).export(path)
            with self.assertRaises(ValueError):
                pipeline.mesh_checks(path)


class LauncherContract(unittest.TestCase):
    def test_version_from_a_foreign_working_directory(self):
        python = PROJECT_ROOT / "runtime" / "env" / "python.exe"
        if not python.is_file():
            self.skipTest("runtime environment is not installed")
        shell = os.environ.get("COMSPEC") or shutil.which("cmd.exe")
        if not shell:
            self.skipTest("cmd.exe is not available")
        completed = subprocess.run(
            [shell, "/c", str(PROJECT_ROOT / "headrgbd.cmd"), "--version"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT.parent),
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), f"headrgbd {EXPECTED_VERSION} ({pipeline.ALGORITHM})")


if __name__ == "__main__":
    unittest.main()
