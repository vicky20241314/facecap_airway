"""Fast contracts that protect acquisition and geometry semantics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from head_rgbd import cli
from head_rgbd.capture_session import parse_view_specs
from head_rgbd.contracts import temporal_edge_allowed
from head_rgbd.paths import (
    asset_paths,
    create_exclusive_dir,
    verify_assets,
)
from head_rgbd.reference_contract import inner_lip_gap_mm, observed_landmarks_m


def _write_manifest(root: Path, expected: str) -> Path:
    manifest = root / "assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "head_rgbd.assets.v1",
                "assets": [
                    {
                        "name": "fixture",
                        "path": "fixture.bin",
                        "sha256": expected,
                        "purpose": "unit test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


class AssetAndOutputContracts(unittest.TestCase):
    def test_asset_verification_detects_missing_and_tampered_files(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            expected = hashlib.sha256(b"trusted").hexdigest()
            manifest = _write_manifest(root, expected)
            paths = asset_paths(root / "runtime")
            model = paths.assets / "fixture.bin"

            missing = verify_assets(paths, manifest)[0]
            self.assertEqual(missing["status"], "missing")

            model.parent.mkdir(parents=True)
            model.write_bytes(b"trusted")
            self.assertEqual(verify_assets(paths, manifest)[0]["status"], "ok")

            model.write_bytes(b"tampered")
            self.assertEqual(
                verify_assets(paths, manifest)[0]["status"], "sha256_mismatch"
            )

    def test_run_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "run"
            self.assertEqual(create_exclusive_dir(target), target.resolve())
            with self.assertRaises(FileExistsError):
                create_exclusive_dir(target)


class TemporalContracts(unittest.TestCase):
    def setUp(self):
        self.previous = {
            "id": "a",
            "status": "candidate",
            "segment": 1,
            "timestamp_s": 10.0,
        }
        self.current = {
            "id": "b",
            "status": "candidate",
            "segment": 1,
            "timestamp_s": 10.1,
            "overlap": {"status": "tracked"},
        }

    def test_only_contiguous_tracked_pairs_are_allowed(self):
        self.assertTrue(temporal_edge_allowed(self.previous, self.current))

        cases = []
        row = {**self.current, "segment": 2}
        cases.append(row)
        row = {**self.current, "timestamp_s": 10.2}
        cases.append(row)
        row = {**self.current, "tracking_barrier": "gap"}
        cases.append(row)
        row = {**self.current, "overlap": {"status": "lost"}}
        cases.append(row)
        row = {**self.current, "status": "excluded"}
        cases.append(row)
        for candidate in cases:
            self.assertFalse(temporal_edge_allowed(self.previous, candidate))


class IdentityAndReferenceContracts(unittest.TestCase):
    def test_fit_summary_preserves_degraded_and_prepared_exclusion_audit(self):
        with tempfile.TemporaryDirectory() as name:
            fit = Path(name)
            (fit / "report.json").write_text(
                json.dumps(
                    {
                        "frames": [
                            {
                                "id": "candidate",
                                "status": "candidate",
                                "degraded": [
                                    "neck_depth_unavailable; head reconstruction continues"
                                ],
                            }
                        ],
                        "prepared_excluded_frames": [
                            {
                                "id": "bad",
                                "status": "excluded",
                                "reason": "face_bbox_not_detected",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = cli._fit_summary(fit)
            self.assertEqual(summary["degraded_frames"][0]["id"], "candidate")
            self.assertEqual(
                summary["excluded_frames"],
                [{"id": "bad", "reason": "face_bbox_not_detected"}],
            )

    def test_guarded_requires_real_closed_oblique_but_research_free_does_not(self):
        without_side = {"has_real_closed_oblique": False}
        with self.assertRaises(ValueError):
            cli._validate_reconstruction_input(without_side, free_identity=False)
        cli._validate_reconstruction_input(without_side, free_identity=True)
        cli._validate_reconstruction_input(
            {"has_real_closed_oblique": True}, free_identity=False
        )

    def test_free_identity_requires_explicit_research_flag(self):
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(free_identity=True, research_only=False)
        with self.assertRaises(SystemExit):
            cli._validate_identity_mode(parser, args)
        args.free_identity = False
        args.research_only = True
        with self.assertRaises(SystemExit):
            cli._validate_identity_mode(parser, args)
        args.free_identity = True
        cli._validate_identity_mode(parser, args)

    def test_missing_inner_lip_depth_stays_null_and_never_becomes_zero(self):
        points = np.zeros((68, 3), dtype=np.float64)
        valid = np.ones(68, dtype=bool)
        points[62] = np.nan
        valid[62] = False
        observed = observed_landmarks_m(points, valid)
        self.assertIsNone(observed[62])
        self.assertIsNone(inner_lip_gap_mm(points, valid))
        self.assertIn('"62": null', json.dumps({str(i): row for i, row in enumerate(observed)}))

    def test_inner_lip_gap_uses_metres_to_millimetres_once(self):
        points = np.zeros((68, 3), dtype=np.float64)
        points[66, 0] = 0.001
        valid = np.ones(68, dtype=bool)
        self.assertAlmostEqual(inner_lip_gap_mm(points, valid), 1.0)


class CaptureAndModelLoadingContracts(unittest.TestCase):
    def test_failed_capture_run_keeps_manifest_audit_record(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            capture = root / "capture"
            capture.mkdir()
            manifest = capture / "session.json"
            manifest.write_text(
                json.dumps({"schema": "head_rgbd.capture.v1", "status": "complete"}),
                encoding="utf-8",
            )
            report_path = root / "run_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema": "head_rgbd.run.v1",
                        "status": "failed",
                        "statuses": {"real_capture_pipeline": "not_run"},
                    }
                ),
                encoding="utf-8",
            )

            cli._attach_capture_report(
                report_path,
                capture,
                validation_status="failed",
                reason="ValueError: face_bbox_not_detected",
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["statuses"]["real_capture_pipeline"], "failed")
            self.assertEqual(report["capture"]["directory"], str(capture))
            self.assertEqual(
                report["capture"]["manifest_sha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                report["capture_validation"]["reason"],
                "ValueError: face_bbox_not_detected",
            )

    def test_capture_requires_front_and_real_oblique_segments(self):
        with self.assertRaises(ValueError):
            parse_view_specs(["closed:front", "open:front"])
        with self.assertRaises(ValueError):
            parse_view_specs(["closed:front", "closed:left", "open:front", "closed:left"])
        self.assertEqual(
            parse_view_specs(["closed:front", "closed:left", "open:front"]),
            [("closed", "front"), ("closed", "left"), ("open", "front")],
        )

    def test_uniface_missing_asset_fails_without_download(self):
        from core import mask_detection

        with tempfile.TemporaryDirectory() as name:
            with mock.patch.dict(
                os.environ, {"UNIFACE_CACHE_DIR": str(Path(name))}, clear=False
            ):
                with self.assertRaises(FileNotFoundError):
                    mask_detection.require_local_parser()


if __name__ == "__main__":
    unittest.main()
