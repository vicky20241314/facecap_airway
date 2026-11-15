"""T5 self-contained delivery: sidecar/manifest contract and not_usable path."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from head_rgbd import delivery as dv  # noqa: E402
from head_rgbd import neck_stage as ns  # noqa: E402

STATES = ("closed", "open")


def _write_mesh(path: Path, count=4):
    V = np.array([[0, 0, 0], [0.01, 0, 0], [0.01, 0.01, 0], [0, 0.01, 0]][:count], float)
    F = np.array([[0, 1, 2], [0, 2, 3]]) if count == 4 else \
        np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
    ns.write_obj(path, V, F, comment="test")


def _record(state):
    return {
        "name": "laryngeal_prominence", "source": "aruco_marker", "state": state,
        "status": "observed", "reason": None,
        "frame_id": f"{state}_front_3",
        "timestamp": {"host_monotonic_s": 1.0},
        "pixel_uv": [100.0, 120.0], "image_size": [1280, 720],
        "point_observed_m": [-0.01, -0.05, -0.48],
        "point_reference_m": [-0.011, -0.051, -0.481],
        "skin_estimate_m": [-0.011, -0.051, -0.480],
        "skin_estimate_reference_m": [-0.012, -0.052, -0.481],
        "reference_frame": f"{state}_front_1",
        "reference_coordinate_system": "reference camera X_right_Y_up_Z_backward",
        "transform": {"status": "observed", "method": "same_frame_observation",
                      "direction": "p_reference_camera = p_observation_camera",
                      "unit": "m", "applies_to": "neck surface observation only",
                      "rotation_3x3": np.eye(3).tolist(),
                      "translation_m": [0, 0, 0]},
        "marker": {"target_id": 0, "dictionary": "DICT_4X4_50",
                   "size": {"black_square_mm": 33.4, "size_source": "measured_from_depth",
                            "size_uncertainty_mm": 0.8},
                   "print": {"black_square_mm": 33.4, "size_source": "measured_from_depth",
                             "size_uncertainty_mm": 0.8, "thickness_mm": 0.1,
                             "thickness_source": "nominal, not measured"},
                   "detected_ids": [0]},
        "marker_plane": {"plane": {"normal": [0.0, 0.0, 1.0]}},
        "quality": {"blur_score": 100.0, "centre_method": "homography_unit_square_centre"},
        "larynx_center": {"t_marker": 0.764, "theta_marker_deg": -3.9,
                          "t_center_used": 0.764, "theta_used_deg": 0.0,
                          "t_gate": 0.90, "source": "aruco_marker"},
        "attempts": [{"frame_id": f"{state}_front_3", "status": "observed",
                      "reason": None}],
    }


class DeliveryContract(unittest.TestCase):
    def _payload(self, observed=True):
        states = {state: _record(state) for state in STATES}
        if not observed:
            states["open"] = {**states["open"], "status": "failed",
                              "reason": "no_marker_detected",
                              "point_reference_m": None,
                              "skin_estimate_reference_m": None,
                              "larynx_center": None}
        return {"schema": "head_rgbd.neck_landmarks.v1", "states": states}

    def test_usable_delivery_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = {state: root / f"final_{state}.obj" for state in STATES}
            measured = {state: root / f"measured_{state}.obj" for state in STATES}
            for state in STATES:
                _write_mesh(final[state], 4)
                _write_mesh(measured[state], 4)
            refs = {state: f"{state}_front_1" for state in STATES}
            info = dv.write_delivery(
                root, final_sources=final, measured_sources=measured,
                landmark_payload=self._payload(), marker_status="usable",
                reference_frames=refs, provenance={"dense": "test"})
            self.assertTrue(info["usable"])
            for role, vertices in (("final", 4), ("measured", 4)):
                d = root / role
                self.assertTrue((d / "head_closed.obj").is_file())
                self.assertTrue((d / "head_open.obj").is_file())
                self.assertTrue((d / "anatomical_landmarks.json").is_file())
                self.assertTrue((d / "manifest.json").is_file())
                side = json.loads((d / "anatomical_landmarks.json").read_text(encoding="utf-8"))
                man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(side["schema"], dv.LANDMARKS_SCHEMA)
                self.assertEqual(side["unit"], "m")
                self.assertEqual(side["effective_marker_mode"], "aruco_marker")
                self.assertFalse(side["observation_is_manual_truth"])
                self.assertEqual(man["schema"], dv.MANIFEST_SCHEMA)
                self.assertEqual(man["status"], "usable")
                self.assertEqual(man["delivery_role"], role)
                self.assertEqual(man["anatomical_landmarks"]["file"],
                                 "anatomical_landmarks.json")
                self.assertEqual(man["anatomical_landmarks"]["schema"],
                                 dv.LANDMARKS_SCHEMA)
                for state in STATES:
                    rec = side["states"][state]
                    self.assertEqual(rec["source"], "aruco_marker")
                    self.assertEqual(rec["main_delivery_mesh_file"],
                                     man["mesh"]["states"][state]["path"])
                    self.assertEqual(rec["reference_frame"],
                                     man["mesh"]["states"][state]["reference_frame"])
                    self.assertEqual(rec["transform"]["status"], "observed")
                    self.assertIsNotNone(rec["skin_estimate_reference_m"])
                    self.assertEqual(rec["skin_estimate"]["thickness_mm"], 0.1)
                    self.assertFalse(rec["skin_estimate"]["is_direct_measurement"])
                    self.assertEqual(rec["skin_estimate"]["normal_unit"], [0.0, 0.0, 1.0])
                    self.assertEqual(rec["marker"]["print"]["size_source"],
                                     "measured_from_depth")
                    self.assertEqual(rec["larynx_center"]["theta_used_deg"], 0.0)
                    self.assertEqual(rec["larynx_center"]["t_center_used"], 0.764)
                    self.assertEqual(rec["mesh_binding"]["status"], "not_completed")
                    self.assertEqual(rec["main_head_mesh_binding"]["status"],
                                     "not_completed")
                    self.assertFalse(rec["main_mesh_binding_completed"])
                    self.assertEqual(man["mesh"]["states"][state]["vertices"], vertices)
                self.assertTrue(man["mesh"]["topology_shared"])

    def test_not_usable_writes_no_pair_able_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = {state: root / f"final_{state}.obj" for state in STATES}
            measured = {state: root / f"measured_{state}.obj" for state in STATES}
            for state in STATES:
                _write_mesh(final[state], 4)
                _write_mesh(measured[state], 4)
            refs = {state: f"{state}_front_1" for state in STATES}
            info = dv.write_delivery(
                root, final_sources=final, measured_sources=measured,
                landmark_payload=self._payload(observed=False),
                marker_status="not_usable", reference_frames=refs,
                provenance={})
            self.assertFalse(info["usable"])
            for role in ("final", "measured"):
                d = root / role
                self.assertTrue((d / "head_closed.obj").is_file())
                self.assertTrue((d / "manifest.json").is_file())
                self.assertFalse((d / "anatomical_landmarks.json").exists())
                man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(man["status"], "not_usable")
                self.assertIsNone(man["anatomical_landmarks"]["file"])

    def test_marker_status_not_usable_overrides_observed_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = {state: root / f"final_{state}.obj" for state in STATES}
            measured = {state: root / f"measured_{state}.obj" for state in STATES}
            for state in STATES:
                _write_mesh(final[state], 4)
                _write_mesh(measured[state], 4)
            info = dv.write_delivery(
                root, final_sources=final, measured_sources=measured,
                landmark_payload=self._payload(), marker_status="not_usable",
                reference_frames={state: f"{state}_front_1" for state in STATES},
                provenance={})
            self.assertFalse(info["usable"])
            self.assertFalse((root / "final" / "anatomical_landmarks.json").exists())
            man = json.loads((root / "final" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(man["status"], "not_usable")


class BaseDeliveryEligibilityTests(unittest.TestCase):
    def test_delivery_with_applied_neck_volume_is_not_a_base(self):
        from head_rgbd import rebuild as rb

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "neck_prominence.json").write_text(json.dumps({
                "mode": "off",
                "states": {"closed": {"status": "applied"},
                           "open": {"status": "applied"}}}), encoding="utf-8")
            self.assertFalse(rb._delivery_base_only(root))

    def test_skipped_or_missing_volume_record_is_a_base(self):
        from head_rgbd import rebuild as rb

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertTrue(rb._delivery_base_only(root))
            (root / "neck_prominence.json").write_text(json.dumps({
                "mode": "off",
                "states": {"closed": {"status": "skipped"},
                           "open": {"status": "skipped"}}}), encoding="utf-8")
            self.assertTrue(rb._delivery_base_only(root))


if __name__ == "__main__":
    unittest.main()