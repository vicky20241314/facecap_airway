"""T3 prepared replay and front-neck position gate."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from head_rgbd import neck_direct as nd  # noqa: E402
from head_rgbd import neck_landmark as nl  # noqa: E402
from head_rgbd import neck_prominence as npn  # noqa: E402
from head_rgbd import neck_stage as ns  # noqa: E402
from head_rgbd.tools import neck_landmark_aruco as nla  # noqa: E402

ZP = PROJECT_ROOT / "runtime/outputs/zp_capwitharuco"
DENSE = ZP / "reconstruction/dense"
PREPARED = ZP / "prepared"


def _subject(state="closed"):
    params = np.load(DENSE / "parameters.npz", allow_pickle=False)
    refs = json.loads((DENSE / "reference_geometry.json").read_text(encoding="utf-8"))["references"]
    stage = refs[state]["stages"]["dense"]
    F = np.asarray(params["dense_faces"], np.int64)
    V = np.asarray(params["dense_vertices"][stage["frame_index"]], float)
    landmarks = np.asarray(stage["landmarks_m"], float)
    _frame, ax, hc = ns.subject_frame(V, F, landmarks)
    curve, _ = ns.spine_curve(V, F, ax, hc)
    return {"V": V, "F": F, "ax": ax, "curve": curve, "h_chin": hc,
            "reference_frame": refs[state]["frame_id"]}


@unittest.skipUnless(DENSE.is_dir() and PREPARED.is_dir(),
                     "zp dense/prepared replay data not present")
class PreparedReplay(unittest.TestCase):
    def test_prepared_replay_writes_observed_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "neck_landmarks.json"
            code = nla.main(["--prepared", str(PREPARED), "--dense", str(DENSE),
                             "--output", str(out)])
            self.assertEqual(code, 0, "both states should be observed on zp")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], nl.LANDMARKS_SCHEMA)
            for state in ("closed", "open"):
                rec = payload["states"][state]
                self.assertEqual(rec["status"], nl.STATUS_OBSERVED, rec.get("reason"))
                self.assertEqual(rec["source"], nl.SOURCE_ARUCO)
                marker = rec["marker"]
                self.assertEqual(marker["target_id"], 0)
                size = marker["size"]
                self.assertEqual(size["size_source"], "measured_from_depth")
                self.assertTrue(8.0 <= size["black_square_mm"] <= 40.0)
                trans = rec["transform"]
                self.assertEqual(trans["status"], nl.STATUS_OBSERVED)
                self.assertEqual(trans["direction"],
                                 "p_reference_camera = R @ p_annotation_camera + t")
                self.assertEqual(rec["reference_frame"],
                                 json.loads((DENSE / "reference_geometry.json").read_text(
                                     encoding="utf-8"))["references"][state]["frame_id"])
                lc = rec["larynx_center"]
                self.assertEqual(lc["source"], "aruco_marker")
                self.assertEqual(lc["theta_used_deg"], 0.0)
                self.assertEqual(lc["t_center_used"], lc["t_marker"])
                self.assertTrue(0.35 <= lc["t_marker"] <= lc["t_gate"] + 0.025)
                self.assertLessEqual(abs(lc["theta_marker_deg"]), 20.0)
                self.assertIsNotNone(rec["point_reference_m"])
                self.assertIsNotNone(rec["skin_estimate_reference_m"])
                self.assertFalse(rec.get("mesh_binding"))


class PositionGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DENSE.is_dir():
            raise unittest.SkipTest("zp dense not present")
        cls.subject = _subject("closed")
        V = cls.subject["V"]
        F = cls.subject["F"]
        ax = cls.subject["ax"]
        _h, th, _r, t = nd.axial_mm(V * 1000.0, ax, cls.subject["curve"],
                                    cls.subject["h_chin"])
        cls.t = t
        cls.theta = np.degrees(th)
        cls.normals = npn.vertex_normals(V, F) @ ax["axis"]

    def _point(self, index):
        return [float(v) for v in self.subject["V"][index]]

    def test_marker_vertex_passes(self):
        result = nl.validate_marker_position(
            self._point(6742), state="closed", **{k: self.subject[k] for k in
                                                  ("V", "F", "ax", "curve", "h_chin")},
            p={"larynx_prior_gate_dn": -0.30})
        self.assertEqual(result["status"], nl.STATUS_OBSERVED, result.get("reason"))
        self.assertLess(result["nearest_surface_distance_mm"], 1e-6)

    def test_t_below_range_rejected(self):
        index = int(np.argmin(np.abs(self.t - 0.20)))
        result = nl.validate_marker_position(
            self._point(index), state="closed",
            **{k: self.subject[k] for k in ("V", "F", "ax", "curve", "h_chin")},
            p={})
        self.assertEqual(result["reason"], "marker_t_below_0_35")

    def test_theta_out_of_range_rejected(self):
        m = (self.t > 0.45) & (self.t < 0.80) & (np.abs(self.theta) > 45.0)
        if not m.any():
            self.skipTest("no lateral vertex in the expected band")
        result = nl.validate_marker_position(
            self._point(int(np.flatnonzero(m)[0])), state="closed",
            **{k: self.subject[k] for k in ("V", "F", "ax", "curve", "h_chin")},
            p={})
        self.assertEqual(result["reason"], "marker_theta_out_of_range")

    def test_normal_gate_rejected(self):
        m = (self.t > 0.55) & (self.t < 0.92) & (np.abs(self.theta) < 20.0) & \
            (self.normals < -0.50)
        if not m.any():
            self.skipTest("no downward-facing anterior vertex in the editable band")
        result = nl.validate_marker_position(
            self._point(int(np.flatnonzero(m)[0])), state="closed",
            **{k: self.subject[k] for k in ("V", "F", "ax", "curve", "h_chin")},
            p={})
        self.assertEqual(result["reason"], "marker_normal_gate_failed")

    def test_far_from_surface_rejected(self):
        base = np.asarray(self._point(6742), float)
        point = base + np.asarray(self.subject["ax"]["anterior"], float) * 0.005
        result = nl.validate_marker_position(
            [float(v) for v in point], state="closed",
            **{k: self.subject[k] for k in ("V", "F", "ax", "curve", "h_chin")},
            p={"marker_max_surface_distance_mm": 2.0})
        self.assertEqual(result["reason"], "marker_far_from_subject_surface")


if __name__ == "__main__":
    unittest.main()