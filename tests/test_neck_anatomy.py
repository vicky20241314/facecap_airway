"""T4 larynx prior centre / gain envelope, pure-function tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from head_rgbd import neck_anatomy as na  # noqa: E402
from tools import build_neck_direct_v2 as bnd  # noqa: E402


def _anat(t_center=0.80, t_gate=0.85):
    return {
        "t_center": t_center, "t_half": 0.045, "t_gate": t_gate,
        "amp_mm": 1.7, "theta_sigma_deg": 10.0, "theta_zero_deg": 45.0,
        "theta_zero_hollow_deg": 55.0, "hollow_deg": 28.0,
        "hollow_sigma_deg": 10.0, "hollow_ratio": 0.35,
    }


class LarynxPriorWindow(unittest.TestCase):
    def test_peak_is_at_marker_centre(self):
        t = np.linspace(0.40, 1.00, 241)
        th = np.linspace(-0.7, 0.7, 81)
        field = na.larynx_prior_field(t, th, _anat(t_center=0.80))
        column = field[:, int(np.argmin(np.abs(th)))]
        self.assertAlmostEqual(t[int(np.argmax(column))], 0.80, delta=0.01)
        self.assertGreater(column.max(), 1.0)

    def test_window_is_clipped_above_the_gate(self):
        t = np.linspace(0.40, 1.00, 241)
        th = np.array([0.0])
        field = na.larynx_prior_field(t, th, _anat(t_center=0.83, t_gate=0.85))
        above = field[t >= 0.88, 0]
        self.assertLess(np.max(np.abs(above)), 1e-9)
        at_gate = float(field[np.argmin(np.abs(t - 0.85)), 0])
        peak = float(field[np.argmin(np.abs(t - 0.83)), 0])
        self.assertLess(abs(at_gate), abs(peak))


class LarynxGainCentre(unittest.TestCase):
    def test_gain_envelope_follows_the_state_centre(self):
        t = np.linspace(0.50, 0.95, 181)
        th = np.linspace(-0.6, 0.6, 61)
        tg = {"t": t, "theta": th,
              "applied": np.ones((len(t), len(th))),
              "R": np.ones((len(t), len(th)))}
        p = {"larynx_gain": 1.5, "larynx_t_center": 0.755,
             "larynx_t_half": 0.08, "larynx_theta_full": 25.0,
             "larynx_theta_zero": 40.0}
        out = bnd._apply_larynx_gain(dict(tg), p, t_center=0.85)
        j = int(np.argmin(np.abs(th)))
        profile = out["applied"][:, j]
        self.assertAlmostEqual(t[int(np.argmax(profile))], 0.85, delta=0.01)
        self.assertAlmostEqual(profile.max(), 1.5, delta=1e-9)
        old = bnd._apply_larynx_gain(dict(tg), p, t_center=0.755)
        self.assertAlmostEqual(old["applied"][:, j].max(), 1.5, delta=1e-9)
        self.assertLess(abs(float(profile[np.argmin(np.abs(t - 0.755))]) - 1.0),
                        abs(float(old["applied"][np.argmin(np.abs(t - 0.755)), j]) - 1.0))


class MarkerCentreOverride(unittest.TestCase):
    def test_marker_t_overrides_each_state_and_theta_is_zero(self):
        from pathlib import Path
        from tools import build_neck_stage as bns
        from head_rgbd import neck_stage as ns

        dense = Path(__file__).resolve().parents[1] / \
            "runtime/outputs/zp_capwitharuco/reconstruction/dense"
        if not dense.is_dir():
            self.skipTest("zp dense not present")
        states_raw, F, _refs = bns.load_subject(dense)
        st = {}
        for key in ("closed", "open"):
            V = states_raw[key]["V"]
            _frame, ax, hc = ns.subject_frame(V, F, states_raw[key]["landmarks"])
            curve, _ = ns.spine_curve(V, F, ax, hc)
            st[key] = {"V": V, "ax": ax, "curve": curve, "h_chin": hc}
        landmarks = {
            "closed": {"status": "observed",
                       "frame_id": "closed_front_3",
                       "reference_frame": "closed_front_1",
                       "larynx_center": {"t_marker": 0.700, "t_center_used": 0.700,
                                        "theta_marker_deg": -3.0, "t_gate": 0.90},
                       "marker": {"target_id": 0,
                                  "size": {"black_square_mm": 33.4,
                                           "size_source": "measured_from_depth",
                                           "size_uncertainty_mm": 0.8}}},
            "open": {"status": "observed",
                     "frame_id": "open_front_2",
                     "reference_frame": "open_front_4",
                     "larynx_center": {"t_marker": 0.805, "t_center_used": 0.805,
                                      "theta_marker_deg": -4.0, "t_gate": 0.95},
                     "marker": {"target_id": 0,
                                "size": {"black_square_mm": 33.0,
                                         "size_source": "measured_from_depth",
                                         "size_uncertainty_mm": 0.8}}},
        }
        p = {"larynx_prior_ratio": 0.35, "jaw_reference_mm": 4.857,
             "larynx_prior_gate_dn": -0.30, "larynx_prior_gate_fallback": 0.78}
        out = bnd._larynx_anatomy_by_state(st, F, states_raw, landmarks, p)
        self.assertEqual(out["closed"]["t_center"], 0.700)
        self.assertEqual(out["open"]["t_center"], 0.805)
        for state in ("closed", "open"):
            self.assertEqual(out[state]["t_center_source"], "aruco_marker")
            self.assertEqual(out[state]["t_center_used"], out[state]["t_marker"])
            self.assertEqual(out[state]["theta_used_deg"], 0.0)
            self.assertAlmostEqual(out[state]["amp_mm"], 0.35 * 4.857, delta=1e-9)
        # a failed marker record falls back to the menton gate, not to a fake t
        failed = dict(landmarks)
        failed["open"] = {**landmarks["open"], "status": "failed",
                          "reason": "no_marker_detected", "larynx_center": None}
        out2 = bnd._larynx_anatomy_by_state(st, F, states_raw, failed, p)
        self.assertEqual(out2["open"]["t_center_source"], "menton_gate")
        self.assertIsNone(out2["open"]["marker"])
        self.assertEqual(out2["closed"]["t_center"], 0.700)


if __name__ == "__main__":
    unittest.main()