"""ArUco laryngeal marker: detection, centre, depth binding, print assets."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from head_rgbd import aruco_marker as am  # noqa: E402
from head_rgbd.prepare import backproject  # noqa: E402

SPEC = am.ArucoSpec()
INTR = np.array([849.22216796875, 849.1190795898438, 640.1256103515625, 361.0])
CFG = {"min_px_per_cell": 3.0, "min_side_px": 18.0, "min_area_px2": 400.0}


def render(spec=SPEC, px_per_mm=24.0, canvas=900):
    """Sticker on a white canvas; returns (rgb, corners-of-black-square)."""
    sticker = am.sticker_image(spec, px_per_mm)
    page = np.full((canvas, canvas), 255, np.uint8)
    side = sticker.shape[0]
    y0 = (canvas - side) // 2
    x0 = (canvas - side) // 2
    page[y0:y0 + side, x0:x0 + side] = sticker
    margin = spec.white_margin_mm * px_per_mm
    square = spec.black_square_mm * px_per_mm
    corners = np.array([[x0 + margin, y0 + margin],
                        [x0 + margin + square, y0 + margin],
                        [x0 + margin + square, y0 + margin + square],
                        [x0 + margin, y0 + margin + square]], np.float64)
    return np.dstack([page] * 3), corners


def depth_of(size, value_mm, centre=None, hole=0, jump=None):
    depth = np.full((size, size), int(value_mm), np.uint16)
    if centre is not None:
        cy, cx = int(round(centre[1])), int(round(centre[0]))
        if hole:
            depth[cy - hole:cy + hole + 1, cx - hole:cx + hole + 1] = 0
        if jump is not None:
            depth[cy - 2:cy + 3, cx - 2:cx - 1] = int(jump)
    return depth


class Detection(unittest.TestCase):
    def test_target_id_is_accepted(self):
        rgb, corners = render()
        result = am.detect_candidates(rgb, SPEC, CFG)
        self.assertEqual(result["rejection"], None)
        self.assertEqual(result["selected"]["detected_id"], 0)
        self.assertEqual(len(result["candidates"]), 1)

    def test_wrong_id_is_rejected(self):
        other = am.ArucoSpec(target_id=7)
        rgb, _ = render(spec=other)
        result = am.detect_candidates(rgb, SPEC, CFG)
        self.assertIsNone(result["selected"])
        self.assertEqual(result["rejection"], "target_id_not_found")
        self.assertEqual(result["candidates"][0]["rejection"], "wrong_id")

    def test_duplicate_target_id_is_rejected(self):
        rgb, _ = render(canvas=1400)
        small = am.sticker_image(SPEC, 12.0)   # 264 px across, clear of the centred copy
        rgb[30:30 + small.shape[0], 30:30 + small.shape[1]] = np.dstack([small] * 3)
        result = am.detect_candidates(rgb, SPEC, CFG)
        self.assertIsNone(result["selected"])
        self.assertEqual(result["rejection"], "duplicate_target_id")

    def test_no_marker(self):
        result = am.detect_candidates(np.full((600, 600, 3), 255, np.uint8), SPEC, CFG)
        self.assertIsNone(result["selected"])
        self.assertEqual(result["rejection"], "no_marker_detected")

    def test_pitch_gate_refuses_a_marker_below_the_limit(self):
        # 0.9 px/mm on a 16 mm square is 2.4 px per cell, below the 3.0 gate.
        rgb, _ = render(px_per_mm=0.9, canvas=60)
        result = am.detect_candidates(rgb, SPEC, CFG)
        if result["candidates"]:
            self.assertIsNone(result["selected"])
            self.assertEqual(result["rejection"], "target_marker_failed_quality")
        else:
            self.assertEqual(result["rejection"], "no_marker_detected")

    def test_12mm_at_470mm_projection_is_at_the_pitch_limit(self):
        # 12 mm across 6 cells at fx=849, d=470 mm is 3.6 px per cell: above the
        # gate, so the smallest candidate is admissible on paper. Whether it is
        # admissible on a print is a field question, not this test's claim.
        rgb, _ = render(spec=am.ArucoSpec(black_square_mm=12.0), px_per_mm=1.35, canvas=60)
        result = am.detect_candidates(rgb, SPEC, {**CFG, "min_px_per_cell": 3.0})
        self.assertAlmostEqual(3.6, 12.0 * 849.22 / 470.0 / 6.0, places=1)


class Centre(unittest.TestCase):
    def test_centre_is_homography_not_corner_mean(self):
        quad = np.array([[100.0, 120.0], [300.0, 100.0], [310.0, 330.0], [90.0, 340.0]])
        centre = np.array(am.homography_center(quad))
        mean = quad.mean(axis=0)
        self.assertGreater(np.linalg.norm(centre - mean), 0.5)
        # The homography centre is the projection of the canonical centre.
        source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        matrix = cv2.getPerspectiveTransform(source, quad.astype(np.float32))
        expected = cv2.perspectiveTransform(np.array([[[0.5, 0.5]]], np.float32), matrix)[0, 0]
        np.testing.assert_allclose(centre, expected, atol=1e-6)

    def test_centre_survives_rotation_and_perspective(self):
        rgb, corners = render()
        # Move the detected quad to a rotated, keystoned target: this is the real
        # 2D projection of a tilted sticker, not a rotation of the image plane.
        target = np.array([[420.0, 260.0], [880.0, 330.0], [820.0, 780.0], [390.0, 700.0]],
                          np.float32)
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
        warped = cv2.warpPerspective(rgb, matrix, (1200, 1200), flags=cv2.INTER_NEAREST,
                                     borderValue=(255, 255, 255))
        result = am.detect_candidates(warped, SPEC, CFG)
        self.assertIsNotNone(result["selected"])
        self.assertEqual(result["selected"]["geometry"]["centre_method"], "homography_unit_square_centre")


class Projection(unittest.TestCase):
    def test_backprojection_matches_project_convention(self):
        depth = np.full((720, 1280), 470, np.uint16)
        depth[300, 500] = 470
        ours = am.backproject_centre([500.0, 300.0], 470.0, INTR)
        theirs, ok = backproject(np.array([[500.0, 300.0]]), depth, INTR)
        self.assertTrue(bool(ok[0]))
        np.testing.assert_allclose(ours, theirs[0], atol=1e-6)
        self.assertLess(ours[2], 0.0)

    def test_principal_point_maps_to_optical_axis(self):
        point = am.backproject_centre([INTR[2], INTR[3]], 470.0, INTR)
        np.testing.assert_allclose(point, [0.0, 0.0, -0.47], atol=1e-9)


class DepthBinding(unittest.TestCase):
    def test_hole_at_centre_fails(self):
        rgb, corners = render()
        centre = am.homography_center(corners)
        depth = depth_of(900, 470, centre, hole=3)
        check = am.centre_depth(centre, depth, INTR, CFG, SPEC)
        self.assertEqual(check["status"] != "ok", True)
        self.assertIn(check["reason"], ("depth_hole_at_centre", "depth_hole_at_centre_pixel"))

    def test_jump_at_centre_fails(self):
        rgb, corners = render()
        centre = am.homography_center(corners)
        depth = depth_of(900, 470, centre, jump=520)
        check = am.centre_depth(centre, depth, INTR, CFG, SPEC)
        self.assertEqual(check["status"] != "ok", True)
        self.assertIn(check["reason"], ("depth_jump_at_centre", "depth_not_single_surface"))

    def test_clean_depth_passes(self):
        rgb, corners = render()
        centre = am.homography_center(corners)
        depth = depth_of(900, 470, centre)
        check = am.centre_depth(centre, depth, INTR, CFG, SPEC)
        self.assertEqual(check["status"], "ok")
        self.assertAlmostEqual(check["depth_mm"], 470.0, places=3)

    def test_centre_outside_image(self):
        depth = depth_of(400, 470)
        check = am.centre_depth([900.0, 900.0], depth, INTR, CFG, SPEC)
        self.assertEqual(check["reason"], "centre_outside_image")


class FrameRecord(unittest.TestCase):
    def test_closed_and_open_are_separate_records(self):
        rgb, corners = render()
        centre = am.homography_center(corners)
        depth = depth_of(900, 470, centre)
        closed = am.evaluate_frame(rgb, depth, INTR, SPEC, state="closed", frame_id="closed_front_0")
        opened = am.evaluate_frame(rgb, depth, INTR, SPEC, state="open", frame_id="open_front_0")
        self.assertEqual(closed["status"], am.STATUS_OBSERVED)
        self.assertEqual(opened["status"], am.STATUS_OBSERVED)
        self.assertNotEqual(closed["frame_id"], opened["frame_id"])
        self.assertNotEqual(closed["depth_sha256"], "open")
        for key in ("marker_type", "dictionary", "target_id", "print", "candidates", "detector"):
            self.assertIn(key, closed["marker"])
        for key in ("corners_px", "centre_px", "point_observed_m", "depth_check",
                    "transform_source", "rgb_sha256", "depth_sha256", "intrinsics_sha256"):
            self.assertIn(key, closed)
        self.assertFalse(closed["observation_is_direct_measurement"])
        self.assertIsNotNone(closed["skin_estimate_m"])

    def test_missing_marker_produces_a_reason_not_a_fallback(self):
        record = am.evaluate_frame(np.full((600, 600, 3), 255, np.uint8),
                                   np.full((600, 600), 470, np.uint16), INTR, SPEC,
                                   state="closed", frame_id="x")
        self.assertEqual(record["status"], am.STATUS_FAILED)
        self.assertEqual(record["reason"], "no_marker_detected")
        self.assertIsNone(record["point_observed_m"])


class PrintAssets(unittest.TestCase):
    def test_spec_geometry(self):
        spec = am.ArucoSpec(black_square_mm=16.0)
        spec.validate()
        self.assertEqual(am.QUAD_SPAN_CELLS, 6)
        self.assertAlmostEqual(spec.cell_mm, 16.0 / 6.0)
        self.assertAlmostEqual(spec.sticker_mm, 22.0)
        with self.assertRaises(ValueError):
            am.ArucoSpec(black_square_mm=16.0, sticker_mm=30.0).validate()

    def test_sticker_includes_white_margin(self):
        spec = am.ArucoSpec(black_square_mm=16.0)
        image = am.sticker_image(spec, 24.0)
        self.assertEqual(image.shape[0], int(round(spec.sticker_mm * 24.0)))
        margin = int(round(spec.white_margin_mm * 24.0))
        self.assertTrue((image[:margin] == 255).all())
        self.assertTrue((image[-margin:] == 255).all())

    def test_assets_name_carries_dictionary_id_and_sizes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            spec = am.ArucoSpec(black_square_mm=12.0)
            paths = am.write_assets(tmp, spec)
            for path in paths.values():
                self.assertTrue(Path(path).is_file())
            self.assertIn("DICT_4X4_50", paths["png"].name)
            self.assertIn("id0", paths["png"].name)
            self.assertIn("black12mm", paths["png"].name)
            import json
            payload = json.loads(paths["spec"].read_text(encoding="utf-8"))
            self.assertEqual(payload["marker_type"], "aruco")
            self.assertEqual(payload["quad_span_cells"], 6)


if __name__ == "__main__":
    unittest.main()
