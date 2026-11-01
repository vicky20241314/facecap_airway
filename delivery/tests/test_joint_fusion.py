"""Mechanism tests for joint front+side shape fusion.

Scene: one flat plate seen by three cameras. Two front views report the plate at
its true depth, one left view reports it 5 mm closer. The shared field must stay
at zero while only the front views vote, and must move once the left view votes.
The same scene shows that a confidence-weighted frontal region can respond where
a hard lock freezes it.
"""
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from head_rgbd import cli  # noqa: E402
from head_rgbd.geometry_d import Settings, solve_field  # noqa: E402

INTRINSICS = np.array([300.0, 300.0, 32.0, 32.0])
PLATE_Y, PLATE_X = slice(16, 48), slice(16, 48)
GRID = 5


def _plate():
    axis = np.linspace(-0.02, 0.02, GRID)
    xs, ys = np.meshgrid(axis, axis)
    vertices = np.column_stack((xs.ravel(), ys.ravel(), np.full(xs.size, -0.5)))
    faces = []
    for row in range(GRID - 1):
        for column in range(GRID - 1):
            a = row * GRID + column
            b, c, d = a + 1, a + GRID, a + GRID + 1
            faces.append((a, b, c))
            faces.append((b, d, c))
    return vertices, np.asarray(faces, np.int64)


def _frame(depth_mm):
    depth = np.full((64, 64), 3000, np.uint16)
    depth[PLATE_Y, PLATE_X] = np.uint16(depth_mm)
    mask = np.zeros((64, 64), bool)
    mask[PLATE_Y, PLATE_X] = True
    return dict(depth=depth, face_mask=mask, intrinsics=INTRINSICS,
                uv=np.full((68, 2), 62.0), xyz=np.zeros((68, 3)), valid=np.zeros(68, bool))


def _scene(frames):
    """frames: list of (view, depth_mm)."""
    vertices, faces = _plate()
    count = len(vertices)
    n = len(frames)
    base = np.repeat(vertices[None], n, axis=0)
    normals = np.tile(np.array([0., 0., -1.]), (n, count, 1))
    data = [_frame(depth) for _, depth in frames]
    rows = [dict(id=f"{view}{i}", view=view, state='closed', segment=f"{view}{i}",
                 timestamp_s=float(i), status='candidate')
            for i, (view, _) in enumerate(frames)]
    settings = Settings(solver='qp', research_v2=True, points=2000, outer_steps=1,
                        gradient_length_mm=8., bending_length_mm=8.,
                        noise_floor_mm=1 / np.sqrt(12), displacement_bound_mm=2.5,
                        fixed_ray_policy='strict-coverage')
    return base, faces, normals, data, rows, np.zeros(count, bool), settings


def _solve(frames, identity_ids, **kwargs):
    base, faces, normals, data, rows, excluded, settings = _scene(frames)
    field, support, log = solve_field(base, faces, normals, data, rows, identity_ids,
                                      excluded, settings, **kwargs)
    return field, support, log


class SharedShapeFromMultipleViews(unittest.TestCase):
    def test_side_view_moves_the_shared_field(self):
        """Front views agree with the model; the side view disagrees by 5 mm."""
        frames = [('front', 500), ('front', 500), ('left', 495)]
        front_field, _, _ = _solve(frames, [0, 1], stage='front_only')
        self.assertLess(abs(front_field).max(), 0.05,
                        "agreeing front views must not invent a correction")

        joint_field, support, log = _solve(frames, [0, 1, 2], stage='joint')
        self.assertGreater(abs(joint_field).max(), 0.3,
                           "the side residual must reach the shared field")
        self.assertTrue(np.isfinite(joint_field).all())
        self.assertEqual(len(log['history']), 1)
        self.assertTrue(support[2].any(), "the side view must be recorded as supporting vertices")

    def test_confidence_weight_lets_the_protected_region_respond(self):
        """Hard lock freezes the frontal region; a confidence weight lets it move."""
        frames = [('front', 500), ('front', 500), ('left', 495)]
        locked, _, locked_log = _solve(frames, [0, 1, 2], initial=np.zeros(GRID * GRID),
                                       locked=np.ones(GRID * GRID, bool), stage='hard_locked')
        self.assertEqual(abs(locked).max(), 0.0)
        self.assertEqual(locked_log['history'][0]['status'], 'skipped_no_free_vertices')

        soft, _, _ = _solve(frames, [0, 1, 2], initial=np.zeros(GRID * GRID),
                            soft_lock=np.full(GRID * GRID, .15), stage='confidence_weighted')
        self.assertGreater(abs(soft).max(), 0.3,
                           "a 15% confidence weight must still admit a correction")

    def test_all_views_agreeing_leave_the_surface_alone(self):
        frames = [('front', 500), ('front', 500), ('left', 500)]
        field, _, _ = _solve(frames, [0, 1, 2], stage='agreeing_views')
        self.assertLess(abs(field).max(), 0.05,
                        "three agreeing views must not manufacture a correction")


class JointModeContract(unittest.TestCase):
    def test_guarded_stays_the_default(self):
        parser = cli.build_parser()
        args = parser.parse_args(['reconstruct', '--capture', 'capture'])
        self.assertEqual(args.shape_fusion, 'guarded')
        self.assertEqual(args.joint_front_confidence, 0.15)
        self.assertEqual(args.side_depth_weight, 1.0)

        joint = parser.parse_args(['reconstruct', '--capture', 'capture', '--shape-fusion', 'joint'])
        self.assertEqual(joint.shape_fusion, 'joint')

    def test_joint_requires_a_front_anchor(self):
        result = subprocess.run(
            [sys.executable, '-m', 'head_rgbd.fit_c', '--sequence', 'sequence.json',
             '--output', 'out', '--shape-fusion', 'joint'],
            cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('joint shape fusion needs a front-only anchor', result.stderr)


if __name__ == "__main__":
    unittest.main()
