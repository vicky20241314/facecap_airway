"""Boolean indexing and lip-constraint gradients in the shared shape fit."""
import numpy as np
import torch

from head_rgbd.experiment.turn_shape import _landmark_masks


def test_landmark_masks_support_boolean_indexing_and_visibility():
    valid3d = np.ones(68, dtype=bool)
    valid2d = np.ones(68, dtype=bool)
    valid3d[30] = False
    valid2d[40] = False
    stable, inner, projected = _landmark_masks(valid3d, valid2d, "cpu")
    assert all(mask.dtype == torch.bool for mask in (stable, inner, projected))
    points = torch.ones(68, 3, requires_grad=True)
    loss = points[stable].sum() + .25 * points[inner].sum()
    loss.backward()
    assert torch.all(points.grad[:17] == 0)
    assert torch.all(points.grad[30] == 0)
    assert torch.all(points.grad[61] == .25)
    assert not projected[40]
    visible = torch.ones(68, dtype=torch.bool)
    visible[61] = False
    assert not (inner & visible)[61]
    assert inner[62]
    assert valid3d[:17].all()


def test_absent_depth_does_not_create_a_landmark_constraint():
    stable, inner, projected = _landmark_masks(np.zeros(68, bool), np.ones(68, bool), "cpu")
    assert not stable.any()
    assert not inner.any()
    assert projected[17:].all()
