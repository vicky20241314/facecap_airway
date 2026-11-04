"""Regression tests for repeat observations, reliability weights and metrics."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from head_rgbd.experiment.turn_config import TurnConfig
from head_rgbd.experiment import turn_field
from head_rgbd.experiment.turn_field import _geometry_change, _metrics
from head_rgbd.experiment.turn_keyframes import select
from head_rgbd.experiment.turn_shape import (
    _frame_weights,
    _refresh_visibility,
    _reliability_weights,
    _weighted_robust,
)
from head_rgbd.motion_v2 import _frame_loss_weights


def _entry(index, yaw, quality=.8, group="turn_g1", source="turn.npz"):
    visible = np.ones(68, dtype=bool)
    return {
        "source_index": index,
        "timestamp_s": .12 * index,
        "yaw_deg": float(yaw),
        "pitch_deg": 0.,
        "roll_deg": 0.,
        "quality": float(quality),
        "quality_components": {},
        "visible_landmarks": visible,
        "state_label": "closed",
        "tracking_group": group,
        "usable": True,
        "source": source,
        "source_frame": index,
    }


def test_same_angle_repeats_are_retained_without_spending_keyframe_budget():
    config = TurnConfig().replace(keyframes={
        "max_keyframes": 9,
        "repeat_observations_per_angle": 2,
        "repeat_max_angle_delta_deg": 6.,
    })
    anchor = _entry(0, 0., quality=.9, group="anchor_g1", source="anchor.npz")
    turn = [_entry(index, value, quality=.80 + index * .005)
            for index, value in enumerate((20., 20.5, 21., 21.5, 22., 22.5), start=1)]

    result = select([anchor], turn, config)

    assert len(result["keyframes"]) == 2
    assert len(result["support_frames"]) == 2
    support = result["support_frames"]
    assert {row["angle_group"] for row in support} == {"yaw+20"}
    assert len({row["id"] for row in support}) == 2
    assert all(row["observation_role"] == "support" for row in support)
    assert all(row["canonical_keyframe_id"] for row in support)
    assert result["budget"]["total_observations"] == 4


def test_one_group_of_repeats_never_exceeds_its_angle_weight():
    rows = [
        {"id": "anchor", "angle_group": "yaw+0", "tracking_quality": .9},
        {"id": "left", "angle_group": "yaw-20", "tracking_quality": .6},
        {"id": "left_support", "angle_group": "yaw-20", "tracking_quality": .8},
    ]
    weights, table = _frame_weights(rows)

    by_group = {item["angle_group"]: item for item in table}
    assert by_group["yaw+0"]["total_weight"] == pytest.approx(.5)
    assert by_group["yaw-20"]["total_weight"] == pytest.approx(.5)
    assert sum(table[1]["frame_weights"]) == pytest.approx(.5)
    assert weights.sum() == pytest.approx(1.)


def test_pointwise_reliability_changes_the_depth_loss_before_reduction():
    residual = torch.tensor([0., 1., 2.], dtype=torch.float32)
    weights = torch.tensor([8., 1., 1.], dtype=torch.float32)
    swapped = weights.flip(0)

    focused = _weighted_robust(residual, weights)
    diffused = _weighted_robust(residual, swapped)

    assert torch.isfinite(focused)
    assert focused < diffused
    assert focused != pytest.approx(float(diffused), rel=0, abs=1e-7)


def test_zero_and_nonfinite_weights_cannot_produce_nan():
    residual = torch.tensor([0., 1., 2.], requires_grad=True)
    weight = torch.tensor([0., 0., 0.])
    loss = _weighted_robust(residual, weight)
    loss.backward()

    assert float(loss) == 0.
    assert torch.isfinite(residual.grad).all()


def test_reliability_weights_use_viewpoint_plane_and_scatter():
    incidence = np.array([.9, .3, .8])
    agreement = np.array([.95, .60, .99])
    scatter = np.array([.25, 1.4, .3])
    weight = _reliability_weights(incidence, scatter, agreement)

    assert np.isfinite(weight).all()
    assert weight[0] > weight[1]
    assert weight[0] > weight[2]
    assert np.isclose(weight[weight > 0].mean(), 1.)
    assert np.array_equal(_reliability_weights([], [], []), np.zeros(0))


def test_visibility_is_recomputed_from_the_immutable_base_mask():
    base = torch.tensor([True, True, True])
    transient = torch.tensor([False, True, True])
    complete = torch.tensor([True, True, True])

    first = _refresh_visibility(base, transient)
    second = _refresh_visibility(base, complete)

    assert first.tolist() == [False, True, True]
    assert second.tolist() == [True, True, True]


def test_geometry_change_uses_shared_shape_control_and_loop_baselines():
    shared = np.zeros((2, 3))
    control = np.array([[.001, 0., 0.], [0., .002, 0.]])
    final = np.array([[.001, .0005, 0.], [0., .002, .001]])

    result = _geometry_change(shared, control, final)

    assert result["shared_shape_to_control"]["median_mm"] == pytest.approx(1.5)
    assert result["control_to_loop"]["median_mm"] == pytest.approx(.75)
    assert result["shared_shape_to_final_total"]["median_mm"] > 1.5
    assert all(value is not None for component in result.values() for value in component.values())


def test_field_metrics_use_the_shared_shape_for_smoothness(monkeypatch):
    shared = np.array([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.], [0., 0., .01]]])
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    config = SimpleNamespace(
        experiment=SimpleNamespace(seed=1),
        field=SimpleNamespace(probe_points_per_frame=100, maximum_slope=.45),
    )
    monkeypatch.setattr(turn_field, "sample_depth",
                        lambda *args, **kwargs: (np.zeros((0, 3)), np.zeros((0, 2), int), {}))

    _, geometry, smoothness = _metrics(
        shared.copy(), faces, [{}], [{"id": "front", "view": "front", "angle_group": "yaw+0",
                                      "fitted_yaw_deg": 0.}],
        shared, shared.copy(), np.zeros(4), config, 0)

    assert geometry["all_frames"]["shared_shape_to_final_total"]["median_mm"]["median_of_frames_mm"] == 0.
    assert np.isfinite(smoothness["p90"])


def test_pose_correction_uses_supplied_angle_weights_with_uniform_fallback():
    supplied = [{"frame_weight": .2}, {"frame_weight": .3}, {"frame_weight": .5}]
    fallback = [{}, {}, {}]

    assert np.allclose(_frame_loss_weights(supplied), [.2, .3, .5])
    assert np.allclose(_frame_loss_weights(fallback), [1. / 3.] * 3)
