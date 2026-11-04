"""Contract tests for the front-anchor turn experiment.

These tests are deliberately free of models, depth data and GPU access: they pin
the properties that are easy to break by accident while tuning the experiment.

1. angle bins, angle labels and the left/right view convention
2. the keyframe budget is a ceiling and is never padded
3. redundant frames at an already covered angle are dropped
4. ``angle_group`` and ``tracking_group`` stay two independent things
5. a colour-only video is rejected instead of silently degrading
6. the configuration refuses values that would change the experiment's meaning
"""
from __future__ import annotations

import numpy as np
import pytest

from head_rgbd.experiment.turn_config import ConfigError, TurnConfig
from head_rgbd.experiment.turn_inputs import UnsupportedInput, plan_source
from head_rgbd.experiment.turn_keyframes import angle_bucket, group_label, select, view_of


def entry(index: int, yaw: float, *, role_source: str = "turn", quality: float = 0.8,
          visible=None, state_label: str = "closed", tracking_group: str = "turn_g1",
          usable: bool = True) -> dict:
    if visible is None:
        visible = np.zeros(68, dtype=bool)
        visible[17:68] = True
    return {
        "source_index": index,
        "timestamp_s": 0.1 * index,
        "yaw_deg": float(yaw),
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "quality": float(quality),
        "quality_components": {"synthetic": True},
        "visible_landmarks": np.asarray(visible),
        "state_label": state_label,
        "tracking_group": tracking_group,
        "usable": usable,
        "source": role_source,
        "source_frame": index,
    }


def anchor_entry(index: int = 0, yaw: float = -2.0, quality: float = 0.9) -> dict:
    return entry(index, yaw, role_source="anchor", quality=quality, tracking_group="anchor_g1")


# ------------------------------------------------------------------ angle helpers
def test_angle_bucket_rounds_to_the_nearest_bin_centre():
    assert angle_bucket(-24.0, 10.0) == -2
    assert angle_bucket(-25.1, 10.0) == -3
    assert angle_bucket(0.0, 10.0) == 0
    assert angle_bucket(18.0, 10.0) == 2


def test_group_label_is_stable_and_readable():
    assert group_label(angle_bucket(-24.0, 10.0), 10.0) == "yaw-20"
    assert group_label(angle_bucket(18.0, 10.0), 10.0) == "yaw+20"


def test_view_convention_negative_yaw_is_the_left_side():
    # The project convention measured on real captures: yaw < 0 is image left.
    assert view_of(-16.0, 8.0) == "left"
    assert view_of(18.0, 8.0) == "right"
    assert view_of(-3.0, 8.0) == "front"


def test_angle_bucket_rejects_a_missing_measurement():
    with pytest.raises(ValueError):
        angle_bucket(float("nan"), 10.0)


# --------------------------------------------------------------- keyframe budget
def test_budget_is_a_ceiling_and_an_under_covered_turn_is_not_padded():
    config = TurnConfig().replace(keyframes={"max_keyframes": 9})
    anchors = [anchor_entry(0)]
    turn = [entry(1, -30.0), entry(2, 30.0)]
    result = select(anchors, turn, config)

    assert len(result["keyframes"]) == 3
    assert result["budget"]["selected"] == 3
    assert result["budget"]["padded"] is False
    assert result["budget"]["stop_reason"] == "no_uncovered_angle_left"


def test_selection_never_exceeds_the_budget():
    config = TurnConfig().replace(keyframes={"max_keyframes": 4})
    anchors = [anchor_entry(0)]
    turn = [entry(index, float(yaw)) for index, yaw in enumerate(range(-50, 51, 10), start=1)]
    result = select(anchors, turn, config)

    assert len(result["keyframes"]) == 4
    assert result["budget"]["max_keyframes"] == 4
    assert result["budget"]["stop_reason"] == "budget_reached"


def test_redundant_frames_at_a_covered_angle_are_dropped_not_stacked():
    config = TurnConfig().replace(keyframes={"max_keyframes": 9})
    anchors = [anchor_entry(0)]
    # Twelve frames within the same 10 degree bin: only the most reliable survives.
    turn = [entry(index, 20.0 + 0.2 * index, quality=0.5 + 0.02 * index) for index in range(1, 13)]
    result = select(anchors, turn, config)

    angles = [row["angle_group"] for row in result["keyframes"]]
    assert len(angles) == len(set(angles)), "one angle group must contribute at most one keyframe"
    assert "same_angle_bucket_as_a_selected_frame" in result["rejected"]
    assert result["budget"]["padded"] is False


def test_the_anchor_is_the_most_reliable_closed_frontal_frame():
    config = TurnConfig().replace(keyframes={"max_keyframes": 9})
    anchors = [anchor_entry(0, yaw=-3.0, quality=0.6), anchor_entry(1, yaw=2.0, quality=0.95)]
    result = select(anchors, [entry(2, -30.0)], config)

    chosen = result["keyframes"][0]
    assert chosen["role"] == "anchor"
    assert chosen["source_index"] == 1
    assert chosen["selection_reason"] == "best_frontal_anchor"


def test_open_mouth_frames_are_never_identity_keyframes():
    config = TurnConfig().replace(keyframes={"max_keyframes": 9})
    result = select([anchor_entry(0)], [entry(1, -30.0, state_label="open")], config)

    assert all(row["state"] == "closed" for row in result["keyframes"])
    assert result["rejected"]["open_mouth_frame"] == 1


# --------------------------------------------- tracking groups versus angle groups
def test_one_tracking_group_can_cover_several_angle_groups():
    config = TurnConfig().replace(keyframes={"max_keyframes": 9})
    turn = [entry(index, yaw, tracking_group="turn_g1")
            for index, yaw in enumerate((-30.0, -15.0, 0.0, 15.0, 30.0), start=1)]
    result = select([anchor_entry(0)], turn, config)

    assert len(result["tracking_groups"]) == 2  # anchor group plus one turn group
    assert len(result["angle_groups"]) >= 4
    turn_groups = [row for row in result["tracking_groups"] if row["label"] == "turn_g1"]
    assert len(turn_groups) == 1
    assert turn_groups[0]["spans_multiple_angles"] is True
    assert len(turn_groups[0]["frames"]) >= 4, "one continuity group must not collapse to one angle"


def test_the_same_angle_can_be_revisited_in_a_later_tracking_group():
    config = TurnConfig().replace(keyframes={"max_keyframes": 9})
    turn = [entry(1, -30.0, tracking_group="turn_g1"),
            entry(2, 30.0, tracking_group="turn_g1"),
            entry(3, -32.0, tracking_group="turn_g2", quality=0.9)]
    result = select([anchor_entry(0)], turn, config)

    reused = [row for row in result["keyframes"] if row["angle_group"] in ("yaw-30", "yaw-40")]
    assert reused, "a revisited angle is a fresh opportunity, not a rejected duplicate"
    assert len({row["angle_group"] for row in result["keyframes"]}) == len(result["keyframes"])


# ------------------------------------------------------------- input type contract
def test_a_colour_video_is_rejected_instead_of_silently_degrading(tmp_path):
    video = tmp_path / "turn.mp4"
    video.write_bytes(b"")
    with pytest.raises(UnsupportedInput) as caught:
        plan_source(video)
    assert "RGB-D" in str(caught.value)


def test_an_unknown_extension_is_rejected(tmp_path):
    strange = tmp_path / "capture.xyz"
    strange.write_bytes(b"")
    with pytest.raises(UnsupportedInput):
        plan_source(strange)


# ------------------------------------------------------------ configuration guards
def test_configuration_round_trips_through_json():
    config = TurnConfig.load()
    again = TurnConfig.from_dict(config.to_dict())
    assert again.to_dict() == config.to_dict()
    assert isinstance(again.delivery.metrics_to_watch, tuple)


def test_overrides_rebuild_the_configuration_without_touching_the_file():
    config = TurnConfig.load()
    changed = config.replace(device="cpu", tracking={"detect_stride": 2},
                             keyframes={"max_keyframes": 6})
    assert changed.experiment.device == "cpu"
    assert changed.tracking.detect_stride == 2
    assert changed.keyframes.max_keyframes == 6
    assert config.keyframes.max_keyframes == 9


@pytest.mark.parametrize("overrides, message", [
    ({"keyframes": {"max_keyframes": 0}}, "max_keyframes"),
    ({"keyframes": {"max_keyframes": 13}}, "max_keyframes"),
    ({"keyframes": {"repeat_observations_per_angle": 5}}, "repeat_observations_per_angle"),
    ({"keyframes": {"repeat_max_angle_delta_deg": 0.0}}, "repeat_max_angle_delta_deg"),
    ({"field": {"displacement_bound_mm": 5.0}}, "displacement_bound_mm"),
    ({"field": {"loop_target_blend": 0.0}}, "loop_target_blend"),
    ({"field": {"points_per_frame": 100}}, "points_per_frame"),
    ({"tracking": {"klt_window": 20}}, "klt_window"),
    ({"joint_shape": {"expression_bound": 0.0}}, "expression_bound"),
])
def test_configuration_refuses_values_that_change_the_experiment(overrides, message):
    with pytest.raises(ConfigError) as caught:
        TurnConfig.load().replace(**overrides)
    assert message in str(caught.value)


def test_frontal_band_must_sit_inside_the_anchor_band():
    with pytest.raises(ConfigError):
        TurnConfig.load().replace(keyframes={"front_yaw_deg": 20.0, "anchor_max_yaw_deg": 15.0})


def test_the_loop_blend_target_is_half_and_not_zero():
    config = TurnConfig.load()
    assert config.field.loop_target_blend == pytest.approx(0.5)
    assert 0.0 < config.field.loop_target_blend <= 1.0
