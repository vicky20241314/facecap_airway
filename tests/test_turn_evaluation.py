"""Pure summary tests for the repeat-observation evaluation tool."""
import pytest

from tools.eval_turn_runs import _delta, _physical_id, _summary


def test_summary_ignores_frames_without_measured_residuals():
    rows = [
        {"median_mm": 1., "p90_mm": 2., "coverage": .9},
        {"median_mm": 2., "p90_mm": 3., "coverage": .8},
        {"median_mm": None, "p90_mm": None, "coverage": 0.},
    ]
    result = _summary(rows)

    assert result["frames"] == 2
    assert result["median_of_frame_medians_mm"] == pytest.approx(1.5)
    assert result["mean_p90_mm"] == pytest.approx(2.5)
    assert result["worst_frame_median_mm"] == pytest.approx(2.)


def test_delta_uses_repeated_minus_static_sign():
    static = {"median_of_frame_medians_mm": .8, "mean_p90_mm": 2.0}
    repeated = {"median_of_frame_medians_mm": .7, "mean_p90_mm": 1.9}
    result = _delta(static, repeated)

    assert result["median_of_frame_medians_mm"] == pytest.approx(-.1)
    assert result["mean_p90_mm"] == pytest.approx(-.1)


def test_physical_frame_identity_prefers_source_and_source_frame():
    assert _physical_id({"source": "turn.npz", "source_frame": 8, "source_index": 9}) == (
        "turn.npz", 8)
    assert _physical_id({"source": "turn.npz", "source_index": 9}) == ("turn.npz", 9)
