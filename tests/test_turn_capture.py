"""Contracts for the independent continuous RGB-D capture entry point."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from head_rgbd.experiment.turn_capture import _validate_frame, _write_role_manifests
from head_rgbd.experiment.turn_config import TurnConfig
from head_rgbd.experiment.turn_inputs import open_source


def _frame():
    return SimpleNamespace(rgb=np.zeros((720, 1280, 3), np.uint8),
                           depth=np.full((720, 1280), 500, np.uint16),
                           intrinsics=np.array([850., 850., 640., 360.]))


def test_aligned_frame_is_accepted():
    _validate_frame(_frame())


@pytest.mark.parametrize("invalid", ["depth_dtype", "rgb_shape", "focal_length", "nan_intrinsics"])
def test_invalid_measurement_is_rejected(invalid):
    frame = _frame()
    if invalid == "depth_dtype":
        frame.depth = frame.depth.astype(np.float32)
    elif invalid == "rgb_shape":
        frame.rgb = frame.rgb[:600]
    elif invalid == "focal_length":
        frame.intrinsics[0] = 0
    else:
        frame.intrinsics[2] = np.nan
    with pytest.raises(ValueError):
        _validate_frame(frame)


def test_role_manifests_open_only_their_own_clip(tmp_path):
    segments = []
    for role, view, count in (("anchor", "front", 2), ("turn", None, 4)):
        filename = f"{role}.npz"
        np.savez(tmp_path / filename, rgb=np.zeros((count, 4, 4, 3), np.uint8),
                 depth=np.full((count, 4, 4), 500, np.uint16),
                 intrinsics=np.tile([850., 850., 2., 2.], (count, 1)),
                 timestamps_s=np.arange(count) * .12)
        segments.append({"id": role, "state": "closed", "view": view,
                         "file": filename, "acquisition_role": role})
    manifest = {"schema": "head_rgbd.capture.v1", "status": "complete",
                "rgb_depth_aligned": True, "depth_unit": "mm", "mesh_unit": "m",
                "coordinate_convention": "X_right_Y_up_Z_backward", "segments": segments}
    _write_role_manifests(tmp_path, manifest)
    for role, count in (("anchor", 2), ("turn", 4)):
        path = tmp_path / f"{role}_session.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert len(saved["segments"]) == 1
        sequence = open_source(path, role, TurnConfig.load())
        assert len(sequence) == count
        assert all(entry.file.name == f"{role}.npz" for entry in sequence.entries)
    with pytest.raises(FileExistsError):
        _write_role_manifests(tmp_path, manifest)


def test_missing_role_does_not_publish_partial_manifests(tmp_path):
    with pytest.raises(ValueError, match="no turn segment"):
        _write_role_manifests(tmp_path, {"segments": [{"acquisition_role": "anchor"}]})
    assert not list(tmp_path.iterdir())
