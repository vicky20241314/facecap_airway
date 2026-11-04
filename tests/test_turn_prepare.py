"""Exercise the actual input-provenance to prepared-sequence handoff."""
import json
from types import SimpleNamespace

import numpy as np

from head_rgbd.audit import sha
from head_rgbd.experiment import turn_prepare
from head_rgbd.experiment.turn_config import TurnConfig
from head_rgbd.experiment.turn_inputs import open_source


def test_prepare_consumes_input_provenance_and_binds_both_sequences(tmp_path, monkeypatch):
    config = TurnConfig.load()
    inputs = {}
    keyframes = []
    for order, role in enumerate(("anchor", "turn")):
        path = tmp_path / f"{role}.npz"
        np.savez(path, rgb=np.zeros((3, 64, 64, 3), np.uint8),
                 depth=np.full((3, 64, 64), 500, np.uint16),
                 intrinsics=np.tile([850., 850., 32., 32.], (3, 1)),
                 timestamps_s=np.arange(3) * .12)
        inputs[role] = open_source(path, role, config).provenance()
        keyframes.append({"id": role, "role": role, "source_index": 0,
                          "tracking_group": f"{role}_track0", "order": order,
                          "view": "front" if role == "anchor" else "left",
                          "angle_group": "yaw0" if role == "anchor" else "yaw-20",
                          "angle_bucket_center_deg": order * -20., "quality": .8,
                          "selection_reason": "contract_test", "yaw_deg": order * -20.})
    supports = [dict(keyframes[1], id="support_turn", source_index=1, order=2,
                     observation_role="support", canonical_keyframe_id="turn",
                     selection_reason="same_angle_repeat")]
    (tmp_path / "selection").mkdir()
    (tmp_path / "selection" / "keyframes.json").write_text(
        json.dumps({"keyframes": keyframes, "support_frames": supports}), encoding="utf-8")
    uv = np.column_stack((np.arange(68) % 10 + 20, np.arange(68) // 10 + 20)).astype(float)
    monkeypatch.setattr(turn_prepare, "detect_landmarks", lambda rgb: (uv, np.array([10, 10, 40, 40])))
    parser = SimpleNamespace(FACE_LABELS=[1], get_face_data_single=lambda *args, **kwargs:
                             {"parsing": np.ones((64, 64), np.uint8)})
    monkeypatch.setattr(turn_prepare, "_prepare_parser", lambda: parser)
    summary = turn_prepare.materialise(tmp_path, config, {"inputs": inputs, "runtime": str(tmp_path)})
    assert summary["prepared_frames"] == 3
    assert summary["keyframe_frames"] == 2
    assert summary["support_frames"] == 1
    assert summary["anchor_frames"] == 1
    prepared = tmp_path / "prepared"
    full = json.loads((prepared / "sequence.json").read_text(encoding="utf-8"))
    anchor = json.loads((prepared / "anchor_sequence.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in full["frames"]] == ["anchor", "turn", "support_turn"]
    assert [row["id"] for row in anchor["frames"]] == ["anchor"]
    assert anchor["frames"][0]["file"] == full["frames"][0]["file"]
    for source in full["sources"]:
        assert source["path"] == inputs[source["role"]]["source"]["path"]
        assert source["sha256"] == sha(source["path"])
    for row in full["frames"]:
        with np.load(prepared / row["file"], allow_pickle=False) as stored:
            assert stored["valid"].shape == (68,)
            assert stored["valid"].all()
            assert np.isfinite(stored["xyz"]).all()
    assert full["frames"][-1]["observation_role"] == "support"
    assert full["frames"][-1]["canonical_keyframe_id"] == "turn"
