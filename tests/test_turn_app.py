"""One-command workflow routing and GUI capture cleanup."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from head_rgbd.experiment import turn_app


def test_capture_inputs_require_both_role_manifests(tmp_path):
    (tmp_path / "session.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="anchor_session.json"):
        turn_app.capture_inputs(tmp_path)
    (tmp_path / "anchor_session.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="turn_session.json"):
        turn_app.capture_inputs(tmp_path)
    (tmp_path / "turn_session.json").write_text("{}", encoding="utf-8")
    assert turn_app.capture_inputs(tmp_path) == (tmp_path / "anchor_session.json", tmp_path / "turn_session.json")


def test_run_routes_new_capture_directly_to_reconstruction(tmp_path, monkeypatch):
    from head_rgbd.experiment import turn_capture, turn_run
    monkeypatch.setattr(turn_run, "_configure_stdio", lambda: None)
    for role in ("anchor", "turn"):
        (tmp_path / f"{role}_session.json").write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(turn_capture, "capture", lambda **kwargs: (calls.append(("capture", kwargs)) or tmp_path))
    monkeypatch.setattr(turn_app, "reconstruct", lambda **kwargs: calls.append(("reconstruct", kwargs)))
    assert turn_app.main(["run", "--device", "cpu", "--turn-frames", "60"]) == 0
    assert [item[0] for item in calls] == ["capture", "reconstruct"]
    assert calls[0][1]["turn_frames"] == 60
    assert calls[1][1]["anchor"] == tmp_path / "anchor_session.json"
    assert calls[1][1]["turn"] == tmp_path / "turn_session.json"
    assert calls[1][1]["device"] == "cpu"


def test_injected_preview_is_closed_and_camera_released_on_cancel(tmp_path, monkeypatch):
    from head_rgbd.experiment import turn_capture

    class Camera:
        def __init__(self, **kwargs):
            self.device = SimpleNamespace(get_depth_work_mode=lambda: SimpleNamespace(name="test"))
            self.color_profile = self.depth_profile = self.pipeline = object()
            self.stopped = False

        def start(self):
            pass

        def stop(self):
            self.stopped = True

    class Preview:
        closed = False

        def close(self):
            self.closed = True

    camera, preview = Camera(), Preview()
    monkeypatch.setattr(turn_capture, "OrbbecCapture", lambda **kwargs: camera)
    monkeypatch.setattr(turn_capture, "configure_environment", lambda runtime: SimpleNamespace(runtime=tmp_path))
    for name in ("_device_info_dict", "_profile_dict", "_camera_param_dict"):
        monkeypatch.setattr(turn_capture, name, lambda obj: {})

    def no_opencv_window():
        raise AssertionError("The GUI must not construct an OpenCV window")

    def cancel(cap, window, *args, **kwargs):
        assert window is preview
        raise KeyboardInterrupt

    monkeypatch.setattr(turn_capture, "_CapturePreview", no_opencv_window)
    monkeypatch.setattr(turn_capture, "_record", cancel)
    output = tmp_path / "captured"
    with pytest.raises(KeyboardInterrupt):
        turn_capture.capture(output=output, runtime=tmp_path, preview_window=preview)
    assert camera.stopped and preview.closed
    partial = json.loads((output / "session.partial.json").read_text(encoding="utf-8"))
    assert partial["status"] == "aborted"
    assert not (output / "session.json").exists()


def test_new_output_path_does_not_depend_on_working_directory(tmp_path, monkeypatch):
    parent = tmp_path / "outputs"
    monkeypatch.chdir(tmp_path)
    path = turn_app.new_run_path(parent)
    assert path.parent == parent
    assert path.name.startswith("gui_turn_")
    assert not path.exists()
