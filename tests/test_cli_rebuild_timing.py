"""The live `run` command must append one capture->rebuild timing record."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from head_rgbd import cli  # noqa: E402


class RunTimingIntegration(unittest.TestCase):
    def test_run_appends_timing_after_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            runtime.mkdir(parents=True)
            capture = root / "capture"
            capture.mkdir()
            (capture / "session.json").write_text(json.dumps({
                "schema": "head_rgbd.capture.v1", "marker_status": "usable",
                "segments": [],
            }), encoding="utf-8")
            output = root / "reconstruction"
            delivery = root / "base_delivery"
            delivery.mkdir()
            neck_result = output / "result"
            args = argparse.Namespace(
                runtime=str(runtime), session_name="test", views=None,
                frames_per_view=2, interval_s=0.12, warmup_s=0.0,
                countdown_s=0, no_preview=True, auto=True,
                allow_marker_fallback=False, output=str(output),
                base_only=False, neck_prominence="off", neck_landmarks=None,
            )

            def fake_pipeline(**kwargs):
                kwargs["output"].mkdir(parents=True, exist_ok=True)
                return 0, delivery

            def fake_neck(a, out, base):
                a.neck_return_code = 0
                Path(out, "result").mkdir(parents=True, exist_ok=True)
                return str(Path(out, "result"))

            with mock.patch("head_rgbd.capture_session.capture_session",
                            return_value=capture), \
                    mock.patch.object(cli, "run_pipeline", side_effect=fake_pipeline), \
                    mock.patch.object(cli, "_attach_capture", return_value=None), \
                    mock.patch.object(cli, "_neck_after_base", side_effect=fake_neck), \
                    mock.patch.object(cli, "_mesh_paths",
                                      return_value={"closed": "x", "open": "y"}):
                code = cli.command_run(args)
            self.assertEqual(code, 0)
            log = runtime / "outputs" / "rebuild_timings.jsonl"
            self.assertTrue(log.is_file())
            line = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(line["schema"], "head_rgbd.rebuild_timing.v1")
            self.assertEqual(line["mode"], "live_run")
            self.assertEqual(line["timer_start"], "capture_complete")
            self.assertEqual(line["status"], "pass")
            self.assertEqual(line["marker_status"], "usable")
            self.assertEqual(line["capture"], str(capture))
            self.assertGreaterEqual(line["rebuild_elapsed_s"], 0.0)
            self.assertTrue((output / "rebuild_timing.json").is_file())
            self.assertEqual(args.capture, str(capture))


if __name__ == "__main__":
    unittest.main()