"""Small tests for the per-run rebuild timer."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from head_rgbd import rebuild_timing as rt  # noqa: E402


class Timing(unittest.TestCase):
    def test_elapsed_is_capture_complete_to_rebuild_complete(self):
        ticks = iter([100.0, 112.5])
        timer = rt.RebuildTimer(clock=lambda: next(ticks))
        self.assertIsNone(timer.elapsed_s)
        timer.stop()
        self.assertAlmostEqual(timer.elapsed_s, 12.5)
        self.assertEqual(timer.started_at is not None, True)
        self.assertEqual(timer.finished_at is not None, True)
        self.assertAlmostEqual(timer.stop(), 12.5)   # idempotent

    def test_elapsed_hms(self):
        self.assertEqual(rt.format_elapsed(0.4), "0s")
        self.assertEqual(rt.format_elapsed(65), "1m05s")
        self.assertEqual(rt.format_elapsed(3725), "1h02m05s")

    def test_jsonl_append_and_per_run_file(self):
        ticks = iter([10.0, 22.0, 30.0, 41.0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "outputs" / "rebuild_timings.jsonl"
            for index in range(2):
                timer = rt.RebuildTimer(clock=lambda: next(ticks))
                record = rt.build_record(
                    timer=timer, mode="live_run", capture=f"capture/{index}",
                    output=f"run/{index}", delivery=f"delivery/{index}",
                    neck_result=f"result/{index}", marker_status="usable",
                    pipeline_code=0, neck_code=0, status="pass")
                written = rt.write_timing(log_path=log,
                                          output_dir=root / "run" / str(index),
                                          record=record)
                self.assertEqual(written["log"], str(log))
                self.assertTrue((root / "run" / str(index) / "rebuild_timing.json").is_file())
            lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["schema"], rt.SCHEMA)
            self.assertEqual(lines[0]["timer_start"], "capture_complete")
            self.assertAlmostEqual(lines[0]["rebuild_elapsed_s"], 12.0)
            self.assertAlmostEqual(lines[1]["rebuild_elapsed_s"], 11.0)
            self.assertEqual(lines[0]["mode"], "live_run")


if __name__ == "__main__":
    unittest.main()