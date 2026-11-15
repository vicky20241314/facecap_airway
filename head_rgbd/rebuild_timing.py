"""Per-run wall-clock timing from capture completion to rebuild completion.

The live ``run`` command is the only place that knows the moment capture has
finished and the moment the reconstruction + neck stage has finished.  This
module keeps that measurement in two places:

* ``<runtime>/outputs/rebuild_timings.jsonl`` -- one append-only line per run,
  so successive runs can be compared;
* ``<reconstruction output>/rebuild_timing.json`` -- the same record next to the
  delivered result.

The timer is wall clock (``time.monotonic``); capture time itself is excluded.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

SCHEMA = "head_rgbd.rebuild_timing.v1"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def format_elapsed(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


class RebuildTimer:
    """Monotonic timer; ``start`` is called right after capture returns."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self.started_monotonic = float(clock())
        self.started_at = now_iso()
        self.finished_at: str | None = None
        self.elapsed_s: float | None = None

    def stop(self) -> float:
        if self.elapsed_s is None:
            self.elapsed_s = float(self._clock() - self.started_monotonic)
            self.finished_at = now_iso()
        return self.elapsed_s


def build_record(*, timer: RebuildTimer, mode: str, capture: str | None,
                 output: str | None, delivery: str | None,
                 neck_result: str | None, marker_status: str | None,
                 pipeline_code: int | None, neck_code: int | None,
                 status: str, error: str | None = None,
                 base_only: bool = False) -> dict:
    elapsed = timer.stop()
    return {
        "schema": SCHEMA,
        "mode": mode,
        "captured_at": timer.started_at,
        "finished_at": timer.finished_at,
        "rebuild_started_at": timer.started_at,
        "rebuild_finished_at": timer.finished_at,
        "rebuild_elapsed_s": round(elapsed, 3),
        "rebuild_elapsed_hms": format_elapsed(elapsed),
        "timer_start": "capture_complete" if mode == "live_run" else "command_start",
        "capture": capture,
        "output": output,
        "delivery": delivery,
        "neck_result": neck_result,
        "marker_status": marker_status,
        "pipeline_code": pipeline_code,
        "neck_code": neck_code,
        "status": status,
        "base_only": bool(base_only),
        "error": error,
    }


def write_timing(*, log_path: str | Path, output_dir: str | Path | None,
                 record: dict) -> dict:
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    written = {"log": str(log)}
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "rebuild_timing.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1,
                                  allow_nan=False) + "\n", encoding="utf-8")
        written["output"] = str(path)
    return written