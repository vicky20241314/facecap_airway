"""Re-read existing validation runs with the current metric collector.

No pipeline is re-run: this only refreshes ``metrics.json`` from the reports
already on disk (e.g. after adding a field to the collector).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.speedup_validate import _downstream, collect


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args(argv)
    for root in args.roots:
        metrics_path = root / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key, old in (metrics.get("runs") or {}).items():
            run_dir = Path(old.get("run_dir") or (root / key))
            record = collect(run_dir)
            record.update({"key": key, "kind": old.get("kind"),
                           "exit_code": old.get("exit_code"),
                           "wall_s": old.get("wall_s"),
                           "capture_name": old.get("capture_name")})
            if old.get("downstream") is not None:
                record["downstream"] = _downstream(run_dir)
            else:
                record["downstream"] = None
            metrics["runs"][key] = record
            print(f"[recollect] {root.name}/{key} ok")
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=1, allow_nan=False) + "\n",
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
