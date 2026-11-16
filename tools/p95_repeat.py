"""Repeated end-to-end runs -> empirical P50/P95/max with the nearest-rank rule.

Each repetition gets a brand-new output directory; nothing is overwritten.
Failures are kept in the denominator and reported separately -- they are never
dropped to make the latency look better.

P95 uses the nearest rank: the ceil(0.95*n)-th item of the ascending sort.

Run:
    python -m tools.p95_repeat --runs 8 --subject zp_full --out <new dir>
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from tools.speedup_validate import (RUNS, PYTHON, _env, _json, collect,
                                    _downstream)


def quantile(sorted_vals, q):
    """Nearest-rank empirical quantile (1-based ceil(q*n)-th item)."""
    n = len(sorted_vals)
    if not n:
        return None
    k = max(1, min(n, int(math.ceil(q * n))))
    return sorted_vals[k - 1], k


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--subject", default="zp_full",
                    help="comma-separated keys from the validation RUNS table")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-s", type=float, default=180.0)
    ap.add_argument("--downstream", action="store_true", default=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)

    wanted = [k.strip() for k in args.subject.split(",") if k.strip()]
    rows = [r for r in RUNS if r["key"] in wanted]
    if not rows:
        ap.error(f"no known subject in {wanted}")

    samples = []
    started = datetime.now().astimezone().isoformat()
    for i in range(args.runs):
        for row in rows:
            capture = PROJ / "runtime" / "captures" / row["capture"]
            output = args.out / f"run{i:02d}_{row['key']}"
            cmd = [str(PYTHON), "-m", "head_rgbd", "rebuild",
                   "--capture", str(capture), "--output", str(output),
                   "--loss-mode", "batch"]
            if row["kind"] == "base":
                cmd.append("--base-only")
            t0 = time.perf_counter()
            cp = subprocess.run(cmd, cwd=str(PROJ), env=_env(),
                                capture_output=True, text=True)
            wall = time.perf_counter() - t0
            (args.out / f"run{i:02d}_{row['key']}_console.log").write_text(
                (cp.stdout or "") + "\n--- stderr ---\n" + (cp.stderr or ""),
                encoding="utf-8", errors="replace")
            rec = collect(output) if output.is_dir() else {}
            neck = rec.get("neck_seconds")
            stages = rec.get("stages") or {}
            base_sum = sum(v.get("elapsed_s") or 0.0 for v in stages.values())
            acc = rec.get("acceptance") or {}
            crease = acc.get("crease") or {}
            full_pos = max([v.get("full_positive_max_deg") or 0.0
                            for v in crease.values()] or [0.0])
            sample = {
                "index": i, "key": row["key"], "kind": row["kind"],
                "output": str(output), "exit_code": cp.returncode,
                "wall_s": round(wall, 4),
                "base_stages_s": round(base_sum, 4),
                "neck_s": round(neck, 4) if neck else None,
                "overhead_s": (round(wall - base_sum - neck, 4)
                               if neck else None),
                "stages": {k: v.get("elapsed_s") for k, v in stages.items()},
                "quick_pass": acc.get("pass"),
                "crease_full_positive_max_deg": full_pos,
                "downstream_all_pass": None,
                "usable": bool((acc.get("final") or {}).get("status") == "usable"
                               and (acc.get("measured") or {}).get("status") == "usable"),
            }
            if row["kind"] == "full" and args.downstream and output.is_dir():
                ds = _downstream(output)
                sample["downstream_all_pass"] = ds.get("all_pass")
            samples.append(sample)
            with (args.out / "samples.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"[p95] run{i:02d} {row['key']}: exit={cp.returncode} "
                  f"wall={wall:.2f}s neck={neck if neck is None else round(neck, 2)} "
                  f"crease={full_pos:.3f}", flush=True)

    full = [s for s in samples if s["kind"] == "full"]
    ok = [s for s in full if s["exit_code"] == 0]
    walls = sorted(s["wall_s"] for s in ok)
    p50 = quantile(walls, 0.50)
    p95 = quantile(walls, 0.95)
    qualified = [s for s in ok
                 if s["wall_s"] <= args.target_s
                 and (s["quick_pass"] or {}).get("critical")
                 and s["downstream_all_pass"]]
    summary = {
        "schema": "p95.repeat.v1", "started_at": started,
        "finished_at": datetime.now().astimezone().isoformat(),
        "target_s": args.target_s,
        "requested_runs": args.runs, "subjects": wanted,
        "n_samples_total": len(samples),
        "n_full": len(full), "n_full_failed": len(full) - len(ok),
        "n_base": len(samples) - len(full),
        "full_wall_s_sorted": walls,
        "p50_s": p50[0] if p50 else None, "p50_rank": p50[1] if p50 else None,
        "p95_s": p95[0] if p95 else None, "p95_rank": p95[1] if p95 else None,
        "max_s": walls[-1] if walls else None,
        "min_s": walls[0] if walls else None,
        "mean_s": round(sum(walls) / len(walls), 4) if walls else None,
        "n_over_target": sum(1 for w in walls if w > args.target_s),
        "timeout_rate": (round(sum(1 for w in walls if w > args.target_s) / len(full), 4)
                         if full else None),
        "n_qualified_and_under_target": len(qualified),
        "qualified_rate_of_all_full": (round(len(qualified) / len(full), 4)
                                       if full else None),
        "p95_rule": "nearest rank: ascending sort, item ceil(0.95*n) (1-based)",
        "limitations": (
            "Offline replay of one recorded capture per subject, not live "
            "acquisition. Repeating the same capture does not establish a "
            "population P95 over new subjects; no confidence claim is made. "
            "Failures stay in the denominator of qualified_rate_of_all_full."),
        "samples": samples,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("n_full", "n_full_failed", "p50_s", "p95_s", "p95_rank",
                       "max_s", "n_over_target", "n_qualified_and_under_target")},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())