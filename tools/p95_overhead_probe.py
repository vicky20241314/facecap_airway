"""Attribute the non-fit overhead of a rebuild run (A0 / section B).

Times the individual pieces that the stage reports do not cover: asset hashing,
GPU probes, environment collection, source hashing, OBJ parsing, and the
interpreter+import startup of each stage module.  Read-only: it changes no
algorithm and writes only into a new --out directory.

Run:
    python -m tools.p95_overhead_probe --run <existing rebuild run dir> --out <new dir>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

PYTHON = PROJ / "runtime" / "env" / "python.exe"
STAGE_MODULES = ("head_rgbd.prepare", "head_rgbd.fit_c", "head_rgbd.motion_v2",
                 "head_rgbd.fit_d", "tools.build_neck_direct_v2")


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["HEADRGBD_RUNTIME_DIR"] = str(PROJ / "runtime")
    env["WITHB_RUNTIME_ROOT"] = str(PROJ / "runtime")
    return env


def timed(fn, reps=1):
    best, out = None, None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return round(best, 4), out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    rec = {}

    from head_rgbd import reporting, paths, pipeline, mesh_integrity as mi

    t, gpu = timed(lambda: reporting.probe_gpu(), reps=3)
    rec["probe_gpu_s"] = t
    rec["probe_gpu_status"] = gpu.get("status")
    t, _ = timed(lambda: reporting.collect_environment(PROJ / "runtime"))
    rec["collect_environment_s"] = t
    t, gate = timed(lambda: pipeline.asset_gate(PROJ / "runtime"))
    rec["asset_gate_s"] = t
    rec["asset_gate_count"] = gate.get("count")
    rec["asset_gate_bytes"] = sum(r.get("size_bytes") or 0 for r in gate["assets"])

    from head_rgbd.audit import sha
    files = [*Path(PROJ / "head_rgbd").glob("*.py"),
             PROJ / "config" / "defaults.json", PROJ / "VERSION"]
    t, _ = timed(lambda: {str(p): sha(p) for p in files})
    rec["source_hashes_s"] = t
    rec["source_hashes_files"] = len(files)

    objs = sorted((args.run / "result").rglob("head_*.obj"))
    per_obj = []
    for p in objs[:8]:
        t, (V, F) = timed(lambda p=p: mi.load_obj(p))
        per_obj.append({"file": str(p.relative_to(args.run)),
                        "lines_approx": int(len(V) + len(F)),
                        "load_obj_s": t})
    rec["load_obj"] = per_obj
    rec["load_obj_total_s"] = round(sum(r["load_obj_s"] for r in per_obj), 4)

    startup = []
    for mod in STAGE_MODULES:
        code = ("import time,sys;t0=time.perf_counter();"
                f"import {mod};"
                "print('IMPORT_S=%.4f'%(time.perf_counter()-t0))")
        t0 = time.perf_counter()
        cp = subprocess.run([str(PYTHON), "-c", code], cwd=str(PROJ),
                            env=_env(), capture_output=True, text=True)
        wall = time.perf_counter() - t0
        inner = None
        for line in cp.stdout.splitlines():
            if line.startswith("IMPORT_S="):
                inner = float(line.split("=")[1])
        startup.append({"module": mod, "exit": cp.returncode,
                        "process_wall_s": round(wall, 4),
                        "import_s": inner,
                        "interpreter_before_import_s": (round(wall - inner, 4)
                                                        if inner is not None else None),
                        "stderr_tail": cp.stderr[-300:] if cp.returncode else ""})
    rec["stage_startup"] = startup
    rec["stage_startup_total_s"] = round(sum(r["process_wall_s"] for r in startup), 4)

    t0 = time.perf_counter()
    subprocess.run([str(PYTHON), "-c", "pass"], cwd=str(PROJ), env=_env(),
                   capture_output=True)
    rec["bare_interpreter_s"] = round(time.perf_counter() - t0, 4)

    (args.out / "overhead_probe.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())