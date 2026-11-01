"""Serial pipeline execution, provenance capture and no-overwrite reports."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np

from head_rgbd import __version__
from head_rgbd.audit import sha
from head_rgbd.paths import create_exclusive_dir, resolve_runtime_dir


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return str(value)


def write_json(path: str | Path, payload: Any, *, exclusive: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with target.open(mode, encoding="utf-8") as stream:
        json.dump(
            jsonable(payload),
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    return target


def timestamp_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


def safe_stage_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-_")
    if not cleaned:
        raise ValueError("stage name has no usable characters")
    return cleaned[:80]


def probe_gpu(timeout_s: float = 5.0) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "unavailable",
            "command": command,
            "error": f"{type(exc).__name__}: {exc}",
        }
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            rows.append({"raw": line})
            continue
        rows.append(
            {
                "name": fields[0],
                "memory_total_mib": int(fields[1]),
                "memory_used_mib": int(fields[2]),
                "memory_free_mib": int(fields[3]),
                "utilization_percent": int(fields[4]),
            }
        )
    return {
        "status": "ok" if completed.returncode == 0 else "error",
        "command": command,
        "exit_code": completed.returncode,
        "gpus": rows,
        "stderr": completed.stderr.strip() or None,
    }


def collect_environment(runtime: str | Path | None = None) -> dict[str, Any]:
    root = resolve_runtime_dir(runtime)
    result: dict[str, Any] = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "runtime": str(root),
        "runtime_env_python": str(root / "env" / "python.exe"),
        "cwd": str(Path.cwd()),
        "environment_variables": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "HEADRGBD_RUNTIME_DIR",
                "HEADRGBD_ASSET_ROOT",
                "HEADRGBD_TUFA_DEVICE",
                "UNIFACE_CACHE_DIR",
            )
            if os.environ.get(name) is not None
        },
        "gpu_before": probe_gpu(),
    }
    try:
        import torch

        result["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    except Exception as exc:
        result["torch"] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    try:
        import pyorbbecsdk

        result["orbbec_sdk"] = {"python_package": str(pyorbbecsdk.get_version())}
    except Exception as exc:
        result["orbbec_sdk"] = {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return result


def inspect_sequence(path: str | Path) -> dict[str, Any]:
    sequence_path = Path(path).expanduser().resolve()
    payload = json.loads(sequence_path.read_text(encoding="utf-8"))
    required = {
        "schema": "head_rgbd.sequence.v1",
        "depth_unit": "mm",
        "mesh_unit": "m",
        "coordinate_convention": "X_right_Y_up_Z_backward",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"sequence {key} must be {expected!r}, got {payload.get(key)!r}"
            )
    frames = payload.get("frames", [])
    candidates = [row for row in frames if row.get("status") == "candidate"]
    if not candidates:
        raise ValueError("sequence contains no candidate frames")
    for row in candidates:
        frame_path = (sequence_path.parent / row["file"]).resolve()
        if frame_path.parent != sequence_path.parent or not frame_path.is_file():
            raise ValueError(f"candidate frame is outside sequence or missing: {frame_path}")
    closed_oblique = [
        row
        for row in candidates
        if row.get("state") == "closed" and row.get("view") in {"left", "right"}
    ]
    segments = list(dict.fromkeys(row.get("segment") for row in candidates))
    return {
        "path": str(sequence_path),
        "sha256": sha(sequence_path),
        "candidate_frames": len(candidates),
        "excluded_frames": len(frames) - len(candidates),
        "segments": segments,
        "closed_oblique_frames": [row["id"] for row in closed_oblique],
        "front_frames": [row["id"] for row in candidates if row.get("view") == "front"],
        "open_frames": [row["id"] for row in candidates if row.get("state") == "open"],
        "has_real_closed_oblique": bool(closed_oblique),
    }


class StageRunner:
    """Execute stages serially and preserve commands, logs and exit status."""

    def __init__(
        self,
        output: str | Path,
        *,
        runtime: str | Path | None = None,
        purpose: str,
    ):
        self.output = create_exclusive_dir(output)
        self.runtime = resolve_runtime_dir(runtime)
        self.logs = self.output / "logs"
        self.logs.mkdir()
        self.purpose = purpose
        self.started_at = datetime.now().astimezone().isoformat()
        self.start = time.perf_counter()
        self.stages: list[dict[str, Any]] = []
        self.environment = collect_environment(self.runtime)
        self.report = {
            "schema": "head_rgbd.run.v1",
            "project_version": __version__,
            "purpose": purpose,
            "status": "running",
            "started_at": self.started_at,
            "output": str(self.output),
            "runtime": str(self.runtime),
            "environment": self.environment,
            "stages": self.stages,
            "statuses": {
                "migration_implementation": "implemented",
                "real_capture_pipeline": "not_run",
                "pipeline_execution": "running",
                "geometry_effect_acceptance": "not_evaluated",
                "independent_accuracy": "unverified",
            },
        }
        self.flush()

    @property
    def report_path(self) -> Path:
        return self.output / "run_report.json"

    def flush(self) -> None:
        write_json(self.report_path, self.report)

    def run(
        self,
        name: str,
        module: str,
        arguments: Iterable[str | Path],
        *,
        timeout_s: float | None = None,
        working_directory: str | Path | None = None,
    ) -> int:
        stage_name = safe_stage_name(name)
        number = len(self.stages) + 1
        command = [
            str(Path(sys.executable).resolve()),
            "-m",
            module,
            *[str(value) for value in arguments],
        ]
        log_path = self.logs / f"{number:02d}_{stage_name}.log"
        env = os.environ.copy()
        env["HEADRGBD_RUNTIME_DIR"] = str(self.runtime)
        env["PYTHONUNBUFFERED"] = "1"
        cwd=Path(working_directory).resolve() if working_directory is not None else Path(__file__).resolve().parents[1]
        row: dict[str, Any] = {
            "name": name,
            "module": module,
            "command": command,
            "cwd": str(cwd),
            "log": str(log_path),
            "started_at": datetime.now().astimezone().isoformat(),
            "gpu_before": probe_gpu(),
            "status": "running",
            "exit_code": None,
            "elapsed_s": None,
        }
        self.stages.append(row)
        self.flush()
        started = time.perf_counter()
        with log_path.open("x", encoding="utf-8") as log:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except OSError as exc:
                row.update(
                    status="failed",
                    exit_code=None,
                    elapsed_s=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.flush()
                return 127
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                exit_code = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                exit_code = 124
                log.write(f"\nstage timeout after {timeout_s}s\n")
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                row.update(
                    status="interrupted",
                    exit_code=process.returncode,
                    elapsed_s=time.perf_counter() - started,
                    finished_at=datetime.now().astimezone().isoformat(),
                )
                self.report["status"] = "interrupted"
                self.report["statuses"]["pipeline_execution"] = "interrupted"
                self.flush()
                raise
        row.update(
            status="passed" if exit_code == 0 else "failed",
            exit_code=exit_code,
            elapsed_s=time.perf_counter() - started,
            finished_at=datetime.now().astimezone().isoformat(),
            gpu_after=probe_gpu(),
        )
        self.flush()
        return exit_code

    def finish(
        self,
        *,
        statuses: dict[str, str] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> int:
        failed = [stage for stage in self.stages if stage["status"] != "passed"]
        self.report["status"] = "passed" if not failed and not self.report.get("error") else "failed"
        self.report["finished_at"] = datetime.now().astimezone().isoformat()
        self.report["elapsed_s"] = time.perf_counter() - self.start
        self.report["statuses"]["pipeline_execution"] = self.report["status"]
        if statuses:
            self.report["statuses"].update(statuses)
        if summary:
            self.report["summary"] = summary
        self.report["environment"]["gpu_after"] = probe_gpu()
        self.flush()
        return 0 if self.report["status"] == "passed" else 1


def require_stage_success(runner: StageRunner, exit_code: int) -> None:
    if exit_code != 0:
        raise RuntimeError(
            f"stage {runner.stages[-1]['name']!r} failed with exit code {exit_code}"
        )
