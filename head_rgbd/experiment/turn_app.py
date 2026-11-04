"""One-command continuous-turn RGB-D application.

No arguments opens the GUI. ``run`` captures then reconstructs and verifies;
``reconstruct --capture <folder>`` reuses a recorded anchor/turn pair.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback

from head_rgbd.paths import PROJECT_ROOT, resolve_runtime_dir


def runtime_python() -> Path:
    """Always launch a console interpreter for subprocess log collection."""
    bundled = PROJECT_ROOT / "runtime" / "env" / "python.exe"
    if bundled.is_file():
        return bundled
    current = Path(sys.executable)
    return current.with_name("python.exe") if current.name.lower() == "pythonw.exe" else current


def new_run_path(parent: Path, prefix: str = "gui_turn") -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return Path(parent).expanduser().resolve() / f"{prefix}_{stamp}"


def capture_inputs(folder) -> tuple[Path, Path]:
    folder = Path(folder).expanduser().resolve()
    anchor = folder / "anchor_session.json"
    turn = folder / "turn_session.json"
    missing = [path.name for path in (anchor, turn) if not path.is_file()]
    if missing:
        raise ValueError(f"采集目录缺少 {', '.join(missing)}。请选择连续转头采集生成的目录。")
    return anchor, turn


def reconstruct(*, anchor, turn, runtime=None, config=None, device=None, output=None,
                max_keyframes=None, detect_stride=None) -> dict:
    from head_rgbd.experiment.turn_run import main as run_experiment
    from head_rgbd.reporting import write_json
    from tools.verify_turn_run import verify

    runtime = resolve_runtime_dir(runtime)
    output = Path(output).expanduser().resolve() if output else new_run_path(runtime / "turn_experiment", "turn")
    arguments = ["--anchor", str(anchor), "--turn", str(turn),
                 "--runtime", str(runtime), "--output", str(output)]
    for name, value in (("--config", config), ("--device", device),
                        ("--max-keyframes", max_keyframes), ("--detect-stride", detect_stride)):
        if value is not None:
            arguments.extend((name, str(value)))
    code = run_experiment(arguments)
    if code:
        raise RuntimeError(f"重建进程退出码为 {code}")
    verification = verify(output)
    write_json(output / "verification.json", verification, exclusive=True)
    if verification["status"] != "passed":
        raise RuntimeError("产物验收失败：" + "; ".join(verification["errors"]))
    result = {"run": str(output), "delivery": str(output / "delivery"),
              "verification": str(output / "verification.json"),
              "elapsed_s": round(verification["elapsed_s"], 3),
              "keyframes": verification["keyframes"], "status": "passed"}
    print("TURN_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def environment_check(runtime=None, device=None) -> dict:
    from importlib.metadata import version
    import torch
    from head_rgbd.pipeline import asset_gate

    runtime = resolve_runtime_dir(runtime)
    assets = asset_gate(runtime)
    cuda = bool(torch.cuda.is_available())
    if device == "cuda" and not cuda:
        raise RuntimeError("当前环境无法使用 CUDA，请选择 CPU 或检查显卡驱动。")
    result = {"runtime": str(runtime), "python": str(runtime_python()),
              "assets": assets["status"], "asset_count": assets["count"],
              "gui": "PyQt6 " + version("PyQt6"), "torch": torch.__version__,
              "cuda_available": cuda,
              "gpu": torch.cuda.get_device_name(0) if cuda else None}
    print("TURN_ENVIRONMENT=" + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _settings(parser):
    parser.add_argument("--runtime", type=Path, help="runtime directory; defaults to the project runtime")
    parser.add_argument("--config", type=Path, help="experiment JSON configuration")
    parser.add_argument("--device", choices=("cuda", "cpu"))


def _fit_options(parser):
    _settings(parser)
    parser.add_argument("--output", type=Path, help="new reconstruction directory; never overwritten")
    parser.add_argument("--max-keyframes", type=int)
    parser.add_argument("--detect-stride", type=int)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    gui = sub.add_parser("gui", help="open the desktop GUI")
    _settings(gui)
    gui.add_argument("--capture", type=Path, help="preselect an existing continuous-turn capture")
    run = sub.add_parser("run", help="capture, reconstruct and verify in one command")
    _fit_options(run)
    run.add_argument("--capture-output", type=Path)
    run.add_argument("--turn-frames", type=int, default=100)
    run.add_argument("--interval-s", type=float, default=.12)
    recorded = sub.add_parser("reconstruct", help="reconstruct and verify recorded RGB-D")
    _fit_options(recorded)
    recorded.add_argument("--capture", type=Path, help="folder containing anchor_session.json and turn_session.json")
    recorded.add_argument("--anchor", type=Path)
    recorded.add_argument("--turn", type=Path)
    capture_parser = sub.add_parser("capture", help="record the two RGB-D clips only")
    capture_parser.add_argument("--runtime", type=Path)
    capture_parser.add_argument("--output", type=Path)
    capture_parser.add_argument("--turn-frames", type=int, default=100)
    capture_parser.add_argument("--interval-s", type=float, default=.12)
    check = sub.add_parser("check", help="check runtime, assets and compute device without opening the camera")
    check.add_argument("--runtime", type=Path)
    check.add_argument("--device", choices=("cuda", "cpu"))
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    command = args.command or "gui"
    if command == "gui":
        try:
            from head_rgbd.experiment.turn_gui import launch
            return launch(runtime=getattr(args, "runtime", None), config=getattr(args, "config", None),
                          device=getattr(args, "device", None), capture_folder=getattr(args, "capture", None))
        except Exception:
            error = traceback.format_exc()
            logs = PROJECT_ROOT / "runtime" / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / "turn_gui_startup_error.log").write_text(error, encoding="utf-8")
            if os.name == "nt" and sys.stderr is None:
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, "GUI 启动失败。\n" + error[-1400:], "连续转头 RGB-D", 0x10)
            else:
                print(error, file=sys.stderr)
            return 1
    from head_rgbd.experiment.turn_run import _configure_stdio
    _configure_stdio()
    try:
        if command == "check":
            environment_check(args.runtime, args.device)
        elif command == "capture":
            from head_rgbd.experiment.turn_capture import capture
            capture(output=args.output, runtime=args.runtime,
                    turn_frames=args.turn_frames, interval_s=args.interval_s)
        else:
            if command == "run":
                from head_rgbd.experiment.turn_capture import capture
                folder = capture(output=args.capture_output, runtime=args.runtime,
                                 turn_frames=args.turn_frames, interval_s=args.interval_s)
                anchor, turn = capture_inputs(folder)
            elif args.capture:
                if args.anchor or args.turn:
                    parser.error("--capture cannot be combined with --anchor/--turn")
                anchor, turn = capture_inputs(args.capture)
            elif args.anchor and args.turn:
                anchor, turn = args.anchor, args.turn
            else:
                parser.error("reconstruct requires --capture or both --anchor and --turn")
            reconstruct(anchor=anchor, turn=turn, runtime=args.runtime, config=args.config,
                        device=args.device, output=args.output, max_keyframes=args.max_keyframes,
                        detect_stride=args.detect_stride)
    except KeyboardInterrupt:
        print("任务已取消；已写入的数据和阶段日志已保留。", flush=True)
        return 130
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
