"""Native new launchers: no-argument GUI routing, quoting and exit codes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import venv

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows launchers")
PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def turn_launcher_project(tmp_path_factory):
    project = tmp_path_factory.mktemp("turn launcher with spaces ")
    for suffix in ("cmd", "ps1"):
        shutil.copyfile(PROJECT / f"headrgbd_turn.{suffix}", project / f"headrgbd_turn.{suffix}")
    runtime = project / "runtime/env"
    venv.EnvBuilder(with_pip=False).create(runtime)
    for name in ("python.exe", "pythonw.exe"):
        shutil.copyfile(runtime / "Scripts" / name, runtime / name)
    package = project / "head_rgbd/experiment"
    package.mkdir(parents=True)
    (project / "head_rgbd/__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "turn_app.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['TURN_LAUNCH_RECORD']).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n"
        "raise SystemExit(int(os.environ.get('TURN_TEST_EXIT', '0')))\n", encoding="utf-8")
    return project


def launch(project, suffix, arguments, record, code=0):
    env = os.environ.copy()
    for key in ("WITHB_RUNTIME_ROOT", "HEADRGBD_RUNTIME_DIR", "PYTHONHOME"):
        env.pop(key, None)
    env.update(TURN_LAUNCH_RECORD=str(record), TURN_TEST_EXIT=str(code))
    script = project / f"headrgbd_turn.{suffix}"
    if suffix == "cmd":
        extra = " " + subprocess.list2cmdline(arguments) if arguments else ""
        command = f'"{os.environ["COMSPEC"]}" /d /s /c ""{script}"{extra}"'
    else:
        command = [shutil.which("pwsh.exe"), "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
                   "Bypass", "-File", str(script), *arguments]
    return subprocess.run(command, cwd=project.parent, env=env, input="\n",
                          capture_output=True, encoding="utf-8", errors="replace", timeout=20)


@pytest.mark.parametrize("suffix", ["cmd", "ps1"])
def test_no_arguments_open_gui(turn_launcher_project, suffix, tmp_path):
    record = tmp_path / "launch.json"
    result = launch(turn_launcher_project, suffix, [], record)
    assert result.returncode == 0, (result.stdout, result.stderr)
    for _ in range(100):
        if record.exists():
            break
        time.sleep(.02)
    assert json.loads(record.read_text(encoding="utf-8")) == ["gui"]


@pytest.mark.parametrize("suffix", ["cmd", "ps1"])
@pytest.mark.parametrize("arguments", [["--help"], ["run", "--help"],
                                       ["reconstruct", "--capture", "中文 recording with spaces"]])
def test_arguments_preserve_spaces_and_unicode(turn_launcher_project, suffix, arguments, tmp_path):
    record = tmp_path / "launch.json"
    result = launch(turn_launcher_project, suffix, arguments, record)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert json.loads(record.read_text(encoding="utf-8")) == arguments


@pytest.mark.parametrize("suffix", ["cmd", "ps1"])
def test_explicit_failure_keeps_exit_code(turn_launcher_project, suffix, tmp_path):
    result = launch(turn_launcher_project, suffix, ["reconstruct"], tmp_path / "launch.json", code=7)
    assert result.returncode == 7, (result.stdout, result.stderr)


def test_missing_runtime_is_reported(tmp_path):
    shutil.copyfile(PROJECT / "headrgbd_turn.cmd", tmp_path / "headrgbd_turn.cmd")
    result = launch(tmp_path, "cmd", [], tmp_path / "launch.json")
    assert result.returncode == 1
    assert "Runtime Python not found" in result.stdout
