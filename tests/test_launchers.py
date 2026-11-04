"""Exercise native Windows launchers without opening a camera or loading models."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import venv

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows launchers")
PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def launcher_project(tmp_path_factory):
    # Test the unmodified wrappers with a real interpreter and a harmless entry
    # point. Spaces and an unrelated CWD exercise Explorer-style path handling.
    project = tmp_path_factory.mktemp("headrgbd launch ")
    for name in ("headrgbd.cmd", "headrgbd.ps1"):
        shutil.copyfile(PROJECT / name, project / name)
    runtime = project / "runtime" / "env"
    venv.EnvBuilder(with_pip=False).create(runtime)
    shutil.copyfile(runtime / "Scripts" / "python.exe", runtime / "python.exe")
    package = project / "head_rgbd"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(
        "import json, os, sys\n"
        "print('LAUNCH_ARGS=' + json.dumps(sys.argv[1:]))\n"
        "raise SystemExit(int(os.environ.get('HEADRGBD_TEST_EXIT_CODE', '0')))\n",
        encoding="utf-8",
    )
    return project


def launch(project, kind, arguments=(), *, exit_code=0):
    env = os.environ.copy()
    for name in ("WITHB_RUNTIME_ROOT", "HEADRGBD_RUNTIME_DIR", "PYTHONHOME"):
        env.pop(name, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HEADRGBD_TEST_EXIT_CODE"] = str(exit_code)
    script = project / f"headrgbd.{kind}"
    if kind == "cmd":
        # CMD /s /c requires an outer quote pair around a quoted script path.
        suffix = " " + subprocess.list2cmdline(list(arguments)) if arguments else ""
        command = f'"{os.environ["COMSPEC"]}" /d /s /c ""{script}"{suffix}"'
    else:
        powershell = shutil.which("pwsh.exe") or shutil.which("powershell.exe")
        assert powershell, "PowerShell must be available on Windows"
        command = [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
                   "Bypass", "-File", str(script), *arguments]
    return subprocess.run(command, cwd=project.parent, env=env, input="\n",
                          encoding="utf-8", errors="replace", capture_output=True,
                          timeout=30)


def launched_arguments(result):
    lines = [line for line in result.stdout.splitlines() if line.startswith("LAUNCH_ARGS=")]
    assert len(lines) == 1, (result.stdout, result.stderr)
    return json.loads(lines[0].partition("=")[2])


@pytest.mark.parametrize("kind", ["cmd", "ps1"])
def test_no_arguments_start_default_run(launcher_project, kind):
    result = launch(launcher_project, kind)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert launched_arguments(result) == ["run"]


@pytest.mark.parametrize("kind", ["cmd", "ps1"])
@pytest.mark.parametrize("arguments", [
    ["--help"], ["--version"], ["run", "--help"],
    ["reconstruct", "--capture", "a capture with spaces"],
])
def test_explicit_arguments_are_preserved(launcher_project, kind, arguments):
    result = launch(launcher_project, kind, arguments)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert launched_arguments(result) == arguments


@pytest.mark.parametrize("arguments, expected", [
    (["-Version"], ["--version"]),
    (["-Help"], ["--help"]),
    (["run", "-Help"], ["run", "--help"]),
])
def test_powershell_switches(launcher_project, arguments, expected):
    result = launch(launcher_project, "ps1", arguments)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert launched_arguments(result) == expected


@pytest.mark.parametrize("kind", ["cmd", "ps1"])
def test_default_failure_keeps_exit_code(launcher_project, kind):
    result = launch(launcher_project, kind, exit_code=7)
    assert result.returncode == 7, (result.stdout, result.stderr)
    assert launched_arguments(result) == ["run"]
    if kind == "cmd":
        assert "HeadRGBD stopped with error code 7." in result.stdout


def test_explicit_cmd_failure_does_not_pause(launcher_project):
    result = launch(launcher_project, "cmd", ["run"], exit_code=7)
    assert result.returncode == 7, (result.stdout, result.stderr)
    assert "HeadRGBD stopped with error code" not in result.stdout


def test_missing_runtime_reports_failure_without_flashing_closed(tmp_path):
    shutil.copyfile(PROJECT / "headrgbd.cmd", tmp_path / "headrgbd.cmd")
    result = launch(tmp_path, "cmd")
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert "Runtime Python not found:" in result.stdout
