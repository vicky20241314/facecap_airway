"""Runtime contracts that require real child processes but no model inference."""
import os
import json
from pathlib import Path
import subprocess
import sys


def test_utf8_logs_work_with_a_gbk_parent_environment():
    env = dict(os.environ, PYTHONIOENCODING="gbk", PYTHONUTF8="0")
    script = ("from head_rgbd.experiment.turn_run import _configure_stdio; "
              "_configure_stdio(); import os; print(chr(0x524)); "
              "print(os.environ['PYTHONIOENCODING'])")
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(__file__).resolve().parents[1], env=env,
                            capture_output=True, encoding="utf-8", timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [chr(0x524), "utf-8"]


def test_delivery_statistics_include_its_own_final_exit_status(tmp_path):
    from head_rgbd.experiment.turn_run import STAGE_ORDER, _finalize_delivery_statistics

    delivery = tmp_path / "delivery"
    delivery.mkdir()
    path = delivery / "stages.json"
    path.write_text(json.dumps({"stages": [{"name": "delivery", "status": "running", "exit_code": None}],
                               "observation_stats": {"keyframes": 7}}), encoding="utf-8")
    stages = [{"name": name, "module": "example", "status": "passed", "exit_code": 0, "elapsed_s": 1.}
              for name in STAGE_ORDER]
    _finalize_delivery_statistics(tmp_path, {"stages": stages, "elapsed_s": 8.5})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["stages"]) == 8
    assert payload["stages"][-1]["status"] == "passed"
    assert payload["stages"][-1]["exit_code"] == 0
    assert payload["total_elapsed_s"] == 8.5
    assert payload["observation_stats"] == {"keyframes": 7}
