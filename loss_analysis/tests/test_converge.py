"""Contracts for the convergence plateau and time-ledger report (constructed data)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from loss_analysis import cli, converge


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def _log_lines(records: list[tuple[int, float, str]], *, mix: bool = True) -> str:
    lines = ["[Timing] Landmark 预测: 1.0s", "noise line"]
    for index, (step, value, phase) in enumerate(records):
        record = {"step": step, "phase": phase, "objective": value}
        if mix and index % 2 == 0:
            lines.append(repr(record))
        else:
            lines.append(json.dumps(record))
        if step % 40 == 0:
            lines.append(json.dumps({"step": step, "phase": phase,
                                     "correspondence_frames": 3}))
    return "\n".join(lines) + "\n"


def _make_run(root: Path, name: str = "run", *, cut: int = 180,
              elapsed: float = 24.0, with_dense: bool = True,
              neck_seconds: float | None = 12.5,
              missing_log: bool = False) -> Path:
    run = root / name
    logs = run / "logs"
    logs.mkdir(parents=True)
    # Phase-local curves: the final phase is flat from `cut` onward.
    values = {
        0: 1.0, 20: 0.9,
        40: 0.5, 60: 0.3, 80: 0.2, 100: 0.16,
        120: 0.15, 140: 0.15,
        160: 0.16,
        180: 0.149 if cut <= 180 else 0.155,
        200: 0.149, 220: 0.149, 240: 0.149,
    }
    phases = {0: "motion_initial", 20: "motion_initial",
              40: "identity_large", 60: "identity_large",
              80: "identity_large", 100: "identity_large",
              120: "identity_surface", 140: "identity_surface",
              160: "motion_final", 180: "motion_final",
              200: "motion_final", 220: "motion_final", 240: "motion_final"}
    records = [(step, value, phases[step]) for step, value in sorted(values.items())]
    fit_log = logs / "02_identity_front_anchor.log"
    fit_log.write_text(_log_lines(records), encoding="utf-8")
    motion_log = logs / "04_fixed_identity_motion.log"
    motion_log.write_text(_log_lines([
        (0, 0.20, "rigid"), (20, 0.19, "rigid"),
        (40, 0.25, "expression_jaw"),
        (60, 0.19, "rigid"), (79, 0.19, "rigid"),
    ]), encoding="utf-8")
    stages = [
        {"name": "prepare_observations", "elapsed_s": 10.0,
         "log": str(logs / "01_prepare_observations.log")},
        {"name": "identity_front_anchor", "elapsed_s": elapsed,
         "log": str(logs / "02_identity_front_anchor.log")},
        {"name": "fixed_identity_motion", "elapsed_s": 8.0,
         "log": str(logs / "04_fixed_identity_motion.log")},
        {"name": "guarded_measured_dense_surface", "elapsed_s": 20.0,
         "log": str(logs / "05_guarded_measured_dense_surface.log")},
    ]
    if missing_log:
        fit_log.unlink()
    if with_dense:
        _write(run / "reconstruction" / "dense" / "report.json", {
            "field_logs": {
                "front_anchor_regularized": {
                    "history": [
                        {"outer": 0, "objective_before": 1.5,
                         "objective_after": 1.2},
                        {"outer": 1, "objective_before": 1.3,
                         "objective_after": 1.1},
                    ]
                }
            }
        })
    _write(run / "run_report.json", {
        "project_version": "test",
        "elapsed_s": 100.0,
        "stages": stages,
    })
    if neck_seconds is not None:
        _write(run / "result" / "build_direct.json", {"seconds": neck_seconds})
    return run


class ObjectiveLogParsing(unittest.TestCase):
    def test_accepts_json_and_python_dicts_and_deduplicates_by_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stage.log"
            path.write_text(_log_lines([(0, 1.0, "a"), (20, 0.5, "a")]) +
                            repr({"step": 20, "phase": "a", "objective": 0.4}) + "\n",
                            encoding="utf-8")
            points = converge.read_objective_points(path)
        self.assertEqual([point["step"] for point in points], [0, 20])
        self.assertAlmostEqual(points[1]["value"], 0.4)

    def test_missing_log_returns_none_and_is_not_zero_filled(self):
        self.assertIsNone(converge.read_objective_points(Path("does-not-exist.log")))


class PhaseAnalysis(unittest.TestCase):
    def test_phase_local_gap_and_plateau_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(Path(temporary))
            report = converge.analyse_run(run)
        stage = next(row for row in report["stages"]
                     if row["name"] == "identity_front_anchor")
        self.assertEqual(stage["cut_scope"], "candidate")
        self.assertAlmostEqual(stage["elapsed_s"], 24.0)
        self.assertAlmostEqual(stage["per_step_s_estimate"], 24.0 / 240.0)
        final_phase = stage["phases"][-1]
        self.assertEqual(final_phase["phase"], "motion_final")
        self.assertEqual(final_phase["status"], "plateau")
        self.assertEqual(final_phase["candidate"]["cut_step"], 180)
        self.assertEqual(final_phase["candidate"]["saved_steps"], 60)
        self.assertEqual(final_phase["candidate"]["application"], "total_budget")
        self.assertAlmostEqual(final_phase["candidate"]["saved_s_estimate"], 6.0)
        large = next(row for row in stage["phases"]
                     if row["phase"] == "identity_large")
        self.assertEqual(large["status"], "no_plateau")
        surface = next(row for row in stage["phases"]
                       if row["phase"] == "identity_surface")
        self.assertEqual(surface["status"], "insufficient_points")

    def test_gap_is_measured_inside_the_same_phase_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(Path(temporary))
            report = converge.analyse_run(run)
        stage = next(row for row in report["stages"]
                     if row["name"] == "identity_front_anchor")
        first = stage["phases"][0]
        # phase-local reference: 0.9 is this phase's final raw value,
        # not the stage final (0.149) and not the next phase.
        self.assertAlmostEqual(first["final_value"], 0.9)
        self.assertAlmostEqual(first["points"][0]["gap_to_phase_final"], 0.1)
        self.assertAlmostEqual(first["points"][0]["relative_gap"], 0.1 / 0.9)
        surface = next(row for row in stage["phases"]
                       if row["phase"] == "identity_surface")
        self.assertAlmostEqual(surface["final_value"], 0.15)

    def test_late_plateau_is_the_earliest_cut_within_the_persist_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(Path(temporary))
            report = converge.analyse_run(run, persist_steps=40)
        stage = next(row for row in report["stages"]
                     if row["name"] == "identity_front_anchor")
        final_phase = stage["phases"][-1]
        self.assertEqual(final_phase["candidate"]["cut_step"], 180)
        surface = next(row for row in stage["phases"]
                       if row["phase"] == "identity_surface")
        self.assertEqual(surface["status"], "insufficient_points")


class LedgerAndGaps(unittest.TestCase):
    def test_dense_and_prepare_are_ledger_only_and_neck_seconds_are_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(Path(temporary))
            report = converge.analyse_run(run)
        stages = {row["name"]: row for row in report["stages"]}
        self.assertEqual(stages["guarded_measured_dense_surface"]["cut_scope"],
                         "ledger_only")
        self.assertIsNone(stages["guarded_measured_dense_surface"]["stage_candidate"])
        self.assertEqual(stages["prepare_observations"]["cut_scope"],
                         "ledger_only")
        self.assertEqual(stages["fixed_identity_motion"]["cut_scope"], "candidate")
        self.assertAlmostEqual(report["neck_seconds"], 12.5)
        neck = next(row for row in report["ledger"] if row["kind"] == "neck")
        self.assertAlmostEqual(neck["elapsed_s"], 12.5)
        self.assertEqual(neck["cut_scope"], "ledger_only")

    def test_missing_solver_log_is_a_gap_without_a_curve(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(Path(temporary), missing_log=True)
            report = converge.analyse_run(run)
        stage = next(row for row in report["stages"]
                     if row["name"] == "identity_front_anchor")
        self.assertEqual(stage["n_points"], 0)
        self.assertEqual(stage["cut_scope"], "ledger_only")
        self.assertIsNone(stage["stage_candidate"])
        self.assertTrue(any("objective log not found" in gap
                            for gap in report["gaps"]))

    def test_total_rebuild_time_is_read_from_the_shared_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _make_run(root)
            with (root / "rebuild_timings.jsonl").open("w", encoding="utf-8") as stream:
                stream.write(json.dumps({"output": str(run),
                                         "rebuild_elapsed_s": 140.0}) + "\n")
                stream.write(json.dumps({"output": str(root / "other"),
                                         "rebuild_elapsed_s": 999.0}) + "\n")
            report = converge.analyse_run(run)
        self.assertAlmostEqual(report["total_rebuild_elapsed_s"], 140.0)
        rest = next(row for row in report["ledger"] if row["kind"] == "rest")
        self.assertAlmostEqual(rest["elapsed_s"], 140.0 - 100.0 - 12.5)


class Aggregation(unittest.TestCase):
    def test_aggregate_takes_the_latest_cut_and_requires_every_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = converge.analyse_run(_make_run(root, "a"), persist_steps=40)
            later = converge.analyse_run(_make_run(root, "b", cut=200),
                                         persist_steps=40)
            partial = converge.analyse_run(_make_run(root, "c", cut=200),
                                           persist_steps=40)
            # Deliberately break the plateau of the third run.
            broken = next(stage for stage in partial["stages"]
                          if stage["name"] == "identity_front_anchor")
            for phase in broken["phases"]:
                phase["candidate"] = None
                phase["status"] = "no_plateau"
            rows = converge.aggregate_candidates([first, later])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["conservative_cut_step"], 200)
            self.assertEqual(rows[0]["status"], "all_runs_plateau")
            mixed = converge.aggregate_candidates([first, later, partial])
            self.assertEqual(mixed[0]["status"], "partial_runs")

    def test_reference_run_is_not_aggregated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate = _make_run(root, "aggregate")
            reference = _make_run(root, "reference", cut=200)
            out = root / "out"
            code = converge.run_converge(SimpleNamespace(
                run=[aggregate], reference_run=[reference],
                epsilon=0.002, persist_steps=60, output=out, no_figures=True,
            ))
            self.assertEqual(code, 0)
            report = json.loads((out / "convergence.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["runs"]), 2)
            self.assertEqual(report["evidence"]["aggregate"], [str(aggregate.resolve())])
            self.assertEqual(report["evidence"]["reference"], [str(reference.resolve())])
            self.assertEqual(len(report["aggregate"]), 1)
            self.assertEqual(report["aggregate"][0]["runs_required"], 1)


class Outputs(unittest.TestCase):
    def test_writes_json_csv_html_and_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _make_run(root)
            args = SimpleNamespace(run=[run], reference_run=[], epsilon=0.002,
                                   persist_steps=60, output=root / "out",
                                   no_figures=True)
            self.assertEqual(converge.run_converge(args), 0)
            for name in ("convergence.json", "convergence.csv",
                         "candidates.csv", "report.md", "report.html"):
                self.assertTrue((root / "out" / name).is_file(), name)
            html = (root / "out" / "report.html").read_text(encoding="utf-8")
            self.assertIn("逐阶段收敛与省时账本", html)
            self.assertIn("insufficient_points", html)
            with self.assertRaises(FileExistsError):
                converge.run_converge(args)

    def test_cli_requires_at_least_one_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            code = cli.main(["converge", "--output", str(output)])
        self.assertEqual(code, 2)

    def test_invalid_epsilon_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _make_run(root)
            code = converge.run_converge(SimpleNamespace(
                run=[run], reference_run=[], epsilon=1.0,
                persist_steps=60, output=root / "out", no_figures=True,
            ))
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
