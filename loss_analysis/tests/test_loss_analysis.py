"""Mathematical and interface contracts for the local loss analyser.

Constructed data only. These tests prove the arithmetic and the interfaces are
correct; they say nothing about reconstruction quality. No real capture, no GPU,
no model weights and no reconstruction are touched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import base64
import unittest

import numpy as np

from loss_analysis import cli, objectives, terms
from loss_analysis import adapter, neck_adapter, optimizer
from loss_analysis.recorder import IterationRecord, Recorder
from loss_analysis.terms import (
    coverage_summary,
    expression_prior_fit_c,
    field_equation_term,
    fixed_ray_energy_mm2,
    aggregate_fixed_ray,
    landmark_term,
    metric_summary,
    motion_frame_acceptance,
    mouth_gap_term,
    quadratic_prior,
    smooth_l1_beta03,
    smooth_l1_beta1,
    strict_acceptance,
    summarise,
    uv_term,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class SmoothL1Contract(unittest.TestCase):
    """The robust kernel must match PyTorch's definition exactly."""

    def test_beta03_quadratic_and_linear_branches(self):
        x = np.array([0.0, 0.1, 0.2999, 0.3, 0.5, 2.0])
        got = smooth_l1_beta03(x)
        expected = np.array([
            0.0,
            0.5 * 0.01 / 0.3,
            0.5 * 0.2999**2 / 0.3,
            0.3 - 0.15,
            0.5 - 0.15,
            2.0 - 0.15,
        ])
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_beta1_branches(self):
        np.testing.assert_allclose(smooth_l1_beta1(np.array([0.0, 0.5, 1.0, 3.0])),
                                   np.array([0.0, 0.125, 0.5, 2.5]))

    def test_known_bias_produces_known_penalty(self):
        """A pure 10 mm offset at scale 1 mm normalises to 10, i.e. 10 - 0.15."""
        per = smooth_l1_beta03(np.array([10.0]))
        self.assertAlmostEqual(float(per[0]), 10.0 - 0.15)


class LandmarkTermContract(unittest.TestCase):
    def _landmarks(self, offset_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target = np.zeros((68, 3))
        model = target.copy()
        model[17:48] = offset_m
        valid = np.ones(68, bool)
        return model, target, valid

    def test_known_offset_with_weight_and_denominator(self):
        model, target, valid = self._landmarks(0.002)
        result = landmark_term(model, target, valid, indices=slice(17, 48), scale_m=0.01,
                               weight=1.0, stage="fit_c", key="landmark_xyz_stable")
        self.assertEqual(result.status, "recomputed")
        self.assertEqual(result.n_valid, 31 * 3)
        # 2 mm / 10 mm = 0.2 -> inside the quadratic branch
        self.assertAlmostEqual(result.raw, 0.5 * 0.2**2 / 0.3)
        self.assertAlmostEqual(result.value, result.raw * 1.0)

    def test_weight_is_applied_linearly(self):
        model, target, valid = self._landmarks(0.002)
        a = landmark_term(model, target, valid, indices=slice(17, 48), scale_m=0.01,
                          weight=1.0, stage="fit_c", key="k")
        b = landmark_term(model, target, valid, indices=slice(17, 48), scale_m=0.01,
                          weight=1.4, stage="fit_c", key="k")
        self.assertAlmostEqual(b.value, 1.4 * a.value)

    def test_invalid_landmarks_are_excluded_not_zeroed(self):
        model, target, valid = self._landmarks(0.002)
        model[20] = 5.0          # gross outlier, valid flag still true -> must count
        reference = landmark_term(model, target, valid, indices=slice(17, 48), scale_m=0.01,
                                  weight=1.0, stage="fit_c", key="k")
        self.assertGreater(reference.raw, 1.0)

        valid[20] = False        # now excluded: the outlier must not enter the mean
        excluded = landmark_term(model, target, valid, indices=slice(17, 48), scale_m=0.01,
                                 weight=1.0, stage="fit_c", key="k")
        self.assertEqual(excluded.n_valid, (31 - 1) * 3)
        self.assertAlmostEqual(excluded.raw, 0.5 * 0.2**2 / 0.3)
        self.assertLess(excluded.raw, reference.raw)

    def test_index_set_outside_window_is_ignored(self):
        model, target, valid = self._landmarks(0.002)
        model[10] = 100.0        # index 10 is outside [17:48] and must never enter
        result = landmark_term(model, target, valid, indices=slice(17, 48), scale_m=0.01,
                               weight=1.0, stage="fit_c", key="k")
        self.assertEqual(result.n_valid, 31 * 3)
        self.assertAlmostEqual(result.raw, 0.5 * 0.2**2 / 0.3)

    def test_zero_valid_samples_is_null_not_zero(self):
        model, target, valid = self._landmarks(0.002)
        valid[:] = False
        result = landmark_term(model, target, valid, indices=slice(17, 48), scale_m=0.01,
                               weight=1.0, stage="fit_c", key="k")
        self.assertIsNone(result.value)
        self.assertEqual(result.status, "no_valid_samples")
        self.assertIn("NOT zero", result.notes)

    def test_nan_target_is_not_a_zero_residual(self):
        model, target, valid = self._landmarks(0.002)
        target[30] = np.nan
        result = landmark_term(model, target, valid, indices=slice(17, 48), scale_m=0.01,
                               weight=1.0, stage="fit_c", key="k")
        self.assertEqual(result.n_valid, 31 * 3 - 3)


class UvTermContract(unittest.TestCase):
    def test_pixel_normalisation(self):
        uv = np.zeros((68, 2))
        target = np.zeros((68, 2))
        uv[48:68, 0] = 10.0  # 10 px offset = 1.0 at the 10 px scale
        valid = np.ones(68, bool)
        result = uv_term(uv, target, valid, indices=slice(48, 68), scale_px=10.0,
                         weight=0.35, stage="fit_c", key="mouth_uv_front")
        # 20 landmarks x 2 interleaved (u,v) components
        self.assertEqual(result.n_valid, 40)
        # only the u component is shifted, so 20 of the 40 components are non-zero:
        # each is 10 px / 10 px = 1.0 -> linear branch of beta=0.3 -> 1.0 - 0.15
        self.assertAlmostEqual(result.raw, 20 * (1.0 - 0.15) / 40)
        self.assertAlmostEqual(result.value, 0.35 * 20 * (1.0 - 0.15) / 40)

    def test_missing_detection_is_null(self):
        uv = np.full((68, 2), np.nan)
        target = np.zeros((68, 2))
        valid = np.zeros(68, bool)
        result = uv_term(uv, target, valid, indices=slice(48, 68), scale_px=10.0,
                         weight=0.35, stage="fit_c", key="k")
        self.assertIsNone(result.value)
        self.assertEqual(result.status, "no_valid_samples")


class MouthGapContract(unittest.TestCase):
    PAIRS = np.array([[61, 67], [62, 66], [63, 65]])

    def test_inside_deadband_is_zero(self):
        xyz = np.zeros((68, 3))
        xyz[61, 0] = 0.001
        xyz[67, 0] = 0.0
        xyz[62, 0] = 0.001
        xyz[66, 0] = 0.0
        xyz[63, 0] = 0.001
        xyz[65, 0] = 0.0
        result = mouth_gap_term(xyz, self.PAIRS, weight=0.15, stage="fit_c")
        self.assertAlmostEqual(result.raw, 0.0)

    def test_hinge_value_matches_formula(self):
        xyz = np.zeros((68, 3))
        for a, b in self.PAIRS:
            xyz[a, 0] = 0.005  # 5 mm gap
        # relu(5-2)/3 = 1.0 -> squared 1.0
        result = mouth_gap_term(xyz, self.PAIRS, weight=0.15, stage="fit_c")
        self.assertAlmostEqual(result.raw, 1.0)
        self.assertAlmostEqual(result.extras["mean_gap_mm"], 5.0)

    def test_invalid_depth_is_not_a_shut_mouth(self):
        xyz = np.full((68, 3), np.nan)
        result = mouth_gap_term(xyz, self.PAIRS, weight=0.15, stage="fit_c")
        self.assertIsNone(result.value)
        self.assertEqual(result.status, "no_valid_samples")


class PriorContract(unittest.TestCase):
    def test_expression_prior_uses_frames_times_ten(self):
        expr = np.ones((4, 30))
        result = expression_prior_fit_c(expr, frames=4)
        # sum = 120, denominator = 40 -> raw 3.0
        self.assertAlmostEqual(result.raw, 3.0)
        self.assertAlmostEqual(result.value, 0.002 * 3.0)

    def test_mean_prior_matches_numpy(self):
        values = np.array([1.0, 2.0, 3.0])
        result = quadratic_prior(values, weight=0.01, stage="fit_c", key="jaw_prior")
        self.assertAlmostEqual(result.raw, float(np.mean(values**2)))


class FieldEquationContract(unittest.TestCase):
    def test_known_linear_residual(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 2.0])
        rw = np.ones(2)
        at_field = field_equation_term(A, b, rw, field_mm=np.array([0.0, 0.0]), stage="field")
        self.assertAlmostEqual(at_field.value, 5.0)
        solved = field_equation_term(A, b, rw, field_mm=b, stage="field")
        self.assertAlmostEqual(solved.value, 0.0)

    def test_weights_scale_quadratically(self):
        A = np.array([[1.0]])
        b = np.array([2.0])
        one = field_equation_term(A, b, np.ones(1), field_mm=np.zeros(1), stage="field")
        two = field_equation_term(A, b, np.full(1, 2.0), field_mm=np.zeros(1), stage="field")
        self.assertAlmostEqual(two.value, 4 * one.value)

    def test_empty_system_is_null(self):
        result = field_equation_term(np.zeros((0, 3)), np.zeros(0), np.zeros(0), stage="field")
        self.assertIsNone(result.value)
        self.assertEqual(result.status, "no_valid_samples")


class FixedRayContract(unittest.TestCase):
    def test_missing_rays_are_penalised_not_dropped(self):
        residual = np.array([0.0, 0.0])
        covered = np.array([True, False])
        strict = fixed_ray_energy_mm2(residual, covered, policy="strict-coverage", stage="motion_v2")
        accounted = fixed_ray_energy_mm2(residual, covered, policy="accounted-research", stage="motion_v2")
        self.assertAlmostEqual(strict, (0.0 + 64.0) / 2)
        self.assertAlmostEqual(accounted, (0.0 + 1024.0) / 2)

    def test_fewer_rays_cannot_look_better(self):
        full = np.array([1.0, 1.0, 1.0, 1.0])
        full_cover = np.ones(4, bool)
        shrinking = fixed_ray_energy_mm2(full, np.array([True, False, False, False]), stage="m")
        complete = fixed_ray_energy_mm2(full, full_cover, stage="m")
        self.assertGreater(shrinking, complete)

    def test_no_ray_returns_null(self):
        self.assertIsNone(fixed_ray_energy_mm2(np.zeros(0), np.zeros(0, bool), stage="m"))
        self.assertIsNone(aggregate_fixed_ray([None, None], [0, 0]))

    def test_strict_acceptance_rule(self):
        self.assertTrue(strict_acceptance(mse_before=1.0, mse_after=1.01, missing_before=3, missing_after=3)["accepted"])
        self.assertFalse(strict_acceptance(mse_before=1.0, mse_after=0.1, missing_before=3, missing_after=4)["accepted"])
        self.assertFalse(strict_acceptance(mse_before=1.0, mse_after=1.5, missing_before=3, missing_after=3)["accepted"])

    def test_motion_acceptance_rule(self):
        ok = motion_frame_acceptance(mse_before=1.0, mse_after=1.0, required_rays=50, covered_required=50)
        self.assertTrue(ok["accepted"])
        worse = motion_frame_acceptance(mse_before=1.0, mse_after=1.0001, required_rays=50, covered_required=50)
        self.assertFalse(worse["accepted"])
        self.assertEqual(worse["reason"], "mse_increased")
        lost = motion_frame_acceptance(mse_before=1.0, mse_after=0.5, required_rays=50, covered_required=49)
        self.assertEqual(lost["reason"], "ray_coverage_lost")
        thin = motion_frame_acceptance(mse_before=1.0, mse_after=0.5, required_rays=10, covered_required=10)
        self.assertEqual(thin["reason"], "insufficient_fixed_rays")


class DiagnosticsContract(unittest.TestCase):
    def test_metric_summary_ignores_invalid(self):
        values = np.array([1.0, 2.0, np.nan, 4.0])
        summary = metric_summary(values)
        self.assertEqual(summary["n_valid"], 3)
        self.assertAlmostEqual(summary["mae_mm"], (1 + 2 + 4) / 3)
        self.assertAlmostEqual(summary["rmse_mm"], np.sqrt((1 + 4 + 16) / 3))

    def test_metric_summary_zero_samples(self):
        summary = metric_summary(np.array([np.nan, np.nan]))
        self.assertEqual(summary["status"], "no_valid_samples")
        self.assertIsNone(summary["mae_mm"])

    def test_coverage_counts(self):
        covered = np.array([True, False, True])
        cov = coverage_summary(covered)
        self.assertAlmostEqual(cov["coverage"], 2 / 3)
        self.assertEqual(cov["n_missing"], 1)
        self.assertIsNone(coverage_summary(np.zeros(0, bool))["coverage"])


class SummariseContract(unittest.TestCase):
    def test_total_excludes_null_terms_and_flags_partial(self):
        good = terms.TermResult(key="a", stage="s", provenance="solver_objective", value=1.5)
        missing = terms.TermResult(key="b", stage="s", provenance="solver_objective", status="not_used")
        result = summarise("s", [good, missing], ("a", "b"))
        self.assertAlmostEqual(result.total, 1.5)
        self.assertEqual(result.total_kind, "sum_of_available_terms_partial")
        self.assertEqual(result.n_terms_missing, 1)

    def test_all_missing_gives_null_total(self):
        missing = terms.TermResult(key="a", stage="s", provenance="solver_objective", status="null")
        result = summarise("s", [missing], ("a",))
        self.assertIsNone(result.total)

    def test_parts_sum_to_total_when_complete(self):
        a = terms.TermResult(key="a", stage="s", provenance="solver_objective", value=0.25)
        b = terms.TermResult(key="b", stage="s", provenance="solver_objective", value=0.75)
        result = summarise("s", [a, b], ("a", "b"))
        self.assertAlmostEqual(result.total, sum(t.value for t in result.terms))
        self.assertEqual(result.total_kind, "sum_of_available_terms")


class RecorderContract(unittest.TestCase):
    def test_tiers_are_separate_and_never_mixed(self):
        recorder = Recorder(run_label="unit", algorithm="alg", software_version="0.2.0")
        recorder.add(IterationRecord(stage="fit_c", iteration=0, tier="solver_objective", value=1.0))
        recorder.add(IterationRecord(stage="fit_c", iteration=0, tier="linearised_objective", value=0.1))
        self.assertEqual(recorder.series("fit_c", "solver_objective"), [(0, 1.0)])
        self.assertEqual(recorder.series("fit_c", "linearised_objective"), [(0, 0.1)])

    def test_unknown_tier_is_rejected(self):
        with self.assertRaises(ValueError):
            IterationRecord(stage="fit_c", iteration=0, tier="total", value=1.0)

    def test_null_value_stays_null_and_is_noted(self):
        record = IterationRecord(stage="s", iteration=1, tier="solver_objective",
                                 value=None, terms={"a": None})
        self.assertIsNone(record.value)
        self.assertTrue(any("null" in note for note in record.notes))

    def test_decisions_round_trip(self):
        recorder = Recorder(run_label="unit", algorithm="alg", software_version="0.2.0")
        for decision, value in (("initial", 2.0), ("candidate", 1.5), ("accepted", 1.5)):
            recorder.add(IterationRecord(stage="field", iteration=0, tier="linearised_objective",
                                         value=value, decision=decision))
        payload = recorder.payload()
        self.assertEqual([r["decision"] for r in payload["iterations"]], ["initial", "candidate", "accepted"])

    def test_json_and_csv_formats(self):
        recorder = Recorder(run_label="unit", algorithm="alg", software_version="0.2.0")
        recorder.add(IterationRecord(stage="fit_c", iteration=0, tier="solver_objective", value=1.25,
                                     phase="motion_initial", decision="initial"))
        recorder.gap("loop_base", "no objective at all", expected_tier="solver_objective")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            json_path = recorder.write_json(root / "records.json")
            csv_path = recorder.write_csv(root / "records.csv")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "loss_analysis.run.v1")
            self.assertEqual(payload["iterations"][0]["value"], 1.25)
            self.assertEqual(payload["gaps"][0]["stage"], "loop_base")
            text = csv_path.read_text(encoding="utf-8-sig")
            self.assertIn("stage,iteration,tier", text.splitlines()[0])
            self.assertIn("__gap__", text)

    def test_no_overwrite(self):
        recorder = Recorder(run_label="unit", algorithm="alg", software_version="0.2.0")
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "records.json"
            recorder.write_json(target)
            with self.assertRaises(FileExistsError):
                recorder.write_json(target)


class ObjectiveCatalogueContract(unittest.TestCase):
    def test_every_term_cites_a_source_line(self):
        for stage in objectives.ALL_STAGES:
            self.assertTrue(stage.terms, f"{stage.key} has no terms")
            for term in stage.terms:
                self.assertTrue(term.source.startswith("head_rgbd/"), f"{stage.key}/{term.key}")
                self.assertTrue(term.denominator, f"{stage.key}/{term.key} lacks a denominator")

    def test_no_unified_total_is_claimed(self):
        self.assertFalse(any(stage.has_shared_total for stage in objectives.ALL_STAGES))

    def test_field_stage_marks_feasibility_only(self):
        loop = objectives.stage_by_key("fit_d.loop_base")
        self.assertEqual(loop.terms[0].key, "geometry_safety_feasibility")
        self.assertIn("no differentiable objective", loop.objective_scope)


class EntryPointContract(unittest.TestCase):
    def _run(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "loss_analysis", *args],
            cwd=str(cwd or PACKAGE_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            env={**os.environ, "PYTHONPATH": str(PACKAGE_ROOT), "PYTHONIOENCODING": "utf-8"},
        )

    def test_help_without_arguments(self):
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("stages", completed.stdout)
        self.assertIn("no input supplied", completed.stdout)

    def test_version(self):
        completed = self._run("--version")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("lossana", completed.stdout)

    def test_stages_command_lists_terms(self):
        completed = self._run("stages")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("fit_c", completed.stdout)
        self.assertIn("fit_d.guarded_dense", completed.stdout)
        self.assertIn("loop_base", completed.stdout)

    def test_missing_run_path_fails_loudly(self):
        completed = self._run("inspect", "--run", str(Path(tempfile.gettempdir()) / "definitely-not-here"))
        self.assertEqual(completed.returncode, 2)
        self.assertIn("error", completed.stdout.lower())

    def test_analyze_requires_explicit_run(self):
        completed = self._run("analyze")
        self.assertNotEqual(completed.returncode, 0)

    def test_no_default_data_is_used(self):
        """Without --run the analyser prints help and analyses nothing."""
        completed = self._run()
        self.assertIn("no input supplied", completed.stdout)
        self.assertNotIn("output_dir", completed.stdout)

    def test_analyze_on_constructed_run(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / "run"
            (run / "logs").mkdir(parents=True)
            (run / "logs" / "03_identity_front_anchor.log").write_text(
                "{'step': 0, 'phase': 'motion_initial', 'objective': 0.9}\n"
                "{'step': 20, 'phase': 'identity_large', 'objective': 0.7}\n"
                "unrelated line\n",
                encoding="utf-8",
            )
            stage = run / "reconstruction" / "dense"
            stage.mkdir(parents=True)
            (stage / "report.json").write_text(json.dumps({
                "field_logs": {"regularized": {
                    "outer": 0, "stage": "guarded_regularized", "objective_before": 4.0,
                    "objective_after": 3.5, "step_accepted": True, "shared_step_scale": 1.0,
                    "free_vertices": 12, "solver_status": 1,
                    "fixed_ray_diagnostic": {
                        "before_mse_mm2": 1.2, "after_mse_mm2": 1.1,
                        "missing_before": 3, "missing_after": 3, "ray_count": 500,
                        "policy": "strict-coverage", "missing_penalty_mm": 8.0,
                    },
                    "safety": {"reason": None, "maximum_slope": 0.12},
                }},
            }), encoding="utf-8")
            out = root / "out"
            completed = self._run("analyze", "--run", str(run), "--output", str(out), "--no-plots")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            # 2 solver objectives + 2 linearised (before/after) + 2 fixed-ray energies
            # + 1 geometric-safety verdict row
            self.assertEqual(summary["iterations"], 7)
            self.assertGreater(summary["gaps"], 0)
            records = json.loads((out / "records.json").read_text(encoding="utf-8"))
            tiers = {r["tier"] for r in records["iterations"]}
            self.assertIn("solver_objective", tiers)
            self.assertIn("linearised_objective", tiers)
            self.assertIn("nonlinear_energy", tiers)
            gap_stages = {g["stage"] for g in records["gaps"]}
            self.assertIn("diagnostics", gap_stages)
            # fit_c stages never persist field_logs: they must appear as explicit gaps
            self.assertIn("identity_front_anchor", gap_stages)
            # other dense sub-stages are simply absent from this fixture
            self.assertIn("dense.front_anchor_low_frequency", gap_stages)
            self.assertTrue((out / "report.md").is_file())
            html = out / "report.html"
            self.assertTrue(html.is_file())
            page = html.read_text(encoding="utf-8")
            self.assertTrue(page.startswith("<!DOCTYPE html>"))
            self.assertIn('charset="utf-8"', page)
            self.assertIn("求解器目标", page)
            self.assertIn("缺口阶段", page)
            self.assertIn("3.5", page)
            # single-file report: no CDN, no external link, no side-car image refs
            self.assertNotIn("http://", page)
            self.assertNotIn("https://", page)
            self.assertNotIn('src="figures', page)


class HtmlReportContract(unittest.TestCase):
    """The HTML report must stay single-file, escaped, and null-preserving."""

    def _payload(self):
        recorder = Recorder(run_label="html <test>", algorithm="fit", software_version="0")
        recorder.add(IterationRecord(
            stage="fit_c", iteration=1, tier="solver_objective", value=0.5,
            phase="identity", decision="accepted",
            notes=["<script>alert(1)</script>"],
        ))
        recorder.add(IterationRecord(
            stage="fit_c", iteration=2, tier="solver_objective", value=None,
            decision="rejected", notes=["missing is <b>not</b> zero"],
        ))
        recorder.gap("loop_base", "no objective <i>by design</i>",
                     expected_tier="solver_objective")
        return recorder.payload()

    def test_html_escapes_dynamic_text_and_keeps_null(self):
        from loss_analysis.html_report import build_analysis_html

        page = build_analysis_html(
            self._payload(), stage_catalogue=[], input_summary={}, extra_notes=[]
        )
        self.assertIn("<!DOCTYPE html>", page)
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)
        self.assertIn("&lt;i&gt;by design&lt;/i&gt;", page)
        self.assertIn("html &lt;test&gt;", page)
        # a missing record renders as null, never as 0
        self.assertIn('<span class="null">null</span>', page)
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)

    def test_figures_are_embedded_not_linked(self):
        from loss_analysis.html_report import build_analysis_html

        with tempfile.TemporaryDirectory() as name:
            png = Path(name) / "fit_c.png"
            png.write_bytes(base64.b64decode(
                b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
                b"hQGAhKmMIQAAAABJRU5ErkJggg=="
            ))
            page = build_analysis_html(
                self._payload(), stage_catalogue=[], input_summary={}, figures=[png]
            )
        self.assertIn('src="data:image/png;base64,', page)
        self.assertNotIn('src="fit_c', page)
        self.assertNotIn('src="figures', page)

    def test_diagnostic_tier_has_its_own_section(self):
        from loss_analysis.html_report import build_analysis_html

        metrics = {"closed_front_2": {
            "metrics": {"status": "ok", "n_valid": 40, "mae_mm": 0.42, "rmse_mm": 0.55,
                        "p95_mm": 0.9, "median_mm": 0.4, "max_mm": 1.2},
            "coverage": {"n_required": 68, "coverage": 0.5882},
        }}
        page = build_analysis_html(
            self._payload(), stage_catalogue=[], input_summary={}, metrics=metrics
        )
        self.assertIn("独立几何诊断", page)
        self.assertIn("closed_front_2", page)
        self.assertIn("非算法 loss", page)
        self.assertIn("0.42", page)


class JointFusionIngestion(unittest.TestCase):
    """v0.4.0 makes one stage key mean two different objectives.

    These tests prove the reader attributes the objective to the variant that the
    RUN reports, that a soft lock is never mistaken for a hard lock, and that a
    stage which disagrees with the rest of the chain is reported instead of being
    silently folded into the chain-level flag.
    """

    def _identity_stage(self, root: Path, **report_fields) -> Path:
        stage = root / "reconstruction" / "low_frequency"
        stage.mkdir(parents=True)
        payload = {
            "shape_fusion": "joint",
            "joint_front_confidence": 0.15,
            "front_locked_vertices": 0,
            "front_downweighted_vertices": 120,
            "stages": [
                {"stage": "joint_low_frequency_all_views", "free_vertices": 900,
                 "history": [{"outer": 0, "objective_before": 3.0, "objective_after": 2.5,
                              "step_accepted": True, "shared_step_scale": 1.0,
                              "free_vertices": 900}]},
                {"stage": "joint_confidence_weighted_low_frequency", "free_vertices": 1200,
                 "history": [{"outer": 0, "objective_before": 2.5, "objective_after": 2.4,
                              "step_accepted": True, "shared_step_scale": 0.5,
                              "free_vertices": 1200}]},
            ],
        }
        payload.update(report_fields)
        (stage / "report.json").write_text(json.dumps(payload), encoding="utf-8")
        return stage

    def test_each_solve_gets_its_own_stage_and_keeps_soft_lock(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            stage = self._identity_stage(root)
            recorder = Recorder(run_label="unit", algorithm="alg", software_version="0.4.0")
            produced = adapter.ingest_identity_stage(
                recorder, stage, "low_frequency_shared_identity", chain_mode="joint")
            self.assertTrue(produced)
            stages = {r.stage for r in recorder.iterations}
            # two different objectives must never share one curve
            self.assertIn("low_frequency_shared_identity.joint_low_frequency_all_views", stages)
            self.assertIn("low_frequency_shared_identity.joint_confidence_weighted_low_frequency", stages)
            soft = [r for r in recorder.iterations if r.extras.get("soft_lock") is not None]
            self.assertTrue(all(r.extras["soft_lock"] is True for r in soft))
            self.assertTrue(all(r.extras["front_downweighted_vertices"] == 120 for r in soft))

    def test_chain_joint_but_stage_guarded_is_an_explicit_inconsistency(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            stage = self._identity_stage(root, shape_fusion="guarded",
                                         front_locked_vertices=2530,
                                         front_downweighted_vertices=0)
            recorder = Recorder(run_label="unit", algorithm="alg", software_version="0.4.0")
            adapter.ingest_identity_stage(
                recorder, stage, "low_frequency_shared_identity", chain_mode="joint")
            reasons = [gap.reason for gap in recorder.gaps
                       if gap.stage == "low_frequency_shared_identity"]
            self.assertTrue(any("mode mismatch" in reason for reason in reasons), reasons)
            self.assertTrue(any("2530" in reason and "0" in reason for reason in reasons), reasons)
            soft = [r for r in recorder.iterations if r.extras.get("soft_lock") is True]
            self.assertFalse(soft, "a guarded stage must not be labelled as confidence-weighted")

    def test_dict_shape_fusion_block_is_not_stringified_into_a_label(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "run"
            reconstruction = root / "reconstruction"
            (reconstruction / "guarded").mkdir(parents=True)
            (reconstruction / "guarded" / "report.json").write_text(
                json.dumps({"shape_fusion": "joint", "joint_shape_delta_from_anchor": 0.73}),
                encoding="utf-8")
            dense = reconstruction / "dense"
            dense.mkdir()
            (dense / "report.json").write_text(json.dumps({
                "shape_fusion": {"mode": "joint", "joint_front_confidence": 0.15,
                                 "shape_frames": ["closed_front_0", "closed_left_0"],
                                 "front_hard_locked_vertices": 0,
                                 "front_downweighted_vertices": 2522},
                "field_logs": {"dense": {"status": "not_executed", "history": []}},
            }), encoding="utf-8")
            self.assertEqual(adapter.chain_fusion(root), "joint")
            recorder = Recorder(run_label="unit", algorithm="alg", software_version="0.4.0")
            adapter.ingest_skills(recorder, root, chain_mode="joint")
            labels = set(recorder.stage_meta) | {gap.stage for gap in recorder.gaps}
            self.assertFalse([label for label in labels if "{" in label], labels)
            notes = " ".join(note for meta in recorder.stage_meta.values() for note in meta["notes"])
            self.assertIn("soft_lock in force", notes)
            self.assertIn("front_downweighted_vertices=2522", notes)

    def test_contradictory_artefacts_declare_no_chain_mode(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            reconstruction = root / "reconstruction"
            for stage, mode in (("guarded", "joint"), ("dense", "guarded")):
                (reconstruction / stage).mkdir(parents=True)
                (reconstruction / stage / "report.json").write_text(
                    json.dumps({"shape_fusion": mode}), encoding="utf-8")
            self.assertIsNone(adapter.chain_fusion(root))

    def test_version_skew_within_one_run_directory_is_flagged(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "delivery").mkdir(parents=True)
            (root / "delivery" / "manifest.json").write_text(
                json.dumps({"software": {"version": "0.3.0", "algorithm": "alg"}}), encoding="utf-8")
            (root / "run_report.json").write_text(json.dumps({"project_version": "0.3.0"}),
                                                  encoding="utf-8")
            dense = root / "reconstruction" / "dense"
            dense.mkdir(parents=True)
            (dense / "report.json").write_text(json.dumps({"project_version": "0.4.0"}),
                                               encoding="utf-8")
            skew = adapter.version_skew(root)
            self.assertFalse(skew["consistent"])
            self.assertEqual(skew["stated"]["reconstruction/dense/report.json"], "0.4.0")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _loss_payload() -> dict:
    return {
        "schema": "neck_direct.loss.v1",
        "tier_note": "solver objective vs diagnostic",
        "params": {"structure_gain": 1.10, "bridge_width_mm": 12.0,
                   "seam_taper_ratio": 2.5, "mu_t_bridge": 0.15,
                   "mu_t_body": 0.50, "anchor_near_A": 2.0},
        "states": {
            "closed": {
                "iterations": [
                    {"iter": 0, "energy": 10.0, "accepted": True,
                     "energy_parts": {"data": 6.0, "bend": 2.0, "tangent": 1.0,
                                      "anchor": 0.9, "small": 0.1},
                     "line_search_alpha": 1.0, "samples": 100, "max_d_mm": 2.0,
                     "step_max_mm": 1.2, "grad_max": 3.0, "grad_rel": 1.0,
                     "energy_rel": 0.0},
                    {"iter": 1, "energy": 9.0, "accepted": False,
                     "energy_parts": {"data": 5.4, "bend": 1.8, "tangent": 0.9,
                                      "anchor": 0.8, "small": 0.1},
                     "line_search_alpha": 0.5, "samples": 100, "max_d_mm": 2.1,
                     "step_max_mm": 0.2, "grad_max": 0.02, "grad_rel": 0.007,
                     "energy_rel": 0.0},
                ],
                "n_iter": 2, "converged": True, "stop_reason": "energy_stationary",
                "energy": 9.0, "n_free": 10, "n_fixed": 90, "data_samples": 100,
                "residual_after": {"vertex_rms_mm": 0.222, "quad_rms_mm": 0.279,
                                   "vertex_p95_mm": 0.5, "quad_p95_mm": 0.6,
                                   "vertex_max_mm": 1.0, "quad_max_mm": 1.1,
                                   "n_samples": 100, "n_samples_vertex": 40,
                                   "n_samples_face": 60},
            },
            "open": {
                "iterations": [
                    {"iter": 0, "energy": 11.0, "accepted": True,
                     "energy_parts": {"data": 6.6, "bend": 2.2, "tangent": 1.1,
                                      "anchor": 1.0, "small": 0.1},
                     "line_search_alpha": 1.0, "samples": 90, "max_d_mm": 2.4,
                     "step_max_mm": 1.4, "grad_max": 3.4, "grad_rel": 1.0,
                     "energy_rel": 0.0},
                ],
                "n_iter": 1, "converged": True, "stop_reason": "energy_stationary",
                "energy": 11.0, "n_free": 10, "n_fixed": 90, "data_samples": 90,
                "residual_after": {"vertex_rms_mm": 0.226, "quad_rms_mm": 0.280,
                                   "n_samples": 90},
            },
        },
        "remesh": {"passes": []},
    }


def _quick_payload(chosen: str) -> dict:
    return {
        "variant": "prior", "chosen": chosen, "variants": {},
        "quick": {
            "signed_crease": {
                "closed": {"full_positive_max_deg": 21.45, "B_positive_max_deg": 14.81},
                "open": {"full_positive_max_deg": 19.40, "B_positive_max_deg": 17.05},
            },
            "protection": {"fixed_vertices": 100, "fixed_max_mm": 0.00086, "pass": True},
            "topology": {"pass": True},
            "shared_topology": True,
            "structure": {"closed": {"larynx_local": {"relief_mm": 5.10}},
                          "open": {"larynx_local": {"relief_mm": 4.69}}},
            "upper_band_0.925_0.97": {"input": {"vertices": 78},
                                      "full": {"vertices": 78}, "no_increase": True},
            "vertices": {"closed": 20009, "open": 20009},
            "faces": {"closed": 39886, "open": 39886},
            "pass": {"critical": True, "connection": False, "density_upper_band": True},
        },
    }


def _dense_payload() -> dict:
    return {"field_logs": {"front_anchor_regularized": {"history": [{
        "outer": 1, "stage": "front_anchor_low_frequency",
        "objective_before": 1.6, "objective_after": 1.5, "step_accepted": True,
        "fixed_ray_diagnostic": {"before_mse_mm2": 1.1, "after_mse_mm2": 1.05,
                                 "missing_before": 9, "missing_after": 10,
                                 "ray_count": 1000, "policy": "strict-coverage"},
        "safety": {"reason": None},
    }]}}}


def _landmarks_payload(t_center: float = 0.7637) -> dict:
    return {
        "schema": "head_rgbd.neck_landmarks.v1",
        "states": {
            "closed": {"status": "observed", "source": "aruco_marker",
                       "frame_id": "closed_front_3",
                       "larynx_center": {"t_center_used": t_center, "theta_used_deg": 0.0,
                                         "t_gate": 0.9},
                       "marker": {"print": {"target_id": 0, "black_square_mm": 33.4076,
                                            "size_source": "measured_from_depth"}}},
            "open": {"status": "observed", "source": "aruco_marker",
                     "frame_id": "open_front_2",
                     "larynx_center": {"t_center_used": 0.8012, "theta_used_deg": 0.0,
                                       "t_gate": 0.95},
                     "marker": {"print": {"target_id": 0, "black_square_mm": 32.9608,
                                          "size_source": "measured_from_depth"}}},
        },
    }


def _delivery_files(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("head_closed.obj", "head_open.obj"):
        (folder / name).write_text("# fake obj\n", encoding="utf-8")
    _write_json(folder / "anatomical_landmarks.json", {
        "schema": "head_rgbd.delivery.anatomical_landmarks.v2",
        "status": "usable",
        "main_delivery_mesh_file": "head_closed.obj",
    })
    _write_json(folder / "manifest.json", {
        "schema": "head_rgbd.delivery.manifest.v1",
        "mesh": {"states": {"closed": {"path": "head_closed.obj"},
                            "open": {"path": "head_open.obj"}}},
    })


def _baseline_tree(root: Path) -> dict[str, Path]:
    result = root / "baseline" / "result"
    _delivery_files(result / "final")
    _delivery_files(result / "measured")
    neck = root / "baseline" / "neck_direct_v2"
    _write_json(neck / "build_direct.json", {
        "params": {"structure_gain": 1.10, "bridge_width_mm": 12.0,
                   "seam_taper_ratio": 2.5, "mu_t_bridge": 0.15,
                   "mu_t_body": 0.50, "anchor_near_A": 2.0},
        "structure": {"reference": {"primary": {"relief_mm": 4.43}}},
        "journal": {"split_new_total": 0, "split_new_cap": 20,
                    "vertices_original": 20018, "vertices_final": 20009,
                    "merged_slots": 9},
    })
    _write_json(neck / "loss_direct.json", _loss_payload())
    _write_json(result / "quick_check.json", _quick_payload(str(neck)))
    _write_json(root / "baseline" / "neck_landmarks.json", _landmarks_payload())
    _write_json(result / "run_report.json", {
        "configuration": {"motion_lowerface_weight": 0.5, "bending_length_mm": 8.0,
                          "low_bending_length_mm": 8.0}
    })
    dense = root / "base" / "dense"
    _write_json(dense / "report.json", _dense_payload())
    return {"result": result, "neck": neck, "dense": dense,
            "landmarks": root / "baseline" / "neck_landmarks.json",
            "root": root}


def _optimizer_args(paths: dict[str, Path], out: Path, **overrides):
    values = dict(
        baseline_result=paths["result"], baseline_dense=paths["dense"],
        baseline_neck=paths["neck"], baseline_trace=paths["neck"],
        capture=None, output=out, candidate=None, limit=12, execute=False,
        max_candidates=1, freeze_landmarks=False, neck_only_dense=False,
        landmarks=paths["landmarks"], timeout_s=60, no_figures=True,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class NeckAdapterContract(unittest.TestCase):
    def test_solver_trace_and_residual_are_separate_tiers(self):
        with tempfile.TemporaryDirectory() as name:
            neck = Path(name) / "neck"
            _write_json(neck / "loss_direct.json", _loss_payload())
            recorder = Recorder(run_label="neck", algorithm="neck_direct",
                                software_version="0.2.0")
            summary = neck_adapter.ingest_neck_direct(recorder, neck)
            self.assertTrue(summary["present"])
            solver = [r for r in recorder.iterations if r.tier == "solver_objective"]
            diagnostic = [r for r in recorder.iterations if r.tier == "diagnostic"]
            self.assertEqual(len(solver), 3)
            self.assertEqual(len(diagnostic), 2)
            first = solver[0]
            # the parts split the solver objective inside one iteration
            self.assertAlmostEqual(sum(first.terms.values()), first.value, places=9)
            self.assertEqual(first.decision, "accepted")
            self.assertEqual(solver[1].decision, "rejected")
            # in-face residual is a diagnostic, never part of the energy
            self.assertFalse(any(r.tier == "solver_objective" and
                                 "vertex_rms_mm" in (r.terms or {})
                                 for r in recorder.iterations))
            for stage, meta in recorder.stage_meta.items():
                if stage.startswith("neck_direct.fit_unified"):
                    self.assertIn("not optimisation steps", " ".join(meta["notes"]))

    def test_missing_trace_is_a_gap_not_zero(self):
        with tempfile.TemporaryDirectory() as name:
            recorder = Recorder(run_label="neck", algorithm="neck_direct",
                                software_version="0.2.0")
            summary = neck_adapter.ingest_neck_direct(recorder, Path(name))
            self.assertFalse(summary["present"])
            self.assertEqual(recorder.iterations, [])
            self.assertTrue(recorder.gaps)
            self.assertEqual(recorder.gaps[0].stage, "neck_direct")


class OptimizerContract(unittest.TestCase):
    def test_plan_is_bounded_and_does_not_execute(self):
        with tempfile.TemporaryDirectory() as name:
            paths = _baseline_tree(Path(name))
            out = Path(name) / "out"
            code = optimizer.run_optimizer(_optimizer_args(paths, out))
            self.assertEqual(code, 0)
            plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
            self.assertLessEqual(len(plan["candidates"]), 12)
            keys = {item["key"] for item in plan["candidates"]}
            self.assertIn("neck_anchor_near_A_2.5", keys)
            self.assertIn("base_bending_length_mm_9", keys)
            self.assertFalse((out / "candidates").exists())
            self.assertTrue((out / "report.html").is_file())
            self.assertTrue((out / "candidates.csv").is_file())
            page = (out / "report.html").read_text(encoding="utf-8")
            self.assertIn("候选计划", page)
            self.assertNotIn("http://", page)
            # baseline residual comes from the supplied trace
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            residual = summary["baseline_assessment"]["neck"]["target_residual"]
            self.assertAlmostEqual(residual["mean_vertex_rms_mm"], 0.224, places=6)
            with self.assertRaises(FileExistsError):
                optimizer.run_optimizer(_optimizer_args(paths, out))

    def test_hard_constraints_reject_marker_drift_and_protection(self):
        with tempfile.TemporaryDirectory() as name:
            paths = _baseline_tree(Path(name))
            baseline = neck_adapter.assess(
                dense_dir=paths["dense"], result_dir=paths["result"],
                neck_dir=paths["neck"], landmarks_path=paths["landmarks"])
            candidate = neck_adapter.assess(
                dense_dir=paths["dense"], result_dir=paths["result"],
                neck_dir=paths["neck"], landmarks_path=paths["landmarks"])
            probe = optimizer._delivery_probe(paths["result"])
            self.assertTrue(all(r["status"] == "pass" for r in
                                optimizer.hard_constraints(baseline, candidate, probe)))
            drifted = json.loads(json.dumps(candidate))
            drifted["neck"]["markers"]["states"]["closed"]["t_center_used"] += 0.01
            rows = optimizer.hard_constraints(baseline, drifted, probe)
            self.assertEqual({r["name"]: r["status"] for r in rows}["marker semantics"], "fail")
            weak = json.loads(json.dumps(candidate))
            weak["neck"]["gates"]["protection"]["pass"] = False
            rows = optimizer.hard_constraints(baseline, weak, probe)
            self.assertEqual({r["name"]: r["status"] for r in rows}["protection"], "fail")
            over = json.loads(json.dumps(candidate))
            over["neck"]["remesh"]["split_new_total"] = 21
            rows = optimizer.hard_constraints(baseline, over, probe)
            self.assertEqual({r["name"]: r["status"] for r in rows}["split budget"], "fail")

    def test_ranking_is_lexicographic_not_weighted_sum(self):
        better_base = {"base": {"fixed_ray_after_mse_mm2": 1.00, "missing_after": 9},
                       "neck": {"gates": {"connection_max_deg": 25.0},
                                "target_residual": {"mean_vertex_rms_mm": 0.5,
                                                    "mean_quad_rms_mm": 0.6}}}
        better_neck = {"base": {"fixed_ray_after_mse_mm2": 1.10, "missing_after": 9},
                       "neck": {"gates": {"connection_max_deg": 19.0},
                                "target_residual": {"mean_vertex_rms_mm": 0.2,
                                                    "mean_quad_rms_mm": 0.3}}}
        ranked = sorted([better_neck, better_base], key=optimizer.ranking_key)
        self.assertIs(ranked[0], better_base)
        tie = json.loads(json.dumps(better_base))
        tie["base"]["fixed_ray_after_mse_mm2"] = 1.10
        ranked = sorted([better_neck, tie], key=optimizer.ranking_key)
        self.assertIs(ranked[0], better_neck)

    def test_cli_requires_explicit_baseline_paths(self):
        completed = subprocess.run(
            [sys.executable, "-m", "loss_analysis", "optimize"],
            cwd=str(PACKAGE_ROOT), capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONPATH": str(PACKAGE_ROOT), "PYTHONIOENCODING": "utf-8"},
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--baseline-result", completed.stderr)


class AnalyzeWithNeckContract(unittest.TestCase):
    """`analyze --neck-result` adds the direct neck trace without inventing gates."""

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "loss_analysis", *args],
            cwd=str(PACKAGE_ROOT), capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONPATH": str(PACKAGE_ROOT), "PYTHONIOENCODING": "utf-8"},
        )

    def test_neck_trace_and_gate_notes_are_read(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / "run"
            (run / "logs").mkdir(parents=True)
            (run / "logs" / "02_identity_front_anchor.log").write_text(
                "{'step': 0, 'phase': 'motion_initial', 'objective': 0.9}\n",
                encoding="utf-8",
            )
            dense = run / "reconstruction" / "dense"
            _write_json(dense / "report.json", _dense_payload())
            paths = _baseline_tree(root)
            out = root / "out"
            completed = self._run(
                "analyze", "--run", str(run), "--output", str(out), "--no-plots",
                "--neck-result", str(paths["result"]),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = json.loads((out / "records.json").read_text(encoding="utf-8"))
            stages = {row["stage"] for row in records["iterations"]}
            self.assertIn("neck_direct.fit_unified.closed", stages)
            self.assertIn("neck_direct.fit_unified.open", stages)
            self.assertIn("neck_direct.target_residual.closed", stages)
            gap_stages = {gap["stage"] for gap in records["gaps"]}
            self.assertNotIn("neck_direct", gap_stages)
            notes = records["stages"]["neck_direct.gates"]["notes"]
            self.assertTrue(any("not a numeric loss" in note for note in notes))
            self.assertTrue(any("connection max positive crease" in note for note in notes))
            page = (out / "report.html").read_text(encoding="utf-8")
            self.assertIn("neck_direct.fit_unified.closed", page)
            self.assertNotIn("http://", page)


if __name__ == "__main__":
    unittest.main()
