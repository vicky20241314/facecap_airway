"""Mathematical and interface contracts for the local loss analyser.

Constructed data only. These tests prove the arithmetic and the interfaces are
correct; they say nothing about reconstruction quality. No real capture, no GPU,
no model weights and no reconstruction are touched.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from loss_analysis import cli, objectives, terms
from loss_analysis import adapter
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


if __name__ == "__main__":
    unittest.main()
