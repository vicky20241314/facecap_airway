"""Contracts for the per-stage termination-gap report (constructed data only)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from loss_analysis import cli, gap_report, neck_adapter, optimizer


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def _make_run(root: Path) -> Path:
    run = root / "run"
    logs = run / "logs"
    logs.mkdir(parents=True)
    _write(run / "run_report.json", {
        "configuration": {"steps": 240, "front_steps": 240, "motion_steps": 80},
    })
    (logs / "03_shared_identity_joint.log").write_text(
        "{'step': 0, 'phase': 'joint_motion_initial', 'objective': 0.200}\n"
        "{'step': 239, 'phase': 'joint_motion_final', 'objective': 0.080}\n",
        encoding="utf-8",
    )
    (logs / "04_fixed_identity_motion.log").write_text(
        "{'step': 0, 'phase': 'rigid', 'objective': 0.180}\n"
        "{'step': 79, 'phase': 'rigid', 'objective': 0.160}\n",
        encoding="utf-8",
    )
    return run


def _make_dense(root: Path) -> Path:
    dense = root / "dense"
    report = {
        "field_logs": {
            "dense": {
                "history": [{
                    "outer": 2,
                    "stage": "dense_surface",
                    "objective_before": 2.0,
                    "objective_after": 1.8,
                    "step_accepted": True,
                    "fixed_ray_diagnostic": {
                        "before_mse_mm2": 1.000,
                        "after_mse_mm2": 1.010,
                        "missing_before": 3,
                        "missing_after": 4,
                        "missing_cap": 10,
                        "missing_stage_baseline": 0,
                        "ray_count": 1000,
                        "policy": "strict-coverage",
                    },
                }],
            },
        },
    }
    _write(dense / "report.json", report)
    return dense


def _make_neck(root: Path, *, full_closed: float = 21.45,
               b_closed: float = 14.81) -> Path:
    neck = root / "neck"
    _write(neck / "build_direct.json", {
        "journal": {
            "split_new_total": 0,
            "split_new_cap": 20,
            "vertices_original": 20018,
            "vertices_final": 20009,
            "merged_slots": 9,
        },
        "structure": {
            "reference": {"primary": {"relief_mm": 4.43}},
        },
    })
    _write(neck / "loss_direct.json", {
        "schema": "neck_direct.loss.v1",
        "params": {"fit_max_iter": 60},
        "states": {
            "closed": {
                "iterations": [
                    {"iter": 0, "energy": 10.0, "energy_parts": {"data": 7.0, "bend": 2.0, "tangent": 1.0},
                     "step_max_mm": 0.05, "energy_rel": 1e-3, "grad_rel": 1e-2, "accepted": True},
                    {"iter": 59, "energy": 8.0, "energy_parts": {"data": 6.0, "bend": 1.0, "tangent": 1.0},
                     "step_max_mm": 0.01, "energy_rel": 1e-4, "grad_rel": 1e-3, "accepted": True},
                ],
                "n_iter": 60,
                "converged": False,
                "stop_reason": "max_iter",
                "energy": 8.0,
                "residual_after": {"vertex_rms_mm": 0.20, "quad_rms_mm": 0.25},
            },
            "open": {
                "iterations": [
                    {"iter": 0, "energy": 9.0, "energy_parts": {"data": 6.0, "bend": 2.0, "tangent": 1.0},
                     "step_max_mm": 0.04, "energy_rel": 1e-3, "grad_rel": 1e-2, "accepted": True},
                    {"iter": 59, "energy": 7.0, "energy_parts": {"data": 5.0, "bend": 1.0, "tangent": 1.0},
                     "step_max_mm": 0.01, "energy_rel": 1e-4, "grad_rel": 1e-3, "accepted": True},
                ],
                "n_iter": 60,
                "converged": False,
                "stop_reason": "max_iter",
                "energy": 7.0,
                "residual_after": {"vertex_rms_mm": 0.22, "quad_rms_mm": 0.27},
            },
        },
    })
    return neck


def _make_result(root: Path, neck: Path, *, full_closed: float = 21.45,
                 b_closed: float = 14.81, full_open: float = 19.40,
                 b_open: float = 17.05) -> Path:
    result = root / "result"
    _write(result / "quick_check.json", {
        "chosen": str(neck),
        "quick": {
            "signed_crease": {
                "closed": {"full_positive_max_deg": full_closed,
                           "B_positive_max_deg": b_closed, "positive_gt20": 2},
                "open": {"full_positive_max_deg": full_open,
                         "B_positive_max_deg": b_open, "positive_gt20": 0},
            },
            "protection": {"fixed_vertices": 19145, "fixed_max_mm": 0.0008,
                           "vertices_above_0_01mm": 0, "pass": True},
            "topology": {"pass": True},
            "shared_topology": True,
            "structure": {
                "closed": {"larynx_local": {"relief_mm": 5.10}},
                "open": {"larynx_local": {"relief_mm": 4.69}},
            },
            "upper_band_0.925_0.97": {
                "input": {"vertices": 78, "faces": 71},
                "full": {"vertices": 78, "faces": 69},
                "no_increase": True,
            },
        },
    })
    for name in ("final", "measured"):
        folder = result / name
        folder.mkdir(parents=True)
        (folder / "head_closed.obj").write_text("", encoding="utf-8")
        (folder / "head_open.obj").write_text("", encoding="utf-8")
        _write(folder / "anatomical_landmarks.json", {
            "status": "usable",
            "states": {
                "closed": {"status": "observed",
                           "main_delivery_mesh_file": "head_closed.obj"},
                "open": {"status": "observed",
                         "main_delivery_mesh_file": "head_open.obj"},
            },
        })
        _write(folder / "manifest.json", {
            "mesh": {"states": {"closed": {}, "open": {}}},
        })
    return result


def _make_landmarks(root: Path) -> Path:
    path = root / "neck_landmarks.json"
    _write(path, {
        "states": {
            "closed": {
                "status": "observed", "source": "aruco_marker",
                "larynx_center": {"t_center_used": 0.7637, "theta_used_deg": 0.0},
                "marker": {"print": {"target_id": 0, "black_square_mm": 33.41}},
            },
            "open": {
                "status": "observed", "source": "aruco_marker",
                "larynx_center": {"t_center_used": 0.8012, "theta_used_deg": 0.0},
                "marker": {"print": {"target_id": 0, "black_square_mm": 32.96}},
            },
        },
    })
    return path


class GapReportContract(unittest.TestCase):
    def test_dominant_connection_and_targeted_seam(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _make_run(root)
            dense = _make_dense(root)
            neck = _make_neck(root)
            result = _make_result(root, neck)
            landmarks = _make_landmarks(root)

            report = gap_report.compute_gaps(
                run_dir=run, dense_dir=dense, result_dir=result,
                neck_dir=neck, landmarks=landmarks,
            )

            dominant = report["dominant"]
            self.assertEqual(dominant["key"], "neck.connection")
            self.assertEqual(dominant["status"], "fail")
            self.assertAlmostEqual(dominant["current"], 21.45, places=6)
            self.assertAlmostEqual(dominant["gap"], 20.0 - 21.45, places=6)
            self.assertEqual(
                [item["key"] for item in report["targeted"]],
                ["neck.structure_gain", "neck.anchor_near_A"],
            )
            regions = report["connection_regions"]
            self.assertEqual(regions["closed"]["worst"], "outside_B")

    def test_bridge_region_targets_bridge_prior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            neck = _make_neck(root)
            result = _make_result(root, neck, full_closed=21.0, b_closed=21.0)
            report = gap_report.compute_gaps(
                run_dir=_make_run(root), dense_dir=_make_dense(root),
                result_dir=result, neck_dir=neck,
            )
            self.assertEqual(report["connection_regions"]["closed"]["worst"], "B")
            self.assertEqual(
                [item["key"] for item in report["targeted"]],
                ["neck.structure_gain", "neck.mu_t_bridge"],
            )

    def test_below_reference_relief_uses_region_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            neck = _make_neck(root)
            result = _make_result(root, neck)
            quick_path = result / "quick_check.json"
            quick = json.loads(quick_path.read_text(encoding="utf-8"))
            quick["quick"]["structure"]["closed"]["larynx_local"]["relief_mm"] = 4.0
            quick["quick"]["structure"]["open"]["larynx_local"]["relief_mm"] = 3.9
            _write(quick_path, quick)
            report = gap_report.compute_gaps(
                run_dir=_make_run(root), dense_dir=_make_dense(root),
                result_dir=result, neck_dir=neck,
            )
            self.assertEqual(
                [item["key"] for item in report["targeted"]],
                ["neck.anchor_near_A", "neck.seam_taper_ratio"],
            )

    def test_dense_acceptance_and_budget_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _make_run(root)
            dense = _make_dense(root)
            neck = _make_neck(root)
            result = _make_result(root, neck)
            report = gap_report.compute_gaps(
                run_dir=run, dense_dir=dense, result_dir=result, neck_dir=neck,
            )
            by_key = {entry["key"]: entry for entry in report["entries"]}
            mse = by_key["dense.fixed_ray_mse"]
            self.assertEqual(mse["status"], "pass")
            self.assertAlmostEqual(mse["target"], 1.0 * 1.02 + 0.002, places=9)
            self.assertAlmostEqual(mse["gap"], 1.022 - 1.010, places=9)
            self.assertEqual(by_key["dense.missing_rays"]["status"], "pass")
            self.assertEqual(by_key["dense.missing_rays"]["gap"], 6)
            self.assertEqual(by_key["shared_identity_joint.step_budget"]["status"],
                             "budget_limited")
            self.assertEqual(by_key["shared_identity_joint.step_budget"]["gap"], 0)
            self.assertEqual(by_key["fixed_identity_motion.step_budget"]["gap"], 0)
            self.assertEqual(by_key["neck_direct.fit_unified.closed.step_max_mm"]["status"],
                             "budget_limited")
            self.assertAlmostEqual(
                by_key["neck_direct.fit_unified.closed.step_max_mm"]["gap"],
                0.01 - 1e-3, places=9,
            )
            self.assertEqual(by_key["neck.protection"]["status"], "pass")
            self.assertEqual(by_key["delivery.self_contained"]["status"], "pass")
            self.assertIsInstance(by_key["neck.topology"]["current"], bool)
            self.assertFalse(by_key["neck.topology"]["current"] == 0)

    def test_energy_stationary_is_not_called_converged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            neck = _make_neck(root)
            loss = json.loads((neck / "loss_direct.json").read_text(encoding="utf-8"))
            block = loss["states"]["closed"]
            block["n_iter"] = 3
            block["converged"] = True
            block["stop_reason"] = "energy_stationary"
            block["iterations"] = [{
                "iter": 2, "energy": 8.0, "energy_parts": {"data": 6.0},
                "step_max_mm": 0.0, "energy_rel": 0.0, "grad_rel": 0.044, "accepted": False,
            }]
            _write(neck / "loss_direct.json", loss)
            report = gap_report.compute_gaps(
                run_dir=_make_run(root), dense_dir=_make_dense(root),
                result_dir=_make_result(root, neck), neck_dir=neck,
            )
            by_key = {entry["key"]: entry for entry in report["entries"]}
            self.assertEqual(
                by_key["neck_direct.fit_unified.closed.iterations"]["status"], "stationary")
            self.assertEqual(
                by_key["neck_direct.fit_unified.closed.grad_rel"]["status"], "stationary")

    def test_missing_trace_is_a_gap_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            neck = _make_neck(root)
            (neck / "loss_direct.json").unlink()
            report = gap_report.compute_gaps(
                run_dir=_make_run(root), dense_dir=_make_dense(root),
                result_dir=_make_result(root, neck), neck_dir=neck,
            )
            self.assertTrue(any(gap["stage"].startswith("neck_direct.fit_unified")
                                for gap in report["gaps"]))
            self.assertFalse(any(entry["key"].startswith("neck_direct.fit_unified")
                                 for entry in report["entries"]))
            self.assertIsNone(neck_adapter.read_loss_direct(neck))

    def test_outputs_are_single_file_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            neck = _make_neck(root)
            report = gap_report.compute_gaps(
                run_dir=_make_run(root), dense_dir=_make_dense(root),
                result_dir=_make_result(root, neck), neck_dir=neck,
            )
            paths = gap_report.write_outputs(report, root / "out")
            html = Path(paths["report"]).read_text(encoding="utf-8")
            self.assertNotIn("http://", html)
            self.assertNotIn("https://", html)
            self.assertIn("逐阶段终止 gap 报告", html)
            self.assertTrue(Path(paths["gaps_json"]).is_file())
            self.assertTrue(Path(paths["csv"]).is_file())
            self.assertTrue((root / "out" / "summary.json").is_file())
            with self.assertRaises(FileExistsError):
                gap_report.write_outputs(report, root / "out")

    def test_targeted_candidates_cap_and_move(self):
        values = {key: param.baseline for key, param in optimizer.PARAMS.items()}
        gaps = {"targeted": [
            {"key": "neck.anchor_near_A", "why": "a"},
            {"key": "neck.seam_taper_ratio", "why": "b"},
            {"key": "neck.structure_gain", "why": "c"},
        ]}
        candidates = optimizer.targeted_candidates(values, gaps)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].overrides, {"neck.anchor_near_A": 2.5})
        self.assertEqual(candidates[1].overrides, {"neck.seam_taper_ratio": 3.0})

    def test_gap_cli_requires_run(self):
        parser = cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["gap"])

    def test_gap_cli_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            neck = _make_neck(root)
            out = root / "gap_out"
            code = cli.main([
                "gap",
                "--run", str(_make_run(root)),
                "--dense", str(_make_dense(root)),
                "--neck-result", str(_make_result(root, neck)),
                "--landmarks", str(_make_landmarks(root)),
                "--output", str(out),
            ])
            self.assertEqual(code, 0)
            self.assertTrue((out / "gaps.json").is_file())
            self.assertTrue((out / "report.html").is_file())
            payload = json.loads((out / "gaps.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["dominant"]["key"], "neck.connection")
            self.assertEqual(
                [item["key"] for item in payload["targeted"]],
                ["neck.structure_gain", "neck.anchor_near_A"],
            )

    def test_resultless_gap_report_does_not_invent_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = gap_report.compute_gaps(
                run_dir=_make_run(root), dense_dir=_make_dense(root),
            )
            self.assertFalse(any(entry["key"].startswith("neck.")
                                 for entry in report["acceptance"]))
            self.assertTrue(any(gap["stage"] == "neck.acceptance" for gap in report["gaps"]))
            self.assertEqual(report["dominant"]["key"], "dense.fixed_ray_mse")
            self.assertEqual(
                [item["key"] for item in report["targeted"]],
                ["base.bending_length_mm", "base.low_bending_length_mm"],
            )


if __name__ == "__main__":
    unittest.main()
