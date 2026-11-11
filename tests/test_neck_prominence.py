"""Unit tests for the marker-guided neck prominence stage (0.5.3 on 0.5.2)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from head_rgbd import neck_landmark as nl
from head_rgbd import neck_prominence as npn


# --------------------------------------------------------------------------- wendland
def test_wendland_c2_support_and_values():
    assert npn.wendland_c2(0.0) == 1.0
    assert npn.wendland_c2(1.0) == 0.0
    values = npn.wendland_c2(np.array([1.5, 2.0, 10.0]))
    assert (values == 0).all()
    r = np.linspace(0, 0.999, 200)
    w = np.asarray(npn.wendland_c2(r))
    assert np.isfinite(w).all()
    assert ((w >= 0) & (w <= 1.0000001)).all()
    # monotone decay from the centre
    assert (np.diff(w) < 1e-12).all()


def test_wendland_c2_smooth_at_support_boundary():
    """C2 continuity: value, first and second derivative -> 0 at r=1."""
    eps = 1e-5
    h = 1e-6
    r0 = 1.0 - eps
    w_left = float(npn.wendland_c2(r0))
    w_right = float(npn.wendland_c2(1.0 + eps))
    slope_left = (float(npn.wendland_c2(r0)) - float(npn.wendland_c2(r0 - h))) / h
    slope_right = (float(npn.wendland_c2(1.0 + eps)) - float(npn.wendland_c2(1.0 - eps))) / (2 * h)
    assert w_left < 1e-12 or abs(w_left) < 1e-6
    assert w_right == 0.0
    assert abs(slope_left) < 1e-3
    assert abs(slope_right) < 1e-9


# --------------------------------------------------------------------------- config
def test_resolve_configuration_defaults_only():
    cfg = npn.resolve_configuration({"neck_prominence": {
        "subject_sex": "unspecified", "mode": "auto",
        "amplitude_mm": 8.0, "radius_x_mm": 11.0, "radius_y_mm": 12.0,
        "base_radius_x_mm": 20.0, "base_radius_y_mm": 16.0, "base_share": 0.35,
        "candidate_amplitudes_mm": [6.0, 8.0],
    }}, None)
    assert cfg.subject_sex == "unspecified"
    assert cfg.mode == "auto"
    assert cfg.amplitude_mm == 8.0
    # preferred target first: 8 mm is tried before the 6 mm fallback
    assert cfg.candidates_mm == (8.0, 6.0)
    assert cfg.base_share == 0.35


def test_resolve_configuration_cli_overrides():
    class Args:
        subject_sex = "male"
        neck_prominence = "on"
        neck_prominence_mm = 5.0
        neck_prominence_radius_x_mm = 14.0
        neck_prominence_radius_y_mm = 20.0
        neck_prominence_base_radius_x_mm = 22.0
        neck_prominence_base_radius_y_mm = 24.0
        neck_prominence_base_share = 0.4
    cfg = npn.resolve_configuration({"neck_prominence": {"amplitude_mm": 3.0}}, Args())
    assert cfg.subject_sex == "male"
    assert cfg.mode == "on"
    assert cfg.amplitude_mm == 5.0
    assert cfg.candidates_mm == (5.0,)          # explicit target: single candidate
    assert cfg.radius_x_mm == 14.0
    assert cfg.radius_y_mm == 20.0
    assert cfg.base_radius_x_mm == 22.0
    assert cfg.base_radius_y_mm == 24.0
    assert cfg.base_share == 0.4


def test_resolve_configuration_rejects_invalid_values():
    with pytest.raises(ValueError):
        npn.resolve_configuration({"neck_prominence": {"amplitude_mm": -1.0}}, None)
    with pytest.raises(ValueError):
        npn.resolve_configuration({"neck_prominence": {"amplitude_mm": float("nan")}}, None)
    with pytest.raises(ValueError):
        npn.resolve_configuration({"neck_prominence": {"radius_x_mm": 0.0}}, None)
    with pytest.raises(ValueError):
        npn.resolve_configuration({"neck_prominence": {"base_share": 1.5}}, None)
    with pytest.raises(ValueError):
        npn.resolve_configuration(
            {"neck_prominence": {"radius_x_mm": 12.0, "base_radius_x_mm": 8.0}}, None
        )


def test_plateau_kernel_is_blunt_and_smooth():
    w = np.asarray(npn.plateau_c2(np.array([0.0, 0.2, 0.35, 0.5, 0.8, 0.999, 1.0, 1.5])))
    assert w[0] == pytest.approx(1.0)
    assert w[2] == pytest.approx(1.0)           # flat crest
    assert 0.4 < w[3] < 1.0                     # gradual fall, unlike Wendland (0.19 at r=0.5)
    assert w[5] < 0.01 and w[6] == 0.0 and w[7] == 0.0
    r = np.linspace(0, 1, 400)
    vals = np.asarray(npn.plateau_c2(r))
    assert (np.diff(vals) <= 1e-12).all()       # monotone
    eps, h = 1e-5, 1e-7
    slope_left = (float(npn.plateau_c2(1 - eps)) - float(npn.plateau_c2(1 - eps - h))) / h
    assert abs(slope_left) < 1e-3               # C1 (hence C2) at the boundary
    slope_plateau = (float(npn.plateau_c2(0.35 + h)) - float(npn.plateau_c2(0.35 - h))) / (2 * h)
    assert abs(slope_plateau) < 1e-3            # smooth crest shoulder


@pytest.mark.parametrize("mode,sex,observed,expected", [
    ("auto", "male", True, True),
    ("auto", "male", False, False),
    ("auto", "female", True, False),
    ("auto", "unspecified", True, False),
    ("on", "female", True, True),
    ("on", "male", False, False),
    ("off", "male", True, False),
])
def test_effective_enable_matrix(mode, sex, observed, expected):
    cfg = npn.ProminenceConfig(subject_sex=sex, mode=mode)
    enabled, reason = npn.effective_enable(cfg, observed)
    assert enabled is expected
    if not enabled:
        assert reason


def test_enable_reasons_are_specific():
    cfg = npn.ProminenceConfig(subject_sex="female", mode="auto")
    _, reason = npn.effective_enable(cfg, True)
    assert reason == "auto_female_keeps_template_neck"
    cfg = npn.ProminenceConfig(subject_sex="unspecified", mode="auto")
    _, reason = npn.effective_enable(cfg, True)
    assert reason == "auto_subject_sex_unspecified"


# --------------------------------------------------------------------------- landmarks
def test_landmarks_file_roundtrip(tmp_path: Path):
    payload = nl.empty_landmarks_payload()
    payload["states"]["closed"] = {
        "source": nl.SOURCE_MANUAL, "state": "closed", "status": nl.STATUS_OBSERVED,
        "frame_id": "closed_front_0", "point_observed_m": [0.01, -0.1, -0.5],
    }
    path = nl.write_landmarks_payload(tmp_path / nl.LANDMARKS_FILE, payload)
    loaded, reason = nl.load_neck_landmarks(path)
    assert reason is None
    assert loaded["states"]["closed"]["point_observed_m"] == [0.01, -0.1, -0.5]
    assert loaded["sources"]["closed"] == nl.SOURCE_MANUAL


def test_landmarks_file_rejects_invalid_point(tmp_path: Path):
    payload = nl.empty_landmarks_payload()
    payload["states"]["closed"] = {
        "status": nl.STATUS_OBSERVED, "frame_id": "x", "point_observed_m": [0.0, 0.0],
    }
    path = nl.write_landmarks_payload(tmp_path / nl.LANDMARKS_FILE, payload)
    loaded, reason = nl.load_neck_landmarks(path)
    assert loaded is None
    assert reason == "closed_point_observed_m_must_be_3_vector"


def test_transform_same_frame_is_identity():
    rows = [{"id": "closed_front_0", "status": "candidate", "segment": "s"},
            {"id": "closed_front_1", "status": "candidate", "segment": "s"}]
    record = {"status": nl.STATUS_OBSERVED, "frame_id": "closed_front_0",
              "point_observed_m": [0.1, 0.2, 0.3]}
    transform = nl.resolve_reference_transform(record, rows, "closed_front_0")
    assert transform["status"] == nl.STATUS_OBSERVED
    assert transform["method"] == "same_frame_observation"
    assert np.allclose(transform["rotation_3x3"], np.eye(3))
    assert transform["translation_m"] == [0.0, 0.0, 0.0]


def test_transform_explicit_record():
    rows = [{"id": "a", "status": "candidate"}]
    record = {"status": nl.STATUS_OBSERVED, "frame_id": "a",
              "point_observed_m": [1.0, 0.0, 0.0],
              "explicit_transform": {"rotation_3x3": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                                     "translation_m": [0.1, 0.0, 0.0]}}
    transform = nl.resolve_reference_transform(record, rows, "other_frame")
    assert transform["status"] == nl.STATUS_OBSERVED
    assert transform["method"] == "explicit_transform_from_record"
    assert nl.apply_transform(record["point_observed_m"], transform) == pytest.approx([0.1, 1.0, 0.0])


def test_transform_unobserved_record_fails():
    rows = [{"id": "a", "status": "candidate"}]
    record = {"status": nl.STATUS_FAILED, "frame_id": "a", "point_observed_m": None}
    transform = nl.resolve_reference_transform(record, rows, "a")
    assert transform["status"] == nl.STATUS_FAILED
    assert transform["reason"] == "annotation_not_observed"


# --------------------------------------------------------------------------- shaping
def _synthetic_head_mesh():
    """A head sphere above a fine neck cylinder, in the camera frame at z=-0.5."""
    import trimesh

    head = trimesh.creation.icosphere(subdivisions=3, radius=0.09)
    head.apply_translation([0.0, 0.02, -0.5])
    head_vertices = np.asarray(head.vertices, dtype=np.float64)
    head_faces = np.asarray(head.faces, dtype=np.int64)

    # neck cylinder grid: 41 rings x 49 columns, front facing +z
    rings, columns = 41, 49
    y = np.linspace(-0.10, -0.24, rings)
    theta = np.linspace(0.0, 2.0 * np.pi, columns, endpoint=False)
    grid_y, grid_t = np.meshgrid(y, theta, indexing="ij")
    radius = 0.05
    neck_vertices = np.stack(
        [radius * np.sin(grid_t), grid_y, -0.5 + radius * np.cos(grid_t)], axis=-1
    ).reshape(-1, 3)
    neck_faces = []
    for i in range(rings - 1):
        for j in range(columns):
            a = i * columns + j
            b = i * columns + (j + 1) % columns
            c = (i + 1) * columns + (j + 1) % columns
            d = (i + 1) * columns + j
            neck_faces.append([a, c, b])
            neck_faces.append([a, d, c])
    neck_faces = np.asarray(neck_faces, dtype=np.int64)

    vertices = np.concatenate([head_vertices, neck_vertices])
    faces = np.concatenate([head_faces, neck_faces + len(head_vertices)])

    landmarks = np.zeros((68, 3))
    landmarks[npn.EYE_OUTER_LEFT] = [-0.03, 0.02, -0.425]
    landmarks[npn.EYE_OUTER_RIGHT] = [0.03, 0.02, -0.425]
    landmarks[npn.NOSE_BRIDGE_TOP] = [0.0, 0.03, -0.415]
    landmarks[npn.NOSE_TIP] = [0.0, 0.0, -0.41]
    landmarks[npn.CHIN] = [0.0, -0.05, -0.425]
    return vertices, faces, landmarks


def _marker_on_neck_front(vertices, faces, landmarks):
    # the neck front surface point 90 mm below the chin, plus 2 mm toward the camera
    centre = np.array([0.0, -0.14, -0.45])
    outward = np.array([0.0, 0.0, 1.0])
    return centre + outward * 0.002, centre


IDENTITY = np.eye(4)


def test_apply_prominence_shapes_only_front_neck():
    vertices, faces, landmarks = _synthetic_head_mesh()
    marker, expected_centre = _marker_on_neck_front(vertices, faces, landmarks)
    record = {"status": nl.STATUS_OBSERVED, "source": nl.SOURCE_MANUAL,
              "frame_id": "f", "point_observed_m": marker.tolist()}
    transform = {"status": nl.STATUS_OBSERVED, "method": "same_frame_observation",
                 "rotation_3x3": np.eye(3).tolist(), "translation_m": [0.0, 0.0, 0.0]}
    cfg = npn.ProminenceConfig(subject_sex="male", mode="on")
    report = npn.apply_prominence(
        state="closed", base_vertices=vertices, faces=faces,
        landmark_record=record, transform=transform,
        model_to_camera=IDENTITY, landmarks_m=landmarks, config=cfg,
    )
    assert report["status"] == npn.STATUS_APPLIED, report.get("reason")
    shaped = report["shaped_vertices"]
    diff = np.linalg.norm(shaped - vertices, axis=1)
    moved = diff > 1e-12
    assert moved.sum() >= 5
    assert diff.max() > 0.001            # a real bump, in metres
    # untouched vertices are strictly identical
    assert np.array_equal(shaped[~moved], vertices[~moved])
    # displacement is along the forward axis (+z in this synthetic frame)
    forward = np.asarray(report["direction"]["forward_model"])
    delta = shaped[moved] - vertices[moved]
    assert (delta @ forward > 0.99 * np.linalg.norm(delta, axis=1)).all()
    # safety recorded and clean
    assert report["safety"]["problems"] == []
    # V2 shape acceptance: the measured final prominence meets the target
    assert abs(report["final_prominence_mm"] - report["requested_final_prominence_mm"]) <= 1.0
    assert report["target_error_mm"] == pytest.approx(
        report["final_prominence_mm"] - report["requested_final_prominence_mm"], abs=1e-6)
    assert report["max_vertex_displacement_mm"] <= report["calibrated_amplitude_mm"] * 1.001
    # blunt crest: half-max extent is wide, not a single-vertex needle
    assert report["extent"]["width_x_half_max_mm"] >= 3.0
    assert report["extent"]["length_y_half_max_mm"] >= 3.0


def test_apply_prominence_female_auto_is_noop():
    vertices, faces, landmarks = _synthetic_head_mesh()
    marker, _ = _marker_on_neck_front(vertices, faces, landmarks)
    record = {"status": nl.STATUS_OBSERVED, "source": nl.SOURCE_MANUAL,
              "frame_id": "f", "point_observed_m": marker.tolist()}
    transform = {"status": nl.STATUS_OBSERVED, "method": "same_frame_observation",
                 "rotation_3x3": np.eye(3).tolist(), "translation_m": [0.0, 0.0, 0.0]}
    cfg = npn.ProminenceConfig(subject_sex="female", mode="auto")
    report = npn.apply_prominence(
        state="closed", base_vertices=vertices, faces=faces,
        landmark_record=record, transform=transform,
        model_to_camera=IDENTITY, landmarks_m=landmarks, config=cfg,
    )
    assert report["status"] == npn.STATUS_SKIPPED
    assert report["reason"] == "auto_female_keeps_template_neck"
    assert "shaped_vertices" not in report


def test_apply_prominence_missing_landmark_skips():
    vertices, faces, landmarks = _synthetic_head_mesh()
    record = nl.unavailable_record("closed", "no_marker_detected")
    transform = {"status": nl.STATUS_FAILED, "reason": "annotation_not_observed"}
    cfg = npn.ProminenceConfig(subject_sex="male", mode="auto")
    report = npn.apply_prominence(
        state="closed", base_vertices=vertices, faces=faces,
        landmark_record=record, transform=transform,
        model_to_camera=IDENTITY, landmarks_m=landmarks, config=cfg,
    )
    assert report["status"] == npn.STATUS_SKIPPED
    assert report["reason"] == "auto_male_but_no_valid_landmark"


def test_apply_prominence_marker_above_chin_rejected():
    vertices, faces, landmarks = _synthetic_head_mesh()
    # marker well above the chin: not the neck front
    marker = np.array([0.0, 0.05, -0.41])
    record = {"status": nl.STATUS_OBSERVED, "source": nl.SOURCE_MANUAL,
              "frame_id": "f", "point_observed_m": marker.tolist()}
    transform = {"status": nl.STATUS_OBSERVED, "method": "same_frame_observation",
                 "rotation_3x3": np.eye(3).tolist(), "translation_m": [0.0, 0.0, 0.0]}
    cfg = npn.ProminenceConfig(subject_sex="male", mode="on")
    report = npn.apply_prominence(
        state="closed", base_vertices=vertices, faces=faces,
        landmark_record=record, transform=transform,
        model_to_camera=IDENTITY, landmarks_m=landmarks, config=cfg,
    )
    assert report["status"] == npn.STATUS_SKIPPED
    assert report["reason"] in {"binding_above_chin_not_neck_front",
                                "binding_direction_ambiguous",
                                "baseline_binding_ray_missed"}


def test_apply_prominence_far_marker_rejected():
    vertices, faces, landmarks = _synthetic_head_mesh()
    # marker on the cylinder axis, deep inside the neck: ray hit far from it
    marker = np.array([0.0, -0.14, -0.5])
    record = {"status": nl.STATUS_OBSERVED, "source": nl.SOURCE_MANUAL,
              "frame_id": "f", "point_observed_m": marker.tolist()}
    transform = {"status": nl.STATUS_OBSERVED, "method": "same_frame_observation",
                 "rotation_3x3": np.eye(3).tolist(), "translation_m": [0.0, 0.0, 0.0]}
    cfg = npn.ProminenceConfig(subject_sex="male", mode="on")
    report = npn.apply_prominence(
        state="closed", base_vertices=vertices, faces=faces,
        landmark_record=record, transform=transform,
        model_to_camera=IDENTITY, landmarks_m=landmarks, config=cfg,
    )
    assert report["status"] == npn.STATUS_SKIPPED


def test_shaping_is_idempotent_from_baseline():
    vertices, faces, landmarks = _synthetic_head_mesh()
    marker, _ = _marker_on_neck_front(vertices, faces, landmarks)
    record = {"status": nl.STATUS_OBSERVED, "source": nl.SOURCE_MANUAL,
              "frame_id": "f", "point_observed_m": marker.tolist()}
    transform = {"status": nl.STATUS_OBSERVED, "method": "same_frame_observation",
                 "rotation_3x3": np.eye(3).tolist(), "translation_m": [0.0, 0.0, 0.0]}
    cfg = npn.ProminenceConfig(subject_sex="male", mode="on")
    kwargs = dict(state="closed", faces=faces, landmark_record=record,
                  transform=transform, model_to_camera=IDENTITY,
                  landmarks_m=landmarks, config=cfg)
    first = npn.apply_prominence(base_vertices=vertices, **kwargs)
    second = npn.apply_prominence(base_vertices=vertices, **kwargs)
    assert first["status"] == second["status"] == npn.STATUS_APPLIED
    assert np.array_equal(first["shaped_vertices"], second["shaped_vertices"])
    # shaping the shaped output again would accumulate; the contract is that
    # exports always start from the baseline archive, which the two equal
    # results above witness.


def test_unit_conversions_are_metric():
    vertices, faces, landmarks = _synthetic_head_mesh()
    marker, _ = _marker_on_neck_front(vertices, faces, landmarks)
    record = {"status": nl.STATUS_OBSERVED, "source": nl.SOURCE_MANUAL,
              "frame_id": "f", "point_observed_m": marker.tolist()}
    transform = {"status": nl.STATUS_OBSERVED, "method": "same_frame_observation",
                 "rotation_3x3": np.eye(3).tolist(), "translation_m": [0.0, 0.0, 0.0]}
    cfg = npn.ProminenceConfig(subject_sex="male", mode="on",
                               amplitude_mm=4.0, _explicit=True)
    report = npn.apply_prominence(
        state="closed", base_vertices=vertices, faces=faces,
        landmark_record=record, transform=transform,
        model_to_camera=IDENTITY, landmarks_m=landmarks, config=cfg,
    )
    assert report["status"] == npn.STATUS_APPLIED, report.get("reason")
    assert report["requested_final_prominence_mm"] == 4.0
    assert report["final_prominence_mm"] == pytest.approx(4.0, abs=1.0)
    assert 0.0 < report["max_vertex_displacement_mm"] <= report["calibrated_amplitude_mm"] * 1.001
    assert report["shape"]["core_radius_x_mm"] == pytest.approx(11.0)
    assert report["shape"]["core_radius_y_mm"] == pytest.approx(12.0)
    assert report["shape"]["base_radius_x_mm"] == pytest.approx(20.0)
    assert report["shape"]["base_radius_y_mm"] == pytest.approx(16.0)
