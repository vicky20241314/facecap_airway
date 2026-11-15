"""Head-region scale regressions: corrected menton (0.9) and same-protocol band.

The 0.8 candidate took the front-most lower-face point as the chin and actually
selected the NOSE TIP (y=+1.52 mm), producing H_head=139.68 mm and a 1.6368
longitudinal scale. These tests pin the corrected measurement.
"""
import json
from pathlib import Path

import numpy as np

from head_rgbd import head_scale as hs
from head_rgbd import neck_volume as nv
from head_rgbd.mesh_integrity import load_obj

PROJECT = Path("D:/proiect/facecap_airway_delivery")
ASSET = PROJECT / "assets/neck_reference/identity_rest_20261109.obj"
ASSET_JSON = PROJECT / "assets/neck_reference/identity_rest_20261109_asset.json"


def test_reference_head_dimensions_are_corrected_and_asset_exists():
    assert ASSET.is_file() and ASSET_JSON.is_file()
    payload = json.loads(ASSET_JSON.read_text(encoding="utf-8"))
    assert payload["original_sha256"].startswith("2e334111")
    h = payload["head"]
    assert 215 < h["H_head_mm"] < 235
    assert -95 < h["chin_y_mm"] < -75
    assert h["nose_tip_y_mm"] > -5  # the nose tip is NOT the chin
    assert abs(h["H_head_mm"] - 225.679) < 0.5
    assert "head_legacy_080_bug" in payload
    assert abs(payload["head_legacy_080_bug"]["H_head_mm"] - 139.68) < 1e-6

    V, _ = load_obj(ASSET)
    measured = hs.reference_head_dimensions(V * 1000.0)
    assert abs(measured["H_head_mm"] - h["H_head_mm"]) < 1e-3
    assert abs(measured["chin_y_mm"] - h["chin_y_mm"]) < 1e-3
    assert measured["menton"]["menton_slope_mm_per_mm"] > 3.0
    # the volume module must use exactly the same canonical detector
    assert nv.detect_reference_menton(V * 1000.0) == measured["menton"]


def test_head_scale_no_longer_declares_a_flattened_reference_head():
    ref = {"H_head_mm": 225.679, "W_head_mm": 176.847, "D_head_mm": 203.641}
    tgt = {"H_head_mm": 228.63, "W_head_mm": 190.36, "D_head_mm": 223.74}
    s = hs.head_scale(ref, tgt)
    assert abs(s["ratios"]["H_head_mm"] - 1.0131) < 2e-3
    assert abs(s["ratios"]["W_head_mm"] - 1.0764) < 2e-3
    assert abs(s["ratios"]["D_head_mm"] - 1.0987) < 2e-3
    assert s["height_width_difference"] < 0.0  # old 0.8 claim was +0.4366
    assert 1.05 < s["S_head_geometric"] < 1.12
    assert "1.6368" in s["note"]


def test_head_dimensions_use_the_same_protocol_for_both_heads():
    rng = np.random.default_rng(7)
    # synthetic ellipsoid head: crown at +225 mm, chin at origin
    V = rng.normal(size=(4000, 3)) * np.array([90., 65., 100.])
    V[:, 1] += 112.0
    up = np.array([0., 1., 0.]); right = np.array([1., 0., 0.]); fwd = np.array([0., 0., 1.])
    a = hs.head_dimensions(V, up, right, fwd, np.zeros(3))
    assert a["H_head_mm"] == a["crown_v_mm"]
    assert a["region"].startswith("crown to menton")
    assert a["W_head_mm"] > 0 and a["D_head_mm"] > 0
