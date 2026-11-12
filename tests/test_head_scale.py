"""Head-region scale and reference proportion asset regressions."""
import json
from pathlib import Path

import numpy as np

from head_rgbd import head_scale as hs

PROJECT = Path("D:/proiect/facecap_airway_delivery")
ASSET = PROJECT / "assets/neck_reference/identity_rest_20261109.obj"
ASSET_JSON = PROJECT / "assets/neck_reference/identity_rest_20261109_asset.json"


def test_reference_head_dimensions_are_stable_and_asset_exists():
    assert ASSET.is_file() and ASSET_JSON.is_file()
    payload = json.loads(ASSET_JSON.read_text(encoding="utf-8"))
    assert payload["original_sha256"].startswith("2e334111")
    h = payload["head"]
    assert 130 < h["H_head_mm"] < 150
    assert 150 < h["W_head_mm"] < 170
    assert abs(h["W_head_mm"] - 158.61) < 1.0
    assert abs(h["H_head_mm"] - 139.68) < 1.0
    assert payload["reference_l50_A_mm"] > 5


def test_head_scale_and_ratio_targets_declare_the_axis_conflict():
    import numpy as np
    ref = {"H_head_mm": 139.68, "W_head_mm": 158.61, "D_head_mm": 186.59}
    tgt = {"H_head_mm": 228.63, "W_head_mm": 190.36, "D_head_mm": 223.74}
    s = hs.head_scale(ref, tgt)
    assert abs(s["ratios"]["W_head_mm"] - 1.2002) < 1e-3
    assert abs(s["ratios"]["D_head_mm"] - 1.1991) < 1e-3
    assert abs(s["ratios"]["H_head_mm"] - 1.6368) < 1e-3
    assert 1.30 < s["S_head_geometric"] < 1.36
    assert s["height_width_difference"] > 0.4
