"""Two-feature reference shape and continuous field regressions."""
from pathlib import Path

import numpy as np
import pytest

from head_rgbd import neck_dual as nd

REF = Path("D:/project_neck/runtime/v8_tail_smoke/experiment_20261109-200612/final/identity_rest.obj")


@pytest.mark.skipif(not REF.is_file(), reason="external reference asset is not available")
def test_reference_has_two_vertical_features_and_a_positive_saddle():
    m = nd.extract_two_feature(REF)
    a, b = m["feature_A"], m["feature_B"]
    assert abs(a["x_mm"]) < 1e-6 and abs(b["x_mm"]) < 1e-6
    assert b["y_mm"] < a["y_mm"] - 12.0          # B is below A
    assert 0.25 <= m["relative"]["B_over_A"] <= 0.50
    assert m["saddle"]["height_mm"] > 0.5         # a positive, shallow saddle
    assert m["saddle"]["height_mm"] < b["height_mm"]
    assert m["saddle"]["relative_to_A"] < 0.5
    assert m["support"]["v_top_mm"] > 5.0 and m["support"]["v_bottom_mm"] < -20.0
    d = nd.design_from_morph(m, 0.8)
    u = np.linspace(-24, 24, 97)
    v = np.linspace(d.v_bottom_mm, d.v_top_mm, 161)
    U, V = np.meshgrid(u, v)
    P = d.evaluate(U, V)
    assert P.min() >= -1e-9
    prof = d.profile(v)
    # two local maxima along v: A above B; both positive; the saddle in between
    from scipy.signal import find_peaks
    pk, _ = find_peaks(prof, prominence=.02)
    assert len(pk) >= 2
    order = pk[np.argsort(-prof[pk])]
    ia = int(order[0])
    ib = int([p for p in order[1:] if v[p] < v[ia] - 5][0])
    assert prof[ia] > prof[ib] > 0
    assert prof[(ia + ib) // 2] > 0
    assert abs(P.max() - d.height_mm) < .01 * d.height_mm
