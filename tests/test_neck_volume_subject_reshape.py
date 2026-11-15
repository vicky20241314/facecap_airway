"""Regression for the subject-adaptive, topology-preserving neck reshape.

Pins the contract used by tools/reshape_capwitharuco_neck.py: the existing
reconstruction is reshaped IN PLACE (same vertex/face arrays), unrelated regions
stay bit-identical, the measured RGB-D patch is not pushed away, and the 0.9
public build_states path is unaffected.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from head_rgbd import neck_volume as nv
from head_rgbd import mesh_integrity as mi

PROJECT = Path("D:/proiect/facecap_airway_delivery")
RUN = PROJECT / "runtime/outputs/zp_capwitharuco"
FIT = RUN / "reconstruction/dense"
ASSET = PROJECT / "assets/neck_reference/identity_rest_20261109.obj"
PARAMS = {"reference_blend": 1.0, "gain": 2.0, "observation_hold": 0.0,
          "observation_pads": (5.0, 5.0), "cut_anchor_floor": 0.0,
          "shape_params": {"submental_fill_mm": 9.0, "submental_h": 92.0,
                           "submental_sigma": 16.0, "full_edit_below_chin": 38.0,
                           "cut_anchor_width": 36.0}}


@lru_cache(maxsize=1)
def _plan():
    needed = [FIT / "parameters.npz", FIT / "reference_geometry.json", ASSET,
              RUN / "delivery/neck_observed_closed_m.obj"]
    if not all(p.is_file() for p in needed):
        pytest.skip("zp_capwitharuco reconstruction / reference asset not available")
    refs = json.loads((FIT / "reference_geometry.json").read_text(encoding="utf-8"))
    with np.load(FIT / "parameters.npz", allow_pickle=False) as z:
        dv, df = z["dense_vertices"], z["dense_faces"]
    return nv.reshape_subject_neck(
        states=("closed", "open"), references=refs["references"],
        dense_vertices=dv, dense_faces=df, reference_path=ASSET,
        observed_neck_paths={
            "closed": str(RUN / "delivery/neck_observed_closed_m.obj"),
            "open": str(RUN / "delivery/neck_observed_open_m.obj")},
        **PARAMS)


def test_topology_is_preserved_exactly():
    p = _plan()
    assert p["accepted"] is True
    assert p["topology_preserved"] is True
    assert p["boundary_loops_before"] == p["boundary_loops_after"] == [64, 60]
    for s in ("closed", "open"):
        st = p["states"][s]
        assert st["run"]["vertices"].shape == st["base"].shape
        assert st["verification"]["faces"] == 39904
        assert np.array_equal(p["faces"], p["faces"])


def test_unrelated_regions_are_bit_identical():
    p = _plan()
    for s in ("closed", "open"):
        st = p["states"][s]
        V0, V1 = st["base"], st["run"]["vertices"]
        d = np.linalg.norm((V1 - V0) * 1000., axis=1)
        frozen = d < 1e-9
        assert frozen.sum() > 18000
        assert np.array_equal(V0[frozen], V1[frozen])
        assert d[np.asarray(p["boundary_vertices"], int)].max() < 1e-9


def test_geometry_guards_and_smooth_connection():
    p = _plan()
    for s in ("closed", "open"):
        g = p["states"][s]["guards"]
        assert g["status"] == "passed"
        assert g["normal_reversal"] == 0
        assert g["nonmanifold_edges"] == 0
        assert g["inconsistent_shared_edges"] == 0
        assert g["degenerate_faces"] == 0 and g["slender_faces"] == 0
        assert g["max_edge_growth"] < 1.5
        d = p["states"][s]["run"]["applied_mm"]
        V0 = p["states"][s]["base"]
        edges, _, counts, _, _ = mi.edge_records(p["faces"])
        e = edges[counts == 2]
        grad = np.linalg.norm(d[e[:, 0]] - d[e[:, 1]], axis=1)
        elen = np.linalg.norm((V0[e[:, 0]] - V0[e[:, 1]]) * 1000., axis=1)
        assert (grad / np.maximum(elen, 1e-6)).max() < 1.0
        assert 5.0 < np.linalg.norm(d, axis=1).max() < 15.0


def test_measured_neck_patch_is_not_pushed_away():
    p = _plan()
    r = p["measured_patch_report"]["closed"]
    assert r["used_as_constraint"] is True
    assert r["distance_after_mm"]["median"] <= r["distance_before_mm"]["median"] + 1.0
    assert r["distance_after_mm"]["median"] < 4.0
    assert p["measured_patch_report"]["open"]["used_as_constraint"] is False


def test_public_build_states_uses_unchanged_defaults():
    """The new options must not leak into the 0.9 public delivery path."""
    from head_rgbd import neck_volume as module
    assert module.DEFAULT_ALPHA == 1.0
    p = _plan()
    assert p["design"]["gain"] == 2.0
    assert p["design"]["reference_blend"] == 1.0
    # public defaults (no shape_params / observation / gain) remain 1.0/None
    import inspect
    src = inspect.getsource(module.build_design)
    assert "gain: float = 1.0" in src
    src2 = inspect.getsource(module.deform_state)
    assert "observation: dict | None = None" in src2
    assert "shape_params: dict | None = None" in src2
