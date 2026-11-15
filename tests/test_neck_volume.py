"""Public 0.9 regression: reference-driven neck volume delivery path.

These tests build the real volume plan once from the saved unshaped dense fit
(no identity reconstruction, no GPU) and check topology, geometry guards,
closed/open correspondence, the corrected head scale and the public wiring.
"""
import inspect
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from head_rgbd import neck_volume as nv
from head_rgbd import pipeline, cli
from head_rgbd import mesh_integrity as mi

PROJECT = Path("D:/proiect/facecap_airway_delivery")
FIT = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/reconstruction/dense"
ASSET = PROJECT / "assets/neck_reference/identity_rest_20261109.obj"


@lru_cache(maxsize=1)
def _plan():
    if not (FIT / "parameters.npz").is_file() or not ASSET.is_file():
        pytest.skip("saved dense fit / reference asset not available")
    refs = json.loads((FIT / "reference_geometry.json").read_text(encoding="utf-8"))
    with np.load(FIT / "parameters.npz", allow_pickle=False) as z:
        dv, df = z["dense_vertices"], z["dense_faces"]
    return nv.build_states(states=("closed", "open"), references=refs["references"],
                           dense_vertices=dv, dense_faces=df,
                           reference_path=ASSET, detail=True)


def test_public_default_and_version_are_switched():
    defaults = json.loads((PROJECT / "config/defaults.json").read_text(encoding="utf-8"))
    assert defaults["neck_prominence"]["algorithm"] == "reference_driven_neck_volume"
    assert defaults["neck_prominence"]["geometry_source"] == "reference_driven_neck_volume"
    assert (PROJECT / "VERSION").read_text(encoding="utf-8").strip() == "0.9.0"
    src = inspect.getsource(pipeline.export_delivery)
    assert "reference_driven_neck_volume" in src
    assert "nvol.build_states" in src
    cli_src = inspect.getsource(cli)
    assert "reference_driven_neck_volume" in cli_src


def test_mode_off_disables_the_volume_branch_before_it_builds():
    """--neck-prominence off must skip the reference volume, not just markers."""
    src = inspect.getsource(pipeline.export_delivery)
    branch = src[src.index('if algorithm == "reference_driven_neck_volume":'):]
    guard = branch[:branch.index("nvol.build_states")]
    assert 'prominence_config.mode == "off"' in guard
    assert "disabled_by_configuration" in guard


def test_plan_is_accepted_with_consistent_subdivision():
    p = _plan()
    assert p["accepted"] is True
    assert p["neck_method"] == nv.NECK_METHOD_VOLUME
    assert p["geometry_source"] == nv.GEOMETRY_SOURCE_VOLUME
    assert p["faces"].shape[0] == 4 * p["base_faces"].shape[0]
    assert len(p["boundary_before"]) == len(p["boundary_after"]) == 2
    assert [len(x) for x in p["boundary_after"]] == [2 * len(x) for x in p["boundary_before"]]
    assert p["detail_applied"] is True


def test_head_scale_corrected_and_shared():
    p = _plan()
    r = p["head_scale"]["ratios"]
    assert abs(r["H_head_mm"] - 1.0131) < 0.02
    assert r["H_head_mm"] < 1.2  # the withdrawn 1.6368 is impossible now
    assert 90 < p["placement"]["h_chin_mm"] < 110
    assert p["placement"]["old_hardcoded_center_vertex"] == 15571
    assert p["reference_morphology"]["menton"]["menton_y_mm"] < -70
    assert p["reference_morphology"]["source_sha256"].startswith("2e334111")


def test_states_share_topology_and_pass_geometry_guards():
    p = _plan()
    closed, open_ = p["states"]["closed"], p["states"]["open"]
    assert closed["run"]["vertices"].shape == open_["run"]["vertices"].shape
    assert closed["verification"]["faces"] == open_["verification"]["faces"]
    for item in (closed, open_):
        g = item["guards"]
        assert g["status"] == "passed"
        assert g["normal_reversal"] == 0
        assert g["nonmanifold_edges"] == 0
        assert g["inconsistent_shared_edges"] == 0
        assert g["degenerate_faces"] == 0
        assert g["slender_faces"] == 0
        assert g["protected_max_displacement_mm"] < 1e-6
        assert g["max_edge_growth"] < 2.2
        assert np.isfinite(item["run"]["vertices"]).all()
    # the closed/open meshes must differ only by the pose/state, not by topology
    assert not np.allclose(closed["run"]["vertices"], open_["run"]["vertices"])


def test_volume_and_detail_are_separate_controlled_fields():
    p = _plan()
    for state in ("closed", "open"):
        run = p["states"][state]["run"]
        vol = 1000.0 * np.linalg.norm(
            run["volume_vertices"] - p["states"][state]["base_subdivided"], axis=1)
        detail = 1000.0 * np.linalg.norm(
            run["vertices"] - run["volume_vertices"], axis=1)
        assert vol.max() > 3.0                      # real volume change, not just smoothing
        assert 0 < detail.max() <= nv.DETAIL_AMPLITUDE_MM * 1.01
        assert np.count_nonzero(run["applied_mm"]) > 1000
    d = p["fields"]["closed"]["volume_disp_mm"] @ np.asarray(p["axial_frame"]["anterior"])
    assert d.min() < -0.5                          # controlled inward motion exists
    assert d.max() > 0.5                           # controlled outward motion exists


def test_bottom_cut_ring_is_preserved_with_in_plane_only_motion():
    p = _plan()
    axis = np.asarray(p["axial_frame"]["axis"])
    base = p["states"]["closed"]["base_subdivided"]
    final = p["states"]["closed"]["run"]["vertices"]
    for lp in p["boundary_before"]:
        for k in lp:
            delta = final[k] - base[k]
            assert abs(float(delta @ axis)) < 1e-9  # cut plane position fixed
    rep = mi.topology_report(final, p["faces"])
    assert rep["nonmanifold_edges"] == 0
    assert rep["inconsistent_shared_edges"] == 0
    assert len(rep["boundary_groups"]) == 2
