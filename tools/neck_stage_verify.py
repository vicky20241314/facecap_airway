"""Acceptance checks for the unified neck stage (head_rgbd.neck_stage v2).

Every number here is measured on the mesh that was RE-READ from the exported
OBJ, not on the in-memory arrays, so a writer bug cannot pass silently.
Geometric safety and structural plausibility are reported separately: passing
one says nothing about the other.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from head_rgbd import mesh_integrity as mi     # noqa: E402
from head_rgbd import neck_stage as ns         # noqa: E402
from head_rgbd import neck_volume as nv        # noqa: E402


def _q(x, qs=(50, 90, 99)):
    x = np.asarray(x, float)
    if not len(x):
        return None
    return {f"p{q}": float(np.percentile(x, q)) for q in qs}


def _health(V, F):
    topo = mi.topology_report(V, F)
    tri = V[F]
    e = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                  np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                  np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)], axis=1) * 1000.0
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                axis=1) * 1e6
    ang = np.degrees(np.arccos(np.clip(
        (e[:, 0] ** 2 + e[:, 1] ** 2 - e[:, 2] ** 2) / np.maximum(2 * e[:, 0] * e[:, 1], 1e-12),
        -1, 1)))
    ang = np.minimum(ang, 180.0 - ang)
    return {
        "vertices": int(len(V)), "faces": int(len(F)),
        "components": topo.get("component_sizes"),
        "nonmanifold_edges": topo.get("nonmanifold_edges"),
        "inconsistent_shared_edges": topo.get("inconsistent_shared_edges"),
        "boundary_edges": topo.get("boundary_edges"),
        "boundary_groups": topo.get("boundary_groups"),
        "degenerate_faces_area_lt_1e-6mm2": int((area < 1e-6).sum()),
        "min_angle_deg": {"min": float(np.nanmin(ang)), "p1": float(np.nanpercentile(ang, 1))},
        "edge_mm": {"p50": float(np.percentile(e, 50)), "p99": float(np.percentile(e, 99)),
                    "max": float(e.max())},
        "face_area_mm2": {"p50": float(np.percentile(area, 50)), "min": float(area.min())},
    }


def _dihedral(V, F, mask=None):
    edges, inverse, counts, _, _ = mi.edge_records(F)
    face_ids = np.tile(np.arange(len(F)), 3)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    tri = V[F] * 1000.0
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    sel = starts[counts == 2]
    i0, i1 = face_ids[order[sel]], face_ids[order[sel + 1]]
    ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", fn[i0], fn[i1]), -1, 1)))
    ev = edges[counts == 2]
    out = {"all": _q(ang), "max": float(ang.max())}
    if mask is not None:
        m = mask[ev].all(axis=1)
        sub = ang[m]
        out["band"] = {"edges": int(m.sum()), **_q(sub),
                       "max": float(sub.max()) if len(sub) else None,
                       "over_30deg": int((sub > 30).sum())}
    return out


def _sections(V, F, ax, h_chin, levels=(0.45, 0.60, 0.72, 0.82, 0.90)):
    out = {}
    th = np.linspace(-np.pi, np.pi, 144, endpoint=False)
    for lv in levels:
        segs = ns.section_segments(V, F, ax, lv * h_chin)
        if not len(segs):
            continue
        pts = segs.reshape(-1, 2)
        r = ns.ray_radius(segs, th, (0.0, 0.0))
        out[f"t={lv:.2f}"] = {
            "n_segments": int(len(segs)),
            "a_range_mm": [float(pts[:, 0].min()), float(pts[:, 0].max())],
            "b_range_mm": [float(pts[:, 1].min()), float(pts[:, 1].max())],
            "r_front_mm": float(r[np.argmin(np.abs(th))]),
            "r_back_mm": float(r[np.argmin(np.abs(np.abs(th) - np.pi))]),
            "r_side_right_mm": float(r[np.argmin(np.abs(th - np.pi / 2))]),
            "r_side_left_mm": float(r[np.argmin(np.abs(th + np.pi / 2))]),
        }
    return out


def _column(field, deg=0.0, t_lo=0.42, t_hi=0.90):
    """Anterior relief column: peaks, saddle and half-width, inside a t window.

    The window matters: above t ~ 0.92 the chin overhang enters the radial field
    and produces a relief maximum that is not neck anatomy at all.  Reporting
    the global maximum would therefore report the jaw, which is why every value
    here is restricted and the peak must be interior to the window.
    """
    j0 = int(np.argmin(np.abs(field["theta"] - np.radians(deg))))
    col = np.asarray(field["relief"], float)[:, j0]
    t = np.asarray(field["t"], float)
    idx = np.flatnonzero((t > t_lo) & (t < t_hi) & np.isfinite(col))
    if len(idx) < 5:
        return {}
    i_pk = idx[int(np.argmax(col[idx]))]
    out = {"window_t": [float(t_lo), float(t_hi)],
           "peak_t": float(t[i_pk]), "peak_mm": float(col[i_pk]),
           "peak_interior": bool(idx[0] < i_pk < idx[-1])}
    half = col[i_pk] / 2.0
    lo = i_pk
    while lo > idx[0] and col[lo] > half:
        lo -= 1
    hi = i_pk
    while hi < idx[-1] and col[hi] > half:
        hi += 1
    out["half_width_mm"] = float((t[hi] - t[lo]) * field["h_chin_mm"])
    below = idx[idx < i_pk]
    if len(below):
        i_s = below[int(np.argmin(col[below]))]
        out["saddle_t"] = float(t[i_s])
        out["saddle_mm"] = float(col[i_s])
        low = below[below < i_s]
        if len(low):
            i_l = low[int(np.argmax(col[low]))]
            out["lower_lobe_t"] = float(t[i_l])
            out["lower_lobe_mm"] = float(col[i_l])
    return out



def _consistency(dr_c, dr_o, shared, ctx, p):
    """Do the two states receive the same design where both of them are free?

    The design is one field in normalised (t, theta).  A vertex can still receive
    a different radial change in the two states for two legitimate reasons: the
    jaw covers the upper anterior neck in the closed state but not in the open
    one, and the state's own geometry decides which vertices are the jaw
    underside.  So the honest measurement is reported twice: over the whole
    shared band, and restricted to the vertices that both states considered
    fully free.
    """
    diff = np.abs(dr_c - dr_o)
    wc = np.asarray(ctx["closed"]["weight"], float)
    wo = np.asarray(ctx["opened"]["weight"], float)
    both = shared & (wc > 0.95) & (wo > 0.95)
    out = {
        "definition": "radial design change (r_target - r_current) per vertex, shared band, "
                      "same normalised (t, theta) grid",
        "band_vertices": int(shared.sum()),
        "max_abs_difference_mm": float(diff[shared].max()) if shared.any() else None,
        "p95_abs_difference_mm": float(np.percentile(diff[shared], 95)) if shared.any() else None,
        "correlation": float(np.corrcoef(dr_c[shared], dr_o[shared])[0, 1]) if shared.sum() > 2 else None,
        "both_states_free_vertices": int(both.sum()),
        "both_states_free_max_mm": float(diff[both].max()) if both.any() else None,
        "both_states_free_p95_mm": float(np.percentile(diff[both], 95)) if both.any() else None,
        "both_states_free_correlation": (float(np.corrcoef(dr_c[both], dr_o[both])[0, 1])
                                         if both.sum() > 2 else None),
        "one_state_only_vertices": int((shared & ~both).sum()),
    }
    return out

def verify(out: Path, obj_closed: Path, obj_open: Path, ctx):
    p = ctx["p"]
    sigma = float(p["structure_sigma_mm"])
    ax_c, ax_o = ctx["ax_c"], ctx["ax_o"]
    h_chin, h_chin_open = ctx["h_chin"], ctx["h_chin_open"]

    Vc, F_c = mi.load_obj(obj_closed)
    Vo, F_o = mi.load_obj(obj_open)
    rep = {"schema": ns.SCHEMA, "method": ns.METHOD,
           "readback": {
               "closed_vertices": int(len(Vc)), "closed_faces": int(len(F_c)),
               "open_vertices": int(len(Vo)), "open_faces": int(len(F_o)),
               "identical_face_arrays": bool(np.array_equal(F_c, F_o)),
               "max_abs_write_error_mm": float(max(
                   np.abs(Vc - np.asarray(ctx["Vc_final"], float)).max() * 1000.0,
                   np.abs(Vo - np.asarray(ctx["Vo_final"], float)).max() * 1000.0)),
           }}

    def state_block(V, V_base, ax, hc_state, key):
        h, a, b = nv.axial_coords(V, ax)
        # same definition as the solver: axial height along THIS state's own axis,
        # divided by the shared design menton height.  A single absolute height in
        # the closed state's camera frame is deliberately not used for the open jaw.
        t_design = h / h_chin
        t_state = h / hc_state
        band = (t_design >= p["t_struct_lo"][0]) & (t_design <= p["t_struct_hi"][1])
        # "face and jaw must not change" is checked in two independent ways:
        # (a) everything above the design menton, in both states;
        # (b) every vertex that the solver itself marked as unmovable.
        prot = t_design >= 1.0
        face = t_state >= 1.0
        solver = ctx["closed" if key == "closed" else "opened"]
        w_moved = np.asarray(solver["weight"], float)
        dn = np.asarray(solver["w_down"], float)
        theta = np.asarray(solver["theta"], float)
        d = np.linalg.norm(V - np.asarray(V_base, float), axis=1) * 1000.0
        return {
            # 0.005 mm: below this the OBJ write/read rounding alone counts as motion
            "moved_vertices": int((d > 5e-3).sum()),
            "max_displacement_mm": float(d.max()),
            "band_mean_mm": float(d[band].mean()) if band.any() else None,
            "band_p95_mm": float(np.percentile(d[band], 95)) if band.any() else None,
            "above_design_menton_vertices": int(prot.sum()),
            "above_design_menton_max_mm": float(d[prot].max()) if prot.any() else 0.0,
            "solver_unmoved_vertices": int((w_moved <= 0.0).sum()),
            "solver_unmoved_max_mm": float(d[w_moved <= 0.0].max()) if (w_moved <= 0.0).any() else 0.0,
            # the jaw underside, per state (its own vertex normals decide it)
            "jaw_underside_vertices": int((dn < 0.05).sum()),
            "jaw_underside_max_mm": float(d[dn < 0.05].max()) if (dn < 0.05).any() else 0.0,
            # above this state's own menton: in the open state the lateral neck
            # legitimately extends above that height, the anterior part must not
            "above_own_menton_vertices": int(face.sum()),
            "above_own_menton_max_mm": float(d[face].max()) if face.any() else 0.0,
            "above_own_menton_front_vertices": int((face & (np.abs(theta) < np.radians(40))).sum()),
            "above_own_menton_front_max_mm": (
                float(d[face & (np.abs(theta) < np.radians(40))].max())
                if (face & (np.abs(theta) < np.radians(40))).any() else 0.0),
            "h_chin_state_mm": hc_state,
            "h_chin_design_mm": h_chin,
        }, band, prot, t_design

    blk_c, band_c, prot_c, td_c = state_block(Vc, ctx["Vc2"], ax_c, h_chin, "closed")
    blk_o, band_o, prot_o, td_o = state_block(Vo, ctx["Vo2"], ax_o, h_chin_open, "open")

    # design consistency between the two states, in the shared (t, theta) frame
    dr_c = np.asarray(ctx["closed"]["dr_mm"], float)
    dr_o = np.asarray(ctx["opened"]["dr_mm"], float)
    w_c = np.asarray(ctx["closed"]["weight"], float)
    shared = w_c > 0.05
    rep["geometry"] = {
        "closed": blk_c, "open": blk_o,
        "design_consistency": _consistency(dr_c, dr_o, shared, ctx, p),
        "two_state": {
            "identical_face_arrays": bool(np.array_equal(F_c, F_o)),
            "max_difference_closed_vs_open_same_index_mm": float(
                np.linalg.norm(Vc - Vo, axis=1).max() * 1000.0),
            "frame_note": "the two states are delivered in their own world frames, as in the "
                          "original delivery; the correspondence is by vertex index",
        },
        "health_closed": _health(Vc, F_c),
        "health_open": _health(Vo, F_o),
        "health_base_refined": _health(np.asarray(ctx["Vc2"], float), np.asarray(ctx["F2"], np.int64)),
        "dihedral": {
            "base_band": _dihedral(np.asarray(ctx["Vc2"], float), ctx["F2"], mask=band_c)["band"],
            "final_band": _dihedral(Vc, F_c, mask=band_c)["band"],
            "final_all": _dihedral(Vc, F_c)["all"],
        },
    }
    rep["geometry"]["self_intersection"] = {
        "closed_band": ns.self_intersections(Vc, F_c, mask=band_c),
        "open_band": ns.self_intersections(Vo, F_o, mask=band_o),
    }
    # the whole-mesh count alone would say nothing: the delivered capture already
    # contains intersections, so the same test is run on the base and on the
    # original dense mesh to attribute what is inherited and what is new
    for key, Vx, Fx in (("closed_whole_mesh", Vc, F_c),
                        ("base_refined_whole_mesh", np.asarray(ctx["Vc2"], float),
                         np.asarray(ctx["F2"], np.int64)),
                        ("original_dense_whole_mesh", np.asarray(ctx["states"]["closed"]["V"], float),
                         np.asarray(ctx["F"], np.int64))):
        try:
            rep["geometry"]["self_intersection"][key] = ns.self_intersections(Vx, Fx)
        except MemoryError as exc:                            # reported, never hidden
            rep["geometry"]["self_intersection"][key] = {"error": str(exc)}

    # distance from the original reconstruction (point -> triangle surface)
    dev = {}
    for key, V0, ax, hcs in (("closed", ctx["states"]["closed"]["V"], ax_c, h_chin),
                             ("open", ctx["states"]["open"]["V"], ax_o, h_chin_open)):
        d, _ = ns.point_to_surface_mm(np.asarray(V0, float), Vc if key == "closed" else Vo,
                                      F_c if key == "closed" else F_o)
        h0, _, _ = nv.axial_coords(np.asarray(V0, float), ax)
        t0 = h0 / hcs
        m_band = (t0 >= p["t_struct_lo"][0]) & (t0 <= p["t_struct_hi"][1])
        m_prot = (h0 / hcs) >= 1.0        # above this state's own menton
        dev[key] = {"method": "point-to-triangle-surface distance (trimesh closest_point)",
                    "band_p50_mm": float(np.median(d[m_band])),
                    "band_p95_mm": float(np.percentile(d[m_band], 95)),
                    "band_max_mm": float(d[m_band].max()),
                    "protected_max_mm": float(d[m_prot].max()) if m_prot.any() else 0.0}
    rep["geometry"]["deviation_from_original"] = dev

    # exact prefix check: the refined mesh keeps the original vertex order, so
    # the displacement of every delivered original vertex can be read directly.
    # The proximity query above is kept because it also says how far the *surface*
    # moved, but at the crown it reports up to 0.24 mm for a vertex that did not
    # move at all, because the delivered cap carries folded slivers there.
    direct = {}
    for key, Vf, ax, hcs in (("closed", Vc, ax_c, h_chin), ("open", Vo, ax_o, h_chin_open)):
        V0 = np.asarray(ctx["states"][key]["V"], float)
        n = len(V0)
        if len(Vf) < n:
            continue
        d = np.linalg.norm(np.asarray(Vf)[:n] - V0, axis=1) * 1000.0
        h0, _, _ = nv.axial_coords(V0, ax)
        t_state = h0 / hcs                     # this state's own menton height
        t_design = h0 / h_chin                 # the solver's protected coordinate
        m_design = t_design >= 1.0
        m_own = t_state >= 1.0
        direct[key] = {
            "method": "direct per-vertex displacement (identical vertex order)",
            "all_max_mm": float(d.max()),
            "design_menton_vertices": int(m_design.sum()),
            "design_menton_max_mm": float(d[m_design].max()) if m_design.any() else 0.0,
            "own_menton_vertices": int(m_own.sum()),
            "own_menton_max_mm": float(d[m_own].max()) if m_own.any() else 0.0,
            "note": "design_menton is the protected set (face, lips, jaw, shared by both "
                    "states); own_menton additionally contains the lateral neck that sits "
                    "above the open state's own (lower) menton and is legitimately reshaped",
            "moved_over_0p01mm": int((d > 0.01).sum())}
    rep["geometry"]["deviation_direct"] = direct

    # fairing of the inherited capture surface (its own evidence)
    fair = {}
    for key, d in (("closed", ctx.get("fair_c")), ("open", ctx.get("fair_o"))):
        if d is None:
            continue
        mag = np.linalg.norm(np.asarray(d, float), axis=1)
        mask = np.asarray(ctx["fair_mask_" + key], bool)
        w = np.asarray(ctx.get("fair_weight_" + key), float)
        fair[key] = {"moved_vertices": int((mag > 5e-3).sum()),
                     "region_vertices": int((w > 0).sum()),
                     "band_p50_mm": float(np.median(mag[mask])),
                     "band_p95_mm": float(np.percentile(mag[mask], 95)),
                     "band_max_mm": float(mag.max()),
                     "bounded_by_mm": float(p["fair_max_mm"]),
                     "weight_zero_at_border": bool(np.isclose(w.min(), 0.0)),
                     "bound_active_vertices": int((mag > p["fair_max_mm"] - 1e-6).sum())}
    rep["geometry"]["base_fairing"] = fair

    # --- structure --------------------------------------------------------
    rax, rh_chin, rcurve = ctx["rx"]
    RV, RF = mi.load_obj(ctx["ref_path"])
    # every field in this report uses the same leak masking as the build, so a
    # reported relief is the relief the design was actually solved against
    ref_field = ns.analyse_surface(RV, RF, rax, rh_chin, curve=rcurve,
                                   structure_sigma=sigma, fine_sigma=p["fine_sigma_mm"],
                                   leak_mask=True)
    fin_field = ns.analyse_surface(Vc, F_c, ax_c, h_chin, curve=ctx["curve_c"],
                                   structure_sigma=sigma, fine_sigma=p["fine_sigma_mm"],
                                   leak_mask=True)
    base_field = ns.analyse_surface(np.asarray(ctx["Vc2"], float), np.asarray(ctx["F2"], np.int64),
                                    ax_c, h_chin, curve=ctx["curve_c"],
                                    structure_sigma=sigma, fine_sigma=p["fine_sigma_mm"],
                                    leak_mask=True)
    open_field = ns.analyse_surface(Vo, F_o, ax_o, h_chin, curve=ctx["curve_o"],
                                    structure_sigma=sigma, fine_sigma=p["fine_sigma_mm"],
                                    leak_mask=True)
    struct = {
        "reference": ns.structure_descriptors(ref_field, "reference"),
        "base_refined": ns.structure_descriptors(base_field, "base"),
        "final_closed": ns.structure_descriptors(fin_field, "final closed"),
        "final_open": ns.structure_descriptors(open_field, "final open", t_hi=0.84),
    }
    if ctx.get("compare") is not None:
        prev = Path(ctx["compare"]) / "head_closed_m.obj"
        if prev.exists():
            PV, PF = mi.load_obj(prev)
            pf = ns.analyse_surface(PV, PF, ax_c, h_chin, curve=ctx["curve_c"],
                                    structure_sigma=sigma, fine_sigma=p["fine_sigma_mm"])
            struct["previous_result"] = ns.structure_descriptors(pf, "previous")
            struct.setdefault("anterior_column", {})["previous_result"] = _column(pf)
    struct.setdefault("anterior_column", {})
    struct["anterior_column"].update({
        "reference": _column(ref_field), "base": _column(base_field),
        "final_closed": _column(fin_field),
        "final_open": _column(open_field, t_hi=0.84)})
    struct["sections"] = {
        "final_closed": _sections(Vc, F_c, ax_c, h_chin),
        "base": _sections(np.asarray(ctx["Vc2"], float), np.asarray(ctx["F2"], np.int64), ax_c, h_chin),
        "reference": _sections(RV, RF, rax, rh_chin),
    }
    rep["structure"] = struct

    fin = struct["anterior_column"]["final_closed"]
    ref = struct["anterior_column"]["reference"]
    rep["design_statement"] = {
        "representation": "one signed radial field R(theta, h) sampled by ray casting the mesh",
        "base": "the subject's own normalised-blur background inside the neck tube "
                "(no inherited facets, no previous reshape result)",
        "structure": "designed axial envelope (main bulge / lower shoulder / shallow saddle) "
                     f"x the reference ring pattern at t = {p['shape_t']:.3f}, amplitude "
                     f"{p.get('amplitude_mm', 0):.2f} mm, axial gate "
                     f"{list(p['structure_t_lo'])} -> {list(p['structure_t_hi'])}",
        "envelope": {"main_t_sigma_mm": [p["env_main"][0], p["env_main"][1] * p["h_chin_proxy_mm"]],
                     "lower_t_sigma_mm": [p["env_second"][0], p["env_second"][1] * p["h_chin_proxy_mm"]],
                     "lower_weight": p["env_second"][2],
                     "saddle_t_sigma_mm": [p["env_saddle"][0], p["env_saddle"][1] * p["h_chin_proxy_mm"]],
                     "saddle_depth": p["env_saddle"][2]},
        "primary_relief": {"final_mm": fin.get("peak_mm"), "reference_mm": ref.get("peak_mm"),
                           "final_t": fin.get("peak_t"), "reference_t": ref.get("peak_t")},
        "secondary_relief": {"final_mm": fin.get("lower_lobe_mm"),
                             "reference_mm": ref.get("lower_lobe_mm"),
                             "final_t": fin.get("lower_lobe_t"),
                             "reference_t": ref.get("lower_lobe_t")},
        "protected": "everything at or above the shared design menton (t_design >= 1) is held "
                     "fixed in both states; the jaw underside is excluded per state by its own "
                     "vertex normal and by the tube-leak test, not by a fixed height",
        "not_claimed": [
            "surface peaks are NOT interpreted as cartilage positions (no label data exists)",
            "the reference carries one prominence (4.43 mm at t=0.75, 27 mm long) plus a broad "
            "lower shoulder; it has no separate second peak. A second *peak* is therefore not "
            "claimed and not fabricated: the lower cricoid-region form is delivered as a weaker, "
            "flatter shoulder (see secondary_relief and the profile figure), because a separate "
            "peak under a 27 mm main bulge would need either a much narrower main bulge than the "
            "reference or a grooved transition, which the brief rules out",
            "the axial envelope is a design decision, not a measurement: only the ring pattern "
            "and the lateral relief come from the reference",
        ],
    }
    return rep
