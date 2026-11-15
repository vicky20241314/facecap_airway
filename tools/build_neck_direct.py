"""Direct neck stage: original shared topology, material partition, 3-D bridge.

    original shared topology (both states)
      -> shared neutral material coordinate  (closed design frame, per index)
      -> A / B / C partition by geodesic distance in material space
      -> continuous neck target field (2-D, independent of the vertex count)
      -> C: scalar radial fit (area-weighted, quadrature samples inside faces)
         B: 3-D bridge solve between the fixed head and the shaped neck
         A: untouched, by construction
      -> bounded re-spacing inside the frozen vertex budget (relax/split/collapse/flip)
      -> final fit + bridge, export, acceptance

No high-density working mesh is built at any step.  The largest array is the
2-D field grid (205 x 360); the largest *mesh* is the delivered one.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from head_rgbd import mesh_integrity as mi          # noqa: E402
from head_rgbd import neck_bridge as nbr            # noqa: E402
from head_rgbd import neck_direct as nd             # noqa: E402
from head_rgbd import neck_fit as nfit              # noqa: E402
from head_rgbd import neck_partition as npart       # noqa: E402
from head_rgbd import neck_remesh as nr             # noqa: E402
from head_rgbd import neck_stage as ns              # noqa: E402
from head_rgbd import neck_prominence as npn        # noqa: E402
from tools import build_neck_stage as bns           # noqa: E402

PARAMS = {
    # ---- reference densities (ratios of the measured head resolution) -----
    # measured on this subject: mandible lateral 4.86 mm, lower occiput 7.89 mm
    "ell_front_ratio": 1.45,        # anterior neck  1.45 x 4.86 = 7.04 mm
    "ell_structure_ratio": 1.25,    # structure zone 1.25 x 4.86 = 6.07 mm
    "ell_side_ratio": 1.00,         # sides / back = measured lower occiput
    "theta_ant_full": 32.0,         # anterior window, narrower than last round
    "theta_ant_zero": 54.0,
    "struct_t": 0.750,              # main bulge (matches the design envelope)
    "struct_t_sigma": 0.075,        # width of the *shape* envelope in t
    "struct_t_radius_mm": 15.0,     # half-support of the *density* window (mm)
    "theta_struct_full": 14.0,
    "theta_struct_zero": 34.0,
    "struct_comp": 2.0,             # compact-support exponent of the weights
    "interface_t": (0.86, 0.98),
    "ell_min_mm": 3.2,
    "ell_max_mm": 12.0,
    "budget_margin": 0.25,
    "chord_tol_frac": 0.05,         # split only if the chord misses by > 5 % ell
    # ---- regularisation lengths (multiples of the mandible reference) -----
    "reg_structure_ratio": 0.25,
    "reg_connection_ratio": 0.50,
    # ---- fit ------------------------------------------------------------
    "max_d_mm": 6.0,
    "data_t": (0.360, 0.420, 0.880, 0.925),  # material band, both states
    "protect_t": 0.97,
    "protect_guard_mm": 5.0,              # unused for the head, kept for the root
    "nape_lo_deg": 100.0,
    "nape_hi_deg": 150.0,
    "nape_keep": 0.45,
    "fit_max_iter": 40,
    # ---- material partition ---------------------------------------------
    "mat_head_seed_t": 1.15,
    "mat_neck_seed_t": 0.60,
    "bridge_width_mm": 12.0,
    # ---- bridge solve ----------------------------------------------------
    "bridge_reg_mm": 4.0,           # bending length inside the bridge
    "bridge_pin": 200.0,            # weight of the C-side pin (x area)
    "bridge_handover": 0.6,         # weight of the hand-over to A (x area)
    "bridge_tangent": 1.0,          # weight of the tangent-plane term (x area)
    "bridge_prior": 2e-3,           # soft pose prior (x area)
    "bridge_ring_depth": 3,
    # The bridge keeps the submental fold, the neck follows the target; where the
    # two meet, the target has to fade or the displacement jumps across one edge
    # (measured on the previous run: 3.04 mm on the neck side against 0.18 mm on
    # the bridge side at t = 0.87, which folds three triangles by 50-65 deg at the
    # anterior end of the B-C line).  The fade is a ring depth into C from the
    # bridge, not a height: the seam is a geodesic boundary, not a t contour.
    # Two rings (~14 mm) measured better than one on every connection number and
    # left the structure untouched: gradient p95 across the seam 0.45 -> 0.33,
    # worst crease change 37.8 -> 26.6 deg, seam p95 27.9 -> 26.3 (input 29.2),
    # peak relief 4.80 -> 4.78 mm, angular width 20 deg unchanged.
    "seam_taper_rings": 2,          # 0 = off
    # ---- open-state pose adaptation (bounded, tried in this order) -------
    "adapt_candidates": [(0.0, 1.00), (0.0, 0.95), (-2.0, 1.00), (-3.0, 1.00),
                         (-4.0, 1.00), (-4.0, 0.90), (0.0, 0.85), (-4.0, 0.85)],
    "adapt_shift_limit_mm": 4.92,    # 5 % of the neutral neck length (98.42 mm)
    "adapt_crease_limit_deg": 6.0,   # p95 of |change| inside the bridge
    "adapt_sharpness_margin_deg": 3.0,  # ... and no sharper than the input + this
    "adapt_gradient_limit": 0.45,    # mm/mm across the bridge
    # ---- remesh ---------------------------------------------------------
    "split_ratio": 4.0 / 3.0,
    "collapse_ratio": 4.0 / 5.0,
    "remesh_passes": 4,
    "split_max_per_pass": None,      # no fixed op cap: the budget decides
    "collapse_max_per_pass": None,
    "split_min_relief_mm": 0.0,      # set from data at run time
    "flip_max_per_pass": 64,
    # a flip is a *shape* operation: it may re-diagonalise a quad, but the new
    # diagonal may never be sharper than the submental fold's own p95 (32 deg) plus
    # a margin.  At 45 deg the flips created new edges at 45 deg and lifted the
    # delivered p95 in B from 31.8 to 38.3 deg; at 36 deg it stays at 31.4 (30.4 in).
    "flip_max_dihedral_deg": 36.0,
    "flip_below_deg": 26.0,
    "flip_min_improve_deg": 2.0,
    "flip_min_angle_deg": 18.0,
    "relax_iters": 6,
    "relax_step": 0.35,
    "relax_max_frac": 0.30,
    "relax_ramp_rings": 3,          # slide fades to zero over this many edges
    "merge_skip_rings": 2,          # never merge the C rings that carry the bridge
    "free_weight": 0.5,
    # ---- structure design (carried over unchanged) ----------------------
    "structure_sigma_mm": 13.0,
    "fine_sigma_mm": 1.2,
    "structure_gain": 1.18,
    "h_chin_proxy_mm": 98.42,
    "shape_t": 0.755,
    "env_main": (0.755, 0.115, 1.00),
    "env_second": (0.565, 0.090, 0.34),
    "env_saddle": (0.655, 0.040, 0.09),
    "env_t_lo": (0.400, 0.500),
    "env_t_hi": (0.880, 0.945),
    "structure_t_lo": (0.42, 0.50),
    "structure_t_hi": (0.86, 0.945),
    "theta_full_deg": 92.0,
    "theta_zero_deg": 112.0,
    "jaw_top_t": (0.940, 1.000),
    "down_face_lo": 0.20,
    "down_face_hi": 0.40,
    # The target field is only valid inside the material band that is star-shaped
    # about the neck axis.  Beyond the shoulder flare it is not, and a vertex just
    # inside the gate with a data weight of 0.19 was carried 6.6 mm inwards by the
    # smoother alone, folding five quads by 32-49 deg on the lower lateral neck.
    # 3 mm of flare is where the field stops being a description of the surface;
    # 5 mm was too generous.  Measured at 3/6 mm: worst fold 49.0 -> 44.6 deg,
    # p99 4.07 -> 3.75 deg, edges over 20 deg 39 -> 28, structure unchanged.
    "leak_gate_lo": 3.0,
    "leak_gate_hi": 6.0,
}


_VERBOSE = False


def log(msg):
    print(msg, flush=True)


def ramp(x, lo, hi):
    return np.clip((np.asarray(x, float) - lo) / (hi - lo), 0.0, 1.0)


class AdaptedField:
    """Target field with a bounded vertical shift and a prominence scale."""

    def __init__(self, field, bg, shift_mm, gain):
        self.f = field
        self.bg = bg
        self.dh = float(shift_mm)
        self.g = float(gain)

    def __call__(self, h, th):
        hh = np.asarray(h, float) + self.dh
        base = np.asarray(self.bg(hh, th), float)
        val = np.asarray(self.f(hh, th), float)
        return base + self.g * (val - base)


# --------------------------------------------------------------------------
# masks and per-state solves
# --------------------------------------------------------------------------


class TaperedWeight:
    """A band weight times a per-vertex ramp, evaluated barycentrically.

    ``fit_radial`` samples the weight at quadrature points and hands back the
    three parent vertices of each sample (``par``), so the ramp is interpolated
    with the same barycentrics as the radius itself.  Used to fade the shaping
    target into the bridge without moving the bridge solve.
    """

    def __init__(self, base, ramp):
        self.base = base
        self.ramp = np.asarray(ramp, float)

    def __call__(self, P, h, th, r, lam, par):
        w = np.asarray(self.base(P, h, th, r, lam, par), float)
        idx = np.asarray(par, np.int64)
        lam = np.asarray(lam, float)
        return w * np.einsum("nk,nk->n", lam, self.ramp[idx])

    def vertex_weight(self, V_m, ax, curve, h_chin):
        Vmm = np.asarray(V_m, float) * 1000.0
        h, th, r, _ = nd.axial_mm(Vmm, ax, curve, h_chin)
        par = np.arange(len(Vmm))[:, None].repeat(3, axis=1)
        lam = np.broadcast_to(np.array([1.0, 0.0, 0.0]), (len(Vmm), 3))
        return np.asarray(self(Vmm, h, th, r, lam, par), float)


def state_masks(st, F, p, h_material, h_chin_design, part, distance_ramp=None):
    """Continuous data weight + the fixed set, from the material partition.

    ``fixed`` is decided **before** any solve and is the same material point set
    in both states: region A (the head), the open rim and the root.  Region B is
    not solved here -- it gets a 3-D displacement in :func:`bridge_state`.
    """
    V = st["V"]
    ax, curve, h_chin = st["ax"], st["curve"], st["h_chin"]
    P = np.asarray(V, float) * 1000.0
    h, th, r, t_state = nd.axial_mm(P, ax, curve, h_chin)
    thd = np.abs(np.degrees(th))

    dn = npn.vertex_normals(V, F) @ ax["axis"]
    rim = ns.boundary_vertices(F)
    if len(rim) < len(V):
        rim = np.concatenate([rim, np.zeros(len(V) - len(rim), bool)])
    tm = np.asarray(h_material, float) / h_chin_design
    fixed = np.asarray(part["A"], bool) | rim | (tm < 0.02)

    weight = nd.BandWeight(st, F, p, h_chin_design, dn, fixed, h_material=h_material)
    # two ways to fade the shaping weight near the connection.  The v2 build
    # passes a *metric* ramp (distance to A in mm); the historical build uses a
    # fixed number of mesh rings.  Both are continuous at in-face samples.
    seam_rings = int(p.get("seam_taper_rings", 0) or 0)
    if distance_ramp is not None:
        weight_c = TaperedWeight(weight, np.asarray(distance_ramp, float))
        seam_taper = {"mode": "distance_mm",
                      "vertices_below_half": int((np.asarray(distance_ramp) < 0.5).sum()),
                      "vertices_below_one": int((np.asarray(distance_ramp) < 0.999).sum())}
    else:
        ramp_v = np.ones(len(V))
        if seam_rings > 0 and part["B"].any():
            depth = nr.ring_depth(F, ~np.asarray(part["B"], bool), seam_rings)
            ramp_v = nd.srise(depth.astype(float), 0.0, float(seam_rings))
            ramp_v[part["B"]] = 1.0
        weight_c = TaperedWeight(weight, ramp_v)
        seam_taper = {"mode": "rings", "rings": seam_rings,
                      "vertices_below_half": int((ramp_v < 0.5).sum()),
                      "vertices_below_one": int((ramp_v < 0.999).sum())}
    data = weight_c.vertex_weight(V, ax, curve, h_chin)
    # the callable keeps the weight continuous through faces that straddle the
    # fixed boundary; the vertex report/mask still has to be zero where the
    # vertex itself is fixed (it is not an unknown)
    data = np.where(fixed, 0.0, data)
    height = (ramp(tm, *p["data_t"][:2]) * (1.0 - ramp(tm, *p["data_t"][2:])))
    bg = np.asarray(st["bg"](h, th), float)
    bg = np.where(np.isfinite(bg), bg, r)
    gate = 1.0 - ramp(r - bg, p["leak_gate_lo"], p["leak_gate_hi"])
    # anchor options.  ``anchor_from_data`` anchors every vertex whose total data
    # weight is low; that includes the jaw-underside normal gate and the region
    # windows, not only the actual field-invalid regions.  ``anchor_validity_only``
    # anchors only where the radial field is not a description of the surface
    # (tube leak + inward guard + shoulder gate), leaving the merely *faded* rows
    # free to follow the bending/continuity into the fixed head.
    if p.get("anchor_validity_only", False):
        rt = np.asarray(st["field"](h, th), float)
        rt = np.where(np.isfinite(rt), rt, r)
        neg = np.maximum(0.0, -(rt - r) -
                         float(p.get("inward_guard_mm", 1.0)))
        gate_far = 1.0 - nd.ramp(neg, 0.0, float(p.get("inward_gate_mm", 1.5)))
        gate_in = 1.0 - (1.0 - gate_far) * nd.ramp(
            tm, float(p.get("inward_t_lo", 0.82)),
            float(p.get("inward_t_hi", 0.88)))
        outside = nd.ramp(thd, float(p.get("side_theta_lo", 82.0)),
                          float(p.get("side_theta_hi", 95.0)))
        side_ok = 1.0 - outside * (1.0 - nd.ramp(
            tm, float(p.get("side_t_lo", 0.45)),
            float(p.get("side_t_hi", 0.55))))
        anchor = 1.0 - gate * gate_in * side_ok
    elif p.get("anchor_from_data", False):
        anchor = 1.0 - data
    else:
        anchor = 1.0 - gate
    return {"data": data, "weight": weight, "weight_c": weight_c, "gate": gate, "dn": dn,
            "height": height, "t_design": tm, "t_state": t_state,
            "theta": thd, "r": r, "h": h, "fixed": fixed,
            "anchor": anchor, "rim": rim,
            "seam_taper": seam_taper}


def radial_state(st, F, p, h_material, h_chin_design, part, field=None,
                 free_mask=None):
    """Scalar radial fit of region C, with A and B held at zero."""
    m = state_masks(st, F, p, h_material, h_chin_design, part)
    st["masks"] = m
    fld = st["field"] if field is None else field
    A_v = nd.vertex_areas(st["V"], F)[0]
    ell_reg = nd.ell_reg_field(m["t_design"], m["theta"], p)
    fixed = m["fixed"].copy()
    if free_mask is None:
        fixed |= np.asarray(part["B"], bool)
        free = np.asarray(part["C"], bool)
    else:
        fixed |= ~np.asarray(free_mask, bool)
    anchor = A_v * m["anchor"]
    if p.get("anchor_from_weight", False):
        # where the target is not trusted the vertex must stay on the capture:
        # the current anchor only knows about the tube-leak gate, so a vertex just
        # inside the gate with a data weight of 0.19 was free to be carried 6.6 mm
        # inwards by the smoother alone (measured, t = 0.39, |theta| = 101 deg)
        anchor = A_v * (m["anchor"] + (1.0 - m["anchor"]) * (1.0 - np.asarray(m["data"])))
    out = nd.fit_band(st["V"], F, st["ax"], st["curve"], st["h_chin"], fld,
                      ell_reg, fixed, m["weight_c"], p, max_mm=p["max_d_mm"],
                      max_iter=p["fit_max_iter"], anchor=anchor)
    s = out["s_mm"]
    e_r = out["e_r"]
    out["d_rad_mm"] = s[:, None] * e_r
    out["s_target_mm"] = s
    st["d_rad_mm"] = out["d_rad_mm"]
    st["e_r"] = e_r
    st["field_used"] = fld
    st["V_rad"] = out["V_mm"] / 1000.0
    return out


def bridge_state(st, F, p, h_material, h_chin_design, part):
    """3-D solve of region B between the fixed head and the shaped neck."""
    m = st["masks"]
    A_v = nd.vertex_areas(st["V"], F)[0]
    P = np.asarray(st["V"], float) * 1000.0
    h, th, r, _ = nd.axial_mm(P, st["ax"], st["curve"], st["h_chin"])
    Rt = np.asarray(st["field_used"](h, th), float)
    Rt = np.where(np.isfinite(Rt), Rt, r)
    s_target = Rt - r
    out = nbr.solve_bridge(st["V"], F, part, st["d_rad_mm"], st["e_r"],
                           s_target, m["data"], A_v, p)
    out["V_mm"] = P + out["d_mm"]
    st["V_bridge"] = out["V_mm"] / 1000.0
    st["d_bridge_mm"] = out["d_mm"]
    return out


def vector_state(st, F, p, h_material, h_chin_design, part, field=None,
                 free_mask=None):
    """Radial fit of C followed by the 3-D bridge of B."""
    t0 = time.time()
    rad = radial_state(st, F, p, h_material, h_chin_design, part, field, free_mask)
    br = bridge_state(st, F, p, h_material, h_chin_design, part)
    st["V"] = st["V_bridge"]
    st["solution"] = {"radial": rad, "bridge": br, "seconds": time.time() - t0}
    return st["solution"]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runtime/outputs/zp_capwitharuco_neck_direct")
    ap.add_argument("--fit", default="runtime/outputs/zp_capwitharuco/reconstruction/dense")
    ap.add_argument("--reference",
                    default="D:/project_neck/runtime/v8_tail_smoke/experiment_20261109-200612/final/identity_rest.obj")
    ap.add_argument("--accepted", default="runtime/outputs/zp_capwitharuco_neck_stage_v2")
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--remesh-passes", type=int, default=None)
    ap.add_argument("--no-remesh", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--seam-taper-rings", type=int, default=None)
    ap.add_argument("--data-t", type=float, nargs=4, default=None,
                    help="axial data band in material t: ramp in, ramp in, ramp out, ramp out")
    ap.add_argument("--anchor-from-weight", action="store_true",
                    help="also anchor proportionally to (1 - data weight), so a vertex the "
                         "target does not trust is held near the capture instead of being "
                         "carried by the smoother")
    ap.add_argument("--leak-gate", type=float, nargs=2, default=None,
                    help="r - background radius (mm) at which the target is fully trusted / "
                         "fully ignored")
    args = ap.parse_args(argv)

    t_start = time.time()
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJ / out
    out.mkdir(parents=True, exist_ok=True)
    p = dict(PARAMS)
    if args.gain is not None:
        p["structure_gain"] = float(args.gain)
    if args.remesh_passes is not None:
        p["remesh_passes"] = int(args.remesh_passes)
    if args.no_remesh:
        p["remesh_passes"] = 0
    if args.seam_taper_rings is not None:
        p["seam_taper_rings"] = int(args.seam_taper_rings)
    if args.data_t is not None:
        p["data_t"] = tuple(float(x) for x in args.data_t)
    if args.anchor_from_weight:
        p["anchor_from_weight"] = True
    if args.leak_gate is not None:
        p["leak_gate_lo"], p["leak_gate_hi"] = (float(x) for x in args.leak_gate)

    # ---- 1. inputs, frames, one topology ---------------------------------
    states_raw, F, refs = bns.load_subject(Path(args.fit))
    log(f"subject: {len(states_raw['closed']['V'])} vertices / {len(F)} faces (shared topology)")
    st = {}
    for key in ("closed", "open"):
        V = states_raw[key]["V"]
        frame, ax, hc = ns.subject_frame(V, F, states_raw[key]["landmarks"])
        curve, spine = ns.spine_curve(V, F, ax, hc)
        st[key] = {"V": V, "V_input": V.copy(), "V_input_ext": V.copy(),
                   "ax": ax, "h_chin": hc,
                   "curve": curve, "rim": ns.boundary_vertices(F), "tag": key}
    hc_c, hc_o = st["closed"]["h_chin"], st["open"]["h_chin"]
    log(f"  h_chin closed {hc_c:.2f} mm / open {hc_o:.2f} mm (each state's own menton)")
    F0 = F.copy()
    n0 = len(st["closed"]["V"])

    # ---- 2. shared neutral material coordinate ---------------------------
    h_mat, mat = material_coordinate(st, F, p, hc_c)
    t_mat = h_mat / hc_c        # dimensionless material t, shared by both states
    log(f"  material: h from the neutral (closed) frame, t = h / {hc_c:.2f} mm; "
        f"open->closed rigid fit on the stable cranium: "
        f"{mat['register']['angle_deg']:.3f} deg, residual median "
        f"{mat['register']['residual_median_mm']:.3f} mm")
    for k, v in mat["correspondence"].items():
        if isinstance(v, dict):
            log(f"    [{k}] n {v['n']} point-to-surface median {v['median_mm']:.3f} mm "
                f"p95 {v['p95_mm']:.3f} mm (index distance median "
                f"{v['raw_index_distance_median_mm']:.1f} mm)")

    # ---- 3. reference + measured head resolution + target edge length -----
    RV, RF, rframe, rax, rhc, menton = ns.reference_frame(args.reference)
    rcurve, _ = ns.spine_curve(RV, RF, rax, rhc)
    scales = nd.head_scales(st["closed"]["V"], F, st["closed"]["ax"],
                            st["closed"]["curve"], hc_c)
    for k, v in scales.items():
        if k.startswith("_"):
            continue
        log(f"  [{k}] {v['vertices']:5d} v {v['edges']:5d} e  p50 {v['edge_p50_mm']:.2f} mm  "
            f"{v['faces_per_cm2']:.2f} faces/cm2")
    p["jaw_reference_mm"] = scales["mandible_lateral"]["edge_p50_mm"]
    p["occiput_reference_mm"] = scales["occiput_lower"]["edge_p50_mm"]
    ell_c, zone_c, anchors = nd.ell_field(st["closed"]["V"], F, st["closed"]["ax"],
                                          st["closed"]["curve"], hc_c, scales, p,
                                          t_material=h_mat)
    log(f"  target edge: anterior {anchors['ell_anterior_mm']:.2f} / structure "
        f"{anchors['ell_structure_mm']:.2f} / side {anchors['ell_side_mm']:.2f} mm "
        f"(jaw ref {anchors['jaw_reference_mm']:.2f}, occiput ref "
        f"{anchors['occiput_reference_mm']:.2f})")
    log(f"  density zones (compact support, structure half-width "
        f"{anchors['support_t_mm']:.1f} mm / {anchors['support_theta_deg']:.0f} deg): "
        + ", ".join(
            f"z{k} {anchors['zone_vertices'][str(k)]} v"
            + (f" ell {anchors['zone_ell_median_mm'][str(k)]:.2f} mm"
               if anchors['zone_ell_median_mm'].get(str(k)) else "")
            for k in range(4) if anchors['zone_vertices'][str(k)]))

    # ---- 4. material partition (once, shared by both states) -------------
    t0 = time.time()
    part = npart.partition(st["closed"]["V"], F, h_mat, hc_c, p)
    log(f"  partition (geodesic, material): A {part['n_A']} v / B {part['n_B']} v / "
        f"C {part['n_C']} v, bridge width {part['bridge_width_measured_mm']:.1f} mm "
        f"(target {p['bridge_width_mm']:.1f}), A-B {len(part['edges_A_B'])} edges, "
        f"B-C {len(part['edges_B_C'])} edges [{time.time()-t0:.1f}s]")
    log(f"    area cm2: " + ", ".join(f"{k} {v/100:.1f}" for k, v in part["area_mm2"].items()))
    st["part"] = part

    # budget: three independent area budgets -- anterior / structure / sides --
    # measured once, before any topology change, and then frozen
    band = (t_mat > p["data_t"][0]) & (t_mat < p["data_t"][3]) & \
           (part["B"] | part["C"])
    bud_by_zone, zone_names = {}, {0: "anterior", 1: "structure", 2: "side_back"}
    for z, name in zone_names.items():
        reg = band & (zone_c == z)
        bud_by_zone[name] = nd.budget(st["closed"]["V"], F, ell_c, reg, p)
        b = bud_by_zone[name]
        log(f"    budget [{name}]: N_est {b['n_estimate']:.0f} -> {b['n_budget']} "
            f"(current {b['n_current']} v, {b['area_mm2']/100:.1f} cm2, ell "
            f"{float(np.median(ell_c[reg])):.2f} mm)")
    bud = nd.budget(st["closed"]["V"], F, ell_c, band, p)
    bud_by_region = {}
    for k in ("B", "C"):
        reg = part[k] & (t_mat > p["data_t"][0]) & (t_mat < p["data_t"][3])
        b = nd.budget(st["closed"]["V"], F, ell_c, reg, p)
        bud_by_region[k] = b
        log(f"    budget [{k}]: N_est {b['n_estimate']:.0f} -> {b['n_budget']} "
            f"(current {b['n_current']} v, {b['area_mm2']/100:.1f} cm2)")
    log(f"  vertex budget total: N_est {bud['n_estimate']:.0f} "
        f"+{100*bud['margin']:.0f}% -> {bud['n_budget']} (band {bud['n_current']} v, "
        f"{bud['area_mm2']/100:.1f} cm2)")

    # ---- 5. continuous target fields (2-D grids, not meshes) -------------
    sig = p["structure_sigma_mm"]
    ref_field = ns.analyse_surface(RV, RF, rax, rhc, curve=rcurve, structure_sigma=sig,
                                   fine_sigma=p["fine_sigma_mm"], leak_mask=True)
    subj = {}
    for key in ("closed", "open"):
        subj[key] = ns.analyse_surface(st[key]["V"], F, st[key]["ax"], hc_o,
                                       curve=st[key]["curve"], structure_sigma=sig,
                                       fine_sigma=p["fine_sigma_mm"], leak_mask=True)
        tg = bns.target_field(ref_field, subj[key], p)
        st[key]["field"] = ns.Field2D(tg["hs_mm"], tg["theta"], tg["R"])
        st[key]["bg"] = ns.Field2D(tg["hs_mm"], tg["theta"], subj[key]["background"])
        st[key]["target"] = tg
    log(f"  target field {st['closed']['target']['R'].shape[0]} x "
        f"{st['closed']['target']['R'].shape[1]} samples, design amplitude "
        f"{st['closed']['target']['amplitude_mm']:.2f} mm (no mesh of this size exists)")
    log(f"  regularisation length: structure "
        f"{p['reg_structure_ratio']*p['jaw_reference_mm']:.2f} mm, connections "
        f"{p['reg_connection_ratio']*p['jaw_reference_mm']:.2f} mm")

    # ---- 6. solve both states on the untouched original mesh -------------
    sol = {}
    for key in ("closed", "open"):
        sol[key] = vector_state(st[key], F, p, h_mat, hc_c, part)
        log_state(key, st[key], sol[key])
    conn = {k: nbr.connection_metrics(st[k]["V_input"], st[k]["V"], F, part,
                                      st[k]["d_bridge_mm"], st[k]["e_r"])
            for k in ("closed", "open")}
    log_conn("original topology", conn)

    # ---- 7. bounded open-state pose adaptation --------------------------
    adapt = {"candidates": [], "chosen": None}
    V_open_keep = st["open"]["V"].copy()
    for shift, gain in p["adapt_candidates"]:
        if abs(shift) > p["adapt_shift_limit_mm"] + 1e-9:
            continue
        if (shift, gain) == (0.0, 1.0):
            m = conn["open"]
        else:
            st["open"]["V"] = st["open"]["V_input"].copy()
            fld = AdaptedField(st["open"]["field"], st["open"]["bg"], shift, gain)
            vector_state(st["open"], F, p, h_mat, hc_c, part, field=fld)
            m = nbr.connection_metrics(st["open"]["V_input"], st["open"]["V"], F,
                                       part, st["open"]["d_bridge_mm"], st["open"]["e_r"])
        # the acceptance is on the *p95* of the change, not on its maximum: a
        # single sharp edge inside the submental fold is physical, a systematic
        # rise of the whole band is not.  Both are relative to this state's own
        # input surface, so the natural sharpness of the fold cancels.
        # Two families of test, because they answer different questions:
        #  * *sharpness* -- is the delivered bridge sharper than the surface it
        #    came from?  The submental fold is naturally sharp (p95 dihedral
        #    ~30 deg on this subject), so an absolute limit on the change would
        #    reject a perfectly smooth bridge.  The output p95 must not exceed
        #    the input p95 by more than a margin;
        #  * *gradient* -- does the displacement ramp across the bridge at a
        #    physical rate (mm of displacement per mm of surface)?
        sharp_ok = (m["dihedral_out_B_p95_deg"]
                    <= m["dihedral_in_B_p95_deg"] + p["adapt_sharpness_margin_deg"]
                    and m["dihedral_out_crossing_p95_deg"]
                    <= m["dihedral_in_crossing_p95_deg"] + p["adapt_sharpness_margin_deg"])
        change_ok = (m["crease_change_crossing_p95_deg"] <= p["adapt_crease_limit_deg"]
                     or sharp_ok)
        ok = (change_ok and
              m["gradient_crossing_p95_mm_per_mm"] <= p["adapt_gradient_limit"]
              and m["gradient_in_B_p95_mm_per_mm"] <= p["adapt_gradient_limit"])
        rec = {"shift_mm": float(shift), "prominence_gain": float(gain),
               "crease_crossing_p95_deg": m["crease_change_crossing_p95_deg"],
               "crease_B_p95_deg": m["crease_change_B_p95_deg"],
               "crease_crossing_max_deg": m["dihedral_change_crossing_deg"],
               "crease_B_max_deg": m["dihedral_change_in_B_deg"],
               "gradient_crossing_mm_per_mm": m["gradient_crossing_mm_per_mm"],
               "gradient_crossing_p95_mm_per_mm": m["gradient_crossing_p95_mm_per_mm"],
               "gradient_B_p95_mm_per_mm": m["gradient_in_B_p95_mm_per_mm"],
               "dihedral_out_B_p95_deg": m["dihedral_out_B_p95_deg"],
               "dihedral_in_B_p95_deg": m["dihedral_in_B_p95_deg"],
               "dihedral_out_crossing_p95_deg": m["dihedral_out_crossing_p95_deg"],
               "sharp_ok": bool(sharp_ok), "pass": bool(ok)}
        adapt["candidates"].append(rec)
        log(f"  adapt shift {shift:+.1f} mm gain {gain:.2f}: crease p95 "
            f"{rec['crease_crossing_p95_deg']:.2f} deg across / "
            f"{rec['crease_B_p95_deg']:.2f} in B (max "
            f"{rec['crease_crossing_max_deg']:.1f}), gradient p95 "
            f"{rec['gradient_crossing_p95_mm_per_mm']:.3f} mm/mm -> "
            f"{'pass' if ok else 'fail'}")
        if ok:
            adapt["chosen"] = rec
            conn["open"] = m
            break
        st["open"]["V"] = V_open_keep.copy()
    if adapt["chosen"] is None:
        adapt["chosen"] = adapt["candidates"][0]
        log("  adapt: no candidate passed the connection conditions; keeping the "
            "neutral design and reporting the conflict")
    adapt["chosen"]["reason"] = ("neutral design already connects within limits"
                                 if adapt["chosen"]["shift_mm"] == 0.0
                                 and adapt["chosen"]["prominence_gain"] == 1.0
                                 else "neutral design did not connect; the smallest "
                                      "bounded adjustment that did was applied")

    # ---- 8. bounded re-spacing, one shared operation log -----------------
    journal = {"passes": [], "max_working_vertices": int(len(st["closed"]["V"])),
               "max_working_faces": int(len(F))}
    recent = {}
    # frozen per-zone budgets: how many points each density zone is allowed
    cap_zone = {z: int(bud_by_zone[zone_names[z]]["n_budget"]) for z in zone_names}
    zone_frozen = zone_c.copy()
    for k in range(p["remesh_passes"]):
        m = state_masks(st["closed"], F, p, h_mat, hc_c, part)
        m_o = state_masks(st["open"], F, p, h_mat, hc_c, part)
        free = (np.asarray(part["B"], bool) | np.asarray(part["C"], bool)) & \
               (~m["fixed"]) & (~m_o["fixed"])
        alive = np.zeros(len(st["closed"]["V"]), bool)
        alive[np.unique(F)] = True
        ell_k, zone_k, _ = nd.ell_field(st["closed"]["V"], F, st["closed"]["ax"],
                                        st["closed"]["curve"], hc_c, scales, p,
                                        t_material=h_mat)
        # zones do not change identity on a split (both endpoints share one), so
        # the frozen assignment is carried, not recomputed from the new positions
        n_k = len(zone_k)
        if len(zone_frozen) >= n_k:
            zone_k = zone_frozen[:n_k].copy()
        else:
            zone_k[:len(zone_frozen)] = zone_frozen
        room_s, room_c = {}, {}
        for z in zone_names:
            # the same population the budget was measured on: the density band
            # of this zone.  Using every vertex of the zone would compare a
            # band-restricted budget against the whole neck and hand out room
            # that was never granted.
            inz = int(((zone_k == z) & band & alive).sum())
            room_s[z] = max(0, cap_zone[z] - inz)
            room_c[z] = max(0, inz - cap_zone[z])
            if _VERBOSE:
                log(f"    [zone {zone_names[z]}] alive-in-band {inz} / cap "
                    f"{cap_zone[z]} -> room +{room_s[z]} / -{room_c[z]}")
        room = {"split": int(sum(room_s.values())),
                "collapse": int(sum(room_c.values())),
                "split_by_zone": room_s, "collapse_by_zone": room_c}
        # re-spacing slides the neck body; the bridge is *not* slid, because a
        # parameter-space step near the submental fold moves a vertex by far more
        # than its 2-D step suggests and folds the triangles there.  The slide is
        # confined to the density band as well: below it the cross-section is no
        # longer star-shaped about the neck axis and ``place`` -- which both the
        # length measurement and the new positions go through -- stops describing
        # the surface.
        rel = nr.relax({"closed": st["closed"], "open": st["open"]}, F,
                       free & np.asarray(part["C"], bool) & band, p, ell_v=ell_k)
        for key in ("closed", "open"):
            vector_state(st[key], F, p, h_mat, hc_c, part,
                         field=st[key].get("field_used"))
        E, nrm, _per = nr._norm_lengths({"closed": st["closed"], "open": st["open"]},
                                        F, free, ell_k)
        n_cand_split = int(((nrm > p["split_ratio"]) &
                            free[np.asarray(E)[:, 0]] & free[np.asarray(E)[:, 1]]).sum())
        n_cand_collapse = int(((nrm < p["collapse_ratio"]) &
                               free[np.asarray(E)[:, 0]] & free[np.asarray(E)[:, 1]]).sum())
        # the bridge is a connection, not a density region: it is never merged
        # away -- and neither are the two rings of C that carry it.  A merge
        # moves the surviving vertex to the parameter midpoint (up to half an
        # edge), which shears the first C ring against the held B ring exactly
        # like a hard-masked relax step does; the merge is still allowed
        # everywhere else in C.
        not_b = ~np.asarray(part["B"], bool)
        depth_b = nr.ring_depth(F, not_b, p["merge_skip_rings"] + 1)
        collapse_ok = not_b & (depth_b > p["merge_skip_rings"])
        F, ops, ext = nr.respace({"closed": st["closed"], "open": st["open"]}, F, free,
                                 ell_k, p, k, recent, room, zone=zone_k,
                                 collapse_ok=collapse_ok, eligible=band)
        # ---- one shared operation log, applied to every per-vertex attribute ---
        if ops["split"]:
            part, h_mat = npart.extend_partition(part, ops, F, h_mat, vlist=h_mat,
                                                 extra_scalars=("d_head", "d_neck", "score"))
            st["part"] = part
            band = np.concatenate([band, band[[q["edge"][0] for q in ops["split"]]]])
            # the "before" surface at the *current* topology: an original vertex
            # keeps its input position, a split vertex takes the midpoint of its
            # parents.  Without this the acceptance cannot compare the delivered
            # mesh with the surface it started from once the ids have grown.
            for key2 in ("closed", "open"):
                Vi = st[key2]["V_input_ext"]
                a2 = Vi[[q["edge"][0] for q in ops["split"]]]
                b2 = Vi[[q["edge"][1] for q in ops["split"]]]
                st[key2]["V_input_ext"] = np.vstack([Vi, 0.5 * (a2 + b2)])
            zone_frozen = np.concatenate([zone_frozen,
                                          np.asarray([zone_k[q["edge"][0]]
                                                      for q in ops["split"]], np.int8)])
        rec = {"pass": k, "free": int(free.sum()), "room_split": int(room["split"]),
               "room_by_zone": room_s, "cap_by_zone": cap_zone,
               "candidate_split": n_cand_split, "candidate_collapse": n_cand_collapse,
               "split": len(ops["split"]), "collapse": len(ops["collapse"]),
               "flip": len(ops["flip"]), "relax_max_move_mm": rel["max_move_mm"],
               "chord_mm_median": (float(np.median([q["chord_mm"] for q in ops["split"]]))
                                   if ops["split"] else None),
               "vertices": int(len(st["closed"]["V"])), "faces": int(len(F)),
               "ops": ops}
        journal["passes"].append(rec)
        log(f"  remesh pass {k}: free {rec['free']} / room +{rec['room_split']} "
            f"{ {zone_names[z]: room_s[z] for z in zone_names} } | candidates "
            f"{n_cand_split} split / {n_cand_collapse} collapse -> executed "
            f"{rec['split']} / {rec['collapse']} / {rec['flip']} flips | "
            f"{rec['vertices']} v / {rec['faces']} f | relax {rel['max_move_mm']:.2f} mm")
        if ops["rejected"]:
            worst = max(ops["rejected"], key=lambda q: q["chord_mm"])
            log(f"    rejected {len(ops['rejected'])} split candidate(s) where the "
                f"parameter midpoint is not on the surface (largest: edge "
                f"{worst['edge']} chord {worst['chord_mm']:.2f} mm over a "
                f"{worst['edge_len_mm']:.2f} mm edge)")
        journal["max_working_vertices"] = max(journal["max_working_vertices"], rec["vertices"])
        journal["max_working_faces"] = max(journal["max_working_faces"], rec["faces"])
        if not (rec["split"] or rec["collapse"] or rec["flip"]):
            break

    # ---- 9. final solve on the final topology ----------------------------
    finals = {}
    for key in ("closed", "open"):
        V_prev = st[key]["V"].copy()
        finals[key] = vector_state(st[key], F, p, h_mat, hc_c, part,
                                   field=st[key].get("field_used"))
        if _VERBOSE:
            # d_bridge_mm / d_rad_mm are mm displacements of THIS solve only; the
            # delivered mesh differs from the input surface by the sum of all of
            # them, so the accumulated value is what the acceptance has to read.
            db = np.linalg.norm(st[key]["d_bridge_mm"], axis=1).max()
            app = np.linalg.norm(st[key]["V"] - V_prev, axis=1).max() * 1000.0
            tot = np.linalg.norm(st[key]["V"] - st[key]["V_input_ext"], axis=1).max() * 1000.0
            log(f"    [displacement {key}] last solve |d| {db:.2f} mm, applied "
                f"{app:.2f} mm, accumulated vs input {tot:.2f} mm")
        log_state(f"{key} final", st[key], finals[key])
    conn = {k: nbr.connection_metrics(st[k]["V_input_ext"], st[k]["V"], F, part,
                                      st[k]["d_bridge_mm"], st[k]["e_r"])
            for k in ("closed", "open")}
    log_conn("final topology", conn)

    # ---- 10. structure of the delivered surface --------------------------
    desc = {}
    for key in ("closed", "open"):
        a = ns.analyse_surface(st[key]["V"], F, st[key]["ax"], hc_o,
                               curve=st[key]["curve"], structure_sigma=sig,
                               fine_sigma=p["fine_sigma_mm"], leak_mask=True)
        desc[key] = ns.structure_descriptors(a, f"direct {key}",
                                             t_hi=0.90 if key == "closed" else 0.84)
    desc["reference"] = ns.structure_descriptors(ref_field, "reference")
    for key in ("closed", "open"):
        q, qr = desc[key]["primary"], desc["reference"]["primary"]
        log(f"  [{key}] peak {q['relief_mm']:.2f} mm at t={q['t']:.3f} "
            f"(ref {qr['relief_mm']:.2f} at {qr['t']:.3f}), theta "
            f"{q['width_theta_deg']:.0f} deg (ref {qr['width_theta_deg']:.0f}), "
            f"axial {q['width_t_mm']:.0f} mm (ref {qr['width_t_mm']:.0f})")

    # ---- 11. export ------------------------------------------------------
    used = np.unique(F)
    remap = np.full(len(st["closed"]["V"]), -1, np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    Fc = remap[F]
    obj = {}
    for key in ("closed", "open"):
        obj[key] = out / f"head_{key}_direct.obj"
        ns.write_obj(obj[key], st[key]["V"][used], Fc,
                     comment=f"neck direct {key}; {len(used)} v / {len(Fc)} f, "
                             f"gain={p['structure_gain']:.2f}, material partition A/B/C, "
                             f"shared topology")
    journal.update({"vertices_original": int(n0), "faces_original": int(len(F0)),
                    "vertices_allocated": int(len(st["closed"]["V"])),
                    "vertices_final": int(len(used)),
                    "faces_final": int(len(Fc)),
                    "vertices_used": int(len(used)),
                    "merged_slots": int(len(st["closed"]["V"]) - len(used)),
                    "n_budget": int(bud["n_budget"]),
                    "slots_alive": int(len(used))})
    topo0 = mi.topology_report(states_raw["closed"]["V"], F0)
    topo1 = mi.topology_report(st["closed"]["V"][used], Fc)
    keys_t = ("nonmanifold_edges", "inconsistent_shared_edges", "boundary_edges",
              "zero_area_faces", "components")
    journal["topology_input"] = {k: topo0[k] for k in keys_t}
    journal["topology_output"] = {k: topo1[k] for k in keys_t}
    log(f"  topology: nonmanifold {topo0['nonmanifold_edges']}->{topo1['nonmanifold_edges']}, "
        f"inconsistent winding {topo0['inconsistent_shared_edges']}->"
        f"{topo1['inconsistent_shared_edges']}, boundary edges "
        f"{topo0['boundary_edges']}->{topo1['boundary_edges']}, zero-area "
        f"{topo0['zero_area_faces']}->{topo1['zero_area_faces']}, components "
        f"{topo0['components']}->{topo1['components']} "
        f"(exported {len(used)} of {len(st['closed']['V'])} slots)")

    # compaction-consistent arrays: everything below is on the exported ids
    nV = len(st["closed"]["V"])
    part_e = {k: np.asarray(part[k], bool)[used] for k in ("A", "B", "C")}

    def _pad_in(V_in, V_now):
        """Vertices created by a split have no "input" position of their own."""
        V_in = np.asarray(V_in, float)
        if len(V_in) >= nV:
            return V_in
        return np.vstack([V_in, np.asarray(V_now, float)[len(V_in):]])

    zone_e = np.asarray(zone_frozen, np.int8)
    zone_e = np.concatenate([zone_e, np.full(max(0, nV - len(zone_e)), 2, np.int8)])[used]
    # the displacement the export actually contains: delivered minus input, in mm.
    # (the per-solve increments ``d_bridge_mm``/``d_rad_mm`` only describe the last
    # pass and must never be quoted as the shape change of the delivered mesh)
    disp = {}
    for key in ("closed", "open"):
        tot = (st[key]["V"] - st[key]["V_input_ext"]) * 1000.0
        disp[key] = tot
        disp[key + "_rad"] = np.einsum("nk,nk->n", tot, st[key]["e_r"])
    npz = {"F": Fc, "V_closed": st["closed"]["V"][used], "V_open": st["open"]["V"][used],
           "V_closed_input": st["closed"]["V_input_ext"][used],
           "V_open_input": st["open"]["V_input_ext"][used],
           "d_closed": disp["closed"][used], "d_open": disp["open"][used],
           "d_rad_closed": disp["closed_rad"][used],
           "d_rad_open": disp["open_rad"][used],
           "e_r_closed": st["closed"]["e_r"][used],
           "e_r_open": st["open"]["e_r"][used],
           "h_material": h_mat[used], "region": np.where(part_e["A"], 0, np.where(
               part_e["B"], 1, 2)).astype(np.int8),
           "density_zone": zone_e,
           "orig_id": used.astype(np.int64),
           "R_target_closed": st["closed"]["target"]["R"],
           "R_target_open": st["open"]["target"]["R"],
           "applied_closed": st["closed"]["target"]["applied"],
           "applied_open": st["open"]["target"]["applied"],
           "background_closed": subj["closed"]["background"],
           "background_open": subj["open"]["background"],
           "theta": st["closed"]["target"]["theta"], "t": st["closed"]["target"]["t"],
           "hs_mm": st["closed"]["target"]["hs_mm"]}
    np.savez_compressed(out / "stage_direct.npz", **npz)

    meta = {"params": p, "scales": scales, "anchors": anchors, "budget": bud,
            "budget_by_region": bud_by_region, "budget_by_zone": bud_by_zone,
            "zone_cap": cap_zone, "structure": desc, "journal": journal,
            "h_chin": hc_c, "h_chin_open": hc_o, "material": mat,
            "partition": {k: v for k, v in part.items()
                          if k not in ("A", "B", "C", "d_head", "d_neck", "score",
                                       "edges_A_B", "edges_B_C", "faces_A_B", "faces_B_C",
                                       "A_idx", "B_idx", "C_idx", "h_material")},
            "connection": conn, "adapt": adapt,
            "fit_closed": {k: v for k, v in finals["closed"]["radial"].items()
                           if k not in ("e_r", "d_rad_mm", "s_target_mm")},
            "fit_open": {k: v for k, v in finals["open"]["radial"].items()
                         if k not in ("e_r", "d_rad_mm", "s_target_mm")},
            "bridge_closed": {k: v for k, v in finals["closed"]["bridge"].items()
                              if k != "d_mm"},
            "bridge_open": {k: v for k, v in finals["open"]["bridge"].items()
                            if k != "d_mm"},
            "masks_closed": {k: (float(np.mean(v)) if isinstance(v, np.ndarray)
                                 else ("callable" if callable(v) else v))
                             for k, v in st["closed"]["masks"].items()},
            "seconds": time.time() - t_start}
    json.dump(meta, open(out / "build_direct.json", "w", encoding="utf-8"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))

    # ---- 12. same-camera acceptance renders ------------------------------
    if not args.no_render:
        from tools import neck_direct_render as ndr
        try:
            rnd = ndr.render_all(out, Path(args.accepted), ref_field=ref_field,
                                 tgt={"applied": st["closed"]["target"]["applied"],
                                      "t": st["closed"]["target"]["t"],
                                      "theta": st["closed"]["target"]["theta"]},
                                 subj_field=subj["closed"], partition=part_e,
                                 h_material=h_mat[used])
            log(f"renders -> {rnd}")
        except Exception as exc:
            log(f"render failed ({type(exc).__name__}: {exc}); meshes are exported")
    log(f"exported in {time.time()-t_start:.1f}s -> {out}")
    return 0


def log_state(tag, st, sol):
    r, b = sol["radial"], sol["bridge"]
    ra = r["residual_after"]
    log(f"  [{tag}] radial: {r['n_free']} free of {len(st['V'])} | max |s| "
        f"{r['max_d_mm']:.2f} mm | {r['n_iter']} iters ({r['stop_reason']}) | "
        f"rms {ra['vertex_rms_mm']:.2f} / quad {ra['quad_rms_mm']:.2f} mm | "
        f"bridge: {b['n_unknown']} unknowns, max |d| in B {b['max_d_B_mm']:.2f} mm | "
        f"{sol['seconds']:.1f}s")


def log_conn(tag, conn):
    for k, m in conn.items():
        log(f"  [{tag} {k}] bridge: gradient {m['gradient_in_B_mm_per_mm']:.3f} in B, "
            f"{m['gradient_crossing_mm_per_mm']:.3f} across the boundary, "
            f"{m['gradient_in_C_mm_per_mm']:.3f} in C mm/mm | dihedral change "
            f"{m['dihedral_change_in_B_deg']:.2f} in B, "
            f"{m['dihedral_change_crossing_deg']:.2f} deg across | p95 "
            f"{m['dihedral_out_B_p95_deg']:.1f} (in {m['dihedral_in_B_p95_deg']:.1f})")


def material_coordinate(st, F, p, h_chin_design):
    """h of the neutral (closed) frame per index, plus the registration check."""
    Vc = st["closed"]["V"]
    Vo = st["open"]["V"]
    h, th, r, t = nd.axial_mm(np.asarray(Vc, float) * 1000.0, st["closed"]["ax"],
                              st["closed"]["curve"], st["closed"]["h_chin"])
    tm = h / h_chin_design
    # registration is estimated on the stable craniofacial region only: the
    # mandible moves by design, and a fit that included it would be meaningless
    stable = tm >= 1.30
    reg = npart.rigid_register(Vo[stable], Vc[stable], trim=0.85)
    Vo_reg = Vo @ reg["R"].T + reg["t"]
    corr = npart.verify_correspondence(Vc, Vo_reg, Vo, F, h, h_chin_design)
    mat = {"register": {"angle_deg": reg["angle_deg"],
                        "inlier_frac": reg["inlier_frac"],
                        "residual_median_mm": float(np.median(reg["residual_mm"])),
                        "residual_p95_mm": float(np.percentile(reg["residual_mm"], 95)),
                        "n_seed": int(stable.sum())},
           "correspondence": corr,
           "h_material_range_mm": [float(h.min()), float(h.max())],
           "note": ("the material axial coordinate is the height in the neutral "
                    "(closed) design frame, one value per vertex index; the design "
                    "band and the protection are functions of it, so they select the "
                    "same material points in both states")}
    return h, mat


if __name__ == "__main__":
    raise SystemExit(main())
