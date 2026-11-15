"""Build the unified neck appearance stage for a captured head.

    python tools/build_neck_stage.py --out runtime/outputs/zp_capwitharuco_neck_stage_v2

The stage replaces the previous accumulate-and-patch neck scripts:

    base  ->  reference structure transfer  ->  connection & two-state fit  ->  acceptance

Everything is solved on one representation (a signed radial field
``R(theta, h)`` sampled by ray casting an actual triangle mesh), and the
finished surface is re-read from the exported OBJ before anything is verified.
See ``head_rgbd/neck_stage.py`` for the algorithm and ``NECK_STAGE_REPORT.md``
in the output directory for the written account.

The two delivered states live in their own world frames (this is how the dense
reconstruction and the original delivery are built), so each state is solved in
its own frame.  The *design* is shared: both are solved against the same
normalised (t, theta) target field, and the axial datum ``h_chin`` of the closed
state is used for both, so the designed relief does not slide when the jaw
opens.  Protection is the opposite: it is taken from each state's own menton,
so the mandible is never reshaped in either state.

Outputs (in --out):
    head_closed_m.obj / head_open_m.obj   the deliverables
    stage_fields.npz                      radial fields and the applied design
    verification.json                     geometric + structural acceptance
    manifest.json                         inputs, parameters, provenance
    render/*.png                          same-camera comparisons and sections
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from head_rgbd import neck_prominence as npn        # noqa: E402
from head_rgbd import neck_stage as ns              # noqa: E402
from head_rgbd import neck_volume as nv             # noqa: E402

SCHEMA = "head_rgbd.neck_stage.v2"
DEFAULT_REFERENCE = (r"D:/project_neck/runtime/v8_tail_smoke/"
                     r"experiment_20261109-200612/final/identity_rest.obj")

# --- stage parameters (all lengths in mm) ----------------------------------
P = {
    "structure_sigma_mm": 13.0,     # scale that separates relief from the neck taper
    "fine_sigma_mm": 1.2,           # noise floor removed from the relief
    "structure_gain": 1.18,
    "h_chin_proxy_mm": 98.42,   # only used to print the envelope in mm         # transfer gain applied to the reference relief
    "structure_t_lo": (0.42, 0.50),  # axial ramp-in of the transferred relief
    "structure_t_hi": (0.86, 0.945),  # axial ramp-out (below the submental corner)
    "t_struct_lo": (0.360, 0.420),  # editable band, root hand-over above the shoulder plate
    "t_struct_hi": (0.880, 0.925),  # editable band, top hand-over below the chin overhang
    "shape_t": 0.755,              # reference ring used for the angular pattern
    "env_main": (0.755, 0.115, 1.00),    # t, sigma_t, weight
    "env_second": (0.565, 0.090, 0.34),  # weaker, flatter lower bulge
    "env_saddle": (0.655, 0.040, 0.09),  # shallow transition between them
    "env_t_lo": (0.400, 0.500),
    "env_t_hi": (0.880, 0.945),
    "theta_full_deg": 92.0,
    "theta_zero_deg": 112.0,
    "protect_t_design": 0.970,
    "jaw_top_t": (0.940, 1.000),   # per state, fraction of its own menton height       # per state: nothing at/above this own-menton height moves
    "down_face_lo": 0.20,           # surface normal . axis below which the design fades out
    "down_face_hi": 0.40,           # (the jaw underside is not a neck surface)
    "t_state_cap": None,           # replaced by the state's own leak/normal tests  # per state: own-menton height where the design stops
                                    # (the chin overhangs the neck earlier when the jaw is open)
    "fair_iters": 32,              # Taubin pairs, bounded fairing of the capture
    "fair_max_mm": 2.60,           # hard bound on that fairing, per vertex
    "fair_t": (-0.32, 0.945),      # includes the root/shoulder plate, per state
    "fair_ramp_mm": 6.0,
    "fair_t_state": 0.940,
    "fair_dn_min": -0.35,          # never fair the jaw's downward-facing underside         # and never above this fraction of a state's own menton           # blend to zero over this distance at the border
    "refine_levels": 2,
    "refine_t": (-0.20, 1.12),  # include the shoulder plate and the submental fold:
                                # tessellation only, the surface itself is unchanged
    "refine_theta_deg": 178.0,
    "regularize_iters": 3,
    "regularize_weight": 0.35,
    "fillet_t": (0.860, 0.966),
    "fillet_theta_deg": 86.0,
    "fillet_iters": 5,
    "fillet_lambda": 0.30,
    "fillet_max_mm": 1.2,
    "radial_clip_mm": 6.0,          # hard bound on the design displacement per vertex
    "leak_gate_lo": 5.0,           # radius above the tube background treated as "not the neck"
    "leak_gate_hi": 9.0,
}


def log(msg):
    print(msg, flush=True)


def band_mask(V, ax, h_chin, curve, t_rng, theta_deg=None, boundary=None,
              ramp_mm=6.0, h_chin_state=None, t_state_max=None, dn=None,
              dn_min=None):
    """Vertices inside the editable neck tube, in one state's own frame.

    Returns ``(mask, weight)``: the weight falls from 1 to 0 over ``ramp_mm``
    inside the two horizontal borders and at the open rim, so a smoothing pass
    driven by this mask never leaves a crease where the region stops.
    """
    h, a, b = nv.axial_coords(V, ax)
    ca, cb = curve(h)
    th = np.degrees(np.arctan2(a - ca, b - cb))
    t = h / h_chin
    inside = (t > t_rng[0]) & (t < t_rng[1])
    if h_chin_state is not None and t_state_max is not None:
        # the same surface operation must not reach into a state's own jaw
        inside &= (h / h_chin_state) < t_state_max
    if dn is not None and dn_min is not None:
        # the jaw underside is not part of the surface this stage repairs
        inside &= dn > dn_min
    if theta_deg is not None:
        inside &= np.abs(th) < theta_deg
    dist = np.minimum((t - t_rng[0]) * h_chin, (t_rng[1] - t) * h_chin)
    w = np.clip(dist / max(ramp_mm, 1e-6), 0.0, 1.0) ** 1.5
    if boundary is not None:
        w = np.where(boundary, 0.0, w)
    w = np.where(inside, w, 0.0)
    return w > 0.0, w


def ramp(x, lo, hi):
    return nv._ramp(np.asarray(x, float), lo, hi)


# ---------------------------------------------------------------------------
def load_subject(fit_dir):
    z = np.load(fit_dir / "parameters.npz", allow_pickle=False)
    refs = json.loads((fit_dir / "reference_geometry.json").read_text(encoding="utf-8"))
    refs = refs["references"]
    F = np.asarray(z["dense_faces"], np.int64)
    states = {}
    for s in ("closed", "open"):
        rc = refs[s]["stages"]["dense"]
        states[s] = {"V": np.asarray(z["dense_vertices"][rc["frame_index"]], float),
                     "landmarks": np.asarray(rc["landmarks_m"], float),
                     "frame_index": rc["frame_index"],
                     "jaw_angle": refs[s].get("jaw_axis_angle_rad")}
    return states, F, refs


def refine_topology(V, F, m0, levels=2):
    """Conforming red-green refinement; the topology is decided once, on the base."""
    if levels <= 0:
        return {"V": V.copy(), "F": F.copy(), "subs": [], "base_face": np.arange(len(F))}
    sub = ns.refine(V, F, m0)
    Vs = ns.apply_refinement(V, sub)
    Fs = np.asarray(sub["faces"], np.int64)
    base_face = np.asarray(sub["parent_faces"], np.int64)
    subs = [sub]
    for _ in range(levels - 1):
        m = np.asarray(m0, bool)[base_face]
        sub = ns.refine(Vs, Fs, m)
        Vs = ns.apply_refinement(Vs, sub)
        base_face = base_face[np.asarray(sub["parent_faces"], np.int64)]
        Fs = np.asarray(sub["faces"], np.int64)
        subs.append(sub)
    return {"V": Vs, "F": Fs, "subs": subs, "base_face": base_face}


def replay_refinement(V, subs):
    """Replay the same midpoint construction on another state of the topology."""
    out = np.asarray(V, float)
    for sub in subs:
        out = ns.apply_refinement(out, sub)
    return out


def face_band(V, F, ax, h_chin, curve, t_band, theta_max_deg):
    h, a, b = nv.axial_coords(V, ax)
    hc = h[F].mean(axis=1)
    ac = a[F].mean(axis=1)
    bc = b[F].mean(axis=1)
    ca, cb = curve(hc)
    th = np.arctan2(ac - ca, bc - cb)
    t = hc / h_chin
    return (t >= t_band[0]) & (t <= t_band[1]) & (np.abs(np.degrees(th)) <= theta_max_deg)


def face_underside(V, F, ax, h_chin, curve, t_max=1.45, theta_max_deg=120.0,
                   dn_max=-0.05):
    """Faces of the submental fold: ABOVE this state's menton height, downward facing.

    The fold is part of the submental-to-neck connection the brief allows us to
    re-shape, but it sits above ``h_chin`` (moving back from the menton goes up),
    so the axial band alone never selects it.  Splitting those faces changes the
    tessellation only; a face is planar, so its geometry is untouched.
    """
    h, a, b = nv.axial_coords(V, ax)
    hc = h[F].mean(axis=1)
    ac = a[F].mean(axis=1)
    bc = b[F].mean(axis=1)
    ca, cb = curve(hc)
    th = np.arctan2(ac - ca, bc - cb)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    n = np.divide(n, np.maximum(ln, 1e-12)[:, None])
    dn = n @ ax["axis"]
    return ((hc / h_chin) <= t_max) & (np.abs(np.degrees(th)) <= theta_max_deg) & (dn <= dn_max)


def target_field(ref_field, sub_field, p):
    """R_target = subject background + gain * gated reference relief, on (t, theta)."""
    hs_s, th_s = sub_field["hs_mm"], sub_field["theta"]
    t_s = sub_field["t"]
    rel_r = np.where(np.isfinite(ref_field["relief"]), ref_field["relief"], 0.0)
    th_r = ref_field["theta"]
    th_ext = np.r_[th_r - 2 * np.pi, th_r, th_r + 2 * np.pi]
    rel_ext = np.hstack([rel_r, rel_r, rel_r])
    idx = np.clip(np.searchsorted(th_ext, th_s), 1, len(th_ext) - 1)
    wt = ((th_s - th_ext[idx - 1]) / np.maximum(th_ext[idx] - th_ext[idx - 1], 1e-12))[None, :]
    rel_th = rel_ext[:, idx - 1] * (1 - wt) + rel_ext[:, idx] * wt
    rel_t = np.empty((len(hs_s), len(th_s)))
    for j in range(len(th_s)):
        rel_t[:, j] = np.interp(t_s, ref_field["t"], rel_th[:, j])
    gate = ramp(t_s, *p["structure_t_lo"]) * (1.0 - ramp(t_s, *p["structure_t_hi"]))
    shape_th = _angular_shape(ref_field, p)
    env = _axial_envelope(t_s, p)
    amp = float(p["structure_gain"]) * float(np.max(np.abs(rel_t * gate[:, None]))) / max(
        float(np.max(np.abs(env))), 1e-9)
    applied = env[:, None] * shape_th[None, :] * amp * gate[:, None]
    return {"R": sub_field["background"] + applied, "applied": applied, "gate": gate,
            "t": t_s, "theta": th_s, "hs_mm": hs_s, "gain": float(p["structure_gain"]),
            "envelope": env, "shape_theta": shape_th, "amplitude_mm": amp}


def _angular_shape(ref_field, p):
    """Reference relief around the ring at the prominence height, made bilateral.

    The axial *envelope* is designed (see ``_axial_envelope``), so the reference is
    used for what it genuinely carries: the cross-sectional pattern at the
    prominence -- one raised anterior column flanked by oblique hollows whose
    depth is ~0.6 of the anterior relief -- and for the lateral fall-off.
    """
    th_r = np.asarray(ref_field["theta"], float)
    t_r = np.asarray(ref_field["t"], float)
    rel = np.where(np.isfinite(ref_field["relief"]), ref_field["relief"], 0.0)
    i0 = int(np.argmin(np.abs(t_r - float(p["shape_t"]))))
    row = rel[i0]
    m = float(np.max(np.abs(row)))
    if m < 1e-9:
        return np.cos(np.clip(np.asarray(ref_field["theta"], float), -np.pi, np.pi))
    row = row / m
    # bilateral: average with the mirror image about theta = 0
    flipped = row[::-1] if len(row) % 2 else np.r_[row[0], row[:0:-1]]
    row = 0.5 * (row + flipped[:len(row)])
    row = np.clip(row, -1.0, 1.3)
    th_s = np.asarray(ref_field["theta"], float)
    return np.interp(th_s, th_r, row)


def _axial_envelope(t, p):
    """Designed axial profile: main bulge, shallow saddle, weaker lower bulge.

    The reference carries one broad rise, not two lobes, so the axial layering
    the brief asks for cannot be transferred from it -- it is designed here in
    normalised height, with three explicit parts:

        main bulge     A1 at t1, width w1   -- the laryngeal-prominence-like form
        lower bulge    A2 at t2, width w2   -- weaker and flatter (cricoid region)
        transition     Ad at t3, width w3   -- a shallow saddle, not a groove

    All three are Gaussian in t and vanish smoothly at the ends of the band, so
    the same envelope serves both states.
    """
    t1, w1, a1 = p["env_main"]
    t2, w2, a2 = p["env_second"]
    t3, w3, ad = p["env_saddle"]
    prof = (a1 * np.exp(-0.5 * ((t - t1) / w1) ** 2)
            + a2 * np.exp(-0.5 * ((t - t2) / w2) ** 2)
            - ad * np.exp(-0.5 * ((t - t3) / w3) ** 2))
    prof = prof * ramp(t, p["env_t_lo"][0], p["env_t_lo"][1]) * (
        1.0 - ramp(t, p["env_t_hi"][0], p["env_t_hi"][1]))
    return prof / max(float(np.max(np.abs(prof))), 1e-9)

def solve_one(name, V, F, ax, h_chin_design, h_chin_state, curve, field, bg_field, p):
    """Re-solve one state's editable band against the shared design field."""
    h, a, b = nv.axial_coords(V, ax)
    t_design = h / h_chin_design
    t_state = h / h_chin_state
    # the region that must never move is defined in the *shared design* frame
    # (above the design menton it is face and jaw in both states); the per-state
    # jaw is handled by the normal and the tube-leak tests below
    protect = t_design >= p["protect_t_design"]
    ca, cb = curve(h)
    theta = np.arctan2(a - ca, b - cb)
    w_t = ramp(t_design, *p["t_struct_lo"]) * (1.0 - ramp(t_design, *p["t_struct_hi"]))
    if p.get("t_state_cap"):
        # optional, and only for a state whose own submental corner is the limit
        w_t = w_t * (1.0 - ramp(t_state, *p["t_state_cap"]))
    w_th = 1.0 - ramp(np.degrees(np.abs(theta)), p["theta_full_deg"], p["theta_zero_deg"])
    # a downward-facing surface inside the band is the jaw underside, not the
    # neck: its normal (not its height) decides that, per state
    dn = npn.vertex_normals(V, F) @ ax["axis"]
    w_down = 1.0 - ramp(-dn, p["down_face_lo"], p["down_face_hi"])

    r_cur = np.hypot(a - ca, b - cb)
    bg_val = bg_field(h, theta)
    bg_val = np.where(np.isfinite(bg_val), bg_val, r_cur)
    # same "outside the tube" test as the background estimate: a vertex whose
    # radius sits far outside its own smooth tube radius is the chin overhang or
    # the shoulder plate, and is not redesigned
    w_leak = 1.0 - ramp(r_cur - bg_val, p["leak_gate_lo"], p["leak_gate_hi"])
    # safety net, per state and only anterior: an anterior vertex above this
    # state's *own* menton is the mandible, not the neck (the lateral neck does
    # legitimately continue upward behind the jaw, so |theta| gates the rule).
    # The normal and the tube-leak tests already catch the jaw underside and the
    # chin overhang; this closes the small forward-facing patch at the chin tip
    # that neither of them can distinguish from the neck.
    ant = 1.0 - ramp(np.degrees(np.abs(theta)), 45.0, 60.0)
    w_jaw = 1.0 - ant * ramp(t_state, p["jaw_top_t"][0], p["jaw_top_t"][1])
    w = np.where(protect, 0.0, w_t * w_th * w_down * w_leak * w_jaw)

    f = field(h, theta)
    r_tgt = np.where(np.isfinite(f), f, r_cur)
    r_tgt = np.clip(r_tgt, r_cur - 14.0, r_cur + 14.0)
    scale = np.where(r_cur > 1e-6, r_tgt / np.maximum(r_cur, 1e-6), 1.0)
    a_new = ca + r_cur * scale * np.sin(theta)
    b_new = cb + r_cur * scale * np.cos(theta)
    P_tgt = (ax["centre_mm"][None, :] + h[:, None] * ax["axis"][None, :] +
             a_new[:, None] * ax["lateral"][None, :] +
             b_new[:, None] * ax["anterior"][None, :]) / 1000.0
    d = w[:, None] * (P_tgt - V) * 1000.0

    d = ns.regularize(F, d, w > 1e-4, iterations=p["regularize_iters"],
                      weight=p["regularize_weight"]) * (w > 0)[:, None]
    mag = np.linalg.norm(d, axis=1)
    n_clip = int((mag > p["radial_clip_mm"]).sum())
    if n_clip:
        d = d * np.minimum(1.0, p["radial_clip_mm"] / np.maximum(mag, 1e-9))[:, None]

    # submental fillet: a bounded tangential fairing of the junction only
    free = ((t_state > p["fillet_t"][0]) & (t_state < p["fillet_t"][1]) &
            (np.abs(np.degrees(theta)) < p["fillet_theta_deg"]) & ~protect & (w > 0.05))
    V_after = V + d / 1000.0
    V_fil = ns.fillet_junction(V_after, F, free, iterations=p["fillet_iters"],
                               lam=p["fillet_lambda"])
    fd = (V_fil - V_after) * 1000.0
    norm = np.linalg.norm(fd, axis=1)
    if norm.max() > p["fillet_max_mm"]:
        fd = fd * (p["fillet_max_mm"] / norm.max())
    V_final = V_after + fd / 1000.0

    log(f"  [{name}] {int((np.abs(d).max(1) > 1e-6).sum())} moved, "
        f"{int(((w_down < 0.5) & ~protect & (t_design > p['t_struct_lo'][0])).sum())} "
        f"excluded as jaw underside, "
        f"design max {np.abs(d).max():.2f} mm (mean {np.abs(d).max(1)[w > 0].mean():.2f}), "
        f"{n_clip} clipped, fillet max {np.linalg.norm(fd, axis=1).max():.2f} mm, "
        f"protected {int(protect.sum())} vertices")
    return {"V": V_final, "d_total_mm": (V_final - V) * 1000.0, "d_design_mm": d,
            "fillet_mm": fd, "weight": w, "theta": theta, "t_design": t_design,
            "t_state": t_state, "protect": protect, "w_down": w_down, "r_current": r_cur,
            "r_target": r_tgt, "n_clipped": n_clip, "dr_mm": r_tgt - r_cur}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runtime/outputs/zp_capwitharuco_neck_stage_v2")
    ap.add_argument("--fit", default="runtime/outputs/zp_capwitharuco/reconstruction/dense")
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--compare", default="runtime/outputs/zp_capwitharuco_neck_reshaped")
    ap.add_argument("--gain", type=float, default=P["structure_gain"])
    ap.add_argument("--structure-sigma", type=float, default=P["structure_sigma_mm"])
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args(argv)
    p = dict(P)
    p["structure_gain"] = args.gain
    p["structure_sigma_mm"] = args.structure_sigma

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJ / out
    out.mkdir(parents=True, exist_ok=True)
    (out / "render").mkdir(exist_ok=True)
    fit = Path(args.fit)
    if not fit.is_absolute():
        fit = PROJ / fit

    # -- 1. inputs ---------------------------------------------------------
    states, F, refs = load_subject(fit)
    Vc = states["closed"]["V"]
    frame_c, ax_c, h_chin = ns.subject_frame(Vc, F, states["closed"]["landmarks"])
    curve_c, spine = ns.spine_curve(Vc, F, ax_c, h_chin)
    Vo = states["open"]["V"]
    frame_o, ax_o, h_chin_open = ns.subject_frame(Vo, F, states["open"]["landmarks"])
    curve_o, _ = ns.spine_curve(Vo, F, ax_o, h_chin_open)
    log(f"subject: {len(Vc)} vertices / {len(F)} faces")
    log(f"  closed h_chin {h_chin:.2f} mm, cut-plane residual {ax_c.get('cut_plane_residual_mm')}")
    log(f"  open   h_chin {h_chin_open:.2f} mm (own menton; jaw position differs)")

    RV, RF, rframe, rax, rh_chin, menton = ns.reference_frame(args.reference)
    rcurve, _ = ns.spine_curve(RV, RF, rax, rh_chin)
    log(f"reference: {len(RV)} vertices / {len(RF)} faces, h_chin {rh_chin:.2f} mm "
        f"(subject/reference = {h_chin / rh_chin:.3f})")

    sig = p["structure_sigma_mm"]
    ref_field = ns.analyse_surface(RV, RF, rax, rh_chin, curve=rcurve,
                                   structure_sigma=sig, fine_sigma=p["fine_sigma_mm"],
                                   leak_mask=True)
    subj_field = ns.analyse_surface(Vc, F, ax_c, h_chin, curve=curve_c,
                                    structure_sigma=sig, fine_sigma=p["fine_sigma_mm"],
                                    leak_mask=True)
    tgt = target_field(ref_field, subj_field, p)
    p["amplitude_mm"] = float(tgt["amplitude_mm"])   # reported in the design statement

    # -- 2. refinement: one topology shared by both states ------------------
    m0 = face_band(Vc, F, ax_c, h_chin, curve_c, p["refine_t"], p["refine_theta_deg"])
    m_fold = face_underside(Vc, F, ax_c, h_chin, curve_c)
    m0 = m0 | m_fold
    base = refine_topology(Vc, F, m0, levels=int(p["refine_levels"]))
    F2 = base["F"]
    Vc2 = base["V"]
    Vo2 = replay_refinement(Vo, base["subs"])
    log(f"refined neck band x{2 ** int(p['refine_levels'])}: {len(Vc2)} vertices / {len(F2)} faces "
        f"(+{len(Vc2) - len(Vc):d})")

    # -- 2b. fair the inherited capture surface inside the band -------------
    rim_c = ns.boundary_vertices(F2)
    dn_c = npn.vertex_normals(Vc2, F2) @ ax_c["axis"]
    dn_o = npn.vertex_normals(Vo2, F2) @ ax_o["axis"]
    mask_c, wfair_c = band_mask(Vc2, ax_c, h_chin, curve_c, p["fair_t"],
                                boundary=rim_c, ramp_mm=p["fair_ramp_mm"],
                                h_chin_state=h_chin, t_state_max=p["fair_t_state"],
                                dn=dn_c, dn_min=p["fair_dn_min"])
    mask_o, wfair_o = band_mask(Vo2, ax_o, h_chin_open, curve_o, p["fair_t"],
                                boundary=rim_c, ramp_mm=p["fair_ramp_mm"],
                                h_chin_state=h_chin_open, t_state_max=p["fair_t_state"],
                                dn=dn_o, dn_min=p["fair_dn_min"])
    Vc2_raw, Vo2_raw = Vc2.copy(), Vo2.copy()
    Vc2, fair_c = ns.fair_band(Vc2, F2, mask_c, p["fair_iters"], p["fair_max_mm"],
                               weight=wfair_c)
    Vo2, fair_o = ns.fair_band(Vo2, F2, mask_o, p["fair_iters"], p["fair_max_mm"],
                               weight=wfair_o)
    for nm, d in (("closed", fair_c), ("open", fair_o)):
        mag = np.linalg.norm(d, axis=1)
        log(f"  [{nm}] base fairing: {int((mag > 1e-6).sum())} vertices, "
            f"p50 {np.percentile(mag[mask_c if nm == 'closed' else mask_o], 50):.3f} mm, "
            f"max {mag.max():.2f} mm (bound {p['fair_max_mm']})")

    # -- 3. solve both states against the shared design ---------------------
    field = ns.Field2D(tgt["hs_mm"], tgt["theta"], tgt["R"])
    bg_c = ns.Field2D(tgt["hs_mm"], tgt["theta"], subj_field["background"])
    closed = solve_one("closed", Vc2, F2, ax_c, h_chin, h_chin, curve_c, field, bg_c, p)

    o_field = ns.analyse_surface(Vo2, F2, ax_o, h_chin, curve=curve_o,
                                 structure_sigma=sig, fine_sigma=p["fine_sigma_mm"],
                                 leak_mask=True)
    o_tgt = target_field(ref_field, o_field, p)
    openf = ns.Field2D(o_tgt["hs_mm"], o_tgt["theta"], o_tgt["R"])
    bg_o = ns.Field2D(o_tgt["hs_mm"], o_tgt["theta"], o_field["background"])
    opened = solve_one("open", Vo2, F2, ax_o, h_chin, h_chin_open, curve_o, openf, bg_o, p)

    # -- 4. export ---------------------------------------------------------
    obj_c = out / "head_closed_m.obj"
    obj_o = out / "head_open_m.obj"
    ns.write_obj(obj_c, closed["V"], F2,
                 comment=f"neck_stage v2 closed; h_chin={h_chin:.3f}mm gain={p['structure_gain']}")
    ns.write_obj(obj_o, opened["V"], F2,
                 comment="neck_stage v2 open; shared topology + vertex correspondence")
    log(f"wrote {obj_c.name} / {obj_o.name}")

    np.savez_compressed(
        out / "stage_fields.npz",
        t=tgt["t"], theta=tgt["theta"], hs_mm=tgt["hs_mm"],
        R_subject=subj_field["R"], bg_subject=subj_field["background"],
        rel_subject=subj_field["relief"], R_reference=ref_field["R"],
        rel_reference=ref_field["relief"], t_reference=ref_field["t"],
        R_target=tgt["R"], applied=tgt["applied"], gate=tgt["gate"],
        R_open_target=o_tgt["R"], bg_open=o_field["background"],
        h_chin=h_chin, h_chin_open=h_chin_open, h_chin_ref=rh_chin,
        base_face=base["base_face"],
        d_closed_mm=closed["d_total_mm"], d_open_mm=opened["d_total_mm"],
        dr_closed_mm=closed["dr_mm"], dr_open_mm=opened["dr_mm"],
        weight_closed=closed["weight"], weight_open=opened["weight"],
        protect_closed=closed["protect"], protect_open=opened["protect"],
        t_state_closed=closed["t_state"], t_state_open=opened["t_state"],
        fillet_closed_mm=closed["fillet_mm"], fillet_open_mm=opened["fillet_mm"],
        fair_closed_mm=fair_c, fair_open_mm=fair_o, fair_mask_closed=mask_c,
        fair_mask_open=mask_o)

    # -- 5. acceptance -----------------------------------------------------
    from tools import neck_stage_verify as nsv
    ctx = {
        "p": p, "states": states, "F": F, "F2": F2,
        "Vc2": Vc2, "Vo2": Vo2, "Vc2_raw": Vc2_raw, "Vo2_raw": Vo2_raw,
        "fair_c": fair_c, "fair_o": fair_o,
        "fair_mask_closed": mask_c, "fair_mask_open": mask_o,
        "fair_weight_closed": wfair_c, "fair_weight_open": wfair_o,
        "Vc_final": closed["V"], "Vo_final": opened["V"],
        "ax_c": ax_c, "ax_o": ax_o, "h_chin": h_chin, "h_chin_open": h_chin_open,
        "curve_c": curve_c, "curve_o": curve_o, "ref_path": args.reference,
        "closed": closed, "opened": opened, "rx": (rax, rh_chin, rcurve),
        "compare": Path(args.compare) if args.compare else None,
    }
    ver = nsv.verify(out=out, obj_closed=obj_c, obj_open=obj_o, ctx=ctx)
    (out / "verification.json").write_text(
        json.dumps(ns._dumpable(ver), ensure_ascii=False, indent=1), encoding="utf-8")

    manifest = {
        "schema": SCHEMA, "method": ns.METHOD,
        "module": "head_rgbd/neck_stage.py", "driver": "tools/build_neck_stage.py",
        "parameters": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()},
        "inputs": {"fit": str(fit), "reference_obj": args.reference,
                   "compare_previous": args.compare},
        "frames": {"h_chin_mm": h_chin, "h_chin_open_mm": h_chin_open,
                   "h_chin_reference_mm": rh_chin,
                   "closed_axis": ax_c["axis"], "open_axis": ax_o["axis"],
                   "closed_centre_mm": ax_c["centre_mm"], "open_centre_mm": ax_o["centre_mm"],
                   "cut_plane_residual_mm": ax_c.get("cut_plane_residual_mm"),
                   "spine_closed": spine},
        "outputs": {"closed": obj_c.name, "open": obj_o.name, "fields": "stage_fields.npz",
                    "verification": "verification.json", "render": "render/"},
        "design_statement": ver.get("design_statement"),
    }
    (out / "manifest.json").write_text(
        json.dumps(ns._dumpable(manifest), ensure_ascii=False, indent=1), encoding="utf-8")

    if not args.no_render:
        from tools import neck_stage_render as nsr
        nsr.render_all(out=out, closed=obj_c, open=obj_o, reference=Path(args.reference),
                       compare=ctx["compare"], ax=ax_c, h_chin=h_chin, rax=rax,
                       rh_chin=rh_chin, curve_c=curve_c, rcurve=rcurve,
                       tgt=tgt, subj_field=subj_field, ref_field=ref_field,
                       base_V=Vc2)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
