"""Direct neck shaping v2: one continuous 3-D displacement, corrected bridge.

Differences from ``tools.build_neck_direct`` (the previous round):

* **One solve.**  The neck body and the submental bridge are no longer two
  geometric targets stitched at a seam.  ``neck_fit.fit_unified`` solves a 3-D
  displacement ``D`` over B + C in one energy: radial data, bending, a strong
  tangential prior in the body and a weak one in the bridge, an anchor where the
  radial target is not trusted, and a Dirichlet fixed head.  A/B/C remain
  weights and acceptance labels only.
* **Interface density.**  The fixed seam inherits the input edge length for one
  local edge length only; further in, the target returns to the design value.
  The inherited dense band is measured, not preserved by construction.
* **Field validity.**  The leak gate is tightened (the radial field is not a
  description of the submental fold or the shoulder flare), and the three
  overlapping top windows of the old round are reduced to one physical fade.
* **Controlled topology.**  No new vertex by default; split only with a
  measured chord error >= 0.35 mm, a per-zone budget and a cumulative cap of 2 %
  of the input neck vertices.  Merges/flips are recorded and checked in both
  states.

The subject base is the 0.5.2-lineage dense reconstruction; the input report is
read and recorded (``neck`` and ``neck_refit`` must show no neck appearance
edit), not inferred from a directory name.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from head_rgbd import mesh_integrity as mi          # noqa: E402
from head_rgbd import neck_bridge as nbr            # noqa: E402
from head_rgbd import neck_anatomy as na             # noqa: E402
from head_rgbd import neck_direct as nd             # noqa: E402
from head_rgbd import neck_fit as nfit              # noqa: E402
from head_rgbd import neck_landmark as nl           # noqa: E402
from head_rgbd import neck_partition as npart       # noqa: E402
from head_rgbd import neck_remesh as nr             # noqa: E402
from head_rgbd import neck_stage as ns              # noqa: E402
from head_rgbd import neck_prominence as npn        # noqa: E402
from tools import build_neck_direct as bnd          # noqa: E402
from tools import build_neck_stage as bns           # noqa: E402

PARAMS = dict(bnd.PARAMS)
PARAMS.update({
    # ---- density: same anchors, one-edge seam inherit ---------------------
    "seam_inherit_ratio": 1.0,      # one local edge length (jaw reference)
    "split_chord_tol_mm": 0.35,     # representation error, not relief
    "max_new_vertex_frac": 0.02,    # of the input B+C vertex count, cumulative
    # ---- field validity ---------------------------------------------------
    # The broad tube-leak gate stays at 3/6 mm: the subject's own anterior
    # prominence is r - background ~1.9 mm and must not be treated as a leak.
    "leak_gate_lo": 3.0,
    "leak_gate_hi": 6.0,
    # A *large negative* target residual above the structure band means the
    # field is describing the jaw underside, not the neck.  The positive
    # laryngeal flank and the shallow saddle are untouched.
    "inward_guard_mm": 1.0,
    "inward_gate_mm": 1.5,
    "inward_t_lo": 0.820,
    "inward_t_hi": 0.880,
    # The shoulder flare (|theta| > ~85 deg below t ~0.55) has no reference
    # structure and its radial field is not the surface.
    "side_theta_lo": 82.0,
    "side_theta_hi": 95.0,
    "side_t_lo": 0.45,
    "side_t_hi": 0.55,
    # v2-only switches; the historical build tool keeps its old behaviour
    "validity_gates": True,
    "weight_fixed_faces": False,
    "anchor_from_data": True,
    # ---- one physical top fade, not three ---------------------------------
    # The *gate* is the single top closure: it also removes the reference's
    # large jaw-underside relief, which is not neck structure.  The designed
    # envelope therefore keeps its full support; fading both the envelope and
    # the gate multiplied two windows and made the last transition abrupt.
    # (Disabling the gate top instead made max(|relief * gate|) jump from 5.2 to
    # 15.8 mm -- the amplitude normaliser then produced a 15.8 mm target, the
    # 6 mm cap saturated and every crease in this round followed from that.)
    "data_t": (0.360, 0.420, 0.940, 1.000),
    "structure_t_hi": (0.860, 0.945),
    "env_t_hi": (1.000, 1.010),        # envelope hands over to the gate only
    "jaw_top_t": (0.940, 1.000),
    # ---- unified solve ----------------------------------------------------
    "mu_t_body": 0.50,              # tangential prior in the neck body
    "mu_t_bridge": 0.150,           # bridge may move in 3-D, but not freely
    "max_t_body_mm": 2.5,
    "max_t_bridge_mm": 2.0,
    "anchor_strength": 0.50,        # anchor = strength * (1 - data weight)
    "anchor_near_A": 2.00,          # extra hand-over stiffness in the first 2.5 edges
    "mu_small": 1e-3,
    "fit_max_iter": 60,
    "seam_prior_edges": 3.0,        # mu_t blend width in local edges
    "seam_taper_ratio": 2.5,        # data weight fades to 0 over 2.5 local edges
                                    # from the protected head (metric, not rings)
    # The reference structure is anterior and lateral; the far side/back of the
    # neck is the subject's own surface.  The old 92/112 deg window kept a full
    # target at |theta| = 85-90 deg, which produced 16-19 deg side folds.
    "theta_full_deg": 78.0,
    "theta_zero_deg": 100.0,
    "pass_rollback_deg": 1.0,       # a pass may not worsen the worst crease
    "crease_hard_cap_deg": 32.0,    # report/gate ceiling for the delivered mesh
    "crease_count_margin": 2,       # allowed increase of signed >10 deg edges
    "crease_gt20_margin": 0,        # no new >20 deg sharpening events
    # ---- bridge width -----------------------------------------------------
    "bridge_width_mm": 12.0,
    # ---- remesh -----------------------------------------------------------
    "seam_taper_rings": 0,          # the unified solve has no hard seam
    "remesh_passes": 12,
    "relax_iters": 0,               # the tested tangential relax added >10 deg
                                     # events; merges/flips are the accepted ops
    "relax_step": 0.35,
    "relax_max_frac": 0.30,
    "relax_ramp_rings": 3,
    "merge_skip_rings": 2,
    "collapse_move_frac": 0.25,     # conformal merge: at most a quarter of the
                                     # edge midpoint shift, never the full 2 mm
    "collapse_max_per_pass": 1,     # one merge per pass, checked by the rollback
    "flip_max_per_pass": 8,
    "structure_gain": 1.10,         # delivered peak 4.50 / 4.29 vs ref 4.43
    # Optional subject-specific *local* gain on the laryngeal band.  It is a
    # smooth 2-D multiplier on the applied relief, not on the whole neck:
    # t 0.755 +/- 0.08, |theta| full 25 deg / zero 40 deg.  Default 1.0 so every
    # previous result is unchanged.
    "larynx_gain": 1.0,
    "larynx_t_center": 0.755,
    "larynx_t_half": 0.080,
    "larynx_theta_full": 25.0,
    "larynx_theta_zero": 40.0,
    # ---- subject-adaptive larynx anatomy + canonical V prior --------------
    # Off by default.  When on, the centre is chosen from the menton landmark
    # and the subject's own anterior normal gate (so the prior fits in the neck,
    # not in the jaw underside); the prior is a narrow V ridge plus shallow
    # lateral hollows, added to the applied relief and scaled by the jaw
    # reference.  It is a semantic prior, not a global gain.
    "larynx_prior": False,
    "larynx_prior_ratio": 0.35,     # x jaw reference (4.03 -> 1.41 mm here)
    "larynx_prior_min_mm": 1.0,
    "larynx_prior_max_mm": 2.5,
    "larynx_prior_t_ref": 0.748,
    "larynx_prior_t_half": 0.045,
    "larynx_prior_t_min": 0.62,
    "larynx_prior_gate_dn": -0.30,
    "larynx_prior_gate_fallback": 0.78,
    "larynx_prior_theta_sigma": 10.0,
    "larynx_prior_theta_zero": 45.0,
    "larynx_prior_theta_zero_hollow": 55.0,
    "larynx_prior_hollow_deg": 28.0,
    "larynx_prior_hollow_sigma": 10.0,
    "larynx_prior_hollow_ratio": 0.35,
    "free_weight": 0.5,
})


def log(msg):
    print(msg, flush=True)


def _bump(x, lo, hi, power=2.0):
    """C1 compact bump: 1 at |x|<=lo, 0 at |x|>=hi."""
    u = np.clip((np.abs(np.asarray(x, float)) - float(lo)) /
                max(float(hi) - float(lo), 1e-12), 0.0, 1.0)
    return (0.5 * (1.0 + np.cos(np.pi * u))) ** float(power)


def _apply_larynx_gain(tg, p, t_center=None):
    """Local smooth multiplier on the applied laryngeal relief.

    It changes only the *designed* relief (``applied``) and its target surface;
    the subject's background and the rest of the neck are untouched.  Default
    gain 1.0 is a no-op, so all previous results remain bit-identical.  When a
    marker centre is supplied, the gain envelope uses the same state-specific
    centre as the V prior.
    """
    lg = float(p.get("larynx_gain", 1.0))
    if abs(lg - 1.0) < 1e-12:
        return tg
    t = np.asarray(tg["t"], float)
    th = np.asarray(tg["theta"], float)
    centre = float(p["larynx_t_center"] if t_center is None else t_center)
    wt = _bump(t - centre, 0.0,
               float(p["larynx_t_half"]), 2.0)
    wth = _bump(th, np.radians(float(p["larynx_theta_full"])),
                np.radians(float(p["larynx_theta_zero"])), 2.0)
    mask = wt[:, None] * wth[None, :]
    base = np.asarray(tg["applied"], float).copy()
    tg["applied_base"] = base
    tg["applied"] = base * (1.0 + (lg - 1.0) * mask)
    tg["R"] = np.asarray(tg["R"], float) + (tg["applied"] - base)
    tg["larynx_gain"] = lg
    tg["larynx_mask"] = mask
    tg["larynx_applied_max_mm"] = float(tg["applied"].max())
    return tg


def _larynx_anatomy_by_state(st, F, states_raw, landmark_states, p):
    """Per-state larynx anatomy, with the marker centre taking priority.

    ``t_center_used = t_marker`` and ``theta_used_deg = 0``; the amplitude and
    half-width stay the anatomy values.  A state without an observed marker
    keeps the menton/normal-gate centre and records the fallback source.
    """
    out: dict[str, dict] = {}
    for key in ("closed", "open"):
        a = na.larynx_anatomy(st[key]["V"], F, st[key]["ax"],
                              st[key]["curve"], st[key]["h_chin"],
                              states_raw[key]["landmarks"], p)
        record = (landmark_states or {}).get(key) or {}
        larynx = record.get("larynx_center") or {}
        marker_ok = (record.get("status") == nl.STATUS_OBSERVED
                     and larynx.get("t_center_used") is not None)
        marker = record.get("marker") or {}
        marker_size = marker.get("size") or {}
        if marker_ok:
            t_center = float(larynx["t_center_used"])
            a.update({
                "t_center": t_center,
                "t_center_used": t_center,
                "t_marker": float(larynx.get("t_marker", t_center)),
                "theta_marker_deg": larynx.get("theta_marker_deg"),
                "theta_used_deg": 0.0,
                "t_center_source": "aruco_marker",
                "t_gate": float(larynx.get("t_gate", a["t_gate"])),
                "d_larynx_mm": float((1.0 - t_center) * st[key]["h_chin"]),
                "marker": {
                    "frame_id": record.get("frame_id"),
                    "reference_frame": record.get("reference_frame"),
                    "transform_status": (record.get("transform") or {}).get("status"),
                    "detected_id": marker.get("target_id"),
                    "black_square_mm": marker_size.get("black_square_mm"),
                    "size_source": marker_size.get("size_source"),
                    "size_uncertainty_mm": marker_size.get("size_uncertainty_mm"),
                },
            })
        else:
            a.update({
                "t_center_source": "menton_gate",
                "marker": None,
                "marker_reason": record.get("reason") or "no_observed_marker",
            })
        out[key] = a
    return out


def _input_provenance(fit_dir: Path, refs):
    """What the input actually is -- read from its report, not from its name."""
    rep_path = fit_dir / "report.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}
    return {
        "fit_dir": str(fit_dir),
        "report_schema": rep.get("schema"),
        "project_version": rep.get("project_version"),
        "algorithm": rep.get("algorithm"),
        "dense_mode": rep.get("dense_mode"),
        "dense_topology": rep.get("dense_topology"),
        "neck_reports": rep.get("neck"),
        "neck_refit": rep.get("neck_refit"),
        "reference_geometry_stages": list(refs["closed"].get("stages", {}).keys()),
        "note": ("neck_reports == [] and neck_refit.enabled == false with status "
                 "disabled_baseline_identical means the dense base has no later "
                 "neck appearance edit; the base head algorithm version is "
                 "recorded in report.json, not inferred from the folder name"),
    }


def _total_d(st):
    return (np.asarray(st["V"], float) - np.asarray(st["V_input_ext"], float)) * 1000.0


def _full_positive_stats(st, F):
    """Worst sharpening and its >10/>20 counts on the complete edge set."""
    E, aout = nbr.dihedral_deg(np.asarray(st["V"], float) * 1000.0, F)
    _e, ain = nbr.dihedral_deg(np.asarray(st["V_input_ext"], float) * 1000.0, F)
    ch = np.abs(aout) - np.abs(ain)
    if not len(ch):
        return 0.0, 0, 0
    return float(np.max(ch)), int((ch > 10.0).sum()), int((ch > 20.0).sum())


def _input_radial(st):
    P = np.asarray(st["V_input_ext"], float) * 1000.0
    return nfit.frame_at(P, st["ax"], st["curve"], st["h_chin"])[4]


class _Marks:
    """Record-only section timer: each call closes the previous section."""

    def __init__(self):
        self.t = time.perf_counter()
        self.out = {}

    def __call__(self, name):
        now = time.perf_counter()
        self.out[name] = round(now - self.t, 6)
        self.t = now


def _mesh_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def solve_state(st, F, p, h_mat, hc_c, part, ell_reg, mu_t, max_t,
                field=None, anchor_mu=None, distance_ramp=None,
                ledger=None, label=""):
    t_wall = time.perf_counter()
    m = bnd.state_masks(st, F, p, h_mat, hc_c, part,
                        distance_ramp=distance_ramp)
    masks_s = time.perf_counter() - t_wall
    st["masks"] = m
    st["field_used"] = st["field"] if field is None else field
    fixed = m["fixed"]
    wfun = m["weight"]
    if anchor_mu is None:
        # ``m["anchor"]`` is 1 - the *delivered* data weight, i.e. it includes
        # the leak gate, the inward gate, the shoulder gate and the metric taper.
        # The extra near-A term is the hand-over to the fixed head; without it a
        # valid but strong target one edge away from A tilted the A-adjacent
        # face by ~20 deg in the first v2 run.
        anchor_mu = p["anchor_strength"] * np.asarray(m["anchor"], float)
        if distance_ramp is not None:
            anchor_mu = anchor_mu + float(p.get("anchor_near_A", 0.0)) * (
                1.0 - np.asarray(distance_ramp, float))
    out = nfit.fit_unified(st["V"], F, st["ax"], st["curve"], st["h_chin"],
                           st["field_used"], ell_reg, fixed, wfun, p,
                           mu_t=mu_t, mu_anchor=anchor_mu,
                           max_d_mm=p["max_d_mm"], max_t_mm=max_t,
                           max_iter=p["fit_max_iter"])
    if ledger is not None:
        rec = dict(out.get("profile") or {})
        rec.update({"label": label, "state": st.get("tag"),
                    "state_masks_s": round(masks_s, 6),
                    "wall_s": round(time.perf_counter() - t_wall, 6),
                    "energy": out["energy"], "n_iter": out["n_iter"],
                    "stop_reason": out["stop_reason"],
                    "vertices": int(len(out["V_mm"])), "faces": int(len(F))})
        ledger.append(rec)
    st["V"] = out["V_mm"] / 1000.0
    st["solve"] = out
    return out


def _structure_side(st, F, p):
    sig = p["structure_sigma_mm"]
    return ns.analyse_surface(st["V"], F, st["ax"], st["h_chin"],
                              curve=st["curve"], structure_sigma=sig,
                              fine_sigma=p["fine_sigma_mm"], leak_mask=True)


def _ell_reg_from(Vc, ax, curve, hc, h_mat, p):
    P = np.asarray(Vc, float) * 1000.0
    h, th, r, _ = nd.axial_mm(P, ax, curve, hc)
    return nd.ell_reg_field(np.asarray(h_mat, float) / float(hc),
                            np.degrees(np.abs(th)), p)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runtime/outputs/zp_capwitharuco_neck_direct_v2")
    ap.add_argument("--fit", default="runtime/outputs/zp_capwitharuco/reconstruction/dense")
    ap.add_argument("--landmarks", type=Path, default=None,
                    help="neck_landmarks.json from neck_landmark_aruco; when a "
                         "state is observed its marker t overrides the larynx "
                         "prior centre (theta=0, amplitude unchanged)")
    ap.add_argument("--reference",
                    default="D:/project_neck/runtime/v8_tail_smoke/experiment_20261109-200612/final/identity_rest.obj")
    ap.add_argument("--no-remesh", action="store_true")
    ap.add_argument("--no-split", action="store_true",
                    help="zero-new-vertex ablation: merges and flips only")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--larynx-gain", type=float, default=None,
                    help="local multiplier on the applied laryngeal relief "
                         "(t 0.755 +/- 0.08, |theta|<25-40 deg); default 1.0")
    ap.add_argument("--larynx-prior", action="store_true",
                    help="add the subject-localized canonical V-shaped larynx "
                         "prior (menton landmark + anterior normal gate)")
    ap.add_argument("--larynx-prior-ratio", type=float, default=None,
                    help="prior amplitude = ratio x jaw reference (default 0.35)")
    ap.add_argument("--larynx-prior-t-half", type=float, default=None,
                    help="prior axial half-width in material t (default 0.045)")
    ap.add_argument("--leak-gate", type=float, nargs=2, default=None)
    ap.add_argument("--bridge-width", type=float, default=None)
    ap.add_argument("--seam-inherit-ratio", type=float, default=None)
    ap.add_argument("--seam-taper-ratio", type=float, default=None,
                    help="data weight fade width from the protected head, in "
                         "local edges (default 2.5)")
    ap.add_argument("--mu-t-bridge", type=float, default=None,
                    help="tangential prior in the submental bridge (default 0.15)")
    ap.add_argument("--mu-t-body", type=float, default=None,
                    help="tangential prior in the neck body (default 0.50)")
    ap.add_argument("--anchor-near-a", type=float, default=None,
                    help="extra hand-over stiffness in the first 2.5 edges from "
                         "the fixed head (default 2.0)")
    ap.add_argument("--relax-iters", type=int, default=None)
    ap.add_argument("--collapse-max", type=int, default=None)
    ap.add_argument("--flip-max", type=int, default=None)
    ap.add_argument("--remesh-passes", type=int, default=None)
    ap.add_argument("--pass-rollback-deg", type=float, default=None)
    ap.add_argument("--crease-hard-cap", type=float, default=None)
    ap.add_argument("--merge-zones", type=str, default=None,
                    help="comma separated zone names allowed to merge, e.g. side_back")
    ap.add_argument("--anchor-validity-only", action="store_true",
                    help="anchor only the field-invalid regions (leak/inward/shoulder), "
                         "not the jaw/nape/taper rows; needed when the jaw underside "
                         "sits inside the structure band (liguanghao)")
    ap.add_argument("--adapt-full-limit", type=float, default=None,
                    help="also require the open-state full-edge positive crease maximum "
                         "to be <= this limit when choosing a pose adaptation")
    args = ap.parse_args(argv)

    t_start = time.time()
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJ / out
    out.mkdir(parents=True, exist_ok=True)
    p = dict(PARAMS)
    if args.gain is not None:
        p["structure_gain"] = float(args.gain)
    if args.larynx_gain is not None:
        p["larynx_gain"] = float(args.larynx_gain)
    if args.larynx_prior:
        p["larynx_prior"] = True
    if args.larynx_prior_ratio is not None:
        p["larynx_prior_ratio"] = float(args.larynx_prior_ratio)
    if args.larynx_prior_t_half is not None:
        p["larynx_prior_t_half"] = float(args.larynx_prior_t_half)
    if args.leak_gate is not None:
        p["leak_gate_lo"], p["leak_gate_hi"] = (float(x) for x in args.leak_gate)
    if args.bridge_width is not None:
        p["bridge_width_mm"] = float(args.bridge_width)
    if args.seam_inherit_ratio is not None:
        p["seam_inherit_ratio"] = float(args.seam_inherit_ratio)
    if args.seam_taper_ratio is not None:
        p["seam_taper_ratio"] = float(args.seam_taper_ratio)
    if args.mu_t_bridge is not None:
        p["mu_t_bridge"] = float(args.mu_t_bridge)
    if args.mu_t_body is not None:
        p["mu_t_body"] = float(args.mu_t_body)
    if args.anchor_near_a is not None:
        p["anchor_near_A"] = float(args.anchor_near_a)
    if args.relax_iters is not None:
        p["relax_iters"] = int(args.relax_iters)
    if args.collapse_max is not None:
        p["collapse_max_per_pass"] = int(args.collapse_max)
    if args.flip_max is not None:
        p["flip_max_per_pass"] = int(args.flip_max)
    if args.remesh_passes is not None:
        p["remesh_passes"] = int(args.remesh_passes)
    if args.pass_rollback_deg is not None:
        p["pass_rollback_deg"] = float(args.pass_rollback_deg)
    if args.crease_hard_cap is not None:
        p["crease_hard_cap_deg"] = float(args.crease_hard_cap)
    merge_zones = (None if args.merge_zones is None else
                   {s.strip() for s in args.merge_zones.split(",") if s.strip()})
    if args.anchor_validity_only:
        p["anchor_validity_only"] = True
    if args.adapt_full_limit is not None:
        p["adapt_full_positive_limit"] = float(args.adapt_full_limit)
    if args.no_remesh:
        p["remesh_passes"] = 0

    # ---- 1. inputs -----------------------------------------------------
    fit_dir = Path(args.fit)
    if not fit_dir.is_absolute():
        fit_dir = PROJ / fit_dir
    states_raw, F, refs = bns.load_subject(fit_dir)
    provenance = _input_provenance(fit_dir, refs)
    log(f"subject: {len(states_raw['closed']['V'])} vertices / {len(F)} faces")
    log(f"input: {provenance['report_schema']} project {provenance['project_version']} "
        f"dense {provenance['dense_topology']}")
    log(f"input neck reports: {provenance['neck_reports']}; neck_refit "
        f"enabled={provenance['neck_refit'].get('enabled') if provenance['neck_refit'] else None} "
        f"status={provenance['neck_refit'].get('status') if provenance['neck_refit'] else None}")
    st = {}
    for key in ("closed", "open"):
        V = np.asarray(states_raw[key]["V"], float)
        frame, ax, hc = ns.subject_frame(V, F, states_raw[key]["landmarks"])
        curve, spine = ns.spine_curve(V, F, ax, hc)
        st[key] = {"V": V, "V_input": V.copy(), "V_input_ext": V.copy(),
                   "ax": ax, "h_chin": hc, "curve": curve,
                   "rim": ns.boundary_vertices(F), "tag": key}
    hc_c, hc_o = st["closed"]["h_chin"], st["open"]["h_chin"]
    F0 = F.copy()
    n0 = len(st["closed"]["V"])
    # A0 time account: every solve of the run, not just the delivered one
    solve_ledger: list[dict] = []
    input_hash = {"vertices_closed": _mesh_hash(st["closed"]["V_input_ext"]),
                  "vertices_open": _mesh_hash(st["open"]["V_input_ext"]),
                  "faces": _mesh_hash(F0),
                  "n_vertices": int(n0), "n_faces": int(len(F0))}
    t_pre = time.perf_counter()
    log(f"  h_chin closed {hc_c:.2f} / open {hc_o:.2f} mm")

    # ---- 2. shared neutral material coordinate -------------------------
    h_mat, mat = bnd.material_coordinate(st, F, p, hc_c)
    t_mat = h_mat / hc_c
    log(f"  material: {mat['register']['angle_deg']:.3f} deg, residual median "
        f"{mat['register']['residual_median_mm']:.3f} mm")

    # ---- 3. partition first, then the density field around it ----------
    t0 = time.time()
    part = npart.partition(st["closed"]["V"], F, h_mat, hc_c, p)
    st["part"] = part
    log(f"  partition A {part['n_A']} / B {part['n_B']} / C {part['n_C']} v, "
        f"width {part['bridge_width_measured_mm']:.1f} mm, B-C "
        f"{len(part['edges_B_C'])} edges [{time.time()-t0:.1f}s]")

    scales = nd.head_scales(st["closed"]["V"], F, st["closed"]["ax"],
                            st["closed"]["curve"], hc_c)
    for k, v in scales.items():
        if k.startswith("_"):
            continue
        log(f"  [{k}] p50 {v['edge_p50_mm']:.2f} mm, {v['faces_per_cm2']:.2f} faces/cm2")
    p["jaw_reference_mm"] = scales["mandible_lateral"]["edge_p50_mm"]
    p["occiput_reference_mm"] = scales["occiput_lower"]["edge_p50_mm"]
    ell_c, zone_c, anchors = nd.ell_field(
        st["closed"]["V"], F, st["closed"]["ax"], st["closed"]["curve"], hc_c,
        scales, p, t_material=h_mat, fixed_labels=part["A"])
    log(f"  target edge: anterior {anchors['ell_anterior_mm']:.2f} / structure "
        f"{anchors['ell_structure_mm']:.2f} / side {anchors['ell_side_mm']:.2f} mm; "
        f"seam inherit {anchors['seam_inherit_mm']:.2f} mm")
    log(f"  zones: " + ", ".join(
        f"z{k} {anchors['zone_vertices'][str(k)]} v"
        for k in range(4) if anchors['zone_vertices'][str(k)]))
    st["ell_input"] = ell_c
    st["zone_input"] = zone_c

    # ---- 4. budgets (three independent zones) --------------------------
    band = ((t_mat > p["data_t"][0]) & (t_mat < p["protect_t"]) &
            (part["B"] | part["C"]))
    bud_by_zone, zone_names = {}, {0: "anterior", 1: "structure", 2: "side_back"}
    for z, name in zone_names.items():
        reg = band & (zone_c == z)
        bud_by_zone[name] = nd.budget(st["closed"]["V"], F, ell_c, reg, p)
        b = bud_by_zone[name]
        log(f"    budget [{name}]: N_est {b['n_estimate']:.0f} -> {b['n_budget']} "
            f"(current {b['n_current']} v, {b['area_mm2']/100:.1f} cm2, ell "
            f"{float(np.median(ell_c[reg])):.2f} mm)")
    bud_by_region = {}
    for k in ("B", "C"):
        reg = part[k] & band
        bud_by_region[k] = nd.budget(st["closed"]["V"], F, ell_c, reg, p)
        log(f"    budget [{k}]: N_est {bud_by_region[k]['n_estimate']:.0f} -> "
            f"{bud_by_region[k]['n_budget']} (current {bud_by_region[k]['n_current']} v)")
    bud = nd.budget(st["closed"]["V"], F, ell_c, band, p)
    cap_new = int(np.ceil(p["max_new_vertex_frac"] * (part["n_B"] + part["n_C"])))
    log(f"  budget total {bud['n_budget']} (band {bud['n_current']} v), "
        f"cumulative new-vertex cap {cap_new} (2% of B+C {part['n_B']+part['n_C']})")

    # ---- 4b. marker landmark sidecar (T3) ------------------------------
    landmark_path = args.landmarks
    if landmark_path is None:
        candidates = [fit_dir / "neck_landmarks.json",
                      fit_dir.parent.parent / "neck_landmarks.json",
                      fit_dir.parent.parent / "prepared" / "neck_landmarks.json"]
        landmark_path = next((path for path in candidates if path.is_file()), None)
    landmark_states: dict[str, dict] = {}
    landmark_info: dict = {"path": None}
    if landmark_path is not None and Path(landmark_path).is_file():
        payload, reason = nl.load_neck_landmarks(landmark_path)
        if payload is not None:
            landmark_states = payload["states"]
            landmark_info = {
                "path": str(landmark_path), "schema": payload.get("schema"),
                "states": {key: {"status": row.get("status"),
                                 "reason": row.get("reason"),
                                 "frame_id": row.get("frame_id"),
                                 "source": row.get("source")}
                           for key, row in landmark_states.items()},
            }
            log(f"  landmark sidecar: {landmark_path} ("
                + ", ".join(f"{key}={row.get('status')}"
                            for key, row in landmark_states.items()) + ")")
        else:
            landmark_info = {"path": str(landmark_path), "error": reason}
            log(f"  landmark sidecar rejected: {reason}")
    else:
        landmark_info = {"path": None, "error": "neck_landmarks_not_found"}
        log("  landmark sidecar: not found (marker-driven centre unavailable)")

    # ---- 5. target fields, each state in its own frame ------------------
    sig = p["structure_sigma_mm"]
    RV, RF, rframe, rax, rhc, menton = ns.reference_frame(args.reference)
    rcurve, _ = ns.spine_curve(RV, RF, rax, rhc)
    ref_field = ns.analyse_surface(RV, RF, rax, rhc, curve=rcurve,
                                   structure_sigma=sig,
                                   fine_sigma=p["fine_sigma_mm"], leak_mask=True)
    anat_by_state: dict[str, dict] = {}
    if p.get("larynx_prior", False):
        anat_by_state = _larynx_anatomy_by_state(st, F, states_raw,
                                                 landmark_states, p)
        for key, a in anat_by_state.items():
            log(f"  larynx {key}: gate t {a['t_gate']:.3f}, centre "
                f"{a['t_center']:.3f} ({a['t_center_source']}), half "
                f"{a['t_half']:.3f}, amp {a['amp_mm']:.2f} mm, d "
                f"{a['d_larynx_mm']:.1f} mm")
    subj = {}
    for key in ("closed", "open"):
        subj[key] = ns.analyse_surface(st[key]["V"], F, st[key]["ax"],
                                       st[key]["h_chin"], curve=st[key]["curve"],
                                       structure_sigma=sig,
                                       fine_sigma=p["fine_sigma_mm"],
                                       leak_mask=True)
        tg = bns.target_field(ref_field, subj[key], p)
        anat_state = anat_by_state.get(key)
        tg = _apply_larynx_gain(tg, p,
                                t_center=(anat_state or {}).get("t_center"))
        if anat_state is not None:
            tg = na.apply_larynx_prior(tg, anat_state)
        st[key]["field"] = ns.Field2D(tg["hs_mm"], tg["theta"], tg["R"])
        st[key]["bg"] = ns.Field2D(tg["hs_mm"], tg["theta"], subj[key]["background"])
        st[key]["target"] = tg
        log(f"  target {key}: grid {tg['R'].shape[0]} x {tg['R'].shape[1]}, "
            f"amplitude {tg['amplitude_mm']:.2f} mm, applied max "
            f"{float(np.max(tg['applied'])):.2f} mm"
            + (f", larynx gain {tg['larynx_gain']:.2f}"
               f" (applied max {tg['larynx_applied_max_mm']:.2f} mm)"
               if abs(float(tg.get("larynx_gain", 1.0)) - 1.0) > 1e-12 else "")
            + (f", larynx prior +/- {tg['larynx_prior_max_mm']:.2f}/"
               f"{tg['larynx_prior_min_mm']:.2f} mm"
               if "larynx_prior" in tg else ""))

    # ---- 6. continuous priors from the geodesic distance to A -----------
    Wgraph, _ = npart.edge_graph(st["closed"]["V"], F)
    d_A = npart.geodesic_from(Wgraph, part["A"])
    d_A = np.where(np.isfinite(d_A), d_A, 0.0)
    blend_lo = float(part["bridge_width_measured_mm"])
    blend_hi = blend_lo + p["seam_prior_edges"] * float(p["occiput_reference_mm"])
    seam_w = nd.srise(d_A, blend_lo, blend_hi)
    mu_t = p["mu_t_bridge"] + (p["mu_t_body"] - p["mu_t_bridge"]) * seam_w
    max_t = p["max_t_bridge_mm"] + (p["max_t_body_mm"] - p["max_t_bridge_mm"]) * seam_w
    # metric taper of the *shaping* weight into the protected head: 0 at A,
    # 1 at 2.5 local edges (12.1 mm here).  This replaces the old fixed "two
    # rings" and is what keeps an A-adjacent face from tilting by 20 deg when
    # its B-side corner moves.
    seam_taper_mm = p["seam_taper_ratio"] * float(p["jaw_reference_mm"])
    w_A = nd.srise(d_A, 0.0, seam_taper_mm)
    log(f"  data taper into A: 0 at the head, 1 at {seam_taper_mm:.1f} mm "
        f"({p['seam_taper_ratio']:.1f} local edges)")
    log(f"  tangential prior: {p['mu_t_bridge']:.3f} at the bridge -> "
        f"{p['mu_t_body']:.3f} in the body (blend {blend_lo:.1f}-{blend_hi:.1f} mm "
        f"from A)")

    ell_reg = _ell_reg_from(st["closed"]["V"], st["closed"]["ax"],
                            st["closed"]["curve"], hc_c, h_mat, p)

    # ---- 7. solve both states on the untouched original topology --------
    setup_s = time.perf_counter() - t_pre
    sol = {}
    for key in ("closed", "open"):
        sol[key] = solve_state(st[key], F, p, h_mat, hc_c, part, ell_reg,
                               mu_t, max_t, distance_ramp=w_A,
                               ledger=solve_ledger, label="initial")
        o = sol[key]
        log(f"  [{key}] unified: {o['n_free']} free / {o['n_fixed']} fixed | "
            f"{o['n_iter']} iters ({o['stop_reason']}) | max |d| "
            f"{o['max_d_mm']:.2f} mm (rad {o['max_d_rad_mm']:.2f}, tan "
            f"{o['max_d_tan_mm']:.2f}) | residual v {o['residual_after']['vertex_rms_mm']:.3f} "
            f"/ q {o['residual_after']['quad_rms_mm']:.3f} mm | "
            f"{o['energy']:.3e}")
    st["mu_t"], st["max_t"], st["d_A"] = mu_t, max_t, d_A
    e_r_in = {k: _input_radial(st[k]) for k in ("closed", "open")}
    conn = {k: nbr.connection_metrics(st[k]["V_input_ext"], st[k]["V"], F, part,
                                      _total_d(st[k]), e_r_in[k])
            for k in ("closed", "open")}
    for k, m in conn.items():
        log(f"  [original topology {k}] crease max B {m['dihedral_change_in_B_deg']:.2f} / "
            f"cross {m['dihedral_change_crossing_deg']:.2f} deg, gradient max "
            f"{m['gradient_in_B_mm_per_mm']:.3f} B / {m['gradient_crossing_mm_per_mm']:.3f} cross")

    # ---- 8. bounded open-state adaptation -------------------------------
    adapt = {"candidates": [], "chosen": None}
    V_open_keep = st["open"]["V"].copy()
    for shift, gain in p["adapt_candidates"]:
        if abs(shift) > p["adapt_shift_limit_mm"] + 1e-9:
            continue
        if (shift, gain) == (0.0, 1.0):
            m = conn["open"]
        else:
            st["open"]["V"] = st["open"]["V_input"].copy()
            fld = bnd.AdaptedField(st["open"]["field"], st["open"]["bg"], shift, gain)
            solve_state(st["open"], F, p, h_mat, hc_c, part, ell_reg, mu_t,
                        max_t, field=fld, distance_ramp=w_A,
                        ledger=solve_ledger, label=f"adapt{shift:+.1f}")
            m = nbr.connection_metrics(st["open"]["V_input_ext"], st["open"]["V"], F,
                                       part, _total_d(st["open"]), e_r_in["open"])
        sharp_ok = (m["dihedral_out_B_p95_deg"]
                    <= m["dihedral_in_B_p95_deg"] + p["adapt_sharpness_margin_deg"]
                    and m["dihedral_out_crossing_p95_deg"]
                    <= m["dihedral_in_crossing_p95_deg"] + p["adapt_sharpness_margin_deg"])
        change_ok = (m["crease_change_crossing_p95_deg"] <= p["adapt_crease_limit_deg"]
                     or sharp_ok)
        full_pos, full_gt10, full_gt20 = _full_positive_stats(st["open"], F)
        full_limit = p.get("adapt_full_positive_limit")
        full_ok = bool(full_limit is None or full_pos <= float(full_limit))
        ok = (change_ok and
              m["gradient_crossing_p95_mm_per_mm"] <= p["adapt_gradient_limit"]
              and m["gradient_in_B_p95_mm_per_mm"] <= p["adapt_gradient_limit"]
              and full_ok)
        rec = {"shift_mm": float(shift), "prominence_gain": float(gain),
               "crease_crossing_p95_deg": m["crease_change_crossing_p95_deg"],
               "crease_B_p95_deg": m["crease_change_B_p95_deg"],
               "crease_crossing_max_deg": m["dihedral_change_crossing_deg"],
               "crease_B_max_deg": m["dihedral_change_in_B_deg"],
               "gradient_crossing_p95_mm_per_mm": m["gradient_crossing_p95_mm_per_mm"],
               "gradient_B_p95_mm_per_mm": m["gradient_in_B_p95_mm_per_mm"],
               "full_positive_max_deg": full_pos,
               "full_positive_gt10": full_gt10, "full_positive_gt20": full_gt20,
               "full_positive_limit": full_limit,
               "sharp_ok": bool(sharp_ok), "pass": bool(ok)}
        adapt["candidates"].append(rec)
        log(f"  adapt {shift:+.1f} mm / gain {gain:.2f}: crease p95 "
            f"{rec['crease_crossing_p95_deg']:.2f} cross / {rec['crease_B_p95_deg']:.2f} B "
            f"(max {rec['crease_B_max_deg']:.2f}), gradient p95 "
            f"{rec['gradient_crossing_p95_mm_per_mm']:.3f}, full sharp "
            f"{full_pos:.2f} -> {'pass' if ok else 'fail'}")
        if ok:
            adapt["chosen"] = rec
            conn["open"] = m
            break
        st["open"]["V"] = V_open_keep.copy()
    if adapt["chosen"] is None:
        adapt["chosen"] = adapt["candidates"][0]
        log("  adapt: no candidate passed; keeping the neutral design and reporting")
    adapt["chosen"]["reason"] = ("neutral design already connects within limits"
                                 if adapt["chosen"]["shift_mm"] == 0.0
                                 and adapt["chosen"]["prominence_gain"] == 1.0
                                 else "neutral did not connect; the smallest bounded "
                                      "adjustment that did was applied")

    # ---- 9. bounded re-spacing, one shared log --------------------------
    journal = {"passes": [], "max_working_vertices": int(len(st["closed"]["V"])),
               "max_working_faces": int(len(F)), "split_new_total": 0,
               "split_new_cap": int(cap_new), "allow_split": bool(not args.no_split)}
    recent = {}
    cap_zone = {z: int(bud_by_zone[zone_names[z]]["n_budget"]) for z in zone_names}
    zone_frozen = zone_c.copy()
    total_splits = 0

    def _crease_stats(states_cur, F_cur):
        """Signed full-edge crease statistics.

        ``max_pos`` is the worst *sharpening* (the surface became sharper than
        the capture); ``max_neg`` is the worst flattening.  The old acceptance
        used ``abs(change)`` and therefore rejected a pass that relieved a 37 deg
        fold (a 26 deg negative change) while accepting new 21 deg folds.  The
        plan only forbids making existing folds sharper, so the rollback uses
        the positive side and the count of positive >10/>20 deg events.
        """
        rows = []
        for key in ("closed", "open"):
            s = states_cur[key]
            _e, aout = nbr.dihedral_deg(np.asarray(s["V"], float) * 1000.0, F_cur)
            _e2, ain = nbr.dihedral_deg(np.asarray(s["V_input_ext"], float) * 1000.0,
                                        F_cur)
            if not len(aout):
                rows.append({"max_pos": 0.0, "max_neg": 0.0,
                             "pos_gt10": 0, "pos_gt20": 0})
                continue
            ch = np.abs(aout) - np.abs(ain)
            rows.append({"max_pos": float(ch.max()), "max_neg": float(ch.min()),
                         "pos_gt10": int((ch > 10.0).sum()),
                         "pos_gt20": int((ch > 20.0).sum())})
        return rows

    for k in range(p["remesh_passes"]):
        mk = _Marks()
        # A pass is a proposal: if the delivered crease gets worse, the whole
        # pass is rolled back.  This is what keeps a merge/flip/relax from
        # trading density for a fold that the next solve then amplifies.
        snap = {"F": F.copy(), "part": copy.deepcopy(part), "h_mat": h_mat.copy(),
                "t_mat": np.asarray(t_mat, float).copy(),
                "zone_frozen": zone_frozen.copy(), "ell_reg": ell_reg.copy(),
                "d_A": d_A.copy(), "mu_t": mu_t.copy(), "max_t": max_t.copy(),
                "w_A": w_A.copy(), "total_splits": total_splits, "recent": dict(recent),
                "V": {key: st[key]["V"].copy() for key in ("closed", "open")},
                "V_input_ext": {key: st[key]["V_input_ext"].copy()
                                for key in ("closed", "open")},
                "solve": {key: copy.deepcopy(st[key].get("solve"))
                          for key in ("closed", "open")},
                "masks": {key: copy.deepcopy(st[key].get("masks"))
                          for key in ("closed", "open")},
                "field_used": {key: st[key].get("field_used")
                               for key in ("closed", "open")}}
        mk("snapshot")
        pre_stats = _crease_stats(st, F)
        pre_pos = max(r["max_pos"] for r in pre_stats)
        pre_gt10 = sum(r["pos_gt10"] for r in pre_stats)
        pre_gt20 = sum(r["pos_gt20"] for r in pre_stats)
        mk("crease_pre")
        m_c = bnd.state_masks(st["closed"], F, p, h_mat, hc_c, part)
        m_o = bnd.state_masks(st["open"], F, p, h_mat, hc_c, part)
        free = (np.asarray(part["B"], bool) | np.asarray(part["C"], bool)) & \
               (~m_c["fixed"]) & (~m_o["fixed"])
        alive = np.zeros(len(st["closed"]["V"]), bool)
        alive[np.unique(F)] = True
        mk("masks_free")
        ell_k, zone_k, _ = nd.ell_field(st["closed"]["V"], F, st["closed"]["ax"],
                                        st["closed"]["curve"], hc_c, scales, p,
                                        t_material=h_mat, fixed_labels=part["A"])
        n_k = len(zone_k)
        if len(zone_frozen) >= n_k:
            zone_k = zone_frozen[:n_k].copy()
        else:
            zone_k[:len(zone_frozen)] = zone_frozen
        room_s, room_c = {}, {}
        band_k = ((t_mat > p["data_t"][0]) & (t_mat < p["protect_t"]) &
                  (part["B"] | part["C"]))
        for z in zone_names:
            inz = int(((zone_k == z) & band_k & alive).sum())
            room_s[z] = max(0, cap_zone[z] - inz)
            room_c[z] = max(0, inz - cap_zone[z])
        if merge_zones is not None:
            for z, name in zone_names.items():
                if name not in merge_zones:
                    room_c[z] = 0
        room = {"split": int(sum(room_s.values())),
                "collapse": int(sum(room_c.values())),
                "split_by_zone": room_s, "collapse_by_zone": room_c}
        mk("ell_field_room")
        E, nrm, _per = nr._norm_lengths({"closed": st["closed"], "open": st["open"]},
                                        F, free, ell_k)
        n_cand_split = int(((nrm > p["split_ratio"]) &
                            free[np.asarray(E)[:, 0]] & free[np.asarray(E)[:, 1]]).sum())
        n_cand_collapse = int(((nrm < p["collapse_ratio"]) &
                               free[np.asarray(E)[:, 0]] & free[np.asarray(E)[:, 1]]).sum())
        not_b = ~np.asarray(part["B"], bool)
        depth_b = nr.ring_depth(F, not_b, p["merge_skip_rings"] + 1)
        collapse_ok = not_b & (depth_b > p["merge_skip_rings"])
        mk("candidates")
        F, ops, ext = nr.respace({"closed": st["closed"], "open": st["open"]}, F, free,
                                 ell_k, p, k, recent, room, zone=zone_k,
                                 collapse_ok=collapse_ok, eligible=band_k,
                                 max_new_vertices=cap_new,
                                 new_vertices_so_far=total_splits,
                                 allow_split=not args.no_split)
        mk("respace")
        if ops["split"]:
            part, h_mat = npart.extend_partition(part, ops, F, h_mat, vlist=h_mat,
                                                 extra_scalars=("d_head", "d_neck", "score"))
            st["part"] = part
            # the material height array grows with the split; recompute t so the
            # next pass's band/zone masks have the same length as the mesh
            t_mat = np.asarray(h_mat, float) / hc_c
            band_k = np.concatenate([band_k, band_k[[q["edge"][0] for q in ops["split"]]]])
            for key2 in ("closed", "open"):
                Vi = st[key2]["V_input_ext"]
                a2 = Vi[[q["edge"][0] for q in ops["split"]]]
                b2 = Vi[[q["edge"][1] for q in ops["split"]]]
                st[key2]["V_input_ext"] = np.vstack([Vi, 0.5 * (a2 + b2)])
            zone_frozen = np.concatenate([zone_frozen,
                                          np.asarray([zone_k[q["edge"][0]]
                                                      for q in ops["split"]], np.int8)])
            total_splits += len(ops["split"])
            # the regularisation length is a material property: extend it too
            ell_reg = np.concatenate([ell_reg, np.asarray(
                [0.5 * (ell_reg[q["edge"][0]] + ell_reg[q["edge"][1]])
                 for q in ops["split"]], float)])
            d_A = np.concatenate([d_A, np.asarray(
                [0.5 * (d_A[q["edge"][0]] + d_A[q["edge"][1]])
                 for q in ops["split"]], float)])
            seam_w = nd.srise(d_A, blend_lo, blend_hi)
            mu_t = p["mu_t_bridge"] + (p["mu_t_body"] - p["mu_t_bridge"]) * seam_w
            max_t = p["max_t_bridge_mm"] + (p["max_t_body_mm"] -
                                            p["max_t_bridge_mm"]) * seam_w
            w_A = nd.srise(d_A, 0.0, seam_taper_mm)
        mk("extend_partition")
        did_ops = bool(ops["split"] or ops["collapse"] or ops["flip"])
        if did_ops:
            # one solve from the capture baseline on the new topology.  Solving
            # again on an already displaced surface accumulated displacements
            # and created spurious 10+ deg crease events even when a pass did
            # nothing; the input baseline is the honest reference.
            for key in ("closed", "open"):
                st[key]["V"] = st[key]["V_input_ext"].copy()
        free_new = np.asarray(ext["free"], bool)
        rel = nr.relax({"closed": st["closed"], "open": st["open"]}, F,
                       free_new & np.asarray(part["C"], bool) & band_k, p,
                       ell_v=ext["ell_v"])
        moved = rel["max_move_mm"] >= 0.01
        mk("relax")
        if did_ops or moved:
            for key in ("closed", "open"):
                solve_state(st[key], F, p, h_mat, hc_c, part, ell_reg, mu_t, max_t,
                            distance_ramp=w_A, ledger=solve_ledger,
                            label=f"remesh{k}")
        else:
            # nothing changed: restore the previous solved state and stop
            F = snap["F"].copy()
            part = copy.deepcopy(snap["part"]); st["part"] = part
            h_mat = snap["h_mat"].copy()
            t_mat = snap["t_mat"].copy()
            zone_frozen = snap["zone_frozen"].copy()
            ell_reg = snap["ell_reg"].copy()
            d_A, mu_t, max_t = snap["d_A"].copy(), snap["mu_t"].copy(), snap["max_t"].copy()
            w_A = snap["w_A"].copy()
            total_splits = int(snap["total_splits"])
            recent = dict(snap["recent"])
            for key in ("closed", "open"):
                st[key]["V"] = snap["V"][key].copy()
                st[key]["V_input_ext"] = snap["V_input_ext"][key].copy()
                st[key]["solve"] = copy.deepcopy(snap["solve"][key])
                st[key]["masks"] = copy.deepcopy(snap["masks"][key])
                st[key]["field_used"] = snap["field_used"][key]
            rel = {"max_move_mm": 0.0}
        mk("solve_two_states")
        post_stats = _crease_stats(st, F)
        post_pos = max(r["max_pos"] for r in post_stats)
        post_gt10 = sum(r["pos_gt10"] for r in post_stats)
        post_gt20 = sum(r["pos_gt20"] for r in post_stats)
        mk("crease_post")
        rec = {"pass": k, "free": int(free.sum()), "room_split": int(room["split"]),
               "room_by_zone": room_s, "cap_by_zone": cap_zone,
               "candidate_split": n_cand_split, "candidate_collapse": n_cand_collapse,
               "split": len(ops["split"]), "collapse": len(ops["collapse"]),
               "flip": len(ops["flip"]), "relax_max_move_mm": rel["max_move_mm"],
               "chord_mm_median": (float(np.median([q["chord_mm"] for q in ops["split"]]))
                                   if ops["split"] else None),
               "vertices": int(len(st["closed"]["V"])), "faces": int(len(F)),
               "split_new_total": int(total_splits),
               "crease_before_deg": float(pre_pos),
               "crease_after_deg": float(post_pos),
               "gt10_before": int(pre_gt10), "gt10_after": int(post_gt10),
               "gt20_before": int(pre_gt20), "gt20_after": int(post_gt20),
               "stats": {"before": pre_stats, "after": post_stats}, "ops": ops}
        bad = (post_pos > pre_pos + p["pass_rollback_deg"] or
               post_pos > p["crease_hard_cap_deg"] or
               post_gt10 > pre_gt10 + p["crease_count_margin"] or
               post_gt20 > pre_gt20 + p["crease_gt20_margin"])
        rec["rolled_back"] = bool(bad)
        if bad:
            F = snap["F"].copy()
            part = copy.deepcopy(snap["part"]); st["part"] = part
            h_mat = snap["h_mat"].copy()
            t_mat = snap["t_mat"].copy()
            zone_frozen = snap["zone_frozen"].copy()
            ell_reg = snap["ell_reg"].copy()
            d_A, mu_t, max_t = snap["d_A"].copy(), snap["mu_t"].copy(), snap["max_t"].copy()
            w_A = snap["w_A"].copy()
            total_splits = int(snap["total_splits"])
            recent = dict(snap["recent"])
            for key in ("closed", "open"):
                st[key]["V"] = snap["V"][key].copy()
                st[key]["V_input_ext"] = snap["V_input_ext"][key].copy()
                st[key]["solve"] = copy.deepcopy(snap["solve"][key])
                st[key]["masks"] = copy.deepcopy(snap["masks"][key])
                st[key]["field_used"] = snap["field_used"][key]
        mk("rollback")
        rec["timing"] = dict(mk.out)
        journal["passes"].append(rec)
        log(f"  remesh {k}: free {rec['free']} / room +{rec['room_split']} "
            f"{ {zone_names[z]: room_s[z] for z in zone_names} } | cand "
            f"{n_cand_split}/{n_cand_collapse} -> {rec['split']}/{rec['collapse']}/"
            f"{rec['flip']} ops | {rec['vertices']} v / {rec['faces']} f | "
            f"relax {rel['max_move_mm']:.2f} mm | new {total_splits}/{cap_new} | "
            f"sharp {pre_pos:.2f}->{post_pos:.2f}, >10 {pre_gt10}->{post_gt10}, "
            f">20 {pre_gt20}->{post_gt20}"
            + (" ROLLED BACK" if bad else ""))
        if ops["rejected"]:
            worst = max(ops["rejected"], key=lambda q: q["chord_mm"])
            log(f"    rejected {len(ops['rejected'])} split(s) off-surface, largest "
                f"{worst['chord_mm']:.2f} mm over {worst['edge_len_mm']:.2f} mm")
        journal["max_working_vertices"] = max(journal["max_working_vertices"],
                                              rec["vertices"])
        journal["max_working_faces"] = max(journal["max_working_faces"], rec["faces"])
        if bad:
            break
        if not (rec["split"] or rec["collapse"] or rec["flip"]):
            break
    journal["split_new_total"] = int(total_splits)
    t_accept = time.perf_counter()

    # ---- 10. acceptance data (the mesh is already solved) --------------
    finals = {key: st[key]["solve"] for key in ("closed", "open")}
    for key in ("closed", "open"):
        log(f"  [{key} delivered] {finals[key]['n_iter']} iters "
            f"({finals[key]['stop_reason']}), last increment max |d| "
            f"{finals[key]['max_d_mm']:.2f} mm")
    e_r_in = {k: _input_radial(st[k]) for k in ("closed", "open")}
    conn = {k: nbr.connection_metrics(st[k]["V_input_ext"], st[k]["V"], F, part,
                                      _total_d(st[k]), e_r_in[k])
            for k in ("closed", "open")}
    for k, m in conn.items():
        log(f"  [final {k}] crease max B {m['dihedral_change_in_B_deg']:.2f} / cross "
            f"{m['dihedral_change_crossing_deg']:.2f} deg, >10 B "
            f"{m['n_edges_crease_gt10_B']} / cross {m['n_edges_crease_gt10_crossing']}")
    desc = {}
    for key in ("closed", "open"):
        a = _structure_side(st[key], F, p)
        desc[key] = ns.structure_descriptors(a, f"direct v2 {key}",
                                             t_hi=0.90 if key == "closed" else 0.84)
    desc["reference"] = ns.structure_descriptors(ref_field, "reference")
    for key in ("closed", "open"):
        q, qr = desc[key]["primary"], desc["reference"]["primary"]
        log(f"  [{key}] peak {q['relief_mm']:.2f} mm at t={q['t']:.3f} "
            f"(ref {qr['relief_mm']:.2f} at {qr['t']:.3f}), theta "
            f"{q['width_theta_deg']:.0f} deg, axial {q['width_t_mm']:.0f} mm")

    # ---- 11. export -----------------------------------------------------
    used = np.unique(F)
    remap = np.full(len(st["closed"]["V"]), -1, np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    Fc = remap[F]
    obj = {}
    for key in ("closed", "open"):
        obj[key] = out / f"head_{key}_direct.obj"
        ns.write_obj(obj[key], st[key]["V"][used], Fc,
                     comment=f"neck direct v2 {key}; {len(used)} v / {len(Fc)} f, "
                             f"unified 3-D displacement, material A/B/C, shared topology")
    journal.update({"vertices_original": int(n0), "faces_original": int(len(F0)),
                    "vertices_allocated": int(len(st["closed"]["V"])),
                    "vertices_final": int(len(used)),
                    "vertices_used": int(len(used)),
                    "faces_final": int(len(Fc)),
                    "merged_slots": int(len(st["closed"]["V"]) - len(used)),
                    "n_budget": int(bud["n_budget"]), "slots_alive": int(len(used))})
    topo0 = mi.topology_report(states_raw["closed"]["V"], F0)
    topo1 = mi.topology_report(st["closed"]["V"][used], Fc)
    keys_t = ("nonmanifold_edges", "inconsistent_shared_edges", "boundary_edges",
              "zero_area_faces", "components")
    journal["topology_input"] = {k: topo0[k] for k in keys_t}
    journal["topology_output"] = {k: topo1[k] for k in keys_t}
    log(f"  topology: nonmanifold {topo0['nonmanifold_edges']}->{topo1['nonmanifold_edges']}, "
        f"winding {topo0['inconsistent_shared_edges']}->{topo1['inconsistent_shared_edges']}, "
        f"boundary {topo0['boundary_edges']}->{topo1['boundary_edges']}, "
        f"components {topo0['components']}->{topo1['components']}")

    nV = len(st["closed"]["V"])
    part_e = {k: np.asarray(part[k], bool)[used] for k in ("A", "B", "C")}
    zone_e = np.asarray(zone_frozen, np.int8)
    zone_e = np.concatenate([zone_e, np.full(max(0, nV - len(zone_e)), 2, np.int8)])[used]
    disp, disp_rad = {}, {}
    for key in ("closed", "open"):
        tot = (st[key]["V"] - st[key]["V_input_ext"]) * 1000.0
        disp[key] = tot
        disp_rad[key] = np.einsum("nk,nk->n", tot, e_r_in[key])
    npz = {"F": Fc, "V_closed": st["closed"]["V"][used],
           "V_open": st["open"]["V"][used],
           "V_closed_input": st["closed"]["V_input_ext"][used],
           "V_open_input": st["open"]["V_input_ext"][used],
           "d_closed": disp["closed"][used], "d_open": disp["open"][used],
           "d_rad_closed": disp_rad["closed"][used],
           "d_rad_open": disp_rad["open"][used],
           "e_r_closed": e_r_in["closed"][used], "e_r_open": e_r_in["open"][used],
           "h_material": h_mat[used],
           "region": np.where(part_e["A"], 0, np.where(part_e["B"], 1, 2)).astype(np.int8),
           "density_zone": zone_e,
           "orig_id": used.astype(np.int64),
           "R_target_closed": st["closed"]["target"]["R"],
           "R_target_open": st["open"]["target"]["R"],
           "applied_closed": st["closed"]["target"]["applied"],
           "applied_open": st["open"]["target"]["applied"],
           "background_closed": subj["closed"]["background"],
           "background_open": subj["open"]["background"],
           "theta": st["closed"]["target"]["theta"],
           "t": st["closed"]["target"]["t"],
           "hs_mm": st["closed"]["target"]["hs_mm"],
           "mu_t": np.asarray(mu_t, float)[used], "max_t": np.asarray(max_t, float)[used],
           "d_A": np.asarray(d_A, float)[used],
           "gate_closed": np.asarray(st["closed"]["masks"]["gate"], float)[used],
           "gate_open": np.asarray(st["open"]["masks"]["gate"], float)[used],
           "data_closed": np.asarray(st["closed"]["masks"]["data"], float)[used],
           "data_open": np.asarray(st["open"]["masks"]["data"], float)[used]}
    if "larynx_prior" in st["closed"]["target"]:
        npz["larynx_prior_closed"] = st["closed"]["target"]["larynx_prior"]
        npz["larynx_prior_open"] = st["open"]["target"]["larynx_prior"]
    np.savez_compressed(out / "stage_direct.npz", **npz)
    meta = {"params": p, "provenance": provenance, "scales": scales,
            "larynx_anatomy": anat_by_state,
            "landmark_sidecar": landmark_info,
            "reference": {"path": args.reference, "vertices": int(len(RV)),
                          "faces": int(len(RF)),
                          "topology_used": False, "density_used": False,
                          "usage": "2-D relief field only (outline, prominence, "
                                   "lower broad lobe, shallow transition, lateral "
                                   "relief, submental connection); no vertex "
                                   "correspondence and no edge-length inheritance"},
            "anchors": anchors, "budget": bud, "budget_by_region": bud_by_region,
            "budget_by_zone": bud_by_zone, "zone_cap": cap_zone,
            "structure": desc, "journal": journal, "h_chin": hc_c, "h_chin_open": hc_o,
            "material": mat,
            "partition": {k: v for k, v in part.items()
                          if k not in ("A", "B", "C", "d_head", "d_neck", "score",
                                       "edges_A_B", "edges_B_C", "faces_A_B",
                                       "faces_B_C", "A_idx", "B_idx", "C_idx",
                                       "h_material")},
            "connection": conn, "adapt": adapt,
            "solve_closed": {k: v for k, v in finals["closed"].items()
                             if k not in ("d_mm", "e_r", "V_mm", "iterations")},
            "solve_open": {k: v for k, v in finals["open"].items()
                           if k not in ("d_mm", "e_r", "V_mm", "iterations")},
            "iterations": {"closed": len(finals["closed"]["iterations"]),
                           "open": len(finals["open"]["iterations"])},
            "priors": {"mu_t_bridge": p["mu_t_bridge"],
                       "mu_t_body": p["mu_t_body"],
                       "blend_mm": [blend_lo, blend_hi],
                       "max_t_bridge_mm": p["max_t_bridge_mm"],
                       "max_t_body_mm": p["max_t_body_mm"]},
            "bridge_ring_check": {
                "c_ring_1": int(nbr._rings(F, np.asarray(part["B"], bool), 1,
                                           np.asarray(part["C"], bool)).sum()),
                "c_ring_3": int(nbr._rings(F, np.asarray(part["B"], bool), 3,
                                           np.asarray(part["C"], bool)).sum()),
                "note": "the previous implementation intersected the seeds with "
                        "C first and returned 0; this is the non-empty ring the "
                        "unified solve uses for the prior blend",
            },
            "timing": {
                "input_hash": input_hash,
                "setup_s": round(setup_s, 6),
                "n_solves": len(solve_ledger),
                "solve_total_s": round(sum(r["wall_s"] for r in solve_ledger), 6),
                "solve_ledger": solve_ledger,
                "remesh_pass_s": round(sum(sum(q["timing"].values())
                                           for q in journal["passes"]), 6),
                "acceptance_export_s": round(time.perf_counter() - t_accept, 6),
                "note": ("record-only A0 account; solve_ledger covers EVERY solve "
                         "of the run (initial + adapt + each remesh pass, both "
                         "states), not just the delivered one. setup_s is "
                         "material/partition/density/target-field construction."),
            },
            "seconds": time.time() - t_start}
    json.dump(meta, open(out / "build_direct.json", "w", encoding="utf-8"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))

    # ---- 11b. record-only loss trace (never used by the solve) ----------
    # The builder already keeps the per-iteration energy in memory; persisting
    # it changes nothing in the algorithm.  ``loss_direct.json`` is the same
    # trace the analysis tool reads, so the offline optimizer can compare a
    # candidate against the baseline without instrumenting the solver.
    loss = {
        "schema": "neck_direct.loss.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "out": str(out),
        "inputs": {"fit": str(fit_dir),
                   "landmarks": (str(args.landmarks) if args.landmarks else None),
                   "reference": str(args.reference)},
        "params": {key: p.get(key) for key in (
            "structure_gain", "bridge_width_mm", "seam_taper_ratio",
            "mu_t_bridge", "mu_t_body", "anchor_near_A", "larynx_gain",
            "larynx_prior", "larynx_prior_ratio", "fit_max_iter",
            "remesh_passes", "anchor_strength", "max_d_mm",
        )},
        "tier_note": ("energy/energy_parts are the unified solver objective; "
                      "residual_after is the data residual recomputed at the end "
                      "of the same solve and is a diagnostic, not a second "
                      "objective; remesh passes are geometry proposals, not "
                      "optimisation steps"),
        "states": {},
        "remesh": {
            "passes": journal.get("passes", []),
            "split_new_total": journal.get("split_new_total"),
            "max_working_vertices": journal.get("max_working_vertices"),
            "max_working_faces": journal.get("max_working_faces"),
            "topology_input": journal.get("topology_input"),
            "topology_output": journal.get("topology_output"),
        },
    }
    for key in ("closed", "open"):
        s = finals[key]
        loss["states"][key] = {
            "iterations": s.get("iterations", []),
            "n_iter": s.get("n_iter"),
            "converged": s.get("converged"),
            "stop_reason": s.get("stop_reason"),
            "energy": s.get("energy"),
            "energy_parts": s.get("energy_parts"),
            "residual_after": s.get("residual_after"),
            "data_samples": s.get("data_samples"),
            "max_d_mm": s.get("max_d_mm"),
            "max_d_rad_mm": s.get("max_d_rad_mm"),
            "max_d_tan_mm": s.get("max_d_tan_mm"),
            "rms_d_mm": s.get("rms_d_mm"),
            "n_free": s.get("n_free"),
            "n_fixed": s.get("n_fixed"),
        }
    json.dump(loss, open(out / "loss_direct.json", "w", encoding="utf-8"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    log(f"exported in {time.time()-t_start:.1f}s -> {out}")

    # ---- 12. renders ----------------------------------------------------
    if not args.no_render:
        try:
            from tools import neck_direct_render_v2 as ndr2
            rnd = ndr2.render_all(out)
            log(f"renders -> {rnd}")
        except Exception as exc:
            log(f"render failed ({type(exc).__name__}: {exc}); meshes are exported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())