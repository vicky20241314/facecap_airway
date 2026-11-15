"""Direct neck shaping on the ORIGINAL shared topology.

This module implements the second-generation neck stage: no high-density working
mesh, no "subdivide first, decimate later".  Everything is defined as

  * a continuous target surface ``R(theta, h)`` sampled on a 2-D grid that is
    independent of the delivered vertex count;
  * a target edge-length field ``ell(p)`` derived from the head's own measured
    resolution, which fixes a *vertex budget* before any topology changes;
  * an area-weighted least-squares fit of the existing vertices, with residual
    samples at vertices, edge midpoints and face centroids (numerical quadrature,
    not new vertices);
  * a physically scaled bending regulariser ``D^T L^T M^-1 L D`` (cotangent
    stiffness + vertex-area mass) instead of adjacency smoothing iterations;
  * a bounded remeshing pass (split / collapse / flip / tangential relax) whose
    every operation is recorded so the other state can replay it by index.

Only ``numpy``/``scipy`` are used; the heavy geometry helpers of
``head_rgbd.neck_stage`` are reused rather than duplicated.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from head_rgbd import mesh_integrity as mi
from head_rgbd import neck_volume as nv

# --------------------------------------------------------------------------
# frames and region definitions
# --------------------------------------------------------------------------


def axial(V, ax, curve, h_chin):
    """h (mm), theta (rad), radial offset from the neck spine, and t."""
    h, a, b = nv.axial_coords(V, ax)
    ca, cb = curve(h)
    da, db = a - ca, b - cb
    return h, np.arctan2(da, db), np.hypot(da, db), h / h_chin


def ramp(x, lo, hi):
    """Linear 0 -> 1 ramp, clamped; the one blending primitive used everywhere."""
    return np.clip((np.asarray(x, float) - lo) / max(float(hi) - float(lo), 1e-12),
                   0.0, 1.0)


def bump(x, lo, hi, power=1.0):
    """Compactly supported C1 bump: 1 at ``lo``, exactly 0 at and beyond ``hi``.

    ``0.5 (1 + cos(pi u))`` with ``u = clamp((x-lo)/(hi-lo), 0, 1)``.  Both the
    value and the first derivative vanish at ``hi``, so a weight built from it
    has *finite* support and joins the surrounding field without a kink.  This
    replaces the broad Gaussian that used to leak over the whole neck.
    """
    u = np.clip((np.asarray(x, float) - lo) / max(float(hi) - float(lo), 1e-12), 0.0, 1.0)
    w = 0.5 * (1.0 + np.cos(np.pi * u))
    return w ** float(power) if power != 1.0 else w


def cband(x, width, power=1.0):
    """Compactly supported C1 bump around 0: 1 at ``x = 0``, 0 for ``|x| >= width``."""
    return bump(np.abs(np.asarray(x, float)), 0.0, float(width), power)


def srise(x, lo, hi, power=1.0):
    """C1 smooth step 0 -> 1: exactly 0 at ``lo``, exactly 1 at and beyond ``hi``."""
    return 1.0 - bump(x, lo, hi, power)


def sfall(x, lo, hi, power=1.0):
    """C1 smooth step 1 -> 0: exactly 1 at ``lo``, exactly 0 at and beyond ``hi``."""
    return bump(x, lo, hi, power)


def _edge_table(F):
    E = mi.edge_records(F)[0]
    return E


def _region_stats(V, F, mask_v, label):
    """Median edge length and face count per mm^2 for edges inside a region."""
    E = _edge_table(F)
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) * 1000.0
    keep = mask_v[E[:, 0]] & mask_v[E[:, 1]]
    if keep.sum() < 12:
        return {"label": label, "vertices": int(mask_v.sum()), "edges": int(keep.sum()),
                "edge_p50_mm": None, "faces_per_cm2": None}
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                axis=1) * 1e6          # mm^2
    fm = mask_v[F].all(axis=1)
    dens = float(fm.sum()) / max(float(area[fm].sum()), 1e-9) * 100.0   # per cm^2
    return {"label": label, "vertices": int(mask_v.sum()), "edges": int(keep.sum()),
            "edge_p50_mm": float(np.median(L[keep])), "faces_per_cm2": dens}


def head_scales(V, F, ax, curve, h_chin):
    """Measure the head's own resolution in four named regions.

    Region definitions (design frame, t = h / h_chin, theta = 0 anterior):

    ``mandible_lateral``  t 1.16-1.32, |theta| 20-50 deg, lateral normal -- the
                          side of the mandible/cheek; this is the reference the
                          brief calls "相邻下颌侧边".
    ``occiput_lower``     t 1.00-1.50, |theta| > 150 deg -- lower back of head.
    ``neck_body``         t 0.15-0.95, all theta -- the neck as captured.
    ``neck_front``        t 0.30-0.90, |theta| < 40 deg -- anterior neck.

    Eyes, nose, mouth and ear fall outside these windows by construction
    (|theta| < 20 deg above t = 1.16 is the mouth/chin, |theta| > 55 deg above
    t = 1.16 and t > 1.30 is the ear), and degenerate faces never enter an edge
    median.
    """
    from head_rgbd import neck_prominence as npn

    h, th, r, t = axial(V, ax, curve, h_chin)
    thd = np.degrees(th)
    n = npn.vertex_normals(V, F)
    lat = np.abs(n @ ax["lateral"])
    up = n @ ax["axis"]
    # the ear is the only place where |theta| 55-105 at t > 1.15 is dense; leave it out
    ear = (t > 1.15) & (np.abs(thd) > 55) & (np.abs(thd) < 105)
    regs = {
        "mandible_lateral": (t > 1.18) & (t < 1.36) & (np.abs(thd) > 18) & (np.abs(thd) < 52)
                            & (lat > 0.45) & (~ear),
        "occiput_lower": (t > 1.00) & (t < 1.50) & (np.abs(thd) > 150),
        "neck_body": (t > 0.15) & (t < 0.95),
        "neck_front": (t > 0.30) & (t < 0.90) & (np.abs(thd) < 40),
        "shoulder": (t > -0.15) & (t < 0.12),
    }
    out = {k: _region_stats(V, F, m, k) for k, m in regs.items()}
    out["_notes"] = {"up_dot_median": float(np.median(up)),
                     "frame": "design frame of this state"}
    return out


# --------------------------------------------------------------------------
# target edge length field and vertex budget
# --------------------------------------------------------------------------


def ell_field(V, F, ax, curve, h_chin, scales, p, t_material=None,
              fixed_labels=None):
    """Smooth target edge length in mm, per vertex.

    The three anchors are ratios of the measured head resolution:

      anterior neck      1.45 x mandible lateral  (~7.0 mm)
      structure zone     1.25 x mandible lateral  (~6.1 mm)
      sides / back       measured lower occiput   (~7.9 mm)

    The interface policy is the one the previous round got wrong.  A fixed seam
    may inherit the tessellation that is already there, but only **the seam
    itself** -- a band of one local edge length (``seam_inherit_ratio`` x the
    measured jaw reference, ~4.9 mm).  Further into the neck the target returns
    to the design value, so an inherited dense band (this subject: p50 4.44 mm at
    t = 0.925-0.97) is not preserved over its whole height.  The previous code
    blended to ``local`` over ``interface_t`` = 0.86-0.98 (~12 mm), which turned
    the entire upper band into "keep the input edge length" and made the dense
    band permanent by construction.

    ``fixed_labels`` -- when given, the protected/head label (A).  The distance
    to it is a geodesic distance on the mesh, so the seam band follows the
    surface (and the submental fold) instead of a horizontal height cut.  With
    ``fixed_labels=None`` the old height-window behaviour is kept for the
    historical build tool.

    ``t_material`` -- when given -- is the shared neutral axial coordinate
    (one value per index); the density then selects the same material band in
    both states instead of following each state's own moved menton.

    **Where the field is valid.**  The whole construction rests on ``R(theta,h)``
    being the surface, which holds only where a cross-section is star-shaped
    about the neck axis -- the material band ``data_t[0] .. protect_t``.  The
    density target is a *layout* target, not a data target: it is defined up to
    the protected boundary and hands over to the input edge length only in the
    seam band.
    """
    jaw = scales["mandible_lateral"]["edge_p50_mm"]
    occ = scales["occiput_lower"]["edge_p50_mm"]
    if jaw is None or occ is None:
        raise RuntimeError("cannot measure head scales")
    ell_ant = p["ell_front_ratio"] * jaw
    ell_struct = p["ell_structure_ratio"] * jaw
    ell_side = p["ell_side_ratio"] * occ

    h, th, r, t_state = axial(V, ax, curve, h_chin)
    t = t_state if t_material is None else np.asarray(t_material, float) / float(h_chin)
    thd = np.degrees(np.abs(th))
    E = _edge_table(F)
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) * 1000.0
    # local current edge length (median over incident edges) -- the interface
    # and the root keep the tessellation the head already has
    acc = np.zeros(len(V))
    cnt = np.zeros(len(V))
    np.add.at(acc, E[:, 0], L)
    np.add.at(acc, E[:, 1], L)
    np.add.at(cnt, E[:, 0], 1.0)
    np.add.at(cnt, E[:, 1], 1.0)
    local = np.divide(acc, np.maximum(cnt, 1.0))

    band = ramp(t, 0.10, 0.22) * (1.0 - ramp(t, 0.86, 0.98))
    # anterior neck: compact in theta, flat in t inside the band
    w_ant = bump(thd, p["theta_ant_full"], p["theta_ant_zero"]) * band
    ell = ell_side + (ell_ant - ell_side) * w_ant

    # structure zone: compact in t *and* theta, so it cannot reach the interface
    w_struct = (cband(t - p["struct_t"], p["struct_t_radius_mm"] / float(h_chin),
                      p["struct_comp"]) *
                bump(thd, p["theta_struct_full"], p["theta_struct_zero"],
                     p["struct_comp"]))
    w_struct = w_struct * band
    ell = ell + (ell_struct - ell) * w_struct

    if fixed_labels is None:
        # historical behaviour (the old build tool); kept for reproduction
        w_far = 1.0 - srise(t, 0.28, float(p["data_t"][0])) * \
            sfall(t, float(p["interface_t"][0]), float(p["interface_t"][1]))
        w_if, w_root = w_far, w_far
        ell = ell + (local - ell) * w_far
        handover = w_far > 0.5
        band_top = float(p["interface_t"][0])
    else:
        # seam band: one local edge length into the editable side, measured
        # along the surface from the protected label A
        from head_rgbd import neck_partition as npart
        W, _ = npart.edge_graph(V, F)
        d_a = npart.geodesic_from(W, np.asarray(fixed_labels, bool))
        seam_mm = float(p.get("seam_inherit_ratio", 1.0)) * float(jaw)
        w_near = 1.0 - srise(d_a, 0.0, seam_mm)
        w_near = np.where(np.asarray(fixed_labels, bool), 1.0, w_near)
        # root hand-over: below the neck body the cross-section is the shoulder
        w_root = 1.0 - srise(t, 0.24, 0.34)
        ell = ell + (local - ell) * w_near
        ell = ell + (local - ell) * w_root
        handover = (w_near > 0.5) | (w_root > 0.5)
        band_top = float(p["protect_t"])
        w_if = np.maximum(w_near, w_root)

    ell = np.clip(ell, p["ell_min_mm"], p["ell_max_mm"])

    # density zones: three independent area budgets, not one global number.
    # 0 anterior, 1 structure, 2 sides/back, 3 hand-over/seam/root (no budget)
    zones = np.full(len(V), 2, np.int8)
    zones[w_ant > 0.5] = 0
    zones[w_struct > 0.5] = 1
    zones[handover] = 3
    return ell, zones, {"ell_anterior_mm": float(ell_ant),
                 "ell_structure_mm": float(ell_struct),
                 "ell_side_mm": float(ell_side), "jaw_reference_mm": float(jaw),
                 "occiput_reference_mm": float(occ),
                 "support_t_mm": float(p["struct_t_radius_mm"]),
                 "support_theta_deg": float(p["theta_struct_zero"]),
                 "anterior_theta_deg": [float(p["theta_ant_full"]),
                                        float(p["theta_ant_zero"])],
                 "struct_weight_mean": float(np.mean(w_struct)),
                 "material_band_t": [float(p["data_t"][0]), band_top],
                 "seam_inherit_mm": (None if fixed_labels is None else
                                     float(p.get("seam_inherit_ratio", 1.0)) * float(jaw)),
                 "zone_vertices": {str(k): int((zones == k).sum()) for k in range(4)},
                 "zone_ell_median_mm": {str(k): (float(np.median(ell[zones == k]))
                                                 if (zones == k).any() else None)
                                        for k in range(3)}}


def vertex_areas(V, F):
    """Barycentric (one third of the incident face areas) vertex areas in mm^2."""
    tri = V[F]
    a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                             axis=1) * 1e6
    out = np.zeros(len(V))
    for k in range(3):
        np.add.at(out, F[:, k], a / 3.0)
    return out, a


def budget(V, F, ell, editable, p):
    """N_est = integral 2 / (sqrt(3) ell^2) dA, plus the requested margin."""
    A, _ = vertex_areas(V, F)
    m = editable
    n_est = float(np.sum(2.0 * A[m] / (np.sqrt(3.0) * ell[m] ** 2)))
    return {"area_mm2": float(A[m].sum()), "n_estimate": n_est,
            "margin": float(p["budget_margin"]),
            "n_budget": int(np.ceil(n_est * (1.0 + p["budget_margin"]))),
            "n_current": int(m.sum())}


# --------------------------------------------------------------------------
# discrete operators
# --------------------------------------------------------------------------


def cotan_operators(V, F):
    """Cotangent stiffness L (N x N, PSD) and diagonal vertex mass M (mm^2).

    Thin wrapper around :func:`head_rgbd.neck_fit.cotan_operators`, which holds
    the corrected indexing (the old local copy applied the ``(k+2)`` offset
    twice and gave every edge the cotangent of the wrong angle).  ``V`` is in
    metres here, the operator lives in millimetres.
    """
    from head_rgbd import neck_fit as nfit
    L, M, _A = nfit.cotan_operators(np.asarray(V, float) * 1000.0, F)
    return L, M


def axial_mm(Pmm, ax, curve, h_chin):
    """Axial coordinates of arbitrary points given in mm."""
    P = np.asarray(Pmm, float) - ax["centre_mm"][None, :]
    h = P @ ax["axis"]
    a = P @ ax["lateral"]
    b = P @ ax["anterior"]
    ca, cb = curve(h)
    da, db = a - ca, b - cb
    return h, np.arctan2(da, db), np.hypot(da, db), h / h_chin


def ell_reg_field(t, theta_deg_abs, p):
    """Physical regularisation length (mm) per vertex.

    Structure: 0.25 x the mandible reference edge (~1.1 mm), so the main bulge is
    not flattened.  Connections: 0.5 x (~2.3 mm), wider smoothing exactly where
    the neck joins the head and the root.  The regulariser below is written so
    that this length -- not a neighbour-iteration count -- sets the scale.
    """
    jaw = p["jaw_reference_mm"]
    l_struct = p["reg_structure_ratio"] * jaw
    l_conn = p["reg_connection_ratio"] * jaw
    t = np.asarray(t, float)
    th = np.asarray(theta_deg_abs, float)

    def ramp(x, lo, hi):
        return np.clip((np.asarray(x, float) - lo) / (hi - lo), 0.0, 1.0)

    w_struct = np.exp(-0.5 * ((t - p["struct_t"]) / (1.6 * p["struct_t_sigma"])) ** 2) * \
               (1.0 - ramp(th, 30.0, 75.0))
    w_struct = np.clip(w_struct, 0.0, 1.0)
    w_conn = np.maximum(ramp(t, 0.80, 0.95), 1.0 - ramp(t, 0.06, 0.22))
    l = l_conn + (l_struct - l_conn) * w_struct
    l = l + (l_conn - l) * w_conn * 0.65
    return l


def _face_area(V, F):
    tri = V[F]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                axis=1) * 1e6


BARY = np.array([[0.5, 0.5, 0.0],
                 [0.0, 0.5, 0.5],
                 [0.5, 0.0, 0.5],
                 [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
BARY_W = np.full(4, 0.25)


def local_edge_length(V, F):
    """Median incident edge length (mm) per vertex."""
    E = _edge_table(F)
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) * 1000.0
    acc = np.zeros(len(V))
    cnt = np.zeros(len(V))
    np.add.at(acc, E[:, 0], L)
    np.add.at(acc, E[:, 1], L)
    np.add.at(cnt, E[:, 0], 1.0)
    np.add.at(cnt, E[:, 1], 1.0)
    return np.divide(acc, np.maximum(cnt, 1.0))


def band_gate(r_mm, bg_mm, p, lo=None, hi=None):
    """1 inside the neck tube, 0 where the ray has left the tube.

    ``r - background`` is the signed relief for a real neck surface and grows to
    10-40 mm on the jaw underside, the shoulder plate and the back of the
    occiput.  Those vertices belong to the head, not to the neck, and the
    target field has no meaning there: without this gate the least squares
    drags them by up to 40 mm.
    """
    lo = p["leak_gate_lo"] if lo is None else lo
    hi = p["leak_gate_hi"] if hi is None else hi
    x = (np.asarray(r_mm, float) - np.asarray(bg_mm, float) - lo) / max(hi - lo, 1e-9)
    return 1.0 - np.clip(x, 0.0, 1.0)


def field_residuals(V, F, ax, curve, h_chin, field, weight=None):
    """Signed field residuals at vertices and at 3 edge midpoints + centroid.

    The quadrature samples are numerical evaluation points, never new vertices;
    they measure what happens *between* vertices, which a vertex-only residual
    cannot see.  Statistics are reported twice: over the editable set (where the
    field is meaningful) and over the whole mesh (where it is not -- the head and
    the shoulder plate are outside the tube and the field there is background).
    """
    Vmm = np.asarray(V, float) * 1000.0
    h, th, r, t = axial_mm(Vmm, ax, curve, h_chin)
    rt = np.asarray(field(h, th), float)
    rt = np.where(np.isfinite(rt), rt, r)
    rv = rt - r
    if weight is None:
        wv = np.ones(len(V))
        wf = np.ones(len(F))
        wq = np.ones((len(F), 4))
    elif callable(weight):
        par = np.arange(len(V))[:, None].repeat(3, axis=1)
        lam = np.broadcast_to(np.array([1.0, 0.0, 0.0]), (len(V), 3))
        wv = np.asarray(weight(Vmm, h, th, r, lam, par), float)
        tri = Vmm[F]
        wq = np.zeros((len(F), 4))
        for k in range(4):
            Pk = np.einsum("j,njc->nc", BARY[k], tri)
            hk, tk, rk, _ = axial_mm(Pk, ax, curve, h_chin)
            wq[:, k] = np.asarray(weight(Pk, hk, tk, rk, np.broadcast_to(BARY[k], (len(F), 3)),
                                         F), float)
        wf = wq.mean(axis=1)
    else:
        wv = np.asarray(weight, float)
        wf = np.asarray(weight, float)[F].min(axis=1)
        wq = np.repeat(wf[:, None], 4, axis=1)
    out = {}
    m = wv > 1e-6
    if m.any():
        out.update({"vertex_rms_mm": float(np.sqrt(np.average(rv[m] ** 2, weights=wv[m]))),
                    "vertex_p95_mm": float(np.percentile(np.abs(rv[m]), 95)),
                    "vertex_max_mm": float(np.max(np.abs(rv[m])))})
    else:
        out.update({"vertex_rms_mm": 0.0, "vertex_p95_mm": 0.0, "vertex_max_mm": 0.0})
    A_f = _face_area(V, F)
    tri = Vmm[F]
    res, wts = [], []
    for k in range(4):
        P = BARY[k] @ tri
        hh, tt, rr, _ = axial_mm(P, ax, curve, h_chin)
        rr_t = np.asarray(field(hh, tt), float)
        rr_t = np.where(np.isfinite(rr_t), rr_t, rr)
        res.append(rr_t - rr)
        wts.append(A_f * BARY_W[k] * np.maximum(wq[:, k], 0.0))
    res = np.concatenate(res)
    wts = np.concatenate(wts)
    mq = wts > 1e-9
    if mq.any():
        out.update({"quad_rms_mm": float(np.sqrt(np.average(res[mq] ** 2, weights=wts[mq]))),
                    "quad_p95_mm": float(np.percentile(np.abs(res[mq]), 95)),
                    "quad_max_mm": float(np.max(np.abs(res[mq])))})
    else:
        out.update({"quad_rms_mm": 0.0, "quad_p95_mm": 0.0, "quad_max_mm": 0.0})
    out["vertex_rms_all_mm"] = float(np.sqrt(np.mean(rv ** 2)))
    out["n_editable"] = int(m.sum())
    out["n_faces_editable"] = int((wf > 1e-6).sum())
    return out


class BandWeight:
    """Continuous editable weight ``w(x)`` of the neck band.

    Every factor is evaluated **at the sample point** from its coordinates:
    the axial height ramp, the angular window (with the nape keep-out), the
    tube-leak gate (from the background radius field) and the jaw cap.  Only the
    normal-based factor cannot be evaluated from coordinates -- it needs the
    surface normal -- so it is interpolated barycentrically from the parents.
    That makes ``w`` continuous across faces: a face straddling a fade boundary
    contributes a proportionally smaller row instead of a step, which is what
    the old "``w[F].min() > 1e-6``" mask could not do.

    The axial coordinate and therefore the design band come from
    ``h_material`` -- the height in the **neutral (closed)** frame, one value
    per vertex index -- not from ``h`` measured against this state's own menton.
    Otherwise the same band would select different material points in the open
    state (whose menton sits 14.6 mm lower) and the protection set would move
    with the jaw.  ``h`` is still used for the background field sample, because
    that grid belongs to the state.
    """

    def __init__(self, st, F, p, h_chin_design, dn, fixed, h_material=None):
        self.p = dict(p)
        self.hc_design = float(h_chin_design)
        self.hc_state = float(st["h_chin"])
        self.bg = st["bg"]
        self.field = st.get("field")
        self.dn = np.asarray(dn, float)
        self.fixed = np.asarray(fixed, bool)
        if h_material is None:
            h_material = np.asarray(st["h_material"], float) \
                if "h_material" in st else None
        self.h_m = None if h_material is None else np.asarray(h_material, float)
        self._cache = None

    def __call__(self, P, h, th, r, lam, par):
        P = np.asarray(P, float)
        h = np.asarray(h, float)
        r = np.asarray(r, float)
        th = np.asarray(th, float)
        lam = np.asarray(lam, float)
        par = np.asarray(par, np.int64)
        p = self.p
        if self.h_m is None:
            hm = h                      # no material coordinate: fall back
        else:
            hm = np.einsum("nk,nk->n", lam, self.h_m[par])
        t_design = hm / self.hc_design
        t_state = h / self.hc_state
        thd = np.abs(np.degrees(th))
        height = (ramp(t_design, *p["data_t"][:2]) *
                  (1.0 - ramp(t_design, *p["data_t"][2:])))
        bg = np.asarray(self.bg(h, th), float)
        bg = np.where(np.isfinite(bg), bg, r)
        gate = 1.0 - ramp(r - bg, p["leak_gate_lo"], p["leak_gate_hi"])
        w_th = 1.0 - ramp(thd, p["theta_full_deg"], p["theta_zero_deg"])
        w_th = w_th * (1.0 - (1.0 - p["nape_keep"]) *
                       ramp(thd, p["nape_lo_deg"], p["nape_hi_deg"]))
        ant = 1.0 - ramp(thd, 45.0, 60.0)
        # the jaw cap is a material definition too: same material points in both
        # states, so it cannot be re-anchored to the moved menton either
        w_jaw = 1.0 - ant * ramp(t_design, *p["jaw_top_t"])
        dn_s = np.einsum("nk,nk->n", lam, self.dn[par])
        w_down = 1.0 - ramp(-dn_s, p["down_face_lo"], p["down_face_hi"])
        w = height * w_th * w_down * gate * w_jaw
        # The fixed set is a Dirichlet boundary, not a face mask: the solver
        # zeroes the Jacobian columns of fixed parents, so a face straddling the
        # protected boundary contributes through its free corners only.  Do not
        # zero ``w`` by ``self.fixed[par]`` here.
        #
        # Two *validity* gates on top of the broad tube-leak gate.  They are
        # evaluated at the sample point and use the continuous field, so they do
        # not introduce a face mask:
        #
        # * inward guard -- above ``inward_t_lo`` the radial field can describe
        #   the jaw underside rather than the neck.  There the target is 2-4 mm
        #   *inside* the actual surface; a positive structure relief is not.
        #   Suppressing only the large negative residual keeps the laryngeal
        #   flank (positive) and the shallow saddle (small negative) while
        #   removing the submental pull that folded the last round at t~0.88.
        # * shoulder gate -- below ``side_t_hi`` and beyond ``side_theta_lo``
        #   the cross-section flares into the shoulder and the radial field is
        #   not the surface.  The reference structure is anterior; there is no
        #   target to defend out there.
        if self.field is not None and p.get("validity_gates", False):
            rt = np.asarray(self.field(h, th), float)
            rt = np.where(np.isfinite(rt), rt, r)
            neg = np.maximum(0.0, -(rt - r) - float(p.get("inward_guard_mm", 1.0)))
            gate_far = 1.0 - ramp(neg, 0.0, float(p.get("inward_gate_mm", 1.5)))
            gate_in = 1.0 - (1.0 - gate_far) * ramp(
                t_design, float(p.get("inward_t_lo", 0.82)),
                float(p.get("inward_t_hi", 0.88)))
            outside = ramp(thd, float(p.get("side_theta_lo", 82.0)),
                           float(p.get("side_theta_hi", 95.0)))
            side_ok = 1.0 - outside * (1.0 - ramp(
                t_design, float(p.get("side_t_lo", 0.45)),
                float(p.get("side_t_hi", 0.55))))
            w = w * gate_in * side_ok
        w = np.maximum(w, 0.0)
        if p.get("weight_fixed_faces", True):
            # legacy path (the old build tool): a face touching the fixed set
            # carries no data row.  The v2 build sets weight_fixed_faces=False
            # and zeroes only the fixed Jacobian columns, which keeps the
            # weight continuous across the protected boundary.
            return np.where(self.fixed[par].any(axis=1), 0.0, w)
        return w

    def vertex_weight(self, V_m, ax, curve, h_chin):
        """``w`` at the vertices (for masks, reports and the free set)."""
        Vmm = np.asarray(V_m, float) * 1000.0
        h, th, r, t = axial_mm(Vmm, ax, curve, h_chin)
        par = np.arange(len(Vmm))[:, None].repeat(3, axis=1)
        lam = np.broadcast_to(np.array([1.0, 0.0, 0.0]), (len(Vmm), 3))
        return np.asarray(self(Vmm, h, th, r, lam, par), float)


def fit_band(V, F, ax, curve, h_chin, field, ell_reg, fixed, data_weight, p,
             max_mm=6.0, max_iter=1, anchor=None):
    """Area-weighted radial fit -- now a thin wrapper on the corrected solver.

    (The maths, the operator and the iteration live in
    :mod:`head_rgbd.neck_fit`; this function only converts units and keeps the
    dictionary keys the build script and the verifier already use.)

    ``data_weight`` is a callable ``w(P, h, th, r, lam, par)`` -- a continuous
    weight evaluated at every quadrature sample.  A plain array is accepted for
    backwards compatibility and is wrapped as a constant-in-the-face weight,
    which is exactly the behaviour this round replaces.
    """
    from head_rgbd import neck_fit as nfit
    if data_weight is None:
        wfun = None
    elif callable(data_weight):
        wfun = data_weight
    else:
        arr = np.asarray(data_weight, float)

        def wfun(P, h, th, r, lam, par):
            return np.einsum("nk,nk->n", lam, arr[par])
    out = nfit.fit_radial(V, F, ax, curve, h_chin, field, ell_reg, fixed, wfun, p,
                          max_mm=max_mm, max_iter=max_iter, anchor=anchor)
    out["rows"] = int(out["data_samples"])
    out["reg_rows"] = int(out["bend_rows"])
    return out
