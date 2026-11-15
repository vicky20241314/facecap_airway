"""Subject-adaptive larynx localization and a canonical laryngeal prior.

The reference relief is a *global* neck-shape transfer.  It is not a guarantee
that a subject will show a laryngeal prominence: the prominence can sit at a
different height, the jaw underside can cut its upper flank, and a smooth or
partially observed capture may simply not carry it.  This module adds the
missing semantic feature:

* **localization** -- use the menton landmark (static-68 index 8) and the
  subject's own anterior normal gate (the highest anterior height where the
  surface still faces up/outward, not the jaw underside) to choose a larynx
  centre and a vertical band that fits inside the editable neck;
* **prior** -- a canonical V-shaped thyroid ridge: a narrow central ridge,
  shallow lateral hollows and a smooth axial envelope.  It is added to the
  applied relief as a continuous 2-D field, with an amplitude scaled by the
  subject's jaw/head reference, so a visible larynx does not depend on the
  capture happening to contain one.

All functions are pure numpy and do not modify the mesh.  The build tool calls
them only when ``--larynx-prior`` is set; default off.
"""
from __future__ import annotations

import numpy as np

from head_rgbd import neck_direct as nd
from head_rgbd import neck_prominence as npn


def _bump(x, lo, hi, power=2.0):
    """C1 compact bump: 1 for |x| <= lo, exactly 0 at |x| >= hi."""
    u = np.clip((np.abs(np.asarray(x, float)) - float(lo)) /
                max(float(hi) - float(lo), 1e-12), 0.0, 1.0)
    return (0.5 * (1.0 + np.cos(np.pi * u))) ** float(power)


def anterior_gate_top(V, F, ax, curve, h_chin, p, theta_deg=15.0):
    """Highest anterior t where the surface still faces up/outward.

    The jaw underside is downward-facing; the same normal gate that protects it
    also cuts the upper half of a laryngeal target.  This function finds the
    subject-specific boundary of the *neck* part of the anterior surface, so the
    prior can end before the jaw instead of being truncated inside it.

    Returns ``(t_gate, profile)``.  ``profile`` is a list of
    ``(t_lo, t_hi, median_dn_anterior, n)`` for the report.
    """
    P = np.asarray(V, float) * 1000.0
    h, th, r, t = nd.axial_mm(P, ax, curve, h_chin)
    thd = np.degrees(th)
    dn = npn.vertex_normals(V, F) @ np.asarray(ax["axis"], float)
    threshold = float(p.get("larynx_prior_gate_dn", -0.30))
    profile = []
    gate = float(p.get("larynx_prior_gate_fallback", 0.78))
    for lo in np.arange(0.55, 0.96, 0.025):
        m = (np.abs(thd) < float(theta_deg)) & (t >= lo) & (t < lo + 0.025)
        if m.sum() < 3:
            continue
        med = float(np.median(dn[m]))
        profile.append([float(lo), float(lo + 0.025), med, int(m.sum())])
        if med > threshold:
            gate = float(min(lo + 0.025, 0.95))
    return gate, profile


def larynx_anatomy(V, F, ax, curve, h_chin, landmarks, p):
    """Subject-adaptive larynx centre and prior parameters.

    The centre is the reference larynx height, but it is **shifted down** when
    the subject's own anterior normal gate is lower: the prior band must fit in
    the neck, not be cut by the jaw underside.  The amplitude is scaled by the
    measured jaw reference (the same head-scale anchor used by the density
    field), clamped to a plausible range.
    """
    gate, profile = anterior_gate_top(V, F, ax, curve, h_chin, p)
    t_ref = float(p.get("larynx_prior_t_ref", 0.748))
    t_half = float(p.get("larynx_prior_t_half", 0.045))
    t_min = float(p.get("larynx_prior_t_min", 0.62))
    # centre so that the upper edge of the prior sits at the gate
    t_center = min(t_ref, max(t_min, gate - t_half))
    jaw = float(p.get("jaw_reference_mm", 4.857))
    amp = float(p.get("larynx_prior_ratio", 0.35)) * jaw
    amp = float(np.clip(amp, float(p.get("larynx_prior_min_mm", 1.0)),
                        float(p.get("larynx_prior_max_mm", 2.5))))
    lm = np.asarray(landmarks, float) if landmarks is not None else None
    chin_h = None
    if lm is not None and len(lm) > 8:
        P = np.asarray(lm[8], float) * 1000.0
        hh, _, _, tt = nd.axial_mm(P[None, :], ax, curve, h_chin)
        chin_h = float(hh[0])
    return {
        "t_gate": float(gate),
        "t_ref": float(t_ref),
        "t_center": float(t_center),
        "t_half": float(t_half),
        "d_larynx_mm": float((1.0 - t_center) * h_chin),
        "chin_landmark_h_mm": chin_h,
        "h_chin_mm": float(h_chin),
        "amp_mm": float(amp),
        "theta_sigma_deg": float(p.get("larynx_prior_theta_sigma", 10.0)),
        "theta_zero_deg": float(p.get("larynx_prior_theta_zero", 45.0)),
        "theta_zero_hollow_deg": float(p.get("larynx_prior_theta_zero_hollow", 55.0)),
        "hollow_deg": float(p.get("larynx_prior_hollow_deg", 28.0)),
        "hollow_sigma_deg": float(p.get("larynx_prior_hollow_sigma", 10.0)),
        "hollow_ratio": float(p.get("larynx_prior_hollow_ratio", 0.35)),
        "gate_dn_threshold": float(p.get("larynx_prior_gate_dn", -0.30)),
        "profile": profile,
        "landmark_index": 8,
        "note": "centre = min(reference t, gate - half); amplitude = ratio x "
                "jaw reference; prior is added, not used to rescale the whole neck",
    }


def larynx_prior_field(t, theta, anat):
    """Canonical V ridge on a (t, theta) grid, in mm.

    ``t`` is one-dimensional height, ``theta`` one-dimensional angle.  The
    central ridge is narrow; the lateral hollow is negative, so the target has
    more *shape* without a proportionally larger peak.  The axial and angular
    windows are compact C1 bumps, so the prior has no boundary kink.
    """
    t = np.asarray(t, float)
    th = np.asarray(theta, float)
    mt = _bump(t - anat["t_center"], 0.0, anat["t_half"], 2.0)
    # The prior must stay inside the subject's anterior neck: above the normal
    # gate the surface is the jaw underside.  A smooth 0.04 t fade is used so the
    # clip cannot introduce a boundary kink.
    gate = anat.get("t_gate")
    if gate is not None:
        mt = mt * (1.0 - nd.ramp(t, float(gate) - 0.02, float(gate) + 0.02))
    mth = _bump(th, 0.0, np.radians(anat["theta_zero_deg"]), 2.0)
    sig = np.radians(anat["theta_sigma_deg"])
    hol = np.radians(anat["hollow_deg"])
    sigh = np.radians(anat["hollow_sigma_deg"])
    mth_h = _bump(th, 0.0,
                  np.radians(anat.get("theta_zero_hollow_deg", 55.0)), 2.0)
    ridge = mth * np.exp(-0.5 * (th / sig) ** 2)
    hollow = mth_h * anat["hollow_ratio"] * np.exp(
        -0.5 * ((np.abs(th) - hol) / sigh) ** 2)
    return anat["amp_mm"] * mt[:, None] * (ridge - hollow)[None, :]


def apply_larynx_prior(tg, anat):
    """Add the prior to the applied relief and the target radius."""
    pr = larynx_prior_field(tg["t"], tg["theta"], anat)
    base = np.asarray(tg["applied"], float).copy()
    tg["applied_base_prior"] = base
    tg["applied"] = base + pr
    tg["R"] = np.asarray(tg["R"], float) + pr
    tg["larynx_prior"] = pr
    tg["larynx_anatomy"] = anat
    tg["larynx_prior_max_mm"] = float(pr.max())
    tg["larynx_prior_min_mm"] = float(pr.min())
    return tg