"""Head-region scale: traceable menton and same-protocol head dimensions (0.9).

0.8 candidate bug (fixed here): ``reference_head_dimensions`` selected the chin
as the front-most lower-face point, which on this reference is the NOSE TIP at
y=+1.52 mm.  That produced a fake 139.68 mm head height, a 1.6368 longitudinal
multiplier, a wrong temporal sampling band and the false conclusion that the
reference head is "relatively flatter".  The public 0.9 path does not use any of
those numbers: the menton is the submental break of the midline anterior
profile, and both heads are measured crown-to-menton with the same band.
"""
from __future__ import annotations

import numpy as np


def _band_stats(values, lateral, forward, band):
    take = band
    return (float(np.ptp(np.percentile(lateral[take], [.5, 99.5]))),
            float(np.ptp(np.percentile(forward[take], [.5, 99.5]))))


def detect_reference_menton(vertices_mm):
    """Traceable geometric menton on the reference midline anterior profile.

    The chin turns UNDER into the submental region: the anterior midline profile
    rises steeply from the throat to the chin.  The menton is that break, found
    as the steepest POSITIVE slope in the lower-face band below the nose tip.
    Crown / nose tip / chin tip are independent stable landmarks.
    """
    V = np.asarray(vertices_mm, float)
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    crown = float(y.max())
    mid = np.abs(x) <= 3.0
    nose_i = int(np.flatnonzero(mid)[np.argmax(z[mid])])
    nose_y, nose_z = float(y[nose_i]), float(z[nose_i])
    ys = np.arange(nose_y - 135.0, nose_y - 20.0, 1.0)
    prof = np.full(len(ys), np.nan)
    for i, yy in enumerate(ys):
        sel = mid & (np.abs(y - yy) <= 1.0)
        if sel.sum() >= 2:
            prof[i] = float(z[sel].max())
    ok = np.isfinite(prof)
    prof = np.interp(ys, ys[ok], prof[ok])
    sm = np.convolve(np.pad(prof, (2, 2), mode="edge"), np.ones(5) / 5.0, mode="valid")
    slope = np.gradient(sm, ys)
    band = (ys >= nose_y - 115.0) & (ys <= nose_y - 55.0)
    ids = np.flatnonzero(band)
    idx = int(ids[np.argmax(slope[band])])
    menton_y = float(ys[idx])
    chin_band = (y > menton_y + 12.0) & (y < nose_y - 20.0)
    chin_tip_y = float(y[chin_band][np.argmax(z[chin_band])])
    jaw_band = (y > menton_y + 5.0) & (y < nose_y - 20.0)
    return {"crown_y_mm": round(crown, 3), "nose_tip_y_mm": round(nose_y, 3),
            "nose_tip_z_mm": round(nose_z, 3),
            "menton_y_mm": round(menton_y, 3),
            "menton_slope_mm_per_mm": round(float(slope[idx]), 3),
            "chin_tip_y_mm": round(chin_tip_y, 3),
            "jaw_width_mm": round(float(np.ptp(np.percentile(x[jaw_band], [.5, 99.5]))), 3),
            "method": "midline anterior profile (|x|<=3mm): steepest rise between "
                      "throat and chin below the nose tip = submental break / menton; "
                      "crown, nose tip and chin tip are independent stable landmarks",
            "old_bug": "max(Z) over y<40mm selected the nose tip at y=+1.52mm -> "
                       "fake H_head 139.68mm and longitudinal ratio 1.6368"}


def head_dimensions(V_mm, up, right, fwd, chin_mm):
    """Crown-to-menton height and temporal width/depth, same protocol for both heads."""
    V = np.asarray(V_mm, float)
    rel = V - np.asarray(chin_mm, float)
    v = rel @ np.asarray(up, float)
    u = rel @ np.asarray(right, float)
    f = rel @ np.asarray(fwd, float)
    crown = float(v.max())
    band = (v > .35 * crown) & (v < .85 * crown)
    W = float(np.ptp(np.percentile(u[band], [.5, 99.5])))
    D = float(np.ptp(np.percentile(f[band], [.5, 99.5])))
    return {"H_head_mm": round(crown, 3), "W_head_mm": round(W, 3),
            "D_head_mm": round(D, 3), "crown_v_mm": round(crown, 3),
            "chin_v_mm": 0.0,
            "region": "crown to menton; temporal band 0.35-0.85 H; "
                      "neck, shoulders and bottom cut excluded",
            "axes": "head-local up/right/forward from stable facial landmarks"}


def reference_head_dimensions(vertices_mm):
    """Corrected reference head dimensions (Y up, +Z anterior), no neck/shoulder."""
    V = np.asarray(vertices_mm, float)
    m = detect_reference_menton(V)
    sel = (np.abs(V[:, 0]) <= 3.) & (np.abs(V[:, 1] - m["menton_y_mm"]) <= 2.)
    chin_z = float(V[sel, 2].max()) if sel.any() else 0.0
    chin = np.array([0., m["menton_y_mm"], chin_z])
    out = head_dimensions(V, np.array([0., 1., 0.]), np.array([1., 0., 0.]),
                          np.array([0., 0., 1.]), chin)
    out.update({"crown_y_mm": m["crown_y_mm"], "chin_y_mm": m["menton_y_mm"],
                "nose_tip_y_mm": m["nose_tip_y_mm"],
                "menton": m,
                "axes": "Y up, +Z anterior",
                "note": "0.9 corrected menton (submental break); the old 139.68mm "
                        "head height came from a nose-tip mis-detection"})
    return out


def target_head_dimensions(vertices_m, up, right, forward, chin_m):
    """Target head: crown=max along head up, chin=menton landmark; same band."""
    V = np.asarray(vertices_m, float) * 1000.0
    up = np.asarray(up, float); right = np.asarray(right, float)
    forward = np.asarray(forward, float); chin = np.asarray(chin_m, float) * 1000.0
    rel = V - chin
    v = rel @ up; u = rel @ right; f = rel @ forward
    H = float(v.max())
    band = (v > .35 * H) & (v < .85 * H)
    W, D = _band_stats(v, u, f, band)
    return {"H_head_mm": round(H, 2), "W_head_mm": round(W, 2),
            "D_head_mm": round(D, 2),
            "region": "crown to chin/menton along the head up axis; temporal band "
                      "0.35-0.85 H; neck excluded",
            "axes": "head-local up/right/forward from the dense landmarks (closed state)"}


def head_scale(reference, target):
    ratios = {k: float(target[k] / reference[k]) for k in ("H_head_mm", "W_head_mm", "D_head_mm")}
    geo = float(np.cbrt(ratios["H_head_mm"] * ratios["W_head_mm"] * ratios["D_head_mm"]))
    return {"reference": reference, "target": target,
            "ratios": {k: round(v, 4) for k, v in ratios.items()},
            "width_depth_agreement": round(abs(ratios["W_head_mm"] - ratios["D_head_mm"]), 4),
            "height_width_difference": round(ratios["H_head_mm"] - ratios["W_head_mm"], 4),
            "S_head_geometric": round(geo, 4),
            "S_head_used_filter": round(geo, 4),
            "note": "0.9: with the real menton the head-height ratio is ~1.01, not "
                    "1.6368; the old 'reference head is relatively shorter/flatter' "
                    "conclusion is withdrawn. Width/depth ratios are used only for "
                    "head-normalised cross-section constraints; the geometric S is "
                    "used only for the isotropic filter and normalized visuals."}


def reference_ratio_targets(morph, ref_head, tgt_head, reference_l50_a, reference_l50_b,
                            reference_w50_b, reference_tip_a):
    """Historical 0.8 two-feature ratio protocol (kept for the archived experiment)."""
    wr = tgt_head["W_head_mm"] / ref_head["W_head_mm"]
    hr = tgt_head["H_head_mm"] / ref_head["H_head_mm"]
    A = morph["feature_A"]; B = morph["feature_B"]; S = morph["saddle"]
    return {
        "historical": True,
        "note": "0.8 candidate protocol; archived only. 0.9 public delivery uses "
                "head_rgbd/neck_volume.py cross-section targets instead.",
        "definitions": {
            "H_A_over_W_head": "reference A background-residual height / head width",
            "W50_A_over_W_head": "reference A lateral 50%-row width / head width",
            "L50_A_over_H_head": "reference A longitudinal 50%-height extent / head height",
            "H_B_over_W_head": "reference B background-residual height / head width",
            "B_local_prominence_over_W_head": "(B peak - saddle) / head width",
            "distance_AB_over_H_head": "A-B centre distance / head height",
            "tip_radius_A_over_W_head": "reference A lateral tip radius / head width",
        },
        "reference_raw": {
            "H_A_mm": A["height_mm"], "W50_A_mm": 32.5, "L50_A_mm": reference_l50_a,
            "H_B_mm": B["height_mm"], "B_local_mm": float(B["height_mm"] - S["height_mm"]),
            "saddle_mm": S["height_mm"], "distance_AB_mm": morph["relative"]["centre_distance_mm"],
            "tip_radius_A_mm": reference_tip_a,
        },
        "reference_ratios": {
            "H_A_over_W_head": A["height_mm"] / ref_head["W_head_mm"],
            "W50_A_over_W_head": 32.5 / ref_head["W_head_mm"],
            "L50_A_over_H_head": reference_l50_a / ref_head["H_head_mm"],
            "H_B_over_W_head": B["height_mm"] / ref_head["W_head_mm"],
            "B_local_prominence_over_W_head": (B["height_mm"] - S["height_mm"]) / ref_head["W_head_mm"],
            "distance_AB_over_H_head": morph["relative"]["centre_distance_mm"] / ref_head["H_head_mm"],
            "tip_radius_A_over_W_head": reference_tip_a / ref_head["W_head_mm"],
            "B_over_A": B["height_mm"] / A["height_mm"],
            "saddle_over_A": S["height_mm"] / A["height_mm"],
        },
        "target_mm": {
            "H_A_mm": A["height_mm"] * wr, "W50_A_mm": 32.5 * wr,
            "L50_A_mm": reference_l50_a * hr, "H_B_mm": B["height_mm"] * wr,
            "B_local_mm": (B["height_mm"] - S["height_mm"]) * wr,
            "saddle_mm": S["height_mm"] * wr,
            "distance_AB_mm": morph["relative"]["centre_distance_mm"] * hr,
            "tip_radius_A_mm": reference_tip_a * wr,
        },
        "scales": {"width_scale": round(wr, 4), "height_scale": round(hr, 4)},
        "tolerances": {"main_height_width_ratio": 0.10,
                       "B_A_and_saddle": 0.15, "distance_L50": 0.15},
    }
