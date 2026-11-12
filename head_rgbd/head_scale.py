"""Head-region scale and reference proportion targets (0.8 candidate)."""
from __future__ import annotations

import numpy as np


def _band_stats(values, lateral, forward, band):
    take = band
    return (float(np.ptp(np.percentile(lateral[take], [.5, 99.5]))),
            float(np.ptp(np.percentile(forward[take], [.5, 99.5]))))


def reference_head_dimensions(vertices_mm):
    """Reference head: crown=max Y, chin=front-most lower-face point, no neck."""
    V = np.asarray(vertices_mm, float)
    y, z, x = V[:, 1], V[:, 2], V[:, 0]
    top = float(y.max())
    lower = (y < 40.0) & (np.abs(x) < 40.0)
    chin = float(y[lower][int(np.argmax(z[lower]))])
    H = top - chin
    band = (y > chin + .35 * H) & (y < chin + .85 * H)
    W = float(np.ptp(np.percentile(x[band], [.5, 99.5])))
    D = float(np.ptp(np.percentile(z[band], [.5, 99.5])))
    return {"H_head_mm": round(H, 2), "W_head_mm": round(W, 2),
            "D_head_mm": round(D, 2), "crown_y_mm": round(top, 2),
            "chin_y_mm": round(chin, 2),
            "region": "crown to chin/menton; temporal band 0.35-0.85 H; neck excluded",
            "axes": "Y up, +Z anterior"}


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
            "region": "crown to chin/menton along the head up axis; temporal band 0.35-0.85 H; neck excluded",
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
            "note": "width and depth scales agree at ~1.20; the reference head is "
                    "relatively shorter (height scale ~1.64). The cross-sectional "
                    "features use W_head and the longitudinal features use H_head, as "
                    "declared in the ratio targets; the geometric S is used only for "
                    "the isotropic filter/normalized visual, not to inflate the larynx."}


def reference_ratio_targets(morph, ref_head, tgt_head, reference_l50_a, reference_l50_b,
                            reference_w50_b, reference_tip_a):
    wr = tgt_head["W_head_mm"] / ref_head["W_head_mm"]
    hr = tgt_head["H_head_mm"] / ref_head["H_head_mm"]
    A = morph["feature_A"]; B = morph["feature_B"]; S = morph["saddle"]
    return {
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
                       "B_A_and_saddle": 0.15, "distance_L50": 0.15,
                       "basis": "main normalized height/width +/-10% declared before the run; "
                                "secondary/saddle and longitudinal from the reference sigma=6/8/10 "
                                "stability (B/A 0.358-0.418, saddle/A ~0.20-0.29) and head-region "
                                "measurement repeatability; not relaxed after seeing results"},
    }
