"""Extract the reference laryngeal shape into a reusable local asset + replay check."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

REF = Path("D:/project_neck/line1/legacy/runtime/outputs/hack_default_verified/final/identity_rest.obj")
OUT = Path("D:/proiect/facecap_airway_delivery/runtime/assets/neck_reference_shape")
VIZ = Path("D:/proiect/facecap_airway_delivery/runtime/outputs/neck_reference_sculpt_ref_preview")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    VIZ.mkdir(parents=True, exist_ok=True)
    m = trimesh.load(str(REF), process=False, maintain_order=True)
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    import hashlib
    sha = hashlib.sha256(REF.read_bytes()).hexdigest()
    up = np.array([0.0, 1.0, 0.0])
    right = np.array([1.0, 0.0, 0.0])
    fwd = np.array([0.0, 0.0, 1.0])          # face side (verified from renders)
    normals = trimesh.load(str(REF), process=False).vertex_normals
    normals = np.asarray(normals, dtype=np.float64)
    u = V @ right
    y = V @ up
    z = V @ fwd

    # ---- midline front outline, chin, submental crease, larynx peak ----
    midline = np.abs(u) < 0.012
    front = (normals @ fwd) > 0.35
    ys = np.arange(-0.150, 0.055, 0.002)
    F_y = np.full(len(ys), np.nan)
    for i, yc in enumerate(ys):
        band = midline & front & (y >= yc - 0.001) & (y < yc + 0.001)
        if band.sum() >= 3:
            F_y[i] = z[band].max()
    valid = np.isfinite(F_y)
    # chin: the highest outline in the lower face band (submental transition)
    lower_face = valid & (ys < 0.0) & (ys > -0.09)
    chin_idx = np.flatnonzero(lower_face)[np.argmax(F_y[lower_face])]
    chin_y = ys[chin_idx]
    # crease: the minimum outline below the chin
    below = valid & (ys < chin_y - 0.004) & (ys > -0.135)
    crease_idx = np.flatnonzero(below)[np.argmin(F_y[below])]
    crease_y = ys[crease_idx]
    # larynx: the local maximum below the crease before the lower neck recedes
    after = valid & (ys < crease_y - 0.004) & (ys > -0.125)
    idx_after = np.flatnonzero(after)
    larynx_idx = idx_after[np.argmax(F_y[idx_after])]
    larynx_y = ys[larynx_idx]
    # background line through the submental crease and the lower-neck tail
    tail = valid & (ys < larynx_y - 0.020) & (ys > -0.135)
    anchors_y = np.concatenate([[crease_y], ys[tail]])
    anchors_f = np.concatenate([[F_y[crease_idx]], F_y[tail]])
    coeffs = np.polyfit(anchors_y, anchors_f, 1)
    bg_at = lambda yy: np.polyval(coeffs, yy)
    peak = float(F_y[larynx_idx] - bg_at(larynx_y))
    # ---- lateral extent of the larynx excess on the front surface ----
    region = front & (y > larynx_y - 0.030) & (y < larynx_y + 0.028) & (np.abs(u) < 0.055)
    excess = z - bg_at(y)
    region &= excess > 0.25 * peak
    u_half = float(np.percentile(np.abs(u[region]), 95)) if region.sum() else 0.02
    v_half = 0.030
    # visible extents on the surface grid
    # ---- bake a normalized 41x41 height field (bilinear resampling) ----
    n_grid = 41
    u_nodes = np.linspace(-1.6, 1.6, n_grid)
    v_nodes = np.linspace(-1.6, 1.6, n_grid)
    field = np.zeros((n_grid, n_grid))
    counts = np.zeros((n_grid, n_grid))
    # vectorized nearest accumulation on the normalized grid
    rn_u = u / u_half
    rn_v = (y - larynx_y) / v_half
    inside = front & (np.abs(rn_u) <= 1.6) & (np.abs(rn_v) <= 1.6)
    iu = np.clip(np.rint((rn_u[inside] + 1.6) / 3.2 * (n_grid - 1)).astype(int), 0, n_grid - 1)
    iv = np.clip(np.rint((rn_v[inside] + 1.6) / 3.2 * (n_grid - 1)).astype(int), 0, n_grid - 1)
    vals = np.clip(excess[inside], 0.0, None)
    np.add.at(field, (iv, iu), vals)
    np.add.at(counts, (iv, iu), 1.0)
    field = field / np.maximum(counts, 1.0)
    # light smoothing to fill empty cells, then normalize
    from scipy.ndimage import gaussian_filter
    field = gaussian_filter(field, sigma=1.0)
    field = np.clip(field, 0.0, None)
    peak_baked = float(field.max())
    if peak_baked > 0:
        field_norm = field / peak_baked
    else:
        field_norm = field
    replay_peak = peak_baked
    # replay check: compare the baked profile against the original outline excess
    uu_profile = field_norm[n_grid // 2, :]
    vv_profile = field_norm[:, n_grid // 2]
    replay = {
        "baked_peak_mm": round(replay_peak, 3),
        "original_peak_mm": round(peak, 3),
        "peak_ratio": round(replay_peak / max(peak, 1e-9), 3),
        "u_profile_peak_offset_norm": float(u_nodes[np.argmax(uu_profile)]),
        "v_profile_peak_offset_norm": float(v_nodes[np.argmax(vv_profile)]),
    }
    asset = {
        "schema": "head_rgbd.neck_reference_shape.v1",
        "source_path": str(REF),
        "source_sha256": sha,
        "source_mesh": {"vertices": int(len(V)), "faces": int(len(F)), "units": "m",
                        "note": "identity_rest in a rest/canonical frame; axes X right / Y up / Z toward the face"},
        "frame": {"up": up.tolist(), "right": right.tolist(), "forward": fwd.tolist()},
        "reference_measurements_mm": {
            "chin_y": round(float(chin_y) * 1000, 2),
            "submental_crease_y": round(float(crease_y) * 1000, 2),
            "larynx_centre_y": round(float(larynx_y) * 1000, 2),
            "larynx_height_mm": round(peak, 2),
            "lateral_half_width_mm": round(u_half * 1000, 2),
            "vertical_half_length_mm": round(v_half * 1000, 2),
            "background_line": [round(float(c), 4) for c in coeffs],
        },
        "grid": {"u_nodes_norm": u_nodes.tolist(), "v_nodes_norm": v_nodes.tolist(),
                 "normalized_by": "peak height and local half width/length",
                 "field": field_norm.round(4).tolist()},
        "replay": replay,
        "processing": "front-facing midline outline, chin/submental/larynx landmarks, linear background "
                      "through submental crease and lower-neck tail, n-excess baked on a 41x41 normalized "
                      "grid, gaussian sigma=1 cell, normalized by the peak",
    }
    asset_path = OUT / "neck_reference_shape.json"
    asset_path.write_text(json.dumps(asset, indent=1), encoding="utf-8")
    # replay render: reconstructed reference local surface vs baked asset profiles
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(u_nodes, uu_profile, "r-o", ms=3)
    axes[0].set_title("baked lateral profile (normalized)")
    axes[1].plot(v_nodes, vv_profile, "b-o", ms=3)
    axes[1].set_title("baked longitudinal profile (normalized)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ / "reference_baked_profiles.png", dpi=110)
    plt.close(fig)
    print(json.dumps({"asset": str(asset_path), "sha": sha[:16], "measurements": asset["reference_measurements_mm"],
                      "replay": replay}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
