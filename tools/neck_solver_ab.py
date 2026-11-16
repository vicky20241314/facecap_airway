"""Same-input A/B of the unified neck solver: reduced kernel vs the original.

The original full-space implementation is reproduced here verbatim as the
reference.  This tool wraps ``neck_fit.fit_unified`` inside a real
``tools.build_neck_direct_v2`` run, so every comparison uses the exact inputs
the production builder produced (same V/F, masks, weights, field, priors).
The wrapper returns the *production* result, so the build itself is unaffected.

Run:
    python -m tools.neck_solver_ab --fit <dense> --landmarks <json> --out <new dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from head_rgbd import neck_fit as nfit              # noqa: E402


# --------------------------------------------------------------------------
# reference: the pre-optimisation implementation, copied verbatim
# --------------------------------------------------------------------------

def ref_build_samples_3d(V_mm, F, D, ax, curve, h_chin, field, fixed, wfun,
                         A_v=None):
    V_mm = np.asarray(V_mm, float)
    D = np.asarray(D, float)
    F = np.asarray(F, np.int64)
    N = len(V_mm)
    if A_v is None:
        _, _, A_v = nfit.cotan_operators(V_mm, F)
    wfun = nfit._no_weight if wfun is None else wfun
    Pv = V_mm + D
    hv, thv, rv, _, _, _ = nfit.frame_at(Pv, ax, curve, h_chin)
    Rv, dRh_v, dRt_v = nfit.field_grad(field, hv, thv)
    Rv = np.where(np.isfinite(Rv), Rv, rv)

    par_v = np.arange(N)[:, None].repeat(3, axis=1)
    lam_v = np.broadcast_to(np.array([1.0, 0.0, 0.0]), (N, 3))
    w_v = np.asarray(wfun(Pv, hv, thv, rv, lam_v, par_v), float)
    w_v = np.where(np.asarray(fixed, bool), 0.0, np.maximum(w_v, 0.0))
    av = 0.5 * A_v

    pts, res, wts, par, lam, jh, jt, kind = [], [], [], [], [], [], [], []
    keepv = w_v > 1e-6
    if keepv.any():
        pts.append(Pv[keepv]); res.append((Rv - rv)[keepv])
        wts.append(av[keepv] * w_v[keepv])
        par.append(par_v[keepv]); lam.append(lam_v[keepv])
        jh.append(dRh_v[keepv]); jt.append(dRt_v[keepv])
        kind.append(np.zeros(int(keepv.sum()), np.int8))

    fixed = np.asarray(fixed, bool)
    tri = Pv[F]
    tri0 = V_mm[F]
    cr = np.cross(tri0[:, 1] - tri0[:, 0], tri0[:, 2] - tri0[:, 0])
    Af = 0.5 * np.linalg.norm(cr, axis=1)
    free_face = ~fixed[F].all(axis=1)
    fid = np.flatnonzero(free_face)
    if len(fid):
        tri_f = tri[fid]
        Af_f = Af[fid]
        for k in range(4):
            Pk = np.einsum("j,njc->nc", BARY[k], tri_f)
            hk, tk, rk, _, _, _ = nfit.frame_at(Pk, ax, curve, h_chin)
            Rk, dRhk, dRtk = nfit.field_grad(field, hk, tk)
            Rk = np.where(np.isfinite(Rk), Rk, rk)
            park = F[fid]
            lamk = np.broadcast_to(BARY[k], (len(fid), 3))
            wk = np.asarray(wfun(Pk, hk, tk, rk, lamk, park), float)
            wk = np.maximum(wk, 0.0)
            keep = wk > 1e-6
            if not keep.any():
                continue
            pts.append(Pk[keep]); res.append((Rk - rk)[keep])
            wts.append(0.5 * Af_f[keep] * BARY_W[k] * wk[keep])
            par.append(park[keep]); lam.append(lamk[keep])
            jh.append(dRhk[keep]); jt.append(dRtk[keep])
            kind.append(np.ones(int(keep.sum()), np.int8))

    if not pts:
        z = np.zeros(0)
        return {"P": np.zeros((0, 3)), "res": z, "w": z,
                "par": np.zeros((0, 3), np.int64), "lam": np.zeros((0, 3)),
                "dRh": z, "dRt": z, "kind": np.zeros(0, np.int8),
                "n_vertex": 0, "n_face": 0, "A_v": A_v}
    out = {"P": np.concatenate(pts), "res": np.concatenate(res),
           "w": np.concatenate(wts), "par": np.concatenate(par),
           "lam": np.concatenate(lam), "dRh": np.concatenate(jh),
           "dRt": np.concatenate(jt), "kind": np.concatenate(kind),
           "A_v": A_v}
    out["n_vertex"] = int((out["kind"] == 0).sum())
    out["n_face"] = int((out["kind"] == 1).sum())
    return out


BARY = nfit.BARY
BARY_W = nfit.BARY_W


def ref_assemble_3d(N, smp, J):
    n = J.shape[0]
    rows = np.repeat(np.arange(n), 9)
    cols = (smp["par"][:, :, None] * 3 + np.arange(3)[None, None, :]).ravel()
    vals = (np.sqrt(smp["w"])[:, None, None] * J).ravel()
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n, 3 * N)).tocsr()


def ref_fit_unified(V, F, ax, curve, h_chin, field, ell_reg, fixed, wfun, p,
                    mu_t=None, mu_anchor=None, max_d_mm=None, max_t_mm=None,
                    max_iter=40, tol_step_mm=1e-3, tol_rel=1e-8, log=None):
    """Original full-space masked solve (reference only)."""
    V_mm = np.asarray(V, float) * 1000.0
    F = np.asarray(F, np.int64)
    N = len(V_mm)
    max_d = float(p.get("max_d_mm", 6.0) if max_d_mm is None else max_d_mm)
    if max_t_mm is None:
        max_t_mm = np.full(N, float(p.get("max_t_mm", 3.0)))
    max_t_mm = np.broadcast_to(np.asarray(max_t_mm, float), (N,)).copy()
    if mu_t is None:
        mu_t = np.full(N, float(p.get("mu_t_c", 0.5)))
    mu_t = np.broadcast_to(np.asarray(mu_t, float), (N,)).copy()
    if mu_anchor is None:
        mu_anchor = np.zeros(N)
    mu_anchor = np.broadcast_to(np.asarray(mu_anchor, float), (N,)).copy()
    fixed = np.asarray(fixed, bool)
    alive = np.zeros(N, bool)
    alive[np.unique(F)] = True
    fixed = fixed | (~alive)
    free = ~fixed
    e_r0 = nfit.frame_at(V_mm, ax, curve, h_chin)[4]

    L, M, A_v = nfit.cotan_operators(V_mm, F)
    Rb = nfit.bending_operator(L, A_v, ell_reg)
    I3 = sparse.eye(3, format="csr")
    Rb3 = sparse.kron(Rb, I3, format="csr")
    mu_small = float(p.get("mu_small", 1e-3))

    D = np.zeros((N, 3))
    smp = ref_build_samples_3d(V_mm, F, D, ax, curve, h_chin, field, fixed, wfun,
                               A_v=A_v)
    T = nfit._projection_rows(V_mm, D, A_v, mu_t, fixed, ax, curve, h_chin,
                              isotropic=False)
    Aanch = nfit._projection_rows(V_mm, D, A_v, mu_anchor, fixed, ax, curve,
                                  h_chin, isotropic=True)
    e_prev, parts = nfit.energy_3d(smp, D, Rb3, T, Aanch, mu_small, A_v)
    parts_new = parts
    hist = []
    converged, reason = False, "max_iter"
    g_ref = None

    for it in range(int(max_iter)):
        if not len(smp["w"]):
            reason = "no_samples"
            break
        J = nfit.jacobian_rows_3d(smp, ax, curve, h_chin, fixed)
        Js = ref_assemble_3d(N, smp, J)
        vec = D.ravel()
        H = (Js.T @ Js + Rb3.T @ Rb3 + T.T @ T + Aanch.T @ Aanch).tocsc()
        H = H + sparse.diags(np.repeat(2.0 * mu_small * A_v, 3))
        g = (np.asarray(Js.T @ (np.sqrt(smp["w"]) * smp["res"])).ravel() +
             np.asarray(Rb3.T @ (Rb3 @ vec)).ravel() +
             np.asarray(T.T @ (T @ vec)).ravel() +
             np.asarray(Aanch.T @ (Aanch @ vec)).ravel() +
             2.0 * mu_small * (A_v[:, None] * D).ravel())
        free3 = np.repeat(free, 3)
        H = (sparse.diags(free3.astype(float)) @ H @
             sparse.diags(free3.astype(float)) +
             sparse.diags((~free3).astype(float)) +
             sparse.diags(free3 * 1e-10))
        g = np.where(free3, g, 0.0)
        gnorm = float(np.abs(g[free3]).max()) if free3.any() else 0.0
        if g_ref is None:
            g_ref = max(gnorm, 1e-30)
        if gnorm <= 1e-9 * g_ref:
            converged, reason = True, "gradient"
            hist.append({"iter": it, "energy": float(e_prev),
                         "grad_max": gnorm, "step_max_mm": 0.0,
                         "line_search_alpha": 0.0, "accepted": False})
            break
        delta = -np.asarray(spsolve(H.tocsc(), g), float)
        delta = np.where(np.isfinite(delta), delta, 0.0).reshape(N, 3)
        delta = np.where(free[:, None], delta, 0.0)
        if float(g @ delta.ravel()) >= 0.0:
            hd = H.diagonal()
            hd = np.where(hd > 0, hd, 1.0)
            delta = (-g / hd).reshape(N, 3)
        if np.max(np.abs(delta)) < tol_step_mm:
            converged, reason = True, "step_below_tolerance"
            hist.append({"iter": it, "energy": float(e_prev),
                         "grad_max": gnorm,
                         "step_max_mm": float(np.max(np.abs(delta))),
                         "line_search_alpha": 0.0, "accepted": False})
            break

        alpha, accepted = 1.0, False
        D_new, smp_new, parts_new, e_new = D, smp, parts, e_prev
        for _ in range(24):
            D_try = D + alpha * delta
            if not nfit._bounds_ok(D_try, e_r0, free, max_d, max_t_mm):
                alpha *= 0.5
                continue
            smp_try = ref_build_samples_3d(V_mm, F, D_try, ax, curve, h_chin,
                                           field, fixed, wfun, A_v=A_v)
            e_try, p_try = nfit.energy_3d(smp_try, D_try, Rb3, T, Aanch,
                                          mu_small, A_v)
            if e_try <= e_prev * (1.0 + 1e-12):
                alpha, accepted = alpha, True
                D_new, smp_new, parts_new, e_new = D_try, smp_try, p_try, e_try
                break
            alpha *= 0.5

        dnorm = float(np.max(np.abs(D_new - D)))
        rel = abs(e_prev - e_new) / max(e_prev, 1e-12)
        hist.append({"iter": it, "energy": float(e_new), "grad_max": gnorm,
                     "step_max_mm": dnorm, "line_search_alpha": float(alpha),
                     "accepted": bool(accepted), "energy_rel": float(rel)})
        if not accepted:
            if alpha <= 1e-4:
                converged, reason = True, "energy_stationary"
            else:
                reason = "line_search_failed"
            break
        D, smp, parts, e_prev = D_new, smp_new, parts_new, e_new
        if dnorm < tol_step_mm and rel < tol_rel:
            converged, reason = True, "step_and_energy"
            break

    P_end = V_mm + D
    smp_end = ref_build_samples_3d(V_mm, F, D, ax, curve, h_chin, field, fixed,
                                   wfun, A_v=A_v)
    kv = smp_end["kind"] == 0
    kf = smp_end["kind"] == 1
    sv = nfit._wstats(smp_end["res"][kv], smp_end["w"][kv])
    sf = nfit._wstats(smp_end["res"][kf], smp_end["w"][kf])
    rad = np.einsum("nk,nk->n", D, e_r0)
    tan = D - rad[:, None] * e_r0
    return {"V_mm": P_end, "d_mm": D, "s_mm": rad, "e_r": e_r0,
            "iterations": hist, "n_iter": len(hist), "converged": converged,
            "stop_reason": reason, "energy": float(e_prev),
            "energy_parts": parts_new if hist else parts,
            "data_samples": int(len(smp_end["w"])),
            "bend_rows": int(Rb3.shape[0]),
            "max_d_mm": float(np.linalg.norm(D, axis=1).max()) if N else 0.0,
            "max_d_rad_mm": float(np.abs(rad).max()) if N else 0.0,
            "max_d_tan_mm": float(np.linalg.norm(tan, axis=1).max()) if N else 0.0,
            "rms_d_mm": float(np.sqrt(np.mean(np.sum(D[free] ** 2, axis=1))))
            if free.any() else 0.0,
            "n_free": int(free.sum()), "n_fixed": int(fixed.sum()),
            "residual_after": {"vertex_rms_mm": sv["rms_mm"],
                               "vertex_p95_mm": sv["p95_mm"],
                               "vertex_max_mm": sv["max_mm"],
                               "quad_rms_mm": sf["rms_mm"],
                               "quad_p95_mm": sf["p95_mm"],
                               "quad_max_mm": sf["max_mm"],
                               "n_samples": int(len(smp_end["w"])),
                               "n_samples_vertex": int(kv.sum()),
                               "n_samples_face": int(kf.sum())}}


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def compare(new, ref, t_new, t_ref):
    dn = np.asarray(new["d_mm"], float)
    dr = np.asarray(ref["d_mm"], float)
    vn = np.asarray(new["V_mm"], float)
    vr = np.asarray(ref["V_mm"], float)
    it_n, it_r = new["iterations"], ref["iterations"]
    per_iter = []
    for k in range(max(len(it_n), len(it_r))):
        a = it_n[k] if k < len(it_n) else {}
        b = it_r[k] if k < len(it_r) else {}
        per_iter.append({
            "iter": k,
            "energy_new": a.get("energy"), "energy_ref": b.get("energy"),
            "energy_abs_diff": (abs(a["energy"] - b["energy"])
                                if "energy" in a and "energy" in b else None),
            "grad_new": a.get("grad_max"), "grad_ref": b.get("grad_max"),
            "alpha_new": a.get("line_search_alpha"),
            "alpha_ref": b.get("line_search_alpha"),
            "accepted_new": a.get("accepted"), "accepted_ref": b.get("accepted"),
            "step_new": a.get("step_max_mm"), "step_ref": b.get("step_max_mm"),
        })
    ra, rb = new["residual_after"], ref["residual_after"]
    return {
        "n_free": new["n_free"], "n_fixed": new["n_fixed"],
        "reduced_unknowns": 3 * new["n_free"],
        "full_unknowns": 3 * (new["n_free"] + new["n_fixed"]),
        "n_iter_new": new["n_iter"], "n_iter_ref": ref["n_iter"],
        "stop_reason_new": new["stop_reason"], "stop_reason_ref": ref["stop_reason"],
        "converged_new": new["converged"], "converged_ref": ref["converged"],
        "data_samples_new": new["data_samples"], "data_samples_ref": ref["data_samples"],
        "energy_new": new["energy"], "energy_ref": ref["energy"],
        "energy_rel_diff": abs(new["energy"] - ref["energy"]) / max(abs(ref["energy"]), 1e-30),
        "energy_parts_new": new["energy_parts"], "energy_parts_ref": ref["energy_parts"],
        "d_max_abs_diff_mm": float(np.abs(dn - dr).max()),
        "d_rms_diff_mm": float(np.sqrt(np.mean((dn - dr) ** 2))),
        "V_max_abs_diff_mm": float(np.abs(vn - vr).max()),
        "d_bit_identical": bool(np.array_equal(dn, dr)),
        "residual_after_new": ra, "residual_after_ref": rb,
        "residual_max_abs_diff_mm": max(abs(ra[k] - rb[k]) for k in
                                        ("vertex_rms_mm", "vertex_p95_mm",
                                         "vertex_max_mm", "quad_rms_mm",
                                         "quad_p95_mm", "quad_max_mm")),
        "s_new": round(t_new, 4), "s_ref": round(t_ref, 4),
        "speedup": round(t_ref / t_new, 2) if t_new else None,
        "iterations": per_iter,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--landmarks", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--remesh-passes", type=int, default=0)
    ap.add_argument("--max-solves", type=int, default=0,
                    help="0 = compare every solve of the run")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        ap.error(f"refusing to overwrite {out}")
    out.mkdir(parents=True)

    from tools import build_neck_direct_v2 as bld

    records = []
    original = nfit.fit_unified

    def wrapped(*a, **kw):
        t0 = time.perf_counter()
        new = original(*a, **kw)
        t_new = time.perf_counter() - t0
        if args.max_solves and len(records) >= args.max_solves:
            return new
        t0 = time.perf_counter()
        ref = ref_fit_unified(*a, **kw)
        t_ref = time.perf_counter() - t0
        rec = compare(new, ref, t_new, t_ref)
        rec["solve_index"] = len(records)
        records.append(rec)
        print(f"[ab] solve {rec['solve_index']}: "
              f"{rec['reduced_unknowns']} vs {rec['full_unknowns']} unknowns | "
              f"iters {rec['n_iter_new']}/{rec['n_iter_ref']} | "
              f"d max diff {rec['d_max_abs_diff_mm']:.3e} mm | "
              f"energy rel {rec['energy_rel_diff']:.3e} | "
              f"{rec['s_new']:.3f}s vs {rec['s_ref']:.3f}s "
              f"({rec['speedup']}x)", flush=True)
        return new

    nfit.fit_unified = wrapped
    bld.nfit.fit_unified = wrapped
    try:
        argv = ["--fit", args.fit, "--out", str(out / "build"),
                "--anchor-validity-only", "--adapt-full-limit", "20.0",
                "--no-render", "--larynx-prior", "--larynx-prior-ratio", "0.35",
                "--remesh-passes", str(args.remesh_passes)]
        if args.landmarks:
            argv += ["--landmarks", args.landmarks]
        code = bld.main(argv)
    finally:
        nfit.fit_unified = original
        bld.nfit.fit_unified = original

    ok = all(r["d_max_abs_diff_mm"] <= 1e-6 and
             r["stop_reason_new"] == r["stop_reason_ref"] and
             r["n_iter_new"] == r["n_iter_ref"] and
             r["residual_max_abs_diff_mm"] <= 1e-6 for r in records)
    payload = {"schema": "neck_solver.ab.v1", "fit": args.fit,
               "remesh_passes": args.remesh_passes,
               "builder_exit": code, "n_solves": len(records),
               "all_bit_identical": all(r["d_bit_identical"] for r in records),
               "max_d_diff_mm": max((r["d_max_abs_diff_mm"] for r in records),
                                    default=0.0),
               "total_s_new": round(sum(r["s_new"] for r in records), 4),
               "total_s_ref": round(sum(r["s_ref"] for r in records), 4),
               "pass": bool(ok), "solves": records}
    (out / "solver_ab.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in
                      ("n_solves", "all_bit_identical", "max_d_diff_mm",
                       "total_s_new", "total_s_ref", "pass")}, indent=1), flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())