"""Benchmark linear-solver options on the *actual* reduced neck systems.

Captures (H, g) from a real builder run by wrapping ``neck_fit.spsolve``, then
times alternative orderings/factorisations and compares their solutions to the
production default.  Read-only with respect to the algorithm: the wrapper
returns the default result, so the build is unchanged.

Run:
    python -m tools.neck_solve_bench --fit <dense> --landmarks <json> --out <new dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from head_rgbd import neck_fit as nfit              # noqa: E402


def _time(fn, reps=3):
    best = None
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--landmarks", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--remesh-passes", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        ap.error(f"refusing to overwrite {out}")
    out.mkdir(parents=True)

    from tools import build_neck_direct_v2 as bld

    captured = []
    original = nfit.spsolve

    def wrapper(H, g, *a, **kw):
        res = original(H, g, *a, **kw)
        captured.append((H.copy(), np.asarray(g, float).copy(),
                         np.asarray(res, float).copy()))
        return res

    nfit.spsolve = wrapper
    try:
        argv = ["--fit", args.fit, "--out", str(out / "build"),
                "--anchor-validity-only", "--adapt-full-limit", "20.0",
                "--no-render", "--larynx-prior", "--larynx-prior-ratio", "0.35",
                "--remesh-passes", str(args.remesh_passes)]
        if args.landmarks:
            argv += ["--landmarks", args.landmarks]
        bld.main(argv)
    finally:
        nfit.spsolve = original

    rows = []
    for k, (H, g, ref) in enumerate(captured):
        H = H.tocsc()
        base_t, _ = _time(lambda: spsolve(H, g), args.reps)
        lu = splu(H)
        asym = H - H.T
        diag = H.diagonal()
        nz = H.data[H.data != 0.0]
        rec = {"system": k, "n": int(H.shape[0]), "nnz": int(H.nnz),
               "density": round(H.nnz / float(H.shape[0] ** 2), 5),
               "stored_explicit_zeros": int(H.nnz - len(nz)),
               "symmetric_max_abs": float(abs(asym).max()) if asym.nnz else 0.0,
               "diag_min": float(diag.min()), "diag_max": float(diag.max()),
               "abs_entry_min": float(np.abs(nz).min()),
               "abs_entry_max": float(np.abs(nz).max()),
               "lu_fill_nnz": int(lu.L.nnz + lu.U.nnz),
               "fill_ratio": round((lu.L.nnz + lu.U.nnz) / float(H.nnz), 2),
               "default_colamd_s": round(base_t, 5), "options": {}}
        # a same-size synthetic SPD control: is 237 ms normal for this shape?
        rng = np.random.default_rng(0)
        n = H.shape[0]
        A = sparse.random(n, n, density=H.nnz / float(n * n), format="csc",
                          random_state=rng)
        A = (A @ A.T + sparse.eye(n, format="csc") * 1.0).tocsc()
        ctl_t, _ = _time(lambda: spsolve(A, np.ones(n)), args.reps)
        rec["synthetic_spd_same_nnz_s"] = round(ctl_t, 5)
        rec["synthetic_nnz"] = int(A.nnz)
        Hz = H.copy()
        Hz.eliminate_zeros()
        ez_t, _ = _time(lambda: spsolve(Hz, g), args.reps)
        rec["options"]["eliminate_zeros"] = {
            "s": round(ez_t, 5), "speedup": round(base_t / ez_t, 2),
            "nnz_after": int(Hz.nnz),
            "max_abs_diff": float(np.abs(spsolve(Hz, g) - ref).max())}
        for spec in ("MMD_AT_PLUS_A", "MMD_ATA", "NATURAL"):
            try:
                t, x = _time(lambda s=spec: spsolve(H, g, permc_spec=s), args.reps)
                rec["options"][spec] = {
                    "s": round(t, 5), "speedup": round(base_t / t, 2),
                    "max_abs_diff": float(np.abs(x - ref).max()),
                    "rel_diff": float(np.linalg.norm(x - ref) /
                                      max(np.linalg.norm(ref), 1e-30))}
            except Exception as exc:
                rec["options"][spec] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            def sym():
                lu = splu(H, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                          options={"SymmetricMode": True})
                return lu.solve(g)
            t, x = _time(sym, args.reps)
            rec["options"]["splu_symmetric_mmd"] = {
                "s": round(t, 5), "speedup": round(base_t / t, 2),
                "max_abs_diff": float(np.abs(x - ref).max()),
                "rel_diff": float(np.linalg.norm(x - ref) /
                                  max(np.linalg.norm(ref), 1e-30))}
        except Exception as exc:
            rec["options"]["splu_symmetric_mmd"] = {"error": f"{type(exc).__name__}: {exc}"}
        rows.append(rec)
        print(json.dumps(rec), flush=True)

    payload = {"schema": "neck_solve.bench.v1", "fit": args.fit,
               "reps": args.reps, "n_systems": len(rows), "systems": rows}
    (out / "solve_bench.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())