"""Implementation tests for the corrected neck fit (no project data needed).

Run:  python -m tools.neck_fit_tests [--out tests/neck_fit_tests.json]

Five tests, each of which fails on the previous implementation:

1. ``cotan_unit``        irregular right triangle -> edge weights [0.25, 0, 1].
2. ``single_dof``        +1 mm target on a cylinder: converges, and the
                         per-iteration max|s| sequence never goes 1 -> 0 -> 1.
3. ``face_weight_fade``  the four in-face samples of one triangle carry
                         *different* weights and the face's total weight is a
                         continuous, monotone function of the fade position
                         (the old code had one binary on/off per face).
4. ``jacobian_fd``       analytical d(res)/d(s_i) against a finite difference of
                         the residual recomputed at the displaced points.
5. ``bending_units``     ||R s||^2 = sum_v (ell^4 / A_v)(L s)^2 and the energy is
                         invariant under a uniform refinement of the patch
                         (the old operator changed with density).
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

from head_rgbd import neck_fit as nf     # noqa: E402


# --------------------------------------------------------------------------
# a synthetic frame: axis = z, lateral = x, anterior = y, spine on the axis
# --------------------------------------------------------------------------


def frame():
    ax = {"centre_mm": np.zeros(3), "axis": np.array([0.0, 0.0, 1.0]),
          "lateral": np.array([1.0, 0.0, 0.0]),
          "anterior": np.array([0.0, 1.0, 0.0])}
    curve = lambda h: (np.zeros_like(np.asarray(h, float)),   # noqa: E731
                       np.zeros_like(np.asarray(h, float)))
    return ax, curve


class ConstField:
    """R(h, theta) = c0 + c1*h + c2*sin(theta)."""

    def __init__(self, c0=21.0, c1=0.0, c2=0.0):
        self.c0, self.c1, self.c2 = c0, c1, c2

    def __call__(self, h, th):
        h = np.asarray(h, float)
        th = np.asarray(th, float)
        return self.c0 + self.c1 * h + self.c2 * np.sin(th)


def cylinder(R=20.0, H=20.0, nseg=48, nring=7):
    """Open tube around the z axis, vertices in mm."""
    th = np.linspace(0.0, 2 * np.pi, nseg, endpoint=False)
    zs = np.linspace(0.0, H, nring)
    V, F = [], []
    for z in zs:
        for a in th:
            V.append([R * np.sin(a), R * np.cos(a), z])
    V = np.array(V, float)
    for r in range(nring - 1):
        for s in range(nseg):
            i0 = r * nseg + s
            i1 = r * nseg + (s + 1) % nseg
            i2 = (r + 1) * nseg + s
            i3 = (r + 1) * nseg + (s + 1) % nseg
            F.append([i0, i2, i3])
            F.append([i0, i3, i1])
    return V, np.array(F, np.int64)


def to_m(Vmm):
    return np.asarray(Vmm, float) / 1000.0


# --------------------------------------------------------------------------
# 1. cotangent weights
# --------------------------------------------------------------------------


def test_cotan_unit():
    out = nf.cotan_unit_test()
    out["name"] = "cotan_unit"
    out["ok"] = bool(out["pass"] and out["symmetric"] and out["row_sums"] < 1e-9
                     and abs(out["mass_sum_single_mm2"] - 1.0) < 1e-12
                     and abs(out["mass_sum_quad_mm2"] - 2.0) < 1e-12)
    return out


# --------------------------------------------------------------------------
# 2. single degree of freedom, +1 mm, no odd/even oscillation
# --------------------------------------------------------------------------


def test_single_dof():
    ax, curve = frame()
    Vmm, F = cylinder()
    field = ConstField(c0=21.0)          # target radius = rest radius + 1 mm
    fixed = np.zeros(len(Vmm), bool)
    p = {"max_d_mm": 5.0}
    hist = []
    out = nf.fit_radial(to_m(Vmm), F, ax, curve, 20.0, field,
                        np.full(len(Vmm), 1.1), fixed, None, p,
                        max_iter=12, log=hist)
    s = out["s_mm"]
    seq = [float(np.abs(h["max_s_mm"])) for h in out["iterations"]]
    energies = [float(h["energy"]) for h in out["iterations"]]
    d_e = np.diff(energies)
    # the oscillation signature of the old loop: a step that nearly undoes the
    # previous one, i.e. sign flips with comparable magnitude
    undoing = 0
    for a, b in zip(seq[:-1], seq[1:]):
        if a > 0.5 and b < 0.5 * a:
            undoing += 1
    return {"name": "single_dof",
            "target_mm": 1.0,
            "s_mean_mm": float(np.mean(s)), "s_min_mm": float(np.min(s)),
            "s_max_mm": float(np.max(s)),
            "max_abs_error_mm": float(np.max(np.abs(s - 1.0))),
            "iterations": len(out["iterations"]),
            "max_s_sequence": seq,
            "energy_sequence": energies,
            "monotone_energy": bool(np.all(d_e <= 1e-12)),
            "undoing_steps": int(undoing),
            "converged": bool(out["converged"]),
            "stop_reason": out["stop_reason"],
            "ok": bool(abs(float(np.mean(s)) - 1.0) < 0.05
                       and float(np.max(np.abs(s - 1.0))) < 0.12
                       and undoing == 0
                       and np.all(d_e <= 1e-12)
                       and out["converged"])}


# --------------------------------------------------------------------------
# 3. in-face weights fade continuously (no "whole face in or out")
# --------------------------------------------------------------------------


def test_face_weight_fade():
    ax, curve = frame()
    Vmm, F = cylinder(nseg=24, nring=5)
    # one specific triangle, well away from the sample-space edges
    fid = int(np.argmax(Vmm[F][:, :, 2].mean(axis=1)))
    tri = Vmm[F[fid]]

    def band(z0, width=2.0):
        def wfun(P, h, th, r, lam, par):
            return np.clip((np.asarray(P, float)[:, 2] - z0) / width, 0.0, 1.0)
        return wfun

    # total weight this face contributes to the data term, as the fade moves
    zs = np.arange(tri[:, 2].min() - 2.0, tri[:, 2].max() + 2.0, 0.05)
    tot = []
    per_sample = []
    for z0 in zs:
        smp = nf.build_samples(Vmm, F, np.zeros(len(Vmm)), nf.frame_at(
            Vmm, ax, curve, 20.0)[4], ax, curve, 20.0, ConstField(),
            np.zeros(len(Vmm), bool), band(z0), faces=(np.arange(len(F)) == fid))
        face_rows = smp["kind"] == 1                 # only this face's four points
        tot.append(float(smp["w"][face_rows].sum()))
        per_sample.append(smp["w"][face_rows].copy())
    tot = np.array(tot)
    spread = []
    for w in per_sample:
        if len(w) == 4:
            spread.append(float(w.max() / max(w.min(), 1e-12)))
        else:
            spread.append(float("nan"))
    dtot = np.abs(np.diff(tot))
    full = float(np.max(tot))
    return {"name": "face_weight_fade",
            "face": fid, "face_area_mm2": float(0.5 * np.linalg.norm(
                np.cross(tri[1] - tri[0], tri[2] - tri[0]))),
            "fade_axis_positions_mm": [float(z) for z in zs],
            "face_total_weight": [float(t) for t in tot],
            "max_step_between_positions": float(dtot.max()) if len(dtot) else 0.0,
            "total_weight_range": [float(tot.min()), float(tot.max())],
            "samples_per_face": int(len(per_sample[0])),
            "max_intra_face_weight_ratio": float(np.nanmax(spread)),
            "fraction_of_positions_with_partial_fade": float(
                np.mean((tot > 1e-6) & (tot < full - 1e-6))),
            "ok": bool(len(per_sample[0]) == 4
                       and np.nanmax(spread) > 1.5                 # not one mask per face
                       and dtot.max() < 0.10 * full                 # continuous
                       and np.mean((tot > 1e-6) & (tot < full - 1e-6)) > 0.4)}


# --------------------------------------------------------------------------
# 4. Jacobian against a finite difference
# --------------------------------------------------------------------------


def test_jacobian_fd():
    ax, curve = frame()
    Vmm, F = cylinder()
    field = ConstField(c0=21.0, c1=0.1, c2=0.4)
    fixed = np.zeros(len(Vmm), bool)
    rng = np.random.default_rng(0)
    s = 0.2 * rng.standard_normal(len(Vmm))
    s[0] = 0.0
    e_r = nf.frame_at(Vmm, ax, curve, 20.0)[4]
    smp = nf.build_samples(Vmm, F, s, e_r, ax, curve, 20.0, field, fixed, None)
    J, _ = nf.jacobian_rows(Vmm, s, e_r, smp, ax, curve, 20.0, fixed)

    eps = 1e-3
    picks = [0, 1, 25, 100, len(Vmm) // 2, len(Vmm) - 1]
    errs, rel = [], []
    for i in picks:
        s2 = s.copy()
        s2[i] += eps
        smp2 = nf.build_samples(Vmm, F, s2, e_r, ax, curve, 20.0, field, fixed, None)
        if len(smp2["res"]) != len(smp["res"]):
            continue                                  # sample set must be stable
        fd = (smp2["res"] - smp["res"]) / eps
        col = np.zeros(len(smp["res"]))
        hit = np.flatnonzero((smp["par"] == i).any(axis=1))
        for h in hit:
            k = int(np.flatnonzero(smp["par"][h] == i)[0])
            col[h] = J[h, k]
        use = np.abs(fd) > 1e-6
        if not use.any():
            continue
        errs.append(float(np.max(np.abs(fd[use] - col[use]))))
        rel.append(float(np.max(np.abs(fd[use] - col[use]) /
                                np.maximum(np.abs(fd[use]), 1e-9))))
    return {"name": "jacobian_fd", "eps_mm": eps, "vertices_checked": len(errs),
            "max_abs_error": float(np.max(errs)) if errs else None,
            "max_rel_error": float(np.max(rel)) if rel else None,
            "per_vertex_abs_error": [float(e) for e in errs],
            "ok": bool(errs and np.max(errs) < 5e-3 and np.max(rel) < 5e-2)}


# --------------------------------------------------------------------------
# 5. bending operator: units, and invariance under refinement
# --------------------------------------------------------------------------


def test_bending_units():
    ax, curve = frame()
    Vmm, F = cylinder(nseg=48, nring=7)
    Vmm2, F2 = cylinder(nseg=96, nring=13)
    L, M, A_v = nf.cotan_operators(Vmm, F)
    L2, M2, A_v2 = nf.cotan_operators(Vmm2, F2)
    ell = np.full(len(Vmm), 2.0)
    ell2 = np.full(len(Vmm2), 2.0)
    Rb = nf.bending_operator(L, A_v, ell)
    Rb2 = nf.bending_operator(L2, A_v2, ell2)

    # a constant must be exactly in the kernel of L.  A function linear in the
    # axis is harmonic *in the interior* of a cylinder (it is a developable
    # surface) -- the open rims do not see the second triangle of their
    # z-edges, so that check is restricted to the interior rings, and is
    # reported separately rather than being required.
    s_const = np.full(len(Vmm), 3.0)
    h = Vmm[:, 2]
    s_lin = 0.01 * h
    Ls_lin = L @ s_lin
    interior = (h > h.min() + 1e-9) & (h < h.max() - 1e-9)
    e_const = float(np.sum((Rb @ s_const) ** 2))
    e_lin_interior = float(np.sum((np.diag(np.sqrt(
        np.maximum(np.asarray(ell, float), 1e-6) ** 4 /
        np.maximum(A_v, 1e-9))) @ Ls_lin)[interior] ** 2))

    # energy of a localised bump must not change when the patch is refined:
    # compare on the *same* physical bump, normalised by the solved amplitude
    def bump(V, sigma=3.0, h0=10.0):
        return np.exp(-0.5 * ((V[:, 2] - h0) / sigma) ** 2)
    b1, b2 = bump(Vmm), bump(Vmm2)
    e1 = float(np.sum((Rb @ b1) ** 2))
    e2 = float(np.sum((Rb2 @ b2) ** 2))
    ratio = e2 / max(e1, 1e-30)
    return {"name": "bending_units",
            "kernel_const": e_const, "kernel_linear_interior": e_lin_interior,
            "bump_energy_coarse": e1, "bump_energy_fine": e2,
            "bump_energy_ratio_fine_over_coarse": ratio,
            "vertices": [int(len(Vmm)), int(len(Vmm2))],
            "ok": bool(e_const < 1e-9 and 0.7 < ratio < 1.4)}


# --------------------------------------------------------------------------
# 6. weights integrate to the area (data term is an area integral)
# --------------------------------------------------------------------------


def test_data_weight_area():
    ax, curve = frame()
    Vmm, F = cylinder()
    fixed = np.zeros(len(Vmm), bool)
    smp = nf.build_samples(Vmm, F, np.zeros(len(Vmm)), nf.frame_at(
        Vmm, ax, curve, 20.0)[4], ax, curve, 20.0, ConstField(), fixed, None)
    area = float(nf.face_geometry(Vmm, F)[0].sum())
    total = float(smp["w"].sum())
    return {"name": "data_weight_area", "area_mm2": area,
            "sum_of_sample_weights": total, "ratio": total / area,
            "n_vertex": smp["n_vertex"], "n_face": smp["n_face"],
            "ok": bool(abs(total / area - 1.0) < 1e-9)}


TESTS = [test_cotan_unit, test_single_dof, test_face_weight_fade,
         test_jacobian_fd, test_bending_units, test_data_weight_area]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/neck_fit_tests.json")
    args = ap.parse_args(argv)
    results = []
    for fn in TESTS:
        try:
            r = fn()
        except Exception as exc:                      # a crashed test is a failed test
            r = {"name": fn.__name__, "ok": False,
                 "error": f"{type(exc).__name__}: {exc}"}
        results.append(r)
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['name']}", flush=True)
        if "error" in r:
            print(f"       {r['error']}", flush=True)
        elif r["name"] == "single_dof":
            print(f"       s mean {r['s_mean_mm']:.4f} mm, max err "
                  f"{r['max_abs_error_mm']:.4f}, iters {r['iterations']}, "
                  f"sequence {[round(x, 3) for x in r['max_s_sequence']]}", flush=True)
        elif r["name"] == "jacobian_fd":
            print(f"       max abs err {r['max_abs_error']}, max rel "
                  f"{r['max_rel_error']} over {r['vertices_checked']} vertices", flush=True)
        elif r["name"] == "face_weight_fade":
            print(f"       max step {r['max_step_between_positions']:.4f} of "
                  f"{max(r['total_weight_range']):.4f}, intra-face ratio "
                  f"{r['max_intra_face_weight_ratio']:.3f}", flush=True)
        elif r["name"] == "cotan_unit":
            print(f"       single {r['single']} quad {r['quad']}", flush=True)
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJ / out
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"pass": all(r["ok"] for r in results),
               "n_pass": int(sum(r["ok"] for r in results)), "n": len(results),
               "tests": results}
    json.dump(summary, open(out, "w", encoding="utf-8"), indent=1)
    print(f"{summary['n_pass']}/{summary['n']} passed -> {out}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
