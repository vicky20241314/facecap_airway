"""Unit tests for the three confirmed bridge defects.

Run:  python -m tools.neck_bridge_tests [--out tests/neck_bridge_tests.json]

1. ``rings_nonempty``   ``_rings(F, B, depth, C)`` must not start from the
                        empty set.  On a tube with A / B / C rings by height it
                        must return the C vertices within ``depth`` graph steps
                        and must cover every C vertex adjacent to B.
2. ``weighted_single_dof``  one data row only: changing the positive weight must
                        not move the solution.  The previous code scaled only
                        the right hand side, so the target itself was multiplied
                        by the weight.
3. ``bending_sign``     for ``||L_u d_u + L_o d_o||^2`` the known contribution
                        on the right hand side is ``-L_o d_o``; the interior of
                        a strip with a known ramp at one end must follow the
                        ramp, not run away from it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from head_rgbd import neck_bridge as nb     # noqa: E402
from head_rgbd import neck_fit as nf        # noqa: E402


def tube(nseg=48, nring=12, radius=20.0, height=60.0):
    """Closed tube around z, vertices in mm, faces wound consistently."""
    th = np.linspace(0.0, 2 * np.pi, nseg, endpoint=False)
    zs = np.linspace(0.0, height, nring)
    V, F = [], []
    for z in zs:
        for a in th:
            V.append([radius * np.sin(a), radius * np.cos(a), z])
    V = np.array(V, float)
    for r in range(nring - 1):
        for s in range(nseg):
            i0 = r * nseg + s
            i1 = r * nseg + (s + 1) % nseg
            i2 = (r + 1) * nseg + s
            i3 = (r + 1) * nseg + (s + 1) % nseg
            F.append([i0, i2, i3])
            F.append([i0, i3, i1])
    return V, np.array(F, np.int64), zs


def test_rings():
    V, F, zs = tube()
    z = V[:, 2]
    A = z < 20.0
    B = (z >= 20.0) & (z < 30.0)
    C = z >= 30.0
    # one-ring C: every C vertex with a B neighbour, computed independently
    from head_rgbd import mesh_integrity as mi
    E = mi.edge_records(F)[0]
    touch = (B[E[:, 0]] & C[E[:, 1]]) | (B[E[:, 1]] & C[E[:, 0]])
    true_one = np.zeros(len(V), bool)
    true_one[E[touch].ravel()] = True
    true_one &= C
    r1 = nb._rings(F, B, 1, C)
    r3 = nb._rings(F, B, 3, C)
    r0 = nb._rings(F, B, 0, C)
    # graph distance (in rings) of every C vertex from B, independently
    dist = np.full(len(V), -1, int)
    dist[B] = 1
    cur = B.copy()
    for d in range(1, 4):
        nxt = np.zeros(len(V), bool)
        nxt[E[cur[E[:, 0]] | cur[E[:, 1]]].ravel()] = True
        nxt &= ~(dist > 0)
        dist[nxt] = d + 1
        cur = nxt
    expected3 = C & (dist > 0) & (dist <= 4)
    return {"name": "rings_nonempty",
            "c_ring_1": int(r1.sum()), "c_ring_3": int(r3.sum()),
            "c_ring_0": int(r0.sum()),
            "true_c_neighbours": int(true_one.sum()),
            "one_ring_covers_true_boundary": bool(np.array_equal(r1, true_one)),
            "three_ring_matches_bfs": bool(np.array_equal(r3, expected3)),
            "ok": bool(r0.sum() == 0 and r1.sum() > 0 and r3.sum() > r1.sum()
                       and np.array_equal(r1, true_one) and np.array_equal(r3, expected3))}


def test_weighted_single_dof():
    # one unknown vector d, one row: w * (e_x . d) = w * target
    out = []
    for w in (0.05, 0.3, 1.0, 4.0, 25.0):
        coef = np.array([[1.0, 0.0, 0.0]])
        res = np.array([2.5])
        A, b = nb._weighted_scalar_row(coef, res, np.array([w]))
        # minimum-norm solution of the single row (rank one by construction)
        d = np.linalg.lstsq(A, b.ravel(), rcond=None)[0]
        out.append({"weight": w, "solution": float(d[0])})
    base = out[0]["solution"]
    ok = all(abs(r["solution"] - base) < 1e-12 for r in out) and abs(base - 2.5) < 1e-12
    return {"name": "weighted_single_dof", "rows": out,
            "target": 2.5, "ok": bool(ok),
            "note": "the old code used coef and w*res, so the solution was "
                    "w * target / coef and moved with the weight"}


def strip(nx=7, ny=3, width=10.0):
    """Triangulated rectangle in the z=0 plane, known ramp along x."""
    xs = np.linspace(0.0, width, nx)
    ys = np.linspace(0.0, 2.0, ny)
    V, F = [], []
    for y in ys:
        for x in xs:
            V.append([x, y, 0.0])
    V = np.array(V, float)
    for j in range(ny - 1):
        for i in range(nx - 1):
            i0 = j * nx + i
            i1 = i0 + 1
            i2 = i0 + nx
            i3 = i2 + 1
            F.append([i0, i1, i2])
            F.append([i1, i3, i2])
    return V, np.array(F, np.int64)


def test_bending_sign():
    V, F = strip()
    L, M, A_v = nf.cotan_operators(V, F)
    ell = np.full(len(V), 2.0)
    Lb = nf.bending_operator(L, A_v, ell).tocsr()
    x = V[:, 0]
    known_mask = (x <= 1e-9) | (x >= x.max() - 1e-9)
    unknown = ~known_mask
    known = np.zeros((len(V), 3))
    # a positive ramp in x: 0 on the left boundary, 1 on the right boundary
    known[known_mask, 0] = x[known_mask] / x.max()
    pos = np.full(len(V), -1, int)
    pos[unknown] = np.arange(unknown.sum())
    rhs, sub, inb = nb.bending_known_rhs(Lb, np.flatnonzero(unknown), pos,
                                         known, int(unknown.sum()))
    # direct check of the sign: rhs == -L_out d_known
    out_cols = ~inb
    manual = np.zeros((int(unknown.sum()), 3))
    if out_cols.any():
        np.add.at(manual, sub.row[out_cols],
                  -sub.data[out_cols, None] * known[sub.col[out_cols]])
    sign_ok = bool(np.allclose(rhs, manual))
    # solve L_u d_u = rhs in the least-squares sense and inspect the interior
    r = sub.row[inb]
    c = pos[sub.col[inb]]
    d = sub.data[inb]
    rows = (r[:, None] * 3 + np.arange(3)[None, :]).ravel()
    cols = (c[:, None] * 3 + np.arange(3)[None, :]).ravel()
    Mb = sparse.coo_matrix((np.repeat(d, 3), (rows, cols)),
                           shape=(3 * int(unknown.sum()), 3 * int(unknown.sum())))
    du = sparse.linalg.lsqr(Mb.tocsr(), rhs.ravel())[0].reshape(-1, 3)
    dx = np.zeros(len(V))
    dx[unknown] = du[:, 0]
    dx[known_mask] = known[known_mask, 0]
    # the correct sign gives an interior that follows the ramp: monotone and
    # inside [0, 1]; the old sign gives values outside that range / wrong slope
    interior = dx[unknown]
    order = np.argsort(V[unknown, 0], kind="stable")
    interior_sorted = interior[order]
    ok = (sign_ok and np.all(interior > -1e-9) and np.all(interior < 1.0 + 1e-9)
          and np.all(np.diff(interior_sorted) > -1e-9))
    return {"name": "bending_sign", "sign_ok": sign_ok,
            "unknown_min": float(interior.min()), "unknown_max": float(interior.max()),
            "monotone": bool(np.all(np.diff(interior_sorted) > -1e-9)),
            "ok": bool(ok),
            "note": "with +L_out d_known the interior leaves [0,1] and the "
                    "solution moves away from the known ramp"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/neck_bridge_tests.json")
    args = ap.parse_args(argv)
    tests = [test_rings(), test_weighted_single_dof(), test_bending_sign()]
    payload = {"tests": tests, "failed": [t["name"] for t in tests if not t["ok"]]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for t in tests:
        print(f"{t['name']}: {'PASS' if t['ok'] else 'FAIL'}  {t}")
    print(f"failed: {payload['failed']} -> {out}")
    return 0 if not payload["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())