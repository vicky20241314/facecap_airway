"""Three-dimensional solve of the submental bridge (region B).

The neck body (region C) is solved as a scalar radial offset, because the target
is a radius field and the radial direction is the only first-order motion that
changes a radius.  The bridge is different: it has to join two *surfaces* -- the
fixed jaw/chin border (region A) and the already shaped neck body -- and a
scalar radius cannot express that.  It needs a genuine 3-D displacement, with
both ends prescribed:

    d = 0          on the A side      (the head is not allowed to move)
    d = d_radial   on the C side      (the shaped neck body)

Between them the field minimises

    E = sum_B  0.5 A_v w |d . e_r - s_target|^2         data (fades out at A)
      + l^4 sum_v (L d)_v^2 / A_v                       area-mass bending
      + mu_p sum_Cring  A_v |d - d_rad|                 pin the C side
      + mu_g sum_B(A)   A_v |d|^2                       hand-over to the head
      + mu_t sum_B(C)   A_v |d - d_pred|^2              tangent plane continuity
      + mu_s sum_B      A_v |d|^2                       pose prior (no drift)

The tangent-plane term is the linear continuation of the C solution across the
interface: ``d_pred`` is the first-order Taylor expansion of ``d_radial`` at the
C-side vertex, so matching it matches the surface derivative there (position
matching alone would only give C0).  The hand-over term is the same construction
against the A side, where the known field is identically zero.

Every term is area weighted (mm^4, like the radial fit) and the whole problem is
*linear* in ``d``, so it is one sparse solve per state -- no iteration, no
line search, no null space: the two Dirichlet ends leave no rigid or constant
mode to drift along, and the ``mu_s`` prior makes the system strictly positive
definite even if a vertex loses every other row.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from head_rgbd import mesh_integrity as mi
from head_rgbd import neck_fit as nfit


def _rings(F, seed, depth, allowed):
    """Breadth-first ``depth`` rings around one or several seed vertices.

    The search **expands from the seed through the whole mesh** and only the
    output is restricted to ``allowed``.  The previous version first intersected
    the seed with ``allowed``, so the canonical call ``_rings(F, B, 3, C)`` --
    seed B, allowed C, two disjoint sets -- started from an empty frontier and
    returned an empty ring.  The C-side pins and the tangent-plane rows of
    :func:`solve_bridge` were therefore silently absent while the report still
    claimed a multi-ring extension.  A test in ``tools/neck_bridge_tests.py``
    asserts that the C ring is non-empty and covers every C vertex adjacent to
    the bridge.
    """
    E = mi.edge_records(F)[0]
    n = len(np.asarray(allowed))
    seed = np.asarray(seed, bool)
    allowed = np.asarray(allowed, bool)
    visited = seed.copy()
    frontier = seed.copy()
    out = seed & allowed
    for _ in range(int(depth)):
        touch = frontier[E[:, 0]] | frontier[E[:, 1]]
        nxt = np.zeros(n, bool)
        nxt[E[touch].ravel()] = True
        nxt &= ~visited
        if not nxt.any():
            break
        visited |= nxt
        frontier = nxt
        out |= nxt & allowed
    return out


def _local_jacobian(X, d, ids, E, ridge=1e-4):
    """Least-squares 3x3 gradient of ``d`` at each id, over its neighbours."""
    nb = {}
    for i, j in E:
        nb.setdefault(int(i), []).append(int(j))
        nb.setdefault(int(j), []).append(int(i))
    out = {}
    for i in np.asarray(ids, np.int64):
        ks = np.array(nb.get(int(i), []), np.int64)
        if len(ks) < 4:
            out[int(i)] = None
            continue
        A = X[ks] - X[i]
        rhs = d[ks] - d[i]
        G = np.linalg.solve(A.T @ A + ridge * np.eye(3), A.T @ rhs)     # (3,3)
        out[int(i)] = G
    return out


def _weighted_scalar_row(coef, res, weight):
    """Rows of ``min ||weight * (coef . d - res)||^2``, coefficient and RHS.

    Both sides carry the same factor.  This is the one place the convention is
    written down; :func:`solve_bridge` calls it and the single-degree-of-freedom
    test uses it directly.
    """
    coef = np.asarray(coef, float).reshape(-1, 3)
    res = np.asarray(res, float).reshape(-1, 1)
    w = np.asarray(weight, float).reshape(-1, 1)
    return w * coef, w * res


def bending_known_rhs(L, rows, unknown_pos, known, n_unknown):
    """Right hand side of ``min ||L_u d_u + L_o d_o||^2``.

    Expanding the squared norm gives the normal equation

        L_u^T L_u d_u = -L_u^T L_o d_o,

    so the known contribution carried to the right hand side is
    ``-L_o d_o``.  The previous code used ``+L_o d_o``, which solves a system
    whose known boundary pulls the interior the wrong way.  The sign is not a
    convention: a test in ``tools/neck_bridge_tests.py`` builds a strip with a
    known ramp on one side and checks that the interior follows the ramp instead
    of running away from it.

    ``L`` is the full (N, N) bending operator, ``rows`` the vertex ids whose
    rows enter the solve, ``unknown_pos`` the vertex -> unknown index map (-1
    outside), ``known`` the (N, 3) displacement of every vertex.  Returns an
    ``(n_unknown, 3)`` array.
    """
    sub = L[rows].tocoo()
    inb = np.asarray(unknown_pos)[sub.col] >= 0
    out = ~inb
    rhs = np.zeros((int(n_unknown), 3))
    if out.any():
        np.add.at(rhs, sub.row[out],
                  -sub.data[out, None] * np.asarray(known, float)[sub.col[out]])
    return rhs, sub, inb


def solve_bridge(V_m, F, part, d_rad_mm, e_r, s_target, w_data, A_v, p):
    """Solve the 3-D displacement of region B; returns mm displacements.

    ``d_rad_mm`` (N,3) is the radial solution on C (zero elsewhere),
    ``s_target`` the radial target (mm), ``w_data`` the continuous data weight.
    """
    Vmm = np.asarray(V_m, float) * 1000.0
    N = len(Vmm)
    A_idx, B_idx, C_idx = part["A_idx"], part["B_idx"], part["C_idx"]
    B = np.zeros(N, bool); B[B_idx] = True
    C = np.zeros(N, bool); C[C_idx] = True
    E = mi.edge_records(F)[0]

    # the C side of the interface plus one more ring, so the displacement
    # gradient of the C solution enters the solve instead of being assumed zero
    ring = _rings(F, B, 3, C)
    ring1 = _rings(F, B, 1, C)
    ext = np.zeros(N, bool)
    ext[B_idx] = True
    ext |= ring
    ext_idx = np.flatnonzero(ext)
    pos = np.full(N, -1, np.int64)
    pos[ext_idx] = np.arange(len(ext_idx))
    nb = len(ext_idx)
    if nb == 0:
        return {"d_mm": np.zeros((N, 3)), "n_unknown": 0, "terms": {}}

    # --- known values on the C side, and the tangent-plane prediction -------
    J = _local_jacobian(Vmm, np.asarray(d_rad_mm, float), np.flatnonzero(ring1), E)
    edge_mask = B[E[:, 0]] & C[E[:, 1]] | B[E[:, 1]] & C[E[:, 0]]
    Eb = E[edge_mask]
    pred_sum = np.zeros((N, 3))
    pred_cnt = np.zeros(N)
    for i, j in Eb:
        if B[i] and C[j]:
            b, c = int(i), int(j)
        else:
            b, c = int(j), int(i)
        G = J.get(c)
        if G is None:
            pred = d_rad_mm[c]
        else:
            pred = d_rad_mm[c] + (Vmm[b] - Vmm[c]) @ G.T
        pred_sum[b] += pred
        pred_cnt[b] += 1
    has_pred = pred_cnt > 0
    pred = np.zeros((N, 3))
    pred[has_pred] = pred_sum[has_pred] / pred_cnt[has_pred, None]

    # --- assemble -----------------------------------------------------------
    L, _M, _Av = nfit.cotan_operators(Vmm, F)
    Lb = nfit.bending_operator(L, A_v, np.full(N, p["bridge_reg_mm"]))

    rows = []
    n_row = 0

    def add_scalar(idx, coef, res, weight):
        """One row per entry: ``weight * (coef_i . d_i) = weight * res_i``.

        The weight has to multiply **both** the coefficients and the right hand
        side.  Scaling only the right hand side changes the target of a single
        constraint from ``res`` to ``weight * res`` (the previous behaviour):
        with one data row the solution was ``weight * res / coef`` and therefore
        moved when the weight changed, which no least-squares weighting may do.
        The caller passes ``sqrt(area * w)``, so the normal equation receives
        ``area * w`` -- the same convention as :func:`add_identity`.
        """
        nonlocal n_row
        idx = np.asarray(idx, np.int64)
        m = len(idx)
        if not m:
            return
        coef = np.asarray(coef, float).reshape(m, 3)
        w = np.asarray(weight, float)
        res = np.asarray(res, float).reshape(m, 1)
        coef_w, res_w = _weighted_scalar_row(coef, res, w)
        r = np.repeat((np.arange(m) + n_row)[:, None], 3, axis=1).ravel()
        c = (pos[idx][:, None] * 3 + np.arange(3)[None, :]).ravel()
        rows.append((r, c, coef_w.ravel(), res_w.ravel()))
        n_row += m

    def add_identity(idx, res, weight):
        """Three rows per entry: ``weight * d_ij = weight * res_ij``."""
        nonlocal n_row
        idx = np.asarray(idx, np.int64)
        m = len(idx)
        if not m:
            return
        res = np.broadcast_to(np.asarray(res, float).reshape(
            1, 3) if np.asarray(res).ndim == 1 else np.asarray(res, float), (m, 3))
        w = np.asarray(weight, float)
        r = (np.arange(m)[:, None] * 3 + np.arange(3)[None, :] + n_row).ravel()
        c = (pos[idx][:, None] * 3 + np.arange(3)[None, :]).ravel()
        rows.append((r, c, w[:, None].repeat(3, axis=1).ravel(),
                     (w[:, None] * res).ravel()))
        n_row += 3 * m

    # data term: only the radial component is constrained (the bridge is free to
    # slide tangentially, which is what makes it a bridge and not a rubber sheet)
    selB = B_idx[w_data[B_idx] > 1e-6]
    if len(selB):
        w = np.sqrt(0.5 * A_v[selB] * w_data[selB])
        add_scalar(selB, e_r[selB],
                   np.clip(s_target[selB], -p["max_d_mm"], p["max_d_mm"]), w)

    # pins on the C side
    selP = np.flatnonzero(ring)
    if len(selP):
        add_identity(selP, d_rad_mm[selP], np.sqrt(p["bridge_pin"] * A_v[selP]))

    # hand-over into A: the displacement must arrive at zero
    na = np.zeros(N, bool)
    for i, j in E:
        if B[i] and not ext[j] and not C[j]:
            na[i] = True
        if B[j] and not ext[i] and not C[i]:
            na[j] = True
    selG = np.flatnonzero(na)
    if len(selG):
        add_identity(selG, np.zeros(3), np.sqrt(p["bridge_handover"] * A_v[selG]))

    # tangent-plane continuity into C
    selT = np.flatnonzero(has_pred)
    if len(selT):
        add_identity(selT, pred[selT], np.sqrt(p["bridge_tangent"] * A_v[selT]))

    # pose prior
    selS = B_idx
    if len(selS):
        add_identity(selS, np.zeros(3), np.sqrt(p["bridge_prior"] * A_v[selS]))

    if not rows:
        return {"d_mm": np.zeros((N, 3)), "n_unknown": 0, "terms": {}}

    R = np.concatenate([r[0] for r in rows])
    Cc = np.concatenate([r[1] for r in rows])
    Vv = np.concatenate([r[2] for r in rows])
    Bb = np.concatenate([r[3] for r in rows])
    M = sparse.coo_matrix((Vv, (R, Cc)), shape=(n_row, 3 * nb)).tocsr()
    rhs = Bb

    # bending rows: keep the rows whose vertex is inside the solve region and
    # carry the known outside values to the right-hand side.  The sign is
    # -L_out d_known (see :func:`bending_known_rhs`).
    Lb = Lb.tocsr()
    rhs_corr, sub, inb = bending_known_rhs(Lb, ext_idx, pos, d_rad_mm, nb)
    rv = sub.row[inb]
    cv = pos[sub.col[inb]]
    dv = sub.data[inb]
    r = (rv[:, None] * 3 + np.arange(3)[None, :]).ravel()
    cc = (cv[:, None] * 3 + np.arange(3)[None, :]).ravel()
    vv = np.repeat(dv, 3)
    Mb = sparse.coo_matrix((vv.ravel(), (r.ravel(), cc.ravel())),
                           shape=(3 * nb, 3 * nb)).tocsr()
    bb = rhs_corr.ravel()
    sol = sparse.vstack([M, Mb]).tocsr()
    allrhs = np.concatenate([rhs, bb])
    d = np.asarray(spsolve((sol.T @ sol).tocsc(), sol.T @ allrhs), float)
    d = np.where(np.isfinite(d), d, 0.0).reshape(nb, 3)

    out = np.zeros((N, 3))
    out[ext_idx] = d
    out[np.asarray(part["A_idx"], np.int64)] = 0.0
    # B is free, the C interior keeps its radial solution
    out[C_idx] = d_rad_mm[C_idx]
    return {"d_mm": out, "n_unknown": int(nb),
            "region": {"B": int(B.sum()), "C_ring": int(ring.sum()),
                       "A_handover": int(len(selG)), "B_tangent": int(len(selT))},
            "rows": {"data": int(3 * len(selB)), "pin": int(3 * len(selP)),
                     "handover": int(3 * len(selG)), "tangent": int(3 * len(selT)),
                     "prior": int(3 * len(selS)), "bending": int(3 * nb)},
            "rhs_bending_outside": float(np.abs(bb).max()),
            "max_d_mm": float(np.linalg.norm(out, axis=1).max()),
            "max_d_B_mm": float(np.linalg.norm(out[B_idx], axis=1).max()) if len(B_idx) else 0.0}


# --------------------------------------------------------------------------
# connection quality of the delivered surface
# --------------------------------------------------------------------------


def face_normals(V_m, F):
    tri = np.asarray(V_m, float)[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]


def dihedral_deg(V_m, F):
    """Signed dihedral angle (deg) across every interior edge."""
    E, inv, counts = mi.edge_records(F)[:3]
    n = face_normals(V_m, F)
    two = np.flatnonzero(counts == 2)
    # directed edge -> face it came from: reconstruct from the face list
    fc = np.concatenate([np.arange(len(F))] * 3)
    order = np.argsort(inv, kind="stable")
    inv_s, fc_s = inv[order], fc[order]
    start = np.searchsorted(inv_s, two)
    f0 = fc_s[start]
    f1 = fc_s[start + 1]
    c = np.einsum("ij,ij->i", n[f0], n[f1])
    ang = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    s = np.einsum("ij,ij->i", np.cross(n[f0], n[f1]),
                  np.asarray(V_m, float)[E[two, 1]] - np.asarray(V_m, float)[E[two, 0]])
    return E[two], ang * np.sign(s)


def connection_metrics(V_in, V_out, F, part, d_mm, e_r):
    """What the bridge actually did to the surface.

    All values are physical (mm, deg, mm/mm) and taken on the *real* surface,
    not on a de-trended relief: the diagonal folds that the previous round
    produced are exactly the kind of thing a relief derivative hides.
    """
    V_in = np.asarray(V_in, float)
    V_out = np.asarray(V_out, float)
    B = np.asarray(part["B"], bool)
    C = np.asarray(part["C"], bool)
    A = np.asarray(part["A"], bool)
    E = mi.edge_records(F)[0]
    Lmm = np.linalg.norm((V_out[E[:, 0]] - V_out[E[:, 1]]) * 1000.0, axis=1)

    # displacement gradient across the bridge, along the surface
    dr = np.einsum("ij,ij->i", d_mm, e_r)
    g = np.abs(dr[E[:, 0]] - dr[E[:, 1]]) / np.maximum(Lmm, 1e-9)
    inB = B[E[:, 0]] & B[E[:, 1]]
    crossB = (B[E[:, 0]] & (A[E[:, 1]] | C[E[:, 1]])) | \
             (B[E[:, 1]] & (A[E[:, 0]] | C[E[:, 0]]))
    inC = C[E[:, 0]] & C[E[:, 1]]

    # dihedral change, inside the bridge and on the two interface lines
    eb, ang_out = dihedral_deg(V_out, F)
    _eb2, ang_in = dihedral_deg(V_in, F)
    change = np.abs(ang_out) - np.abs(ang_in)
    eB = B[eb[:, 0]] & B[eb[:, 1]]
    eX = (B[eb[:, 0]] & (A[eb[:, 1]] | C[eb[:, 1]])) | \
         (B[eb[:, 1]] & (A[eb[:, 0]] | C[eb[:, 0]]))
    rel = np.zeros(len(V_out))
    rel[B] = np.linalg.norm(d_mm[B], axis=1)
    chB = np.abs(change[eB])
    chX = np.abs(change[eX])
    out = {
        "gradient_in_B_mm_per_mm": float(g[inB].max()) if inB.any() else 0.0,
        "gradient_crossing_mm_per_mm": float(g[crossB].max()) if crossB.any() else 0.0,
        "gradient_in_C_mm_per_mm": float(g[inC].max()) if inC.any() else 0.0,
        # the maximum alone is dominated by the shortest edges (a 0.5 mm edge
        # turns a 0.3 mm jump into 0.6 mm/mm), so the percentile is the number
        # the acceptance uses and the maximum is reported next to it
        "gradient_in_B_p95_mm_per_mm": float(np.percentile(g[inB], 95)) if inB.any() else 0.0,
        "gradient_crossing_p95_mm_per_mm": (float(np.percentile(g[crossB], 95))
                                            if crossB.any() else 0.0),
        "gradient_in_C_p95_mm_per_mm": float(np.percentile(g[inC], 95)) if inC.any() else 0.0,
        "crossing_edge_len_p50_mm": float(np.median(Lmm[crossB])) if crossB.any() else 0.0,
        "crossing_edge_len_min_mm": float(Lmm[crossB].min()) if crossB.any() else 0.0,
        "dihedral_change_in_B_deg": float(chB.max()) if eB.any() else 0.0,
        "dihedral_change_crossing_deg": float(chX.max()) if eX.any() else 0.0,
        "crease_change_B_p95_deg": float(np.percentile(chB, 95)) if eB.any() else 0.0,
        "crease_change_crossing_p95_deg": float(np.percentile(chX, 95)) if eX.any() else 0.0,
        "crease_change_B_p50_deg": float(np.percentile(chB, 50)) if eB.any() else 0.0,
        "n_edges_crease_gt10_B": int((chB > 10.0).sum()),
        "n_edges_crease_gt10_crossing": int((chX > 10.0).sum()),
        "dihedral_out_B_p95_deg": float(np.percentile(np.abs(ang_out[eB]), 95)) if eB.any() else 0.0,
        "dihedral_in_B_p95_deg": float(np.percentile(np.abs(ang_in[eB]), 95)) if eB.any() else 0.0,
        "dihedral_out_crossing_p95_deg": float(np.percentile(np.abs(ang_out[eX]), 95)) if eX.any() else 0.0,
        "dihedral_in_crossing_p95_deg": float(np.percentile(np.abs(ang_in[eX]), 95)) if eX.any() else 0.0,
        "n_edges_B": int(eB.sum()), "n_edges_crossing": int(eX.sum()),
        "disp_B_max_mm": float(rel[B].max()) if B.any() else 0.0,
        "disp_A_max_mm": float(rel[A].max()) if A.any() else 0.0,
        "nonmanifold_edges": int(mi.topology_report(V_out / 1000.0, F)["nonmanifold_edges"]),
    }
    # where the worst crease change sits, in the design frame -- a number without
    # a location cannot be acted on
    if eB.any():
        k = int(np.argmax(chB))
        out["worst_crease_B"] = {"edge_mm": float(np.linalg.norm(
            (V_out[eb[k, 0]] - V_out[eb[k, 1]]) * 1000.0)),
            "in_deg": float(ang_in[eB][k]), "out_deg": float(ang_out[eB][k]),
            "edge": [int(eb[k, 0]), int(eb[k, 1])]}
    if eX.any():
        k = int(np.argmax(chX))
        out["worst_crease_crossing"] = {"in_deg": float(ang_in[eX][k]),
                                        "out_deg": float(ang_out[eX][k]),
                                        "edge": [int(eb[k, 0]), int(eb[k, 1])]}
    return out
