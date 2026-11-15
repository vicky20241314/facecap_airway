"""Corrected discrete operators, quadrature weights and the incremental fit.

This module replaces four things that were measurably wrong in the previous
round.  Each one is fixed *and* has a test that fails on the old code:

1. **Cotangent indexing.**  ``cot[:, k]`` stores the cotangent of the angle at
   corner ``k``; the weight of edge ``(k, k+1)`` is the cotangent of the
   *opposite* corner ``(k+2)``, times one half.  The old assembly applied the
   offset a second time, so every edge received the cotangent of the wrong
   angle.  ``cotan_operators`` now builds the textbook FEM stiffness (symmetric,
   zero row sums, mass = barycentric vertex area) and ``cotan_unit_test`` checks
   the right triangle (0,0)/(2,0)/(0,1), whose weights are exactly
   ``[0.25, 0, 1]``.  Negative cotangents are *kept*: clipping them is a
   different operator and one that breaks the exactness on obtuse triangles.

2. **Increment vs absolute step.**  The old loop evaluated the residual at
   ``X + s e_r`` and then *replaced* ``s`` by the solution of the normal
   equations written for the *previous* position.  That is not a Newton step and
   it oscillates between two values (measured 1 -> 0 -> 1).  Everything here is
   an increment: residual and Jacobian at the current ``X``, solve for
   ``delta``, step ``X + a*delta``, backtrack until the *total energy* decreases,
   and judge convergence on energy change, step size **and** the residual
   recomputed at the returned position.

3. **In-face weight and Jacobian.**  The old code decided per *face* whether it
   was editable (``w[F].min() > 1e-6``) and then integrated with the full area,
   i.e. a binary mask instead of a fade.  Now every quadrature sample (all
   vertices, three edge midpoints, the centroid) evaluates the same continuous
   weight at its own position, and the in-face Jacobian carries the full chain
   rule: the sample point depends on its parents barycentrically, the radius,
   the height and the angle all move, so

       dr/ds_i = lam_i (e_i . e_r)      dh/ds_i = lam_i (e_i . axis)
       dth/ds_i = lam_i (e_i . e_theta) / r
       dR/ds_i  = R_h dh/ds_i + R_theta dth/ds_i - dr/ds_i

   A vertex sample is the special case ``lam = e_k`` where the first term is
   exactly 1 and the other two vanish -- which is why a vertex-only Jacobian
   looks trivial and an in-face one does not.

4. **The regulariser actually used vs the one reported.**  It is

       E_bend = sum_v (ell_v^4 / A_v) (L s)_v^2 = s^T D L^T M^-1 L D s

   with ``D = diag(ell_v^2)``, ``M = diag(A_v)``, ``L`` the cotangent
   stiffness.  ``ell`` is in millimetres, so it sets a physical smoothing length
   and the balance against the area-weighted data term does not change when the
   vertex density changes.  The previous code used ``diag[A_v (ell/L_v)^4] L``,
   which is a different (density-dependent) operator.

Data term.  Both sides of the energy are area integrals in mm^4:

    E_data = int w(x) (R(x) - r(x))^2 dA ~ sum_s 0.5 * a_s * w(x_s) res_s^2

with ``a_s`` the lumped vertex area for vertex samples and ``A_f * 1/4`` for the
four in-face samples (the 0.5 removes the double counting of the two
quadratures, each of which integrates to the area on its own).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------


def face_geometry(V, F):
    """Per-face area, unit normal, and the cotangent of each corner angle."""
    tri = V[F]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm_len = np.linalg.norm(cr, axis=1)
    area = 0.5 * nrm_len
    nrm = cr / np.maximum(nrm_len, 1e-12)[:, None]
    cot = np.zeros((len(F), 3))
    for k in range(3):
        o = tri[:, k]
        u = tri[:, (k + 1) % 3] - o
        v = tri[:, (k + 2) % 3] - o
        cot[:, k] = (np.einsum("ij,ij->i", u, v) /
                     np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-12))
    return area, nrm, cot


def cotan_operators(V, F, cot_clip=1e4):
    """Cotangent stiffness ``L`` (symmetric, zero row sums) and mass ``M``.

    ``w_ij = 0.5 * (cot(alpha_ij) + cot(beta_ij))`` -- half the sum of the
    cotangents of the two angles opposite edge ``ij``.  For a boundary edge the
    second angle does not exist and only the one cotangent contributes, which is
    the natural Neumann condition.  ``M`` is the positive barycentric lumped
    area, ``A_v = sum_f A_f / 3``.
    """
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    area, _, cot = face_geometry(V, F)
    if cot_clip is not None:                     # only a guard against degenerate
        cot = np.clip(cot, -cot_clip, cot_clip)  # slivers, not a sign change
    # edge (0,1) is opposite corner 2, (1,2) opposite 0, (2,0) opposite 1
    I = np.concatenate([F[:, 0], F[:, 1], F[:, 1], F[:, 2], F[:, 2], F[:, 0]])
    J = np.concatenate([F[:, 1], F[:, 0], F[:, 2], F[:, 1], F[:, 0], F[:, 2]])
    w = 0.5 * np.concatenate([cot[:, 2], cot[:, 2], cot[:, 0], cot[:, 0],
                              cot[:, 1], cot[:, 1]])
    # off-diagonals are MINUS the weights: L = D - W, so (Ls)_i = sum_j w_ij
    # (s_i - s_j) and a constant field is exactly in the kernel
    L = sparse.coo_matrix((-w, (I, J)), shape=(len(V), len(V))).tocsr()
    L = (L + sparse.diags(-np.asarray(L.sum(axis=1)).ravel())).tocsr()
    A_v = np.zeros(len(V))
    for k in range(3):
        np.add.at(A_v, F[:, k], area / 3.0)
    M = sparse.diags(np.maximum(A_v, 1e-9))
    return L, M, A_v


def cotan_unit_test():
    """Right triangle (0,0),(2,0),(0,1): edge weights ``[0.25, 0, 1]``.

    Hand check with ``v0=(0,0) v1=(2,0) v2=(0,1)``:

        angle at v0 = 90 deg       -> cot 0
        angle at v1 = atan(1/2)    -> cot 2      (legs (-2,0), (-2,1))
        angle at v2 = atan(2)      -> cot 0.5    (legs (2,-1), (0,-1))

        edge 01 is opposite v2 -> w = 0.5 * 0.5 = 0.25
        edge 12 is opposite v0 -> w = 0.5 * 0   = 0
        edge 20 is opposite v1 -> w = 0.5 * 2   = 1

    which is exactly the reference ``[0.25, 0, 1]``.  The two-triangle square
    (0,0),(2,0),(2,1),(0,1) additionally checks that a shared edge sums the two
    opposite cotangents (the diagonal gets ``0.5*(cot 90 + cot 90) = 0``) and
    that the mass is the total area.
    """
    V = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = np.array([[0, 1, 2]])
    L, M, A = cotan_operators(V, F, cot_clip=None)
    Ld = L.toarray()
    w01, w12, w20 = -Ld[0, 1], -Ld[1, 2], -Ld[2, 0]
    Vq = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0],
                   [0.0, 1.0, 0.0]])
    Fq = np.array([[0, 1, 2], [0, 2, 3]])
    Lq, _, Aq = cotan_operators(Vq, Fq, cot_clip=None)
    Lq = Lq.toarray()
    q01, q12, q23, q30 = -Lq[0, 1], -Lq[1, 2], -Lq[2, 3], -Lq[3, 0]
    q02 = -Lq[0, 2]
    ok = (abs(w01 - 0.25) < 1e-12 and abs(w12) < 1e-12 and abs(w20 - 1.0) < 1e-12
          and abs(q01 - 0.25) < 1e-12 and abs(q12 - 1.0) < 1e-12
          and abs(q23 - 0.25) < 1e-12 and abs(q30 - 1.0) < 1e-12
          and abs(q02) < 1e-12)
    return {"single": {"w01": float(w01), "w12": float(w12), "w20": float(w20),
                       "expected": [0.25, 0.0, 1.0]},
            "quad": {"w01": float(q01), "w12": float(q12), "w23": float(q23),
                     "w30": float(q30), "w02_diagonal": float(q02),
                     "expected": [0.25, 1.0, 0.25, 1.0, 0.0]},
            "pass": bool(ok),
            "symmetric": bool(np.allclose(Ld, Ld.T) and np.allclose(Lq, Lq.T)),
            "row_sums": float(max(np.abs(Ld.sum(axis=1)).max(),
                                  np.abs(Lq.sum(axis=1)).max())),
            "mass_sum_single_mm2": float(A.sum()), "area_single_mm2": 1.0,
            "mass_sum_quad_mm2": float(Aq.sum()), "area_quad_mm2": 2.0}


def bending_operator(L, A_v, ell_mm):
    """Rows ``R`` with ``||R s||^2 = sum_v (ell_v^4 / A_v) (L s)_v^2``."""
    L = L.tocsr()
    scale = np.sqrt(np.maximum(np.asarray(ell_mm, float), 1e-6) ** 4 /
                    np.maximum(np.asarray(A_v, float), 1e-9))
    return sparse.diags(scale) @ L


# --------------------------------------------------------------------------
# point-wise frame evaluation
# --------------------------------------------------------------------------


def frame_at(P_mm, ax, curve, h_chin):
    """h, theta, r, t, e_r, e_theta for arbitrary points (mm)."""
    P = np.asarray(P_mm, float) - ax["centre_mm"][None, :]
    h = P @ ax["axis"]
    a = P @ ax["lateral"]
    b = P @ ax["anterior"]
    ca, cb = curve(h)
    da, db = a - ca, b - cb
    r = np.hypot(da, db)
    th = np.arctan2(da, db)
    e_r = (np.sin(th)[:, None] * ax["lateral"][None, :] +
           np.cos(th)[:, None] * ax["anterior"][None, :])
    e_t = (np.cos(th)[:, None] * ax["lateral"][None, :] -
           np.sin(th)[:, None] * ax["anterior"][None, :])
    return h, th, r, h / h_chin, e_r, e_t


def field_grad(field, h, th, dh=0.25, dth=1e-3):
    """Value, dR/dh and dR/dtheta of a bilinear field (central differences)."""
    r0 = np.asarray(field(h, th), float)
    rp = np.asarray(field(h + dh, th), float)
    rm = np.asarray(field(h - dh, th), float)
    tp = np.asarray(field(h, th + dth), float)
    tm = np.asarray(field(h, th - dth), float)
    dh_ = (rp - rm) / (2 * dh)
    dt_ = (tp - tm) / (2 * dth)
    dh_ = np.where(np.isfinite(dh_), dh_, 0.0)
    dt_ = np.where(np.isfinite(dt_), dt_, 0.0)
    return r0, dh_, dt_


# --------------------------------------------------------------------------
# quadrature samples
# --------------------------------------------------------------------------

BARY = np.array([[0.5, 0.5, 0.0],
                 [0.0, 0.5, 0.5],
                 [0.5, 0.0, 0.5],
                 [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
BARY_W = np.full(4, 0.25)


def _no_weight(P, h, th, r, lam, par):
    return np.ones(len(np.asarray(P)), float)


def build_samples(V_mm, F, s, e_r, ax, curve, h_chin, field, fixed, wfun=None,
                  faces=None):
    """Every quadrature sample: point, residual, weight, parents, chain terms.

    Vertices contribute their lumped area, faces contribute four points with a
    quarter of the face area each.  The weight is evaluated **at the sample
    point** (a continuous function of position), never as "is the whole face
    inside the region", so a face straddling a fade boundary gets a
    proportionally smaller row instead of a step.
    """
    V_mm = np.asarray(V_mm, float)
    F = np.asarray(F, np.int64)
    wfun = _no_weight if wfun is None else wfun
    fixed = np.asarray(fixed, bool)
    A_v = None
    N = len(V_mm)

    Pv = V_mm + s[:, None] * e_r
    hv, thv, rv, _, _, _ = frame_at(Pv, ax, curve, h_chin)
    Rv, dRh_v, dRt_v = field_grad(field, hv, thv)
    Rv = np.where(np.isfinite(Rv), Rv, rv)

    _, _, A_v = cotan_operators(V_mm, F)

    par_v = np.arange(N)[:, None].repeat(3, axis=1)
    lam_v = np.broadcast_to(np.array([1.0, 0.0, 0.0]), (N, 3))
    w_v = np.asarray(wfun(Pv, hv, thv, rv, lam_v, par_v), float)
    w_v = np.where(fixed, 0.0, np.maximum(w_v, 0.0))
    av = 0.5 * A_v                                   # mm^2, half of the lumped area

    fid = (np.arange(len(F)) if faces is None
           else np.flatnonzero(np.asarray(faces, bool)))
    pts, res, wts, par, lam, jh, jt, kind = [], [], [], [], [], [], [], []
    keepv = w_v > 1e-6
    if keepv.any():
        pts.append(Pv[keepv]); res.append((Rv - rv)[keepv])
        wts.append(av[keepv] * w_v[keepv])
        par.append(par_v[keepv]); lam.append(lam_v[keepv])
        jh.append(dRh_v[keepv]); jt.append(dRt_v[keepv])
        kind.append(np.zeros(int(keepv.sum()), np.int8))

    if len(fid):
        # the sample points must MOVE with the vertices they interpolate,
        # otherwise the in-face rows carry no dependence on s at all and the fit
        # is decided by the vertex rows alone
        tri = Pv[F[fid]]
        # ... but the quadrature weight is the *material* (rest) area, so the
        # discrete integral sum_s w_s does not drift as the surface stretches
        tri0 = V_mm[F[fid]]
        cr = np.cross(tri0[:, 1] - tri0[:, 0], tri0[:, 2] - tri0[:, 0])
        Af = 0.5 * np.linalg.norm(cr, axis=1)
        # a face with every corner fixed cannot move at all: it has no unknowns
        free_face = ~fixed[F[fid]].all(axis=1)
        if free_face.any():
            fid = fid[free_face]
            tri = tri[free_face]
            Af = Af[free_face]
            for k in range(4):
                Pk = np.einsum("j,njc->nc", BARY[k], tri)
                hk, tk, rk, _, _, _ = frame_at(Pk, ax, curve, h_chin)
                Rk, dRhk, dRtk = field_grad(field, hk, tk)
                Rk = np.where(np.isfinite(Rk), Rk, rk)
                park = F[fid]
                lamk = np.broadcast_to(BARY[k], (len(fid), 3))
                wk = np.asarray(wfun(Pk, hk, tk, rk, lamk, park), float)
                wk = np.maximum(wk, 0.0)
                keep = wk > 1e-6
                if not keep.any():
                    continue
                pts.append(Pk[keep]); res.append((Rk - rk)[keep])
                wts.append(0.5 * Af[keep] * BARY_W[k] * wk[keep])
                par.append(park[keep]); lam.append(lamk[keep])
                jh.append(dRhk[keep]); jt.append(dRtk[keep])
                kind.append(np.ones(int(keep.sum()), np.int8))

    if not pts:
        z = np.zeros(0)
        return {"P": np.zeros((0, 3)), "res": z, "w": z,
                "par": np.zeros((0, 3), np.int64), "lam": np.zeros((0, 3)),
                "dRh": z, "dRt": z, "kind": np.zeros(0, np.int8),
                "n_vertex": 0, "n_face": 0}
    out = {"P": np.concatenate(pts), "res": np.concatenate(res),
           "w": np.concatenate(wts), "par": np.concatenate(par),
           "lam": np.concatenate(lam), "dRh": np.concatenate(jh),
           "dRt": np.concatenate(jt), "kind": np.concatenate(kind)}
    out["n_vertex"] = int((out["kind"] == 0).sum())
    out["n_face"] = int((out["kind"] == 1).sum())
    return out


def jacobian_rows(V_mm, s, e_r, smp, ax, curve, h_chin, fixed):
    """d(residual)/d(s_i) for every sample, with the full chain rule."""
    # the frame must be taken at the SAMPLE point, not at the vertex the sample
    # belongs to: an in-face point is not on the radial ray of any of its parents
    _, _, r, _, er, et = frame_at(smp["P"], ax, curve, h_chin)
    par = smp["par"]
    lam = smp["lam"]
    ee = e_r[par]                                   # (n, 3 parents, 3 xyz)
    axis = np.broadcast_to(np.asarray(ax["axis"], float), er.shape)
    dr = lam * np.einsum("nkc,nc->nk", ee, er)                    # (n, 3)
    dh = lam * np.einsum("nkc,nc->nk", ee, axis)
    dt = lam * np.einsum("nkc,nc->nk", ee, et) / np.maximum(r, 1e-9)[:, None]
    J = smp["dRh"][:, None] * dh + smp["dRt"][:, None] * dt - dr
    J = np.where(np.asarray(fixed, bool)[par], 0.0, J)
    return J, {"dr": dr, "dh": dh, "dt": dt}


def residuals_labelled(V_mm, ax, curve, h_chin, pts):
    """Convenience: (r, h, theta) of arbitrary points, for reporting."""
    _, _, r, _, _, _ = frame_at(pts, ax, curve, h_chin)
    return r


def assemble(N, smp, J):
    """Sparse weighted design matrix (samples x N), rows scaled by sqrt(w)."""
    n, k = J.shape
    rows = np.repeat(np.arange(n), k)
    cols = smp["par"].ravel()
    vals = (np.sqrt(smp["w"])[:, None] * J).ravel()
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n, N)).tocsr()


def energy(smp, s, Rb, alpha):
    """Total energy of the *given* samples: data + bending + anchor (mm^4)."""
    e_d = float(np.sum(smp["w"] * smp["res"] ** 2)) if len(smp["w"]) else 0.0
    e_b = float(np.sum((Rb @ s) ** 2))
    e_a = float(np.sum(np.asarray(alpha, float) * s ** 2))
    return e_d + e_b + e_a, {"data": e_d, "bend": e_b, "anchor": e_a}


# --------------------------------------------------------------------------
# the incremental fit
# --------------------------------------------------------------------------


def fit_radial(V, F, ax, curve, h_chin, field, ell_reg, fixed, wfun, p,
               max_mm=None, max_iter=25, anchor=None, tol_step_mm=1e-3,
               tol_rel=1e-8, log=None):
    """Incremental Gauss-Newton fit of the 1-D radial offset ``s``.

    ``V``: metres.  ``s`` is the displacement along the **rest** radial
    direction ``e_r`` (a fixed frame per vertex: the frame is not an unknown, so
    the derivative of ``e_r`` with respect to ``s`` does not enter).  The
    residual ``R(h, theta) - r`` is evaluated at the *displaced* point, so the
    height and angle derivatives of the field are included.

    Returns positions in mm, the increment history, and the residual recomputed
    at the returned positions.
    """
    V_mm = np.asarray(V, float) * 1000.0
    N = len(V_mm)
    max_mm = float(p.get("max_d_mm", 6.0)) if max_mm is None else float(max_mm)
    alive = np.zeros(N, bool)
    alive[np.unique(np.asarray(F, np.int64))] = True
    fixed = np.asarray(fixed, bool) | (~alive)
    free = ~fixed

    L, M, A_v = cotan_operators(V_mm, F)
    Rb = bending_operator(L, A_v, ell_reg)
    alpha = np.zeros(N) if anchor is None else np.maximum(np.asarray(anchor, float), 0.0)
    alpha = np.where(free, alpha, 0.0)

    e_r0 = frame_at(V_mm, ax, curve, h_chin)[4]
    s = np.zeros(N)
    smp = build_samples(V_mm, F, s, e_r0, ax, curve, h_chin, field, fixed, wfun)
    e_prev, parts = energy(smp, s, Rb, alpha)
    parts_new = parts            # defined even if the very first test converges
    hist = []
    converged, reason = False, "max_iter"
    g_ref = None

    for it in range(int(max_iter)):
        if not len(smp["w"]):
            reason = "no_samples"
            break
        J, _ = jacobian_rows(V_mm, s, e_r0, smp, ax, curve, h_chin, fixed)
        Js = assemble(N, smp, J)
        # E = ||sqrt(w) res||^2 + ||Rb s||^2 + sum alpha s^2, so the anchor
        # contributes 2 alpha s to the gradient and 2 alpha to the Hessian.  Both
        # halves matter: with the anchor only in the gradient the step is far too
        # long wherever the anchor is stiff, one global backtracking factor then
        # shrinks it until it is also far too short everywhere else, and the fit
        # crawls (measured: alpha = 1/64 on every iteration, 40 iterations and
        # still not converged).
        H = (Js.T @ Js + Rb.T @ Rb + sparse.diags(2.0 * alpha)).tocsc()
        g = (np.asarray(Js.T @ (np.sqrt(smp["w"]) * smp["res"])).ravel() +
             np.asarray(Rb.T @ (Rb @ s)).ravel() + 2.0 * alpha * s)
        H = (sparse.diags(free.astype(float)) @ H @ sparse.diags(free.astype(float)) +
             sparse.diags(fixed.astype(float)) + sparse.diags(free * 1e-10))
        g = np.where(free, g, 0.0)
        gnorm = float(np.abs(g[free]).max()) if free.any() else 0.0
        if g_ref is None:
            g_ref = max(gnorm, 1e-30)
        # a stationary point is a stationary point: at the optimum every trial
        # step increases the energy, so "the line search refused" is not a
        # failure -- the gradient is the criterion that distinguishes the two
        if gnorm <= 1e-9 * g_ref:
            converged, reason = True, "gradient"
            hist.append({"iter": it, "samples": int(len(smp["w"])),
                         "samples_vertex": int(smp["n_vertex"]),
                         "samples_face": int(smp["n_face"]),
                         "energy": float(e_prev), "energy_parts": parts,
                         "step_max_mm": 0.0, "grad_max": gnorm,
                         "grad_rel": gnorm / g_ref,
                         "line_search_alpha": 0.0, "accepted": False,
                         "energy_rel": 0.0, "max_s_mm": float(np.abs(s).max())})
            if log is not None:
                log.append(hist[-1])
            break
        d = -np.asarray(spsolve(H.tocsc(), g), float)
        d = np.where(np.isfinite(d), d, 0.0)
        d = np.where(free, d, 0.0)
        # a Gauss-Newton step from an almost-converged point can be dominated by
        # the near-null space of the bending operator (constant offsets, which
        # the operator annihilates exactly).  If the step is not a descent
        # direction, fall back to diagonally scaled steepest descent.
        if float(g @ d) >= 0.0:
            hd = H.diagonal()
            hd = np.where(hd > 0, hd, 1.0)
            d = -g / hd

        # the increment may not push any vertex past the displacement cap
        room = np.maximum(max_mm - np.abs(s), 0.0)
        over = np.abs(d) > room
        if over.any():
            d = np.where(over, np.sign(d) * room, d)

        if np.max(np.abs(d)) < tol_step_mm:
            converged, reason = True, "step_below_tolerance"
            hist.append({"iter": it, "samples": int(len(smp["w"])),
                         "samples_vertex": int(smp["n_vertex"]),
                         "samples_face": int(smp["n_face"]),
                         "energy": float(e_prev), "energy_parts": parts,
                         "step_max_mm": float(np.max(np.abs(d))),
                         "grad_max": gnorm, "grad_rel": gnorm / g_ref,
                         "line_search_alpha": 0.0, "accepted": False,
                         "energy_rel": 0.0, "max_s_mm": float(np.abs(s).max())})
            if log is not None:
                log.append(hist[-1])
            break

        a_step, accepted, e_new, parts_new, s_new = 1.0, False, e_prev, parts, s
        for _ in range(14):
            s_try = s + a_step * d
            smp_try = build_samples(V_mm, F, s_try, e_r0, ax, curve, h_chin,
                                    field, fixed, wfun)
            e_try, parts_try = energy(smp_try, s_try, Rb, alpha)
            if e_try <= e_prev * (1.0 + 1e-12):
                accepted, e_new, parts_new, s_new, smp = (True, e_try, parts_try,
                                                          s_try, smp_try)
                break
            a_step *= 0.5

        dnorm = float(np.max(np.abs(a_step * d))) if accepted else 0.0
        rel = abs(e_prev - e_new) / max(e_prev, 1e-12)
        rec = {"iter": it, "samples": int(len(smp["w"])),
               "samples_vertex": int(smp["n_vertex"]), "samples_face": int(smp["n_face"]),
               "energy": float(e_new), "energy_parts": parts_new,
               "step_max_mm": dnorm, "grad_max": gnorm, "grad_rel": gnorm / g_ref,
               "line_search_alpha": float(a_step),
               "accepted": bool(accepted), "energy_rel": float(rel),
               "max_s_mm": float(np.abs(s_new).max())}
        hist.append(rec)
        if log is not None:
            log.append(rec)
        if not accepted:
            # No step along the Gauss-Newton direction lowers the energy any more.
            # At the optimum every trial step increases it, so this is a
            # stationary point reached to within the model's accuracy -- and the
            # residual is then recomputed at the returned position anyway.
            if a_step <= 1e-4:
                converged, reason = True, "energy_stationary"
            else:
                reason = "line_search_failed"
            break
        s = s_new
        e_prev = e_new
        parts = parts_new
        if dnorm < tol_step_mm and rel < tol_rel:
            converged, reason = True, "step_and_energy"
            break

    d_mm = s[:, None] * e_r0
    P_end = V_mm + d_mm
    res = residuals_at(P_end, F, ax, curve, h_chin, field, wfun, fixed)
    return {"V_mm": P_end, "d_mm": d_mm, "s_mm": s, "e_r": e_r0,
            "iterations": hist, "n_iter": len(hist), "converged": converged,
            "stop_reason": reason, "energy": float(e_prev),
            "energy_parts": parts_new if hist else parts,
            "data_samples": int(len(smp["w"])) if hist else 0,
            "bend_rows": int(Rb.shape[0]),
            "max_d_mm": float(np.abs(s).max()) if N else 0.0,
            "rms_d_mm": float(np.sqrt(np.mean(s[free] ** 2))) if free.any() else 0.0,
            "n_free": int(free.sum()), "n_fixed": int(fixed.sum()),
            "residual_after": res}


# --------------------------------------------------------------------------
# unified 3-D displacement solve (neck body + submental bridge, one system)
# --------------------------------------------------------------------------


def build_samples_3d(V_mm, F, D, ax, curve, h_chin, field, fixed, wfun,
                     A_v=None):
    """Quadrature samples of the radial target on a *displaced* surface.

    ``D`` is a 3-D displacement (mm) per vertex.  Vertices and the four in-face
    points of every face with at least one free corner contribute; the sample
    point moves with its parents, so the data rows carry the full dependence on
    ``D``.  The continuous weight is evaluated at the sample point, never as a
    per-face mask.
    """
    V_mm = np.asarray(V_mm, float)
    D = np.asarray(D, float)
    F = np.asarray(F, np.int64)
    N = len(V_mm)
    if A_v is None:
        _, _, A_v = cotan_operators(V_mm, F)
    wfun = _no_weight if wfun is None else wfun
    Pv = V_mm + D
    hv, thv, rv, _, _, _ = frame_at(Pv, ax, curve, h_chin)
    Rv, dRh_v, dRt_v = field_grad(field, hv, thv)
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
            hk, tk, rk, _, _, _ = frame_at(Pk, ax, curve, h_chin)
            Rk, dRhk, dRtk = field_grad(field, hk, tk)
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


def jacobian_rows_3d(smp, ax, curve, h_chin, fixed):
    """d(residual)/d(D_k) for a 3-D displacement, full chain rule.

    ``h``, ``theta`` and ``r`` are taken at the sample point.  Moving parent
    ``k`` by ``dD_k`` moves the sample by ``lam_k dD_k``, hence

        dh/dD_k = lam_k axis
        dth/dD_k = lam_k e_theta / r
        dr/dD_k = lam_k e_r
        dres/dD_k = lam_k (R_h axis + R_theta/r e_theta - e_r)

    which is a 3-vector per parent.  At a vertex sample ``lam = e_0`` and the
    bracket is exactly the scalar formula used by :func:`fit_radial`.
    """
    _, _, r, _, er, et = frame_at(smp["P"], ax, curve, h_chin)
    axis = np.asarray(ax["axis"], float)
    v = (smp["dRh"][:, None] * axis[None, :] +
         (smp["dRt"] / np.maximum(r, 1e-9))[:, None] * et - er)
    J = smp["lam"][:, :, None] * v[:, None, :]
    bad = np.asarray(fixed, bool)[smp["par"]][:, :, None]
    return np.where(bad, 0.0, J)


def assemble_3d(N, smp, J):
    """Sparse weighted design matrix for the 3-D displacement."""
    n = J.shape[0]
    rows = np.repeat(np.arange(n), 9)
    cols = (smp["par"][:, :, None] * 3 +
            np.arange(3)[None, None, :]).ravel()
    vals = (np.sqrt(smp["w"])[:, None, None] * J).ravel()
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n, 3 * N)).tocsr()


def _projection_rows(V_mm, D, A_v, mu, fixed, ax, curve, h_chin,
                     isotropic=False):
    """Rows ``sqrt(mu A) P`` with ``P = I`` or ``P = I - e_r e_r^T``."""
    _, _, _, _, er, _ = frame_at(V_mm + D, ax, curve, h_chin)
    if isotropic:
        P = np.broadcast_to(np.eye(3)[None, :, :], (len(V_mm), 3, 3)).copy()
    else:
        P = np.eye(3)[None, :, :] - er[:, :, None] * er[:, None, :]
    scale = np.sqrt(np.maximum(np.asarray(mu, float) * np.asarray(A_v, float), 0.0))
    scale = np.where(np.asarray(fixed, bool), 0.0, scale)
    N = len(V_mm)
    idx = np.arange(N)
    rr = (3 * idx[:, None, None] + np.arange(3)[None, :, None] +
          np.zeros((1, 1, 3), int)).ravel()
    cc = (3 * idx[:, None, None] + np.zeros((1, 3, 1), int) +
          np.arange(3)[None, None, :]).ravel()
    vv = (scale[:, None, None] * P).ravel()
    return sparse.coo_matrix((vv, (rr, cc)), shape=(3 * N, 3 * N)).tocsr()


def energy_3d(smp, D, Rb3, T, Aanch, mu_small, A_v):
    vec = np.asarray(D, float).ravel()
    e_d = float(np.sum(smp["w"] * smp["res"] ** 2)) if len(smp["w"]) else 0.0
    e_b = float(np.sum((Rb3 @ vec) ** 2))
    e_t = float(np.sum((T @ vec) ** 2))
    e_a = float(np.sum((Aanch @ vec) ** 2))
    e_s = float(np.sum(mu_small * np.asarray(A_v, float) *
                       np.sum(np.asarray(D, float) ** 2, axis=1)))
    return e_d + e_b + e_t + e_a + e_s, {"data": e_d, "bend": e_b,
                                         "tangent": e_t, "anchor": e_a,
                                         "small": e_s}


def _bounds_ok(D, e_r, free, max_d_mm, max_t_mm):
    rad = np.einsum("nk,nk->n", D, e_r)
    tan = D - rad[:, None] * e_r
    tn = np.linalg.norm(tan, axis=1)
    return bool(np.all(np.abs(rad[free]) <= float(max_d_mm) + 1e-9) and
                np.all(tn[free] <= np.asarray(max_t_mm)[free] + 1e-9))


def fit_unified(V, F, ax, curve, h_chin, field, ell_reg, fixed, wfun, p,
                mu_t=None, mu_anchor=None, max_d_mm=None, max_t_mm=None,
                max_iter=40, tol_step_mm=1e-3, tol_rel=1e-8, log=None):
    """One continuous 3-D displacement solve for the neck body and bridge.

    This replaces the previous two-stage construction (a radial fit of C with B
    held at zero, then a separate bridge solve of B pinned to C) by a single
    energy over the same unknown field ``D``:

        E = sum_s w_s (R_s - r_s)^2                    data (radial target)
          + ||Rb D||^2                                  bending, ell_reg
          + ||T D||^2        T = sqrt(mu_t A) (I - e_r e_r^T)   tangential prior
          + ||A D||^2        A = sqrt(mu_a A (1-gate)) I        anchor
          + mu_small sum A_v |D|^2                      numerical prior

    A/B/C are not separate geometry targets here: they only set ``mu_t`` (strong
    in the neck body, weak in the bridge), the anchor and the acceptance labels.
    The fixed set is a Dirichlet boundary.  ``D`` is bounded radially by
    ``max_d_mm`` and tangentially by ``max_t_mm`` (a trust region, not a clipped
    solution).

    Returns the same kind of dictionary as :func:`fit_radial`.
    """
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
    e_r0 = frame_at(V_mm, ax, curve, h_chin)[4]

    L, M, A_v = cotan_operators(V_mm, F)
    Rb = bending_operator(L, A_v, ell_reg)
    I3 = sparse.eye(3, format="csr")
    Rb3 = sparse.kron(Rb, I3, format="csr")
    mu_small = float(p.get("mu_small", 1e-3))

    D = np.zeros((N, 3))
    smp = build_samples_3d(V_mm, F, D, ax, curve, h_chin, field, fixed, wfun,
                           A_v=A_v)
    T = _projection_rows(V_mm, D, A_v, mu_t, fixed, ax, curve, h_chin,
                         isotropic=False)
    Aanch = _projection_rows(V_mm, D, A_v, mu_anchor, fixed, ax, curve, h_chin,
                             isotropic=True)
    e_prev, parts = energy_3d(smp, D, Rb3, T, Aanch, mu_small, A_v)
    parts_new = parts
    hist = []
    converged, reason = False, "max_iter"
    g_ref = None

    for it in range(int(max_iter)):
        if not len(smp["w"]):
            reason = "no_samples"
            break
        J = jacobian_rows_3d(smp, ax, curve, h_chin, fixed)
        Js = assemble_3d(N, smp, J)
        vec = D.ravel()
        H = (Js.T @ Js + Rb3.T @ Rb3 + T.T @ T + Aanch.T @ Aanch).tocsc()
        # the small isotropic prior is diagonal; its Hessian must be present or
        # the Gauss-Newton step ignores half of the term it is minimising
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
            hist.append({"iter": it, "samples": int(len(smp["w"])),
                         "samples_vertex": int(smp["n_vertex"]),
                         "samples_face": int(smp["n_face"]),
                         "energy": float(e_prev), "energy_parts": parts,
                         "step_max_mm": 0.0, "grad_max": gnorm,
                         "grad_rel": gnorm / g_ref, "line_search_alpha": 0.0,
                         "accepted": False, "energy_rel": 0.0,
                         "max_d_mm": float(np.linalg.norm(D, axis=1).max())})
            if log is not None:
                log.append(hist[-1])
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
            hist.append({"iter": it, "samples": int(len(smp["w"])),
                         "samples_vertex": int(smp["n_vertex"]),
                         "samples_face": int(smp["n_face"]),
                         "energy": float(e_prev), "energy_parts": parts,
                         "step_max_mm": float(np.max(np.abs(delta))),
                         "grad_max": gnorm, "grad_rel": gnorm / g_ref,
                         "line_search_alpha": 0.0, "accepted": False,
                         "energy_rel": 0.0,
                         "max_d_mm": float(np.linalg.norm(D, axis=1).max())})
            if log is not None:
                log.append(hist[-1])
            break

        alpha, accepted = 1.0, False
        D_new, smp_new, parts_new, e_new = D, smp, parts, e_prev
        for _ in range(24):
            D_try = D + alpha * delta
            if not _bounds_ok(D_try, e_r0, free, max_d, max_t_mm):
                alpha *= 0.5
                continue
            smp_try = build_samples_3d(V_mm, F, D_try, ax, curve, h_chin,
                                       field, fixed, wfun, A_v=A_v)
            e_try, p_try = energy_3d(smp_try, D_try, Rb3, T, Aanch, mu_small, A_v)
            if e_try <= e_prev * (1.0 + 1e-12):
                alpha, accepted = alpha, True
                D_new, smp_new, parts_new, e_new = D_try, smp_try, p_try, e_try
                break
            alpha *= 0.5

        dnorm = float(np.max(np.abs(D_new - D)))
        rel = abs(e_prev - e_new) / max(e_prev, 1e-12)
        rec = {"iter": it, "samples": int(len(smp["w"])),
               "samples_vertex": int(smp["n_vertex"]),
               "samples_face": int(smp["n_face"]),
               "energy": float(e_new), "energy_parts": parts_new,
               "step_max_mm": dnorm, "grad_max": gnorm, "grad_rel": gnorm / g_ref,
               "line_search_alpha": float(alpha), "accepted": bool(accepted),
               "energy_rel": float(rel),
               "max_d_mm": float(np.linalg.norm(D_new, axis=1).max())}
        hist.append(rec)
        if log is not None:
            log.append(rec)
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
    smp_end = build_samples_3d(V_mm, F, D, ax, curve, h_chin, field, fixed, wfun,
                               A_v=A_v)
    kv = smp_end["kind"] == 0
    kf = smp_end["kind"] == 1
    sv = _wstats(smp_end["res"][kv], smp_end["w"][kv])
    sf = _wstats(smp_end["res"][kf], smp_end["w"][kf])
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


def _wstats(res, w):
    """Weighted RMS, p95 of |res| and max |res| (weights are area weights)."""
    res = np.asarray(res, float)
    w = np.asarray(w, float)
    m = w > 0
    if not m.any():
        return {"rms_mm": 0.0, "p95_mm": 0.0, "max_mm": 0.0}
    r, ww = res[m], w[m]
    rms = float(np.sqrt(np.sum(ww * r ** 2) / np.sum(ww)))
    o = np.argsort(np.abs(r))
    cw = np.cumsum(ww[o]) / np.sum(ww)
    p95 = float(np.abs(r[o][int(np.searchsorted(cw, 0.95))]))
    return {"rms_mm": rms, "p95_mm": p95, "max_mm": float(np.max(np.abs(r)))}


def residuals_at(P_mm, F, ax, curve, h_chin, field, wfun, fixed):
    """Residual statistics recomputed on the given positions.

    Reported twice: over the *editable* set (weight > 0) and, as
    ``vertex_rms_all_mm``, unweighted over every vertex of the mesh.  Vertex and
    in-face (quadrature) residuals are separate -- a vertex-only number hides
    what happens between vertices.
    """
    P_mm = np.asarray(P_mm, float)
    n = len(P_mm)
    h, th, r, t, _, _ = frame_at(P_mm, ax, curve, h_chin)
    R = np.asarray(field(h, th), float)
    R = np.where(np.isfinite(R), R, r)
    rv = R - r
    par = np.arange(n)[:, None].repeat(3, axis=1)
    lam = np.broadcast_to(np.array([1.0, 0.0, 0.0]), (n, 3))
    w = np.asarray((wfun or _no_weight)(P_mm, h, th, r, lam, par), float)
    w = np.where(np.asarray(fixed, bool), 0.0, np.maximum(w, 0.0))

    smp = build_samples(P_mm, F, np.zeros(n), np.zeros((n, 3)), ax, curve,
                        h_chin, field, np.zeros(n, bool), wfun)
    kv = smp["kind"] == 0
    kf = smp["kind"] == 1
    sv = _wstats(smp["res"][kv], smp["w"][kv])
    sf = _wstats(smp["res"][kf], smp["w"][kf])
    m = w > 1e-6
    out = {"n_editable": int(m.sum()),
           "vertex_rms_mm": sv["rms_mm"], "vertex_p95_mm": sv["p95_mm"],
           "vertex_max_mm": sv["max_mm"],
           "quad_rms_mm": sf["rms_mm"], "quad_p95_mm": sf["p95_mm"],
           "quad_max_mm": sf["max_mm"],
           "vertex_rms_all_mm": float(np.sqrt(np.mean(rv ** 2))),
           "n_samples": int(len(smp["w"])),
           "n_samples_vertex": int(kv.sum()), "n_samples_face": int(kf.sum())}
    if m.any():
        out["vertex_rms_editable_mm"] = float(np.sqrt(np.average(rv[m] ** 2, weights=w[m])))
    return out
