"""Shared neutral material coordinates and the A / B / C region partition.

Why this module exists
----------------------
The two delivered states are two reconstructions of the same subject with the
same topology, but each has its own global pose and its own reconstruction
noise: every vertex of the "open" mesh sits ~39 mm away from its "closed"
counterpart, and even the stable cranium is not related by an exact rigid
transform.  What *is* shared is the **vertex index** -- the same index is the
same material point of the same subject.

So the design coordinate cannot be "the height above this state's menton".  That
is exactly the conflict the previous round ran into: with the closed menton at
98.424 mm and the open one at 83.798 mm, the design band that ends at
``0.925 * 98.424 = 91.0 mm`` is *entirely* inside the open state's protected
zone (which starts at ``83.798 - 5 = 78.8 mm``).  Instead:

* the **material axial coordinate** ``h_m`` of a vertex is its height in the
  neutral (closed) design frame -- one number per index, shared by both states;
* the design band, the structure window, the density targets and the protection
  are all functions of ``h_m``, so they select the *same material points* in
  both states;
* the *pose* is handled separately and locally (see the bounded adaptation in
  ``tools/build_neck_direct.py``), not by redefining the axis.

The registration of the open state onto the closed one is still computed and
verified here, because it is the evidence that index correspondence is
meaningful: the stable craniofacial seed (no mandible) must land on the closed
surface to within the reconstruction noise, and the mandible must *not*.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from head_rgbd import mesh_integrity as mi


# --------------------------------------------------------------------------
# rigid registration (robust, trimmed)
# --------------------------------------------------------------------------


def rigid_fit(A, B, w=None):
    """Least-squares rotation+translation mapping ``A`` onto ``B``."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if w is None:
        w = np.ones(len(A))
    w = np.asarray(w, float)
    w = w / max(w.sum(), 1e-12)
    ca = (w[:, None] * A).sum(0)
    cb = (w[:, None] * B).sum(0)
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    return R, t


def rigid_register(A, B, iters=8, trim=0.8):
    """Robust rigid fit with iteratively trimmed correspondences."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    w = np.ones(len(A))
    R, t = rigid_fit(A, B, w)
    keep_frac = float(trim)
    for _ in range(iters):
        r = np.linalg.norm((A @ R.T + t) - B, axis=1)
        thr = np.quantile(r, keep_frac)
        w = (r <= thr).astype(float)
        R, t = rigid_fit(A, B, w)
    r = np.linalg.norm((A @ R.T + t) - B, axis=1)
    return {"R": R, "t": t, "residual_mm": r * 1000.0,
            "angle_deg": float(np.degrees(np.arccos(
                np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))),
            "inlier_frac": float(np.mean(w > 0))}


def _closest_on_tri(p, T):
    """Closest point on each triangle ``T`` (m,3,3) to the point ``p`` (m,3).

    Ericson's region test, fully vectorised: the seven Voronoi regions of the
    triangle are selected with ``np.where`` chains, so a point outside the
    triangle lands on the correct vertex or edge instead of being projected by
    the (extrapolating) barycentric formula.
    """
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    inv = np.where(np.abs(den) > 1e-30, 1.0 / np.where(np.abs(den) > 1e-30, den, 1.0), 0.0)
    v = vb * inv
    w = vc * inv
    q = a + ab * v[:, None] + ac * w[:, None]           # interior (or extrapolated)
    # T1
    m = (d3 >= 0) & (d4 <= d3)
    q = np.where(m[:, None], b, q)
    # edge AB
    m2 = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.where(np.abs(d1 - d3) > 1e-30, d1 - d3, 1.0)
    q = np.where(m2[:, None], a + ab * t[:, None], q)
    # T2
    m = (d6 >= 0) & (d5 <= d6)
    q = np.where(m[:, None], c, q)
    # edge AC
    m3 = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.where(np.abs(d2 - d6) > 1e-30, d2 - d6, 1.0)
    q = np.where(m3[:, None], a + ac * t[:, None], q)
    # edge BC
    m4 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    den2 = (d4 - d3) + (d5 - d6)
    t = (d4 - d3) / np.where(np.abs(den2) > 1e-30, den2, 1.0)
    q = np.where(m4[:, None], b + (c - b) * t[:, None], q)
    # T0 covers d1 <= 0 & d2 <= 0
    m0 = (d1 <= 0) & (d2 <= 0)
    q = np.where(m0[:, None], a, q)
    return q


def _point_to_surface(Vq, V, F):
    """Distance (mm) from query points to a triangulated surface.

    Brute force over the nearest face centroids; only used on the small
    verification subsets, so no acceleration structure is needed and the
    three-component mesh (with the open rim) stays out of the way.
    """
    tri = V[F]
    cen = tri.mean(axis=1)
    tre = cKDTree(cen)
    k = min(24, len(cen))
    done = np.zeros(len(Vq), bool)
    _d, idx = tre.query(Vq, k=k)
    idx = np.atleast_2d(idx)
    best = np.full(len(Vq), np.inf)
    for j in range(idx.shape[1]):
        fi = idx[:, j]
        q = _closest_on_tri(Vq, tri[fi])
        best = np.minimum(best, np.linalg.norm(Vq - q, axis=1))
    done[:] = True
    return best * 1000.0


def verify_correspondence(V_closed, V_open_reg, V_open, F, h_material,
                          h_chin_design, stable_t=1.30, mandible_t=(0.95, 1.30)):
    """Does index correspondence mean "same material point"?

    Two checks on the *registered* open mesh: (a) in the stable craniofacial
    seed the open vertex must sit on the closed surface to within the
    reconstruction noise; (b) the mandible must sit clearly *off* it, otherwise
    the two states would not be an open/closed pair at all.
    """
    Vq = np.asarray(V_open_reg, float)
    Vc = np.asarray(V_closed, float)
    # subsample: the surface query is brute force
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Vq))
    pick = idx[:4000]
    dist = np.full(len(Vq), np.nan)
    dist[pick] = _point_to_surface(Vq[pick], Vc, F)
    h = np.asarray(h_material, float) / h_chin_design
    stable = (h >= stable_t) & np.isfinite(dist)
    mand = (h >= mandible_t[0]) & (h < mandible_t[1]) & np.isfinite(dist)
    raw = np.linalg.norm(Vq - Vc, axis=1) * 1000.0
    out = {"n_sampled": int(len(pick))}
    for name, m in (("stable_craniofacial", stable), ("mandible_band", mand)):
        if m.sum() < 10:
            out[name] = None
            continue
        d = dist[m]
        out[name] = {"n": int(m.sum()), "median_mm": float(np.median(d)),
                     "p95_mm": float(np.percentile(d, 95)),
                     "max_mm": float(np.max(d)),
                     "raw_index_distance_median_mm": float(np.median(raw[m]))}
    return out


# --------------------------------------------------------------------------
# geodesic partition
# --------------------------------------------------------------------------


def edge_graph(V_m, F, alive=None):
    """Symmetric sparse graph of the mesh with edge lengths in mm."""
    E = mi.edge_records(F)[0]
    Vmm = np.asarray(V_m, float) * 1000.0
    L = np.linalg.norm(Vmm[E[:, 0]] - Vmm[E[:, 1]], axis=1)
    n = len(Vmm)
    W = sparse.coo_matrix((L, (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    W = W.maximum(W.T)
    return W, E


def geodesic_from(W, seeds):
    """Distance (mm) to the nearest of ``seeds`` along the surface graph."""
    seeds = np.flatnonzero(np.asarray(seeds, bool))
    if not len(seeds):
        return np.full(W.shape[0], np.inf)
    d = dijkstra(W, directed=False, indices=seeds, min_only=True)
    return np.asarray(d, float).ravel()


def partition(V_m, F, h_material, h_chin_design, p, extra_fixed=None):
    """A (fixed head) / B (submental bridge) / C (neck body) by geodesic distance.

    ``A``  the head: face, lips, teeth, the mandible outer contour and the rest
           of the head.  Never moves.
    ``B``  the submental bridge: a band of geodesic width ``bridge_width_mm``
           measured *from the A/B boundary along the surface*, between the fixed
           jaw/chin border and the neck body.  Bounded 3-D displacement.
    ``C``  the neck body and root: the part that the radial design shapes.

    The boundary is the equidistant set between two surface seeds, so it follows
    the surface (and therefore the submental fold) instead of one global height
    threshold, and it is computed once in material space and used by both
    states.
    """
    t_m = np.asarray(h_material, float) / float(h_chin_design)
    W, E = edge_graph(V_m, F)
    head_seed = t_m >= p["mat_head_seed_t"]
    neck_seed = t_m <= p["mat_neck_seed_t"]
    d_head = geodesic_from(W, head_seed)
    d_neck = geodesic_from(W, neck_seed)
    score = d_head - d_neck
    reach = np.isfinite(score)
    A = (score < 0) & reach
    # B is a geodesic band measured along the crossing direction; the score
    # changes at 2 mm per mm travelled across the boundary, so a band of width
    # w in score is w/2 wide on the surface
    B = (score >= 0) & (score <= 2.0 * p["bridge_width_mm"]) & reach
    C = (score > 2.0 * p["bridge_width_mm"]) & reach
    # anything unreachable from the seeds (the two small extra components) is
    # simply protected: it is not part of the neck
    unreached = ~reach
    A = A | unreached
    if extra_fixed is not None:
        A = A | np.asarray(extra_fixed, bool)

    # B may not touch the open rim or the root, and it must not extend into A
    B = B & ~A
    C = C & ~A & ~B
    A_idx = np.flatnonzero(A)
    B_idx = np.flatnonzero(B)
    C_idx = np.flatnonzero(C)

    # interface edges, both directions
    eA = A[E[:, 0]] & B[E[:, 1]] | A[E[:, 1]] & B[E[:, 0]]
    eB = B[E[:, 0]] & C[E[:, 1]] | B[E[:, 1]] & C[E[:, 0]]
    L = np.linalg.norm(np.asarray(V_m, float)[E[:, 0]] * 1000.0 -
                       np.asarray(V_m, float)[E[:, 1]] * 1000.0, axis=1)
    # measured geodesic thickness of B: the distance from the A/B boundary
    # *along the surface*, which is the quantity the spec calls the transition
    # width.  (``max(d_head) - min(d_head)`` would measure the spread of the
    # band around the whole contour, not its thickness across the crossing.)
    d_A = geodesic_from(W, A)
    dB = d_A[B_idx]
    dB = dB[np.isfinite(dB)]
    width_meas = float(dB.max()) if len(dB) else 0.0
    width_med = float(np.median(dB)) if len(dB) else 0.0
    return {"A": A, "B": B, "C": C,
            "A_idx": A_idx, "B_idx": B_idx, "C_idx": C_idx,
            "d_head": d_head, "d_neck": d_neck, "score": score,
            "edges_A_B": E[eA], "edges_B_C": E[eB],
            "faces_A_B": np.flatnonzero(A[F].any(axis=1) & B[F].any(axis=1)),
            "faces_B_C": np.flatnonzero(B[F].any(axis=1) & C[F].any(axis=1)),
            "n_A": int(A.sum()), "n_B": int(B.sum()), "n_C": int(C.sum()),
            "bridge_width_target_mm": float(p["bridge_width_mm"]),
            "bridge_width_measured_mm": width_meas,
            "bridge_width_median_mm": width_med,
            "geodesic_head_to_neck_mm": float(np.nanmax(np.minimum(d_head, d_neck))),
            "area_mm2": {"A": float(_area(V_m, F, A)),
                         "B": float(_area(V_m, F, B)),
                         "C": float(_area(V_m, F, C))},
            "head_seed_t": float(p["mat_head_seed_t"]),
            "neck_seed_t": float(p["mat_neck_seed_t"]),
            "unreached": int(unreached.sum())}


def _area(V_m, F, mask_v):
    tri = np.asarray(V_m, float)[F]
    a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                      tri[:, 2] - tri[:, 0]), axis=1) * 1e6
    m = mask_v[F].all(axis=1)
    return float(a[m].sum())


def boundary_loops(F, mask_v):
    """Vertices of ``mask_v`` that have at least one neighbour outside it."""
    E = mi.edge_records(F)[0]
    inside = mask_v[E[:, 0]] & mask_v[E[:, 1]]
    bedges = E[~inside]
    ids = np.unique(bedges.ravel())
    return ids[mask_v[ids]]


def extend_partition(part, ops, F, h_material, vlist=None, extra_scalars=()):
    """Carry the A/B/C partition across a re-spacing pass.

    A new vertex cannot be re-partitioned from scratch -- the geodesic solve is
    defined on the *material* graph, which the split just changed -- so it
    inherits from the edge it was inserted on.  That is not a shortcut: the
    re-spacing pass refuses to refine an edge whose endpoints sit in different
    regions, so the edge has exactly one region and the inheritance is the same
    answer the geodesic solve would give.

    ``vlist`` is the material height array (one value per vertex); it is extended
    here as the *parameter midpoint* of the split edge, which is where the new
    vertex actually sits in the material frame.
    ``extra_scalars`` names further per-vertex arrays in ``part`` that should be
    extended by the same midpoint rule (``d_head``, ``d_neck``, ``score``).
    """
    n_old = len(part["A"])
    pairs = ops["split"]
    out = dict(part)
    if pairs:
        add = {}
        for k in ("A", "B", "C"):
            m0 = np.asarray(part[k], bool)
            add[k] = m0[[q["edge"][0] for q in pairs]] & m0[[q["edge"][1] for q in pairs]]
        for k in ("A", "B", "C"):
            out[k] = np.concatenate([np.asarray(part[k], bool), add[k]])
        for k in extra_scalars:
            if k in part and isinstance(part[k], np.ndarray) and len(part[k]) == n_old:
                a = np.asarray(part[k], float)[[q["edge"][0] for q in pairs]]
                b = np.asarray(part[k], float)[[q["edge"][1] for q in pairs]]
                out[k] = np.concatenate([np.asarray(part[k], float), 0.5 * (a + b)])
    hm = None
    if vlist is not None:
        hm = np.asarray(vlist, float)
        if pairs:
            a = hm[[q["edge"][0] for q in pairs]]
            b = hm[[q["edge"][1] for q in pairs]]
            hm = np.concatenate([hm, 0.5 * (a + b)])
        out["h_material"] = hm
    # boundary sets are topology, not attributes: recompute them from the new F
    for key, (u, w) in (("edges_A_B", ("A", "B")), ("edges_B_C", ("B", "C"))):
        E = mi.edge_records(F)[0]
        m = ((out[u][E[:, 0]] & out[w][E[:, 1]]) | (out[u][E[:, 1]] & out[w][E[:, 0]]))
        out[key] = E[m]
    for key, (u, w) in (("faces_A_B", ("A", "B")), ("faces_B_C", ("B", "C"))):
        out[key] = np.flatnonzero(out[u][F].any(axis=1) & out[w][F].any(axis=1))
    out["n_A"] = int(out["A"].sum())
    out["n_B"] = int(out["B"].sum())
    out["n_C"] = int(out["C"].sum())
    out["A_idx"] = np.flatnonzero(out["A"])
    out["B_idx"] = np.flatnonzero(out["B"])
    out["C_idx"] = np.flatnonzero(out["C"])
    out["n_split_inherited"] = int(len(pairs))
    return out, hm
