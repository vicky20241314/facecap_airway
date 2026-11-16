"""Recorded, budgeted local re-spacing on the shared topology.

Every operation is expressed as an *index* record (which edge, which endpoint is
kept, the order in which new vertices appear), so the very same list can be
replayed on the other state: the two states keep one topology and one vertex
correspondence and are never re-meshed independently.

Only three primitives touch the topology, each with an explicit validity test:

``split``     one edge, one new vertex whose parameter position is decided once
              and whose radius comes from each state's own continuous field;
``collapse``  one edge, link condition + valence + both-state normal test;
``flip``      one edge, strictly improving the worst face angle.

Decisions are made on the normalised edge length ``L / ell(p)`` with hysteresis
(``> 4/3`` to be a split candidate, ``< 4/5`` to be a collapse candidate).  A
candidate is not an operation: it must also sit inside the editable band, fit
the vertex budget computed before any topology change, improve shape or
spacing, be safe in *both* states, and keep the mesh manifold with its boundary
and the protected set intact.  Candidates that touch the protected set are never
used, so an interface edge cannot start a refinement cascade.
"""
from __future__ import annotations

import os

import numpy as np

from head_rgbd import mesh_integrity as mi
from head_rgbd import neck_direct as nd

_DBG = bool(os.environ.get("NECK_DBG"))


def _dbg_spacing(tag, states, F, P):
    """Temporary diagnostic: longest edge and worst vertex far from its star."""
    st = list(states.values())[0]
    V = st["V"]
    E = mi.edge_records(F)[0]
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) * 1000.0
    k = int(np.argmax(L))
    print(f"    [dbg] {tag}: max edge {L.max():.1f} mm at {E[k].tolist()}, "
          f"vertices {len(V)}", flush=True)


# --------------------------------------------------------------------------
# small geometric helpers
# --------------------------------------------------------------------------


def face_angles(V, F):
    tri = V[F]
    ang = np.zeros((len(F), 3))
    for k in range(3):
        a, b, c = tri[:, k], tri[:, (k + 1) % 3], tri[:, (k + 2) % 3]
        u, w = b - a, c - a
        cos = np.einsum("ij,ij->i", u, w) / np.maximum(
            np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-12)
        ang[:, k] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return ang


def face_normals(V, F):
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]


def face_area_mm2(V, F):
    tri = V[F]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                axis=1) * 1e6


def edge_map(F):
    """sorted (i, j) -> face rows containing it."""
    F = np.asarray(F)
    d = {}
    for k in range(3):
        i, j = F[:, k], F[:, (k + 1) % 3]
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        for a, b, r in zip(lo.tolist(), hi.tolist(), range(len(F))):
            d.setdefault((a, b), []).append(r)
    return d


def valence(F, n):
    return np.bincount(np.asarray(F, np.int64).ravel(), minlength=n)


def star_normal(V, F, v, rows):
    """Area-weighted normal of the faces in ``rows`` that touch ``v``."""
    n = np.zeros(3)
    for r in rows:
        tri = V[F[r]]
        c = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        n += c
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else n


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def split_edges(F, pairs):
    """Split a set of *vertex-disjoint* edges in one pass.

    ``pairs`` is a list of ``(i, j, nv)``.  No two selected edges share a vertex,
    so no triangle contains two of them (any two edges of a triangle share a
    vertex) and each affected face is simply cut in two: the result stays
    conforming with no red-green bookkeeping.
    """
    F = np.asarray(F, dtype=np.int64)
    sel = {}
    for i, j, nv in pairs:
        sel[(min(i, j), max(i, j))] = nv
    if not sel:
        return F, 0
    keep, new, hit = [], [], 0
    for tri in F.tolist():
        found = None
        for s in range(3):
            p0, p1, p2 = tri[s], tri[(s + 1) % 3], tri[(s + 2) % 3]
            key = (min(p0, p1), max(p0, p1))
            if key in sel:
                found = (p0, p1, p2, sel[key])
                break
        if found is None:
            keep.append(tri)
        else:
            p0, p1, p2, nv = found
            keep.append([p0, nv, p2])
            new.append([nv, p1, p2])
            hit += 1
    return np.asarray(keep + new, dtype=np.int64), hit


def collapse_edge(F, keep, drop, em=None, min_val=3):
    """Merge ``drop`` into ``keep``; None when the topology would break."""
    em = edge_map(F) if em is None else em
    rows = em.get((min(keep, drop), max(keep, drop)), [])
    if len(rows) != 2:
        return None
    opp = []
    for r in rows:
        opp += [int(x) for x in F[r] if x != keep and x != drop]
    if len(set(opp)) != 2:                      # link condition
        return None
    val = valence(F, int(F.max()) + 1)
    if val[keep] + val[drop] - 2 < 3:
        return None
    if val[drop] <= min_val and val[keep] <= min_val:
        return None
    F2 = F.copy()
    F2[F2 == drop] = keep
    degen = (F2[:, 0] == F2[:, 1]) | (F2[:, 1] == F2[:, 2]) | (F2[:, 0] == F2[:, 2])
    F2 = F2[~degen]
    if mi.has_duplicate_face(F2):
        return None
    return F2


def flip_edge(F, em, i, j, V=None):
    """Flip edge (i, j), preserving each triangle's cyclic winding exactly.

    A flip rewrites the two adjacent triangles, so their orientation must be
    read from the data rather than assumed: writing ``(i, l, k)`` into an
    arbitrarily ordered row inverts the winding of that triangle, which is
    culled by the rasteriser (a black hole) and shows up as
    ``inconsistent_shared_edges`` in the topology report.  Two cases exist,
    ``(i, j, k)`` and ``(i, k, j)``; each has its own pair of output windings.
    ``V`` is only used for an optional safety assert.
    """
    rows = em.get((min(i, j), max(i, j)), [])
    if len(rows) != 2:
        return None
    tri0 = F[rows[0]].tolist()
    opp = []
    for r in rows:
        opp += [x for x in F[r].tolist() if x != i and x != j]
    if len(set(opp)) != 2:
        return None
    k, l = opp
    p = tri0.index(i)
    if tri0[(p + 1) % 3] == j:            # cyclic  i -> j -> k
        new = [[i, l, k], [j, k, l]]
    else:                                  # cyclic  i -> k -> j
        new = [[i, k, l], [j, l, k]]
    if V is not None:
        tri = np.asarray(V, float)
        ref = np.zeros(3)
        for r in rows:
            ref += np.cross(tri[F[r, 1]] - tri[F[r, 0]], tri[F[r, 2]] - tri[F[r, 0]])
        for t in new:
            n = np.cross(tri[t[1]] - tri[t[0]], tri[t[2]] - tri[t[0]])
            if float(n @ ref) < 0.0:       # degenerate or folded pair: reject
                return None
    F2 = F.copy()
    F2[rows[0]] = new[0]
    F2[rows[1]] = new[1]
    if mi.has_duplicate_face(F2):
        return None
    return F2


# --------------------------------------------------------------------------
# per-state parameter layout
# --------------------------------------------------------------------------


def layout(st, V=None):
    """(h, a, b) of every vertex in this state's own frame, plus its residual.

    ``res`` is how far the vertex sits from the continuous target field along the
    radius.  Everything that *creates* a point (a split) or *measures* an edge
    on the current surface needs it: the field alone is the target, not the mesh.
    """
    V = st["V"] if V is None else V
    P = np.asarray(V, float) * 1000.0
    h, th, r, t = nd.axial_mm(P, st["ax"], st["curve"], st["h_chin"])
    ca, cb = st["curve"](h)
    rt = np.asarray(st["field"](h, th), float)
    res = np.where(np.isfinite(rt), r - rt, 0.0)
    return {"h": h, "a": np.sin(th) * r + ca, "b": np.cos(th) * r + cb, "t": t,
            "th": th, "r": r, "res": res}


def place(st, h, a, b, res=None):
    """Position from this state's own frame and own continuous field.

    ``res`` -- optional radial residual per entry.  The field is a *target*, and
    a vertex of the delivered mesh sits at ``field + residual``; a point created
    by a split must inherit the interpolated residual of its two parents,
    otherwise it lands on the target while its parents sit up to ``max|s|`` mm
    off it and the edge turns into a kink that no later solve can smooth away.
    """
    ca, cb = st["curve"](h)
    aa, bb = a - ca, b - cb
    rr = np.hypot(aa, bb)
    th = np.arctan2(aa, bb)
    rt = np.asarray(st["field"](h, th), float)
    rt = np.where(np.isfinite(rt), rt, rr)
    if res is not None:
        rt = np.maximum(rt + np.asarray(res, float), 1.0)
    sa = ca + np.sin(th) * rt
    sb = cb + np.cos(th) * rt
    return (st["ax"]["centre_mm"][None, :] + h[:, None] * st["ax"]["axis"][None, :] +
            sa[:, None] * st["ax"]["lateral"][None, :] +
            sb[:, None] * st["ax"]["anterior"][None, :]) / 1000.0


def ring_depth(F, mask, rings=3):
    """BFS distance inside ``mask`` from its border, capped at ``rings``.

    Depth 0 is the first ring *inside* ``mask`` (i.e. the vertices of ``mask``
    that touch something outside it).  Used both for the sliding ramp of
    :func:`relax` and to keep merges away from the bridge interface.
    """
    mask = np.asarray(mask, bool)
    n = len(mask)
    depth = np.full(n, int(rings), np.int32)
    if not mask.any():
        return depth
    E = mi.edge_records(F)[0]
    outside = ~mask
    touch = outside[E[:, 0]] | outside[E[:, 1]]
    layer = np.zeros(n, bool)
    layer[E[touch].ravel()] = True
    layer &= mask
    depth[layer] = 0
    d = 0
    while d < int(rings) and layer.any():
        d += 1
        touch = layer[E[:, 0]] | layer[E[:, 1]]
        nxt = np.zeros(n, bool)
        nxt[E[touch].ravel()] = True
        nxt &= mask
        nxt &= ~(depth < d)
        depth[nxt] = d
        layer = nxt
    return depth


def boundary_ramp(F, mask, rings=3):
    """C1 weight: 0 on the border of ``mask``, 1 at ``rings`` edges inside it.

    A tangential relaxation is a *sliding* boundary value problem: where the
    sliding set stops, the displacement has to go to zero smoothly, or the last
    ring shears against its fixed neighbour.  In this mesh that neighbour is the
    bridge (region B), whose vertices are held by the 3-D solve, so a hard mask
    put a 2.4 mm one-pass tangential jump across a single 4 mm edge and turned a
    2 deg face pair into a 150 deg fold.  The step is therefore weighted by this
    ramp instead of by a binary mask.
    """
    mask = np.asarray(mask, bool)
    depth = ring_depth(F, mask, rings)
    out = nd.srise(depth.astype(float), 0.0, float(rings))
    return np.where(mask, out, 0.0)


def relax(states, F, free, p, iters=None, step=None, ell_v=None, weight=None):
    """Tangential re-spacing in the (h, a, b) parameter plane.

    The neck surface is a radial field over the spine, so every vertex has exact
    2-D parameters: height ``h`` and offset ``(a, b)``.  A Laplacian step on that
    triple, followed by re-evaluating the radius from the same continuous field,
    cannot drift off the surface and needs no projection, no self-intersection
    repair and no normal smoothing.  The fixed set is a Dirichlet boundary.

    This is a *layout increment*: the same step and the same neighbour structure
    in both states, each state's own coordinates and own field.  Both states
    move, so the correspondence is preserved by construction.

    The step is *weighted by* :func:`boundary_ramp`, not gated by the mask: a
    tangential slide that stops abruptly at the bridge shears the last ring into
    a fold, so the sliding amplitude has to decay to zero over a few edges.
    """
    iters = p["relax_iters"] if iters is None else iters
    step = p["relax_step"] if step is None else step
    free = np.asarray(free, bool)
    # sliding must fade out towards every place where it is not allowed to slide
    wstep = (boundary_ramp(F, free, p.get("relax_ramp_rings", 3))
             if weight is None else np.asarray(weight, float))
    E = mi.edge_records(F)[0]
    keep = free[E[:, 0]] & free[E[:, 1]]
    e = E[keep]
    n = len(free)
    deg = np.zeros(n)
    np.add.at(deg, e[:, 0], 1.0)
    np.add.at(deg, e[:, 1], 1.0)
    move = free & (deg > 0)
    moved = 0.0
    for st in states.values():
        L = layout(st)
        h, a, b = L["h"].copy(), L["a"].copy(), L["b"].copy()
        # the radial residual to the target is *carried along*, so the step stays
        # tangential: without it the re-evaluated radius would re-project every
        # vertex onto the field and a partially weighted vertex would jump
        r_t = np.asarray(st["field"](L["h"], L["th"]), float)
        r_t = np.where(np.isfinite(r_t), r_t, L["r"])
        res = L["r"] - r_t
        h0, a0, b0 = h.copy(), a.copy(), b.copy()
        ell_e = 0.5 * (ell_v[e[:, 0]] + ell_v[e[:, 1]])
        for _ in range(iters):
            # spring equaliser, not a Laplacian: every edge pushes or pulls
            # along its own direction until it matches its target length.  A
            # Laplacian would shrink the parameter distribution, which never
            # stops manufacturing "too short" edges.
            # measure lengths on the surface the vertex actually sits on
            # (field + carried residual), not on the target
            P = place(st, h, a, b, res)
            d3 = P[e[:, 1]] - P[e[:, 0]]
            L3 = np.linalg.norm(d3, axis=1) * 1000.0
            u = d3 / np.maximum(np.linalg.norm(d3, axis=1)[:, None], 1e-9)
            corr = u * (L3 - ell_e)[:, None]
            acc3 = np.zeros((n, 3))
            np.add.at(acc3, e[:, 0], corr)
            np.add.at(acc3, e[:, 1], -corr)
            acc3 = np.divide(acc3, np.maximum(deg, 1.0)[:, None])
            acc3 = acc3 * wstep[:, None]
            h += step * (acc3 @ st["ax"]["axis"])
            a += step * (acc3 @ st["ax"]["lateral"])
            b += step * (acc3 @ st["ax"]["anterior"])
        # bound the slide in millimetres: a re-spacing step must stay local
        cap = p["relax_max_frac"] * (np.asarray(ell_v, float) if ell_v is not None
                                     else np.full(n, 6.0))
        dh, da, db = h - h0, a - a0, b - b0
        mag = np.sqrt(dh ** 2 + da ** 2 + db ** 2)
        over = mag > cap
        if over.any():
            k = np.where(over, cap / np.maximum(mag, 1e-9), 1.0)
            h = h0 + dh * k
            a = a0 + da * k
            b = b0 + db * k
        ca, cb = st["curve"](h)
        aa, bb = a - ca, b - cb
        rr = np.hypot(aa, bb)
        th = np.arctan2(aa, bb)
        rt = np.asarray(st["field"](h, th), float)
        rt = np.where(np.isfinite(rt), rt, rr)
        rr = np.where(move, np.maximum(rt + res, 1.0), rr)
        sa, sb = ca + np.sin(th) * rr, cb + np.cos(th) * rr
        V = (st["ax"]["centre_mm"][None, :] + h[:, None] * st["ax"]["axis"][None, :] +
             sa[:, None] * st["ax"]["lateral"][None, :] +
             sb[:, None] * st["ax"]["anterior"][None, :]) / 1000.0
        d = np.linalg.norm(V[move] - st["V"][move], axis=1) * 1000.0
        moved = max(moved, float(d.max()) if len(d) else 0.0)
        st["V"] = np.where(free[:, None], V, st["V"])
    return {"max_move_mm": moved}


def _norm_lengths(states, F, free, ell_v):
    """Normalised edge length, aggregated over the two states by the maximum.

    Both tests read the same aggregate, in opposite directions:

    ``split``     ``max(L/ell) > 4/3``  -- long in *either* state is enough to
                  ask for a vertex, and a vertex added for one state is harmless
                  for the other;
    ``collapse``  ``max(L/ell) < 4/5``  -- short in *every* state is required,
                  because a merge decided on the closed mesh alone can coarsen
                  the crest of the open one.

    Aggregating with ``min`` would invert the second test into "short in at
    least one state", which is exactly the case the shared decision forbids.

    The neutral value for an edge that may not move is ``1.0``: it sits inside
    ``[4/5, 4/3]``, so a protected edge is neither split nor merged, and no
    separate sentinel is needed per direction.

    Returns ``(E, nrm, per_state)``; ``per_state`` keeps each state's own
    ``L / ell`` so the journal can say *which* state asked for an operation and
    prove it was still decided on the shared log.
    """
    E = mi.edge_records(F)[0]
    Ls = [np.linalg.norm(st["V"][E[:, 0]] - st["V"][E[:, 1]], axis=1) * 1000.0
          for st in states.values()]
    LA = np.max(Ls, axis=0)
    ok = free[E[:, 0]] & free[E[:, 1]]
    ell_e = 0.5 * (ell_v[E[:, 0]] + ell_v[E[:, 1]])
    nrm = np.where(ok, LA / np.maximum(ell_e, 1e-6), 1.0)
    per = {key: np.where(ok, L / np.maximum(ell_e, 1e-6), 1.0)
           for key, L in zip(states.keys(), Ls)}
    return E, nrm, per


def chord_error_mm(states, E, sel, with_len=False):
    """Sagitta of the *continuous* target surface over the selected edges.

    A long edge is not by itself a reason to add a vertex: on a flat or a
    gently curved part of the field a 9 mm edge represents the surface to a few
    hundredths of a millimetre, and splitting it only spends budget and adds
    facets.  What actually needs a new point is a place where the straight
    segment between the two endpoints departs from the target surface.

    The test therefore samples the continuous field at the *parameter midpoint*
    of the edge (``place``), and measures how far that true surface point sits
    from the chord joining the two endpoint surface points.  Aggregated over the
    two states by the maximum, like the length test.  Threshold:
    ``chord_tol_frac * ell``.

    **The measurement is only meaningful while the midpoint is actually on the
    surface between the endpoints.**  Where the cross-section is not star-shaped
    about the neck axis the radial field -- and hence ``place`` -- is not a
    description of the surface, and the "sagitta" can come out larger than the
    edge itself.  That is not a missing vertex, it is a broken parameterisation,
    so ``with_len=True`` also returns the shortest of the two 3-D edge lengths and
    the caller rejects any candidate whose chord exceeds half of it (a sagitta of
    a smooth arc can never be more than half its chord).
    """
    if not len(sel):
        return (np.zeros(0), np.zeros(0)) if with_len else np.zeros(0)
    e = E[sel]
    out, lmin = [], []
    for st in states.values():
        L = layout(st)
        h = 0.5 * (L["h"][e[:, 0]] + L["h"][e[:, 1]])
        a = 0.5 * (L["a"][e[:, 0]] + L["a"][e[:, 1]])
        b = 0.5 * (L["b"][e[:, 0]] + L["b"][e[:, 1]])
        res = 0.5 * (L["res"][e[:, 0]] + L["res"][e[:, 1]])
        Pm = place(st, h, a, b, res)
        P0 = st["V"][e[:, 0]]
        P1 = st["V"][e[:, 1]]
        mid = 0.5 * (P0 + P1)
        out.append(np.linalg.norm(Pm - mid, axis=1) * 1000.0)
        lmin.append(np.linalg.norm(P1 - P0, axis=1) * 1000.0)
    C = np.max(np.asarray(out), axis=0)
    return (C, np.min(np.asarray(lmin), axis=0)) if with_len else C


def _cap(v):
    """``None`` means "no fixed cap" -- the budget and the criteria decide."""
    return 10 ** 9 if v is None else int(v)


def respace(states, F, free, ell_v, p, pass_index, recent, budget,
            zone=None, chord_tol_frac=None, collapse_ok=None, eligible=None,
            max_new_vertices=None, new_vertices_so_far=0, allow_split=True):
    """One re-spacing pass: split / collapse / flip, recorded as index ops.

    ``budget`` is either an int (total split room) or a dict with
    ``{"split": int, "collapse": int, "split_by_zone": {z: int},
    "collapse_by_zone": {z: int}}``.  The per-zone rooms come from the *area
    budget of each density zone separately* (anterior / structure / sides), so a
    surplus in one zone can never be spent in another and the anterior neck
    cannot be densified by vertices granted to the structure zone.

    ``eligible`` -- the set of vertices an operation may touch at all.  It is the
    density band (material ``t`` inside ``data_t[0] .. interface_t[0]``): the only
    place where an area budget was measured *and* where ``place()`` is a valid
    description of the surface.  Both endpoints of an edge must be eligible.

    There is no fixed operation cap: how many operations happen is decided by the
    budget, the length test and the chord test.  A pass ends when no candidate
    satisfies all of them (which is also the convergence signal the caller
    watches).
    """
    ops = {"split": [], "collapse": [], "flip": [], "rejected": []}
    E, nrm, per = _norm_lengths(states, F, free, ell_v)
    n_v = len(list(states.values())[0]["V"])
    zone = np.zeros(n_v, np.int8) if zone is None else np.asarray(zone, np.int8)
    elig = np.ones(n_v, bool) if eligible is None else np.asarray(eligible, bool)
    chord_tol_frac = float(p.get("chord_tol_frac", 0.05) if chord_tol_frac is None
                           else chord_tol_frac)

    # ---- splits ---------------------------------------------------------
    if isinstance(budget, dict):
        room_split = int(budget.get("split", 10 ** 9))
        room_collapse = int(budget.get("collapse", 10 ** 9))
        room_s = budget.get("split_by_zone", {})
        room_c = budget.get("collapse_by_zone", {})
    else:
        room_split = int(budget)
        room_collapse = 10 ** 9
        room_s, room_c = {}, {}

    def _room_left(tbl, z):
        return tbl.get(int(z), tbl.get(str(int(z)), 10 ** 9))

    order = np.argsort(-nrm)
    cand = [int(ei) for ei in order.tolist() if nrm[ei] > p["split_ratio"]]
    used, pairs, spent = set(), [], {}
    chord_tol_mm = float(p.get("split_chord_tol_mm", 0.0) or 0.0)
    if cand:
        # the chord test is only run on the length candidates, so a long but
        # flat edge is skipped instead of costing a solve
        C, CL = chord_error_mm(states, E, np.asarray(cand, np.int64), with_len=True)
    else:
        C, CL = np.zeros(0), np.zeros(0)
    for k, ei in enumerate(cand):
        if not allow_split:
            break
        if max_new_vertices is not None and \
                new_vertices_so_far + len(pairs) >= int(max_new_vertices):
            break
        if len(pairs) >= _cap(p.get("split_max_per_pass")) or len(pairs) >= room_split:
            break
        i, j = int(E[ei, 0]), int(E[ei, 1])
        if not (elig[i] and elig[j]):
            continue                       # outside the density band
        if i in used or j in used or not _fresh((i, j), recent, pass_index):
            continue
        z = int(zone[i])
        if z != int(zone[j]):
            continue                       # never refine across a zone border
        if spent.get(z, 0) >= _room_left(room_s, z):
            continue                       # this zone's own area budget is used up
        ell_e = 0.5 * (ell_v[i] + ell_v[j])
        # A split has to buy surface expression, not merely a long edge: the
        # chord has to miss the continuous target by at least the requested
        # millimetre error (0.35 mm in the current brief).  Relief magnitude is
        # not used as a proxy -- a large but smooth bulge does not need a vertex.
        if C[k] < max(chord_tol_frac * ell_e, chord_tol_mm):
            continue                       # long, but the chord already fits
        if C[k] > 0.5 * max(CL[k], 1e-9):
            # the parameter midpoint is not on the surface between the two
            # endpoints: the radial field is not valid here, so the "sagitta"
            # says nothing about resolution.  Splitting would place the new
            # vertex off the surface (measured up to 39 mm, i.e. a spike), so
            # the candidate is rejected *and reported*, not approved.
            ops["rejected"].append({"edge": [i, j], "why": "chord_exceeds_edge",
                                    "chord_mm": round(float(C[k]), 3),
                                    "edge_len_mm": round(float(CL[k]), 3)})
            continue
        used.add(i)
        used.add(j)
        spent[z] = spent.get(z, 0) + 1
        pairs.append((i, j, n_v + len(pairs), float(nrm[ei]), ei, float(C[k]),
                      float(ell_e)))
    if pairs:
        # parameter midpoint decided once, radius from each state's own field
        # plus the interpolated residual of the two parents, so the new vertex
        # is *on the delivered surface* and not merely on the target
        for st in states.values():
            L = layout(st)
            newpos, newres = [], []
            for i, j, nv, _, _, _, _ in pairs:
                h = 0.5 * (L["h"][i] + L["h"][j])
                a = 0.5 * (L["a"][i] + L["a"][j])
                b = 0.5 * (L["b"][i] + L["b"][j])
                newpos.append([h, a, b])
                newres.append(0.5 * (L["res"][i] + L["res"][j]))
            P = np.asarray(newpos, float)
            Vn = place(st, P[:, 0], P[:, 1], P[:, 2], np.asarray(newres, float))
            st["V"] = np.vstack([st["V"], Vn])
        F, hit = split_edges(F, [(i, j, nv) for i, j, nv, _, _, _, _ in pairs])
        for i, j, nv, r, ei, c, ell_e in pairs:
            ops["split"].append({"edge": [i, j], "new_vertex": int(nv),
                                 "norm": round(r, 3),
                                 "chord_mm": round(c, 3),
                                 "chord_frac": round(c / max(ell_e, 1e-6), 3),
                                 "zone": int(zone[i]),
                                 "norm_closed": round(float(per["closed"][ei]), 3),
                                 "norm_open": round(float(per["open"][ei]), 3)})
            recent[(min(i, nv), max(i, nv))] = pass_index
            recent[(min(j, nv), max(j, nv))] = pass_index
        # a new vertex is not "free because it is new": it is free because it
        # sits on an edge whose two endpoints were free and inside the same
        # density zone.  Cell protection is inherited *and re-checked*; nothing
        # is switched on unconditionally.
        inherit = np.asarray([free[i] and free[j] for i, j, _, _, _, _, _ in pairs], bool)
        free = np.concatenate([np.asarray(free, bool), inherit])
        zone = np.concatenate([zone, np.asarray([zone[i] for i, _, _, _, _, _, _ in pairs],
                                                np.int8)])
        ell_v = np.concatenate([np.asarray(ell_v, float),
                                [0.5 * (ell_v[i] + ell_v[j]) for i, j, _, _, _, _, _ in pairs]])
        elig = np.concatenate([elig, np.asarray([elig[i] and elig[j]
                                                 for i, j, _, _, _, _, _ in pairs], bool)])
        if _DBG:
            _dbg_spacing(f"after {len(pairs)} splits", states, F, None)

    # ---- collapses ------------------------------------------------------
    E, nrm, per = _norm_lengths(states, F, free, ell_v)
    # zones may have grown with the splits: keep them addressable
    if len(zone) < len(free):
        zone = np.concatenate([zone, np.full(len(free) - len(zone), 2, np.int8)])
    if len(elig) < len(free):
        elig = np.concatenate([elig, np.zeros(len(free) - len(elig), bool)])
    cok = np.ones(len(free), bool) if collapse_ok is None else np.asarray(collapse_ok, bool)
    if len(cok) < len(free):
        cok = np.concatenate([cok, np.ones(len(free) - len(cok), bool)])
    em = edge_map(F)
    dropped = set()
    spent = {}
    for ei in np.argsort(nrm).tolist():
        if len(ops["collapse"]) >= _cap(p.get("collapse_max_per_pass")) or \
                len(ops["collapse"]) >= room_collapse:
            break
        if nrm[ei] >= p["collapse_ratio"]:
            break
        i, j = int(E[ei, 0]), int(E[ei, 1])
        if not (free[i] and free[j]):
            continue
        if not (elig[i] and elig[j]):
            continue          # outside the density band: nothing was budgeted there
        if not (cok[i] and cok[j]):
            continue          # connection band: never merged away
        if i in dropped or j in dropped or not _fresh((i, j), recent, pass_index):
            continue
        z = int(zone[i])
        if z == 1:
            continue          # the structure zone is never coarsened this round
        if z != int(zone[j]):
            continue          # a merge across a zone border would blur the hand-over
        if spent.get(z, 0) >= _room_left(room_c, z):
            continue
        keep, drop = (i, j) if i < j else (j, i)
        F2 = collapse_edge(F, keep, drop, em=em)
        if F2 is None:
            continue
        # The surviving vertex is kept on the captured surface.  A merge is a
        # *conformality* operation: removing a redundant vertex must not move the
        # surface.  Moving the survivor to the parameter midpoint (the previous
        # behaviour) shifted it by up to 2.4 mm here and, after the continuous
        # solve, turned a 16 deg face pair into a 44-49 deg fold.  The layout can
        # be evened afterwards by the tangential relax; geometry comes first.
        move_frac = float(p.get("collapse_move_frac", 1.0))
        Ls = {key: layout(st) for key, st in states.items()}
        cand = {key: place(st,
                           np.array([0.5 * (Ls[key]["h"][i] + Ls[key]["h"][j])]),
                           np.array([0.5 * (Ls[key]["a"][i] + Ls[key]["a"][j])]),
                           np.array([0.5 * (Ls[key]["b"][i] + Ls[key]["b"][j])]),
                           np.array([0.5 * (Ls[key]["res"][i] + Ls[key]["res"][j])]))[0]
                for key, st in states.items()}
        old = {key: st["V"][keep].copy() for key, st in states.items()}
        for key, st in states.items():
            st["V"][keep] = old[key] + move_frac * (cand[key] - old[key])
        if not _collapse_safe(states, F, F2, keep, drop, ell_v=ell_v,
                              split_ratio=p["split_ratio"]):
            for key, st in states.items():
                st["V"][keep] = old[key]
            continue
        # surface expression error of the merged patch: the new longest edge must
        # still represent the continuous field (same test as the split, reversed)
        star = np.unique(F2[np.flatnonzero((F2 == keep).any(axis=1))])
        star = star[star != keep]
        if len(star) >= 2:
            se = np.stack([np.minimum(keep, star), np.maximum(keep, star)], axis=1)
            sc = chord_error_mm(states, se, np.arange(len(se)))
            if (sc > 0.10 * np.maximum(ell_v[keep], 3.0)).any():
                for key, st in states.items():
                    st["V"][keep] = old[key]
                continue
        F = F2
        em = edge_map(F)
        dropped.add(drop)
        spent[z] = spent.get(z, 0) + 1
        ops["collapse"].append({"edge": [i, j], "keep": int(keep), "drop": int(drop),
                                "norm": round(float(nrm[ei]), 3),
                                "zone": int(z),
                                "norm_closed": round(float(per["closed"][ei]), 3),
                                "norm_open": round(float(per["open"][ei]), 3),
                                "moved_mm": round(float(np.linalg.norm(
                                    states[list(states)[0]]["V"][keep] -
                                    old[list(states)[0]]) * 1000.0), 2)})

    # ---- flips ----------------------------------------------------------
    em = edge_map(F)
    slist = list(states.values())
    ang = np.min([face_angles(st["V"], F).min(axis=1) for st in slist], axis=0)
    # a fold must not be re-diagonalised: improving the *in-plane* minimum angle
    # of a pair of triangles that meet at 150 deg in 3-D swaps the diagonal
    # across the fold and produces a knife edge with a perfect 60/60/60 shape.
    # The dihedral is therefore a hard precondition, not a tie-breaker.
    dih_abs = np.zeros(len(F), float)
    try:
        from head_rgbd.neck_bridge import face_max_dihedral_deg
        dih_abs = face_max_dihedral_deg(slist[0]["V"], F)
    except Exception:
        dih_abs = np.zeros(len(F), float)
    fold = dih_abs > float(p.get("flip_max_dihedral_deg", 45.0))
    touched, done = set(), 0
    for r in np.flatnonzero(ang < p["flip_below_deg"]).tolist():
        if done >= p["flip_max_per_pass"]:
            break
        if fold[r]:
            continue                       # sliver inside a fold: leave it alone
        tri = F[r].tolist()
        for s in range(3):
            i, j = tri[s], tri[(s + 1) % 3]
            key = (min(i, j), max(i, j))
            if key in touched or key not in em:
                continue
            if not (free[i] and free[j]) or not _fresh(key, recent, pass_index):
                continue
            rows = list(em[key])
            before = min(float(face_angles(st["V"], F[rows]).min()) for st in slist)
            # the flip must keep the winding consistent in EVERY state
            F2 = flip_edge(F, em, i, j, V=slist[0]["V"])
            if F2 is None:
                continue
            bad = False
            for st in slist[1:]:
                tri0 = np.asarray(st["V"], float)
                ref = np.cross(tri0[F[rows[0], 1]] - tri0[F[rows[0], 0]],
                               tri0[F[rows[0], 2]] - tri0[F[rows[0], 0]])
                for rr in rows:
                    n = np.cross(tri0[F2[rr, 1]] - tri0[F2[rr, 0]],
                                 tri0[F2[rr, 2]] - tri0[F2[rr, 0]])
                    if float(n @ ref) < 0.0:
                        bad = True
            if bad:
                continue
            after = min(float(face_angles(st["V"], F2[rows]).min()) for st in slist)
            if after < before + p["flip_min_improve_deg"] or after < p["flip_min_angle_deg"]:
                continue
            if _flip_fold(slist[0]["V"], F, F2, em, rows,
                          float(p.get("flip_max_dihedral_deg", 45.0))):
                continue
            F = F2
            em = edge_map(F)
            touched.add(key)
            ops["flip"].append({"edge": [int(i), int(j)],
                                "min_angle": [round(before, 1), round(after, 1)]})
            done += 1
            break
    return F, ops, {"free": np.asarray(free, bool), "zone": zone,
                    "ell_v": np.asarray(ell_v, float), "n_vertices": len(free)}


def _flip_fold(V, F, F2, em, rows, threshold_deg):
    """Does the flip create a fold on any of the five edges it changes?

    A flip is a *planar* operation.  When the two triangles of the quad meet at
    a sharp angle in 3-D, swapping the diagonal keeps the in-plane minimum angle
    and replaces a real fold with a knife edge -- the result looks perfect to
    every 2-D test and renders as a crease.  The only reliable test is
    geometric: the new diagonal and the four outer edges must each stay below
    the fold threshold *after* the flip.
    """
    tri = np.asarray(V, float)

    def nrm(r, FF):
        t = tri[FF[r]]
        c = np.cross(t[1] - t[0], t[2] - t[0])
        l = np.linalg.norm(c)
        return c / l if l > 1e-12 else c

    shared = set(F2[rows[0]].tolist()) & set(F2[rows[1]].tolist())
    if len(shared) != 2:
        return True                       # malformed: let the caller reject it
    d = np.degrees(np.arccos(np.clip(float(nrm(rows[0], F2) @ nrm(rows[1], F2)),
                                     -1.0, 1.0)))
    if d > threshold_deg:
        return True
    quad = sorted(set(F2[rows[0]].tolist()) | set(F2[rows[1]].tolist()))
    for a in range(len(quad)):
        for b in range(a + 1, len(quad)):
            u, v = quad[a], quad[b]
            rr = em.get((u, v), [])
            outside = [r for r in rr if r not in rows]
            if not outside:
                continue
            inside = [r for r in rows
                      if u in F2[r].tolist() and v in F2[r].tolist()]
            if len(inside) != 1:
                continue
            nn = np.degrees(np.arccos(np.clip(
                float(nrm(inside[0], F2) @ nrm(outside[0], F)), -1.0, 1.0)))
            if nn > threshold_deg:
                return True
    return False


def _fresh(edge, recent, pass_index, gap=1):
    key = (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1])))
    t = recent.get(key)
    return t is None or (pass_index - t) > gap


def _collapse_safe(states, F, F2, keep, drop, ell_v=None, split_ratio=4.0 / 3.0,
                   min_angle_deg=8.0):
    """Is this merge safe and non-degrading in *every* state?

    Three tests, because the average-normal test alone is far too permissive:

    1. the kept vertex's star must keep facing roughly the same way (the star can
       still fold locally while its mean normal barely moves);
    2. every rewritten face must keep its own orientation with respect to the
       face it came from -- a single inverted face is a visible hole and a
       ray-cast artefact;
    3. no rewritten face may become a sliver, and no edge of the merged star may
       end up longer than ``split_ratio * ell``: otherwise the merge immediately
       manufactures a split candidate and the mesh oscillates (add, delete, add).
    """
    rows_a = np.flatnonzero((F == keep).any(axis=1) | (F == drop).any(axis=1))
    rows_b = np.flatnonzero((F2 == keep).any(axis=1))
    if len(rows_b) == 0:
        return False
    # the face each rewritten row is descended from: normalise ``drop`` to
    # ``keep`` on the old rows and match vertex sets exactly
    anc = {}
    for r0 in rows_a.tolist():
        key = frozenset(keep if int(x) == drop else int(x) for x in F[r0])
        anc[key] = int(r0)
    for st in states.values():
        V = st["V"]
        na = star_normal(V, F, keep, rows_a.tolist())
        nb = star_normal(V, F2, keep, rows_b.tolist())
        if np.dot(na, nb) < 0.35:
            return False
        n2 = face_normals(V, F2[rows_b])
        for k, r in enumerate(rows_b.tolist()):
            r0 = anc.get(frozenset(int(x) for x in F2[r]))
            if r0 is not None:
                n0 = face_normals(V, F[[r0]])[0]
                if float(n2[k] @ n0) < 0.30:      # that face just turned over
                    return False
        if face_angles(V, F2[rows_b]).min() < min_angle_deg:
            return False
        if (face_area_mm2(V, F2[rows_b]) < 1e-7).any():
            return False
        if ell_v is not None:
            nb_v = np.unique(F2[rows_b])
            nb_v = nb_v[nb_v != keep]
            Lk = np.linalg.norm(V[nb_v] - V[keep], axis=1) * 1000.0
            if len(Lk) and (Lk / np.maximum(ell_v[nb_v], 1e-6)).max() > split_ratio:
                return False
    return True
