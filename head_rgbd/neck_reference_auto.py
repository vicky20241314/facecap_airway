"""Reference-derived anterior-neck appearance editing; never an observation claim.

The reference is a Y-up, +Z-front rest mesh in metres. A bounded anterior ROI
is separated from its broad neck curvature. The resulting signed, smooth field
is scaled with neck width and transported through a shared template binding.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline
from scipy.ndimage import gaussian_filter
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from head_rgbd import neck_appearance as appearance
from head_rgbd import neck_prominence as prominence
from head_rgbd import neck_subject_v3 as subject
from head_rgbd.mesh_integrity import edge_records, load_obj


def smooth_gate(t):
    t = np.clip(t, 0., 1.)
    return t ** 3 * (10. + t * (-15. + 6. * t))


def extract_reference(path, *, centre_y_mm=-105.5):
    """Bake this reference's bounded laryngeal profile, not a global maximum.

    ROI bounds are explicit design coordinates for the supplied reference.
    Missing interpolation cells are rejected; no unsupported gap is filled.
    """
    path = Path(path)
    vertices, faces = load_obj(path)
    points = vertices * 1000.
    normals = prominence.vertex_normals(vertices, faces)
    select = ((normals[:, 2] > .15) & (np.abs(points[:, 0]) < 50.)
              & (points[:, 1] > -140.) & (points[:, 1] < -70.))
    interp = LinearNDInterpolator(points[select, :2], points[select, 2])
    x = np.linspace(-34., 34., 137)
    y = np.linspace(-125., -81., 89)
    X, Y = np.meshgrid(x, y)
    Z = interp(X, Y)
    if not np.isfinite(Z).all():
        raise ValueError("reference anterior ROI has missing observations")
    U, Q = X / 34., (Y + 103.) / 22.
    basis = np.stack([np.ones_like(X), U, Q, U*U, U*Q, Q*Q], axis=-1)
    outer = (np.abs(U) > .72) | (np.abs(Q) > .77)
    coef = np.linalg.lstsq(basis[outer], Z[outer], rcond=None)[0]
    residual = gaussian_filter(Z - basis @ coef, sigma=1.0)
    # Exclude the under-chin turn (y > -86) and the remote neck/shoulder.
    gate = (smooth_gate((34. - np.abs(X)) / 9.)
            * smooth_gate((Y + 125.) / 9.)
            * smooth_gate((-86. - Y) / 9.))
    field = residual * gate
    core = (np.abs(X) < 20.) & (Y > -116.) & (Y < -94.)
    peak_index = np.unravel_index(np.argmax(np.where(core, field, -np.inf)), field.shape)
    slab = np.abs(points[:, 1] - centre_y_mm) < 3.
    width = float(np.ptp(np.percentile(points[slab, 0], [2, 98])))
    asset = {
        "schema": "head_rgbd.neck_reference_field.v1",
        "source_path": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_axes": "Y up, +Z anterior", "source_units": "m",
        "editing_mode": True, "subject_observation_matched": False,
        "centre_y_mm": float(centre_y_mm), "reference_neck_width_mm": width,
        "x_mm": x.tolist(), "y_mm": y.tolist(), "field_mm": field.tolist(),
        "background_coefficients_mm": coef.tolist(),
        "valid_fraction": 1., "core_peak_mm": float(field[peak_index]),
        "core_peak_xy_mm": [float(X[peak_index]), float(Y[peak_index])],
        "negative_extent_mm": float(field.min()),
        "method": "anterior ROI minus outer-ring quadratic; 0.5-mm smoothing; C2 edge fade",
        "excluded": "under-chin turn above -86 mm; shoulders; all posterior surfaces",
    }
    return asset, vertices, faces


def evaluate_field(asset, u_mm, v_mm, *, scale=1., strength=1.):
    """Similarity-scaled reference field. No peak-height normalisation."""
    x, y = np.asarray(asset["x_mm"]), np.asarray(asset["y_mm"])
    field = np.asarray(asset["field_mm"])
    spline = RectBivariateSpline(y, x, field, kx=3, ky=3, s=0)
    u = np.asarray(u_mm) / scale
    v = np.asarray(v_mm) / scale + asset["centre_y_mm"]
    inside = (u >= x[0]) & (u <= x[-1]) & (v >= y[0]) & (v <= y[-1])
    result = np.zeros_like(u, dtype=float)
    result[inside] = spline.ev(v[inside], u[inside]) * scale * strength
    return result


def local_intersections(vertices, faces, moved):
    """Exact affected-edge / triangle interior crossings, t in [0,1].

    Broad phase uses bounding spheres and boxes; the reported count is the
    number of crossing edge/face pairs. Shared-edge contacts are legal. A
    separate coplanar strict-interior test covers overlapping flat triangles.
    """
    tri = vertices[faces]
    centres = tri.mean(axis=1)
    radii = np.linalg.norm(tri - centres[:, None], axis=2).max(axis=1)
    tree = cKDTree(centres)
    maximum = float(radii.max())
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    edges, _, _, _, _ = edge_records(faces)
    edges = edges[moved[edges].any(axis=1)]
    examples = []
    tested = 0
    for i, j in edges:
        a, b = vertices[i], vertices[j]
        d = b-a
        length = np.linalg.norm(d)
        if length < 1e-12:
            continue
        candidates = np.asarray(tree.query_ball_point((a+b)/2., maximum+length/2.+1e-10), dtype=int)
        box = ((hi[candidates] >= np.minimum(a,b)-1e-10).all(axis=1)
               & (lo[candidates] <= np.maximum(a,b)+1e-10).all(axis=1))
        candidates = candidates[box]
        candidates = candidates[~((faces[candidates] == i).any(axis=1)
                                   & (faces[candidates] == j).any(axis=1))]
        if not len(candidates):
            continue
        tested += len(candidates)
        T = tri[candidates]
        e1, e2 = T[:,1]-T[:,0], T[:,2]-T[:,0]
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        norm = np.linalg.norm(np.cross(e1,e2),axis=1)
        ok = np.abs(det) > 1e-10*length*norm
        inv = np.divide(1.,det,out=np.zeros_like(det),where=ok)
        tv = a-T[:,0]
        u = np.einsum("ij,ij->i",tv,p)*inv
        q = np.cross(tv,e1)
        v = q@d*inv
        t = np.einsum("ij,ij->i",e2,q)*inv
        crossing = ok & (t>1e-8)&(t<1.-1e-8)&(u>1e-8)&(v>1e-8)&(u+v<1.-1e-8)
        for fi in candidates[crossing]:
            examples.append({"edge":[int(i),int(j)],"face":int(fi),"type":"interior_crossing"})
        # Coplanar crossings are tested in a dimensionless projected plane.
        coplanar = (~ok) & (np.abs(np.einsum("ij,ij->i",tv,np.cross(e1,e2))) < 1e-10*norm)
        for k in np.flatnonzero(coplanar):
            triangle = T[k]
            normal = np.cross(e1[k],e2[k]); drop = int(np.argmax(np.abs(normal)))
            if np.linalg.norm(normal)<1e-18:
                continue
            dims = [j for j in range(3) if j != drop]
            A,B,C = triangle[:,dims]
            pa,pb = a[dims],b[dims]
            denom = np.cross(B-A,C-A)
            if abs(denom)<1e-20:
                continue
            s0,s1 = np.cross(pa-A,C-A)/denom,np.cross(pb-A,C-A)/denom
            r0,r1 = np.cross(B-A,pa-A)/denom,np.cross(B-A,pb-A)/denom
            lower,upper = 1e-8,1.-1e-8
            for start,finish in ((s0,s1),(r0,r1),(1.-s0-r0,1.-s1-r1)):
                slope = finish-start
                if abs(slope)<1e-15:
                    if start<=1e-8:
                        upper=-1.
                elif slope>0:
                    lower=max(lower,(1e-8-start)/slope)
                else:
                    upper=min(upper,(1e-8-start)/slope)
            if upper>lower:
                examples.append({"edge":[int(i),int(j)],"face":int(candidates[k]),"type":"coplanar_interior"})
        if len(examples)>=20:
            break
    return {"count":len(examples),"examples":examples[:20],"affected_edges":int(len(edges)),
            "candidate_tests":int(tested),"exhaustive":len(examples)<20,
            "method":"dimensionless segment t; strict triangle interior; local AABB broad phase"}


def build_states(references, dense_vertices, dense_faces, asset, *, strength=1.):
    states = ("closed", "open")
    base = {}
    for state in states:
        ref = references[state]
        V = np.asarray(dense_vertices[ref["stages"]["dense"]["frame_index"]],float)
        M = np.asarray(ref["animated_model_to_reference_camera_4x4"])
        R,t = M[:3,:3],M[:3,3]
        model = (V-t)@R
        lm = (np.asarray(ref["stages"]["dense"]["landmarks_m"])-t)@R
        axes = prominence._direction_frame(lm)
        N = prominence.vertex_normals(model,dense_faces)
        protection = prominence._protected_mask(model,lm,N,axes)["mask"]
        base[state] = {"vertices":V,"model":model,"protection":protection,
                       "up":R@axes["up"],"forward":R@axes["forward"],
                       "normals":N@R.T,"landmarks_camera":np.asarray(ref["stages"]["dense"]["landmarks_m"])}
    # Automatic anterior midline binding, about 30 mm below the closed chin.
    # Minimise positional distance, not maximum height; reject protected sites.
    closed = base["closed"]
    M = np.asarray(references["closed"]["animated_model_to_reference_camera_4x4"])
    lm = (closed["landmarks_camera"]-M[:3,3])@M[:3,:3]
    target_y = lm[prominence.CHIN,1]-.030
    Q = closed["model"]
    possible = (np.abs(Q[:,0])<.012)&(np.abs(Q[:,1]-target_y)<.012)
    possible &= ~base["closed"]["protection"] & ~base["open"]["protection"]
    for s in states:
        possible &= (base[s]["normals"]@base[s]["forward"])>.45
    if not possible.any():
        raise ValueError("no shared, unprotected anterior-neck binding")
    score = (Q[:,0]/.004)**2+((Q[:,1]-target_y)/.006)**2
    centre = int(np.argmin(np.where(possible,score,np.inf)))
    slab = abs(Q[:,1]-Q[centre,1])<.003
    width = float(np.ptp(np.percentile(Q[slab,0],[2,98]))*1000.)
    scale = width/asset["reference_neck_width_mm"]
    if not .65<scale<1.5:
        raise ValueError("target/reference neck scale is implausible; inspect coordinates")
    for s,item in base.items():
        V = item["vertices"]
        nearby = (np.linalg.norm(V-V[centre],axis=1)<.012) & ~item["protection"]
        normal = item["normals"][nearby].mean(axis=0); normal/=np.linalg.norm(normal)
        up = item["up"]-np.dot(item["up"],normal)*normal; up/=np.linalg.norm(up)
        u = np.cross(up,normal); u/=np.linalg.norm(u)
        v = np.cross(normal,u)
        item["frame"] = subject.LocalNeckFrame(V[centre].copy(),normal,up,u,v,
                    {"source":"shared_anterior_template_binding","base_vertex":centre})
    mask = np.zeros(len(dense_faces),bool)
    for item in base.values():
        mask |= np.linalg.norm(item["vertices"][dense_faces].mean(axis=1)-item["frame"].origin,axis=1)<.046*scale
    sub1 = subject.subdivide_local(closed["vertices"],dense_faces,face_mask=mask)
    e1 = sub1["midpoint_edges"]
    mask = np.zeros(len(sub1["faces"]),bool)
    for item in base.values():
        V = appearance._rebuild_refined(item["vertices"],e1,[])
        mask |= np.linalg.norm(V[sub1["faces"]].mean(axis=1)-item["frame"].origin,axis=1)<.036*scale
    sub2 = subject.subdivide_local(sub1["vertices"],sub1["faces"],face_mask=mask)
    e2,F = sub2["midpoint_edges"],sub2["faces"]
    results = {}
    for s,item in base.items():
        V = appearance._rebuild_refined(item["vertices"],e1,e2)
        protected = list(item["protection"])
        for edges in (e1,e2):
            protected.extend([protected[i] or protected[j] for i,j in edges])
        protected = np.asarray(protected,bool)
        frame = item["frame"]
        uv = frame.uv(V)
        field = evaluate_field(asset,uv[:,0],uv[:,1],scale=scale,strength=strength)
        # C2 distance fade, restricted to the anterior sheet and protected rim.
        distance = cKDTree(V[protected]).query(V)[0]*1000.
        gate = smooth_gate(distance/12.)
        gate *= subject._region_window(V,frame,protected)
        applied = field*gate
        shaped = V+applied[:,None]*1e-3*frame.n
        moved = np.linalg.norm(shaped-V,axis=1)>1e-10
        guards = appearance.geometry_guards(rest=V,shaped=shaped,faces=F,moved=moved)
        intersections = local_intersections(shaped,F,moved)
        peak = float(applied.max())
        metrics = {"peak_outward_mm":peak,"max_inward_mm":float(applied.min()),
                   "protected_displacement_mm":float(np.linalg.norm(shaped[protected]-V[protected],axis=1).max()*1000.),
                   "centre_camera_m":frame.origin.tolist(),"centre_clearance_mm":float(distance[centre]),
                   "moved_vertices":int(moved.sum())}
        for level in (.1,.5):
            take = applied>=peak*level
            metrics[f"width_{int(level*100)}_mm"] = float(np.ptp(uv[take,0]))
            metrics[f"length_{int(level*100)}_mm"] = float(np.ptp(uv[take,1]))
        results[s] = {**item,"refined_vertices":V,"shaped_vertices":shaped,
                      "protected_refined":protected,"applied_mm":applied,"uv_mm":uv,
                      "guards":guards,"intersections":intersections,"metrics":metrics}
    return {"states":results,"faces":F,"centre_vertex":centre,"scale":float(scale),
            "target_neck_width_mm":width,"strength":strength,
            "accepted":all(appearance.accept_guards(r["guards"]) and r["intersections"]["count"]==0
                           and r["metrics"]["protected_displacement_mm"]==0 for r in results.values())}


def _selected_components(selected, faces):
    """Connected components of a vertex subset using mesh-edge adjacency."""
    idx = np.flatnonzero(selected)
    if not len(idx):
        return idx, np.zeros(0, dtype=int), 0
    lookup = -np.ones(len(selected), dtype=int)
    lookup[idx] = np.arange(len(idx))
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    keep = selected[edges[:, 0]] & selected[edges[:, 1]]
    sub = lookup[edges[keep]]
    graph = sparse.coo_matrix((np.ones(len(sub)), (sub[:, 0], sub[:, 1])),
                              shape=(len(idx), len(idx)))
    count, labels = connected_components(graph, directed=False)
    return idx, labels, count


def _component_area_mm2(vertices_m, faces, subset):
    mask = np.isin(faces, subset).all(axis=1)
    tri = vertices_m[faces[mask]]
    if not len(tri):
        return 0.
    cross = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return float(cross.sum() * .5 * 1e6)


def _component_table(uv, applied, vertices_m, faces, selected, apex_index, limit=12, valid=None):
    if valid is not None:
        selected = selected & valid
    idx, labels, count = _selected_components(selected, faces)
    rows = []
    for k in range(count):
        sub = idx[labels == k]
        hi = sub[int(np.argmax(applied[sub]))]
        lo = sub[int(np.argmin(applied[sub]))]
        rows.append({
            "vertices": int(len(sub)),
            "uv_centroid_mm": [round(float(v), 2) for v in uv[sub].mean(axis=0)],
            "uv_bbox_mm": [round(float(v), 2) for v in
                           (uv[sub, 0].min(), uv[sub, 0].max(), uv[sub, 1].min(), uv[sub, 1].max())],
            "max_mm": round(float(applied[hi]), 3),
            "min_mm": round(float(applied[lo]), 3),
            "area_mm2": round(_component_area_mm2(vertices_m, faces, sub), 2),
            "contains_apex": bool(apex_index in sub),
        })
    rows.sort(key=lambda r: r["vertices"], reverse=True)
    return {"component_count": int(count), "components": rows[:limit],
            "truncated": bool(count > limit)}


def _profiles(uv, applied, apex_uv, valid):
    mask = valid & ((np.linalg.norm(uv - apex_uv, axis=1) < 60.) | (np.abs(applied) > .05))
    result = {}
    try:
        interp = LinearNDInterpolator(uv[mask], applied[mask])
    except Exception:  # noqa: BLE001 - degenerate projection is reported, not hidden
        return {"error": "profile interpolation failed"}
    for name, axis, fixed in (
            ("longitudinal_u0", "v", 0.),
            ("transverse_apex", "u", float(apex_uv[1]))):
        if name == "transverse_apex":
            span = np.arange(-36., 36. + 1e-9, .5)
            pts = np.column_stack([span, np.full_like(span, float(apex_uv[1]))])
        else:
            span = np.arange(float(apex_uv[1]) - 32., float(apex_uv[1]) + 24. + 1e-9, .5)
            pts = np.column_stack([np.full_like(span, fixed), span])
        values = interp(pts)
        result[name] = {
            "axis": axis,
            "axis_mm": [round(float(v), 2) for v in span],
            "disp_mm": [None if not np.isfinite(v) else round(float(v), 3) for v in values],
            "finite_fraction": float(np.isfinite(values).mean()),
        }
    return result


def _apex_curvature(uv, applied, apex_index, valid):
    apex = uv[apex_index]
    rows = []
    for window in (2., 3., 4.):
        take = valid & (np.linalg.norm(uv - apex, axis=1) <= window)
        if int(take.sum()) < 6:
            continue
        du, dv = uv[take, 0] - apex[0], uv[take, 1] - apex[1]
        design = np.stack([du*du, du*dv, dv*dv, du, dv, np.ones_like(du)], axis=1)
        coef = np.linalg.lstsq(design, applied[take], rcond=None)[0]
        residual = applied[take] - design @ coef
        hessian = np.array([[2.*coef[0], coef[1]], [coef[1], 2.*coef[2]]])
        eig = np.linalg.eigvalsh(hessian)
        curvature = float(np.max(np.abs(eig)))
        rows.append({
            "window_mm": window,
            "points": int(take.sum()),
            "radius_mm": round(1./curvature, 2) if curvature > 1e-9 else None,
            "anisotropy": round(float(np.max(np.abs(eig))/max(np.min(np.abs(eig)), 1e-9)), 2),
            "fit_rms_mm": round(float(np.sqrt(np.mean(residual**2))), 3),
            "bump_sign_ok": bool(eig.max() < 0.),
        })
    return rows


def analyse_applied(state, faces, *, residual_threshold_mm=.05):
    """Audit the realised appearance field: contours, islands, profiles, bluntness.

    All extents are in the shared anterior (u, v) frame; areas are the summed
    3D areas of triangles whose three vertices pass the threshold, so they are
    a conservative lower bound of the level-set interior. Back-of-neck vertices
    project into the same (u, v) frame, so the audit keeps only front-facing
    vertices and reports how many back-side projections were excluded.
    """
    uv = np.asarray(state["uv_mm"], float)
    applied = np.asarray(state["applied_mm"], float)
    vertices = np.asarray(state["refined_vertices"], float)
    normals = prominence.vertex_normals(vertices, faces)
    front = (normals @ np.asarray(state["frame"].n)) > .2
    projected_back = (~front) & (np.linalg.norm(uv, axis=1) < 45.)
    peak = float(applied[front].max())
    apex_index = int(np.argmax(np.where(front, applied, -np.inf)))
    apex_uv = uv[apex_index]
    levels = {}
    for fraction in (.5, .1):
        take = front & (applied >= peak*fraction)
        table = _component_table(uv, applied, vertices, faces, take, apex_index, valid=front)
        central = next((r for r in table["components"] if r["contains_apex"]), None)
        levels[f"{fraction}"] = {
            "level": fraction,
            "threshold_mm": round(peak*fraction, 3),
            "vertex_count": int(take.sum()),
            "global_bbox_u_mm": [round(float(uv[take, 0].min()), 2),
                                 round(float(uv[take, 0].max()), 2)],
            "global_bbox_v_mm": [round(float(uv[take, 1].min()), 2),
                                 round(float(uv[take, 1].max()), 2)],
            "global_width_mm": round(float(np.ptp(uv[take, 0])), 2),
            "global_length_mm": round(float(np.ptp(uv[take, 1])), 2),
            "central": central,
            "component_count": table["component_count"],
            "components": table["components"],
            "truncated": table["truncated"],
        }
    positive = applied >= residual_threshold_mm
    negative = applied <= -residual_threshold_mm
    return {
        "peak_outward_mm": round(peak, 3),
        "apex_uv_mm": [round(float(v), 2) for v in apex_uv],
        "levels": levels,
        "residual_audit": {
            "threshold_mm": residual_threshold_mm,
            "positive": _component_table(uv, applied, vertices, faces, positive, apex_index,
                                         valid=front),
            "negative": _component_table(uv, applied, vertices, faces, negative, apex_index,
                                         valid=front),
        },
        "profiles": _profiles(uv, applied, apex_uv, front),
        "apex_curvature": _apex_curvature(uv, applied, apex_index, front),
        "support": {
            "uv_range_u_mm": [round(float(uv[front, 0].min()), 2),
                              round(float(uv[front, 0].max()), 2)],
            "uv_range_v_mm": [round(float(uv[front, 1].min()), 2),
                              round(float(uv[front, 1].max()), 2)],
            "front_facing_vertices": int(front.sum()),
            "back_projections_excluded": int(projected_back.sum()),
            "positive_vertices": int((applied[front] > 0).sum()),
            "zero_vertices": int((applied[front] == 0).sum()),
            "negative_vertices": int((applied[front] < 0).sum()),
            "negative_min_mm": round(float(applied[front].min()), 3),
        },
        "method": "front-facing vertex sheet (normal . frame_n > 0.2; back-of-neck uv "
                  "projections excluded and counted); level sets in the shared (u,v) frame; "
                  "3D triangle area summed inside each component (conservative); profiles by "
                  "linear interpolation; apex radius from a local quadratic height fit.",
    }


def background_sensitivity(path, *, centre_y_mm=-105.5):
    """Diagnose whether the extracted peak depends on the background setting.

    The delivered field uses the outer-ring quadratic.  Two alternatives are
    evaluated on the same ROI and gate: an outer-ring cubic and a quadratic on
    a wider outer ring.  This function only reports sensitivity; it does not
    change the delivered field or mesh.
    """
    path = Path(path)
    vertices, faces = load_obj(path)
    points = vertices * 1000.
    normals = prominence.vertex_normals(vertices, faces)
    select = ((normals[:, 2] > .15) & (np.abs(points[:, 0]) < 50.)
              & (points[:, 1] > -140.) & (points[:, 1] < -70.))
    interp = LinearNDInterpolator(points[select, :2], points[select, 2])
    x = np.linspace(-34., 34., 137)
    y = np.linspace(-125., -81., 89)
    X, Y = np.meshgrid(x, y)
    Z = interp(X, Y)
    if not np.isfinite(Z).all():
        raise ValueError("reference anterior ROI has missing observations")
    U, Q = X / 34., (Y + 103.) / 22.
    gate = (smooth_gate((34. - np.abs(X)) / 9.)
            * smooth_gate((Y + 125.) / 9.)
            * smooth_gate((-86. - Y) / 9.))
    core = (np.abs(X) < 20.) & (Y > -116.) & (Y < -94.)
    variants = []
    reference_core = None
    for name, order, outer in (
            ("outer_quadratic_used", 2, (np.abs(U) > .72) | (np.abs(Q) > .77)),
            ("outer_cubic", 3, (np.abs(U) > .72) | (np.abs(Q) > .77)),
            ("outer_quadratic_wider_ring", 2, (np.abs(U) > .62) | (np.abs(Q) > .67))):
        if order == 2:
            basis = np.stack([np.ones_like(X), U, Q, U*U, U*Q, Q*Q], axis=-1)
        else:
            basis = np.stack([np.ones_like(X), U, Q, U*U, U*Q, Q*Q,
                              U**3, U*U*Q, U*Q*Q, Q**3], axis=-1)
        coef = np.linalg.lstsq(basis[outer], Z[outer], rcond=None)[0]
        field = gaussian_filter(Z - basis @ coef, sigma=1.0) * gate
        peak_index = np.unravel_index(np.argmax(np.where(core, field, -np.inf)), field.shape)
        peak = float(field[peak_index])
        row = field[peak_index[0]]
        take = np.flatnonzero(row >= peak*.5)
        lo = take[take <= peak_index[1]]
        hi = take[take >= peak_index[1]]
        width = float(x[hi.max()] - x[lo.min()]) if len(lo) and len(hi) else None
        column = field[:, peak_index[1]]
        rows = np.flatnonzero(column >= peak*.5)
        lo_v = rows[rows <= peak_index[0]]
        hi_v = rows[rows >= peak_index[0]]
        length = float(y[hi_v.max()] - y[lo_v.min()]) if len(lo_v) and len(hi_v) else None
        if reference_core is None:
            reference_core = field[core]
            correlation = 1.
            max_diff = 0.
        else:
            correlation = float(np.corrcoef(reference_core, field[core])[0, 1])
            max_diff = float(np.max(np.abs(field[core] - reference_core)))
        variants.append({
            "setting": name, "background_order": order,
            "outer_ring": "|U|>0.72 or |Q|>0.77" if "wider" not in name
                          else "|U|>0.62 or |Q|>0.67",
            "peak_mm": round(peak, 3),
            "peak_xy_mm": [round(float(X[peak_index]), 2), round(float(Y[peak_index]), 2)],
            "negative_extent_mm": round(float(field.min()), 3),
            "width50_mm": None if width is None else round(width, 2),
            "length50_mm": None if length is None else round(length, 2),
            "correlation_core": round(correlation, 5),
            "max_abs_diff_core_mm": round(max_diff, 3),
        })
    return {
        "primary": "outer_quadratic_used",
        "note": "diagnostic only; the delivered field and mesh always use the primary "
                "outer-ring quadratic setting",
        "core": "|x|<20 mm, y in (-116,-94) mm",
        "variants": variants,
    }
