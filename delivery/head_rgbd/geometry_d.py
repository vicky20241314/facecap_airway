"""Research-only, density-normalized physical displacement reconstruction.

All geometry is metres; the linear unknown is millimetres. One absolute shared
field is solved per topology. Subdivision never resets its +/-3 mm budget.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import cv2
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import lsmr

from head_rgbd.fit_b import normals
from head_rgbd.geometry_c import ray_surface, sample_depth


@dataclass(frozen=True)
class Settings:
    solver: str = 'least-squares'
    points: int = 5000
    outer_steps: int = 3
    gradient_length_mm: float = 3.0
    bending_length_mm: float = 4.0
    noise_floor_mm: float = 0.7
    maximum_slope: float = 0.45
    maximum_normal_change_deg: float = 35.0
    displacement_bound_mm: float = 3.0
    minimum_signed_area_ratio: float = 0.05
    dense_residual_bound_mm: float = 0.5
    dense_prior_strength: float = 2.0
    research_v2: bool = False
    fixed_ray_policy: str = 'strict-coverage'

    def __post_init__(self):
        if self.fixed_ray_policy not in ('strict-coverage','accounted-research'):
            raise ValueError('unknown fixed-ray policy')
        if self.fixed_ray_policy!='strict-coverage' and not self.research_v2:
            raise ValueError('accounted ray policy is an explicit v2 research option')


def fixed_ray_energy(depth_error_mm,covered,policy):
    # Every original ray remains in the denominator. The 32mm missing penalty
    # exceeds the 30mm intersection query gate; losing a hit cannot make that
    # individual observation cheaper. This is NOT a sensor-accuracy parameter.
    penalty=32. if policy=='accounted-research' else 8.
    return float(np.mean(np.where(covered,depth_error_mm,penalty)**2))


def aggregate_fixed_ray_energy(frame_scores,ray_counts,policy):
    if not frame_scores:return 0.
    return float(np.average(frame_scores,weights=ray_counts)) if policy=='accounted-research' else float(np.mean(frame_scores))


def edges_of(faces):
    return np.unique(np.sort(np.concatenate((faces[:, :2], faces[:, 1:], faces[:, [0, 2]])), axis=1), axis=0)


def area_quadrature(face_indices, areas):
    unique, frequency = np.unique(face_indices, return_counts=True)
    counts = np.zeros(len(areas)); counts[unique] = frequency
    return areas[face_indices]/counts[face_indices]/max(areas[unique].sum(), 1e-12)


def subdivide(faces, vertex_count):
    """Exact parent/child mapping, no shape modification or evidence inheritance."""
    edges = edges_of(faces)
    lookup = {tuple(edge): vertex_count+i for i, edge in enumerate(edges)}
    mids = np.array([[lookup[tuple(sorted((a, b)))], lookup[tuple(sorted((b, c)))],
                      lookup[tuple(sorted((c, a)))]] for a, b, c in faces])
    a, b, c = faces.T
    ab, bc, ca = mids.T
    children = np.stack((np.column_stack((a, ab, ca)), np.column_stack((ab, b, bc)),
                         np.column_stack((ca, bc, c)), np.column_stack((ab, bc, ca))), axis=1).reshape(-1, 3)
    rows = np.r_[np.arange(vertex_count), np.repeat(np.arange(len(edges))+vertex_count, 2)]
    cols = np.r_[np.arange(vertex_count), edges.ravel()]
    values = np.r_[np.ones(vertex_count), np.full(2*len(edges), .5)]
    prolongation = sparse.coo_matrix((values, (rows, cols)), shape=(vertex_count+len(edges), vertex_count)).tocsr()
    return children, prolongation, edges, np.repeat(np.arange(len(faces)), 4)


def child_landmarks(parent_faces, barycentric):
    """Map parent triangle barycentrics to the exact child triangle coordinates."""
    barycentric = np.asarray(barycentric)
    which = np.full(len(barycentric), 3, int)
    result = np.empty_like(barycentric)
    remaining = np.ones(len(which), bool)
    for corner in range(3):
        selected = remaining & (barycentric[:, corner] >= .5)
        which[selected] = corner
        b = barycentric[selected]
        if corner == 0:
            result[selected] = np.column_stack((2*b[:, 0]-1, 2*b[:, 1], 2*b[:, 2]))
        elif corner == 1:
            result[selected] = np.column_stack((2*b[:, 0], 2*b[:, 1]-1, 2*b[:, 2]))
        else:
            result[selected] = np.column_stack((2*b[:, 0], 2*b[:, 1], 2*b[:, 2]-1))
        remaining[selected] = False
    b = barycentric[remaining]
    result[remaining] = np.column_stack((1-2*b[:, 2], 1-2*b[:, 0], 1-2*b[:, 1]))
    return np.asarray(parent_faces)*4+which, result


def physical_regularizers(vertices, faces, settings):
    """Area quadrature of |grad d|² and |laplacian d|² on the base surface.

    Positive finite-volume weights avoid negative cotangents near thin eyelid
    triangles. Every coefficient is computed in mm, independent of ray count.
    """
    v = np.asarray(vertices)*1000
    edges = edges_of(faces)
    triangles = v[faces]
    areas = .5*np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    mass = np.zeros(len(v))
    for j in range(3):
        np.add.at(mass, faces[:, j], areas/3)
    total = max(mass.sum(), 1e-12)
    length = np.linalg.norm(v[edges[:, 0]]-v[edges[:, 1]], axis=1)
    degree = np.bincount(edges.ravel(), minlength=len(v))
    edge_mass = .5*(mass[edges[:, 0]]/np.maximum(degree[edges[:, 0]], 1)
                    + mass[edges[:, 1]]/np.maximum(degree[edges[:, 1]], 1))
    g = np.sqrt(edge_mass/total)/np.maximum(length, .05)
    incidence = sparse.coo_matrix((np.tile([1., -1.], len(edges)),
        (np.repeat(np.arange(len(edges)), 2), edges.ravel())), shape=(len(edges), len(v))).tocsr()
    grad = incidence.multiply(g[:, None])
    weights = edge_mass/np.maximum(length, .05)**2
    lap = incidence.T @ incidence.multiply(weights[:, None])
    lap = sparse.diags(1/np.maximum(mass, .01)) @ lap
    bend = sparse.diags(np.sqrt(mass/total)) @ lap
    damping = sparse.diags(np.sqrt(mass/total))
    matrix = sparse.vstack((settings.gradient_length_mm*grad,
                            settings.bending_length_mm**2*bend, .08*damping)).tocsr()
    return matrix, edges, length, areas, mass


def eye_mask(data):
    mask = np.zeros(data['depth'].shape, np.uint8)
    for ids in (slice(36, 42), slice(42, 48)):
        uv = data['uv'][ids]
        if np.isfinite(uv).all():
            cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(uv).astype(np.int32)), 1)
    return cv2.dilate(mask, np.ones((7, 7), np.uint8)).astype(bool)


def anatomical_exclusion(base, data, front_ids, mouth):
    excluded = mouth.copy()
    for i in front_ids:
        v = base[i]
        fx, fy, cx, cy = data[i]['intrinsics']
        z = np.maximum(-v[:, 2], .01)
        uv = np.rint(np.column_stack((fx*v[:, 0]/z+cx, -fy*v[:, 1]/z+cy))).astype(int)
        mask = eye_mask(data[i])
        valid = (uv[:, 0] >= 0)&(uv[:, 0] < mask.shape[1])&(uv[:, 1] >= 0)&(uv[:, 1] < mask.shape[0])
        ids = np.flatnonzero(valid)
        excluded[ids[mask[uv[ids, 1], uv[ids, 0]]]] = True
    return excluded


def observations(data, identity_ids, settings, seed=20261031):
    rng = np.random.default_rng(seed)
    points = {}
    metadata = []
    for i in identity_ids:
        xyz, uv, stats = sample_depth(data[i], settings.points, rng, True, stride=3)
        safe = ~eye_mask(data[i])[uv[:, 1], uv[:, 0]]
        points[i] = xyz[safe]
        metadata.append(dict(frame_index=int(i), **stats, eye_region_rejected=int((~safe).sum())))
    return points, metadata


def surface_state(base, directions, field_mm):
    return base+directions*field_mm[None, :, None]*.001


def safety(base, directions, faces, candidate, edges, lengths_mm, settings):
    if np.abs(candidate).max(initial=0) > settings.displacement_bound_mm+1e-7:
        return False, dict(reason='total_displacement_bound')
    slope = np.abs(candidate[edges[:, 0]]-candidate[edges[:, 1]])/np.maximum(lengths_mm, .05)
    if slope.max(initial=0) > settings.maximum_slope+1e-8:
        return False, dict(reason='physical_slope', maximum=float(slope.max()))
    min_ratio, max_angle = np.inf, 0.
    for v, n in zip(base, directions):
        tri = v[faces]
        old = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        vv = v+n*candidate[:, None]*.001
        tri = vv[faces]
        new = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        a2 = (old*old).sum(1)
        valid = a2 > 1e-24
        ratio = (old*new).sum(1)[valid]/a2[valid]
        cosine = (old*new).sum(1)[valid]/np.maximum(np.linalg.norm(old[valid], axis=1)*np.linalg.norm(new[valid], axis=1), 1e-30)
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        min_ratio = min(min_ratio, float(ratio.min()))
        max_angle = max(max_angle, float(angle.max()))
        if min_ratio < settings.minimum_signed_area_ratio or max_angle > settings.maximum_normal_change_deg:
            return False, dict(reason='orientation_or_normal_change', minimum_signed_area_ratio=min_ratio, maximum_normal_change_deg=max_angle)
    return True, dict(minimum_signed_area_ratio=min_ratio, maximum_normal_change_deg=max_angle,
                      maximum_slope=float(slope.max(initial=0)))


def solve_field(base, faces, directions, data, rows, identity_ids, excluded, settings,
                *, initial=None, locked=None, stage='regularized',reference_base=None,proxy_field=None):
    count = base.shape[1]
    field = np.zeros(count) if initial is None else initial.copy()
    locked = np.zeros(count, bool) if locked is None else locked
    regularizer, edges, lengths, areas, mass = physical_regularizers(base[0] if reference_base is None else reference_base[0], faces, settings)
    regularizer_bias=np.zeros(regularizer.shape[0])
    if settings.research_v2:
        from head_rgbd.surface_v2 import total_regularizer
        regularizer,regularizer_bias,areas,mass=total_regularizer(base,base if reference_base is None else reference_base,directions,faces,settings)
    if reference_base is not None:
        regularizer=sparse.vstack((regularizer,settings.dense_prior_strength*sparse.diags(np.sqrt(mass/max(mass.sum(),1e-12))))).tocsr()
        regularizer_bias=np.r_[regularizer_bias,np.zeros(count)]
        from head_rgbd.hierarchy_d import safety as affine_safety
        check_field=lambda candidate:affine_safety(base,reference_base,directions,faces,candidate,edges,lengths,proxy_field,settings)
    else:
        check_field=lambda candidate:safety(base,directions,faces,candidate,edges,lengths,settings)
    points, sampling = observations(data, identity_ids, settings)
    plane_data={}
    if settings.research_v2:
        from head_rgbd.surface_v2 import local_depth_planes
        for i in identity_ids:
            fx,fy,cx,cy=data[i]['intrinsics'];p=points[i];z=-p[:,2]
            uv=np.rint(np.column_stack((fx*p[:,0]/z+cx,-fy*p[:,1]/z+cy))).astype(int)
            plane_data[i]=local_depth_planes(data[i],uv)
    diagnostic_rays={}
    if settings.research_v2:
        for i in identity_ids:
            c,_=ray_surface(surface_state(base,directions,field)[i],faces,points[i],.008,.005)
            if c is not None:
                keep=~excluded[faces[c['face']]].any(1)&plane_data[i][2][c['ray']]
                diagnostic_rays[i]=np.unique(c['ray'][keep])
    def fixed_diagnostic(candidate):
        posed=surface_state(base,directions,candidate);scores=[];counts=[];missing=0
        for i,required in diagnostic_rays.items():
            if not len(required):continue
            c,_=ray_surface(posed[i],faces,points[i][required],.030,None)
            residual=np.full(len(required),8.)
            covered=np.zeros(len(required),bool)
            if c is not None:
                covered[c['ray']]=True
                residual[c['ray']]=np.abs(c['hit'][:,2]-c['target'][:,2])*1000
            missing+=int((~covered).sum());scores.append(fixed_ray_energy(residual,covered,settings.fixed_ray_policy))
            counts.append(len(required))
        return aggregate_fixed_ray_energy(scores,counts,settings.fixed_ray_policy),missing
    history = []
    support_union = np.zeros((len(identity_ids), count), bool)
    frame_weight_sum = sum(3. if rows[i]['view']=='front' else 1. for i in identity_ids)
    if settings.research_v2:
        from head_rgbd.surface_v2 import burst_weights
        normalized_frame_weights=burst_weights(rows,identity_ids)
    else:
        normalized_frame_weights={i:(3. if rows[i]['view']=='front' else 1.)/frame_weight_sum for i in identity_ids}
    qp_constraints = None
    qp_active = None
    for outer in range(settings.outer_steps):
        posed = surface_state(base, directions, field)
        matrices, targets, weights, face_medians = [], [], [], []
        support = np.zeros_like(support_union)
        accepted = []
        blocks = []
        for k, i in enumerate(identity_ids):
            corr, stats = ray_surface(posed[i], faces, points[i], .008, .005)
            accepted.append(dict(frame_index=int(i), **stats))
            if corr is None:
                continue
            ids = faces[corr['face']]
            keep = ~excluded[ids].any(1)
            if settings.research_v2:
                observed_normal,scatter,plane_good=plane_data[i]
                agreement=(corr['normal']*observed_normal[corr['ray']]).sum(1)
                keep &= plane_good[corr['ray']] & (agreement>.5)
                accepted[-1]['local_plane_or_normal_rejected']=int((~(plane_good[corr['ray']]&(agreement>.5))).sum())
            ids, bc = ids[keep], corr['bary'][keep]
            normal = plane_data[i][0][corr['ray'][keep]] if settings.research_v2 else corr['normal'][keep]
            fi = corr['face'][keep]
            accepted[-1]['identity_equations_after_anatomical_and_reliability_masks']=len(fi)
            value = bc*(directions[i][ids]*normal[:, None]).sum(2)
            A = sparse.coo_matrix((value.ravel(), (np.repeat(np.arange(len(ids)), 3), ids.ravel())),
                                   shape=(len(ids), count)).tocsr()
            measured_plane=((corr['hit'][keep]-corr['target'][keep])*normal).sum(1)
            b = -measured_plane*1000 + A@field
            unique = np.unique(fi)
            # Surface-area quadrature: duplicating rays does not increase a face's vote.
            w = area_quadrature(fi, areas)
            w *= normalized_frame_weights[i]
            w *= corr['incidence'][keep]**2
            sensitivity=value.sum(1)
            if settings.research_v2:
                from head_rgbd.surface_v2 import canonical_targets
                comparable=canonical_targets(b,sensitivity)
                # A burst gets one view vote; denominator also counts bursts.
                distance=np.linalg.norm(corr['target'][keep],axis=1)
                w*=np.minimum(1.,(np.median(distance)/np.maximum(distance,.1))**4)
                local_sigma=np.maximum(plane_data[i][1][corr['ray'][keep]],1/np.sqrt(12))
                w*=settings.noise_floor_mm**2/np.maximum(local_sigma,settings.noise_floor_mm)**2
            else:comparable=b
            support[k, np.unique(ids[bc > .05])] = True
            med = np.full(len(faces), np.nan)
            for face in unique:
                med[face] = np.nanmedian(comparable[fi==face])
            face_medians.append(med)
            blocks.append((A, b, w, fi, sensitivity))
        if not blocks:
            raise ValueError('no reliable depth correspondences for identity refinement')
        medians = np.array(face_medians)
        # Across-frame disagreement lowers confidence; absence is never a zero measurement.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            center = np.nanmedian(medians, axis=0)
            noise = 1.4826*np.nanmedian(np.abs(medians-center), axis=0)
        noise = np.maximum(np.nan_to_num(noise, nan=settings.noise_floor_mm), settings.noise_floor_mm)
        for A, b, w, fi, sensitivity in blocks:
            matrices.append(A); targets.append(b)
            projected_noise=noise[fi]*np.maximum(np.abs(sensitivity),.15) if settings.research_v2 else noise[fi]
            weights.append(w/projected_noise**2)
        A = sparse.vstack(matrices).tocsr()
        b, w = np.concatenate(targets), np.concatenate(weights)
        support_union |= support
        free_mask = (support.sum(0) >= 2)&~excluded&~locked
        free = np.flatnonzero(free_mask)
        if not len(free):
            history.append(dict(stage=stage, outer=outer, free_vertices=0, accepted=accepted))
            break
        fixed = field.copy(); fixed[free] = 0
        rw = np.sqrt(w/np.maximum(1., np.abs(A@field-b)/2.))
        system = sparse.vstack((A[:, free].multiply(rw[:, None]), regularizer[:, free])).tocsr()
        target = np.r_[(b-A@fixed)*rw, -(regularizer@fixed+regularizer_bias)]
        if settings.solver == 'qp':
            from head_rgbd.qp_d import constraints
            from head_rgbd.qp_d import solve
            if qp_active is None or not np.array_equal(qp_active, free_mask):
                if reference_base is None:
                    qp_constraints=constraints(base,directions,faces,edges,lengths,settings,free_mask)
                else:
                    from head_rgbd.hierarchy_d import constraints as affine_constraints
                    qp_constraints=affine_constraints(base,reference_base,directions,faces,edges,lengths,proxy_field,settings,free_mask)
                qp_active=free_mask.copy()
            solution, solver_info = solve(system, target, free, fixed, field, *qp_constraints)
            solver_status = solver_info['status']
            optimality = solver_info['dual_residual']
        else:
            fit = lsq_linear(system, target, bounds=(-settings.displacement_bound_mm, settings.displacement_bound_mm),
                             tol=1e-7, lsmr_tol=1e-7, max_iter=60)
            solution=fit.x; solver_status=int(fit.status); optimality=float(fit.optimality)
            solver_info=dict(method='bounded_least_squares')
        if not np.isfinite(solution).all():
            raise ValueError('nonfinite regularized field')
        proposed = field.copy(); proposed[free] = solution
        # A shared global line search preserves the smooth solution. No local clipping.
        alpha = 1.
        previous_objective = float(np.linalg.norm((A@field-b)*rw)**2+np.linalg.norm(regularizer@field+regularizer_bias)**2)
        found = False
        check = {}
        diagnostic_before,missing_before=fixed_diagnostic(field) if settings.research_v2 else (0.,0)
        diagnostic_after,missing_after=diagnostic_before,missing_before
        trials=[]
        for _ in range(18):
            candidate = field+alpha*(proposed-field)
            valid, check = check_field(candidate)
            geometric_valid=valid
            diagnostic_evaluated=False
            objective = float(np.linalg.norm((A@candidate-b)*rw)**2+np.linalg.norm(regularizer@candidate+regularizer_bias)**2)
            reason=check.get('reason') if not valid else ('linear_objective' if objective>previous_objective+1e-12 else None)
            if valid and objective <= previous_objective+1e-12 and settings.research_v2:
                diagnostic_after,missing_after=fixed_diagnostic(candidate)
                diagnostic_evaluated=True
                if settings.fixed_ray_policy=='accounted-research':
                    valid=diagnostic_after<=diagnostic_before+1e-10
                    if not valid:reason='fixed_ray_penalized_mse'
                else:
                    valid=missing_after<=missing_before and diagnostic_after<=diagnostic_before*1.02+.002
                    if not valid:reason='fixed_ray_coverage' if missing_after>missing_before else 'fixed_ray_depth_mse'
            if settings.research_v2:
                trials.append(dict(alpha=alpha,geometry_valid=bool(geometric_valid),reason=reason,
                    objective=objective,fixed_ray_mse_mm2=diagnostic_after if diagnostic_evaluated else None,
                    missing_rays=missing_after if diagnostic_evaluated else None,safety=check.copy()))
            if valid and objective <= previous_objective+1e-12:
                field = candidate; found = True; break
            alpha *= .5
        record = dict(stage=stage, outer=outer, free_vertices=len(free), solver_status=solver_status,
                      solver=solver_info, optimality=optimality, step_accepted=found, shared_step_scale=alpha,
                      accepted=accepted, safety=check, objective_before=previous_objective,
                      objective_after=objective, noise_median_mm=float(np.median(noise)))
        record['cross_frame_statistic']='canonical_identity_scalar_mm' if settings.research_v2 else 'camera_plane_target_mm'
        record['regularizer']='FEM total vector curvature change plus scalar gradient' if settings.research_v2 else 'finite_volume_scalar'
        if settings.research_v2:
            record['line_search_trials']=trials
            record['fixed_ray_diagnostic']=dict(before_mse_mm2=diagnostic_before,after_mse_mm2=diagnostic_after,
                missing_before=missing_before,missing_after=missing_after,ray_count=sum(map(len,diagnostic_rays.values())),
                policy=settings.fixed_ray_policy,missing_penalty_mm=32. if settings.fixed_ray_policy=='accounted-research' else 8.,
                aggregation='uniform over every original fixed ray' if settings.fixed_ray_policy=='accounted-research' else 'equal weight per frame MSE',
                allowance='penalized MSE must not increase; every original ray including missing hits retained; coverage independently evaluated' if settings.fixed_ray_policy=='accounted-research' else 'at most 2 percent MSE plus 0.002 mm2 per accepted step; all original diagnostic rays retained')
        history.append(record)
        print(f'{stage} outer={outer} free={len(free)} alpha={alpha:.5f} accepted={found}', flush=True)
    valid, check = check_field(field)
    if not valid:
        raise ValueError(f'final shared geometry violates safety: {check}')
    return field, support_union, dict(settings=asdict(settings), sampling=sampling, history=history, final_safety=check,
                                     maximum_absolute_displacement_mm=float(np.abs(field).max()))


def evidence(vertices, faces, data, points=5000):
    """Independent visibility queries on THIS topology; no label interpolation."""
    rng = np.random.default_rng(7891)
    observed = np.zeros((len(data), vertices.shape[1]), bool)
    records = []
    for i, d in enumerate(data):
        xyz, _, sampling = sample_depth(d, points, rng, stride=3)
        corr, stats = ray_surface(vertices[i], faces, xyz, .008, .005)
        if corr is not None:
            ids = faces[corr['face']]
            observed[i, np.unique(ids[corr['bary'] > .05])] = True
        records.append(dict(frame_index=i, **sampling, **stats))
    counts = observed.sum(0)
    return dict(observed_frame_mask=observed, observed_frames=counts,
                region_label=np.where(counts >= 2, 2, np.where(counts > 0, 1, 0)).astype(np.uint8),
                confidence=np.minimum(counts/3., 1.)), records
