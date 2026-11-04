"""Low-frequency identity pass for alternating fixed-shape motion refinement."""
import argparse,json,time
from pathlib import Path
import numpy as np
from head_rgbd.fit_d import load_npz
from head_rgbd.fit_b import normals
from head_rgbd.geometry_d import Settings,solve_field,anatomical_exclusion,surface_state
from head_rgbd.geometry_c import front_protected_vertices
from head_rgbd.reporting import write_json
from head_rgbd.audit import sha


def run(sequence,initial,output,fixed_ray_policy='strict-coverage',
        shape_fusion='guarded',joint_front_confidence=.15,
        support_min_information=0.,coverage_allowance_ratio=0.,anchor_prior_weight=0.,
        identity_views='all',information_measure='barycentric',information_damping_cap=10.,view_balance=False,
        reuse_anchor=False):
    joint=shape_fusion=='joint'
    start=time.perf_counter();output.mkdir(parents=True,exist_ok=False)
    source=json.loads((initial/'report.json').read_text(encoding='utf-8'))
    if source['sequence_sha256']!=sha(sequence):raise ValueError('sequence mismatch')
    info=json.loads(sequence.read_text(encoding='utf-8'));rows=source['frames'];p=load_npz(initial/'parameters.npz')
    data=[load_npz(sequence.parent/next(r['file'] for r in info['frames'] if r['id']==row['id'])) for row in rows]
    base=p['posed_vertices'].astype(float);f=p['faces'];direction=np.array([normals(v,f) for v in base])
    identity=[int(i) for i in p['identity_selected']];front=[i for i,row in enumerate(rows) if row['view']=='front']
    # Joint fusion: every closed identity view feeds the shared low-frequency field.
    # Guarded control keeps the original front-only anchor stage.
    front_identity=[i for i in identity if rows[i]['view']=='front']
    ids=identity if (joint and identity_views=='all') else front_identity
    excluded=anatomical_exclusion(base,data,front,p['mouth_identity_excluded'])
    s=Settings(solver='qp',research_v2=True,points=5000,outer_steps=2,gradient_length_mm=8.,bending_length_mm=8.,
               noise_floor_mm=1/np.sqrt(12),displacement_bound_mm=2.5,fixed_ray_policy=fixed_ray_policy,
               support_min_information=support_min_information,
               coverage_allowance_ratio=coverage_allowance_ratio,
               anchor_prior_weight=anchor_prior_weight,
               information_measure=information_measure,
               information_damping_cap=information_damping_cap,view_balance=view_balance)
    anchor,support_anchor,log=solve_field(base,f,direction,data,rows,ids,excluded,s,
                             stage='joint_low_frequency_all_views' if joint else 'alternating_low_frequency_front')
    protected=front_protected_vertices(surface_state(base,direction,anchor),f,data,front)
    # Effective shape frames: the frames that may update shared geometry. In the
    # side ablation every shared-geometry solve uses this set, not just the first.
    shape_ids=ids if joint else identity
    if reuse_anchor:
        # The second solve moved the field 0.0014 mm on the reference capture; it
        # re-solves the same observations from an already-converged start.
        field,support,side_log=anchor,support_anchor,dict(stage='reused_first_solve',reused=True,
            maximum_absolute_displacement_mm=float(np.abs(anchor).max()),history=[])
        policy='first solve reused; no second low-frequency solve'
        moved=float(np.abs(field).max())
        steps=[h for log_pair in (log,) for h in (log_pair.get('history') or [])]
        write_json(output/'report.json',dict(sequence_sha256=sha(sequence),initial=str(initial),elapsed_s=time.perf_counter()-start,
            shape_fusion=shape_fusion,joint_front_confidence=joint_front_confidence,
            effective_shape_frame_ids=[rows[i]['id'] for i in shape_ids],
            front_locked_vertices=0,front_downweighted_vertices=0,
            maximum_absolute_displacement_mm=moved,free_vertices=[h.get('free_vertices') for h in steps],
            stages=[log,side_log],policy=policy))
        np.savez_compressed(output/'identity_field.npz',deviation_m=field*.001,posed_vertices=base,faces=f,
            regularized_vertices=surface_state(base,direction,field),identity_selected=identity,
            effective_shape_frame_ids=[rows[i]['id'] for i in shape_ids],front_anchor_deviation_m=anchor*.001)
        return
    if joint and anchor_prior_weight > 0:
        # Observation confidence stays per equation; the frontal anchor is an
        # explicit prior on the value, so an oblique observation that sees the
        # region keeps its own weight and can outvote the prior.
        strength=np.where(protected,1.,s.anchor_side_strength)
        field,support,side_log=solve_field(base,f,direction,data,rows,shape_ids,excluded,s,initial=anchor,
            anchor_field=anchor,anchor_strength=strength,stage='joint_anchor_prior_low_frequency')
        policy=('shared low frequency field over every closed view; front-covered vertices held by an '
                f'explicit anchor prior (lambda {anchor_prior_weight:g} mm^-2, side {s.anchor_side_strength:g}) '
                'instead of a hard lock; passed to next motion correction as FIXED identity')
    elif joint:
        # Continuous confidence instead of a hard front lock: a front-covered
        # vertex keeps only this fraction of the data weight, so a reliable oblique
        # observation can still correct it against the prior.
        soft=np.where(protected,joint_front_confidence,1.)
        field,support,side_log=solve_field(base,f,direction,data,rows,shape_ids,excluded,s,initial=anchor,
            soft_lock=soft,stage='joint_confidence_weighted_low_frequency')
        policy=('shared low frequency field over every closed view; front-covered vertices '
                f'weighted {joint_front_confidence:g} instead of locked; passed to next motion correction as FIXED identity')
    else:
        field,support,side_log=solve_field(base,f,direction,data,rows,identity,excluded,s,initial=anchor,
            locked=protected,stage='alternating_guarded_low_frequency')
        policy='shared low frequency field, guarded original front; passed to next motion correction as FIXED identity'
    np.savez_compressed(output/'identity_field.npz',deviation_m=field*.001,posed_vertices=base,faces=f,
        regularized_vertices=surface_state(base,direction,field),identity_selected=identity,
        effective_shape_frame_ids=[rows[i]['id'] for i in shape_ids],front_anchor_deviation_m=anchor*.001)
    moved = float(np.abs(field).max())
    steps = [h for log_pair in (log, side_log) for h in (log_pair.get('history') or [])]
    if joint and all(h.get('status') == 'skipped_no_free_vertices' for h in steps):
        # A joint chain whose identity stage solved nothing is not a degraded run,
        # it is a chain that never applied the fusion it claims. Fail loudly instead
        # of shipping a mesh that looks normal.
        raise ValueError(
            'joint low-frequency identity solved no free vertices; '
            f'front_locked={int(protected.sum())} identity_frames={ids} '
            'reason=no vertex has two supported observations outside the excluded set')
    write_json(output/'report.json',dict(sequence_sha256=sha(sequence),initial=str(initial),elapsed_s=time.perf_counter()-start,
        shape_fusion=shape_fusion,joint_front_confidence=joint_front_confidence,
        effective_shape_frame_ids=[rows[i]['id'] for i in shape_ids],
        support_min_information=support_min_information,
        coverage_allowance_ratio=coverage_allowance_ratio,
        anchor_prior_weight=anchor_prior_weight,identity_views=identity_views,
        front_locked_vertices=int(protected.sum()) if not joint else 0,
        front_downweighted_vertices=int(protected.sum()) if joint else 0,
        maximum_absolute_displacement_mm=moved,
        free_vertices=[h.get('free_vertices') for h in steps],
        stages=[log,side_log],policy=policy))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sequence','initial-fit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fixed-ray-policy',choices=('strict-coverage','accounted-research'),default='strict-coverage')
    p.add_argument('--shape-fusion',choices=('guarded','joint'),default='guarded')
    p.add_argument('--joint-front-confidence',type=float,default=.15)
    p.add_argument('--support-min-information',type=float,default=0.)
    p.add_argument('--coverage-allowance-ratio',type=float,default=0.)
    p.add_argument('--anchor-prior-weight',type=float,default=0.)
    p.add_argument('--identity-views',choices=['all','front'],default='all')
    p.add_argument('--information-measure',choices=['barycentric','curvature'],default='barycentric')
    p.add_argument('--view-balance',action='store_true')
    p.add_argument('--reuse-anchor',action='store_true',help='skip the second low-frequency solve')
    p.add_argument('--information-damping-cap',type=float,default=10.)
    a=p.parse_args();run(a.sequence.resolve(),a.initial_fit.resolve(),a.output.resolve(),a.fixed_ray_policy,
                         a.shape_fusion,a.joint_front_confidence,a.support_min_information,
                         a.coverage_allowance_ratio,a.anchor_prior_weight,a.identity_views,
                         a.information_measure,a.information_damping_cap,a.view_balance,a.reuse_anchor)
