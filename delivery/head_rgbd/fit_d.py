"""Explicit research entry: regularized shared identity and measured dense detail."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import trimesh

from head_rgbd import __version__
from head_rgbd.audit import sha
from head_rgbd.fit_b import normals
from head_rgbd.geometry_c import front_protected_vertices, neck_surfaces
from head_rgbd.geometry_d import (Settings, anatomical_exclusion, child_landmarks, evidence,
                                 solve_field, subdivide, surface_state)
from head_rgbd.paths import configure_environment
from head_rgbd.reference_contract import inner_lip_gap_mm, observed_landmarks_m
from head_rgbd.reporting import write_json


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def reconstruct(sequence, initial_fit, output, settings, mode,dense_mode='absolute',regularized_fit=None):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads((initial_fit/'report.json').read_text(encoding='utf-8'))
    if mode=='guarded' and not source.get('front_anchor'):
        raise ValueError('guarded D requires the actual C front-anchor initialization')
    if mode=='joint-research' and source.get('front_anchor'):
        raise ValueError('joint identity requires an explicitly unanchored all-closed-view coarse fit')
    if sha(sequence) != source['sequence_sha256']:
        raise ValueError('initialization must bind the exact prepared sequence')
    info = json.loads(sequence.read_text(encoding='utf-8'))
    if (info['depth_unit'], info['mesh_unit'], info['coordinate_convention']) != ('mm', 'm', 'X_right_Y_up_Z_backward'):
        raise ValueError('unsupported units/coordinate contract')
    params = load_npz(initial_fit/'parameters.npz')
    rows = source['frames']
    if any(row['status'] != 'candidate' for row in rows):
        raise ValueError('initialization contains excluded rows; explicitly reprepare/reinitialize first')
    data = [load_npz(sequence.parent/next(r['file'] for r in info['frames'] if r['id']==row['id'])) for row in rows]
    base = params['posed_vertices'].astype(float)
    faces = params['faces']
    direction = np.array([normals(v, faces) for v in base])
    identity = [int(i) for i in params['identity_selected']]
    front = [i for i, row in enumerate(rows) if row['view']=='front']
    front_identity = [i for i in identity if rows[i]['view']=='front']
    assert all(rows[i]['state']=='closed' for i in identity)
    excluded = anatomical_exclusion(base, data, front, params['mouth_identity_excluded'])
    if dense_mode in ('hierarchical','loop-research') and settings.solver!='qp':
        raise ValueError('hierarchical residual requires the explicit constrained QP solver')
    if dense_mode=='loop-research' and not settings.research_v2:
        raise ValueError('Loop candidate requires --research-v2')
    if settings.research_v2 and regularized_fit is not None:
        raise ValueError('v2 requires fresh staged control optimization; old scalar fields cannot be reused')
    # Reuse is restricted to an exact previously computed D control surface.
    if regularized_fit is not None:
        previous=json.loads((regularized_fit/'report.json').read_text(encoding='utf-8'))
        saved=load_npz(regularized_fit/'parameters.npz')
        if previous['sequence_sha256']!=sha(sequence) or previous['mode']!=mode:
            raise ValueError('regularized reuse must bind the same sequence and candidate mode')
        np.testing.assert_array_equal(saved['posed_vertices'],base)
        np.testing.assert_array_equal(saved['faces'],faces)
        np.testing.assert_array_equal(saved['identity_selected'],identity)
        for key in ('solver','gradient_length_mm','bending_length_mm','maximum_slope','displacement_bound_mm','minimum_signed_area_ratio'):
            if previous['field_logs']['regularized']['settings'][key]!=getattr(settings,key):
                raise ValueError('regularized reuse settings mismatch: '+key)
        anchor=saved['front_anchor_deviation_m']*1000
        regularized=saved['deviation_m']*1000
        protected=saved['front_protected_vertices']
        support_regular=load_npz(regularized_fit/'support_regularized.npz')['refinement_frame_mask']
        anchor_log=previous['field_logs']['front_anchor_regularized']
        regular_log=previous['field_logs']['regularized']
    else:
        # Restart from the coarse model, never smooth the noisy v0.1 field.
        low_settings=replace(settings,gradient_length_mm=8.,bending_length_mm=8.,outer_steps=2,
                             displacement_bound_mm=2.5) if settings.research_v2 else settings
        anchor, _, anchor_log = solve_field(base, faces, direction, data, rows, front_identity,
                                           excluded, low_settings, stage='front_anchor_low_frequency' if settings.research_v2 else 'front_anchor_regularized')
        if settings.research_v2:
            low_anchor=anchor.copy()
            anchor,_,mid_log=solve_field(base,faces,direction,data,rows,front_identity,excluded,
                replace(settings,displacement_bound_mm=2.5),initial=anchor,stage='front_anchor_mid_frequency')
            anchor_log['mid_frequency']=mid_log
    regular_anchor_vertices = surface_state(base, direction, anchor)
    if regularized_fit is None and mode=='guarded':
        protected = front_protected_vertices(regular_anchor_vertices, faces, data, front)
        regularized, support_regular, regular_log = solve_field(
            base, faces, direction, data, rows, identity, excluded, replace(settings,displacement_bound_mm=2.5) if settings.research_v2 else settings,
            initial=anchor, locked=protected, stage='guarded_regularized')
    elif regularized_fit is None:
        protected = np.zeros(base.shape[1], bool)
        regularized, support_regular, regular_log = solve_field(
            base, faces, direction, data, rows, identity, excluded, replace(settings,displacement_bound_mm=2.5) if settings.research_v2 else settings,
            initial=anchor, stage='joint_closed_regularized_research')
    regular_vertices = surface_state(base, direction, regularized)
    dense_faces, mapping, midpoint_edges, parent_faces = subdivide(faces, base.shape[1])
    dense_base = np.array([mapping@v for v in base])
    dense_direction = np.array([mapping@n for n in direction])
    dense_direction /= np.maximum(np.linalg.norm(dense_direction, axis=2, keepdims=True), 1e-12)
    dense_excluded = anatomical_exclusion(dense_base, data, front, (mapping@excluded.astype(float)) > 0)
    dense_settings = replace(settings, gradient_length_mm=settings.gradient_length_mm,
                             bending_length_mm=settings.bending_length_mm)
    interpolation = mapping@regularized
    # Exact prolongation of geometry, NOT normalize(P*N)*(P*d).
    interpolation_vertices=np.array([mapping@v for v in regular_vertices])
    loop_log=None;smooth_mapping=mapping
    dense_fit_base=interpolation_vertices if dense_mode in ('hierarchical','loop-research') else dense_base
    dense_locked=np.zeros(dense_base.shape[1],bool)
    options={}
    if dense_mode in ('hierarchical','loop-research'):
        dense_locked[:base.shape[1]]=True
        options=dict(reference_base=dense_base,proxy_field=interpolation)
    if dense_mode=='loop-research':
        from head_rgbd.surface_v2 import loop_mapping
        from head_rgbd.geometry_d import edges_of
        from head_rgbd.hierarchy_d import safety as dense_safety
        _,_,S,_,_,loop_fixed=loop_mapping(faces,base.shape[1],excluded|(support_regular.sum(0)<2))
        S=S.tolil()
        for vertex in np.flatnonzero(dense_excluded):
            row=mapping.getrow(vertex);S.rows[vertex]=row.indices.tolist();S.data[vertex]=row.data.tolist()
        S=S.tocsr()
        edges=edges_of(dense_faces)
        lengths=np.linalg.norm((dense_base[0,edges[:,0]]-dense_base[0,edges[:,1]])*1000,axis=1)
        trials=[];alpha=1.
        for _ in range(20):
            proposed_mapping=mapping+alpha*(S-mapping)
            proposed_base=np.array([proposed_mapping@v for v in regular_vertices])
            ok,record=dense_safety(proposed_base,dense_base,dense_direction,dense_faces,
                                   np.zeros(len(interpolation)),edges,lengths,interpolation,settings)
            trials.append(dict(blend=alpha,safe=ok,check=record))
            if ok:
                smooth_mapping=proposed_mapping;dense_fit_base=proposed_base;break
            alpha*=.5
        else:raise ValueError('no feasible Loop base within ORIGINAL total budget')
        dense_locked[:]=False
        loop_log=dict(research_only=True,blend=alpha,nonlinear_trials=trials,
            original_controls_fixed_in_dense_layer=False,anatomical_fixed_controls=int(loop_fixed.sum()),
            maximum_move_from_linear_mm=float(np.linalg.norm(dense_fit_base-interpolation_vertices,axis=2).max()*1000),
            base_is_not_a_scan=True,limit_surface=False,
            budget_reference='exact linear prolongation of ORIGINAL FLAME coarse, not the smoothed base')
    # Fresh equations on the dense topology; the residual has its own trust bound.
    dense_anchor, _, dense_anchor_log = solve_field(
        dense_fit_base, dense_faces, dense_direction, data, rows, front_identity, dense_excluded,
        dense_settings, locked=dense_locked,stage='front_anchor_dense',**options)
    dense_anchor_vertices = surface_state(dense_fit_base, dense_direction, dense_anchor)
    if mode=='guarded':
        dense_protected = front_protected_vertices(dense_anchor_vertices, dense_faces, data, front)
        dense, support_dense, dense_log = solve_field(
            dense_fit_base, dense_faces, dense_direction, data, rows, identity, dense_excluded,
            dense_settings, initial=dense_anchor, locked=dense_protected|dense_locked, stage='guarded_dense',**options)
    else:
        dense_protected = np.zeros(dense_base.shape[1], bool)
        dense, support_dense, dense_log = solve_field(
            dense_fit_base, dense_faces, dense_direction, data, rows, identity, dense_excluded,
            dense_settings, initial=dense_anchor,locked=dense_locked, stage='joint_closed_dense_research',**options)
    dense_vertices = surface_state(dense_fit_base, dense_direction, dense)
    if settings.research_v2:
        from head_rgbd.surface_v2 import reliable_evidence
        evidence_regular,evidence_log_regular=reliable_evidence(regular_vertices,faces,data,rows)
        evidence_dense,evidence_log_dense=reliable_evidence(dense_vertices,dense_faces,data,rows)
    else:
        evidence_regular, evidence_log_regular = evidence(regular_vertices, faces, data)
        evidence_dense, evidence_log_dense = evidence(dense_vertices, dense_faces, data)
    np.savez_compressed(output/'support_regularized.npz', **evidence_regular,
                        refinement_frame_mask=support_regular, refinement_frame_indices=identity,
                        excluded_identity_vertices=excluded, deviation_m=regularized*.001)
    np.savez_compressed(output/'support_dense.npz', **evidence_dense,
                        refinement_frame_mask=support_dense, refinement_frame_indices=identity,
                        excluded_identity_vertices=dense_excluded, deviation_m=dense*.001,
                        total_displacement_vectors_m=dense_vertices-dense_base)
    np.savez_compressed(output/'mapping.npz', prolongation_data=mapping.data,
                        prolongation_indices=mapping.indices, prolongation_indptr=mapping.indptr,
                        prolongation_shape=mapping.shape, midpoint_edges=midpoint_edges,
                        parent_face_indices=parent_faces,
                        smooth_data=smooth_mapping.data,smooth_indices=smooth_mapping.indices,
                        smooth_indptr=smooth_mapping.indptr,smooth_shape=smooth_mapping.shape)
    stages = dict(coarse=(base, faces), regularized=(regular_vertices, faces),
                  dense=(dense_vertices, dense_faces), dense_interpolation=(interpolation_vertices, dense_faces))
    if loop_log is not None:
        stages['dense_smooth_base']=(dense_fit_base,dense_faces)
        stages['low_frequency_anchor']=(surface_state(base,direction,low_anchor),faces)
    references = {}
    for state in ('closed', 'open'):
        frame_id = source['references'][state]['frame_id']
        i = next(i for i, row in enumerate(rows) if row['id']==frame_id)
        dense_fi, dense_bc = child_landmarks(params['landmark_faces'][i], params['landmark_barycentric'][i])
        reference = dict(frame_id=frame_id, source=rows[i]['source'], source_frame=rows[i]['source_frame'],
                         timestamp_s=rows[i]['timestamp_s'], coordinate_system='reference camera X_right_Y_up_Z_backward',
                         mesh_unit='m', expression=params['expression'][i], jaw_axis_angle_rad=params['jaw'][i],
                         observed_landmarks_m=observed_landmarks_m(data[i]['xyz'], data[i]['valid']),
                         observed_valid=data[i]['valid'], observed_inner_lip_gap_mm=inner_lip_gap_mm(data[i]['xyz'], data[i]['valid']),
                         observation_is_manual_truth=False, stages={})
        model_to_camera=np.eye(4);model_to_camera[:3,:3]=params['R'][i];model_to_camera[:3,3]=params['t'][i]
        reference['animated_model_to_reference_camera_4x4']=model_to_camera
        reference['reference_camera_to_animated_model_4x4']=np.linalg.inv(model_to_camera)
        reference['transform_translation_unit']='m'
        reference['transform_scope']='rigid coordinate transform only; expression/jaw remains; NOT a scan registration'
        for stage, (vertices, topology) in stages.items():
            fi, bc = (dense_fi, dense_bc) if stage.startswith('dense') else (params['landmark_faces'][i], params['landmark_barycentric'][i])
            landmarks = (vertices[i][topology[fi]]*bc[:, :, None]).sum(1)
            name = f'{state}_{stage}_m.obj'
            mesh = trimesh.Trimesh(vertices[i], topology, process=False)
            mesh.export(output/name, include_normals=True)
            # Verify actual on-disk vertex/index order and landmark interpolation.
            exported = trimesh.load(output/name, process=False, maintain_order=True)
            assert len(exported.vertices)==len(vertices[i])
            measured = (np.asarray(exported.vertices)[np.asarray(exported.faces)[fi]]*bc[:, :, None]).sum(1)
            np.testing.assert_allclose(measured, landmarks, atol=2e-7)
            reference['stages'][stage] = dict(mesh=name, vertices=len(mesh.vertices), faces=len(mesh.faces),
                landmark_triangle_indices=fi, landmark_barycentric=bc, landmarks_m=landmarks,
                inner_lip_gap_mm=float(np.linalg.norm(landmarks[62]-landmarks[66])*1000))
            if stage in ('regularized', 'dense'):
                ev = evidence_regular if stage=='regularized' else evidence_dense
                palette = np.array([[120, 120, 125, 255], [230, 175, 55, 255], [55, 175, 215, 255]], np.uint8)
                trimesh.Trimesh(vertices[i], topology, vertex_colors=palette[ev['region_label']], process=False).export(output/f'{state}_{stage}_support_m.ply')
        references[state] = reference
    write_json(output/'reference_geometry.json', dict(schema='head_rgbd.d.reference.v1', references=references,
        landmark_indices='FLAME dynamic contour + static 68; zero based; per-stage triangle mapping'))
    neck = neck_surfaces(rows, data, output)
    np.savez_compressed(output/'parameters.npz', shape=params['shape'], expression=params['expression'], jaw=params['jaw'],
        R=params['R'], t=params['t'], posed_vertices=base, faces=faces, regularized_vertices=regular_vertices,
        dense_base_vertices=dense_base, dense_vertices=dense_vertices, dense_faces=dense_faces,
        dense_fit_base_vertices=dense_fit_base,dense_total_displacement_m=dense_vertices-dense_base,
        low_frequency_anchor_vertices=surface_state(base,direction,low_anchor) if settings.research_v2 and regularized_fit is None else regular_anchor_vertices,
        dense_solver_scalar_m=dense*.001,
        dense_interpolation_vertices=interpolation_vertices, deviation_m=regularized*.001, dense_deviation_m=dense*.001,
        front_anchor_deviation_m=anchor*.001, dense_front_anchor_deviation_m=dense_anchor*.001,
        front_anchor_vertices=regular_anchor_vertices, dense_front_anchor_vertices=dense_anchor_vertices,
        front_protected_vertices=protected, dense_front_protected_vertices=dense_protected,
        mouth_identity_excluded=params['mouth_identity_excluded'], anatomical_excluded=excluded,
        dense_anatomical_excluded=dense_excluded, identity_selected=identity)
    logs = dict(front_anchor_regularized=anchor_log, regularized=regular_log,
                front_anchor_dense=dense_anchor_log, dense=dense_log)
    write_json(output/'report.json', dict(schema='head_rgbd.d.fit.v1', project_version=__version__,
        algorithm='v2 research: alternating fixed-identity motion initialization, FEM total deformation and budgeted measured Loop surface' if settings.research_v2 else 'D research: measured shared normal field, area quadrature and physical regularization',
        mode=mode,dense_mode=dense_mode,research_v2=settings.research_v2,loop_base=loop_log,research_only=True, full_facecap_reproduction=False,
        dense_scalar_representation='shared measured residual over budgeted Loop regularized control surface' if dense_mode=='loop-research' else ('residual over exact interpolated regularized control surface' if dense_mode=='hierarchical' else 'absolute normal scalar over FLAME base'),
        regularized_reuse=dict(path=str(regularized_fit),sha256=sha(regularized_fit/'parameters.npz')) if regularized_fit else None,
        sequence_sha256=sha(sequence), initialization=str(initial_fit), initialization_sha256=sha(initial_fit/'parameters.npz'),
        old_refinement_ignored=True, frames=rows, prepared_excluded_frames=source.get('prepared_excluded_frames', []),
        identity_frames=[rows[i]['id'] for i in identity], motion_only_frames=[r['id'] for i,r in enumerate(rows) if i not in identity],
        temporal_edges=source['temporal_edges'], inherited_motion_expression_policy='fresh C fit; fixed during D; never average raw expression depths',
        regularized_topology=dict(vertices=base.shape[1], faces=len(faces)),
        dense_topology=dict(vertices=dense_base.shape[1], faces=len(dense_faces)),
        dense_change_from_interpolated_regularized_mm=float(np.linalg.norm(dense_vertices-interpolation_vertices, axis=2).max()*1000),
        dense_reliable_observation_vertices=int((evidence_dense['region_label']==2).sum()),
        maximum_dense_total_displacement_mm=float(np.linalg.norm(dense_vertices-dense_base,axis=2).max()*1000),
        dense_label_counts=np.bincount(evidence_dense['region_label'], minlength=3),
        references={s:dict(frame_id=r['frame_id']) for s,r in references.items()},
        field_logs=logs, evidence_logs=dict(regularized=evidence_log_regular, dense=evidence_log_dense), neck=neck,
        compute_device='cpu', cuda_peak_allocated_mib=0., cuda_metric_scope='D uses CPU; initialization GPU statistics are in its own report',
        elapsed_s=time.perf_counter()-started, timing_scope='D CPU correspondences, two topologies, evidence, independent neck and exports',
        source_hashes={name:sha(Path(__file__).parent/name) for name in ('geometry_d.py','fit_d.py','qp_d.py','normal_cone_d.py','hierarchy_d.py')},
        statuses=dict(implementation='implemented', pipeline='passed', geometry_quality='not_evaluated', independent_accuracy='unverified'),
        limitations=['research candidate, not a replacement of default C guarded',
            'FLAME per-frame expression approximation is inherited; residual expression/pose bias may remain',
            'closed keyframes only; repeated short-burst frames are correlated',
            '3mm total normal bound remains; larger or tangential identity errors can remain',
            'dense midpoint base is a parameterization only; detail must be supported by fresh depth queries',
            'occluded back/ears remain model prior; no independent scan, contour or calibrated external pose']))
    print(json.dumps(dict(output=str(output), mode=mode, elapsed_s=time.perf_counter()-started,
                          coarse_vertices=base.shape[1], dense_vertices=dense_base.shape[1])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--initial-fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--mode', choices=('guarded', 'joint-research'), default='guarded')
    parser.add_argument('--research-only', action='store_true', required=True)
    parser.add_argument('--points', type=int, default=5000)
    parser.add_argument('--solver', choices=('least-squares','qp'), default='least-squares')
    parser.add_argument('--dense-mode',choices=('absolute','hierarchical','loop-research'),default='absolute')
    parser.add_argument('--research-v2',action='store_true',help='FEM total deformation, observed depth planes, canonical noise, staged identity')
    parser.add_argument('--fixed-ray-policy',choices=('strict-coverage','accounted-research'),default='strict-coverage')
    parser.add_argument('--regularized-fit',type=Path,help='exact completed D control field; verified before reuse')
    parser.add_argument('--dense-residual-bound-mm',type=float,default=.5)
    parser.add_argument('--gradient-length-mm', type=float, default=3.)
    parser.add_argument('--bending-length-mm', type=float, default=4.)
    args = parser.parse_args()
    if args.points < 500 or min(args.gradient_length_mm, args.bending_length_mm,args.dense_residual_bound_mm) <= 0 or args.dense_residual_bound_mm> .5:
        parser.error('invalid sampling/physical regularization')
    configure_environment(args.runtime)
    reconstruct(args.sequence.resolve(), args.initial_fit.resolve(), args.output.resolve(),
                Settings(solver=args.solver, points=args.points, research_v2=args.research_v2,fixed_ray_policy=args.fixed_ray_policy,noise_floor_mm=1/np.sqrt(12) if args.research_v2 else .7,gradient_length_mm=args.gradient_length_mm,
                         bending_length_mm=args.bending_length_mm,dense_residual_bound_mm=args.dense_residual_bound_mm), args.mode,
                args.dense_mode,args.regularized_fit.resolve() if args.regularized_fit else None)


if __name__=='__main__':
    main()
