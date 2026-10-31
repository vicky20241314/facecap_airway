"""Same-real-sequence comparisons; acceptance is separate from execution success."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import trimesh

from head_rgbd.audit import sha
from head_rgbd.contracts import temporal_edge_allowed
from head_rgbd.evaluate import raster_depth
from head_rgbd.fit_b import normals
from head_rgbd.fit_d import load_npz
from head_rgbd.geometry_d import subdivide
from head_rgbd.quality_d import physical_roughness, summary, triangle_risks
from head_rgbd.reporting import write_json
from head_rgbd.visualize_d import comparison_views


def regions(data):
    h,w=data['depth'].shape; yy,xx=np.indices((h,w));uv=data['uv']
    mouth=np.zeros((h,w),np.uint8)
    if np.isfinite(uv[48:]).all():
        cv2.fillConvexPoly(mouth,cv2.convexHull(np.rint(uv[48:]).astype(np.int32)),1)
        mouth=cv2.dilate(mouth,np.ones((17,17),np.uint8))
    return dict(full=np.ones((h,w),bool),mouth=mouth.astype(bool),
                lower_face_jaw_proxy=yy>uv[33,1],
                left_cheek=(xx<uv[30,0])&(yy>uv[27,1])&(yy<uv[57,1])&~mouth.astype(bool),
                right_cheek=(xx>=uv[30,0])&(yy>uv[27,1])&(yy<uv[57,1])&~mouth.astype(bool))


def evaluate(sequence, baseline, guarded, joint, output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    sequence_info=json.loads(sequence.read_text(encoding='utf-8'))
    baseline_fit=baseline/'reconstruction'/'guarded'
    original=load_npz(baseline_fit/'parameters.npz')
    old_report=json.loads((baseline_fit/'report.json').read_text(encoding='utf-8'))
    old_ids={r['id']:i for i,r in enumerate(old_report['frames'])}
    candidates={name:(root,load_npz(root/'parameters.npz'),json.loads((root/'report.json').read_text(encoding='utf-8')))
                for name,root in (('D_guarded',guarded),('D_joint',joint)) if root is not None}
    reference_report=candidates['D_guarded'][2]
    rows=reference_report['frames']
    methods=dict(coarse=(original['posed_vertices'],original['faces'],old_ids,False),
                 v010=(original['refined_vertices'],original['faces'],old_ids,False))
    for name,(root,p,report) in candidates.items():
        index={r['id']:i for i,r in enumerate(report['frames'])}
        methods[name+'_regularized']=(p['regularized_vertices'],p['faces'],index,False)
        methods[name+'_dense']=(p['dense_vertices'],p['dense_faces'],index,True)
        if name=='D_guarded':methods[name+'_interpolation']=(p['dense_interpolation_vertices'],p['dense_faces'],index,True)
        if name=='D_joint':methods['joint_coarse']=(p['posed_vertices'],p['faces'],index,False)
    results=[];guard_checks=[]
    for row in rows:
        source=next(r for r in sequence_info['frames'] if r['id']==row['id'])
        data=load_npz(sequence.parent/source['file'])
        observed=data['depth'][::4,::4]*.001
        valid=data['face_mask'][::4,::4]&(observed>.1)&(observed<1.5)
        rendered={name:raster_depth(vertices[index[row['id']]],faces,data['intrinsics'],data['depth'].shape,4)
                  for name,(vertices,faces,index,_) in methods.items()}
        common=valid.copy()
        for image in rendered.values():common&=np.isfinite(image)
        record=dict(frame_id=row['id'],view=row['view'],state=row['state'],fitted_yaw_deg=row['fitted_yaw_deg'],
                    identity_keyframe=bool(row.get('identity_keyframe')),
                    diagnostic_role='identity_training' if row.get('identity_keyframe') else 'motion_expression_only_not_independent_validation',
                    valid_pixels=int(valid.sum()),common_pixels=int(common.sum()),methods={})
        for name,image in rendered.items():
            covered=valid&np.isfinite(image)
            region_result={}
            for region,mask in regions(data).items():
                pixels=common&mask[::4,::4]
                error=np.abs(image[pixels]-observed[pixels])*1000
                region_result[region]=dict(**summary(error),above_5mm_fraction=float(np.mean(error>5)) if len(error) else None)
            shared=valid&np.isfinite(rendered['v010'])&np.isfinite(image)
            pair_error=np.abs(image[shared]-observed[shared])*1000
            old_error=np.abs(rendered['v010'][shared]-observed[shared])*1000
            record['methods'][name]=dict(coverage=float(covered.sum()/max(valid.sum(),1)),regions=region_result,
                pairwise_v010_common_pixels=int(shared.sum()),pairwise_v010_residual_mm=summary(pair_error),
                v010_on_pairwise_pixels_mm=summary(old_error))
        root,p,r=candidates['D_guarded']
        if row['view']=='front':
            i=next(i for i,r in enumerate(r['frames']) if r['id']==row['id'])
            for stage,key,faces in (('regularized','front_anchor_vertices',p['faces']),('dense','dense_front_anchor_vertices',p['dense_faces'])):
                anchor=raster_depth(p[key][i],faces,data['intrinsics'],data['depth'].shape,4)
                actual=rendered['D_guarded_'+stage]
                old_valid=valid&np.isfinite(anchor);new_valid=valid&np.isfinite(actual)
                same=bool(np.array_equal(old_valid,new_valid))
                delta=float(np.max(np.abs(actual[old_valid]-anchor[old_valid]))*1000) if same and old_valid.any() else None
                guard_checks.append(dict(frame_id=row['id'],stage=stage,coverage_identical=same,maximum_depth_change_mm=delta,
                                         passed=same and delta is not None and delta<.0001))
        results.append(record)
        print(f"evaluated {row['id']} on {len(methods)} actual geometries",flush=True)
    by_id={r['id']:r for r in rows}
    assert all(temporal_edge_allowed(by_id[a],by_id[b]) for a,b in reference_report['temporal_edges'])
    support_old=load_npz(baseline_fit/'support.npz')
    supported_faces=(support_old['region_label'][original['faces']]==2).all(1)
    closed_id=old_report['references']['closed']['frame_id']
    base_closed=original['posed_vertices'][old_ids[closed_id]]
    geometry={};visual=[];risk_reference={}
    original_dense_faces,original_mapping,_,_=subdivide(original['faces'],original['posed_vertices'].shape[1])
    for topology in ('coarse','dense'):
        risk_reference[topology]={}
        for state in ('closed','open'):
            fid=old_report['references'][state]['frame_id']
            v=original['posed_vertices'][old_ids[fid]].astype(float)
            f=original['faces']
            if topology=='dense':v=original_mapping@v;f=original_dense_faces
            risk_reference[topology][state]=triangle_risks(v,f)
    old_i=old_ids[closed_id]
    for name,(vertices,faces,index,dense) in methods.items():
        i=index[closed_id]
        if name.startswith('D_'):
            family='D_guarded' if name.startswith('D_guarded') else 'D_joint'
            pp=candidates[family][1]
        elif name=='joint_coarse':pp=candidates['D_joint'][1]
        else:pp=original
        aligned=(vertices[i]-pp['t'][i])@pp['R'][i]@original['R'][old_i].T+original['t'][old_i]
        roughness=physical_roughness(aligned,faces,base_closed,original['faces'],supported_faces,dense)
        risks={}
        for state in ('closed','open'):
            fid=old_report['references'][state]['frame_id'];j=index[fid]
            risks[state]=triangle_risks(vertices[j],faces)
        geometry[name]=dict(physical_smoothness=roughness,risks=risks)
        if name in ('coarse','v010','D_guarded_regularized','D_guarded_dense','D_joint_dense'):
            visual.append((name,aligned,faces))
        print(f'physical quality and intersection screen: {name}',flush=True)
    reference_checks={}
    for name,(root,p,report) in candidates.items():
        refs=json.loads((root/'reference_geometry.json').read_text(encoding='utf-8'))
        assert np.isfinite(p['regularized_vertices']).all() and np.isfinite(p['dense_vertices']).all()
        assert p['dense_vertices'].shape[1]>p['posed_vertices'].shape[1] and len(p['dense_faces'])>len(p['faces'])
        for state,reference in refs['references'].items():
            assert reference['frame_id']==old_report['references'][state]['frame_id']
            for stage,detail in reference['stages'].items():
                mesh=trimesh.load(root/detail['mesh'],process=False,maintain_order=True)
                assert np.isfinite(mesh.vertices).all() and .1<max(mesh.extents)<.5
                assert len(mesh.vertices)==detail['vertices'] and len(mesh.faces)==detail['faces']
                lm=(np.asarray(mesh.vertices)[np.asarray(mesh.faces)[detail['landmark_triangle_indices']]]*np.asarray(detail['landmark_barycentric'])[:,:,None]).sum(1)
                np.testing.assert_allclose(lm,detail['landmarks_m'],atol=2e-7)
            for valid,point in zip(reference['observed_valid'],reference['observed_landmarks_m']):
                if not valid:assert point is None
        assert np.abs(p['deviation_m']).max()<=.003000001
        assert np.abs(p['dense_deviation_m']).max()<=.003000001
        if 'dense_total_displacement_m' in p:
            assert np.linalg.norm(p['dense_total_displacement_m'],axis=2).max()<=.003000001
        if report.get('dense_mode')=='hierarchical':
            np.testing.assert_allclose(p['dense_vertices'][:,:p['posed_vertices'].shape[1]],p['regularized_vertices'],atol=1e-12,rtol=0)
            assert np.abs(p['dense_deviation_m']).max()<=.000500001
        assert np.all(p['deviation_m'][p['anatomical_excluded']]==0)
        assert np.all(p['dense_deviation_m'][p['dense_anatomical_excluded']]==0)
        evidence=load_npz(root/'support_dense.npz')
        np.testing.assert_array_equal(evidence['observed_frames'],evidence['observed_frame_mask'].sum(0))
        assert evidence['observed_frame_mask'].shape==(len(rows),p['dense_vertices'].shape[1])
        new_vertex=np.arange(p['dense_vertices'].shape[1])>=p['posed_vertices'].shape[1]
        closed_support=evidence['observed_frame_mask'][p['identity_selected']].sum(0)>=2
        ci=next(i for i,r in enumerate(report['frames']) if r['id']==closed_id)
        measured_residual=np.linalg.norm(p['dense_vertices'][ci]-p['dense_interpolation_vertices'][ci],axis=1)
        depth_detail=new_vertex&closed_support&(measured_residual>1e-5)
        reference_checks[name]=dict(status='passed',reference_frames={s:r['frame_id'] for s,r in refs['references'].items()},
            support_counts=np.bincount(evidence['region_label'],minlength=3),dense_support_requeried=True,
            dense_difference_from_interpolation_mm=report['dense_change_from_interpolated_regularized_mm'],
            new_vertices_with_closed_depth_support_and_nonzero_detail=int(depth_detail.sum()),
            new_supported_detail_mm=summary(measured_residual[depth_detail]*1000),
            detail_definition='new vertices: repeated closed depth support AND actual difference from exact interpolated control surface >0.01mm',
            maximum_total_dense_displacement_mm=float(np.linalg.norm(p['dense_vertices']-p['dense_base_vertices'],axis=2).max()*1000))
    acceptance={}
    for name in methods:
        if not name.startswith('D_') or name.endswith('_interpolation'):continue
        front_regressions=[];side_regressions=[];coverage_regressions=[]
        for row in results:
            values=row['methods'][name]
            if values['coverage'] < row['methods']['v010']['coverage']-1e-12:
                coverage_regressions.append(dict(frame_id=row['frame_id'],change=values['coverage']-row['methods']['v010']['coverage']))
            new=values['pairwise_v010_residual_mm'];old=values['v010_on_pairwise_pixels_mm']
            if row['view']=='front' and new['p95']>old['p95']:
                front_regressions.append(dict(frame_id=row['frame_id'],p95_change_mm=new['p95']-old['p95']))
            if row['view']!='front':
                baseline_coarse=row['methods']['coarse']['regions']['full']
                current=values['regions']['full']
                if current['p95']>baseline_coarse['p95'] or current['median']>baseline_coarse['median']:
                    side_regressions.append(row['frame_id'])
        current=geometry[name]['physical_smoothness'];old=geometry['v010']['physical_smoothness']
        smooth=all(current[key]['p95']<old[key]['p95'] for key in ('normal_spread_rms_deg','local_best_plane_rms_mm'))
        topology='dense' if name.endswith('dense') else 'coarse'
        intersections=all(geometry[name]['risks'][state]['proper_intersection_pairs']<=risk_reference[topology][state]['proper_intersection_pairs']
                          and geometry[name]['risks'][state]['coplanar_overlap_pairs']<=risk_reference[topology][state]['coplanar_overlap_pairs'] for state in ('closed','open'))
        stage='dense' if name.endswith('dense') else 'regularized'
        guard_passed=all(r['passed'] for r in guard_checks if r['stage']==stage) if name.startswith('D_guarded') else None
        family='D_guarded' if name.startswith('D_guarded') else 'D_joint'
        measured_detail=reference_checks[family]['new_vertices_with_closed_depth_support_and_nonzero_detail']>0 if stage=='dense' else True
        acceptance[name]=dict(physical_smoothness_improved=bool(smooth),front_p95_regressions_vs_v010=front_regressions,
            side_p95_or_median_regressions_vs_coarse=side_regressions,no_increase_in_screened_intersections=intersections,
            coverage_regressions_vs_v010=coverage_regressions,guarded_front_preserved=guard_passed,
            actual_depth_driven_detail=measured_detail,
            overall='passed' if smooth and not front_regressions and not side_regressions and not coverage_regressions
                and intersections and measured_detail and guard_passed is not False else 'not_passed')
    comparison_views(visual,output,(base_closed.min(0)+base_closed.max(0))/2)
    write_json(output/'comparison.json',dict(schema='head_rgbd.d.comparison.v1',sequence_sha256=sha(sequence),
        metric_scope='same-real-training-data camera-axis diagnostics, not independent accuracy',
        frames=results,geometry=geometry,matched_topology_intersection_reference=risk_reference,
        guard_checks=guard_checks,references=reference_checks,
        no_cross_gap_constraints=True,acceptance=acceptance,independent_accuracy='unverified',
        elapsed_s=time.perf_counter()-started,
        limitations=['local masks derive from automatic 2D landmarks, not manual jaw contours',
                     'self-intersection screen excludes incident triangles and tangency certification',
                     'matched-coverage residuals use fitted poses/expressions; no independent reference scan']))
    print(json.dumps(dict(comparison=str(output/'comparison.json'),acceptance=acceptance)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sequence','baseline','guarded','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--joint',type=Path)
    a=p.parse_args();evaluate(a.sequence,a.baseline,a.guarded,a.joint,a.output)


if __name__=='__main__':main()
