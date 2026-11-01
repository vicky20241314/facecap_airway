"""Fixed-identity pose then expression correction, with fixed-ray acceptance.

No scale, shape, camera calibration or per-frame identity variables are optimized.
Only a new output directory is written. This is a research motion ablation.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rodrigues
from head_rgbd.fit_d import load_npz
from head_rgbd.fit_c import landmark_embedding
from head_rgbd.geometry_c import sample_depth,ray_surface,mouth_pixels
from head_rgbd.geometry_d import eye_mask
from head_rgbd.paths import configure_environment
from head_rgbd.reporting import write_json
from head_rgbd.audit import sha
from head_rgbd.contracts import temporal_edge_allowed
from head_rgbd.reference_contract import observed_landmarks_m,inner_lip_gap_mm
from core.integrated_fitting import make_flame
from core.fit_options import FitOptions


def fixed_ray_score(v,faces,points,required=None):
    corr,_=ray_surface(v,faces,points,.030,None)
    residual=np.full(len(points),8.)
    covered=np.zeros(len(points),bool)
    if corr is not None:
        covered[corr['ray']]=True
        residual[corr['ray']]=np.abs(corr['hit'][:,2]-corr['target'][:,2])*1000
    mask=np.ones(len(points),bool) if required is None else required
    return (float(np.mean(residual[mask]**2)) if np.any(mask) else float('inf')),covered


def correct(sequence,initial,output,runtime,device='cuda',steps=80,identity_field=None):
    configure_environment(runtime);start=time.perf_counter();torch.set_num_threads(4)
    output.mkdir(parents=True,exist_ok=False)
    report=json.loads((initial/'report.json').read_text(encoding='utf-8'))
    if report['sequence_sha256']!=sha(sequence):raise ValueError('sequence mismatch')
    saved=load_npz(initial/'parameters.npz');rows=report['frames'];n=len(rows)
    if report.get('front_anchor'):
        report['front_anchor'].setdefault('original_guard_policy',report['front_anchor'].get('policy'))
        report['front_anchor']['policy']='original C anchor is initialization lineage; this explicit research mode corrects motion before building a NEW guarded identity anchor'
        report['front_anchor']['motion_refit_before_new_anchor']=True
    for row in rows:
        if row.get('acquisition',{}).get('calibration_status')=='session_alignment_unverified':
            row['alignment_evidence']='session alignment and distortion unverified; no calibration metadata invented'
            row['degraded']=list(dict.fromkeys(row.get('degraded',[])+['session_alignment_unverified']))
    info=json.loads(sequence.read_text(encoding='utf-8'))
    data=[load_npz(sequence.parent/next(r['file'] for r in info['frames'] if r['id']==row['id'])) for row in rows]
    T=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32)
    faces=saved['faces'];ft=torch.as_tensor(faces,device=device)
    flame=make_flame(FitOptions(shape_components=saved['shape'].shape[1],expression_components=saved['expression'].shape[1]),device,batch=n)
    if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    shape=T(saved['shape']);expr0=T(saved['expression']);jaw0=T(saved['jaw']);R0=T(saved['R']);t0=T(saved['t'])
    dr=torch.nn.Parameter(T(np.zeros((n,3))));dt=torch.nn.Parameter(T(np.zeros((n,3))))
    de=torch.nn.Parameter(torch.zeros_like(expr0));dj=torch.nn.Parameter(torch.zeros_like(jaw0))
    field=None
    fixed_reference=None
    if identity_field:
        f=load_npz(identity_field)
        np.testing.assert_array_equal(f['posed_vertices'],saved['posed_vertices'])
        np.testing.assert_array_equal(f['faces'],saved['faces'])
        field=T(f['deviation_m'])
        fixed_reference=next(i for i,row in enumerate(rows) if row['id']==report['references']['closed']['frame_id'])
    def forward():
        R=batch_rodrigues(dr)@R0
        raw,lm=flame(shape_params=shape.expand(n,-1),expression_params=expr0+de,
            pose_params=torch.cat((torch.zeros_like(dj),jaw0+dj),1),transl=torch.zeros_like(dt),view_rotation=R)
        v=torch.einsum('bij,bvj->bvi',R,raw)+t0[:,None]+dt[:,None]
        l=torch.einsum('bij,bkj->bki',R,lm)+t0[:,None]+dt[:,None]
        fit_v=v
        if field is not None:
            tri=v[:,ft];normal=torch.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=2)
            vn=torch.zeros_like(v)
            for j in range(3):vn.index_add_(1,ft[:,j],normal)
            vn=torch.nn.functional.normalize(vn,dim=2)
            fit_v=v+vn*field[None,:,None]
        return raw,v,l,R,fit_v
    rng=np.random.default_rng(7652);points={}
    for i,d in enumerate(data):
        p,uv,_=sample_depth(d,1800,rng,True,stride=3)
        stable=~eye_mask(d)[uv[:,1],uv[:,0]] & (uv[:,1]<d['uv'][33,1])
        points[i]=p[stable]
        if len(points[i])<80:
            rows[i]['degraded']=rows[i].get('degraded',[])+['insufficient_stable_skin_motion_correction_skipped']
    with torch.no_grad():old_forward=forward();old_fit=old_forward[-1].cpu().numpy()
    old_scores=[];required=[]
    for i in range(n):
        _,c=fixed_ray_score(old_fit[i],faces,points[i])
        s,_=fixed_ray_score(old_fit[i],faces,points[i],c);old_scores.append(s);required.append(c)
    opt=torch.optim.Adam([{'params':[dr],'lr':.0006},{'params':[dt],'lr':.00008},
                          {'params':[de],'lr':.006},{'params':[dj],'lr':.0006}])
    temporal=[(i-1,i) for i in range(1,n) if temporal_edge_allowed(rows[i-1],rows[i])]
    history=[];correspond={}
    for step in range(steps):
        expression_phase=steps//2<=step<3*steps//4
        opt.zero_grad();raw,v,l,R,fit_v=forward();loss=v.sum()*0
        if step%10==0 or step in (steps//2,3*steps//4):
            correspond={}
            for i in range(n):
                if len(points[i])<80:continue
                c,stats=ray_surface(fit_v[i].detach().cpu().numpy(),faces,points[i],.008,.005)
                if c is not None:
                    correspond[i]=(ft[torch.as_tensor(c['face'],device=device)],T(c['bary']),T(c['normal']),T(c['target']))
        for i,(tri,bc,normal,target) in correspond.items():
            surf=(fit_v[i,tri]*bc[:,:,None]).sum(1)
            residual=((surf-target)*normal).sum(1)/.002
            loss+=torch.nn.functional.smooth_l1_loss(residual,torch.zeros_like(residual),beta=1.)/n
            valid=data[i]['valid'].copy();valid[:17]=False;valid[48:]=False
            if valid.any():
                difference=(l[i][valid]-T(data[i]['xyz'][valid]))/.003
                loss+=.12*torch.nn.functional.smooth_l1_loss(difference,torch.zeros_like(difference))/n
            if expression_phase:
                uv=T(data[i]['uv']);finite=torch.isfinite(uv).all(1);finite[:48]=False
                fx,fy,cx,cy=data[i]['intrinsics'];z=(-l[i,:,2]).clamp(min=.05)
                projected=torch.stack((fx*l[i,:,0]/z+cx,-fy*l[i,:,1]/z+cy),1)
                if finite.any():loss+=.08*torch.nn.functional.smooth_l1_loss(projected[finite]/3,uv[finite]/3)/n
                if rows[i]['state']=='closed':
                    gaps=torch.linalg.norm(l[i,[61,62,63]]-l[i,[67,66,65]],dim=1)
                    loss+=.1*(torch.relu(gaps-.002)/.003).square().mean()/n
        loss+=.01*(dr/.02).square().mean()+.01*(dt/.003).square().mean()
        loss+=.02*(de/.2).square().mean()+.02*(dj/.03).square().mean()
        for i,j in temporal:
            loss+=.002*((de[j]-de[i]).square().mean()+(dj[j]-dj[i]).square().mean())/max(len(temporal),1)
        loss.backward()
        if expression_phase:dr.grad=None;dt.grad=None
        else:de.grad=None;dj.grad=None
        opt.step()
        with torch.no_grad():
            dr.mul_(torch.clamp(.02/torch.linalg.norm(dr,dim=1,keepdim=True).clamp(min=1e-12),max=1.))
            dt.mul_(torch.clamp(.003/torch.linalg.norm(dt,dim=1,keepdim=True).clamp(min=1e-12),max=1.))
            de.clamp_(-.2,.2);dj.clamp_(-.03,.03)
            if fixed_reference is not None:
                for variable in (dr,dt,de,dj):variable[fixed_reference].zero_()
        if step%20==0:
            history.append(dict(step=step,phase='expression_jaw' if expression_phase else 'rigid',objective=float(loss.detach())))
            print(history[-1],flush=True)
    # Each frame is accepted against its ORIGINAL fixed rays; disappearing rays
    # retain a penalty. Never count only the surviving easy ICP correspondences.
    correction=[]
    proposed=[x.detach().clone() for x in (dr,dt,de,dj)]
    for i in range(n):
        accepted=False
        if required[i].sum()<30:
            with torch.no_grad():
                for variable in (dr,dt,de,dj):variable[i].zero_()
            rows[i]['degraded']=rows[i].get('degraded',[])+['insufficient_fixed_motion_rays_original_motion_retained']
            correction.append(dict(frame_id=rows[i]['id'],scale=0.,accepted=False,reason='insufficient_fixed_rays',
                original_fixed_ray_mse_mm2=None,final_fixed_ray_mse_mm2=None,fixed_ray_count=int(required[i].sum())))
            continue
        for alpha in (1.,.5,.25,.125,0.):
            with torch.no_grad():
                for variable,p in zip((dr,dt,de,dj),proposed):variable[i].copy_(p[i]*alpha)
                candidate=forward()[-1][i].cpu().numpy()
            score,covered=fixed_ray_score(candidate,faces,points[i],required[i])
            if score<=old_scores[i]+1e-8 and np.all(covered[required[i]]):accepted=True;break
        correction.append(dict(frame_id=rows[i]['id'],scale=alpha,accepted=accepted,
            original_fixed_ray_mse_mm2=old_scores[i],final_fixed_ray_mse_mm2=score,
            fixed_ray_count=int(required[i].sum()),gauge_reference_frozen=i==fixed_reference,
            rotation_delta_deg=float(torch.linalg.norm(dr[i]).detach().cpu())*180/np.pi,
            translation_delta_mm=float(torch.linalg.norm(dt[i]).detach().cpu())*1000))
    with torch.no_grad():
        raw,v,l,R,fit_v=forward();fi,bc=landmark_embedding(flame,raw,jaw0+dj,R)
        out={**saved,'shape':saved['shape'],'raw_vertices':raw.cpu().numpy(),'posed_vertices':v.cpu().numpy(),
            'R':R.cpu().numpy(),'t':(t0+dt).cpu().numpy(),'expression':(expr0+de).cpu().numpy(),
            'jaw':(jaw0+dj).cpu().numpy(),'landmarks':l.cpu().numpy(),'landmark_faces':fi.cpu().numpy(),
            'landmark_barycentric':bc.cpu().numpy()}
        out['refined_vertices']=fit_v.cpu().numpy()
        out['before_side_detail_vertices']=fit_v.cpu().numpy()
        out['deviation_m']=field.cpu().numpy() if field is not None else np.zeros(v.shape[1])
        out['effective_deviation_m']=np.tile(out['deviation_m'],(n,1))
    np.savez_compressed(output/'parameters.npz',**out)
    np.savez_compressed(output/'motion_delta.npz',rotation=dr.detach().cpu().numpy(),translation_m=dt.detach().cpu().numpy(),
                        expression=de.detach().cpu().numpy(),jaw_rad=dj.detach().cpu().numpy())
    for i,row in enumerate(rows):row['fitted_yaw_deg']=float(Rotation.from_matrix(out['R'][i]).as_euler('xyz',degrees=True)[1])
    if device=='cuda':torch.cuda.synchronize()
    reference_geometry={}
    for state,ref in report['references'].items():
        i=next(i for i,row in enumerate(rows) if row['id']==ref['frame_id'])
        item=dict(frame_id=ref['frame_id'],mesh_unit='m',coordinate_system='reference camera X_right_Y_up_Z_backward',
            observed_valid=data[i]['valid'],observed_landmarks_m=observed_landmarks_m(data[i]['xyz'],data[i]['valid']),
            observed_inner_lip_gap_mm=inner_lip_gap_mm(data[i]['xyz'],data[i]['valid']),stages={})
        for stage,key in (('motion_only','posed_vertices'),('fixed_identity_surface','refined_vertices')):
            name=f'{state}_{stage}_m.obj';vv=out[key][i]
            trimesh.Trimesh(vv,faces,process=False).export(output/name)
            lm=(vv[faces[out['landmark_faces'][i]]]*out['landmark_barycentric'][i,:, :,None]).sum(1)
            item['stages'][stage]=dict(mesh=name,landmark_triangle_indices=out['landmark_faces'][i],
                landmark_barycentric=out['landmark_barycentric'][i],landmarks_m=lm,
                inner_lip_gap_mm=float(np.linalg.norm(lm[62]-lm[66])*1000))
        reference_geometry[state]=item
    write_json(output/'reference_geometry.json',dict(schema='head_rgbd.motion.reference.v2',references=reference_geometry))
    report.update(motion_correction=dict(research_only=True,source=str(initial),source_parameters_sha256=sha(initial/'parameters.npz'),
        shape_exactly_fixed=bool(np.array_equal(out['shape'],saved['shape'])),scale_variable=False,calibration_fixed=True,
        translation_bound_mm=3.,rotation_bound_rad=.02,expression_delta_bound=.2,jaw_delta_bound_rad=.03,
        fixed_ray_acceptance=correction,history=history,elapsed_s=time.perf_counter()-start,
        identity_field=str(identity_field) if identity_field else None,
        fixed_reference_frame=rows[fixed_reference]['id'] if fixed_reference is not None else None,
        gauge='original FLAME coordinate frame, fixed shape and scale, original reference camera axes; only per-frame motion changes'),
        elapsed_s=time.perf_counter()-start,algorithm='research fixed-identity motion/expression correction',
        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0.,
        cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device=='cuda' else 0.,
        limitations=report.get('limitations',[])+['motion refinement is initialized locally; not proof of full identity/expression disentanglement'])
    write_json(output/'report.json',report)
    print(json.dumps(dict(output=str(output),elapsed_s=report['elapsed_s'],accepted_frames=sum(x['scale']>0 and not x.get('gauge_reference_frozen',False) for x in correction))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sequence','initial-fit','output','runtime'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda');p.add_argument('--steps',type=int,default=80)
    p.add_argument('--identity-field',type=Path)
    a=p.parse_args();correct(a.sequence.resolve(),a.initial_fit.resolve(),a.output.resolve(),a.runtime.resolve(),a.device,a.steps,a.identity_field)
