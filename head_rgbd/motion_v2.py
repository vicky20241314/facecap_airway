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
from head_rgbd.motion_acceptance import batch_backtrack


def fixed_ray_residual(v,faces,points):
    corr,_=ray_surface(v,faces,points,.030,None)
    residual=np.full(len(points),8.)
    covered=np.zeros(len(points),bool)
    if corr is not None:
        covered[corr['ray']]=True
        residual[corr['ray']]=np.abs(corr['hit'][:,2]-corr['target'][:,2])*1000
    return residual,covered


def fixed_ray_score(v,faces,points,required=None):
    residual,covered=fixed_ray_residual(v,faces,points)
    mask=np.ones(len(points),bool) if required is None else required
    return (float(np.mean(residual[mask]**2)) if np.any(mask) else float('inf')),covered


def _frame_loss_weights(rows):
    """Use supplied angle-balanced weights, with the historic uniform fallback."""
    values=np.asarray([row.get('frame_weight',np.nan) for row in rows],dtype=np.float64)
    if np.isfinite(values).all() and np.all(values>0):
        return values/values.sum()
    return np.full(len(rows),1./max(len(rows),1),dtype=np.float64)


def correct(sequence,initial,output,runtime,device='cuda',steps=80,identity_field=None,lowerface_weight=.5,
            correspondence_interval=10,expression_start=None,expression_end=None,loss_mode='legacy'):
    if not np.isfinite(lowerface_weight) or lowerface_weight<0:
        raise ValueError('lowerface weight must be finite and nonnegative')
    expr_start=steps//2 if expression_start is None else int(expression_start)
    expr_end=3*steps//4 if expression_end is None else int(expression_end)
    if not 0<=expr_start<expr_end<=steps:
        raise ValueError('expression phase window must satisfy 0 <= start < end <= steps')
    if int(correspondence_interval)<1:raise ValueError('correspondence interval must be >= 1')
    if loss_mode not in ('legacy','batch'):raise ValueError('unsupported loss mode')
    configure_environment(runtime);start=time.perf_counter();torch.set_num_threads(4)
    output.mkdir(parents=True,exist_ok=False)
    report=json.loads((initial/'report.json').read_text(encoding='utf-8'))
    if report['sequence_sha256']!=sha(sequence):raise ValueError('sequence mismatch')
    saved=load_npz(initial/'parameters.npz');rows=report['frames'];n=len(rows)
    frame_loss_weights=_frame_loss_weights(rows)
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
    rng=np.random.default_rng(7652);points={};lower_points={}
    for i,d in enumerate(data):
        p,uv,_=sample_depth(d,1800,rng,True,stride=3)
        skin=~eye_mask(d)[uv[:,1],uv[:,0]]
        stable=skin & (uv[:,1]<d['uv'][33,1])
        points[i]=p[stable]
        lower=skin & (uv[:,1]>=d['uv'][33,1])
        if lowerface_weight>0 and rows[i]['state']=='open' and rows[i]['view']=='front' and lower.sum()>=80 and stable.sum()>=80:
            lower_points[i]=p[lower]
        if len(points[i])<80:
            rows[i]['degraded']=rows[i].get('degraded',[])+['insufficient_stable_skin_motion_correction_skipped']
    with torch.no_grad():old_forward=forward();old_fit=old_forward[-1].cpu().numpy()
    old_scores=[];required=[]
    for i in range(n):
        residual,c=fixed_ray_residual(old_fit[i],faces,points[i])
        s=float(np.mean(residual[c]**2)) if c.any() else float('inf')
        old_scores.append(s);required.append(c)
    lower_required={};lower_old_scores={}
    for i,p in lower_points.items():
        residual,covered=fixed_ray_residual(old_fit[i],faces,p)
        if covered.sum()>=30:
            lower_required[i]=covered
            lower_old_scores[i]=float(np.mean(residual[covered]**2))
    # ---- P2 frame-dimension batch loss tensors (legacy loop untouched) ----
    from head_rgbd.loss_batch import pack_correspond,masked_mean,smooth_l1
    fw_t=torch.as_tensor(frame_loss_weights,device=device,dtype=torch.float32)
    lm_valid_np=[]
    for d in data:
        vld=d['valid'].copy();vld[:17]=False;vld[48:]=False;lm_valid_np.append(vld)
    lm_valid_t=torch.as_tensor(np.stack(lm_valid_np),device=device)
    xyz_t=torch.nan_to_num(torch.stack([T(d['xyz']) for d in data]))
    uv_t=torch.nan_to_num(torch.stack([T(d['uv']) for d in data]))
    cols=torch.arange(68,device=device)
    finite_uv_t=torch.as_tensor(np.stack([np.isfinite(d['uv']).all(1) for d in data]),
                                device=device)&(cols>=48)[None,:]
    closed_t=torch.tensor([row['state']=='closed' for row in rows],device=device)
    intr_t=torch.as_tensor(np.asarray([d['intrinsics'] for d in data]),
                           device=device,dtype=torch.float32)
    corr_pack=None;lower_pack=None
    opt=torch.optim.Adam([{'params':[dr],'lr':.0006},{'params':[dt],'lr':.00008},
                          {'params':[de],'lr':.006},{'params':[dj],'lr':.0006}])
    temporal=[(i-1,i) for i in range(1,n) if temporal_edge_allowed(rows[i-1],rows[i])]
    history=[];correspond={};lower_correspond={}
    loop_timing=dict(steps=int(steps),correspondence_steps=0,correspondence_s=0.,step_s=0.,
                     forward_s=0.,loss_s=0.,backward_s=0.,opt_s=0.)
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    def batch_loss():
        """Frame-dimension equivalent of the legacy per-frame motion loss."""
        total=v.sum()*0
        if corr_pack is not None:
            tri,bc,normal,target,cmask=corr_pack
            idx=torch.arange(n,device=device)[:,None,None]
            surf=(fit_v[idx,tri]*bc[:,:,:,None]).sum(2)
            res=((surf-target)*normal).sum(2)/.002
            total=total+(fw_t*masked_mean(smooth_l1(res,1.),cmask)).sum()
            present=cmask.any(1)
            total=total+(fw_t*.12*masked_mean(smooth_l1((l-xyz_t)/.003,1.),lm_valid_t)*present).sum()
            if expression_phase:
                z=(-l[:,:,2]).clamp(min=.05)
                proj=torch.stack((intr_t[:,0:1]*l[:,:,0]/z+intr_t[:,2:3],
                                  -intr_t[:,1:2]*l[:,:,1]/z+intr_t[:,3:4]),2)
                total=total+(fw_t*.08*masked_mean(smooth_l1((proj-uv_t)/3,1.),finite_uv_t)*present).sum()
                gaps=torch.linalg.norm(l[:,[61,62,63]]-l[:,[67,66,65]],dim=2)
                gt=(torch.relu(gaps-.002)/.003).square().mean(1)
                total=total+(fw_t*.1*gt*closed_t*present).sum()
        if expression_phase and lower_pack is not None:
            tri,bc,normal,target,cmask=lower_pack
            idx=torch.arange(n,device=device)[:,None,None]
            surf=(fit_v[idx,tri]*bc[:,:,:,None]).sum(2)
            res=((surf-target)*normal).sum(2)/.002
            total=total+(fw_t*lowerface_weight*masked_mean(smooth_l1(res,1.),cmask)).sum()
        return total
    for step in range(steps):
        sync();loop_started=time.perf_counter();correspondence_s=0.
        expression_phase=expr_start<=step<expr_end
        opt.zero_grad();raw,v,l,R,fit_v=forward();loss=v.sum()*0
        sync();forward_end=time.perf_counter()
        if step%int(correspondence_interval)==0 or step in (expr_start,expr_end):
            sync();correspondence_started=time.perf_counter()
            correspond={}
            for i in range(n):
                if len(points[i])<80:continue
                c,stats=ray_surface(fit_v[i].detach().cpu().numpy(),faces,points[i],.008,.005)
                if c is not None:
                    correspond[i]=(ft[torch.as_tensor(c['face'],device=device)],T(c['bary']),T(c['normal']),T(c['target']))
            lower_correspond={}
            if expression_phase:
                for i in lower_required:
                    c,_=ray_surface(fit_v[i].detach().cpu().numpy(),faces,lower_points[i],.008,.005)
                    if c is not None:
                        lower_correspond[i]=(ft[torch.as_tensor(c['face'],device=device)],T(c['bary']),T(c['normal']),T(c['target']))
            sync();correspondence_s=time.perf_counter()-correspondence_started
            if loss_mode=='batch':
                corr_pack=pack_correspond(correspond,n,device)
                lower_pack=pack_correspond(lower_correspond,n,device) if expression_phase else None
            loop_timing['correspondence_s']+=correspondence_s
            loop_timing['correspondence_steps']+=1
        for i,(tri,bc,normal,target) in (correspond.items() if loss_mode=='legacy' else ()):
            surf=(fit_v[i,tri]*bc[:,:,None]).sum(1)
            residual=((surf-target)*normal).sum(1)/.002
            loss+=frame_loss_weights[i]*torch.nn.functional.smooth_l1_loss(
                residual,torch.zeros_like(residual),beta=1.)
            valid=data[i]['valid'].copy();valid[:17]=False;valid[48:]=False
            if valid.any():
                difference=(l[i][valid]-T(data[i]['xyz'][valid]))/.003
                loss+=frame_loss_weights[i]*.12*torch.nn.functional.smooth_l1_loss(
                    difference,torch.zeros_like(difference))
            if expression_phase:
                uv=T(data[i]['uv']);finite=torch.isfinite(uv).all(1);finite[:48]=False
                fx,fy,cx,cy=data[i]['intrinsics'];z=(-l[i,:,2]).clamp(min=.05)
                projected=torch.stack((fx*l[i,:,0]/z+cx,-fy*l[i,:,1]/z+cy),1)
                if finite.any():loss+=frame_loss_weights[i]*.08*torch.nn.functional.smooth_l1_loss(
                    projected[finite]/3,uv[finite]/3)
                if rows[i]['state']=='closed':
                    gaps=torch.linalg.norm(l[i,[61,62,63]]-l[i,[67,66,65]],dim=1)
                    loss+=frame_loss_weights[i]*.1*(torch.relu(gaps-.002)/.003).square().mean()
        lower_loss=v.sum()*0
        if expression_phase:
            for i,(tri,bc,normal,target) in (lower_correspond.items() if loss_mode=='legacy' else ()):
                surf=(fit_v[i,tri]*bc[:,:,None]).sum(1)
                residual=((surf-target)*normal).sum(1)/.002
                lower_loss+=frame_loss_weights[i]*lowerface_weight*torch.nn.functional.smooth_l1_loss(
                    residual,torch.zeros_like(residual),beta=1.)
        loss+=lower_loss
        if loss_mode=='batch':
            loss+=batch_loss()
        loss+=.01*(dr/.02).square().mean()+.01*(dt/.003).square().mean()
        loss+=.02*(de/.2).square().mean()+.02*(dj/.03).square().mean()
        for i,j in temporal:
            loss+=.002*((de[j]-de[i]).square().mean()+(dj[j]-dj[i]).square().mean())/max(len(temporal),1)
        sync();loss_end=time.perf_counter()
        loss.backward()
        sync();backward_end=time.perf_counter()
        if expression_phase:dr.grad=None;dt.grad=None
        else:de.grad=None;dj.grad=None
        opt.step()
        with torch.no_grad():
            dr.mul_(torch.clamp(.02/torch.linalg.norm(dr,dim=1,keepdim=True).clamp(min=1e-12),max=1.))
            dt.mul_(torch.clamp(.003/torch.linalg.norm(dt,dim=1,keepdim=True).clamp(min=1e-12),max=1.))
            de.clamp_(-.2,.2);dj.clamp_(-.03,.03)
            if fixed_reference is not None:
                for variable in (dr,dt,de,dj):variable[fixed_reference].zero_()
        sync();opt_end=time.perf_counter()
        loop_timing['forward_s']+=forward_end-loop_started
        loop_timing['loss_s']+=loss_end-forward_end-correspondence_s
        loop_timing['backward_s']+=backward_end-loss_end
        loop_timing['opt_s']+=opt_end-backward_end
        loop_timing['step_s']+=opt_end-loop_started-correspondence_s
        # Record-only: per-20-step samples plus the last step of each phase.
        if step%20==0 or step in (expr_start-1,expr_end-1,steps-1):
            history.append(dict(step=step,phase='expression_jaw' if expression_phase else 'rigid',objective=float(loss.detach()),
                                lowerface_depth_objective=float(lower_loss.detach()),objective_scope='before_optimizer_update'))
            print(history[-1],flush=True)
    # Each frame is accepted against its ORIGINAL fixed rays; disappearing rays
    # retain a penalty. Never count only the surviving easy ICP correspondences.
    correction=[None]*n
    proposed=[x.detach().clone() for x in (dr,dt,de,dj)]
    scales=np.zeros(n)
    pending=[]
    for i in range(n):
        if required[i].sum()<30:
            rows[i]['degraded']=rows[i].get('degraded',[])+['insufficient_fixed_motion_rays_original_motion_retained']
            correction[i]=dict(frame_id=rows[i]['id'],scale=0.,accepted=False,reason='insufficient_fixed_rays',
                original_fixed_ray_mse_mm2=None,final_fixed_ray_mse_mm2=None,fixed_ray_count=int(required[i].sum()))
            continue
        pending.append(i)
    def evaluate_scale(alpha):
        with torch.no_grad():
            for variable,p in zip((dr,dt,de,dj),proposed):variable.copy_(p*alpha)
            return forward()[-1].cpu().numpy()
    def accept_frame(i,candidate):
        score,covered=fixed_ray_score(candidate,faces,points[i],required[i])
        accepted=score<=old_scores[i]+1e-8 and np.all(covered[required[i]])
        lower_metrics={}
        if i in lower_required:
            lower_score,lower_covered=fixed_ray_score(candidate,faces,lower_points[i],lower_required[i])
            lower_ok=lower_score<=lower_old_scores[i]+1e-8 and np.all(lower_covered[lower_required[i]])
            accepted=accepted and lower_ok
            lower_metrics=dict(lowerface_original_mse_mm2=lower_old_scores[i],lowerface_final_mse_mm2=lower_score,
                               lowerface_fixed_ray_count=int(lower_required[i].sum()))
        return dict(accepted=bool(accepted),original_fixed_ray_mse_mm2=old_scores[i],final_fixed_ray_mse_mm2=score,
                    fixed_ray_count=int(required[i].sum()),**lower_metrics)
    accepted_frames,acceptance_forward_calls=batch_backtrack(pending,evaluate_scale,accept_frame)
    for i,info in accepted_frames.items():
        alpha=info['scale'];scales[i]=alpha
        correction[i]=dict(frame_id=rows[i]['id'],**info,gauge_reference_frozen=i==fixed_reference,
            rotation_delta_deg=float(torch.linalg.norm(proposed[0][i]).cpu())*alpha*180/np.pi,
            translation_delta_mm=float(torch.linalg.norm(proposed[1][i]).cpu())*alpha*1000)
    with torch.no_grad():
        for variable,p in zip((dr,dt,de,dj),proposed):variable.copy_(p*T(scales)[:,None])
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
    np.savez(output/'parameters.npz',**out)
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
            vv=out[key][i]
            lm=(vv[faces[out['landmark_faces'][i]]]*out['landmark_barycentric'][i,:, :,None]).sum(1)
            item['stages'][stage]=dict(mesh=None,storage='parameters.npz',vertex_array=key,frame_index=i,
                landmark_triangle_indices=out['landmark_faces'][i],
                landmark_barycentric=out['landmark_barycentric'][i],landmarks_m=lm,
                inner_lip_gap_mm=float(np.linalg.norm(lm[62]-lm[66])*1000))
        reference_geometry[state]=item
    write_json(output/'reference_geometry.json',dict(schema='head_rgbd.motion.reference.v2',references=reference_geometry))
    report.update(motion_correction=dict(research_only=True,steps=steps,source=str(initial),source_parameters_sha256=sha(initial/'parameters.npz'),
        lowerface_weight=lowerface_weight,lowerface_frames=[rows[i]['id'] for i in lower_required],
        frame_weighting=dict(policy=('angle-group totals from the shared fit; repeated observations '
                                    'split their angle-group vote'),weights=frame_loss_weights.tolist()),
        initial_fixed_ray_queries=n+len(lower_points),
        acceptance_forward_calls=acceptance_forward_calls,acceptance_strategy='batch_by_shared_backtracking_scale',
        shape_exactly_fixed=bool(np.array_equal(out['shape'],saved['shape'])),scale_variable=False,calibration_fixed=True,
        translation_bound_mm=3.,rotation_bound_rad=.02,expression_delta_bound=.2,jaw_delta_bound_rad=.03,
        fixed_ray_acceptance=correction,history=history,elapsed_s=time.perf_counter()-start,
        timing=loop_timing,
        timing_step_scope='loop wall time split: correspondence_s = CPU ray_surface refresh block; step_s = forward_s + loss_s + backward_s + opt_s; torch.cuda.synchronize() at each boundary; acceptance/exports excluded',
        identity_field=str(identity_field) if identity_field else None,
        fixed_reference_frame=rows[fixed_reference]['id'] if fixed_reference is not None else None,
        gauge='original FLAME coordinate frame, fixed shape and scale, original reference camera axes; only per-frame motion changes'),
        elapsed_s=time.perf_counter()-start,algorithm='research fixed-identity motion/expression correction',
        output_policy=dict(meshes='none',parameter_archive='stored_npz'),
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
    p.add_argument('--lowerface-weight',type=float,default=.5,
                   help='reliable open-front lower-face depth in expression phase; 0 disables it')
    p.add_argument('--correspondence-interval',type=int,default=10,
                   help='steps between surface correspondence refreshes (phase start always refreshes)')
    p.add_argument('--expression-start',type=int,default=None,
                   help='first expression/jaw step; default steps//2 (keeps the historic 40 at 80 steps)')
    p.add_argument('--expression-end',type=int,default=None,
                   help='exclusive expression/jaw end step; default 3*steps//4 (historic 60)')
    p.add_argument('--loss-mode',choices=('legacy','batch'),default='legacy',
                   help='legacy per-frame Python loss loop, or frame-dimension batch loss '
                        'with identical weights/denominators/beta')
    a=p.parse_args();correct(a.sequence.resolve(),a.initial_fit.resolve(),a.output.resolve(),a.runtime.resolve(),
        a.device,a.steps,a.identity_field,a.lowerface_weight,a.correspondence_interval,
        a.expression_start,a.expression_end,a.loss_mode)
