"""Exact control-surface prolongation and bounded measured residual geometry.

The residual is shared, but the interpolated animated control displacement is
a per-state vector. Never replace P(N*d) with normalize(P*N)*(P*d).
"""
import numpy as np
from scipy import sparse


def residual_bounds(base, reference, direction, settings):
    offset=(base-reference)*1000
    along=(offset*direction).sum(2)
    perpendicular=np.maximum((offset*offset).sum(2)-along*along,0)
    radius=settings.displacement_bound_mm
    if perpendicular.max()>radius**2+1e-6:
        raise ValueError('control surface already exceeds total geometric budget')
    remaining=np.sqrt(np.maximum(radius**2-perpendicular,0))
    lower=np.maximum((-along-remaining).max(0),-settings.dense_residual_bound_mm)
    upper=np.minimum((-along+remaining).min(0),settings.dense_residual_bound_mm)
    if np.any(lower>1e-6) or np.any(upper< -1e-6):
        raise ValueError('interpolated control surface is not a feasible zero residual')
    return lower,upper


def constraints(base, reference, direction, faces, edges, lengths, proxy, settings, active):
    count=base.shape[1]
    lower,upper=residual_bounds(base,reference,direction,settings)
    edges=edges[active[edges].any(1)]
    # Lengths refer to material edges on the original, undeformed dense base.
    all_lengths=np.linalg.norm((reference[0][edges[:,0]]-reference[0][edges[:,1]])*1000,axis=1)
    matrices=[sparse.eye(count,format='csr')];lows=[lower];highs=[upper]
    edge=sparse.coo_matrix((np.tile([1.,-1.],len(edges)),
        (np.repeat(np.arange(len(edges)),2),edges.ravel())),shape=(len(edges),count)).tocsr()
    shift=proxy[edges[:,0]]-proxy[edges[:,1]]
    limit=settings.maximum_slope*np.maximum(all_lengths,.05)
    matrices.append(edge);lows.append(-limit-shift);highs.append(limit-shift)
    faces=faces[active[faces].any(1)]
    cone=np.tan(np.radians(settings.maximum_normal_change_deg))*np.cos(np.pi/8)
    for v,ref,d in zip(base,reference,direction):
        rt=ref[faces];original=np.cross(rt[:,1]-rt[:,0],rt[:,2]-rt[:,0])
        area=np.maximum(np.linalg.norm(original,axis=1),1e-14)
        unit=original/area[:,None]
        tangent=rt[:,1]-rt[:,0]
        tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-14)
        tangent2=np.cross(unit,tangent)
        tri=v[faces];n=d[faces];e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
        current=np.cross(e1,e2)/area[:,None]
        jac=np.stack((-np.cross(n[:,0],e2)-np.cross(e1,n[:,0]),
                      np.cross(n[:,1],e2),np.cross(e1,n[:,2])),axis=1)*.001/area[:,None,None]
        normal=(jac*unit[:,None]).sum(2);current_normal=(current*unit).sum(1)
        def matrix(values):
            return sparse.coo_matrix((values.ravel(),(np.repeat(np.arange(len(faces)),3),faces.ravel())),
                                     shape=(len(faces),count)).tocsr()
        # Inscribed 8-sided cone: linear approximation only; actual 35deg check remains.
        for angle in np.arange(8)*np.pi/4:
            axis=np.cos(angle)*tangent+np.sin(angle)*tangent2
            transverse=(jac*axis[:,None]).sum(2)
            matrices.append(matrix(transverse-cone*normal))
            lows.append(np.full(len(faces),-np.inf))
            highs.append(cone*current_normal-(current*axis).sum(1))
        matrices.append(matrix(normal))
        lows.append(settings.minimum_signed_area_ratio-current_normal)
        highs.append(np.full(len(faces),np.inf))
    return sparse.vstack(matrices).tocsr(),np.concatenate(lows),np.concatenate(highs)


def safety(base, reference, direction, faces, residual, edges, lengths, proxy, settings):
    if np.abs(residual).max(initial=0)>settings.dense_residual_bound_mm+1e-7:
        return False,dict(reason='dense_residual_bound')
    actual=base+direction*residual[None,:,None]*.001
    total=np.linalg.norm(actual-reference,axis=2).max()*1000
    if total>settings.displacement_bound_mm+1e-7:
        return False,dict(reason='total_vector_displacement_bound',maximum_total_displacement_mm=float(total))
    field=proxy+residual
    slope=np.abs(field[edges[:,0]]-field[edges[:,1]])/np.maximum(lengths,.05)
    if slope.max(initial=0)>settings.maximum_slope+1e-8:
        return False,dict(reason='physical_slope',maximum=float(slope.max()))
    min_ratio,max_angle=np.inf,0.
    for old_v,new_v in zip(reference,actual):
        old=old_v[faces];new=new_v[faces]
        a=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0])
        b=np.cross(new[:,1]-new[:,0],new[:,2]-new[:,0])
        a2=(a*a).sum(1);valid=a2>1e-24
        ratio=(a*b).sum(1)[valid]/a2[valid]
        cosine=(a*b).sum(1)[valid]/np.maximum(np.linalg.norm(a[valid],axis=1)*np.linalg.norm(b[valid],axis=1),1e-30)
        min_ratio=min(min_ratio,float(ratio.min()))
        max_angle=max(max_angle,float(np.degrees(np.arccos(np.clip(cosine,-1,1))).max()))
    record=dict(minimum_signed_area_ratio=min_ratio,maximum_normal_change_deg=max_angle,
                maximum_slope=float(slope.max(initial=0)),maximum_total_displacement_mm=float(total),
                maximum_residual_mm=float(np.abs(residual).max(initial=0)))
    return min_ratio>=settings.minimum_signed_area_ratio and max_angle<=settings.maximum_normal_change_deg,record
