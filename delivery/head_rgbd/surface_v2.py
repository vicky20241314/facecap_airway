"""Research FEM total-deformation energies and budgeted Loop stencils.

Independent mathematical implementation; no libigl/OpenSubdiv source copied.
All differential operators use millimetres and lumped surface area quadrature.
"""
import numpy as np
from scipy import sparse
from head_rgbd.geometry_d import subdivide, edges_of


def fem_operators(vertices, faces):
    v=np.asarray(vertices,dtype=float)*1000
    t=v[faces];cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
    double_area=np.linalg.norm(cross,axis=1)
    tolerance=max(float(double_area.max())*1e-12,1e-15)
    if np.any(double_area<tolerance):raise ValueError('FEM reference has degenerate triangles')
    unit=cross/double_area[:,None];area=double_area*.5
    gradients=np.stack((np.cross(unit,t[:,2]-t[:,1]),np.cross(unit,t[:,0]-t[:,2]),
                        np.cross(unit,t[:,1]-t[:,0])),axis=2)/double_area[:,None,None]
    row=np.repeat(np.arange(len(faces)*3),3)
    col=np.repeat(faces[:,None,:],3,axis=1).ravel()
    G=sparse.coo_matrix((gradients.ravel(),(row,col)),shape=(len(faces)*3,len(v))).tocsr()
    W=np.repeat(area,3)
    K=(G.T@G.multiply(W[:,None])).tocsr()
    mass=np.bincount(faces.ravel(),weights=np.repeat(area/3,3),minlength=len(v))
    inv=np.zeros_like(mass);valid=mass>tolerance;inv[valid]=1/mass[valid]
    L=sparse.diags(inv)@K
    return G,L,area,mass


def total_regularizer(base,reference,direction,faces,settings):
    """Penalize curvature CHANGE from the original coarse prior, not facial curvature.

    Dense base offsets (including Loop motion) enter the affine right hand side.
    All animation states share one scalar; three representative states contribute
    energy to avoid overweighting near-duplicate frames. Safety still checks ALL.
    """
    G,L,area,mass=fem_operators(reference[0],faces)
    total=mass.sum();grad=G.multiply(np.sqrt(np.repeat(area,3)/total)[:,None])
    B=sparse.diags(np.sqrt(mass/total))@L
    matrices=[settings.gradient_length_mm*grad,.08*sparse.diags(np.sqrt(mass/total))]
    biases=[np.zeros(G.shape[0]),np.zeros(len(mass))]
    # Direction fields transformed to the first state's camera orientation produce
    # equivalent vector norm energies under rigid rotations. Use every state for
    # expression-sensitive bending, normalized by count.
    states=np.unique(np.linspace(0,len(base)-1,min(4,len(base))).astype(int))
    for i in states:
        scale=settings.bending_length_mm**2/np.sqrt(len(states))
        offset=(base[i]-reference[i])*1000
        for axis in range(3):
            matrices.append(scale*B@sparse.diags(direction[i,:,axis]))
            biases.append(scale*(B@offset[:,axis]))
    return sparse.vstack(matrices).tocsr(),np.concatenate(biases),area,mass


def loop_mapping(faces,count,excluded):
    """One Loop step, with anatomical and open/nonmanifold boundary stencils fixed.

    The same triangle child parameter domain is retained. A global blend with the
    exact linear prolongation is selected by nonlinear geometry safety.
    """
    child,P,edges,parent=subdivide(faces,count)
    adjacent={tuple(edge):[] for edge in edges}
    neighbors=[set() for _ in range(count)]
    for a,b,c in faces:
        for x,y,z in ((a,b,c),(b,c,a),(c,a,b)):
            adjacent[tuple(sorted((x,y)))].append(z);neighbors[x].add(y);neighbors[y].add(x)
    lock=np.asarray(excluded,bool).copy()
    for (a,b),opposite in adjacent.items():
        if len(opposite)!=2:lock[[a,b]]=True
    # No stencil can mix across an excluded anatomical region.
    for i,ns in enumerate(neighbors):
        if any(excluded[j] for j in ns):lock[i]=True
    S=P.tolil()
    for i,ns in enumerate(neighbors):
        if lock[i] or len(ns)<3:continue
        n=len(ns);beta=(5/8-(3/8+.25*np.cos(2*np.pi/n))**2)/n
        S.rows[i]=[i]+sorted(ns);S.data[i]=[1-n*beta]+[beta]*n
    for k,(a,b) in enumerate(edges):
        opposite=adjacent[(a,b)]
        if len(opposite)==2 and not lock[a] and not lock[b] and not np.any(excluded[opposite]):
            S.rows[count+k]=[int(a),int(b),int(opposite[0]),int(opposite[1])]
            S.data[count+k]=[.375,.375,.125,.125]
    return child,P,S.tocsr(),edges,parent,lock


def local_depth_planes(d,uv):
    """Small observed planes, rejecting holes, cavities, semantic/depth boundaries.

    The scatter estimates local consistency, NOT calibrated sensor accuracy.
    """
    from head_rgbd.geometry_c import mouth_pixels
    from head_rgbd.geometry_d import eye_mask
    depth=d['depth'].astype(float)*.001;h,w=depth.shape
    offsets=np.array([(x,y) for y in range(-2,3) for x in range(-2,3)])
    pixel=uv[:,None,:]+offsets[None]
    x=np.clip(pixel[:,:,0],0,w-1);y=np.clip(pixel[:,:,1],0,h-1)
    z=depth[y,x];semantic=d['face_mask']&~mouth_pixels(d)&~eye_mask(d)
    valid=(pixel[:,:,0]>=0)&(pixel[:,:,0]<w)&(pixel[:,:,1]>=0)&(pixel[:,:,1]<h)
    valid&=(z>.1)&(z<1.5)&semantic[y,x]
    fx,fy,cx,cy=d['intrinsics']
    points=np.stack(((x-cx)*z/fx,-(y-cy)*z/fy,-z),axis=2)
    centered=points-points.mean(1,keepdims=True)
    covariance=np.einsum('nki,nkj->nij',centered,centered)/25
    eigen,vectors=np.linalg.eigh(covariance)
    normal=vectors[:,:,0]
    flip=(normal*(-points.mean(1))).sum(1)<0;normal[flip]*=-1
    rms=np.sqrt(np.maximum(eigen[:,0],0))*1000
    good=valid.all(1)&((z.max(1)-z.min(1))<.010)&(eigen[:,1]>1e-9)
    good&=rms<1.5
    return normal,rms,good


def canonical_targets(b,sensitivity):
    """Convert camera point-plane targets to shared scalar identity units."""
    return np.divide(b,sensitivity,out=np.full_like(b,np.nan),where=np.abs(sensitivity)>.15)


def burst_weights(rows,identity_ids):
    groups={rows[i]['segment'] for i in identity_ids}
    group_weights={g:(3. if next(rows[i]['view'] for i in identity_ids if rows[i]['segment']==g)=='front' else 1.) for g in groups}
    total=sum(group_weights.values())
    return {i:group_weights[rows[i]['segment']]/total/sum(rows[j]['segment']==rows[i]['segment'] for j in identity_ids) for i in identity_ids}


def reliable_evidence(vertices,faces,data,rows,points=5000):
    """Requery THIS surface; distinguish correlated repeats from view diversity."""
    from head_rgbd.geometry_c import sample_depth,ray_surface
    from head_rgbd.geometry_d import eye_mask
    rng=np.random.default_rng(7891);observed=np.zeros(vertices.shape[:2],bool);logs=[]
    for i,d in enumerate(data):
        xyz,uv,sampling=sample_depth(d,points,rng,True,stride=3)
        normal,scatter,good=local_depth_planes(d,uv)
        c,stats=ray_surface(vertices[i],faces,xyz,.008,.005)
        if c is not None:
            keep=good[c['ray']]&((normal[c['ray']]*c['normal']).sum(1)>.5)
            ids=faces[c['face'][keep]];bc=c['bary'][keep]
            observed[i,np.unique(ids[bc>.05])]=True
            stats['local_plane_accepted']=int(keep.sum())
        logs.append(dict(frame_index=i,**sampling,**stats))
    segments=sorted(set(row['segment'] for row in rows if row['state']=='closed'))
    groups=np.array([observed[[i for i,row in enumerate(rows) if row['segment']==s and row['state']=='closed']].any(0) for s in segments])
    closed_count=groups.sum(0)
    yaw=np.array([row['fitted_yaw_deg'] for row in rows])
    minimum=np.full(vertices.shape[1],np.inf);maximum=-minimum.copy()
    for i,row in enumerate(rows):
        if row['state']=='closed':
            minimum[observed[i]]=np.minimum(minimum[observed[i]],yaw[i]);maximum[observed[i]]=np.maximum(maximum[observed[i]],yaw[i])
    span=np.where(np.isfinite(minimum),maximum-minimum,0)
    repeated_distinct=(closed_count>=2)&(span>=8.)
    counts=observed.sum(0)
    return dict(observed_frame_mask=observed,observed_frames=counts,
        closed_burst_support_mask=groups,closed_burst_ids=np.array(segments),closed_burst_count=closed_count,
        observed_closed_yaw_span_deg=span,
        region_label=np.where(repeated_distinct,2,np.where(counts>0,1,0)).astype(np.uint8),
        confidence=np.minimum(closed_count/3.,1.)*np.minimum(span/30.+.25,1.),
        label_names=np.array(['unobserved_model_completion','single_or_correlated_observation_prior','distinct_closed_burst_and_angle_supported']),
        confidence_is_calibrated=np.array(False)),logs
