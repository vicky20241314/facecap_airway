import numpy as np
from head_rgbd.geometry_d import Settings,subdivide
from head_rgbd.hierarchy_d import residual_bounds,constraints,safety


def scene():
    v=np.array([[[0.,0.,0.],[.02,0.,0.],[0.,.02,0.]]])
    n=np.zeros_like(v);n[:,:,2]=1
    return v,n,np.array([[0,1,2]])


def test_vector_prolongation_is_not_normal_scalar_prolongation():
    v,n,f=scene();n[0,1]=[.6,0,.8]
    _,p,_,_=subdivide(f,3);d=np.array([.001,.003,0.])
    exact=p@(v[0]+n[0]*d[:,None])
    pn=p@n[0];pn/=np.linalg.norm(pn,axis=1,keepdims=True)
    old=p@v[0]+pn*(p@d)[:,None]
    assert np.max(np.linalg.norm(exact-old,axis=1))>1e-4
    np.testing.assert_allclose(exact[:3],v[0]+n[0]*d[:,None])


def test_residual_cannot_add_half_mm_on_top_of_total_three_mm():
    ref,n,_=scene();base=ref+n*.0029
    lo,hi=residual_bounds(base,ref,n,Settings())
    np.testing.assert_allclose(hi,.1,atol=1e-10)
    np.testing.assert_allclose(lo,-.5)
    edges=np.array([[0,1],[1,2],[0,2]]);length=np.array([20.,np.sqrt(800),20.])
    ok,record=safety(base,ref,n,np.array([[0,1,2]]),np.full(3,.2),edges,length,np.full(3,2.9),Settings())
    assert not ok and record['reason']=='total_vector_displacement_bound'


def test_affine_cone_accepts_safe_deformed_zero_residual():
    ref,n,f=scene();base=ref.copy();base[0,1,2]=.002
    edges=np.array([[0,1],[1,2],[0,2]])
    length=np.linalg.norm((ref[0,edges[:,0]]-ref[0,edges[:,1]])*1000,axis=1)
    a,lo,hi=constraints(base,ref,n,f,edges,length,np.array([0.,2.,0.]),Settings(),np.ones(3,bool))
    assert np.all(a@np.zeros(3)>=lo-1e-10) and np.all(a@np.zeros(3)<=hi+1e-10)
    ok,record=safety(base,ref,n,f,np.zeros(3),edges,length,np.array([0.,2.,0.]),Settings())
    assert ok and record['maximum_total_displacement_mm']<=2.000001
