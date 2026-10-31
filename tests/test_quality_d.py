import numpy as np
from head_rgbd.quality_d import segment_triangle, triangle_risks, physical_roughness


def test_physical_neighborhood_does_not_mix_close_disconnected_layers():
    xy=np.array([[x*.002,y*.002] for y in range(3) for x in range(3)])
    v=np.column_stack((xy,np.zeros(9)))
    f=[]
    for y in range(2):
        for x in range(2):
            a=y*3+x;f.extend([[a,a+1,a+3],[a+1,a+4,a+3]])
    f=np.array(f)
    vv=np.vstack((v,v+[0.,0.,.001]));ff=np.vstack((f,f[:,::-1]+9))
    result=physical_roughness(vv,ff,vv,ff,np.ones(len(ff),bool))
    assert result['normal_spread_rms_deg']['count']>0
    assert result['normal_spread_rms_deg']['maximum']<1e-6


def test_triangle_crossing_and_separation_are_distinguished():
    triangle=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    assert segment_triangle(np.array([[.2,.2,-1]]),np.array([[.2,.2,1]]),triangle)[0]
    assert not segment_triangle(np.array([[2.,2.,-1]]),np.array([[2.,2.,1]]),triangle)[0]
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[.2,.2,-1],[.2,.2,1],[.8,.2,0.]])
    result=triangle_risks(v,np.array([[0,1,2],[3,4,5]]))
    assert result['proper_intersection_pairs']==1


def test_coplanar_overlapping_nonincident_faces_are_reported():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[.1,.1,0.],[.3,.1,0.],[.1,.3,0.]])
    assert triangle_risks(v,np.array([[0,1,2],[3,4,5]]))['coplanar_overlap_pairs']==1
