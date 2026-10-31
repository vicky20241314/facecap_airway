import numpy as np
from head_rgbd.normal_cone_d import normal_cones


def test_area_contraction_cannot_hide_an_excessive_normal_tilt():
    base=np.array([[[0.,0.,0.],[.001,0.,0.],[0.,.001,0.]]])
    # Unit animated directions at a sharply curved part of a control surface.
    direction=np.array([[[0.,1.,0.],[-.8,0.,.6],[0.,1.,0.]]])
    faces=np.array([[0,1,2]]);field=np.full(3,.625)
    moved=base[0]+direction[0]*field[:,None]*.001
    cross=np.cross(moved[1]-moved[0],moved[2]-moved[0])
    angle=np.degrees(np.arccos(cross[2]/np.linalg.norm(cross)))
    assert angle>35
    # The previous separate tangent/area bounds accepted this linear example.
    limit=np.tan(np.radians(35*.8))/np.sqrt(2)
    assert abs(cross[0]/1e-6)<limit and cross[2]/1e-6>.1
    blocks=list(normal_cones(base,direction,faces,35))
    assert any(np.any(A@field>upper+1e-10) for A,lower,upper in blocks)


def test_every_actual_pose_contributes_a_normal_constraint():
    base=np.array([[[0.,0.,0.],[.001,0.,0.],[0.,.001,0.]]])
    base=np.repeat(base,3,axis=0)
    n=np.zeros_like(base);n[:,:,2]=1
    assert len(list(normal_cones(base,n,np.array([[0,1,2]]),35)))==15
    from head_rgbd.qp_d import constraints
    from head_rgbd.geometry_d import Settings
    A,l,u=constraints(base,n,np.array([[0,1,2]]),np.array([[0,1],[1,2],[0,2]]),np.array([1.,1.414,1.]),Settings())
    assert A.shape==(21,3) and len(l)==len(u)==21
