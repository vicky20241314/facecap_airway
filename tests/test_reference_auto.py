"""Physical scale, appearance-field transport, and intersection regressions."""
import numpy as np
from head_rgbd.neck_broad_volume import BroadDesign,broad_profile
from head_rgbd.neck_reference_auto import analyse_applied,evaluate_field,local_intersections
from head_rgbd.neck_appearance import geometry_guards,accept_guards


def test_ten_percent_dimensions_mean_actual_ten_percent_level():
    d=BroadDesign(3.,36.,32.)
    np.testing.assert_allclose(broad_profile(np.array([18.,0.]),np.array([0.,16.]),d),[.3,.3])


def test_similarity_field_scales_dimensions_and_amplitude_and_stays_bounded():
    x=np.linspace(-2,2,9); y=np.linspace(-2,2,9)
    X,Y=np.meshgrid(x,y)
    asset={"x_mm":x,"y_mm":y,"field_mm":(1.-X*X/4.)*(1.-Y*Y/4.),"centre_y_mm":0.}
    a=evaluate_field(asset,np.array([0.,1.,3.]),np.zeros(3))
    b=evaluate_field(asset,np.array([0.,2.,6.]),np.zeros(3),scale=2.)
    np.testing.assert_allclose(b,2.*a)
    assert a[-1]==0.


def test_crossing_segment_uses_dimensionless_parameter_at_small_metric_scale():
    # Crossing at t=0.5, segment length=.002 m; the old t<length check missed it.
    V=np.array([[0.,0.,-.001],[0.,0.,.001],[.001,0.,.001],
                [-.002,-.002,0.],[.002,-.002,0.],[0.,.002,0.]])
    F=np.array([[0,1,2],[3,4,5]])
    assert local_intersections(V,F,np.array([1,1,1,0,0,0],bool))["count"]>0


def test_coplanar_overlap_and_shared_edge_contact_are_distinct():
    V=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.],
                [.1,.1,0.],[.2,.1,0.],[.1,.2,0.]])*.001
    assert local_intersections(V,np.array([[0,1,2],[1,3,2]]),np.ones(7,bool))["count"]==0
    assert local_intersections(V,np.array([[0,1,2],[4,5,6]]),np.ones(7,bool))["count"]>0


def test_no_motion_still_checks_inconsistent_winding():
    V=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])*.001
    guards=geometry_guards(rest=V,shaped=V,faces=np.array([[0,1,2],[0,3,2]]),moved=np.zeros(4,bool))
    assert guards["inconsistent_shared_edges"]==1
    assert not accept_guards(guards)


def _bump_state():
    step=.5
    xs=np.arange(-10.,35.+1e-9,step); ys=np.arange(-10.,10.+1e-9,step)
    uu,vv=np.meshgrid(xs,ys,indexing="ij")
    h=(4.*np.exp(-(uu**2+vv**2)/50.)+0.5*np.exp(-((uu-30.)**2+vv**2)/8.)).ravel()
    uv=np.stack([uu.ravel(),vv.ravel()],axis=1)
    verts=np.stack([uv[:,0]*1e-3,uv[:,1]*1e-3,h*1e-3],axis=1)
    nx,ny=len(xs),len(ys); faces=[]
    for i in range(nx-1):
        for j in range(ny-1):
            a=i*ny+j; b=(i+1)*ny+j; c=i*ny+j+1; d=(i+1)*ny+j+1
            faces.extend([[a,b,c],[b,d,c]])
    faces=np.asarray(faces,int)
    # A back-of-neck triangle whose projection sits under the apex and must not
    # contaminate contours, profiles or the curvature fit.
    back=np.array([[0.,0.,-.05],[0.,.001,-.05],[.001,0.,-.05]])
    verts=np.vstack([verts,back])
    uv=np.vstack([uv,np.zeros((3,2))])
    applied=np.concatenate([h,np.zeros(3)])
    faces=np.vstack([faces,[[len(verts)-3,len(verts)-2,len(verts)-1]]])
    class Frame: n=np.array([0.,0.,1.])
    return {"uv_mm":uv,"applied_mm":applied,"refined_vertices":verts,"frame":Frame()},faces


def test_field_audit_excludes_back_projection_and_reports_outer_islands():
    state,faces=_bump_state()
    a=analyse_applied(state,faces)
    assert a["support"]["back_projections_excluded"]>=1
    assert abs(a["peak_outward_mm"]-4.)<.05
    central=a["levels"]["0.5"]["central"]
    width=central["uv_bbox_mm"][1]-central["uv_bbox_mm"][0]
    assert 10.<width<13.5   # 2*sigma*sqrt(2*ln2) = 11.77 mm for this synthetic bump
    assert a["levels"]["0.1"]["component_count"]>=2   # second lobe at u=+30 mm
    assert a["levels"]["0.1"]["global_width_mm"]>a["levels"]["0.5"]["global_width_mm"]
    assert a["residual_audit"]["negative"]["component_count"]==0
    r=a["apex_curvature"][0]
    assert 4.<r["radius_mm"]<9. and r["fit_rms_mm"]<.05   # analytic r=sigma^2/A=6.25 mm
    assert all(p["finite_fraction"]>.3 for p in a["profiles"].values())


def _synthetic_morphology():
    return {
        "peak_mm": 4.6, "reference_neck_width_mm": 127.0,
        "fit": {"vertical_support_up_mm": 13.3, "vertical_support_down_mm": 13.9,
                "lateral_half_width_mm": 11.5, "tip_radius_mm": 4.2,
                "centre_offset_mm": 0.2},
    }


def test_continuous_design_nonnegative_single_body_and_c2_boundary():
    from head_rgbd import neck_continuous as nc
    design, audit = nc.design_from_morphology(_synthetic_morphology(), .95, .9)
    u = np.arange(-60., 60.01, .1)
    p = nc.na.sculpt_profile(u, np.zeros_like(u), design)
    assert p.min() >= -1e-9
    assert p.max() > 3.0
    assert abs(p[0]) < 1e-9 and abs(p[-1]) < 1e-9
    edge_u = design.half_width_mm + design.tip_radius_mm
    near = np.abs(np.abs(u) - edge_u) < .15
    assert np.max(np.abs(np.gradient(p, u)[near])) < .05   # slope vanishes at the lateral edge
    v = np.arange(-60., 60.01, .1)
    env = nc._plateau(v, design.length_up_mm, design.length_down_mm)
    assert abs(env[0]) < 1e-9 and abs(env[-1]) < 1e-9
    near_v = np.abs(np.abs(v) - max(design.length_up_mm, design.length_down_mm)) < .15
    assert np.max(np.abs(np.gradient(env, v)[near_v])) < .05
    # one connected positive region at 10% of the peak, no multiple bumps
    from scipy.ndimage import label
    U, V = np.meshgrid(u, v)
    field = nc.na.sculpt_profile(U, V, design)
    lab, n = label(field >= .1 * field.max())
    assert n == 1
    apex = field.max()
    row = field[np.argmin(np.abs(v)), :]
    assert np.all(np.diff(row[:np.argmax(row) + 1]) >= -.01)
    assert np.all(np.diff(row[np.argmax(row):]) <= .01)
