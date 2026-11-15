"""Reference neck surface transfer on an existing reconstruction; isolated experiment.

The public reconstruction/defaults are deliberately not changed.  A signed,
two-dimensional reference relief replaces the old neck's medium-scale relief.
All coordinates inside the atlas are mm; exported meshes remain in metres.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.signal import find_peaks
from scipy import sparse

from head_rgbd import mesh_integrity as mi
from head_rgbd import neck_volume as nv
from head_rgbd import neck_subject_v3 as n3
from head_rgbd.neck_prominence import vertex_normals

PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "runtime/outputs/zp_capwitharuco"
BASE = PROJECT / "runtime/outputs/zp_capwitharuco_neck_reshaped"
REF = Path("D:/project_neck/runtime/v8_tail_smoke/experiment_20261109-200612/final/identity_rest.obj")
OUT = PROJECT / "runtime/outputs/zp_capwitharuco_neck_structured"
STEP = .5


def dump(path, obj):
    def encode(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=encode), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raster(P, F, xs, ys, values=None):
    """Orthographic triangle rasterizer: maximal depth, optional Gouraud values.

    Interpolates the actual triangles, with no Delaunay bridging of holes and
    no averaging front and back surfaces. P columns are screen x,y,depth.
    """
    z = np.full((len(ys), len(xs)), -np.inf)
    colour = np.zeros_like(z) if values is not None else None
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    for f in F:
        p = P[f]
        ix0 = max(0, int(np.ceil((p[:, 0].min() - xs[0]) / dx - 1e-7)))
        ix1 = min(len(xs)-1, int(np.floor((p[:, 0].max() - xs[0]) / dx + 1e-7)))
        iy0 = max(0, int(np.ceil((p[:, 1].min() - ys[0]) / dy - 1e-7)))
        iy1 = min(len(ys)-1, int(np.floor((p[:, 1].max() - ys[0]) / dy + 1e-7)))
        if ix1 < ix0 or iy1 < iy0:
            continue
        den = ((p[1,1]-p[2,1])*(p[0,0]-p[2,0])
               + (p[2,0]-p[1,0])*(p[0,1]-p[2,1]))
        if abs(den) < 1e-10:
            continue
        xx = xs[ix0:ix1+1][None, :]
        yy = ys[iy0:iy1+1][:, None]
        u = ((p[1,1]-p[2,1])*(xx-p[2,0]) + (p[2,0]-p[1,0])*(yy-p[2,1])) / den
        v = ((p[2,1]-p[0,1])*(xx-p[2,0]) + (p[0,0]-p[2,0])*(yy-p[2,1])) / den
        w = 1-u-v
        zz = u*p[0,2] + v*p[1,2] + w*p[2,2]
        slab = z[iy0:iy1+1, ix0:ix1+1]
        update = (u >= -1e-7) & (v >= -1e-7) & (w >= -1e-7) & (zz > slab)
        slab[update] = zz[update]
        if values is not None:
            cc = u*values[f[0]] + v*values[f[1]] + w*values[f[2]]
            colour[iy0:iy1+1, ix0:ix1+1][update] = cc[update]
    return z if values is None else (z, colour)


def fill_grid(z):
    valid = np.isfinite(z)
    if not valid.any():
        raise ValueError("front surface does not overlap atlas")
    inds = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return z[tuple(inds)], valid


def local(V, ax):
    h, a, b = nv.axial_coords(V, ax)
    return np.column_stack([a, h, b])


def atlas(P, F, xs, hs, sigma=10.):
    raw, valid = fill_grid(raster(P, F, xs, hs))
    fine = gaussian_filter(raw, 1.0 / STEP)
    broad = gaussian_filter(raw, sigma / STEP)
    return {"raw": raw, "fine": fine, "broad": broad,
            "relief": fine-broad, "valid": valid}


def sample(field, xs, hs, a, h):
    return RegularGridInterpolator((hs, xs), field, bounds_error=False,
                                  fill_value=0.)(np.column_stack([h, a]))


def fair_neck(V, F, weight, iterations=18):
    """Constrained Taubin fairing; frozen vertices remain exactly unchanged.

    Local refinement by itself is piecewise planar. This stage removes that
    inherited faceting before imposing the independent reference structure.
    """
    edges=mi.edge_records(F)[0]
    rows=np.r_[edges[:,0],edges[:,1]]
    cols=np.r_[edges[:,1],edges[:,0]]
    count=np.bincount(rows,minlength=len(V))
    A=sparse.coo_matrix((1./np.maximum(count[rows],1),(rows,cols)),shape=(len(V),len(V))).tocsr()
    result=V.copy()
    for _ in range(iterations):
        result+=.50*weight[:,None]*(A@result-result)
        result-=.53*weight[:,None]*(A@result-result)
    result[weight==0]=V[weight==0]
    return result


def geometry_check(V0, V1, F, immutable, boundary):
    rep0, rep = mi.topology_report(V0, F), mi.topology_report(V1, F)
    edges, _, _, _, _ = mi.edge_records(F)
    l0 = np.linalg.norm(V0[edges[:,0]] - V0[edges[:,1]], axis=1)
    l1 = np.linalg.norm(V1[edges[:,0]] - V1[edges[:,1]], axis=1)
    d = np.linalg.norm(V1-V0, axis=1)*1000
    tri0, tri1 = V0[F], V1[F]
    n0 = np.cross(tri0[:,1]-tri0[:,0], tri0[:,2]-tri0[:,0])
    n1 = np.cross(tri1[:,1]-tri1[:,0], tri1[:,2]-tri1[:,0])
    cos = np.einsum("ij,ij->i", n0, n1) / np.maximum(np.linalg.norm(n0,axis=1)*np.linalg.norm(n1,axis=1),1e-30)
    flips = int((cos <= 0).sum())
    angles = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    checks = {"finite": rep["finite"], "zero_normal_reversals": flips == 0,
              "zero_degenerate_faces": rep["zero_area_faces"] == 0,
              "zero_nonmanifold_edges": rep["nonmanifold_edges"] == 0,
              "consistent_shared_edges": rep["inconsistent_shared_edges"] == 0,
              "components_unchanged": rep["components"] == rep0["components"],
              "boundary_loops_unchanged": sorted(len(x) for x in nv._boundary_loops(F)) == [60,64],
              "boundary_vertices_exact": np.array_equal(V0[boundary],V1[boundary]),
              "identity_vertices_exact": np.array_equal(V0[immutable],V1[immutable]),
              "bounded_edge_growth": float((l1/np.maximum(l0,1e-12)).max()) < 1.8}
    return {"checks": checks, "passed": all(checks.values()), "topology": rep,
            "normal_reversals": flips, "max_normal_change_deg": float(angles.max()),
            "max_edge_growth": float((l1/np.maximum(l0,1e-12)).max()),
            "max_displacement_mm": float(d.max()), "moved_vertices": int((d>1e-6).sum()),
            "identity_vertices": int(immutable.sum()), "identity_max_change_mm": float(d[immutable].max()),
            "intersection_check": "normal reversal and degeneracy only; no global triangle-pair intersection test"}


def render_mesh(P,F,azimuth,xlim,ylim,res=650,flat=False):
    angle=np.radians(azimuth)
    cam=np.array([np.sin(angle),0.,np.cos(angle)])
    right=np.array([np.cos(angle),0.,-np.sin(angle)])
    up=np.array([0.,1.,0.])
    projected=np.column_stack([P@right, P@up, P@cam])
    light=.35*right+.55*up+cam
    light/=np.linalg.norm(light)
    # (lateral, up, anterior) is a reflected coordinate basis: reverse normals
    # for display only. Mesh winding in original camera coordinates is intact.
    normals=-vertex_normals(P,F)
    if flat:
        tri=P[F]; ns=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        ns/=-np.maximum(np.linalg.norm(ns,axis=1)[:,None],1e-20)
        P2=projected[F].reshape(-1,3)
        vals=np.repeat(.25+.70*np.clip(ns@light,0,1),3)
        F2=np.arange(len(P2)).reshape(-1,3)
    else:
        P2,F2,vals=projected,F,.25+.70*np.clip(normals@light,0,1)
    ys=np.linspace(*ylim,res)
    xs=np.linspace(*xlim,round(res*(xlim[1]-xlim[0])/(ylim[1]-ylim[0])))
    z,c=raster(P2,F2,xs,ys,vals)
    c[~np.isfinite(z)]=.10
    return c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--gain",type=float,default=1.0)
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():
        raise FileExistsError(f"use a new output directory: {out}")
    out.mkdir(parents=True)
    fit=RUN/"reconstruction/dense"
    refs=json.loads((fit/"reference_geometry.json").read_text(encoding="utf-8"))["references"]
    source_paths=[BASE/f"head_{s}_m.obj" for s in ("closed","open")]
    source_paths += [fit/"reference_geometry.json", fit/"parameters.npz",REF,
                     RUN/"delivery/neck_observed_closed_m.obj",BASE/"manifest.json"]
    before={str(p):sha(p) for p in source_paths}
    old={s:mi.load_obj(BASE/f"head_{s}_m.obj")[0] for s in ("closed","open")}
    V,F=mi.load_obj(BASE/"head_closed_m.obj")
    assert np.array_equal(F,mi.load_obj(BASE/"head_open_m.obj")[1])
    lm={s:np.asarray(refs[s]["stages"]["dense"]["landmarks_m"]) for s in old}
    frame=nv.head_frame(lm["closed"])
    ax=nv.axial_frame(V,F,frame)
    state_axes={s:nv.axial_frame(old[s],F,nv.head_frame(lm[s])) for s in old}
    hc=ax["h_chin_mm"]
    RV,RF=mi.load_obj(REF)
    menton=nv.detect_reference_menton(RV*1000)
    rchin=np.array([0.,menton["menton_y_mm"],0.])
    sel=(np.abs(RV[:,0]*1000)<3)&(np.abs(RV[:,1]*1000-rchin[1])<2)
    rchin[2]=(RV[sel,2]*1000).max()
    rax=nv.axial_frame(RV,RF,{"up":np.array([0.,1.,0.]),"right":np.array([1.,0.,0.]),
                              "forward":np.array([0.,0.,1.]),"chin":rchin/1000})
    rhc=rax["h_chin_mm"]
    audit=json.loads((BASE/"verification.json").read_text(encoding="utf-8"))["head_scale"]
    sx=audit["ratios"]["W_head_mm"]
    P=local(V,ax); RP0=local(RV,rax)
    # Use the subject and reference section centre curves to remove frame offsets.
    levels=np.asarray(nv.SECTION_LEVELS)
    tc=nv.centre_curve_from_sections(nv.measure_sections(V,F,ax,levels*hc,"target"),hc)
    rc=nv.centre_curve_from_sections(nv.measure_sections(RV,RF,rax,levels*rhc,"reference"),rhc)
    RP=RP0.copy()
    RP[:,1]*=hc/rhc
    rca,rcb=rc(RP0[:,1]); tca,tcb=tc(RP[:,1])
    RP[:,0]=(RP0[:,0]-rca)*sx+tca
    RP[:,2]=(RP0[:,2]-rcb)*sx+tcb
    xs=np.arange(-85.,85.+STEP/2,STEP)
    hs=np.arange(0.,hc+10.+STEP/2,STEP)
    ref_at=atlas(RP,RF,xs,hs)
    tgt_at=atlas(P,F,xs,hs)
    # No manual anatomy bumps: retain the signed 2-D reference residual.
    replacement=tgt_at["broad"]+args.gain*ref_at["relief"]
    correction=replacement-tgt_at["raw"]
    valid=ref_at["valid"]&tgt_at["valid"]
    distance=distance_transform_edt(valid)*STEP
    valid_ramp=nv._ramp(distance,1.,7.)
    correction*=valid_ramp

    # Quantify the old patch hold independently on the unedited dense mesh.
    with np.load(fit/"parameters.npz",allow_pickle=False) as z:
        dv=z["dense_vertices"][refs["closed"]["stages"]["dense"]["frame_index"]]
    dP=local(dv,ax); ca,cb=tc(dP[:,1])
    theta=np.arctan2(dP[:,0]-ca,dP[:,2]-cb)
    OV,OF=mi.load_obj(RUN/"delivery/neck_observed_closed_m.obj")
    hold=nv.measured_region_hold(dv,ax,theta,{"points_mm":OV*1000,"hold":0.,"h_pad_mm":5.,"a_pad_mm":5.})
    op=local(OV,ax)
    front=(np.abs(theta)<.6)&(dP[:,1]>.35*hc)&(dP[:,1]<.85*hc)
    old_hold={"patch_h_range_mm":[float(op[:,1].min()),float(op[:,1].max())],
              "patch_t_range":[float(op[:,1].min()/hc),float(op[:,1].max()/hc)],
              "zero_hold_front_vertices":int(((hold<1e-6)&front).sum()),
              "front_vertices_in_band":int(front.sum())}

    # Conforming LOCAL midpoint subdivision. Existing face/identity vertices
    # are retained; no Loop smoothing, no change to the face surface.
    orig_n=len(V)
    refinement=[]
    for iteration in range(2):
        pp=local(old["closed"],ax)
        centers=pp[F].mean(axis=1)
        ns=vertex_normals(old["closed"],F)@ax["anterior"]
        red=(centers[:,1]>8)&(pp[F,1].max(axis=1)<hc-2)
        sub=n3.subdivide_local(old["closed"],F,face_mask=red)
        ed=np.asarray(sub["midpoint_edges"],dtype=int)
        for s in old:
            old[s]=np.vstack([old[s],old[s][ed].mean(axis=1)])
        F=sub["faces"]
        refinement.append({"level":iteration+1,"added_vertices":len(ed),"faces":len(F)})
    original_refined={s:vs.copy() for s,vs in old.items()}
    V=old["closed"]
    P=local(V,ax)
    smoothing_weight=nv._ramp(P[:,1],3.,18.)
    # Protect the UNION of both states' own jaw/identity regions. A plane from
    # the closed camera cannot be used to label the open camera's face.
    for s,vs in old.items():
        sh,_,_=nv.axial_coords(vs,state_axes[s])
        sch=state_axes[s]["h_chin_mm"]
        smoothing_weight*=1.-nv._ramp(sh,sch-14.,sch-3.)
    boundary=np.unique(mi.edge_records(F)[0][mi.edge_records(F)[2]==1])
    smoothing_weight[boundary]=0.
    for s in old:
        old[s]=fair_neck(old[s],F,smoothing_weight)
    V=old["closed"]
    P=local(V,ax)
    ca,cb=tc(P[:,1]);theta=np.arctan2(P[:,0]-ca,P[:,2]-cb)
    t=P[:,1]/hc
    window=(nv._ramp(t,.18,.34)*(1.-nv._ramp(t,.83,.99))
            *(1.-nv._ramp(np.abs(theta),.78,1.38)))
    window*=smoothing_weight
    window[boundary]=0.
    smoothed_at=atlas(P,F,xs,hs)
    correction=(tgt_at["broad"]+args.gain*ref_at["relief"]-smoothed_at["raw"])*valid_ramp
    delta=sample(correction,xs,hs,P[:,0],P[:,1])*window
    # Front-sheet gating avoids moving any inner mouth or posterior sheet.
    front_b=sample(smoothed_at["raw"],xs,hs,P[:,0],P[:,1])
    on_front=np.abs(P[:,2]-front_b)<3.
    delta*=on_front
    delta=np.clip(delta,-8.,12.)
    dclosed=delta[:,None]*ax["anterior"][None,:]/1000
    immutable=smoothing_weight==0
    closed_basis=np.column_stack([frame[k] for k in ("right","up","forward")])
    final={}; reports={}
    for s,V0 in old.items():
        if s=="closed":
            displacement=dclosed
        else:
            sf=nv.head_frame(lm[s])
            sb=np.column_stack([sf[k] for k in ("right","up","forward")])
            displacement=dclosed@(closed_basis@sb.T)
        V1=V0+displacement
        final[s]=V1
        reports[s]=geometry_check(original_refined[s],V1,F,immutable,boundary)
        path=out/f"head_{s}_m.obj"
        mi.export_obj(path,V1,F)
        back,bf=mi.load_obj(path)
        reports[s]["roundtrip_max_error_mm"]=float(np.abs(back-V1).max()*1000)
        reports[s]["roundtrip_faces_identical"]=np.array_equal(bf,F)
        orig,_=mi.load_obj(BASE/f"head_{s}_m.obj")
        h0=local(orig,state_axes[s])[:,1]
        face=h0>=state_axes[s]["h_chin_mm"]-3.
        reports[s]["original_face_vertex_max_change_mm"]=float(np.linalg.norm((back[:orig_n]-orig)[face],axis=1).max()*1000)
    print(json.dumps({"phase":"geometry","states":{s:{k:r[k] for k in ("passed","moved_vertices","max_displacement_mm","max_edge_growth","normal_reversals","original_face_vertex_max_change_mm")} for s,r in reports.items()}}),flush=True)

    # Actual post-export surface profiles; never score the design field alone.
    actual=atlas(local(final["closed"],ax),F,xs,hs)
    xx=np.argmin(np.abs(xs))
    band=(hs>.25*hc)&(hs<.88*hc)
    profile={}
    for name,A in (("reference",ref_at),("current",tgt_at),("candidate",actual)):
        curve=A["relief"][:,xx]
        peaks,props=find_peaks(curve,prominence=.15,distance=int(8/STEP))
        peaks=peaks[band[peaks]]
        profile[name]={"peaks":[{"t":float(hs[i]/hc),"h_mm":float(hs[i]),"relief_mm":float(curve[i])} for i in peaks]}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for name,A,c in (("reference",ref_at,"#888888"),("current",tgt_at,"#d77a2b"),("candidate",actual,"#1679b5")):
        axes[0].plot(A["raw"][:,xx],hs/hc,label=name,color=c)
        axes[1].plot(A["relief"][:,xx],hs/hc,label=name,color=c)
    for axis in axes:
        axis.set_ylim(.12,1.01);axis.grid(alpha=.25);axis.legend();axis.set_ylabel("h / chin height (bottom to top)")
    axes[0].set_xlabel("anterior depth (mm)");axes[0].set_title("Exported midline front contour")
    axes[1].set_xlabel("signed relief above 10 mm background (mm)");axes[1].set_title("Exported structural relief")
    fig.tight_layout();fig.savefig(out/"profiles.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for axis,(name,A) in zip(axes,[("reference",ref_at),("current",tgt_at),("candidate",actual)]):
        im=axis.imshow(A["relief"],origin="lower",extent=[xs[0],xs[-1],hs[0],hs[-1]],vmin=-6,vmax=6,cmap="coolwarm",aspect="equal")
        axis.set_xlim(-60,60);axis.set_ylim(10,hc);axis.set_title(name);axis.set_xlabel("lateral (mm)")
    fig.colorbar(im,ax=axes.ravel().tolist(),label="signed relief (mm)",shrink=.7)
    fig.savefig(out/"relief_maps.png",dpi=160);plt.close(fig)
    for angle,label in [(0,"front"),(45,"oblique"),(90,"side")]:
        fig,axes=plt.subplots(1,3,figsize=(12,5),facecolor="#191919")
        for axis,(name,pp,ff) in zip(axes,[("Current",local(original_refined["closed"],ax),F),("Reference (adapted)",RP,RF),("Candidate",local(final["closed"],ax),F)]):
            # Shared depth offset for comparable camera framing in side views.
            shift=float(tc(.55*hc)[1]); qq=pp.copy();qq[:,2]-=shift
            img=render_mesh(qq,ff,angle,(-86,86),(-5,hc+28))
            axis.imshow(img,origin="lower",cmap="gray",vmin=0,vmax=1);axis.axis("off");axis.set_title(name,color="white")
        fig.tight_layout();fig.savefig(out/f"compare_{label}.png",dpi=150,facecolor=fig.get_facecolor());plt.close(fig)
        print(json.dumps({"phase":"render","view":label}),flush=True)
    # Both expression states and flat shading are also made available.
    for s in ("closed","open"):
        fig,axes=plt.subplots(1,2,figsize=(9,5),facecolor="#191919")
        for axis,flat in zip(axes,(False,True)):
            pp=local(final[s],ax); pp[:,2]-=float(tc(.55*hc)[1])
            axis.imshow(render_mesh(pp,F,35,(-78,78),(-5,hc+28),flat=flat),origin="lower",cmap="gray",vmin=0,vmax=1)
            axis.axis("off");axis.set_title(f"{s} / {'flat' if flat else 'smooth'}",color="white")
        fig.tight_layout();fig.savefig(out/f"verify_{s}.png",dpi=150,facecolor=fig.get_facecolor());plt.close(fig)
    after={str(p):sha(p) for p in source_paths}
    checks={"geometry":all(r["passed"] for r in reports.values()),
            "obj_roundtrip":all(r["roundtrip_faces_identical"] and r["roundtrip_max_error_mm"]<1e-6 for r in reports.values()),
            "face_unchanged":all(r["original_face_vertex_max_change_mm"]<1e-7 for r in reports.values()),
            "inputs_unchanged":before==after}
    dump(out/"verification.json",{"checks":checks,"states":reports,"source_sha256":before,
          "refinement":refinement,"old_hold_audit":old_hold,"profiles":profile,
          "target_chin_height_mm":hc,"reference_chin_height_mm":rhc,"width_scale":sx})
    dump(out/"manifest.json",{"status":"geometry_passed_candidate" if all(checks.values()) else "failed",
          "source":BASE,"reference":REF,"algorithm":"signed_2d_reference_relief_transfer",
          "mesh_unit":"m","editing_mode":True,"subject_observation_matched":False,
          "parameters":{"gain":args.gain,"broad_sigma_mm":10.,"fine_sigma_mm":1.,"atlas_step_mm":STEP,"neck_fairing":"18 constrained Taubin iterations; 0.50/-0.53"},
          "checks":checks,"topology_identical_to_input":False,"closed_open_shared_topology":True,
          "note":"Reference-inspired external skin relief; peak identities are design labels, not observed cartilage segmentation. Global self intersections not checked."})
    np.savez_compressed(out/"structure_fields.npz",xs=xs,hs=hs,reference_relief=ref_at["relief"],current_relief=tgt_at["relief"],candidate_relief=actual["relief"],correction_mm=delta,faces=F)
    print(json.dumps({"status":"candidate_ready" if all(checks.values()) else "rejected","checks":checks,"output":str(out),"hold_audit":old_hold,"profiles":profile},ensure_ascii=False),flush=True)
    return 0 if all(checks.values()) else 2


if __name__=="__main__":
    raise SystemExit(main())
