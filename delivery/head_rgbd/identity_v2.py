"""Low-frequency identity pass for alternating fixed-shape motion refinement."""
import argparse,json,time
from pathlib import Path
import numpy as np
from head_rgbd.fit_d import load_npz
from head_rgbd.fit_b import normals
from head_rgbd.geometry_d import Settings,solve_field,anatomical_exclusion,surface_state
from head_rgbd.geometry_c import front_protected_vertices
from head_rgbd.reporting import write_json
from head_rgbd.audit import sha


def run(sequence,initial,output,fixed_ray_policy='strict-coverage'):
    start=time.perf_counter();output.mkdir(parents=True,exist_ok=False)
    source=json.loads((initial/'report.json').read_text(encoding='utf-8'))
    if source['sequence_sha256']!=sha(sequence):raise ValueError('sequence mismatch')
    info=json.loads(sequence.read_text(encoding='utf-8'));rows=source['frames'];p=load_npz(initial/'parameters.npz')
    data=[load_npz(sequence.parent/next(r['file'] for r in info['frames'] if r['id']==row['id'])) for row in rows]
    base=p['posed_vertices'].astype(float);f=p['faces'];direction=np.array([normals(v,f) for v in base])
    identity=[int(i) for i in p['identity_selected']];front=[i for i,row in enumerate(rows) if row['view']=='front']
    ids=[i for i in identity if rows[i]['view']=='front'];excluded=anatomical_exclusion(base,data,front,p['mouth_identity_excluded'])
    s=Settings(solver='qp',research_v2=True,points=5000,outer_steps=2,gradient_length_mm=8.,bending_length_mm=8.,
               noise_floor_mm=1/np.sqrt(12),displacement_bound_mm=2.5,fixed_ray_policy=fixed_ray_policy)
    anchor,_,log=solve_field(base,f,direction,data,rows,ids,excluded,s,stage='alternating_low_frequency_front')
    protected=front_protected_vertices(surface_state(base,direction,anchor),f,data,front)
    field,support,side_log=solve_field(base,f,direction,data,rows,identity,excluded,s,initial=anchor,locked=protected,stage='alternating_guarded_low_frequency')
    np.savez_compressed(output/'identity_field.npz',deviation_m=field*.001,posed_vertices=base,faces=f,
        regularized_vertices=surface_state(base,direction,field),identity_selected=identity,front_anchor_deviation_m=anchor*.001)
    write_json(output/'report.json',dict(sequence_sha256=sha(sequence),initial=str(initial),elapsed_s=time.perf_counter()-start,
        stages=[log,side_log],policy='shared low frequency field, guarded original front; passed to next motion correction as FIXED identity'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sequence','initial-fit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fixed-ray-policy',choices=('strict-coverage','accounted-research'),default='strict-coverage')
    a=p.parse_args();run(a.sequence.resolve(),a.initial_fit.resolve(),a.output.resolve(),a.fixed_ray_policy)
