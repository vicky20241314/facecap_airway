"""Reproducible geometry checks on a frozen reference domain, outside run timing."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from head_rgbd.fit_b import normals
from head_rgbd.fit_d import load_npz
from head_rgbd.geometry_d import Settings, edges_of, subdivide
from head_rgbd.surface_v2 import loop_mapping, safe_loop_base

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runtime/validation/default_smooth_fast'
REFERENCE = ROOT / 'runtime/outputs/joint_fast_v3'
PREPARED = ROOT / 'runtime/outputs/zhoupei_cap/prepared'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def bind_prepared(run):
    dest = run / 'prepared'
    dest.mkdir(exist_ok=True)
    for p in PREPARED.iterdir():
        if p.is_file() and not (dest / p.name).exists():
            import os
            os.link(p, dest / p.name)


def regenerate(run, output, minimum_support=2):
    """No identity/field fit: reuse the exact saved controls and evidence mask."""
    source = run / 'reconstruction/dense'
    z = load_npz(source / 'parameters.npz')
    report = json.loads((source / 'report.json').read_text())
    old = report['field_logs']['front_anchor_regularized']['settings']
    keys = {f.name for f in fields(Settings)}
    settings = Settings(**{**{k: v for k, v in old.items() if k in keys}, 'loop_target_blend': .5})
    f, P, _, _ = subdivide(z['faces'], z['posed_vertices'].shape[1])
    support = load_npz(source / 'support_regularized.npz')['refinement_frame_mask']
    _, _, S, _, _, _ = loop_mapping(z['faces'], z['posed_vertices'].shape[1],
                                   z['anatomical_excluded'] | (support.sum(0) < minimum_support))
    S = S.tolil()
    for vertex in np.flatnonzero(z['dense_anatomical_excluded']):
        row = P.getrow(vertex)
        S.rows[vertex], S.data[vertex] = row.indices.tolist(), row.data.tolist()
    direction = np.array([normals(v, z['faces']) for v in z['posed_vertices']])
    d = np.array([P @ n for n in direction])
    d /= np.maximum(np.linalg.norm(d, axis=2, keepdims=True), 1e-12)
    _, v, log = safe_loop_base(P, S.tocsr(), z['regularized_vertices'], z['dense_base_vertices'],
                              d, f, P @ (z['deviation_m'] * 1000), settings)
    dest = output / 'reconstruction/dense'
    dest.mkdir(parents=True, exist_ok=False)
    z['dense_vertices'] = v
    np.savez_compressed(dest / 'parameters.npz', **z)
    write(output / 'regeneration.json', dict(source=str(source), loop=log,
          source_sha256=hashlib.sha256((source / 'parameters.npz').read_bytes()).hexdigest(),
          minimum_loop_support_frames=minimum_support,
          field_recomputed=False, identity_recomputed=False))
    bind_prepared(output)
    print(json.dumps(log, indent=2))


def domain():
    path = WORK / 'evaluation_domain.npz'
    if path.exists():
        return load_npz(path)
    z = load_npz(REFERENCE / 'reconstruction/dense/parameters.npz')
    ev = load_npz(REFERENCE / 'reconstruction/dense/support_dense.npz')
    # The reference alone fixes the domain BEFORE any candidate is scored.
    # Repeat real observations; remove mouth/cavity/eye and anatomical boundaries.
    keep = (ev['observed_frames'] >= 2) & ~z['dense_anatomical_excluded']
    faces = z['dense_faces']
    face_keep = keep[faces].all(1)
    mesh = trimesh.Trimesh(z['dense_vertices'][0], faces, process=False)
    adjacency = mesh.face_adjacency
    adjacency = adjacency[face_keep[adjacency].all(1)]
    np.savez_compressed(path, faces=faces, adjacency=adjacency, vertex_mask=keep,
                        face_mask=face_keep)
    write(WORK / 'evaluation_domain.json', dict(
        source=str(REFERENCE), source_parameters_sha256=hashlib.sha256(
            (REFERENCE / 'reconstruction/dense/parameters.npz').read_bytes()).hexdigest(),
        rule='baseline observed_frames >= 2 AND NOT baseline dense_anatomical_excluded; '
             'both faces and all vertices of each adjacent pair must satisfy this fixed mask',
        vertices=int(keep.sum()), faces=int(face_keep.sum()), pairs=len(adjacency),
        weighting='one vote per shared edge; degrees; all 20 same-frame posed states',
        domain_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return load_npz(path)


def smooth(run):
    z = load_npz(run / 'reconstruction/dense/parameters.npz')
    dom = domain()
    np.testing.assert_array_equal(z['dense_faces'], dom['faces'])
    seq = json.loads((PREPARED / 'sequence.json').read_text())
    entries = [e for e in seq['frames'] if e['status'] == 'candidate']
    ref_z = load_npz(REFERENCE / 'reconstruction/dense/parameters.npz')
    refs = json.loads((REFERENCE / 'reconstruction/dense/reference_geometry.json').read_text())['references']
    ri = next(i for i, e in enumerate(entries) if e['id'] == refs['closed']['frame_id'])
    canonical = (ref_z['dense_vertices'][ri] - ref_z['t'][ri]) @ ref_z['R'][ri]
    lm = (np.asarray(refs['closed']['stages']['dense']['landmarks_m']) - ref_z['t'][ri]) @ ref_z['R'][ri]
    center = canonical[dom['faces']].mean(1)[dom['adjacency']].mean(1)
    regions = dict(forehead=center[:, 1] > lm[17:27, 1].mean(),
                   jaw=center[:, 1] < lm[48:60, 1].mean())
    regions['cheeks'] = (~regions['forehead'] & ~regions['jaw'] &
                         (abs(center[:, 0]) > np.max(abs(lm[31:36, 0]))))
    values = []
    for i, v in enumerate(z['dense_vertices']):
        t = v[dom['faces']]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
        a, b = dom['adjacency'].T
        angle = np.degrees(np.arccos(np.clip((n[a] * n[b]).sum(1), -1, 1)))
        values.append(dict(frame=entries[i]['id'], state=entries[i]['state'],
                           rms_deg=float(np.sqrt(np.mean(angle**2))),
                           p95_deg=float(np.percentile(angle, 95)),
                           regions={name: dict(pairs=int(mask.sum()),
                               rms_deg=float(np.sqrt(np.mean(angle[mask]**2))),
                               p95_deg=float(np.percentile(angle[mask], 95)))
                               for name, mask in regions.items()}))
    ed = edges_of(z['faces'])
    field = z['deviation_m'] * 1000
    jump = abs(field[ed[:, 0]] - field[ed[:, 1]])
    # A local spike is departure from the one-ring mean, distinct from a broad correction.
    A = sparse.coo_matrix((np.ones(2*len(ed)), (ed.ravel(), ed[:, ::-1].ravel())),
                          shape=(len(field), len(field))).tocsr()
    spike = abs(field - (A @ field) / np.maximum(np.asarray(A.sum(1)).ravel(), 1))
    def stats(x):
        return dict(rms_mm=float(np.sqrt(np.mean(x*x))), p95_mm=float(np.percentile(x, 95)),
                    max_mm=float(x.max()))
    return dict(run=str(run), frames=values, field_max_mm=float(abs(field).max()),
                edge_difference=stats(jump), one_ring_spike=stats(spike),
                supported_edge_difference=stats(jump[dom['vertex_mask'][:len(field)][ed].all(1)]),
                supported_one_ring_spike=stats(spike[dom['vertex_mask'][:len(field)]]),
                local_region_definition='fixed baseline canonical coordinates: forehead above mean brow y; '
                    'jaw below outer lip y; cheeks between these y and lateral to nose alae')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('regenerate')
    r.add_argument('--run', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--minimum-support', type=int, default=2)
    s = sub.add_parser('smooth')
    s.add_argument('--run', type=Path, action='append', required=True)
    s.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.command == 'regenerate':
        regenerate(a.run, a.output, a.minimum_support)
    else:
        result = [smooth(run) for run in a.run]
        write(a.output, result)
        for entry in result:
            for state in ('closed', 'open'):
                f = [f for f in entry['frames'] if f['state'] == state]
                print(Path(entry['run']).name, state,
                      'RMS', np.mean([x['rms_deg'] for x in f]),
                      'P95', np.mean([x['p95_deg'] for x in f]), entry['edge_difference'])


if __name__ == '__main__':
    main()
