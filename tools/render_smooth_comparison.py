"""Flat shaded geometry only: identical orthographic views, lighting and material."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.validate_smooth_fast import PREPARED, REFERENCE, load_npz


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    entries = json.loads((PREPARED / 'sequence.json').read_text())['frames']
    refs = json.loads((REFERENCE / 'reconstruction/dense/report.json').read_text())['references']
    fig, axes = plt.subplots(len(a.run)*2, 3, figsize=(12, len(a.run)*6), squeeze=False)
    light = np.array([-.25, .5, 1.]); light /= np.linalg.norm(light)
    for row, run in enumerate(a.run):
        z = load_npz(run / 'reconstruction/dense/parameters.npz')
        for s, state in enumerate(('closed', 'open')):
            i = next(i for i, e in enumerate(entries) if e['id'] == refs[state]['frame_id'])
            canonical = (z['dense_vertices'][i] - z['t'][i]) @ z['R'][i]
            for col, yaw in enumerate((0, -55, 55)):
                angle = np.radians(yaw)
                rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                                [-np.sin(angle), 0, np.cos(angle)]])
                v = canonical @ rot.T * 1000
                tri = v[z['dense_faces']]
                normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
                normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-20)
                visible = normal[:, 2] > 0
                tri, normal = tri[visible], normal[visible]
                order = tri[:, :, 2].mean(1).argsort()
                intensity = .22 + .78*np.clip(normal @ light, 0, 1)
                color = intensity[:, None] * np.array([[.79, .83, .88]])
                ax = axes[2*row+s, col]
                ax.add_collection(PolyCollection(tri[order, :, :2], facecolors=color[order],
                                                 edgecolors='none', antialiased=False))
                ax.set(xlim=(-110, 110), ylim=(-125, 140), aspect='equal')
                ax.axis('off')
                ax.set_title(f'{run.name} | {state} | yaw {yaw}', fontsize=9)
    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=160, facecolor='white')
    a.output.with_suffix('.json').write_text(json.dumps(dict(
        runs=[str(r) for r in a.run], reference_frames=refs, yaw_deg=[0, -55, 55],
        material='same matte gray; face-normal flat shading; no texture or smooth normal interpolation',
        light=light.tolist(), bounds_mm=[[-110,110],[-125,140]],
        coordinates='camera rigid inverse R,t only; no shape alignment or smoothing'), indent=2))


if __name__ == '__main__':
    main()
