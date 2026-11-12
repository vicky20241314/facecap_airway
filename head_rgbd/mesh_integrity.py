"""OBJ geometry checks and conservative winding repair (no vertex movement)."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def load_obj(path):
    vertices, faces = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == "f":
            indices = [int(x.split("/")[0]) for x in fields[1:]]
            indices = [x - 1 if x > 0 else len(vertices) + x for x in indices]
            faces.extend((indices[0], indices[i], indices[i + 1])
                         for i in range(1, len(indices) - 1))
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def edge_records(faces):
    faces = np.asarray(faces, dtype=np.int64)
    directed = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                      return_inverse=True, return_counts=True)
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    balance = np.bincount(inverse, weights=signs, minlength=len(edges))
    return edges, inverse, counts, signs, balance


def topology_report(vertices, faces):
    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=np.int64)
    edges, _, counts, _, balance = edge_records(faces)
    triangle = vertices[faces]
    cross = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    area_mm2 = np.linalg.norm(cross, axis=1) * 5e5
    graph = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                              shape=(len(vertices), len(vertices))).tocsr()
    n_components, labels = connected_components(graph, directed=False)
    boundary = edges[counts == 1]
    groups = []
    if len(boundary):
        ids = np.unique(boundary)
        bgraph = sparse.coo_matrix((np.ones(len(boundary)), (boundary[:, 0], boundary[:, 1])),
                                   shape=(len(vertices), len(vertices))).tocsr()[ids][:, ids]
        number, blabels = connected_components(bgraph, directed=False)
        for i in range(number):
            selected = ids[blabels == i]
            groups.append({"vertices": int(len(selected)),
                           "centroid_m": vertices[selected].mean(axis=0).tolist()})
    return {
        "vertices": int(len(vertices)), "faces": int(len(faces)),
        "finite": bool(np.isfinite(vertices).all()),
        "components": int(n_components),
        "component_sizes": np.bincount(labels).tolist(),
        "nonmanifold_edges": int((counts > 2).sum()),
        "inconsistent_shared_edges": int(((counts == 2) & (balance != 0)).sum()),
        "boundary_edges": int(len(boundary)), "boundary_groups": groups,
        "zero_area_faces": int((area_mm2 < 1e-8).sum()),
        "small_faces_below_0_01_mm2": int((area_mm2 < .01).sum()),
        "min_area_mm2": float(area_mm2.min()) if len(area_mm2) else None,
    }


def orient_faces_consistently(faces):
    """Preserve the majority orientation of each component; reject nonmanifolds."""
    faces = np.asarray(faces, dtype=np.int64)
    _, inverse, counts, signs, _ = edge_records(faces)
    if np.any(counts > 2):
        raise ValueError("cannot orient a nonmanifold mesh")
    face_ids = np.tile(np.arange(len(faces)), 3)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    adjacency = [[] for _ in range(len(faces))]
    for start in starts[counts == 2]:
        a, b = order[start:start + 2]
        i, j = int(face_ids[a]), int(face_ids[b])
        opposite_flip = bool(signs[a] == signs[b])
        adjacency[i].append((j, opposite_flip))
        adjacency[j].append((i, opposite_flip))
    parity = np.full(len(faces), -1, dtype=np.int8)
    for seed in range(len(faces)):
        if parity[seed] >= 0:
            continue
        parity[seed] = 0
        queue, component = deque([seed]), []
        while queue:
            i = queue.popleft()
            component.append(i)
            for j, toggle in adjacency[i]:
                required = parity[i] ^ int(toggle)
                if parity[j] < 0:
                    parity[j] = required
                    queue.append(j)
                elif parity[j] != required:
                    raise ValueError("non-orientable face adjacency")
        if parity[component].sum() > len(component) / 2:
            parity[component] ^= 1
    result = faces.copy()
    changed = np.flatnonzero(parity == 1)
    result[changed] = result[changed][:, [0, 2, 1]]
    return result, {"reversed_faces": int(len(changed)), "face_ids": changed.tolist(),
                    "vertex_movement_mm": 0.0}


def export_obj(path, vertices, faces):
    """Write fresh area-weighted vertex normals, no material/colour overrides."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vertex_normal = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(vertex_normal, faces[:, corner], normal)
    length = np.linalg.norm(vertex_normal, axis=1)
    vertex_normal /= np.maximum(length[:, None], 1e-30)
    lines = ["# facecap: consistent winding, recomputed vertex normals; units=m\n"]
    lines.extend("v %.10f %.10f %.10f\n" % tuple(p) for p in vertices)
    lines.extend("vn %.10f %.10f %.10f\n" % tuple(n) for n in vertex_normal)
    lines.extend("f %d//%d %d//%d %d//%d\n" %
                 (a + 1, a + 1, b + 1, b + 1, c + 1, c + 1) for a, b, c in faces)
    path.write_text("".join(lines), encoding="utf-8")
    return path
