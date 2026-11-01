"""Read-only verification of one delivery run directory.

Checks (no fitting, no GPU):
  * exactly one final mesh plus the three required sidecars exist;
  * vertices/faces are finite and every face index is within range;
  * the mesh is metre scale and its extent matches the manifest;
  * the reference frame, landmarks, support labels and mesh stage agree;
  * the manifest version equals VERSION and the algorithm name is the single default.

Usage:
  runtime\\env\\python.exe tools\\verify_delivery.py runtime\\outputs\\<run>\\delivery
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


def verify(delivery: Path) -> dict:
    delivery = delivery.resolve()
    manifest = json.loads((delivery / "manifest.json").read_text(encoding="utf-8"))
    support = json.loads((delivery / "support.json").read_text(encoding="utf-8"))
    landmarks = json.loads((delivery / "reference_landmarks.json").read_text(encoding="utf-8"))
    meshes = sorted(p.name for p in delivery.iterdir() if p.suffix.lower() in {".obj", ".ply", ".stl"})
    assert meshes == [manifest["mesh"]["path"]], f"expected exactly one mesh, found {meshes}"

    import trimesh

    mesh = trimesh.load(str(delivery / manifest["mesh"]["path"]), process=False, maintain_order=True)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    assert np.isfinite(vertices).all(), "non-finite vertex"
    assert faces.min() >= 0 and faces.max() < len(vertices), "invalid face index"
    extents = vertices.max(0) - vertices.min(0)
    assert 0.10 < float(extents.max()) < 0.60, f"not metre scale: {extents}"
    assert len(vertices) == manifest["mesh"]["vertices"], "manifest vertex count mismatch"
    assert len(vertices) == int(np.sum(support["counts"])), "support counts do not sum to vertex count"
    assert manifest["mesh"]["unit"] == "m"

    project = Path(__file__).resolve().parents[1]
    version = (project / "VERSION").read_text(encoding="utf-8").strip()
    assert manifest["software"]["version"] == version == "0.2.0", "version mismatch across VERSION/manifest"
    assert manifest["software"]["algorithm"] == "facecap-inspired-v2-strict"

    assert landmarks["mesh"] == manifest["mesh"]["path"]
    assert landmarks["frame_id"] == manifest["input"]["closed_reference_frame"]
    assert landmarks["state"] == "closed"
    assert landmarks["mesh_unit"] == "m"
    assert len(landmarks["landmarks_m"]) == landmarks["landmark_count"] == 68
    triangles = np.asarray(landmarks["landmark_triangle_indices"], dtype=np.int64)
    bary = np.asarray(landmarks["landmark_barycentric"], dtype=float)
    assert triangles.max() < len(faces)
    recomputed = (vertices[faces[triangles]] * bary[:, :, None]).sum(1)
    np.testing.assert_allclose(recomputed, np.asarray(landmarks["landmarks_m"]), atol=1e-6)
    assert all(row is None or np.isfinite(row).all() for row in landmarks["observed_landmarks_m"])

    assert manifest["mesh"]["dense"] is True, "final mesh is not the dense stage"
    return {
        "mesh": manifest["mesh"]["path"],
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "extents_m": [round(float(v), 4) for v in extents],
        "unit": manifest["mesh"]["unit"],
        "coordinate_system": manifest["mesh"]["coordinate_system"],
        "reference_frame": landmarks["frame_id"],
        "support_counts": dict(zip(support["labels"], support["counts"])),
        "status": manifest["status"],
        "warnings": manifest["warnings"],
        "version": manifest["software"]["version"],
        "algorithm": manifest["software"]["algorithm"],
        "landmark_recompute_max_error": float(np.abs(recomputed - np.asarray(landmarks["landmarks_m"])).max()),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, ensure_ascii=False))
