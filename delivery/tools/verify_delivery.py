"""Read-only verification of one delivery run directory.

Checks (no fitting, no GPU):
  * exactly one closed mesh and one open mesh, plus the three required sidecars;
  * both meshes are finite, share one dense topology, and use valid face indices;
  * both are metre scale and their extents match the manifest;
  * each state's reference frame, 68 landmarks and rigid transform agree, and the
    landmarks can be recomputed from that state's own mesh;
  * closed/open are genuinely different states (different reference frame and
    different geometry) rather than one mesh exported twice;
  * support labels and counts cover every vertex of the shared topology;
  * the manifest version equals VERSION and the algorithm name is the single default.

Usage:
  runtime\\env\\python.exe tools\\verify_delivery.py runtime\\outputs\\<run>\\delivery
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


def _mesh(delivery: Path, name: str):
    import trimesh

    mesh = trimesh.load(str(delivery / name), process=False, maintain_order=True)
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64)


def verify(delivery: Path) -> dict:
    delivery = delivery.resolve()
    manifest = json.loads((delivery / "manifest.json").read_text(encoding="utf-8"))
    support = json.loads((delivery / "support.json").read_text(encoding="utf-8"))
    landmarks = json.loads((delivery / "reference_landmarks.json").read_text(encoding="utf-8"))

    mesh_payload = manifest["mesh"]
    states = ["closed", "open"]
    expected = sorted(mesh_payload["states"][state]["path"] for state in states)
    found = sorted(p.name for p in delivery.iterdir() if p.suffix.lower() in {".obj", ".ply", ".stl"})
    assert found == expected, f"expected exactly the closed/open meshes {expected}, found {found}"

    geometry = {}
    topology = None
    for state in states:
        item = mesh_payload["states"][state]
        vertices, faces = _mesh(delivery, item["path"])
        assert np.isfinite(vertices).all(), f"{state}: non-finite vertex"
        assert faces.min() >= 0 and faces.max() < len(vertices), f"{state}: invalid face index"
        extents = vertices.max(0) - vertices.min(0)
        assert 0.10 < float(extents.max()) < 0.60, f"{state}: not metre scale: {extents}"
        assert len(vertices) == item["vertices"], f"{state}: manifest vertex count mismatch"
        assert np.allclose(extents, item["extents_m"], atol=1e-6), f"{state}: extent mismatch"
        if topology is None:
            topology = faces.copy()
        else:
            assert np.array_equal(topology, faces), "closed and open meshes do not share one topology"
        geometry[state] = {"vertices": vertices, "faces": faces, "extents": extents}

    assert mesh_payload["unit"] == "m"
    assert mesh_payload.get("topology_shared") is True
    assert len(topology) == mesh_payload["states"]["closed"]["faces"] == mesh_payload["states"]["open"]["faces"]
    assert len(geometry["closed"]["vertices"]) == int(np.sum(support["counts"])), (
        "support counts do not sum to vertex count"
    )

    closed = geometry["closed"]["vertices"]
    opened = geometry["open"]["vertices"]
    per_vertex = np.linalg.norm(closed - opened, axis=1) * 1000.0
    assert float(per_vertex.max()) > 0.1, "closed and open meshes are identical; the open state is missing"
    assert (
        mesh_payload["states"]["closed"]["reference_frame"]
        != mesh_payload["states"]["open"]["reference_frame"]
    )

    project = Path(__file__).resolve().parents[1]
    version = (project / "VERSION").read_text(encoding="utf-8").strip()
    assert manifest["software"]["version"] == version == "0.3.0", "version mismatch across VERSION/manifest"
    assert manifest["software"]["algorithm"] == "facecap-inspired-v2-strict"

    state_report = {}
    for state in states:
        payload = landmarks["states"][state]
        item = mesh_payload["states"][state]
        assert payload["mesh"] == item["path"]
        assert payload["frame_id"] == item["reference_frame"]
        assert payload["mesh_unit"] == "m"
        assert len(payload["landmarks_m"]) == payload["landmark_count"] == 68
        triangles = np.asarray(payload["landmark_triangle_indices"], dtype=np.int64)
        bary = np.asarray(payload["landmark_barycentric"], dtype=float)
        assert triangles.max() < len(geometry[state]["faces"])
        recomputed = (geometry[state]["vertices"][geometry[state]["faces"][triangles]] * bary[:, :, None]).sum(1)
        np.testing.assert_allclose(recomputed, np.asarray(payload["landmarks_m"]), atol=1e-6)
        assert all(row is None or np.isfinite(row).all() for row in payload["observed_landmarks_m"])
        assert np.isfinite(np.asarray(payload["animated_model_to_reference_camera_4x4"])).all()
        observed_gap = payload["observed_inner_lip_gap_mm"]
        state_report[state] = {
            "mesh": item["path"],
            "vertices": int(len(geometry[state]["vertices"])),
            "faces": int(len(geometry[state]["faces"])),
            "extents_m": [round(float(v), 4) for v in geometry[state]["extents"]],
            "reference_frame": payload["frame_id"],
            "model_inner_lip_gap_mm": round(float(payload["model_inner_lip_gap_mm"]), 3),
            "observed_inner_lip_gap_mm": None if observed_gap is None else round(float(observed_gap), 3),
            "landmark_recompute_max_error": float(np.abs(recomputed - np.asarray(payload["landmarks_m"])).max()),
        }

    return {
        "unit": mesh_payload["unit"],
        "topology_shared": True,
        "states": state_report,
        "closed_to_open_vertex_motion_mm": {
            "max": round(float(per_vertex.max()), 3),
            "mean": round(float(per_vertex.mean()), 3),
        },
        "support_counts": dict(zip(support["labels"], support["counts"])),
        "status": manifest["status"],
        "warnings": manifest["warnings"],
        "version": manifest["software"]["version"],
        "algorithm": manifest["software"]["algorithm"],
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, ensure_ascii=False))
