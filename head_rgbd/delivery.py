"""Self-contained final/ and measured/ delivery directories.

Each directory carries the two same-named meshes, a downstream v2
``anatomical_landmarks.json`` and a minimal ``manifest.json``.  The two files are
written from the marker record produced by ``neck_landmark_aruco``; nothing is
invented and no mesh triangle binding is fabricated.

The module is pure file assembly so it can be tested without running the
camera, the base reconstruction or the neck solve.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from head_rgbd import __version__
from head_rgbd import neck_landmark as nl
from head_rgbd import mesh_integrity as mi

LANDMARKS_SCHEMA = "head_rgbd.delivery.anatomical_landmarks.v2"
MANIFEST_SCHEMA = "head_rgbd.delivery.manifest.v1"
SIDECAR_FILE = "anatomical_landmarks.json"
MANIFEST_FILE = "manifest.json"
STATES = ("closed", "open")
MESH_FILES = {state: f"head_{state}.obj" for state in STATES}
SKIN_ESTIMATE_UNCERTAINTY_MM = 1.5
SKIN_THICKNESS_MM = 0.10
# The downstream PowerShell launcher reads a JSON probe with the console code
# page while forcing PYTHONIOENCODING=utf-8; non-ASCII definitions make that
# probe unparsable.  The delivered sidecar keeps the semantic definitions in
# ASCII (the intermediate neck_landmarks.json may stay Chinese).
SIDECAR_DEFINITION = ("Operator-placed body-surface point at the laryngeal "
                      "prominence; not an internal thyroid-cartilage measurement")
SIDECAR_DEFINITION_SCOPE = ("Body-surface observation only; not an independent "
                            "geometric ground truth")
NOT_COMPLETED_REASON = (
    "ArUco skin anchor is a surface observation; the downstream projects S0 "
    "onto the current head mesh. No main-mesh triangle binding is fabricated.")


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _record_states(landmark_payload: dict | None) -> dict:
    return (landmark_payload or {}).get("states") or {}


def _observed(record: dict | None) -> bool:
    return bool(record and record.get("status") == nl.STATUS_OBSERVED
                and record.get("point_reference_m") is not None)


def usable_delivery(landmark_payload: dict | None, marker_status: str | None) -> bool:
    """Pair-ability: every state observed and the live gate not in fallback."""
    if marker_status == "not_usable":
        return False
    states = _record_states(landmark_payload)
    return all(_observed(states.get(state)) for state in STATES)


def _sidecar_state(record: dict, state: str) -> dict:
    marker = record.get("marker") or {}
    size = marker.get("size") or {}
    print_spec = marker.get("print") or {}
    transform = record.get("transform") or {}
    plane = (record.get("marker_plane") or {}).get("plane") or {}
    return {
        "name": nl.NAME,
        "definition": SIDECAR_DEFINITION,
        "definition_scope": SIDECAR_DEFINITION_SCOPE,
        "source": nl.SOURCE_ARUCO,
        "state": state,
        "status": nl.STATUS_OBSERVED,
        "reason": None,
        "frame_id": record.get("frame_id"),
        "timestamp": record.get("timestamp") or {},
        "pixel_uv": record.get("pixel_uv"),
        "image_size": record.get("image_size"),
        "point_observed_m": record.get("point_observed_m"),
        "coordinate_system": nl.OBSERVATION_FRAME_COORDINATES,
        "unit": "m",
        "depth_check": record.get("depth_check"),
        "quality": {
            "blur_score": record.get("blur_score"),
            "centre_method": (record.get("quality") or {}).get("centre_method"),
            "black_square_mm": size.get("black_square_mm"),
            "size_uncertainty_mm": size.get("size_uncertainty_mm"),
        },
        "reference_frame": record.get("reference_frame"),
        "reference_coordinate_system": "reference camera X_right_Y_up_Z_backward",
        "transform": {
            "status": transform.get("status"),
            "method": transform.get("method"),
            "direction": transform.get("direction"),
            "unit": transform.get("unit", "m"),
            "applies_to": transform.get("applies_to",
                                       "neck surface observation only"),
            "rotation_3x3": transform.get("rotation_3x3"),
            "translation_m": transform.get("translation_m"),
            "links": transform.get("links") or [],
        },
        "point_reference_m": record.get("point_reference_m"),
        "skin_estimate": {
            "status": "estimated",
            "method": "marker_surface_minus_thickness_along_local_marker_normal",
            "thickness_mm": SKIN_THICKNESS_MM,
            "is_direct_measurement": False,
            "uncertainty_mm": SKIN_ESTIMATE_UNCERTAINTY_MM,
            "normal_unit": plane.get("normal"),
        },
        "skin_estimate_reference_m": record.get("skin_estimate_reference_m"),
        "skin_estimate_method": "marker_surface_minus_thickness_along_local_marker_normal",
        "skin_estimate_uncertainty_mm": SKIN_ESTIMATE_UNCERTAINTY_MM,
        "marker": {
            "marker_type": "aruco",
            "dictionary": marker.get("dictionary", "DICT_4X4_50"),
            "detected_id": marker.get("target_id"),
            "print": {
                **print_spec,
                "black_square_mm": size.get("black_square_mm"),
                "size_source": size.get("size_source"),
                "size_uncertainty_mm": size.get("size_uncertainty_mm"),
            },
            "quality": {
                "blur_score": record.get("blur_score"),
                "detected_ids": marker.get("detected_ids"),
                "size": size,
            },
            "attempts": record.get("attempts") or [],
        },
        "larynx_center": record.get("larynx_center"),
        "main_delivery_mesh_file": MESH_FILES[state],
        "mesh_binding": {"status": "not_completed", "reason": NOT_COMPLETED_REASON},
        "main_head_mesh_binding": {"status": "not_completed",
                                   "reason": NOT_COMPLETED_REASON},
        "main_mesh_binding_completed": False,
    }


def build_sidecar(landmark_payload: dict | None, *, created_at: str | None = None) -> dict | None:
    """Downstream v2 sidecar, or None when a state is not observed."""
    if not usable_delivery(landmark_payload, "usable"):
        return None
    states = _record_states(landmark_payload)
    return {
        "schema": LANDMARKS_SCHEMA,
        "created_at": created_at or _now(),
        "software_version": __version__,
        "unit": "m",
        "distance_unit": "mm",
        "depth_unit": "mm",
        "coordinate_convention": "X_right_Y_up_Z_backward",
        "landmark_name": nl.NAME,
        "definition": SIDECAR_DEFINITION,
        "definition_scope": SIDECAR_DEFINITION_SCOPE,
        "observation_is_manual_truth": False,
        "effective_marker_mode": "aruco_marker",
        "states": {state: _sidecar_state(states[state], state) for state in STATES},
        "note": ("marker centre is a body-surface observation on the laryngeal "
                 "prominence, not an internal cartilage measurement; L_obs, "
                 "L_skin_est and L_anchor are recorded separately"),
    }


def _mesh_record(source: Path) -> dict:
    V, F = mi.load_obj(source)
    return {"vertices": int(len(V)), "faces": int(len(F))}


def build_manifest(*, role: str, status: str, meshes: dict[str, Path],
                   reference_frames: dict[str, str],
                   landmark_payload: dict | None, marker_status: str | None,
                   provenance: dict, include_sidecar: bool,
                   created_at: str | None = None) -> dict:
    states = _record_states(landmark_payload)
    mesh_states = {}
    for state in STATES:
        record = _mesh_record(Path(meshes[state]))
        mesh_states[state] = {
            "path": MESH_FILES[state],
            "reference_frame": reference_frames.get(state),
            "vertices": record["vertices"],
            "faces": record["faces"],
        }
    fmask = {state: np.asarray(mi.load_obj(Path(meshes[state]))[1], np.int64)
             for state in STATES}
    topology_shared = bool(fmask["closed"].shape == fmask["open"].shape
                           and np.array_equal(fmask["closed"], fmask["open"]))
    landmark_states = {}
    for state in STATES:
        rec = states.get(state) or {}
        landmark_states[state] = {
            "source": rec.get("source"),
            "status": rec.get("status"),
            "frame_id": rec.get("frame_id"),
            "point_observed_m": rec.get("point_observed_m"),
            "reference_frame": rec.get("reference_frame") or reference_frames.get(state),
        }
    return {
        "schema": MANIFEST_SCHEMA,
        "created_at": created_at or _now(),
        "software": {"name": "head_rgbd", "version": __version__},
        "status": status,
        "delivery_role": role,
        "mesh": {
            "unit": "m",
            "coordinate_system": "reference camera X_right_Y_up_Z_backward",
            "topology_shared": topology_shared,
            "states": mesh_states,
        },
        "anatomical_landmarks": {
            "schema": LANDMARKS_SCHEMA,
            "file": SIDECAR_FILE if include_sidecar else None,
            "states": landmark_states,
        },
        "provenance": {
            **provenance,
            "marker_status": marker_status,
            "sidecar_written": bool(include_sidecar),
        },
    }


def write_delivery(root: str | Path, *, final_sources: dict[str, Path],
                   measured_sources: dict[str, Path],
                   landmark_payload: dict | None, marker_status: str | None,
                   reference_frames: dict[str, str], provenance: dict) -> dict:
    """Write final/ and measured/; root-level files are left to the caller."""
    root = Path(root)
    usable = usable_delivery(landmark_payload, marker_status)
    status = "usable" if usable else "not_usable"
    created_at = _now()
    result = {}
    for role, sources in (("final", final_sources), ("measured", measured_sources)):
        directory = root / role
        directory.mkdir(parents=True, exist_ok=True)
        for state in STATES:
            source = Path(sources[state])
            target = directory / MESH_FILES[state]
            # The fast path can write the measured base straight into the
            # delivery directory; copying a file onto itself would fail.
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
        sidecar = build_sidecar(landmark_payload, created_at=created_at) if usable else None
        if sidecar is not None:
            (directory / SIDECAR_FILE).write_text(
                json.dumps(sidecar, ensure_ascii=False, indent=1, allow_nan=False) + "\n",
                encoding="utf-8")
        manifest = build_manifest(
            role=role, status=status, meshes={state: directory / MESH_FILES[state]
                                              for state in STATES},
            reference_frames=reference_frames, landmark_payload=landmark_payload,
            marker_status=marker_status, provenance=provenance,
            include_sidecar=sidecar is not None, created_at=created_at)
        (directory / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1, allow_nan=False) + "\n",
            encoding="utf-8")
        result[role] = {
            "dir": str(directory),
            "manifest": str(directory / MANIFEST_FILE),
            "sidecar": str(directory / SIDECAR_FILE) if sidecar is not None else None,
            "status": status,
        }
    return {"usable": usable, "status": status, "delivery": result,
            "created_at": created_at}