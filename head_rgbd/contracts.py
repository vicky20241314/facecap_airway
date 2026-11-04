"""Small, testable contracts shared by acquisition, preparation and fitting."""
from __future__ import annotations

from typing import Mapping, Any


MAX_TEMPORAL_GAP_S = 0.2
SEQUENCE_SCHEMA = "head_rgbd.sequence.v1"
CAPTURE_SCHEMA = "head_rgbd.capture.v1"
FRAME_MANIFEST_SCHEMA = "head_rgbd.raw.v1"
DEPTH_UNIT = "mm"
MESH_UNIT = "m"
COORDINATE_CONVENTION = "X_right_Y_up_Z_backward"


def temporal_edge_allowed(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Return true only for a contiguous, successfully tracked adjacent pair."""
    if previous.get("status") == "excluded" or current.get("status") == "excluded":
        return False
    if previous.get("segment") != current.get("segment"):
        return False
    if current.get("tracking_barrier"):
        return False
    try:
        delta = float(current["timestamp_s"]) - float(previous["timestamp_s"])
    except (KeyError, TypeError, ValueError):
        return False
    delta = round(delta, 9)
    if not 0 < delta < MAX_TEMPORAL_GAP_S:
        return False
    return current.get("overlap", {}).get("status") == "tracked"
