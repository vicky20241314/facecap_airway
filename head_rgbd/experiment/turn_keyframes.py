"""Angle-bin plus quality keyframe selection, with a hard budget.

Pure NumPy, no model and no image access, so it can be unit tested without assets.

Two different groupings are produced and must not be confused:

``angle_group``
    A yaw bucket. It answers "which part of the head surface does this view add".
    The shared-identity fit gives every angle group the same total vote.
``tracking_group``
    A continuity segment inherited from :mod:`turn_tracking`. It answers "which
    frames are temporally connected". One tracking group normally spans many angle
    groups, and the same angle can be revisited in a later tracking group.

Selection never pads. If the recorded turn covers only a few angles, fewer than
``max_keyframes`` frames are returned and ``coverage`` says what is missing.
"""
from __future__ import annotations

import numpy as np

ROLE_ANCHOR = "anchor"
ROLE_TURN = "turn"


def angle_bucket(yaw_deg: float, bin_deg: float) -> int:
    """Stable integer bucket for one yaw angle."""
    if not np.isfinite(yaw_deg):
        raise ValueError("yaw must be finite to be bucketed")
    return int(np.floor(float(yaw_deg) / float(bin_deg) + 0.5))


def group_label(bucket: int, bin_deg: float) -> str:
    """Human readable angle-group name, e.g. ``yaw-20`` for bucket -2 at 10 deg."""
    return f"yaw{int(round(bucket * float(bin_deg))):+d}"


def view_of(yaw_deg: float, front_yaw_deg: float) -> str:
    """Map a yaw angle to the sequence ``view`` label used by the shared modules.

    The recorded convention in this project is that a negative yaw is the left side
    of the image and a positive yaw is the right side.
    """
    if abs(yaw_deg) <= front_yaw_deg:
        return "front"
    return "left" if yaw_deg < 0 else "right"


def select(anchor_entries, turn_entries, config) -> dict:
    """Choose at most ``max_keyframes`` closed identity keyframes."""
    keyframes = config.keyframes
    rejected: dict[str, int] = {}
    pool: list[dict] = []
    for role, entries in ((ROLE_ANCHOR, anchor_entries), (ROLE_TURN, turn_entries)):
        for entry in entries:
            reason = _reject_reason(entry, keyframes)
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            candidate = dict(entry)
            candidate["role"] = role
            candidate["yaw_deg"] = float(entry["yaw_deg"])
            candidate["quality"] = float(entry["quality"])
            candidate["bucket"] = angle_bucket(candidate["yaw_deg"], keyframes.angle_bin_deg)
            pool.append(candidate)
    if not pool:
        raise ValueError("no keyframe candidate survived the quality, state and yaw filters")

    anchor, anchor_note = _choose_anchor(pool, keyframes)
    selected = [anchor]
    covered = {anchor["bucket"]}
    reasons = {id(anchor): anchor_note}
    stop_reason = "budget_reached"
    while len(selected) < keyframes.max_keyframes:
        covered_landmarks = _union_landmarks(selected)
        best, best_score, best_gain = None, -np.inf, 0.0
        for candidate in pool:
            if any(candidate is item for item in selected):
                continue
            if candidate["bucket"] in covered:
                # Redundant at the same angle: the more reliable representative stays.
                continue
            if _min_separation(selected, candidate["yaw_deg"]) < keyframes.min_yaw_separation_deg:
                continue
            gain = _surface_gain(covered_landmarks, candidate)
            score = candidate["quality"] * (1.0 + keyframes.surface_gain_weight * gain) \
                + keyframes.side_balance_bonus * _balance_bonus(candidate, selected, keyframes)
            if score > best_score:
                best, best_score, best_gain = candidate, score, gain
        if best is None:
            stop_reason = "no_uncovered_angle_left"
            break
        if best_score < keyframes.min_selection_score:
            stop_reason = "best_remaining_candidate_below_score_floor"
            break
        selected.append(best)
        covered.add(best["bucket"])
        reasons[id(best)] = (f"new_angle_bin_{group_label(best['bucket'], keyframes.angle_bin_deg)}"
                             f"_surface_gain_{best_gain:.2f}")
    else:
        stop_reason = "budget_reached"

    keyframe_rows = []
    for order, candidate in enumerate(selected):
        label = group_label(candidate["bucket"], keyframes.angle_bin_deg)
        keyframe_rows.append({
            "id": f"{candidate['role']}_{candidate['source_index']:03d}",
            "order": order,
            "role": candidate["role"],
            "source_index": int(candidate["source_index"]),
            "source": candidate.get("source"),
            "source_frame": candidate.get("source_frame"),
            "timestamp_s": float(candidate["timestamp_s"]),
            "yaw_deg": round(candidate["yaw_deg"], 4),
            "pitch_deg": None if candidate.get("pitch_deg") is None else round(float(candidate["pitch_deg"]), 4),
            "roll_deg": None if candidate.get("roll_deg") is None else round(float(candidate["roll_deg"]), 4),
            "quality": round(candidate["quality"], 4),
            "quality_components": candidate.get("quality_components"),
            "tracking_group": candidate.get("tracking_group"),
            "angle_group": label,
            "angle_bucket_center_deg": int(round(candidate["bucket"] * keyframes.angle_bin_deg)),
            "visible_landmark_count": int(_visible(candidate).sum()),
            "view": view_of(candidate["yaw_deg"], keyframes.front_yaw_deg),
            "selection_reason": reasons[id(candidate)],
            "state": "closed",
            "observation_role": "keyframe",
            "canonical_keyframe_id": f"{candidate['role']}_{candidate['source_index']:03d}",
        })

    support_candidates = _choose_support_observations(selected, pool, keyframes)
    support_rows = []
    for order, (canonical, candidate) in enumerate(support_candidates):
        label = group_label(canonical["bucket"], keyframes.angle_bin_deg)
        support_rows.append({
            "id": f"support_{candidate['role']}_{candidate['source_index']:03d}",
            "order": len(keyframe_rows) + order,
            "role": candidate["role"],
            "source_index": int(candidate["source_index"]),
            "source": candidate.get("source"),
            "source_frame": candidate.get("source_frame"),
            "timestamp_s": float(candidate["timestamp_s"]),
            "yaw_deg": round(candidate["yaw_deg"], 4),
            "pitch_deg": None if candidate.get("pitch_deg") is None else round(float(candidate["pitch_deg"]), 4),
            "roll_deg": None if candidate.get("roll_deg") is None else round(float(candidate["roll_deg"]), 4),
            "quality": round(candidate["quality"], 4),
            "quality_components": candidate.get("quality_components"),
            "tracking_group": candidate.get("tracking_group"),
            "angle_group": label,
            "angle_bucket_center_deg": int(round(canonical["bucket"] * keyframes.angle_bin_deg)),
            "visible_landmark_count": int(_visible(candidate).sum()),
            "view": view_of(candidate["yaw_deg"], keyframes.front_yaw_deg),
            "selection_reason": (
                f"same_angle_repeat_delta_{abs(candidate['yaw_deg'] - canonical['yaw_deg']):.2f}deg"
            ),
            "state": "closed",
            "observation_role": "support",
            "canonical_keyframe_id": f"{canonical['role']}_{canonical['source_index']:03d}",
        })

    used_identities = {
        _source_identity(item)
        for item in [*selected, *(candidate for _, candidate in support_candidates)]
    }
    # Final pass: explain only candidates that are not already support observations.
    for candidate in pool:
        if any(candidate is item for item in selected):
            continue
        if _source_identity(candidate) in used_identities:
            continue
        if candidate["bucket"] in covered:
            rejected["same_angle_bucket_as_a_selected_frame"] = \
                rejected.get("same_angle_bucket_as_a_selected_frame", 0) + 1
        elif _min_separation(selected, candidate["yaw_deg"]) < keyframes.min_yaw_separation_deg:
            rejected["too_close_in_yaw_to_a_selected_frame"] = \
                rejected.get("too_close_in_yaw_to_a_selected_frame", 0) + 1
        else:
            rejected["not_needed_within_keyframe_budget"] = \
                rejected.get("not_needed_within_keyframe_budget", 0) + 1

    return {
        "keyframes": keyframe_rows,
        "support_frames": support_rows,
        "angle_groups": _angle_groups(keyframe_rows),
        "tracking_groups": _tracking_groups(keyframe_rows),
        "coverage": _coverage(keyframe_rows, keyframes),
        "rejected": dict(sorted(rejected.items())),
        "budget": {
            "max_keyframes": keyframes.max_keyframes,
            "selected": len(keyframe_rows),
            "anchor_budget": keyframes.anchor_budget,
            "repeat_observations_per_angle": keyframes.repeat_observations_per_angle,
            "support_observations": len(support_rows),
            "total_observations": len(keyframe_rows) + len(support_rows),
            "padded": False,
            "stop_reason": stop_reason,
            "policy": ("keyframes stay one-per-angle pose anchors; a bounded set of nearby reliable "
                       "repeats is retained as support observations and every angle group keeps one vote"),
        },
        "policy": {
            "angle_group": "yaw bucket; the shared shape fit equalises total weight per angle group",
            "tracking_group": "continuity segment from turn_tracking; independent of the angle",
            "anchor": "the most reliable closed frontal frame initialises shape, scale and frontal form",
            "no_padding": "an under-covered turn returns fewer keyframes than the budget",
            "repeats": ("repeats are pose-registered separately; raw depth is never averaged and "
                        "their total vote is capped by the angle-group weight"),
        },
    }


def _choose_support_observations(selected, pool, keyframes):
    """Choose nearby reliable repeats without spending the keyframe budget."""
    limit = int(keyframes.repeat_observations_per_angle)
    if not limit:
        return []
    by_bucket: dict[int, list[dict]] = {}
    selected_ids = {id(item) for item in selected}
    for candidate in pool:
        if id(candidate) in selected_ids:
            continue
        by_bucket.setdefault(candidate["bucket"], []).append(candidate)

    chosen = []
    used_sources = {_source_identity(item) for item in selected}
    for canonical in selected:
        candidates = []
        for candidate in by_bucket.get(canonical["bucket"], []):
            identity = _source_identity(candidate)
            if identity in used_sources:
                continue
            delta = abs(float(candidate["yaw_deg"]) - float(canonical["yaw_deg"]))
            if delta > float(keyframes.repeat_max_angle_delta_deg):
                continue
            close = max(0.0, 1.0 - delta / float(keyframes.repeat_max_angle_delta_deg))
            independent_group = candidate.get("tracking_group") != canonical.get("tracking_group")
            score = (float(candidate["quality"]) + 0.20 * close
                     + (0.03 if independent_group else 0.0))
            candidates.append((score, -delta, float(candidate["quality"]), candidate))
        candidates.sort(key=lambda item: item[:3], reverse=True)
        for _, _, _, candidate in candidates[:limit]:
            identity = _source_identity(candidate)
            if identity in used_sources:
                continue
            used_sources.add(identity)
            chosen.append((canonical, candidate))
    return chosen


def _source_identity(entry: dict):
    """Identify a physical frame even when one source is supplied under both roles."""
    source = str(entry.get("source") or "")
    frame = entry.get("source_frame")
    if frame is None:
        frame = entry.get("source_index")
    return source, int(frame)


def _reject_reason(entry: dict, keyframes) -> str | None:
    if entry.get("state_label") not in (None, "closed"):
        return "open_mouth_frame"
    if not entry.get("usable", True):
        return "frame_unusable_in_tracking"
    yaw = entry.get("yaw_deg")
    if yaw is None or not np.isfinite(yaw):
        return "no_pose_measurement"
    if abs(float(yaw)) > keyframes.max_yaw_deg:
        return "yaw_beyond_max"
    if float(entry.get("quality", 0.0)) < keyframes.min_quality:
        return "quality_below_floor"
    return None


def _choose_anchor(pool, keyframes):
    frontal = [item for item in pool
               if item["role"] == ROLE_ANCHOR and abs(item["yaw_deg"]) <= keyframes.anchor_max_yaw_deg]
    note = "best_frontal_anchor"
    if not frontal:
        frontal = [item for item in pool if item["role"] == ROLE_ANCHOR]
        note = "frontal_anchor_outside_expected_yaw_used_as_anchor"
    if not frontal:
        frontal = [item for item in pool if abs(item["yaw_deg"]) <= keyframes.anchor_max_yaw_deg]
        note = "no_anchor_frame_available_best_frontal_turn_frame_used_as_anchor"
    if not frontal:
        frontal = list(pool)
        note = "no_frontal_candidate_best_available_frame_used_as_anchor"
    return max(frontal, key=lambda item: (item["quality"], -abs(item["yaw_deg"]), -item["timestamp_s"])), note


def _visible(candidate: dict) -> np.ndarray:
    visible = candidate.get("visible_landmarks")
    if visible is None:
        return np.ones(68, dtype=bool)
    return np.asarray(visible, dtype=bool)


def _union_landmarks(selected) -> np.ndarray:
    union = np.zeros(68, dtype=bool)
    for item in selected:
        union |= _visible(item)
    return union


def _surface_gain(covered_landmarks: np.ndarray, candidate: dict) -> float:
    visible = _visible(candidate)
    total = int(visible.sum())
    if total == 0:
        return 0.0
    return float(int((visible & ~covered_landmarks).sum()) / total)


def _min_separation(selected, yaw_deg: float) -> float:
    if not selected:
        return float("inf")
    return min(abs(yaw_deg - item["yaw_deg"]) for item in selected)


def _balance_bonus(candidate: dict, selected, keyframes) -> float:
    if abs(candidate["yaw_deg"]) <= keyframes.front_yaw_deg:
        return 0.0
    side = "left" if candidate["yaw_deg"] < 0 else "right"
    same = sum(1 for item in selected
               if abs(item["yaw_deg"]) > keyframes.front_yaw_deg
               and ("left" if item["yaw_deg"] < 0 else "right") == side)
    other = sum(1 for item in selected
                if abs(item["yaw_deg"]) > keyframes.front_yaw_deg
                and ("left" if item["yaw_deg"] < 0 else "right") != side)
    return 1.0 if same <= other else 0.0


def _angle_groups(rows) -> list[dict]:
    groups: dict[str, dict] = {}
    for row in rows:
        item = groups.setdefault(row["angle_group"], {"label": row["angle_group"],
                                                      "center_deg": row["angle_bucket_center_deg"],
                                                      "frames": [], "tracking_groups": []})
        item["frames"].append(row["id"])
        if row["tracking_group"] not in item["tracking_groups"]:
            item["tracking_groups"].append(row["tracking_group"])
    for item in groups.values():
        item["frame_count"] = len(item["frames"])
    return [groups[key] for key in sorted(groups, key=lambda key: groups[key]["center_deg"])]


def _tracking_groups(rows) -> list[dict]:
    groups: dict[str, dict] = {}
    for row in rows:
        item = groups.setdefault(row["tracking_group"], {"label": row["tracking_group"],
                                                         "frames": [], "angle_groups": []})
        item["frames"].append(row["id"])
        if row["angle_group"] not in item["angle_groups"]:
            item["angle_groups"].append(row["angle_group"])
    for item in groups.values():
        item["frame_count"] = len(item["frames"])
        item["spans_multiple_angles"] = len(item["angle_groups"]) > 1
    return [groups[key] for key in sorted(groups)]


def _coverage(rows, keyframes) -> dict:
    yaw = np.asarray([row["yaw_deg"] for row in rows], dtype=np.float64)
    left = [row["id"] for row in rows if row["yaw_deg"] < -keyframes.front_yaw_deg]
    right = [row["id"] for row in rows if row["yaw_deg"] > keyframes.front_yaw_deg]
    front = [row["id"] for row in rows if abs(row["yaw_deg"]) <= keyframes.front_yaw_deg]
    missing = []
    if not left:
        missing.append("left_side_not_observed")
    if not right:
        missing.append("right_side_not_observed")
    if not front:
        missing.append("no_near_frontal_frame")
    if min(abs(yaw.min()), abs(yaw.max())) < keyframes.min_yaw_separation_deg and len(rows) > 1:
        missing.append("narrow_yaw_span")
    return {
        "selected_frames": len(rows),
        "yaw_min_deg": round(float(yaw.min()), 4),
        "yaw_max_deg": round(float(yaw.max()), 4),
        "yaw_span_deg": round(float(yaw.max() - yaw.min()), 4),
        "angle_group_count": len({row["angle_group"] for row in rows}),
        "front_frames": front,
        "left_frames": left,
        "right_frames": right,
        "two_sided": bool(left and right),
        "missing": missing,
    }
