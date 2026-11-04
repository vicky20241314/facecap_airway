"""Input discovery for the front-anchor turn experiment.

The one rule this module enforces: a plain RGB video (or an image folder, or an
``.npz`` without depth) is **not** a synchronised RGB-D sequence. Such an input is
rejected with an explicit error and never degraded into a landmark-only or
monocular run, because this experiment measures a shared metric surface.

Supported RGB-D inputs, all of them already produced by this project:

* ``head_rgbd.capture.v1`` session (``session.json`` plus per-segment ``.npz``)
* ``head_rgbd.raw.v1`` frame manifest (``frames.json`` plus chunk ``.npz``)
* ``head_rgbd.sequence.v1`` prepared sequence (``sequence.json`` plus frame ``.npz``)
* a single segment ``.npz`` containing stacked ``rgb``/``depth``/``intrinsics``
* a directory of one-frame ``.npz`` files that each contain ``rgb`` and ``depth``

Only the first three carry a declared ``state``/``view`` label; the last two are
accepted as one continuous closed segment and the absence of a label is recorded
as a note, never invented as metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from head_rgbd.audit import sha
from head_rgbd.contracts import CAPTURE_SCHEMA, FRAME_MANIFEST_SCHEMA, SEQUENCE_SCHEMA

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

ROLE_ANCHOR = "anchor"
ROLE_TURN = "turn"


class InputError(RuntimeError):
    """The input cannot be used by this experiment; nothing is silently repaired."""


class UnsupportedInput(InputError):
    """The input is recognised but is not a synchronised RGB-D sequence."""


@dataclass(frozen=True)
class SourcePlan:
    path: Path
    kind: str
    schema: str | None = None
    frame_count: int = 0
    depth_available: bool = False
    alignment_declared: bool | None = None
    alignment_evidence: str | None = None
    declared_state: bool = False
    declared_view: bool = False
    notes: tuple[str, ...] = ()
    files: tuple[Path, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "schema": self.schema,
            "frame_count": self.frame_count,
            "depth_available": self.depth_available,
            "rgb_depth_alignment_declared": self.alignment_declared,
            "alignment_evidence": self.alignment_evidence,
            "state_declared_by_input": self.declared_state,
            "view_declared_by_input": self.declared_view,
            "notes": list(self.notes),
        }


@dataclass
class FrameEntry:
    file: Path
    index_in_file: int | None
    timestamp_s: float
    intrinsics: np.ndarray
    state: str | None
    view: str | None
    group: str
    source: str
    source_frame: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InputMetadata:
    """What the caller must know before treating frames as measurements."""

    kind: str
    alignment_declared: bool | None
    calibration_status: str
    depth_unit: str
    notes: tuple[str, ...]

    @property
    def alignment_verified(self) -> bool:
        return self.alignment_declared is True

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "rgb_depth_alignment_declared": self.alignment_declared,
            "calibration_status": self.calibration_status,
            "depth_unit": self.depth_unit,
            "alignment_is_independent_calibration": False,
            "notes": list(self.notes),
        }


class FrameSequence:
    """Random access to one RGB-D source with a single decoded chunk cached.

    Frames stored as stacked arrays (capture segments, frame chunks) are decoded
    per file, so peak memory is bounded by the largest chunk, not by the video.
    Small arrays (timestamps, intrinsics) are read once when the sequence opens.
    """

    def __init__(self, entries: list[FrameEntry], *, plan: SourcePlan, metadata: InputMetadata,
                 role: str, dropped: list[dict[str, Any]] | None = None):
        if not entries:
            raise InputError(f"no usable frame left for role {role!r} in {plan.path}")
        self.entries = entries
        self.plan = plan
        self.metadata = metadata
        self.role = role
        self.dropped = dropped or []
        self._cache_path: Path | None = None
        self._cache: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[tuple[int, FrameEntry]]:
        return iter(enumerate(self.entries))

    @property
    def timestamps(self) -> np.ndarray:
        return np.asarray([entry.timestamp_s for entry in self.entries], dtype=np.float64)

    @property
    def groups(self) -> list[str]:
        return list(dict.fromkeys(entry.group for entry in self.entries))

    def _arrays(self, path: Path) -> dict[str, np.ndarray]:
        if self._cache_path != path or self._cache is None:
            with np.load(path, allow_pickle=False) as data:
                self._cache = {key: data[key] for key in data.files}
            self._cache_path = path
        return self._cache

    def read(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        arrays = self._arrays(entry.file)
        rgb = _take(arrays.get("rgb"), entry.index_in_file, "rgb", entry.file)
        depth = _take(arrays.get("depth"), entry.index_in_file, "depth", entry.file)
        if rgb is None:
            raise UnsupportedInput(f"{entry.file} has no RGB array; this experiment needs RGB-D")
        if depth is None:
            raise UnsupportedInput(
                f"{entry.file} has no depth array for frame {entry.source_frame}. "
                "A colour video cannot be used: the shared metric surface needs synchronised depth."
            )
        rgb = np.asarray(rgb)
        depth = np.asarray(depth)
        if rgb.dtype != np.uint8:
            raise InputError(f"RGB must be uint8 in {entry.file}, got {rgb.dtype}")
        if depth.dtype != np.uint16:
            raise UnsupportedInput(
                f"depth must be uint16 millimetres in {entry.file}, got {depth.dtype}; "
                "a normalised or float depth map has no declared unit"
            )
        if rgb.shape != (*depth.shape, 3):
            raise UnsupportedInput(
                f"RGB {rgb.shape} and depth {depth.shape} are not pixel-aligned in {entry.file}; "
                "aligned RGB-D is required and no resampling is guessed"
            )
        frame: dict[str, Any] = {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": np.asarray(entry.intrinsics, dtype=np.float64),
            "timestamp_s": float(entry.timestamp_s),
            "source": entry.source,
            "source_frame": int(entry.source_frame),
            "group": entry.group,
        }
        for key, contract in (("uv", "uv"), ("xyz", "xyz"), ("valid", "valid"),
                              ("landmark2d_valid", "landmark2d_valid"),
                              ("face_mask", "face_mask"), ("neck_mask", "neck_mask")):
            if key in arrays:
                frame[key] = _take_single(arrays[key], entry.index_in_file)
        return frame

    def provenance(self) -> dict[str, Any]:
        files = list(dict.fromkeys(str(entry.file) for entry in self.entries))
        return {
            "role": self.role,
            "source": self.plan.as_dict(),
            "metadata": self.metadata.as_dict(),
            "frames": len(self.entries),
            "groups": self.groups,
            "files": [{"path": path, "sha256": sha(Path(path))} for path in files],
            "dropped_frames": self.dropped,
        }


# --------------------------------------------------------------------- discovery
def plan_source(path: str | Path) -> SourcePlan:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"input does not exist: {target}")
    if target.is_dir():
        for name, kind in (("session.json", "capture_session"), ("frames.json", "frame_manifest"),
                           ("sequence.json", "sequence")):
            candidate = target / name
            if candidate.is_file():
                return _plan_json(candidate, kind)
        npz = sorted(item for item in target.iterdir() if item.suffix.lower() == ".npz")
        if npz:
            return _plan_frame_directory(target, npz)
        return _plan_colour_only_directory(target)
    suffix = target.suffix.lower()
    if suffix == ".json":
        return _plan_json(target, _kind_from_schema(target))
    if suffix == ".npz":
        return _plan_segment(target)
    if suffix in VIDEO_SUFFIXES:
        raise UnsupportedInput(_colour_only_message(target))
    if suffix in IMAGE_SUFFIXES:
        raise UnsupportedInput(_colour_only_message(target))
    raise UnsupportedInput(f"unsupported input type: {target} ({suffix or 'no suffix'})")


def open_source(path: str | Path, role: str, config) -> FrameSequence:
    """Open one role of the experiment input and validate the RGB-D contract."""
    plan = plan_source(path)
    notes = list(plan.notes)
    if role not in (ROLE_ANCHOR, ROLE_TURN):
        raise ValueError(f"unknown input role: {role!r}")
    if plan.kind in ("capture_session", "frame_manifest", "sequence"):
        payload = json.loads(plan.path.read_text(encoding="utf-8"))
        _check_depth_and_alignment(plan, payload, config)
        entries, dropped = _entries_from_payload(plan, payload, role)
    elif plan.kind == "segment_npz":
        _check_npz_segment(plan)
        entries, dropped = _entries_from_segment(plan), []
        notes.append(
            "single segment .npz carries no state/view label; it is treated as one "
            "continuous closed segment and the turn rate is measured from the data"
        )
    elif plan.kind == "frame_directory":
        _check_npz_segment(plan, plan.files[0])
        entries, dropped = _entries_from_directory(plan), []
        notes.append(
            "one-frame .npz directory carries no state/view label; frames are treated "
            "as one continuous closed segment ordered by file name"
        )
    else:  # pragma: no cover - plan_source never returns a colour-only kind
        raise UnsupportedInput(_colour_only_message(plan.path))

    if role == ROLE_ANCHOR:
        entries = [entry for entry in entries if entry.state in (None, "closed")]
        entries = [entry for entry in entries if entry.view in (None, "front")] or entries
        if len(entries) < config.input.min_anchor_frames:
            raise InputError(
                f"anchor input {plan.path} has {len(entries)} usable closed frontal frame(s); "
                f"at least {config.input.min_anchor_frames} is required"
            )
    else:
        for entry in entries:
            if entry.state == "open":
                dropped.append({"source_frame": entry.source_frame, "group": entry.group,
                                "reason": "open_mouth_frame_excluded_from_closed_identity_turn"})
        entries = [entry for entry in entries if entry.state in (None, "closed")]
        if len(entries) < config.input.min_source_frames:
            raise InputError(
                f"turn input {plan.path} has {len(entries)} usable closed frame(s); "
                f"at least {config.input.min_source_frames} is required"
            )
    metadata = InputMetadata(
        kind=plan.kind,
        alignment_declared=plan.alignment_declared,
        calibration_status=_calibration_status(plan),
        depth_unit="mm",
        notes=tuple(notes),
    )
    return FrameSequence(entries, plan=plan, metadata=metadata, role=role, dropped=dropped)


# ------------------------------------------------------------------------ detail
def _kind_from_schema(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{path} is not valid JSON: {exc}") from exc
    schema = payload.get("schema")
    if schema == CAPTURE_SCHEMA:
        return "capture_session"
    if schema == FRAME_MANIFEST_SCHEMA:
        return "frame_manifest"
    if schema == SEQUENCE_SCHEMA:
        return "sequence"
    if path.name == "session.json":
        raise InputError(f"{path} does not declare {CAPTURE_SCHEMA}")
    if path.name == "frames.json":
        raise InputError(f"{path} does not declare {FRAME_MANIFEST_SCHEMA}")
    if path.name == "sequence.json":
        raise InputError(f"{path} does not declare {SEQUENCE_SCHEMA}")
    raise UnsupportedInput(
        f"unrecognised JSON input {path}: expected one of {CAPTURE_SCHEMA}, "
        f"{FRAME_MANIFEST_SCHEMA}, {SEQUENCE_SCHEMA}"
    )


def _plan_json(path: Path, kind: str) -> SourcePlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    notes: list[str] = []
    files: list[Path] = []
    declared_state = declared_view = False
    frame_count = 0
    if kind == "capture_session":
        schema = payload.get("schema")
        if schema != CAPTURE_SCHEMA:
            raise InputError(f"{path} must declare {CAPTURE_SCHEMA}, got {schema!r}")
        for segment in payload.get("segments", []):
            segment_file = (path.parent / segment["file"]).resolve()
            if segment_file.parent != path.parent or not segment_file.is_file():
                raise InputError(f"capture segment is outside the session or missing: {segment_file}")
            files.append(segment_file)
            frame_count += _frame_count(segment_file)
        declared_state = declared_view = True
        alignment = payload.get("rgb_depth_aligned")
        calibration = str(payload.get("calibration_status") or "declared_by_capture")
        evidence = "capture session declares hardware D2C alignment"
    elif kind == "frame_manifest":
        schema = payload.get("schema")
        if schema != FRAME_MANIFEST_SCHEMA:
            raise InputError(f"{path} must declare {FRAME_MANIFEST_SCHEMA}, got {schema!r}")
        for chunk in payload.get("chunks", []):
            chunk_file = (path.parent / chunk["file"]).resolve()
            if chunk_file.parent != path.parent or not chunk_file.is_file():
                raise InputError(f"frame chunk is outside the manifest or missing: {chunk_file}")
            files.append(chunk_file)
            frame_count += _frame_count(chunk_file)
        declared_state = declared_view = True
        alignment = payload.get("rgb_depth_aligned")
        calibration = str(payload.get("calibration_status") or "declared_by_manifest")
        evidence = str(payload.get("alignment_evidence") or "external manifest declares aligned RGB-D")
    else:
        schema = payload.get("schema")
        if schema != SEQUENCE_SCHEMA:
            raise InputError(f"{path} must declare {SEQUENCE_SCHEMA}, got {schema!r}")
        candidates = [row for row in payload.get("frames", []) if row.get("status") == "candidate"]
        if not candidates:
            raise InputError(f"{path} contains no candidate frames")
        for row in candidates:
            frame_file = (path.parent / row["file"]).resolve()
            if frame_file.parent != path.parent or not frame_file.is_file():
                raise InputError(f"prepared frame is outside the sequence or missing: {frame_file}")
            files.append(frame_file)
        frame_count = len(candidates)
        declared_state = all("state" in row for row in candidates)
        declared_view = all("view" in row for row in candidates)
        alignment = payload.get("independent_extrinsics") is False and True
        calibration = str(payload.get("calibration_status") or "declared_by_sequence")
        evidence = "prepared sequence was built from a capture whose alignment is declared, not calibrated"
    for key, expected in (("depth_unit", "mm"), ("mesh_unit", "m"),
                          ("coordinate_convention", "X_right_Y_up_Z_backward")):
        if payload.get(key) != expected:
            raise InputError(f"{path} {key} must be {expected!r}, got {payload.get(key)!r}")
    if payload.get("status") not in (None, "prepared", "complete") and kind != "frame_manifest":
        notes.append(f"input status is {payload.get('status')!r}; frames are used as recorded")
    return SourcePlan(path=path, kind=kind, schema=payload.get("schema"), frame_count=frame_count,
                      depth_available=True, alignment_declared=alignment,
                      alignment_evidence=f"{evidence} ({calibration})",
                      declared_state=declared_state, declared_view=declared_view,
                      notes=tuple(dict.fromkeys(notes)), files=tuple(files))


def _plan_segment(path: Path) -> SourcePlan:
    count = _frame_count(path)
    return SourcePlan(path=path, kind="segment_npz", schema=None, frame_count=count,
                      depth_available=True, alignment_declared=None,
                      alignment_evidence="single segment file; alignment is not described by the file",
                      declared_state=False, declared_view=False,
                      notes=("segment .npz carries no calibration record",), files=(path,))


def _plan_frame_directory(path: Path, files: list[Path]) -> SourcePlan:
    return SourcePlan(path=path, kind="frame_directory", schema=None, frame_count=len(files),
                      depth_available=True, alignment_declared=None,
                      alignment_evidence="directory of per-frame files; alignment is not described",
                      declared_state=False, declared_view=False,
                      notes=("directory of one-frame .npz files ordered by name",),
                      files=tuple(files))


def _plan_colour_only_directory(path: Path) -> SourcePlan:
    contents = list(path.iterdir())
    if any(item.suffix.lower() == ".npz" for item in contents):
        raise UnsupportedInput(f"{path} contains .npz files but none could be planned")
    raise UnsupportedInput(_colour_only_message(path))


def _colour_only_message(path: Path) -> str:
    return (
        f"{path} is a colour-only input. This experiment needs a synchronised RGB-D "
        "sequence (RGB plus depth with intrinsics and a depth unit). It will not fall "
        "back to monocular landmarks or an estimated depth map, because the shared "
        "surface is measured in metres."
    )


def _frame_count(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if "timestamps_s" in keys:
            return int(np.asarray(data["timestamps_s"]).reshape(-1).size)
        if "rgb" in keys:
            rgb = data["rgb"]
            return int(rgb.shape[0]) if rgb.ndim == 4 else 1
    raise InputError(f"{path} has neither timestamps_s nor rgb; cannot count frames")


def _check_depth_and_alignment(plan: SourcePlan, payload: dict[str, Any], config) -> None:
    if plan.kind == "capture_session":
        if payload.get("rgb_depth_aligned") is not True:
            raise UnsupportedInput(
                f"{plan.path} does not declare rgb_depth_aligned: true. Unaligned RGB-D is "
                "rejected instead of being resampled with a guessed transform."
            )
        if payload.get("status") not in (None, "complete"):
            raise InputError(f"{plan.path} capture status is {payload.get('status')!r}, not 'complete'")
        return
    if plan.kind == "frame_manifest":
        aligned = payload.get("rgb_depth_aligned")
        unverified = payload.get("calibration_status") == "session_alignment_unverified"
        if aligned is not True and not (config.input.allow_unverified_alignment and aligned is None and unverified):
            raise UnsupportedInput(
                f"{plan.path} does not declare rgb_depth_aligned: true. Set "
                "input.allow_unverified_alignment to true only if you accept that the session "
                "alignment is unverified and uncalibrated."
            )
        return
    # A prepared sequence already passed its own alignment gate; re-check the schema only.
    if payload.get("schema") != SEQUENCE_SCHEMA:
        raise InputError(f"{plan.path} is not a {SEQUENCE_SCHEMA}")


def _check_npz_segment(plan: SourcePlan, path: Path | None = None) -> None:
    target = path or plan.path
    with np.load(target, allow_pickle=False) as data:
        keys = set(data.files)
        missing = [key for key in ("rgb", "depth", "intrinsics", "timestamps_s") if key not in keys]
        if missing:
            raise UnsupportedInput(
                f"{target} is missing {missing}. A synchronised RGB-D segment must carry rgb, "
                "depth, intrinsics and timestamps_s; a colour-only source is not accepted."
            )


def _calibration_status(plan: SourcePlan) -> str:
    if plan.alignment_declared is True:
        return "declared_by_input_not_independently_calibrated"
    return "alignment_unverified"


def _entries_from_payload(plan: SourcePlan, payload: dict[str, Any], role: str) -> tuple[list[FrameEntry], list[dict]]:
    entries: list[FrameEntry] = []
    dropped: list[dict] = []
    if plan.kind == "sequence":
        rows = [row for row in payload.get("frames", []) if row.get("status") == "candidate"]
        for row in rows:
            frame_file = (plan.path.parent / row["file"]).resolve()
            with np.load(frame_file, allow_pickle=False) as data:
                intrinsics = np.asarray(data["intrinsics"], dtype=np.float64).reshape(-1)
                timestamp = float(np.asarray(data["timestamp_s"]).reshape(-1)[0]) \
                    if "timestamp_s" in data.files else float(row["timestamp_s"])
            entries.append(FrameEntry(
                file=frame_file, index_in_file=None, timestamp_s=timestamp,
                intrinsics=intrinsics, state=row.get("state"), view=row.get("view"),
                group=str(row.get("segment") or row.get("id") or "group_0"),
                source=str(plan.path), source_frame=int(row.get("source_frame", 0)),
                extra={"id": row.get("id"), "segment": row.get("segment"),
                       "angle_group": row.get("angle_group"), "sharpness": row.get("sharpness")}))
        return entries, dropped

    segments_key = "segments" if plan.kind == "capture_session" else "chunks"
    for segment in payload.get(segments_key, []):
        segment_file = (plan.path.parent / segment["file"]).resolve()
        with np.load(segment_file, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        intrinsics = np.asarray(arrays["intrinsics"], dtype=np.float64)
        if intrinsics.ndim == 1:
            intrinsics = np.tile(intrinsics, (int(np.asarray(arrays["timestamps_s"]).size), 1))
        timestamps = np.asarray(arrays["timestamps_s"], dtype=np.float64).reshape(-1)
        group = str(segment.get("segment") or segment.get("view_id") or f"group_{len(entries)}")
        for index in range(len(timestamps)):
            entries.append(FrameEntry(
                file=segment_file, index_in_file=index, timestamp_s=float(timestamps[index]),
                intrinsics=np.asarray(intrinsics[index], dtype=np.float64),
                state=segment.get("state"), view=segment.get("view"), group=group,
                source=str(segment_file), source_frame=index,
                extra={"segment_id": segment.get("id") or segment.get("view_id")}))
    return entries, dropped


def _entries_from_segment(plan: SourcePlan) -> list[FrameEntry]:
    with np.load(plan.path, allow_pickle=False) as data:
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
        timestamps = np.asarray(data["timestamps_s"], dtype=np.float64).reshape(-1)
    if intrinsics.ndim == 1:
        intrinsics = np.tile(intrinsics, (len(timestamps), 1))
    return [FrameEntry(file=plan.path, index_in_file=index, timestamp_s=float(timestamps[index]),
                       intrinsics=np.asarray(intrinsics[index], dtype=np.float64),
                       state=None, view=None, group="segment_000", source=str(plan.path),
                       source_frame=index)
            for index in range(len(timestamps))]


def _entries_from_directory(plan: SourcePlan) -> list[FrameEntry]:
    entries: list[FrameEntry] = []
    for order, frame_file in enumerate(plan.files):
        with np.load(frame_file, allow_pickle=False) as data:
            intrinsics = np.asarray(data["intrinsics"], dtype=np.float64).reshape(-1)
            timestamp = float(np.asarray(data["timestamps_s"]).reshape(-1)[0])
        entries.append(FrameEntry(file=frame_file, index_in_file=None, timestamp_s=timestamp,
                                  intrinsics=intrinsics, state=None, view=None,
                                  group="directory_000", source=str(frame_file),
                                  source_frame=order))
    entries.sort(key=lambda entry: entry.timestamp_s)
    return entries


def _take(array, index: int | None, kind: str, path: Path):
    if array is None:
        return None
    array = np.asarray(array)
    if kind == "rgb":
        if array.ndim == 4:
            return None if index is None else array[index]
        if array.ndim == 3 and array.shape[2] in (3, 4):
            return array[..., :3]
        raise InputError(f"RGB array in {path} has unexpected shape {array.shape}")
    if kind == "depth":
        if array.ndim == 3:
            return None if index is None else array[index]
        if array.ndim == 2:
            return array
        raise InputError(f"depth array in {path} has unexpected shape {array.shape}")
    raise ValueError(f"unknown array kind: {kind}")


def _take_single(array, index: int | None):
    array = np.asarray(array)
    if index is None:
        return array
    if array.ndim >= 3:
        return array[index]
    if array.ndim == 2 and array.shape[0] > 1 and array.shape[0] != array.shape[1]:
        return array[index]
    if array.ndim == 1 and array.size > 1:
        return array[index]
    return array
