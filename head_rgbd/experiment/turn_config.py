"""Frozen configuration sections for the front-anchor turn experiment.

Every section is an explicit dataclass instead of a free-form dict so a typo in
``config/turn_experiment.json`` fails loudly instead of silently falling back to a
default. ``validate()`` performs the checks that must hold across sections.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field as dc_field, fields, is_dataclass
import json
from pathlib import Path
from typing import Any

from head_rgbd.experiment import CONFIG_SCHEMA

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "turn_experiment.json"


class ConfigError(ValueError):
    """The experiment configuration is unusable; never silently repaired."""


@dataclass(frozen=True)
class ExperimentSection:
    name: str = "front_anchor_turn_rgbd"
    version: str = "0.1.0"
    device: str = "cuda"
    seed: int = 20261102
    output_root_name: str = "turn_experiment"
    output_suffix: str = "turn"


@dataclass(frozen=True)
class FlameSection:
    shape_components: int = 50
    expression_components: int = 10


@dataclass(frozen=True)
class InputSection:
    max_adjacent_gap_s: float = 0.2
    allow_unverified_alignment: bool = False
    min_source_frames: int = 3
    min_anchor_frames: int = 1


@dataclass(frozen=True)
class QualitySection:
    min_sharpness: float = 40.0
    reference_sharpness: float = 180.0
    min_face_depth_pixels: int = 200
    reference_face_depth_pixels: int = 2600
    min_valid_landmarks: int = 24
    reference_residual_mm: float = 1.0
    max_residual_mm: float = 8.0


@dataclass(frozen=True)
class TrackingSection:
    detect_stride: int = 3
    klt_window: int = 21
    klt_levels: int = 3
    klt_min_landmarks: int = 40
    klt_forward_backward_px: float = 1.0
    max_yaw_jump_deg: float = 18.0
    max_pitch_deg: float = 28.0
    max_roll_deg: float = 22.0
    max_fit_residual_mm: float = 8.0
    min_valid_upper_landmarks: int = 8
    expression_gap_mm: float = 2.5
    expression_ratio: float = 0.045
    quality: QualitySection = dc_field(default_factory=QualitySection)


@dataclass(frozen=True)
class KeyframeSection:
    max_keyframes: int = 9
    anchor_budget: int = 1
    repeat_observations_per_angle: int = 2
    repeat_max_angle_delta_deg: float = 6.0
    front_yaw_deg: float = 8.0
    anchor_max_yaw_deg: float = 15.0
    angle_bin_deg: float = 10.0
    min_yaw_separation_deg: float = 6.0
    max_yaw_deg: float = 55.0
    min_quality: float = 0.25
    min_selection_score: float = 0.2
    surface_gain_weight: float = 0.5
    side_balance_bonus: float = 0.15


@dataclass(frozen=True)
class FrontInitSection:
    steps: int = 240
    points_per_frame: int = 1200


@dataclass(frozen=True)
class JointShapeSection:
    pose_steps: int = 60
    joint_steps: int = 180
    points_per_frame: int = 1200
    correspondence_interval: int = 10
    landmark_visibility_mm: float = 0.012
    expression_prior_weight: float = 0.05
    jaw_delta_prior_weight: float = 0.05
    expression_bound: float = 0.8
    jaw_delta_bound_rad: float = 0.10
    anchor_shape_prior: float = 0.02
    max_correspondence_depth_mm: float = 8.0
    max_correspondence_plane_mm: float = 5.0
    min_correspondences: int = 30


@dataclass(frozen=True)
class PoseCorrectionSection:
    steps: int = 60


@dataclass(frozen=True)
class FieldSection:
    solver: str = "qp"
    points_per_frame: int = 5000
    outer_steps: int = 2
    gradient_length_mm: float = 8.0
    bending_length_mm: float = 8.0
    low_bending_length_mm: float = 8.0
    displacement_bound_mm: float = 2.5
    maximum_slope: float = 0.45
    maximum_normal_change_deg: float = 35.0
    minimum_signed_area_ratio: float = 0.05
    noise_floor_mm: float = 0.28867513459481287
    research_v2: bool = True
    fixed_ray_policy: str = "strict-coverage"
    support_min_information: float = 0.5
    coverage_allowance_ratio: float = 0.001
    information_measure: str = "barycentric"
    information_damping_cap: float = 10.0
    view_balance: bool = True
    loop_target_blend: float = 0.5
    probe_points_per_frame: int = 1500
    probe_max_depth_mm: float = 30.0


@dataclass(frozen=True)
class DeliverySection:
    keep_stage_directories: bool = True
    metrics_to_watch: tuple[str, ...] = (
        "reconstruction_elapsed_s",
        "per_view_depth_consistency_mm",
        "side_geometry_change_mm",
        "surface_smoothness_mm_per_mm",
    )


@dataclass(frozen=True)
class TurnConfig:
    experiment: ExperimentSection = dc_field(default_factory=ExperimentSection)
    flame: FlameSection = dc_field(default_factory=FlameSection)
    input: InputSection = dc_field(default_factory=InputSection)
    tracking: TrackingSection = dc_field(default_factory=TrackingSection)
    keyframes: KeyframeSection = dc_field(default_factory=KeyframeSection)
    front_init: FrontInitSection = dc_field(default_factory=FrontInitSection)
    joint_shape: JointShapeSection = dc_field(default_factory=JointShapeSection)
    pose_correction: PoseCorrectionSection = dc_field(default_factory=PoseCorrectionSection)
    field: FieldSection = dc_field(default_factory=FieldSection)
    delivery: DeliverySection = dc_field(default_factory=DeliverySection)
    source_path: str | None = None
    schema: str = CONFIG_SCHEMA

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path | None = None) -> "TurnConfig":
        target = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG_PATH
        if not target.is_file():
            raise ConfigError(f"experiment configuration not found: {target}")
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"experiment configuration is not valid JSON: {exc}") from exc
        return cls.from_dict(payload, source_path=str(target))

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, source_path: str | None = None) -> "TurnConfig":
        if not isinstance(payload, dict):
            raise ConfigError("experiment configuration must be a JSON object")
        schema = payload.get("schema", CONFIG_SCHEMA)
        if schema != CONFIG_SCHEMA:
            raise ConfigError(f"unsupported experiment configuration schema: {schema!r}")
        known = {item.name for item in fields(cls)} - {"source_path", "schema"}
        unknown = sorted(set(payload) - known - {"schema", "note"})
        if unknown:
            raise ConfigError(f"unknown experiment configuration section(s): {', '.join(unknown)}")
        config = cls(
            experiment=_section(ExperimentSection, payload.get("experiment")),
            flame=_section(FlameSection, payload.get("flame")),
            input=_section(InputSection, payload.get("input")),
            tracking=_tracking(payload.get("tracking")),
            keyframes=_section(KeyframeSection, payload.get("keyframes")),
            front_init=_section(FrontInitSection, payload.get("front_init")),
            joint_shape=_section(JointShapeSection, payload.get("joint_shape")),
            pose_correction=_section(PoseCorrectionSection, payload.get("pose_correction")),
            field=_section(FieldSection, payload.get("field")),
            delivery=_delivery(payload.get("delivery")),
            source_path=source_path,
            schema=schema,
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        payload = {key: value for key, value in asdict(self).items() if key != "source_path"}
        payload["delivery"]["metrics_to_watch"] = list(payload["delivery"]["metrics_to_watch"])
        return payload

    def replace(self, **overrides: Any) -> "TurnConfig":
        payload = {key: value for key, value in asdict(self).items() if key != "source_path"}
        for key, value in overrides.items():
            if key == "device":
                payload["experiment"]["device"] = value
            elif key in payload and isinstance(payload[key], dict):
                payload[key].update(value)
            else:
                payload[key] = value
        return TurnConfig.from_dict(json.loads(json.dumps(payload)), source_path=self.source_path)

    # ----------------------------------------------------------------- checking
    def validate(self) -> None:
        if self.schema != CONFIG_SCHEMA:
            raise ConfigError(f"unsupported experiment configuration schema: {self.schema!r}")
        if self.experiment.device not in ("cpu", "cuda"):
            raise ConfigError("experiment.device must be 'cpu' or 'cuda'")
        if not self.experiment.output_root_name or not self.experiment.output_suffix:
            raise ConfigError("experiment output names must not be empty")

        if not 10 <= self.flame.shape_components <= 300:
            raise ConfigError("flame.shape_components must be within [10, 300]")
        if not 1 <= self.flame.expression_components <= 50:
            raise ConfigError("flame.expression_components must be within [1, 50]")

        if self.input.max_adjacent_gap_s <= 0:
            raise ConfigError("input.max_adjacent_gap_s must be positive")
        if self.input.min_source_frames < 2:
            raise ConfigError("input.min_source_frames must be at least 2")
        if self.input.min_anchor_frames < 1:
            raise ConfigError("input.min_anchor_frames must be at least 1")

        tracking = self.tracking
        if tracking.detect_stride < 1:
            raise ConfigError("tracking.detect_stride must be at least 1")
        if tracking.klt_window < 5 or tracking.klt_window % 2 == 0:
            raise ConfigError("tracking.klt_window must be an odd window size of at least 5")
        if tracking.klt_levels < 0:
            raise ConfigError("tracking.klt_levels must not be negative")
        if tracking.klt_min_landmarks < 6:
            raise ConfigError("tracking.klt_min_landmarks must be at least 6")
        for name, value in (("max_yaw_jump_deg", tracking.max_yaw_jump_deg),
                            ("max_pitch_deg", tracking.max_pitch_deg),
                            ("max_roll_deg", tracking.max_roll_deg),
                            ("max_fit_residual_mm", tracking.max_fit_residual_mm)):
            if value <= 0:
                raise ConfigError(f"tracking.{name} must be positive")
        if tracking.min_valid_upper_landmarks < 4:
            raise ConfigError("tracking.min_valid_upper_landmarks must be at least 4")
        quality = tracking.quality
        if quality.reference_sharpness <= quality.min_sharpness:
            raise ConfigError("tracking.quality.reference_sharpness must exceed min_sharpness")
        if quality.reference_face_depth_pixels <= quality.min_face_depth_pixels:
            raise ConfigError("tracking.quality.reference_face_depth_pixels must exceed min_face_depth_pixels")
        if quality.max_residual_mm <= quality.reference_residual_mm:
            raise ConfigError("tracking.quality.max_residual_mm must exceed reference_residual_mm")

        keyframes = self.keyframes
        if not 1 <= keyframes.max_keyframes <= 12:
            raise ConfigError("keyframes.max_keyframes must be within [1, 12]")
        if not 0 <= keyframes.anchor_budget <= keyframes.max_keyframes:
            raise ConfigError("keyframes.anchor_budget must fit inside max_keyframes")
        if not 0 <= keyframes.repeat_observations_per_angle <= 4:
            raise ConfigError("keyframes.repeat_observations_per_angle must be within [0, 4]")
        if keyframes.repeat_max_angle_delta_deg <= 0:
            raise ConfigError("keyframes.repeat_max_angle_delta_deg must be positive")
        if keyframes.angle_bin_deg <= 0:
            raise ConfigError("keyframes.angle_bin_deg must be positive")
        if not 0 <= keyframes.front_yaw_deg <= keyframes.anchor_max_yaw_deg <= keyframes.max_yaw_deg:
            raise ConfigError("keyframe yaw limits must satisfy front <= anchor_max <= max_yaw")
        if keyframes.min_yaw_separation_deg < 0:
            raise ConfigError("keyframes.min_yaw_separation_deg must not be negative")
        if not 0 <= keyframes.min_quality <= 1 or not 0 <= keyframes.min_selection_score <= 1:
            raise ConfigError("keyframe quality/score floors must be within [0, 1]")
        if keyframes.surface_gain_weight < 0 or keyframes.side_balance_bonus < 0:
            raise ConfigError("keyframe selection weights must not be negative")

        if self.front_init.steps < 40:
            raise ConfigError("front_init.steps must be at least 40")
        if self.front_init.points_per_frame < 100:
            raise ConfigError("front_init.points_per_frame must be at least 100")

        joint = self.joint_shape
        if joint.pose_steps < 0 or joint.joint_steps < 20:
            raise ConfigError("joint_shape.joint_steps must be at least 20")
        if joint.points_per_frame < 100:
            raise ConfigError("joint_shape.points_per_frame must be at least 100")
        if joint.correspondence_interval < 1:
            raise ConfigError("joint_shape.correspondence_interval must be at least 1")
        if not 0 < joint.expression_bound <= 4:
            raise ConfigError("joint_shape.expression_bound must be within (0, 4]")
        if not 0 < joint.jaw_delta_bound_rad <= 0.7:
            raise ConfigError("joint_shape.jaw_delta_bound_rad must be within (0, 0.7]")
        if joint.min_correspondences < 10:
            raise ConfigError("joint_shape.min_correspondences must be at least 10")

        if self.pose_correction.steps < 10:
            raise ConfigError("pose_correction.steps must be at least 10")

        volume = self.field
        if volume.solver not in ("qp", "least-squares"):
            raise ConfigError("field.solver must be 'qp' or 'least-squares'")
        if volume.points_per_frame < 500:
            raise ConfigError("field.points_per_frame must be at least 500")
        if volume.outer_steps < 1:
            raise ConfigError("field.outer_steps must be at least 1")
        if min(volume.gradient_length_mm, volume.bending_length_mm, volume.low_bending_length_mm) <= 0:
            raise ConfigError("field regularization lengths must be positive")
        if not 0 < volume.displacement_bound_mm <= 3:
            raise ConfigError("field.displacement_bound_mm must be within (0, 3]")
        if volume.low_bending_length_mm > volume.bending_length_mm:
            raise ConfigError("field.low_bending_length_mm must not exceed bending_length_mm")
        if not 0 < volume.maximum_slope:
            raise ConfigError("field.maximum_slope must be positive")
        if not 0 < volume.maximum_normal_change_deg <= 60:
            raise ConfigError("field.maximum_normal_change_deg must be within (0, 60]")
        if not 0 < volume.minimum_signed_area_ratio <= 1:
            raise ConfigError("field.minimum_signed_area_ratio must be within (0, 1]")
        if volume.noise_floor_mm <= 0:
            raise ConfigError("field.noise_floor_mm must be positive")
        if volume.fixed_ray_policy not in ("strict-coverage", "accounted-research"):
            raise ConfigError("field.fixed_ray_policy is unknown")
        if volume.fixed_ray_policy != "strict-coverage" and not volume.research_v2:
            raise ConfigError("field accounted ray policy needs research_v2")
        if volume.information_measure not in ("barycentric", "curvature"):
            raise ConfigError("field.information_measure is unknown")
        if min(volume.support_min_information, volume.coverage_allowance_ratio,
               volume.information_damping_cap) < 0:
            raise ConfigError("field information budgets must not be negative")
        if not 0 < volume.loop_target_blend <= 1:
            raise ConfigError("field.loop_target_blend must be within (0, 1]")
        if volume.probe_points_per_frame < 100:
            raise ConfigError("field.probe_points_per_frame must be at least 100")
        if volume.probe_max_depth_mm <= 0:
            raise ConfigError("field.probe_max_depth_mm must be positive")


def _section(kind, payload: dict[str, Any] | None):
    if payload is None:
        return kind()
    if not isinstance(payload, dict):
        raise ConfigError(f"{kind.__name__} must be a JSON object")
    names = {item.name for item in fields(kind)}
    unknown = sorted(set(payload) - names)
    if unknown:
        raise ConfigError(f"unknown keys in {kind.__name__}: {', '.join(unknown)}")
    values = {}
    for item in fields(kind):
        if item.name not in payload:
            continue
        if is_dataclass(item.default) or callable(item.default_factory):
            raise ConfigError(f"{kind.__name__}.{item.name} must be built by its own loader")
        values[item.name] = _coerce(item.type, payload[item.name], f"{kind.__name__}.{item.name}")
    return kind(**values)


def _tracking(payload: dict[str, Any] | None) -> TrackingSection:
    if payload is None:
        return TrackingSection()
    payload = dict(payload)
    quality = payload.pop("quality", None)
    section = _section(TrackingSection, payload) if payload else TrackingSection()
    return TrackingSection(**{**asdict(section), "quality": _section(QualitySection, quality)})


def _delivery(payload: dict[str, Any] | None) -> DeliverySection:
    if payload is None:
        return DeliverySection()
    payload = dict(payload)
    metrics = payload.pop("metrics_to_watch", None)
    section = _section(DeliverySection, payload) if payload else DeliverySection()
    values = asdict(section)
    if metrics is not None:
        if not isinstance(metrics, list) or not all(isinstance(item, str) and item for item in metrics):
            raise ConfigError("delivery.metrics_to_watch must be a list of non-empty strings")
        values["metrics_to_watch"] = tuple(metrics)
    return DeliverySection(**values)


def _coerce(kind, value: Any, label: str):
    if isinstance(kind, str):
        # ``from __future__ import annotations`` leaves the annotation as a string.
        if kind in ("int", "bool", "float", "str"):
            return _primitives(kind, value, label)
        return value
    return _primitives(getattr(kind, "__name__", ""), value, label)


def _primitives(kind: str, value: Any, label: str):
    if kind == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{label} must be true or false")
        return value
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{label} must be an integer")
        if float(value) != int(value):
            raise ConfigError(f"{label} must be a whole number")
        return int(value)
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{label} must be a number")
        return float(value)
    if kind == "str":
        if not isinstance(value, str):
            raise ConfigError(f"{label} must be a string")
        return value
    return value
