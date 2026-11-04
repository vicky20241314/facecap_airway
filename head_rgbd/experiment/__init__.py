"""Independent experiment: frontal RGB-D anchor plus a continuous head-turn RGB-D sequence.

This package is deliberately separate from :mod:`head_rgbd.pipeline`. It adds no
CLI subcommand to :mod:`head_rgbd.cli`, changes no default parameter and writes
only into its own run directory, so the production chain keeps its behaviour.

Entry point::

    python -m head_rgbd.experiment.turn_run --anchor <path> --turn <path>

The stage modules are:

``turn_config``
    Frozen configuration sections and validation.
``turn_inputs``
    Input discovery, explicit RGB-video versus synchronised-RGB-D rejection.
``turn_tracking``
    Lightweight continuous per-frame yaw/pitch/roll tracking and continuity groups.
``turn_keyframes``
    Angle-bin plus quality keyframe selection with a hard budget.
``turn_prepare``
    Materialise exactly the selected keyframes as fitting observations.
``turn_shape``
    One shared neutral identity fitted to all selected views at once.
``turn_field``
    One low/mid-frequency displacement field and the final smooth surface.
``turn_run``
    Command line orchestration and delivery assembly.
"""
from __future__ import annotations

EXPERIMENT = "front_anchor_turn_rgbd"
EXPERIMENT_VERSION = "0.1.0"
CONFIG_SCHEMA = "head_rgbd.turn_experiment.config.v1"

__all__ = ["EXPERIMENT", "EXPERIMENT_VERSION", "CONFIG_SCHEMA"]
