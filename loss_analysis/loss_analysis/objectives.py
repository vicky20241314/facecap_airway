"""Stage/term taxonomy transcribed from the actual delivery source.

Every entry here was read from `head_rgbd/*.py` (delivery commit) and records the
source location, the exact normalisation, the weight and its unit. Nothing is
invented: a term that the source does not compute is simply absent, and a term
whose value cannot be reconstructed offline is marked ``UNRECOVERABLE``.

Reading rules used below
------------------------
* ``smooth_l1(x)`` is PyTorch ``smooth_l1_loss(x, 0, beta=0.3)`` on **already
  normalised** residuals (the source divides by a unit scale before calling it),
  so its value is dimensionless and per-element (``reduction='mean'``).
* Frame weight ``frame_weight = view_weight[i] / denominator`` with
  ``view_weight = 3`` for front frames during an identity phase else ``1``.
* A stage's scalar objective is the Adam loss of that step. It is a *penalised
  surrogate*, not the metric residual; it is never called "error".
"""
from __future__ import annotations

from dataclasses import dataclass, field


# --- provenance tiers -------------------------------------------------------
SOLVER_OBJECTIVE = "solver_objective"          # the scalar the optimiser minimised
LINEARISED = "linearised_objective"            # frozen-correspondence linear system value
NONLINEAR_ENERGY = "nonlinear_energy"          # re-evaluated with refreshed correspondences
DIAGNOSTIC = "diagnostic"                      # our own metric, never the algorithm's loss

# --- availability -----------------------------------------------------------
RECORDED = "recorded"          # taken verbatim from a real run artefact
RECOMPUTED = "recomputed"      # recomputed by this tool from real arrays
NOT_USED = "not_used"          # the algorithm does not have this term
NULL = "null"                  # the algorithm has it but the value is unavailable


@dataclass(frozen=True)
class Term:
    """One additive summand of a stage objective."""

    key: str
    label_cn: str
    source: str                 # file:line in the delivery source
    kind: str                   # "data" | "prior" | "temporal" | "state"
    normaliser: str             # denominator / unit scale applied before scoring
    normaliser_unit: str        # "", "mm", "px", "1"
    weight: str                 # constant weight multiplying the term
    weight_unit: str            # "", "1/mm^2"
    reduction: str              # how samples collapse to a scalar
    denominator: str            # what the reduction divides by
    notes: str = ""


@dataclass(frozen=True)
class Stage:
    """A stage of the single default algorithm and what it actually optimises."""

    key: str
    label_cn: str
    module: str
    parameters: tuple[str, ...]
    terms: tuple[Term, ...]
    objective_scope: str
    solver: str
    has_shared_total: bool = False   # False => stages have no common total loss
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModeVariant:
    """One (stage, mode) combination with a different objective than its sibling.

    v0.4.0 makes the SAME stage key produce different objectives depending on a
    flag, and in one case (``identity_v2``) the flag is not even forwarded by the
    pipeline. A variant entry is therefore the only place where "which objective
    does this stage key mean in this run" is answered, and it is answered from
    artefacts, never from the command line.
    """

    variant_key: str               # the name the run itself writes into the log/report
    stage_key: str                 # the catalogue stage it belongs to
    label_cn: str
    activated_by: str              # what makes the run choose it
    data_frames: str               # which frames contribute data equations
    term_overrides: str            # how the term set differs from the parent stage
    evidence: str                  # where the run records which variant it took


# --------------------------------------------------------------------------
# fit_c: front anchor and guarded shared identity (Adam, per-step scalar loss)
# --------------------------------------------------------------------------
FIT_C_TERMS = (
    Term(
        key="landmark_xyz_stable",
        label_cn="稳定标点三维残差",
        source="head_rgbd/fit_c.py:200 (v0.4.0)",
        kind="data",
        normaliser="residual / 0.01 m",
        normaliser_unit="mm->1 (÷10 mm)",
        weight="frame_weight * landmark_weight * 1.0, landmark_weight=.2 when an anchor exists and stage>0",
        weight_unit="1",
        reduction="smooth_l1 mean over valid stable landmarks",
        denominator="count of valid stable landmarks in the frame",
        notes="indices 0..59 (inner lip 60..67 moved to its own term); indices <17 always disabled; "
              "v0.4.0 additionally multiplies by 0.2 (fit_c.py:199) in every post-anchor stage",
    ),
    Term(
        key="landmark_xyz_inner_lip",
        label_cn="内唇标点三维残差",
        source="head_rgbd/fit_c.py:202 (v0.4.0)",
        kind="data",
        normaliser="residual / 0.01 m",
        normaliser_unit="mm->1 (÷10 mm)",
        weight="frame_weight * 0.25",
        weight_unit="1",
        reduction="smooth_l1 mean",
        denominator="count of valid inner-lip landmarks",
        notes="down-weighted because cavity-edge depth is noisy",
    ),
    Term(
        key="landmark_uv",
        label_cn="标点二维重投影残差",
        source="head_rgbd/fit_c.py:208 (v0.4.0)",
        kind="data",
        normaliser="(uv - uv_target) / 10 px",
        normaliser_unit="px->1",
        weight="frame_weight * landmark_weight * 0.15",
        weight_unit="1",
        reduction="smooth_l1 mean",
        denominator="count of valid 2D landmarks",
        notes="needs no depth; oblique frames keep indices 48..67 disabled",
    ),
    Term(
        key="mouth_uv_front",
        label_cn="正面口部二维残差",
        source="head_rgbd/fit_c.py:212 (v0.4.0)",
        kind="data",
        normaliser="(uv - uv_target) / 10 px",
        normaliser_unit="px->1",
        weight="frame_weight * 0.35",
        weight_unit="1",
        reduction="smooth_l1 mean",
        denominator="count of valid mouth landmarks (48..67)",
        notes="front frames only, non-identity phases only",
    ),
    Term(
        key="mouth_gap_closed",
        label_cn="闭口内唇贴合 hinge",
        source="head_rgbd/fit_c.py:216 (v0.4.0)",
        kind="state",
        normaliser="relu(gap - 2 mm) / 3 mm",
        normaliser_unit="mm->1",
        weight="frame_weight * 0.15",
        weight_unit="1",
        reduction="squared mean over 3 inner-lip pairs",
        denominator="3 pairs",
        notes="closed-state frames only; keeps the mouth shut where the state is known",
    ),
    Term(
        key="depth_surface_normal",
        label_cn="真实深度射线法向残差",
        source="head_rgbd/fit_c.py:222 (v0.4.0)",
        kind="data",
        normaliser="residual_mm / 10 mm",
        normaliser_unit="mm->1",
        weight="frame_weight * 1.4 * depth_weight, depth_weight=1 for a front view else --side-depth-weight",
        weight_unit="1",
        reduction="smooth_l1 mean over accepted triangle correspondences",
        denominator="number of accepted correspondences in the frame",
        notes="frozen between correspondences refreshes; only phases > 'motion_initial'; "
              "v0.4.0 adds --side-depth-weight (fit_c.py:221), so side-view depth can be discounted "
              "independently of the landmark terms",
    ),
    Term(
        key="shape_prior",
        label_cn="身份形状先验",
        source="head_rgbd/fit_c.py:224 (v0.4.0)",
        kind="prior",
        normaliser="shape^2 (raw FLAME coefficients)",
        normaliser_unit="coefficient^2",
        weight="0.002",
        weight_unit="1",
        reduction="mean over coefficients",
        denominator="number of shape coefficients",
    ),
    Term(
        key="expression_prior",
        label_cn="表情先验",
        source="head_rgbd/fit_c.py:224 (v0.4.0)",
        kind="prior",
        normaliser="expr^2",
        normaliser_unit="coefficient^2",
        weight="0.002",
        weight_unit="1",
        reduction="sum over all values / (n_frames * 10)",
        denominator="n_frames * 10 (NOT the component count)",
        notes="asymmetric with shape_prior; documented as-is, not 'fixed'. Because the denominator is "
              "the total frame count, the SAME formula is weaker on a run that has more frames",
    ),
    Term(
        key="jaw_prior",
        label_cn="下颌先验",
        source="head_rgbd/fit_c.py:224 (v0.4.0)",
        kind="prior",
        normaliser="jaw^2",
        normaliser_unit="rad^2",
        weight="0.01",
        weight_unit="1",
        reduction="mean over (frames, 3)",
        denominator="n_frames * 3",
    ),
    Term(
        key="joint_shape_prior",
        label_cn="联合身份回拉先验（v0.4.0 新增）",
        source="head_rgbd/fit_c.py:228 (v0.4.0)",
        kind="prior",
        normaliser="(shape - anchor_shape)^2 in raw FLAME coefficients",
        normaliser_unit="coefficient^2",
        weight="--joint-shape-prior, default 0.02",
        weight_unit="1",
        reduction="mean over coefficients",
        denominator="number of shape coefficients",
        notes="only when --shape-fusion joint (a front anchor exists and joint=True); this is the ONLY "
              "term that still ties the shared identity to the frontal fit, because joint mode removes "
              "the hard front lock and gives every view the same weight",
    ),
    Term(
        key="temporal_smoothness",
        label_cn="相邻帧时间平滑",
        source="head_rgbd/fit_c.py:232 (v0.4.0)",
        kind="temporal",
        normaliser="(coefficient difference)^2",
        normaliser_unit="rad^2 / coefficient^2",
        weight="factor = (0.033 / dt)^2, then 0.005 / 0.02 / 0.01 per group",
        weight_unit="1/s^2 * 1",
        reduction="sum over temporal edges / n_edges",
        denominator="number of temporal edges (frames inside one tracked segment only)",
        notes="skipped entirely during identity phases",
    ),
)

FIT_C = Stage(
    key="fit_c",
    label_cn="共享身份拟合（正面锚 / 受保护身份，Adam）",
    module="head_rgbd.fit_c",
    parameters=("shape(1x50)", "expression(nx30)", "jaw(nx3)", "rotation(nx3)", "translation(nx3)"),
    terms=FIT_C_TERMS,
    objective_scope=(
        "one scalar per optimiser step; each data term is averaged inside its own frame and "
        "then multiplied by frame_weight, so the total is a weighted mean over frames"
    ),
    solver="torch.optim.Adam, 5 np.rint phase cuts of --steps, grad clip 10",
    has_shared_total=False,
    notes=(
        "the 'front-anchor' run solves the same objective restricted to front frames",
        "identity phases zero the expression/jaw gradients, so those terms only scale the value",
        "no convergence threshold is applied; the run always executes exactly --steps iterations",
        "v0.4.0 frame_weight rule (fit_c.py:166-172): in joint mode EVERY active view gets 1.0 during an "
        "identity phase, so the front view loses the 3x it has in guarded mode; outside an identity phase "
        "front stays 3.0",
        "v0.4.0 phase names carry a 'joint_' prefix when --shape-fusion joint, so guarded and joint curves "
        "can never be silently stitched together",
        "v0.4.0 joint additionally reports joint_anchor_update_mm and joint_shape_delta_from_anchor. The "
        "former is NOT a shape figure: it also contains the pose and expression freedom of the anchored "
        "frames (9-11 mm on the reference capture), so only joint_shape_delta_from_anchor may be read as "
        "how far the shared identity moved away from the front anchor",
        "the 5 phase boundaries change the ACTIVE TERM SET (landmark-only phase 0, then ray-surface "
        "correspondences, then identity phases without expression/jaw gradients); the objective is "
        "therefore only comparable WITHIN one phase",
    ),
)

# --------------------------------------------------------------------------
# motion_v2: per-frame pose/expression/jaw with fixed identity
# --------------------------------------------------------------------------
MOTION_V2_TERMS = (
    Term(
        key="depth_surface_normal",
        label_cn="固定射线法向深度残差",
        source="head_rgbd/motion_v2.py:114",
        kind="data",
        normaliser="residual / 0.002 m (2 mm)",
        normaliser_unit="mm->1 (÷2 mm)",
        weight="1.0",
        weight_unit="1",
        reduction="smooth_l1(beta=1) over correspondences, divided by n_frames",
        denominator="n_frames (not the correspondence count)",
        notes="correspondences refreshed every 10 steps with max_plane 5 mm",
    ),
    Term(
        key="landmark_xyz",
        label_cn="标点三维残差",
        source="head_rgbd/motion_v2.py:118",
        kind="data",
        normaliser="residual / 0.003 m (3 mm)",
        normaliser_unit="mm->1 (÷3 mm)",
        weight="0.12",
        weight_unit="1",
        reduction="smooth_l1 mean, divided by n_frames",
        denominator="n_frames",
        notes="indices <17 and 48..67 excluded",
    ),
    Term(
        key="expression_uv",
        label_cn="表情阶段二维残差",
        source="head_rgbd/motion_v2.py:124",
        kind="data",
        normaliser="uv / 3 px",
        normaliser_unit="px->1",
        weight="0.08",
        weight_unit="1",
        reduction="smooth_l1 mean, divided by n_frames",
        denominator="n_frames",
        notes="only inside the expression phase (second quarter) on indices 48..67",
    ),
    Term(
        key="mouth_gap_closed",
        label_cn="闭口 hinge",
        source="head_rgbd/motion_v2.py:127",
        kind="state",
        normaliser="relu(gap - 2 mm) / 3 mm",
        normaliser_unit="mm->1",
        weight="0.1",
        weight_unit="1",
        reduction="squared mean over 3 pairs, divided by n_frames",
        denominator="n_frames",
        notes="expression phase, closed frames only",
    ),
    Term(
        key="pose_prior",
        label_cn="旋转/平移先验",
        source="head_rgbd/motion_v2.py:129",
        kind="prior",
        normaliser="(dr/0.02 rad)^2 , (dt/0.003 m)^2",
        normaliser_unit="1",
        weight="0.01 / 0.01",
        weight_unit="1",
        reduction="mean per parameter group",
        denominator="n_frames",
    ),
    Term(
        key="expression_jaw_prior",
        label_cn="表情/下颌先验",
        source="head_rgbd/motion_v2.py:130",
        kind="prior",
        normaliser="(de/0.2)^2 , (dj/0.03 rad)^2",
        normaliser_unit="1",
        weight="0.02 / 0.02",
        weight_unit="1",
        reduction="mean per parameter group",
        denominator="n_frames",
    ),
    Term(
        key="temporal_smoothness",
        label_cn="时间平滑",
        source="head_rgbd/motion_v2.py:132",
        kind="temporal",
        normaliser="(de_j-de_i)^2 + (dj_j-dj_i)^2",
        normaliser_unit="rad^2",
        weight="0.002",
        weight_unit="1",
        reduction="sum / n_edges",
        denominator="number of temporal edges",
    ),
)

MOTION_V2 = Stage(
    key="motion_v2",
    label_cn="逐帧姿态/表情/下颌（固定身份，Adam）",
    module="head_rgbd.motion_v2",
    parameters=("dr(nx3)", "dt(nx3)", "de(nx30)", "dj(nx3)"),
    terms=MOTION_V2_TERMS,
    objective_scope="one scalar per step; every data term is divided by n_frames (not by its own sample count)",
    solver="torch.optim.Adam; hard per-frame caps |dr|<=0.02 rad, |dt|<=3 mm, |de|<=0.2, |dj|<=0.03",
    has_shared_total=False,
    notes=(
        "the pose and expression gradients are alternated by phase; the loss still sums both groups",
        "acceptance is NOT a loss comparison: each frame is re-scored on its ORIGINAL fixed rays "
        "and a half-step ladder 1/1,1/2,1/4,1/8,0 keeps the first candidate with "
        "mse <= old_mse + 1e-8 AND full coverage of the required rays",
        "rejected frames fall back to alpha=0, i.e. the old motion is retained",
    ),
)

# --------------------------------------------------------------------------
# identity_v2 / fit_d: constrained linear least squares over a scalar field
# --------------------------------------------------------------------------
FIELD_TERMS = (
    Term(
        key="depth_field_equation",
        label_cn="射线深度方程（加权）",
        source="head_rgbd/geometry_d.py:314-327,347-349",
        kind="data",
        normaliser="b in mm; rw = sqrt(w / max(1, |A d - b| / 2))",
        normaliser_unit="mm",
        weight="w = area_quadrature * frame_weight * incidence^2 * distance taper * "
               "(noise_floor / sigma)^2 / sigma_proj^2",
        weight_unit="1/mm^2",
        reduction="sum of squares over all accepted equations",
        denominator="weight itself (no extra averaging); area quadrature stops ray duplication from outvoting a face",
        notes="canonical targets (b / sensitivity) only enter the cross-frame noise estimate; the solved b stays in mm",
    ),
    Term(
        key="soft_lock_confidence",
        label_cn="软锁置信权重（v0.4.0 新增，不新增项，只改权重）",
        source="head_rgbd/geometry_d.py:314 (v0.4.0)",
        kind="prior",
        normaliser="w *= soft_lock[vertex].mean(over the rays hitting that vertex)",
        normaliser_unit="1",
        weight="--joint-front-confidence, default 0.15 on front-protected vertices, 1.0 elsewhere",
        weight_unit="1",
        reduction="scales the existing depth equation weights; the vertex stays in the free set",
        denominator="unchanged",
        notes="NOT an extra additive term: it replaces the hard locked= mask. A vertex at confidence 0.15 "
              "keeps 15% of its own data weight, so observations can outvote the frontal prior instead of "
              "being removed from the linear system",
    ),
    Term(
        key="fem_total_regularizer",
        label_cn="FEM 总形变能 + 标量梯度",
        source="head_rgbd/surface_v2.py:27-59 (FEM operators + total_regularizer)",
        kind="prior",
        normaliser="millimetres of displacement; biases use (base - reference)",
        normaliser_unit="mm",
        weight="gradient_length_mm=3 (or 8 for low frequency), 0.08 damping, bending scaled by 1/sqrt(n_states)",
        weight_unit="1",
        reduction="sum of squares of the stacked regulariser rows",
        denominator="mass-normalised (lumped area quadrature)",
        notes="research_v2 geometry; penalises curvature CHANGE from the original prior, not facial curvature",
    ),
    Term(
        key="dense_prior",
        label_cn="密集层先验",
        source="head_rgbd/geometry_d.py:213",
        kind="prior",
        normaliser="sqrt(mass / total_mass) * field",
        normaliser_unit="mm",
        weight="dense_prior_strength = 2.0",
        weight_unit="1",
        reduction="sum of squares",
        denominator="total mass",
        notes="only when reference_base is given (hierarchical / loop-research)",
    ),
)

FIELD_STAGE = Stage(
    key="field",
    label_cn="共享标量场（约束最小二乘 / QP）",
    module="head_rgbd.geometry_d.solve_field + head_rgbd.qp_d",
    parameters=("field(count,)  # one scalar per vertex, millimetres along the animated normal",),
    terms=FIELD_TERMS,
    objective_scope=(
        "the QP minimises ||rw*(A f - b)||^2 + ||R f + bias||^2 subject to a hard feasible set; "
        "that quadratic is the LINEARISED objective under frozen correspondences"
    ),
    solver="OSQP (eps 1e-6) or scipy lsq_linear; 18-step global halving line search per outer step",
    has_shared_total=False,
    notes=(
        "acceptance requires: geometric safety AND candidate objective <= previous + 1e-12 AND "
        "(research_v2) fixed-ray coverage non-decreasing AND penalised MSE <= before*1.02 + 0.002",
        "v0.5.0 replaced the absolute zero-tolerance coverage rule with a per-stage allowance "
        "(--coverage-allowance-ratio, default 0.001): fixed_ray_diagnostic now records "
        "missing_allowance / missing_cap, so a step with a handful of missing rays can be accepted. "
        "A report must carry that number next to the MSE, because the same step would have been refused "
        "under the 0.2.0-0.4.0 rule",
        "the fixed-ray MSE/coverage check is a NONLINEAR energy recomputed by ray casting, "
        "not part of the solver quadratic",
        "the solved field is millimetres; exported geometry converts with *.001",
        "v0.4.0: `soft_lock` multiplies the depth-equation weight and keeps the vertex INSIDE the free "
        "set, while `locked=` removes it from the unknown vector. Therefore `free_vertices` counts a "
        "0.15-confidence vertex as free and is not comparable between a locked stage and a confidence "
        "stage, even when both report the same number",
        "v0.4.0: both solvers implement both branches (fit_d.py:114-124, identity_v2.py:32-44); which one "
        "ran is a property of the STAGE REPORT, not of the chain flag, because the pipeline does not "
        "forward --shape-fusion everywhere (see MODE_VARIANTS)",
        "`status=skipped_no_free_vertices` means the stage RAN and found nothing solvable: its objective "
        "curve is empty by construction, not converged. `status=not_executed` means the stage was bypassed "
        "by a switch (v0.4.0 --fast). Neither may be plotted as a zero",
    ),
)

# --------------------------------------------------------------------------
# fit_d research-v2 stage sequence (the delivery chain)
# --------------------------------------------------------------------------
FIT_D_STAGES = (
    Stage(
        key="fit_d.front_anchor_low_frequency",
        label_cn="密集层：正面锚 低频",
        module="head_rgbd/fit_d.py:78-88 -> geometry_d.solve_field",
        parameters=FIELD_STAGE.parameters,
        terms=FIELD_TERMS,
        objective_scope="linearised quadratic under frozen correspondences (low frequency: gradient/bending 8 mm, 2.5 mm bound, 2 outers)",
        solver=FIELD_STAGE.solver,
        notes=("same objective shape as identity_v2 but on the coarse topology",
               "v0.4.0 joint: the control field is solved from EVERY closed identity view "
               "(fit_d.py:60-64, shape_frames=identity), while the stage keeps its historical "
               "'front_anchor_*' name for log compatibility; the label must not be read as 'front-only', "
               "and its data term therefore averages over a different frame set than in guarded mode",),
    ),
    Stage(
        key="fit_d.front_anchor_mid_frequency",
        label_cn="密集层：正面锚 中频",
        module="head_rgbd/fit_d.py:89-91",
        parameters=FIELD_STAGE.parameters,
        terms=FIELD_TERMS,
        objective_scope="same quadratic, displacement bound relaxed to 3 mm, initialised from the low-frequency field",
        solver=FIELD_STAGE.solver,
        notes=("v0.4.0 --fast reduces this round to --mid-rounds; the round is stored NESTED inside "
               "field_logs['front_anchor_regularized']['mid_frequency'], not as its own key",),
    ),
    Stage(
        key="fit_d.guarded_regularized",
        label_cn="密集层：受保护正则场",
        module="head_rgbd/fit_d.py:92-100",
        parameters=FIELD_STAGE.parameters,
        terms=FIELD_TERMS,
        objective_scope="same quadratic over all identity frames with front-protected vertices locked",
        solver=FIELD_STAGE.solver,
        notes=("this is the guarded (non-joint) path; with --shape-fusion joint the stage is recorded as "
               "joint_confidence_weighted_regularized and stores the SAME field_logs['regularized'] slot",),
    ),
    Stage(
        key="fit_d.joint_confidence_weighted_regularized",
        label_cn="密集层：联合置信加权正则场（v0.4.0）",
        module="head_rgbd/fit_d.py:114-120",
        parameters=FIELD_STAGE.parameters,
        terms=FIELD_TERMS,
        objective_scope="same quadratic, but the front-protected vertices are only downweighted "
                        "(--joint-front-confidence 0.15) instead of being locked out of the free set",
        solver=FIELD_STAGE.solver,
        notes=("v0.4.0 only; recorded in report.json field_logs['regularized'] so the log key alone does "
               "not reveal which path produced it -- the stage name inside each history entry does",),
    ),
    Stage(
        key="fit_d.front_anchor_dense",
        label_cn="密集层：细网格正面锚",
        module="head_rgbd/fit_d.py:147-149",
        parameters=FIELD_STAGE.parameters,
        terms=FIELD_TERMS,
        objective_scope="quadratic on the subdivided topology with the dense prior; residual bound 0.5 mm",
        solver=FIELD_STAGE.solver,
    ),
    Stage(
        key="fit_d.guarded_dense",
        label_cn="密集层：受保护密集残差（最终网格）",
        module="head_rgbd/fit_d.py:151-158",
        parameters=FIELD_STAGE.parameters,
        terms=FIELD_TERMS,
        objective_scope="quadratic on the dense topology; this field produces head_closed_m.obj / head_open_m.obj",
        solver=FIELD_STAGE.solver,
        notes=("only stage whose output is exported",),
    ),
    Stage(
        key="fit_d.loop_base",
        label_cn="Loop 基面混合（无梯度）",
        module="head_rgbd/fit_d.py:119-145",
        parameters=("global blend alpha in (0,1]",),
        terms=(
            Term(
                key="geometry_safety_feasibility",
                label_cn="几何安全可行性（可行性判定，非可微 loss）",
                source="head_rgbd/fit_d.py:127-133 -> hierarchy_d.safety",
                kind="state",
                normaliser="total displacement vs 3 mm, slope vs 0.45, signed area >= 5%, normal change <= 35 deg",
                normaliser_unit="mm / 1",
                weight="n/a (binary accept/reject)",
                weight_unit="n/a",
                reduction="first feasible alpha from the 1, 1/2, ..., 2^-19 ladder",
                denominator="n/a",
                notes="NOT an optimiser objective; there is no gradient and no loss value",
            ),
        ),
        objective_scope="feasibility-only search, no differentiable objective",
        solver="20 halving trials",
        notes=("no loss curve exists for this stage; the tool records trials only",),
    ),
)

# --------------------------------------------------------------------------
# Identity / motion alternation runs
# --------------------------------------------------------------------------
IDENTITY_V2 = Stage(
    key="identity_v2",
    label_cn="低频共享身份（约束最小二乘）",
    module="head_rgbd/identity_v2.py:23-51 -> geometry_d.solve_field",
    parameters=FIELD_STAGE.parameters,
    terms=FIELD_TERMS,
    objective_scope=("two solve_field calls: an anchor solve (front frames only when guarded, every closed "
                     "identity view when joint) then a second solve over all identity frames that either "
                     "hard-locks the front-protected set or gives it --joint-front-confidence"),
    solver=FIELD_STAGE.solver,
    notes=("same quadratic family as fit_d; the front stage result initialises the guarded stage", 
           "v0.4.0 implements BOTH branches here (identity_v2.py:23-44): joint feeds every closed "
           "identity view and passes soft_lock, guarded keeps the front-only anchor and the hard locked= "
           "mask. But pipeline.py:202-217 does not forward --shape-fusion to this module, so a joint "
           "chain still executes the GUARDED branch",
           "therefore the branch may not be inferred from the run's --shape-fusion flag: read "
           "report.json (shape_fusion, front_locked_vertices>0 => hard lock, "
           "front_downweighted_vertices>0 => confidence weights). A joint chain reporting "
           "shape_fusion='guarded' with front_locked_vertices=2530 and front_downweighted_vertices=0 ran "
           "the hard lock, and on a capture whose usable support lies entirely inside the "
           "front-protected set both solves return status='skipped_no_free_vertices' and the field is "
           "identically zero",
           "its report.json uses a `stages` LIST (one entry per solve) rather than `field_logs`, so a "
           "reader that only looks for field_logs records this stage as 'not persisted' and misses the "
           "far more important fact that it did nothing",
           "v0.5.0: `chain=lean` (the default) removes this stage from the pipeline entirely "
           "(pipeline.py:198-205, 'the joint fit ends with its own motion phase and fit_d rebuilds the "
           "low/mid field'). Its absence is therefore a chain decision, not missing data, and no "
           "low-frequency identity curve should be expected for such a run"),
)


ALL_STAGES = (FIT_C, MOTION_V2, IDENTITY_V2, FIELD_STAGE, *FIT_D_STAGES)


# --------------------------------------------------------------------------
# neck_direct (v2 builder): one 3-D displacement solve per state
# --------------------------------------------------------------------------
NECK_DIRECT_TERMS = (
    Term(
        key="data_radial",
        label_cn="径向数据项（目标场残差）",
        source="head_rgbd/neck_fit.py:659-667 (energy_3d), residual from build_samples_3d",
        kind="data",
        normaliser="(R_target - r)^2 in mm^2; weight = area quadrature x validity gate",
        normaliser_unit="mm^2",
        weight="per-sample w (area x data weight), no extra constant",
        weight_unit="mm^2",
        reduction="sum over vertex and in-face samples",
        denominator="none (sum, not mean) — the sample count changes between iterations",
        notes="this is the only term that measures distance to the target field; its "
              "post-solve RMS/P95 is recorded as a diagnostic, not as a second objective",
    ),
    Term(
        key="bending",
        label_cn="弯曲项（沿轴向的二阶差分）",
        source="head_rgbd/neck_fit.py:724,662 (bending_operator, ||Rb D||^2)",
        kind="prior",
        normaliser="(Rb D)^2 with Rb = L_out restricted to the editable set",
        normaliser_unit="mm^2",
        weight="ell_reg per material position (axial length in mm)",
        weight_unit="mm",
        reduction="sum over 3 x vertices",
        denominator="none",
        notes="penalises curvature change of the displacement, not the surface curvature",
    ),
    Term(
        key="tangent",
        label_cn="切向先验",
        source="head_rgbd/neck_fit.py:639-656,663 (sqrt(mu_t A)(I - e_r e_r^T))",
        kind="prior",
        normaliser="(T D)^2, T = sqrt(mu_t A) (I - e_r e_r^T)",
        normaliser_unit="mm^2",
        weight="mu_t: bridge 0.15 / body 0.50 (blended over seam_prior_edges)",
        weight_unit="1",
        reduction="sum over 3 x vertices",
        denominator="none",
        notes="bridge may move in 3-D but not freely; the body is strongly tangential",
    ),
    Term(
        key="anchor",
        label_cn="固定头部锚定",
        source="head_rgbd/neck_fit.py:639-656,664 (sqrt(mu_a A))",
        kind="prior",
        normaliser="(A D)^2, A = sqrt(mu_a A_v) I",
        normaliser_unit="mm^2",
        weight="anchor_strength * (1 - data weight) + anchor_near_A near the fixed head",
        weight_unit="1",
        reduction="sum over free vertices",
        denominator="none",
        notes="Dirichlet fixed set is separate: those rows are removed from the unknowns",
    ),
    Term(
        key="small",
        label_cn="数值先验（各向同性）",
        source="head_rgbd/neck_fit.py:665-666",
        kind="prior",
        normaliser="mu_small * sum A_v |D|^2",
        normaliser_unit="mm^2",
        weight="mu_small = 1e-3",
        weight_unit="1",
        reduction="sum over vertices",
        denominator="none",
        notes="keeps the Hessian well posed; not a shape prior",
    ),
)

NECK_DIRECT = Stage(
    key="neck_direct",
    label_cn="颈部直接塑形（统一 3-D 位移，逐状态）",
    module="tools/build_neck_direct_v2.py:280-330 -> head_rgbd/neck_fit.py:680-872 (fit_unified)",
    parameters=("D (每顶点 3-D 位移, mm)",),
    terms=NECK_DIRECT_TERMS,
    objective_scope=("one Gauss-Newton solve per state over B+C; A and the fixed set are "
                     "Dirichlet boundaries. The solver objective is the sum of the five terms, "
                     "minimised under radial/tangential trust-region bounds. Remesh passes are "
                     "geometry proposals checked by a crease rollback, not optimisation steps"),
    solver="Gauss-Newton + backtracking line search (max 60 iterations)",
    notes=("the builder now persists the per-iteration energy/energy_parts in loss_direct.json; "
           "before that only the final solve summary existed in build_direct.json",
           "target residual (vertex/quadrature RMS, P95, max) is recomputed at the end of the "
           "solve and is a diagnostic; it is not added to the objective",
           "remesh merges/flips/splits and the seam density budget belong to structure/density "
           "acceptance, not to this objective",
           "marker-derived t_center and amplitude semantics are inputs; an offline candidate may "
           "not change them"),
)

ALL_STAGES = (FIT_C, MOTION_V2, IDENTITY_V2, FIELD_STAGE, *FIT_D_STAGES, NECK_DIRECT)


# --------------------------------------------------------------------------
# v0.4.0 mode variants: same stage key, different objective
# --------------------------------------------------------------------------
MODE_VARIANTS = (
    ModeVariant(
        variant_key="shared_identity_guarded",
        stage_key="fit_c",
        label_cn="共享身份（guarded，默认）",
        activated_by="--shape-fusion guarded (the default)",
        data_frames="front identity frames get view_weight 3 during identity phases, other closed views 1",
        term_overrides="no joint_shape_prior; shape gradient is frozen in the side-anchored mode",
        evidence="report.json shape_fusion='guarded'",
    ),
    ModeVariant(
        variant_key="shared_identity_joint",
        stage_key="fit_c",
        label_cn="共享身份（joint，v0.4.0）",
        activated_by="--shape-fusion joint",
        data_frames="every closed view gets view_weight 1.0 in identity phases (front loses its 3x)",
        term_overrides="adds joint_shape_prior = --joint-shape-prior (0.02) * mean((shape-anchor_shape)^2); "
                       "side depth scaled by --side-depth-weight",
        evidence="report.json shape_fusion='joint' + joint_anchor_update_mm / joint_shape_delta_from_anchor",
    ),
    ModeVariant(
        variant_key="alternating_low_frequency_front",
        stage_key="identity_v2",
        label_cn="低频身份：锚定解（guarded 分支的前帧解）",
        activated_by="guarded branch, first solve_field call (identity_v2.py:25,29)",
        data_frames="front identity frames only",
        term_overrides="no lock at all: this solve produces the anchor the second solve starts from",
        evidence="stages[i].stage in report.json",
    ),
    ModeVariant(
        variant_key="joint_low_frequency_all_views",
        stage_key="identity_v2",
        label_cn="低频身份：锚定解（joint 分支的全体闭口帧解）",
        activated_by="joint branch, first solve_field call (identity_v2.py:30)",
        data_frames="every closed identity view",
        term_overrides="no lock; the anchor already carries side evidence",
        evidence="stages[i].stage in report.json",
    ),
    ModeVariant(
        variant_key="alternating_guarded_low_frequency",
        stage_key="identity_v2",
        label_cn="低频身份：受保护解（硬锁）",
        activated_by="guarded branch, second solve_field call (identity_v2.py:42-43)",
        data_frames="every closed identity view",
        term_overrides="locked=front_protected removes those vertices from the unknown vector, so "
                       "free_vertices excludes them",
        evidence="report.json front_locked_vertices>0, front_downweighted_vertices=0",
    ),
    ModeVariant(
        variant_key="joint_confidence_weighted_low_frequency",
        stage_key="identity_v2",
        label_cn="低频身份：置信加权解（软锁，v0.4.0）",
        activated_by="joint branch, second solve_field call (identity_v2.py:36-38)",
        data_frames="every closed identity view",
        term_overrides="soft_lock=--joint-front-confidence (0.15) on the front-protected set; those "
                       "vertices stay free but keep only 15% of their depth weight",
        evidence="report.json front_downweighted_vertices>0, front_locked_vertices=0",
    ),
    ModeVariant(
        variant_key="guarded_regularized",
        stage_key="fit_d.guarded_regularized",
        label_cn="密集正则场（硬锁）",
        activated_by="--shape-fusion guarded (fit_d.py:121-124)",
        data_frames="every closed identity view",
        term_overrides="locked=front_protected removes the front region from the free set",
        evidence="field_logs['regularized'].history[i].stage",
    ),
    ModeVariant(
        variant_key="joint_confidence_weighted_regularized",
        stage_key="fit_d.guarded_regularized",
        label_cn="密集正则场（软锁，v0.4.0）",
        activated_by="--shape-fusion joint (fit_d.py:114-120)",
        data_frames="every closed identity view",
        term_overrides="soft_lock=0.15 on the front-protected set; same free-set size as guarded is "
                       "NOT expected because the downweighted vertices stay free",
        evidence="field_logs['regularized'].history[i].stage + report.shape_fusion block",
    ),
    ModeVariant(
        variant_key="reused_low_mid_field",
        stage_key="fit_d.guarded_regularized",
        label_cn="密集正则场：直接复用前锚的低/中频场（v0.5.0）",
        activated_by="--reuse-field, a 0.5.0 default",
        data_frames="none: no equations are assembled for this stage",
        term_overrides="field_logs['regularized'] becomes {stage: 'reused_low_mid_field', reused: true, "
                       "history: []}; its maximum_absolute_displacement_mm is inherited from the field "
                       "it reused, so this stage has NO objective and must not be reported as solved",
        evidence="field_logs['regularized'].reused == true with an empty history",
    ),
    ModeVariant(
        variant_key="fast",
        stage_key="fit_d.guarded_dense",
        label_cn="快速路径：密集残差整段不执行（0.5.0 起为默认）",
        activated_by="--fast (default since 0.5.0; pass --no-fast to execute the residual)",
        data_frames="none: no dense equations are assembled",
        term_overrides="front_anchor_dense / guarded_dense are recorded with status='not_executed' and "
                       "dense_solver_scalar_m is identically 0; the delivered surface is the Loop base, "
                       "so dense_change_from_interpolated_regularized_mm equals "
                       "loop_base.maximum_move_from_linear_mm bit for bit",
        evidence="statuses / field_logs status field; dense_solver_scalar_m absmax = 0",
    ),
)


def variants_for(stage_key: str) -> tuple[ModeVariant, ...]:
    return tuple(variant for variant in MODE_VARIANTS if variant.stage_key == stage_key)


def variant_by_key(variant_key: str) -> ModeVariant:
    for variant in MODE_VARIANTS:
        if variant.variant_key == variant_key:
            return variant
    raise KeyError(f"unknown mode variant {variant_key!r}")



def stage_by_key(key: str) -> Stage:
    for stage in ALL_STAGES:
        if stage.key == key:
            return stage
    raise KeyError(f"unknown stage {key!r}; known: {[s.key for s in ALL_STAGES]}")


def terms_of(stage_key: str) -> tuple[Term, ...]:
    return stage_by_key(stage_key).terms


def term_keys(stage_key: str) -> tuple[str, ...]:
    return tuple(term.key for term in terms_of(stage_key))
