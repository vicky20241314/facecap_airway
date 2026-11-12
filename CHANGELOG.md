# Changelog

## 0.5.7 - 2026-11-12

Proportional sculpt candidate (NOT the public default; 0.5.6 fixed design stays
the default because the proportion preferences were not fully met).

- New scale measurement on the unshaped mesh: W_head (front-facing lower-face
  band, ears excluded by orientation), W_neck (median full cross-section width
  over the middle of the neck), L_neck (submental start where the section
  reaches 90% of the neck reference width, to a neck root found from a
  cross-section widening; the mesh cut is only a documented fallback), robust
  neck axis from section centroids. All methods, confidence and fallbacks are
  recorded (W_head 94.0, W_neck 121.1, L_neck 42.0 mm on this template).
- Presets become relative strengths (0.6/1.0/1.4); sizes are derived:
  H = 0.055 W_neck, L10 = 0.46 L_neck, and a coordinated W10 = L10/1.25/(0.65/0.686)
  so a short neck shrinks the whole design instead of squashing it; slope
  (<=1.1 mm/mm), head-width and space limits are applied and recorded.
- Normalised position s in [0,1] along the front midline; the natural centre is
  chosen from a candidate scan that maximises the weaker-state clearance from
  the chin protection (s = 0.38 here, recorded together with the label's s and
  the offset). The centre is a shared base-topology vertex binding, so closed
  and open cannot drift apart.
- Vertical envelope option B: C2 plateau (1-t^2)^3 retains effective height much
  longer than the Wendland profile at the same support; A/B evaluated.
- Result on this template: H/W_neck 0.054 and L10/L_neck 0.37 meet the design
  ranges, geometry is safe (0 folds/self-intersections/slender faces, facial
  hit shift 0.0000 mm, disk OBJ verified), but W10/W_neck 0.14 (coordinated
  width for the short neck, outside the nominal 0.26-0.34 band) and the
  measured L50/W50 0.85-0.99 remain below the preferred 1.1-1.5; the measured
  apex curvature fit is polluted by the flanks. These are recorded as blocking
  items in docs/NECK_PROPORTIONAL_SCULPT_REPORT.html; the default entry keeps
  the 0.5.6 geometry.
## 0.5.6 - 2026-11-12

Manually designed laryngeal appearance sculpt (user-authorised non-observed
geometry). The public export entry now defaults to `neck_prominence.algorithm =
manual_edit`, which is this sculpt; it is explicitly an appearance edit, not a
subject reconstruction.

- One continuous analytic field p(u,v) = A(v)*C(s): Wendland C2 vertical ridge
  envelope with independent upper/lower support lengths, a rounded-V cross
  section with a finite tip radius and C2 lateral window, and a width envelope;
  displacement along one constant local neck normal, multiplied only by a
  continuous protection taper. Values and first derivatives vanish at the
  support boundary; no stacked bumps, no negative-ridge or depth-pull terms, no
  post-export smoothing.
- Presets: light 3 mm / natural 5 mm (default) / pronounced 7 mm forward
  height; natural uses a 29 mm lateral support, 15/24 mm up/down transitions
  and a 2.5 mm tip radius. Appearance parameters only, not anatomy.
- Positioning uses the recorded label when available and otherwise estimates
  the larynx position from the mesh (neck midline, jaw and neck base) with
  `position_source=estimated_from_mesh`; no ArUco requirement.
- Local two-level red-green refinement in the target patch (union masks for
  closed/open, per-state midpoint positions, protection inherited) - core edges
  about 1-2 mm; the experiment driver and the public entry share the same
  kernel.
- Acceptance is appearance/geometry based (no depth MAE requirement): zero
  self-intersections, degeneration, winding errors, true folds, slender faces,
  edge growth <= 2.2x, protected vertices unchanged, facial first-hit shifts
  <= 0.05 mm, shared refined topology, and the disk OBJ read back within 1 um.
  `requested/realized height, width, length, tip radius, transitions and
  safety_fallback` are recorded per state.
- Finding: the three presets are geometrically safe on both states with
  realized heights equal to the design (5.0 -> 4.96/4.97 mm at natural); the
  delivered geometry is exactly the analytic design sampled on the refined
  mesh. Public entry export is byte-identical to the experiment's natural
  preset. Report: docs/NECK_APPEARANCE_SCULPT_REPORT.html.
## 0.5.5 - 2026-11-11

Subject-observed neck reconstruction V3. Replaces the fixed-height procedural
prominence with an observation-driven local fit on the saved 0.5.2 geometry
(no FLAME re-run).

- Local neck frame per state from the depth observations (label origin, plane
  normal, observed midline refinement; head pose only initializes).
- Frozen low-order quadratic background fitted to all reliable observations
  (six coefficients, cannot form a local bump), plus a signed rounded-ridge
  model (height, lateral width, tip rounding, centre offset/tilt, peak position,
  independent up/down transitions; |height| bounded to 3 mm) and a strongly
  priored bounded residual field. The fit is robust and binned; the depth term
  uses the true first hit with a per-pixel normal/ray response factor.
- Deterministic red-green refinement of the front-neck patch (shared refined
  topology for closed/open, per-state midpoint positions, midpoint protection
  inherited); delivery marks the refined geometry.
- Acceptance on a fixed observation set: exact first-hit depth, no-hit 30 mm
  penalty, no truncation, MAE/P95/coverage vs baseline and V2; facial first-hit
  protection with a 0.05 mm engineering tolerance on facial-surface hits;
  pre-existing degenerate FLAME faces excluded from the reversal count; formal
  geometry guards (area/normal/slender/edge-growth cap/self-intersection).
- Finding for this capture: the depth data does not support a clear laryngeal
  prominence. After the background correction the local excess at the label is
  -0.5 mm (closed) / -0.7 mm (open) with 0.3-0.6 mm background spread; V2's
  8 mm shape has no observation basis. Delivered geometry follows the skin
  (closed first-hit MAE 13.6 -> 7.4 mm, open 5.8 -> 3.3 mm) and the report
  states "geometry safe, shape fidelity unproven".
- Public entry: `neck_prominence.algorithm` = `observation` (default) or
  `manual_edit`; the manual two-field editor stays available but is marked as
  an appearance edit, not a subject reconstruction. Report:
  docs/NECK_SUBJECT_V3_REPORT.html and the run directory.
## 0.5.4 - 2026-11-11

Depth-conditioned procedural neck prominence V2 on the 0.5.2 template. Base head
reconstruction is untouched; the optional stage is a single light pass between
the saved baseline dense surface and the unified mesh write.

- Retarget the stage from a fixed displacement amplitude to a FINAL prominence
  height H measured on a fixed sagittal plane/mesh section against a baseline
  reference trend anchored outside the support before shaping. Both closed and
  open states reach the 8 mm target (error 0.0 mm) and are accepted only when
  the shape goal (|H_final - H_target| <= 1 mm) AND every geometry check pass.
- Replace the single narrow Wendland core with a wide Wendland C2 base plus a
  flat-topped C2 compact-support crest (base share 0.35); a single-field control
  at the same target fails the open state with a displacement spike.
- Snap the shaping centre onto the template neck's own front ridge when the
  binding lands on the ridge slope (template neck laterally offset from the
  subject's real neck), limited to 20 mm and fully recorded.
- Replace the graph-ring taper with a continuous taper over the actual distance
  (6 mm ramp) from protected vertices; keep face/lips/jaw/chin/ears/back/bottom
  protection and the exact camera-ray binding.
- Add in-stage self-intersection detection (segment/triangle with legal
  shared-edge contact excluded) plus a displacement spike test; edge-growth
  limits become a hard measured cap inside the shaping region (1.58-1.75x smooth
  flank growth recorded) instead of the plain 1.5x/+2 mm rule.
- Record requested final height, baseline height, final height, target error,
  binding barycentric forward displacement, max vertex displacement, actual
  width/length, peak offset, shape parameters, depth conditioning, snap and
  fallback reasons per state (neck_prominence.json v2).
- Use depth only to verify direction, bound support radii against the observed
  neck width and record reliability; target height stays a configured
  appearance prior, never derived from unreliable depth protrusion.
- Fix the manual landmark ROI (10% -> 2% of face-box height below the chin: the
  old gate excluded the laryngeal region and forced clicks onto the collar) and
  correct this capture's landmarks from recorded RGB/depth evidence; female +
  auto still keeps every vertex identical, off still matches the clean 0.5.2
  baseline byte for byte.
- 31 neck tests plus the V2 verification tool (8 legacy invariants + fixed
  shape acceptance + same-view renders and plane/mesh sections). Report:
  docs/NECK_PROMINENCE_V2_REPORT.html.

## 0.5.3 - 2026-11-11

Marker-guided front-neck template prominence on the 0.5.2 baseline: minimal
landmark localisation port (ArUco detection + manual fallback, per-state
records), label-bound local Wendland C2 shaping with strict face protection,
neck_prominence.json sidecar and an 8-item end-to-end verification. Superseded
by 0.5.4 for shape quality; the 0.5.3 run and report remain archived.

## Unreleased - output performance

- Default reconstruction writes only the final closed/open OBJ files in delivery.
  Remove unused motion comparisons, intermediate dense meshes, support PLY and
  independent neck diagnostic surfaces; preserve fitting and geometric checks.
- Store default fitting checkpoints as uncompressed NPZ with unchanged arrays.
  Export the final geometry directly from checkpoints, while retaining legacy
  OBJ re-export and validating written mesh indices, units and landmarks.
- Three interleaved dense-plus-delivery runs per version: median 31.657 s to
  23.389 s. Final meshes, support and landmarks remain byte-identical; the full
  default CLI run also passes. See `docs/MINIMAL_OUTPUT_REPORT.md` for scope,
  storage tradeoff and all samples.

## 0.5.2 - 2026-11-03

- Add confidence-gated lower-face depth to the existing open-front expression/jaw
  phase, weight 0.5. Keep upper-face rigid support, mouth/depth-edge exclusion,
  shared identity, motion bounds and the original fixed-ray acceptance; require
  lower-face fixed-ray coverage and non-increasing MSE as well.
- Share whole-batch motion evaluations across frames at each backtracking scale;
  retain per-frame first-accepted scales. Reuse initial ray residuals and coverage
  instead of issuing the same query twice. Disabled new constraints reproduce
  the previous parameters and vertices exactly on both validation captures.
- Keep 240 front / 240 joint / 80 motion steps, one mid round, 8 mm bending and
  geometry-checked Loop 0.5. Forward one JSON default through public entry points
  and report actual lower-face settings, accepted frames and evaluation counts.
- Record the final motion iteration. Add regressions for coverage-preserving
  batch acceptance, finite weights, shared defaults and historical re-export.
- Validate two captures, same-frame meshes, disjoint motion-sampling pixels and
  common rays; reject weight 1.0 because it regresses the second capture. Twelve
  full reconstructions retain all timing samples, including large variations in
  unmodified stages. Supplementary warm motion timing does not establish stable
  acceleration. The gain is small local depth consistency, not scan accuracy.
- Results, meshes, hashes, limits and all timings:
  `docs/LOSS_GUIDED_IMPROVEMENT_REPORT.html`.

## 0.5.1 - 2026-11-02

- Keep one preview window alive throughout all four capture poses. Press SPACE
  in that window to start the live countdown and recording; preview mode no
  longer asks for a second confirmation in the terminal.
- Treat ESC and the window close button as cancellation, including during
  recording and missing-frame retries. Always release the owned preview and
  camera on completion or failure; closing a preview never recreates it.
- Retain terminal Enter/countdown for `--no-preview`, and automatic countdown
  for `--auto`. Reconstruction settings and quality are unchanged from 0.5.0.
- Forward PowerShell subcommand help (`run --help`, `capture --help`) instead
  of consuming the help switch and accidentally starting capture.

## 0.5.0 - 2026-11-02

Default smooth-fast reconstruction. Public `run` and `reconstruct`, including
both Windows launchers, share the JSON defaults: joint, fast, lean, minimal
postfit, one mid-frequency round, field reuse, barycentric information 0.5,
coverage allowance 0.001, target Loop blend 0.5, no depth candidate selection.

- Try the target Loop blend first and backtrack only for original geometry
  safety; record target, adopted blend, failure checks and fallback reason.
- Make prior stencil eligibility consistent with reliable single-view
  barycentric controls, while retaining anatomical and unobserved locks.
  Measurement labels and optimization free sets stay separate from smoothing.
  Reuse the low/mid support union without new evidence queries.
- Select 8 mm mid-frequency bending after 4/6/8 comparisons. Keep low-frequency
  bending at 8 mm and gradient at 3 mm; stronger exploratory settings were not
  adopted. The local field remains a measured correction, with no shrink filter.
- Add an independent frontal initialization budget through the common argument
  builder. The proposed 120 steps lost open-mouth coverage, so defaults remain
  240 front / 240 joint, plus one 80-step motion correction.
- Record actual stage settings and source hashes in run reports and delivery
  manifests, including overrides, re-export, and Loop fallback. Reject invalid
  budgets before camera capture. Keep launchers as argument forwarders.
- Preserve a hashed pre-change full joint-fast source snapshot and interleave
  three prepared-input runs per variant. Fixed-domain geometry, same-frame depth,
  safety, reproducibility and timing evidence: `docs/DEFAULT_SMOOTH_FAST_REPORT.md`.
- Add regression cases for independent budgets, shared public defaults,
  evidence/stencil separation, safe fallback and actual configuration reporting.

## 0.4.0 - 2026-11-02

Shape-capability release: front, left and right observations now jointly determine
one shared face shape. Algorithm name is unchanged
(`facecap-inspired-v2-strict`); what changes is how shape evidence is used. The
default stays `guarded`, so existing behaviour is reproduced unless joint is
requested.

- **Why this was needed.** Up to 0.3.0 the side views could not contribute to the
  shared shape at all, and that was structural rather than a tuning problem:
  `fit_c` set `shape.grad = None` in the side-anchored mode (identity frozen),
  dropped front frames from the identity phases, `identity_v2` hard-locked every
  front-covered vertex, and `fit_d` fed only front identity frames into the
  low/mid-frequency control field. Measured consequence on the reference
  capture: `changed_side_detail_vertices = 0` and
  `maximum_side_detail_change_mm = 0.0` — the side stage moved nothing.
- **New `--shape-fusion joint`.** All closed views update one FLAME identity and
  one smooth scalar normal field; each frame keeps its own pose, expression and
  jaw. The front anchor becomes an initialisation with a finite soft pull
  (`--joint-shape-prior`, default 0.02) instead of a fixed endpoint, identity
  phases give every closed view one vote instead of a 3x front boost, and side
  depth terms scale with `--side-depth-weight` (0 reproduces the ablation).
- **Hard lock replaced by continuous confidence.** `geometry_d.solve_field`
  gains `soft_lock`, a per-vertex weight multiplied into the equation weights
  while the vertex **stays in the free set**, so the displacement, slope, normal
  and fold checks still apply. `identity_v2` and `fit_d` use
  `--joint-front-confidence` (default 0.15) on the previously locked region
  instead of `locked=`.
- **Control field from every closed view.** `fit_d` low/mid-frequency solves use
  the full closed identity set in joint mode rather than the front subset, so the
  delivered Loop base inherits the fused shape. `report.json` gains a
  `shape_fusion` block (mode, identity views, hard-locked and downweighted vertex
  counts); `fit_c` records `joint_anchor_update_mm` and
  `joint_shape_delta_from_anchor` separately because the former also contains
  pose/expression freedom (9-11 mm) and is not a shape figure.
- **Measured on the 20-frame reference capture** (same code, same input, same
  scoring as the control): front hard-locked vertices 2797 -> 0, local field
  actually moved 1776 vertices up to 2.999 mm where guarded moved zero. Closed-view
  accuracy is **level with guarded** (largest per-view |dMAE| 0.040 mm, inside the
  pre-declared 0.05 mm gate). Open-mouth MAE 1.737 -> 1.581 mm, coverage
  95.74% -> 98.94%, missing rays 1278 -> 319, and front jaw/contour P95
  9.384 -> 7.756 mm. **Nose accuracy does not improve** (0.99-1.06 mm across all
  three views in every arm) and this release does not claim it.
- **Side ablation.** Running joint with `--side-depth-weight 0` keeps the same
  degrees of freedom but removes the side depth term: side lips degrade by
  0.7-1.0 mm and side jaw/contour by 0.65-0.84 mm. That is the evidence that the
  side residual carries usable lateral geometry, as opposed to the extra freedom
  doing the work.
- **Cost.** joint+fast 249.8 s against guarded+fast 265.0 s on the reference
  capture: the fusion stage costs +14.9 s (38.1 -> 53.0 s) while the dense stage
  gets 25 s faster, so there is no measurable net cost rather than a speedup.
- **New tools.** `tools/eval_multiview.py` scores every frame against its own
  observation, grouped by view, mouth state, shape-training role and face region
  (nose / jaw-contour / lips / rest), which is what makes a held-out side
  evaluation possible without separate mesh exports.
  `tools/import_multiview_capture.py` converts an `airway3d.rgbd.multiview.v1`
  capture into this pipeline's capture directory (one reference frame per view).
- **Tests.** `tests/test_joint_fusion.py` (5 cases): a side residual reaches the
  shared field while agreeing front views stay put; a hard lock freezes the
  frontal region and a 15% confidence weight does not; three agreeing views
  manufacture no correction; guarded stays the default; joint refuses to run
  without a front anchor.
- **Documentation.** `docs/JOINT_FUSION_REPORT.md` carries the mechanism,
  the per-view and per-region tables, the ablation and the scope limits.
- **Scope.** One capture, 20 frames, three views; the held-out closed frames are
  within one burst, not a second session. Closed-view accuracy does not beat
  guarded on this capture — the front anchor was already accurate laterally —
  so the supported claims are "side data is necessary" and "the fusion mechanism
  is active", not "joint is more accurate than guarded".

## 0.3.0 - 2026-11-01

Reconstruction-speed release. Same fitting chain, same output contract, same
quality gates; the visibility kernel is replaced and a fast path is added.
Evaluation tooling and loss accounting are out of scope for this release and are
not shipped here.

- **Ray intersection replaced** (`head_rgbd/ray_index.py`). Every visibility
  query in this pipeline casts from the camera centre, so a ray is fixed by the
  surface point it was backprojected to. Triangles are projected onto the camera
  plane (`u = x/-z`, `v = y/-z`) and looked up in a uniform grid, with an exact
  Möller-Trumbore narrow phase. Measured 0.644 ms/ray -> 0.0072 ms/ray (about
  90x, index build included); against the previous rtree query on real meshes
  and real depth-sample rays it returns identical hit counts, 100% identical
  face ids and hit points differing by 0.15 um. The kernel sits under
  `ray_surface` and `front_protected_vertices`, so identity fitting, motion,
  frontal protection, acceptance and the final evidence pass all benefit - this
  is why the early stages shrink too, not just the dense stage.
- **Optional fast path** (`--fast --mid-rounds N` on `head_rgbd.fit_d`, and the
  same flags on `headrgbd reconstruct`): the delivered surface is the budgeted
  Loop base grown from the guarded low/mid control field, i.e. no dense residual
  QP is solved; the mid-frequency correction runs one round instead of three;
  the intermediate `regularized` evidence is not computed because only a preview
  image and the reuse sidecar consume it. `report.json` gains a `fast_path`
  block and each skipped stage records `status: not_executed` with a reason, so
  a skipped solve can never be read as a successful one.
- **Measured end to end on the 20-frame verification capture**: 1945.7 s ->
  298.3 s (6.52x). Replacing only the kernel, with no stage skipped, takes
  865.4 s (2.24x). The two gains are reported separately and are not additive.
- **Quality, against the same code without the fast flags**: whole-face MAE
  +0.012 mm (closed) / +0.008 mm (open); lips +0.005 / +0.003, jaw and contour
  +0.024 / +0.017, rest of the face +0.011 / +0.007; coverage, missing-ray count
  and flipped-face count unchanged. That is inside the agreed gates
  (MAE/RMSE +0.05 mm, P95 +0.1 mm, coverage -0.1 pp).
- **Not bit-identical to 0.2.0, and this is stated rather than hidden.** The new
  kernel agrees with the old one to 1e-7 m, but the chain reacts to that: the
  barycentric support threshold `bc > .05` flips three vertices
  (`free_vertices` 949 -> 946), which changes the QP active set and moves the
  low-frequency solution. Against the 0.2.0 reference the whole-face MAE reads
  +0.39 mm - and a run with only the kernel replaced and no stage skipped reads
  the same +0.39 mm. **The 0.2.0 acceptance therefore does not carry over**, and
  the comparison that isolates this release is the one against that no-skip
  control, not against 0.2.0.
- Fixed: `first_hits` raised `ValueError` whenever a whole 32-ray batch missed
  (trimesh returns a `(0,)` shaped array that cannot be concatenated with
  `(n,3)`). Empty batches now have an explicit `(0,3)/(0,)/(0,)` contract, and
  ray numbering stays global.
- Also in this release: `fit_d` reports per-outer `timing` and
  `actual_update_mm` plus a section breakdown in `report.json`; the outer loop
  stops when the accepted step moves the surface less than one depth
  quantisation step and the fixed-ray MSE improves by less than 1e-4, with the
  step scale deliberately excluded from the rule (a very small step can be the
  only one that survives the coverage check); an empty free set now records
  `status: skipped_no_free_vertices` and returns the unchanged field instead of
  entering the solver.

- Documentation: `README.md` gains §5.3 fast-path usage, §7.1 the visibility
  kernel, §7.2 what the fast path skips and what it keeps (with the measured
  time/MAE trade-off of each round count), §7.3 the `report.json` evidence
  fields, and §9.1 the three-way per-stage timing table. The technical write-up
  ships as `docs/FAST_PATH_REPORT.md`, which also lists the early design
  assumptions this work disproved so they are not reused.

## 0.2.0 - 2026-11-01

Final-flow consolidation, entry-point reduction and output convergence. This
release does not claim any geometry-accuracy improvement.

- One entry point (`headrgbd run | capture | reconstruct | device | assets`) that
  always runs the frozen `facecap-inspired-v2-strict` chain; research modes,
  policy switches and evaluation/ablation modules are no longer shipped.
- Default export converged to two final metric meshes, one per mouth state
  (`head_closed_m.obj`, `head_open_m.obj`, same shared identity and topology),
  plus `manifest.json`, `reference_landmarks.json` and `support.json`. Coarse,
  anchor, low-frequency, smooth-base, interpolation and candidate-comparison
  meshes are not exported.
- Runtime root defaults to `<project>/runtime`, resolved from the source
  location; no silent fallback to a research directory.
- Added `device` (camera + model load check) and `assets` (SHA256) commands.
- Added a real-acquisition importer for the verification capture and a trimmed
  contract test suite.

## 0.1.0 - 2026-10-31

Migrated RGB-D head reconstruction research worktree (GB/T style pipeline,
research modes B/C/D and v2 refinement candidates).
