# Changelog

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
