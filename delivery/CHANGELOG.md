# Changelog

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
