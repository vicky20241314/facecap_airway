# Changelog

## 0.2.0 - 2026-11-01

Final-flow consolidation, entry-point reduction and output convergence. This
release does not claim any geometry-accuracy improvement.

- One entry point (`headrgbd run | capture | reconstruct | device | assets`) that
  always runs the frozen `facecap-inspired-v2-strict` chain; research modes,
  policy switches and evaluation/ablation modules are no longer shipped.
- Default export converged to one final metric mesh (`head_m.obj`) plus
  `manifest.json`, `reference_landmarks.json` and `support.json`. Coarse,
  anchor, low-frequency, smooth-base, interpolation and candidate-comparison
  meshes are no longer exported.
- Runtime root defaults to `<project>/runtime`, resolved from the source
  location; no silent fallback to a research directory.
- Added `device` (camera + model load check) and `assets` (SHA256) commands.
- Added a real-acquisition importer for the verification capture and a trimmed
  contract test suite.

## 0.1.0 - 2026-10-31

Migrated RGB-D head reconstruction research worktree (GB/T style pipeline,
research modes B/C/D and v2 refinement candidates).
