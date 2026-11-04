# Third-party notices

The source repository contains code from the projects below. Runtime model
weights are not committed; each must be obtained under its own license before
the pipeline can run.

| Component | Location | License / terms | Notes |
|---|---|---|---|
| FLAME model and PyTorch implementation | `third_party/flame/`, `runtime/assets/flame/` | FLAME license (MPI, non-commercial research) | `flame2023.pkl`, static/dynamic landmark embeddings. Read `third_party/flame/LICENSE` and the FLAME terms before any commercial use. |
| TUFA (landmark network) | `third_party/tufa/`, `runtime/assets/tufa/` | MIT (see `third_party/tufa/LICENSE`); weights under the authors' terms | `Face_prompt_base.pth` and `Prompt/shape_68.npz`. |
| UniFace (face parsing) | `runtime/assets/uniface/parsing_resnet34.onnx` | UniFace project terms | Loaded through the `uniface` pip package; the ONNX file is fetched separately. |
| YuNet face detector | `runtime/assets/models/yunet/` | OpenCV Zoo / YuNet terms (Apache-2.0 family) | `face_detection_yunet_2023mar.onnx`. |
| SMPL-X (`smplx` pip package) | dependency | MPI non-commercial research | Used only for linear blend skinning helpers. |
| trimesh, rtree, SciPy, NumPy, OpenCV, PyTorch, OSQP | dependencies | MIT / BSD / Apache-2.0 | Standard pip installs. |
| Orbbec SDK Python bindings (`pyorbbecsdk`) | dependency | Orbbec SDK license | Required for Gemini 215 acquisition. |

## Distribution rules

- Do not commit `runtime/env/`, `runtime/assets/`, `runtime/models/`, captures,
  caches or generated meshes.
- Model redistribution, remote use and commercial deployment require the
  license-holder's permission for each model above.
- `third_party/tufa/Prompt/shape_*.npz` are small prompt offsets that ship with
  the TUFA source and are the only binary files kept in the repository.
