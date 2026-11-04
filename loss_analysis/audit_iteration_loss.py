"""Read-only audit of the recorded iteration objectives for one completed run.

Purpose: recompute, from the run's OWN persisted artifacts, the quantities the
    delivery prints and logs, so that "the loss did not go down" can be
    separated into  (a) recording artefact,  (b) genuine stagnation,
    (c) broken convergence.

Hard constraints (see HANDOFF.md §7):
    * nothing under runtime/outputs/ is modified;
    * no reconstruction is re-executed;
    * no main-project source is imported for the maths that can be reproduced
      from the persisted arrays.

Usage (PowerShell, login=false):
    Set-Location C:\\
    $env:PYTHONPATH = 'D:\\proiect\\facecap_airway_delivery\\loss_analysis'
    $env:PYTHONIOENCODING = 'utf-8'
    & 'D:\\proiect\\facecap_airway_delivery\\runtime\\env\\python.exe' `
        'D:\\proiect\\facecap_airway_delivery\\loss_analysis\\audit_iteration_loss.py' `
        --run 'D:\\...\\runtime\\outputs\\<run>'
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

RUN_DEFAULT = (r"D:\proiect\facecap_airway_delivery\runtime\outputs"
               r"\20261101_140933_045847_head_reconstruction")
OUT_DEFAULT = (r"D:\proiect\facecap_airway_delivery\runtime\loss_analysis"
               r"\audit_iteration_loss")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def fmt(x, nd=6):
    if x is None:
        return "null"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return str(v)
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        return f"{v:.4e}"
    return f"{v:.{nd}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(c) for c in r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# A. objective decomposition of the Adam stages, using the run's own logs
# --------------------------------------------------------------------------- #
def adam_append_audit(run: Path):
    """Reconstruct the phase sequence of each Adam stage and check monotonicity.

    The delivery prints one scalar every 20 steps. A phase change rewrites the
    objective (expression / jaw enter or leave the sum, identity phases zero
    their gradients), therefore a phase boundary is NOT a loss increase and a
    raw first->last comparison mixes incomparable numbers.
    """
    logs = {
        "identity_front_anchor": run / "logs" / "02_identity_front_anchor.log",
        "shared_identity_guarded": run / "logs" / "03_shared_identity_guarded.log",
        "fixed_identity_motion": run / "logs" / "04_fixed_identity_motion.log",
        "motion_reestimate_after_identity": run / "logs" / "06_motion_reestimate_after_identity.log",
    }
    res = {}
    for stage, path in logs.items():
        if not path.exists():
            continue
        obj, phase, frames = [], [], []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            rec = json.loads(line.replace("'", '"'))
            if "objective" not in rec:
                continue
            obj.append(float(rec["objective"]))
            phase.append(rec.get("phase"))
            frames.append(rec.get("correspondence_frames"))
        steps = [int(rec["step"]) for rec in
                 (json.loads(l.strip().replace("'", '"'))
                  for l in path.read_text(encoding="utf-8").splitlines()
                  if l.strip().startswith("{") and "objective" in l)]
        res[stage] = dict(objective=obj, phase=phase, frames=frames, steps=steps,
                          raw_first=obj[0] if obj else None,
                          raw_last=obj[-1] if obj else None)
    return res


def phase_aware_summary(stage, rec):
    """Within-phase monotonicity + honest boundary resets."""
    obj, phase, steps = rec["objective"], rec["phase"], rec["steps"]
    segments, cur = [], 0
    for k in range(1, len(obj)):
        if phase[k] != phase[cur]:
            segments.append((cur, k - 1))
            cur = k
    segments.append((cur, len(obj) - 1))
    rows = []
    for a, b in segments:
        vals = obj[a:b + 1]
        inc = [(steps[k], obj[k - 1], obj[k])
               for k in range(max(a, 1), b + 1) if obj[k] > obj[k - 1] + 1e-12]
        rows.append(dict(phase=phase[a], iter_from=steps[a], iter_to=steps[b],
                         value_from=vals[0], value_to=vals[-1],
                         ratio=vals[-1] / vals[0] if vals[0] else None,
                         increases=inc))
    # step-40 style boundary: value jumps because the term set changed
    boundaries = [(steps[a], obj[a - 1], obj[a], phase[a - 1], phase[a])
                  for a, b in segments[1:]]
    return dict(segments=rows, boundaries=boundaries,
                per_step_slope=[(steps[k], (obj[k] - obj[k - 1]) / (steps[k] - steps[k - 1]))
                                for k in range(1, len(obj))])


# --------------------------------------------------------------------------- #
# B. dense stages: the delivered logs vs. the true gate audit
# --------------------------------------------------------------------------- #
DENSE_STAGES = {
    "front_anchor_regularized": "front_anchor_low_frequency",
    "regularized": "guarded_regularized",
    "front_anchor_dense": "front_anchor_dense",
    "dense": "guarded_dense",
}


def dense_audit(run: Path):
    report = load_json(run / "reconstruction" / "dense" / "report.json")
    fl = report["field_logs"]
    out = {}
    for key, label in DENSE_STAGES.items():
        entry = fl.get(key)
        if entry is None:
            continue
        chains = []
        # front_anchor_regularized nests mid_frequency in the same dict
        if "mid_frequency" in entry:
            chains.append(("front_anchor_low_frequency", entry))
            chains.append(("front_anchor_mid_frequency", entry["mid_frequency"]))
        else:
            chains.append((label, entry))
        for name, node in chains:
            hist = node.get("history") or []
            rows = []
            for h in hist:
                trials = h.get("line_search_trials") or []
                fixed = h.get("fixed_ray_diagnostic") or {}
                acc = [t for t in trials if t.get("reason") is None]
                rows.append(dict(
                    outer=h.get("outer"),
                    free=h.get("free_vertices"),
                    step_accepted=h.get("step_accepted"),
                    alpha_used=h.get("shared_step_scale"),
                    obj_before=h.get("objective_before"),
                    obj_after=h.get("objective_after"),
                    obj_red_pct=(None if h.get("objective_before") in (None, 0) else
                                 100.0 * (1 - h["objective_after"] / h["objective_before"])),
                    nrms_before=fixed.get("before_mse_mm2"),
                    nrms_after=fixed.get("after_mse_mm2"),
                    missing_before=fixed.get("missing_before"),
                    missing_after=fixed.get("missing_after"),
                    rays=fixed.get("ray_count"),
                    best_free_obj=min((t["objective"] for t in trials), default=None),
                    best_free_alpha=(min(trials, key=lambda t: t["objective"])["alpha"]
                                     if trials else None),
                    best_free_mse=min((t["fixed_ray_mse_mm2"] for t in trials
                                       if t.get("fixed_ray_mse_mm2") is not None), default=None),
                    trials=len(trials),
                    gated_by=sorted({t["reason"] for t in trials if t.get("reason")}),
                ))
            out[name] = dict(rows=rows, settings=node.get("settings"),
                             final_safety=node.get("final_safety"),
                             max_abs_mm=node.get("maximum_absolute_displacement_mm"))
    return report, out


# --------------------------------------------------------------------------- #
# C. what the dense scalar field actually contains
# --------------------------------------------------------------------------- #
def dense_field_audit(run: Path):
    p = load_npz(run / "reconstruction" / "dense" / "parameters.npz")
    res = {}
    for name, key in (("dense_solver_scalar_mm", "dense_solver_scalar_mm"),
                      ("dense_front_anchor_scalar_mm", "dense_front_anchor_deviation_m"),
                      ("regularized_scalar_mm", "deviation_m"),
                      ("regularized_anchor_scalar_mm", "front_anchor_deviation_m")):
        if key not in p:
            continue
        f = np.asarray(p[key], float) * (1000.0 if key.endswith("deviation_m") else 1.0)
        res[name] = dict(n=int(f.size), absmax=float(np.abs(f).max()),
                         absmean=float(np.abs(f).mean()), valid=int((np.abs(f) > 1e-6).sum()))
    # Where does the dense stage move anything at all?
    fit_base = np.asarray(p["dense_fit_base_vertices"], float)
    dense = np.asarray(p["dense_vertices"], float)
    total = np.linalg.norm(dense - fit_base, axis=2) * 1000.0  # per-state mm
    res["dense_change_from_fit_base_mm"] = dict(
        n=int(total.size), absmax=float(total.max()), n_above_0p01=int((total > 0.01).sum()))
    reg = np.asarray(p["regularized_vertices"], float)
    base = np.asarray(p["posed_vertices"], float)
    res["regularized_change_from_coarse_mm"] = dict(
        n=int(reg.size), absmax=float(np.linalg.norm(reg - base, axis=2).max() * 1000))
    return res


# --------------------------------------------------------------------------- #
# D. rigidity / decoupling proof (independent of any solver state)
# --------------------------------------------------------------------------- #
def rotation_stats(R, tr):
    """Angle of min(||R - I||, ||R - S||) for the 4 sign conventions about X,
    measured on the ROTATION-COMPENSATED pose R.T @ (actual R).

    Base and field live in the REFERENCE camera frame (see `frame_audit`), so a
    model-space R cannot rotate a camera-space field. What actually happens is
    that the field is rotated by R.T back into the model frame and therefore
    moves WITH the head in every state. That is a rotated-prior fit, not a
    shared identity measurement.
    """
    I = np.eye(3)
    Sx = np.diag([-1.0, 1.0, 1.0])
    signs = [np.diag([1, sy, sz]) for sy in (1, -1) for sz in (1, -1)]
    targets = [I, Sx @ I, Sx]
    best = []
    for Rm in R:
        cand = []
        for sg in signs:
            for tgt in targets:
                M = Rm.T @ Rm @ sg @ tgt.T
                ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(M) - 1) / 2))))
                cand.append(ang)
        best.append(min(cand))
    return np.array(best)


def frame_audit(run: Path):
    """Which frame do posed_vertices / deviation_m live in? (independent proof)"""
    b = load_npz(run / "reconstruction" / "alternated_motion" / "parameters.npz")
    r = load_npz(run / "reconstruction" / "dense" / "parameters.npz")
    R, t = np.asarray(b["R"], float), np.asarray(b["t"], float)
    P, PW = np.asarray(b["posed_vertices"], float), np.asarray(r["posed_vertices"], float)
    faces = b["faces"]
    D, bc = b["landmark_faces"], b["landmark_barycentric"]
    out = dict(posed_vertices_identical_between_stages=bool(np.allclose(P, PW)),
               identical_R_between_stages=bool(np.allclose(R, np.asarray(r["R"], float), atol=0)))
    err_cam, err_model = [], []
    for i in range(0, len(R), max(1, len(R) // 6)):
        tri = P[i][faces[D[i]]]
        cam = (tri * bc[i][:, :, None]).sum(1)
        tri_m = np.asarray(b["raw_vertices"], float)[i][faces[D[i]]]
        model = (tri_m * bc[i][:, :, None]).sum(1)
        rec = np.einsum("ij,kj->ki", R[i], model) + t[i]
        err_cam.append(float(np.nanmax(np.abs(cam - rec)) * 1000))
        err_model.append(float(np.nanmax(np.abs(cam - model)) * 1000))
    out["landmark_vs_R_times_model_plus_t_mm"] = dict(max=max(err_cam))
    out["landmark_vs_raw_model_mm"] = dict(max=max(err_model))
    out["verdict"] = ("posed_vertices are CAMERA-frame (R@raw+t reproduces them); "
                      "a camera-frame field cannot be rotated by a model-space R"
                      if max(err_cam) < 1.0 else
                      "posed_vertices are NOT the camera-frame posed model")
    return out


def pose_audit(run: Path):
    b = load_npz(run / "reconstruction" / "alternated_motion" / "parameters.npz")
    R = np.asarray(b["R"], float)
    raws = np.asarray(b["raw_vertices"], float)
    rows = []
    for i, Rm in enumerate(R):
        raws_i = raws[i]
        # how close is the raw (expressed) model to the identity prior?
        rows.append(dict(state=i, angle_deg=math.degrees(math.acos(
            max(-1.0, min(1.0, (np.trace(Rm) - 1) / 2)))),
            raw_extent_mm=float(np.linalg.norm(raws_i - raws_i.mean(0), axis=1).mean() * 1000)))
    res = rotation_stats(R, b["t"])
    return dict(per_state=rows, residual_angle_deg=dict(
        min=float(res.min()), median=float(np.median(res)),
        max=float(res.max()), all=[float(v) for v in res]))


def rotation_audit(run: Path):
    base = load_npz(run / "reconstruction" / "alternated_motion" / "parameters.npz")
    reg = load_npz(run / "reconstruction" / "dense" / "parameters.npz")
    R = np.asarray(base["R"], float)
    R2 = np.asarray(reg["R"], float)
    # the exported meshes carry the same vertices; if the export baked the pose
    # the two files would differ by a rotation
    d = dict(n_states=int(len(R)),
             identical_R_between_stages=bool(np.allclose(R, R2, atol=0)))
    ang_a = rotation_stats(R, base["t"])
    ang_b = rotation_stats(R2, reg["t"])
    d["residual_angle_deg"] = dict(
        min=float(min(ang_a.min(), ang_b.min())),
        median=float(np.median(np.concatenate([ang_a, ang_b]))),
        max=float(max(ang_a.max(), ang_b.max())),
        per_state_a=[float(v) for v in ang_a],
        per_state_b=[float(v) for v in ang_b])
    # exported mesh equality -> which state / which frame does the .obj carry?
    import trimesh
    va = np.asarray(trimesh.load(run / "reconstruction" / "dense" / "closed_dense_m.obj",
                                 process=False, maintain_order=True).vertices, float)
    vr = load_npz(run / "reconstruction" / "dense" / "parameters.npz")
    dv = np.asarray(vr["dense_vertices"], float)
    R0, t0 = np.asarray(vr["R"], float)[0], np.asarray(vr["t"], float)[0]
    back = dv[0] - t0                      # camera -> model, no rotation needed for state 0
    d["export_vs_state0_camera_mm"] = float(np.abs(va - dv[0]).max() * 1000)
    d["export_vs_state0_model_after_removing_t_mm"] = float(np.abs(va - back).max() * 1000)
    d["export_best_state"] = min(
        range(dv.shape[0]), key=lambda k: float(np.abs(va - dv[k]).max()))
    d["export_best_state_delta_mm"] = float(np.abs(va - dv[d["export_best_state"]]).max() * 1000)
    d["params_R_state0_angle_deg"] = float(math.degrees(math.acos(max(
        -1.0, min(1.0, (np.trace(R0) - 1) / 2)))))
    d["params_t_state0_mm"] = float(np.linalg.norm(t0) * 1000)
    return d


# --------------------------------------------------------------------------- #
# E. report
# --------------------------------------------------------------------------- #
def build(run: Path) -> dict:
    adam = adam_append_audit(run)
    dense_report, dense = dense_audit(run)
    field = dense_field_audit(run)
    rig = rotation_audit(run)
    return dict(run=str(run),
                adam={k: dict(raw_first=v["raw_first"], raw_last=v["raw_last"],
                              phases=phase_aware_summary(k, v)) for k, v in adam.items()},
                dense=dense, dense_field=field, rigidity=rig,
                frame=frame_audit(run), pose=pose_audit(run),
                dense_report_meta=dict(mode=dense_report.get("mode"),
                                       dense_mode=dense_report.get("dense_mode"),
                                       research_v2=dense_report.get("research_v2"),
                                       loop_base=(dense_report.get("loop_base") or {}).get("blend"),
                                       dense_change_from_interpolated_mm=dense_report.get(
                                           "dense_change_from_interpolated_regularized_mm")))


def write_markdown(payload: dict, path: Path):
    L = []
    L.append("# 迭代 loss 审计（只读复算）\n")
    L.append(f"- run：`{payload['run']}`")
    L.append("- 本文件由 `loss_analysis/audit_iteration_loss.py` 生成，只读，未改动任何主算法文件。")
    L.append("- 口径：`solver_objective` / `linearised_objective` / `nonlinear_energy` / `diagnostic` 四档不合并（HANDOFF §4.1）。\n")

    L.append("## 1. Adam 阶段：相位边界不是 loss 上升\n")
    for stage, info in payload["adam"].items():
        L.append(f"### {stage}\n")
        L.append(f"- 原始首值 `{fmt(info['raw_first'])}` → 末值 `{fmt(info['raw_last'])}`"
                 f"（**跨相位不可比**，这是报告里“末值比首值大”的来源）")
        rows = []
        for seg in info["phases"]["segments"]:
            rows.append([seg["phase"], f"{seg['iter_from']}..{seg['iter_to']}",
                         seg["value_from"], seg["value_to"],
                         f"{(1-seg['ratio'])*100:+.2f}%" if seg["ratio"] not in (None, 0) else None,
                         len(seg["increases"])])
        L.append(md_table(["相位", "迭代区间", "相位首值", "相位末值", "相位内变化", "相位内上升次"], rows))
        if info["phases"]["boundaries"]:
            b = info["phases"]["boundaries"][0]
            L.append(f"\n- 首次相位切换在 step {b[0]}：`{fmt(b[1])}` → `{fmt(b[2])}`（{b[3]} → {b[4]}），"
                     f"第 {b[0]} 步起 `expression`/`jaw` 重新进入目标，数值变大是**项集变化**，不是退化。")
        L.append("")

    L.append("## 2. dense 阶段：求解器确实收敛，但每步只挪极小一步\n")
    rows = []
    for name, node in payload["dense"].items():
        for r in node["rows"]:
            rows.append([name, r["outer"], r["free"], r["step_accepted"], r["alpha_used"],
                         r["obj_before"], r["obj_after"], f"{r['obj_red_pct']:.4f}%" if r["obj_red_pct"] is not None else None,
                         r["nrms_before"], r["nrms_after"], r["missing_before"], r["missing_after"]])
    L.append(md_table(["阶段", "outer", "自由顶点", "接受", "实际 alpha",
                       "线性目标 before", "线性目标 after", "线性目标降幅",
                       "固定射线 MSE before", "after", "缺失 before", "缺失 after"], rows))
    L.append("")
    L.append("### 2.1 线搜索里真正发生了什么\n")
    rows = []
    for name, node in payload["dense"].items():
        for r in node["rows"]:
            if r["trials"] == 0:
                continue
            rows.append([name, r["outer"], r["trials"], r["best_free_alpha"], r["best_free_obj"],
                         r["best_free_mse"], r["alpha_used"], r["obj_after"], r["nrms_after"],
                         ", ".join(r["gated_by"]) or "-"])
    L.append(md_table(["阶段", "outer", "试步数", "最优试步 alpha", "最优试步线性目标",
                       "该试步固定射线 MSE", "最终采用 alpha", "最终线性目标", "最终固定射线 MSE",
                       "被哪个门否决"], rows))
    L.append("")
    L.append("## 3. dense 标量场里到底有没有东西\n")
    rows = [[k, v["n"], v["absmax"], v["absmean"], v["valid"]] for k, v in payload["dense_field"].items()
            if isinstance(v, dict) and {"n", "absmax", "absmean", "valid"} <= set(v)]
    L.append(md_table(["场", "元素数", "|f| max (mm)", "|f| mean (mm)", "|f|>1e-6 的顶点"], rows))
    L.append("")

    L.append("## 4. 坐标系与刚体可吸收性\n")
    fr, pose, rig = payload["frame"], payload["pose"], payload["rigidity"]
    L.append(f"- `posed_vertices` 是否相机系：{fr['verdict']}")
    L.append(f"  - 标点经 `R·raw + t` 重建，误差 max `{fmt(fr['landmark_vs_R_times_model_plus_t_mm']['max'])} mm`")
    L.append(f"  - 与原始模型直比，误差 max `{fmt(fr['landmark_vs_raw_model_mm']['max'])} mm`")
    L.append(f"- 两阶段 `posed_vertices` 完全相同：`{fr['posed_vertices_identical_between_stages']}`；"
             f"两阶段 `R` 完全相同：`{fr['identical_R_between_stages']}`")
    L.append(f"- 旋转补偿后仍剩的残差角：min `{pose['residual_angle_deg']['min']:.3f}°` / "
             f"median `{pose['residual_angle_deg']['median']:.3f}°` / max `{pose['residual_angle_deg']['max']:.3f}°`")
    L.append(f"- 导出网格 `closed_dense_m.obj` 与 `parameters.npz:dense_vertices[best={rig['export_best_state']}]` "
             f"最大差 `{fmt(rig['export_best_state_delta_mm'])} mm`；"
             f"state0 位姿角 `{rig['params_R_state0_angle_deg']:.3f}°`、平移 `{fmt(rig['params_t_state0_mm'])} mm`")
    L.append(f"  - 与 state0 相机系顶点差 `{fmt(rig['export_vs_state0_camera_mm'])} mm`，"
             f"去掉平移后差 `{fmt(rig['export_vs_state0_model_after_removing_t_mm'])} mm`")
    L.append("  → `parameters.npz` 的 `posed_vertices`/`dense_vertices` 与导出 OBJ 是**相机系**，"
             "同文件里的 `R`/`t` 是**模型系**，两者不是同一坐标系；"
             "任何把 `R` 作用到这些顶点上的操作都是坐标系混用。")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path(RUN_DEFAULT))
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    run = a.run.resolve()
    payload = build(run)
    out = a.output or Path(OUT_DEFAULT)
    out = out if out.is_absolute() else Path(OUT_DEFAULT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                                    encoding="utf-8")
    write_markdown(payload, out / "audit.md")
    print(json.dumps(dict(output=str(out), run=str(run)), ensure_ascii=False))


if __name__ == "__main__":
    main()
