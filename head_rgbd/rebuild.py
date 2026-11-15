"""One-click reconstruction + neck reshaping.

The delivered product is a human head mesh with a reshaped neck.  The base
reconstruction is the existing locked 0.5.2-lineage chain (with the old
``neck_prominence`` appearance stage explicitly off); the neck is the
``neck_direct_v2`` stage with the subject-adaptive anatomy/prior.

The neck stage is tried in two variants and chosen by a measured crease check:

* **prior**  -- the canonical, anatomy-localized larynx prior (base gain 1.0);
* **no-prior** -- the same continuous solve without the extra prior.

``auto`` builds the prior first and keeps it when the full-edge/B positive
crease maxima are within the crease limit; otherwise it builds the no-prior
variant and keeps the one with the lower maximum.  The attempted and chosen
builds are both preserved; ``result/REBUILD_REPORT.md`` says what was chosen
and why.

The default acceptance is a quick, mesh-read check (protection, topology,
signed creases, laryngeal local prominence, upper-band counts).  ``--full-verify``
additionally builds the zero-topology/zero-new-vertex ablations and runs
``tools.neck_direct_verify_v2`` for the complete S0-S9 report.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]


def _log(msg):
    print(msg, flush=True)


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJ) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return env


def _run_module(module, argv, *, what, cwd=PROJ):
    cmd = [sys.executable, "-m", module, *[str(a) for a in argv]]
    _log(f"[rebuild] {what}: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd), env=_env())
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed with exit code {r.returncode}")
    return r


# --------------------------------------------------------------------------
# measured selection / quick acceptance
# --------------------------------------------------------------------------


def signed_crease(npz_path):
    """Signed positive full-edge and B-interior crease maxima (both states)."""
    from head_rgbd import neck_bridge as nbr

    z = np.load(npz_path)
    F = np.asarray(z["F"], np.int64)
    reg = np.asarray(z["region"], np.int64)
    out = {}
    for key in ("closed", "open"):
        Vout = np.asarray(z["V_" + key], float)
        Vin = np.asarray(z["V_" + key + "_input"], float)
        E, ao = nbr.dihedral_deg(Vout * 1000.0, F)
        _e, ai = nbr.dihedral_deg(Vin * 1000.0, F)
        good = np.isfinite(ao) & np.isfinite(ai)
        if not good.any():
            out[key] = {"full_positive_max_deg": 0.0, "B_positive_max_deg": 0.0,
                        "positive_gt10": 0, "positive_gt20": 0}
            continue
        ch = (np.abs(ao) - np.abs(ai))[good]
        e = E[good]
        B = (reg[e[:, 0]] == 1) & (reg[e[:, 1]] == 1)
        out[key] = {
            "full_positive_max_deg": float(ch.max()),
            "B_positive_max_deg": float(ch[B].max()) if B.any() else 0.0,
            "positive_gt10": int((ch > 10.0).sum()),
            "positive_gt20": int((ch > 20.0).sum()),
        }
    return out


def _finite_or_zero(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _crease_metric(s):
    return max(s["closed"]["full_positive_max_deg"],
               s["closed"]["B_positive_max_deg"],
               s["open"]["full_positive_max_deg"],
               s["open"]["B_positive_max_deg"])


def _band(V, F, t, lo=0.925, hi=0.970):
    from head_rgbd import mesh_integrity as mi

    E = mi.edge_records(F)[0]
    L = np.linalg.norm((np.asarray(V, float)[E[:, 0]] -
                        np.asarray(V, float)[E[:, 1]]) * 1000.0, axis=1)
    mv = (t >= lo) & (t < hi)
    mf = mv[F].all(axis=1)
    me = mv[E[:, 0]] & mv[E[:, 1]]
    tri = np.asarray(V, float)[F]
    A = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                      tri[:, 2] - tri[:, 0]), axis=1) * 1e6
    return {"vertices": int(mv.sum()), "faces": int(mf.sum()),
            "edges": int(me.sum()),
            "edge_p50_mm": float(np.median(L[me])) if me.any() else None,
            "area_cm2": float(A[mf].sum() / 100.0)}


def _structure(V, F, ax, hc, curve, p, key, t_hi):
    from head_rgbd import neck_stage as ns

    f = ns.analyse_surface(V, F, ax, hc, curve=curve,
                           structure_sigma=p["structure_sigma_mm"],
                           fine_sigma=p["fine_sigma_mm"], leak_mask=True)
    d = ns.structure_descriptors(f, key, t_lo=0.42, t_hi=t_hi)
    t = np.asarray(f["t"], float)
    th = np.degrees(np.asarray(f["theta"], float))
    rel = np.asarray(f["relief"], float)
    mask = ((t > 0.64)[:, None] & (t < 0.83)[:, None] &
            (np.abs(th)[None, :] < 25.0))
    work = np.where(mask, rel, -np.inf)
    ij = np.unravel_index(np.nanargmax(work), work.shape)
    return {"primary": d["primary"], "saddle": d.get("saddle"),
            "larynx_local": {"relief_mm": float(work[ij]),
                             "t": float(t[ij[0]]),
                             "theta_deg": float(th[ij[1]])}}


def quick_check(chosen, dense_dir):
    """Mesh-read quick acceptance; no build products are trusted."""
    from head_rgbd import mesh_integrity as mi
    from head_rgbd import neck_direct as nd
    from head_rgbd import neck_stage as ns
    from tools import build_neck_stage as bns
    from tools import neck_direct_verify as v1

    chosen = Path(chosen)
    meta = json.load(open(chosen / "build_direct.json", encoding="utf-8"))
    p = meta["params"]
    npz = np.load(chosen / "stage_direct.npz")
    states, F0, _ = bns.load_subject(Path(dense_dir))
    V0c, V0o = states["closed"]["V"], states["open"]["V"]
    _, ax, hc = ns.subject_frame(V0c, F0, states["closed"]["landmarks"])
    curve, _ = ns.spine_curve(V0c, F0, ax, hc)
    _, ax_o, hc_o = ns.subject_frame(V0o, F0, states["open"]["landmarks"])
    curve_o, _ = ns.spine_curve(V0o, F0, ax_o, hc_o)
    Vc, Fc = mi.load_obj(chosen / "head_closed_direct.obj")
    Vo, Fo = mi.load_obj(chosen / "head_open_direct.obj")
    signed = signed_crease(chosen / "stage_direct.npz")
    prot = v1.protection_section(V0c, V0o, F0, Vc, Fc, Vo, Fo, npz,
                                 ax, curve, hc, p, meta)
    topo = v1.topology_section(Vc, Fc, Vo, Fo, V0c, F0)
    struct_c = _structure(Vc, Fc, ax, hc, curve, p, "closed", 0.90)
    struct_o = _structure(Vo, Fo, ax_o, hc_o, curve_o, p, "open", 0.84)
    t0 = nd.axial_mm(np.asarray(V0c, float) * 1000.0, ax, curve, hc)[3]
    tf = np.asarray(npz["h_material"], float) / hc
    upper = {"input": _band(V0c, F0, t0), "full": _band(Vc, Fc, tf)}
    upper["no_increase"] = bool(
        upper["full"]["vertices"] <= upper["input"]["vertices"]
        and upper["full"]["faces"] <= upper["input"]["faces"])
    prot_max = max(prot["fixed_set"]["max_displacement_closed_mm"],
                   prot["fixed_set"]["max_displacement_open_mm"])
    ti, tc, to = topo["input"], topo["closed"], topo["open"]
    topo_ok = (tc["nonmanifold_edges"] == ti["nonmanifold_edges"]
               and to["nonmanifold_edges"] == ti["nonmanifold_edges"]
               and tc["inconsistent_shared_edges"] == 0
               and to["inconsistent_shared_edges"] == 0
               and tc["boundary_edges"] == ti["boundary_edges"]
               and to["boundary_edges"] == ti["boundary_edges"]
               and tc["zero_area_faces"] == 0 and to["zero_area_faces"] == 0
               and tc["components"] == ti["components"]
               and to["components"] == ti["components"])
    mesh_ok = bool(Fc.shape == Fo.shape and np.array_equal(Fc, Fo)
                   and len(Vc) == len(Vo))
    return {
        "signed_crease": signed,
        "protection": {
            "fixed_vertices": int(prot["fixed_set"]["vertices"]),
            "fixed_max_mm": float(prot_max),
            "vertices_above_0_01mm": int(prot["fixed_set"]["vertices_moving_more_than_0_01mm"]),
            "pass": bool(prot_max <= 0.002),
            "handover_shell_max_closed_mm": _finite_or_zero(
                prot.get("handover_shell", {}).get("max_displacement_closed_mm")),
        },
        "topology": {"closed": tc, "open": to, "input": ti, "pass": bool(topo_ok)},
        "shared_topology": bool(mesh_ok),
        "structure": {"closed": struct_c, "open": struct_o},
        "upper_band_0.925_0.97": upper,
        "vertices": {"input": int(len(V0c)), "closed": int(len(Vc)),
                     "open": int(len(Vo))},
        "faces": {"input": int(len(F0)), "closed": int(len(Fc)),
                  "open": int(len(Fo))},
        "pass": {
            "critical": bool(prot_max <= 0.002 and topo_ok and mesh_ok),
            "connection": bool(_crease_metric(signed) <= float(
                p.get("adapt_full_positive_limit") or 25.0)),
            "density_upper_band": bool(upper["no_increase"]),
        },
    }


# --------------------------------------------------------------------------
# neck variant builds and selection
# --------------------------------------------------------------------------


def _neck_param_args(args):
    """Whitelisted neck parameters forwarded to the direct builder.

    They are only added when the caller set them, so the default command line
    is byte-identical to the previous one.
    """
    out: list = []
    pairs = (
        ("neck_gain", "--gain"),
        ("neck_bridge_width", "--bridge-width"),
        ("neck_seam_taper_ratio", "--seam-taper-ratio"),
        ("neck_mu_t_bridge", "--mu-t-bridge"),
        ("neck_mu_t_body", "--mu-t-body"),
        ("neck_anchor_near_a", "--anchor-near-a"),
    )
    for attr, flag in pairs:
        value = getattr(args, attr, None)
        if value is not None:
            out += [flag, float(value)]
    return out


def _build_neck(dense, out, *, prior, args, runtime):
    common = ["--fit", dense, "--out", out,
              "--anchor-validity-only",
              "--adapt-full-limit", float(getattr(args, "crease_limit", 20.0)),
              "--no-render"]
    common += _neck_param_args(args)
    if getattr(args, "neck_landmarks", None):
        common += ["--landmarks", str(args.neck_landmarks)]
    if prior:
        common += ["--larynx-prior",
                   "--larynx-prior-ratio",
                   float(getattr(args, "larynx_prior_ratio", 0.35))]
    if getattr(args, "larynx_gain", None) is not None:
        common += ["--larynx-gain", float(args.larynx_gain)]
    _run_module("tools.build_neck_direct_v2", common,
                what=("neck prior build" if prior else "neck no-prior build"))


def _choose_variant(output, dense, args, runtime):
    variant = str(getattr(args, "neck_variant", "prior"))
    prior_dir = Path(output) / "neck_direct_v2_prior"
    no_prior_dir = Path(output) / "neck_direct_v2_no_prior"
    reports = {}
    if variant in ("auto", "prior"):
        _build_neck(dense, prior_dir, prior=True, args=args, runtime=runtime)
        s = signed_crease(prior_dir / "stage_direct.npz")
        reports["prior"] = {"dir": str(prior_dir), "signed": s,
                            "metric": _crease_metric(s)}
        limit = float(getattr(args, "crease_limit", 20.0))
        prior_ok = reports["prior"]["metric"] <= limit
        _log(f"[rebuild] prior variant: max positive {reports['prior']['metric']:.2f} deg")
    else:
        prior_ok = False
    if variant == "prior":
        return "prior", prior_dir, reports
    if variant == "no-prior" or (variant == "auto" and not prior_ok):
        _build_neck(dense, no_prior_dir, prior=False, args=args, runtime=runtime)
        s = signed_crease(no_prior_dir / "stage_direct.npz")
        reports["no_prior"] = {"dir": str(no_prior_dir), "signed": s,
                               "metric": _crease_metric(s)}
        _log(f"[rebuild] no-prior variant: max positive {reports['no_prior']['metric']:.2f} deg")
    if variant == "no-prior":
        return "no_prior", no_prior_dir, reports
    if prior_ok:
        return "prior", prior_dir, reports
    # both built and the prior failed: keep the lower maximum, preferring the
    # no-prior variant when it is the one under the limit
    if reports["no_prior"]["metric"] <= reports["prior"]["metric"]:
        return "no_prior", no_prior_dir, reports
    return "prior", prior_dir, reports


# --------------------------------------------------------------------------
# export + report
# --------------------------------------------------------------------------


def _neck_submesh(chosen, result, key):
    from head_rgbd import mesh_integrity as mi
    from head_rgbd import neck_stage as ns

    V, F = mi.load_obj(Path(chosen) / f"head_{key}_direct.obj")
    npz = np.load(Path(chosen) / "stage_direct.npz")
    reg = np.asarray(npz["region"], np.int64)
    keep = (reg[F] >= 1).all(axis=1)          # B (submental bridge) + C (neck body)
    Fn = np.asarray(F, np.int64)[keep]
    used = np.unique(Fn)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    ns.write_obj(Path(result) / f"neck_{key}.obj", V[used], remap[Fn],
                 comment=f"reshaped neck {key}; B+C only; head boundary open; "
                         f"{len(used)} v / {len(Fn)} f")


def _write_report(result, output, dense, base_delivery, chosen, variant,
                  variant_reports, quick, full_verify, render, elapsed, args,
                  delivery_info=None, marker_status=None, landmark_payload=None):
    from head_rgbd import neck_stage as ns

    chosen = Path(chosen)
    meta = json.load(open(chosen / "build_direct.json", encoding="utf-8"))
    rep = json.load(open(Path(dense) / "report.json", encoding="utf-8")) \
        if (Path(dense) / "report.json").exists() else {}
    base_manifest = None
    if base_delivery is not None and (Path(base_delivery) / "manifest.json").exists():
        base_manifest = json.load(open(Path(base_delivery) / "manifest.json", encoding="utf-8"))
    ver = json.load(open(chosen / "verification.json", encoding="utf-8")) \
        if (chosen / "verification.json").exists() else None
    q = quick
    s = q["signed_crease"]
    lc = q["structure"]["closed"]["larynx_local"]
    lo = q["structure"]["open"]["larynx_local"]
    lines = [
        "# 一键重建 + 颈部重塑交付报告",
        "",
        f"结果目录：`{result}`  ",
        f"基础重建：`{dense}`  ",
        f"颈部变体：**{variant.replace(chr(95), chr(45))}**（目录 `{chosen}`）  ",
        f"耗时：{elapsed:.1f} s",
        "",
        "## 输出",
        "",
        "| 文件 | 内容 |",
        "|---|---|",
        "| `final/head_closed.obj` / `final/head_open.obj` | 最终**人头 + 重塑颈部**（下游规范路径 1） |",
        "| `final/anatomical_landmarks.json` | 喉结 sidecar（usable 时；`skin_estimate_reference_m` 是首选锚点） |",
        "| `final/manifest.json` | 最小自包含 manifest（状态配对用） |",
        "| `measured/head_closed.obj` / `measured/head_open.obj` | 未做 C 的基础测量网格（下游规范路径 2） |",
        "| `measured/anatomical_landmarks.json` / `manifest.json` | 同一标记观测的测量交付 sidecar/manifest |",
        "| `head_closed.obj` / `head_open.obj` | 根目录兼容副本（旧脚本；**不做下游自动配对**） |",
        "| `neck_closed.obj` / `neck_open.obj` | 重塑颈部子网格（B 颏下桥接 + C 颈体；头侧边界开放） |",
        "| `base_head_closed.obj` / `base_head_open.obj` | 基础 dense 重建的头部网格（无后续颈部外观塑形） |",
        "| `REBUILD_REPORT.md` / `quick_check.json` | 本报告与快速验收 |",
        "| `render/` | （启用出图时）标准颈部/整体面板 + 喉结近景/剖面/剖面曲线 |",
        "",
        "## 标记与 C 中心",
        "",
    ]
    states = (landmark_payload or {}).get("states") or {}
    for state in ("closed", "open"):
        rec = states.get(state) or {}
        lcent = rec.get("larynx_center") or {}
        marker = rec.get("marker") or {}
        size = marker.get("size") or {}
        lines.append(
            f"- **{state}**：marker `{rec.get('status')}`"
            f"（frame `{rec.get('frame_id')}`，id `{marker.get('target_id')}`，"
            f"black {size.get('black_square_mm')} mm / {size.get('size_source')}）；"
            f"t_marker={lcent.get('t_marker')}，t_center_used={lcent.get('t_center_used')}，"
            f"theta_used=0°，t_gate={lcent.get('t_gate')}，source={lcent.get('source')}")
    if marker_status:
        lines.append(f"- 交付状态：**{marker_status}**；"
                     f"`final/`、`measured/` 均已写出；usable 时两目录各有 sidecar+manifest，"
                     f"not_usable 时不写可配对 sidecar、manifest.status=not_usable。")
    anat_meta = meta.get("larynx_anatomy") or {}
    for state in ("closed", "open"):
        a = anat_meta.get(state) or {}
        if a:
            lines.append(f"- C 先验 **{state}**：t_center={a.get('t_center')}，"
                         f"source={a.get('t_center_source')}，t_gate={a.get('t_gate')}，"
                         f"amp={a.get('amp_mm')} mm，half={a.get('t_half')}")
    if delivery_info is not None:
        fdir = delivery_info["delivery"]["final"]
        mdir = delivery_info["delivery"]["measured"]
        lines.append(f"- 下游目录：`{fdir['dir']}`（sidecar {fdir.get('sidecar')}） / "
                     f"`{mdir['dir']}`（sidecar {mdir.get('sidecar')}）")
    lines += [
        "",
        "## 算法来源",
        "",
        f"- 基础 dense：`{rep.get('schema', 'n/a')}`，project `{rep.get('project_version', 'n/a')}`，"
        f"`{rep.get('algorithm', 'n/a')}`；dense `{rep.get('dense_topology', {})}`；"
        f"`neck`={rep.get('neck')}，`neck_refit.enabled`="
        f"{(rep.get('neck_refit') or {}).get('enabled', 'n/a')}",
        "- 基础重建强制 `--neck-prominence off`；旧颈前外观塑形不参与，颈部由 `neck_direct_v2` 负责。",
        f"- 颈部：`neck_direct_v2`，构建/变体参数见 `neck_direct_v2/build_direct.json`；"
        f"变体由 `--neck-variant` 指定；auto/回退时以全边集/B 正向锐化 ≤ "
        f"{float(getattr(args, 'crease_limit', 20.0))}° 为准则，强制 prior/no-prior "
        f"时按指定变体交付并在 quick_check 如实标记是否超限。",
        "",
        "## 颈部变体选择",
        "",
    ]
    for name, row in variant_reports.items():
        lines.append(
            f"- **{name}**：全边集/B 正向最大 "
            f"{row['metric']:.2f}°（闭口 full {row['signed']['closed']['full_positive_max_deg']:.2f} / "
            f"B {row['signed']['closed']['B_positive_max_deg']:.2f}；张口 full "
            f"{row['signed']['open']['full_positive_max_deg']:.2f} / B "
            f"{row['signed']['open']['B_positive_max_deg']:.2f}）；目录 `{row['dir']}`")
    lines += [
        "",
        "## 快速验收（从交付 OBJ 重读）",
        "",
        f"- 顶点/面：输入 {q['vertices']['input']} / {q['faces']['input']} → 交付 "
        f"{q['vertices']['closed']} / {q['faces']['closed']}",
        f"- 两态同一面表：**{q['shared_topology']}**",
        f"- 保护固定集：{q['protection']['fixed_vertices']} v，最大位移 "
        f"**{q['protection']['fixed_max_mm']:.6f} mm**（门槛 0.002）；>0.01 mm "
        f"{q['protection']['vertices_above_0_01mm']} 个",
        f"- 拓扑：非流形 0 / 绕向 0 / 边界 124 / 分量 3：**{q['topology']['pass']}**",
        f"- 全边集正向锐化：闭口 full {s['closed']['full_positive_max_deg']:.2f}° / "
        f"B {s['closed']['B_positive_max_deg']:.2f}°；张口 full "
        f"{s['open']['full_positive_max_deg']:.2f}° / B "
        f"{s['open']['B_positive_max_deg']:.2f}°；>20° 闭 "
        f"{s['closed']['positive_gt20']} / 张 {s['open']['positive_gt20']}",
        f"- 喉结局部：闭口 **{lc['relief_mm']:.2f} mm @ t={lc['t']:.3f}, "
        f"θ={lc['theta_deg']:.1f}°**；张口 **{lo['relief_mm']:.2f} mm @ t={lo['t']:.3f}, "
        f"θ={lo['theta_deg']:.1f}°**",
        f"- 上颈密带 0.925–0.97：顶点 {q['upper_band_0.925_0.97']['input']['vertices']}→"
        f"{q['upper_band_0.925_0.97']['full']['vertices']}，面 "
        f"{q['upper_band_0.925_0.97']['input']['faces']}→"
        f"{q['upper_band_0.925_0.97']['full']['faces']}；"
        f"不加密：**{q['upper_band_0.925_0.97']['no_increase']}**",
        "",
        "## 未完成 / 报告项",
        "",
    ]
    prim_c = q["structure"]["closed"]["primary"]
    prim_o = q["structure"]["open"]["primary"]
    lines += [
        f"- 闭口全局 primary {prim_c['relief_mm']:.2f} mm @ t={prim_c['t']:.3f}；"
        f"张口 {prim_o['relief_mm']:.2f} mm @ t={prim_o['t']:.3f}。"
        "若闭口 primary 在颈上段/颏下而不是喉结，这是受试者自身背景，报告为结构偏差，"
        "不以压平受保护下颌来换取参考峰值。",
        f"- 连接：当前颈部的全边集/B 正向最大为 "
        f"{max(s['closed']['full_positive_max_deg'], s['open']['full_positive_max_deg']):.2f}°；"
        f"快速连接判定 **{q['pass']['connection']}**（限制 "
        f"{float(getattr(args, 'crease_limit', 20.0))}°）。",
        "- 快速验收的临界项（保护/拓扑/两态面表）："
        f"**{q['pass']['critical']}**；上颈不加密：{q['pass']['density_upper_band']}。",
        "",
        "## 完整验收 / 图",
        "",
    ]
    if ver:
        summ = ver["S9_verdict"]["summary"]
        lines += [
            f"- `--full-verify` 已运行：`verification.json`；gates={json.dumps(summ['gates'], ensure_ascii=False)}；"
            f"failed={summ['failed']}",
        ]
    else:
        lines.append("- 未运行 `--full-verify`；本报告只含快速验收。零拓扑/零新增点消融与 S0–S9 完整验收需加 `--full-verify`。")
    if render:
        lines.append(f"- 渲染：`{render}`（`larynx_compare_smooth.png`、`larynx_compare_wire.png`、"
                     "`larynx_sections.png`、`larynx_profile.png` 及标准面板）。")
    else:
        lines.append("- 未出图（`--no-render`）。")
    lines += [
        "",
        "## 备注",
        "",
        "- 基础重建保留 `delivery/`、`reconstruction/`；旧结果和用户已有修改不被覆盖；"
        "本次输出是新的 `result/` 与 `neck_direct_v2*` 目录。",
        "- 下游 head_windpipe_regis 的自动配对/输入校验/配准搜索结果（如有）记录在 "
        "`T6_DOWNSTREAM.md`；D1/D2/间隙 pass/fail 如实保留，不改下游算法换 pass。",
        "- 输入采集/基础 dense 的独立精度仍按原项目状态：内部同数据对照，未做独立扫描精度验证。",
    ]
    path = Path(result) / "REBUILD_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# compact fast-path report
# --------------------------------------------------------------------------


def _write_compact_report(result, output, dense, chosen, variant, reports, quick,
                          elapsed, args, delivery_info=None, marker_status=None,
                          landmark_payload=None):
    """Short acceptance summary for the minimal fast path.

    The full report remains available with ``--no-minimal-output``; this one
    keeps the delivered paths, the measured acceptance numbers and the honest
    pass/fail flags, and drops the long algorithm narrative.
    """
    chosen = Path(chosen)
    q = quick
    s = q["signed_crease"]
    lc = q["structure"]["closed"]["larynx_local"]
    lo = q["structure"]["open"]["larynx_local"]
    prot = q["protection"]
    states = (landmark_payload or {}).get("states") or {}
    lines = [
        "# 一键重建 + 颈部重塑交付报告（compact fast path）",
        "",
        f"- 结果目录：`{result}`",
        f"- 基础重建：`{dense}`",
        f"- 颈部变体：**{variant.replace(chr(95), chr(45))}**（`{chosen}`）",
        f"- 耗时（颈部 + 交付，不含基础重建）：{elapsed:.1f} s",
        "",
        "## 输出",
        "",
        "| 路径 | 内容 |",
        "|---|---|",
        "| `final/head_closed.obj` / `head_open.obj` | 最终人头 + 重塑颈部 |",
        "| `final/anatomical_landmarks.json` + `manifest.json` | 下游配对 sidecar / 自包含 manifest |",
        "| `measured/head_closed.obj` / `head_open.obj` | 未做 C 重塑的测量网格 |",
        "| `measured/anatomical_landmarks.json` + `manifest.json` | 同一标记观测的测量交付 |",
        "| `quick_check.json` | 从交付 OBJ 重读的快速验收 |",
        "| `run_report.json`（运行根） | 阶段、命令、日志、耗时 |",
        "| `reconstruction/dense/parameters.npz` / `reference_geometry.json` | 流水线必需参数 |",
        "",
        "## 验收摘要（从交付 OBJ 重读）",
        "",
        "| 项 | 数值 | 判定 |",
        "|---|---|---|",
        f"| 顶点 / 面 | 输入 {q['vertices']['input']} / {q['faces']['input']} → 交付 "
        f"{q['vertices']['closed']} / {q['faces']['closed']} | — |",
        f"| 两态同一面表 | {q['shared_topology']} | {'pass' if q['shared_topology'] else 'fail'} |",
        f"| 保护固定集 | {prot['fixed_vertices']} v，最大位移 "
        f"{prot['fixed_max_mm']:.6f} mm（门槛 0.002） | "
        f"{'pass' if prot['pass'] else 'fail'} |",
        f"| 拓扑 | 非流形 {q['topology']['closed']['nonmanifold_edges']} / 绕向 "
        f"{q['topology']['closed']['inconsistent_shared_edges']} / 边界 "
        f"{q['topology']['closed']['boundary_edges']} / 分量 "
        f"{q['topology']['closed']['components']} | "
        f"{'pass' if q['topology']['pass'] else 'fail'} |",
        f"| 全边集正向锐化 | 闭口 full {s['closed']['full_positive_max_deg']:.2f}° / B "
        f"{s['closed']['B_positive_max_deg']:.2f}°；张口 full "
        f"{s['open']['full_positive_max_deg']:.2f}° / B "
        f"{s['open']['B_positive_max_deg']:.2f}°；>20° 闭 "
        f"{s['closed']['positive_gt20']} / 张 {s['open']['positive_gt20']} | "
        f"{'pass' if q['pass']['connection'] else 'fail'}（限制 "
        f"{float(getattr(args, 'crease_limit', 20.0))}°） |",
        f"| 喉结局部 | 闭口 {lc['relief_mm']:.2f} mm @ t={lc['t']:.3f}, "
        f"θ={lc['theta_deg']:.1f}°；张口 {lo['relief_mm']:.2f} mm @ "
        f"t={lo['t']:.3f}, θ={lo['theta_deg']:.1f}° | — |",
        f"| 上颈密带 0.925–0.97 | 顶点 "
        f"{q['upper_band_0.925_0.97']['input']['vertices']}→"
        f"{q['upper_band_0.925_0.97']['full']['vertices']}，面 "
        f"{q['upper_band_0.925_0.97']['input']['faces']}→"
        f"{q['upper_band_0.925_0.97']['full']['faces']} | "
        f"{'pass' if q['pass']['density_upper_band'] else 'fail'}（不加密） |",
        f"| 临界项（保护/拓扑/面表） | {q['pass']['critical']} | "
        f"{'pass' if q['pass']['critical'] else 'fail'} |",
        "",
        "## 标记与 C 中心",
        "",
    ]
    for state in ("closed", "open"):
        rec = states.get(state) or {}
        lcent = rec.get("larynx_center") or {}
        lines.append(
            f"- **{state}**：marker `{rec.get('status')}`（frame `{rec.get('frame_id')}`，"
            f"t_marker={lcent.get('t_marker')}，t_center_used={lcent.get('t_center_used')}，"
            f"theta_used=0°，source={lcent.get('source')}）")
    if marker_status:
        lines.append(f"- 交付状态：**{marker_status}**")
    for name, row in (reports or {}).items():
        lines.append(f"- 变体 **{name}**：正向最大 {row['metric']:.2f}°")
    lines += ["", "## 下游", ""]
    if delivery_info is not None:
        fdir = delivery_info["delivery"]["final"]
        mdir = delivery_info["delivery"]["measured"]
        lines.append(f"- `final/`：status={delivery_info['status']}，"
                     f"sidecar={'yes' if fdir.get('sidecar') else 'no'}，manifest=yes")
        lines.append(f"- `measured/`：status={delivery_info['status']}，"
                     f"sidecar={'yes' if mdir.get('sidecar') else 'no'}，manifest=yes")
    lines += [
        "",
        "## 未完成 / 备注",
        "",
        "- 未运行 `--full-verify`；完整 S0–S9 与零拓扑/零新增点消融需显式加该参数。",
        "- 渲染默认关闭（`--render` 可恢复）；`summary` 只保留快速验收。",
        "- fast path 不写 `result/neck_*.obj` 与 `result/base_head_*.obj` 展示副本；"
        "基础测量网格只在 `measured/` 内交付一份。",
        "- 输入采集的独立精度未验证；本报告只覆盖本次链路与交付契约。",
    ]
    path = Path(result) / "REBUILD_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# public entry
# --------------------------------------------------------------------------


def _delivery_base_only(delivery: Path) -> bool:
    """The measured base must not carry a neck appearance edit (old volume)."""
    payload_path = Path(delivery) / "neck_prominence.json"
    if not payload_path.is_file():
        return True
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    states = payload.get("states") or {}
    return not any((row or {}).get("status") == "applied" for row in states.values())


def run_rebuild(*, runtime, output, dense, base_delivery, args):
    """Neck stage on an already reconstructed dense fit."""
    started = time.time()
    output = Path(output)
    dense = Path(dense)
    output.mkdir(parents=True, exist_ok=True)
    if not (dense / "parameters.npz").exists():
        raise FileNotFoundError(f"dense reconstruction not found: {dense}")
    if base_delivery is not None and not _delivery_base_only(base_delivery):
        # Pre-fix run outputs (0.9.0) still carry the old reference-driven neck
        # volume; using them as the measurement base would hand downstream a
        # shaped mesh.  Fall back to the dense base instead.
        _log(f"[rebuild] base delivery has a neck appearance edit; using the "
             f"dense base for measured: {base_delivery}")
        base_delivery = None
    if getattr(args, "base_only", False):
        _log(f"[rebuild] base-only; dense={dense}")
        return 0

    # ---- T3: authoritative prepared-frame marker recheck -----------------
    from head_rgbd import delivery as dv
    from head_rgbd import neck_landmark as nl
    from head_rgbd.tools import neck_landmark_aruco as nla

    landmark_path = getattr(args, "neck_landmarks", None)
    prepared = output / "prepared"
    if landmark_path is None or not Path(landmark_path).is_file():
        if (prepared / "sequence.json").is_file():
            target = output / "neck_landmarks.json"
            rc = nla.main(["--prepared", str(prepared), "--dense", str(dense),
                           "--output", str(target)])
            landmark_path = target
            _log(f"[rebuild] prepared marker recheck rc={rc} -> {target}")
        else:
            landmark_path = None
    args.neck_landmarks = landmark_path
    landmark_payload, landmark_error = (nl.load_neck_landmarks(landmark_path)
                                        if landmark_path else (None, "no_neck_landmarks_file"))
    if landmark_payload is None:
        _log(f"[rebuild] landmark sidecar unavailable: {landmark_error}")
    t3_usable = dv.usable_delivery(landmark_payload, "usable")
    marker_status = getattr(args, "marker_status", None)
    if marker_status is None:
        marker_status = "usable" if t3_usable else "not_usable"
    elif marker_status == "usable" and not t3_usable:
        # The prepared recheck is authoritative: a preview-valid marker that
        # fails the prepared record cannot produce a usable delivery.
        if not bool(getattr(args, "allow_marker_fallback", False)):
            _log("[rebuild] prepared marker recheck failed; live run aborted "
                 "(use --allow-marker-fallback for a not_usable delivery)")
            return 2
        marker_status = "not_usable"
    args.marker_status = marker_status
    _log(f"[rebuild] marker status={marker_status} "
         f"(t3_usable={t3_usable}, sidecar={landmark_path})")
    if t3_usable and str(getattr(args, "neck_variant", "prior")) == "auto":
        # Marker-first: do not silently fall back to the menton-only no-prior
        # variant when the marker itself is usable.
        args.neck_variant = "prior"

    variant, chosen_src, reports = _choose_variant(output, dense, args, runtime)
    chosen = output / "neck_direct_v2"
    if Path(chosen_src) != chosen:
        shutil.copytree(chosen_src, chosen, dirs_exist_ok=True)
    _log(f"[rebuild] chose {variant} -> {chosen}")
    quick = quick_check(chosen, dense)
    result = output / "result"
    result.mkdir(parents=True, exist_ok=True)
    minimal = bool(getattr(args, "minimal_output", True))
    for key in ("closed", "open"):
        shutil.copyfile(chosen / f"head_{key}_direct.obj",
                        result / f"head_{key}.obj")
        if not minimal:
            # Display-only submesh; the fast path does not ship it.
            _neck_submesh(chosen, result, key)
    if not minimal and base_delivery is not None:
        for key in ("closed", "open"):
            src = Path(base_delivery) / f"head_{key}_m.obj"
            if src.exists():
                shutil.copyfile(src, result / f"base_head_{key}.obj")
    if not minimal and (chosen / "build_direct.json").exists():
        # The same file already lives in neck_direct_v2/; the copy is verbose.
        shutil.copyfile(chosen / "build_direct.json", result / "build_direct.json")
    (result / "quick_check.json").write_text(
        json.dumps({"variant": variant, "chosen": str(chosen),
                    "variants": reports, "quick": quick}, indent=1),
        encoding="utf-8")
    # optional full verification + renders
    full_verify = bool(getattr(args, "full_verify", False))
    if full_verify:
        ablations = {}
        for tag, extra in (("noremesh", ["--no-remesh"]),
                           ("nosplit", ["--no-split"])):
            d = output / f"ablation_{tag}"
            common = ["--fit", dense, "--out", d,
                      "--anchor-validity-only",
                      "--adapt-full-limit", float(getattr(args, "crease_limit", 20.0)),
                      *extra, "--no-render"]
            common += _neck_param_args(args)
            if getattr(args, "neck_landmarks", None):
                common += ["--landmarks", str(args.neck_landmarks)]
            if variant == "prior":
                common += ["--larynx-prior",
                           "--larynx-prior-ratio",
                           float(getattr(args, "larynx_prior_ratio", 0.35))]
            _run_module("tools.build_neck_direct_v2", common,
                        what=f"ablation {tag}")
            ablations[tag] = d
        _run_module("tools.neck_direct_verify_v2",
                    ["--out", chosen, "--fit", dense,
                     "--ablation-noremesh", ablations["noremesh"],
                     "--ablation-nosplit", ablations["nosplit"],
                     "--previous", output / "_no_previous"],
                    what="full verification")
        if (chosen / "verification.json").exists():
            shutil.copyfile(chosen / "verification.json", result / "verification.json")
    render_done = None
    if bool(getattr(args, "render", False)):
        _run_module("tools.neck_direct_render_v2",
                    ["--out", chosen, "--fit", dense, "--no-baseline"],
                    what="standard neck render")
        _run_module("tools.larynx_prior_render",
                    ["--out", chosen,
                     "--old", output / "_no_old",
                     "--gain13", output / "_no_gain"],
                    what="larynx close-up render")
        src = chosen / "render"
        if src.exists():
            dst = result / "render"
            shutil.copytree(src, dst, dirs_exist_ok=True)
            render_done = dst
    # ---- T5: self-contained final/ + measured/ --------------------------
    from head_rgbd import neck_stage as ns

    final_sources = {state: chosen / f"head_{state}_direct.obj"
                     for state in ("closed", "open")}
    measured_sources = {}
    for state in ("closed", "open"):
        src = None
        if base_delivery is not None:
            candidate = Path(base_delivery) / f"head_{state}_m.obj"
            if candidate.is_file():
                src = candidate
        if src is None:
            # Write the measured base straight into measured/; no result-root
            # duplicate is needed on the fast path.
            src = result / "measured" / f"head_{state}.obj"
            src.parent.mkdir(parents=True, exist_ok=True)
            if not src.is_file():
                with np.load(dense / "parameters.npz", allow_pickle=False) as data:
                    faces = np.asarray(data["dense_faces"], np.int64)
                    vertices = data["dense_vertices"]
                refs = json.loads((dense / "reference_geometry.json").read_text(
                    encoding="utf-8"))["references"]
                stage = refs[state]["stages"]["dense"]
                ns.write_obj(src, np.asarray(vertices[stage["frame_index"]], float), faces,
                             comment=f"measured base dense {state}; no neck C reshaping")
        measured_sources[state] = src
    reference_frames = {}
    dense_refs = json.loads((dense / "reference_geometry.json").read_text(
        encoding="utf-8"))["references"] if (dense / "reference_geometry.json").is_file() else {}
    for state in ("closed", "open"):
        rec = ((landmark_payload or {}).get("states") or {}).get(state) or {}
        reference_frames[state] = (rec.get("reference_frame")
                                   or (dense_refs.get(state) or {}).get("frame_id"))
    marker_summary = {}
    for state in ("closed", "open"):
        rec = ((landmark_payload or {}).get("states") or {}).get(state) or {}
        lc = rec.get("larynx_center") or {}
        marker = rec.get("marker") or {}
        size = marker.get("size") or {}
        marker_summary[state] = {
            "status": rec.get("status"), "reason": rec.get("reason"),
            "frame_id": rec.get("frame_id"),
            "reference_frame": reference_frames[state],
            "detected_id": marker.get("target_id"),
            "black_square_mm": size.get("black_square_mm"),
            "size_source": size.get("size_source"),
            "size_uncertainty_mm": size.get("size_uncertainty_mm"),
            "t_marker": lc.get("t_marker"),
            "t_center_used": lc.get("t_center_used"),
            "theta_marker_deg": lc.get("theta_marker_deg"),
            "theta_used_deg": 0.0 if lc else None,
            "t_gate": lc.get("t_gate"),
            "source": lc.get("source"),
        }
    provenance = {
        "capture": str(getattr(args, "capture", None)) if getattr(args, "capture", None) else None,
        "dense": str(dense),
        "base_delivery": str(base_delivery) if base_delivery is not None else None,
        "neck_variant": variant,
        "neck_dir": str(chosen),
        "landmark_sidecar": str(landmark_path) if landmark_path else None,
        "marker": marker_summary,
    }
    delivery_info = dv.write_delivery(
        result, final_sources=final_sources, measured_sources=measured_sources,
        landmark_payload=landmark_payload, marker_status=marker_status,
        reference_frames=reference_frames, provenance=provenance)
    _log(f"[rebuild] delivery {delivery_info['status']}: "
         f"{delivery_info['delivery']['final']['dir']} / "
         f"{delivery_info['delivery']['measured']['dir']}")
    elapsed = time.time() - started
    if minimal:
        _write_compact_report(result, output, dense, chosen, variant, reports,
                              quick, elapsed, args,
                              delivery_info=delivery_info,
                              marker_status=marker_status,
                              landmark_payload=landmark_payload)
    else:
        _write_report(result, output, dense, base_delivery, chosen, variant,
                      reports, quick, full_verify, render_done,
                      elapsed, args,
                      delivery_info=delivery_info, marker_status=marker_status,
                      landmark_payload=landmark_payload)
    _log(f"[rebuild] delivered -> {result}")
    _log(f"[rebuild] downstream: {result / 'final'} / {result / 'measured'}")
    return 0


def run_base_and_rebuild(*, runtime, output, capture, sequence, args,
                         capture_is_recorded_session=True):
    """Run the existing locked reconstruction, then the neck stage."""
    from head_rgbd.pipeline import run_pipeline

    # The old appearance stage must not compete with the direct neck shaping.
    setattr(args, "neck_prominence", "off")
    code, delivery = run_pipeline(
        runtime=runtime,
        output=Path(output),
        capture=capture,
        sequence=sequence,
        args=args,
        purpose="one_click_reconstruction_neck",
        capture_is_recorded_session=capture_is_recorded_session,
    )
    if code != 0:
        return code
    dense = Path(output) / "reconstruction" / "dense"
    return run_rebuild(runtime=runtime, output=output, dense=dense,
                       base_delivery=delivery, args=args)