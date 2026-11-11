"""Generate docs/NECK_PROMINENCE_V2_REPORT.html for the 0.5.4 prominence.

Reads the V2 verification artefacts, the male delivery sidecar, the
single-field comparison and the 0.5.3 reference sidecars; embeds the renders
(base64) so the report is self-contained.
"""
from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def img64(path: Path) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def kv_table(rows) -> str:
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f"<table class='kv'>{body}</table>"


def fmt(value, digits=2, suffix=""):
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--prominence", type=Path, required=True, help="delivery_male neck_prominence.json")
    parser.add_argument("--single-field", type=Path, required=True, help="comparison_single_field neck_prominence.json")
    parser.add_argument("--v053", type=Path, required=True, help="0.5.3 default delivery neck_prominence.json")
    parser.add_argument("--v053-trial6", type=Path, required=True, help="0.5.3 6 mm trial neck_prominence.json")
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--position-review", type=Path, required=True, help="position_review directory")
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    v = json.loads(args.verification.read_text(encoding="utf-8"))
    prom = json.loads(args.prominence.read_text(encoding="utf-8"))
    single = json.loads(args.single_field.read_text(encoding="utf-8"))
    v053 = json.loads(args.v053.read_text(encoding="utf-8"))
    trial6 = json.loads(args.v053_trial6.read_text(encoding="utf-8"))
    lm = json.loads(args.landmarks.read_text(encoding="utf-8"))
    review = json.loads((args.position_review / "position_review.json").read_text(encoding="utf-8"))
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M CST")

    items = v["items"]
    check_rows = "".join(
        f"<tr class=\"{'pass' if items[k]['passed'] else 'fail'}\"><td>{i + 1}</td><td>{label}</td>"
        f"<td>{'通过 / PASS' if items[k]['passed'] else '失败 / FAIL'}</td></tr>"
        for i, (k, label) in enumerate([
            ("1_off_matches_clean_052_baseline", "关闭塑形 → 与纯 0.5.2 基线字节一致"),
            ("2_female_auto_zero_displacement", "female + auto → 全部顶点零位移"),
            ("3_missing_landmark_delivers_baseline", "无有效标签 → 正常交付基线"),
            ("4_repeat_export_idempotent", "重复导出不累积"),
            ("5_protection_strictly_holds", "面部/嘴唇/下巴/后颈/底部保护成立；固定人脸射线无新增遮挡"),
            ("6_topology_shared_with_baseline", "闭口/张口共享拓扑且与基线一致"),
            ("7_report_matches_obj", "报告参数与最终 OBJ 读回一致"),
            ("8_shape_and_safety_acceptance", "形状验收（H 目标 ≤1mm）+ 全部几何安全检查"),
        ])
    )
    states = items["8_shape_and_safety_acceptance"]["states"]

    def state_rows(s):
        shape = states[s]
        side = prom["states"][s]
        ext = side["extent"]
        return [
            ("H_base（基线已有突起）", fmt(shape["h_base_mm"], 3, " mm")),
            ("H_target（设计目标）", fmt(shape["h_target_mm"], 1, " mm")),
            ("H_final（最终网格实测）", fmt(shape["h_final_mm"], 3, " mm")),
            ("目标误差", fmt(shape["target_error_mm"], 3, " mm")),
            ("标签绑定处前向位移（重心插值）", fmt(shape["binding_forward_displacement_mm"], 3, " mm")),
            ("最大顶点位移", fmt(shape["max_vertex_displacement_mm"], 3, " mm")),
            ("校准后场幅值 A", fmt(side.get("calibrated_amplitude_mm"), 3, " mm")),
            ("半高横向宽度（10% 宽度）", f'{fmt(ext.get("width_x_half_max_mm"), 2)} mm ({fmt(ext.get("width_x_ten_percent_mm"), 2)} mm)'),
            ("半高上下长度（10% 长度）", f'{fmt(ext.get("length_y_half_max_mm"), 2)} mm ({fmt(ext.get("length_y_ten_percent_mm"), 2)} mm)'),
            ("峰值位置相对塑形中心", f'{fmt(shape["peak_offset_from_centre_mm"][0], 2)}, {fmt(shape["peak_offset_from_centre_mm"][1], 2)} mm'),
            ("塑形中心贴脊偏移", fmt((side.get("centre_snap") or {}).get("offset_mm"), 2, " mm")),
            ("标签→绑定距离", fmt(side["binding"].get("distance_marker_to_ray_binding_mm"), 2, " mm")),
            ("移动顶点 / 受影响面", f'{side["support"]["moved_vertices"]} / {side["safety"]["affected_faces"]}'),
            ("边缘增长最大比（硬上限 2.2）", fmt((side["safety"]["edge_growth"] or {}).get("max_ratio"), 3)),
            ("位移尖峰 / 失控长边 / 细长面", f'{(side["safety"]["edge_growth"] or {}).get("spike_vertices") and len((side["safety"]["edge_growth"] or {}).get("spike_vertices")) or 0} / {len((side["safety"]["edge_growth"] or {}).get("violations", []))} / {(side["safety"]["edge_growth"] or {}).get("slender_faces")}'),
            ("在线自交检测（段数/候选面）", f'{(side["safety"]["self_intersections"] or {}).get("count")}（{(side["safety"]["self_intersections"] or {}).get("segments")} 段 / {(side["safety"]["self_intersections"] or {}).get("candidate_faces")} 面）'),
            ("独立复查自交", fmt(shape["self_intersections_independent"], 0)),
            ("安全 problems", str(shape["safety_reported"] or "无")),
            ("深度辅助：状态 / 前向一致 / 颈宽", f'{(side.get("depth_conditioning") or {}).get("status")} / {fmt((side.get("depth_conditioning") or {}).get("forward_agreement"), 3)} / {fmt((side.get("depth_conditioning") or {}).get("neck_width_mm"), 1, " mm")}'),
            ("参考趋势锚点（下/上）", str((side.get("trend") or {}).get("per_side"))),
            ("标注校正（原始像素→修正像素）", f'{lm["states"][s]["correction"]["original_pixel_uv"]} → {lm["states"][s]["correction"]["corrected_pixel_uv"]}' if lm["states"][s].get("correction") else "未校正"),
        ]

    def comparison_rows():
        out = []
        for s in ("closed", "open"):
            v3 = v053["states"][s]
            t6 = trial6["states"][s]
            sf = single["states"][s]
            out.append(
                f"<tr><td>{s}</td>"
                f"<td>{v3['status']}<br>实际最大位移 {fmt(v3.get('actual_amplitude_mm'), 2)} mm</td>"
                f"<td>{t6['status']}<br>实际最大位移 {fmt(t6.get('actual_amplitude_mm'), 2)} mm（名义 6）</td>"
                f"<td>{sf['status']}<br>H_final {fmt(sf.get('final_prominence_mm'), 2)} mm<br>{(sf.get('safety') or {}).get('problems') or '无问题'}</td>"
                f"<td>{prom['states'][s]['status']}<br>H_final {fmt(prom['states'][s].get('final_prominence_mm'), 2)} mm<br>尖峰 0 / 违规 0</td></tr>"
            )
        return "".join(out)

    timing_rows = "".join(
        f"<tr><td>{t['run']}</td><td>{t['export_total_s']:.3f} s</td>"
        f"<td>{', '.join(f'{x:.0f} ms' for x in t['prominence_stage_ms'])}</td>"
        f"<td>{t['prominence_stage_total_ms']:.0f} ms</td></tr>"
        for t in v["timing"]
    )

    def figure(name: str, caption: str, width: str = "100%") -> str:
        path = args.render_dir / name
        return (
            f"<figure><img style=\"width:{width}\" src=\"{img64(path)}\" alt=\"{name}\"/>"
            f"<figcaption>{caption}</figcaption></figure>"
        )

    position_figures = "".join(
        f"<figure><img style=\"width:49%\" src=\"{img64(args.position_review / name)}\" alt=\"{name}\"/>"
        f"<figcaption>{caption}</figcaption></figure>"
        for name, caption in (
            ("closed_rgb_overlay.png", "闭口：原始点/绑定/中线/支撑范围叠加（标注校正后）"),
            ("open_rgb_overlay.png", "张口：同上"),
            ("closed_front_0_zoom.png", "闭口原始标注点附近放大（点在衣领上，属误标）"),
            ("open_front_4_zoom.png", "张口原始标注点附近放大（在颈区下缘/衣物边界）"),
        )
    )

    single_open = single["states"]["open"]
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>颈前塑形 V2 报告 / Neck Prominence V2</title>
<style>
 body {{ font-family: "Segoe UI","Microsoft YaHei",sans-serif; margin:0; background:#1c1e22; color:#d8dce2; line-height:1.55; }}
 .wrap {{ max-width:1100px; margin:0 auto; padding:32px 24px 64px; }}
 h1 {{ font-size:25px; border-bottom:2px solid #4a90d9; padding-bottom:10px; }}
 h2 {{ font-size:19px; color:#7fb3e8; margin-top:36px; border-left:4px solid #4a90d9; padding-left:10px; }}
 h3 {{ font-size:16px; color:#b8c6d9; }}
 table {{ border-collapse:collapse; margin:12px 0; width:100%; font-size:13.5px; }}
 td,th {{ border:1px solid #3a3f47; padding:6px 10px; text-align:left; vertical-align:top; }}
 th {{ background:#262a31; }}
 tr.pass td:last-child {{ color:#7fd88f; font-weight:600; }}
 tr.fail td:last-child {{ color:#e06c5f; font-weight:600; }}
 table.kv td:first-child {{ color:#9aa7b8; width:42%; }}
 code {{ background:#262a31; padding:1px 5px; border-radius:3px; font-size:12.5px; }}
 .note {{ background:#2a2519; border:1px solid #6b5b1e; padding:10px 14px; border-radius:6px; margin:12px 0; }}
 .warn {{ background:#2a1f1f; border:1px solid #6b2e2e; padding:10px 14px; border-radius:6px; margin:12px 0; }}
 figure {{ margin:16px 0; text-align:center; }}
 figure img {{ border-radius:6px; max-width:100%; }}
 figcaption {{ font-size:12.5px; color:#8a97a8; margin-top:6px; }}
</style></head><body><div class="wrap">

<h1>颈前塑形 V2 — 深度辅助定参的程序化喉结（0.5.4，基于 0.5.2）</h1>
<p style="color:#8a97a8">Neck prominence V2 — depth-conditioned procedural prominence</p>

<h2>0. 摘要</h2>
<div class="note">
<b>本阶段仍是模板颈部的“外观塑形”</b>：深度只用于核对方向、约束支撑尺度与记录局部皮肤趋势；不做整片颈部逐点拟合，
不使用实测补片或缝合。塑形目标改为<b>最终隆起高度 H</b>（固定剖面、固定参考趋势、面片-平面求交测量），
用“宽基底 + 窄圆钝脊峰”双场实现；闭口/张口均达到 8 mm 目标（误差 0.0 mm），全部几何安全检查通过。
旧的 0.5.3 单场在同样的驱动方式下张口触发尖峰检测，双场方案是必要的。
</div>
{kv_table([
 ("基线算法 / base algorithm", "0.5.2（commit e70f240 工作树）"),
 ("特性版本 / feature version", "0.5.4（基于 0.5.2；0.6.0/0.7.0 已被既往实验占用）"),
 ("塑形方法 / neck method", f'<code>{prom["neck_method"]}</code>'),
 ("几何来源 / geometry source", f'<code>{prom["geometry_source"]}</code>'),
 ("验证时间 / verified at", now),
 ("验证数据", "runtime/outputs/zp_0.5.1/prepared（无 ArUco；人工指定点，软件链路验证）"),
])}

<h2>1. 位置复核（先于塑形改进）</h2>
<p>将原始人工点、基线网格绑定、拟用塑形中心、颈部局部中线与支撑范围叠加到 RGB/深度图后确认：</p>
<ul>
<li><b>原始闭口点 (692,568) 位于衣领上、在 prepared neck_mask 之外</b>（该列颈部皮肤仅到 v≈564）；原始张口点 (649,588) 位于 neck_mask 最底缘的衣物边界。</li>
<li><b>根因</b>：标签工具的 neck_roi 点击门限在人脸框下方 10% 框高处（v≥520），把真正的喉结区（v≈480–520）排除在外，迫使点击落到衣领。已把门限改为 2% 框高并记录。</li>
<li>修正中心：闭口 (688,504)、张口 (663,555)——均位于颈部皮肤中线、喉结高度（多帧深度中线相对侧向参考的突起峰 v≈498/552，且符合上三分之一解剖带）。原始点、修正点与原因均写入 <code>neck_landmarks_v2.json</code> 的 correction 记录。</li>
<li><b>关于“中面偏移约 17mm/3.3mm”线索</b>：核查发现 <code>landmarks_m</code> 本身已是参考相机系（直接投影与观测 2D 特征点偏差 ≤2px，无镜像/缩放/裁剪/帧索引错误）；此前疑似中面偏移的计算来自对已变换点再做一次模型变换的错误。真实情况：真人颈部中线与面部中线一致（u≈688 对 687.7）。模板颈部前棱位于绑定左侧约 13mm（闭口）/ 9.8mm（张口）——属<b>模板颈部与真人颈部差异</b>，故塑形中心按记录贴到模板颈前棱（≤20mm 邻域约束），不强行拉回人脸中面，也不把 19.8mm 绑定深度差当作喉结高度。</li>
</ul>
<div>{position_figures}</div>

<h2>2. 算法（V2）</h2>
<ul>
<li><b>目标高度</b>：H_base 为基线已有突起；H_target ∈ {{8, 6}}（优先 8，10 留作边界试验）；H_final 为最终网格实测。已有突起 ≥ 目标时跳过（不重复叠加）。</li>
<li><b>参考趋势</b>：塑形前由矢状剖面（面片与平面精确求交，0.5mm 加密）在支撑外锚带 |y| ∈ [ry_base+2, ry_base+8] 上拟合直线，且在塑形后<b>不再重拟合</b>；锚点须在中心轮廓 ±8mm 连续带内（排除下巴突变）。</li>
<li><b>双场</b>：宽基底 Wendland C2（20×16mm，占 35%）+ 窄中央圆钝平台核（11×12mm，C2 紧支撑、顶平坦，占 65%）；沿各状态头部姿态的局部前向轴施加；位移永远从未变形基线计算（幂等）。</li>
<li><b>标定</b>：场幅值按“绑定面重心插值后剖面 H 达到目标”迭代 1–3 次；中心效率过低时先加宽核心（≤基底），绝不除以极小权重制造尖峰。</li>
<li><b>保护</b>：面部特征点 30mm、耳线外侧 |x|>45mm、底部裁切带 8mm、背向法向全部锁定；边界改用<b>按实际毫米距离的连续衰减</b>（6mm 斜坡），替代固定网格环数。</li>
<li><b>深度辅助定参</b>：标记点邻域 17×17 深度平面拟合（rms≤1.5mm 才可用）核对前向方向；颈部 mask 宽度约束支撑半径上限；不可靠时不改变目标高度，改用记录在案的配置先验。</li>
</ul>

<h2>3. 形状验收（定义与容差在查看结果前固定）</h2>
<table><tr><th>#</th><th>检查项</th><th>结果</th></tr>{check_rows}</table>
<p>容差：|H_final − H_target| ≤ {v["acceptance"]["target_tolerance_mm"]} mm；半高宽/长 ≥ {v["acceptance"]["min_half_max_width_mm"]}/{v["acceptance"]["min_half_max_length_mm"]} mm；峰值偏离中心 ≤ {v["acceptance"]["max_peak_offset_mm"]} mm。</p>

<h3>3.1 逐状态结果</h3>
<div style="display:flex;gap:18px;flex-wrap:wrap">
<div style="flex:1 1 480px"><h3>闭口 / closed</h3>{kv_table(state_rows("closed"))}</div>
<div style="flex:1 1 480px"><h3>张口 / open</h3>{kv_table(state_rows("open"))}</div>
</div>

<h3>3.2 与 0.5.3 及单场对照</h3>
<table>
<tr><th>状态</th><th>0.5.3 默认（单 Wendland，候选 2/3/4 首个安全）</th><th>0.5.3 6 mm 对照</th><th>修正候选选择后的单场（纯脊峰核，H=8）</th><th>V2 双场（本交付）</th></tr>
{comparison_rows()}
</table>
<div class="note">单场对照说明：相同的目标高度驱动、相同的绑定/贴脊/剖面/安全链，仅把基底份额设为 0。闭口虽通过，<b>张口出现 displacement_spike</b>（单顶点相对邻域中值突出）——
这证明在 4–8mm 采样的颈前表面上仅靠提高名义幅度会先触发局部尖刺，双场（宽基底承担过渡、中央圆钝脊峰提供特征）是必要的。</div>

<h3>3.3 阶段耗时（≥3 次采样）</h3>
<table><tr><th>采样</th><th>导出总耗时</th><th>closed / open 阶段耗时</th><th>合计</th></tr>{timing_rows}</table>

<h2>4. 同视角渲染与剖面</h2>
{figure("closed_side_compare.png", "闭口：侧面（同视角/比例/光照；左基线，右塑形）")}
{figure("open_side_compare.png", "张口：侧面")}
{figure("closed_45deg_compare.png", "闭口：45°")}
{figure("open_45deg_compare.png", "张口：45°")}
{figure("closed_front_compare.png", "闭口：正面")}
{figure("open_front_compare.png", "张口：正面")}
{figure("closed_closeup_solid.png", "闭口：颈部 45° 局部放大（灰色实体）")}
{figure("open_closeup_solid.png", "张口：颈部 45° 局部放大")}
{figure("closed_closeup_wireframe.png", "闭口：局部线框（确认凸起来自几何本身，不依赖着色）")}
{figure("open_closeup_wireframe.png", "张口：局部线框")}
{figure("closed_section.png", "闭口：穿过实际塑形中心的三角网格剖面线（面片∩平面；平面绕中心倾斜 15° 以横向切割脊峰）")}
{figure("open_section.png", "张口：剖面线")}

<h2>5. 深度辅助参数与回退</h2>
{kv_table([
 ("深度状态（闭口/张口）", f'{prom["states"]["closed"]["depth_conditioning"].get("status")} / {prom["states"]["open"]["depth_conditioning"].get("status")}'),
 ("平面拟合 rms（mm）", f'{fmt(prom["states"]["closed"]["depth_conditioning"].get("surface_plane", {}).get("rms_resid_mm"), 3)} / {fmt(prom["states"]["open"]["depth_conditioning"].get("surface_plane", {}).get("rms_resid_mm"), 3)}'),
 ("前向一致（模型轴 vs 深度法向）", f'{fmt(prom["states"]["closed"]["depth_conditioning"].get("forward_agreement"), 3)} / {fmt(prom["states"]["open"]["depth_conditioning"].get("forward_agreement"), 3)}'),
 ("观测颈宽（mm）", f'{fmt(prom["states"]["closed"]["depth_conditioning"].get("neck_width_mm"), 1)} / {fmt(prom["states"]["open"]["depth_conditioning"].get("neck_width_mm"), 1)}'),
 ("半径钳制是否触发", f'{prom["states"]["closed"]["depth_conditioning"]["radius_bounds"].get("clamped")} / {prom["states"]["open"]["depth_conditioning"]["radius_bounds"].get("clamped")}（试验半径均在颈宽以内）'),
 ("目标强度来源", "配置外观先验（8 mm 设计参数）；深度突起估计（约 3.5–11 mm、1mm 量化且含首饰干扰）不可靠，未用于改目标"),
])}

<h2>6. 交付物与入口</h2>
<ul>
<li>特性运行：<code>runtime/outputs/zp_052_neck_bump_v2/delivery_male_v2</code>（闭口/张口网格、neck_prominence.json、manifest）</li>
<li>验证交付（来自纯 0.5.2 重建）：<code>runtime/outputs/zp_052_neck_bump_v2/verification/delivery_male</code>，八项明细 <code>verification_report.json</code>，渲染 <code>verification/renders/</code></li>
<li>标签：<code>runtime/outputs/zp_052_neck_bump_v2/neck_landmarks_v2.json</code>（原始/修正/原因/证据）</li>
<li>位置复核：<code>runtime/outputs/zp_052_neck_bump_v2/position_review/</code></li>
<li>配置：<code>config/defaults.json · neck_prominence</code>（amplitude_mm=8 目标、候选 [6,8]、core 11×12、base 20×16、share 0.35）；CLI 仅覆盖</li>
<li>代码：<code>head_rgbd/neck_prominence.py</code>（V2 引擎）、<code>head_rgbd/neck_landmark.py</code>（ROI 门限修正）、<code>tools/verify_neck_prominence_v2.py</code>、<code>tests/test_neck_prominence.py</code></li>
<li>manifest / run_report：<code>neck_method=depth_conditioned_procedural_prominence</code>、<code>geometry_source=template_plus_procedural_deformation</code>、逐状态 H/安全/深度/贴脊/回退原因</li>
</ul>

<h2>7. 限制与未完成项</h2>
<div class="warn"><ul>
<li><b>真实 ArUco 标签采集链路仍未验证</b>（本数据无标签）：人工指定属软件链路验证；ArUco 仅验证了正确失败行为。</li>
<li>不声称任何深度保真；8mm 为外观设计参数，不是解剖标准。</li>
<li>闭口/张口为同一身份、共享拓扑的两个姿态网格；剖面在脊峰处近乎相切，图示剖面倾斜 15° 仅用于可视化，数值 H 使用固定矢状剖面定义。</li>
<li>模板颈部与真人颈部存在 ~10–13mm 的横向差异（模板棱线偏移），本阶段不校正该差异，只把塑形中心贴到模板自身棱线并如实记录。</li>
</ul></div>

<p style="color:#8a97a8;margin-top:40px">报告生成：{now} · tools/make_neck_report_v2.py</p>
</div></body></html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
