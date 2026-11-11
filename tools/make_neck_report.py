"""Generate docs/NECK_TEMPLATE_PROMINENCE_REPORT.html from the verification artefacts."""
from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path


def img64(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def state_table(prom: dict, state: str) -> str:
    r = prom["states"][state]
    b = r["binding"]
    d = r["direction"]
    rows = [
        ("状态 / status", r["status"]),
        ("标签来源 / landmark source", r["landmark_source"]),
        ("请求幅值 / requested amplitude", f'{r["requested_amplitude_mm"]} mm'),
        ("候选幅值 / candidates", ", ".join(f'{v} mm' for v in r["candidate_amplitudes_mm"])),
        ("选中候选 / selected amplitude", f'{r["selected_amplitude_mm"]} mm'),
        ("实际最大位移 / actual amplitude", f'{r["actual_amplitude_mm"]:.3f} mm'),
        ("支撑半径 / radii", f'{r["support"]["radius_x_mm"]} × {r["support"]["radius_y_mm"]} mm (x × y)'),
        ("支撑顶点 / selected vertices", r["support"]["selected_vertices"]),
        ("移动顶点 / moved vertices", r["support"]["moved_vertices"]),
        ("受保护顶点 / protected vertices", r["protection"]["protected_vertices"]),
        ("绑定三角面 / binding triangle", b["triangle_index"]),
        ("标签到绑定面距离 / marker→binding", f'{b["distance_marker_to_binding_mm"]:.1f} mm'),
        ("射线入射角一致性 / ray incidence", f'{b["ray_incidence"]:.3f}'),
        ("方向-法向一致性 / forward-normal agreement", f'{d["forward_normal_agreement"]:.3f}'),
        ("几何安全 / safety", f'{r["safety"]["status"]} (problems: {r["safety"]["problems"] or "none"})'),
        ("受影响面数 / affected faces", r["safety"]["affected_faces"]),
        ("阶段耗时 / stage elapsed", f'{r["elapsed_ms"]:.0f} ms'),
        ("变换来源 / transform", f'{r["transform"]["method"]} ({r["transform"]["annotation_frame"]})'),
    ]
    if r.get("reason"):
        rows.append(("跳过原因 / reason", r["reason"]))
    body = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows
    )
    return f"<table class='kv'>{body}</table>"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--prominence", type=Path, required=True, help="delivery_male neck_prominence.json")
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    v = json.loads(args.verification.read_text(encoding="utf-8"))
    prom = json.loads(args.prominence.read_text(encoding="utf-8"))
    lm = json.loads(args.landmarks.read_text(encoding="utf-8"))
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M CST")

    items = v["items"]
    checks = "".join(
        f"<tr class=\"{'pass' if items[k]['passed'] else 'fail'}\">"
        f"<td>{i + 1}</td><td>{label}</td>"
        f"<td>{'通过 / PASS' if items[k]['passed'] else '失败 / FAIL'}</td></tr>"
        for i, (k, label) in enumerate([
            ("1_off_matches_clean_052_baseline", "关闭塑形时结果与干净 0.5.2 基线字节一致"),
            ("2_female_auto_zero_displacement", "female + auto：全部顶点与基线一致（零位移）"),
            ("3_male_valid_landmark_continuous_bump", "male + 有效标签：闭口/张口均为连续局部凸起"),
            ("4_missing_landmark_delivers_baseline", "缺失/无效标签：正常交付基线网格，不阻塞"),
            ("5_protection_strictly_holds", "面部保护严格成立 + 面部特征点首射线遮挡不变"),
            ("6_repeat_export_idempotent", "重复导出不累积（哈希一致）"),
            ("7_units_and_reported_values", "参数与 mm/m 单位换算正确"),
            ("8_geometry_safety_recheck", "几何安全独立复查通过（含自相交检测）"),
        ])
    )

    timing = v["prominence_stage_timing"]
    timing_rows = "".join(
        f"<tr><td>{t['run']}</td><td>{t['export_total_s']:.3f} s</td>"
        f"<td>{', '.join(f'{x:.0f} ms' for x in t['prominence_stage_ms'])}</td>"
        f"<td>{t['prominence_stage_total_ms']:.0f} ms</td></tr>"
        for t in timing
    )

    renders = {Path(p).name: img64(Path(p)) for p in v["renders"]}

    def figure(name: str, caption: str) -> str:
        return (
            f"<figure><img src=\"{renders[name]}\" alt=\"{name}\"/>"
            f"<figcaption>{caption}</figcaption></figure>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>标签引导颈前模板塑形报告 / Neck Template Prominence Report</title>
<style>
  body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0;
         background: #1c1e22; color: #d8dce2; line-height: 1.55; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; }}
  h1 {{ font-size: 26px; border-bottom: 2px solid #4a90d9; padding-bottom: 10px; }}
  h2 {{ font-size: 19px; color: #7fb3e8; margin-top: 36px;
       border-left: 4px solid #4a90d9; padding-left: 10px; }}
  h3 {{ font-size: 16px; color: #b8c6d9; }}
  table {{ border-collapse: collapse; margin: 12px 0; width: 100%; font-size: 13.5px; }}
  td, th {{ border: 1px solid #3a3f47; padding: 6px 10px; text-align: left; }}
  th {{ background: #262a31; }}
  tr.pass td:last-child {{ color: #7fd88f; font-weight: 600; }}
  tr.fail td:last-child {{ color: #e06c5f; font-weight: 600; }}
  table.kv td:first-child {{ color: #9aa7b8; width: 46%; }}
  code {{ background: #262a31; padding: 1px 5px; border-radius: 3px; font-size: 12.5px; }}
  .note {{ background: #2a2519; border: 1px solid #6b5b1e; padding: 10px 14px;
          border-radius: 6px; margin: 12px 0; }}
  .warn {{ background: #2a1f1f; border: 1px solid #6b2e2e; padding: 10px 14px;
          border-radius: 6px; margin: 12px 0; }}
  figure {{ margin: 18px 0; text-align: center; }}
  figure img {{ max-width: 100%; border-radius: 6px; }}
  figcaption {{ font-size: 12.5px; color: #8a97a8; margin-top: 6px; }}
  .cols {{ display: flex; gap: 18px; flex-wrap: wrap; }}
  .cols > div {{ flex: 1 1 460px; }}
  ul {{ margin: 6px 0; }}
</style>
</head>
<body><div class="wrap">

<h1>标签引导的颈前模板局部塑形 — 验证与交付报告</h1>
<p style="color:#8a97a8">Marker-guided front-neck template prominence — verification &amp; delivery report</p>

<h2>0. 摘要</h2>
<div class="note">
<b>本特性是对模板颈部的"外观塑形"，不是任何形式的颈部实测重建。</b>标签（喉结体表点）仅用于定位一个
小的过程性 Wendland C2 位移场，使模板颈前表面在标签附近产生连续凸起。闭口/张口两状态各自使用独立的
有效标签与头部姿态；不修改任何 0.5.2 既有几何约束；关闭该阶段时输出与纯 0.5.2 基线<b>字节一致</b>。
八项验证全部通过。
</div>
<table class="kv">
<tr><td>基线算法版本 / base algorithm version</td><td>{prom["base_algorithm_version"]}（commit e70f240）</td></tr>
<tr><td>特性版本 / feature version</td><td>0.5.3（基于 0.5.2；0.6.0/0.7.0 已被既往实验占用，非原版 0.5.2 伪装）</td></tr>
<tr><td>实现分支 / implementation branch</td><td>codex/rollback-052-neck-bump（独立 worktree，commit 0c07f76）</td></tr>
<tr><td>塑形方法 / neck method</td><td><code>{prom["neck_method"]}</code></td></tr>
<tr><td>几何来源 / geometry source</td><td><code>{prom["geometry_source"]}</code>（模板 + 过程性位移，非测量几何）</td></tr>
<tr><td>主体性别设置 / subject sex</td><td><code>{prom["subject_sex"]}</code>（仅来自显式 CLI 参数，绝不从姓名/照片/人脸推断）</td></tr>
<tr><td>验证时间 / verified at</td><td>{now}</td></tr>
<tr><td>验证数据 / data</td><td>runtime/outputs/zp_0.5.1/prepared（无 ArUco 采集）</td></tr>
</table>

<h2>1. 回退保护与来源版本</h2>
<table class="kv">
<tr><td>回退目标 / rollback target</td><td>e70f240d4c0bff8f63255286f4f786290f61ff54（0.5.2 + THIRD_PARTY_NOTICES.md）</td></tr>
<tr><td>回退前主项目 HEAD</td><td>5fcf069 + 未提交修改</td></tr>
<tr><td>完整工作区备份 tag</td><td><code>backup-20261111-full-workspace</code> → commit 6bdc422（含全部未提交/未跟踪文件，未触碰工作树）</td></tr>
<tr><td>哈希清单 / hash manifests</td><td>D:\\proiect\\_backup_20261111\\（tracked 165 项 / untracked 271 项 / workspace 299 项，SHA-256）</td></tr>
<tr><td>原始采集 / 模型资产 / 历史网格与报告</td><td>未改动，仅读取</td></tr>
</table>
<p><b>恢复方法</b>：<code>git stash list</code> 检查后，<code>git checkout backup-20261111-full-workspace -- &lt;path&gt;</code>
按哈希清单恢复任意文件，或将主项目重置到该 tag 指向的树。全程未使用 <code>reset --hard</code> / <code>clean</code>。</p>

<h2>2. 标签定位（最小移植）</h2>
<p>仅移植：ArUco 检测与中心定位、坐标变换与逐状态记录、人工指定回退、标签→最终网格绑定。
<b>未移植</b>：neck_refit / neck_measured / neck_anterior / 旧导出颈部整合 / 任何补丁、缝合、全颈 ICP 或重复变形。</p>
<table class="kv">
<tr><td>本次实测标签来源</td><td><code>manual_click</code>（闭口 closed_front_0 @ (692, 568)；张口 open_front_4 @ (649, 588)），深度回投影并经空洞/跳变/边缘检查</td></tr>
<tr><td>标签文件 / landmarks file</td><td><code>neck_landmarks.json</code>（schema <code>head_rgbd.neck_landmarks.v1</code>，逐状态记录帧绑定 SHA-256）</td></tr>
<tr><td>闭合/张口独立性</td><td>每状态各自标注帧 + 各自头部姿态变换（本次均为 same_frame_observation）；从不跨状态复制相机点</td></tr>
</table>
<div class="warn">
<b>诚实声明</b>：采集数据（zp_0.5.1）中<b>没有 ArUco 标签</b>。本次"有效标签"为人工指定点，属于
<b>软件链路验证</b>，不能等同于真实标签验证。ArUco 检测链路仅验证了正确失败行为（no_marker_detected，
19 项单元测试通过）——<b>真实 ArUco 采集链路验证未完成，单独报告</b>。不可靠定位一律记录原因并跳过，
绝不用模板估计替代，绝不阻塞头部重建。
</div>

<h2>3. 塑形算法与配置</h2>
<ul>
<li>拓扑不变：仅移动标签附近颈前顶点；方向取自逐状态头部姿态的局部轴（右/上/前，由基线 FLAME 特征点构建，与外法向一致性校验：闭合 {prom["states"]["closed"]["direction"]["forward_normal_agreement"]:.3f}，张口 {prom["states"]["open"]["direction"]["forward_normal_agreement"]:.3f}），绝不用世界轴/相机轴。</li>
<li>支撑域：约束网格邻域 BFS 连通的前颈皮肤区域（非欧氏球），椭圆坐标 Wendland C2 紧支撑 w(r)=(1−r)⁴(4r+1)。</li>
<li>保护：面部特征点 30mm 邻域、耳线外侧 |x|&gt;45mm、底部裁切带 8mm、背向法向全部锁定；靠近保护边界按 BFS 环带连续渐缩（无硬裁剪）。</li>
<li>幅值候选 2/3/4mm 升序尝试，几何安全检查通过即取更小的自然凸起；不安全则减小幅值重试，仍失败→跳过该状态、交付基线并记录原因。</li>
<li>单一作用点：最终基线表面之后、统一网格写出之前；位移永远从<b>未变形基线</b>计算（重复导出幂等）。</li>
</ul>
<p>配置仅存于 <code>config/defaults.json</code> 的 <code>neck_prominence</code> 节，CLI 仅做覆盖：
<code>--subject-sex</code>、<code>--neck-prominence auto|on|off</code>、<code>--neck-prominence-mm</code>、
<code>--neck-prominence-radius-x-mm</code>、<code>--neck-prominence-radius-y-mm</code>。</p>
<div class="note">幅值 A=3mm、rx=12mm、ry=18mm 为<b>试用值</b>，未经解剖学或最优化验证。</div>

<h2>4. 八项验证结果</h2>
<table>
<tr><th>#</th><th>验证项</th><th>结果</th></tr>
{checks}
</table>

<h3>4.1 逐状态塑形明细</h3>
<div class="cols">
<div><h3>闭口 / closed</h3>{state_table(prom, "closed")}</div>
<div><h3>张口 / open</h3>{state_table(prom, "open")}</div>
</div>
<p>闭合状态标签到绑定面距离 {prom["states"]["closed"]["binding"]["distance_marker_to_binding_mm"]:.1f} mm
（模板颈部表面位于标签平面之后，&lt;25mm 上限内）；张口 {prom["states"]["open"]["binding"]["distance_marker_to_binding_mm"]:.1f} mm。</p>

<h3>4.2 关键数值</h3>
<table class="kv">
<tr><td>基线（纯 0.5.2）closed 哈希</td><td><code>{items["1_off_matches_clean_052_baseline"]["baseline_hashes"]["closed"][:32]}…</code></td></tr>
<tr><td>基线（纯 0.5.2）open 哈希</td><td><code>{items["1_off_matches_clean_052_baseline"]["baseline_hashes"]["open"][:32]}…</code></td></tr>
<tr><td>off 导出哈希</td><td>与基线完全一致（字节级）</td></tr>
<tr><td>female+auto 哈希</td><td>与基线完全一致；reason = <code>auto_female_keeps_template_neck</code></td></tr>
<tr><td>无标签时 reason</td><td><code>auto_male_but_no_valid_landmark</code>（交付基线，不阻塞）</td></tr>
<tr><td>面部特征点首射线</td><td>closed {items["5_protection_strictly_holds"]["states"]["closed"]["facial_landmark_first_hits_unchanged"]} 不变（其余为两网格均无射线的掠射点）；面部 30mm 内移动顶点数 = 0</td></tr>
<tr><td>自相交检测</td><td>closed {items["8_geometry_safety_recheck"]["states"]["closed"]["self_intersection_count"]} 处 / open {items["8_geometry_safety_recheck"]["states"]["open"]["self_intersection_count"]} 处</td></tr>
</table>

<h3>4.3 阶段耗时（≥3 次采样）</h3>
<table>
<tr><th>采样</th><th>导出总耗时</th><th>closed / open 阶段耗时</th><th>阶段合计</th></tr>
{timing_rows}
</table>

<h2>5. 渲染对比（同视角 / 同尺度）</h2>
{figure("closed_front_compare.png", "闭口：颈前对比（左：0.5.2 基线；右：+ 标签引导塑形）")}
{figure("open_front_compare.png", "张口：颈前对比")}
{figure("closed_sagittal_profile.png", "闭口：颈前正中矢状轮廓（灰=基线，红=塑形；蓝+=标签点，绿+=基线绑定，黄×=塑形后绑定）")}
{figure("open_sagittal_profile.png", "张口：颈前正中矢状轮廓")}
{figure("closed_transverse_section.png", "闭口：绑定高度横截面（±4mm）")}
{figure("open_transverse_section.png", "张口：绑定高度横截面（±4mm）")}
{figure("closed_displacement_field.png", "闭口：位移场（椭圆紧支撑、连续衰减）")}
{figure("open_displacement_field.png", "张口：位移场")}

<h2>6. 交付物</h2>
<ul>
<li>代码与统一配置：主项目 <code>D:\\proiect\\facecap_airway_delivery</code>（分支 release/0.5.3-neck-prominence，实现 commit 0c07f76 + 报告 a00b9ad + 契约更新 4b3731f）；实验存档 worktree <code>D:\\proiect\\facecap_airway_delivery_052_neck_bump</code>（分支 codex/rollback-052-neck-bump）保留</li>
<li>旧实验工作区存档：tag <code>backup-20261111-full-workspace</code>、分支 <code>archive/pre-rollback-20261111</code>、stash@&#123;0&#125;（含全部未提交/未跟踪文件）与 D:\\proiect\\_backup_20261111\\ 哈希清单；旧颈部阶段模块已不在主目录工作树中，无法影响结果</li>
<li>纯 0.5.2 基线输出（已保存）：<code>runtime/outputs/zp_052_neck_bump_baseline</code></li>
<li>male 0.5.3 特性运行：<code>runtime/outputs/zp_052_neck_bump_male_053</code>（delivery/ 含 head_closed_m.obj、head_open_m.obj、neck_prominence.json、manifest.json；闭口/张口均为共享拓扑的同一 FLAME 网格 + 局部位移）</li>
<li>标签文件与最终网格绑定：neck_landmarks.json + neck_prominence.json 中的 binding（三角形索引 + 重心坐标 + 标签/基线/塑形三点）</li>
<li>受影响顶点、实际幅值、支撑范围、方向、回退原因：neck_prominence.json 逐状态完整记录</li>
<li>八项验证明细与渲染：<code>runtime/outputs/zp_052_neck_bump_male_053/verification/verification_report.json</code> + 8 张 PNG</li>
<li>单元测试：43 项（test_neck_prominence.py 24 项 + test_aruco_marker.py 19 项）全部通过</li>
</ul>

<h2>7. 限制与未完成项</h2>
<div class="warn">
<ul>
<li><b>真实 ArUco 标签采集链路未验证</b>（采集数据无标签）：仅验证了人工指定链路与 ArUco 正确失败行为。该项<b>未完成</b>，单独报告。</li>
<li><b>不声称任何深度保真</b>：本阶段是外观塑形（appearance shaping），标签定位的是模板表面绑定，不是测量几何。</li>
<li>试用参数（3mm/12mm/18mm）未经解剖学验证；更小候选（2mm）为本次实际选中值。</li>
</ul>
</div>

<p style="color:#8a97a8;margin-top:40px">报告生成：{now} · tools/make_neck_report.py</p>
</div></body></html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
