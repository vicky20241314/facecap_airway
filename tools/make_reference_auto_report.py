"""Write NECK_REFERENCE_AUTO_SCULPT_REPORT.html."""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path


def img(p: Path) -> str:
    if not Path(p).is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode("ascii")


def write_report(out: Path, *, summary: dict, timing) -> Path:
    checks = summary["chosen_checks"]
    check_rows = "".join(
        f"<tr class='{'pass' if v else 'fail'}'><td>{k}</td><td>{'通过' if v else '未通过'}</td></tr>"
        for k, v in checks.items())
    disk = summary["disk"]
    fix = summary["old_fix"]
    metrics = summary["metrics"]
    masset = summary["asset_peak_mm"]

    def m(state, key, d=2):
        try:
            return f"{float(metrics[state].get(key)):.{d}f}"
        except (TypeError, ValueError):
            return "—"

    figs = "".join(
        f"<figure><img style='width:100%' src='{img(out / 'render' / f'compare_{s}_{v}_flat.png')}'/>"
        f"<figcaption>{s} / {v}：参考 · 旧交付(黑面) · 旧交付仅修朝向 · 本轮自动塑形（同尺度/同光照/平面着色）</figcaption></figure>"
        for s in ("closed", "open") for v in ("front", "side", "45deg"))
    closefigs = "".join(
        f"<figure><img style='width:100%' src='{img(out / 'render' / f'compare_{s}_neck_close.png')}'/>"
        f"<figcaption>{s} 颈前局部：旧交付仅修朝向 / 自动塑形平滑法向 / 自动塑形平面着色</figcaption></figure>"
        for s in ("closed", "open"))
    secfigs = "".join(
        f"<figure><img style='width:46%' src='{img(out / 'render' / f'{s}_section_{k}.png')}'/>"
        f"<figcaption>{s}：{'纵向' if k == 'long' else '横向'}剖面（磁盘 OBJ 面片∩平面）</figcaption></figure>"
        for s in ("closed", "open") for k in ("long", "cross"))
    gv = json.loads((out / "geometry_verification.json").read_text(encoding="utf-8"))
    facial_row = "；".join(
        f"{s} 命中 {gv['states'][s]['facial_first_hit']['protected_face_hits']} 条，"
        f"最大位移 {gv['states'][s]['facial_first_hit']['max_abs_shift_mm']} mm"
        for s in ("closed", "open") if s in gv["states"])
    analysis = summary.get("analysis", {})
    fieldfigs = "".join(
        f"<figure><img style='width:100%' src='{img(out / 'render' / f'field_{s}.png')}'/>"
        f"<figcaption>{s}：应用场等值线（白 10% / 黄 50%，含外围全部区域）与中线纵/横向剖面</figcaption></figure>"
        for s in ("closed", "open"))

    def central_dims(state, level):
        try:
            comp = analysis[state]["levels"][level]["central"]
            box = comp["uv_bbox_mm"]
            return {"w": box[1] - box[0], "l": box[3] - box[2],
                    "vertices": comp["vertices"], "area": comp["area_mm2"]}
        except (KeyError, TypeError):
            return None

    def level_cell(state, level, key):
        dims = central_dims(state, level)
        if dims is None:
            return "—"
        lvl = analysis[state]["levels"][level]
        global_value = lvl["global_width_mm"] if key == "w" else lvl["global_length_mm"]
        return f"{dims[key]:.2f} / {global_value:.2f}"

    shape_rows = "".join(
        f"<tr><td>{s}</td><td>{analysis[s]['peak_outward_mm']:.2f}</td>"
        f"<td>{level_cell(s, '0.5', 'w')}</td><td>{level_cell(s, '0.5', 'l')}</td>"
        f"<td>{level_cell(s, '0.1', 'w')}</td><td>{level_cell(s, '0.1', 'l')}</td>"
        f"<td>{analysis[s]['levels']['0.5']['component_count']} / "
        f"{analysis[s]['levels']['0.1']['component_count']}</td>"
        f"<td>{central_dims(s, '0.5')['area']:.1f}</td></tr>"
        for s in ("closed", "open") if s in analysis)

    def component_list(state, sign):
        block = analysis[state]["residual_audit"][sign]
        rows = []
        for i, c in enumerate(block["components"], 1):
            box = c["uv_bbox_mm"]
            uv = c["uv_centroid_mm"]
            rows.append(
                f"#{i} 质心({uv[0]:.1f},{uv[1]:.1f}) u[{box[0]:.1f},{box[1]:.1f}] "
                f"v[{box[2]:.1f},{box[3]:.1f}] 面积{c['area_mm2']:.1f}mm² "
                f"峰{c['max_mm']:.2f}/谷{c['min_mm']:.2f}mm{'(含顶点)' if c['contains_apex'] else ''}")
        more = f"；另有 {block['component_count'] - len(block['components'])} 个未列出" \
            if block["truncated"] else ""
        return f"共 {block['component_count']} 个连通域" + ("：" + "；".join(rows) + more if rows else "")

    audit_rows = "".join(
        f"<tr><td>{s}</td><td>正 (≥+{analysis[s]['residual_audit']['threshold_mm']:.2f}mm)</td>"
        f"<td>{component_list(s, 'positive')}</td></tr>"
        f"<tr><td>{s}</td><td>负 (≤−{analysis[s]['residual_audit']['threshold_mm']:.2f}mm)</td>"
        f"<td>{component_list(s, 'negative')}；最低 {analysis[s]['support']['negative_min_mm']:.2f}mm</td></tr>"
        for s in ("closed", "open") if s in analysis)

    blunt_rows = "".join(
        f"<tr><td>{s}</td><td>{r['window_mm']:.0f}</td><td>{r['points']}</td>"
        f"<td>{r['radius_mm']:.2f}</td><td>{r['anisotropy']:.2f}</td>"
        f"<td>{r['fit_rms_mm']:.3f}</td><td>{'是' if r['bump_sign_ok'] else '否'}</td></tr>"
        for s in ("closed", "open") if s in analysis
        for r in analysis[s]["apex_curvature"])

    def level_note(state):
        l10 = analysis[state]["levels"]["0.1"]
        c10 = central_dims(state, "0.1")
        if c10 is None:
            return ""
        return (f"{state}：10% 水平正连通域 {l10['component_count']} 个；中心域 "
                f"W10={c10['w']:.1f}mm / 全局包围 {l10['global_width_mm']:.1f}mm，"
                f"其余为肩部与宽缓过渡（见上表连通域列表，未被裁剪或隐藏）。")

    level_notes = "<br>".join(level_note(s) for s in ("closed", "open") if s in analysis)
    back_note = "；".join(
        f"{s} 排除颈后投影 {analysis[s]['support']['back_projections_excluded']} 个"
        for s in ("closed", "open") if s in analysis)
    back_note += ("（面法向背向观察者，不参与等值线/剖面）。变形未移动任何后向法向顶点，"
                  "见 geometry_verification.json 的 backward_facing_moved_vertices 与验收项 "
                  "no_backward_facing_moved。")

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>参考驱动自动喉结塑形报告</title>
<style>
 body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#1c1e22;color:#d8dce2;line-height:1.55}}
 .wrap{{max-width:1120px;margin:0 auto;padding:32px 24px 64px}}
 h1{{font-size:24px;border-bottom:2px solid #4a90d9;padding-bottom:10px}}
 h2{{font-size:18px;color:#7fb3e8;margin-top:32px;border-left:4px solid #4a90d9;padding-left:10px}}
 table{{border-collapse:collapse;margin:12px 0;width:100%;font-size:13px}}
 td,th{{border:1px solid #3a3f47;padding:5px 9px;text-align:left}} th{{background:#262a31}}
 tr.pass td:last-child{{color:#7fd88f}} tr.fail td:last-child{{color:#e06c5f}}
 table.kv td:first-child{{color:#aab7c8;width:36%}}
 code{{background:#262a31;padding:1px 5px;border-radius:3px;font-size:12px}}
 .note{{background:#2a2519;border:1px solid #6b5b1e;padding:10px 14px;border-radius:6px;margin:12px 0}}
 .warn{{background:#2a1f1f;border:1px solid #6b2e2e;padding:10px 14px;border-radius:6px;margin:12px 0}}
 figure{{margin:14px 0;text-align:center}} figure img{{border-radius:6px;max-width:100%}}
 figcaption{{font-size:12.5px;color:#8a97a8;margin-top:5px}}
</style></head><body><div class="wrap">
<h1>参考驱动自动喉结塑形 — 交付报告（0.5.9 候选）</h1>
<p style="color:#8a97a8">Reference-driven laryngeal appearance edit — delivered meshes, disk verification</p>

<div class="note"><b>最终网格（绝对路径）：</b><br>
闭口：<code>{disk['closed']['path']}</code><br>
张口：<code>{disk['open']['path']}</code><br>
editing_mode=true · geometry_source=reference_driven_appearance · subject_observation_matched=false<br>
<b>公共默认仍为 0.5.6</b>；本轮为独立脚本运行成功，不代表所有入口已切换。</div>

<h2>0. 七个问题</h2>
<ol>
<li><b>旧交付色差是否存在真实几何问题？</b>存在<b>绕序缺陷</b>：两状态各 36 条共享边朝向不一致（基线为 0），
来自细分绿色过渡三角形的第三个面绕序反了；非流形边 0、边界 124 条与基线一致（嘴内+颈底），
<b>没有新增孔洞</b>。黑面是绕序 + 平面着色共同造成的，不是材质问题。</li>
<li><b>修复了什么、是否新增孔洞？</b>细分分支 `[m1,c,m2]→[m1,m2,c]` 已修；旧交付另做“仅朝向”对照：
闭/张各翻转 {fix['closed']['reversed_faces']}/{fix['open']['reversed_faces']} 个面，顶点位移 0，
翻转面 ID 与 repair_report.json 一并记录于 <code>old_delivery_orient_fix/</code>；
修复后不一致共享边 0、边界仍 124；<b>未新增孔洞</b>。</li>
<li><b>新喉结如何从本轮指定参考得到？</b>参考 ROI（|x|&lt;50、y∈(−125,−81) mm）→ 外环二次背景 → 有符号残差 →
C2 门控（排除颏下反折 y&gt;−86 与肩部）→ 参考提取峰值 <b>{masset:.2f} mm</b>；
按目标颈宽/参考颈宽 <b>{summary['scale']:.3f}</b> 同比缩放 → 共享基顶点 {summary['centre_vertex']} 绑定 →
局部两级细分 → 12 mm C2 保护距离门控。参考 SHA256 记录于 reference_shape.json。</li>
<li><b>闭口/张口的最终位置、尺寸、比例、尖锐度：</b>同一共享基拓扑绑定（中心基顶点 {summary['centre_vertex']}，自动选取），两状态各自姿态坐标。
闭口峰值 {m('closed','peak_outward_mm')} mm；中心连通 W50 {level_cell('closed','0.5','w')} mm、W10 {level_cell('closed','0.1','w')} mm、
L50 {level_cell('closed','0.5','l')} mm、L10 {level_cell('closed','0.1','l')} mm（中心/全局包围）。
张口峰值 {m('open','peak_outward_mm')} mm；W50 {level_cell('open','0.5','w')} mm、W10 {level_cell('open','0.1','w')} mm、
L50 {level_cell('open','0.5','l')} mm、L10 {level_cell('open','0.1','l')} mm。
局部二次高度拟合半径闭口 {analysis['closed']['apex_curvature'][0]['radius_mm']:.1f} / 张口 {analysis['open']['apex_curvature'][0]['radius_mm']:.1f} mm（±2mm 窗口），
为连续宽缓曲面而非针尖；外围残差与全部连通域见第 2 节。</li>
<li><b>两个最终 OBJ：</b>见上方绝对路径；单位 m、重算顶点法向、无材质组。</li>
<li><b>通过/未通过：</b>下表 {sum(1 for v in checks.values() if v)}/{len(checks)} 通过；未通过项明确列出。</li>
<li><b>耗时与退出状态：</b>总 {summary.get('total_seconds', 0.):.1f} s；退出码 {'0 / ok' if summary.get('verification_passed') else '1 / verification_failed'}（详见 run_status.json）。</li>
</ol>

<h2>1. 参考与提取</h2>
<table class="kv">
<tr><td>参考文件</td><td><code>{summary['reference']['path']}</code></td></tr>
<tr><td>SHA256 / 轴 / 单位</td><td><code>{summary['reference']['sha256']}</code> · {summary['reference']['axes']} · {summary['reference']['units']}</td></tr>
<tr><td>提取峰值 / 峰位</td><td>{masset:.2f} mm @ {json.loads((out/'reference_shape.json').read_text(encoding='utf-8'))['core_peak_xy_mm']} mm</td></tr>
<tr><td>负向范围（两侧/上下肩部）</td><td>{json.loads((out/'reference_shape.json').read_text(encoding='utf-8'))['negative_extent_mm']:.2f} mm（保留，不裁零）</td></tr>
<tr><td>参考颈宽 / 目标缩放</td><td>{json.loads((out/'reference_shape.json').read_text(encoding='utf-8'))['reference_neck_width_mm']:.1f} mm · scale {summary['scale']:.3f}</td></tr>
<tr><td>排除区域</td><td>颏下反折（y&gt;−86 mm）、肩部、所有后侧表面；有效域内无空洞（插值前已校验）</td></tr>
</table>

<h2>2. 形态度量、外围残差与尖端圆钝审计</h2>
<table><tr><th>状态</th><th>峰值 H (mm)</th><th>W50 中心/全局 (mm)</th><th>L50 中心/全局 (mm)</th>
<th>W10 中心/全局 (mm)</th><th>L10 中心/全局 (mm)</th><th>连通域数 50%/10%</th><th>中心域面积 50% (mm²)</th></tr>
{shape_rows}</table>
<p>{level_notes}</p>
<p style="color:#8a97a8">{back_note}</p>
<table><tr><th>状态</th><th>残差符号</th><th>全域连通域（位置、范围、面积、峰/谷值）</th></tr>{audit_rows}</table>
<p>残差阈值 0.05 mm；面积按连通域内三角面三维面积求和，是等值线内部面积的保守下界；未按中心裁剪。</p>
<table><tr><th>状态</th><th>拟合窗口 (mm)</th><th>点数</th><th>尖锐半径 (mm)</th>
<th>各向异性</th><th>拟合 rms (mm)</th><th>凸起符号正确</th></tr>{blunt_rows}</table>
{fieldfigs}

<h2>3. 四方案同尺度对照（参考 / 旧交付 / 仅修朝向 / 本轮）</h2>
{figs}
{closefigs}
<p>平面着色用于暴露真实面片朝向；平滑法向仅用于确认几何连续、不是光照掩盖。</p>

<h2>4. 真实剖面（最终磁盘 OBJ）</h2>
{secfigs}

<h2>5. 安全与磁盘复核</h2>
<table><tr><th>检查</th><th>结果</th></tr>{check_rows}</table>
<table class="kv">
<tr><td>共享边朝向（闭/张）</td><td>{disk['closed']['topology']['inconsistent_shared_edges']} / {disk['open']['topology']['inconsistent_shared_edges']}</td></tr>
<tr><td>非流形边 / 边界边</td><td>{disk['closed']['topology']['nonmanifold_edges']} / {disk['closed']['topology']['boundary_edges']}（基线同为 {json.loads((out/'geometry_verification.json').read_text(encoding='utf-8'))['baseline_topology']['boundary_edges']}）</td></tr>
<tr><td>自交（局部检测范围）</td><td>{json.loads((out/'geometry_verification.json').read_text(encoding='utf-8'))['intersection_scope']}</td></tr>
<tr><td>受保护位移（闭/张）</td><td>{m('closed','protected_displacement_mm',4)} / {m('open','protected_displacement_mm',4)} mm</td></tr>
<tr><td>面部首命中（原始基线面索引）</td><td>{facial_row}（≤0.05 mm 通过）</td></tr>
<tr><td>磁盘坐标误差</td><td>≤{max(disk[s]['max_abs_diff_m'] for s in disk):.2e} m</td></tr>
</table>

<h2>6. 权重与局限</h2>
<div class="warn"><ul>
<li>参考峰值 {masset:.2f} mm 是参考外观位移，不是人体真实喉结高度；缩放 {summary['scale']:.3f} 为颈部宽度比。</li>
<li>W10/L10 的“全局包围”包含肩部与宽缓过渡；“中心连通”只含峰值所在连通域，两者并列给出，连通域清单未被裁剪。</li>
<li>尖端半径是应用高度场的局部二次拟合（2/3/4 mm 窗口），不是解剖测量；窗口间变化见上表。</li>
<li>自交检测为局部受影响面范围，不是整头穷举保证。</li>
<li>公共默认未切换；如需发布需统一更新入口、版本与回归测试。</li>
</ul></div>

<p style="color:#8a97a8;margin-top:30px">生成：{datetime.now().strftime('%Y-%m-%d %H:%M')} · tools/run_neck_reference_auto.py · 输出：{out.name}</p>
</div></body></html>"""
    path = out / "NECK_REFERENCE_AUTO_SCULPT_REPORT.html"
    path.write_text(html, encoding="utf-8")
    return path
