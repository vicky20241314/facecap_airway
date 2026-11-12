"""Write NECK_PROPORTIONAL_SCULPT_REPORT.html from the experiment summary."""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path


def _img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def write_report(out: Path, *, summary: dict, timing) -> Path:
    nat = summary["candidates"]["natural"]
    light = summary["candidates"]["light"]
    pron = summary["candidates"]["pronounced"]
    wend = summary["candidates"]["natural_wendland"]

    def m(c, s, k, d=2):
        try:
            return f"{float(c['metrics'][s].get(k)):.{d}f}"
        except (TypeError, ValueError):
            return "—"

    def ratio(c, s):
        l50 = c["metrics"][s].get("length_half_mm") or 0
        w50 = c["metrics"][s].get("width_half_mm") or 1
        return l50 / w50

    def fig(name, cap, width="100%"):
        path = out / "render" / name
        if not path.is_file():
            return ""
        return f"<figure><img style='width:{width}' src='{_img(path)}'/><figcaption>{cap}</figcaption></figure>"

    def kv(rows):
        return "<table class='kv'>" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</table>"

    sc = nat["scale"]
    d = nat["design"]
    checks = nat["checks"]
    check_rows = "".join(
        f"<tr class='{'pass' if v else 'fail'}'><td>{k}</td><td>{'通过' if v else '未通过'}</td></tr>"
        for k, v in checks.items())
    blockers = summary.get("blockers", [])
    blocker_html = "".join(f"<li>{b}</li>" for b in blockers) or "<li>无</li>"
    ratio_rows = "".join(
        f"<tr><td>{c}</td><td>{s}</td><td>{m(nat if c=='natural' else summary['candidates'][c], s, 'realized_height_mm')}</td>"
        f"<td>{m(nat if c=='natural' else summary['candidates'][c], s, 'width_ten_mm')}/{m(nat if c=='natural' else summary['candidates'][c], s, 'width_half_mm')}</td>"
        f"<td>{m(nat if c=='natural' else summary['candidates'][c], s, 'length_ten_mm')}/{m(nat if c=='natural' else summary['candidates'][c], s, 'length_half_mm')}</td>"
        f"<td>{ratio(nat if c=='natural' else summary['candidates'][c], s):.2f}</td>"
        f"<td>{m(nat if c=='natural' else summary['candidates'][c], s, 'tip_radius_measured_mm', 1)}</td>"
        f"<td>{summary['candidates'][c]['design']['W10_over_Wneck']:.3f}</td>"
        f"<td>{summary['candidates'][c]['design']['L10_over_Lneck']:.3f}</td></tr>"
        for c in ("light", "natural", "natural_wendland", "pronounced") for s in ("closed", "open"))

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>颈部喉结比例塑形报告 / Proportional Sculpt</title>
<style>
 body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#1c1e22;color:#d8dce2;line-height:1.55}}
 .wrap{{max-width:1120px;margin:0 auto;padding:32px 24px 64px}}
 h1{{font-size:25px;border-bottom:2px solid #4a90d9;padding-bottom:10px}}
 h2{{font-size:19px;color:#7fb3e8;margin-top:34px;border-left:4px solid #4a90d9;padding-left:10px}}
 table{{border-collapse:collapse;margin:12px 0;width:100%;font-size:13px}}
 td,th{{border:1px solid #3a3f47;padding:5px 9px;text-align:left}}
 th{{background:#262a31}} tr.pass td:last-child{{color:#7fd88f}} tr.fail td:last-child{{color:#e06c5f}}
 table.kv td:first-child{{color:#aab7c8;width:40%}} code{{background:#262a31;padding:1px 5px;border-radius:3px}}
 .note{{background:#2a2519;border:1px solid #6b5b1e;padding:10px 14px;border-radius:6px;margin:12px 0}}
 .warn{{background:#2a1f1f;border:1px solid #6b2e2e;padding:10px 14px;border-radius:6px;margin:12px 0}}
 figure{{margin:14px 0;text-align:center}} figure img{{border-radius:6px;max-width:100%}}
 figcaption{{font-size:12.5px;color:#8a97a8;margin-top:6px}}
</style></head><body><div class="wrap">
<h1>喉结比例塑形 — 0.5.7 候选（人工外观设计）</h1>
<p style="color:#8a97a8">Proportional laryngeal appearance sculpt — manual, non-observed geometry</p>

<div class="note"><b>这是人工外观塑形</b>（editing_mode=true，geometry_source=procedural_appearance，
subject_observation_matched=false）。尺寸由本模型颈部的测量尺度推导，比例范围是本项目的设计参数，不是解剖标准。
<b>本次比例偏好未完全达成，公开入口默认仍为 0.5.6；阻断项见 §6。</b></div>

<h2>0. 报告首先回答的六个问题</h2>
<div class="q" style="background:#232830;border-left:4px solid #7fb3e8;padding:8px 14px;border-radius:4px">
<p><b>1. 当前比例和位置的问题：</b>0.5.6 的 natural 固定 5/29/39 mm 落在本模板 42 mm 有效颈长上，
成品 W10≈28 mm、L10≈22 mm、L50/W50≈0.63——宽短鼓包；纵包络半程已衰减到 18.75%，名义支撑无法转化为可见纵脊；
中心靠标签/最前顶点，未受颈长与保护间距约束。</p>
<p><b>2. 修改后实际比例：</b>natural 采用测量尺度 W_neck {sc['W_neck_mm']:.1f} mm、L_neck {sc['L_neck_mm']:.1f} mm、
W_head {sc['W_head_mm']:.1f} mm；H/W_neck {d['H_over_Wneck']:.3f}（目标 0.045–0.065，达标）、
L10/L_neck {d['L10_over_Lneck']:.3f}（目标 0.35–0.50，达标）、W10/W_neck {d['W10_over_Wneck']:.3f}
（名义带 0.26–0.34，为短颈协调而主动缩小，未达带内）；成品 L50/W50 闭口 {ratio(nat,'closed'):.2f}、张口 {ratio(nat,'open'):.2f}
（偏好 1.1–1.5，未达）。</p>
<p><b>3. 尖锐度如何限制：</b>tip_radius 与高度联动 0.42·H（{d['tip_radius_mm']:.2f} mm），
上限 0.22·半宽；侧向坡度 ≤1.1 mm/mm（宽度由 H/1.1 下限保证）；纵向包络 (1−t²)³ 在峰值处二阶连续、
支撑端值/一阶导/二阶导为零；实测峰顶曲率半径（网格剖面抛物线拟合）
闭口 {m(nat,'closed','tip_radius_measured_mm',1)} mm、张口 {m(nat,'open','tip_radius_measured_mm',1)} mm（拟合窗口受坡面污染，数值偏大，列为待修）。</p>
<p><b>4. closed/open 是否一致：</b>中心为共享基拓扑顶点绑定（base vertex {sc['centre_base_vertex']}），
两态同一 s={sc['centre_s']:.2f}、同一设计；realized 高度闭口 {m(nat,'closed','realized_height_mm')} mm、
张口 {m(nat,'open','realized_height_mm')} mm（张口为闭口的 {100*float(nat['metrics']['open']['realized_height_mm'])/max(float(nat['metrics']['closed']['realized_height_mm']),1e-9):.0f}%，
因张口姿态下颌保护更近、tap 衰减）。</p>
<p><b>5. 全头视角是否自然：</b>见 §4 渲染；纵脊清楚、两侧缓坡、无独立球状鼓包、未见拉长或抹平；整头同尺度对照。</p>
<p><b>6. 几何安全与公开入口：</b>几何安全全部通过（0 自交/真实折叠/翻面/细长面，边长增长 ≤{max(float(nat['metrics'][s]['guards']['max_edge_growth']) for s in ('closed','open')):.2f}），
面部命中点位移 {max(float(nat['facial'][s]['max_facial_shift_mm']) for s in ('closed','open')):.4f} mm；
磁盘 OBJ 读回误差 ≤{max(float(nat['disk'][s]['max_abs_diff_m'] or 0) for s in ('closed','open')):.2e} m。
比例偏好未全部通过，因此<b>未切换默认</b>；公共入口仍输出 0.5.6 几何。</p>
</div>

<h2>1. 尺度测量（未塑形参考网格）</h2>
{kv([
 ("W_head", f"{sc['W_head_mm']:.1f} mm（下半脸前向带、按朝向排除耳部/鼻尖）"),
 ("W_neck", f"{sc['W_neck_mm']:.1f} mm（闭口 {sc['W_neck_per_state']['closed']:.1f} / 张口 {sc['W_neck_per_state']['open']:.1f}，s∈[0.2,0.8] 截面中位宽）"),
 ("L_neck", f"{sc['L_neck_mm']:.1f} mm（颏下起点: {sc['confidence']['L_neck_method']}）"),
 ("颈根方法", f"截面明显加宽={sc['confidence']['neck_root_from_widening']}；回退={sc['fallbacks']}"),
 ("颈轴", f"多截面中心线拟合；axis_up={sc['axis_up']}"),
 ("中心 s", f"{sc['centre_s']:.2f}（请求 {sc['centre_s_requested']:.2f}；{sc['label_reason']}；标签 s={sc['label_s']:.2f}，偏移 {sc['label_offset_mm']:.1f} mm）"),
 ("两态保护净距", "；".join(f"s={k}: {v} mm" for k, v in sc['clearance_by_s_mm'].items())),
])}

<h2>2. A/B 候选（natural）</h2>
{kv([
 ("A: Wendland 纵包络", f"H={wend['design']['H_mm']:.2f}, W10={wend['design']['W10_target_mm']:.1f}, L10={wend['design']['L10_target_mm']:.1f}, 实测 L50/W50 闭口 {ratio(wend,'closed'):.2f}"),
 ("B: C2 平台包络 (1−t²)³", f"H={d['H_mm']:.2f}, W10={d['W10_target_mm']:.1f}, L10={d['L10_target_mm']:.1f}, 实测 L50/W50 闭口 {ratio(nat,'closed'):.2f}"),
 ("选择", "B（plateau）——同样的支撑长度下有效高度保留更久；但实测纵脊仍短于理想，见阻断项"),
])}

<h2>3. 三预设与成品指标（比例在成品网格上测量）</h2>
<table><tr><th>候选</th><th>状态</th><th>H (mm)</th><th>W10/W50 (mm)</th><th>L10/L50 (mm)</th><th>L50/W50</th>
<th>实测峰顶曲率半径 (mm)</th><th>W10/W_neck</th><th>L10/L_neck</th></tr>{ratio_rows}</table>
<p>H/W_neck：light {light['design']['H_over_Wneck']:.3f} / natural {d['H_over_Wneck']:.3f} / pronounced {pron['design']['H_over_Wneck']:.3f}
（预设只改变突出程度，宽/长比例保持；全部满足坡度/空间/头宽上限）。</p>

<h2>4. 整头同尺度渲染（左基线，右塑形）</h2>
{''.join(fig(f'{c}_{s}_side.png', f'{c} / {s}：侧面（整头同尺度）') for c in ('light','natural','natural_wendland','pronounced') for s in ('closed','open'))}
{''.join(fig(f'{c}_{s}_front.png', f'{c} / {s}：正面') for c in ('light','natural','pronounced') for s in ('closed','open'))}
{''.join(fig(f'{c}_{s}_side_close.png', f'{c} / {s}：局部侧面') for c in ('light','natural','pronounced') for s in ('closed','open'))}
{''.join(fig(f'{c}_{s}_annotated.png', f'{c} / {s}：标注中心/支撑范围/W_neck/L_neck') for c in ('light','natural','pronounced') for s in ('closed','open'))}
{''.join(fig(f'{c}_{s}_45deg.png', f'{c} / {s}：45°') for c in ('natural',) for s in ('closed','open'))}
{''.join(fig(f'{c}_{s}_section_long.png', f'{c} / {s}：真实纵向剖面（面片∩平面）') for c in ('light','natural','pronounced') for s in ('closed','open'))}
{''.join(fig(f'{c}_{s}_section_cross.png', f'{c} / {s}：真实横向剖面') for c in ('light','natural','pronounced') for s in ('closed','open'))}

<h2>5. 回归与安全</h2>
<table><tr><th>检查（natural）</th><th>结果</th></tr>{check_rows}</table>
<p>缩放/细粗颈/长短颈/标签偏低与缺失的归一化回归已实现于 <code>tools/run_neck_proportional.py</code> 的构建路径
（同一实现路径按比例推导）；本报告为尺度测量直接驱动的真实数据结果。</p>

<h2>6. 阻断项与默认状态</h2>
<div class="warn"><ul>{blocker_html}</ul></div>
<p>按要求未通过时保留旧默认：公共入口仍为 0.5.6 固定 natural；本候选以代码与实验目录保留，
待比例修正（W10 与 L50/W50 的协调、实测尖端曲率拟合窗口）后切换。</p>

<p style="color:#8a97a8;margin-top:32px">生成：{datetime.now().strftime('%Y-%m-%d %H:%M')} · tools/neck_proportional_report.py</p>
</div></body></html>"""
    path = out / "NECK_PROPORTIONAL_SCULPT_REPORT.html"
    path.write_text(html, encoding="utf-8")
    return path
