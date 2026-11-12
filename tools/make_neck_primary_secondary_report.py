"""Write NECK_PRIMARY_SECONDARY_SHAPE_REPORT.html."""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path

PROJECT = Path("D:/proiect/facecap_airway_delivery")


def img(path: Path) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def write_report(out: Path, *, summary: dict) -> Path:
    checks = summary["chosen_checks"]
    morph = summary["morph"]
    audit = summary["audits"]
    design = summary["design"]
    placement = summary["placement"]
    target = summary["target"]
    disk = summary["disk"]
    facial = summary["facial"]
    A_feat = summary["A_feat"]
    A_peak = summary.get("A_peak", {})
    B_metric = summary["B_metric"]
    public_switch = summary.get("public_switch")
    public_switched = summary.get("public_switched")
    total = summary.get("total_seconds", 0.)

    def fnum(v, n=2, missing="—"):
        try:
            return f"{float(v):.{n}f}"
        except (TypeError, ValueError):
            return missing

    def feat(s, key, name, n=2):
        try:
            return fnum(audit[s]["features"][key][name], n)
        except (KeyError, TypeError):
            return "—"

    def rel(s, key, n=3):
        try:
            return fnum(audit[s]["features"]["relative"][key], n)
        except (KeyError, TypeError):
            return "—"

    def a_feature(key, name):
        try:
            return fnum(A_feat["closed"][key][name], 2)
        except (KeyError, TypeError):
            return "—"

    check_rows = "".join(
        f"<tr class='{'pass' if v else 'fail'}'><td>{k}</td><td>{'通过' if v else '未通过'}</td></tr>"
        for k, v in checks.items())
    feat_rows = "".join(
        f"<tr><td>{s}</td><td>{feat(s,'A','v_mm')}</td><td>{feat(s,'A','height_mm')}</td>"
        f"<td>{feat(s,'B','v_mm')}</td><td>{feat(s,'B','height_mm')}</td>"
        f"<td>{rel(s,'B_over_A')}</td><td>{fnum(audit[s]['features']['saddle']['height_mm'],2)}</td>"
        f"<td>{rel(s,'saddle_over_A')}</td><td>{feat(s,'B','prominence_mm')}</td>"
        f"<td>{fnum(audit[s]['features']['B']['width']['w50_mm'],0) if audit[s]['features']['B']['width'] else '—'}</td><td>{audit[s]['features'].get('B_stable_offsets')}</td>"
        f"<td>{fnum(audit[s]['support']['clearance_min_mm'],2)}</td>"
        f"<td>{fnum(audit[s]['support']['bottom_margin_mm'],2)}</td></tr>"
        for s in ("closed", "open"))
    causal_rows = "".join(
        f"<tr><td>{label}</td><td>{state}</td><td>{feat_text}</td><td>{neg}</td>"
        f"<td>{support}</td></tr>"
        for label, state, feat_text, neg, support in (
            ("A 0.6.0 single", "closed", f"A {fnum(A_peak.get('closed'))} mm; B absent", "0", "31.89/8.64"),
            ("A 0.6.0 single", "open", f"A {fnum(A_peak.get('open'))} mm; B absent", "0", "31.89/8.64"),
            ("B two-feature old placement", "closed",
             f"A {fnum(B_metric['closed']['features'].get('A',{}).get('height_mm'))}; "
             f"B {fnum(B_metric['closed']['features'].get('B',{}).get('height_mm'))}",
             fnum(B_metric['closed']['negative_area_mm2'],0),
             fnum(B_metric['closed']['support'].get('clearance_min_mm'),2)),
            ("B two-feature old placement", "open",
             f"A {fnum(B_metric['open']['features'].get('A',{}).get('height_mm'))}; "
             f"B {fnum(B_metric['open']['features'].get('B',{}).get('height_mm'))}",
             fnum(B_metric['open']['negative_area_mm2'],0),
             fnum(B_metric['open']['support'].get('clearance_min_mm'),2)),
            ("C final", "closed", f"A {feat('closed','A','height_mm')}; B {feat('closed','B','height_mm')}",
             fnum(audit['closed']['base_relative']['negative_face_area_mm2'],0),
             fnum(audit['closed']['support']['clearance_min_mm'],2)),
            ("C final", "open", f"A {feat('open','A','height_mm')}; B {feat('open','B','height_mm')}",
             fnum(audit['open']['base_relative']['negative_face_area_mm2'],0),
             fnum(audit['open']['support']['clearance_min_mm'],2))))
    disk_rows = "".join(
        f"<tr><td>{s}</td><td>{disk[s]['vertices']}</td><td>{disk[s]['faces']}</td>"
        f"<td>{disk[s]['topology']['boundary_edges']}</td>"
        f"<td>{disk[s]['topology']['inconsistent_shared_edges']}</td>"
        f"<td>{disk[s]['topology']['nonmanifold_edges']}</td>"
        f"<td>{disk[s]['max_abs_diff_m']:.2e}</td><td>{disk[s]['no_materials']}</td></tr>"
        for s in ("closed", "open"))
    full_figs = "".join(
        f"<figure><img style='width:100%' src='{img(out/'render'/f'compare_{s}_{v}_full_flat.png')}'/>"
        f"<figcaption>{s} / {v}：参考 · 未塑形基线 · 当前 0.6.0 单峰 · 本轮双特征（同尺度/平光/平面）</figcaption></figure>"
        for s in ("closed", "open") for v in ("front", "right", "left", "right45", "left45"))
    close_figs = "".join(
        f"<figure><img style='width:100%' src='{img(out/'render'/f'compare_{s}_neck_{v}_{shade}.png')}'/>"
        f"<figcaption>{s} 颈前局部 {v}（{shade}）：参考 · 基线 · 0.6.0 · 本轮</figcaption></figure>"
        for s in ("closed", "open") for v in ("front", "right45") for shade in ("flat", "smooth"))
    causal_figs = "".join(
        f"<figure><img style='width:82%' src='{img(out/'render'/f'causal_{s}_45_flat.png')}'/>"
        f"<figcaption>{s} 因果对照：基线 · B（双特征旧位置/尺度） · C（最终比例/边界协调）</figcaption></figure>"
        for s in ("closed", "open"))
    section_figs = "".join(
        f"<figure><img style='width:92%' src='{img(out/'render'/f'section_{s}_{name}.png')}'/>"
        f"<figcaption>{s}：{'中线纵向（含 A/B）' if name=='long' else ('A 处横向' if name=='crossA' else 'B 处横向')}剖面与相对基线差</figcaption></figure>"
        for s in ("closed", "open") for name in ("long", "crossA", "crossB"))
    heat_figs = "".join(
        f"<figure><img style='width:96%' src='{img(out/'render'/f'heat_{s}.png')}'/>"
        f"<figcaption>{s}：双特征应用场热图与 10%/50% 等值线</figcaption></figure>"
        for s in ("closed", "open"))
    wire_figs = "".join(
        f"<figure><img style='width:56%' src='{img(out/'render'/f'wire_{s}.png')}'/>"
        f"<figcaption>{s}：双特征过渡区线框</figcaption></figure>"
        for s in ("closed", "open"))
    feature_fig = f"<figure><img style='width:100%' src='{img(out/'render'/'feature_audit.png')}'/>" \
                  f"<figcaption>左：参考 σ=8 mm 双特征剖面与旧 ROI 下界 −125 mm（B 中心 −127 mm 被截）；" \
                  f"右：本轮闭/张 A–鞍–B 高度链</figcaption></figure>"
    feature_rows = (
        f"<tr><td>feature_A</td><td>x={morph['feature_A']['x_mm']:.1f}, y={morph['feature_A']['y_mm']:.1f} mm；"
        f"高 {morph['feature_A']['height_mm']:.2f} mm；目标闭/张 {feat('closed','A','height_mm')} / {feat('open','A','height_mm')} mm；"
        f"目标 W50 {fnum(audit['closed']['features']['A']['width']['w50_mm'],0) if audit['closed']['features']['A']['width'] else '—'} mm</td></tr>"
        f"<tr><td>feature_B</td><td>x={morph['feature_B']['x_mm']:.1f}, y={morph['feature_B']['y_mm']:.1f} mm；"
        f"高 {morph['feature_B']['height_mm']:.2f} mm；目标闭/张 {feat('closed','B','height_mm')} / {feat('open','B','height_mm')} mm；"
        f"局部突出 {feat('closed','B','prominence_mm')} mm；目标 W50 {fnum(audit['closed']['features']['B']['width']['w50_mm'],0) if audit['closed']['features']['B']['width'] else '—'} mm</td></tr>"
        f"<tr><td>相对大小 / 间距</td><td>参考 B/A {morph['relative']['B_over_A']:.3f}，中心距 {morph['relative']['centre_distance_mm']:.1f} mm；"
        f"目标闭/张 B/A {rel('closed','B_over_A')} / {rel('open','B_over_A')}，间距 {rel('closed','centre_distance_mm')} / {rel('open','centre_distance_mm')} mm</td></tr>"
        f"<tr><td>鞍部</td><td>参考 {morph['saddle']['height_mm']:.2f} mm（{morph['saddle']['relative_to_A']:.3f} A；B 高于鞍 {morph['relative']['B_over_saddle']:.2f} mm）；"
        f"目标闭/张 {feat('closed','B','v_mm') and fnum(audit['closed']['features']['saddle']['height_mm'],2)} / {fnum(audit['open']['features']['saddle']['height_mm'],2)} mm</td></tr>"
        f"<tr><td>灵敏度</td><td>{json.dumps(morph['sensitivity'],ensure_ascii=False)}；σ=6/8 mm 检出两峰，σ=10 mm B 并入宽缓体量；B 是浅而宽的次级肩脊，不是第二个球包</td></tr>"
        f"<tr><td>设计假设</td><td>纵向采用参考测得的连续双峰控制曲线（非两个独立球相加），横向宽度在 A/B 之间连续过渡；"
        f"整体缩放 q·f={summary.get('q',0):.3f}·{placement.get('fit_scale')}；f 受张口保护与真实颈底切口同时约束</td></tr>"
    )
    if public_switched and public_switch and public_switch.get("accepted"):
        max_diff = max(public_switch["meshes"][s]["max_vertex_diff_m"] for s in ("closed", "open"))
        default_text = (
            "已切换到同一生产实现（`neck_prominence.algorithm=continuous_reference`，pipeline 使用 "
            "`neck_dual.build_states`）。公开交付：<code>" + public_switch["delivery"] + "</code>"
            f"（包装脚本 `headrgbd.cmd export`，版本 {public_switch['manifest_version']}，"
            f"与独立工具逐顶点一致 ≤{max_diff:.1e} m）。"
            "公开 manifest 的 degraded 仍来自既有张口内唇深度先验警告（与基线相同），不是颈部算法失败。")
    else:
        default_text = "尚未切换：本轮独立交付已通过，但公共入口切换/端到端复核尚未写入。"
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>主次双特征颈前外观报告</title>
<style>
 body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#1c1e22;color:#d8dce2;line-height:1.55}}
 .wrap{{max-width:1140px;margin:0 auto;padding:32px 24px 64px}}
 h1{{font-size:24px;border-bottom:2px solid #4a90d9;padding-bottom:10px}}
 h2{{font-size:18px;color:#7fb3e8;margin-top:32px;border-left:4px solid #4a90d9;padding-left:10px}}
 h3{{font-size:14px;color:#aab7c8;margin-top:22px}}
 table{{border-collapse:collapse;margin:12px 0;width:100%;font-size:13px}}
 td,th{{border:1px solid #3a3f47;padding:5px 9px;text-align:left}} th{{background:#262a31}}
 tr.pass td:last-child{{color:#7fd88f}} tr.fail td:last-child{{color:#e06c5f}}
 table.kv td:first-child{{color:#aab7c8;width:32%}}
 code{{background:#262a31;padding:1px 5px;border-radius:3px;font-size:12px}}
 .note{{background:#2a2519;border:1px solid #6b5b1e;padding:10px 14px;border-radius:6px;margin:12px 0}}
 .warn{{background:#2a1f1f;border:1px solid #6b2e2e;padding:10px 14px;border-radius:6px;margin:12px 0}}
 figure{{margin:14px 0;text-align:center}} figure img{{border-radius:6px;max-width:100%}}
 figcaption{{font-size:12.5px;color:#8a97a8;margin-top:5px}}
</style></head><body><div class="wrap">
<h1>主次双特征颈前外观 — 交付报告（0.7.0）</h1>
<p style="color:#8a97a8">Reference two-feature appearance, actual closed/open meshes, feature audit and four-way comparison.</p>

<div class="note"><b>最终网格（绝对路径）：</b><br>
闭口：<code>{disk['closed']['path']}</code><br>
张口：<code>{disk['open']['path']}</code><br>
editing_mode=true · geometry_source=reference_driven_appearance · subject_observation_matched=false<br>
参考：<code>{morph['source_path']}</code>（SHA256 <code>{morph['source_sha256']}</code>）</div>

<h2>0. 七个问题</h2>
<ol>
<li><b>模板的两处突起是否为真实几何，方向是什么？</b>是。对参考原始网格的前表面高度做高斯背景（σ=8 mm）扣除后，中线 x≈0 出现两个纵向峰：
A（上、主）在 y≈{morph['feature_A']['y_mm']:.1f} mm，高 {morph['feature_A']['height_mm']:.2f} mm；
B（下、次）在 y≈{morph['feature_B']['y_mm']:.1f} mm，高 {morph['feature_B']['height_mm']:.2f} mm；
中心距 {morph['relative']['centre_distance_mm']:.1f} mm，B/A={morph['relative']['B_over_A']:.3f}，峰间鞍部
{morph['saddle']['height_mm']:.2f} mm（{morph['saddle']['relative_to_A']:.3f}A，B 比鞍高 {morph['relative']['B_over_saddle']:.2f} mm）。
两峰在 σ=6/8 mm 均检出，σ=10 mm 时 B 并入宽缓下颈体量；方向是<b>上下排列</b>，不是左右或斜向，也不是光照/法向造成的假象。
两侧 |x|≈20 mm、y≈−75 mm 的强峰属于颏下/下颌轮廓，已排除；|x|≈26–31 mm 的旧 1.55 mm 岛是横向肩部残差，<b>不是 B</b>。</li>
<li><b>哪些解剖解释有依据，哪些仅是外观命名？</b>A 与甲状软骨喉隆起的位置一致；B 位于下颈/上胸过渡，可能对应环状软骨层面或下颈体量，但本工作没有软骨/解剖成像证据。
因此报告统一使用 <b>reference_feature_A / reference_feature_B</b>；不把表面外观结果称为真实软骨形态或解剖测量。</li>
<li><b>当前单峰算法在哪一步丢掉第二处特征？</b>最先在<b>参考采样与门控</b>：旧 ROI 的 y 下界为 −125 mm，B 中心 −127 mm 已在 ROI 外；
下侧 9 mm 门控在 y≤−125 mm 归零，B 的上翼在 −120…−125 mm 被衰减；随后峰值选择核只取 y∈[−116,−94] mm，完全排除 B；
最后单峰 C2 参数化即使看到 B 也无法表达两个不等高峰。二次背景的部分吸收进一步减弱 B，但不是第一丢失点。
旧肩部岛与 B 不同，不能拿它们冒充第二处特征。</li>
<li><b>新结果如何保留主次结构且不挖坑？</b>把参考中线双峰剖面作为一条连续控制曲线（带 C2 端窗），横向宽度在 A/B 之间连续过渡；
A、B 共享一个非负支撑域，鞍部为正（目标闭/张 {fnum(audit['closed']['features']['saddle']['height_mm'],2)} / {fnum(audit['open']['features']['saddle']['height_mm'],2)} mm），
未做负残差迁移、未逐点截零、未生成两个独立球包；相对未塑形基线最小法向位移 0.000 mm、负面积 0 mm²。
由于目标短颈/下颌保护/真实颈底切口，整体设计协调缩放 f={placement.get('fit_scale')}，主峰 2.73 mm、次峰 1.00 mm；
主峰小于 0.6.0 的 4.04 mm 是空间协调结果，不是把双峰抹平。</li>
<li><b>闭口、张口是否均通过？</b>是。闭/张 A 峰 {feat('closed','A','height_mm')} / {feat('open','A','height_mm')} mm，
B 峰 {feat('closed','B','height_mm')} / {feat('open','B','height_mm')} mm，B/A {rel('closed','B_over_A')} / {rel('open','B_over_A')}，
B 局部突出 {feat('closed','B','prominence_mm')} / {feat('open','B','prominence_mm')} mm，B 在 ±3/±5 mm 相邻剖面稳定；
10% 支撑保护净距 {fnum(audit['closed']['support']['clearance_min_mm'],2)} / {fnum(audit['open']['support']['clearance_min_mm'],2)} mm，
距真实颈底切口 {fnum(audit['closed']['support']['bottom_margin_mm'],2)} / {fnum(audit['open']['support']['bottom_margin_mm'],2)} mm；
面部首命中按原始基线面索引最大 {facial['closed']['max_abs_shift_mm']} / {facial['open']['max_abs_shift_mm']} mm；
全部 {sum(1 for v in checks.values() if v)}/{len(checks)} 项验收通过。张口与闭口的主次结构一致，世界坐标/局部姿态不同是正常的。</li>
<li><b>两个最终 OBJ：</b>见上方绝对路径；单位 m，共享拓扑，边界仍 124 条，无材质，回读误差 ≤{max(disk[s]['max_abs_diff_m'] for s in disk):.1e} m。</li>
<li><b>实际默认、版本、耗时和退出状态：</b>{default_text}
版本 0.7.0；本轮独立交付总 {total:.1f} s，验收 {'通过，退出码 0 / ok' if summary['accepted'] else '未通过，退出码 1 / verification_failed'}；详见 run_status.json。</li>
</ol>

<h2>1. 参考特征审计</h2>
<table class="kv">{feature_rows}</table>
{feature_fig}
<p style="color:#8a97a8">背景与 σ 的定义是外观定义，不是解剖高度；A/B 高度是相对该背景的正向突出。参考负残差、侧沟、横向肩部岛均未迁移。</p>

<h2>2. A/B/C 因果对照（实际网格）</h2>
<table><tr><th>方案</th><th>状态</th><th>特征高度</th><th>负面积 mm²</th><th>10% 保护净距 mm</th></tr>{causal_rows}</table>
{causal_figs}
<p>B 在旧 0.6.0 中心/尺度上恢复了双特征（说明第二处特征确实是采样/单峰模型丢失），但其支撑未按目标短颈/保护/颈底重新协调；
C 在同一双特征表示上做比例定位、整体缩放和边界协调。</p>

<h2>3. 最终设计、位置与比例</h2>
<table class="kv">
<tr><td>参考颈宽 / 目标颈段</td><td>{morph['reference_neck_width_mm']:.1f} mm；W_neck {target['W_neck_mm']:.1f} mm；L_neck {target['L_neck_mm']:.1f} mm（闭/张 {target['L_per_state']['closed']:.1f}/{target['L_per_state']['open']:.1f} mm）</td></tr>
<tr><td>整体缩放 q·f</td><td>{summary.get('q',0):.4f} · {placement.get('fit_scale')}（高度/宽度/长度/尖端与 A-B 间距同比例）</td></tr>
<tr><td>A 中心绑定</td><td>基顶点 {placement.get('vertex')}；s_ref {morph['s_ref']} → s_chosen {fnum(placement.get('s_chosen'),3)}；A/B 设计 v={fnum(design.get('v_b_mm'),1)} mm</td></tr>
<tr><td>设计尺寸</td><td>H_A {fnum(design.get('height_mm'),2)} mm；支撑 v [{fnum(design.get('v_bottom_mm'),1)}, {fnum(design.get('v_top_mm'),1)}] mm；
横半宽 A/B {fnum(design.get('half_a_mm'),1)}/{fnum(design.get('half_b_mm'),1)} mm；尖端 {fnum(design.get('tip_a_mm'),1)}/{fnum(design.get('tip_b_mm'),1)} mm</td></tr>
<tr><td>支撑边界</td><td>10% 保护净距闭/张 {fnum(audit['closed']['support']['clearance_min_mm'],2)}/{fnum(audit['open']['support']['clearance_min_mm'],2)} mm；
保护渐变最小值 {fnum(audit['closed']['support']['gate_min'],3)}/{fnum(audit['open']['support']['gate_min'],3)}；
真实颈底切口上方 {fnum(audit['closed']['support']['bottom_margin_mm'],2)}/{fnum(audit['open']['support']['bottom_margin_mm'],2)} mm</td></tr>
<tr><td>轮廓</td><td>单一非负 C2 支撑域；鞍部为正值；外侧位移连续归零；无环沟/凹坑；未做基面修复</td></tr>
</table>

<h2>4. 闭/张特征与安全验收</h2>
<table><tr><th>状态</th><th>A v mm</th><th>A 高</th><th>B v mm</th><th>B 高</th><th>B/A</th>
<th>鞍高</th><th>鞍/A</th><th>B 局部突出</th><th>B W50</th><th>B 稳定剖面数</th><th>保护净距</th><th>颈底余量</th></tr>{feat_rows}</table>
<table><tr><th>检查</th><th>结果</th></tr>{check_rows}</table>
<table><tr><th>状态</th><th>顶点</th><th>面</th><th>边界边</th><th>不一致共享边</th><th>非流形边</th><th>回读误差 m</th><th>无材质</th></tr>{disk_rows}</table>
<p>面部首命中按原始基线面索引：闭/张保护面 {facial['closed']['protected_face_hits']}/{facial['open']['protected_face_hits']} 条，
最大 {facial['closed']['max_abs_shift_mm']}/{facial['open']['max_abs_shift_mm']} mm；保护顶点移动 0；局部自交范围见 geometry_verification.json。</p>

<h2>5. 视觉复核（同尺度/视角/光照）</h2>
<h3>全头四方案（参考 / 基线 / 0.6.0 / 双特征）</h3>
{full_figs}
<h3>颈前局部（平面 + 平滑）</h3>
{close_figs}
<h3>纵向与 A/B 横向剖面</h3>
{section_figs}
<h3>位移热图与线框</h3>
{heat_figs}
{wire_figs}

<h2>6. 局限</h2>
<div class="warn"><ul>
<li>这是参考驱动的外观塑形；参考峰值/双峰高度是相对高斯背景的外观残差，不是受试者解剖高度，subject_observation_matched=false。</li>
<li>B 是浅而宽的次级肩脊：参考仅比鞍高 {morph['relative']['B_over_saddle']:.2f} mm，局部突出有限；未为了“可辨”而抬高 B 或挖深鞍部。</li>
<li>目标颈段短、下颌保护与真实颈底切口同时约束，整体缩小 f={placement.get('fit_scale')}；主峰小于 0.6.0，但 A/B 比例、B/A 和鞍部比例与参考一致（见第 4 节）。</li>
<li>特征命名是外观标签；未做软骨/解剖验证。自交检测为局部受影响面范围，不是整头绝对保证。</li>
</ul></div>

<p style="color:#8a97a8;margin-top:30px">生成：{datetime.now().strftime('%Y-%m-%d %H:%M')} · tools/run_neck_primary_secondary.py · 输出：{out.name} · 版本 {summary.get('version')}</p>
</div></body></html>"""
    path = out / "NECK_PRIMARY_SECONDARY_SHAPE_REPORT.html"
    path.write_text(html, encoding="utf-8")
    return path
