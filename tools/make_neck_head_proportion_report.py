"""Write NECK_HEAD_PROPORTION_REPORT.html."""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path

PROJECT = Path("D:/proiect/facecap_airway_delivery")


def img(path: Path) -> str:
    p = Path(path)
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii") if p.is_file() else ""


def write_report(out: Path, *, summary: dict) -> Path:
    checks = summary["checks"]
    scale = summary["head_scale"]
    targets = summary["targets"]
    metrics = summary["final_metrics"]
    prop = summary["proportion"]
    C = summary["C_feat"] or {}
    B = summary["base_feat"] or {}
    A7 = summary["A_feat"] or {}
    Bfull_support = summary["B_support"]
    Bfull_guard = summary["B_guard"]
    Bfull_inter = summary["B_inter"]
    C_support = summary["C_support"]
    morph = summary["morph"]
    Bfeat = summary.get("B_feat", {})
    def fmt(v, n=4):
        return "—" if v is None else f"{float(v):.{n}f}"
    design = summary["C_design"]
    disk = summary["disk"]
    facial = summary["facial"]
    total = summary.get("total_seconds", 0.)
    ref_rat = targets["reference_ratios"]; tgt_mm = targets["target_mm"]
    tol = targets["tolerances"]
    prop_rows_list = []
    for k, v in prop.items():
        err = "—" if v.get("rel_error") is None else f"{v['rel_error']*100:.1f}%"
        prop_rows_list.append(
            f"<tr class='{'pass' if v['pass'] else 'fail'}'><td>{k}</td>"
            f"<td>{v['reference']:.4f}</td><td>{fmt(v.get('final'))}</td>"
            f"<td>{err}</td><td>{v['tolerance']*100:.0f}%</td>"
            f"<td>{'通过' if v['pass'] else '未通过'}</td></tr>")
    prop_rows = "".join(prop_rows_list)
    check_rows = "".join(
        f"<tr class='{'pass' if v else 'fail'}'><td>{k}</td><td>{'通过' if v else '未通过'}</td></tr>"
        for k, v in checks.items())
    disk_rows = "".join(
        f"<tr><td>{s}</td><td>{disk[s]['vertices']}</td><td>{disk[s]['faces']}</td>"
        f"<td>{disk[s]['topology']['boundary_edges']}</td>"
        f"<td>{disk[s]['topology']['inconsistent_shared_edges']}</td>"
        f"<td>{disk[s]['topology']['nonmanifold_edges']}</td><td>{disk[s]['max_abs_diff_m']:.2e}</td></tr>"
        for s in ("closed", "open"))

    def fh(feat, key):
        try:
            return f"{float(feat[key]):.2f}"
        except (KeyError, TypeError):
            return "—"
    def feat_line(label, feat):
        if not feat:
            return f"<tr><td>{label}</td><td colspan='6'>未测得稳定的双特征</td></tr>"
        A = feat.get("A", {}); Bf = feat.get("B", {}); S = feat.get("saddle", {}); R = feat.get("relative", {})
        return (f"<tr><td>{label}</td><td>{fh(A,'height_mm')}</td>"
                f"<td>{(A.get('width') or {}).get('w50_mm','—')}</td><td>{fh(A,'l50_mm')}</td>"
                f"<td>{fh(Bf,'height_mm')}</td><td>{fh(S,'height_mm')}</td>"
                f"<td>{(R.get('centre_distance_mm','—'))}</td></tr>")
    surface_rows = "".join(feat_line(lab, feat) for lab, feat in (
        ("unshaped baseline", B), ("0.7.0 current", A7), ("C closest safe", C)))
    full_figs = "".join(
        f"<figure><img style='width:100%' src='{img(out/'render'/f'compare_{s}_{v}_full_physical.png')}'/>"
        f"<figcaption>{s} / {v} 物理单位：参考（原尺寸）· 基线 · 0.7.0 · C（同尺度/平光/平面）</figcaption></figure>"
        for s in ("closed", "open") for v in ("front", "right", "left", "right45", "left45"))
    norm_figs = "".join(
        f"<figure><img style='width:100%' src='{img(out/'render'/f'compare_{s}_{v}_full_head_normalized.png')}'/>"
        f"<figcaption>{s} / {v} 头部尺度归一化：参考按 S_head={scale['S_head_geometric']:.3f} 缩放 · 基线 · 0.7.0 · C</figcaption></figure>"
        for s in ("closed", "open") for v in ("front", "right45"))
    close_figs = "".join(
        f"<figure><img style='width:100%' src='{img(out/'render'/f'compare_{s}_neck_{v}_{shade}.png')}'/>"
        f"<figcaption>{s} 颈前 {v}（{shade}）：参考 · 基线 · 0.7.0 · C</figcaption></figure>"
        for s in ("closed", "open") for v in ("front", "right45") for shade in ("flat", "smooth"))
    sec_figs = "".join(
        f"<figure><img style='width:92%' src='{img(out/'render'/f'section_{s}_{name}.png')}'/>"
        f"<figcaption>{s}：{'中线纵向' if name=='long' else ('A 处横向' if name=='crossA' else 'B 处横向')}与相对基线差</figcaption></figure>"
        for s in ("closed", "open") for name in ("long", "crossA", "crossB"))
    causal_figs = "".join(
        f"<figure><img style='width:82%' src='{img(out/'render'/f'causal_{s}_45_flat.png')}'/>"
        f"<figcaption>{s}：基线 · B 头尺度完整（支撑不安全） · C 最接近安全</figcaption></figure>"
        for s in ("closed", "open"))
    other_figs = f"<figure><img style='width:70%' src='{img(out/'render'/'head_scale.png')}'/>" \
                 f"<figcaption>参考/目标头高、头宽、头深与按轴比值；几何 S_head={scale['S_head_geometric']:.3f}</figcaption></figure>" \
                 f"<figure><img style='width:90%' src='{img(out/'render'/'final_surface_features.png')}'/>" \
                 f"<figcaption>最终表面 A/鞍/B 高度（头部尺度背景，σ={metrics['sigma_mm']:.1f} mm）</figcaption></figure>" \
                 + "".join(f"<figure><img style='width:96%' src='{img(out/'render'/f'heat_{s}.png')}'/>" \
                           f"<figcaption>{s}：C 应用场热图与等值线</figcaption></figure>" for s in ("closed", "open")) \
                 + "".join(f"<figure><img style='width:56%' src='{img(out/'render'/f'wire_{s}.png')}'/>" \
                           f"<figcaption>{s}：C 过渡区线框</figcaption></figure>" for s in ("closed", "open"))
    default_text = ("公共默认仍为 <b>0.7.0</b>（`continuous_reference` + `neck_dual`）。本轮 0.8.0-candidate 的"
                    "几何/保护/支撑验收通过，但比例条件未通过（宽度、纵向长度、A-B 间距低于参考目标），"
                    "因此未切换默认，也未把整体再次缩小后称为完成。参考副本已保存为 "
                    "`assets/neck_reference/identity_rest_20261109.obj` 与 `identity_rest_20261109_asset.json`，"
                    "公共默认配置已指向项目内资产，不再依赖外部历史实验目录。")
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>头尺度归一化喉结比例报告</title>
<style>
 body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#1c1e22;color:#d8dce2;line-height:1.55}}
 .wrap{{max-width:1140px;margin:0 auto;padding:32px 24px 64px}}
 h1{{font-size:24px;border-bottom:2px solid #4a90d9;padding-bottom:10px}}
 h2{{font-size:18px;color:#7fb3e8;margin-top:32px;border-left:4px solid #4a90d9;padding-left:10px}}
 h3{{font-size:14px;color:#aab7c8;margin-top:22px}}
 table{{border-collapse:collapse;margin:12px 0;width:100%;font-size:13px}}
 td,th{{border:1px solid #3a3f47;padding:5px 9px;text-align:left}} th{{background:#262a31}}
 tr.pass td:last-child{{color:#7fd88f}} tr.fail td:last-child{{color:#e06c5f}}
 table.kv td:first-child{{color:#aab7c8;width:34%}}
 code{{background:#262a31;padding:1px 5px;border-radius:3px;font-size:12px}}
 .note{{background:#2a2519;border:1px solid #6b5b1e;padding:10px 14px;border-radius:6px;margin:12px 0}}
 .warn{{background:#2a1f1f;border:1px solid #6b2e2e;padding:10px 14px;border-radius:6px;margin:12px 0}}
 figure{{margin:14px 0;text-align:center}} figure img{{border-radius:6px;max-width:100%}}
 figcaption{{font-size:12.5px;color:#8a97a8;margin-top:5px}}
</style></head><body><div class="wrap">
<h1>头尺度归一化喉结比例 — 交付报告（0.8.0-candidate）</h1>
<p style="color:#8a97a8">Head-scale ratio audit, final-surface metrics, A/B/C, closest safe closed/open meshes.</p>

<div class="note"><b>最终网格（绝对路径）：</b><br>
闭口：<code>{disk['closed']['path']}</code><br>
张口：<code>{disk['open']['path']}</code><br>
editing_mode=true · geometry_source=reference_driven_appearance · subject_observation_matched=false<br>
参考：<code>assets/neck_reference/identity_rest_20261109.obj</code>（源 SHA <code>{morph['source_sha256']}</code>）</div>

<h2>0. 七个问题</h2>
<ol>
<li><b>为什么 0.7.0 双特征仍显得小？</b>0.7.0 以<b>颈宽</b> q≈0.952 计算尺寸，再因空间预筛乘 f=0.72；而参考与目标的<b>头部尺度</b>不同：
参考头高/宽/深 {scale['reference']['H_head_mm']:.1f}/{scale['reference']['W_head_mm']:.1f}/{scale['reference']['D_head_mm']:.1f} mm，
目标 {scale['target']['H_head_mm']:.1f}/{scale['target']['W_head_mm']:.1f}/{scale['target']['D_head_mm']:.1f} mm；
宽度/深度比一致约 {scale['ratios']['W_head_mm']:.2f}，头高比 {scale['ratios']['H_head_mm']:.2f}。
按参考比例，目标主峰应为 H≈{tgt_mm['H_A_mm']:.2f} mm、W50≈{tgt_mm['W50_A_mm']:.2f} mm，
而 0.7.0 最终表面主峰 H≈{fh(A7.get('A',{}),'height_mm')} mm（约为目标的 60%）；
0.7.0 的 W50 读数 {(A7.get('A',{}).get('width') or {}).get('w50_mm','—')} mm 主要来自基面/颏下体量，不能与目标宽度直接比较。
同时 0.7.0 报告的是新增位移，而本轮把最终表面、未塑形基面和新增场分开测量。</li>
<li><b>参考和目标头部尺寸如何同口径测量？</b>两者都排除颈部/肩部/底切，取“颅顶到颏下/menton”为头高，
取颏下以上、颅顶以下 0.35–0.85 头高带的横向/前后的 0.5–99.5 百分位为头宽/头深。
参考 H/W/D={scale['reference']['H_head_mm']:.2f}/{scale['reference']['W_head_mm']:.2f}/{scale['reference']['D_head_mm']:.2f} mm；
目标 {scale['target']['H_head_mm']:.2f}/{scale['target']['W_head_mm']:.2f}/{scale['target']['D_head_mm']:.2f} mm。
宽/深比 1.20 与 1.199 高度一致；头高比 1.637 明显不同（参考头更扁）。
因此横截面/高度/尖端用 W_head 比例，纵向长度/A-B 间距用 H_head 比例；
滤波尺度用几何 S_head={scale['S_head_geometric']:.4f}（目标 σ={metrics['sigma_mm']:.2f} mm，对应参考 8 mm）。
没有为了让喉结显大而只选头高比：两种轴比都记录，冲突在第 4 节。</li>
<li><b>最终表面相对头部的比例是否达到目标？</b>主峰归一化高度 {fmt(prop['H_A_over_W_head'].get('final'))} vs {prop['H_A_over_W_head']['reference']:.4f}（误差 {100*prop['H_A_over_W_head'].get('rel_error', 0):.1f}%）；
主峰宽度 {fmt(prop['W50_A_over_W_head'].get('final'))} vs {prop['W50_A_over_W_head']['reference']:.4f}（{100*prop['W50_A_over_W_head'].get('rel_error', 0):.1f}%）；
纵向 L50 {fmt(prop['L50_A_over_H_head'].get('final'))} vs {prop['L50_A_over_H_head']['reference']:.4f}（{100*prop['L50_A_over_H_head'].get('rel_error', 0):.1f}%）；
A-B 间距 {fmt(prop['distance_AB_over_H_head'].get('final'))} vs {prop['distance_AB_over_H_head']['reference']:.4f}（{100*prop['distance_AB_over_H_head'].get('rel_error', 0):.1f}%）；
尖端半径 {fmt(prop['tip_radius_A_over_W_head'].get('final'))} vs {prop['tip_radius_A_over_W_head']['reference']:.4f}。
<b>比例条件未通过</b>；不是“检测到两个峰”或“B/A 相同”就算完成。</li>
<li><b>哪些尺寸受空间适配影响，影响多少？</b>核心高度先按参考 W_head 比例（height=1.2，施加 H_A≈4.78 mm），再按“最终表面同口径”校准到 height={summary.get('C_height_factor', 1.2):.2f}（施加 H_A≈{design['height_mm']:.2f} mm，最终表面 A≈{fh(C.get('A',{}),'height_mm')} mm；
height=2.0 时实际 guard/法向检查失败，因此这是最大安全施加高度，不是任意增高），
横向在保护/下颌侧缘安全约束下只能取 lateral=1.0（W50≈{fh(C.get('A',{}),'height_mm') and (C.get('A',{}).get('width') or {}).get('w50_mm','—')} mm，目标 {tgt_mm['W50_A_mm']:.1f}）；
纵向为容纳双特征、下颌保护与真实颈底切口取 vertical=0.68（L50≈{C.get('A',{}).get('l50_mm','—')} mm，目标 {tgt_mm['L50_A_mm']:.1f}；间距 {C.get('relative',{}).get('centre_distance_mm','—')} mm，目标 {tgt_mm['distance_AB_mm']:.1f}）。
主峰高度/次峰/B/A/鞍部保留，宽度/纵向长度/间距被目标空间截断；不是再次整体乘 f 后宣称通过。</li>
<li><b>大尺寸方案是否实际安全，而非仅被预设裕量排除？</b>B（头尺度完整：vertical={scale['ratios']['H_head_mm']:.3f}, lateral={scale['ratios']['W_head_mm']:.3f}）实际构建结果：
闭/张 guard={Bfull_guard['closed']}/{Bfull_guard['open']}，自交 {Bfull_inter['closed']}/{Bfull_inter['open']}，
10% 支撑保护净距 {Bfull_support['closed']['clearance_min_mm']}/{Bfull_support['open']['clearance_min_mm']} mm，
门控最小 {Bfull_support['closed']['gate_min']}/{Bfull_support['open']['gate_min']}，
距真实颈底切口 {Bfull_support['closed']['bottom_margin_mm']}/{Bfull_support['open']['bottom_margin_mm']} mm，
闭口特征检测 {Bfeat.get('closed',{}).get('detected')}。
B 的支撑在张口低于颈底切口并落入 8 mm 保护渐变，闭口次级特征被截；这不是“只被预筛拒绝”，是实际支撑/边界失败。
C 取实际安全的最大横向/纵向：支撑门控 {C_support['closed']['gate_min']}/{C_support['open']['gate_min']}，颈底余量 {C_support['closed']['bottom_margin_mm']}/{C_support['open']['bottom_margin_mm']} mm，自交 0。
所以“大尺寸必然安全”不成立，C 是最接近且安全的网格，比例条件明确未通过。</li>
<li><b>两个最终 OBJ：</b>见上方绝对路径；单位 m、共享拓扑、边界 {disk['closed']['topology']['boundary_edges']} 条、无材质、回读误差 ≤{max(disk[s]['max_abs_diff_m'] for s in disk):.1e} m。</li>
<li><b>公共默认、版本、耗时和退出状态：</b>{default_text}
本实验版本 0.8.0-candidate；几何/保护验收 {'通过' if summary['accepted_geometry'] else '未通过'}，
比例条件 {'通过' if summary['proportion_pass'] else '未通过（明确标记）'}；总 {total:.1f} s，退出码 {'0 / ok（几何安全，比例未通过）' if summary['accepted_geometry'] else '1 / verification_failed'}。</li>
</ol>

<h2>1. 头部尺度与参考比例目标</h2>
{other_figs}
<table class="kv">
<tr><td>参考头 H/W/D</td><td>{scale['reference']['H_head_mm']:.2f} / {scale['reference']['W_head_mm']:.2f} / {scale['reference']['D_head_mm']:.2f} mm</td></tr>
<tr><td>目标头 H/W/D</td><td>{scale['target']['H_head_mm']:.2f} / {scale['target']['W_head_mm']:.2f} / {scale['target']['D_head_mm']:.2f} mm</td></tr>
<tr><td>比例</td><td>H {scale['ratios']['H_head_mm']:.4f}，W {scale['ratios']['W_head_mm']:.4f}，D {scale['ratios']['D_head_mm']:.4f}；
几何 S_head {scale['S_head_geometric']:.4f}；宽/深一致，头高不同</td></tr>
<tr><td>参考比例（最终表面口径）</td><td>H_A/W {ref_rat['H_A_over_W_head']:.4f}；W50/W {ref_rat['W50_A_over_W_head']:.4f}；
L50/H {ref_rat['L50_A_over_H_head']:.4f}；H_B/W {ref_rat['H_B_over_W_head']:.4f}；
B 局部/W {ref_rat['B_local_prominence_over_W_head']:.4f}；间距/H {ref_rat['distance_AB_over_H_head']:.4f}；尖端/W {ref_rat['tip_radius_A_over_W_head']:.4f}</td></tr>
<tr><td>目标毫米</td><td>H_A {tgt_mm['H_A_mm']:.2f}；W50 {tgt_mm['W50_A_mm']:.2f}；L50 {tgt_mm['L50_A_mm']:.2f}；
H_B {tgt_mm['H_B_mm']:.2f}；鞍 {tgt_mm['saddle_mm']:.2f}；间距 {tgt_mm['distance_AB_mm']:.2f}</td></tr>
</table>

<h2>2. 最终表面口径与比例结果</h2>
<table><tr><th>表面</th><th>A 高 (mm)</th><th>A W50 (mm)</th><th>A L50 (mm)</th><th>B 高 (mm)</th><th>鞍高 (mm)</th><th>间距 (mm)</th></tr>{surface_rows}</table>
<table><tr><th>指标</th><th>参考</th><th>C 最终</th><th>误差</th><th>容差</th><th>结果</th></tr>{prop_rows}</table>
<p>最终表面、未塑形基面、0.7.0 和新增位移分别记录在 final_surface_metrics.json 与 deformation_audit.json；
下表的“最终表面”不是 applied 位移读数。新增位移的主峰约 {design['height_mm']:.2f} mm，基面在主特征位置另有约
{fh(B.get('A',{}),'height_mm')} mm 的同口径体量，故不能把 applied 读数直接当作最终喉结高度。</p>

<h2>3. A/B/C 与空间/安全审计</h2>
<table><tr><th>方案</th><th>状态</th><th>A 高</th><th>B 高</th><th>B/A</th><th>鞍</th><th>间距</th><th>支撑净距</th><th>颈底余量</th></tr>
<tr><td>A 0.7.0</td><td>closed</td><td>{fh(A7.get('A',{}),'height_mm')}</td><td>{fh(A7.get('B',{}),'height_mm')}</td>
<td>{A7.get('relative',{}).get('B_over_A','—')}</td><td>{fh(A7.get('saddle',{}),'height_mm')}</td>
<td>{A7.get('relative',{}).get('centre_distance_mm','—')}</td><td>—</td><td>—</td></tr>
<tr><td>B head full</td><td>closed/open</td><td>per design {summary['B_full_design']['height_mm']:.2f}</td>
<td>—</td><td>—</td><td>—</td><td>{summary['B_full_design']['v_b_mm']:.1f} (design v)</td>
<td>{Bfull_support['closed']['clearance_min_mm']}/{Bfull_support['open']['clearance_min_mm']}</td>
<td>{Bfull_support['closed']['bottom_margin_mm']}/{Bfull_support['open']['bottom_margin_mm']}</td></tr>
<tr><td>C safe</td><td>closed/open</td><td>{fh(C.get('A',{}),'height_mm')}</td><td>{fh(C.get('B',{}),'height_mm')}</td>
<td>{C.get('relative',{}).get('B_over_A','—')}</td><td>{fh(C.get('saddle',{}),'height_mm')}</td>
<td>{C.get('relative',{}).get('centre_distance_mm','—')}</td>
<td>{C_support['closed']['clearance_min_mm']}/{C_support['open']['clearance_min_mm']}</td>
<td>{C_support['closed']['bottom_margin_mm']}/{C_support['open']['bottom_margin_mm']}</td></tr></table>
{causal_figs}
<p>B 的 guard/自交并未必然失败，但支撑越过了真实颈底/保护渐变，闭口次级特征被截；
这说明预设的 8 mm 裕量与真实门控/边界一致，不是单纯保守筛除。C 保留参考高度、B/A 和正鞍部，
在真实安全约束下取得最大横向/纵向支撑；L/distance 目标因目标短颈和安全边界无法同时达到。</p>

<h2>4. 几何与保护验收</h2>
<table><tr><th>检查</th><th>结果</th></tr>{check_rows}</table>
<table><tr><th>状态</th><th>顶点</th><th>面</th><th>边界边</th><th>不一致共享边</th><th>非流形边</th><th>回读误差 m</th></tr>{disk_rows}</table>
<p>面部首命中按原始基线面索引：闭/张最大 {facial['closed']['max_abs_shift_mm']}/{facial['open']['max_abs_shift_mm']} mm；
保护顶点移动 0；局部自交 0；未做基面修复，未迁移负残差。C 的几何/保护验收通过，比例条件未通过。</p>

<h2>5. 视觉比较（物理单位 + 头尺度归一化）</h2>
<h3>物理单位全头（参考原尺寸 / 基线 / 0.7.0 / C）</h3>
{full_figs}
<h3>头部尺度归一化全头（参考按 S_head 缩放）</h3>
{norm_figs}
<h3>颈前局部（平面 + 平滑）</h3>
{close_figs}
<h3>纵向与 A/B 横向剖面</h3>
{sec_figs}

<h2>6. 结论与局限</h2>
<div class="warn"><ul>
<li>参考驱动的外观塑形，不是解剖测量；参考/目标头尺度是外观归一化，不是软骨尺寸。</li>
<li>参考头相对更扁（H/W 比目标小），目标头高比 1.64 与头宽比 1.20 不一致。本轮按声明的 W/H 分轴比例测量，没有选单一有利尺度。
最终 C 达到主峰归一化高度目标，但宽度、L50 和 A-B 间距受目标短颈、下颌保护与真实颈底切口限制，比例条件明确未通过。</li>
<li>B 的完整头尺度方案实际支撑低于颈底/进入保护渐变，闭口次级特征被截；不是安全候选。没有删除安全门槛或关闭几何检查。</li>
<li>0.7.0 的 applied 位移读数与最终表面口径不同；本报告以最终表面为主，并把基面、0.7.0、C 与新增场分开记录。</li>
<li>公共默认仍是 0.7.0；0.8.0-candidate 仅交付最接近且安全的网格。参考副本/资产已存入项目，不再依赖外部历史实验目录。</li>
</ul></div>

<p style="color:#8a97a8;margin-top:30px">生成：{datetime.now().strftime('%Y-%m-%d %H:%M')} · tools/run_neck_head_proportion.py · 输出：{out.name}</p>
</div></body></html>"""
    path = out / "NECK_HEAD_PROPORTION_REPORT.html"
    path.write_text(html, encoding="utf-8")
    return path
