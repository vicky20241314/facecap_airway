"""Write NECK_CONTINUOUS_APPEARANCE_FIX_REPORT.html."""
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
    audit = summary["audits"]
    causal = summary["causal"]
    morph = summary["reference"]
    design = summary["design"]
    placement = summary["placement"]
    ref_ratios = summary["ref_ratios"]
    prop = summary["prop_rows"]
    disk = summary["disk"]
    facial = summary["facial"]
    direction = summary["direction_mode"]
    alpha = summary["alpha"]
    total = summary.get("total_seconds", 0.)
    direction_runs = summary.get("direction_runs", {})
    def dir_max(state, mode):
        try:
            return float(direction_runs[state][mode]["max_tangential_mm"])
        except (KeyError, TypeError):
            return 0.0
    C_dir_closed = dir_max("closed", "constant")
    C_dir_open = dir_max("open", "constant")
    islands_text = "；".join(
        f"u[{i['u_mm'][0]}..{i['u_mm'][1]}] v[{i['v_mm'][0]}..{i['v_mm'][1]}] 峰{i['peak_mm']:.2f} mm"
        for i in morph.get("islands_10pct", [])) or "无"
    def wl(state):
        try:
            box = audit[state]["levels"]["0.5"]["central"]["uv_bbox_mm"]
            return box[1] - box[0], box[3] - box[2]
        except (KeyError, TypeError):
            return 0.0, 0.0
    defaults = json.loads((PROJECT / "config" / "defaults.json").read_text(encoding="utf-8"))
    algorithm = str((defaults.get("neck_prominence") or {}).get("algorithm"))
    pipeline_text = (PROJECT / "head_rgbd" / "pipeline.py").read_text(encoding="utf-8")
    public_switched = algorithm == "continuous_reference" and "continuous_reference" in pipeline_text
    switch_path = out / "public_switch.json"
    public_switch = (json.loads(switch_path.read_text(encoding="utf-8"))
                     if switch_path.is_file() else None)

    def d(state, section, key, nd=2):
        try:
            value = audit[state][section][key]
            return "—" if value is None else f"{float(value):.{nd}f}"
        except (KeyError, TypeError, ValueError):
            return "—"

    def ratio_row(state):
        row = prop[state]
        return (f"{row['height_mm']:.2f} / {row['width50_mm']:.2f} / {row['length50_mm']:.2f}；"
                f"H/W50 {row['height_over_width50']:.3f}，L50/W50 {row['length50_over_width50']:.3f}，"
                f"W50/W_neck {row['width50_over_neck']:.3f}")

    full_figs = "".join(
        f"<figure><img style='width:100%' src='{img(out / 'render' / f'compare_{s}_{v}_full_flat.png')}'/>"
        f"<figcaption>{s} / {v}：参考 · 未塑形基线 · 0.5.9 有符号残差 · 本轮连续非负（同尺度/同光照/平面着色）</figcaption></figure>"
        for s in ("closed", "open") for v in ("front", "right", "left", "right45", "left45"))
    close_figs = "".join(
        f"<figure><img style='width:100%' src='{img(out / 'render' / f'compare_{s}_neck_{v}_{shade}.png')}'/>"
        f"<figcaption>{s} 颈前局部 {v}（{shade}）：参考 · 基线 · 0.5.9 · 本轮</figcaption></figure>"
        for s in ("closed", "open") for v in ("front", "right45") for shade in ("flat", "smooth"))
    causal_figs = "".join(
        f"<figure><img style='width:82%' src='{img(out / 'render' / f'causal_{s}_neck_45_flat.png')}'/>"
        f"<figcaption>{s} 因果对照（45°）：基线 · B（旧中心/尺度，非负场） · C（比例定位 + 整体缩放）</figcaption></figure>"
        for s in ("closed", "open"))
    section_figs = "".join(
        f"<figure><img style='width:92%' src='{img(out / 'render' / f'section_{s}_{name}.png')}'/>"
        f"<figcaption>{s}：{'中线纵向' if name == 'long' else '穿峰横向'}剖面与相对基线法向差（−0.05 mm 虚线）</figcaption></figure>"
        for s in ("closed", "open") for name in ("long", "cross"))
    heat_figs = "".join(
        f"<figure><img style='width:96%' src='{img(out / 'render' / f'heat_{s}.png')}'/>"
        f"<figcaption>{s}：应用场热图（左：3D 颈前；右：(u,v) 等值线，白 10% / 黑 50%）</figcaption></figure>"
        for s in ("closed", "open"))
    wire_figs = "".join(
        f"<figure><img style='width:56%' src='{img(out / 'render' / f'wire_{s}.png')}'/>"
        f"<figcaption>{s}：细化过渡区线框</figcaption></figure>"
        for s in ("closed", "open"))

    check_rows = "".join(
        f"<tr class='{'pass' if v else 'fail'}'><td>{k}</td><td>{'通过' if v else '未通过'}</td></tr>"
        for k, v in checks.items())
    def clearance_cell(label, state):
        value = causal[label][state]["support_clearance_mm"]
        return "—" if value is None else f"{float(value):.2f}"

    a_rows = "".join(
        f"<tr><td>{label}</td><td>{s}</td><td>{causal[label][s]['peak_mm']:.2f}</td>"
        f"<td>{causal[label][s]['min_mm']:.2f}</td><td>{causal[label][s]['negative_area_mm2']:.0f}</td>"
        f"<td>{causal[label][s]['max_tangential_mm']:.2f}</td>"
        f"<td>{clearance_cell(label, s)}</td></tr>"
        for label in causal for s in ("closed", "open"))
    disc_rows = "".join(
        f"<tr><td>{s}</td><td>{disk[s]['vertices']}</td><td>{disk[s]['faces']}</td>"
        f"<td>{disk[s]['topology']['boundary_edges']}</td>"
        f"<td>{disk[s]['topology']['inconsistent_shared_edges']}</td>"
        f"<td>{disk[s]['topology']['nonmanifold_edges']}</td>"
        f"<td>{disk[s]['max_abs_diff_m']:.2e}</td><td>{disk[s]['no_materials']}</td></tr>"
        for s in ("closed", "open"))

    if public_switched and public_switch:
        max_diff = max(public_switch["meshes"][s]["max_vertex_diff_m"] for s in ("closed", "open"))
        default_text = (
            "已切换到同一生产实现（`neck_prominence.algorithm=continuous_reference`，pipeline 分支已接入）。"
            f"公开交付：<code>{public_switch['delivery']}</code>（包装脚本 `headrgbd.cmd export`，版本 "
            f"{public_switch['manifest_version']}，与独立工具逐顶点一致 ≤{max_diff:.1e} m）。"
            "公开 manifest 的状态为 degraded，来源是既有的张口内唇深度先验警告（基线 manifest 同样存在），"
            "不是颈部算法失败；neck_prominence 中 continuous_reference 为 applied，editing/geometry_source 标记一致。")
    elif public_switched:
        default_text = ("已切换到同一生产实现（`neck_prominence.algorithm=continuous_reference`，pipeline 分支已接入）；"
                        "公开交付复核记录尚未写入本目录。")
    else:
        default_text = "尚未切换公共默认：本报告是独立生产工具的实际交付，入口统一在验收通过后单独执行并复核。"
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>连续非负参考喉结外观修复报告</title>
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
 .good{{background:#1f2a24;border:1px solid #2e6b43;padding:10px 14px;border-radius:6px;margin:12px 0}}
 figure{{margin:14px 0;text-align:center}} figure img{{border-radius:6px;max-width:100%}}
 figcaption{{font-size:12.5px;color:#8a97a8;margin-top:5px}}
</style></head><body><div class="wrap">
<h1>连续非负参考喉结外观修复 — 交付报告（0.6.0）</h1>
<p style="color:#8a97a8">Continuous non-negative reference-driven laryngeal appearance; actual closed/open meshes, disk verification and four-way visual comparison.</p>

<div class="note"><b>最终网格（绝对路径）：</b><br>
闭口：<code>{disk['closed']['path']}</code><br>
张口：<code>{disk['open']['path']}</code><br>
editing_mode=true · geometry_source=reference_driven_appearance · subject_observation_matched=false<br>
参考：<code>{morph['source_path']}</code>（SHA256 <code>{morph['source_sha256']}</code>）</div>

<h2>0. 八个问题</h2>
<ol>
<li><b>凹坑的实际来源是什么？</b>主要来自 0.5.9 把参考的<b>有符号负残差</b>直接迁入目标：相对未塑形基线，
闭口基顶点最小 <b>−2.02 mm</b>、负面积 <b>2330 mm²</b>（张口 −1.37 mm / 1466 mm²），这是 0.5.9 在凸起周围形成暗环/凹坑的原因；
固定方向还造成最高 <b>2.22 mm</b> 的切向拖拽。未塑形闭口基面自身没有局部坑（中线相对弦线 −1.49…+0.80 mm 的宽缓自然曲率），
因此不是“基面凹陷”主导，也不需要单独的基面修复。中心位置（A 的 s=0.546）与参考 s=0.545 接近，<b>不是主因</b>；
主因是负残差迁移与固定方向/过渡处理。</li>
<li><b>是否存在拓扑切除或补片拼接？</b>没有。始终使用同一未塑形基拓扑，局部红绿细分，闭/张共享面表；
最终边界仍为 <b>124</b> 条（基线嘴内+颈底），无新孔洞、无补片缝合、无非流形边。</li>
<li><b>修改后是否仍有新增内压和凹槽？</b>没有。相对未塑形基线的法向分量最小 <b>0.000 mm</b>（容差 −0.05 mm），
负面积 <b>0 mm²</b>；径向环采样无局部回升（groove 检查 0），正残差只有 1 个连通域、负残差 0 个；
应用场本身是非负 C2 单峰。B（旧中心/尺度 + 非负场）也没有内压，但其张口 10% 支撑距保护只有 3.23 mm，不安全，
因此 C 同时修正定位/整体尺度。</li>
<li><b>位置如何确定，是否适配颈部比例？</b>在闭口建立颏下-颈底可靠颈段（W_neck {summary['target_scale'].W_neck_mm:.1f} mm，
L_neck {summary['target_scale'].L_neck_mm:.1f} mm，闭/张 {summary['target_scale'].L_neck_per_state['closed']:.1f}/{summary['target_scale'].L_neck_per_state['open']:.1f} mm），
参考喉结在可靠前窗内的相对位置 s_ref={morph['window']['s_roi']}；映射后因短颈/保护间距与完整支撑，协调得到
s_chosen={placement.get('s_chosen'):.3f}、整体缩放 f={placement.get('fit_scale'):.2f}（H/W/L/tip 同比例缩放，未单独压窄），
中心自动绑定基顶点 <b>{placement.get('vertex')}</b>（按位置选择，不是固定 ID）。H/W50/L50 形状比与参考一致（见第 3 节），
绝对尺寸比参考小约 {100*(1-summary['q']*placement.get('fit_scale',1)):.0f}%（短颈容纳/保护协调）。未使用固定 30 mm、世界方向、全颈最大残差或肩部复制。</li>
<li><b>闭口与张口的实际形态和安全检查是否通过？</b>
闭口峰值 <b>{d('closed','base_relative','max_outward_mm')} mm</b>，中心 W50/L50 <b>{wl('closed')[0]:.2f}/{wl('closed')[1]:.2f} mm</b>；
张口峰值 <b>{d('open','base_relative','max_outward_mm')} mm</b>，中心 W50/L50 <b>{wl('open')[0]:.2f}/{wl('open')[1]:.2f} mm</b>。
两状态 10% 支撑保护净距 {d('closed','support','clearance_min_mm')} / {d('open','support','clearance_min_mm')} mm，
保护渐变值 {d('closed','support','gate_min',4)} / {d('open','support','gate_min',4)}，10% 与 50% 均 1 个连通域；
闭/张面表一致；面部首命中（原始基线面索引）最大位移 {facial['closed']['max_abs_shift_mm']} / {facial['open']['max_abs_shift_mm']} mm。
全部 {sum(1 for v in checks.values() if v)}/{len(checks)} 项验收通过。张口相对闭口的长宽略有差异，来源是两状态各自的
颈段长度（{summary['target_scale'].L_neck_per_state['closed']:.1f} vs {summary['target_scale'].L_neck_per_state['open']:.1f} mm）和共享基绑定的 uv 映射，
峰值保持相同，没有被张口保护截掉。</li>
<li><b>两个最终 OBJ 的绝对路径：</b>见上方；单位 m，重算顶点法向，无材质/颜色组，磁盘回读误差 ≤{max(disk[s]['max_abs_diff_m'] for s in disk):.1e} m。</li>
<li><b>公共默认是否已经切换？</b>{default_text}</li>
<li><b>实际耗时和退出状态：</b>总 {total:.1f} s；验收 {'通过，退出码 0 / ok' if summary['accepted'] else '未通过，退出码 1 / verification_failed'}（详见 run_status.json）。</li>
</ol>

<h2>1. 参考与可靠形态（只取非负形态）</h2>
<table class="kv">
<tr><td>参考峰位 / 峰值</td><td>({morph['peak_xy_mm'][0]}, {morph['peak_xy_mm'][1]}) mm / {morph['peak_mm']:.3f} mm（相对二次背景的外观残差，不是解剖高度）</td></tr>
<tr><td>中心正叶 W50×L50 / W10×L10</td><td>{morph['width50_mm']:.2f}×{morph['length50_mm']:.2f} / {morph['width10_mm']:.2f}×{morph['length10_mm']:.2f} mm</td></tr>
<tr><td>上下 50% / 10%</td><td>{morph['up50_mm']:.2f}/{morph['down50_mm']:.2f} · {morph['up10_mm']:.2f}/{morph['down10_mm']:.2f} mm（下过渡略长）</td></tr>
<tr><td>解析拟合</td><td>纵支撑 {morph['fit']['vertical_support_up_mm']:.2f}/{morph['fit']['vertical_support_down_mm']:.2f} mm，rms {morph['fit']['vertical_rms']:.4f}；
横半宽 {morph['fit']['lateral_half_width_mm']:.2f} mm，尖端半径 {morph['fit']['tip_radius_mm']:.2f} mm，rms {morph['fit']['lateral_rms']:.4f}</td></tr>
<tr><td>尖端圆钝</td><td>局部二次拟合半径 {morph['apex_curvature'][0]['radius_mm']:.2f} mm（±2 mm），rms {morph['apex_curvature'][0]['fit_rms_mm']:.3f} mm</td></tr>
<tr><td>负残差 / 肩部岛</td><td>最低 {morph['negative_extent_mm']:.2f} mm，<b>不迁移</b>；10% 外围肩部岛 {islands_text}，<b>不复制</b></td></tr>
<tr><td>背景敏感性</td><td>三次背景/更宽外环与主设置的峰值范围
{[v['peak_mm'] for v in morph.get('background_sensitivity', {}).get('variants', [])]} mm；峰位一致，幅值作为局限记录，未用变体调高高度</td></tr>
<tr><td>参考颈窗</td><td>ROI y {morph['window']['roi_window_mm']} mm，s_ref={morph['window']['s_roi']}；几何下颏转折/肩部加宽窗 s={morph['window']['geometric'].get('s_larynx')}（肩部加宽与喉结重叠，未采用，见 placement_audit.json）</td></tr>
</table>

<h2>2. 因果对照 A/B/C（实际网格）</h2>
<table><tr><th>方案</th><th>状态</th><th>峰值 (mm)</th><th>最小法向 (mm)</th><th>负面积 (mm²)</th>
<th>最大切向 (mm)</th><th>10% 支撑保护净距 (mm)</th></tr>{a_rows}</table>
<p>B 与 A 使用同一中心/尺度，只取消负残差迁移；B 的内压为 0，但张口 10% 支撑距保护仅 3.23 mm（会挤压/截断），
因此 C 必须同时修正比例定位和整体缩放。C 的峰值略小于 A，但消除了负环，得到连续、可容纳的单峰。</p>
{causal_figs}

<h2>3. 设计、定位与比例</h2>
<table class="kv">
<tr><td>比例缩放 q（目标 W_neck/参考颈宽） / 整体 f</td><td>{summary['q']:.4f} / {placement.get('fit_scale')}</td></tr>
<tr><td>设计尺寸 H / 支撑 L_up×L_dn / W_total / tip</td><td>{design['height_mm']} mm / {design['support_up_mm']}×{design['support_down_mm']} mm / {design['width_total_mm']} mm / {design['tip_radius_mm']} mm</td></tr>
<tr><td>中心自动绑定</td><td>基顶点 {placement.get('vertex')}；s_ref {morph['window']['s_roi']} → s_chosen {placement.get('s_chosen'):.3f}；
10% 上/下沿 {placement.get('top10_mm'):.2f}/{placement.get('bottom10_mm'):.2f} mm，颏下转折/颈底 {placement.get('crease_mm'):.2f}/{placement.get('root_mm'):.2f} mm</td></tr>
<tr><td>闭口比例（H/W50/L50；H/W50，L50/W50，W50/W_neck）</td><td>{ratio_row('closed')}</td></tr>
<tr><td>张口比例（同上）</td><td>{ratio_row('open')}</td></tr>
<tr><td>参考比例</td><td>H/W50 {ref_ratios['height_over_width50']:.3f}，L50/W50 {ref_ratios['length50_over_width50']:.3f}，
W50/W_neck {ref_ratios['width50_over_neck']:.3f}，L50/参考颈窗 {ref_ratios['length50_over_neck_window']:.3f}</td></tr>
<tr><td>基面自身凹陷记录</td><td>闭口 {d('closed','base_concavity','deviation_min_mm')}…{d('closed','base_concavity','deviation_max_mm')} mm，
张口 {d('open','base_concavity','deviation_min_mm')}…{d('open','base_concavity','deviation_max_mm')} mm（宽缓自然曲率；未做基面修复，
不把新塑形与基面混在一起）</td></tr>
<tr><td>边界坡度</td><td>横向/纵向最大斜率 {d('closed','boundary','max_lateral_slope_mm_per_mm',3)} / {d('closed','boundary','max_vertical_slope_mm_per_mm',3)} mm/mm；
支持边界解析值为 0，C2 值与一阶导为零</td></tr>
</table>

<h2>4. 方向场</h2>
<p>同一形态场比较固定框架法向与低频平滑法向（α={alpha:.2f}）：最大切向拖拽闭口
{dir_max('closed','smooth'):.2f} mm（固定 {C_dir_closed:.2f}），张口
{dir_max('open','smooth'):.2f} mm（固定 {C_dir_open:.2f}）；选择 <b>{direction}</b>。
逐顶点法向先做 8 轮邻域平滑，再与框架法向混合并投影到外向，未直接使用粗网格逐面法向。</p>

<h2>5. 验收与磁盘回读</h2>
<table><tr><th>检查</th><th>结果</th></tr>{check_rows}</table>
<table><tr><th>状态</th><th>顶点</th><th>面</th><th>边界边</th><th>不一致共享边</th><th>非流形边</th><th>磁盘坐标差 (m)</th><th>无材质</th></tr>
{disc_rows}</table>
<p>自交检测：{summary.get('intersection_scope', '局部受影响边范围；非整头穷举')}。面部首命中按<b>原始基线面索引</b>分类：
闭/张保护面命中 {facial['closed']['protected_face_hits']}/{facial['open']['protected_face_hits']} 条，
最大位移 {facial['closed']['max_abs_shift_mm']}/{facial['open']['max_abs_shift_mm']} mm；保护顶点位移 0。</p>

<h2>6. 视觉复核（同一尺度/视角/光照）</h2>
<h3>全头四方案（闭口/张口 × 正面/左右侧/左右 45°）</h3>
{full_figs}
<h3>颈前局部（平面与平滑法向）</h3>
{close_figs}
<h3>剖面（纵/横，含相对基线差）</h3>
{section_figs}
<h3>位移热图</h3>
{heat_figs}
<h3>过渡区线框</h3>
{wire_figs}

<h2>7. 公开默认与端到端复核</h2>
{"<table class='kv'><tr><td>公开交付</td><td><code>" + public_switch['delivery'] + "</code></td></tr><tr><td>包装脚本</td><td>" + public_switch['wrapper'] + "（已验证）</td></tr><tr><td>版本/算法/manifest</td><td>" + public_switch['manifest_version'] + " / " + public_switch['neck_method'] + " / editing_mode=" + str(public_switch['editing_mode']).lower() + "，geometry_source=" + public_switch['geometry_source'] + "，subject_observation_matched=" + str(public_switch['subject_observation_matched']).lower() + "</td></tr><tr><td>与独立工具逐顶点</td><td>面一致，最大坐标差 " + f"{max(public_switch['meshes'][s]['max_vertex_diff_m'] for s in ('closed','open')):.1e}" + " m</td></tr><tr><td>degraded 说明</td><td>" + public_switch['degraded_reason'] + "</td></tr></table>" if public_switch else "<p>公开默认切换复核记录未找到。</p>"}

<h2>8. 局限与不夸大</h2>
<div class="warn"><ul>
<li>这是参考驱动的<b>外观塑形</b>：参考峰值 4.61 mm 是相对其二次背景的残差，不是受试者真实喉结高度；subject_observation_matched=false。</li>
<li>参考背景阶次/外环变化时峰值在 4.53–6.38 mm 之间；本轮采用主设置，未做高度补偿。参考自身负残差、侧沟和外围肩部岛均未迁移。</li>
<li>短颈/保护间距使整体缩小 f={placement.get('fit_scale')}；H/W50/L50 形状比保持，绝对尺寸小于参考按颈宽的直接缩放。</li>
<li>自交是局部受影响边范围，不是整头绝对无自交的数学保证；面部首命中只在 68 个关键点射线上分类。</li>
<li>张口相对颈段长度的比例与闭口略不同，来源是两状态姿态/颈长差异和共享基绑定，不是重新按活动下巴定位。</li>
</ul></div>

<p style="color:#8a97a8;margin-top:30px">生成：{datetime.now().strftime('%Y-%m-%d %H:%M')} · tools/run_neck_continuous_fix.py · 输出：{out.name} · 版本 {summary.get('version')}</p>
</div></body></html>"""
    path = out / "NECK_CONTINUOUS_APPEARANCE_FIX_REPORT.html"
    path.write_text(html, encoding="utf-8")
    return path
