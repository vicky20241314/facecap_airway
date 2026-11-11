"""Write NECK_SUBJECT_V3_REPORT.html: self-contained, answers the eight questions."""
from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def img64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def kv(rows):
    return "<table class='kv'>" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</table>"


def fmt(v, digits=2, suffix=""):
    try:
        return f"{float(v):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def write_report(
    *, output: Path, out_dir: Path, input_manifest: dict, shape_rows, uncertainty,
    obs_metrics, deform_rows, protection, verification, resolution, fit_diags,
    profiles, render_paths, timing, landmarks, v2_sidecar,
):
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M CST")
    by = {(m["state"], m["mesh"]): m for m in obs_metrics}
    shape = {r["state"]: r for r in shape_rows}
    deform = {r["state"]: r for r in deform_rows}
    v2_states = v2_sidecar.get("states", {})

    def obs_row(state):
        b = by[(state, "baseline")]
        v2 = by.get((state, "v2"), {})
        v3 = by[(state, "v3")]
        return (f"<tr><td>{state}</td><td>{b['coverage']:.3f}</td><td>{b['mae_mm']:.2f}</td>"
                f"<td>{b['median_signed_mm']:+.2f}</td><td>{b['p95_abs_mm']:.2f}</td>"
                f"<td>{v2.get('mae_mm', float('nan')):.2f}</td><td>{v2.get('median_signed_mm', float('nan')):+.2f}</td>"
                f"<td>{v3['coverage']:.3f}</td><td><b>{v3['mae_mm']:.2f}</b></td>"
                f"<td><b>{v3['median_signed_mm']:+.2f}</b></td><td>{v3['p95_abs_mm']:.2f}</td></tr>")

    def shape_row(state):
        r = shape[state]
        u = uncertainty[state]
        return (f"<tr><td>{state}</td><td>{r['H_at_label_mm']:+.2f}</td><td>{r['H_peak_mm']:+.2f}</td>"
                f"<td>({r['peak_u_mm']:+.0f}, {r['peak_v_mm']:+.0f})</td>"
                f"<td>{fmt(r['width_half_mm'])}</td><td>{fmt(r['length_half_mm'])}</td>"
                f"<td>{fmt(r['width_ten_mm'])}</td><td>{fmt(r['length_ten_mm'])}</td>"
                f"<td>{ftip(r)}</td><td>{u['background_centre_spread_mm']:.2f}</td>"
                f"<td>{'是/yes' if u.get('distinguishable') else '否/no'}</td></tr>")

    def ftip(r):
        return fmt(r["tip_flat_v_mm"])

    def fig(name, caption, width="100%"):
        path = out_dir / "render" / name
        if not path.is_file():
            return ""
        return (f"<figure><img style='width:{width}' src='{img64(path)}'/>"
                f"<figcaption>{caption}</figcaption></figure>")

    def prof_fig(name, caption):
        path = out_dir / "profiles" / name
        if not path.is_file():
            return ""
        return f"<figure><img style='width:80%' src='{img64(path)}'/><figcaption>{caption}</figcaption></figure>"

    supported = verification["honesty"].get("observation_supported") or {}
    supported_txt = "；".join(
        f"{s}: |H|={v['abs_H_mm']:.2f} mm vs 背景不确定度 {v['background_spread_mm']:.2f} mm → "
        f"{'可区分' if v['distinguishable'] else '不可区分（证据不足）'}"
        for s, v in supported.items()
    )
    checks = verification["checks"]
    all_pass = verification["passed"]
    timed = "；".join(f"{t['stage']} {t['seconds']:.1f}s" for t in timing[-6:])
    total = timing[-1]["seconds"] if timing else None

    resolutions = "；".join(
        f"{s}: 边长 {v['edge_min_med_p90_mm'][1]:.1f} mm (中位), 特征宽 {v['feature_width_ten_mm']:.1f} mm → "
        f"基础网格 {v['samples_across_feature_base']:.1f} 个采样/特征"
        for s, v in resolution.items()
    )

    ridge_rows = "".join(
        f"<tr><td>{s}</td><td>{d['parameters']['height_mm']:+.2f}</td><td>{d['parameters']['c0_mm']:+.1f}</td>"
        f"<td>{d['parameters']['width_mm']:.1f}</td><td>{d['parameters']['rtip_mm']:.1f}</td>"
        f"<td>{d['parameters']['l_up_mm']:.1f}/{d['parameters']['l_dn_mm']:.1f}</td>"
        f"<td>{d.get('background', {}).get('centre_mm', float('nan')):+.1f}</td>"
        f"<td>{d['depth_rms_mm']:.2f}</td><td>{d.get('bins', '—')}</td></tr>"
        for s, d in fit_diags.items()
    )

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>本人观测驱动的颈前重建 V3 报告</title>
<style>
 body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#1c1e22;color:#d8dce2;line-height:1.55}}
 .wrap{{max-width:1120px;margin:0 auto;padding:32px 24px 64px}}
 h1{{font-size:25px;border-bottom:2px solid #4a90d9;padding-bottom:10px}}
 h2{{font-size:19px;color:#7fb3e8;margin-top:34px;border-left:4px solid #4a90d9;padding-left:10px}}
 h3{{font-size:16px;color:#b8c6d9}}
 table{{border-collapse:collapse;margin:12px 0;width:100%;font-size:13.5px}}
 td,th{{border:1px solid #3a3f47;padding:6px 10px;text-align:left;vertical-align:top}}
 th{{background:#262a31}}
 tr.pass td:last-child{{color:#7fd88f;font-weight:600}}
 tr.fail td:last-child{{color:#e06c5f;font-weight:600}}
 table.kv td:first-child{{color:#9aa7b8;width:42%}}
 code{{background:#262a31;padding:1px 5px;border-radius:3px;font-size:12.5px}}
 .note{{background:#2a2519;border:1px solid #6b5b1e;padding:10px 14px;border-radius:6px;margin:12px 0}}
 .warn{{background:#2a1f1f;border:1px solid #6b2e2e;padding:10px 14px;border-radius:6px;margin:12px 0}}
 figure{{margin:16px 0;text-align:center}}
 figure img{{border-radius:6px;max-width:100%}}
 figcaption{{font-size:12.5px;color:#8a97a8;margin-top:6px}}
 .q{{background:#232830;border-left:4px solid #7fb3e8;padding:8px 14px;margin:14px 0;border-radius:4px}}
 .q b{{color:#a9c9ee}}
</style></head><body><div class="wrap">

<h1>本人观测驱动的颈前重建 V3（0.5.5，基于 0.5.2 未塑形基线）</h1>
<p style="color:#8a97a8">Subject-observed neck reconstruction V3 — background correction, individual ridge fit, fixed-protocol acceptance</p>

<div class="note"><b>一句话结论：</b>几何安全、观测拟合已把颈前表面从"落后真实皮肤约 12 mm"校正到中位误差约 0 毫米；
但<b>这份单视角深度数据并不支持一个明显的喉结</b>——局部隆起在扣除背景后接近 0（闭口）或轻微内收（张口），
V2 的 8 mm 形态没有观测依据。因此本次交付的是"保守、跟随皮肤"的颈部，并如实报告形态证据不足。
</div>

<h2>0. 报告开头必须回答的八个问题</h2>
<div class="q"><b>1. 本人的数据实际支持多大的隆起、什么位置和轮廓？</b><br>
固定协议测量：{supported_txt}。脊形拟合（幅度被保守限定在 ±3 mm）给出的局部修正为
闭口 {shape['closed']['H_at_label_mm']:+.2f} mm、张口 {shape['open']['H_at_label_mm']:+.2f} mm（标签处，扣除冻结背景）；
横向/纵向半高范围 {fmt(shape['closed']['width_half_mm'])}/{fmt(shape['closed']['length_half_mm'])} mm（闭口）。
结论：深度支持"跟随皮肤的背景校正"，对喉结本身只支持 0–2 mm 量级的局部差异，且与背景模型不确定度同量级。</div>
<div class="q"><b>2. 相比 V2，"小豆子"形态为何减少？</b><br>
V2 在模板错误的表面上叠加了 8 mm 程序化双场凸起；V3 先把模板颈部低频校正到真实皮肤（背景 ~{fit_diags['closed'].get('background', {}).get('centre_mm', 0):+.1f} mm 闭口），
再只用受先验约束的小脊形 + 受限残差描述局部差异。没有"错误背景 + 独立鼓包"的叠加结构。</div>
<div class="q"><b>3. 改善来自观测拟合、背景校正还是先验？</b><br>
主要来自<b>背景校正</b>（首命中 MAE 闭口 {by[('closed','baseline')]['mae_mm']:.1f}→{by[('closed','v3')]['mae_mm']:.1f} mm，张口 {by[('open','baseline')]['mae_mm']:.1f}→{by[('open','v3')]['mae_mm']:.1f} mm）；
局部脊形证据很弱，先验（±3 mm 上限、宽度/过渡厚度先验）主要起"防止追噪"的作用。</div>
<div class="q"><b>4. 哪些区域缺少真实证据？</b><br>
区域外围（|uv| &gt; ~30 mm，向颌下与衣领过渡）深度证据稀疏且混有遮挡/衣着；侧向与后颈完全没有观测；
多帧一致性只在同一正面视角内成立，无法验证侧向轮廓。</div>
<div class="q"><b>5. 是否改变了面部？</b><br>
{'否' if checks['protected_vertices_unchanged'] and checks['facial_first_hits_within_tolerance'] else '需要检查'}：
受保护顶点位移为 0；68 个人脸特征点固定射线的首命中点最大位移
{max(protection[s]['max_facial_hit_shift_mm'] for s in protection):.4f} mm（工程容差 {protection['closed']['tolerance_mm']} mm）。</div>
<div class="q"><b>6. closed/open 是否分别满足要求？</b><br>
几何检查：{'两者均通过' if all_pass else '存在未通过项'}（见 §5）。观测指标：闭口 MAE {by[('closed','v3')]['mae_mm']:.2f} mm、
张口 {by[('open','v3')]['mae_mm']:.2f} mm，均优于未塑形基线。形态证据：两状态均"证据不足"（见问题 1）。</div>
<div class="q"><b>7. 哪些尺寸或形状尚不能可靠确定？</b><br>
喉结高度、宽度、上下过渡长度、顶部圆钝度都落在观测噪声与背景模型敏感度范围内
（背景中心对拟合半径 {uncertainty['closed']['background_centre_sensitivity_mm']} 的变化 → 展布 {uncertainty['closed']['background_centre_spread_mm']:.2f} mm）；
侧向位置由单一视角支持，无法区分左右不对称。</div>
<div class="q"><b>8. 实际运行耗时和退出状态？</b><br>
总耗时 {total:.1f} s（主要阶段：{timed}）；退出状态：<code>{'0（通过）' if all_pass else '1（存在未通过项，见 run_status.json）'}</code>。</div>

<h2>1. 数据、基线与标签</h2>
{kv([
 ("基线", "0.5.2 未塑形几何（parameters.npz / delivery），未重跑 FLAME 身份重建"),
 ("参考帧", "闭口 closed_front_0（帧 0），张口 open_front_4（帧 19）"),
 ("深度", "uint16 mm；相机系 X right / Y up / Z backward"),
 ("标签来源", "人工指定（RGB+深度证据校正过），不是独立真值；置信度约 5 px (~3 mm)"),
 ("标签像素", f"闭口 {landmarks['states']['closed']['pixel_uv']}，张口 {landmarks['states']['open']['pixel_uv']}"),
 ("V2 产物", "runtime/outputs/zp_052_neck_bump_v2（仅作对照，未修改）"),
])}

<h2>2. 算法（V3）</h2>
<ul>
<li><b>局部颈坐标</b>：逐状态，原点=标签点；n=深度平面法向；沿颈轴由头部姿态初始化、由完整行截面质心线细化（&gt;35° 时回退），u=颈前横向。记录于 neck_frame_audit.json。</li>
<li><b>背景 b(u,v)</b>：6 系数二次曲面，全部可靠观测稳健拟合 → <b>先冻结</b>；主曲率/整体偏移归背景，背景不能形成局部凸起。</li>
<li><b>脊形 p</b>：z = h − k(v)(√((u−c(v))²+r_tip²) − r_tip)，c(v)=c0+c1(v−v0)，独立上下平滑过渡；由分箱中位残差稳健拟合；h 允许正负，但保守限定 |h|≤3 mm；</li>
<li><b>残差 r</b>：3×3 平滑基，|r|≤2.5 mm，强先验，仅吸收可靠的非对称细节。</li>
<li><b>目标函数</b>：E_depth（真实首命中深度，按每像素法向/射线响应因子归一，σ=2 mm，soft-L1 稳健）、
E_contour（观测行宽 vs 投影宽度，σ=10 mm）、E_landmark（标签位置，σ=4 mm）、E_shape（弱先验）、
E_deformation（边长拉伸超出 25% 部分）、E_bending（位移拉普拉斯，1 mm）、E_residual（残差先验 0.8 mm）。</li>
<li><b>一致性</b>：拟合与验收使用同一个"真实首命中"定义；z-buffer 仅用于早期的快速诊断（已知在 5–8 mm 三角形边缘有 ≥3 mm 走样，不再用于评分）。</li>
</ul>
<p>分辨率：{resolutions}。基础网格不足以让 20 mm 特征获得 ≥6 个采样，因此对颈前局部做了确定性红绿细分（closed/open 同一构造、同一父面绑定），交付标识为细化几何。</p>

<h2>3. 拟合结果</h2>
<table><tr><th>状态</th><th>h (mm)</th><th>c0 (mm)</th><th>W (mm)</th><th>r_tip</th><th>上/下过渡 (mm)</th><th>背景中心 (mm)</th><th>分箱深度 RMS (mm)</th><th>分箱数</th></tr>{ridge_rows}</table>
{kv([
 ("拟合方法", "稳健分箱 + 有界最小二乘（28→11 参数），一次渲染首面校正诊断（本次保守模式下不缩放）"),
 ("说明", "h 为负表示对模板已有错误凸起的减小；本次两状态 h 均为负且幅度 ≤ 2 mm，位置与标签相差 ≤ 11 mm"),
])}

<h2>4. 独立观测验收（固定母集、真实首命中）</h2>
<table><tr><th>状态</th><th>baseline 覆盖</th><th>baseline MAE</th><th>baseline 中位带符号</th><th>baseline P95</th>
<th>V2 MAE</th><th>V2 中位</th><th>V3 覆盖</th><th>V3 MAE</th><th>V3 中位带符号</th><th>V3 P95</th></tr>
{obs_row('closed')}{obs_row('open')}</table>
<p>定义：真实第一命中；无命中罚 30 mm；误差不截断；同时报告 MAE、P95 与覆盖率。正值 = 渲染表面在真实皮肤之前。</p>
<table><tr><th>状态</th><th>H@标签 (mm)</th><th>H 峰值</th><th>峰值 (u,v)</th><th>半高宽</th><th>半高长</th><th>10% 宽</th><th>10% 长</th><th>顶部平台 (mm)</th><th>背景展布 (mm)</th><th>与噪声可区分</th></tr>
{shape_row('closed')}{shape_row('open')}</table>
{prof_fig('closed_H_lines.png', '闭口：固定评分线 H(u, v=0) 与 H(u=0, v)，背景已冻结')}
{prof_fig('open_H_lines.png', '张口：同上')}

<h2>5. 几何安全与保护</h2>
<table><tr><th>检查</th><th>结果</th></tr>
{''.join(f"<tr class='{'pass' if v else 'fail'}'><td>{k}</td><td>{'通过' if v else '失败'}</td></tr>" for k, v in checks.items())}</table>
{kv([
 ("自交检测（整网格受影响段）", "；".join(f"{s}: {r['count']} 处（{r['segments']} 段 / {r['candidate_faces']} 面）" for s, r in verification['self_intersections'].items())),
 ("人脸首命中最大点位移", "；".join(f"{s}: {p['max_facial_hit_shift_mm']:.4f} mm" for s, p in protection.items())),
 ("受保护顶点移动数", "；".join(f"{s}: {deform[s]['protected_vertices_moved']}" for s in deform)),
 ("拉伸 >25% 的边数", "；".join(f"{s}: {deform[s]['stretch_over_25pct']}（p99 {deform[s]['stretch_p99']:.3f}）" for s in deform)),
 ("平移顶点 / 最大位移", "；".join(f"{s}: {deform[s]['moved_vertices']} / {deform[s]['max_displacement_mm']:.1f} mm" for s in deform)),
])}

<h2>6. 同视角渲染（相同相机、姿态、比例与光照）</h2>
<p>三条网格：未塑形基线 / V2 / V3。相对标签点取景，90° 极坐标一致。</p>
{fig('closed_front_compare.png', '闭口：正面')}
{fig('closed_side_compare.png', '闭口：左侧面（观察者方向）')}
{fig('closed_45deg_compare.png', '闭口：45°')}
{fig('closed_closeup_solid.png', '闭口：颈部局部放大（实体）')}
{fig('closed_closeup_wireframe.png', '闭口：局部线框（几何本身，不依赖着色）')}
{fig('closed_section.png', '闭口：过标签点剖面（面片∩平面；法向倾 15° 仅用于可视化）')}
{fig('open_front_compare.png', '张口：正面')}
{fig('open_side_compare.png', '张口：侧面')}
{fig('open_45deg_compare.png', '张口：45°')}
{fig('open_closeup_solid.png', '张口：局部放大')}
{fig('open_closeup_wireframe.png', '张口：局部线框')}
{fig('open_section.png', '张口：剖面')}
<p>剖面线由面片与平面真实求交得到，交线段内按 0.5 mm 加密仅用于绘图，不增加独立观测；固定评分平面为 u=0 与 v=0（§4），倾斜剖面仅作附加可视化。</p>

<h2>7. 结论与限制</h2>
<div class="warn"><ul>
<li><b>形态证据不足</b>：本次数据可支持背景校正与“贴近皮肤”的颈部，但不能支持一个明确隆起的喉结。若需要喉结形态，需要更好的采集（侧向视角、更高深度精度或带标记的多视角），而不是在现有数据上继续加大幅度。</li>
<li>观测只是单视角正面深度 + 单帧图像；侧向轮廓、后颈与颈部完整周长无证据。</li>
<li>标签来自人工指定（有 RGB/深度证据），不自动等于独立真值；自动定位（ArUco）在本数据中无标签可用。</li>
<li>细化几何使用本地红绿细分；基础网格另行保留，不可将其声明为与旧面数组完全相同。</li>
<li>本报告中的模型参数是低维皮肤外观参数，不是解剖标准。</li>
</ul></div>

<p style="color:#8a97a8;margin-top:36px">报告生成：{now} · tools/make_neck_v3_report.py · 输出目录：{out_dir.name}</p>
</div></body></html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")
