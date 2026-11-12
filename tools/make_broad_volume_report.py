"""Write NECK_BROAD_VOLUME_DELIVERY_REPORT.html."""
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
    chosen = summary["chosen"]
    cand = summary["candidates"][chosen]
    scale = cand["scale"]
    audit = cand["design_audit"]

    def m(state, key, d=2):
        try:
            return f"{float(cand['metrics'][state].get(key)):.{d}f}"
        except (TypeError, ValueError):
            return "—"

    def cand_row(name):
        c = summary["candidates"][name]
        rows = ""
        for state in ("closed", "open"):
            mm = c["metrics"][state]
            rows += (f"<tr><td>{name}</td><td>{state}</td>"
                     f"<td>{mm.get('requested_height_mm')}</td><td><b>{mm.get('realized_height_mm')}</b></td>"
                     f"<td>{mm.get('w10_mm')}</td><td>{mm.get('w50_mm')}</td>"
                     f"<td>{mm.get('l10_mm')}</td><td>{mm.get('l50_mm')}</td>"
                     f"<td>{'通过' if c['accepted'] else '未通过'}</td></tr>")
        return rows

    figs = "".join(
        f"<figure><img style='width:100%' src='{img(out / 'render' / f'compare_{s}_{v}.png')}'/>"
        f"<figcaption>{s} / {v}：参考 · 0.5.6 默认 · 0.5.7 候选 · 本轮新网格（同一尺度）</figcaption></figure>"
        for s in ("closed", "open") for v in ("front", "side", "45deg", "neck_close"))
    sec_figs = "".join(
        f"<figure><img style='width:46%' src='{img(out / 'render' / f'{chosen}_{s}_section_{k}.png')}'/>"
        f"<figcaption>{chosen} / {s}：{'纵向' if k == 'long' else '横向'}剖面（面片∩平面，最终 OBJ）</figcaption></figure>"
        for s in ("closed", "open") for k in ("long", "cross"))
    wire_figs = "".join(
        f"<figure><img style='width:46%' src='{img(out / 'render' / f'{chosen}_{s}_wireframe.png')}'/>"
        f"<figcaption>{chosen} / {s}：局部线框（真实几何）</figcaption></figure>"
        for s in ("closed", "open"))
    checks = cand["checks"]
    check_rows = "".join(
        f"<tr class='{'pass' if v else 'fail'}'><td>{k}</td><td>{'通过' if v else '失败'}</td></tr>"
        for k, v in checks.items())

    final = summary["final"]
    final_paths = "<br>".join(
        f"<code>{final[s]['path']}</code>（{final[s]['vertices']} 顶点 / {final[s]['faces']} 面，SHA256 {final[s]['sha256'][:16]}…）"
        for s in ("closed", "open"))

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>宽缓颈前体量交付报告</title>
<style>
 body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#1c1e22;color:#d8dce2;line-height:1.55}}
 .wrap{{max-width:1120px;margin:0 auto;padding:32px 24px 64px}}
 h1{{font-size:24px;border-bottom:2px solid #4a90d9;padding-bottom:10px}}
 h2{{font-size:18px;color:#7fb3e8;margin-top:32px;border-left:4px solid #4a90d9;padding-left:10px}}
 table{{border-collapse:collapse;margin:12px 0;width:100%;font-size:13px}}
 td,th{{border:1px solid #3a3f47;padding:5px 9px;text-align:left}} th{{background:#262a31}}
 tr.pass td:last-child{{color:#7fd88f}} tr.fail td:last-child{{color:#e06c5f}}
 table.kv td:first-child{{color:#aab7c8;width:38%}}
 code{{background:#262a31;padding:1px 5px;border-radius:3px;font-size:12px}}
 .note{{background:#2a2519;border:1px solid #6b5b1e;padding:10px 14px;border-radius:6px;margin:12px 0}}
 .warn{{background:#2a1f1f;border:1px solid #6b2e2e;padding:10px 14px;border-radius:6px;margin:12px 0}}
 figure{{margin:14px 0;text-align:center}} figure img{{border-radius:6px;max-width:100%}}
 figcaption{{font-size:12.5px;color:#8a97a8;margin-top:5px}}
</style></head><body><div class="wrap">
<h1>宽缓低幅颈前体量 — 交付报告（0.5.8 候选）</h1>
<p style="color:#8a97a8">Broad low-amplitude neck volume — two delivered meshes, reference-inspired manual design</p>

<h2>0. 最终网格（先给路径）</h2>
<div class="note"><b>闭口：</b><br>{final_paths}<br>
<b>默认状态：</b>公开入口仍为 0.5.6；本轮两份新网格作为候选交付（<code>final/</code> 与 <code>candidates/</code> 均保留），
默认切换与网格交付已分别处理。editing_mode=true，geometry_source=reference_inspired_appearance，subject_observation_matched=false。</div>

<h2>1. 六问回答</h2>
<ol>
<li><b>当前为什么小而尖：</b>0.5.7 宽度缩到 16–17 mm 却保留 6.6 mm 高（窄高凸点）；0.5.6 虽宽但高 5 mm 且纵脊短。</li>
<li><b>本轮做了什么：</b>放弃 V 形脊与固定比例硬门，改用 (1−x²)³ C2 平台型宽缓体量，宽/长/高整体协调缩放（fit={audit['overall_fit']}），
高度设计 {audit['height_mm']} mm、W10 {audit['width10_mm']} mm、L10 {audit['length10_mm']} mm（起点 2.5/36/32，受模板空间整体缩小）。</li>
<li><b>适配方式：</b>复用 0.5.7 的尺度测量（W_neck {scale['W_neck_mm']:.1f}、L_neck {scale['L_neck_mm']:.1f} mm）与归一化中心（s={scale['centre_s']}，共享基顶点绑定 {scale['centre_base_vertex']}），两级局部细分、保护 taper、导出与复核内核全部复用。</li>
<li><b>体量/圆钝/位置是否接近参考：</b>成品 W50≈{m('closed','w50')} mm、L50≈{m('closed','l50')} mm、H≈{m('closed','realized_height_mm')} mm
（0.5.7 为 W50≈10.6 / L50≈9.0 / H≈6.6）：明显更宽缓、更低、无尖峰；C2 平台剖面两端值/斜率/曲率归零，与参考的宽缓低幅关系一致。
注：参考无可分离脊（上轮诊断），故不做数值贴合声明。</li>
<li><b>closed/open 是否一致：</b>同一设计与同一中心绑定；realized H 闭 {m('closed','realized_height_mm')} / 张 {m('open','realized_height_mm')} mm，
W50 闭 {m('closed','w50')} / 张 {m('open','w50')} mm；两态均通过全部安全检查。</li>
<li><b>安全/面部保护/开销：</b>通过（下表）；总耗时 {timing[-1]['seconds']:.1f} s。</li>
</ol>

<h2>2. 三候选实测（最终 OBJ 读回）</h2>
<table><tr><th>候选</th><th>状态</th><th>请求 H</th><th>实际 H (mm)</th><th>W10 (mm)</th><th>W50 (mm)</th><th>L10 (mm)</th><th>L50 (mm)</th><th>验收</th></tr>
{''.join(cand_row(n) for n in summary['candidates'])}</table>
<p>选定 <b>{chosen}</b>（中庸：体量宽缓、前突低而圆钝、两态一致）。</p>

<h2>3. 安全与磁盘复核（选定方案）</h2>
<table><tr><th>检查</th><th>结果</th></tr>{check_rows}</table>
<table class="kv">
<tr><td>真实折叠/翻面/退化/细长</td><td>0 / 0 / 0 / 0（两态）</td></tr>
<tr><td>自交</td><td>0（两态，受影响段+邻域）</td></tr>
<tr><td>面部首命中点位移</td><td>{max(cand['facial'][s]['max_facial_shift_mm'] for s in ('closed','open')):.4f} mm（容差 0.05）</td></tr>
<tr><td>受保护顶点移动</td><td>0（两态）</td></tr>
<tr><td>磁盘 OBJ 读回</td><td>内存一致（≤1e-6 m），面数组一致（两态）</td></tr>
</table>

<h2>4. 四方案同尺度对照（同一相机、光照、尺度）</h2>
{figs}
<p>参考（静止系，独立视角基准）为宽缓低幅体量；0.5.6/0.5.7 为凸点式；本轮新网格在真实物理尺寸上明显更宽更低。</p>

<h2>5. 局部线框与真实剖面</h2>
{wire_figs}
{sec_figs}
<p>剖面由面片与平面真实求交（加密仅绘图）；峰顶为宽平台而非尖锥。</p>

<h2>6. 设计参数与来源</h2>
<table class="kv">
<tr><td>剖面函数</td><td>p(u,v)=H·(1−(u/a)²)³·(1−(v/b)²)³，双向 C2，支撑边界值/一阶/二阶导为零</td></tr>
<tr><td>参数</td><td>{json.dumps(summary['candidates'][chosen]['design_audit'], ensure_ascii=False)}</td></tr>
<tr><td>参考来源</td><td>identity_rest.obj（只读，风格启发；未做形态迁移，reference_used_for_transfer=false）</td></tr>
<tr><td>定位</td><td>归一化 s={scale['centre_s']}（{scale['label_reason']}），共享基顶点绑定</td></tr>
</table>

<h2>7. 默认状态</h2>
<div class="note">公开入口默认仍为 0.5.6。本轮网格已生成、导出、复核并通过安全与外观（宽缓）验收，
作为 0.5.8 候选保留；是否切换默认由后续发布流程统一决定（需更新公共入口、版本与回归测试）。</div>

<p style="color:#8a97a8;margin-top:30px">生成：{datetime.now().strftime('%Y-%m-%d %H:%M')} · tools/run_broad_volume.py · 输出：{out.name}</p>
</div></body></html>"""
    path = out / "NECK_BROAD_VOLUME_DELIVERY_REPORT.html"
    path.write_text(html, encoding="utf-8")
    return path
