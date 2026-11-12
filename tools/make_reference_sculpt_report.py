import base64
import json
from datetime import datetime
from pathlib import Path


def img(p):
    p = Path(p)
    if not p.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def main():
    preview = Path("runtime/outputs/neck_reference_sculpt_ref_preview")
    asset = json.loads(Path("runtime/assets/neck_reference_shape/neck_reference_shape.json").read_text(encoding="utf-8"))
    v056 = Path("runtime/outputs/zp_056_neck_sculpt_20261112_115219/render")
    v057 = Path("runtime/outputs/zp_057_neck_proportional_20261112_124238/render")
    m = asset["reference_measurements_mm"]
    kv_rows = "".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(k, v)
        for k, v in [
            ("参考文件", asset["source_path"]),
            ("SHA256", asset["source_sha256"]),
            ("顶点/面", "{} / {}（quad 已三角化）".format(asset["source_mesh"]["vertices"], asset["source_mesh"]["faces"])),
            ("单位/坐标", "mesh_units=m；静止系 X 右 / Y 上 / Z 朝脸（identity_rest 为静止几何，未直接叠加到当前相机系）"),
            ("只读状态", "未修改 line1 目录；仅读取并记录哈希"),
            ("关键测量", "喉结候选 y={} mm；中线残差峰值 {} mm（被颏下/肩部污染）；参考 W_neck={} mm；参考 L_neck={} mm".format(
                m["larynx_centre_y"], m["height_mm"], m["W_neck_ref_mm"], m["L_neck_ref_mm"])),
        ]
    )
    figs = "".join(
        "<figure><img src='{}'/><figcaption>{}</figcaption></figure>".format(img(preview / name), cap)
        for name, cap in (("ref_z_neg.png", "参考正面（沿 −Z）"), ("ref_x_pos.png", "参考侧面"),
                          ("ref_z_pos.png", "参考背面（沿 +Z）"), ("ref_side_neck.png", "颈前局部（参考）"))
    )
    figs2 = "".join(
        "<figure><img src='{}'/><figcaption>{}</figcaption></figure>".format(img(path), cap)
        for path, cap in ((v056 / "natural_closed_side_close_compare.png", "0.5.6 公开默认 闭口"),
                          (v057 / "natural_closed_side_close.png", "0.5.7 候选 闭口"),
                          (v056 / "natural_open_side_close_compare.png", "0.5.6 公开默认 张口"),
                          (v057 / "natural_open_side_close.png", "0.5.7 候选 张口"))
    )
    html = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>参考喉结形态雕塑报告</title>
<style>
 body{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#1c1e22;color:#d8dce2;line-height:1.55}
 .wrap{max-width:1080px;margin:0 auto;padding:32px 24px 64px}
 h1{font-size:24px;border-bottom:2px solid #4a90d9;padding-bottom:10px}
 h2{font-size:18px;color:#7fb3e8;margin-top:32px;border-left:4px solid #4a90d9;padding-left:10px}
 table{border-collapse:collapse;margin:12px 0;width:100%;font-size:13px}
 td,th{border:1px solid #3a3f47;padding:5px 9px;text-align:left} th{background:#262a31}
 table.kv td:first-child{color:#aab7c8;width:36%}
 code{background:#262a31;padding:1px 5px;border-radius:3px}
 .note{background:#2a2519;border:1px solid #6b5b1e;padding:10px 14px;border-radius:6px;margin:12px 0}
 .warn{background:#2a1f1f;border:1px solid #6b2e2e;padding:10px 14px;border-radius:6px;margin:12px 0}
 figure{margin:14px 0;text-align:center} figure img{border-radius:6px;max-width:100%}
 figcaption{font-size:12.5px;color:#8a97a8;margin-top:5px}
</style></head><body><div class="wrap">
<h1>参考喉结形态雕塑 — 诊断与交付（默认保持不变）</h1>
<p style="color:#8a97a8">Reference-shape laryngeal sculpt — extraction diagnostics</p>
<div class="warn"><b>结论：参考形态迁移未通过回放验证，按任务规则停止迁移并保留当前公开默认（0.5.6 固定设计）。
本报告交付参考核对、提取诊断、对照与阻断项；未提高高度、未修改验收阈值、未把未验证形态设为默认。</b></div>

<h2>0. 六个问题</h2>
<ol>
<li><b>当前为什么小而尖：</b>0.5.7 natural 为适配短颈把宽度缩到 16–17 mm、长度 15–19 mm，却保留 H=6.58 mm（按颈宽计算），
宽/长/高/圆钝未协同缩放；成品 W50≈10 mm、L50≈9 mm，是窄而高的凸点。</li>
<li><b>提取了参考的哪些形态：</b>参考网格（%%VERTS%% 顶点 / %%FACES%% 三角面，SHA256 <code>%%SHA%%</code>，单位 m，静止系 +Z 朝脸）。
扣除参考自身颈基面后，中线上方约 <b>+2.2 mm</b>、两侧 −1.6~−2.3 mm，跨度 ±25–40 mm——是<b>宽缓、低幅、长过渡的颈前体量</b>，
不存在可分离的离散喉结脊；峰值搜索落在颏下反折与肩部（|u|≈54 mm、残差 26 mm）。</li>
<li><b>如何适配当前模板：</b>建立项目内资产 <code>runtime/assets/neck_reference_shape/neck_reference_shape.json</code>
（41×41 归一化高度场 + 来源路径/哈希/单位/方法），按目标 W_neck/L_neck 协同缩放；但回放无法还原参考局部形态
（资产峰值 %%PEAK%% mm 来自颏下/肩部污染，半长 %%HALFV%% mm 覆盖整段颈部），故未将迁移结果用于生产。</li>
<li><b>体量/圆钝/位置是否接近参考：</b>未验证（迁移未发布）。对照：0.5.6 H≈5.0 / W50≈18.1 / L50≈11.5；
0.5.7 H≈6.6 / W50≈10.6 / L50≈9.0；参考提示“更低更宽”，但参考本身缺少可分离脊，无法给出可验收目标。</li>
<li><b>closed/open 是否一致：</b>0.5.7 候选中心 s=0.38 为共享基顶点绑定，闭/张 realize 6.58/6.21 mm；迁移未发布，不做一致性声明。</li>
<li><b>安全/面部保护/开销：</b>候选几何安全已通过（0 自交/折叠/翻面/细长面、面部命中位移 0.0000 mm、磁盘一致）；
本次未新增生产几何，不产生新的安全风险；参考扫描为秒级只读处理。</li>
</ol>

<h2>1. 参考核对</h2>
<table class="kv">%%KV%%</table>

<h2>2. 参考渲染</h2>
%%FIGS%%
<p>参考侧视显示颈前是一条连续的、宽缓体量，喉结区域没有形成独立凸点——这正是“体量、圆钝、左右展开、上下连接”的来源。</p>

<h2>3. 当前公开默认（0.5.6）与 0.5.7 候选对照（同尺度）</h2>
%%FIGS2%%
<p>两者都是“凸点”式形态；参考则是宽缓体量。按真实物理尺寸对照，0.5.7 比 0.5.6 更高更窄，与参考关系相反。</p>

<h2>4. 形态资产与回放</h2>
<p>资产：<code>runtime/assets/neck_reference_shape/neck_reference_shape.json</code>（head_rgbd.neck_reference_shape.v1）。</p>
<table class="kv"><tr><td>回放峰值</td><td>%%PEAK%% mm（原始峰值 %%PEAK0%% mm）</td></tr>
<tr><td>结论</td><td>回放无法把参考还原为可分离的喉结局部形态 → 停止迁移</td></tr></table>

<h2>5. 阻断项</h2>
<div class="warn"><ul>
<li>参考缺少可分离的喉结脊：扣除自身颈基面后中线上方仅约 +2.2 mm，左右为负值；无法据此给定可验收的目标高度/宽度/长度。</li>
<li>残差场被颏下反折与肩部污染（峰值 26 mm 位于 |u|≈54 mm），回放未通过。</li>
<li>0.5.7 候选的尖端曲率测量仍受坡面污染，不能用于圆钝度验收。</li>
<li>公开入口仍为 0.5.6（editing_mode=true，subject_observation_matched=false）。</li>
</ul></div>

<h2>6. 后续路径</h2>
<ol>
<li>若参考另有专门雕刻喉结的版本（head_neck_closed/open 与 identity_rest 的差异），应以“雕刻版 − 静止版”差值场作为参考形态；本轮未核对这一对应关系。</li>
<li>或直接采用参考的宽缓关系作为目标：低幅 2–3 mm、W10≈0.3 W_neck、L10≈0.4 L_neck、tip≈0.5 H，并在成品网格上验收比例与安全。</li>
</ol>

<p style="color:#8a97a8;margin-top:30px">生成：%%NOW%% · 参考只读 · 默认未变 · 0.5.7 候选保留</p>
</div></body></html>"""
    html = html.replace("%%VERTS%%", str(asset["source_mesh"]["vertices"])) \
               .replace("%%FACES%%", str(asset["source_mesh"]["faces"])) \
               .replace("%%SHA%%", asset["source_sha256"][:24] + "…") \
               .replace("%%PEAK%%", str(asset["replay"]["peak_baked_mm"])) \
               .replace("%%PEAK0%%", str(asset["replay"]["peak_original_mm"])) \
               .replace("%%HALFV%%", str(m["half_length_mm"])) \
               .replace("%%KV%%", kv_rows).replace("%%FIGS%%", figs).replace("%%FIGS2%%", figs2) \
               .replace("%%NOW%%", datetime.now().strftime("%Y-%m-%d %H:%M"))
    Path("docs/NECK_REFERENCE_SHAPE_SCULPT_REPORT.html").write_text(html, encoding="utf-8")
    print("report written")


if __name__ == "__main__":
    main()
