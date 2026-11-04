"""Assemble the default acceptance report from preserved measurements."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runtime/validation/default_smooth_fast'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    timing = read(WORK / 'interleaved_v1/timings.json')
    smooth = read(WORK / 'final_smoothness.json')
    quality = read(WORK / 'quality_final/multiview.json')
    baseline, final = smooth[0], smooth[1]
    bframes = {f['frame']: f for f in baseline['frames']}
    angle_pass = all(f[k] <= bframes[f['frame']][k] for f in final['frames']
                     for k in ('rms_deg', 'p95_deg'))
    qb, qf = quality['runs'][:2]
    q120 = next((r for r in quality['runs'][2:] if 'front120' in Path(r['run']).name), None)
    g120 = next((g for g in q120['groups'] if g['state']=='open'), None) if q120 else None
    bg = {(g['view'], g['state'], g['role']): g for g in qb['groups']}
    br = {(g['view'], g['state'], g['role'], g['region']): g for g in qb['regions']}
    groups = []
    for g in qf['groups']:
        b = bg[g['view'], g['state'], g['role']]
        groups.append(dict(**g, delta_mae=g['mae_mm']-b['mae_mm'],
                           delta_p95=g['p95_mm']-b['p95_mm'], delta_coverage=g['coverage']-b['coverage']))
    regions = []
    for g in qf['regions']:
        b = br[g['view'], g['state'], g['role'], g['region']]
        if g['mae_mm'] is not None and b['mae_mm'] is not None:
            regions.append(dict(**g, delta_mae=g['mae_mm']-b['mae_mm']))
    replay_comparison = None
    if len(quality['runs']) > 3:
        replay = {(g['view'], g['state'], g['role']): g for g in quality['runs'][3]['groups']}
        replay_comparison = {key: max(
            g[key]-replay[g['view'], g['state'], g['role']][key] for g in qf['groups'])
            for key in ('mae_mm', 'p95_mm')}
        replay_comparison['worst_coverage_delta'] = min(
            g['coverage']-replay[g['view'], g['state'], g['role']]['coverage'] for g in qf['groups'])
    depth_pass = all(g['delta_mae'] <= .05 for g in groups)
    region_pass = all(g['delta_mae'] <= .1 for g in regions if g['region'] in ('nose', 'lips', 'jaw_contour'))
    coverage_pass = all(g['delta_coverage'] >= -.001 for g in groups)
    final_run = Path(final['run']).resolve()
    fit = read(final_run / 'reconstruction/dense/report.json')
    loop = fit['loop_base']
    check = next(t['check'] for t in reversed(loop['nonlinear_trials']) if t['safe'])
    safety_pass = (check['maximum_total_displacement_mm'] <= 3.0000001 and
                   check['maximum_normal_change_deg'] <= 35 and
                   check['minimum_signed_area_ratio'] >= .05 and check['maximum_slope'] <= .4500001)
    time_pass = timing['default']['median_s'] <= timing['joint_fast']['median_s']
    runs = [Path(r['output']) for r in timing['records'] if r['label'] == 'default']
    with np.load(runs[0] / 'reconstruction/dense/parameters.npz') as z:
        vertices = z['dense_vertices']
    repetition_differences = []
    for run in runs[1:]:
        with np.load(run / 'reconstruction/dense/parameters.npz') as z:
            repetition_differences.append(float(abs(z['dense_vertices']-vertices).max()))
    sources = [read(run / 'run_report.json')['source_hashes'] for run in runs]
    same_code = all(s == sources[0] for s in sources)
    current_source_match = all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == h
                               for name, h in sources[0].items())
    frozen_hashes_ok = all(hashlib.sha256((WORK / 'source_before' / name).read_bytes()).hexdigest() == h
                          for name, h in read(WORK / 'source_before_sha256.json').items())
    config = read(final_run / 'delivery/manifest.json')['configuration']
    reports_agree = all(read(run / 'run_report.json')['configuration'] ==
                        read(run / 'delivery/manifest.json')['configuration'] for run in runs)
    acceptance = dict(smooth_every_frame=angle_pass, view_mae=depth_pass, key_region_mae=region_pass,
                      coverage=coverage_pass, geometric_safety=safety_pass, reconstruction_time=time_pass,
                      same_candidate_code=same_code, report_manifest_agree=reports_agree,
                      current_source_matches_measurement=current_source_match, baseline_snapshot_intact=frozen_hashes_ok,
                      repeated_vertex_max_difference_m=repetition_differences,
                      replay_baseline_depth_comparison=replay_comparison,
                      final_run=str(final_run), configuration=config, groups=groups, regions=regions)
    (WORK / 'acceptance.json').write_text(json.dumps(acceptance, indent=2), encoding='utf-8')
    ok = all((angle_pass, depth_pass, region_pass, coverage_pass, safety_pass, time_pass,
              same_code, reports_agree, current_source_match, frozen_hashes_ok))
    lines = ['# 默认平滑快速重建验收（0.5.0）', '', f'状态：{"已完成" if ok else "未完成：见下列未通过项"}。', '',
      '1. **默认流程**：观测准备 → 正面初始化 240 步 → 正侧联合身份 240 步 → 一次运动校正 80 步 → '
      '低频 2 轮 / 中频 1 轮 → 安全 Loop 目标 0.5 → 闭口与张口交付。',
      f'2. **光滑度达到 joint_fast_v3**：{angle_pass}；固定区域内全部 20 帧逐帧 RMS 与 P95 验收。',
      f'3. **深度误差和覆盖通过**：MAE {depth_pass}，关键区域 {region_pass}，coverage {coverage_pass}，几何安全 {safety_pass}。',
      f'4. **相同条件耗时不高于 joint-fast**：{time_pass}；外部墙钟中位数 '
      f'{timing["joint_fast"]["median_s"]:.3f} → {timing["default"]["median_s"]:.3f} s。', '',
      '## 实际默认值与入口', '', '```json', json.dumps(config, ensure_ascii=False, indent=2), '```', '',
      '`python -m head_rgbd run` 与 `reconstruct` 共用 `_add_fit_options(load_defaults())`；'
      '`headrgbd.cmd` 与 `headrgbd.ps1` 原样透传 `-m head_rgbd`。正式计时中的新方案没有填写任何模式参数。'
      '默认不执行未消费的 fit_c 后处理、独立低频阶段、第二次运动、重复锚定和密集残差求解。'
      '`reuse-anchor` 仅在 full 对照链有效，不是 lean 的优化项。', '',
      (f'120 步初始化在最终模板规则下复核，张口 coverage {g120["coverage"]*100:.4f}%，'
       f'基准 {next(g["coverage"] for g in qb["groups"] if g["state"]=="open")*100:.4f}%，未采用。'
       if g120 else '120 步初始化的初始实验张口 coverage 97.40%，基准 98.94%，未采用。') +
      '180、220 步是模板资格修复前的预算探索，未继续作为发布候选；保留已通过的 240 步，'
      '时间目标通过精简链满足。没有新增早停、缓存或求解器。', '',
      '## 几何改动与选择依据', '',
      'Loop 直接尝试目标 0.5，只有原位移、坡度、法向和防翻面检查失败才逐次减半，最后检查零混合；'
      '不安全的控制网格即便零混合也拒绝。报告记录目标、实际值、全部检查及回退原因。'
      '默认取消三候选深度打分，显式实验打分也逐候选检查安全。', '',
      '关键修复是先验模板的证据资格与 barycentric 控制场一致：连续信息允许可靠单视角更新，'
      'Loop 模板也允许至少一帧真实支持。未观测控制点、解剖禁区及其边界继续固定。'
      '这只改变先验模板，未扩大优化自由集，未改变证据标签，不将平滑计为新增实测。'
      '中频已计算的支持与低频支持合并复用；本例该合并未新增两帧支持顶点，不将其算作质量收益。', '',
      '先保持原设置比较中频弯曲 4/6/8 mm。恢复 Loop 后仅增大中频仍不足，'
      '10/12 mm 中频、低频 10/12/20 mm、梯度 6/12 mm、160 步运动和 180/220 步初始化也留下独立实验。'
      '最终在证据资格修复后重新比较 6/8：6 未通过，8 是已测试强度中最弱通过值。'
      '低频弯曲恢复原 8 mm、梯度恢复原 3 mm、运动恢复原 80 步。', '',
      'FEM 惩罚相对原模型的向量曲率变化，保留原身份和解剖结构；没有整体收缩、切向自由度、'
      '曲率自由集扩展、新密集残差或新后处理。缺失射线 8 mm 惩罚、逐帧 coverage、'
      '2% + 0.002 mm² 数据验收未改变。', '', '## 固定光滑度评价域', '',
      '评价域仅由 joint_fast_v3 的重复实测支持与解剖掩码固定：observed_frames ≥ 2 且非解剖禁区，'
      '相邻两面的所有顶点均满足。4218 顶点、7837 面、11459 相邻面对。全部候选验证相同拓扑，'
      '每条共享边等权，单位度；逐帧用面法向，不使用显示平滑法向。', '',
      f'域哈希：`{read(WORK / "evaluation_domain.json")["domain_sha256"]}`。'
      '完整索引保存在 `runtime/validation/default_smooth_fast/evaluation_domain.npz`。'
      '原历史 5.95° 不直接套用；本报告所有基准数字均在本固定域重算。', '',
      '| 状态 | 基准平均 RMS | 新版平均 RMS | 基准平均 P95 | 新版平均 P95 |',
      '|---|---:|---:|---:|---:|']
    for state in ('closed', 'open'):
        bf = [f for f in baseline['frames'] if f['state'] == state]
        ff = [f for f in final['frames'] if f['state'] == state]
        lines.append(f'| {state} | {np.mean([f["rms_deg"] for f in bf]):.6f} | '
                     f'{np.mean([f["rms_deg"] for f in ff]):.6f} | {np.mean([f["p95_deg"] for f in bf]):.6f} | '
                     f'{np.mean([f["p95_deg"] for f in ff]):.6f} |')
    lines += ['', '表为逐帧统计的状态平均；验收逐帧执行，完整结果见 `final_smoothness.json`。', '',
      '| 区域/状态 | 基准平均 RMS | 新版平均 RMS | 基准平均 P95 | 新版平均 P95 |', '|---|---:|---:|---:|---:|']
    for name in ('forehead', 'cheeks', 'jaw'):
        for state in ('closed', 'open'):
            bf = [f['regions'][name] for f in baseline['frames'] if f['state'] == state]
            ff = [f['regions'][name] for f in final['frames'] if f['state'] == state]
            lines.append(f'| {name}/{state} | {np.mean([f["rms_deg"] for f in bf]):.4f} | '
                         f'{np.mean([f["rms_deg"] for f in ff]):.4f} | {np.mean([f["p95_deg"] for f in bf]):.4f} | '
                         f'{np.mean([f["p95_deg"] for f in ff]):.4f} |')
    lines += ['', '局部区域按基准 canonical 坐标和眉、鼻翼、外唇关键点固定，定义见工具输出。', '',
      f'新版支持区相邻场差 RMS/P95/max：`{final["supported_edge_difference"]}`；'
      f'相对一环邻居均值的尖峰：`{final["supported_one_ring_spike"]}`。'
      '完整全网格统计也保留，未观测边界不混同于稳定实测区。', '',
      '统一几何图：相同正交视角 0/±55°、相同光源和灰材质，仅面法向平面着色；'
      '仅逆转相机刚体变换，无形状配准、无纹理、无平滑法向。额头、面颊与下颌波纹改善来自几何；'
      '鼻尖、唇缘、下颌结构保留。', '',
      '![基准与默认方案](../runtime/validation/default_smooth_fast/final_geometry.png)', '',
      '## 同帧深度一致性与覆盖', '',
      '20 帧实际观测数组逐项与历史 prepared 比较一致，sequence.json 仅描述信息不同。'
      '每帧固定 seed=20261101、6000 点、face_mask 且 depth>0，首个正向命中，沿光轴误差，'
      '|残差|≤30 mm，缺失计入 coverage。按同一历史训练/留出划分和每帧自身网格评价；'
      '关键区域按最近观测关键点固定。此为深度一致性，并非独立扫描精度。', '',
      '| 视角/状态/角色 | 基准 MAE | 新版 MAE | ΔMAE mm | 基准 P95 | 新版 P95 | 基准 coverage | 新版 coverage |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        b = bg[g['view'], g['state'], g['role']]
        lines.append(f'| {g["view"]}/{g["state"]}/{g["role"]} | {b["mae_mm"]:.4f} | {g["mae_mm"]:.4f} | '
          f'{g["delta_mae"]:+.4f} | {b["p95_mm"]:.4f} | {g["p95_mm"]:.4f} | {b["coverage"]*100:.4f}% | {g["coverage"]*100:.4f}% |')
    lines += ['', f'关键区域全部分组最大 ΔMAE：{max(g["delta_mae"] for g in regions if g["region"] in ("nose","lips","jaw_contour")):+.6f} mm（预算 +0.1 mm）。'
      '鼻、唇、下颌逐组明细见 `quality_final/regions_by_view.md` 与 `acceptance.json`（包括张口）。', '',
      f'最终 Loop 安全检查：`{check}`；目标 {loop["target_blend"]}，实际 {loop["blend"]}，'
      f'回退原因 `{loop["fallback_reason"]}`。', '', '## 同批交错耗时', '',
      '同一 RTX 3060 Laptop、同一环境、同一个 prepared 绝对路径；串行交错各 3 次，'
      '不并行运行其他重建或重型评测。外部墙钟从启动公共入口到进程退出，含导入、资产检查、环境记录、'
      '重建与导出；两组均不含采集、预处理或本报告评测。run_report 耗时单列。', '',
      '| 次数 | joint-fast 外部 s | 新默认外部 s | joint-fast run_report s | 新默认 run_report s |', '|---|---:|---:|---:|---:|']
    records = timing['records']
    for i in range(1, 4):
        b = next(r for r in records if r['label']=='joint_fast' and r['repetition']==i)
        f = next(r for r in records if r['label']=='default' and r['repetition']==i)
        lines.append(f'| {i} | {b["wall_s"]:.3f} | {f["wall_s"]:.3f} | {b["run_report_s"]:.3f} | {f["run_report_s"]:.3f} |')
    for label in ('joint_fast', 'default'):
        t = timing[label]
        lines += ['', f'{label} 外部中位数 {t["median_s"]:.3f} s；范围 {t["min_s"]:.3f}–{t["max_s"]:.3f} s，幅 {t["max_s"]-t["min_s"]:.3f} s。']
    lines += ['', '历史 249.8 s 含约 37.8 s 预处理；与 194.6 s prepared 复用结果可比的历史值约 '
      '212 与 195 s，不能宣称节省 55 s。本报告速度结论只使用上表同批实测。', '',
      '基线是修改前源码快照（0.4.0）的 full joint-fast：240 步、fast、mid_rounds=1、'
      'postfit=full、support=0、coverage=0、anchor_prior=0，旧 Loop 自动安全回退。'
      '源码与第三方依赖 69 项 SHA256 见 `source_before_sha256.json`，所有实际命令、工作目录、阶段日志、'
      '退出状态见 `interleaved_v1/timings.json`。历史原提交不在此非 Git 目录中；快照与历史记录有 7 个模块字节不同，'
      '因此这是可追溯的 full joint-fast 重放基线，不冒称历史源码逐字复原；质量参照始终是保留的 joint_fast_v3。', '',
      f'相对本批 full joint-fast 重放产物，新默认最差的分组 ΔMAE/ΔP95 与 coverage 差值：`{replay_comparison}`。'
      '本批基线的逐视角深度原始表也包含在 `quality_final/multiview.json` 中。', '',
      f'新方案三次源码哈希完全相同：{same_code}；报告与 manifest 实际配置一致：{reports_agree}；'
      f'重复运行顶点最大差值 m：{repetition_differences}。', '', '## 交付与复核', '',
      f'归档基线 SHA256 全部保持一致：{frozen_hashes_ok}；当前运行源码与计时版本一致：{current_source_match}。', '',
      f'- 闭口：`{final_run / "delivery/head_closed_m.obj"}`',
      f'- 张口：`{final_run / "delivery/head_open_m.obj"}`',
      f'- 实际参数：`{final_run / "delivery/manifest.json"}` 与 `run_report.json`。',
      '- 原始采集、资产、历史输出保留；全部新实验位于 `runtime/validation/default_smooth_fast`。',
      '- 工具：`validate_smooth_fast.py`（保存控制网格直接重生成、固定域与局部统计）、'
      '`eval_multiview.py`、`render_smooth_comparison.py`、`benchmark_smooth_fast.py`、`report_smooth_fast.py`。',
      '- 回归与公共包装入口烟雾测试结果见同目录 `verification.txt`。', '',
      '### 筛选实验索引', '', '| 实验 | 闭口平均 RMS | 张口平均 RMS | 闭口平均 P95 | 张口平均 P95 |', '|---|---:|---:|---:|---:|']
    seen = set()
    for path in sorted(WORK.glob('*smoothness.json')):
        for entry in read(path):
            name = Path(entry['run']).name
            if name in seen: continue
            seen.add(name)
            cf = [f for f in entry['frames'] if f['state']=='closed']
            of = [f for f in entry['frames'] if f['state']=='open']
            lines.append(f'| {name} | {np.mean([f["rms_deg"] for f in cf]):.6f} | {np.mean([f["rms_deg"] for f in of]):.6f} | '
              f'{np.mean([f["p95_deg"] for f in cf]):.6f} | {np.mean([f["p95_deg"] for f in of]):.6f} |')
    (ROOT / 'docs/DEFAULT_SMOOTH_FAST_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in acceptance.items() if k not in ('groups','regions')}, indent=2))
    return 0 if ok else 3


if __name__ == '__main__':
    raise SystemExit(main())
