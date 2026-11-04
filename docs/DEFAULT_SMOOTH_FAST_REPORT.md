# 默认平滑快速重建验收（0.5.0）

状态：已完成。

1. **默认流程**：观测准备 → 正面初始化 240 步 → 正侧联合身份 240 步 → 一次运动校正 80 步 → 低频 2 轮 / 中频 1 轮 → 安全 Loop 目标 0.5 → 闭口与张口交付。
2. **光滑度达到 joint_fast_v3**：True；固定区域内全部 20 帧逐帧 RMS 与 P95 验收。
3. **深度误差和覆盖通过**：MAE True，关键区域 True，coverage True，几何安全 True。
4. **相同条件耗时不高于 joint-fast**：True；外部墙钟中位数 214.726 → 124.680 s。

## 实际默认值与入口

```json
{
  "device": "cuda",
  "steps": 240,
  "front_steps": 240,
  "shape_fusion": "joint",
  "fast": true,
  "chain": "lean",
  "postfit": "minimal",
  "mid_rounds": 1,
  "reuse_field": true,
  "information_measure": "barycentric",
  "support_min_information": 0.5,
  "coverage_allowance_ratio": 0.001,
  "loop_data_selection": false,
  "loop_target_blend": 0.5,
  "bending_length_mm": 8.0,
  "low_bending_length_mm": 8.0,
  "keyframes_per_segment": 3,
  "expression_components": 30,
  "dense_points": 5000,
  "points_per_frame": 1200,
  "loop_actual_blend": 0.5,
  "loop_fallback_reason": null,
  "motion_steps": 80,
  "fixed_ray_policy": "strict-coverage"
}
```

`python -m head_rgbd run` 与 `reconstruct` 共用 `_add_fit_options(load_defaults())`；`headrgbd.cmd` 与 `headrgbd.ps1` 原样透传 `-m head_rgbd`。正式计时中的新方案没有填写任何模式参数。默认不执行未消费的 fit_c 后处理、独立低频阶段、第二次运动、重复锚定和密集残差求解。`reuse-anchor` 仅在 full 对照链有效，不是 lean 的优化项。

120 步初始化在最终模板规则下复核，张口 coverage 97.4067%，基准 98.9367%，未采用。180、220 步是模板资格修复前的预算探索，未继续作为发布候选；保留已通过的 240 步，时间目标通过精简链满足。没有新增早停、缓存或求解器。

## 几何改动与选择依据

Loop 直接尝试目标 0.5，只有原位移、坡度、法向和防翻面检查失败才逐次减半，最后检查零混合；不安全的控制网格即便零混合也拒绝。报告记录目标、实际值、全部检查及回退原因。默认取消三候选深度打分，显式实验打分也逐候选检查安全。

关键修复是先验模板的证据资格与 barycentric 控制场一致：连续信息允许可靠单视角更新，Loop 模板也允许至少一帧真实支持。未观测控制点、解剖禁区及其边界继续固定。这只改变先验模板，未扩大优化自由集，未改变证据标签，不将平滑计为新增实测。中频已计算的支持与低频支持合并复用；本例该合并未新增两帧支持顶点，不将其算作质量收益。

先保持原设置比较中频弯曲 4/6/8 mm。恢复 Loop 后仅增大中频仍不足，10/12 mm 中频、低频 10/12/20 mm、梯度 6/12 mm、160 步运动和 180/220 步初始化也留下独立实验。最终在证据资格修复后重新比较 6/8：6 未通过，8 是已测试强度中最弱通过值。低频弯曲恢复原 8 mm、梯度恢复原 3 mm、运动恢复原 80 步。

FEM 惩罚相对原模型的向量曲率变化，保留原身份和解剖结构；没有整体收缩、切向自由度、曲率自由集扩展、新密集残差或新后处理。缺失射线 8 mm 惩罚、逐帧 coverage、2% + 0.002 mm² 数据验收未改变。

## 固定光滑度评价域

评价域仅由 joint_fast_v3 的重复实测支持与解剖掩码固定：observed_frames ≥ 2 且非解剖禁区，相邻两面的所有顶点均满足。4218 顶点、7837 面、11459 相邻面对。全部候选验证相同拓扑，每条共享边等权，单位度；逐帧用面法向，不使用显示平滑法向。

域哈希：`e0761ca8148a000da2ff4bfaff22548b56a9f4a627b75ad2123a7d33d92eeabb`。完整索引保存在 `runtime/validation/default_smooth_fast/evaluation_domain.npz`。原历史 5.95° 不直接套用；本报告所有基准数字均在本固定域重算。

| 状态 | 基准平均 RMS | 新版平均 RMS | 基准平均 P95 | 新版平均 P95 |
|---|---:|---:|---:|---:|
| closed | 5.913811 | 5.821471 | 12.268505 | 12.107229 |
| open | 5.775331 | 5.678570 | 12.303113 | 12.148147 |

表为逐帧统计的状态平均；验收逐帧执行，完整结果见 `final_smoothness.json`。

| 区域/状态 | 基准平均 RMS | 新版平均 RMS | 基准平均 P95 | 新版平均 P95 |
|---|---:|---:|---:|---:|
| forehead/closed | 3.2705 | 3.7310 | 6.7427 | 8.3165 |
| forehead/open | 3.3337 | 3.7848 | 6.7963 | 8.4841 |
| cheeks/closed | 5.4653 | 5.3627 | 10.9328 | 10.8109 |
| cheeks/open | 5.1847 | 5.0371 | 10.6277 | 10.2838 |
| jaw/closed | 5.1705 | 5.4344 | 12.3052 | 12.6348 |
| jaw/open | 5.7304 | 5.8893 | 13.5725 | 13.6242 |

局部区域按基准 canonical 坐标和眉、鼻翼、外唇关键点固定，定义见工具输出。

新版支持区相邻场差 RMS/P95/max：`{'rms_mm': 0.33690567599317445, 'p95_mm': 0.7599327566313294, 'max_mm': 1.7942286643172056}`；相对一环邻居均值的尖峰：`{'rms_mm': 0.15741693365200102, 'p95_mm': 0.350875711593477, 'max_mm': 0.8956155186057151}`。完整全网格统计也保留，未观测边界不混同于稳定实测区。

统一几何图：相同正交视角 0/±55°、相同光源和灰材质，仅面法向平面着色；仅逆转相机刚体变换，无形状配准、无纹理、无平滑法向。额头、面颊与下颌波纹改善来自几何；鼻尖、唇缘、下颌结构保留。

![基准与默认方案](../runtime/validation/default_smooth_fast/final_geometry.png)

## 同帧深度一致性与覆盖

20 帧实际观测数组逐项与历史 prepared 比较一致，sequence.json 仅描述信息不同。每帧固定 seed=20261101、6000 点、face_mask 且 depth>0，首个正向命中，沿光轴误差，|残差|≤30 mm，缺失计入 coverage。按同一历史训练/留出划分和每帧自身网格评价；关键区域按最近观测关键点固定。此为深度一致性，并非独立扫描精度。

| 视角/状态/角色 | 基准 MAE | 新版 MAE | ΔMAE mm | 基准 P95 | 新版 P95 | 基准 coverage | 新版 coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| front/closed/held_out_closed | 1.3449 | 0.9560 | -0.3890 | 3.8085 | 2.8949 | 99.8333% | 99.8167% |
| front/closed/shape_training | 1.3682 | 0.9896 | -0.3786 | 3.8744 | 2.8261 | 99.7167% | 99.7056% |
| front/open/open | 1.5813 | 1.2676 | -0.3137 | 4.5573 | 3.7765 | 98.9367% | 98.9900% |
| left/closed/held_out_closed | 1.2457 | 0.9980 | -0.2477 | 3.2824 | 2.9783 | 99.3417% | 99.5500% |
| left/closed/shape_training | 1.1674 | 0.9255 | -0.2418 | 2.9713 | 2.6379 | 99.5278% | 99.6556% |
| right/closed/held_out_closed | 1.1651 | 0.8741 | -0.2910 | 3.0343 | 2.5627 | 99.5167% | 99.5250% |
| right/closed/shape_training | 1.1194 | 0.7957 | -0.3238 | 3.0242 | 2.4470 | 99.6444% | 99.6667% |

关键区域全部分组最大 ΔMAE：-0.139007 mm（预算 +0.1 mm）。鼻、唇、下颌逐组明细见 `quality_final/regions_by_view.md` 与 `acceptance.json`（包括张口）。

最终 Loop 安全检查：`{'minimum_signed_area_ratio': 0.6752914687076297, 'maximum_normal_change_deg': 26.694787233055564, 'maximum_slope': 0.25672429899770294, 'maximum_total_displacement_mm': 1.7595492837364908, 'maximum_residual_mm': 0.0}`；目标 0.5，实际 0.5，回退原因 `None`。

## 同批交错耗时

同一 RTX 3060 Laptop、同一环境、同一个 prepared 绝对路径；串行交错各 3 次，不并行运行其他重建或重型评测。外部墙钟从启动公共入口到进程退出，含导入、资产检查、环境记录、重建与导出；两组均不含采集、预处理或本报告评测。run_report 耗时单列。

| 次数 | joint-fast 外部 s | 新默认外部 s | joint-fast run_report s | 新默认 run_report s |
|---|---:|---:|---:|---:|
| 1 | 200.241 | 137.193 | 198.961 | 136.031 |
| 2 | 224.532 | 124.680 | 223.196 | 123.309 |
| 3 | 214.726 | 123.740 | 213.549 | 122.566 |

joint_fast 外部中位数 214.726 s；范围 200.241–224.532 s，幅 24.291 s。

default 外部中位数 124.680 s；范围 123.740–137.193 s，幅 13.453 s。

历史 249.8 s 含约 37.8 s 预处理；与 194.6 s prepared 复用结果可比的历史值约 212 与 195 s，不能宣称节省 55 s。本报告速度结论只使用上表同批实测。

基线是修改前源码快照（0.4.0）的 full joint-fast：240 步、fast、mid_rounds=1、postfit=full、support=0、coverage=0、anchor_prior=0，旧 Loop 自动安全回退。源码与第三方依赖 69 项 SHA256 见 `source_before_sha256.json`，所有实际命令、工作目录、阶段日志、退出状态见 `interleaved_v1/timings.json`。历史原提交不在此非 Git 目录中；快照与历史记录有 7 个模块字节不同，因此这是可追溯的 full joint-fast 重放基线，不冒称历史源码逐字复原；质量参照始终是保留的 joint_fast_v3。

相对本批 full joint-fast 重放产物，新默认最差的分组 ΔMAE/ΔP95 与 coverage 差值：`{'mae_mm': -0.0647068553900716, 'p95_mm': -0.10104511816414297, 'worst_coverage_delta': -0.00033333333333340764}`。本批基线的逐视角深度原始表也包含在 `quality_final/multiview.json` 中。

新方案三次源码哈希完全相同：True；报告与 manifest 实际配置一致：True；重复运行顶点最大差值 m：[0.0, 0.0]。

## 交付与复核

归档基线 SHA256 全部保持一致：True；当前运行源码与计时版本一致：True。

- 闭口：`D:\proiect\facecap_airway_delivery\runtime\validation\default_smooth_fast\interleaved_v1\default_1\delivery\head_closed_m.obj`
- 张口：`D:\proiect\facecap_airway_delivery\runtime\validation\default_smooth_fast\interleaved_v1\default_1\delivery\head_open_m.obj`
- 实际参数：`D:\proiect\facecap_airway_delivery\runtime\validation\default_smooth_fast\interleaved_v1\default_1\delivery\manifest.json` 与 `run_report.json`。
- 原始采集、资产、历史输出保留；全部新实验位于 `runtime/validation/default_smooth_fast`。
- 工具：`validate_smooth_fast.py`（保存控制网格直接重生成、固定域与局部统计）、`eval_multiview.py`、`render_smooth_comparison.py`、`benchmark_smooth_fast.py`、`report_smooth_fast.py`。
- 回归与公共包装入口烟雾测试结果见同目录 `verification.txt`。

### 筛选实验索引

| 实验 | 闭口平均 RMS | 张口平均 RMS | 闭口平均 P95 | 张口平均 P95 |
|---|---:|---:|---:|---:|
| joint_fast_v3 | 5.913811 | 5.775331 | 12.268505 | 12.303113 |
| bend6 | 6.109469 | 5.943432 | 12.498814 | 12.416460 |
| bend8 | 5.977579 | 5.820333 | 12.160541 | 12.275337 |
| default_1 | 5.821471 | 5.678570 | 12.107229 | 12.148147 |
| default_2 | 5.821471 | 5.678570 | 12.107229 | 12.148147 |
| default_3 | 5.821471 | 5.678570 | 12.107229 | 12.148147 |
| loop_one_support_bend4 | 6.275188 | 6.113712 | 13.114349 | 13.010855 |
| front120_one_support | 5.804721 | 5.786532 | 12.157778 | 12.440557 |
| front120_bend8 | 5.952609 | 5.921965 | 12.262846 | 12.569614 |
| front180_low12_mid12 | 5.900350 | 5.769962 | 12.214979 | 12.497704 |
| front220_low12_mid12 | 5.906536 | 5.772796 | 12.259438 | 12.408542 |
| front240_bend10 | 5.952278 | 5.797363 | 12.158600 | 12.254524 |
| front240_bend12 | 5.952747 | 5.797460 | 12.155430 | 12.245166 |
| gradient6_bend8 | 5.967685 | 5.810368 | 12.145131 | 12.266451 |
| gradient12_bend8 | 5.951437 | 5.795139 | 12.142259 | 12.259063 |
| zhoupei_lite_C_20261102_133517 | 7.709141 | 7.493262 | 17.005011 | 16.933545 |
| C_loop05 | 6.434199 | 6.263221 | 13.242898 | 13.117934 |
| loop_one_support_bend8 | 5.821471 | 5.678570 | 12.107229 | 12.148147 |
| low10_mid10 | 5.931092 | 5.781378 | 12.172239 | 12.332787 |
| low12_mid12 | 5.919610 | 5.773732 | 12.213164 | 12.368665 |
| low20_mid20 | 5.920401 | 5.780800 | 12.283872 | 12.362095 |
| motion160_low12_mid12 | 5.945034 | 5.798636 | 12.233888 | 12.392419 |
| front120_bend10 | 5.924065 | 5.896867 | 12.228560 | 12.571182 |
| front120_bend12 | 5.924291 | 5.896720 | 12.229747 | 12.569858 |
| support_union_bend8 | 5.977579 | 5.820333 | 12.160541 | 12.275337 |
