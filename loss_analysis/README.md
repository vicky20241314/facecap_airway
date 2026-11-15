# loss_analysis — 本地 loss 分析工具

> 该模块**不属于**主重建算法。主项目 `head_rgbd` / `core` 不导入它，启动流程不依赖它，
> 它也不在主算法必装依赖里。它做三件事：**读一次已经跑完的重建，把源码里真实存在的
> 优化目标按档位记录下来，并给出独立诊断**；**逐阶段报告当前值距终止目标还差多远，
> 并把主导 gap 映射到 ≤2 个白名单参数**；以及在此之上**离线**试跑少量白名单参数候选，
> 按分层终止目标排序。每次分析或优化都会生成一份单文件 HTML 报告。
>
> 本目录及其输出**只保留在本机**，不进入 GitHub。

---

## 1. 先说清楚：这些数字分别是什么

工具把四种性质完全不同的量**分档记录，绝不合并成一个 "total loss"**：

| 档位 | 含义 | 是否等于优化器在最小化的那个量 |
|---|---|---|
| `solver_objective` | 优化器某一步实际最小化的标量（`fit_c` / `motion_v2` 的 Adam loss） | 是 |
| `linearised_objective` | 冻结对应关系下的线性系统值（`solve_field` 的 `objective_before/after`） | 是，但只在该线性化下成立 |
| `nonlinear_energy` | 刷新对应关系后重新射线求交得到的固定射线 MSE | **不是**求解器目标，是接受判据 |
| `diagnostic` | 本工具自己算的 MAE / RMSE / P95 / 覆盖率 | **不是算法 loss** |

`neck_direct` 沿用同一套档位：`fit_unified` 每轮 `energy`（及其 `data/bend/tangent/anchor/small`
分项）是 `solver_objective`；solve 结束时的目标场残差（vertex/quad RMS）是 `diagnostic`；
有符号折痕、保护、拓扑、密度是**验收门（布尔）**，不会被折算成 1/0 的假 loss。

另外三条硬规则：

1. **不产出跨阶段 total loss**。`fit_c`、`motion_v2`、`solve_field`、Loop 混合各有各的目标，
   阶段之间没有共同总目标，源码里也不存在把它们加起来的量。
2. **缺失就是 `null`**。拿不到的值写 `null` 并在 notes 说明原因，**绝不填 0**。
   零有效样本时 `value=None, status="no_valid_samples"`。
3. **没有真实迭代状态就不画曲线**。例如 Loop 基面混合只有可行性试步、没有目标值，
   该阶段只会出现在"缺口面板"里，不会用最终网格反推一条假曲线。

---

## 2. 各阶段优化目标（来自实际源码）

完整表格由 `python -m loss_analysis stages` 打印，`objectives.py` 里边一项一条 `file:line`。
摘要如下。

### 2.1 `fit_c` — 共享身份拟合（Adam）

- **参数**：`shape(1×50)`、`expression(n×30)`、`jaw(n×3)`、`rotation(n×3)`、`translation(n×3)`
- **阶段切分**：按 `--steps` 的 `np.rint([0,.167,.458,.625,.833,1])` 切成 5 个阶段；
  身份阶段把 `expr` / `jaw` 梯度置零，因此这两项**只改变目标值、不产生更新**
- **分项**（权重为常数，`frame_weight = 该帧视角权重 / Σ视角权重`）：

| 分项 | 归一化 | 权重 | 分母 |
|---|---|---|---|
| 稳定标点 XYZ | `/0.01 m` | `frame_weight·landmark_weight·1.0` | 该帧有效标点数 |
| 内唇标点 XYZ | `/0.01 m` | `frame_weight·0.25` | 有效内唇点数 |
| 标点二维重投影 | `/10 px` | `frame_weight·landmark_weight·0.15` | 有效 2D 标点数 |
| 正面口部 2D | `/10 px` | `frame_weight·0.35` | 有效口部点数 |
| 闭口 hinge | `relu(gap−2mm)/3mm` | `frame_weight·0.15` | 3 对视点 |
| 深度射线法向 | `residual_mm/10mm` | `frame_weight·1.4` | 该帧有效对应数 |
| shape 先验 | `shape²` | `0.002` | 系数个数 |
| expression 先验 | `expr²` | `0.002` | **`n_frames×10`（不是系数个数）** |
| jaw 先验 | `jaw²` | `0.01` | `n_frames×3` |
| 时间平滑 | 系数差² | `(0.033/dt)²` 再乘 `0.005/0.02/0.01` | 时间边数 |

- 鲁棒核：`smooth_l1(beta=0.3)`，作用在**已归一化**的残差上。
- expression 先验的分母是 `n_frames×10`，与 shape 的 `mean` 不对称——这是源码实际写法，
  工具**如实记录，不做"修正"**。

### 2.2 `motion_v2` — 逐帧姿态/表情/下颌（Adam，身份固定）

- **参数**：`dr(rad)`、`dt(m)`、`de`、`dj(rad)`；硬上限 `|dr|≤0.02`、`|dt|≤3mm`、`|de|≤0.2`、`|dj|≤0.03`
- **分项**：射线法向 `/2mm` 权重 `1.0`；标点 XYZ `/3mm` 权重 `0.12`；
  表情阶段 2D `/3px` 权重 `0.08`；闭口 hinge 权重 `0.1`；
  姿态先验 `(dr/0.02)²`、`(dt/0.003)²` 权重 `0.01`；表情/下颌先验 `(de/0.2)²`、`(dj/0.03)²` 权重 `0.02`；
  时间平滑 `0.002`
- **所有数据项都除以 `n_frames`**，而不是各自的样本数
- **接受判据不是比较 loss**：每帧在**原始固定射线**上重打分，
  `mse_after ≤ mse_before + 1e-8` 且覆盖不减少才接受，否则沿 `1,½,¼,⅛,0` 阶梯退到`保留旧运动`

### 2.3 `solve_field`（`identity_v2` / `fit_d`）— 约束加权最小二乘 / QP

- **参数**：每个顶点一个标量场 `f`，单位**毫米**，沿动画法向
- **目标**：`‖rw·(A f − b)‖² + ‖R f + bias‖²`，其中
  `rw = sqrt(w / max(1, |A f − b| / 2))`（IRLS 风格），
  `w = 面积求积 × 帧权重 × 入射角² × 距离锥 × (noise_floor/σ)² / σ_proj²`
- **正则**：research-v2 用 FEM 总形变能（惩罚**曲率变化**而非面部曲率）+ 标量梯度；
  密集层再加 `dense_prior_strength = 2.0`
- **约束**：`|f| ≤ 3mm`、坡度 ≤0.45、有向面积 ≥5%、法向变化 ≤35°
- **接受判据**（三重）：
  1. 几何安全；
  2. 候选线性目标 ≤ 当前 + `1e-12`；
  3. research-v2 下，固定射线 **覆盖不减少** 且 `mse ≤ before×1.02 + 0.002`（strict-coverage）
- **关键区别**：第 3 条是**另行射线求交**得到的非线性能量，**不在**求解器二次型里。

### 2.4 `fit_d.loop_base` — Loop 基面混合

**没有目标函数**，只有 20 次折半的**可行性**搜索（几何安全）。因此该阶段在工具里只有
"试步记录/缺口"，不存在 loss 曲线。

---

## 3. 目录与接口

```
loss_analysis/
├─ README.md                      本文件
├─ requirements-analysis.txt      仅分析侧依赖（主算法不需要）
├─ loss_analysis/
│  ├─ objectives.py               阶段/分项目录，逐条标注源码行、权重、分母
│  ├─ terms.py                    纯函数：每个分项的计算 + 零样本/无效值处理
│  ├─ recorder.py                 迭代记录（四档位）+ CSV/JSON 写入
│  ├─ adapter.py                  只读读取已完成运行目录
│  ├─ diagnostics.py              固定观测射线误差、标点重投影、支持度统计
│  ├─ plots.py                    曲线（仅真实记录）+ 中文报告（Markdown）
│  ├─ html_report.py              单文件 HTML 报告（内联样式、图 base64 内嵌、无外链）
│  ├─ neck_adapter.py             只读读取 loss_direct.json / 验收门 / 标记语义
│  ├─ gap_report.py               逐阶段终止 gap（求解器层 + 验收层）+ 定向候选映射
│  ├─ converge.py                 逐阶段/phase 收敛平台、阶段内 gap 与均摊省时账本
│  ├─ optimizer.py                离线白名单候选：分层终止目标 + 硬约束 + 字典序
│  └─ cli.py                      独立入口
└─ tests/                         85 项数学与接口契约测试（构造数据）
```

### 3.1 主要函数

| 函数 | 作用 | 关键约定 |
|---|---|---|
| `terms.landmark_term / uv_term / mouth_gap_term` | 标点、重投影、闭口 hinge | 无效/NaN 被掩码排除；零样本返回 `None` |
| `terms.expression_prior_fit_c` | fit_c 特有的 `/ (n_frames×10)` 先验 | 与源码一致，不做"对称化" |
| `terms.field_equation_term` | `‖rw(Af−b)‖²` | 有 `field_mm` 时算残差，否则明确标注为"目标不自洽" |
| `terms.fixed_ray_energy_mm2` | 固定射线 MSE（缺失射线加罚 8mm/32mm） | 射线变少**不可能**变好 |
| `terms.strict_acceptance` / `motion_frame_acceptance` | 复刻两条接受规则 | 纯判定，返回布尔与原因 |
| `terms.metric_summary / coverage_summary` | MAE/RMSE/P95/覆盖率 | 零样本 → `null` |
| `recorder.Recorder` | 记录 + 写 `records.json` / `records.csv` | 同路径二次写入会 `FileExistsError` |
| `html_report.build_analysis_html` | 由同一 payload 生成单文件 HTML 报告 | 只用标准库；动态文本转义；`null` 不写 0；无外部资源 |
| `neck_adapter.ingest_neck_direct` | 读 `loss_direct.json` 记录每轮能量与残差 | 能量=求解器目标；残差=诊断；缺文件记缺口 |
| `neck_adapter.assess` | 组装基重建/颈部/下游的终止评估 | 验收门保持布尔，不折算成数值 loss |
| `gap_report.compute_gaps` | 逐阶段当前值、终止目标、剩余/超出量 | 求解器层与验收层分开；正 gap=余量，负 gap=超出；缺失为 `null` |
| `gap_report.targeted_keys` | 主导 gap → ≤2 个白名单参数 | 连接折痕按 full/B 区域选择 A 侧或桥接侧；不由白名单控制的门不硬凑参数 |
| `optimizer.plan_candidates` | 白名单单参数计划（≤12） | 只读基线值；不执行、不改默认参数 |
| `optimizer.targeted_candidates` | 由 gap 报告生成 ≤2 个定向候选 | 单参数、同向、白名单内；不扫全量 |
| `optimizer.run_optimizer` | 计划/执行候选 + 硬约束 + 字典序 + HTML | 默认不执行；输出目录不覆盖 |

### 3.2 记录字段（`records.json` / `records.csv`）

`stage, iteration, tier, phase, value, terms, weights, raw, n_valid, denominators,
decision, step_scale, coverage, source, notes`

- `decision`：`initial | candidate | accepted | rejected`
- 一次迭代可能同时有 before/after 两条，绘图时**按键位与决策拆成独立序列**，
  不会连成锯齿假振荡
- 缺口记录：`kind="gap", value=null, reason=...`

---

## 4. 依赖

Python 3.10（复用主项目 `runtime/env` 即可）。分析侧实际用到：

```
numpy>=1.24
matplotlib>=3.7      # 绘图
trimesh>=4.0         # 深度诊断的射线求交（仅在 --sequence 时）
pytest>=7            # 测试
```

安装（可选，通常直接复用主环境）：

```pwsh
D:\proiect\facecap_airway_delivery\runtime\env\python.exe -m pip install -r requirements-analysis.txt
```

**不引入 torch / CUDA / 相机 SDK**：分析不加载模型、不跑重建。

---

## 5. 使用

所有命令都要求**显式传入路径**。没有默认采集、没有默认运行目录、不会自动搜索。

```pwsh
cd D:\proiect\facecap_airway_delivery
$env:PYTHONPATH = "D:\proiect\facecap_airway_delivery\loss_analysis"
$py = "D:\proiect\facecap_airway_delivery\runtime\env\python.exe"
```

### 5.1 看阶段目录

```pwsh
& $py -m loss_analysis stages
& $py -m loss_analysis stages --json          # 机器可读
```

### 5.2 读一个已完成运行（只读）

```pwsh
& $py -m loss_analysis inspect --run runtime\outputs\<run>
```

输出：是否存在 `run_report` / `manifest`、各阶段日志里的真实迭代条数、有哪些阶段目录、
是否已准备 `sequence.json`。

### 5.3 完整分析（将来传入真实数据用这个）

```pwsh
& $py -m loss_analysis analyze `
    --run      "D:\...\runtime\outputs\<run>" `
    --sequence "D:\...\runtime\outputs\<run>\prepared\sequence.json"
```

- `--sequence` 可选；**不给就不做深度诊断**（并记录为缺口），不会去猜。
- `--neck-result` 可选，指向一次 `rebuild` 的 `result/`；给出时读取颈部
  `loss_direct.json`（每轮能量与目标残差）和 `quick_check.json` 验收门（布尔，不折算成 loss）。
  不给则把 `neck_direct` 记为缺口。
- `--mesh-dir` 默认 `<run>\delivery`，里面应有 `head_closed_m.obj` / `head_open_m.obj`。
- `--output` 不给时写到 `runtime\loss_analysis\<时间戳>_<label>\`；**已存在则直接报错，不覆盖**。
- `--no-plots` 只出 CSV/JSON/报告。
- `--max-depth-mm` 默认 30（与算法 30mm 查询同量级）。

输出：

```
<output>/
├─ records.json     四档位迭代记录（含缺口）
├─ records.csv      长表：一行一个 (stage, iteration, tier, term)
├─ summary.json     本次分析摘要
├─ report.md        中文报告（含公式来源与结论边界）
├─ report.html      单文件中文报告（样式与图内嵌，可离线单独分发）
└─ figures/
   ├─ <stage>.png           每个阶段一张，按档位/决策分线
   ├─ _gaps.png             缺口面板：哪些阶段没有真实状态、为什么
   └─ _diagnostics.png      独立诊断 MAE/RMSE/P95（非算法 loss）
```

`report.html` 与 `report.md` 由同一份 payload 生成，内容一致：档位图例、阶段目录、
逐阶段序列摘要、逐条记录（含 `null` 与决策）、缺口表、诊断表、图、结论边界。
它**不引用任何外部资源**（无 CDN、无外链、无 side-car 图片路径），图以 base64 内嵌，
因此单个 `.html` 文件即可离线打开或转发；每次 `analyze` 都会生成。

### 5.4 逐阶段终止 gap（`gap`）

```pwsh
& $py -m loss_analysis gap `
    --run        "D:\...\runtime\outputs\zp_capwitharuco" `
    --neck-result "D:\...\runtime\outputs\zp_capwitharuco_neck_delivery_t5\result" `
    --trace      "D:\...\runtime\outputs\_loss_neck_trace\neck_direct_v2_prior" `
    --landmarks  "D:\...\zp_capwitharuco_neck_delivery_t5\neck_landmarks.json"
```

- **求解器层**：`fit_c` / `motion_v2` 只报告步数预算消耗（源码没有收敛阈值，不发明阈值）；
  `solve_field` 用 fixed-ray 接受带；`neck_fit.fit_unified` 用步长/能量/梯度阈值与 `stop_reason`。
- **验收层**：连接 20° 全边集、保护 0.002 mm、上颈密度、拓扑/两态面表、拆分预算、
  标记语义、交付 sidecar/manifest 自包含。布尔值保持布尔，缺失写 `null`。
- **主导 gap**：先取硬性失败（按检测优先级），否则取最小相对余量；映射到 ≤2 个白名单参数。
  连接折痕先看 delivered relief 是否高于参考峰值：高于时首选 `structure_gain`（全局目标幅值），
  否则按 full 与 B 的区域关系选择 A 侧（`anchor_near_A` / `seam_taper_ratio`）或桥接侧
  （`mu_t_bridge`）；保护、密度、拓扑等不由白名单控制的项**不生成候选**。
- 输出：`gaps.json`、`gaps.csv`、`summary.json`、单文件 `report.html`；只读，不启动重建。
- 没有 `--trace` / `loss_direct.json` 时，颈部求解器轨迹记为缺口，**不会**从最终网格反推曲线。

### 5.5 逐阶段收敛与省时账本（`converge`）

```pwsh
& $py -m loss_analysis converge `
    --run "D:\...\runtime\outputs\<run-1>" `
    --run "D:\...\runtime\outputs\<run-2>" `
    --reference-run "D:\...\runtime\outputs\<reference>" `
    --persist-steps 60 --epsilon 0.002
```

- 读 `run_report.json` 的 `elapsed_s`、各阶段 `logs/*.log` 的每 20 步 objective、
  dense `report.json` 的 `field_logs`；只读，不启动重建。
- **参考**：同一次运行、同一 phase 内的最终记录值。跨 phase/跨阶段的 objective
  不可比，差值记为缺口，不相减。
- **平台判据**：从候选切点到该 phase 结束至少 `--persist-steps` 步，且此间相对
  剩余潜力 ≤ `--epsilon`。候选只是切点，不是源码收敛阈值。
- **耗时**：阶段 `elapsed_s` / 记录步数（均摊估计，非实测每步）。
- dense 只有少量 outer、时间主要在对应/射线/导出，只入账本，不出切点；prepare 与
  颈部同为只读账本。跨阶段共同几何诊断因没有每阶段网格，记为缺口。
- 输出：`convergence.json` / `convergence.csv` / `candidates.csv` / `report.md` /
  `report.html` / `summary.json` / `figures/`。没有候选时 CSV 只有表头，不补零。

---

## 6. 离线优化器（`optimize`）

只做两件事：**读基线** + **可选地重跑白名单候选**。默认只生成计划；`--execute` 才执行。
它不改变默认参数、接受条件、标记语义或下游算法；候选写入新目录，不覆盖基线。

```pwsh
& $py -m loss_analysis optimize `
    --baseline-result "D:\...\zp_capwitharuco_neck_delivery_t5\result" `
    --baseline-dense  "D:\...\zp_capwitharuco\reconstruction\dense" `
    --baseline-neck   "D:\...\zp_capwitharuco_neck_delivery_t5\neck_direct_v2_prior" `
    --baseline-trace  "D:\...\runtime\outputs\_loss_neck_trace\neck_direct_v2_prior" `
    --landmarks       "D:\...\zp_capwitharuco_neck_delivery_t5\neck_landmarks.json" `
    --capture         "D:\...\runtime\captures\20261108_164853_124297_head"
```

- 不给 `--candidate` → 按白名单生成单参数单向计划（≤12）；给 `--candidate KEY=VALUE` 可显式指定。
- `--targeted` → 先算逐阶段 gap，只从主导 gap 映射出 ≤2 个单参数候选；主导 gap 不在白名单内时
  **不生成候选**（不会为了凑数扫全量）。
- 计划不执行任何重建；加 `--execute --max-candidates 1` 才跑 1 个候选。
- 默认候选都走 `--capture`（基重建 → prepared → T3 → C → T5）。只改颈部参数时可用
  `--neck-only-dense` 复用基线 dense，但新目录没有 `prepared/`，T3 会跳过、交付标 not_usable，
  因此这只是省时间的降级选项，报告会按硬约束拒绝它。
- `--freeze-landmarks` 会把基线 sidecar 直接转发给候选；**默认关闭**，让候选重跑 live
  prepared 复检，否则"标记语义不变"的硬约束检查会变成空检查。
- `--baseline-trace` 指向一个带 `loss_direct.json` 的颈部目录（例如 `_loss_neck_trace/neck_direct_v2_prior`）；
  旧基线没有该文件时残差记为缺口，只在候选之间比较。

**白名单（9 个，每个候选只改一个）**

| 键 | 默认 | 方向 | 目的 |
|---|---|---|---|
| `neck.structure_gain` | 1.10 | −0.05 | 检验连接折痕是否跟随 over-relief |
| `neck.mu_t_bridge` | 0.15 | +0.05 | 桥接切向先验，抑制 B/C 交叉剪切 |
| `neck.anchor_near_A` | 2.00 | +0.50 | 固定头附近的手递刚度（最大折痕在 A 旁） |
| `neck.seam_taper_ratio` | 2.50 | +0.50 | 更宽的度量过渡 |
| `neck.mu_t_body` | 0.50 | +0.10 | 颈体切向先验 |
| `neck.bridge_width_mm` | 12.0 | +1.0 | 更宽的桥接区 |
| `base.motion_lowerface_weight` | 0.50 | +0.10 | open 下面部深度 |
| `base.bending_length_mm` | 8.0 | +1.0 | 更硬的稠密弯曲先验 |
| `base.low_bending_length_mm` | 8.0 | +1.0 | 更硬的低频弯曲先验 |

**排序**：硬约束（标记语义逐态不变、保护、拓扑与两态面表、上颈密度、累计拆分预算、交付自包含）全部
通过后，按字典序比较：基重建固定射线 MSE/缺失 → 颈部连接折痕 → 目标残差 vertex/quad
→ 结构与密度。层间不相加，低层不能补偿高层。

输出：

```
<output>/
├─ plan.json           基线读数 + 候选计划
├─ candidates.json     执行后的评估（硬约束、排名、残差）
├─ candidates.csv      一行一个候选
├─ summary.json        摘要（executed / ranked / winner / notes）
├─ report.html         单文件中文报告（规则、基线、计划、结果、硬约束矩阵、轨迹图）
└─ candidates/
   └─ c01_<key>/       每个候选一个全新目录（run.log + 完整 rebuild 产物）
```

---

## 7. 没有参数时会怎样

| 输入 | 行为 |
|---|---|
| 无任何参数 | 打印帮助并提示 `no input supplied`，退出码 0 |
| `inspect --run <不存在的路径>` | 输出 `{"status":"error", ...}`，退出码 2 |
| `analyze` 缺 `--run` | argparse 直接报 `the following arguments are required: --run`，退出码 2 |
| 输出目录已存在 | `FileExistsError`，退出码 3，不覆盖 |

---

## 8. 与主算法的隔离（已落实）

- **未修改任何主项目已跟踪源码**；`head_rgbd` / `core` 不认识本模块。
- 分析侧状态全部通过**只读读取已落盘产物**获得
  （阶段日志里的 objective 行、各 stage 的 `report.json`、`loss_direct.json`）。
- 唯一新增的主项目侧写入是 `tools/build_neck_direct_v2.py` 额外落盘 `loss_direct.json`：
  只记录求解器已经算出的每轮能量/残差，不改变求解、参数、采样、种子、接受条件或几何保护。
- `optimize --execute` 会以子进程调用正常的 `headrgbd rebuild` 入口，候选写进全新目录；
  分析模块不被主算法导入，主算法也不读取优化器的结论。
- 不改变默认参数、默认接受条件、标记语义；不自动把最优候选写回主项目。
- 分析本身不启动重建、不占用 GPU；只有显式 `--execute` 才会启动候选重建。

### 8.1 已接通 / 尚未接通

| 项目 | 状态 |
|---|---|
| `fit_c` 每步 solver objective | ✅ 可从 `logs/*.log` 读取 |
| `motion_v2` 每步 solver objective | ✅ 可从 `logs/*.log` 读取 |
| `fit_c` / `motion_v2` **分项拆分** | ❌ 运行未持久化分项，记为 gap（需插桩才能得到，本模块不插桩） |
| `solve_field` 线性化目标（before/after） | ✅ 从 dense 阶段 `report.json` 的 `field_logs[*].history` |
| 固定射线非线性能量 + 缺失射线数 | ✅ 同上（`fixed_ray_diagnostic`） |
| 几何安全判定 | ✅ 同上（`safety`） |
| `neck_direct` 每轮 solver objective 与分项 | ✅ 从新构建的 `loss_direct.json` 读取（旧构建记为 gap） |
| `neck_direct` 目标场残差 | ✅ `loss_direct.json`（诊断档，不进能量） |
| 颈部验收门（保护/拓扑/连接/密度） | ✅ 读 `result/quick_check.json`、`verification.json`（布尔，不折算成 loss） |
| `identity_v2` / `front_anchor` / `guarded` 的 `field_logs` | ❌ 这些 `report.json` **不含** `field_logs`，记为 gap |
| `loop_base` 目标 | ❌ 算法本身没有目标，只有可行性试步 |
| 深度误差 MAE/RMSE/P95/覆盖 | ✅ 给出 `--sequence` 时计算（构造数据已验证接口，真实数据待你指定） |
| 标点重投影诊断 | ✅ 函数就绪，需 `sequence` + 网格 + `reference_landmarks.json` 三者同时提供 |
| 支持度标签统计 | ✅ 读 `delivery/support.json` |
| 下游输入接口探针 | ✅ `optimize` 读 final/measured 的 sidecar/manifest 结构（不做配准） |
| 逐阶段终止 gap | ✅ `gap` 读日志/loss_direct/quick_check；求解器层与验收层分开，缺失记缺口 |
| gap → 参数定向 | ✅ 连接折痕先看 relief 是否高于参考，再按 full/B 区域选第二参数；保护/密度/拓扑等不硬凑参数 |

---

## 9. 测试与入口验证（本轮实际结果）

```pwsh
cd D:\proiect\facecap_airway_delivery\loss_analysis
..\runtime\env\python.exe -m pytest tests -q
# 72 passed
```

覆盖：平滑 L1 两分支、已知 2mm/10mm 偏差、权重线性、无效标点排除、索引窗口外数据被忽略、
零有效样本为 `null`、2D 归一化、hinge 死区与公式、`/(n_frames×10)` 先验、线性系统二次缩放、
固定射线缺失加罚、射线变少不可能变好、两条接受判据、四档位不混、缺口记录、
CSV/JSON/HTML 格式、HTML 单文件不外链且转义动态文本、图以 base64 内嵌、
颈部 solver 分项与残差诊断不混档、旧构建缺 `loss_direct.json` 记缺口、
优化器计划 ≤12 且默认不执行、硬约束拒绝标记漂移与保护失败、字典序不是加权和、
输出目录不覆盖、入口无参数/`--version`/`stages`/缺路径/缺 `--run`/缺 `optimize` 必填参数；
逐阶段 gap 的接受带余量、20° 连接失败与 full/B 区域定向、预算耗尽不等于收敛、
缺失 trace 记缺口、嵌套 sidecar 的 `main_delivery_mesh_file`、gap CSV/HTML 单文件。

**构造数据只证明计算与接口正确，不证明真实重建效果。**

入口实测（`login=false`，PowerShell）：

| 命令 | 结果 |
|---|---|
| `python -m loss_analysis` | 打印帮助 + `no input supplied`，退出码 0 |
| `python -m loss_analysis --version` | `lossana 0.1.0`，退出码 0 |
| `python -m loss_analysis stages` | 打印 10 个阶段与全部分项，退出码 0 |
| `inspect --run <不存在>` | 明确报错，退出码 2 |
| `analyze`（缺 `--run`） | 参数缺失报错，退出码 2 |
| `optimize`（缺基线路径） | 参数缺失报错，退出码 2 |

---

## 10. 将来传入真实数据的命令格式

```pwsh
Set-Location D:\proiect\facecap_airway_delivery
$env:PYTHONPATH = "D:\proiect\facecap_airway_delivery\loss_analysis"
& ".\runtime\env\python.exe" -m loss_analysis analyze `
    --run      ".\runtime\outputs\<你的运行目录>" `
    --sequence ".\runtime\outputs\<你的运行目录>\prepared\sequence.json" `
    --mesh-dir ".\runtime\outputs\<你的运行目录>\delivery" `
    --neck-result ".\runtime\outputs\<你的重建+颈部目录>\result" `
    --label    "<自定义标签>" `
    --max-depth-mm 30
```

要求：`<run>` 必须是一次**已完成**的 `headrgbd reconstruct` 目录
（有 `run_report.json`、`logs/`、`reconstruction/`）。若只有 `capture` 而无重建，
`analyze` 会把这些阶段全部记为 gap 并说明原因。

---

## 11. Git 与本地保留

**事实（已用 Git 命令核实，不是假设）**：

1. `D:\proiect\facecap_airway_delivery` **本身不是 Git 仓库**
   （`git rev-parse --is-inside-work-tree` → `fatal: not a git repository`）。
2. 最近的仓库是 `D:\proiect\withB-depth-hires-20261031`，其
   `--git-dir` 是 `D:/proiect/withB/.git/worktrees/withB-depth-hires-20261031`（`.git` 是**文件指针**，不是目录），
   `--git-common-dir` 是 `D:/proiect/withB/.git`。
3. 该仓库**把交付目录视为仓库之外**：
   `git check-ignore -v D:/proiect/facecap_airway_delivery/loss_analysis`
   → `fatal: ... is outside repository at 'D:/proiect/withB-depth-hires-20261031'`。

**因此**：

- 交付目录内**没有任何被 Git 跟踪的文件**：
  `git ls-files | grep -c loss_analysis` → `0`（`git ls-files -- <外部路径>` 只会返回同样的
  `fatal:` 提示，不是文件）。
- 已按"用 Git 命令查询真实位置、只追加不覆盖"的要求，向
  `D:/proiect/withB/.git/info/exclude` **追加**了两条规则（原内容完整保留）：

  ```
  loss_analysis/
  runtime/loss_analysis/
  ```

- 验证：`git check-ignore -v loss_analysis/` →
  `D:/proiect/withB/.git/info/exclude:12:loss_analysis/`（退出码 0，规则命中）。
  绝对路径形式会被 Git 以"在仓库之外"拒绝，这是该目录位置的**客观限制**，
  不是规则写错；一旦交付目录被移入任一 worktree，上述规则会立即生效。
- 已确认 `runtime/loss_analysis` 命中仓库既有 `.gitignore:7:runtime/`。

**本轮未执行 `git add` / `commit` / `push`，未创建 PR，未修改远程配置，未发布模块。**

---

## 12. 已知限制

1. `fit_c` / `motion_v2` 的**分项拆分**拿不到（运行只记总量），要拿到必须插桩，本模块刻意不做。
2. `identity_v2` 与 `front_anchor` / `guarded` 阶段的 `field_logs` 未落盘，只有 dense 阶段有。
3. 深度诊断只做**绝对深度差**（沿光轴），未包含法向/平面距离分项；也未做跨状态配准。
4. 支持度标签是整次运行累计的，**不能**按闭口/张口分别重算。
5. 曲线上的点只有 2–3 个时（因为该阶段只有 2–3 个 outer step），不构成趋势结论。
6. 构造数据测试**不验证**真实重建精度或算法优劣。
7. 旧的颈部构建没有 `loss_direct.json`；`--baseline-trace` 只能指向同参数重建的 trace，
   基线交付本身不重跑，报告会明确标注这一点。
8. 优化器第一轮只做单参数移动；一次只跑 1 个候选的冒烟，不等于全量搜索结果。
9. 候选的 `rebuild` 会重新拟合基重建；相同随机种子下应确定，但若上游环境变化，
   基重建层的数值移动会一并出现在候选里，排序的第一层正是为此保留。
10. 下游只做输入接口探针；气管几何、D1/D2 间隙与配准 pass/fail 不在优化范围内。

---

## 13. 结论边界

- 分析命令只观察、只统计、只报告；`optimize` 只在显式 `--execute` 时重跑候选，
  不写回默认参数、不自动采用最优候选。
- 四档数值**不可互相比较、不可相加**；跨阶段没有 total；分层终止目标之间也不能互相补偿。
- 硬约束失败或无法验证的候选不参与排名；不能用标记漂移、保护放松或拓扑变化换指标。
- "loss 下降"不等于"几何更准"；精度仍需独立扫描/人工轮廓验证。
