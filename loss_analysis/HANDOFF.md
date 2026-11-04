# 交接文档：loss 计算任务 → 另一个 agent

交接时间：2026-11-01 15:45（GMT+8）
发起方：当前会话（负责**改进重建算法**，不是评估/loss 算法）
接收方：接手 **loss 计算与分析** 的 agent

---

## 0. 一句话交接

`loss_analysis` 模块的**实现与单元测试已完成**，但**尚未用你指定的真实数据跑过**。
本会话把这条线交出去，自己转向改进重建算法。接收方需要做的是：用真实采集跑通分析、
补上缺失的分项记录、把结论反馈给算法改进方向。

---

## 1. 背景与分工边界

| 角色 | 负责什么 | 不负责什么 |
|---|---|---|
| 本会话（交出的部分） | 重建算法本身的改进 | 评估、loss 指标、分析报告 |
| 接收方 | loss 计算、迭代记录、诊断指标、分析报告 | 改重建算法、改参数、改版本 |

**硬性边界（接收方也必须遵守）**：

- 主算法不得依赖分析工具：`head_rgbd/` 与 `core/` 里 **0 处** 引用 `loss_analysis`（已用 grep 核实）。
- 不修改主项目已跟踪源码；若必须要状态，只能**只读读取已落盘产物**。
- 不得为了拿到 loss 而改动目标函数、参数更新、采样、随机种子、迭代次数、接受条件、几何保护。
- 分析工具、环境、日志、图表、报告**只留本机**，不进 GitHub。
- 主项目版本保持 `0.2.0`，**不要升级**。

---

## 2. 当前真实状态（已核实，不是回忆）

### 2.1 目录

```
D:\proiect\facecap_airway_delivery\          ← 主项目（版本 0.2.0）
├─ head_rgbd/ core/ config/ tests/ tools/    ← 主算法，本次未改动
├─ runtime/
│  ├─ env/                                    ← Python 3.10.22 + torch 2.0.1(cu117)
│  ├─ assets/                                 ← FLAME / TUFA / UniFace / YuNet（7 个）
│  ├─ outputs/zhou_full_20261101/             ← 唯一一次完整真实重建（20 帧）
│  └─ loss_analysis/20261101_152049_.../      ← 分析输出（只读验证那次）
└─ loss_analysis/                             ← 本交接的模块（12 个文件）
   ├─ README.md
   ├─ requirements-analysis.txt
   ├─ loss_analysis/{__init__,__main__,objectives,terms,recorder,adapter,diagnostics,plots,cli}.py
   └─ tests/test_loss_analysis.py
```

### 2.2 验证状态

| 项目 | 状态 | 证据 |
|---|---|---|
| 单元测试 | ✅ **46 passed** | 见 §3 的运行方式（有陷阱） |
| 入口 `--version` | ✅ `lossana 0.1.0`，退出码 0 | |
| 入口无参数 | ✅ 打印帮助 + `no input supplied`，退出码 0 | |
| `stages` | ✅ 打印 9 个阶段全部分项，退出码 0 | |
| 缺 `--run` | ✅ 参数报错，退出码 2 | |
| 不存在路径 | ✅ `{"status":"error"}`，退出码 2 | |
| 只读接线（历史运行） | ✅ 95 条真实记录 / 7 个缺口 / 10 张图 | `runtime\loss_analysis\20261101_152049_zhou_full_20261101\` |
| **真实指定数据** | ❌ **未运行** | 用户尚未指定分析哪份采集 |

### 2.3 Git 事实（用命令核实）

- `D:\proiect\facecap_airway_delivery` **本身不是 Git 仓库**
  （`git rev-parse --is-inside-work-tree` → `fatal: not a git repository`）。
- 最近仓库：`D:\proiect\withB-depth-hires-20261031`，分支 `codex/facecap-airway-delivery-0.2.0`，
  HEAD `6bc4268`；其 `--git-dir` 是**文件指针** `D:/proiect/withB/.git/worktrees/...`，
  common-dir 是 `D:/proiect/withB/.git`。
- 该仓库判定交付目录**在仓库之外**，故其规则对交付目录不生效（客观限制）。
- `loss_analysis/` 与 `runtime/loss_analysis/` 已追加到
  `D:/proiect/withB/.git/info/exclude`，`git check-ignore -v loss_analysis/`
  → `info/exclude:12:loss_analysis/`（命中）。
- `git ls-files | grep -c loss_analysis` → **0**（无跟踪文件）。
- 本次交接**未执行** `git add/commit/push`，未建 PR，未改远程配置。

---

## 3. 如何运行（⚠️ 有坑，务必按此执行）

### 3.1 测试必须在模块目录下跑

```pwsh
cd D:\proiect\facecap_airway_delivery\loss_analysis
..\runtime\env\python.exe -m pytest tests -q        # → 46 passed
```

**坑**：在项目根目录执行 `pytest loss_analysis/tests` 会 **collection error**，
因为 `loss_analysis` 这个包名与目录名同名导致导入路径歧义。
坑在**运行位置**，不在代码；若要修，可在根目录加 `pyproject.toml` 或用 `-p no:cacheprovider`
配合 `PYTHONPATH`，但**本交接不建议改**（改了要重测）。

### 3.2 分析命令（PowerShell，login=false）

```pwsh
Set-Location C:\
$env:PYTHONPATH = 'D:\proiect\facecap_airway_delivery\loss_analysis'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8    # 否则中文乱码
$py = 'D:\proiect\facecap_airway_delivery\runtime\env\python.exe'

& $py -m loss_analysis stages
& $py -m loss_analysis inspect --run 'D:\...\runtime\outputs\<run>'
& $py -m loss_analysis analyze `
    --run      'D:\...\runtime\outputs\<run>' `
    --sequence 'D:\...\runtime\outputs\<run>\prepared\sequence.json'
```

`--sequence` 不给 → 不做深度诊断，并记为缺口（不会去猜）。
`--output` 已存在 → `FileExistsError`，退出码 3，不覆盖。

---

## 4. 已实现的能力（可直接用）

| 能力 | 入口 | 说明 |
|---|---|---|
| 分项计算 | `terms.landmark_term / uv_term / mouth_gap_term / quadratic_prior / expression_prior_fit_c / field_equation_term` | 纯函数，返回 `raw / weight / value / n_valid / denominator` |
| 固定射线能量 | `terms.fixed_ray_energy_mm2 / aggregate_fixed_ray` | 缺失射线加罚 8mm(strict)/32mm(accounted)，**射线变少不可能变好** |
| 两条接受判据 | `terms.strict_acceptance / motion_frame_acceptance` | 纯判定，返回布尔 + 原因 |
| 诊断指标 | `terms.metric_summary / coverage_summary` | MAE/RMSE/P95/中位/最大/覆盖；零样本 → `null` |
| 迭代记录 | `recorder.Recorder` | 四档位；写 `records.json` / `records.csv`，不覆盖 |
| 只读适配 | `adapter.ingest_solver_logs / ingest_skills / describe_inputs` | 读阶段日志 objective 行、`field_logs[*].history` |
| 深度诊断 | `diagnostics.depth_surface_error_mm / landmark_reprojection_px / support_label_summary` | 固定真实射线，缺失算缺失不算 0 |
| 绘图 | `plots.plot_stage_series / plot_gap_panel / plot_metrics` | before/after 拆线；无真实状态不画线 |
| 报告 | `plots.build_report` | 中文，四档位释义 + 源码依据 + 结论边界 |

### 4.1 四档位（核心约定，不可合并）

| 档位 | 含义 |
|---|---|
| `solver_objective` | 优化器实际最小化的标量（`fit_c` / `motion_v2`） |
| `linearised_objective` | 冻结对应下的线性系统值（`solve_field` before/after） |
| `nonlinear_energy` | 刷新对应后射线求交的固定射线 MSE（**不是**求解器目标） |
| `diagnostic` | 本工具自己的 MAE/RMSE/P95/覆盖（**不是**算法 loss） |

**不产出跨阶段 total loss**（各阶段无共同总目标）；缺失写 `null` 不写 0。

---

## 5. 各阶段目标速查（详细版见 `loss_analysis/README.md` §2）

| 阶段 | 参数 | 求解器 | 记录可得性 |
|---|---|---|---|
| `fit_c` | shape/expr/jaw/rot/trans | Adam，5 阶段切分 | ✅ 每步总量（日志）；❌ 分项 |
| `motion_v2` | dr/dt/de/dj | Adam + 硬上限 | ✅ 每步总量（日志）；❌ 分项 |
| `identity_v2` | 毫米标量场 | OSQP/lsq_linear | ❌ `report.json` 无 `field_logs` |
| `dense.*`（4 个子阶段） | 毫米标量场 | OSQP | ✅ 线性化 before/after + 固定射线能量 + 安全判定 |
| `loop_base` | 全局 α | 20 次折半 | ❌ 算法本身无目标（只有可行性） |

关键细节（接手的 agent 容易踩）：

1. `fit_c` 的 **expression 先验分母是 `n_frames×10`**，不是系数个数；与 shape 的 `mean` 不对称。
   这是源码实际写法，**不要"修正"成对称**，否则算出的值对不上。
2. `motion_v2` 所有数据项**除以 `n_frames`**，不是各自样本数。
3. `solve_field` 的固定射线 MSE 是**另外射线求交**得到的，不在求解器二次型里；
   把它当成"solver objective 的一部分"是错的。
4. `uv` 的 u/v 是**交错**存储，做均值时分母是 `2N`，不是 `N`（测试里踩过这个坑）。
5. 同一次迭代有 before/after 两条记录，**必须拆成独立序列**，否则曲线呈锯齿假振荡。

---

## 6. 尚未接通 / 需要接收方处理

优先级从高到低：

1. **【最高】用真实数据跑一次**：用户尚未指定采集。拿到路径后按 §3.2 执行，
   确认结论能否支撑算法改进。这一步**没有做**，是本次交接的核心待办。
2. **`fit_c` / `motion_v2` 分项拆分**：运行只落总量。要拿分项，只能在
   `loss_analysis/` 内建**带插桩的源码副本**（记录原文件 SHA256 + 插桩 diff），
   **不得覆盖主代码**。当前刻意没做。
3. **`identity_v2` / `front_anchor` / `guarded` 的 `field_logs`**：`report.json` 里没有。
   要拿只能等主算法侧改动（属于算法改进范畴，需本会话配合）或插桩。
4. **深度诊断仅沿光轴绝对差**：未含法向/平面距离；未做跨状态配准。
5. **支持度标签是整次运行累计**：无法按闭口/张口分别重算。

---

## 7. 给接收方的"可做 / 不可做"

### 可以做

- 读 `runtime/outputs/*/` 的任意产物（只读）。
- 在 `runtime/loss_analysis/` 下新建输出目录（不覆盖已有）。
- 在 `loss_analysis/` 内加分析脚本、图、测试。
- 报告"某阶段目标与预期不符"，作为**算法改进线索**反馈给本会话。

### 不可以做

- ❌ 改 `head_rgbd/`、`core/`、`config/`、`headrgbd.cmd` 等主项目源码。
- ❌ 为了让 loss 好看而调参数、改接受条件、改权重。
- ❌ 用最终网格反推迭代曲线（拿不到真实状态就写缺口）。
- ❌ 把 `solver_objective` 与 `diagnostic` 相加或互相比较。
- ❌ `git add/commit/push`、建 PR、发布模块、改远程配置。
- ❌ 升级主项目版本（保持 `0.2.0`）。
- ❌ 删除任何旧数据、旧运行、分析输出。

---

## 8. 反馈给算法改进方的格式

接收方得出结论后，按下面格式反馈（本会话据此改进算法）：

```
【阶段】dense.guarded_dense
【现象】固定射线 MSE 在 outer 1→2 几乎不变（2.99827 → 2.99827）
【数据】records.csv 具体行 / 图路径
【判断】可能是 X
【建议方向】（不要求实现，只描述）
```

**注意**：结论必须带图/CSV 路径，不能只给一句"loss 不降"。

---

## 9. 参考产物位置

| 内容 | 路径 |
|---|---|
| 分析模块说明 | `D:\proiect\facecap_airway_delivery\loss_analysis\README.md` |
| 阶段/分项源码对照 | `loss_analysis\loss_analysis\objectives.py` |
| 只读验证输出 | `D:\proiect\facecap_airway_delivery\runtime\loss_analysis\20261101_152049_zhou_full_20261101\` |
| 唯一完整真实重建 | `D:\proiect\facecap_airway_delivery\runtime\outputs\zhou_full_20261101\` |
| 重建算法入口 | `D:\proiect\facecap_airway_delivery\headrgbd.cmd`（`run` / `capture` / `reconstruct` / `export`） |

---

## 10. 交接结论

- **模块实现与单元测试**：✅ 完成（46 passed，入口与缺输入行为已验证）
- **真实数据**：❌ 尚未指定，**尚未运行**
- **GitHub**：❌ **没有提交或推送**；分析目录已本地排除
- **主项目**：版本仍 `0.2.0`，主算法零改动、零依赖，旧数据未删除
