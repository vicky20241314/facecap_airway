# loss_analysis — 本地 loss 分析工具

> 该模块**不属于**主重建算法。主项目 `head_rgbd` / `core` 不导入它，启动流程不依赖它，
> 它也不在主算法必装依赖里。它只做一件事：**读一次已经跑完的重建，把源码里真实存在的
> 优化目标按档位记录下来，并给出独立诊断**，用于判断优化过程是否合理、寻找改进方向。
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
│  ├─ plots.py                    曲线（仅真实记录）+ 中文报告
│  └─ cli.py                      独立入口
└─ tests/test_loss_analysis.py    46 项数学与接口契约测试（构造数据）
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
└─ figures/
   ├─ <stage>.png           每个阶段一张，按档位/决策分线
   ├─ _gaps.png             缺口面板：哪些阶段没有真实状态、为什么
   └─ _diagnostics.png      独立诊断 MAE/RMSE/P95（非算法 loss）
```

---

## 6. 没有参数时会怎样

| 输入 | 行为 |
|---|---|
| 无任何参数 | 打印帮助并提示 `no input supplied`，退出码 0 |
| `inspect --run <不存在的路径>` | 输出 `{"status":"error", ...}`，退出码 2 |
| `analyze` 缺 `--run` | argparse 直接报 `the following arguments are required: --run`，退出码 2 |
| 输出目录已存在 | `FileExistsError`，退出码 3，不覆盖 |

---

## 7. 与主算法的隔离（已落实）

- **未修改任何主项目已跟踪源码**；`head_rgbd` / `core` 不认识本模块。
- 不插桩、不复制插桩源码：状态全部通过**只读读取已落盘产物**获得
  （阶段日志里的 objective 行、各 stage 的 `report.json`）。
- 不改变目标、参数更新、采样、随机种子、迭代次数、接受条件、几何保护。
- 分析不启动重建、不占用 GPU。

### 7.1 已接通 / 尚未接通

| 项目 | 状态 |
|---|---|
| `fit_c` 每步 solver objective | ✅ 可从 `logs/*.log` 读取 |
| `motion_v2` 每步 solver objective | ✅ 可从 `logs/*.log` 读取 |
| `fit_c` / `motion_v2` **分项拆分** | ❌ 运行未持久化分项，记为 gap（需插桩才能得到，本模块不插桩） |
| `solve_field` 线性化目标（before/after） | ✅ 从 dense 阶段 `report.json` 的 `field_logs[*].history` |
| 固定射线非线性能量 + 缺失射线数 | ✅ 同上（`fixed_ray_diagnostic`） |
| 几何安全判定 | ✅ 同上（`safety`） |
| `identity_v2` / `front_anchor` / `guarded` 的 `field_logs` | ❌ 这些 `report.json` **不含** `field_logs`，记为 gap |
| `loop_base` 目标 | ❌ 算法本身没有目标，只有可行性试步 |
| 深度误差 MAE/RMSE/P95/覆盖 | ✅ 给出 `--sequence` 时计算（构造数据已验证接口，真实数据待你指定） |
| 标点重投影诊断 | ✅ 函数就绪，需 `sequence` + 网格 + `reference_landmarks.json` 三者同时提供 |
| 支持度标签统计 | ✅ 读 `delivery/support.json` |

---

## 8. 测试与入口验证（本轮实际结果）

```pwsh
cd D:\proiect\facecap_airway_delivery\loss_analysis
..\runtime\env\python.exe -m pytest tests -q
# 46 passed
```

覆盖：平滑 L1 两分支、已知 2mm/10mm 偏差、权重线性、无效标点排除、索引窗口外数据被忽略、
零有效样本为 `null`、2D 归一化、hinge 死区与公式、`/(n_frames×10)` 先验、线性系统二次缩放、
固定射线缺失加罚、射线变少不可能变好、两条接受判据、四档位不混、缺口记录、
CSV/JSON 格式、不覆盖、目录完整性、入口无参数/`--version`/`stages`/缺路径/缺 `--run`。

**构造数据只证明计算与接口正确，不证明真实重建效果。**

入口实测（`login=false`，PowerShell）：

| 命令 | 结果 |
|---|---|
| `python -m loss_analysis` | 打印帮助 + `no input supplied`，退出码 0 |
| `python -m loss_analysis --version` | `lossana 0.1.0`，退出码 0 |
| `python -m loss_analysis stages` | 打印 9 个阶段与全部分项，退出码 0 |
| `inspect --run <不存在>` | 明确报错，退出码 2 |
| `analyze`（缺 `--run`） | 参数缺失报错，退出码 2 |

---

## 9. 将来传入真实数据的命令格式

```pwsh
Set-Location D:\proiect\facecap_airway_delivery
$env:PYTHONPATH = "D:\proiect\facecap_airway_delivery\loss_analysis"
& ".\runtime\env\python.exe" -m loss_analysis analyze `
    --run      ".\runtime\outputs\<你的运行目录>" `
    --sequence ".\runtime\outputs\<你的运行目录>\prepared\sequence.json" `
    --mesh-dir ".\runtime\outputs\<你的运行目录>\delivery" `
    --label    "<自定义标签>" `
    --max-depth-mm 30
```

要求：`<run>` 必须是一次**已完成**的 `headrgbd reconstruct` 目录
（有 `run_report.json`、`logs/`、`reconstruction/`）。若只有 `capture` 而无重建，
`analyze` 会把这些阶段全部记为 gap 并说明原因。

---

## 10. Git 与本地保留

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

## 11. 已知限制

1. `fit_c` / `motion_v2` 的**分项拆分**拿不到（运行只记总量），要拿到必须插桩，本模块刻意不做。
2. `identity_v2` 与 `front_anchor` / `guarded` 阶段的 `field_logs` 未落盘，只有 dense 阶段有。
3. 深度诊断只做**绝对深度差**（沿光轴），未包含法向/平面距离分项；也未做跨状态配准。
4. 支持度标签是整次运行累计的，**不能**按闭口/张口分别重算。
5. 曲线上的点只有 2–3 个时（因为该阶段只有 2–3 个 outer step），不构成趋势结论。
6. 构造数据测试**不验证**真实重建精度或算法优劣。

---

## 12. 结论边界

- 工具只观察、只统计、只报告；它不参与优化，也不改进算法。
- 四档数值**不可互相比较、不可相加**；跨阶段没有 total。
- "loss 下降"不等于"几何更准"；精度仍需独立扫描/人工轮廓验证。
