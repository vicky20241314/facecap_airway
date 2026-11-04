# 精简重建主流程：删除未消费计算与链路缩短

> 历史实验记录。0.5.0 的正式默认值与几何/耗时验收见
> [DEFAULT_SMOOTH_FAST_REPORT.md](DEFAULT_SMOOTH_FAST_REPORT.md)。本报告的 C 使用 blend=0，
> 深度误差通过不等于表面平顺度通过。lean 不调用独立低频阶段，`reuse-anchor` 在 lean 下无效；
> 旧推荐命令不再代表当前默认方案。

对象：`D:\proiect\facecap_airway_delivery`，算法 `facecap-inspired-v2-strict`。
数据：`runtime/outputs/zhoupei_cap/prepared`（20 帧：闭口正面 5 / 左 5 / 右 5，张口正面 5）。
口径见 §5。所有数字来自本机真实重建。

---

## 1. 最终结果

主流程从 **7 个阶段缩到 5 个**，交错重复计时下耗时中位数从 **239.0 s 降到 210.7 s（−11.8%）**，
三个视角的精度全部落在工程预算内（±0.05 mm），其中四个分组还略有改善。

| 变体 | 阶段数 | 耗时 | 相对基线 |
|---|---|---|---|
| **BASE** 当前全链路 | 7 | 221.3 s | — |
| **A** 跳过未消费的后处理 | 7 | 218.3 s | −3.0 s |
| **B** + 删除重复锚定求解 | 7 | 218.1 s | −0.2 s |
| **C** + 缩短低频—运动交替链 | **5** | **194.6 s** | **−26.7 s（−12.1%）** |

质量（同帧深度一致性，MAE mm，留出闭口帧）：

| 分组 | BASE | C | Δ | 预算 | 判定 |
|---|---|---|---|---|---|
| 正面留出 | 0.882 | **0.864** | −0.018 | ≤ +0.05 | 通过 |
| 正面训练 | 0.887 | 0.877 | −0.010 | ≤ +0.05 | 通过 |
| 正面张口 | 1.215 | 1.221 | +0.006 | ≤ +0.05 | 通过 |
| 左侧留出 | 0.980 | **0.962** | −0.018 | ≤ +0.05 | 通过 |
| 左侧训练 | 0.884 | 0.885 | +0.001 | ≤ +0.05 | 通过 |
| 右侧留出 | 0.822 | 0.838 | +0.016 | ≤ +0.05 | 通过 |
| 右侧训练 | 0.751 | 0.757 | +0.006 | ≤ +0.05 | 通过 |

P95 同时改善的分组：正面留出 2.832 → 2.680，左侧留出 3.014 → 2.873，右侧留出 2.498 → 2.432，
右侧训练 2.328 → 2.304。覆盖率全部持平或上升。**三个视角共同约束共享身份的能力没有削弱**：
侧面帧仍进入身份拟合、低频场和 Loop 打分，`effective_shape_frame_ids` 未变。

---

## 2. 最终流程

```
prepare_observations
  → identity_front_anchor        fit_c --front-only            （初始化）
  → shared_identity_joint        fit_c --shape-fusion joint     （正侧联合身份 + 末段运动）
  → fixed_identity_motion        motion_v2                      （一次运动校正）
  → guarded_measured_dense_surface  fit_d --fast                （低频/中频共享形变 + Loop + 交付）
```

被删除的两个阶段：`low_frequency_shared_identity`（16.2 s）与
`motion_reestimate_after_identity`（24.7 s）。

---

## 3. 删掉的计算及其依赖依据

### 3.1 A：fit_c 拟合后处理（−3.0 s）

`fit_c` 在参数拟合之后还做五件事：局部精修 `refine_shared`、`safe_refinement`、支持证据射线
（每帧 12000 点）、颈部生成、阶段网格导出。**这五件的产物在 joint+fast 主链里没有被读**：

| 产物 | 消费者 | 结论 |
|---|---|---|
| `deviation_m`（局部场） | `motion_v2` | 读，但无 `--identity-field` 时 `field=None`，直接用 FLAME 原始几何 → 丢弃 |
| `refined_vertices` / `effective_deviation_m` | 仅 guarded 模式的锚点逻辑 | joint 分支不读 |
| `observed` / `labels` / `confidence` | 支持度 ply、`reference_geometry.json` | 交付包的 `support.json` 由 `fit_d` 生成 |
| 颈部补片 | `fit_c` 自身报告 | 交付不读 |
| `{state}_{coarse,refined}_m.obj`、`_support_m.ply` | 人工查看 | 交付不读 |

**一次修正**：第一次尝试把整段跳掉，`motion_v2` 立刻抛 `KeyError: 'closed'`——它读
`report['references']['closed']['frame_id']`（`--identity-field` 分支需要固定参考帧）。
所以保留 `references` 的记账（帧 id、位姿、标点），只跳过求解、证据射线、颈部和文件导出。
这说明"没有消费者"这个判断必须逐项验证，不能整段假定。

**实测拆分**（不是预估）：正面锚阶段 34.4 → 28.0 s（−6.4 s），联合阶段 54.4 → 54.7 s（+0.3 s，波动），
密集阶段 62.2 → 65.6 s（+3.4 s，波动）。净 −3.0 s。
任务给出的"两处后处理合计约 8.2 s"在这份采集上并未全部转化为节省——约 6 s 落在正面锚阶段，
其余被其他阶段的波动吃掉了，所以净收益按 3 s 记。

### 3.2 B：重复锚定求解（−0.2 s）

| 位置 | 上一版实测位移 | 处理 |
|---|---|---|
| `identity_v2` 第二次 `solve_field` | 0.0014 mm | `--reuse-anchor`：直接继承第一次求解的场与支持 |
| `fit_d` `guarded_regularized` | 0.0013–0.0014 mm | `--reuse-field`：直接用低/中频场 |

两者是"对同一批观测、从已收敛的点再解一次"。移除后几何差异在 0.0014 mm 量级。
**但时间只省了 0.2 s**——在 barycentric 配置下这两个求解本身就不贵（自由顶点 1127，1 轮外迭代）。
之前报告里"约 8 s"是把整个正则阶段含数据装配一起算的，装配仍要做。

诚实结论：这两个阶段的删除在**复杂度**上有价值（少一条调用、少一次 QP 装配），
在时间上几乎为零。不作为加速项宣传。

### 3.3 C：缩短低频—运动交替链（−23.5 s）

原链路有两次运动校正夹一个独立低频身份场。删掉这个中间环节后：

```
联合身份拟合 → 一次运动校正 → 一套低频/中频共享形变 → 细分交付
```

为什么能删：`fit_c` 的联合阶段的最后一个相位本身就是 `joint_motion_final`（只更新姿态/表情/下颌），
已经承担了"身份定了之后校正运动"的职责；`fit_d` 又从 `posed_vertices` 重新解低频/中频场，
而 `motion_v2 --identity-field` 带来的额外姿态修正没有在最终几何上留下可测差异。

代价与收益都能量化：

| 阶段 | BASE | C | Δ |
|---|---|---|---|
| identity_front_anchor | 34.4 | 30.7 | −3.7 |
| shared_identity_joint | 54.4 | 57.6 | +3.2 |
| fixed_identity_motion | 22.7 | 28.7 | +6.0 |
| low_frequency_shared_identity | 18.8 | — | −18.8 |
| motion_reestimate_after_identity | 21.5 | — | −21.5 |
| guarded_measured_dense_surface | 62.2 | 70.8 | +8.6 |
| **合计** | **221.3** | **194.6** | **−26.7** |

删掉 40.3 s，其中 18.7 s 被"下游要自己补做"吃回去：唯一的运动校正要多走一点（+6.0），
密集阶段的低频/中频场起点离收敛更远（+8.6），联合阶段略慢（+3.2）。净 −26.7 s。

这是本轮唯一量级上的节省，也是唯一需要质量背书的一步——见 §1，三视角误差变化在 ±0.018 mm。

---

## 4. 没有做的优化

以下都在任务清单里，但本轮未实施，如实列出：

- **正面锚阶段的迭代预算**：`--steps` 由两个 `fit_c` 阶段共用，降低它会同时削弱联合身份拟合。
  没有为正面锚单独开一个预算参数，因此没有测。
- **只对活跃帧跑 FLAME 前向**：`fit_c` 的身份相位仍对全部 20 帧前向。改成活跃帧子集会牵动
  时间平滑项（相邻帧的 `expr`/`jaw` 差分），属于真实依赖，没有贸然改。
- **阶段内早停**：现有早停在外迭代层，未在 Adam 相位内加。
- **Loop 单候选**：`--loop-data-selection` 仍评 3 个候选（约 2 s），没有改成固定混合量。
- **缓存**：没有新增缓存层；观测采样本身在阶段内已被复用。

这些项都不与已做的删减冲突，可作为下一轮。

---

## 5. 评价口径

同帧网格配同帧观测：每帧用自己那一行 `dense_vertices` 对自己的深度图求残差，每帧 6000 点，
`face_mask & depth>0`，seed=20261101，残差为沿光轴深度差，门限 |残差| ≤ 30 mm，
缺失计入覆盖率而不记 0。工具 `tools/eval_multiview.py`。

训练/留出划分沿用前几轮：`choose()` 在每个闭口段选 3 帧作为身份帧（9 帧），其余 6 帧留出；
张口 5 帧不参与身份。所有变体使用同一划分。

**这些数值是采集深度一致性，不是独立几何精度。** 没有扫描配准、没有人工轮廓。
工程验收预算是筛选加速方案用的（各视角 MAE ≤ +0.05 mm、关键区域 ≤ +0.1 mm），不代表测量精度。

---

## 6. 耗时测量

交错执行，BASE 与 lean 交替各 3 次，取 `run_report.json` 的端到端耗时（含进程启动与导出）：

| 轮次 | BASE | LEAN |
|---|---|---|
| 1 | 221.3 s | 194.6 s |
| 2 | 239.0 s | 210.7 s |
| 3 | 273.4 s | 216.5 s |
| **中位数** | **239.0 s** | **210.7 s** |
| 范围 | 221.3–273.4（幅 52.1） | 194.6–216.5（幅 21.9） |

**中位节省 28.3 s（−11.8%）。** 三对比较中 BASE 的最小值（221.3）仍大于 lean 的最大值（216.5），
按秩完全分离；但单次波动最高达 52 s，所以只有中位数是有意义的量，
§3.3 的单次阶段拆分只用于解释"节省落在哪个阶段"，不用于声称精确加速比。

基线本身在不同时段漂移明显：上一轮同一配置记录 187.6 s，本轮 221.3 s（+18%），中间没有代码差异。
**历史单次耗时不能当基线**，这也是本轮重测的原因。

准备阶段未单独计时：本次全部运行复用 `zhoupei_cap/prepared`，`prepare_observations` 不在链路内。
评价（`eval_multiview.py`）单次约 40 s，不计入重建耗时。

---

## 7. 推荐配置

```pwsh
cd D:\proiect\facecap_airway_delivery
.\runtime\env\python.exe -m head_rgbd reconstruct `
  --sequence "runtime/outputs/zhoupei_cap/prepared/sequence.json" `
  --output   "runtime/outputs/<新目录>" `
  --fast --mid-rounds 1 `
  --shape-fusion joint `
  --anchor-prior-weight 1.0 `
  --coverage-allowance-ratio 0.001 `
  --support-min-information 0.5 `
  --chain lean --postfit minimal --reuse-anchor --reuse-field
```

| 参数 | 值 | 依据 |
|---|---|---|
| `--chain` | `lean` | 删掉两个阶段，−23.5 s，质量在预算内（§3.3） |
| `--postfit` | `minimal` | 跳过未被读的局部场/证据/颈部/阶段网格，−3.0 s（§3.1） |
| `--reuse-anchor` / `--reuse-field` | 开 | 去掉两次 0.0014 mm 量级的重复求解（§3.2，时间收益近似 0） |
| `--shape-fusion joint` | — | 保留正侧共同约束；侧面数据在三视角上的收益见 `MULTIVIEW_CONFLICT_REDUCTION_REPORT.md` |
| `--fast --mid-rounds 1` | — | 密集残差仍不执行（既定取舍） |

需要与历史结果逐位对照时，去掉最后四个开关即回到 BASE。

---

## 8. 产物

| 内容 | 绝对路径 |
|---|---|
| BASE（全链路基线） | `runtime\outputs\zhoupei_lite_base_20261102_130908` |
| A 跳过后处理 | `runtime\outputs\zhoupei_lite_A_20261102_132101` |
| B +删除重复求解 | `runtime\outputs\zhoupei_lite_B_20261102_133138` |
| **C 精简主流程（推荐）** | `runtime\outputs\zhoupei_lite_C_20261102_133517` |
| 质量对比 | `runtime\eval\zhoupei_lite` |
| 交付网格 | 各 run 的 `delivery\head_closed_m.obj`、`head_open_m.obj` |
| 结构网格 | `reconstruction\{front_anchor,guarded,motion_only,dense}`（lean 链无 `low_frequency`/`alternated_motion`） |

`zhoupei_cap`、历史输出、采集与模型资产未改动。

## 9. 代码改动

| 文件 | 改动 |
|---|---|
| `head_rgbd/fit_c.py` | `--postfit {full,minimal}`：minimal 跳过 `refine_shared`、支持证据射线、颈部、阶段网格导出；保留 `references` 记账（`motion_v2` 依赖） |
| `head_rgbd/identity_v2.py` | `--reuse-anchor`：继承第一次求解的场与支持，不再二次求解；报告写 `stage=reused_first_solve`、`reused=True` |
| `head_rgbd/fit_d.py` | `--reuse-field`：直接用低/中频场，跳过 `guarded_regularized`；报告写 `stage=reused_low_mid_field` |
| `head_rgbd/pipeline.py` | `--chain lean` 跳过 `low_frequency_shared_identity` 与 `motion_reestimate_after_identity`，`fit_d` 直接以 `motion_only` 为输入；四个开关贯通 |
| `head_rgbd/cli.py` | `--chain`、`--postfit`、`--reuse-anchor`、`--reuse-field` |

跳过阶段在报告里如实记录：`reused=True` 与 `history=[]`，不会以"收敛到零更新"的形式出现。

回归：`pytest tests -q` → **44 passed, 1 skipped**。几何安全（位移上限、坡度、法向、防翻面）
与覆盖率验收在 lean 链上照常执行，未放宽。
