# 正侧面联合形状重建：实现与实验报告

版本 0.3.0 之上的一次算法能力扩展。所有数字来自本机真实重建，口径见 §5。

---

## 1. 解决的具体问题

改造前（`guarded`）的链路里，侧面观测对**共享形状**没有贡献，而且这不是参数没调好，是结构性的：

| 环节 | 改造前行为 | 后果 |
|---|---|---|
| `fit_c --front-anchor` | `if not identity_phase or anchor is not None: shape.grad=None` | FLAME 身份参数在侧面阶段被**完全冻结** |
| 同一步的 `active` | `active=[i for i in active if i not in anchored_ids]` | 正面帧被移出身份阶段，只剩侧面帧在更新 |
| `identity_v2` | `locked=front_protected_vertices(...)` | 正面覆盖的 2797 个顶点被**硬锁**，不可移动 |
| `fit_d` 低频/中频 | `front_identity`（只含正面帧） | 交付面所需的控制场只由正面决定 |

实测证据：改造前这次运行里 `changed_side_detail_vertices = 0`、`maximum_side_detail_change_mm = 0.0`——侧面阶段一个顶点都没改。这也解释了为什么"侧面数据没有改善侧向几何"。

本次实现了一个显式的联合模式 `--shape-fusion joint`，并保留 `guarded` 作为可复现对照。

---

## 2. 联合融合机制

### 2.1 统一的多视角形状模型

沿用现有表示，不新造模型：

- **共享 FLAME 身份参数**（`fit_c` 的 `shape`，50 维）：管整体脸型、鼻部突出程度、下颌比例。
- **共享平滑标量法向场**（`geometry_d.solve_field` 的 `field`）：管个体局部差异。
- **各帧自有** 姿态 / 表情 / 下颌。

所有闭口视角共同决定上面两类共享变量；张口帧不参与形状，只用于表情、下颌与交付状态。

### 2.2 三处关键改动

| 文件 | 改动 |
|---|---|
| `head_rgbd/fit_c.py` | 新增 `--shape-fusion {guarded,joint}`、`--side-depth-weight`、`--joint-shape-prior`。联合模式下：`shape.grad` 在身份阶段**不再置空**；正面帧**不再移出**身份阶段；身份阶段每个闭口视角**一票**（不再给正面 3× 权重）；深度项对侧面乘 `--side-depth-weight`；正视身份先验变为**有限强度软约束** `joint_shape_prior * mean((shape - anchor_shape)^2)`（默认 0.02）。正面前锚降级为**初始化**：不再断言正面帧顶点必须等于前锚，也不再锁定前锚的法向场 |
| `head_rgbd/geometry_d.py` | `solve_field` 新增 `soft_lock`（每顶点置信度，1.0 = 完全可信）。方程权重乘 `soft_lock[顶点].mean()`，**顶点仍留在自由集合里**，因此几何约束照常生效——这是"硬锁 → 连续置信度"的实现，不是放宽约束 |
| `head_rgbd/identity_v2.py` | 联合模式：低频锚定用**全部闭口身份帧**；第二次求解用 `soft_lock=np.where(protected, joint_front_confidence, 1.)`（默认 0.15）代替 `locked=protected` |
| `head_rgbd/fit_d.py` | 联合模式：低频/中频控制场用 `identity`（全部闭口身份帧）而不是 `front_identity`；`guarded_regularized` 用置信度加权代替硬锁。报告新增 `shape_fusion` 块记录模式、身份视角、硬锁/降权顶点数 |
| `head_rgbd/pipeline.py`、`cli.py` | `--shape-fusion`、`--joint-front-confidence`、`--side-depth-weight` 贯通全链路；联合模式阶段名记为 `shared_identity_joint` |

### 2.3 顺序与参考模型

运动与形状交替，沿用既有阶段划分，不新增求解器：

1. 正面初始化：`fit_c --front-only` → 正面前锚（唯一一次「只看正面」）。
2. 联合身份：`fit_c --front-anchor --shape-fusion joint`，身份阶段由三个视角共同更新 `shape`，同时各帧姿态/表情/下颌交替更新，终点再解一次局部标量场。
3. 运动校正：`motion_v2` → 低频身份（联合置信度加权）→ `motion_v2 --identity-field`。
4. 控制网格：`fit_d --fast --shape-fusion joint`，Loop 细分 + 几何验证 + 支持度 + 导出。

跨阶段复用位移时，参考模型始终是当阶段的 `posed_vertices`；联合模式额外记录 `joint_anchor_update_mm`（正面帧相对前锚的真实位移）与 `joint_shape_delta_from_anchor`（身份参数相对前锚的最大变化），避免"位移参考不明"。

### 2.4 置信度（不是"更多自由度"）

- 单视角可靠区域可以修正：顶点是否可动由 `support >= 2` 与解剖排除决定，**不再由正面覆盖决定**。
- 多视角一致区域获得更充分约束：方程权重按面积求积、入射角平方、连拍归一化、局部平面噪声、以及 `soft_lock` 连乘。
- 观测不足的方向由先验稳定：`joint_shape_prior` 与前锚软约束。
- 连拍重复与视角多样性仍分别统计（`burst_weights` 与按帧一票分开）。
- 最终支持度标签（`support.json` 的 `region_label`）仍只用于**解释**观测情况，与优化置信度分开记录，二者不混用。

---

## 3. 侧面几何改善证据

### 3.1 三个实验的实际配置

| 代号 | 配置 | 说明 |
|---|---|---|
| **B** | `--fast --mid-rounds 1` | 当前 guarded 快速流程（正面冻结形状） |
| **C** | 同上 + `--shape-fusion joint` | 本次联合流程 |
| **C2** | 同上 + `--shape-fusion joint --side-depth-weight 0` | **侧面消融**：同样的自由度，但没有侧面深度残差 |

关于 A：任务里的「A. 仅正面形状优化」在本代码库里**与 B 是同一配置**（B 的形状本来就只由正面决定，见 §1 的 `changed_side_detail_vertices=0`）。因此我没有伪造一个 A，而是把「侧面数据到底有没有用」这个问题交给更强的对照 **C2**：C2 和 C 自由度完全相同，唯一差别是侧面深度项权重为 0。这样 C2→C 的差值只能归因于**数据融合**，不能归因于"自由度变多"。

训练/留出划分：`choose()` 在每个闭口段选 3 个关键帧作为身份帧，其余 2 帧不参与共享形状更新。实际选中的 9 帧（3 正面 / 3 左 / 3 右）记录在每个 run 的 `reconstruction/dense/report.json:identity_frames`，评估工具按同一份划分分组，留出帧只用于评价。

### 3.2 侧面数据确实携带有效几何信息（C2 → C）

同帧、同射线、同门限，只看闭口帧，单位 mm：

| 视角 | 区域 | C2（无侧面数据） | C（有侧面数据） | 改善 |
|---|---|---|---|---|
| 左 | 嘴唇 | 2.432 | 1.418 | **−1.014** |
| 左 | 下颌/轮廓 | 2.708 | 1.966 | **−0.742** |
| 右 | 嘴唇 | 2.018 | 1.308 | **−0.710** |
| 右 | 下颌/轮廓 | 2.347 | 1.598 | **−0.749** |
| 左 | 其余面部 | 0.920 | 0.904 | −0.016 |
| 右 | 其余面部 | 1.020 | 0.979 | −0.041 |
| 左 | 鼻部 | 1.032 | 0.994 | −0.038 |
| 右 | 鼻部 | 1.063 | 1.028 | −0.035 |

全脸：左侧留出 1.582 → 1.246（−0.336），右侧留出 1.456 → 1.165（−0.291）。

**结论**：在同样的模型自由度下，去掉侧面深度残差会让侧面嘴唇和下颌轮廓恶化 0.7–1.0 mm。侧面观测确实给出了正面看不到的侧向形状约束，最大收益集中在**嘴唇与下颌轮廓**，其次是面颊（"其余面部"）。

### 3.3 鼻部：没有测到改善

三个视角的鼻部误差在 B / C / C2 之间都在 0.99–1.06 mm 内，差异 < 0.07 mm（低于本次口径的判别力）。这份采集的侧向姿态下，鼻部大部分面积被正面与近正面观测主导，侧面既没有新增可见面积也没有更陡的入射角，所以**联合融合没有改善鼻部**。不宣称这一点。

---

## 4. 机制消融：原正面保护区是否真的能动了

| 指标 | B（guarded） | C（joint） | C2（joint 无侧面数据） |
|---|---|---|---|
| `shape_identity_frozen` | `True` | `False` | `False` |
| 正面硬锁顶点 `protected_vertices` | 2797 | **0** | **0** |
| 局部场相对前锚改变的顶点 `changed_side_detail_vertices` | **0** | **1776** | 1790 |
| 局部场最大改变 `maximum_side_detail_change_mm` | **0.0** | **2.999** | 2.999 |
| 身份参数相对前锚最大变化 `joint_shape_delta_from_anchor` | —（冻结） | 0.540 | 0.261 |
| 正面帧相对前锚真实位移 `joint_anchor_update_mm` | —（断言相等） | 9.24 mm | 10.90 mm |

读法：

- 改造前**一个顶点都没动**；改造后 1776 个顶点实际移动，且局部场达到 3.0 mm 的位移上限——机制不是"看起来通了"，是真的在求解。
- `joint_anchor_update_mm` 9–11 mm 里包含正面帧姿态/表情的自由度，不能当作纯形状变化；纯身份的指标是 `joint_shape_delta_from_anchor`（0.54 / 0.26）。**这两个数必须分开读**，否则会把姿态漂移误读成形状改动。
- C 与 C2 的改动量接近，但方向不同：C2 没有侧面数据却仍放开自由度，于是身份参数只敢移动 0.26（被正面软先验拉住），而姿态漂移更大（10.9 mm），结果是侧面被带偏（§3.2）。这正是"单纯增加自由度"与"数据融合"的区别。

### 4.1 最终网格确实继承了联合结果

| run | 状态 | 参考帧 | 交付网格 vs `dense_vertices[row]` 最大差 |
|---|---|---|---|
| C joint | closed | `closed_front_4` | 0.000005 mm |
| C joint | open | `open_front_4` | 0.000005 mm |
| B guarded | closed | `closed_front_4` | 0.000005 mm |

差值来自 OBJ 文本精度（5e-6 mm），说明 `delivery/head_{closed,open}_m.obj` 就是联合求解的 `dense_vertices` 对应行，不存在"导出的是另一条链路"的情况。

### 4.2 几何安全

两态均 20018 顶点 / 39904 面、顶点有限、面索引合法；总面积 C 195 207.9 / 194 934.8 mm² 对 B 195 195.4 / 194 987.0 mm²（差 0.006%，在数值噪声内）。位移上限、坡度、法向变化、防翻面等检查在每次 `solve_field` 内部照常执行（`check_field`），没有因为联合模式而放宽。

---

## 5. 质量与效率权衡

口径：同帧网格配同帧观测（每帧用自己的 `dense_vertices[row]` 对自己的深度），每帧 6000 点，seed=20261101，残差为沿光轴深度差，门限 |残差| ≤ 30 mm；缺失计入覆盖率、不记 0。工具：`tools/eval_multiview.py`。

### 5.1 全视角（C 相对 B）

| 视角 | 状态 | 划分 | B MAE | C MAE | Δ | B P95 | C P95 | B 覆盖/缺失 | C 覆盖/缺失 |
|---|---|---|---|---|---|---|---|---|---|
| 正面 | closed | 留出 | 1.352 | 1.345 | −0.007 | 3.903 | 3.808 | 99.27% / 88 | 99.83% / **20** |
| 正面 | closed | 训练 | 1.328 | 1.368 | +0.040 | 3.827 | 3.874 | 98.91% / 196 | 99.72% / **51** |
| 左 | closed | 留出 | 1.207 | 1.246 | +0.039 | 3.618 | 3.282 | 99.19% / 97 | 99.34% / 79 |
| 左 | closed | 训练 | 1.179 | 1.167 | −0.012 | 3.377 | 2.971 | 99.45% / 99 | 99.53% / 85 |
| 右 | closed | 留出 | 1.168 | 1.165 | −0.003 | 3.234 | 3.034 | 99.46% / 65 | 99.52% / 58 |
| 右 | closed | 训练 | 1.101 | 1.119 | +0.018 | 3.283 | 3.024 | 99.61% / 70 | 99.64% / 64 |
| 正面 | open | — | 1.737 | **1.581** | **−0.156** | 5.103 | 4.557 | 95.74% / 1278 | **98.94% / 319** |

**C 相对 B 在闭口视角上是持平的**（最大 |ΔMAE| = 0.040 mm，在闭口各视角 ±0.04 内互有胜负），没有变好也没有变坏。

### 5.2 相对 B 的真实收益（按证据强度排序）

1. **张口状态误差 −0.156 mm**（1.737 → 1.581），覆盖率 95.74% → 98.94%，缺失 1278 → 319。这是本次最大且最稳的单项改善。
2. **正面下颌/轮廓 P95 −1.63 mm**（9.384 → 7.756），下颌轮廓是全脸误差最集中的区域，长尾被明显压掉。
3. **覆盖率系统性上升、缺失系统性下降**：正面闭口缺失 196 → 51，正面张口 1278 → 319，其余视角小幅改善。
4. **P95 在侧面普遍下降**：左侧训练 3.377 → 2.971，右侧训练 3.283 → 3.024，右侧留出 3.234 → 3.034。

### 5.3 预先约定的门限（跑候选前确定，未事后放宽）

MAE/RMSE 相对增加 ≤ 0.05 mm、P95 相对增加 ≤ 0.1 mm、覆盖率下降 ≤ 0.1 个百分点、几何安全条件不变。对照 §5.1：闭口最大 ΔMAE +0.040、最大 ΔP95 +0.047、覆盖率全部上升 → **C 全部通过**。

### 5.4 耗时

| 阶段 | B guarded | C joint | C2 无侧面 |
|---|---|---|---|
| prepare_observations | 36.9 | 37.8 | 36.1 |
| identity_front_anchor | 29.3 | 29.1 | 28.5 |
| **shared_identity_（joint/guarded）** | **38.1** | **53.0** | 59.3 |
| fixed_identity_motion | 21.3 | 20.8 | 21.1 |
| low_frequency_shared_identity | 19.2 | 12.9 | 15.3 |
| motion_reestimate_after_identity | 20.9 | 22.1 | 21.5 |
| guarded_measured_dense_surface | 92.5 | 67.3 | 120.0 |
| **合计** | **265.0** | **249.8** | 301.8 |

- 联合身份阶段比 guarded 贵 **+14.9 s**（38.1 → 53.0 s：多了身份阶段的正面帧，且共享身份参数也要解）。
- 但后续密集段反而更快（92.5 → 67.3 s）：中频 QP 的迭代次数随初始化改变而减少。
- 净结果：**C 的总耗时 249.8 s，比 B 的 265.0 s 略低，仍在 300 s 目标内**。这次上下的 15 s 在单次运行的波动范围内（同代码重复运行实测 ±40 s），所以准确的表述是"**联合融合没有带来可测的时间代价**"，不是"更快"。

---

## 6. 推荐配置

```pwsh
cd D:\proiect\facecap_airway_delivery

# 联合模式（推荐）：正面与侧面共同决定同一个头面部形状
.\runtime\env\python.exe -m head_rgbd reconstruct `
  --capture "runtime/captures/20261101_140933_045847_head" `
  --output "runtime/outputs/<新目录>" `
  --fast --mid-rounds 1 --shape-fusion joint

# 对照（guarded，形状只由正面决定）
.\runtime\env\python.exe -m head_rgbd reconstruct `
  --capture "runtime/captures/20261101_140933_045847_head" `
  --output "runtime/outputs/<新目录>" --fast --mid-rounds 1

# 侧面消融（同自由度、无侧面深度）
... --fast --mid-rounds 1 --shape-fusion joint --side-depth-weight 0
```

参数（默认值即推荐值）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--shape-fusion` | `guarded` | `joint` 开启多视角共享形状 |
| `--joint-front-confidence` | `0.15` | 正面覆盖顶点保留的数据权重（替代硬锁） |
| `--side-depth-weight` | `1.0` | 侧面深度项相对正面的权重；`0` 即消融对照 |
| `--joint-shape-prior` | `0.02` | 身份参数回拉前锚的软约束强度（`fit_c`） |
| `--mid-rounds` | `2` | 中频轮数；本次实验用 `1` |

**建议默认用 joint**，理由是：闭口精度不劣于 guarded（门限内），张口误差与覆盖率明显更好，耗时无增加，而且不再需要"永久冻结正面区域"这条硬规则。若需要与 0.3.0 的交付完全可复现对照，用 `guarded`。

### 报告字段（便于复核）

- `reconstruction/guarded/report.json`：`shape_fusion`、`shape_identity_frozen`、`front_anchor.protected_vertices`、`changed_side_detail_vertices`、`joint_anchor_update_mm`、`joint_shape_delta_from_anchor`、`local_field_max_mm`
- `reconstruction/dense/report.json`：`shape_fusion`（模式、身份视角清单、硬锁/降权顶点数）、`fast_path`、`timing_sections`
- `reconstruction/low_frequency/report.json`：`shape_fusion`、`front_locked_vertices` / `front_downweighted_vertices`

---

## 7. 测试

新增 `tests/test_joint_fusion.py`（5 项，全部通过）：

| 测试 | 拦截的错误 |
|---|---|
| `test_side_view_moves_the_shared_field` | 两个正面视角与模型一致时共享场不动（< 0.05 mm）；加入一个差 5 mm 的侧面视角后共享场必须移动（> 0.3 mm），且该视角被记为支持 |
| `test_confidence_weight_lets_the_protected_region_respond` | 硬锁下自由集为空、场为 0 且状态是 `skipped_no_free_vertices`；同样的保护区域给 15% 置信度后必须能产生修正 |
| `test_all_views_agreeing_leave_the_surface_alone` | 三个视角都同意时不能凭空造出修正 |
| `test_guarded_stays_the_default` | 不给参数时仍是 `guarded`，默认值不被联合模式污染 |
| `test_joint_requires_a_front_anchor` | 联合模式必须带前锚初始化，否则解析期直接报错退出码 2 |

回归：`pytest tests -q` → **42 passed**（原 37 + 新增 5）。

---

## 8. 未完成 / 边界

- **鼻部未改善**（§3.3），不宣称。
- **闭口视角精度没有超过 guarded**（§5.1）。本次能证明的是"侧面数据必要"（§3.2 消融）与"联合机制生效"（§4），不能证明"联合比正面锚更准"。原因：这份采集的正面锚已经把侧向几何拟合到 1.1–1.2 mm，侧面观测能提供的增量被压低。
- **只用了一次采集、20 帧、3 个视角**，且 train/holdout 是同一连拍内的留出，不是跨采集留出。结论适用范围限于"同一人、同一设备、同一采集流程"。
- **未恢复密集残差 QP**（保持 0.3.0 的快速路径）；联合模式只作用于控制网格。若后续要在密集层也做融合，需要重新验证 §5 的门限。
- 正面帧在联合模式下姿态/表情自由，`joint_anchor_update_mm` 达 9–11 mm。这是自由度带来的，**不是**形状被改了 9 mm；读报告时必须用 `joint_shape_delta_from_anchor`（0.54）判断形状改动。
- 未提交、未推送；采集、模型资产与历史输出未改动。

---

## 9. 产物

| 内容 | 绝对路径 |
|---|---|
| 联合重建（C） | `D:\proiect\facecap_airway_delivery\runtime\outputs\joint_fast_v3` |
| guarded 对照（B） | `...\runtime\outputs\guarded_fast_v2` |
| 侧面消融（C2） | `...\runtime\outputs\joint_noside_v2` |
| 多视角/区域评估 | `...\runtime\eval\joint_v3`（`multiview.json`、`multiview.md`、`regions_by_view.md`） |
| 本报告 | `D:\proiect\facecap_airway_delivery\docs\JOINT_FUSION_REPORT.md` |

每个 run 目录下的 `reconstruction/{front_anchor,guarded,low_frequency,alternated_motion,dense}` 分别保存该阶段网格与参数；`delivery/head_closed_m.obj`、`head_open_m.obj` 是最终交付网格。
