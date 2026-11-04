# 多视角联合约束下的形状融合：改进与验证

对象：`D:\proiect\facecap_airway_delivery`，算法 `facecap-inspired-v2-strict`。
验证数据：`runtime/outputs/zhoupei_cap/prepared`（20 帧：闭口正面 5 / 闭口左 5 / 闭口右 5 / 张口正面 5）。
所有数字来自本机真实重建，口径见 §4.2。

---

## 1. 结果

多视角联合约束这条链路此前是**结构上不通的**：侧面观测对共享身份没有任何贡献。把链路接通、
把脆弱的两条硬门限换成连续量、把正面锚点从"降权"改成"显式先验"之后，同一份数据的三个视角
同时改善，且更快。

| 变体 | 正面留出 | 正面训练 | 张口 | 左侧留出 | 左侧训练 | 右侧留出 | 右侧训练 | 总耗时 |
|---|---|---|---|---|---|---|---|---|
| **B0** 现状 joint+fast | 1.345 | 1.368 | 1.581 | 1.246 | 1.167 | 1.165 | 1.119 | 244.8 s |
| **B1** 链路修复 + 连续支持 + 锚点先验 | 1.083 | 1.111 | 1.353 | 1.076 | 0.986 | 0.949 | 0.879 | 200.1 s |
| **B2** + 覆盖率比例容差 | 0.889 | 0.894 | 1.216 | 0.975 | 0.882 | 0.822 | 0.751 | 181.5 s |
| **B3** + 数据驱动 Loop 选择 | **0.882** | **0.887** | **1.215** | **0.980** | **0.884** | **0.822** | **0.751** | 189.6 s |
| **D** 侧面数据消融（同机制，仅正面更新形状） | 0.786 | 0.803 | 1.146 | 1.159 | 1.027 | 0.959 | 0.901 | 195.8 s |

单位 mm，MAE。相对 B0：

- 正面留出 **−34.4%**，左侧留出 **−21.3%**，右侧留出 **−29.4%**，张口 **−23.2%**；
- 总耗时 **−22.5%**（244.8 → 189.6 s）。

侧向几何的改善集中在**下颌轮廓与侧向鼻部**（§5.3）：左侧下颌 1.676→2.145 的反向对照说明，
这 0.47 mm 来自侧面数据本身，不是"多给了自由度"。

---

## 2. 改进的能力与它此前为什么不存在

### 2.1 侧面观测到不了共享身份

四道结构门把它们全挡住了，全部来自源码，不是推测：

| 位置 | 原行为 | 后果 |
|---|---|---|
| `fit_c.py`：`if not identity_phase or anchor is not None: shape.grad=None` | 侧锚模式下 FLAME 身份参数梯度被清空 | 身份在侧面阶段停止更新 |
| 同上：`active=[i for i in active if i not in anchored_ids]` | 正面帧被移出身份阶段 | 只剩侧面帧，但它们的身份梯度又为空 |
| `pipeline.py`：`low_frequency_shared_identity` 的调用参数 | **没有转发 `--shape-fusion`/`--joint-front-confidence`** | 即使命令行开了 joint，`identity_v2` 仍跑 guarded 分支 |
| `identity_v2.py`：`locked=front_protected_vertices(...)` | 正面覆盖顶点被硬移出未知向量 | 可观测区域 100% 被锁，自由集精确为空 |

实测后果（B0，`reconstruction/low_frequency/report.json`）：

```
shape_fusion = "guarded"        front_locked_vertices = 2670
front_downweighted_vertices = 0 maximum_absolute_displacement_mm = 0.0
solve[1].history[0].status = "skipped_no_free_vertices"
```

低频身份阶段两次求解，第二次**一个自由度都没有**，整个阶段空转，却仍以 `passed` 收尾。

### 2.2 支持集由两个硬门限决定

`geometry_d.solve_field` 用 `bc > 0.05` 记支持、用 `support.sum(0) >= 2` 决定可优化顶点。
两条都是阶跃：对应关系的一点数值变化就能让顶点进出自由集，而"只有一个可靠视角覆盖的侧脸"
被结构性排除——这正是侧面数据最该起作用的地方。

### 2.3 正面锚点被实现成"给观测降权"

`soft_lock` 把落在正面覆盖区的**方程权重**统一乘 0.15。该区域里的正面观测和侧面观测一起被
削弱，所以它表达不了"保留可靠正面形状、同时接受侧面修正"——它只是把两种证据一起压小。

---

## 3. 改进机制

### 3.1 打通联合链路（B1）

`pipeline.py` 的 `low_frequency_shared_identity` 调用补上 `--shape-fusion` 与
`--joint-front-confidence`，与 `fit_d` 一致；`identity_v2` 在 joint 下用全部闭口身份帧做锚定解，
不再只用正面。同时给该阶段加了一条结构性检查：joint 模式下若两次求解都是
`skipped_no_free_vertices`，直接报错并给出锁定数、帧号与原因，**不再以 `passed` 收尾**。

验收（B1 实测）：`front_locked_vertices 2670 → 0`，`front_downweighted_vertices 0 → 2675`，
`maximum_absolute_displacement_mm 0.000 → 1.091 mm`，自由顶点 `0 → 1136`。

### 3.2 连续支持替代支持计数（B1）

`geometry_d.solve_field` 每收到一条射线，就把它的**重心坐标**累加到命中三角形的三个顶点上：

```python
np.add.at(information[k], ids.ravel(), bc.ravel())
```

三个重心权重每条射线求和为 1，所以"擦边命中"贡献小、"命中中心"贡献大，全程没有阈值。
自由集由累计信息量决定：

```python
free_mask = (information.sum(0) >= settings.support_min_information) & ~excluded & ~locked
```

`support_min_information = 0.5` 的含义是"约等于一次穿过顶点中心的有效命中"。旧的
`seen by two frames` 规则保留为 `0`，用于复现历史结果。新增报告字段 `free_set_rule`、
`information_total`、`information_positive`，让"这次到底按哪条规则选的点"可查。

单视角因此可以更新它真正约束到的自由度；把它拉住的是 §3.3 的先验，不是帧数门限。
测试 `test_information_rule_admits_a_single_view` 直接对照两条规则：同一份单视角数据，
旧规则 `free=0` 且状态 `skipped_no_free_vertices`，新规则 `free>0` 且场发生位移。

### 3.3 观测置信度与形状锚点先验分离（B1）

观测置信度留在原处，按入射角平方、局部平面噪声、连拍归一化逐条方程计算，**不再被区域掩码
统一打折**。正面锚点改成求解目标里的显式项：

E = Σ 数据项(逐条方程置信度) + Σ 正则项 + **Σ_v λ_v (f_v − f_anchor_v)²**

λ_v 由调用方按顶点给出：正面覆盖区取满强度，其余区域取 `anchor_side_strength`（默认 0.2）。
于是"侧面观测看到正面区域"时，它的权重是完整的，只是要和 λ_v 竞争——可靠多视角证据能赢，
单条弱证据赢不了。实现为向系统追加对角行：

```python
prior_lam = settings.anchor_prior_weight * anchor_strength
system = vstack((A[:, free] * rw, diag(sqrt(prior_lam))[:, free], regularizer[:, free]))
```

线性化目标与非线性目标都由同一个 `energy()` 计算，先验项在两处口径一致。
`test_anchor_prior_holds_the_field_where_evidence_is_thin` 验证：同一份三视角数据，无先验时
场幅值 >0.3 mm，λ=50 时被压到不足一半。

### 3.4 覆盖率从"绝对零容忍"改成"相对阶段起点的比例容差"（B2）

原规则 `missing_after <= missing_before` 把"1 条射线扫到边界"和"整片区域失配"同等否决。
B0 的证据是硬的：`regularized` 阶段 29 次候选，**27 次被 `fixed_ray_coverage` 否决**。

新规则：

```python
coverage_allowance = round(settings.coverage_allowance_ratio * ray_count)   # 0.001
coverage_cap = max(missing_before, missing_baseline + coverage_allowance)
```

四个约束同时满足：① 评估射线集合在阶段开始时就固定（`diagnostic_rays` 只建一次）；
② 所有候选共用同一分母 `ray_count`；③ 容差相对**阶段起点** `missing_baseline` 计算，
不会逐步累积；④ 缺失惩罚与残差鲁棒函数不变（`fixed_ray_energy`，strict 8 mm / accounted 32 mm）。
每次候选都把 `missing_stage_baseline`、`missing_allowance`、`missing_cap` 写进
`line_search_trials`，可逐条复核。

验收（B2 实测）：`regularized` 阶段 `29 trials / 27 rejected → 1 trial / 0 rejected`；
`maximum_absolute_displacement_mm 1.091 → 1.711 mm`；三个视角 MAE 同步下降（§1）。

### 3.5 Loop 基底由数据在同一可行域内选（B3）

原实现只问几何安全：从 α=1.0 往下减半，**第一个可行就停**，数据代价从不参与。B0/B2 都因此
停在 α=0.5，把表面从线性插值推开 0.72 mm。

现在先在安全范围内求出最大可行 α（`alpha_safe`），再对 `{0, alpha_safe/2, alpha_safe}`
用固定的训练射线集合打分（每帧 400 点，seed=20261102，缺失按 8 mm 惩罚），取最小者：

```
candidate 0.00  cost 1.882790
candidate 0.25  cost 1.882402   ← chosen
candidate 0.50  cost 1.885584
```

α=0.5（原来自动选中的那个）恰好是三者中最差的。选择结果与全部候选代价写入
`report.json:loop_base.data_selection`，`alpha_safe` 与 `blend` 分开记录。

这一项的收益是三个机制里最小的（正面留出 0.889 → 0.882 mm），它修的是"最后一个没有数据依据的
选择"，不是主要来源。

---

## 4. 实验设计

### 4.1 数据与划分

复用 `zhoupei_cap/prepared` 的 20 帧观测，所有变体输入完全相同。身份关键帧由 `choose()`
在每个闭口段选 3 帧，实测 9 帧（正面 `closed_front_0/1/4`、左侧 `closed_left_0/2/4`、
右侧 `closed_right_1/3/4`），三个变体一致。每段剩余 2 帧不参与共享形状更新，作为**留出**评价。

张口 5 帧从不参与身份，是形状层面的天然留出。

### 4.2 评价口径

同帧网格配同帧观测：每帧用自己那一行 `dense_vertices` 对**自己**的深度图求残差，不存在跨帧
姿态/表情差。每帧 6000 点，`face_mask & depth>0`，seed=20261101，残差为沿光轴深度差，
门限 |残差| ≤ 30 mm，缺失计入覆盖率而不记 0。工具：`tools/eval_multiview.py`（分视角/状态/
训练角色/区域）。

区域按最近 68 点决定：鼻部 27–35、下颌与轮廓 0–16、口部 48–67、其余为 `face_other`。

### 4.3 变体

| 变体 | 配置 |
|---|---|
| B0 | `--fast --mid-rounds 1 --shape-fusion joint`（当前代码） |
| B1 | + `--support-min-information 0.5 --anchor-prior-weight 1.0`（含链路修复） |
| B2 | + `--coverage-allowance-ratio 0.001` |
| B3 | + `--loop-data-selection` |
| D | B3 + `--identity-views front`（侧面数据消融） |

D 的消融是**贯穿全链路**的：`fit_c` 身份相位、`identity_v2` 锚定解、`fit_d` 低/中频与正则场
都只吃正面身份帧。只把 `fit_c` 的 `side_depth_weight` 设为 0 不算消融——那样侧面数据仍会通过
低频场和正则场进入交付曲面。

### 4.4 口径的边界

§5 的所有数值是**采集深度一致性**：同一台相机、同一套内参下，重建表面对原始深度图的重合程度。
它**不是**独立几何精度——没有扫描配准、没有人工轮廓、没有独立外参。区域内比较（同一区域、
同一批射线、同一门限）是有效的；跨区域、跨设备的绝对值不具备测量学意义。

---

## 5. 实验依据

### 5.1 侧面数据到底买到了什么（B3 vs D）

留出闭口帧，单位 mm：

| 区域 | 正面 B3 | 正面 D | 左侧 B3 | 左侧 D | 右侧 B3 | 右侧 D |
|---|---|---|---|---|---|---|
| 鼻部 | 0.769 | 0.749 | **0.787** | 0.932 | **0.844** | 0.953 |
| 下颌/轮廓 | 1.238 | **0.871** | **1.676** | 2.145 | **0.996** | 1.249 |
| 口部 | 1.087 | **0.940** | **1.317** | 1.372 | **1.142** | 1.154 |
| 其余面部 | **0.668** | 0.702 | **0.583** | 0.663 | **0.643** | 0.777 |

侧面数据带来的净变化（B3 − D）：

- 左侧：下颌轮廓 **−0.469**，鼻部 **−0.145**，口部 −0.055，其余 −0.080；
- 右侧：下颌轮廓 **−0.253**，鼻部 **−0.109**，口部 −0.012，其余 −0.134；
- 正面：下颌轮廓 +0.367，口部 +0.147，鼻部 +0.020，其余 −0.034。

这是这次改进的**核心事实**：侧面观测把侧向的下颌轮廓和鼻部拉近 0.1–0.5 mm，代价是正面
下颌轮廓让出 0.37 mm。两者来自同一个共享身份——侧向证据在争用身份参数时，把正面轮廓的
一部分拟合优势换走了。

这个取舍是可以选择的：`--identity-views front` 就是"要正面最优"的配置，`all` 是"要侧向
一致"的配置。默认 `all`，因为交付物是人头整体，而侧向轮廓的 0.47 mm 大于正面的 0.37 mm。

### 5.2 三个机制各自的贡献

| 步骤 | 正面留出 | 左侧留出 | 右侧留出 | 关键证据 |
|---|---|---|---|---|
| B0 → B1 | −0.262 | −0.170 | −0.216 | 低频空转 0 → 位移 1.091 mm；自由顶点 0 → 1136 |
| B1 → B2 | −0.194 | −0.101 | −0.127 | 覆盖率否决 27/29 → 0/1；正则场位移 1.091 → 1.711 mm |
| B2 → B3 | −0.007 | +0.005 | 0.000 | Loop 候选 0.5 代价最高，改选 0.25 |

B1 与 B2 是主要来源，量级相当：前者让侧面证据终于能进入身份，后者让本来更优的候选步不再
被一条射线否决。B3 只关掉最后一个无依据选择。

### 5.3 为什么 B1 的收益比 B2 大在正面

B1 修复的是低频身份链。该阶段此前空转，所以交付曲面完全由正面锚 + Loop 决定；接通后，
低频场在**全部闭口视角**上重解，把正面残差也一起压下去了——正面的改善不是来自侧面证据，
而是来自"正面自己也被低频场重新解了一次"。

### 5.4 表情与身份的耦合：本采集上不是问题

`fit_c` 报告新增 `expression_bound`。B0–B3、D 全部 20 帧的 `per_frame_at_bound` 都是 0，
没有系数触到 ±4 的钳位。也就是说 P4 描述的"联合模式把表情推到硬钳位"在这份数据上**没有复现**，
本轮不据此调整先验。

关于先验归一化：`0.002 * sum(e²)/(frames*10)` 对帧数**不敏感**——`sum` 随帧数线性增长，
分母也线性增长，商不变。把它改成 `/(frames*30)` 会让该项减弱到三分之一，那是**主动削弱**先验，
不是修 bug。本轮保留原式，并在报告里写出这个事实，避免下一次把"报告字段变了"误判成"先验被稀释"。

`joint_shape_delta_from_anchor`（身份参数相对正锚的最大变化）B3 = 0.540、D = 0.596，量级相同；
`joint_anchor_update_mm`（正面帧总位移）包含姿态与表情自由度，不能当形状指标引用。

---

## 6. 时间成本

| 阶段 | B0 | B1 | B2 | B3 | D |
|---|---|---|---|---|---|
| identity_front_anchor | 37.6 | 29.8 | 27.5 | 25.3 | 25.3 |
| shared_identity_joint | 64.4 | 54.3 | 45.8 | 46.1 | 44.8 |
| fixed_identity_motion | 23.4 | 21.8 | 18.5 | 18.1 | 22.5 |
| low_frequency_shared_identity | 13.6 | 18.5 | 14.9 | 14.4 | 17.1 |
| motion_reestimate_after_identity | 21.8 | 18.6 | 18.4 | 18.0 | 22.1 |
| guarded_measured_dense_surface | 75.6 | 50.5 | 50.8 | 61.6 | 57.4 |
| **合计** | **244.8** | **200.1** | **181.5** | **189.6** | **195.8** |

改进后的流程**比基线快 22.5%**。机制不是"少算"，而是"少浪费"：自由集从 0 变成 1136 之后，
低频阶段不再空转后靠后续阶段补；覆盖率容差让求解器在第一次候选就接受，而不是耗掉 27 次
被否决的线搜索。

真实代价在三处，都可量化：

- 低频阶段 13.6 → 14.4 s（+0.8 s）：多了两次带自由度的求解；
- 密集阶段 75.6 → 61.6 s（−14 s）：表面更贴合，Loop 与残差阶段更早满足收敛条件；
- Loop 数据选择 0 → 约 2 s（3 个候选 × 9 帧 × 400 射线）。

---

## 7. 推荐配置

```pwsh
cd D:\proiect\facecap_airway_delivery
.\runtime\env\python.exe -m head_rgbd reconstruct `
  --sequence "runtime/outputs/<run>/prepared/sequence.json" `
  --output   "runtime/outputs/<新目录>" `
  --fast --mid-rounds 1 `
  --shape-fusion joint `
  --support-min-information 0.5 `
  --anchor-prior-weight 1.0 `
  --coverage-allowance-ratio 0.001 `
  --loop-data-selection
```

| 参数 | 值 | 作用 | 不建议改动的原因 |
|---|---|---|---|
| `--support-min-information` | 0.5 | 连续信息量门限，约"一次穿过顶点中心" | 再小会把擦边命中也算进来 |
| `--anchor-prior-weight` | 1.0 | 正面锚点先验强度，mm⁻² | 与数据项同一量纲；0 等于取消正面锚点 |
| `--coverage-allowance-ratio` | 0.001 | 允许的缺失比例（约 13/13156 条） | 再大就接近放弃覆盖约束 |
| `--loop-data-selection` | 开 | 在安全域内按数据选 Loop 混合量 | 关掉就等于退回"第一个可行即停" |
| `--identity-views` | `all` | 三视角共同定形状（`front` = 要正面最优的消融配置） | — |

需要与 0.4.0 完全可复现对照时，去掉除 `--fast --mid-rounds 1 --shape-fusion joint` 以外的全部参数，
即得到 B0。

---

## 8. 未完成与仍未解决的问题

- **B0 与 B3 不是同一进程两次运行**，而是各跑一次。本机单次运行波动约 ±10%，所以 §1 的
  耗时差（−55 s）大于该波动、可信；而 B2→B3 的 0.007 mm 差异低于口径判别力，不能当作收益。
- **覆盖率容差只做了单一取值（0.001）**。没有扫 0.0005/0.002/0.005，所以"0.001 是最优"没有证据，
  只有"0.001 明显优于 0"。
- **Loop 候选只有三档**（0、α_safe/2、α_safe），且训练/评价共用同一批身份帧射线
  （留出帧没有参与选择，但也没有专门用于选 Loop）。更严格的做法需要独立的 Loop 选择集。
- **密集残差仍未执行**（`--fast` 的既定取舍）。因此本条链路对"密集层质量"没有发言权。
- **单采集、单人、单设备**。9 个身份帧、12 个留出闭口帧加上 5 个张口帧，都来自同一次采集。
  跨采集、跨设备、跨头型的泛化没有验证。
- **§4.4 的边界仍然成立**：以上都是采集深度一致性，不是独立几何精度。要声称精度提升，
  需要扫描配准或人工轮廓真值，本轮没有。

---

## 9. 产物

| 内容 | 绝对路径 |
|---|---|
| 基线 B0 | `runtime\outputs\zhoupei_joint_baseline_20261102_101330` |
| B1 链路修复 | `runtime\outputs\zhoupei_joint_b1_20261102_102818` |
| B2 +覆盖率容差 | `runtime\outputs\zhoupei_joint_b2_20261102_103525` |
| **B3 推荐配置** | `runtime\outputs\zhoupei_joint_b3_20261102_103828` |
| D 侧面消融 | `runtime\outputs\zhoupei_joint_noSide_clean_20261102_104640` |
| 质量评估（五变体） | `runtime\eval\zhoupei_all`（`multiview.md` / `regions_by_view.md` / `multiview.json`） |
| 消融评估（B3 vs D） | `runtime\eval\zhoupei_ablation` |
| 交付网格 | 各 run 的 `delivery\head_closed_m.obj`、`head_open_m.obj` |
| 阶段网格与参数 | 各 run 的 `reconstruction\{front_anchor,guarded,low_frequency,alternated_motion,dense}` |

`zhoupei_cap` 原始目录、历史输出、采集与模型资产均未改动。

## 10. 代码改动

| 文件 | 改动 |
|---|---|
| `head_rgbd/pipeline.py` | 向 `identity_v2` 转发 `--shape-fusion`/`--joint-front-confidence`/`--identity-views` 与三个新预算（P1 的根因） |
| `head_rgbd/identity_v2.py` | joint 下三视角锚定 + 锚点先验；`identity_views` 消融开关；空自由集时显式报错；报告增加 `free_vertices`、`maximum_absolute_displacement_mm` |
| `head_rgbd/geometry_d.py` | `support_min_information` 连续支持；`anchor_field`/`anchor_strength` 显式先验；`coverage_allowance_ratio` 比例容差（相对阶段起点）；`energy()` 统一线性/非线性目标；`free_set_rule` 等报告字段 |
| `head_rgbd/fit_d.py` | `identity_views` 作用于低/中频与正则场；`--loop-data-selection` 按数据在安全域内选 Loop；三个新预算的 CLI 与报告 |
| `head_rgbd/fit_c.py` | `identity_views` 消融开关；`expression_bound` 触界报告 |
| `head_rgbd/cli.py` | 五个新参数的入口 |
| `tests/test_joint_fusion.py` | 新增 3 项：单视角可更新其约束的自由度、锚点先验回拉、负预算被拒 |
| `tests/test_contracts.py` | 阶段顺序测试的 args 补齐新参数 |

测试：`pytest tests -q` → **44 passed, 1 skipped**（跳过项为 `test_eval_sameframe` 的
"real run not available in this checkout"，与本改动无关）。
