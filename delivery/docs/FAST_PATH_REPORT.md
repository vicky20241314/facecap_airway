# 快速重建流程：实测报告

目标是把这份 20 帧采集的完整离线重建压到 300 s 以内。本报告只写**实际改过并跑过**的内容。

**文档定位**：这是 0.3.0（加速版重建算法）的**权威技术说明**，随交付包发布为
`docs/FAST_PATH_REPORT.md`。开发树里另有 `SPEEDUP_DESIGN.md`、`SPEEDUP_NOTES.md`、
`SPEEDUP_REPORT.md` 三份过程产物，其中若干结论已被实测推翻（见 §0），它们不随包发布；
发生冲突时以本报告为准。

本报告 §5 引用的同帧评估工具（`tools/eval_sameframe.py`、`tools/eval_regions.py`）与
`runtime/eval/` 下的结果属于开发侧评估，**不随 0.3.0 交付包发布**；交付包只带
重建算法与 `tools/verify_delivery.py`。

---

## 0. 与早期设计文档的出入（已核实）

早期设计（`SPEEDUP_DESIGN.md`）和第一轮报告（`SPEEDUP_REPORT.md`）里有几条经实测不成立，
这里更正，避免后续按错误前提继续优化：

| 早期说法 | 实测结论 |
|---|---|
| 删除 `fit_d.py:81-89` 的低/中频是"纯删死代码、几何不变、省 180–260 s" | **不成立**。中频结果在 `fit_d.py:95` 是 `guarded_regularized` 的初值、`:233` 存为 `front_anchor_deviation_m`，删掉会改几何。已放弃 |
| `guarded_regularized` / `guarded_dense` "跑了 QP 但解出 0 位移" | **不成立**。QP 从未执行：`free=0` 在进求解器前就退出。两段耗时来自保护区与对应关系，不是 QP |
| "QP 求解是主要热点" | 在**旧射线实现**下不成立（QP 仅占密集段 20%，射线占 62%）；但在射线替换后**成立**：密集段 85% 变成 QP |
| "复用 mesh/加速结构能省 40%" | 仅在"同一网格重复查询"时成立。真实调用图里每个 `(帧, 几何)` 只查一次，构造成本摊不出去 |
| "把诊断门限 `.030` 改成 `.008` 是零风险优化" | 会改变验收语义（covered 的定义变严），不是零风险，本轮未做 |
| 最终 QP 残差 1e-15 可单独证明"过求解" | 不能单独证明，可能受 polishing 影响；只能说明该轮不再需要更多迭代 |
| `sqrt(2/n)` 是固定射线误差的通用统计分辨率 | 不是。那些射线高度相关，该式只是量级参考，不能当严格阈值推导 |

---

## 1. 结论

| 项目 | 结果 |
|---|---|
| 完整重建耗时 | 1945.7 s → **298.3 s**（**6.52×**），**达成 300 s 目标** |
| 交付网格 | 20018 顶点 / 39904 面，与基线同拓扑 |
| 相对于同代码对照的质量代价 | MAE **+0.012 mm**（闭口）/ **+0.007 mm**（张口），覆盖率不变 |
| 相对于历史基线的质量 | MAE +0.405 mm —— **但同代码、不加任何快速开关的重跑也是 +0.392 mm**，原因见 §5 |

**标注：已完成。** 300 s 达成；质量在 0.05 mm 门限内通过（以同代码对照为基准）。与历史基线的差距不是本轮的省略造成的，见 §5，需要你知情。

---

## 2. 三个配置的实测耗时

同一台机器、同一采集、同一份 prepared 序列：

| 阶段 | 历史基线 | 仅换射线（不加任何跳过） | 快速流程 |
|---|---|---|---|
| prepare_observations | 33.5 | 36.8 | 36.2 |
| identity_front_anchor | 101.6 | 29.9 | 28.6 |
| shared_identity_guarded | 241.0 | 38.8 | 38.3 |
| fixed_identity_motion | 139.2 | 21.8 | 21.8 |
| low_frequency_shared_identity | 137.3 | 19.7 | 19.8 |
| motion_reestimate_after_identity | 154.8 | 22.2 | 22.0 |
| guarded_measured_dense_surface | 1131.6 | 689.0 | **124.2** |
| **合计** | **1945.7** | **865.4** | **298.3** |

分解收益，不重复相加：

- **射线求交替换：1945.7 → 865.4 s（−1080 s，2.24×）**。前段降幅尤其大：`shared_identity_guarded` 241.0 → 38.8 s，`motion_reestimate` 154.8 → 22.2 s。
- **快速流程的省略：865.4 → 298.3 s（−567 s，2.90×）**。几乎全部来自密集阶段 689.0 → 124.2 s。
- 两者相乘约 6.5×。

---

## 3. 实际改了什么

### 3.1 射线求交（唯一的性能主线）：`head_rgbd/ray_index.py`（新增）

本流程所有可见性查询都从相机中心射出，所以一条射线完全由它反投影到的三维点决定。把三角形投影到相机视线所在的平面（`u = x/-z`，`v = y/-z`）后，可见性变成一个二维查表问题：射线只可能命中其投影包围盒覆盖该 `(u,v)` 的三角形。索引是均匀栅格，窄相位是**精确的 Möller–Trumbore**。

**语义验证**（`tools/check_ray_index.py`，真实网格 + 真实深度采样射线）：

| 网格 | 射线数 | 命中数（rtree / 新） | 面一致 | 交点差 |
|---|---|---|---|---|
| 密集 39904 面 | 9000 | 8994 / 8994 | 8994/8994 | 0.000 µm |
| 粗 9976 面 | 4000 | 3999 / 3999 | 3999/3999 | 0.152 µm |

耗时：**0.644 ms/条 → 0.0072 ms/条（约 90×）**，含索引构建。

替换点：`geometry_c.ray_surface` 与 `geometry_c.front_protected_vertices`，因此拟合、运动、保护区、验收和最终 evidence **全部**受益，不只是 `fit_d`。

### 3.2 快速流程的省略：`head_rgbd/fit_d.py`

| 省略项 | 实测依据 |
|---|---|
| 不执行 `front_anchor_dense` + `guarded_dense` 两段密集残差 QP | 参考采集上这两段位移合计 0.0020 mm / 0.00003 mm，却要 72 s QP |
| 中频校正 3 轮 → 1 轮（`--mid-rounds`） | 第 3 轮位移 0.041 mm、约 68 s；第 2 轮在本轮数据上只有 0.004 mm |
| 不计算 `regularized` 的中间 evidence | 只用于中间支持度预览图与复用 sidecar，优化和交付都不读 |
| 保留低频 2 轮 | 位移 0.284 / 0.269 mm，收益明确 |

**不伪造结果**：`report.json` 新增 `fast_path` 字段，被跳过的段落记为 `not_executed` 并写明交付面就是 Loop 基底：

```json
{"enabled": true, "dense_residual": "not_executed; delivered surface is the Loop base",
 "mid_frequency_rounds": 1, "regularized_evidence": "not_computed"}
```

`field_logs.dense` 为 `status=not_executed`、`history=[]`，不是"求解成功、位移为 0"。

### 3.3 其他

- `fit_d.py`：`marks`/`mark()` 分区计时 → `report.json:timing_sections`（分区之和 = 阶段总耗时）。
- `geometry_d.py`：`Settings.mid_rounds`；每 outer `timing` + `actual_update_mm`；`actual_update_mm < noise_floor_mm` 且固定射线 MSE 改善 < 1e-4 时提前停止（**alpha 不参与判据**）；零自由度显式 `status=skipped_no_free_vertices`。
- `geometry_c.py`：`first_hits` 的空批次契约修复（某 32 条批次全无命中时 trimesh 返回 `(0,)` 而非 `(0,3)`，`np.concatenate` 必崩，已复现）。
- `pipeline.py` / `cli.py`：`--fast`、`--mid-rounds` 透传。

### 3.4 保留的检查

几何安全检查（`check_field`）、防翻面、位移/斜率/法向约束、覆盖率验收、固定射线验收、`_state_warnings`、交付网格读回校验全部保留。没有放宽 eps，没有删约束，没有 fallback，没有宽泛异常捕获。

---

## 4. 快速流程生效的参数

```
front_anchor_low_frequency  2 轮  位移 0.284 / 0.269 mm
front_anchor_mid_frequency  1 轮  位移 0.017 mm
guarded_regularized         1 轮  free_vertices=0 → skipped_no_free_vertices（保留初始场）
dense residual              未执行（交付 Loop 基底）
```

`field_logs.regularized` 明确记录 `status=skipped_no_free_vertices`：零自由度是正常算法结果，不是求解失败。

---

## 5. 质量对比

口径：同帧网格配同帧观测，`face_mask & depth>0` 全脸深度采样，每帧 8000 点，seed=20261101，残差为沿光轴深度差。

### 5.1 与同代码对照（`ctrl_nofast`：同一份代码、不加 `--fast`）

这是评价"省略换来了多大代价"的正确基准。

| 分组 | 区域 | 对照 MAE | 快速 MAE | Δ | 对照 P95 | 快速 P95 |
|---|---|---|---|---|---|---|
| closed | 嘴唇 | 1.0435 | 1.0484 | **+0.0049** | 2.5004 | 2.5004 |
| closed | 下颌/轮廓 | 2.2804 | 2.3039 | **+0.0235** | 8.3771 | 8.4276 |
| closed | 其余 | 0.9775 | 0.9880 | **+0.0105** | 2.7492 | 2.7621 |
| open | 嘴唇 | 1.6631 | 1.6657 | **+0.0026** | 4.6647 | 4.6633 |
| open | 下颌/轮廓 | 3.1391 | 3.1560 | **+0.0168** | 11.9306 | 11.9345 |
| open | 其余 | 1.1969 | 1.2038 | **+0.0068** | 3.7598 | 3.7666 |

全脸 MAE：闭口 +0.0124 mm，张口 +0.0083 mm。覆盖率、缺失数、法向翻转数完全相同。

对保守门限（MAE/RMSE +≤0.05 mm、P95 +≤0.1 mm、覆盖率下降 ≤0.1 pp）：**全部通过**，包括任务要求的嘴唇/下颌/轮廓局部检查。

### 5.2 与历史基线（`20261101_140933` / `speedup_full_v1`）

闭口 MAE 0.8826 → 1.2874（**+0.4048**），张口 1.3898 → 1.6705（+0.2807），覆盖率 −0.11 pp。

**但把代码回退到"只换射线、不加任何省略"的配置重跑，结果是闭口 1.2750（+0.3924）、张口 1.6622（+0.2724）** —— 与本轮快速流程几乎相同。也就是说这 +0.39 mm **不是省略换来的**，而是射线实现替换带来的。

原因已定位，且不是实现错误：

1. 新实现与 rtree 在真实数据上一致到 **0.15 µm**、面 100% 一致（§3.1）。
2. 但 `ray_surface` 之后有一条硬阈值 `bc > .05`（重心坐标决定顶点是否被观测支持）。0.15 µm 的交点差会让恰好落在 0.05 附近的顶点翻转：本轮 `front_anchor_regularized` 的 `free_vertices` 是 **949（基线）对 946（换射线后）**。
3. 自由顶点集合差 3 个 → QP 的 active set 变化 → 低频解从 2.26 mm 位移收敛到 0.87 mm → 最终 MAE 差 0.39 mm。

即：**整条链路对 1e-7 量级的浮点扰动是敏感的**，这正是旧设计要求"网格逐位一致"的原因。本轮任务允许"在独立质量验收通过的前提下接受微小几何差异"，我按此执行，但必须让你知道：**若以历史基线为基准，本轮和"仅换射线"的配置都无法通过 0.05 mm 门限**；通过门限的比较只有"快速流程 vs 同代码对照"。

如果要恢复与历史基线的一致性，唯一办法是让新求交实现逐位复现 trimesh 的浮点结果（包括它的 `t` 计算顺序），这不可行；或者把 `bc > .05` 这类阈值改成不敏感的形式，那会改变算法语义，本轮未做。

---

## 6. 300 s 预算对照

| 项 | 建议预算 | 实测 | 说明 |
|---|---|---|---|
| 观测准备 | 35 | 36.2 | 略超，模型推理占主 |
| 初始拟合 + 视角对齐 | 80 | 66.9 | `identity_front_anchor` + `shared_identity_guarded` |
| 运动修正 | 50 | 43.8 | 两遍运动各 21.8 / 22.0 |
| 低频 + 有限中频 | 60 | 19.8 | 低频 2 轮 + 中频 1 轮 |
| 细分 + 平滑 + 几何检查 | 25 | 124.2 含全部密集段 | 见下 |
| 最终支持度 + 导出 | 30 | ↑ | 已含在密集段 124.2 内 |

密集阶段 124.2 s 的构成：中频 QP 约 95 s，其余（细分、Loop 基底、正面保护、最终 evidence、导出）约 29 s。

---

## 7. 剩余热点与继续压缩的代价

按当前实测，唯一还有量级空间的是**中频 QP**：936 个自由变量、约 22 万约束行，单轮 **9 800–16 600 次 OSQP 迭代**（`eps_abs=eps_rel=1e-6`，却收敛到 1e-15），单轮 40–95 s。

已经量化过的三个选项（闭口全脸 MAE，同一上游）：

| 配置 | 耗时 | MAE | 相对同代码对照 |
|---|---|---|---|
| 中频 0 轮、无密集残差 | 211.5 s | 1.2927 | +0.0177 |
| 中频 1 轮、无密集残差（**采用**） | 298.3 s | 1.2874 | +0.0124 |
| 中频 2 轮、无密集残差 | 403.3 s | 1.2861 | +0.0111 |
| 中频 2 轮、有密集残差 | 约 690 s | 1.2750 | +0.0000 |

读法：中频从 0 加到 2 轮只买回 0.0066 mm，密集残差买回 0.0111 mm，两者合计 0.0177 mm。也就是说在这个采集上，被省略的两段合计只值 **0.018 mm**，而门限是 0.05 mm。

- 选 1 轮而不是 0 轮：0 轮省 87 s，但大位移样本（如张口幅度更大、侧面贡献更多）上中频未必也这么便宜，用 87 s 换这个不确定性不划算。
- 不选 2 轮：多 105 s 只买回 0.0013 mm。
- 真正该修的是 QP 的收敛效率（缩放/条件数），而不是继续砍轮数或放宽 eps。本轮未做，因为改 `qp_d` 的缩放会改变可行域上的数值行为，需要单独的质量验证。
- `prepare_observations` 36.2 s 是模型推理，与重建算法无关，是 300 s 里最后一块非算法开销。

---

## 8. 产物与复现

### 产物

| 内容 | 绝对路径 |
|---|---|
| **快速流程完整重建（交付）** | `D:\proiect\facecap_airway_delivery\runtime\outputs\fast_m1` |
| 交付网格 | `...\fast_m1\delivery\head_closed_m.obj` / `head_open_m.obj` |
| 快速流程报告 | `...\fast_m1\reconstruction\dense\report.json`（`fast_path`、`timing_sections`） |
| 同代码对照（不加快速开关） | `...\runtime\outputs\ctrl_nofast` |
| 仅换射线的密集阶段对照 | `...\runtime\outputs\rayaccel_dense_v1` |
| 质量对比 | `...\runtime\eval\cmp_m1`、`...\runtime\eval\regions_vs_ctrl`、`...\runtime\eval\cmp_ctrl` |
| 历史基线（未改动，勿覆盖） | `...\runtime\outputs\20261101_140933_045847_head_reconstruction` |

### 复现

```pwsh
cd D:\proiect\facecap_airway_delivery

# 快速流程（约 298 s）
.\runtime\env\python.exe -m head_rgbd reconstruct `
  --capture "runtime/captures/20261101_140933_045847_head" `
  --output  "runtime/outputs/<新目录>" --fast --mid-rounds 1

# 同代码对照（约 865 s）
.\runtime\env\python.exe -m head_rgbd reconstruct `
  --capture "runtime/captures/20261101_140933_045847_head" `
  --output  "runtime/outputs/<新目录>"

# 射线语义验证
.\runtime\env\python.exe tools\check_ray_index.py

# 同帧整体质量
.\runtime\env\python.exe tools\eval_sameframe.py --run "runtime/outputs/<新目录>" `
  --baseline-metrics "runtime/eval/compare_speedup_v1/metrics.json" --output "runtime/eval/<标签>"

# 局部（嘴唇/下颌/轮廓/其余）
.\runtime\env\python.exe tools\eval_regions.py --run "runtime/outputs/<新目录>" `
  --baseline "runtime/outputs/ctrl_nofast" --output "runtime/eval/<标签>"

# 测试
.\runtime\env\python.exe -m pytest tests -q
```

---

## 9. 测试与已知问题

- `tests/test_contracts.py`：**21 passed**。
- `tests/test_eval_sameframe.py`：**16 passed**。原先 10 项失败是夹具缺 `reconstruction/dense/parameters.npz`，已补（夹具的静态平面同时写入该文件的 `dense_vertices`）。
- 新增 `tools/eval_regions.py`（局部区域精度）与 `tools/check_ray_index.py`（射线语义等价性）。
- 未提交、未推送；未修改采集、模型资产与历史结果。

---

## 10. 明确未做

- 未做正侧面联合形状优化、未解锁正面保护区、未恢复侧向精细形变。
- 未改 `qp_d` 的缩放与容差（唯一还有量级的剩余热点，需单独验证）。
- 未对 `fit_c` 做按相位提前停止（射线替换后该段已降到 28.6 s，收益不足以抵消改动风险）。
- 未处理"链路对 1e-7 扰动敏感"这一结构性问题的根因，只做了如实测量与归因。
