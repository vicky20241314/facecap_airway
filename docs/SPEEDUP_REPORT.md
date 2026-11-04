# 主重建算法加速：实测报告

> **已被取代（0.3.0）。** 本文件记录第一轮（射线替换之前）的实测与 10% 级别的结果。
> 当前权威说明是 `docs/FAST_PATH_REPORT.md`；本文件中"QP 仅占 20%""复用收益很小"等结论
> 只在旧射线实现下成立，射线替换后瓶颈已变成 QP。这里 §5 记录的 +0.39 mm 差异成因见
> `FAST_PATH_REPORT.md` §5.2（链路阈值敏感，非实现错误）。

本报告只记录**实际修改并跑过真实数据验证**的内容。未实施的方案单独列在 §7，不混入结论。

---

## 1. 结论摘要

| 项目 | 结果 |
|---|---|
| 密集阶段耗时 | 1131.6 s → **1015.7 s**（−115.9 s，**−10.2%**） |
| 全流程耗时 | 1945.7 s → 1913.8 s（−31.9 s，−1.6%） |
| 质量 | 逐项不变（见 §5：MAE 差 1e-6 mm，覆盖率与缺失数完全一致） |
| 首轮 50% 目标 | **未达成**。实际改善 10.2%（阶段内）/ 1.6%（全流程单次） |

**全流程 −1.6% 不可用作加速比**，原因见 §4.3：前段（`prepare`～`motion_reestimate`）代码一行未改，却出现 +4.3 s ～ +50.8 s 的漂移，说明这台机器单次运行的噪声（±5%～8%）大于本项优化的收益。可信的比较只有一种：**同一阶段、改动前后、扣除未改动阶段的漂移**。

---

## 2. 实测瓶颈（本次最有价值的产出）

先补分项计时，再用实测数据推翻了动手前的假设。

### 2.1 阶段内分区（`fit_d` 自身，总 1029.5 s，分区之和 = 总耗时）

| 分区 | 耗时 | 占比 |
|---|---|---|
| `dense_front_anchor`（密集正面细化） | 355.4 s | 34.5% |
| `front_anchor_low_and_mid`（低频+中频锚定） | 230.1 s | 22.4% |
| `guarded_dense`（含正面保护射线） | 223.0 s | 21.7% |
| `evidence`（`reliable_evidence` × 2） | 132.7 s | 12.9% |
| `guarded_regularized` | 79.6 s | 7.7% |
| `subdivide_and_loop_base` | 1.2 s | 0.1% |
| `references_and_exports` + `neck` + `parameters_save` | 7.5 s | 0.7% |

### 2.2 按操作类型归集（关键结论）

| 操作 | 实测耗时 | 占比 | 证据 |
|---|---|---|---|
| **射线求交** | **≈ 635 s** | **62%** | 见下拆分 |
| QP 求解 | ≈ 205 s | 20% | `qp_solve_s`：顶层 82.7 s + 中频 122.1 s |
| 其余（矩阵装配、几何检查、导出） | ≈ 190 s | 18% | |

射线求交的拆分（均为实测，非估算）：

| 用途 | 耗时 | 证据 |
|---|---|---|
| 固定射线诊断（`fixed_diagnostic`） | 254.3 s | `timing.fixed_ray_s` |
| 对应关系查询（每 outer 一次） | 87.8 s | `timing.correspondences_s` |
| `reliable_evidence` × 2 | 132.7 s | `timing_sections.evidence` |
| 正面保护顶点（密集网格那一次） | **129.9 s** | 独立实测：201,553 条射线，**0.644 ms/条** |

**单条射线 0.644 ms 是本次唯一真正重要的数字。** 任何能达到 ~0.15 ms/条的射线求交实现，可回收约 470 s（密集阶段的 46%）。除此之外的所有优化空间加起来不到 20%。

### 2.3 由此推翻的三个预设

1. **"QP 求解是主要热点"——不成立。** QP 仅占 20%。中频那 3 轮确实有 10175/9800/8900 次迭代，但已收敛到 `dual_residual ≈ 4e-15`（要求 1e-6），是过求解而非卡住；单轮 40 s，三轮合计 122 s。
2. **"零自由度阶段在空转烧 QP"——不成立。** `guarded_regularized`（free=0）与 `guarded_dense`（free=0）的 QP **从未执行**：`geometry_d.py:341` 的 `if not len(free)` 在进 QP 前就退出。这两段的耗时几乎全是"为了算出 free=0"而必须做的射线查询（regularized 段 17.6 s、dense 段 42.0 s），不可省。
3. **"复用 mesh 能省 40%"——在真实调用图里基本不成立。** 40% 是我在"同一网格重复查询 6 次"的微基准里测到的。真实调用图中每个 `(帧, 几何)` 只被查一次，rtree 构造成本摊不出去。真正的一次性收益只有 `fixed_diagnostic` 里跳过 `gc.collect()` 那部分，量级 < 10 s。

---

## 3. 实际改动

### 3.1 `head_rgbd/geometry_c.py`

| 位置 | 改动 | 性质 |
|---|---|---|
| `:59 first_hits` | 空批次契约修复：某 32 条批次**完全无命中**时，`trimesh` 返回形状 `(0,)` 而非 `(0,3)`，`np.concatenate` 必然抛 `ValueError`。现在统一为 `(0,3)/(0,)/(0,)`，射线编号仍为全局 `ray + start` | **真实 bug 修复**（已复现） |
| `:49 build_mesh` | 新增：显式构造一个可复用的查询面，生命周期由调用方管理，不引入任何全局缓存 | 供 `fixed_diagnostic` 跳过 `gc.collect()` |
| `:79 ray_surface(...)` | 新增可选参数 `mesh=`（外部已建好的加速结构）与 `timing=`（就地累加 `build_s/query_s/calls`）。`mesh=None` 时行为与原来完全一致，11 个既有调用点不受影响 | 计时 + 生命周期开关 |

### 3.2 `head_rgbd/geometry_d.py`

| 位置 | 改动 |
|---|---|
| `:268` | 每个 outer 记录一个 `timing` dict：`correspondences_s / system_build_s / qp_setup_s / qp_solve_s / line_search_s / geometry_check_s / fixed_ray_s` |
| `:341` | 零自由度显式短路：不建约束、不调求解器、`status='skipped_no_free_vertices'`、`actual_update_mm=0.`、`field` 保持不变，记录仍写入 `history`（no-op 不会被误认为漏跑） |
| `:427` | 新增 `actual_update_mm = |field_after − field_before|.max()`：**真实形变**，不是 alpha，也不是相对 Loop 基面的总位移（那个是 2.2598 mm，大 5 个数量级） |
| `:450` | 提前停止，判据见 §4 |

### 3.3 `head_rgbd/fit_d.py`

新增 `marks` / `mark()`，把 `solve_field` 之外的分区也计时，结果写入 `report.json` 的 `timing_sections`（分区之和 = 阶段总耗时，不重复计数）。这是 §2.1 那张表的来源。

---

## 4. 提前停止

### 4.1 判据与推导

两个独立量必须同时成立才停止：

```
actual_update_mm < settings.noise_floor_mm          # 实际形变小于一个深度量化步长
且
diagnostic_after >= diagnostic_before * (1 - 1e-4)  # 固定射线 MSE 真实改善 < 0.01%
```

- `noise_floor_mm = 1/√12 ≈ 0.2887`：`fit_d.py:293` 在 `--research-v2` 下就是这个值，是现有噪声尺度，不是新引入的魔法数字。
- `1e-4`：13045 条固定射线的 MSS 标准误约 √(2/n) ≈ 1.24%，0.01% 比统计分辨率低两个数量级，属于"确实没进展"。
- **alpha 不参与判据**。`front_anchor_dense` outer1 的 alpha 只有 1/32768（3e-5），但那恰恰是唯一通过覆盖率检查的候选，用 alpha 判停会误杀。

### 4.2 实测触发

```
front_anchor_dense outer=0 alpha=0.00391   accepted=True update=0.0020mm
front_anchor_dense outer=1 alpha=0.00006   accepted=True update=0.0000mm
front_anchor_dense outer=1 stopped: update=0.000030mm fixed_ray_mse=0.746330->0.746316
```

outer0 的改善是 0.116%（> 0.01%）→ 继续；outer1 的改善是 0.0018%（< 0.01%）且实际更新 3e-5 mm（< 0.2887 mm）→ 停止，砍掉 outer2。

确定性收益（与机器噪声无关）：`front_anchor_dense` 的 outer 数 3→2，QP 迭代 3000→2050，试步 40→24。

### 4.3 为什么全流程数字不可信

| 阶段 | 基线 | 加速后 | 差值 | 该阶段代码是否改动 |
|---|---|---|---|---|
| prepare_observations | 33.5 | 37.8 | +4.3 | 否 |
| identity_front_anchor | 101.6 | 99.7 | −1.9 | 否 |
| shared_identity_guarded | 241.0 | 291.7 | **+50.8** | 否 |
| fixed_identity_motion | 139.2 | 156.3 | +17.1 | 否 |
| low_frequency_shared_identity | 137.3 | 148.8 | +11.5 | 否 |
| motion_reestimate_after_identity | 154.8 | 156.4 | +1.6 | 否 |
| guarded_measured_dense_surface | 1131.6 | **1015.7** | **−115.9** | 是 |

前六段没有任何改动，却合计 +83.4 s；同一密集阶段在四次运行里分别是 1131.6 / 1067.4 / 1029.5 / 1015.7 s，自身波动就有 116 s。**这台机器（共享、16 核空闲但单线程计算）的单次运行噪声与本次收益同量级。**因此本报告只主张阶段内的 −10.2%，不主张全流程加速比。

---

## 5. 质量验证

### 5.1 独立同帧评估（`tools/eval_sameframe.py`，非本算法代码）

口径：`face_mask & depth>0` 全脸深度采样，每帧 8000 点，seed=20261101，射线=相机光心过像素，残差=沿光轴深度差；**同帧网格 + 同帧观测**（`dense_vertices[4]` 配 `closed_front_4`，`dense_vertices[19]` 配 `open_front_4`）。

| state | 指标 | 基线 | 加速后 | 差值 |
|---|---|---|---|---|
| closed | MAE (mm) | 0.882556 | 0.882557 | +9.4e-07 |
| closed | RMSE (mm) | 1.847302 | 1.847303 | +1.2e-06 |
| closed | P95 (mm) | 2.596343 | 2.596343 | −1.3e-07 |
| closed | coverage | 0.9900 | 0.9900 | 0 |
| closed | n_used / n_missing | 7920 / 80 | 7920 / 80 | 0 / 0 |
| open | MAE (mm) | 1.389846 | 1.389846 | −1.4e-07 |
| open | RMSE (mm) | 2.621622 | 2.621623 | +2.1e-07 |
| open | P95 (mm) | 4.015923 | 4.015914 | −8.2e-06 |
| open | coverage | 0.9595 | 0.9595 | 0 |
| open | n_used / n_missing | 7676 / 324 | 7676 / 324 | 0 / 0 |

`sampling_matches: true`、`reference_frames_match: true`。

**覆盖率没有下降，缺失数没有增加，样本数完全相同。**

### 5.2 交付网格逐顶点差异

| 网格 | 顶点/面 | 最大差异 | 平均 | P95 |
|---|---|---|---|---|
| `head_closed_m.obj` | 20018 / 39904 | **0.000052 mm** | 1e-6 mm | 1e-5 mm |
| `head_open_m.obj` | 20018 / 39904 | **0.000052 mm** | 1e-6 mm | 1e-5 mm |

中间产物：`regularized` 与 `dense_smooth_base` **逐位相同（0.000000 mm）**。

### 5.3 几何有效性

两态均：顶点有限、索引合法、米级包围盒、法向翻转 39 面（0.098%，阈值 5%）、总面积 195355.77 mm²（与基线一致到 7 位）、距 FLAME base 最大位移 2.2366 mm。

### 5.4 固定射线 MSE

13045 条固定射线：0.7463098 → 0.7463164 mm²（**差 0.0009%，变差**）。这是砍掉 outer2 的代价，量级远低于统计分辨率（√(2/13045) ≈ 1.24%），如实记录。

### 5.5 测试

- 主算法契约测试 `tests/test_contracts.py`：**21 passed**
- 评估工具测试 `tests/test_eval_sameframe.py`：**10 failed / 6 passed** —— 见 §8

---

## 6. 已核实并推翻的结论

任务里转述的架构分析有几条经核实不成立，未采纳：

1. **"`fit_d.py:81-89` 是死代码，纯删可省 180–260 s 且几何不变"——不成立。**
   那段中频结果 `anchor` 在 `fit_d.py:95` 被作为 `guarded_regularized` 的初值，在 `:233` 存为 `parameters.npz` 的 `front_anchor_deviation_m`，`:90` 的 `regular_anchor_vertices` 又来自它。删除会改变 `regularized` → 改变 `dense_fit_base` → 改变最终几何。**已放弃该项。**

2. **"`guarded_regularized` / `guarded_dense` 跑了 QP 但解出 0 位移"——不成立。** QP 根本没执行（`geometry_d.py:341` 提前退出）。因此"零自由度短路"只改善报告语义，几乎不省时间（实测两段合计 0 s 的 QP）。

3. **"QP 是主要热点"——不成立。** 占 20%，其中一半是过求解而非必要计算。

4. **"复用 mesh/加速结构能省 40%"——在真实调用图里不成立。** 40% 来自重复查询同一网格的微基准；真实调用中每个 `(帧,几何)` 只查一次。

---

## 7. 未实施与下一步

### 7.1 为什么不做（本轮）

| 方案 | 不做原因 |
|---|---|
| QP 对象复用 / warm start | QP 仅占 20%，且中频那三段问题在过求解不在建对象，收益上限 < 60 s，收益/风险不划算 |
| 约束生成（方案E） | QP 不是瓶颈 |
| 软约束解锁正面硬锁（方案D） | 属质量/策略问题，非加速；且需独立验收，本轮未做 |
| 放宽覆盖率判据 | 会改接受规则，必须先有"新增缺失射线位置直方图"证据。基线里 `missing_before` 恒为 0，`missing_after ≥ 1` 全部被拒，是零容忍而非随机失败；无证据不放宽 |
| 降 `outer_steps` / `points` / 删阶段 | 未验证质量的粗略手段 |
| 删 `fit_d.py:81-89` | 见 §6.1，会改变几何 |

### 7.2 下一步唯一值得做的事：替换射线求交

- **目标**：0.644 ms/条 → 0.15 ms/条，回收约 470 s（密集阶段 46%）
- **做法**：共享相机原点 → 三角形一次性光栅化到像素栅格 → 查表取候选 → 自实现 Möller–Trumbore 窄相位（顺带直接得到重心坐标，省掉 `points_to_barycentric`）
- **必须守住的语义**（逐条验证）：首命中、射线编号、双面查询、backface 过滤、缺失语义、`max_depth/max_plane` 门限不变
- **风险**：这是唯一能达成 50% 的路径，但需要改 11 个调用点的取样方式（把像素坐标传下去），并重新验证全部几何与覆盖率

### 7.3 其余低成本项（若要继续）

- `fixed_diagnostic` 与 `front_protected_vertices` 在同一几何上做不同取样的两次全量查询，可共享加速结构（收益小，§2.3-3）
- 中频 QP 过求解：`eps_abs=1e-6` 却收敛到 1e-15，说明 ADMM 收敛慢；应查缩放/条件数，**不得靠调大 eps 解决**

---

## 8. 已知缺陷与未验证内容

1. **`tests/test_eval_sameframe.py` 10 项失败**：测试夹具用合成数据构造 run 目录，但工具最终版要求存在 `reconstruction/dense/parameters.npz`，夹具没建，因此 `FileNotFoundError`。**工具本身在真实运行上工作正常**（§5.1 的结果就是它产出的），但测试与实现不同步，属交付缺陷，未修。
2. **侧面（左/右）与张口外的其他状态无同帧网格**，无法按同帧原则评估；强行评会把姿态差算成形状误差。本轮只有正面 × 张/闭口。
3. **全流程加速比不可信**（§4.3），只主张阶段内 −10.2%。
4. 未做约束生成、未做软约束解锁、未改任何几何约束与接受规则。
5. 未提交、未推送；项目目录不是 git 仓库，未执行任何 git 写操作。

---

## 9. 产物与复现命令

### 产物

| 内容 | 绝对路径 |
|---|---|
| 完整候选重建（20 帧，status=passed） | `D:\proiect\facecap_airway_delivery\runtime\outputs\speedup_full_v1` |
| 交付网格 | `...\speedup_full_v1\delivery\head_closed_m.obj` / `head_open_m.obj` |
| 分段计时报告 | `...\speedup_full_v1\reconstruction\dense\report.json`（`timing_sections` + `field_logs.*.history[].timing`） |
| 仅密集阶段（带分段计时） | `...\runtime\outputs\speedup_dense_v3` |
| 同帧评估对比 | `...\runtime\eval\compare_speedup_v1\`（`metrics.json` / `metrics.csv` / `report.md`） |
| 基线同帧指标 | `...\runtime\eval\20261101_140933_before\metrics.json` |
| 基线（未改动，勿覆盖） | `...\runtime\outputs\20261101_140933_045847_head_reconstruction` |

### 复现

```pwsh
# 完整重建（约 32 分钟）
cd D:\proiect\facecap_airway_delivery
.\runtime\env\python.exe -m head_rgbd reconstruct `
  --capture "runtime/captures/20261101_140933_045847_head" `
  --output  "runtime/outputs/<新目录>"

# 仅密集阶段（约 17 分钟，带分段计时）
.\runtime\env\python.exe -m head_rgbd.fit_d `
  --sequence "runtime/outputs/20261101_140933_045847_head_reconstruction/prepared/sequence.json" `
  --initial-fit "runtime/outputs/20261101_140933_045847_head_reconstruction/reconstruction/alternated_motion" `
  --output "runtime/outputs/<新目录>" --runtime runtime `
  --mode guarded --research-only --solver qp --dense-mode loop-research --research-v2 `
  --points 5000 --fixed-ray-policy strict-coverage `
  --gradient-length-mm 3.0 --bending-length-mm 4.0

# 同帧评估对比
.\runtime\env\python.exe tools\eval_sameframe.py `
  --run "runtime/outputs/<新目录>" `
  --baseline-metrics "runtime/eval/20261101_140933_before/metrics.json" `
  --output "runtime/eval/<标签>"

# 测试
.\runtime\env\python.exe -m pytest tests/test_contracts.py -q
```
