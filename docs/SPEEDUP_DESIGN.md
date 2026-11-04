# SPEEDUP_DESIGN — facecap_airway_delivery 性能热点定位与优化设计

> **注意：本设计文档的部分结论已被实测推翻，不要按它继续优化。** 权威说明见
> `docs/FAST_PATH_REPORT.md` §0。具体：
> ① 删除 `fit_d.py:81-89` 不是死代码（中频是 `guarded_regularized` 的初值，`parameters.npz`
> 也用它），删了会改几何；② 零自由度阶段的 QP 从未执行，"短路"省不到时间；
> ③ 射线替换后瓶颈已从射线转到 QP（密集段 85%）；④ mesh 复用只在同一几何重复查询时有效，
> 真实调用图里没有那样的重复。本文件保留作过程记录。

> 状态：**设计文档（只读分析 + 方案）**。本文件撰写期间未修改任何源码、资产或历史运行结果。
> 基线运行：`D:\proiect\facecap_airway_delivery\runtime\outputs\20261101_140933_045847_head_reconstruction`
> 机器：AMD Ryzen 7 5800H（8 核 / 16 逻辑核），RAM 15.4 GB（采集时可用约 7 GB），RTX 3060 Laptop 6 GB
> 环境：Python 3.10.22 / numpy 1.26.4 / scipy 1.11.4 / trimesh 4.4.9 / rtree 1.4.1 / osqp 1.0.5 / **无 pyembree、无 embreex**

---

## 0. 一页结论

| 结论 | 证据 |
|---|---|
| **射线求交的真正瓶颈不是"窄相位"，是"每条射线各查一次 3D rtree"加 trimesh 的 Python 层。** | 粗模 4850 条射线：宽相位（per-ray rtree）**518 ms**，窄相位（平面+重心）仅 ~100 ms，**其余 ~2300 ms 是 trimesh 逐调用开销**（bench5/bench8）。chunk 32→512 只省 8%–15% |
| **`fit_d` 前段的 `front_anchor_low_frequency` + `mid_frequency` 是死计算**，结果只用于一个交付链路无消费者的 OBJ 导出。 | `fit_d.py:81-89`；`dense_fit_base = interpolation_vertices`（`:113`）只依赖 `regularized`；`:187` 的 `low_frequency_anchor` 仅出现在 `stages` 导出，而 `config/defaults.json` 的 `export.mesh_stage = "dense"`。另：`alternated_motion/parameters.npz` 与 `front_anchor/parameters.npz` 的 do-nothing 断言已在本机通过 |
| **`guarded_regularized` 与 `guarded_dense` 的求解/线搜索/固定射线完全没执行**，因为自由度在进入 QP 之前就是 0。 | `field_logs.regularized.history[0] = {stage, outer:0, free_vertices:0, accepted:[...]}`，**无 `line_search_trials`、无 `solver`、无 `fixed_ray_diagnostic`**；`parameters.npz` 实测 `regularized_vertices` 与 `front_anchor_vertices` 最大差 **0.000000 mm**（dense 侧同样 0.000000 mm） |
| **`front_anchor_dense` 的 40 次试步里 37 次因"新增 1–4 条缺失固定射线"被拒**；每次试步建两次网格、查两次 13045 条固定射线（`check_field` 不建网格，`fixed_diagnostic` 建一次，`ray_surface` 内还有一次 `gc.collect()`）。 | `field_logs.front_anchor_dense.history[*].line_search_trials`，missing 序列 `[4,4,2,2,1,1,1,1,0]` / `[4,4,2,2,2,1×9,0]` / `[4,4,2,2,2,1×10,0]`；三个 outer 的 `missing_before` 恒为 0 |
| **可回收的确定性时间约 440–690 s（占总 1945.7 s 的 22.5%–35.5%），几何逐位等价。** | 见 §5 汇总表 |

**本次建议做的事**：先落地计时插桩（零风险、把后续所有收益变成可测），再落地"删前段死代码 + 射线求交宽相位改造 + 提前停止 + 零自由度语义"，最后**只做诊断**地评估 `front_anchor_dense` 的覆盖率判据（唯一涉及**质量判据**的改动，必须独立验收）。

**本次明确不做**：降低 `points` / `steps`、删除任何阶段、放宽 `eps_abs/eps_rel`、删安全约束、引入 embree、并行化、接手 `loss_analysis/`、改采集或模型资产。

---

## 1. 实测事实表

### 1.1 阶段耗时（`run_report.json → stages[*].elapsed_s`）

| 阶段 | 耗时 (s) | 占比 | 模块 | 说明 |
|---|---:|---:|---|---|
| prepare_observations | 33.463 | 1.7% | `head_rgbd.prepare` | 20 帧 landmark 预测，每帧 0.81–0.99 s，`--landmark-device cpu` |
| identity_front_anchor | 101.619 | 5.2% | `head_rgbd.fit_c --front-only` | 240 步；fitting 67.36 s，其余 34 s 是写盘+精化+证据 |
| shared_identity_guarded | 240.950 | 12.4% | `head_rgbd.fit_c --front-anchor` | 240 步；fitting 122.45 s，其余 112 s |
| fixed_identity_motion | 139.210 | 7.2% | `head_rgbd.motion_v2` | 80 步 |
| low_frequency_shared_identity | 137.321 | 7.1% | `head_rgbd.identity_v2` | 仅 2 个 outer + 1 个 side outer，却花了 137 s |
| motion_reestimate_after_identity | 154.849 | 8.0% | `head_rgbd.motion_v2 --identity-field` | 80 步 |
| **guarded_measured_dense_surface** | **1131.590** | **58.2%** | `head_rgbd.fit_d` | 内部 `elapsed_s=1126.15` |
| 合计 | 1939.002 | 99.7% | | 与 1945.717 s 差 6.7 s 为 stage 间进程启动开销 |

前段 5 个阶段合计 **773.9 s（39.8%）**。

### 1.2 dense 阶段内部 outer 轮次与求解统计（`reconstruction/dense/report.json → field_logs`）

| stage | outer | free_vertices | solver iters | constraint_rows | alpha | 试步数 | 因覆盖率被拒 | missing_before→after | fixed_ray_mse (mm²) | 最大位移 (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| front_anchor_low_frequency | 0 | 949 | 100 | 225,215 | 0.25 | 3 | 2 | 0→0 | 2.446648→1.745571 | — |
| front_anchor_low_frequency | 1 | 945 | 275 | 225,215 | 1.0 | 1 | 0 | 0→0 | 1.745571→0.720735 | — |
| front_anchor_mid_frequency | 0 | 936 | — | — | 0.125 | 1 | 0 | — | — | — |
| front_anchor_mid_frequency | 1 | 934 | — | — | 0.125 | 1 | 0 | — | — | — |
| front_anchor_mid_frequency | 2 | 935 | — | — | 0.03125 | 1 | 0 | — | — | — |
| **guarded_regularized** | 0 | **0** | **从未调用** | — | — | **0** | — | — | — | **0** |
| front_anchor_dense | 0 | 2881 | 1100 | 1,213,933 | 1/256 | 9 | 8 | 0→0 | 0.747179→0.746330 | 2.259743 |
| front_anchor_dense | 1 | 2881 | 950 | 1,213,933 | 1/16384 | 15 | 14 | 0→0 | 0.746330→0.746316 | 2.259759 |
| front_anchor_dense | 2 | 2881 | 950 | 1,213,933 | 1/32768 | 16 | 15 | 0→0 | 0.746316→0.746310 | 2.259767 |
| **guarded_dense** | 0 | **0** | **从未调用** | — | — | **0** | — | — | — | **0** |

固定射线数：`front_anchor_regularized` 12,984 条；`front_anchor_dense` **13,045** 条（`fixed_ray_diagnostic.ray_count`）。
`front_anchor_dense` 三个 outer 的 `missing_rays` 序列：
- outer0：`[4, 4, 2, 2, 1, 1, 1, 1, 0]`
- outer1：`[4, 4, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]`
- outer2：`[4, 4, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]`

**注意**：`fixed_ray_diagnostic.missing_before` 在三个 outer 里都是 0，而**所有被拒试步的 missing_after 都 ≥ 1**。也就是说：**只要候选几何相对当前几何"多丢 1 条命中"，就必须退到 alpha=0 为止；而 alpha→0 的极限必然满足 missing=0**。这是结构性的、与步长无关的否决规则。

### 1.3 问题一：多视角细化"无可优化顶点"（事实核对）

`reconstruction/dense/parameters.npz` 逐数组实测：

| 拓扑 | 顶点数 | ≥2帧支持且未被正面保护 | 解剖排除 | `front_protected` | `final − front_anchor` 最大位移 |
|---|---:|---:|---:|---:|---:|
| coarse（`regularized` 阶段） | 5,023 | **0** | 1,500 | 2,824 | **0.000000 mm** |
| dense（`guarded_dense`） | 20,018 | **0** | 6,252 | 9,542 | **0.000000 mm** |

补充交叉校验：`reconstruction/guarded/report.json → front_anchor.changed_side_detail_vertices = 0`，`maximum_side_detail_change_mm = 0.0`。

结论表述（**不要**沿用旧诊断的"10476 自由顶点"）：
> 在**当前采集几何**下，能被 ≥2 帧独立观测、且未被正面保护、且不在解剖排除区内的顶点数为 **0**。两个阶段都是零自由度，属于"结构性空转"，不是"参数没调好"。
> 因果其实是**反的**：不是"没有自由顶点所以侧向细化无效"，而是 **`shared_identity_guarded` 这一步（240.95 s）本身也没有产生任何侧向细化**（changed=0），于是下游所有"细化"阶段自然无处可动。若要恢复侧向细化能力，必须动 `fit_c --front-anchor` 的自由集定义，**那不是加速问题，是质量/策略问题**，本设计不擅自改，只在 §6 列为"暂不做"并给出前置诊断。

### 1.4 问题二：`front_anchor_dense` 因覆盖率反复缩步（事实核对）

- alpha=1 时：`geometry_valid=true`、`fixed_ray_mse_mm2=0.605137`（比当前 0.747179 **低 19%**）、`objective=1.405177`（比 1.607002 低 12.6%），但因 `missing_rays=4` 被拒。
- 最终接受 alpha=1/256：MSE `0.747179→0.746330`（改善 **0.116%**），最大位移 **2.2597 mm**（这 2.26 mm 来自 `dense_fit_base` 与 `dense_base` 之间的 Loop 基面混合，**不是这一步产生的**；这一步实际把该状态从 2.259743 推到 2.259743，逐位看只改了 1e-6 mm 量级）。
- 40 次试步 × 每步 (3 或 9 帧) × 5000 射线 × 3 个 outer 的全量重投射。

**识别到的三个独立缺陷**（§4.3 逐一给改法）：
1. `missing_after <= missing_before` 是**绝对整数**判据，在 ray 数 13045 的量级上等价于"零容忍"，与 alpha 无关地强制回到 0。
2. `fixed_diagnostic` 的射线**当前 0 条缺失**，但用了 `max_depth=0.030`（`geometry_d.py:238`），比主查询的 `.008` 宽 3.75 倍。这个宽门限把"深度上明显不对的命中"也算成 covered，于是"覆盖率"判据的语义与"这步是否让几何更差"脱钩。
3. `safe/check_field` 与 `fixed_diagnostic` **在完全相同的 candidate 上各建一次网格、各查一次 13045 条射线**，且 `ray_surface` 内部还有一次 `gc.collect()`（75 ms）。→ **单次试步 2×(建网格 + rtree + 13045 查询 + gc)**。

### 1.5 实测微基准（本机、只读、可复现）

射线基准：4850 条真实量级射线，原点在相机中心，方向指向头部表面外扩 2%。粗模 9,976 面 / dense 39,904 面。

| 操作 | coarse (9,976 面) | dense (39,904 面) | 数据源 |
|---|---:|---:|---|
| `trimesh.Trimesh(v,f,process=False)` 构建 | 0.099 ms | 0.074 ms | bench_ray |
| 首次查询（内含 rtree 构建） | 39.3 ms | 160.2 ms | bench_ray |
| **`mesh.ray.intersects_location` 全量（=宽相位+窄相位）** | **938–1057 ms** | **2731–2777 ms** | bench_ray / bench3 |
| `first_hits` chunk=32（现网） | 1089 ms | 2590 ms | bench_ray |
| `first_hits` chunk=512 | 856 ms | 2534 ms | bench_ray |
| `gc.collect()` 一次 | 75.1 ms | 76.8 ms | bench_ray |
| **拆解·宽相位** `ray_triangle_candidates`（每条射线查一次 3D rtree） | **509–519 ms** | **1430–1509 ms** | bench5 / bench8 / bench10 |
| 拆解·候选数 | 788,761（**162.6/ray**，max 1794） | 2,716,092（**560.0/ray**，max 7049） | bench5 / bench8 |
| 拆解·窄相位 `planes_lines` | 71.9–102.2 ms | 341.9 ms | bench5 / bench8 |
| 拆解·窄相位 `points_to_barycentric` | ~30 ms | ~150 ms | bench5 |
| **自实现 Möller–Trumbore 窄相位（numpy，全候选）** | 141.6 ms | 633.6 ms | bench6 |
| 首命中 argmin（`lexsort`） | 10.8 ms | 2.0 ms | bench6 |
| 命中统计（tol 1e-12/1e-9/1e-6 完全一致） | 10,966 hits / 4850 ray | 10,992 hits / 4850 ray | bench8 |
| 锥台合并查询 `tree.intersection(frustum_aabb)` | 3.1 ms（选中全部 9,976 面） | 17.2 ms（全部 39,904 面） | bench11 |
| 每射线 AABB 与锥台内三角形做**成对**测试 | 1,915 ms | 7,959 ms | bench11（**不可用**） |
| rtree 包住整束射线的 AABB → 候选 1330.9/ray | 2,801 ms | 9,081 ms | bench15（**不可用**） |
| 图像尺寸 | 720 × 1280 | | `prepared/closed_front_0.npz` |

**读法**：现网 2927 ms（coarse 全量）里，宽相位 ≈ 510 ms（17%），窄相位 ≈ 100 ms（3%），**其余 ≈ 2300 ms 是 trimesh 自身的 Python 层开销**（`intersects_location` → `intersects_id` → `ray_triangle_id` → `grouping.unique_rows` 等）。所以**只调 chunk 大小几乎没用（实测 32→512 只省 8%）**，必须换掉"每条射线一次 rtree 查询"这个结构。

OSQP 复现实测（同尺寸合成系统：20018 未知量、2881 自由、13k 观测行、22,828 有效约束行）：

| 操作 | 耗时 |
|---|---:|
| 构造 system / H=systemᵀsystem | 0.006 s |
| 约束矩阵装配 + `eliminate_zeros` + 行裁剪 | 0.008 s |
| **`OSQP.setup()`（含 KKT 因式分解）** | **0.950 s** |
| **首次 `solve()`** | **1.144 s / 75 iter** |
| `warm_start(x)+solve()`（P、q 不变） | **0.086 s / 25 iter** |
| `update(q)` | 0.00004 s |
| `update(q)+solve()` | 0.086 s / 25 iter |
| `update(P)` | 0.0022 s |
| `update(P)+solve()` | 0.085 s / 25 iter |

**读法**：setup 的 0.95 s **不可省**（P 与约束集变了就必须重新分解），但**同一 outer 内若只有 q/P 变化、约束集与自由集不变，可省掉 setup 与首次冷启动**。基线 dense 阶段只做了 6 次 setup（2+1+3），setup 总量 ≈ 5.7 s，**不是主要热点**；真正的浪费是"零自由度也照做一次 setup"和"每 outer 冷启动 1100/950 次迭代"。

---

## 2. 热点排序（按可回收的绝对墙钟时间）

| # | 热点 | 估计耗时 | 占比 | 证据来源 | 性质 |
|---|---|---:|---:|---|---|
| **H1** | `fit_d` 前段：`front_anchor_low_frequency` + `mid_frequency` 的观测采样、本地平面、诊断射线、5 个 outer 的 QP、4 次试步——**结果只进一个交付链路无消费者的 OBJ** | **≈ 180–260 s** | 9–13% | `fit_d.py:81-89`；`dense_fit_base = interpolation_vertices`（`:113`）只依赖 `regularized`；`:187` 的 `stages['low_frequency_anchor']` 无导出消费者（`defaults.json: export.mesh_stage="dense"`）；两个上游 `parameters.npz` 的 do-nothing 断言已在本机通过 | **死计算** |
| **H2** | 射线求交"每条射线一次 3D rtree 查询" + trimesh Python 层，在 dense 阶段被调用 ~127 次粗模 + ~20 次粗模几何分析 + 11 次 dense | **≈ 180–260 s 可回收** | 9–13% | bench5/bench8（宽相位 162.6/560 候选每射线，rtree 查询 518/1430 ms，窄相位仅 100/490 ms）；`trimesh/ray/ray_triangle.py:333` 的 `for i, bounds in enumerate(bounding)` 逐射线循环；`field_logs` 的 `sampling`/`accepted` 帧数与射线数 | **算法结构** |
| **H3** | `front_anchor_dense` 的 40 次试步 × (3 帧 5000 射线主查询 + 13045 固定射线) × 建网格 + rtree + `gc.collect()` | **≈ 60–100 s**（其中 40–70 s 由提前停止直接省掉，其余靠判据） | 3–5% | `field_logs.front_anchor_dense`：40 trials / 37 拒；`geometry_d.py:238` 与 `:264` 各一次 `ray_surface`；`geometry_c.py:72` 的 `gc.collect()` | 算法结构 + 判据 |
| **H4** | `motion_v2` 跑两遍（各自 80 步 × 10 次 `%10` 刷新 × 20 帧 × 1800 射线）+ `identity_v2` 用同参数重算一遍采样/平面 | **≈ 250–300 s** 中约 **100–140 s 可回收**（需 T01 测点确认） | 5–7% | `motion_v2.py:103`（`step%10`）、`:91-95`（每帧两次 `fixed_ray_score`，第二次多余）、`:156-160`；`identity_v2.py:22` 新建 `Settings` 触发全新 `observations()` | 重复计算 |
| **H5** | `fit_c` 每 10 步全量重建对应（两阶段各 116/152 次 `ray_surface`，每次 1200 射线）+ `refine_shared` 9 帧 × 9307 射线 | **≈ 60–90 s** 中约 **30–50 s 可回收** | 2–3% | `fit_c.py:152`（`(step-cuts[stage])%10==0`）；`correspondence_log` 152 条/阶段，`refinement_log` 9 帧 | 重复计算 |
| H6 | `guarded_*` 的零自由度语义（QP 与线搜索**本来就没执行**，缺的只是 report 可读性） | ≈ 0–25 s | ≤1% | §1.3；`field_logs.regularized.history[0]` 无 `solver` 字段 | **correctness/可读性** |
| H7 | OSQP 冷启动（6 次 setup × 0.95 s）+ 每 outer 冷启动 950–1100 iter | ≈ 6–15 s | <1% | §1.5 OSQP 表 | 次要 |

> 排序说明：H1 最大且**完全是死代码**，改动最安全；H2 收益同级且被 4 个阶段共享；H3 依赖判据诊断，先做诊断；H4/H5 依赖 T01 的测点数据才能精确切分；H6/H7 是语义/整洁性问题，顺手做。
> **H1 与 H6 的区别很关键**：H1 是"算了但没人要"（可以删）；H6 是"没算但 report 没说清"（只能补字段，删不掉多少时间）。

---

## 3. A. 计时插桩设计（最小必要）

**原则**：不建通用监控框架。只在 4 个文件里加局部 `perf_counter()` 前后对，字段直接进现有 `history` / `report.json` 结构。所有新增字段一律**追加**，不改既有键名与语义，旧 report 仍可读。

### 3.1 `head_rgbd/geometry_c.py`

**（1）`ray_surface` 返回 `stats['timing']`**（替代现在只返回计数的做法）

```python
def ray_surface(v, f, points, max_depth=.025, max_plane=None):
    ...
    t0 = time.perf_counter()
    mesh = trimesh.Trimesh(v, f, process=False)          # 原 :67
    t1 = time.perf_counter()
    hit, ray, face = first_hits(mesh, points)            # 原 :68
    t2 = time.perf_counter()
    del mesh
    t3 = time.perf_counter()
    gc.collect()                                          # 原 :72
    t4 = time.perf_counter()
    # ... 原 :73-92 的 bary/normal/incidence/plane/keep 计算
    t5 = time.perf_counter()
    stats = dict(..., timing=dict(
        mesh_build_ms=(t1-t0)*1e3, intersect_ms=(t2-t1)*1e3,
        del_ms=(t3-t2)*1e3, gc_ms=(t4-t3)*1e3, post_ms=(t5-t4)*1e3,
        n_rays=len(points)))
```

记录字段（每次调用一条）：`mesh_build_ms`、`intersect_ms`（内部再按 `first_hits` 的 chunk 聚合）、`gc_ms`、`post_ms`、`n_rays`。

**（2）`first_hits` 增可选 `stats` 出口**，累计 `broad_ms`（候选搜索）与 `narrow_ms`（平面+重心）——**只有在换实现后才需要细分**；改造前保持现状即可。

**（3）`front_protected_vertices`**：按帧记录 `rays`、`hits`、`mesh_build_ms`、`intersect_ms`、`gc_ms`，返回值的第 4 元（新增，默认 `None`）写入 `dense_front_protected_log`。

### 3.2 `head_rgbd/geometry_d.py` → `solve_field`

在 `history` 的每条 record 上**追加**下列键（不改现有键）：

```python
record = dict(existing...,  # 保持不变
    timing=dict(
        setup_ms=...,            # physical_regularizers / total_regularizer
        observations_ms=...,     # 采样 + local_depth_planes
        diagnostic_init_ms=...,  # 建立 diagnostic_rays 的那一次粗查
        outer_total_ms=...,
        correspondence_ms=...,   # 逐帧 ray_surface（含其 timing 的汇总）
        system_build_ms=...,     # 逐帧 A/b/w 组装 + noise 统计 + vstack
        qp_build_ms=...,         # constraints(...) 矩阵装配（不含 setup 分解）
        qp_setup_ms=...,         # OSQP.setup 的因式分解
        qp_solve_ms=...,         # OSQP.solve 的迭代
        line_search_ms=...,      # 18 次折半循环总计
        geometry_check_ms=...,   # 其中 check_field
        fixed_ray_ms=...,        # 其中 fixed_diagnostic
        fixed_ray_calls=...,     # 本 outer 内 fixed_diagnostic 的调用次数
    ),
    # 物理量（D 段要求）
    update_max_mm=float(np.abs(field - field_before).max(initial=0)),  # 见下，field 已是本 outer 的终值
    backtrack_count=int(len(trials)),          # = 实际试步次数
    free_vertices=int(len(free)),
    constraint_rows=int(qp_constraints[0].shape[0]) if settings.solver=='qp' else 0,
    qp_iterations=int(solver_info.get('iterations', -1)),
    fixed_ray_real_energy_before=float(diagnostic_before),   # 已存在
    fixed_ray_real_energy_after=float(diagnostic_after),     # 已存在
    field_changed=bool(np.abs(proposed - field_before).max(initial=0) > 0),
)
```

**`update_max_mm` 的定义（必须写清，避免歧义）**：
```python
field_before = field.copy()          # 在 outer 循环开始、任何求解之前抓一次
# ... 本 outer 全部逻辑 ...
update_max_mm = float(np.abs(field - field_before).max(initial=0))
```
`field` 是「每顶点沿 `directions`（单位法向）的 mm 标量」，outer 循环内被整体替换（`field = candidate`，`geometry_d.py:381`），所以上面对 `field` 与快照做差即可。对 coarse 与 dense 都成立（`surface_state` 用 `base + directions * field_mm * .001`）。
**不要**用 `||dense_vertices - dense_base||`——那是含 Loop 基面偏移的总位移（2.2598 mm），与本 outer 的实际更新（1e-5 mm 量级）差 5 个数量级，见 §1.4。

**`skipped` 语义（D 段硬性要求）**：在 `if not len(free):` 分支（现 `:324-326`）写全：
```python
if not len(free):
    history.append(dict(
        stage=stage, outer=outer, free_vertices=0,
        status='skipped', reason='no_free_vertices',
        free_rule=dict(support_ge_2=int((support.sum(0) >= 2).sum()),
                       excluded=int(excluded.sum()), locked=int(locked.sum())),
        update_max_mm=0.0, backtrack_count=0, fixed_ray_calls=0,
        skipped_savings_ms=None,   # 由调用方填
        accepted=accepted))
    break
```

**重要**：这个 `break` 是**逐 outer 的 break，不是整阶段 break**。`front_anchor_dense` 三个 outer 的 `free_vertices` 都是 2881，所以它不会触发；但 `guarded_*` 会在 outer 0 就退出——与现在的行为一致，只是多了可读原因。**不要**改成"跳过整个阶段"（会改变 `support_union` 语义）。

### 3.3 `head_rgbd/qp_d.py` → `solve`

返回值 `solver_info` **追加** `timing=dict(matrix_build_ms, eliminate_ms, qp_build_ms, setup_ms, warm_start_ms, solve_ms)`，并区分 `setup_ms`（`OSQP.setup`）与 `solve_ms`（`OSQP.solve`）。**禁止**在此处放宽 `eps_abs/eps_rel` 或删约束行。

### 3.4 `head_rgbd/fit_d.py` → 阶段级汇总

`report.json` 顶层追加：

```python
dense_timing = dict(
    stage_total_ms=...,
    load_ms=...,               # 读 prepared/npz
    dense_topology_ms=...,     # subdivide + mapping + anatomical_exclusion(dense)
    loop_base_ms=...,          # loop_mapping + 20 次试步
    front_anchor_dense_ms=...,
    guarded_dense_ms=...,
    evidence_regular_ms=...,   # reliable_evidence(regular_vertices)
    evidence_dense_ms=...,
    export_ms=...,             # 每个 state × 每个 stage 的 OBJ 写盘 + 读回 + assert
    neck_ms=...,
    unused_front_section_ms=0.0,   # 实现 §4.2 后置 0
)
```

同时在 `logs/07_*.log` 打一行 JSON，便于 `run_report.json` 之外做 grep：
```python
print(json.dumps(dict(stage='dense', **{k: round(v/1000, 2) for k, v in dense_timing.items()})), flush=True)
```

### 3.5 `head_rgbd/fit_c.py` / `motion_v2.py`（前段，只加最少几个点）

按 §3.6 的"先测再改"清单，只加 3 个计数器，不做完整剖析：
- `fit_c`：`ray_surface_total_ms`、`ray_surface_calls`、`refine_shared_ms`、`evidence_ms`，写进 `report['phase_timing']`。
- `motion_v2`：`corr_refresh_calls`、`corr_rays_total`、`fixed_ray_calls`、`forward_calls`（每条都廉价），写进 `report['motion_correction']['timing']`。

### 3.6 插桩验收标准（必须先做完 T01 并保存为**新**运行目录）

1. 新旧运行（同输入）**几何输出逐位一致**：`dense_vertices`、`dense_deviation_m`、`front_anchor_deviation_m` 用 `np.testing.assert_array_equal`。
2. 新 report 的 `dense_timing.stage_total_ms` 与 `run_report.json` 的 `elapsed_s` 相对误差 < 3%。
3. 各子项之和 ≤ `stage_total_ms`，缺口 ≤ 5%（缺口超过 5% 说明还有未覆盖热点，写进报告再决定下一步）。
4. 插桩本身的开销 < 0.5 s（不可能超过：全部是 `perf_counter` 与 dict 构造）。

---

## 4. 优化方案（逐条：改哪里 / 改什么 / 收益 / 风险 / 验证）

### 4.1 【P0】零自由度短路（H6）

**改哪个文件**：`head_rgbd/geometry_d.py`、`head_rgbd/fit_d.py`
**改哪几行**：`geometry_d.py:324-326`；`fit_d.py:91-101`、`:152-156`

**改成什么**

`geometry_d.py`：把 `:324-326` 的 `history.append(...); break` 扩展成 §3.2 的 `status='skipped'` 记录（含 `reason`、`free_rule`、`update_max_mm=0.0`、`backtrack_count=0`）。

`fit_d.py`：在计算完 `free_mask` 的**同一规则**下提前判定（规则必须与 `solve_field` 内 `:322` 完全一致，否则会改变行为）：
```python
# 与 solve_field 内 free_mask = (support.sum(0) >= 2) & ~excluded & ~locked 完全一致
def free_count(support, excluded, locked):
    return int(((support.sum(0) >= 2) & ~excluded & ~locked).sum())
```
- 对 `guarded_regularized` / `guarded_dense`：**不要在 fit_d 里提前跳过** —— `solve_field` 必须至少跑一次 outer 才能拿到 `support_union`（下游 `loop_mapping` 的 `support_regular.sum(0)<2` 需要它，`fit_d.py:123`）。正确做法是**保留 outer 0 的采样与对应构建**（这是 `support` 的唯一来源），只**跳过** outer 1..N-1 的重复工作——而现状本来 `break` 了，所以**这一步本身不需要改代码，只需要修 `report.json` 的可读性**（§3.2）。
- **真正要改的是**：`guarded_*` 阶段既然 free=0，`check_field`/`fixed_diagnostic`/QP 全都没有意义。当前 `break` 已经在 `:324` 提前退出，**QP 与线搜索确实没有执行**（`report.json` 无 `solver` 字段即证据）。所以 H6 的"60–90 s"里，**大部分是 outer 0 的采样 + 9 帧×5000 射线粗查 + 13045 条诊断射线**，这部分**是必须保留的**（它产出 `support_regular`）。

> **因此 4.1 的真实内容收缩为两条**：
> - **(a)** `guarded_dense` 的 `support` 对下游**无人使用**（`fit_d.py` 只在 `support_regular` 上做 `loop_mapping`），但 `report.json` 仍写 `support_dense.npz`。**保留计算与写盘**（交付物的一部分），只在耗时里标清。
> - **(b)** `front_protected_vertices` 在 `guarded_dense` 里被调用一次（`fit_d.py:153`，dense 拓扑 × 3 帧正面），它唯一的消费者是 `locked=`，而 free 恒为 0 → 结果无影响。**这一处可以安全跳过**，条件严格为：`(support_regular.sum(0)>=2) & ~dense_excluded` 在 dense 拓扑上的支持集已被 `support_regular` 的 prolongation 完全覆盖。**保守做法**：不跳过，只在日志里记录 `dense_protected` 与 free 的关系，等 T01 数据再说。

**收益**：0–25 s（诚实估计；原估 60–90 s 偏高，因为 QP 与线搜索本来就没执行）。
**风险**：低（几乎不删计算，主要是修 report 语义）。
**验证**：新 report 里 `guarded_regularized.history[0].status == 'skipped'`、`reason == 'no_free_vertices'`、`update_max_mm == 0.0`；`free_rule` 三项分解与 `parameters.npz` 的 `front_protected_vertices` / `anatomical_excluded` 对得上；几何逐位一致。

---

### 4.2 【P0】删除 `fit_d` 前段死计算（H1）—— **本次最大、最安全的一笔**

**改哪个文件**：`head_rgbd/fit_d.py`
**改哪几行**：**只删 `:81-89`**（`low_settings` / `anchor` / `low_anchor` / `mid_log` 的 `solve_field` 调用），`:80` 的注释同步；**保留 `:91-101` 的 `guarded_regularized`**；`:187` 与 `:230` 的 `low_anchor` / `low_frequency_anchor_vertices` 引用按下文处理。

**证据链（三条独立证据，缺一不可）**

1. **数据流**：`dense_fit_base = interpolation_vertices`（`:113`），而 `interpolation_vertices = [mapping @ v for v in regular_vertices]`（`:111`），`regular_vertices = surface_state(base, direction, regularized)`（`:101`）。**`anchor` 与 `low_anchor` 不参与任何一条下游路径。** `dense_anchor` 的初值来自 `initial=dense_anchor`，但那是 `front_anchor_dense` 自己的输出（`:148-150`）。
2. **唯一残留引用**：`:187` 的 `stages['low_frequency_anchor']`，只用于**导出** `closed_low_frequency_anchor_m.obj` / `open_low_frequency_anchor_m.obj`。而 `config/defaults.json → export.mesh_stage = "dense"`，`pipeline.export_delivery` 只读 `stages['dense']`。**该 OBJ 在交付链路上无消费者。**
3. **逐字节同源（本机已跑通）**：`regularized_vertices`（`fit_d` 内部）与 `alternated_motion/parameters.npz['posed_vertices']` 相同；`front_anchor_vertices` 与 `front_anchor/parameters.npz['posed_vertices']` 相同（do-nothing 测试）。因此 `front_anchor_low_frequency` / `mid_frequency` 重算的几何**在上游已经存在**。

**改成什么**

```python
# 删除 :81-89 整段（low_settings / anchor / low_anchor / mid_log 的 solve_field 调用）
# 保留 anchor 名字以维持下游引用，直接取上游已有值：
prev = load_npz(initial_fit / 'parameters.npz')       # initial_fit == alternated_motion
anchor = prev['deviation_m'] * 1000.0                  # 与 solve_field 的 mm 标量同单位
anchor_log = dict(source='inherited_from_initial_fit',
                  note='fit_d 前段 anchor 不参与下游；几何取自上游 parameters.npz，逐位一致',
                  inherited_from=str(initial_fit / 'parameters.npz'))
low_anchor = prev['deviation_m'] * 1000.0              # 保持 :187 的低频锚点 OBJ 与旧运行一致
```

**必须在实现时加的守卫**（不是 defensive fallback，而是**契约断言**，与现有 `fit_d.py:32-38` 的风格一致）：
```python
np.testing.assert_array_equal(prev['posed_vertices'], base)
np.testing.assert_array_equal(prev['faces'], faces)
np.testing.assert_array_equal(prev['identity_selected'], identity)
# 单位契约：parameters.npz 里 deviation_m 是米
assert prev['deviation_m'].max() < 1e-2
```
若上游不是 `alternated_motion`（`mode='joint-research'` 等），这条断言会失败 → **必须显式报错**，不能 fallback。即：
```python
if mode != 'guarded':
    raise ValueError('fit_d 前段死计算消除仅对 guarded 链验证过；其它 mode 保留原路径')
```
> 这**不是**防御性编程，是**缩小改动面**：只在被验证过的 guarded 链上启用，其它 mode 原样保留。

**收益**（按 amdahl，用 dense 内部 1126 s 做分母）：
- 去掉 `front_anchor_low_frequency`（2 outer：free=949/945）与 `front_anchor_mid_frequency`（3 outer：free=936/934/935）两次 `solve_field` 的全部开销：3 帧观测采样 + 3 帧 `local_depth_planes` + 一次 `diagnostic_rays` 建立（12,984 条粗查）+ 5 个 outer 的 QP（setup 5×0.95 s + solve 100 + 275 + mid 三段）+ 4 次试步。
- 按 §1.2 的 outer 结构估计 **180–260 s**。
- 外加 `:230` 的 `low_frequency_anchor_vertices` 写盘（2.3 MB）与 `:187` 的两次 OBJ 导出。

**风险**：中低。
- 风险 1：`parameters.npz` 里 `deviation_m` 单位是**米**（`fit_c.py:286` 写 `deviation_m=deviation`，`deviation` 是米；`fit_d.py:232` 写 `deviation_m=regularized*.001`），必须 `*1000` 转 mm。已在上面显式写出，实现时**必须断言 max < 1e-2**。
- 风险 2：`alternated_motion` 的 `deviation_m` 是 `motion_v2.py:175` 的 `field.cpu().numpy()`，来自 `identity_field.npz['deviation_m']`（`identity_v2.py:27` 写 `field*.001`）→ 单位一路是米。**已核对。**
- 风险 3：`support_regular`。`fit_d` 用 `support_regular = solve_field(...)['support_union']`（`:93`），下游 `loop_mapping` 用它冻结控制点（`:123`）。**若把 `guarded_regularized` 一起删，`support_regular` 就没有来源**：上游 `alternated_motion/support.npz` 的 `refinement_frame_mask` 是 `fit_c.refine_shared` 的 support（16000 点/帧、需同时满足边界框与 `>0.05` 重心），与 `fit_d` 的（5000 点/帧、`support>=2`、`~excluded`、`~locked`）**不是同一个集合**，不能替换。
  - **结论**：**`guarded_regularized`（`fit_d.py:91-101`）必须保留**，只删 `:81-89`。它产出 `support_regular`，且 free=0 时 outer 0 已经完成一次全量对应构建，这部分**不是浪费**。
  - 因此收益定格为 **≈ 180–260 s**（仍是最大单笔）。

**验证**
1. `np.testing.assert_array_equal(new['dense_vertices'], old['dense_vertices'])`、`dense_deviation_m`、`dense_front_anchor_deviation_m` 全部逐位相等。
2. `support_regularized.npz['refinement_frame_mask']` 逐位相等。
3. `report.json` 里 `field_logs` 的键应从 `{front_anchor_regularized, regularized, front_anchor_dense, dense}` 变为 `{regularized, front_anchor_dense, dense}`，且新增 `front_section_removed` 说明字段。**不要保留空壳键。**
4. 时间：`dense_timing.stage_total_ms` 下降 ≥ 180 s。
5. **导出物核对**：`closed_low_frequency_anchor_m.obj` / `open_low_frequency_anchor_m.obj` 若保留导出，需与旧文件**逐字节一致**（可用 `sha256` 比对）；若决定不再导出，必须同时从 `stages` 里去掉并在 `report.json` 的 `limitations` 里注明"低频谱锚点 OBJ 不再导出（交付链路无消费者）"。

---

### 4.3 【P0】射线求交：换掉"每条射线一次 3D rtree 查询"（H2）

**改哪个文件**：`head_rgbd/geometry_c.py`（`first_hits`），其余调用点**零改动**（保持 `ray_surface` 签名与返回语义）

#### 4.3.1 批量大小的实测依据（现在 32）

- `chunk=32`：1088.9 ms（coarse）/ 2590.1 ms（dense）
- `chunk=16`：1099.0 / 2824.4
- `chunk=64`：1025.5 / 2518.0
- `chunk=128`：907.9 / 2497.0
- `chunk=256`：860.7 / 2591.1
- `chunk=512`：856.1 / 2533.8
- `chunk=1024`：891.8 / 2459.2
- 全量 4850：906.7 / 2613.7

**结论**：chunk 从 32 调到 128–512 只省 **~15%**（coarse）/ **~4%**（dense），且噪声 ±10%。**没有值得单独落地的批量调参**。`chunk` 这个参数现在唯一的真实作用是"限制瞬态 rtree 交点数组的内存峰值"（注释 `geometry_c.py:49` 说的就是这个）。→ **保留 `chunk` 作为内存上界参数，但把它从 32 提到 512**（省 ~130–230 ms 粗模、~60 ms dense，零风险，零语义变化），**不作为主要方案**。

#### 4.3.2 正确生命周期：哪些调用可以共用 mesh/rtree

`trimesh.Trimesh(v,f,process=False)` 的构建只要 **0.07–0.10 ms**（实测），rtree 构建 **39 ms（coarse）/ 160 ms（dense）**。所以问题不在"建网格贵"，而在"**每条射线查一次**"。

| 场景 | 几何是否变化 | 可否共用 mesh + rtree |
|---|---|---|
| `solve_field` 单个 outer 内的**逐帧** `ray_surface(posed[i], ...)` | `posed` 每帧不同，但**同一帧在本 outer 内只查一次** | 无需共用（浪费为 0） |
| `fixed_diagnostic(candidate)` 内**逐帧** `ray_surface(posed[i][required], ...)` | 每帧变一次 | **不可跨帧共用** |
| `check_field` → `hierarchy_d.safety` / `geometry_d.safety` | 纯 numpy，不建网格 | — |
| `evidence()` / `reliable_evidence()` 逐帧 | 每帧一次 | 无需共用 |
| `front_protected_vertices` 逐帧 | 每帧一次 | 无需共用 |
| **`fit_c.refine_shared` 内 `ray_surface`** | 每帧一次 | 无需共用 |

**真正可共用、且现在被浪费的**：`fixed_diagnostic(candidate)` 与**同一 candidate 上的 `check_field`**——但 `check_field` 不建网格。**所以"共用 mesh"这条路的收益是 0。** 必须换实现。

#### 4.3.3 具体方案：单次栅格（raster）候选 + 自实现 Möller–Trumbore + `lexsort` 首命中

**核心想法**：所有射线**共享同一个相机原点**（实测 `origins = np.zeros_like(points)`，`geometry_c.py:53` 就是这么写的）。因此"这束射线的锥体"投影到图像平面上就是一张 (H=720, W=1280) 的图。把每个三角形**一次性**光栅化到这张图上（每条射线只标记自己那一个像素，`r=2` 邻域），得到"三角形 → 像素"的**静态**关联表；之后每来一条射线，只需查表拿到候选三角形，无需任何空间查询。

**改动（只动 `geometry_c.py`）**

```python
def build_ray_raster(vertices, faces, intrinsics, shape, radius=2):
    """一次性把三角形光栅化到相机像素栅格；返回每个像素的候选面索引。

    *均质相机原点*：只支持 origin=(0,0,0) 的透视相机（与 ray_surface 的调用方一致）。
    结果只依赖 (vertices, faces, intrinsics, shape)，几何变化后必须重建。
    """
    # 1) 三角形三个顶点投影到像素
    # 2) 对每个三角形算像素 bbox，按 bbox 切片累加到 pixels -> face 的扁平列表
    # 3) 用 np.repeat/np.searchsorted 建 CSR：pixel_ptr (H*W+1), pixel_faces
    ...

def first_hits_raster(raster, points, intrinsics, max_iters_per_ray=64):
    """返回 (hits, ray_indices, face_indices)，语义与 first_hits 逐项一致：
    首命中（同一射线多命中取 t 最小）、保留原始射线编号、缺失射线不出现。
    """
    # 1) 像素索引 = 反投影 points 的 uv
    # 2) 候选面 = raster[pixel]
    # 3) Möller–Trumbore（向量化，见下）
    # 4) lexsort((t, ray)) 取每组首行
```

Möller–Trumbore（已在 bench6/bench8 验证，命中数与 trimesh 完全一致 10,966 / 10,992，tol 1e-12/1e-9/1e-6 结果不变）：
```python
v0 = T[:, 0]; v1 = T[:, 1]; v2 = T[:, 2]
e1 = v1 - v0; e2 = v2 - v0
pv = np.cross(D, e2); det = (e1 * pv).sum(1)
ok = np.abs(det) > 1e-14
inv = np.zeros_like(det); inv[ok] = 1.0 / det[ok]
tv = O - v0
u = (tv * pv).sum(1) * inv
qv = np.cross(tv, e1); vv = (D * qv).sum(1) * inv
t = (e2 * qv).sum(1) * inv
inside = ok & (u >= -1e-12) & (u <= 1+1e-12) & (vv >= -1e-12) & (u+vv <= 1+1e-12) & (t > 0)
```

**必须保持的五项语义（逐条对照现有实现）**

| 语义 | 现有实现 | 新实现如何保证 |
|---|---|---|
| **首命中** | `intersects_location(multiple_hits=False)` 内部按 `location·direction` 排序取首个 | `lexsort((t, ray_index))` 后每组首行；`t>0` 保证在射线正方向 |
| **射线编号** | `first_hits` 里 `r + start` 还原全局编号 | 直接使用输入 `points` 的行号，天然一致；缺失射线**不出现**在返回里（调用方 `ray_surface` 用 `points[ray]` 回取，`ray_surface:82,96`） |
| **双面查询** | rtree 的 AABB 与三角形朝向无关，`intersects_location` 对两面都返回 | Möller–Trumbore 的 `det` 取绝对值 → 双面 |
| **backface 过滤** | **不在求交层**，由 `ray_surface:84` 的 `incidence > .15` 完成 | 完全不动 `ray_surface:77-92` 的前/背判定 |
| **缺失语义** | 无交点 → 该 ray 不出现在 `hit/ray/face` 中，`stats['no_surface_hit'] = len(points) - len(hit)` | 同上；`stats` 的计算代码**一行不改** |

**唯一新增的语义差异（必须显式声明并测试）**：`max_iters_per_ray`。栅格候选理论上包含所有可能相交的三角形；为防病态（一条射线穿过极薄的边界），设上限 64 并**在 `stats['raster_truncated']` 计数**。基线实测 max 命中/射线 = 8（bench8），64 有 8 倍余量。**若 `raster_truncated > 0`，必须让这次查询直接 `raise RuntimeError`**（不是 fallback），否则首命中可能错。

**收益**
- 宽相位：`per-ray rtree` 518 / 1430 ms → **栅格建表 ~40–50 ms（一次性，按拓扑分摊）+ 查表 ~5–15 ms**。
- 窄相位：`planes_lines` + `barycentric`（102 / 492 ms）→ **141 / 634 ms**。**注意自实现更慢**（float64 三重叉积），但**结构上可省掉求交后的 `points_to_barycentric` 重算**吗？不能——`ray_surface:76` 需要**交点处的重心坐标**做后续 `value = bc*(...)`。Möller–Trumbore 已经顺手算出 `u, v`，`bc = (1-u-v, u, v)`，可以**直接复用**。→ 把 `ray_surface:76` 的 `points_to_barycentric` 换成直接用 `(1-u-v, u, v)`，**省掉 30 / 150 ms**，且更精确（同一套算术）。
- **合计（单次调用，4850 射线）**：coarse 938 → **约 190 ms**（−80%）；dense 2771 → **约 700 ms**（−75%）。
- 按 dense 阶段 107 次粗模 + 20 次粗模几何分析 + 11 次 dense 粗查估算 → **可回收 200–280 s**。
- 若栅格表能按 `(topology, poses)` 缓存（**注意：几何一变就必须重建**，`solve_field` 每 outer 的 `posed` 都变），则只有 `evidence`/`front_protected` 这类"同一几何多次同点查"的场景能复用。**本设计不引入缓存框架**：只在 `fit_d` 里对 `front_protected_vertices` 的两个拓扑各建一次表并传参，其余每次重建（40–50 ms 相对 500+ ms 可忽略）。

**风险**：中。
- 风险 1：栅格化实现错误导致漏候选 → 表现为 `no_surface_hit` 上升。**必须做逐射线比对测试**。
- 风险 2：极薄/极倾斜三角形在像素上面积 < 1 像素，可能被 `r=2` 邻域漏掉 → 用 `r=2` 并测试 `stats['hits']` 与旧实现**逐帧完全相等**。
- 风险 3：`intrinsics` 在 `solve_field` 里是**逐帧**的（`data[i]['intrinsics']`），不同帧可能不同 → 栅格表必须**按帧建**，或断言全序列 `intrinsics` 相同（本采集是同一次会话，应相同；实现时**断言**，不同则退化为"按帧建表"，不是 fallback 而是分支）。

**验证（硬门槛，任一不过就不合并）**
1. 对 `front_anchor_dense` outer0 的 3 帧、`guarded_regularized` outer0 的 9 帧，逐帧比对 `ray_surface` 返回的 `ray`、`face`、`bary`（`atol=1e-9`）、`hit`（`atol=1e-9`）、以及 `stats` 的**全部计数字段**，要求**完全相等**。
2. 全流程几何逐位一致：`dense_vertices`、`regularized_vertices`、`dense_deviation_m` 全部 `assert_array_equal`。
3. `report.json` 里所有 `accepted[*]` 的 `hits`/`no_surface_hit`/`accepted` 与旧运行**完全相同**。
4. `raster_truncated == 0` 全程。
5. `dense_timing.correspondence_ms` 下降 ≥ 60%。

**分两步落地**：**T02a** 只把 `chunk 32→512` 与 `gc` 降频（零风险，~0.3 s/阶段）；**T02b** 才换栅格实现（需过上面 5 条门槛）。

#### 4.3.4 `gc.collect()` 的取舍

**现有生命周期分析**

- `ray_surface:67` 建 `Trimesh`，`:68` 用它的 `mesh.ray`（内部持 `rtree.Index`），`:71` `del mesh`，`:72` `gc.collect()`。
- 注释（`:69-70`）的假设是"trimesh ray/cache 对象有引用环，离线 16GB 机器上跑几百次 ICP 会积累 rtree"。**该假设在 trimesh 4.4.9 上不成立**：`RayMeshIntersector.__init__` 只存 `self.mesh` 与 `self._cache = caching.Cache(mesh.__hash__)`（`ray_triangle.py:26-28`）；`rtree.Index` 只被 `mesh.triangles_tree`（`caching.Cache` 装饰的 property）持有。`del mesh` 之后引用计数即归零，**无需循环 GC**。
- **实测**：`gc.collect()` 每次 **75.1 / 76.8 ms**。
- **峰值内存风险判断**：
  - 单次 `ray_surface` 的瞬态峰值 = `Trimesh`（v,f 视图，UBOUND ~0）+ rtree（~39 MB/万面 × 4 ≈ 160 MB 上界，实测进程 RSS 增量 < 40 MB）+ `first_hits` 的交点数组（chunk×命中，chunk=512 时 < 1 MB）。**远低于 7 GB 可用内存。**
  - 与现有 `dense` 阶段同时存活的大对象：`dense_fit_base` 9.2 MB、`dense_vertices` 9.2 MB、`constraint_matrix`（1,213,933 × 20018 的 vstack，`qp_d.py:26`）— **这一项才是内存大头**，但它在 `qp_constraints` 里跨 outer 复用（`geometry_d.py:340`），与 `gc.collect()` 无关。
- **结论**：`gc.collect()` **可以删**。但为最小风险，**改为按次数降频**：
  ```python
  # geometry_c.py 模块级
  _GC_INTERVAL = 200
  _gc_counter = 0
  # ray_surface 内
  del mesh
  _gc_counter += 1
  if _gc_counter % _GC_INTERVAL == 0:
      gc.collect()
  ```
  **不要**用"全局开关/策略"包装。`_GC_INTERVAL=200` 的理由：粗模 rtree ~40 MB 峰值，200 次最坏 8 GB 的不释放上界——但引用计数已归零，实际 RSS 增长为 0（实测 500 次 `ray_surface` 后 RSS 增长 < 20 MB，可在 T02a 里复核）。
  **若 T02a 实测 RSS 增长 < 100 MB**，则直接删掉 `gc.collect()`（更简单、更快）。
- **收益**：粗模每次 75 ms × 107+ 次 ≈ **8 s**（加 dense 11 次 ≈ 0.9 s）。**不大，但零风险，顺手做。**

---

### 4.4 【P1】`front_anchor_dense` 的覆盖率冲突（H3 / E 段）

#### 4.4.1 先做诊断：那 4 条/2 条/1 条到底在哪？

**这是本轮唯一必须"先测后改"的判断。** 加一个只读诊断，在 `fixed_diagnostic` 里对每个被拒 trial 记录**新增缺失射线的身份**：

```python
# geometry_d.py fixed_diagnostic 内
before_covered = ...            # 上一次 accepted 状态的 covered 布尔
after_covered  = covered        # 当前 candidate 的 covered
newly_missing  = np.flatnonzero(before_covered & ~after_covered)
return energy, missing, dict(newly_missing_global_index=newly_missing.tolist()[:16],
                             newly_missing_ray_uv=... ,      # 由 points 反投影得到
                             newly_missing_t_min=...,        # 该射线的新 t 与旧 t
                             newly_missing_face=...,         # 命中的面是否在 silhouette 邻域
                             newly_missing_at_silhouette=... # 该面法向与视线夹角是否 > 某阈值
                             )
```

然后**只跑 `front_anchor_dense` 一个阶段**（3 个 outer，约 5 min），检查新增的 1–4 条缺失是否满足：
- **(a) 都在轮廓/遮挡边界**（面的两跳邻域内含 silhouette 边，或该像素的 depth 邻域方差 > 10 mm）→ 属于**几何上合理的遮挡改变**，应允许；
- **(b) 都在数值不稳区**（射线与三角形近乎共面，`|det| < 1e-9`；或交点距某条边 < 1e-9 m）→ 属于**数值退化**，应允许（并同时修容差）；
- **(c) 真的是"几何变差"**（面正对相机、深度稳定，候选让它穿透/翻面）→ **不许放宽，当前的严格判据是对的**。

**判定口径（可计算、无魔法数字）**：一条缺失射线被判为 (a)/(b)，当且仅当
```
在 points[ray] 的 5×5 深度邻域内：max(z)-min(z) > 2 * noise_floor_mm * DIST_FACTOR
```
或
```
该射线在 candidate 几何上仍存在交点，且 |hit_t_after - hit_t_before| < max_depth/2，但命中面绕射线方向旋转 > 30°
```
其中 `noise_floor_mm = 1/sqrt(12) = 0.28868 mm`（`fit_d.py:293` 的同一常量），`DIST_FACTOR` 由 `local_depth_planes` 的 `rms < 1.5` 门限反推（`surface_v2.py:110`），取 `1.5/0.28868 ≈ 5.2`。**写进代码时直接写 `2 * settings.noise_floor_mm * (1.5 / settings.noise_floor_mm)` 并注释来源，不要把 5.2 当常量。**

#### 4.4.2 按风险从低到高的改法（每条都给了"如何证明不是换指标"）

| 序 | 改法 | 风险 | 如何证明不是靠换指标宣称改善 |
|---|---|---|---|
| **L1** | **不做任何判据改动**，只把 `fixed_diagnostic` 的 `max_depth` 从 `.030` 改成与主查询一致的 `.008`（`geometry_d.py:238`） | **低** | 这是**让判据变严**（更多命中被判为缺失），只可能减少接受步，不可能让某步"看起来变好"。跑完若接受步数下降 → 说明原判据确实在纵容坏几何，记录并接受；若上升 → 说明原来的 `.030` 把"深度错 3 cm 的命中"算成 covered，判据本身失效。**用 `missing_before` 与 `missing_after` 的变化方向自证。** |
| **L2** | 把 `missing_after <= missing_before` 改为**与固定射线数无关的相对判据**：`missing_after <= missing_before + max(0, ceil(REG_COVER_TOL * n_fixed))`，`REG_COVER_TOL = 1/10000`（13045 条 → 允许 2 条） | **中低** | **必须同时**证明三件事：① 被放行的 trial 的 `fixed_ray_mse_mm2` **用原始 13045 条射线、原始 8.0 mm 缺失罚**（`fixed_ray_energy` 的 `penalty=8.`，不是 `.030` 口径）重算后**下降**；② **独立**的 `evidence`/`reliable_evidence`（`surface_v2.py:126`）在同一几何上重查，`region_label==2` 的顶点数**不减少**；③ 最终 `dense_change_from_interpolated_regularized_mm` 与 `maximum_dense_total_displacement_mm` **都不增加**。三者全过才叫改善。 |
| **L3** | 位置敏感判据：**只有被判为 (a)/(b) 的缺失射线才豁免**，其余仍然绝对禁止 | **中** | 豁免规则**完全由 4.4.1 的诊断脚本先量化**（先跑出 37 次被拒 trial 的 37×4 条缺失射线的分类直方图）。若直方图显示 90% 落在 (a)/(b)，L3 有依据；若 < 50%，**放弃 L3**。证明方式同 L2，且额外要求：被豁免射线的分类标签与几何量（`local rms`、`|det|`）**全部写进 report**，可被第三方复核。 |
| **L4** | 把 `front_anchor_dense` 的 `outer_steps` 从 3 降到 1（现状 outer1/2 的相对改善是 1.8e-5 与 8.7e-6 mm，ΔMSE 1.4e-5 / 6.6e-6 mm²） | **低（但需说明）** | 这是**减少步数**，任务书明确说"不要当作主要方案"。但这里的依据是**物理量已被穷尽**：`update_max_mm` 在 outer1 = 0.000017 mm、outer2 = 0.000008 mm，均 < `noise_floor_mm/10000 = 2.9e-8 m = 2.9e-5 mm`… **注意**：`0.000017 mm` 已接近该阈值但仍大于，**因此 L4 必须等 §4.5 的 D 判据落地后自动生效，而不是手工改配置**。→ L4 不单独实施，由 `§4.5` 的提前停止覆盖。 |

**推荐**：**先做 L1（低风险、纯诊断价值）+ 4.4.1 诊断**，用实测直方图决定是否做 L2/L3。**L1 本身就值得做，因为现在的 `.030` 与主查询的 `.008` 不一致，是一个可证实的判据不一致。**

**关键提醒**：三个 outer 的 `missing_before` 始终是 0，说明**当前几何命中了全部 13045 条**。在 13045/13045 全覆盖的情况下，"允许 1–2 条缺失"（L2）在**几何上等价于承认"轮廓处 1 像素的覆盖抖动是允许的"**。这个论点必须用 4.4.1 的直方图数据支撑，不能靠断言。

---

### 4.5 【P1】提前停止判据（D 段）

#### 4.5.1 三个可计算判据

**判据 1：实际最大顶点更新（主判据，单位 mm，直接可比）**
```
update_max_mm = np.abs(field_after - field_before).max()
```
**阈值推导**（不引入新魔法数字）：
- 未知量是"沿单位法向的位移（mm）"。
- 该几何的最低可分辨量由深度噪声决定：`noise_floor_mm = 1/sqrt(12) = 0.28868 mm`（`fit_d.py:293`，建模为 ±0.5 mm 均匀量化）。
- 一个顶点位移 `d` 若要在**跨 3 帧**、**面积求积加权**的正常方程里产生 ≥ 噪声量级的目标变化，需要 `d` ≥ `noise_floor_mm / sqrt(3) ≈ 0.1667 mm`。这是**修改有意义**的下界。
- 反过来，`d = noise_floor_mm / 100 = 0.00289 mm` 时，该位移在 `local_depth_planes` 的局部一致性门限（`rms < 1.5 mm`，`surface_v2.py:110`）下**不可分辨**：0.00289 / 1.5 ≈ 1/519，已低于该门限的两个数量级余量。
- 再小到 `noise_floor_mm / 1000 = 2.9e-4 mm` 时，其对任何下游量（法向、面积比、重心）的影响都低于 float64 在米单位下的有效精度，**数值上无意义**。
- **取两个门限**（都写成 `noise_floor_mm` 的显式分式，不引入独立常量）：
  - `update_meaningful_mm = noise_floor_mm / sqrt(3) ≈ 0.1667` → 低于此值，**这一步的更新在跨 3 帧的噪声尺度下不可观测**；
  - `update_noise_mm = noise_floor_mm / 100 ≈ 0.00289` → 低于此值，**这一步的更新在局部平面 rms 门限下不可分辨**。
- `front_anchor_dense` 实测：outer0 接受的是 alpha=1/256，`fixed_ray_mse` 从 0.747179 到 0.746330（Δ=8.5e-4 mm²）；outer1/2 分别是 1.4e-5 / 6.6e-6 mm²。**`update_max_mm` 必须由 T01 实测**，不得推算。

**判据 2：固定射线真实能量改善（辅判据，必须用原始口径）**
```
rel_improvement = (energy_before - energy_after) / energy_before
energy = aggregate_fixed_ray_energy(..., policy='strict-coverage')
```
- 要求 `rel_improvement >= 1e-3`（0.1%）。**依据**：13,045 条射线的 MSE 的标准误约为 `sqrt(2/n) ≈ 1.24%`（卡方近似），**1e-3 远低于统计分辨率**，所以"改善 < 0.1%"应视为**测不出**。
- 现状三层改善分别是 `0.116% / 0.0018% / 0.0009%` → **只有 outer0 勉强到 0.1%**，outer1/2 明确不显著。

**判据 3：场是否真的改变（结构判据）**
```
field_changed = bool((field_after != field_before).any())
```
- 若 `field_changed == False`，**立即停止该阶段**（并记 `status='converged_exact'`）。
- 若 `field_changed == True` 但两条数值判据都不满足 → 记 `status='converged_insignificant'`，停止。

#### 4.5.2 停止规则（写在 `solve_field` 的 outer 循环尾部）

```python
# 每 outer 结束后
if not found:                                   # 本 outer 未接受任何步
    stop = 'no_accepted_step'
elif not field_changed:
    stop = 'converged_exact'
elif update_max_mm < settings.update_noise_mm and rel_improvement < settings.energy_rel_min:
    stop = 'converged_insignificant'
elif update_max_mm < settings.update_meaningful_mm and outer >= 1:
    stop = 'converged_sub_noise_budget'   # 连续第二轮低于噪声尺度
else:
    stop = None
if stop:
    record['status'] = stop
    record['stop_reason_detail'] = dict(update_max_mm=update_max_mm,
                                        rel_improvement=rel_improvement,
                                        update_noise_mm=settings.update_noise_mm,
                                        noise_floor_mm=settings.noise_floor_mm)
    break
```

**两个新 Settings 字段**（`geometry_d.Settings` 追加，全部由 `noise_floor_mm` 导出，**不暴露 CLI**）：
```python
@property
def update_meaningful_mm(self): return self.noise_floor_mm / np.sqrt(3.)
@property
def update_noise_mm(self):      return self.noise_floor_mm / 100.
```
`energy_rel_min` 不设成 Settings，直接写模块常量 `ENERGY_REL_MIN = 1e-3` 并在注释里写明卡方依据。**不要**做成可配置开关。

> **注意**：`Settings` 是 `@dataclass(frozen=True)` 且被 `asdict(settings)` 写进 report（`geometry_d.py:401`）。加 `@property` 不会进 `asdict`，所以 report 的 `settings` 块不变——**这正是我们要的**（阈值由 `noise_floor_mm` 唯一决定，不需要单独记录）。若希望 report 里能看到实际生效的阈值，把它们放进 `record['stop_reason_detail']`（已在上面写了）。

#### 4.5.3 零自由度阶段的显式语义（D 段硬性要求）

已在 §3.2 写出。补充：`guarded_regularized` 与 `guarded_dense` 的 `status` 应为 `'skipped'`，`reason='no_free_vertices'`，`update_max_mm=0.0`，`backtrack_count=0`，并把 `free_rule` 的**三项分解**（`support_ge_2` / `excluded` / `locked`）写出来，让读者一眼看出是"支持不够"还是"全被锁"。

#### 4.5.4 收益与风险

- `front_anchor_dense`：期望从 3 个 outer 收敛到 **1 个 outer**。依据是 outer1/2 的相对改善为 1.8e-5 / 8.7e-6 mm（`dense_front_anchor_deviation_m` 的差分），远低于 `update_noise_mm = 0.00289 mm`。省 2×（QP 950 iter + 31 次试步 × 11/3 帧全量重投射）≈ **60–100 s**。
- `front_anchor_low_frequency`：现状 2 个 outer（alpha 0.25 → 1.0，`fixed_ray_mse` 2.4466→1.7456→0.7207，改善 28.6% / 58.7%）—— **明显未收敛，不会被提前停止触发**。
- `front_anchor_mid_frequency`：3 个 outer（alpha 0.125 / 0.125 / 0.03125，free 936/934/935）—— **需 T01 实测 `update_max_mm` 才能判定**。**不预设结论。**
- **风险**：中低。改变的是"什么时候停"，不改变"接受什么"。**验证**：期望几何逐位一致（停在当前几何，只是不再多走一步）。若出现逐位不一致，说明被省掉的那个 outer 的接受步本身低于噪声尺度——记录 `update_max_mm` 并**接受该差异**，**不修改阈值去凑逐位一致**。

---

### 4.6 【P1】前段（13.6 分钟）可回收部分（F 段）

#### 4.6.1 `fit_c` 与 `identity_v2` 是否重复求解？

**是，但不是同一件事。**

| 项 | `fit_c --front-anchor`（`shared_identity_guarded`） | `identity_v2`（`low_frequency_shared_identity`） |
|---|---|---|
| 求解对象 | FLAME `shape`（50）冻结，`expr/jaw/rotation/trans` 240 步 Adam | 顶点法向标量场 `deviation`（5023） |
| 对应方式 | PyTorch 可微表面 + 每 10 步刷新 `ray_surface` 的固定对应 | `solve_field` 的 `ray_surface`（每 outer 重算） |
| 采样 | `fit_c.py:128` `sample_depth(d, 1200)` 无 `exclude_mouth` | `identity_v2` → `solve_field` → `observations()` → `sample_depth(d, 5000, exclude_mouth=True, stride=3)` |
| 遮挡/可见性 | `landmark_visibility` | `plane_data` + `excluded` + `support>=2` |

**结论**：**不是重复求解**（不同未知量、不同分辨率），**但有大量重复的准备工作**：
1. `sample_depth` 被调用 4 次/帧（`fit_c:128` 两次 × 2 个阶段；`identity_v2` 经 `solve_field` 1 次；`motion_v2:86` 1 次）。`sample_depth` 内含 `binary_erosion` + `minimum_filter` + `maximum_filter`（`geometry_c.py:32-35`），这是 4 个滤波器 × 20 帧 × 6 次 = **480 次滤波**。
2. `local_depth_planes`（`surface_v2.py:87`）在 `solve_field` 每个**阶段**重算一次（`front_anchor_regularized` / `regularized` / `front_anchor_dense` / `dense` 各 3/9/3/9 帧），**5 帧×5 邻域 = 25 点/射线 × 5000 射线 × 24 帧次**。这是 O(120k × 25) 的 `np.linalg.eigh(3×3)` 批处理，**在 `fit_d` 里被算了 4 遍**（两遍随 §4.2 一起消失）。
3. **`identity_v2` 的 `Settings` 是新建的**（`identity_v2.py:22`），`gradient=8, bending=8, outer=2`；`fit_d` 的 `low_settings`（`fit_d.py:81`）是**完全相同**的参数。**两者算的是同一件事**，只是 `identity_v2` 用上游 `motion_only` 的 `posed_vertices`，`fit_d` 用 `alternated_motion` 的。

**先测再改的具体测点（T01 必须加的 3 个）**
```python
# fit_c
report['phase_timing'] = dict(
    ray_surface_calls=int(...), ray_surface_total_ms=...,
    refine_shared_ms=..., evidence_ms=...,
    sample_depth_ms=...,          # 在 fit_c.py:128 前后加
)
# identity_v2
report['timing'] = dict(sample_depth_ms=..., local_planes_ms=...,
                        solve_field_ms=..., front_protected_ms=...)
# motion_v2
report['motion_correction']['timing'] = dict(
    corr_refresh_calls=..., corr_rays_total=...,
    fixed_ray_calls=..., forward_calls=...,
    accept_loop_ms=...,
)
```

#### 4.6.2 `motion_v2` 跑两遍能否复用状态？

**不能直接复用优化器状态**（第二遍的 `initial-fit` 是 `motion_only` 而非 `guarded`，且带 `--identity-field`），但**有三处明确的重复**：

| 重复点 | 位置 | 证据 | 可回收 |
|---|---|---|---|
| **(a) `%10` 刷新对应** | `motion_v2.py:103` `if step%10==0 or step in (steps//2, 3*steps//4)` | 80 步 → **10 次刷新** × 20 帧 × 1800 射线 = 360k ray casts/阶段，两阶段 720k | 每次刷新 1800×20 帧，耗时可测（T01）；**刷新频率 10 是过密的**：`fit_c` 的 240 步才每 10 步刷新一次（24 次），而 `motion_v2` 只有 80 步却刷 10 次。**但 `motion_v2` 的 lr 更大（`.0006/.00008`），对应过期更快。** → **不预设结论，先用 `corr_refresh_calls` + `accept_loop` 数据判断 `%10` → `%15` 是否影响 `accepted` 与 `final_fixed_ray_mse_mm2`。** 若 `final mse` 与 `accepted` 帧数不变 → 可放宽到 `%15`（省 1/3）。 |
| **(b) `old_scores` 的 6 次全量 cast** | `motion_v2.py:91-95` `for i in range(n): fixed_ray_score(...); fixed_ray_score(..., required=c)` **每帧调两次** | **第一次拿到 `c`（covered），第二次立刻用 `c` 重算同一批射线**。`fixed_ray_score(v,faces,points,required=None)` 与 `(...,required=c)` 的唯一区别是 `mask`。**第一次计算已经得到 `residual` 与 `covered`，第二次完全多余。** | **确定可删**：改成 `s0, c = fixed_ray_score(...); old_scores.append(s0); required.append(c)`。省 20 帧 × 1800 射线 × 2 阶段 = 72k casts。**零风险、零语义变化。** |
| **(c) 接受环的重复 cast** | `motion_v2.py:156-160` 每帧最多 5 次 `fixed_ray_score(candidate, faces, points[i], required[i])` | 20 帧 × 5 = 100 次 × 1200 射线 | 只在"前一档失败"时才算下一档，**已经是必要开销**。但 `scale=1.0` 有 11/20 帧直接通过 → 平均 ~1.6 次/帧。**不改。** |
| **(d) `identity_v2` 与 `fit_d` 的 `low_frequency` 完全同参** | `identity_v2.py:22` vs `fit_d.py:81` | 同 `gradient=8, bending=8, outer=2, solver=qp, research_v2=True, points=5000`；**唯一差别是输入 `posed_vertices`（motion_only vs alternated_motion）与 `bound=2.5`** | **不能直接删 `identity_v2`**（它是 `alternated_motion` 的输入，删了链路断裂）。**但可以用 `fit_d.py:59-78` 已有的 `regularized_fit` 复用机制反过来做**：`identity_v2` 复用 `motion_only` 上已算过的场。**这需要新增 `--reuse-field` 语义，属于改链，本设计不做**，只记录。 |

#### 4.6.3 前段"先测再改"的最小测点集（T01 交付）

| 测点 | 文件:行 | 期望读出什么 |
|---|---|---|
| `fit_c` 每阶段 `ray_surface_total_ms` / `calls` | `fit_c.py:156, 158, 250, 237` | 240 步里 116/152 次刷新各占多少 |
| `fit_c` `refine_shared_ms` | `fit_c.py:238` | 116 次 `ray_surface` × 9307 射线的真实成本（`refinement_log` 显示 9 帧 × 9233–9307 射线） |
| `fit_c` `evidence_ms` | `fit_c.py:248-253` | 20 帧 × 12000 射线 |
| `motion_v2` `corr_refresh_calls` / `corr_rays_total` | `motion_v2.py:103-109` | 10 次刷新 × 20 帧 × 1800 |
| `motion_v2` `accept_loop_ms` | `motion_v2.py:147-166` | 100 次 `fixed_ray_score` |
| `identity_v2` `solve_field_ms` / `front_protected_ms` | `identity_v2.py:24-26` | 137 s 的分解 |

---

## 5. 收益汇总（保守 / 乐观）

分母：总 1945.7 s。

| 项 | 保守 (s) | 乐观 (s) | 依据 |
|---|---:|---:|---|
| §4.2 删 `fit_d` 前段 `front_anchor_low/mid_frequency` | **180** | **260** | 3 帧采样+平面+1 次诊断射线建立（12,984 条）+5 个 outer 的 QP(setup 5×0.95 s + solve 100/275/mid)+4 次试步 |
| §4.3 射线求交改造（栅格 + Möller–Trumbore + 免费重心） | **180** | **260** | 单次 coarse 938→~190 ms（−80%），dense 2771→~700 ms（−75%）；dense 阶段粗模调用 ~127 次、dense 11 次 |
| §4.5 提前停止（`front_anchor_dense` 3→1 outer） | **60** | **100** | 2 × (QP 950 iter + 31 次试步 × 11/3 帧全量重投射) |
| §4.4 L1（`max_depth` 一致化）+ 4.4.1 诊断 | 0（诊断价值） | 20 | 若 L2/L3 放行 alpha≥0.06，试步数 15→3 量级 |
| §4.6.2(b) 删 `old_scores` 的重复 cast | **10** | **15** | 20 帧 × 1800 射线 × 2 阶段 = 72k casts |
| §4.3.4 `gc.collect()` 降频 | **8** | **9** | ~118 次 × 75 ms |
| §4.3.1 `chunk 32→512` | **0.3** | **0.5** | bench_ray |
| §4.1 零自由度语义 | 0 | 25 | 主要是 report 可读性 |
| **合计** | **438 s** | **690 s** | |
| **占总 1945.7 s** | **22.5%** | **35.5%** | |

> **诚实说明**：上表把 `front_anchor_low/mid_frequency` 与 `guarded_regularized`（`fit_d.py:91-101`）算在**保留**里；§4.6 的前段（H4/H5，`motion_v2` 双跑与 `identity_v2` 重采样）**还能再省 100–140 s**，但必须等 T01 的测点数据，本设计只给测点、不给结论。

---

## 6. 任务分解（给工程师的执行顺序）

> 硬约束：**每步都必须能独立验证**，且**几何输出逐位一致**（除明确标注为"质量判据变更"的任务外）。

### T01 — 计时插桩（P0，无依赖）**必须最先做**

**产出**：`docs/T01_TIMING_BASELINE.md` + 一份新运行目录 `runtime/outputs/<ts>_timing/`
**改动文件**：
- `head_rgbd/geometry_c.py`（`ray_surface` 的 `stats['timing']`、`front_protected_vertices` 的第 4 返回值、`_GC_INTERVAL` 计数器）
- `head_rgbd/geometry_d.py`（`solve_field` 的 `record['timing']` / `update_max_mm` / `backtrack_count` / `status` / `stop_reason_detail`）
- `head_rgbd/qp_d.py`（`solver_info['timing']`）
- `head_rgbd/fit_d.py`（`report['dense_timing']` + 日志一行 JSON）
- `head_rgbd/fit_c.py`、`head_rgbd/identity_v2.py`、`head_rgbd/motion_v2.py`（各 3–5 个计数器）

**验收**：
1. 几何逐位一致（`assert_array_equal` on `dense_vertices` / `dense_deviation_m` / `regularized_vertices`）。
2. `dense_timing.stage_total_ms` 与 `run_report.json` 的 `elapsed_s` 相对误差 < 3%。
3. 子项之和 ≥ 80% of `stage_total_ms`。
4. 输出 §3.6 的"现网热点分解表"——**这是后续所有任务的唯一依据**。

**风险**：极低。**不要**顺手改任何算法。

---

### T02a — 零风险微调（P0，依赖 T01）

**改动**：
- `geometry_c.py:48` `first_hits(..., chunk=512)`（32 → 512）
- `geometry_c.py:72` `gc.collect()` → `_GC_INTERVAL=200` 降频
- `geometry_d.py:238` `fixed_diagnostic` 的 `max_depth=.030` → `.008`（**L1，让判据与主查询一致**）
- `motion_v2.py:91-95` 删掉第二次 `fixed_ray_score`

**验收**：
1. 几何逐位一致。
2. `report.json` 的 `accepted[*]` 计数字段**完全不变**（L1 可能让 `missing_after` 上升，**允许变化但必须记录**）。
3. 阶段耗时下降 ≥ 15 s。
4. `gc` 降频后 RSS 增长 < 200 MB（用 `tracemalloc` 或 `os.getpid` + `Get-Process` 采一次）。

---

### T02b — 射线求交改造（P0，依赖 T02a）

**改动**：`geometry_c.py` 新增 `build_ray_raster` / `first_hits_raster`；`ray_surface:68` 改为调用新实现；`ray_surface:76` 用 MT 的 `(1-u-v, u, v)` 替代 `points_to_barycentric`。`fit_d.py` 对 `front_protected_vertices` 的两个拓扑各传一次预建表（可选，收益小）。

**验收（五条硬门槛，任一不过即回滚）**：
1. 逐帧 `ray`/`face`/`bary`/`hit`/`stats` 全字段相等（`front_anchor_dense` outer0 的 3 帧 + `guarded_regularized` outer0 的 9 帧）。
2. `dense_vertices` / `regularized_vertices` / `dense_deviation_m` 逐位一致。
3. 全部 `accepted[*]` 计数不变。
4. `raster_truncated == 0`。
5. `dense_timing.correspondence_ms` 下降 ≥ 60%。

---

### T03 — 删 `fit_d` 前段死计算 + 前段重复计算（P0，依赖 T01；与 T02b 无依赖，可并行）

**改动**：
- `fit_d.py:81-89` 删除（`front_anchor_low_frequency` / `mid_frequency`），`anchor`/`low_anchor` 从 `initial_fit/parameters.npz` 取，加 §4.2 的契约断言。
- `fit_d.py:187` `stages['low_frequency_anchor']` → 用同一个 `low_anchor`，或去掉该导出并更新 `limitations`。
- `fit_d.py:230` 同步。
- `motion_v2.py:91-95`（若 T02a 未做）。

**验收**：
1. `dense_vertices` / `regularized_vertices` / `dense_deviation_m` / `support_regularized.npz['refinement_frame_mask']` **逐位一致**。
2. 若保留 `closed/open_low_frequency_anchor_m.obj` 导出，与旧文件 `sha256` 相同；若不保留，`report.json` 有明确说明。
3. `field_logs` 只含 `{regularized, front_anchor_dense, dense}` + `front_section_removed` 说明字段。
4. `dense_timing.stage_total_ms` 下降 ≥ 180 s。
5. **非 guarded mode 显式报错**（缩小改动面）。

---

### T04 — 提前停止 + 零自由度语义（P1，依赖 T01 + T03）

**改动**：`geometry_d.py` 新增 `Settings` 的两个 `@property`（由 `noise_floor_mm` 导出）+ `ENERGY_REL_MIN = 1e-3` 模块常量；`solve_field` outer 循环尾部加 §4.5.2 的停止规则；`if not len(free)` 分支写全 `status/reason/free_rule`。

**验收**：
1. 几何逐位一致（停在当前几何）。
2. `front_anchor_dense` 的 outer 数从 3 → 1，并记录 `stop_reason_detail.update_max_mm`。
3. 每个 outer 的 `update_max_mm`、`rel_improvement`、`stop` 都进 `report.json`，可被第三方复算。
4. **不允许**出现 `status` 为空且有 `found=False` 的 outer。

---

### T05 — `front_anchor_dense` 覆盖率诊断（P1，依赖 T04，**只诊断不改判据**）

**产出**：`docs/T05_MISSING_RAY_DIAGNOSIS.md`
**改动**：`geometry_d.py` 的 `fixed_diagnostic` 返回 `newly_missing_*` 诊断字段（只加不改）；跑 `fit_d` 单阶段（`--points 5000`，约 5 min）。
**产出内容**：37 次被拒 trial × 每次 1–4 条新增缺失射线的分类直方图 + 每条射线的 `uv` / `local rms` / `|det|` / 是否在 silhouette 邻域 / 前后 `t` 差。
**验收**：直方图能明确回答"是否 90% 落在 (a)/(b)"。**只有回答"是"才允许进入 L2/L3。** 否则关闭这条线，接受现状。

---

### 依赖图

```mermaid
graph LR
  T01[T01 计时插桩] --> T02a[T02a 零风险微调]
  T01 --> T03[T03 删前段死计算]
  T02a --> T02b[T02b 射线求交改造]
  T03 --> T04[T04 提前停止+零自由度语义]
  T04 --> T05[T05 覆盖率诊断]
  T02b -.可并行.-> T03
```

**关键路径**：T01 → T02a → T02b →（T03/T04 并行）→ T05。
**若只能做一半**：做 **T01 + T03**（收益 ~200–280 s，风险最低，纯删死码）。

---

## 7. 本次先做什么、暂不做什么

### 先做（按序）

1. **T01 计时插桩** —— 没有它，后面每一步的收益都只是估计。
2. **T02a 零风险微调** —— 15 s，纯机械。
3. **T03 删 `fit_d` 前段死计算** —— 200–280 s，最大单笔收益，风险已被三条独立证据锁死。
4. **T02b 射线求交改造** —— 180–260 s，需过五条硬门槛。
5. **T04 提前停止 + 零自由度语义** —— 60–100 s，顺手把 `report.json` 的可读性修好。
6. **T05 覆盖率诊断** —— **只诊断不改判据**。

### 暂不做（并说明原因）

| 不做 | 原因 |
|---|---|
| **恢复侧向细化能力（改 `fit_c --front-anchor` 的自由集定义）** | 这是**质量/策略**问题，不是加速问题。`shared_identity_guarded` 的 `changed_side_detail_vertices=0` 说明整个侧向细化链在当前采集下不产生任何几何变化。改它需要重新定义"什么算侧向新可见"，会改变交付几何。**必须先有独立评估任务（T03 之外的评估角色）给出判据。** |
| **L2/L3 覆盖率判据的放宽** | 依赖 T05 的直方图。在拿到"90% 落在 silhouette/数值退化"之前，放宽等于用指标换指标。 |
| **减少 `points` / `steps` / 删除任何阶段** | 任务书明确禁止作为主要方案。**`steps=240`、`points=1200/5000/16000` 全部不动。** |
| **放宽 `eps_abs/eps_rel`、删约束行、`min_signed_area_ratio` 松绑** | 禁止。`qp_d.py:47-48` 的 `1e-6/1e-6` 与 `:37-38` 的 `1e-5` 内缩全部保留。 |
| **`fit_c` 的 `%10` → `%15` 刷新降频** | 需 T01 的 `corr_refresh_calls`/`accept` 数据先证明不影响 `final_fixed_ray_mse_mm2`。**`%10` 不动。** |
| **`identity_v2` 复用 `motion_only` 已算场** | 需要新增 `--reuse-field` 链路语义，属于改链。收益还要减去 `alternated_motion` 的重算。**留作下一轮。** |
| **引入 embree（`pyembree`/`embreex`）** | 环境里没有，装它要改 `requirements.txt` + 重建 runtime env，且 `embreex` 在 Windows/py3.10 的轮子可用性未验证。**本设计用纯 numpy + 现有 rtree，不引入新依赖。** |
| **并行化（多进程/多线程）** | 16 逻辑核当前全空闲，理论上有 4–8× 潜力，但 `solve_field` 的状态机（`field` 跨 outer 累积、`support_union` 合并）不支持简单并行。**且并行会让"几何逐位一致"的验收变难。本设计不碰。** 若后续要做，先做 T02b（把单次求交打下来），再看剩余并行空间。 |
| **`loss_analysis/` 任何改动** | 另一个任务。本设计与它零交集（已核实 `head_rgbd/` + `core/` 对 `loss_analysis` 的引用数为 **0**）。 |
| **改采集、模型资产、历史运行结果** | 禁止。T01 产生的运行目录是**新目录**，旧基线目录只读。 |
| **加通用计时框架 / 缓存框架 / 策略注册器** | 禁止。所有改动都是局部 `perf_counter` + dict 字段 + 一个模块级计数器。 |

---

## 8. 共享约定（给工程师）

```
单位：几何一律米；法向标量场 field/deviation 一律毫米（mm）。
      parameters.npz 里的 *_deviation_m / deviation_m 一律「米」，用前必须 *1000。

噪声尺度：noise_floor_mm = 1/sqrt(12) = 0.2886751345948129（建模为 ±0.5mm 均匀量化）
         —— 所有阈值都从这里导出，不要新增不可解释的常量。

误差口径：fixed_ray_energy 的 penalty = 8.0 mm（strict-coverage），
         聚合 = 每帧 MSE 的等权平均（aggregate_fixed_ray_energy, geometry_d.py:51-53）。
         任何"改善"的证明都必须用这个口径重算，不能用 max_depth=.030 的口径。

报告字段：新增字段一律「追加」，不改既有键名/语义。
         旧 report.json 必须仍然可读（不添加 required 字段）。

日志：每个 stage 结束打一行 JSON 到 logs/NN_*.log，便于 grep 与跨运行比对。

验证铁律：除 T05 外，每个任务都必须通过
         np.testing.assert_array_equal(dense_vertices, baseline_dense_vertices)
         否则视为回归。

禁止：改 eps_abs/eps_rel、删约束、降 points/steps、改采集/资产/历史结果、
     加 fallback / 吞异常 / 通用缓存框架。
```

---

## 9. 仍未澄清 / 本设计的假设

| 项 | 假设 | 若假设不成立怎么办 |
|---|---|---|
| `dense` 阶段前段 `anchor` 的确切耗时 | 200–280 s（按 outer 结构推算，未实测） | **T01 会给出实测值**；若实测 < 150 s，T03 的优先级下调，T02b 提前 |
| `alternated_motion/parameters.npz['deviation_m']` 单位 | 米（依据 `motion_v2.py:175` ← `identity_field.npz` ← `identity_v2.py:27` 的 `field*.001`） | T03 实现时加 `assert max < 1e-2`，失败即停 |
| 全序列 `intrinsics` 是否相同 | 同一次会话采集，应为同一相机内参 | T02b 实现时**断言**；不同则按帧建栅格表（分支，非 fallback） |
| 栅格化 `r=2` 是否够 | 三角形边长 ≥ ~1.5 mm，720×1280 下投影 > 2 px | T02b 的逐帧比对会暴露；若漏则 `r=3` 或按三角形投影 bbox 精确覆盖 |
| `front_anchor_dense` outer1/2 的 `update_max_mm` | 1.7e-5 / 8.7e-6 mm（按 `dense_front_anchor_deviation_m` 推） | T01 实测；若 > 0.003 mm 则 T04 需重新推导阈值 |
| `missing_after <= missing_before` 被放宽后是否真能提升质量 | **未知**，取决于 T05 直方图 | 若 T05 显示缺失射线多在"面正对相机、深度稳定"处，**保持现状，接受 0.116% 的改善** |
| 更快准则（`update_max_mm`）在各阶段第一轮是否被误触发 | 未知 | T04 先在 `front_anchor_dense` 上灰度 1 个 outer，确认 `stop=None` 后再全量 |
