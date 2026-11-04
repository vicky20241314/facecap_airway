# SPEEDUP_NOTES — 对当前实现改动的即时核对与纠偏

> **过程记录，机制已变。** 本文件写于中间方案（`build_mesh`、`ray_surface(mesh=)`）时期；
> 最终实现是 `head_rgbd/ray_index.py`：索引对象自带几何、由调用方按需构建，
> `ray_surface` 不再接受 `mesh=`，也不再有 `owned`/`del`/`gc.collect()` 的生命周期分支。
> 结论层（不要调批大小、不要跨几何复用、要查首命中语义）仍然成立；
> 权威说明见 `docs/FAST_PATH_REPORT.md`。

写于 `docs/SPEEDUP_DESIGN.md` 之后。工程师已在改 `geometry_c.py`（`build_mesh` / `first_hits` 空批归一 / `ray_surface(mesh=, timing=)`）。
以下四块只讲判断，不贴源码。

---

## 1. 热点排序：dense 阶段（1131 s）最耗时三项

1. **`fit_d.py` 前段死计算（~180–260 s）** —— `front_anchor_low_frequency` + `mid_frequency` 跑 5 个 QP outer、4 次试步、13 帧次采样与本地平面；结果只进 `stages['low_frequency_anchor']` 这个 OBJ，而 `defaults.json` 的 `export.mesh_stage="dense"`，交付链路无人读。依据：`dense_fit_base = interpolation_vertices` 只依赖 `regularized`；`alternated_motion/parameters.npz` 与 `front_anchor/parameters.npz` 的 do-nothing 断言在本机通过。

2. **射线求交（~180–260 s 可回收）** —— 不是批大小、不是窄相位，是"每条射线一次 3D rtree 查询"叠加 trimesh Python 层。依据：4850 条射线实测宽相位 518 ms（粗模）/ 1430 ms（dense），窄相位仅 ~100 / ~490 ms，其余 ~2300 / ~900 ms 是 trimesh 逐调用开销；`trimesh/ray/ray_triangle.py:333` 是 `for i, bounds in enumerate(bounding): tree.intersection(bounds)`。chunk 32 已接近最优（你实测一致），**不要再调批大小**。

3. **`front_anchor_dense` 的 40 次试步（~60–100 s）** —— 其中 37 次因覆盖率被拒；每次试步都全量重投射 13045 条固定射线 + 3 帧 5000 条主查询。相对改善 0.116% / 0.0018% / 0.0009%，后两轮在噪声下不可分辨。

---

## 2. 你给的任务里我认为有问题的地方

### 2.1 "只在顶点数组完全相同时复用 mesh" —— **约束不够，会把正确性绑在浮点相等上**

现在的注释写的是 "must never pass a mesh built for different vertices"，但**没有定义"不同"**。三种真实危险：

- **同一批顶点、不同 `max_depth` / `max_plane`**：可以共用，安全。
- **同一批顶点、不同 `f`**：危险。`ray_surface` 用 `v[f[face]]` 取三角形，但 mesh 的 rtree 是绑定 `(v,f)` 那一对的。只比 `v` 不比 `f` → 陈旧面索引。
- **同一批顶点、同一 `f`、但顶点被原地改写（in-place）**：**最危险**。`v is mesh.vertices` 为真、`np.array_equal` 也可能为真（看你比较的时机），但 rtree 已经在旧包围盒上建好。这类错误不报错，只返回"另一个表面"的对应，属于你明令禁止的"陈旧结果"。

**建议的契约（比"完全相等"强，且低成本）**：
- 复用方**必须**同时传 `v`、`f`，并在内部断言 `mesh.vertices is v or np.shares_memory(mesh.vertices, v)` **且** `mesh.faces is f or np.shares_memory(mesh.faces, f)`。共享内存断言比 `array_equal` 便宜且更准确——它同时排掉"拷贝了一份相同的数组"（那种情况下 mesh 可以共用，但没有共享内存就不能证明没被原地改）。
- **不要**在 `ray_surface` 内部做这个断言（热路径，每次 5000 射线都做一次相等比较是浪费）。放在**调用方**：`fit_d` / `geometry_d` 里那些"我要复用"的点，显式写出 `assert mesh.vertices is posed_i`。
- **跨几何状态复用一律禁止**，且不能靠"我这次记得重建"来保证——**把 mesh 的生命周期绑在一个局部作用域里**，出了作用域就 `del`。现在的 `owned` 参数方向是对的（只有自建才 `del` + `gc.collect()`），但**外部传入的 mesh 不释放**意味着调用方要自己管，这正是出错的地方。

**更重要的判断：这条优化的收益被高估了。** 你的实测里"新建 trimesh + rtree"只占约 40%（第一次查询内含 tree 构建 39 ms / 160 ms）。即使把**所有**可复用的点都复用，dense 阶段的上限收益也在 60–90 s 量级，远小于 §1 的两项。**不要让它成为主线**，只作为顺手的附带项。

### 2.2 "提前停止用 `actual_update_mm < noise_floor_mm` 且 MSE 改善 < 0.01%" —— **两个阈值都错，会误停**

**阈值 1：`actual_update_mm < noise_floor_mm` 太松一个数量级。**
`noise_floor_mm = 0.2887 mm`。`front_anchor_dense` outer0 实际最大更新只有约 1e-3 mm 量级（alpha=1/256），outer1/2 在 1e-5 量级。这个判据在**第一轮就会触发**，把唯一有实际改善的 outer0 也停掉。正确阈值应是从 `noise_floor_mm` **导出**的更小量（见 §3）。

**阈值 2：`MSE 改善 < 0.01%` 太严，把 outer0 也判为"无改善"。**
outer0 的改善是 **0.116%**，是 0.01% 的 11.6 倍——**这一轮会通过**，但 outer1（0.0018%）和 outer2（0.0009%）会被截断，而那两轮本来就该截断。所以 `0.01%` 这一侧其实**方向对、数值偏严**：13045 条射线的 MSE 标准误约 `sqrt(2/n) ≈ 1.24%`，用 `0.01%` 当门限等于要求"改善到统计分辨率的 1/100"，这不合理；但也正因为严格，它**不会误停**。真正危险的是阈值 1。

**结论**：`actual_update_mm < noise_floor_mm` 这一条**必须改**，否则会停掉 outer0，直接丢掉整个 `front_anchor_dense` 阶段（而这是唯一产生几何变化的阶段）。`MSE 改善 < 0.01%` 可以保留但建议放到 `0.1%`（= 统计分辨率的 1/12，仍远低于 1.24% 的标准误，不会放行噪声）。

### 2.3 你对 `missing_before` 恒为 0 的解释是对的，且比我的更准确

`diagnostic_rays` 每个 outer 用当前场重新定义（`geometry_d.py:229-235`），所以 `missing_before` 恒为 0 是**结构性的**，"新增缺失"是相对本 outer 起点的**相对量**，不是绝对覆盖。这意味着：

- 严格覆盖规则 `missing_after <= missing_before` 在 `missing_before ≡ 0` 时**恒等价于"零新增缺失"**，与 alpha 无关。
- 后果正是你观察到的：alpha=1 时 MSE 改善 19% 却被拒，因为 miss=4；alpha→0 时候选≈当前场，覆盖必然恢复，所以"接受"接近 no-op。

**这不只是"判据太严"，而是判据本身在这个实现下退化了**：它测的不是"覆盖率好不好"，而是"这一轮有没有动过轮廓"。

### 2.4 关于你实测的 `first_hits` bug —— 工程师的修法是对的，但**验收标准不完整**

`h=np.empty((0,3))` / `r=np.empty(0)` / `f=np.empty(0)` 的归一化是正确的，`r+start` 保持全局编号也正确。但缺两件事：
- **`np.concatenate` 之前应断言每个 list 元素的 ndim 一致**（现在只处理了"整批 0 命中"，没处理"部分命中但形状异常"）。
- **需要一个能真正拦截它的测试**：用基线 dense 网格 + 一批**全部 miss** 的射线（例如全部朝相机反方向）+ 一批**全部命中**的，混在同一 `points` 里跨越 32 的批次边界。纯"首命中一致性"测试不会触发这个 bug。

---

## 3. 被漏掉的高收益点

- **`fit_d.py:81-89` 整段是死代码**，删掉零风险、几何逐位不变——这是最大单笔，却不在任何任务里。
- **`guarded_regularized` / `guarded_dense` 的 QP 和线搜索从来没执行过**（`free_vertices=0` 在进 QP 前就 `break`），所以"零自由度短路"省不到时间，只是 report 可读性；但它**掩盖了一个真相**：`shared_identity_guarded` 本身 `changed_side_detail_vertices=0`、`final−front_anchor` 最大差 **0.000000 mm**，侧向细化整条链在当前采集下不产生任何几何变化。
- **`fixed_diagnostic` 用 `max_depth=.030`（`geometry_d.py:238`），主查询用 `.008`** —— 判据口径不一致，3.75 倍的宽门限把"深度错 3 cm 的命中"也算 covered。
- **`motion_v2.py` 的 `old_scores` 每帧调两次 `fixed_ray_score`**，第一次已拿到 `covered`，第二次纯多余（20 帧 × 1800 射线 × 2 阶段）。
- **`gc.collect()` 每次 75 ms**，trimesh 4.4.9 的 `RayMeshIntersector` 只持 `self.mesh` + `Cache`，无循环引用，`del mesh` 后引用计数即归零——可降频或删。

---

## 4. 风险提示：最可能在哪一步破坏正确性

1. **mesh 复用的"陈旧几何"**（最高风险，且静默）—— 改了顶点没重建 rtree，返回另一个表面的对应。表现为"几何看起来正常但数值略偏"，任何 assert 都抓不到。**只能靠作用域绑定 + 共享内存断言防。**
2. **删 `fit_d.py:81-89` 时的 `support_regular` 归属** —— `guarded_regularized`（`:91-101`）**不能一起删**，它产出 `support_regular`，下游 `loop_mapping` 要用；上游 `alternated_motion/support.npz` 的 mask 是 16000 点/帧、语义不同，**不能替换**。误删 → `loop_mapping` 的固定控制点集合变了 → 几何全变，且不会报错。
3. **`anchor` 单位** —— `parameters.npz` 的 `deviation_m` 是**米**，`solve_field` 的场是**毫米**。从上游读回来少乘 1000 → 位移放大 1000 倍，会被 `safety` 拦住（这里反而是安全的）。
4. **提前停止的阈值取了 `noise_floor_mm`** —— 会在 outer0 就停掉 `front_anchor_dense`，静默丢掉唯一的几何改善，且几何仍"合法"、所有 assert 都过。**这是最隐蔽的一条。**
5. **`first_hits` 空批修复后仍可能在"部分命中 + 极端形状"上崩** —— 需要跨批次边界的混合测试，纯首命中一致性测不出来。
