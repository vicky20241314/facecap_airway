# 正面 RGB-D 锚点 + 连续转头 RGB-D 重建实验

独立实验模块：`head_rgbd/experiment/`，入口 `python -m head_rgbd.experiment.turn_run`。

本模块**不进入**生产链 `head_rgbd.pipeline`，不给 `head_rgbd.cli` 增加子命令，不改任何默认参数，
所有产物只写进自己的 run 目录（默认 `runtime/turn_experiment/<stem>_turn_<时间戳>/`）。
原有 `reconstruct` 链路、`config/defaults.json` 与既有输出行为保持原样。

---

## 1. 交付状态（如实说明）

| 项目 | 状态 |
| --- | --- |
| 独立模块与入口 | 已完成，包含独立的连续转头现场采集入口 |
| 配置与校验 | 已完成，越界值直接报错，不做静默回退 |
| 语法 / 导入 | 已用 `runtime/env/python.exe` 验证：全部模块可导入，CLI `--help` 正常 |
| 单元测试 | 全仓库 **173 passed / 1 skipped / 2 subtests passed** |
| 真实数据重建 | 2026-11-02 与 2026-11-03 的默认/对照运行均完成 8 阶段并通过交付验收 |
| 同角度留出精度 | 7 个未参与拟合帧全部改善；中位数 `0.763 → 0.723 mm`，p90 `2.011 → 1.953 mm` |
| 独立几何真值 | **仍不可用**；不能把同采集留出结果外推为通用精度或优于静态采集 |

2026-11-02 用户进一步授权现场采集和完整测试，使用 5 帧正面锚点 + 100 帧连续转头 RGB-D，
选出 7 个关键帧并生成初始化、联合拟合、最终三张网格。修复日志编码、输入字段、布尔掩码和交付统计问题后，
从头完整运行 122.787 秒通过；最终网格 20,018 顶点 / 39,904 面。
详见 [本次完整验证报告](TURN_EXPERIMENT_VALIDATION_20261102.md)。此结论仅覆盖该数据和环境的流程执行及产物完整性。

2026-11-03 的算法更新进一步修复了逐点权重归约、瞬时可见性永久累积和几何变化基准问题，
并把每角度最多 2 个附近可靠帧作为支持观测加入联合约束。静态/视频成对实测报告见
[连续观测算法与留出评测](TURN_ALGORITHM_ACCURACY_20261104.md)。

---

## 2. 输入契约与假设

### 2.1 两个输入角色

| 角色 | 内容 | 用途 |
| --- | --- | --- |
| `--anchor` | 一张或一小组正面闭口 RGB-D | 初始化形状、尺度、五官与正面形态；作为锚点关键帧 |
| `--turn` | 一段闭口、自然表情、左→右转头 RGB-D 序列 | 提供侧面与斜侧视角的观测 |

两个输入都必须是**同步 RGB-D**，且相机固定、运动只来自头部。

### 2.2 可接受的来源形式（`turn_inputs.plan_source`）

| 形式 | 判定 | 必需内容 |
| --- | --- | --- |
| `session.json`（capture 会话） | `capture_session` | `rgb_depth_aligned: true`、`status` 为 `complete` 或空、指向的分段 npz |
| `frames.json`（帧清单） | `frame_manifest` | `rgb_depth_aligned: true`（或在显式允许时接受 `session_alignment_unverified`） |
| `sequence.json`（已准备序列） | `sequence` | schema 必须为 `head_rgbd.sequence.v1` |
| 单个 `segment_00x.npz` | `segment_npz` | 同时含 `rgb`、`depth`、`intrinsics`、`timestamps_s` |
| 一目录的逐帧 npz | `frame_directory` | 同上 |
| 视频 / 图片目录 | **拒收** | 抛 `UnsupportedInput`，提示需要同步 RGB-D |

### 2.3 明确报错、不静默退化

- 彩色视频（`.mp4/.mov/.avi/...`）或图片目录 → `UnsupportedInput`，消息里写明需要同步 RGB-D 序列。
- npz 缺 `depth` 或缺 `intrinsics` → `UnsupportedInput`，列出缺失键名。
- 会话未声明 `rgb_depth_aligned: true` → `UnsupportedInput`。仅当配置里显式打开
  `input.allow_unverified_alignment` 且清单自称 `session_alignment_unverified` 时才放行，并把
  `calibration_status` 记为 `alignment_unverified` 写进产物。
- 深度单位：约定为毫米 `uint16`，内部统一换成米；`depth_unit` 与 `calibration_status` 会写进
  `tracking/*.json`、`sequence.json` 与 `delivery/experiment_config.json`。
- 帧率/时间戳：相邻帧间隔超过 `input.max_adjacent_gap_s`（默认 0.2 s）会被记为新的跟踪组，
  而不是当成连续运动。

坐标系沿用项目约定 `X_right_Y_up_Z_backward`，内部单位米。实测约定：**yaw 负值 = 画面左侧**，
正值 = 右侧（同一次采集里 `closed_left ≈ −15…−16°`、`closed_right ≈ +18°`、`front ≈ −3…−6°`）。

---

## 3. 总体流程（单遍，8 个阶段，无迭代交替）

```
tracking → keyframe_selection → prepared_keyframes → front_anchor
        → joint_shared_identity → pose_correction → control_field → delivery
```

每个阶段都由 `head_rgbd.reporting.StageRunner` 起**独立子进程**执行，命令、日志
（`audit/logs/NN_<stage>.log`）、退出码和耗时写入 `audit/run_report.json`。任一阶段非零退出即中止，
不会带着半成品继续。

| # | 阶段 | 做什么 | 产物 |
| --- | --- | --- | --- |
| 1 | `tracking` | 逐帧轻量跟踪，得 yaw/pitch/roll、质量、连续性分组 | `tracking/{anchor,turn}.json`、`.npz`、`summary.json` |
| 2 | `keyframe_selection` | 角度分桶 + 质量排序选帧，硬预算 9、不凑数 | `selection/keyframes.json` |
| 3 | `prepared_keyframes` | 只对入选帧重跑检测/关键点/解析，落观测 | `prepared/<id>.npz`、`prepared/sequence.json`、`prepared/anchor_sequence.json` |
| 4 | `front_anchor` | `head_rgbd.fit_c --front-only` 做正面初始化 | `front_anchor/parameters.npz`、`report.json` |
| 5 | `joint_shared_identity` | 所有关键帧共拟一个中性形状 + 每帧刚性姿态 | `joint_shape/parameters.npz`、`report.json`、`reference_geometry.json` |
| 6 | `pose_correction` | `head_rgbd.motion_v2` 跑一次姿态修正 | `pose_correction/parameters.npz`、`report.json` |
| 7 | `control_field` | 一次低/中频位移场 + 一次 Loop 混合 | `field/final_vertices.npz`、`final_mesh.obj`、`report.json` |
| 8 | `delivery` | 只导出 6 个交付文件 | `delivery/*` |

---

## 4. 逐阶段算法与函数职责

### 4.1 `turn_tracking` — 轻量连续跟踪

`analyze(sequence, config, role)` 返回 `{records, arrays, summary}`。

- **检测频率**：YuNet 检测框 + TUFA 68 点只在 `i % detect_stride == 0`（默认 3）或上一帧跟踪失败时执行；
  中间帧用前后向 KLT（`winSize=(klt_window, klt_window)`，默认 21，`maxLevel=klt_levels`）把 68 点 uv 传播过去，
  前后向误差超阈或有效点少于 `klt_min_landmarks` 就重定位（重跑检测）。
- **姿态**：用中性 FLAME 模板的上半脸关键点（17–27、36–47）与实测 3D 关键点做 SVD 刚体配准
  （`prepare.rigid`），得到 `R, t` 与残差；`yaw/pitch/roll = Rotation.as_euler('xyz')` 的
  `[1]/[0]/[2]`（度）。残差超过 `max_fit_residual_mm` 视为不可用。
- **质量**：清晰度（`_sharpness`，拉普拉斯方差相对 `reference_sharpness`）、脸部有效深度像素占比
  （相对 `reference_face_depth_pixels`）、有效关键点比例、拟合残差（相对 `reference_residual_mm`）四项的
  加权几何均值，落在 `[0,1]`。
- **表情代理**：内唇 3D 间距（`reference_contract.inner_lip_gap_mm`）与嘴部比例相对滑动 EMA 参考的变化量；
  超过 `expression_gap_mm` 或 `expression_ratio` 就判定表情变化。
- **连续性分组**（`tracking_group`）：以下任一情况新开一组 —— 序列起点、时间间隔超限
  （`temporal_gap`）、时间戳非单调（`nonmonotonic_timestamp`）、跟踪丢失后重定位
  （`tracking_relocalized_after_loss`）、相邻帧 yaw 跳变超 `max_yaw_jump_deg`、表情变化
  （`expression_change`）。分组的含义只是"时间上连续"，不是"角度相同"。

### 4.2 `turn_keyframes` — 选帧（纯 NumPy，可单测）

`select(anchor_entries, turn_entries, config)`：

1. 过滤：张口帧、跟踪不可用帧、无姿态帧、`|yaw| > max_yaw_deg`、质量低于 `min_quality` 的候选直接剔除并计数。
2. 角度分桶：`angle_bucket(yaw, angle_bin_deg)`（默认 10° 一桶，四舍五入到桶心）；
   `group_label` 生成可读名 `yaw-20` / `yaw+20`。
3. 锚点：优先取 `role == anchor` 且 `|yaw| ≤ anchor_max_yaw_deg` 中最可靠的一帧；不可用时按
   "放宽到 turn 帧的正面帧" → "最可靠候选" 依次降级，并在 `selection_reason` 里如实写明用了哪条降级。
4. 补帧：每轮在**尚未覆盖的角度桶**里挑 score 最高的候选
   （`score = quality × (1 + surface_gain_weight × 新增可见关键点比例) + side_balance_bonus × 左右平衡`），
   直到 `max_keyframes`（默认 9）用尽或没有新角度可取。
5. **不凑数**：`budget.padded = false`，帧不够就返回更少帧，`coverage.missing` 明确写出
   `left_side_not_observed` / `right_side_not_observed` / `no_near_frontal_frame` / `narrow_yaw_span`。
6. 每个角度桶仍只选一个关键帧作为姿态/覆盖锚点；此外按
   `repeat_observations_per_angle`（默认 2）和 `repeat_max_angle_delta_deg`（默认 6°）
   选择附近可靠支持帧，写入 `support_frames`。支持帧不占关键帧预算，原始深度图不直接平均。

输出里 `angle_groups`（角度桶）与 `tracking_groups`（连续性段）是**两个独立列表**，
`tracking_groups[*].spans_multiple_angles` 直接标出"一个连续段跨了多个角度"这一常见情形。

### 4.3 `turn_prepare` — 物化关键帧与支持观测

`materialise(run_dir, config, experiment)`：对每个关键帧和支持帧重新走一次
`detect_face_bbox`（YuNet）→ TUFA 68 点 → UniFace 解析，写出与 `head_rgbd.prepare` 同字段的 npz
（`rgb, depth, intrinsics, uv, xyz, valid, landmark2d_valid, face_mask, neck_mask, timestamp_s, source_frame`）。

同时写两个序列文件：

- `prepared/sequence.json`：锚点帧在前，其余按角度桶中心升序；`rows` 带
  `id / segment(=tracking_group) / state='closed' / view / index / timestamp_s / source / source_sha256 /
  source_frame / status='candidate' / file / sharpness / overlap / degraded / angle_group /
  tracking_quality / yaw_deg / observation_role / canonical_keyframe_id`，并声明
  `continuous_full_turn: true`。
- `prepared/anchor_sequence.json`：只含 `role == anchor` 且 `observation_role == keyframe` 的帧。

**为什么是两个文件**：`fit_c --front-anchor` 会校验锚点报告的 `sequence_sha256` 与当前 `--sequence`
完全一致。让"正面初始化"在 `anchor_sequence.json` 上独立跑一次 `fit_c --front-only`，再让联合拟合
校验同一份文件的 sha，就既能复用现成的正面初始化，又不需要改动 `fit_c`。

`view` 映射：锚点帧恒为 `front`；其余按 `|yaw| ≤ front_yaw_deg → front`、`yaw < 0 → left`、否则 `right`。

### 4.4 `front_anchor` — 正面初始化（复用 `fit_c`）

直接调用现有命令，不复制实现：

```
python -m head_rgbd.fit_c --sequence <run>/prepared/anchor_sequence.json \
  --output <run>/front_anchor --front-only --postfit minimal \
  --keyframes-per-segment <max_keyframes> --points <front_init.points_per_frame> \
  --steps <front_init.steps> --expression-components <flame.expression_components> \
  --runtime <runtime> --device <device>
```

`--keyframes-per-segment` 显式给到 `max_keyframes`，因为 `fit_b.choose` 的"每 segment 最多 N 帧"
限制在这里没有意义——选帧已经由 `turn_keyframes` 决定过，不能再被截断。
`--postfit minimal` 跳过局部细化、颈部与网格导出，只保留 `parameters.npz` 与 `report.json`。

### 4.5 `turn_shape` — 共享形状联合拟合（本实验的核心）

`run(run_dir, config, device)`。变量极少：

| 变量 | 维度 | 说明 |
| --- | --- | --- |
| `shape` | `(1, S)` | 全场唯一的中性形状 |
| `expression` | `(n, E)` | 每帧表情，初值取锚点均值，偏移受 `expression_bound` 约束 |
| `jaw` | `(n, 3)` | 每帧下颌，相对参考受 `jaw_delta_bound_rad` 约束 |
| `rotation_delta` | `(n, 3)` | 相对锚点/刚体初值的轴角修正 |
| `translation_delta` | `(n, 3)` | 相对初值的平移修正 |

初始化：锚点覆盖的帧直接用 `fit_c` 的 `R/t/expression/jaw`；其余帧用中性模板上半脸关键点与实测 3D
的刚体配准给初值（点数不足 8 时退回 `z = −0.5 m` 的保底值，并在 `report.json` 里标
`pose_source=fallback_*`）。

两个阶段：

1. `pose_steps`：只优化 `rotation/translation`，`shape` 不参与（关键点项），把每帧先摆正。
2. `joint_steps`：加入深度项，每 `correspondence_interval` 步刷新一次对应关系
   （`geometry_c.ray_surface` 精确首命中 + 重心坐标，`max_depth=8 mm`、`max_plane=5 mm`）。

**帧权重（角度组平衡）**：`_frame_weights` 按 `angle_group` 分组，每组总权重固定为 `1/G`。
关键帧和同角度支持帧在组内按 `tracking_quality` 分配这唯一一票，因此增加重复帧不会放大该角度
的总影响。`report.json` 的 `angle_weighting` 表逐组给出总权重与逐帧权重。

**观测权重**（每个对应点）：`incidence² × plane_confidence ×
(noise_floor / max(σ_local, noise_floor))²`。对应点先通过有限性、首命中、深度/平面门限、
`local_depth_planes` 可用性和模型/观测局部法向一致性检查；随后先计算逐点 Smooth-L1，
再以 `sum(weight × pointwise_loss) / sum(weight)` 聚合。空对应、全零权重和非有限值回退为零损失，
不会用默认标量归约抵消权重。

**关键点项**（沿用 `fit_c` 形式）：3D 稳定点 17–47 权重 1；内唇 60–67 权重 0.25；2D 投影 0.15；
正面帧嘴部 2D 追加 0.35；闭口内唇 gap 约束（`relu(gap − 2 mm)`）权重 0.15。

**先验**：`0.002·shape² + expression_prior_weight·mean(expression²) +
jaw_delta_prior_weight·mean((jaw − ref)²) + anchor_shape_prior·(shape − anchor_shape)²`。
锚点提供的是**先验**而不是硬约束，普通面区不会被永久固定在正面形态上。

不使用：逐顶点自由形变、全视频逐帧非刚性拟合、TSDF、神经网络、多轮交替重建。

### 4.6 `pose_correction` — 一次姿态修正（复用 `motion_v2`）

```
python -m head_rgbd.motion_v2 --sequence <run>/prepared/sequence.json \
  --initial-fit <run>/joint_shape --output <run>/pose_correction \
  --steps <pose_correction.steps> --runtime <runtime> --device <device>
```

只跑一次。`motion_v2` 的边界（旋转 ≤ 0.02 rad、平移 ≤ 3 mm、表情 ≤ 0.2、下颌 ≤ 0.03 rad）
和逐帧固定射线接受规则原样生效。实验入口把 `turn_shape.report.json` 中按角度组归一化的
`frame_weight` 传给 `motion_v2`，因此 21 个 `7 关键帧 + 14 支持帧` 观测不会按帧数无条件放大
某一角度；旧 run 没有该字段时自动回退到原来的逐帧均匀目标。

为了让 `motion_v2` 能直接吃本实验的输出，`turn_shape` 写出的 `parameters.npz` 与 `report.json`
字段和 `fit_c` 兼容（`shape/expression/jaw/R/t/posed_vertices/faces/landmarks/landmark_faces/
landmark_barycentric/identity_selected/mouth_identity_excluded/deviation_m/selected`、
`sequence_sha256/frames/references`）。

### 4.7 `turn_field` — 一次低/中频形变 + 一次 Loop 混合

`run(run_dir, config, device)`：

1. `base = pose_correction/parameters.npz['posed_vertices']`，`direction = 顶点法向`。
2. 排除集：`anatomical_exclusion(base, data, front_ids, mouth_identity_excluded)`（眼、口、颈部等）。
3. **只调用一次** `geometry_d.solve_field`，`Settings` 关键取值：`solver='qp'`、`research_v2=True`、
   `view_balance=True`、`points=5000`、`outer_steps=2`、`gradient/bending=8 mm`、
   `displacement_bound=2.5 mm`、`support_min_information=0.5`、`coverage_allowance_ratio=0.001`、
   `loop_target_blend=0.5`。
4. 传进去的 `rows` 是**副本**，`segment` 被重写成 `angle_group`：

   ```python
   rows_field = [dict(row, segment=row["angle_group"]) for row in rows]
   ```

   配合 `view_balance=True`（`front_boost=1.0`），`surface_v2.burst_weights` 就给出
   "每组总权重相等、组内按帧数均分"的角度平衡权重。**公共函数一行未改。**
5. 细分与混合：`subdivide` → `loop_support_exclusion` → `loop_mapping`（线性映射覆盖到加密拓扑）→
   `safe_loop_base(...)`。目标混合系数固定 0.5，只有几何安全检查失败时才降低，
   并把这个原因写进 `report.json` 的 `subdivision.loop_blend.fallback_reason`。
   **不会**因为几个深度采样点的微小差异把系数判成 0。
6. 导出参考帧最终网格，并计算指标（见 §7）。

几何变化指标分别记录三段：`shared_shape_to_control`（控制场相对共享形状）、
`control_to_loop`（Loop 相对控制场）、`shared_shape_to_final_total`（最终相对共享形状）。
三者都使用同一帧、同一坐标系和同一顶点对应；变化量明确不是精度分数。

### 4.8 `turn_run` — 编排与交付

- `main` → `_run_experiment`：读配置 → 覆盖 `--device/--detect-stride/--max-keyframes` →
  `resolve_runtime_dir` → `asset_gate` → **先校验两个输入**（失败时尚未创建目录）→
  独占创建 run 目录（`create_exclusive_dir`，已存在即报错）→ 写 `experiment.json` →
  依次跑 8 个阶段 → 汇总。
- `--mode <stage> --run <dir>`：对已存在的 run 目录单独重跑某一阶段，便于排错。
- `--stop-after <stage>`：跑到指定阶段后停下。

---

## 5. 关键设计决策（为什么这样做）

| 决策 | 原因 |
| --- | --- |
| 不给 `head_rgbd.cli` 加子命令 | `tests/test_contracts.py` 断言 `headrgbd` 的命令集合恰为 6 个，加子命令会破坏契约 |
| 共用一份 prepared 观测、两个 sequence 文件 | `fit_c --front-anchor` 要求 sha 完全一致；`--front-only` 自己按 `view=='front'` 过滤 |
| 把 `rows[*]['segment']` 改写成 `angle_group` 再传给 `solve_field` | `burst_weights` 按 `segment` 分组；沿用原 segment 会把整段转头当成"一个视角"归一化 |
| 设 `view_balance=True` 且 `front_boost=1.0` | 让每组总权重相等，而不是让 `front` 帧拿 3 倍权重 |
| 联合形状拟合自己实现（`turn_shape`） | `fit_c` 的 joint 阶段是逐帧等权，无法按角度组平衡 |
| 调 `fit_c` 时 `--keyframes-per-segment = max_keyframes` | 选帧已由本模块决定，不能被"每段最多 3 帧"二次截断 |
| 只写 `references={'closed': ...}` | 本实验全是闭口帧；写 `front_anchor` 之类的键会改变 `motion_v2` 的 policy 语义 |
| `FrameSequence` 只缓存一个 chunk | 现有 npz 是堆叠数组，整块解码；长视频建议改用逐帧 npz 或分块，内存随最大 chunk 而非总长 |

---

## 6. 配置与运行

配置：`config/turn_experiment.json`（schema `head_rgbd.turn_experiment.config.v1`）。
分节：`experiment / flame / input / tracking(+quality) / keyframes / front_init / joint_shape /
pose_correction / field / delivery`。`TurnConfig.validate()` 会拒绝越界值，例如
`1 ≤ max_keyframes ≤ 12`、`0 ≤ repeat_observations_per_angle ≤ 4`、
`front_yaw_deg ≤ anchor_max_yaw_deg ≤ max_yaw_deg`、
`0 < displacement_bound_mm ≤ 3`、`0 < loop_target_blend ≤ 1`、`field.points_per_frame ≥ 500`。

### Windows PowerShell 现场采集与运行

使用独立采集入口录制正面锚点和连续闭口转头，不改变生产链的四姿态采集方式。
预览窗口中每段按一次空格，倒计时后录制；ESC 取消并保留 partial 清单。
锚点默认 5 帧，转头默认 100 帧、间隔 0.12 秒，按提示缓慢完成左→右→正面。

```powershell
Set-Location -LiteralPath 'D:\proiect\facecap_airway_delivery'
& '.\runtime\env\python.exe' -m head_rgbd.experiment.turn_capture `
    --output '.\runtime\captures\my_continuous_turn'

& '.\runtime\env\python.exe' -m head_rgbd.experiment.turn_run `
    --anchor '.\runtime\captures\my_continuous_turn\anchor_session.json' `
    --turn '.\runtime\captures\my_continuous_turn\turn_session.json' `
    --config '.\config\turn_experiment.json' `
    --device cuda
```

采集目录必须尚不存在。两份角色清单分别指向锚点和转头片段，避免未标注固定视角的转头帧进入锚点输入。
完整运行后，可执行 `runtime/env/python.exe tools/verify_turn_run.py --run <本次结果目录>` 验收产物。

### 使用已有采集的示例命令

```bash
cd /d/proiect/facecap_airway_delivery

runtime/env/python.exe -m head_rgbd.experiment.turn_run \
  --anchor runtime/captures/liguanghao_cap_20261102 \
  --turn   runtime/captures/liguanghao_cap_20261102 \
  --config config/turn_experiment.json \
  --device cuda
```

`--anchor` 与 `--turn` 可以指向同一个 `session.json`/目录（角色不同，选帧时各自按 `view`
与 `role` 取用），也可以分别指向两次采集。常用变体：

```bash
# 只看跟踪与选帧，不拟合（排查输入是否可用）
... --stop-after selection

# 覆盖检测频率与关键帧预算
... --detect-stride 4 --max-keyframes 7

# 已存在 run 目录，单独重跑某一阶段
... --mode field --run runtime/turn_experiment/<run 名>
```

### 输出目录

```
runtime/turn_experiment/<stem>_turn_<YYYYmmdd_HHMMSS>/
├── experiment.json              # 配置快照、输入来源与 sha、阶段顺序、交付清单
├── tracking/{anchor,turn}.{json,npz}, summary.json
├── selection/keyframes.json     # 角度/质量/分组/入选原因/覆盖/被拒原因
├── prepared/sequence.json, anchor_sequence.json, <frame_id>.npz, prepare_summary.json
├── front_anchor/                # fit_c 产物：parameters.npz, report.json
├── joint_shape/                 # parameters.npz, report.json, reference_geometry.json
├── pose_correction/             # motion_v2 产物
├── field/                       # final_vertices.npz, final_mesh.obj, report.json
├── audit/                       # run_report.json + logs/NN_<stage>.log
└── delivery/                    # 只有 6 个文件
    ├── keyframe_selection.json
    ├── init_mesh.obj            # 正面初始化网格（参考帧）
    ├── joint_mesh.obj           # 联合拟合网格（参考帧）
    ├── final_mesh.obj           # 最终平滑网格（参考帧）
    ├── experiment_config.json   # 实际生效配置 + 覆盖项 + 输入信息
    └── stages.json              # 各阶段耗时、有效观测统计、待观察指标
```

三个网格都写在**参考帧的相机坐标**下，各自带该阶段的模型姿态，因此可比但不做配准；
`stages.json.coordinate_note` 会明确写出这一点。

---

## 7. 后续测试关注的指标（以及在哪里看）

| 指标 | 位置 | 读法 |
| --- | --- | --- |
| 重建耗时 | `delivery/stages.json → stage_elapsed_s / total_elapsed_s` | 跟踪与关键帧选择仍近似随视频帧数增长；准备和拟合成本随“关键帧 + 支持帧”数增长 |
| 各视角深度一致性 | `delivery/stages.json → observation_stats.metrics.depth_consistency`；另有 `field/report.json` 逐帧明细 | 逐帧 `median_mm` / `p90_mm` / `coverage`，以及 `by_view`、`by_angle_group` 汇总；侧面 `median_mm` 明显高于正面属预期 |
| 几何变化 | `field/report.json → metrics.side_geometry_change_from_joint.components` | 分开读控制场贡献、Loop 贡献和总变化；不要把它当精度 |
| 表面光滑性 | `metrics.smoothness`（位移场梯度分位数，mm/mm）与 `field/report.json → subdivision.loop_blend` | 目标 `blend = 0.5`；`fallback_reason` 非空说明触发了几何安全降级 |

---

## 8. 复用与改动清单

**新增文件**

| 文件 | 职责 |
| --- | --- |
| `config/turn_experiment.json` | 独立实验配置 |
| `head_rgbd/experiment/__init__.py` | 实验标识与 schema 常量 |
| `head_rgbd/experiment/turn_config.py` | 冻结 dataclass 分节 + 校验 + 覆盖构造 |
| `head_rgbd/experiment/turn_inputs.py` | 输入识别、RGB-D 契约校验、按需解码的 `FrameSequence` |
| `head_rgbd/experiment/turn_capture.py` | 独立现场采集正面锚点 + 连续闭口转头，生成角色清单 |
| `head_rgbd/experiment/turn_tracking.py` | 轻量连续跟踪与连续性分组 |
| `head_rgbd/experiment/turn_keyframes.py` | 角度分桶 + 质量选帧（纯 NumPy） |
| `head_rgbd/experiment/turn_prepare.py` | 入选帧观测重算与序列文件生成 |
| `head_rgbd/experiment/turn_shape.py` | 共享形状联合拟合 |
| `head_rgbd/experiment/turn_field.py` | 一次场求解 + Loop 混合 + 指标 |
| `head_rgbd/experiment/turn_run.py` | CLI 编排、阶段执行、交付汇总 |
| `tests/test_turn_experiment.py` | 24 条纯 NumPy 契约测试 |
| `tests/test_turn_capture.py`、`tests/test_turn_prepare.py`、`tests/test_turn_shape_masks.py`、`tests/test_turn_runtime.py` | 本次现场运行发现的问题对应的 12 项回归测试 |
| `tools/verify_turn_run.py` | 完整运行的文件、哈希、阶段退出码和网格验收 |
| `docs/TURN_ANCHOR_EXPERIMENT.md` | 本文档 |
| `docs/HANDOFF_TURN_EXPERIMENT.md` | 交接说明（完成度、红线、剩余排错路线） |

**复用（未修改）的现有接口**

`core.integrated_fitting.make_flame` / `core.fit_options.FitOptions`；`head_rgbd.prepare`（`backproject`、`rigid`）；
`core.face_test.detect_face_bbox`、`core.mask_detection.get_face_data_single`、`third_party/tufa`；
`head_rgbd.fit_c`（正面初始化，作为子进程）；`head_rgbd.motion_v2`（一次姿态修正，作为子进程）；
`head_rgbd.geometry_c`（`ray_surface`、`sample_depth`、`landmark_validity`）；
`head_rgbd.geometry_d`（`Settings`、`solve_field`、`subdivide`、`anatomical_exclusion`、`edges_of`）；
`head_rgbd.surface_v2`（`burst_weights`、`local_depth_planes`、`loop_mapping`、`loop_support_exclusion`、`safe_loop_base`）；
`head_rgbd.hierarchy_d.safety`；`head_rgbd.paths`（`resolve_runtime_dir`、`configure_environment`、`create_exclusive_dir`）；
`head_rgbd.reporting`（`StageRunner`、`write_json`）；`head_rgbd.pipeline.asset_gate`；
`head_rgbd.reference_contract`（`inner_lip_gap_mm`、`observed_landmarks_m`）。

**必要改动**

- 现有生产链与公共函数：**零改动**（`cli.py`、`pipeline.py`、`fit_*.py`、`geometry_*.py` 均未修改）。
- 初次实现为新增实验文件及 README 指针；本次真实运行修复集中在实验模块 `turn_run.py`、`turn_prepare.py`、`turn_shape.py`，并补充独立采集、验收和测试。

---

## 9. 已知限制与需要实测验证的部分

**已验证（2026-11-02）**：完整测试集 99 passed / 1 skipped；现场新采集的 105 帧通过输入检查，
选出 7 帧、覆盖 −29.6801° 至 +27.4955°，`coverage.missing=[]`。完整 8 阶段连续运行退出码均为 0；
`fit_c` 正面初始化、共享形状拟合、`motion_v2` 和一次场求解均被实际执行；Loop 系数为 0.5，无安全回退。
交付恰有 6 文件，三张网格均重新加载通过，序列/输入/配置/源码绑定验收通过。

**仍需另行验证**：其他人的数据、不同距离/光照/转头范围、完整 CPU 路径、相对于独立几何真值的误差，
以及与生产链的速度/精度比较。三张网格保留各阶段自身姿态，并未额外做跨阶段刚体配准。
同帧深度残差和光滑性诊断的本次数字见 [验证报告](TURN_EXPERIMENT_VALIDATION_20261102.md)。

**本模块不宣称**：速度提升、精度提升、或与生产链相比的优劣。这些都必须在真实数据上测过才能谈。
