# 交接文档：正面锚点 + 连续转头 RGB-D 实验

> **后续更新（2026-11-04）：** 此前的逐点权重、多帧输入、可见性累积和几何变化基准诊断均已复核；其中
> 逐点权重确实会在标量归约后失效，按角度分桶也确实会丢弃同角度重复观测。当前默认配置保留 1 个角度
> 关键帧并增加最多 2 个同角度支持帧，21 个观测按角度等权进入联合拟合；可见性每轮从基础掩码重算；
> `change_from_joint` 已改为共享形状到最终表面的真实比较，并分别报告控制场和 Loop 贡献。
> 当前代码已重新完成静态/重复观测两组 8 阶段运行，产物验收均通过；全仓库测试为
> **173 passed / 1 skipped / 2 subtests passed**。重复观测组的最大场位移由触顶的 `2.49999 mm`
> 降至 `2.07603 mm`，Loop 系数由 `3.0517578125e-05` 恢复到 `0.5`；7/7 个同角度未参与拟合
> 留出帧改善，但 7 个共同训练关键帧残差略差，且没有独立扫描真值。
> 详见 [连续转头算法修正与验证报告](TURN_ALGORITHM_ACCURACY_20261104.md)。
> 原 2026-11-02 更新块及以下正文保留为历史记录。

> **后续更新（2026-11-02）：** 用户已进一步授权现场采集及完整测试。
> 新采集 5 帧正面锚点 + 100 帧连续转头 RGB-D，修复运行时问题后，默认配置完整 8 阶段与产物验收均通过；
> 最终运行 `runtime/turn_experiment/turn_e2e_live_20261102_attempt04`，耗时 122.787 秒，测试集 99 passed / 1 skipped。
> 详见 [完整验证报告](TURN_EXPERIMENT_VALIDATION_20261102.md)。
> 以下正文保留初次交接时的历史记录，其中“真实数据未执行”等状态不再代表当前状态；独立精度与相对性能仍未验证。

> 交接对象：接手的 agent。
> 交接时间：2026-11-02。
> 代码位置：`D:\proiect\facecap_airway_delivery`，模块 `head_rgbd/experiment/`。
> 本文档假设你**没有**看过之前的对话，因此把背景、约束、已完成项、未验证项、以及不能踩的坑都写全。

---

## 0. 三分钟快速上手

```bash
cd /d/proiect/facecap_airway_delivery

# 1) 语法与导入（约 5 秒）
runtime/env/python.exe -m py_compile head_rgbd/experiment/*.py

# 2) 全仓库测试（约 35 秒，应为 86 passed, 2 skipped）
runtime/env/python.exe -m pytest tests/ -q

# 3) 入口自检
runtime/env/python.exe -m head_rgbd.experiment.turn_run --help
```

读三份东西即可掌握全貌：

| 文件 | 内容 |
| --- | --- |
| `docs/TURN_ANCHOR_EXPERIMENT.md` | 算法流程、函数职责、输入契约、示例命令、待观察指标 |
| `config/turn_experiment.json` | 全部可调参数（分 10 节） |
| `head_rgbd/experiment/turn_run.py` 顶部 docstring | 8 个阶段及顺序 |

---

## 1. 用户要的是什么

实现一个**独立的**"正面 RGB-D 锚点 + 连续转头 RGB-D 视频"重建实验算法。用户的原始约束（必须继续遵守）：

1. **只实现代码并说明算法**；不执行数据准备、重建、参数搜索、性能测试。实际重建由用户自己跑。
2. **不破坏现有主链**：保留现有算法、入口、默认参数、输出行为；不给原流程加大量条件分支。
3. 新算法放在**独立实验模块**，单独入口与配置；可复用现有几何模型/相机/损失/求解器，行为变化用新模块封装。
4. **不安装依赖、不下载模型、不自动运行重建**；不构建通用框架。
5. 输入检查必须**明确区分 RGB 视频与同步 RGB-D**；缺深度或缺标定要给明确错误，**不静默退化**。
6. 不引入：逐帧全视频非刚性拟合、TSDF、新神经网络、多轮完整重建交替、高维逐顶点自由形变、为微小指标变化加的求解分支。
7. Loop 与线性混合以 **0.5 为目标**，只由几何安全检查决定是否降低；**不得**因几个深度采样点的微小差异把系数选成 0。
8. **不要宣称**速度或精度提升；必须明确说明"完成到什么程度、哪些要用户实测"。

---

## 2. 当前状态：代码已完成，真实数据未跑过

### 2.1 文件清单（全部为新增，生产链零修改）

| 文件 | 行数 | 职责 |
| --- | ---: | --- |
| `config/turn_experiment.json` | 112 | 独立配置，schema `head_rgbd.turn_experiment.config.v1` |
| `head_rgbd/experiment/__init__.py` | 36 | `EXPERIMENT` / `EXPERIMENT_VERSION` / `CONFIG_SCHEMA` |
| `head_rgbd/experiment/turn_config.py` | 417 | 冻结 dataclass 分节、`load/from_dict/to_dict/replace/validate` |
| `head_rgbd/experiment/turn_inputs.py` | 568 | 输入识别、RGB-D 契约校验、`FrameSequence`（单 chunk 缓存） |
| `head_rgbd/experiment/turn_tracking.py` | 437 | YuNet+TUFA 抽样检测、KLT 传播、yaw/pitch/roll、质量、连续性分组 |
| `head_rgbd/experiment/turn_keyframes.py` | 291 | 角度分桶 + 质量选帧（纯 NumPy，可单测） |
| `head_rgbd/experiment/turn_prepare.py` | 255 | 只为入选帧重算观测；生成 `sequence.json` + `anchor_sequence.json` |
| `head_rgbd/experiment/turn_shape.py` | 515 | 共享形状联合拟合（角度组等权） |
| `head_rgbd/experiment/turn_field.py` | 318 | 一次 `solve_field` + 一次 `safe_loop_base` + 指标 |
| `head_rgbd/experiment/turn_run.py` | 477 | CLI 编排、阶段子进程、交付汇总 |
| `tests/test_turn_experiment.py` | 215 | 24 条纯 NumPy 契约测试 |
| `docs/TURN_ANCHOR_EXPERIMENT.md` | 400 | 算法说明（交付给用户的文档） |
| `docs/HANDOFF_TURN_EXPERIMENT.md` | — | 本交接文档 |
| `README.md` | +13 | 末尾新增 §11 指向实验，纯说明 |

### 2.2 已验证（可以信任）

- 全部模块可导入，无未定义名字（用 `symtable` 遍历所有作用域查过全局自由变量）。
- `turn_run --help` 正常；`TurnConfig` 加载 / JSON 往返 / `replace` 覆盖 / 越界拒绝均正常。
- `tests/` 全绿：**86 passed, 2 skipped**（含既有 `test_contracts.py` 的 21 条，说明未破坏命令集合契约）。
- `tests/test_turn_experiment.py` 24 条覆盖：角度分桶与左右约定、预算不凑数、同角度去冗余、
  `tracking_group` 与 `angle_group` 相互独立、彩色视频被拒、配置越界被拒。

### 2.3 未验证（**这是交接的重点**）

**没有一张真实网格是由本模块产出的。** 所有需要模型、深度数据、GPU 的路径都还没执行过。
因此"能不能跑通"仍是未知数，"效果好不好的"更未测。

---

## 3. 代码地图与数据流

```
--anchor / --turn
   │
   ├─ turn_inputs.plan_source / open_source      ← 只认同步 RGB-D，彩色视频直接抛错
   ▼
turn_tracking.analyze(role)                       → tracking/{anchor,turn}.{json,npz} + summary.json
   ▼
turn_keyframes.select(anchor_rows, turn_rows, cfg) → selection/keyframes.json（≤9 帧，不凑数）
   ▼
turn_prepare.materialise(run_dir, cfg, exp)        → prepared/<id>.npz + sequence.json + anchor_sequence.json
   ▼
head_rgbd.fit_c --front-only（子进程，只用 anchor_sequence.json）
   ▼
turn_shape.run(run_dir, cfg, device)              → joint_shape/parameters.npz（fit_c/motion_v2 兼容）
   ▼
head_rgbd.motion_v2（子进程，读 joint_shape）      → pose_correction/
   ▼
turn_field.run(run_dir, cfg)                      → field/final_vertices.npz, final_mesh.obj, report.json
   ▼
turn_run._stage_delivery                          → delivery/ 6 个文件
```

关键约定：

- 阶段之间只通过文件通信，每个阶段在独立子进程里跑（`StageRunner`），日志在 `audit/logs/NN_<stage>.log`。
- 阶段目录名：`tracking / selection / prepared / front_anchor / joint_shape / pose_correction / field / delivery / audit`。
- `--mode <stage> --run <dir>` 可单跑某阶段；`--stop-after <stage>` 可提前停。

---

## 4. 五条红线（改了就会破坏既有契约或用户要求）

### 4.1 不要给 `head_rgbd.cli` 加子命令

`tests/test_contracts.py` 断言 `headrgbd` 的子命令集合恰为
`{"run","capture","reconstruct","export","device","assets"}`。加子命令会让既有测试红。
入口只能是 `python -m head_rgbd.experiment.turn_run`。

### 4.2 不要试图让 `fit_c --front-anchor` 直接吃联合序列

`fit_c` 里有一句硬校验：锚点报告的 `sequence_sha256` 必须等于当前 `--sequence` 的 sha。
本实验的做法是：**正面初始化在 `prepared/anchor_sequence.json` 上独立跑 `fit_c --front-only`**，
联合拟合在自己的模块里做，并校验同一份文件的 sha。别改成"一个序列跑两次"。

### 4.3 传给 `solve_field` 的 `segment` 必须被改写成 `angle_group`

`surface_v2.burst_weights` 按 `rows[i]['segment']` 分组，并且 `view == 'front'` 会给 3 倍权重。
如果沿用原 segment（= 跟踪连续性分组），整段转头会被当成"一个视角"归一化——这正是用户明确禁止的。

现有写法（`turn_field.py:71`）：

```python
rows_field = [dict(row, segment=str(row.get("angle_group") or "one_group")) for row in rows]
```

配合 `Settings(view_balance=True)`（`burst_weights` 内部 `front_boost=1.0`），得到
"每组总权重相等、组内按帧数均分"。**公共函数一行都没改**，这是刻意设计。

### 4.4 不要"每 segment 最多 3 帧"

`fit_b.choose` 有这个限制，但本实验的选帧完全由 `turn_keyframes.select` 决定。
调用 `fit_c` 时必须显式传 `--keyframes-per-segment <max_keyframes>`（默认 9），否则每个 segment 会被截到 3 帧。

### 4.5 不要把 Loop 混合系数做成"按数据挑 α"

目标固定 0.5，只允许几何安全检查（`hierarchy_d.safety`）触发降级，降级原因写进
`field/report.json → subdivision.loop_blend.fallback_reason`。用户明确禁止用几个采样点的微小差异把系数判成 0。

---

## 5. 剩余工作清单

### P0 — 首次真实运行的分阶段排错（用户跑或你被明确授权跑时才做）

用户说"实际重建由我自行运行"，所以**除非用户明确要求你跑**，不要执行重建。
若要跑，按阶段推进，不要一把梭：

```bash
# 阶段 1-2：只验证输入能不能用、选帧是否合理（不加载 FLAME，最快）
runtime/env/python.exe -m head_rgbd.experiment.turn_run \
  --anchor runtime/captures/liguanghao_cap_20261102 \
  --turn   runtime/captures/liguanghao_cap_20261102 \
  --stop-after selection
```

看 `tracking/summary.json`（可用帧数、yaw 跨度、分组数）与 `selection/keyframes.json`
（每帧角度/质量/入选原因、`coverage.missing`、`rejected` 计数）。

通过后再 `--stop-after prepared`、`--stop-after front_anchor`、`--stop-after joint_shape`、
`--stop-after pose_correction`、`--stop-after field`，逐段核对产物字段。

**验收标准**：每个阶段退出码 0；相邻阶段报告的 sha 互相绑定（`sequence_sha256` 串联）；
delivery 目录恰好 6 个文件；三张网格顶点/面数非零且能被 trimesh 重新加载（`_export` 已内置往返校验）。

### P1 — 四处需要人工确认的调用契约（最可能出运行时错）

按"最可能先炸"排序，每处都给了定位方式和判据：

1. **`turn_field.py:92-98` 用 lil_matrix 覆盖排除行**
   `stencil.rows[vertex] = ...` / `stencil.data[vertex] = ...` 直接改写内部列表。
   本机 scipy 是 **1.11.4**，这个写法在该版本可用（`scipy>=1.14` 才收紧），所以现在不是错误，
   但属于脆弱写法：一旦升级 scipy 就会失效。
   判据：`runtime/env/python.exe -c "import scipy;print(scipy.__version__)"`；
   若升级到 1.14+ 后报错，改用 `stencil[vertex, :] = row` 逐行赋值（更慢但安全）。
   同时注意这里覆盖用的是 `subdivide` 的线性映射 `mapping`，而不是 `loop_mapping` 返回的线性映射（已被丢弃）——
   两者理论一致，但若发现排除区被平滑进去，先查这一点。

2. **`turn_field.py:73` 只调一次 `solve_field`，未传 `initial/locked/soft_lock`**
   默认 `initial=0`、`soft_lock=1.0`。生产链 `fit_d` 是"低/中频两轮 settings"。
   本实验刻意只跑一轮（用户要求"一次低/中频形变"）。
   判据：看 `field/report.json → field.free_set_rule`、`final_safety`、`maximum_absolute_displacement_mm`；
   若 `maximum_absolute_displacement_mm` 贴近 2.5（上界），说明约束在硬顶，需下调 `field.displacement_bound_mm` 或放宽正则。

3. **`turn_shape.py` 的对应刷新参数**（`ray_surface(..., .008, .005)`，每 10 步刷新）
   判据：`joint_shape/report.json → frames[*].depth_correspondences`。
   若大量帧的对应点数为 0，先放宽 `max_depth`/`max_plane`，再考虑加步数；不要反过来先加步数。

4. **`turn_run.py:187-199` 传给 `fit_c` 的参数组合**
   `--front-only --postfit minimal --keyframes-per-segment <9>`。
   `--postfit minimal` 会跳过网格导出，但 `report.json` 的 `references[*].mesh` 仍写有文件名（文件不存在）。
   已确认 `motion_v2` 只用 `references[*]['frame_id']`，不需要那个文件；但**如果以后有人在别处读 `mesh` 字段会踩空**。
   判据：`front_anchor/report.json` 能被 `motion_v2` 正常消费即可。

### P2 — 文档与可观测性（低风险，可在真跑后补）

1. 首次真实运行后，把"实测耗时 / 每个视角深度残差 / 侧面变化量 / 光滑性分位数"填进
   `docs/TURN_ANCHOR_EXPERIMENT.md` 的 §9，并把"未验证"改成"已验证 + 具体数字"。
2. `README.md` §10 的版本号文字（写 `0.4.0`）与 `VERSION`（`0.5.0`）本来就不一致，**属既有问题，不在本任务范围**，
   不要顺手改（会牵动既有文档口径）。
3. 若用户要求"更省时间"，优先看 `tracking` 与 `keyframe_selection` 的耗时占比：
   帧数增加主要增加这两段成本（`detect_stride` 是唯一有效旋钮），**不要**去动拟合阶段。

---

## 6. 需要知道的事实（省得你再翻代码）

| 事实 | 值 / 位置 |
| --- | --- |
| Python | `runtime/env/python.exe`（**必须用它**，系统 python 缺依赖） |
| 依赖 | numpy 1.26.4 / scipy / torch 2.0.1 / opencv / trimesh / smplx / osqp 1.0.5 |
| 坐标约定 | `X_right_Y_up_Z_backward`，内部单位米，深度 `uint16` 毫米 |
| **yaw 符号** | **负 = 画面左侧，正 = 右侧**（实测 left ≈ −15…−16°，right ≈ +18°，front ≈ −3…−6°） |
| 模型 | FLAME，`core.integrated_fitting.make_flame(FitOptions(...), device, batch)` |
| 检测 | `core/face_test.detect_face_bbox`（YuNet）+ `third_party/tufa` 68 点 + UniFace 解析 |
| 现成真实数据 | `runtime/captures/liguanghao_cap_20261102/`（session.json + segment npz） |
| 既有 prepared 样例 | `runtime/outputs/0.5.0/prepared/sequence.json`（可对照 rows 字段） |
| 资产闸门 | `head_rgbd.pipeline.asset_gate(runtime)` |
| 报告工具 | `head_rgbd.reporting.{StageRunner, write_json, require_stage_success}` |
| 配置默认值 | `config/turn_experiment.json`；代码默认值在 `turn_config.py` 各 dataclass |

---

## 7. 配置项速查（改这些就够，不要加新旋钮）

| 节 | 关键项 | 默认 | 作用 |
| --- | --- | --- | --- |
| `experiment` | `device` | `cuda` | `--device` 可覆盖 |
| `flame` | `shape_components` / `expression_components` | 50 / 10 | 表达式刻意压低（闭口强约束） |
| `input` | `max_adjacent_gap_s` | 0.2 | 超时即切断连续性 |
| | `allow_unverified_alignment` | `false` | 打开前必须让用户明确同意 |
| `tracking` | `detect_stride` | 3 | 帧数成本的主要旋钮 |
| | `klt_window` / `klt_levels` | 21 / 3 | 奇数窗口，校验会拒绝偶数 |
| `keyframes` | `max_keyframes` | 9 | 硬预算，`--max-keyframes` 可覆盖 |
| | `angle_bin_deg` | 10 | 角度分桶粒度 |
| | `front_yaw_deg ≤ anchor_max_yaw_deg ≤ max_yaw_deg` | 8/15/55 | 越界即报错 |
| `front_init` | `steps` / `points_per_frame` | 240 / 1200 | 传给 `fit_c` |
| `joint_shape` | `pose_steps` / `joint_steps` | 60 / 180 | 两阶段步数 |
| | `expression_bound` / `jaw_delta_bound_rad` | 0.8 / 0.1 | 闭口约束与下颌界限 |
| `pose_correction` | `steps` | 60 | 传给 `motion_v2` |
| `field` | `points_per_frame` / `outer_steps` | 5000 / 2 | 一次场求解 |
| | `displacement_bound_mm` | 2.5 | 上限 3，校验会拒绝更大值 |
| | `loop_target_blend` | 0.5 | 目标值，不允许设 0 |

---

## 8. 完成定义（DoD）

本次交接的范围**已经完成**的部分：代码、配置、测试、两份文档、README 指针。

判断"整体任务完成"还差一步：**用户自己在真实数据上跑通并接受结果**。
在此之前，任何关于效果/速度/精度的陈述都必须加限定词，交付口径只能是：

> 代码已实现并通过静态与单元测试；真实数据重建未执行，效果需用户实测验证。

---

## 9. 千万不要做的事

1. 不要自动执行重建、参数搜索或性能测试（用户明确保留这一步）。
2. 不要宣称速度或精度提升。
3. 不要修改 `head_rgbd/cli.py`、`pipeline.py`、`fit_*.py`、`geometry_*.py`、`surface_v2.py` 的现有行为。
4. 不要为了"跑通"而在输入检查里放水（缺深度/未对齐必须报错）。
5. 不要给 `solve_field` / `burst_weights` / `safe_loop_base` 加新参数或分支——用 `Settings` 现有开关。
6. 不要把每轮迭代的网格导出（用户只要 init / joint / final 三张 + 选帧清单 + 配置 + 阶段统计）。
