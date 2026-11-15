# ArUco live 一键采集与喉结交付规格

- 日期：2026-11-15
- 状态：已确认（基于 2026-11-15 的 grill 共识）
- 术语：以根目录 [CONTEXT.md](../../CONTEXT.md) 为准
- 下游：`D:\proiect\head_windpipe_regis`（只消费输出，不在本规格内修改）

## 1. 问题陈述

当前 live 一键采集没有任何 ArUco 入口；默认 C 只用 menton + 颈前法向门推断喉结中心。项目里已有 ArUco 贴纸检测、深度绑定和 `neck_landmarks.json`，但检测发生在采集之后，且当前一键 `result/` 没有 `anatomical_landmarks.json` 和 `manifest.json`，`head_windpipe_regis` 无法自动配对，C 的喉结位置也没有使用标记观测。

本次要补齐的是一条可追溯链路：live 采集时确认两态标记有效；采集后自动识别 ID 与物理尺寸，把标记中心映射到受试者坐标并驱动逐态 C 中心；最终在网格同目录写出下游可直接消费的喉结 sidecar 和 manifest，同时保留 C 网格与基础测量网格两套自包含交付。

## 2. 方案概述

1. **采集**：live `run` 在预览中自动检测前视画面的 ArUco，显示 ID、尺寸估计和质量；每个状态结束前，已保存的前视帧至少一帧必须有效。
2. **复检**：采集后从 prepared 前视帧重新检测，preview 只作提示，复检结果是权威记录。
3. **定位**：把标记皮肤估计点从标注帧变换到参考帧，在受试者坐标下计算高度 `t` 与左右角 `θ`；`t` 驱动该状态 C 的喉结先验中心，`θ` 固定为 0，幅度仍由解剖/比例决定。
4. **交付**：`result/final/` 与 `result/measured/` 各自包含标准命名的头模、`anatomical_landmarks.json` 和最小 `manifest.json`；根 `result/` 保留报告、quick_check、render 与诊断。
5. **下游**：`head_windpipe_regis` 指向任一子目录的头模即可自动配对 sidecar/manifest，完成输入校验和配准搜索；D1/D2/间隙等几何结论如实 pass/fail。
6. **范围**：只有 live `run` 强制两态标记；`reconstruct`、`rebuild --capture`、`rebuild --dense` 无标记时可回退 C，但结果标 `not_usable`、不写可配对 sidecar。

## 3. 接缝（seams）

优先复用已有接缝；测试靶心放在这些稳定接口上，不绑死内部实现。

| 接缝 | 位置 | 职责 | 测试 |
|---|---|---|---|
| 采集帧证据 | `head_rgbd/capture_session.py`、`head_rgbd/aruco_marker.py` | 预览检测、状态结束门禁；输入 RGB/depth/intrinsics，输出每状态标记记录 | 用保存帧/合成帧驱动，不需要真机 |
| 标记观测 | `head_rgbd/tools/neck_landmark_aruco.py` | 扫描 prepared 前视帧，自动 ID/尺寸，写每状态观测 | 对 `zp_capwitharuco` 回放 |
| 标注→参考变换 | `head_rgbd/neck_landmark.py` | `resolve_reference_transform` + `apply_transform` | 同帧/跨帧/失败三类 |
| C 中心 | `head_rgbd/neck_anatomy.py`、`tools/build_neck_direct_v2.py` | 逐态 marker `t` 覆盖先验中心，`θ=0`，幅度不变 | 有标记/无标记/位置超界 |
| 交付组装 | `head_rgbd/rebuild.py`、`head_rgbd/cli.py` | 生成 `final/`、`measured/`、sidecar、manifest、报告 | 端到端回放 |
| 下游契约 | `head_windpipe_regis/src/input_resolution.py`、`landmark_policy.py` | 自动配对、字段/状态/帧/单位校验 | 直接运行下游 |

## 4. 用户故事

### U1 打印/准备标记
现场可以获得 ArUco 打印资产；贴纸可来自现有库存，尺寸和 ID 不必预先固定。
**验收**：`python -m head_rgbd.aruco_marker --output <dir>` 能生成可打印 PNG/SVG/打印页；文档说明贴纸中心贴在可触及的喉结体表位置，不是内部甲状软骨测量。

### U2 预览自动检测
live `run` 的 front 预览自动检测 ArUco，不要求操作者输入 ID 或尺寸。
**验收**：画面显示检测到的 ID、黑方块尺寸估计、质量状态；无标记时显示明确原因，不显示伪造坐标；镜像预览不能改变检测所用的原始帧。

### U3 自动识别 ID 和尺寸
检测颈部 ROI 内的 ArUco，自动读取 ID，并根据深度和四角三维边长估计黑方块尺寸；不假定 16 mm。
**验收**：任意 `DICT_4X4_50` ID 均可识别；记录 `black_square_mm`、`size_source=measured_from_depth`、`size_uncertainty_mm`；尺寸在 8–40 mm 外、少于 3 个有效角点或估计不稳定时判无效；多个不同 ID 同时出现判 `multiple_marker_ids`，不按检测顺序选。

### U4 状态结束门禁
每个状态（closed/open）采集结束前，检查该状态已保存的 front 帧。
**验收**：至少一帧通过质量+唯一性+尺寸校验，才允许完成；不通过时人工模式提示重拍，`--auto` 模式中止；显式 `--allow-marker-fallback` 才允许继续，并标记 `not_usable`。

### U5 采集后权威复检
prepared 生成后，对每个状态的前视帧重新检测，preview 不作为最终记录。
**验收**：复检结果与 preview 冲突时以复检为准；每状态选择质量最好的有效帧（清晰度优先），记录所有尝试和失败原因；没有有效帧则按 U4/U9 处理。

### U6 标注帧到参考帧
把标记观测从标注帧变换到该状态 dense 参考帧。
**验收**：使用现有 `resolve_reference_transform`；记录 `status/method/direction/rotation_3x3/translation_m/applies_to`；同帧用 identity，跨帧用已有 neck-region 光流链；无有效变换时 live 默认失败，不能把标注帧坐标冒充参考帧坐标。

### U7 颈前位置校验
标记必须落在该受试者的颈前合理范围。
**验收**：用受试者轴线、颈前法向门和 t 范围检查；要求 `0.35 <= t <= t_gate + 0.025`（`t_gate` 的 0.025 量化容差）、`|θ| <= 20°`，最近表面法向通过现有 gate；超出范围或落在下颌/锁骨/肩部时判无效，不用错误位置驱动 C。

### U8 逐态驱动 C 中心
closed 用 closed 标记，open 用 open 标记；只取高度 `t`，`θ` 固定 0。
**验收**：每个状态的 `t_center_used = t_marker`；幅度仍为 `larynx_prior_ratio × jaw_reference`，不因标记改变；larynx gain 的轴向中心同步到该状态 `t_marker`；先验窗在颈前范围内裁剪，越界部分不进入 C；`build_direct.json` 记录标记来源、ID、尺寸、`t_marker`、`t_center_used`、`theta_used_deg=0`、`t_gate`。

### U9 标记优先，验收如实
标记驱动 C 后，如果折痕/连接验收超限，不自动回退 menton。
**验收**：C 仍使用标记中心；`quick_check`/`REBUILD_REPORT.md` 如实标 `connection=false` 等；网格和 sidecar 仍交付；不为了通过而压平受保护区域或改标记。

### U10 双套自包含交付
结果同时生成最终 C 网格和基础测量网格两套目录。
**验收**：`result/final/` 含 `head_closed.obj`、`head_open.obj`、`anatomical_landmarks.json`、`manifest.json`；`result/measured/` 同样含四件；根 `result/` 保留现有报告、quick_check、render、neck 子网格；两套同名文件在不同目录，互不覆盖。

### U11 喉结 sidecar
每个子目录写出 `anatomical_landmarks.json`，字段满足下游 v2 策略。
**验收**：见 §6.2；`source=aruco_marker`，包含 `point_observed_m`、`point_reference_m`、`skin_estimate_reference_m`、`skin_estimate`、`transform`、`reference_frame`、`reference_coordinate_system`、`main_delivery_mesh_file`；不伪造 `mesh_binding.point_mesh_m` 或主头模三角绑定；`main_head_mesh_binding.status=not_completed` 并写明原因。

### U12 最小 manifest
每个子目录写出最小自包含 `manifest.json`，用于下游状态配对。
**验收**：见 §6.3；`mesh.states.{closed,open}.path` 与 sidecar `main_delivery_mesh_file` 同名；`reference_frame` 与 sidecar 一致；`anatomical_landmarks.schema` 与 sidecar schema 一致；不复制基础 delivery 中已失效的颈部外观字段。

### U13 下游自动配对
`head_windpipe_regis` 指向 `final/` 或 `measured/` 的头模时，自动找到同目录 sidecar 和 manifest。
**验收**：`closed` 头模只配对 closed 记录，`open` 只配对 open；能通过字段/状态/帧/单位校验；不按文件名猜测；`head_completeness` 通过；把 `final/head_closed.obj` 和 `measured/head_closed.obj` 分别作为输入都应能解析。

### U14 下游配准可跑，几何如实
下游能完成配准搜索，但本规格不承诺 D1/D2/间隙 pass。
**验收**：运行下游入口后生成 `input_validation.json`、`surface_anchor.json` 等批次记录；`summary.status` 可以是 pass 或 fail；失败时如实记录 D1/D2/间隙/覆盖原因；不为了 pass 改本算法头模或缩小气管。

### U15 离线回退
`reconstruct`、`rebuild --capture`、`rebuild --dense` 没有有效标记时仍可完成 C。
**验收**：使用现有 menton 解剖中心；`result/final/` 与 `result/measured/` 仍可生成，但 `manifest.status=not_usable`、不写可配对 `anatomical_landmarks.json`；旧数据不被强制重采。

### U16 live 失败与显式回退
live `run` 默认两态必须有效。
**验收**：任一状态缺失/无效/变换失败/位置超界时，命令非零退出；不生成 `final/`、`measured/`；保留 capture、prepared、日志和 `marker_validation.json`；只有 `--allow-marker-fallback` 才继续 C，并把结果标 `not_usable`、不写可配对 sidecar。

### U17 报告与 CLI
一键入口输出两套路径和标记状态。
**验收**：stdout 打印 `result/final`、`result/measured`、sidecar 路径和 marker 状态；`REBUILD_REPORT.md` 记录标记 ID/尺寸/t、C 中心来源、下游可用的两个目录、已知几何失败；失败原因可机器读取。

### U18 向后兼容
不破坏现有 C、旧输出和旧入口。
**验收**：根 `result/head_closed.obj`/`head_open.obj` 仍存在（兼容旧脚本），但规范下游路径是子目录；旧输出不被覆盖；无标记离线结果仍走现有 C，不新增必需参数。

## 5. 已定决策

1. **标记角色**：ArUco 是 live 采集的喉结观测源；标记中心同时驱动 C 中心并写入 sidecar，不只是下游锚点。
2. **中心维度**：只取高度 `t`；`θ` 固定 0。左右偏移记录在 sidecar，但不改变 C 的对称喉结脊。
3. **幅度**：不因标记改变；仍是 `larynx_prior_ratio × jaw_reference`，clamp 到现有范围。
4. **检测时点**：预览实时检测 + prepared 后权威复检；前者提示，后者落盘。
5. **门禁**：live `run` 默认两态都必须有效；状态结束前检查；显式 fallback 才允许无效继续。
6. **失败策略**：live 无效默认不出 final/measured；离线可回退 C，但标 `not_usable`、不写可配对 sidecar。
7. **位置校验**：按颈前法向门和 t/θ 范围；超界失败，不钳制。
8. **输出组织**：`final/` 与 `measured/` 两个自包含目录；每个都有 sidecar + 最小 manifest；根 `result/` 只做报告/诊断。
9. **下游验收**：自动配对 + 字段/状态/帧/单位校验 + 配准可跑；几何结论如实，不追求强行 pass。
10. **自动规格**：不要求操作员输入 ID/尺寸；自动识别 `DICT_4X4_50` ID，尺寸由深度+四角三维边长估计并记录不确定度。

## 6. 数据契约

### 6.1 目录

```text
result/
  final/
    head_closed.obj
    head_open.obj
    anatomical_landmarks.json
    manifest.json
  measured/
    head_closed.obj
    head_open.obj
    anatomical_landmarks.json
    manifest.json
  neck_closed.obj            # 现有 C 子网格，保留
  neck_open.obj
  head_closed.obj            # 现有兼容副本，保留；不以根目录做下游配对
  head_open.obj
  base_head_closed.obj
  base_head_open.obj
  REBUILD_REPORT.md
  quick_check.json
  render/
```

`measured/head_closed.obj` 是未做 C 的基础测量网格（来自 base dense）；`final/head_closed.obj` 是 C 重塑结果。两者坐标、单位、参考帧一致，但几何不同；各自 sidecar 使用同一次标记观测。

### 6.2 `anatomical_landmarks.json` 必填字段

顶层：

- `schema`: `head_rgbd.delivery.anatomical_landmarks.v2`
- `created_at`、`software_version`
- `unit`: `m`
- `distance_unit`: `mm`、`depth_unit`: `mm`（不用于坐标）
- `coordinate_convention`: `X_right_Y_up_Z_backward`
- `landmark_name`: `laryngeal_prominence`
- `definition`、`definition_scope`
- `observation_is_manual_truth`: `false`
- `effective_marker_mode`: `aruco_marker`
- `states`: 见下

每个状态 `closed` / `open`：

- `name`、`definition`、`definition_scope`
- `source`: `aruco_marker`
- `state`
- `status`: `observed`（或明确的失败状态；失败状态不写可配对 sidecar）
- `reason`: 成功时 `null`
- `frame_id`、`timestamp`、`pixel_uv`、`image_size`
- `point_observed_m`: 标注帧的贴纸表面观测点
- `coordinate_system`: `annotation frame camera X_right_Y_up_Z_backward; metres`
- `unit`: `m`
- `depth_check`、`quality`
- `reference_frame`
- `reference_coordinate_system`: `reference camera X_right_Y_up_Z_backward`
- `transform`: `status=observed`、`method`、`direction=p_reference_camera = R @ p_annotation_camera + t`、`unit=m`、`applies_to` 含 `neck`、`rotation_3x3`、`translation_m`
- `point_reference_m`
- `skin_estimate`: `status=estimated`、`method=marker_surface_minus_thickness_along_local_marker_normal`、`thickness_mm=0.1`、`is_direct_measurement=false`、`uncertainty_mm=1.5`、`normal_unit`
- `skin_estimate_reference_m`
- `skin_estimate_method`、`skin_estimate_uncertainty_mm`
- `marker`: `marker_type=aruco`、`dictionary=DICT_4X4_50`、`detected_id`、`print`（`black_square_mm` 为深度估计值、`size_source=measured_from_depth`、`size_uncertainty_mm`、`thickness_mm`、`thickness_source`）、`quality`、`attempts`
- `larynx_center`: `t_marker`、`theta_marker_deg`、`t_center_used`、`theta_used_deg=0`、`t_gate`、`source=aruco_marker`
- `main_delivery_mesh_file`: `head_closed.obj` 或 `head_open.obj`
- `mesh_binding`: `status=not_completed`、`reason`（ArUco 皮肤锚点不绑定主头模三角面；下游自行投影 S0）
- `main_head_mesh_binding`: `status=not_completed`、`reason`
- `main_mesh_binding_completed`: `false`

下游策略会优先使用 `skin_estimate_reference_m`；`point_reference_m` 是原始标记表面在参考帧的位置，`point_observed_m` 保留原始标注帧观测；三者分开记录，不互相覆盖。

### 6.3 `manifest.json` 必填字段

- `schema`: `head_rgbd.delivery.manifest.v1`（本算法自有；下游只读字段，不依赖本 schema 名）
- `created_at`、`software`（name/version）
- `status`: `usable` / `not_usable`
- `delivery_role`: `final` / `measured`
- `mesh`: `unit=m`、`coordinate_system=reference camera X_right_Y_up_Z_backward`、`topology_shared=true`、`states`：
  - `closed`: `path=head_closed.obj`、`reference_frame`、`vertices`、`faces`
  - `open`: `path=head_open.obj`、`reference_frame`、`vertices`、`faces`
- `anatomical_landmarks`: `schema`（与 sidecar 一致）、`file=anatomical_landmarks.json`、`states`（每状态 `source`、`status`、`frame_id`、`point_observed_m`、`reference_frame`）
- `provenance`: capture、dense、base、C 变体、marker ID/尺寸估计、sidecar 生成时间

`manifest.mesh.states.path` 与 sidecar `main_delivery_mesh_file` 必须同名；`reference_frame` 必须一致；不复制基础 delivery 的 `neck_prominence`/旧颈部外观字段。

### 6.4 坐标与单位

- 头模、sidecar 坐标均为米；坐标约定 `X_right_Y_up_Z_backward`。
- 标记 `point_observed_m` 是标注帧；`point_reference_m`/`skin_estimate_reference_m` 是 dense 参考帧，和头模顶点同一坐标系。
- 下游 `head_windpipe_regis` 以 `-Unit m` 读取；sidecar `unit=m`。
- 标注帧到参考帧的变换只用于颈部体表观测，不用于气管放置矩阵。

## 7. 测试决策与完成定义

### 7.1 单元

- 自动 ID：任意 `DICT_4X4_50` ID；多个不同 ID 拒绝；同 ID 双份拒绝；错误字典报错。
- 自动尺寸：3/4 个有效角点的三维边长估计；尺寸范围 8–40 mm；记录不确定度；角点无效/跳变时拒绝。
- 门禁：状态结束前至少一帧有效；`--auto` 失败中止；`--allow-marker-fallback` 只改变失败策略，不伪造 sidecar。
- 位置：`t` 边界、`θ` 边界、法向 gate、超界拒绝。
- sidecar：字段/状态/帧/单位/transform 校验；target 的 `resolve_anchor` 选中 `skin_estimate_reference_m`。
- manifest：状态配对、`reference_frame`、schema 一致性。

### 7.2 集成

- 用 `zp_capwitharuco` 的 prepared 帧回放：自动 ID/尺寸、closed/open 观测、transform、sidecar、C 中心；该历史采集的深度尺寸估计约 33.6 mm，不能被固定 16 mm 假设拒绝。
- 无标记的 `jzh_0.5.1` / `liguanghao` 离线路径：仍出 C，`not_usable`，不写可配对 sidecar。
- live 门禁用保存帧模拟；不需要真机即可测试“状态结束前有效/无效/重拍/回退”。
- 现有 C 回归：标记只改中心，不改幅度/拓扑/保护；原有 quick_check 门槛不降低。

### 7.3 下游

- 对 `result/final/` 和 `result/measured/` 分别运行 `head_windpipe_regis`：
  - 自动配对 sidecar + manifest；
  - `input_validation.json` 通过字段/状态/帧/单位校验；
  - 喉结锚点加载为 `skin_estimate_reference_m`；
  - 配准搜索可完成；`summary.status` 允许 pass 或 fail，失败必须是 D1/D2/间隙等真实几何原因。
- 当前 C/measured 头模已通过下游 `head_completeness`（无大型前颈开口）；该检查作为回归门槛。

### 7.4 完成定义

1. live `run` 用有效标记能生成两套自包含交付，下游自动配对并通过输入校验。
2. live `run` 无有效标记默认失败，不产生 final/measured；显式 fallback 只产生 `not_usable`。
3. 自动 ID/尺寸在历史标记帧上工作，不依赖固定 16 mm。
4. C 逐态中心由标记 `t` 决定，`θ=0`；超限如实报告。
5. 旧 C/旧输出不回归；现有测试通过；新增测试覆盖上述接缝。

## 8. 明确不做

- 不支持 `DICT_4X4_50` 之外的 ArUco 字典；QR 标记只保留旧兼容路径，不接入本次 live 门禁。
- 不支持同一画面多个不同 ID 的自动消歧；多标记直接拒绝。
- 不测量贴纸实际厚度；sidecar 记录 0.1 mm 名义值和 1.5 mm 不确定度。
- 不保证 `head_windpipe_regis` 的 D1/D2/间隙 pass；不修改其算法或气管几何来换 pass。
- 不用标记改变 C 的幅度、带宽或 V 形先验形状；只改中心 `t`。
- `--auto` 不自动重采；失败即中止并保留证据。
- 不清理/移除贴纸在 RGB 或纹理中的外观；不重建贴纸下面的皮肤。
- 不给历史无标记数据补造 sidecar；离线回退只标 `not_usable`。
- 不在本次运行完整 `--full-verify`；需要时显式另跑。
- 不把根 `result/` 做成下游自动配对目录；规范下游路径是 `final/` 和 `measured/`。

## 9. 补充说明

- 旧 `zp_capwitharuco` 的 sidecar 标称黑方块 16 mm，但用深度和四角三维边长估计约 33.6 mm；这说明现场实际贴纸尺寸与标称可能不一致，自动尺寸估计必须记录测量值和不确定度，不能假定打印规格。
- 标记是体表贴纸中心，不是甲状软骨内部测量；`skin_estimate_reference_m` 是贴纸表面沿局部法向减去名义厚度后的估计，下游必须保留这个语义。
- 当前 C 在部分数据上已有真实失败（例如 jzh 张口全边集 85.17°、liguanghao 张口 22.16° > 20°）；标记驱动中心后可能变化，但本规格要求如实报告，不为此回退或压平。
- 下游 `head_windpipe_regis` 的 `mesh_binding.point_mesh_m` 对 ArUco 不是首选锚点；本算法不伪造主头模三角绑定，避免把贴纸点说成主网格绑定。
- 本次输出是新目录、新文件；不覆盖旧 `jzh_0.5.1`、`zp_capwitharuco` 等已有结果；未要求不提交 git。
