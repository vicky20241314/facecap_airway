# 连续转头 RGB-D 算法修正与验证报告（2026-11-04）

本次修改只涉及连续转头实验算法、配置默认值和评测工具。GUI、`headrgbd.cmd`、
`headrgbd_turn.cmd`、采集交互以及交付目录的 6 文件约定保持兼容。

## 1. 结论

此前诊断仍成立，并已完成修正：

- 逐点可靠权重此前会在标量鲁棒损失之后才乘入，实际不能改变点间相对作用；现已改为逐点
  Smooth-L1 后按有效权重质量归一化。
- 单纯按角度桶保留一个关键帧会丢弃大量重复观测；现保留关键帧作为姿态和覆盖锚点，同时把
  同角度附近的可靠帧作为支持观测加入联合拟合。
- 可见性掩码曾可能跨迭代累积取交集；现每轮从不可变基础掩码和当前自遮挡结果重算，并且
  即使本轮深度对应点不足也会先刷新，避免沿用上一轮失效的跟踪状态。
- `change_from_joint` 等几何变化统计此前没有稳定对应到正确的中间表面；现分别在每帧同一
  坐标系、同一姿态和同一顶点拓扑下比较共享形状、控制场和 Loop 结果。
- 静态单帧结果会推动法向控制场触及 `2.5 mm` 位移上限，迫使 Loop 系数降到
  `3.0517578125e-05`；重复观测组在当前安全边界不放宽的情况下，最大场位移降到
  `2.07603 mm`，Loop 系数保持目标值 `0.5`，无安全回退。

精度结论需要分开表述：

- 稳定性明显改善：控制场不再触顶，Loop 不再因总位移越界而几乎关闭。
- 同角度未参与拟合的留出帧有小幅改善：7/7 帧的残差中位数都下降。
- 7 个共同训练关键帧的残差略差，因此不能宣称所有输入帧或所有场景都更准。
- 当前没有独立扫描、人工三维真值或标定式外部位姿，不能把以下同采集评测外推为通用精度。

## 2. 算法修改

### 2.1 逐点可靠权重

`head_rgbd/experiment/turn_shape.py` 的 `_weighted_robust` 现在：

1. 先检查残差和权重形状一致。
2. 只保留有限且为正的权重。
3. 对每个有效点计算 `smooth_l1`，不先做标量归约。
4. 按有效权重之和归一化。
5. 空数组、全零权重和非有限权重返回可反向传播的零损失，不产生 `NaN`。

深度对应点进入模型前还会经过光线搜索、深度门限、局部平面门限、法向一致性、入射角、
局部散度和自遮挡检查。没有足够对应点时，该帧不提交深度约束，避免空集合或伪造零深度。

### 2.2 重复观测

当前策略：

- 每个角度桶保留一个关键帧，关键帧负责姿态、覆盖和正面锚点。
- 每个关键帧最多保留 2 个同角度附近、质量达标的支持帧。
- 支持帧与关键帧的 yaw 差不超过 `6°`，且不占用 `max_keyframes` 预算。
- 每个角度组总权重固定为 `1/G`。同组内一个关键帧和多个支持帧共同分摊该权重，因此正面
  或任一侧面的大量帧不能靠数量扩大影响。
- 每帧独立初始化并更新刚性姿态，不直接平均未配准深度图。
- 深度约束按配准残差、局部平面一致性和入射角逐点加权，并在共享形状联合优化中重刷对应点。

本次实际运行使用 7 个角度组，静态组有 7 个观测；重复观测组有 7 个关键帧和 14 个支持帧，
共 21 个观测。两组每个角度组的总权重均为 `0.142857`。

### 2.3 姿态、可见性和身份约束

- 正面锚点继续初始化身份形状、尺度、五官形态和正面姿态。
- 其余帧使用逐帧 landmark 刚体对齐；共享形状与逐帧刚性姿态联合更新。
- 表情和下颌仅允许在正面锚点附近的有限范围内变化，并保留对正面形状的先验。
- 每次刷新对应点时，landmark 自遮挡和有限深度误差重新计算。可见性取
  `不可变基础有效掩码 AND 当前自遮挡/跟踪结果`，不再与历史轮次永久取交集。
- 即使某轮深度点不足，也会更新当前 landmark 可见性，避免旧状态影响后续损失。

### 2.4 几何变化指标

`head_rgbd/experiment/turn_field.py` 现在分别报告：

- `shared_shape_to_control`：控制场相对共享形状的位移；
- `control_to_loop`：Loop 细分结果相对控制场的位移；
- `shared_shape_to_final_total`：最终表面相对共享形状的总位移。

三个量都来自同一帧、同一坐标系、同一姿态和同一顶点对应关系，单位是毫米。几何变化量仅
描述表面改动，不是精度分数。旧字段 `change_from_joint_*` 保留兼容，现绑定到
`shared_shape_to_final_total` 的真实基准。平滑度统计也改为引用共享形状的边，修复了原实现
可能触发未定义变量错误的问题。

### 2.5 几何细化边界

未改动或放宽以下安全约束：

- 总位移上限 `2.5 mm`；
- 最大斜率 `0.45`；
- 最大法向变化 `35°`；
- 最小有向面积比 `0.05`；
- 表情、下颌、解剖排除和防翻面保护；
- 最终网格拓扑仍为 `20,018` 顶点 / `39,904` 面。

因此本次稳定性改善来自更可靠的输入观测和约束修正，不是通过增加顶点、放宽边界或提高
平滑强度获得。

## 3. 端到端运行

测试数据为：

- 正面锚点：5 帧；
- 连续转头：100 帧；
- 来源：`runtime/captures/turn_e2e_live_20261102`；
- 设备：NVIDIA GeForce RTX 3060 Laptop GPU，CUDA 可用。

两组均用当前代码重新完成 8 个阶段：

`tracking → keyframe_selection → prepared_keyframes → front_anchor → joint_shared_identity → pose_correction → control_field → delivery`

| 项目 | 静态对照 | 重复观测 |
| --- | ---: | ---: |
| 关键帧 | 7 | 7 |
| 支持帧 | 0 | 14 |
| 总观测 | 7 | 21 |
| 阶段完成 | 8/8 | 8/8 |
| 主进程退出码 | 0 | 0 |
| 总耗时 | 122.874 s | 163.023 s |
| 产物验收退出码 | 0 | 0 |
| 验收错误 | 0 | 0 |
| 最大场位移 | 2.49999 mm | 2.07603 mm |
| Loop 系数 | 3.0517578125e-05 | 0.5 |
| Loop 安全回退 | 总位移越界 | 无 |
| 最终网格 | 20,018 / 39,904 | 20,018 / 39,904 |

静态运行：

- `runtime/validation/turn_algorithm_20261104/static_run/audit/run_report.json`
- `runtime/validation/turn_algorithm_20261104/static_run/delivery/`

重复观测运行：

- `runtime/validation/turn_algorithm_20261104/video_run/audit/run_report.json`
- `runtime/validation/turn_algorithm_20261104/video_run/delivery/`

两次运行的一键入口兼容性检查也通过：runtime Python、7 项资产、PyQt6 和 CUDA 均可用。

## 4. 评测结果

完整机器可读结果：

- `runtime/validation/turn_algorithm_20261104/comparison.json`

评测使用共同关键帧、逐帧固定 seed 和同一批像素。共同训练关键帧是拟合输入，因此其残差只
作为训练数据一致性诊断；真正的比较重点是未进入重复观测组拟合的同角度留出帧。

### 4.1 共同训练关键帧

| 指标 | 静态 | 重复观测 | 重复 − 静态 |
| --- | ---: | ---: | ---: |
| 帧中位数 | 0.479 mm | 0.523 mm | +0.044 mm |
| 平均 p90 | 1.659 mm | 1.764 mm | +0.105 mm |
| 最差帧中位数 | 0.551 mm | 0.572 mm | +0.021 mm |
| 平均覆盖率 | 99.65% | 99.60% | −0.05 个百分点 |

重复观测组在这 7 个训练关键帧上略差。不能忽略或改写这一结果，也不能仅凭留出帧改善宣称
全面精度提升。

### 4.2 同角度留出帧

每个角度组选择 1 个未参与拟合的可靠重复帧，两组使用完全相同的相对位姿初始化和 5 步 ICP：

| 指标 | 静态 | 重复观测 | 相对改善 |
| --- | ---: | ---: | ---: |
| 帧中位数（跨帧中位数） | 0.763 mm | 0.723 mm | 0.040 mm（5.2%） |
| 平均 p90 | 2.011 mm | 1.953 mm | 0.058 mm（2.9%） |
| 最差帧中位数 | 0.850 mm | 0.839 mm | 0.011 mm |
| 平均覆盖率 | 99.92% | 99.92% | 基本不变 |
| 改善帧数 | — | 7/7 | 全部改善 |

这组结果说明，在相同采集、相同角度但未参与拟合的帧上，多帧观测产生了可重复的小幅收益。
它仍不是独立扫描真值，样本量也只有 7 个角度组。

### 4.3 几何变化

下表为共同关键帧中各几何变化量的帧间 p90 均值。它们只描述几何改动，不等于误差：

| 几何变化 | 静态 | 重复观测 |
| --- | ---: | ---: |
| 共享形状 → 控制场 | 0.337 mm | 0.298 mm |
| 控制场 → Loop | 0.000 mm | 0.092 mm |
| 共享形状 → 最终表面 | 0.337 mm | 0.341 mm |

静态组的 Loop 几乎被安全回退关闭，因此“控制场 → Loop”接近零；重复观测组能安全保留
目标 Loop 系数，Loop 数值贡献为 `0.092 mm`。这证明 Loop 与几何变化分开统计后，能够
区分“没有变化”和“变化被安全策略关闭”。

## 5. 测试覆盖

未发现缺失的必要代码文件、模块、依赖或配置，因此没有新增外部依赖。

本次最终验证结果：

- 语法/导入编译检查：退出码 `0`；
- 全仓库测试：`173 passed, 1 skipped, 2 subtests passed`，耗时 `24.65 s`；
- 一键入口环境检查：退出码 `0`；
- 静态端到端：退出码 `0`；
- 重复观测端到端：退出码 `0`；
- 两个运行目录的产物验收：均为退出码 `0` 且 `errors=[]`。

测试覆盖包括：

- 重复帧选择、关键帧预算与支持帧预算相互独立；
- 同角度组总权重上限、角度平衡和姿态修正权重传递；
- 逐点鲁棒损失在标量归约前乘权；
- 零权重、非有限权重和空数组不产生 `NaN`；
- 入射角、局部平面一致性和局部散度的可靠权重；
- 可见性从不可变基础掩码刷新，不跨轮次永久累积；
- 共享形状、控制场和 Loop 三种几何变化的正确基准；
- 平滑度统计和姿态修正兼容旧数据；
- 共同关键帧、固定像素、物理帧身份和留出帧汇总；
- 支持帧进入 `prepared/sequence.json`，正面初始化仍只读取
  `prepared/anchor_sequence.json`；
- 8 阶段执行、阶段日志、序列哈希、输入哈希、三张网格和交付 6 文件完整性。

## 6. 复现命令

以下命令均在项目根目录 PowerShell 中执行。

语法和全量测试：

```powershell
& '.\runtime\env\python.exe' -m py_compile (Get-ChildItem -LiteralPath '.\head_rgbd\experiment' -Filter '*.py' | ForEach-Object { $_.FullName })
& '.\runtime\env\python.exe' -m pytest tests -q --disable-warnings --maxfail=1
```

静态对照：

```powershell
& '.\runtime\env\python.exe' -m head_rgbd.experiment.turn_run --anchor '.\runtime\captures\turn_e2e_live_20261102\anchor_session.json' --turn '.\runtime\captures\turn_e2e_live_20261102\turn_session.json' --config '.\runtime\validation\turn_algorithm_20261103\static_config.json' --device cuda --output '.\runtime\validation\turn_algorithm_20261104\static_run'
```

重复观测：

```powershell
& '.\runtime\env\python.exe' -m head_rgbd.experiment.turn_run --anchor '.\runtime\captures\turn_e2e_live_20261102\anchor_session.json' --turn '.\runtime\captures\turn_e2e_live_20261102\turn_session.json' --config '.\config\turn_experiment.json' --device cuda --output '.\runtime\validation\turn_algorithm_20261104\video_run'
```

验收和比较：

```powershell
& '.\runtime\env\python.exe' '.\tools\verify_turn_run.py' --run '.\runtime\validation\turn_algorithm_20261104\static_run'
& '.\runtime\env\python.exe' '.\tools\verify_turn_run.py' --run '.\runtime\validation\turn_algorithm_20261104\video_run'
& '.\runtime\env\python.exe' '.\tools\eval_turn_runs.py' --static '.\runtime\validation\turn_algorithm_20261104\static_run' --repeated '.\runtime\validation\turn_algorithm_20261104\video_run' --output '.\runtime\validation\turn_algorithm_20261104\comparison.json' --samples 6000 --seed 20261103
```

日常一键入口：

```powershell
& '.\headrgbd_turn.cmd'
```

无参数双击或执行上述命令会打开 GUI；`.\headrgbd_turn.cmd run` 会采集、重建和验收；
`.\headrgbd_turn.cmd reconstruct --capture <采集目录>` 会复用已有 RGB-D 采集。

## 7. 剩余风险

- 只有一名受试者、一次实际采集和 7 个留出角度组。
- 留出帧与训练帧来自同一次采集，仍可能共享相机偏差、表情和当次身份误差。
- 没有独立扫描或标定式三维真值，不能给出绝对几何误差。
- 重复观测使运行时间从 122.874 秒增加到 163.023 秒，这是稳定性的实际成本。
- 现有安全边界保持不变。后续只有在独立数据证明有收益时，才应重新评估细化支持范围或边界。
