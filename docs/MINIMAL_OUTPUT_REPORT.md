# 默认仅输出最终网格：实现与验证

日期：2026-11-03。基于 0.5.2 的输出与存储优化，重建算法不变。

## 结果

- 默认运行只在 `delivery` 生成闭口、张口两个最终 OBJ。运动对照、coarse、regularized、插值、平滑基面、支持度着色及独立颈部诊断网格均不再生成。
- 保留参数、参考帧、观测支持度、运行报告和最终几何检查。最终 OBJ 直接从参数导出，写盘后仍核对有限性、索引、米制尺寸和关键点插值。
- 完整公共入口运行成功；四阶段共 88 组参数数组与历史同输入结果逐元素完全一致。最终 OBJ、支持度 JSON、关键点 JSON 逐字节一致。新参数结果和历史 OBJ 结果的重新导出均通过。
- 同批交错测试的 dense 重建加交付阶段，中位数由 **31.657 秒**降为 **23.389 秒**，减少 **8.267 秒（26.1%）**。
- 全部回归：138 passed、1 skipped、2 subtests passed，退出码 0。

## 实现范围

1. `fit_d` 仅保存最终曲面的参考信息和参数，不再逐阶段生成和重新读取中间 OBJ，也不再导出支持度 PLY。
2. 独立颈部观测网格从不反馈最终头部曲面，因此取消这项诊断构建与导出；采集帧中的颈部跟踪记录仍保留。
3. `motion_v2` 不再输出四份运动对照 OBJ，继续保存全部运动参数和验收记录。
4. 默认 minimal 拟合、运动、dense 参数使用未压缩 NPZ，保持全部数组键、类型和值，减少 CPU 压缩时间。
5. `export_delivery` 同时兼容新的参数存储和旧 OBJ 存储；新结果只在最终位置写 OBJ 一次，保留禁止覆盖及写盘后检查。
6. 共享身份、形变正则、迭代预算、Loop 混合量、覆盖与几何安全规则均未更改。

## 同条件计时

输入为 `runtime/outputs/20261103_124548_306894_head_reconstruction/prepared/sequence.json`，初始化复用该次已保存的 `motion_only`。两版使用相同 runtime、设备、输入与原始 dense 参数。

边界：每次启动全新 Python 进程，执行 dense 重建和最终交付导出；不含采集、观测准备、正面拟合、联合身份和运动优化。两版各预热一次，正式按 AB / BA / AB 交错各运行三次，所有样本保留。

| 轮次 | 修改前（秒） | 修改后（秒） |
|---|---:|---:|
| 预热，不计入统计 | 26.099 | 21.490 |
| 1 | 30.763 | 23.418 |
| 2 | 31.657 | 23.389 |
| 3 | 32.284 | 22.914 |

修改前范围 30.763–32.284 秒；修改后 22.914–23.418 秒。预热与正式的 8 次工作进程均退出 0，并通过最终网格与参数一致性检查。

各内部计时块的正式样本中位数如下。各块中位数不应相加当作总中位数，计时也包含机器负载波动。

| 内部计时块 | 修改前（秒） | 修改后（秒） |
|---|---:|---:|
| `front_anchor_low_and_mid` | 11.378 | 10.724 |
| `guarded_regularized` | 0.003 | 0.003 |
| `subdivide_and_loop_base` | 1.340 | 1.236 |
| `dense_front_anchor` | 1.310 | 1.277 |
| `guarded_dense` | 0.000 | 0.000 |
| `evidence` | 2.851 | 2.935 |
| `references_and_exports` | 3.107 | 0.253 |
| `neck` | 1.315 | 0.010 |
| `parameters_save` | 4.099 | 0.074 |

## 公共入口验证

执行 `python -m head_rgbd --runtime <runtime> reconstruct --sequence <prepared/sequence.json> --output <独立验证目录>`，未额外指定模式参数，未启动相机。

整次命令耗时 **149.667 秒**，退出码 0。它包含输入与资产检查、完整拟合、运动、dense 和交付，复用了 prepared 数据。这是一次入口验证，不是完整流程的交错性能基准；不能将前面的 26.1% 当作完整重建提速，也不与历史单次耗时直接比较。

新运行中 OBJ/PLY 文件仅为：

```text
delivery/head_closed_m.obj
delivery/head_open_m.obj
```

输出位于 `runtime/validation/20261103_minimal_output/public_reconstruct/delivery/`。实际配置记录 `mesh_output=delivery_only`、`parameter_archive=stored_npz`，并保留 joint、fast、lean、minimal、240/240/80 预算、中频 1 轮、8 mm 弯曲及安全 Loop 0.5。

## 存储与限制

完整运行的 reconstruction + delivery 占用由历史结果的 81.57 MiB 变为 93.59 MiB；其中网格文件数量由 23 变为 2。
未压缩参数用更多内部存储空间换取更少压缩计算，最终交付网格的内容与大小不变。参数仍保留全部诊断数组，未通过删减数组换取时间。
本次证据支持输出和存储开销下降，没有重新评价或声称几何精度提升。原始采集、模型资产及历史输出均保留。

## 可追溯记录

- `runtime/validation/20261103_minimal_output/comparison.json`：全部计时、范围、工作进程命令及一致性检查。
- `source_hashes.json`、`baseline_source/`、`candidate_source/`：两版源代码快照；两版共享项目原有第三方依赖。
- `public_verification.json`、`public_cli_timing.json`、`public_cli.log`：完整入口、存储格式、实际参数、重新导出与源代码一致性验证。
- `final_summary.json`、`test_result.json`：汇总、存储占用、回归结果。
- `measurements/`：8 次成功的预热和正式输出。
- 验证脚本在正式计时前曾出现括号遗漏和第三方导入路径错误，修正后才开始完整采样；失败预热保留在 `timing/baseline_0/worker.log` 与 `failed_preflight_timings.json`，未计入正式统计。
