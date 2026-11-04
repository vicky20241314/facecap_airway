# 连续转头 RGB-D 实验运行验证 — 2026-11-02

状态：**已完成；本次真实数据端到端运行及产物验收通过。**

## 测试数据与环境

本次在用户明确授权并配合下，使用 Orbbec Gemini 215 现场采集新数据，未采用历史分段采集或合成观测替代真实输入。

- 数据：`D:\proiect\facecap_airway_delivery\runtime\captures\turn_e2e_live_20261102`。
- 正面闭口锚点 5 帧，连续闭口转头 100 帧；转头片段实际 13.36 秒。
- RGB 为 `1280×720 uint8`；深度为 `1280×720 uint16` 毫米；每帧携带彩色相机内参和时间戳。
- 所有 105 帧通过输入读取与 RGB-D 契约检查；最大相邻间隔：锚点 0.141 秒、转头 0.172 秒，均小于默认 0.2 秒阈值。
- 对齐依据为相机硬件 D2C 和逐帧尺寸、内参检查；没有做独立物理标定。
- Python 3.10.22、PyTorch 2.0.1、SciPy 1.11.4，NVIDIA GeForce RTX 3060 Laptop GPU / CUDA。
- 7 个模型资产均通过 SHA-256 校验；没有发现运行所需依赖、模型或默认配置缺失，本次无需安装依赖或下载模型。
- 使用 `config/turn_experiment.json` 默认参数，设备为 CUDA；没有降低优化步数或放宽输入、质量、安全检查来取得通过结果。

## 发现的问题与补充

| 文件 | 问题 / 补充内容 | 验证 |
| --- | --- | --- |
| `head_rgbd/experiment/turn_capture.py`（新增） | 现有生产采集入口要求四个固定姿态，缺少连续闭口转头采集方式。新增独立两段采集，复用现有相机驱动、预览和 NPZ 保存格式；生成独立锚点/转头清单，带格式、内参检查和无帧超时处理 | 已现场采集 5+100 帧；7 项采集契约测试通过 |
| `head_rgbd/experiment/turn_run.py` | Windows GBK 标准输出无法转发某些 Unicode 日志，首轮抛出 `UnicodeEncodeError`。统一实验入口和子进程的 UTF-8 编码 | GBK 环境下真实子进程回归测试通过；最终完整运行日志正常 |
| `head_rgbd/experiment/turn_prepare.py` | 将输入溯源结构误读为 `inputs[role].path`，实际路径为 `inputs[role].source.path`；来源清单也有相同问题 | 修复两处读取和来源哈希生成；实际 prepared 输出及回归测试通过 |
| `head_rgbd/experiment/turn_shape.py` | 布尔关键点掩码和可见性掩码被转成浮点张量，索引报错；内唇掩码与排除内唇的稳定点掩码相交，内唇项失效 | 使用布尔张量并独立应用内唇可见性；索引/梯度回归测试通过；真实 7 帧联合拟合通过 |
| `head_rgbd/experiment/turn_run.py` | 交付子进程写统计时还不知道自己的退出码，最终快照可能遗留 `running` | 主流程在交付退出后刷新阶段统计；回归测试和实际 8 阶段报告验收通过 |
| `tools/verify_turn_run.py`（新增） | 补充可复用产物验收入口 | 检查阶段顺序与退出码、6 文件交付、源码/输入/配置/序列哈希绑定、三张网格和最终参考帧表面一致性；退出码 0 |

没有修改生产链 `cli.py`、`pipeline.py`、`fit_*.py`、`geometry_*.py` 或公共求解器。
修改前的实验源文件和文档备份位于 `runtime/validation/turn_e2e_20261102/source_before/`。
失败运行目录保留供追溯；正式通过结果是 `attempt04`。

## 完整流程覆盖与结果

最终完整命令一次从头运行全部阶段，主进程退出码 **0**，审计总耗时 **122.787 秒**（不包含现场采集时间）。
各阶段耗时含对应子进程启动与退出；总耗时还包含编排及环境记录，因此不要求等于表内简单求和。

| 阶段 | 实际覆盖内容 | 退出码 | 耗时 / 秒 |
| --- | --- | ---: | ---: |
| tracking | YuNet/TUFA 检测、KLT 传播、深度反投影、刚体姿态和质量/连续性分组；105/105 帧可用 | 0 | 43.496 |
| keyframe_selection | 角度分桶和质量选帧；选择 7 帧/7 个角度组，覆盖 −29.6801° 至 +27.4955°，缺失视角列表为空 | 0 | 0.445 |
| prepared_keyframes | 7 个选中帧重算关键点与语义解析；写入 NPZ、全序列及仅锚点序列 | 0 | 15.984 |
| front_anchor | 原 `fit_c --front-only --postfit minimal`，默认 240 步正面初始化 | 0 | 12.246 |
| joint_shared_identity | 默认 60 步姿态 + 180 步共享形状联合拟合，7 帧均建立深度对应 | 0 | 24.223 |
| pose_correction | 原 `motion_v2` 默认 60 步；6 帧接受修正 | 0 | 9.748 |
| control_field | 一次低/中频位移场求解（2 个 outer step）、一次 Loop 混合，混合系数 0.5，无回退 | 0 | 10.950 |
| delivery | 导出 3 张网格、选帧清单、配置及完整阶段统计 | 0 | 2.213 |

交付目录：`D:\proiect\facecap_airway_delivery\runtime\turn_experiment\turn_e2e_live_20261102_attempt04\delivery`。

| 网格 | 顶点数 | 面数 | 重新加载 / 有限坐标 / 面索引 |
| --- | ---: | ---: | --- |
| init_mesh.obj | 5,023 | 9,976 | 通过 |
| joint_mesh.obj | 5,023 | 9,976 | 通过 |
| final_mesh.obj | 20,018 | 39,904 | 通过 |

该目录恰有 6 个文件：上述 3 张网格和 `keyframe_selection.json`、`experiment_config.json`、`stages.json`。
最终 OBJ 与 `field/final_vertices.npz` 中对应参考帧的顶点在 `1e-7 m` 绝对容差内一致。
控制场最大绝对位移为 2.49999 mm，未超出配置的 2.5 mm 上限；Loop 目标 0.5 被几何安全检查接受。

同次采集深度诊断如下，仅用于检查本次拟合与输入数据的一致性：

| 角度组 | median / mm | p90 / mm | 命中覆盖率 |
| --- | ---: | ---: | ---: |
| yaw+0 | 0.585 | 1.964 | 99.27% |
| yaw-30 | 0.499 | 1.698 | 99.80% |
| yaw-20 | 0.528 | 2.006 | 99.53% |
| yaw-10 | 0.542 | 1.851 | 99.40% |
| yaw+10 | 0.492 | 1.636 | 99.60% |
| yaw+20 | 0.559 | 1.717 | 99.40% |
| yaw+30 | 0.538 | 1.893 | 98.73% |

位移梯度 `|d_a-d_b| / edge_length` 的 p90/p99/最大值为 0.0377 / 0.1231 / 0.2555，配置上限为 0.45。
这些数字来自参与拟合的同一份深度数据，不能解释为独立几何精度，也没有与旧算法比较速度或精度。

## 测试证据

- 完整测试集：**99 passed, 1 skipped in 12.77s，退出码 0**；本次新增 12 项回归测试。
- 唯一跳过：`tests.test_eval_sameframe.SameFrame.test_real_run_uses_front_4_not_front_0`，原因是该用例依赖的旧重建样例在当前目录中不存在。该跳过不代表本次新采集的端到端运行被跳过。
- JUnit 与日志：`runtime/validation/turn_e2e_20261102/unit_tests.xml`、`unit_tests.log`。
- 独立产物验收：**passed，errors=[]，退出码 0**；JSON：`runtime/validation/turn_e2e_20261102/verification.json`。
- 完整运行审计及逐阶段日志：`runtime/turn_experiment/turn_e2e_live_20261102_attempt04/audit/`。

## 复现命令（PowerShell）

输出目录必须尚不存在；以下使用自动时间戳创建新的运行目录。

```powershell
Set-Location -LiteralPath 'D:\proiect\facecap_airway_delivery'
& '.\runtime\env\python.exe' -m head_rgbd.experiment.turn_run `
    --anchor '.\runtime\captures\turn_e2e_live_20261102\anchor_session.json' `
    --turn '.\runtime\captures\turn_e2e_live_20261102\turn_session.json' `
    --config '.\config\turn_experiment.json' `
    --device cuda

& '.\runtime\env\python.exe' '.\tools\verify_turn_run.py' `
    --run '.\runtime\turn_experiment\turn_e2e_live_20261102_attempt04'
```

本次验证证明此设备、环境、默认配置和这份真实采集上的完整流程能够执行并交付有效文件。
其他采集、CPU 全流程、独立扫描精度和相对速度/精度提升均不属于本次通过结论。
