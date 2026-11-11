# 交付验证记录（2026-11-01）

本文件记录本机上的**实际**验证。§1–§5 是 0.2.0 交付包的那次全流程验证记录，
保留作历史；§7 是 0.3.0（加速版重建算法）本轮实际执行过的检查。所有数值来自真实运行，
不是从旧研究目录抄来的。

## 1. 环境与依赖

| 检查 | 结果 |
|---|---|
| Python | 3.10.22（自 `runtime/env`，与共享研究环境相互独立） |
| PyTorch | 2.0.1 + CUDA 11.7，`cuda_available=True` |
| pyorbbecsdk / OpenCV / NumPy / SciPy / trimesh | 2.2.8 / 4.8.1 / 1.26.4 / 1.11.4 / 4.4.9 |
| 模型 | 7 个文件 SHA256 全部 `ok`（`headrgbd assets`） |
| 模型加载 | `headrgbd device` → `models.status = passed`（FLAME 5023 顶点模板、TUFA 权重、UniFace 解析器） |

## 2. 契约测试

```
runtime\env\python.exe -m pytest tests -q
20 passed
```

覆盖：版本一致、命令集合、默认算法唯一、冻结阶段顺序、strict-coverage 接受规则、
路径解析（跨工作目录）、时间边、无效深度、单网格导出、单位检查。

## 3. 统一入口

| 命令 | 结果 |
|---|---|
| `headrgbd.cmd --version`（从 `C:\` 启动） | `headrgbd 0.2.0 (facecap-inspired-v2-strict)`，退出码 0 |
| `headrgbd.cmd assets` | 7/7 `ok`，退出码 0 |
| `headrgbd.cmd device` | 相机打开成功，1280×720 RGB-D、`rgb_depth_aligned=true`，内参读取成功 |

相机：Orbbec Gemini 215，序列号 `AYZ81630069`，USB3.0，固件 1.0.9。
彩色内参 `fx=849.222 fy=849.119 cx=640.126 cy=361.012`。

## 4. 真实采集离线重建

输入：`runtime/_validation/zhou_cap_10310944`（4 段 × 5 帧真实 Gemini 采集，
源目录 `D:\proiect\zhou_cap_10310944`，复制后 SHA256 与源一致）。
该采集没有逐会话标定记录，因此准备阶段标记 `session_alignment_unverified`。

命令：

```pwsh
.\headrgbd.cmd reconstruct --capture runtime\_validation\zhou_cap_10310944 --output runtime\outputs\<run>
```

结果见运行目录下的 `delivery/`（由 `tools/verify_delivery.py` 复核，见第 5 节）。

## 5. 输出与几何检查

由 `tools/verify_delivery.py` 独立读取磁盘文件复核：

- `delivery/` 中只有 2 个网格：`head_closed_m.obj`、`head_open_m.obj` + 3 个侧车；
- 两个网格顶点有限、面索引合法、包围盒为米级、`unit = m`，且**面表完全相同**（共享拓扑）；
- 每个状态各自的参考帧、68 个标点（由该状态网格 + 重心坐标重算）、刚体变换一致；
- 闭口/张口确实是两个不同状态：参考帧不同（`closed_front_2` / `open_front_0`），
  内唇距 1.92 mm / 26.95 mm，顶点相对位移最大 64.8 mm；
- 支持度计数覆盖共享拓扑的全部 20 018 个顶点；
- `manifest.json` 版本与 `VERSION` 一致，算法名唯一。
- 与 frozen 参考逐点比对：闭口 0.0003 mm、张口 0.0003 mm（面表全等）。

## 6. 未验证项

- **几何精度未验证**：没有扫描配准、没有人工轮廓、没有独立外参。
- **真实相机新采集未在本轮执行**：相机已成功打开并读到对齐 RGB-D 与内参，
  但完整采集需要人员配合摆姿势，本轮没有用旧数据冒充新采集。
- 颈部补片来自真实深度观测，但不做跨段颈部身份配准（与算法原有限制一致）。

## 7. 0.3.0（加速版重建算法）本轮验证

本节只覆盖本轮实际执行过的检查。重建算法改动见 `CHANGELOG.md` 的 0.3.0 段；
评估工具与 loss 统计不在本版本范围内，未随包交付。

| 检查 | 命令 | 结果 |
|---|---|---|
| 版本一致性 | `python -m head_rgbd --version` | `headrgbd 0.3.0 (facecap-inspired-v2-strict)`，退出码 0 |
| 契约测试 | `pytest tests -q`（在本目录，使用项目运行时） | **20 passed, 1 skipped**；跳过项为 `tests/test_contracts.py:490` “runtime environment is not installed”，因为交付包按惯例不带 `runtime/env` |
| 完整重建（快速流程） | `head_rgbd reconstruct --capture <验证采集> --fast --mid-rounds 1` | **298.3 s**（0.2.0 基线 1945.7 s，6.52×），status=passed，导出闭口/张口各一份 |
| 完整重建（只换射线、不跳过任何阶段） | `head_rgbd reconstruct --capture <验证采集>` | **865.4 s**（2.24×），用于把射线收益与省略收益分开报告 |
| 交付网格 | 上述两次运行的 `delivery/` | 20 018 顶点 / 39 904 面，顶点有限、面索引合法、米级包围盒、两面表相同 |

同一采集、同一台机器、同一份 prepared 序列。逐阶段耗时、消融表与剩余热点见
`CHANGELOG.md`；本轮没有重新做相机采集，也没有做扫描配准，因此 §6 的未验证项
**仍然成立**。

### 与 0.2.0 的关系（必须知道）

- 新的射线实现与旧的 rtree 在真实网格+真实深度射线上一致到 **0.15 µm**、命中面 100% 相同。
- 但流水线在重心支持阈值 `bc > .05` 与 QP active set 处对 1e-7 量级扰动敏感：
  `free_vertices` 949 → 946，低频解随之改变。
- 因此**交付网格与 0.2.0 不是逐位一致**；以 0.2.0 为基准的全脸 MAE 读数为 +0.39 mm，
  而“只换射线、不跳过任何阶段”的同机运行也是 +0.39 mm。
- 结论：**§1–§5 的 0.2.0 验收结论不能直接沿用到 0.3.0**，需要用同机同代码的对照重新验收。

## 8. 0.4.0（正侧面联合形状融合）本轮验证

新增能力：`--shape-fusion joint`。默认仍是 `guarded`，所以 0.3.0 的行为可原样复现。

| 检查 | 命令 | 结果 |
|---|---|---|
| 版本一致性 | `python -m head_rgbd --version` | `headrgbd 0.4.0 (facecap-inspired-v2-strict)`，退出码 0 |
| 测试 | `pytest tests -q`（本目录，使用项目运行时） | **25 passed, 1 skipped**；跳过项仍是 `runtime environment is not installed` |
| 联合重建（20 帧参考采集） | `reconstruct --capture <参考采集> --fast --mid-rounds 1 --shape-fusion joint` | **249.8 s**，status=passed |
| guarded 对照（同一采集、同一代码） | 同上去掉 `--shape-fusion joint` | **265.0 s**，status=passed |
| 侧面消融 | 同上加 `--side-depth-weight 0` | **301.8 s**，status=passed |
| 联合重建（airway3d 四视角采集） | 先 `tools/import_multiview_capture.py` 转换，再同上 | **118.3 s**，status=passed，闭口/张口各一份 |
| 交付网格 | 上述各运行的 `delivery/` | 20018 顶点 / 39904 面，顶点有限、索引合法、米级包围盒、两面表相同 |
| 交付继承联合结果 | 交付 OBJ vs `parameters.npz:dense_vertices[row]` | 最大差 0.000005 mm（OBJ 文本精度） |
| 机制生效 | `reconstruction/guarded/report.json` | 正面硬锁顶点 2797 → **0**；局部场改变顶点 0 → **1776**，最大 0 → **2.999 mm** |

### 质量（同帧、每帧 6000 点、|残差| ≤ 30 mm，`tools/eval_multiview.py`）

| 视角 | 状态 | 划分 | guarded MAE | joint MAE | Δ |
|---|---|---|---|---|---|
| 正面 | closed | 留出 | 1.352 | 1.345 | −0.007 |
| 正面 | closed | 训练 | 1.328 | 1.368 | +0.040 |
| 左 | closed | 留出 | 1.207 | 1.246 | +0.039 |
| 左 | closed | 训练 | 1.179 | 1.167 | −0.012 |
| 右 | closed | 留出 | 1.168 | 1.165 | −0.003 |
| 右 | closed | 训练 | 1.101 | 1.119 | +0.018 |
| 正面 | open | — | 1.737 | 1.581 | **−0.156** |

外观损失：闭口持平（最大 +0.040 mm，门限 0.05 mm 内），张口与覆盖率明显改善。侧面消融
（无侧面深度）会让侧面嘴唇恶化 0.7–1.0 mm、下颌轮廓恶化 0.65–0.84 mm，说明侧面数据是
必要的；但 **joint 在闭口精度上没有超过 guarded**，鼻部也无改善——见
`docs/JOINT_FUSION_REPORT.md` 与 `CHANGELOG.md` 的 0.4.0 段。`§6 未验证项` 仍然成立。
# 0.5.0 默认方案验收入口

最新默认流程与实际逐项验收见 [DEFAULT_SMOOTH_FAST_REPORT.md](docs/DEFAULT_SMOOTH_FAST_REPORT.md)。
`config/defaults.json` 是公共 run/reconstruct 的统一来源，cmd/PowerShell 包装脚本保持透传。
本文件后续 0.2/0.3/0.4 表格是历史记录，其中“默认 guarded”等描述只适用于对应版本。
本轮新实验、逐帧平顺度、同批交错三次计时、实际参数和命令均保存在
`runtime/validation/default_smooth_fast`，原始采集、模型与历史输出未覆盖。

## 0.5.1 单窗口采集修复验证（2026-11-02）

默认采集四个姿态共用一个 `RGB-D head capture` 窗口。每段在窗口中按 SPACE，
实时倒计时后录制，不再额外到终端按 Enter。ESC 或关闭窗口取消；结束、异常和
取消均释放窗口与相机。无预览模式保留终端确认，`--auto` 自动倒计时。

原来逐段创建/销毁预览，关闭窗口后下次 `imshow` 可能重建窗口；预览模式还同时
要求终端 Enter 和窗口 SPACE。简单合成测试没有复现用户原先看到的两个窗口，
因此不把上述代码问题等同于已复现全部现场原因。修复后对窗口数量与按键做了
以下实际验证。

| 检查 | 结果 |
|---|---|
| 全套 `python -m pytest tests -q` | **63 passed, 1 skipped**，15.24 s，退出码 0 |
| 合成画面 + 原生 Windows HighGUI | 四段同一窗口句柄，四次 SPACE 均启动；关闭取消、ESC 取消、无遗留窗口，6/6 检查通过，退出码 0 |
| Orbbec 实际 RGB 画面 + 原生窗口 | 四段各读入 2 帧，窗口句柄始终 `14354656`；相同 6/6 检查通过，退出码 0 |
| cmd 版本 / capture 帮助 | 均退出码 0，版本 0.5.1 |
| PowerShell 版本 / run 帮助 | 均退出码 0，版本 0.5.1 |

回归覆盖：连续复用窗口、第一轮事件中的 SPACE、倒计时顺序、录制与缺帧重试时
ESC、关闭后不重开、自动/无预览、会话取消后的 partial manifest 与相机释放。
原生事件由 `PostMessageW` 投递到测试进程自己的窗口，不使用模拟的 `waitKey`。
相机验证只在内存中使用实际画面，没有保存照片，也没有录制新的真人姿态数据。

帮助入口检查还发现 PowerShell 将 `run --help` 中的帮助参数绑定后丢弃，误进入
采集等待。该次检查失败（退出码 1），已结束测试进程。修正为向统一 CLI 转发
子命令帮助，并增加 `run --help --device __help_guard__` 回归：即使帮助转发退步，
无效设备参数也会阻止启动硬件。修正后所有入口检查均通过。

可追溯记录：

- `runtime/validation/capture_single_window/native_windows.json`
- `runtime/validation/capture_single_window/live_camera_windows.json`
- `runtime/validation/capture_single_window/pytest.txt`
- `runtime/validation/capture_single_window/entrypoints.json`

复核工具：`tools/verify_capture_preview.py --output <文件>`，使用实际相机时加
`--camera`。本次没有修改重建算法或默认拟合参数，未重跑重建质量和计时实验；
0.5.0 的对应验收仍以其独立报告和当次源代码快照为准。



## 0.5.2 Loss 驱动局部改进验收（2026-11-03）

默认采用权重 0.5 的可靠张口下半脸深度项，在原 80 步表情/下颌校正内执行。
保留原上脸定位、共享身份、口内/边缘排除与运动边界，并同时验收下半脸固定射线。
批量复用回退比例的前向结果、初始化去重射线查询；240/240 拟合、中频 1 轮、
8 mm 弯曲及安全 Loop 0.5 均保持。

两套采集、24 帧、12 次完整重建全部退出 0；两套像素网格与共同覆盖射线均有小幅
下颌残差收益，逐帧平顺度、覆盖与几何安全通过。权重 1.0 在第二套采集退步，未采用。
109 项项目回归通过、1 项跳过；隔离候选回归 108 通过、2 跳过。
公共 Python/cmd/PowerShell 入口均报告 0.5.2，实际生效参数写入交付 manifest。

计时存在明显波动，全部样本保留；补充运动阶段中位数约持平，未证明稳定加速。
收益指同源深度残差改善，不代表独立扫描精度。HTML 报告包含全部误差、区域代价、
覆盖、法向 RMS/P95、耗时范围、图像、命令及哈希：
[LOSS_GUIDED_IMPROVEMENT_REPORT.html](docs/LOSS_GUIDED_IMPROVEMENT_REPORT.html)。
原始证据与独立源码快照：`runtime/validation/20261103_loss_guided`。

## 0.5.4 颈前塑形 V2 验收（2026-11-11）

环境与数据：runtime/env Python 3.10.22；基线为纯 0.5.2 运行 `runtime/outputs/zp_052_neck_bump_baseline`
（重建自 zp_0.5.1/prepared，无 ArUco）；全部特性导出复用该重建，仅经公开导出入口。

位置复核：原始闭口人工点 (692,568) 位于 prepared neck_mask 之外的衣领上，原始张口点 (649,588) 位于
neck_mask 底缘衣物边界；根因是 neck_roi 点击门限在人脸框下方 10% 框高，排除了真正的喉结区。门限改为
2% 框高；修正中心为闭口 (688,504)、张口 (663,555)，原因与证据写入 neck_landmarks_v2.json。

八项验证（工具 `tools/verify_neck_prominence_v2.py`，明细 `verification_report.json`）：

| # | 检查 | 结果 |
|---|---|---|
| 1 | 关闭塑形 → 与纯 0.5.2 基线字节一致 | 通过 |
| 2 | female + auto → 全部顶点零位移 | 通过 |
| 3 | 无有效标签 → 正常交付基线 | 通过 |
| 4 | 重复导出不累积 | 通过 |
| 5 | 保护/固定人脸射线 | 通过（下巴射线命中三角形不变，点位漂移 0.017 mm） |
| 6 | 闭口/张口共享拓扑且与基线一致 | 通过（20018 顶点 / 39904 面） |
| 7 | 报告参数与 OBJ 读回一致 | 通过 |
| 8 | 形状验收 + 全部几何安全 | 通过 |

形状结果（固定剖面 H，容差 ±1 mm 在看结果前固定）：闭口 H 0.12 → 8.00 mm（误差 0.00），
半高宽/长 9.57/8.33 mm；张口 H −0.49 → 8.00 mm（误差 0.00），半高宽/长 11.30/11.31 mm；
两状态 0 自交、0 尖峰、0 失控长边、0 细长面；峰值偏离塑形中心 0 mm；绑定重心位移 7.88/8.49 mm。
单场对照（基底份额 0）张口触发 displacement_spike，证明双场必要。

阶段耗时（同一导出入口，3 次采样，闭口+张口合计）：319–355 ms（与 V1 同量级）。

限制：真实 ArUco 采集链路未验证（数据无标签，人工指定属软件链路验证）；8 mm 为外观设计参数，
非解剖标准；模板颈部与真人颈部存在约 10–13 mm 横向差异，未被本阶段校正。

## 0.5.5 颈前重建 V3（观测驱动）验收（2026-11-11）

数据：复用 `runtime/outputs/zp_052_neck_bump_baseline` 的 0.5.2 未塑形几何与
`runtime/outputs/zp_0.5.1/prepared` 原始深度（closed_front_0 / open_front_4），未重跑 FLAME。

关键事实：
- 模板颈前表面落后真实皮肤 ~11.6 mm（中位），V2 的 8 mm 鼓包叠加在错误背景上。
- 扣除冻结二次背景后，标签处局部残差仅 -0.5 mm（闭口）/ -0.7 mm（张口），背景模型敏感度 ±0.3/±0.6 mm；
  **数据不支持明显喉结**。
- 首命中（固定母集、无命中罚 30 mm、不截断）：闭口 MAE 13.62 → **7.39** mm（中位 −11.76 → −0.95）；
  张口 5.79 → **3.30** mm（中位 2.06 → +0.04）。V2 分别为 13.24 / 6.37 mm，无改善。
- 面部保护：受保护顶点位移 0；面部表面命中点位移 0.0000 mm（工程容差 0.05 mm）；闭口有 1 条射线
  首命中颈面（基线即如此），其点位移 4.85 mm，不构成新增遮挡。
- 几何：0 自交、0 翻面、0 面积塌陷、0 细长面，最大边长增长 1.59/1.19（上限 2.2）；预存在的 8 个
  FLAME 近退化面单独记录。
- 局部红绿细分：+985 顶点，closed/open 同一构造与父面绑定；H 测量限制在 |u|,|v|≤18 mm 核心区。

结论：几何安全、观测拟合改善明显；喉结形态证据不足，报告标注“安全但保真未证实”。
