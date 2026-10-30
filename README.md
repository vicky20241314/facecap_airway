# withB: 单台 Orbbec RGB-D 人头重建

本项目把方案 C 的离线 RGB-D 人头重建链路迁移为独立程序，并接入
Orbbec Gemini 215 的“引导采集后离线重建”入口。算法是 FaceCap 启发的
FLAME 原型，不是完整 FaceCap 复现。

## 当前结论

以下状态必须分开使用：

| 项目 | 状态 |
|---|---|
| 迁移实现 | 完成 |
| 真实数据离线回归 | 流程跑通 |
| 真实相机连接和原始数据保存 | 已验证 |
| 真实人脸采集到米制网格链路 | 未跑通；现有最小采集无人脸、深度无效 |
| 训练数据几何诊断 | 失败；斜侧整脸仍未优于原 B |
| 独立真实精度 | 未验证；缺少独立外参、人工轮廓和参考扫描 |

最新真实数据回归位于：

`D:\proiect\withB-runtime\outputs\regression-20261031-final\run_report.json`

流水线本身执行通过，但同一报告中的 `geometry_effect_acceptance` 为
`failed`。6 个闭口斜侧身份帧的整脸 P95 或中位数没有优于原 B；张口
P95 未超过旧仅正面对照。不能宣称真实侧面脸型、下颌角或前后厚度已经改善。

## 一键启动

在任意当前目录运行：

```powershell
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' capture-run --session-name head --device cuda
```

该命令依次执行：

1. 连接 Gemini 215 并读取 RGB-D、相机参数和对齐状态。
2. 引导采集闭口正面、闭口左斜侧、闭口右斜侧、张口正面四段。
3. 将原始 RGB、uint16 毫米深度、时间戳、内参和分段状态保存到新的采集目录。
4. 离线准备语义、标点、可见性和深度有效区域。
5. 串行拟合共享身份及逐帧姿态、表情和下颌。
6. 按默认 guarded 模式执行正面锚定、有界侧面细化、颈部处理和米制导出。
7. 执行几何、参考标点和支持度契约验证。

运行目录默认是 `D:\proiect\withB-runtime`。所有采集和输出目录都以独占方式
创建，已有目录不会被覆盖。

仅检查相机时：

```powershell
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' probe --samples 3
```

## 采集操作

默认四段和每段 5 帧：

| 段 | 状态 | 要求 |
|---|---|---|
| `closed:front` | 闭口 | 正对相机，自然闭唇 |
| `closed:left` | 闭口 | 头向画面左侧转约 25-35 度 |
| `closed:right` | 闭口 | 头向画面右侧转约 25-35 度 |
| `open:front` | 张口 | 正对相机，明显张口 |

每段会先显示局部预览。确认人脸完整、RGB/深度位置一致、深度不是黑色空洞后，
按 `SPACE` 开始；`ESC` 取消。每段独立保存，头和身体尽量保持稳定。斜侧段必须
来自真实采集，程序不会复制或旋转正面帧冒充侧面。

采集契约要求至少一个真实闭口斜侧段。时间约束只连接同一段内、间隔不超过
0.2 秒且重叠跟踪有效的相邻帧，不跨越分段缺口。失跟时可在新段通过标点重定位。
闭口和张口参考帧分别绑定真实正面帧；无效内唇深度在侧车中写为 JSON `null`。

## 运行模式

默认 guarded 模式先只用真实正面闭口帧恢复大尺度身份，再固定正面重叠区域的
粗身份、姿态、表情和下颌，只允许正面保护范围外、至少两个真实斜侧帧支持的
顶点发生不超过 3 mm 的细化。细化后执行三角形翻转保护。

自由共享身份模式只用于研究对照，必须同时显式给出两个开关：

```powershell
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' reconstruct --sequence 'D:\path\to\sequence.json' --output 'D:\proiect\withB-runtime\outputs\free-study-01' --free-identity --research-only --device cuda
```

自由身份模式在现有数据上发生过正面回归，不是默认算法，也不能作为效果验收结果。

## 输出说明

`capture-run` 的输出目录默认是
`D:\proiect\withB-runtime\outputs\<capture-name>_reconstruction`：

| 路径 | 内容 |
|---|---|
| `run_report.json` | 分段、阶段命令、退出码、耗时、显存、排除/降级原因和总状态 |
| `prepared\sequence.json` | 候选帧、源文件和帧号、时间戳、状态、视角、重叠和拒绝记录 |
| `reconstruction\guarded\closed_coarse_m.obj` | 闭口粗几何，米 |
| `reconstruction\guarded\closed_refined_m.obj` | 闭口有界细化几何，米 |
| `reconstruction\guarded\open_coarse_m.obj` | 张口粗几何，米 |
| `reconstruction\guarded\open_refined_m.obj` | 张口有界细化几何，米 |
| `reference_geometry.json` | 闭口/张口参考帧、源帧号、坐标系、粗/细重心标点及唇距 |
| `report.json` | 每帧身份/运动角色、姿态、表情、下颌、可见性、对应拒绝和翻转保护 |
| `parameters.npz` | 共享身份、逐帧运动/表情/下颌、实际网格和安全偏移 |
| `support.npz` | 逐帧顶点观测、观测次数、细化帧/次数、位移、标签和支持分数 |
| `*_support_m.ply` | 支持度着色米制网格 |
| `neck_track_*` | 独立颈部观测表面及支持度；不反馈头部身份 |
| `validation\verification.json` | 有限性、米制尺度、参考坐标、时间边和防翻转检查 |
| `validation\reference_geometry.json` | 从实际导出网格重建的参考标点验证 |

`support.npz` 标签：

- 2：至少两次可靠重复观测支持。
- 1：观测不足，保持模型先验。
- 0：未获得可靠深度支持，属于模型补全。

支持分数只表示重复观测计数，不是校准后的概率或独立覆盖率。模板后脑、被遮挡
耳部和未观测侧面属于先验/补全，不是扫描得到的真实表面。

## 环境与资产

运行目录：

`D:\proiect\withB-runtime`

Python：

`D:\proiect\withB-runtime\env\python.exe`

资产清单位于仓库 `config\assets.json`，实际资产只放在运行目录
`D:\proiect\withB-runtime\assets`。当前 7 个 FLAME、YuNet、UniFace 和 TUFA
资产都用 SHA256 验证。检查命令：

```powershell
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' assets
```

完整 Python 依赖见 `requirements.txt`。不要修改共享研究环境；需要变更时创建
新的隔离环境。大型模型、环境、采集和输出均不得提交 Git。

许可边界：

- `third_party\flame` 代码为 MIT；FLAME 模型须遵守官方模型许可。
- `third_party\tufa` 为 GPL-2.0。
- UniFace、YuNet 和下载资产须遵守各自上游许可。

## 效果与验证边界

最新 20 帧真实数据回归总耗时 388.89 秒，guarded C 拟合耗时 163.28 秒，
PyTorch allocator 峰值分配/保留为 65.77/88.00 MiB。GPU 作业按 B 全视角、
B 仅正面、C 正面锚、C guarded 的顺序串行运行，没有终止其他进程。

代表帧整脸相机轴残差（中位数/P95，mm）：

| 帧 | 原 B | guarded C |
|---|---:|---:|
| `closed_front_1` | 1.128 / 6.037 | 0.453 / 2.409 |
| `closed_left_0` | 0.532 / 2.829 | 0.742 / 3.339 |
| `closed_right_3` | 0.533 / 2.773 | 0.735 / 3.416 |
| `open_front_2` | 1.051 / 4.312 | 0.706 / 4.319 |

闭口参考 `closed_front_2` 的粗/细内唇距为 1.689601 / 1.689598 mm；原始自动
深度标点为 1.035963 mm，但该数字不是人工真值，也没有被强制拟合。张口参考
`open_front_1` 的粗/细内唇距为 28.531110 / 28.531118 mm，原始内唇深度无效，
侧车明确输出 `null`。

相机已实际识别为 Gemini 215，序列号 `AYZ81630069`，USB 3.0，SDK 2.2.8，
1280x720 HW D2C。最小采集的 6/6 帧均因未检测到人脸被排除，三段有效深度
中位数均为 0。失败审计位于：

`D:\proiect\withB-runtime\outputs\live-minimal-audit-failure\run_report.json`

因此当前不能报告“真实采集到重建链路跑通”。

## 历史文档提示

`head_rgbd\C_METHOD.md` 和 `head_rgbd\C_RESULTS.md` 是来源研究的原样审计文档，
其中旧 worktree、旧运行目录和 `reproduce_c.ps1` 命令不是本迁移项目的运行入口。
本 README 和 `headrgbd.ps1` 才是 `withB` 的权威入口。迁移记录和清理候选见
`MIGRATION.md`。
