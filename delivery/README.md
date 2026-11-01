# facecap_airway_delivery

用一台 Orbbec Gemini 215 采集成人头部 RGB-D 数据，离线重建出**一份米制高分辨率人头网格**。

```
真实相机采集 → 离线重建 → 导出最终网格
```

一条命令走完：

```pwsh
.\headrgbd.cmd run
```

---

## 1. 这是什么、能做到什么、做不到什么

**是**：FaceCap 思路启发的 **FLAME 离线原型**。用一次真实、有引导的 RGB-D 采集，先在全序列上拟合**同一个身份**，再逐帧估计**姿态、表情和下颌**，最后在与 FLAME 同拓扑的密集表面上用真实深度做残差细化，导出一个闭口状态的密集网格。

**不是**：

- 不是 FaceCap 的完整复现（没有视频级实时跟踪、没有多视角装置、没有其训练数据与网络）。
- **没有经过独立精度验证**。本项目的几何质量只做过内部同数据对照，没有扫描配准、没有人工轮廓、没有独立外参。因此本项目**不声称**任何"多少毫米真实精度"。
- 不是产品级扫描仪。遮挡区域（后脑、耳后）、内唇、口腔内部属于模型先验或完全未观测，`support.json` 会如实标注。

> 工程上能跑通 ≠ 几何精度已通过。请把输出当作"带真实深度支持的 FLAME 形变"，而不是测量级扫描。

---

## 2. 硬件与系统要求

| 项目 | 要求 |
|---|---|
| 相机 | Orbbec Gemini 215（`pyorbbecsdk` 2.2.8） |
| 系统 | Windows 10/11 x64 |
| GPU | NVIDIA（重建用 CUDA；已验证 RTX 3060 Laptop 6 GB）。可用 `--device cpu`，但慢很多 |
| 内存 | ≥ 16 GB |
| 磁盘 | 运行目录约 7 GB（主要是 Python 环境），一次采集约 20 MB，一次输出约 25 MB |
| 采集距离 | 0.3–0.7 m，正对镜头 |

---

## 3. 目录结构

```
facecap_airway_delivery/
├─ README.md                    本说明
├─ VERSION                      0.2.0
├─ CHANGELOG.md
├─ LICENSE / THIRD_PARTY_NOTICES.md
├─ requirements.txt             重建环境所需的精确版本
├─ headrgbd.cmd / headrgbd.ps1  唯一启动入口
├─ config/
│  ├─ defaults.json             默认算法与参数（唯一一套）
│  └─ assets.json               模型文件 SHA256 清单
├─ head_rgbd/                   流水线源码
│  ├─ cli.py                    入口
│  ├─ pipeline.py               唯一算法链路 + 导出契约
│  ├─ capture_session.py        引导采集
│  ├─ prepare.py                数据准备、关键帧、追踪
│  ├─ fit_c.py                  共享身份拟合
│  ├─ motion_v2.py              逐帧姿态/表情/下颌
│  ├─ identity_v2.py            低频共享身份（正面保护）
│  ├─ fit_d.py                  密集表面与最终导出
│  ├─ geometry_c.py / geometry_d.py / surface_v2.py   对应、保护、Loop、证据
│  └─ ...
├─ core/                        相机 SDK 封装、FLAME 装载、人脸/颈部解析
├─ third_party/                 FLAME 与 TUFA 源码（含各自 LICENSE）
├─ tests/test_contracts.py      小型核心契约测试
└─ runtime/                     ← 不进入 Git
   ├─ env/                      Python 3.10 环境（含 CUDA 版 PyTorch）
   ├─ assets/                   模型权重
   ├─ models/
   ├─ cache/
   ├─ captures/                 每次采集一个时间戳目录
   ├─ outputs/                  每次重建一个目录
   ├─ logs/                     device / assets 检查记录
   └─ _validation/              交付验证用的一份真实采集
```

---

## 4. 安装

### 4.1 直接使用已准备好的环境（推荐）

如果 `runtime/env` 和 `runtime/assets` 已经存在，跳过本节，直接做 4.3 自检。

### 4.2 从零重建环境

```pwsh
# 1) 准备 Python 3.10 x64，装到 runtime\env
#    例如：conda create -p .\runtime\env python=3.10.22 -y
# 2) 安装依赖
.\runtime\env\python.exe -m pip install -r requirements.txt
#    PyTorch 需要 CUDA 11.7 版本：
#    pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117
# 3) Orbbec SDK：pyorbbecsdk 2.2.8 自带 Gemini 215 驱动，插上相机即可
```

### 4.3 模型获取与许可

模型**不在 Git 里**，必须按各自许可单独获取，放到下表路径。放好后 `.\headrgbd.cmd assets`
会逐个核对 SHA256（清单见 `config/assets.json`）。

| 文件 | 路径 | 来源与许可 |
|---|---|---|
| `flame2023.pkl` | `runtime/assets/flame/` | FLAME（MPI），**仅限非商用研究**，见 `third_party/flame/LICENSE` |
| `flame_static_embedding.pkl` | `runtime/assets/flame/` | 同上 |
| `flame_dynamic_embedding.npy` | `runtime/assets/flame/` | 同上 |
| `face_detection_yunet_2023mar.onnx` | `runtime/assets/models/yunet/` | OpenCV Zoo / YuNet |
| `parsing_resnet34.onnx` | `runtime/assets/uniface/` | UniFace 人脸/颈部解析 |
| `Face_prompt_base.pth` | `runtime/assets/tufa/` | TUFA 68 点网络权重 |
| `shape_68.npz` | `runtime/assets/tufa/Prompt/` | TUFA 源码自带（已在 Git 中） |

许可细节见 `THIRD_PARTY_NOTICES.md`。**商用、云端部署或对外分发模型，需各权利方授权。**

### 4.4 自检

```pwsh
.\headrgbd.cmd --version      # headrgbd 0.2.0 (facecap-inspired-v2-strict)
.\headrgbd.cmd assets         # 7 个模型 SHA256 全部 ok
.\headrgbd.cmd device         # 打开相机 + 载入模型
```

---

## 5. 采集与重建：一条命令

```pwsh
cd D:\proiect\facecap_airway_delivery
.\headrgbd.cmd run
```

`run` 会依次完成：

1. 打开相机，检查 RGB/深度尺寸、对齐、内参与同步；
2. 按 4 个视角引导采集（每段 5 帧）；
3. 保存原始数据 + 分段信息到 `runtime/captures/<时间戳>_head/`；
4. 离线重建（CPU/GPU 串行，单卡）；
5. 导出 `runtime/outputs/<采集名>_reconstruction/delivery/` 下的最终网格。

### 5.1 怎样站位、转头、闭口、张口

程序会在屏幕上逐段提示，每段先按一次 `Enter`（或 `--auto` 自动倒数）：`SPACE` 开始、`ESC` 取消。

| 段 | 状态 | 要求 |
|---|---|---|
| 1 | closed:front | 正脸看镜头，**嘴唇自然闭合**（不要抿紧），头不要歪 |
| 2 | closed:left | 保持闭口，头向**画面左侧**转约 25–35°（以图像左右为准） |
| 3 | closed:right | 保持闭口，头向**画面右侧**转约 25–35° |
| 4 | open:front | 正脸，**自然张口**（能看到内唇），身体不要动 |

要点：

- 距离 0.4–0.6 m，让整个头部（含下巴）在画面中，尽量避开强反光；
- 每段只需保持 1 秒左右（5 帧 × 0.12 s）；
- 继续转动时**不要**换表情，表情变化会被当作身份变化污染拟合；
- 刘海、手、眼镜不要挡脸。

### 5.2 什么时候开始、结束、重试

- **开始**：屏幕出现该段提示、倒计时结束即开始录制；倒计时期间调整好姿势即可。
- **结束**：4 段都出现 `saved closed_front (5 frames)` 之类的输出后，采集结束，自动进入重建。
- **重试整段**：`ESC` 取消，或重建失败后重跑。**不要**手动改 `session.json`。
- 单段人动了导致重建失败时，直接重跑 `run`（每次采集都是新目录，不覆盖旧的）。

### 5.3 只采集 / 重建已有采集

```pwsh
# 只采集，不重建
.\headrgbd.cmd capture --session-name subject01

# 重建已有采集（session.json 所在目录）
.\headrgbd.cmd reconstruct --capture runtime\captures\20261101_113000_123456_head

# 也可以重建已准备好的 sequence.json（高级）
.\headrgbd.cmd reconstruct --sequence runtime\outputs\<run>\prepared\sequence.json

# 指定输出目录（必须是新目录，绝不覆盖）
.\headrgbd.cmd reconstruct --capture <dir> --output runtime\outputs\my_run

# 只有原始 frames.json 的第三方采集（无 session.json）：传该目录即可，
# 程序会先把各段原始 npz 字节一致地复制到 run 目录再重建
.\headrgbd.cmd reconstruct --capture <含 frames.json 的目录>
```

`reconstruct --capture` 接受两种目录：程序自己采集的（有 `session.json`，声明了 HW D2C 对齐），
以及只有 `frames.json` 的外部原始采集（对齐状态未知）。后者自动以
`--allow-unverified-alignment` 降级准备，并在 `prepared/sequence.json` 与
`manifest.json` 中标记 `session_alignment_unverified`，**不会**伪造成已验证对齐。

需要的实时参数（一般不用改）：

```pwsh
.\headrgbd.cmd run --frames-per-view 5 --interval-s 0.12 --device cuda
.\headrgbd.cmd reconstruct --capture <dir> --device cpu        # 无 GPU
.\headrgbd.cmd reconstruct --capture <dir> --landmark-device cpu

# 只从已完成的重建再导出（不重跑拟合，秒级完成）
.\headrgbd.cmd export --run runtime\outputs\<run>
```

---

## 6. 输出：只有一份最终网格

`runtime/outputs/<采集名>_reconstruction/delivery/`：

| 文件 | 内容 |
|---|---|
| `head_m.obj` | **唯一最终网格**：闭口参考帧的密集表面（20 018 顶点 / 39 904 面），单位**米** |
| `manifest.json` | 运行清单（见下） |
| `support.json` | 每个顶点的真实观测支持度标签 |
| `reference_landmarks.json` | 闭口参考帧的 68 个标点（模型侧 + 原始观测侧） |

同目录的 `run_report.json` 与 `reconstruction/` 是审计记录（阶段命令、耗时、日志），
**不是**给用户看的中间网格。粗网格、锚点网格、Loop 基面、插值网格、对比网格都不再导出。

### 6.1 单位、坐标与参考帧

- **单位**：米（OBJ 顶点数值即米）。深度原始数据为毫米，内部换算后求解。
- **坐标系**：`X 向右 / Y 向上 / Z 向后`（相机右手系，Z 朝相机后方，即脸在 z<0 一侧）。
  单位与坐标都会写进 `manifest.json` 的 `mesh.unit` / `mesh.coordinate_system`。
- **参考帧**：`closed_front_*` 中真实存在、且深度有效的那一帧闭口正面帧（默认 `closed_front_2`）。
  网格是"该参考帧相机坐标下的闭口几何"，不是世界坐标，也不是扫描仪坐标。
- `reference_landmarks.json` 里同时给出 `animated_model_to_reference_camera_4x4`
  与逆变换，说明是纯刚体坐标变换，**不是**扫描配准。

### 6.2 manifest.json 记录什么

软件版本与算法来源、配置参数、输入采集与参考帧、网格单位/坐标系/顶点数/面数、
`success | degraded` 状态、关键质量警告、支持度统计、颈部独立补片、限制说明。

### 6.3 支持度怎么读

`support.json` 的 `labels` 三类，`vertex_label` 与 `head_m.obj` 顶点顺序一一对应：

| 标签 | 含义 |
|---|---|
| `distinct_closed_bursts_and_angle_supported` | 至少两次闭口短段、且转角差 ≥ 8° 的真实观测支持 |
| `single_or_correlated_burst_plus_prior` | 只有单次/相关性很强的观测，叠加模型先验 |
| `unobserved_completion` | **没有真实观测**，完全由模型补全 |

**插值出来的标签不算新增真实观测**；`confidence` 只是启发式权重，不是标定概率。

---

## 7. 算法流程（唯一默认算法）

默认算法固定为 `facecap-inspired-v2-strict`，来源：`candidate_frozen` 链（2026-10-31，
源码提交 `9ef3837`）。**没有** B/C/free/guarded/accounted 等模式开关。

| # | 阶段 | 模块 | 做什么 |
|---|---|---|---|
| 0 | 采集 | `capture_session` | 分段采集原始 RGB-D，保留时间戳、闭/张口、质量 |
| 1 | 准备 | `prepare` | 人脸/颈部解析、TUFA 68 点、深度有效性、相邻帧追踪、关键帧 |
| 2 | 共享身份 | `fit_c --front-only` → `fit_c --front-anchor` | 全序列**一个身份**，正面深度受保护 |
| 3 | 运动 | `motion_v2` | **先**解释每帧姿态/表情/下颌，再谈深度 |
| 4 | 低频身份 | `identity_v2` | 带正面保护的低频身份修正 |
| 5 | 运动重估 | `motion_v2 --identity-field` | 身份固定后重估运动 |
| 6 | 密集细化 | `fit_d --dense-mode loop-research` | 同拓扑 Loop 基面 + 真实深度残差，最终网格 |

必须保留的约束（不是可选优化）：

- 无效内唇深度**不产生**伪三维约束（NaN + 显式掩码，绝不当 0）；
- 不直接平均不同表情的原始深度；
- 固定射线覆盖：接受一个新步时**覆盖不得减少**、MSE 不得超 2 %+0.002 mm²（strict-coverage）；
- 位移上限 3 mm、法向变化 ≤ 35°、有向面积 ≥ 5 %（防翻转）；
- 时间约束**不跨**采集分段缺口、不跨追踪丢失帧；
- 颈部单独处理（独立跟踪、按段输出观测补片，不参与身份拟合）；
- 每帧仍导出真实支持度标签。

---

## 8. 常见失败与恢复

| 现象 | 原因 | 处理 |
|---|---|---|
| `Runtime Python not found` | `runtime/env` 未就绪 | 见 4.2 |
| `runtime assets failed verification: ...=missing` | 模型未放好 | 按 4.3 放模型，再跑 `assets` |
| `... = sha256_mismatch` | 模型文件不对/被改 | 重新获取对应文件 |
| `No Orbbec device reported` | 相机没插/被占用 | 换 USB3 口，关闭其他相机软件，`device` 重试 |
| `guarded reconstruction requires a real closed left/right observation` | 没有真实闭口侧脸帧 | 重采集，确保第 2、3 段真的转了头（不能被程序用正面帧冒充） |
| `face_bbox_not_detected` | 某一帧没检出人脸 | 重采集该段；或 `--frames-per-view` 调大 |
| 重建中途失败 | 看 `run_report.json` 的 `error` 和 `logs/` | 修复后重跑；旧输出目录不会被覆盖 |
| `no feasible Loop base within ORIGINAL total budget` | 深度噪声太大/约束太紧 | 改善光照与距离后重采集；不要放宽预算参数 |
| 网格看起来"糊"或有波纹 | 采集帧不足、侧脸角度小、存在模型先验区域 | 看 `support.json`；补采侧脸；不要期待未观测区域有真实细节 |

每次运行都是新目录；失败不会破坏已有结果。日志在 `runtime/outputs/<run>/logs/`。

---

## 9. 测试

```pwsh
.\runtime\env\python.exe -m pytest tests -q
```

测试只覆盖契约层（版本一致、命令集合、固定阶段顺序、strict 接受规则、路径解析、
时间边、无效深度、单网格导出、单位检查），**不**验证几何精度，也不跑 GPU 重建。

交付输出的独立复核（只读，不重算）：

```pwsh
.\runtime\env\python.exe tools\verify_delivery.py runtime\outputs\<run>\delivery
```

它检查：目录里只有一个网格、顶点/面有限且索引合法、米级尺度、参考帧与 68 点可从
网格重算一致、支持度计数与顶点数一致、manifest 版本与 `VERSION` 一致。

---

## 9.1 实测耗时（RTX 3060 Laptop 6 GB，20 帧真实采集）

| 阶段 | 耗时 |
|---|---|
| 原始采集导入 + 准备（TUFA/YuNet/UniFace，CPU 推理） | 35 s |
| 正面锚 + 共享身份（C，GPU） | 328 s |
| 固定身份运动 → 低频身份 → 运动重估（GPU） | 477 s |
| 密集 Loop + 真实深度残差（CPU 求解，GPU 仅初始化） | 1 811 s |
| **合计（`reconstruct` 全程，含导出）** | **约 44 min** |
| 仅再导出（`export`） | 约 2 s |

口径：`run_report.json` 中每个阶段的墙钟时间；GPU 段包含进程启动与导出。
采集本身另有约 40 s（4 段，含人工倒数）。

---

## 10. 版本

`0.2.0`。`VERSION`、`--version`、`README`、`manifest.json` 四处一致。
历史见 `CHANGELOG.md`：本次变化是**最终流程整合、入口精简、输出收敛**，
不包含任何算法精度提升的声明。
