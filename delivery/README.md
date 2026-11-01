# facecap_airway_delivery

用一台 Orbbec Gemini 215 采集成人头部 RGB-D 数据，离线重建出**米制高分辨率人头网格**：
**一份闭口、一份张口**，同一共享身份、同一套拓扑。

```
真实相机采集 → 离线重建 → 导出闭口网格 + 张口网格
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
| 磁盘 | 运行目录约 7 GB（主要是 Python 环境），一次采集约 20 MB，一次输出约 50 MB（两个网格） |
| 采集距离 | 0.3–0.7 m，正对镜头 |

---

## 3. 目录结构

```
facecap_airway_delivery/
├─ README.md                    本说明
├─ VERSION                      0.4.0
├─ CHANGELOG.md
├─ VALIDATION.md                本机实际验证记录
├─ docs/FAST_PATH_REPORT.md     0.3.0 加速版算法：实测、消融与剩余热点（权威）
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
│  ├─ ray_index.py              可见性内核：投影栅格 + 精确 Möller–Trumbore
│  ├─ geometry_c.py / geometry_d.py / surface_v2.py   对应、保护、Loop、证据
│  └─ ...
├─ core/                        相机 SDK 封装、FLAME 装载、人脸/颈部解析
├─ third_party/                 FLAME 与 TUFA 源码（含各自 LICENSE）
├─ tests/test_contracts.py      小型核心契约测试
├─ tools/verify_delivery.py     只读复核交付输出
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
.\headrgbd.cmd --version      # headrgbd 0.4.0 (facecap-inspired-v2-strict)
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
5. 导出 `runtime/outputs/<采集名>_reconstruction/delivery/` 下的闭口与张口两个最终网格。

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

#### 快速流程（0.3.0 起可用）

```pwsh
# 约 5 分钟的完整重建：跳过密集残差求解，中频校正跑 1 轮
.\headrgbd.cmd reconstruct --capture <dir> --fast --mid-rounds 1

# 更保守：中频 2 轮（更慢，见 §7.2 的代价表）
.\headrgbd.cmd reconstruct --capture <dir> --fast --mid-rounds 2

# 不加 --fast 就是默认流程，与 0.2.0 的算法链一致，只是换掉了射线求交内核
.\headrgbd.cmd reconstruct --capture <dir>
```

`--fast` 不是"质量旋钮"，而是**一个明确的算法定义**：交付面取 Loop 基面，不再求解密集
残差。被跳过的计算会在 `report.json` 里标成 `not_executed` 并写明原因，不会伪装成
"求解成功、位移为 0"。取舍数据见 §7.2。

#### 正侧面联合形状（0.4.0 起可用）

```pwsh
# 正面与侧面共同决定同一个头面部形状（推荐）
.\headrgbd.cmd reconstruct --capture <dir> --fast --mid-rounds 1 --shape-fusion joint

# 对照：形状只由正面决定（0.3.0 及以前的默认行为）
.\headrgbd.cmd reconstruct --capture <dir> --fast --mid-rounds 1

# 侧面消融：同样的自由度，但侧面深度项权重为 0
.\headrgbd.cmd reconstruct --capture <dir> --fast --mid-rounds 1 `
  --shape-fusion joint --side-depth-weight 0
```

`--shape-fusion` 与 `--fast` 是**两个正交维度**：前者决定形状从哪来，后者决定算到哪一步。
默认仍是 `guarded`（只正面），要显式打开 joint。机制与实测见 §7.4。

---

## 6. 输出：闭口 + 张口，各一份最终网格

`runtime/outputs/<采集名>_reconstruction/delivery/`：

| 文件 | 内容 |
|---|---|
| `head_closed_m.obj` | **闭口最终网格**：闭口参考帧的密集表面（20 018 顶点 / 39 904 面），单位**米** |
| `head_open_m.obj` | **张口最终网格**：张口参考帧的同一套密集表面，单位**米** |
| `manifest.json` | 运行清单（见下），两个状态都在 `mesh.states` 下 |
| `support.json` | 每个顶点的真实观测支持度标签（两网格共享顶点顺序） |
| `reference_landmarks.json` | 两个状态各自的 68 点（模型侧 + 原始观测侧）与刚体变换 |

同目录的 `run_report.json` 与 `reconstruction/` 是审计记录（阶段命令、耗时、日志），
**不是**给用户看的中间网格。粗网格、锚点网格、Loop 基面、插值网格、对比网格都不再导出。

两个网格来自**同一次运行、同一个共享身份、同一套拓扑**（面表完全相同），
差别只来自参考帧本身的姿态/表情/下颌：

| | 闭口 | 张口 |
|---|---|---|
| 参考帧 | `closed_front_2` | `open_front_0` |
| 内唇距（模型） | 1.92 mm | 26.95 mm |
| 内唇距（原始观测） | 2.39 mm | 26.47 mm |

任何一步缺失都会**直接报错**，不会用闭口帧冒充张口、也不会用模型先验伪造张口观测：
张口参考帧必须是真实存在、且内唇深度有效的张口帧。

### 6.1 单位、坐标与参考帧

- **单位**：米（OBJ 顶点数值即米）。深度原始数据为毫米，内部换算后求解。
- **坐标系**：`X 向右 / Y 向上 / Z 向后`（相机右手系，Z 朝相机后方，即脸在 z<0 一侧）。
  单位与坐标写进 `manifest.json` 的 `mesh.unit` / `mesh.states.<state>.coordinate_system`。
- **参考帧**：闭口取 `closed_front_*` 中真实存在且深度有效的那一帧（默认 `closed_front_2`），
  张口取 `open_front_*` 中的一帧（默认 `open_front_0`）。
  两个网格各自在**自己的参考帧相机坐标**下，**不是**世界坐标，也不是扫描仪坐标。
- **不要直接把两个网格叠在一起比较**：它们分属两个不同的参考相机坐标系，
  两者之间没有做任何外部配准；`reference_landmarks.json` 里给出的
  `animated_model_to_reference_camera_4x4` 只是各自状态内的纯刚体变换。
- 要对比闭口/张口，请在**同一坐标系**下用 FLAME 参数或标点做刚体对齐后再比。

### 6.2 manifest.json 记录什么

软件版本与算法来源、配置参数、输入采集、两个参考帧、每个状态的网格路径/单位/坐标系/
顶点数/面数/包围盒、内唇距（模型与观测）、`success | degraded` 状态、关键质量警告、
支持度统计、颈部独立补片、限制说明。

### 6.3 支持度怎么读

`support.json` 的 `labels` 三类，`vertex_label` 与**两个网格共同**的顶点顺序一一对应：

| 标签 | 含义 |
|---|---|
| `distinct_closed_bursts_and_angle_supported` | 至少两次闭口短段、且转角差 ≥ 8° 的真实观测支持 |
| `single_or_correlated_burst_plus_prior` | 只有单次/相关性很强的观测，叠加模型先验 |
| `unobserved_completion` | **没有真实观测**，完全由模型补全 |

**插值出来的标签不算新增真实观测**；`confidence` 只是启发式权重，不是标定概率。
支持度在同一次共享身份拟合中累计，**不是**按闭口/张口分别重算，
因此只在张口或只在闭口阶段看到的区域无法在这份文件里区分。
张口状态的内唇、口腔内侧本来就缺少深度，属于模型先验。

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
| 6 | 密集细化 | `fit_d --dense-mode loop-research` | 同拓扑 Loop 基面 + 真实深度残差，产出闭口/张口两个最终网格 |

必须保留的约束（不是可选优化）：

- 无效内唇深度**不产生**伪三维约束（NaN + 显式掩码，绝不当 0）；
- 不直接平均不同表情的原始深度；
- 固定射线覆盖：接受一个新步时**覆盖不得减少**、MSE 不得超 2 %+0.002 mm²（strict-coverage）；
- 位移上限 3 mm、法向变化 ≤ 35°、有向面积 ≥ 5 %（防翻转）；
- 时间约束**不跨**采集分段缺口、不跨追踪丢失帧；
- 颈部单独处理（独立跟踪、按段输出观测补片，不参与身份拟合）；
- 每帧仍导出真实支持度标签。

### 7.1 可见性内核（0.3.0 起）

所有可见性查询都从相机中心射出，所以一条射线完全由它反投影到的三维点决定。据此把
三角形投影到相机视线平面（`u = x/-z`，`v = y/-z`）建均匀栅格索引，窄相位用精确的
Möller–Trumbore。实现：`head_rgbd/ray_index.py`。

- 实测 **0.644 ms/条 → 0.0072 ms/条**（约 90×，含索引构建）；
- 与旧的 rtree 查询在真实网格 + 真实深度采样射线上：命中数相同、命中面 **100 %** 相同、
  交点差 **0.15 µm**；
- 它位于 `ray_surface` 与 `front_protected_vertices` 之下，所以身份拟合、运动、正面保护、
  覆盖验收和最终支持度**全部**受益——这也是前段能从 11 分钟降到 2 分钟的原因，不只是密集段；
- 语义未变：首命中、射线编号、双面查询、遮挡与缺失语义、`max_depth/max_plane` 门限全部保留。

### 7.2 快速流程：跳过什么、保留什么

| 项 | 默认流程 | `--fast` |
|---|---|---|
| 低频身份校正 | 2 轮 | 2 轮（**保留**，位移 0.28 / 0.27 mm） |
| 中频校正 | 3 轮 | `--mid-rounds` 轮，默认 2 |
| 密集残差 QP | `front_anchor_dense` + `guarded_dense` | **不执行**，交付 Loop 基面 |
| regularized 中间支持度 | 计算 | 不计算（只服务预览图与复用 sidecar） |
| 几何/解剖/覆盖验收 | 全部 | **全部保留** |

同一采集、同一上游下的取舍（闭口全脸 MAE，同帧同观测）：

| 配置 | 完整耗时 | MAE (mm) |
|---|---|---|
| 中频 0 轮，无密集残差 | 211.5 s | 1.2927 |
| **中频 1 轮，无密集残差（`--fast --mid-rounds 1`，推荐）** | **298.3 s** | **1.2874** |
| 中频 2 轮，无密集残差（`--fast` 默认） | 403.3 s | 1.2861 |
| 中频 2 轮，有密集残差（默认流程） | 865.4 s | 1.2750 |

读法：中频从 0 加到 2 轮只买回 0.0066 mm，密集残差买回 0.0111 mm；被省略的部分在这份
采集上合计只值 **0.018 mm**。`--mid-rounds` 默认取 2 是保守值，若时间优先用 1。

### 7.3 `report.json` 里的执行证据

| 字段 | 含义 |
|---|---|
| `fast_path` | `{enabled, dense_residual, mid_frequency_rounds, regularized_evidence}` |
| `field_logs.dense.status` | `not_executed` + `reason`（跳过时），不是"求解成功" |
| `field_logs.regularized.status` | `skipped_no_free_vertices`（零自由度，保留初始场） |
| `field_logs.*.history[].timing` | 对应关系 / 系统装配 / QP setup / QP 求解 / 线搜索 / 几何检查 / 固定射线 的分项秒数 |
| `field_logs.*.history[].actual_update_mm` | 该轮接受后表面的**真实最大位移**（mm），不是步长系数 |
| `field_logs.*.stop_reason` | 提前停止原因；判据是"实际位移小于一个深度量化步长 且 固定射线 MSE 改善 < 1e-4"，**步长系数不参与判据** |
| `timing_sections` | 阶段内各分区的墙钟耗时，分区之和等于阶段总耗时 |


### 7.4 正侧面联合形状融合（0.4.0）

0.3.0 及以前，侧面观测对**共享形状**没有任何贡献，这是结构性的：`fit_c` 在侧锚模式下
`shape.grad=None`（身份被冻结），正面帧被移出身份阶段，`identity_v2` 用硬掩码锁住正面
覆盖的顶点，`fit_d` 的低/中频控制场只吃正面帧。实测后果是
`changed_side_detail_vertices = 0`——侧面阶段一个顶点都没改。

`--shape-fusion joint` 的改法：

| 环节 | guarded | joint |
|---|---|---|
| FLAME 身份参数 | 冻结 | 每个闭口视角一票，共同更新 |
| 正面前锚 | 最终值（断言顶点相等） | 只作**初始化** + 有限强度软先验（`--joint-shape-prior`，默认 0.02） |
| 正面覆盖区域 | 硬锁（`locked`） | 连续置信度 `--joint-front-confidence`（默认 0.15），顶点仍在自由集 |
| 低/中频控制场 | 只用正面身份帧 | 全部闭口身份帧 |
| 张口帧 | 不参与身份 | 不参与身份（同上） |

`soft_lock` 不是放宽几何约束（位移/坡度/法向/防翻面照常检查），只是把"这块区域的数据该
信多少"从 0/1 改成连续权重，让可靠的多视角观测能够压过先验。

实测（20 帧参考采集；joint+fast 249.8 s，guarded+fast 265.0 s）：

- **正面硬锁顶点 2797 → 0**；局部场实际改动 1776 个顶点、最大 2.999 mm（guarded 为 0）；
- 闭口精度相对 guarded **持平**（三视角最大 |ΔMAE| 0.040 mm，在门限内）；
- 张口 MAE 1.737 → **1.581 mm**，覆盖率 95.74% → **98.94%**，缺失 1278 → **319**；
- 正面下颌/轮廓 P95 9.384 → **7.756 mm**；
- **鼻部没有改善**（三视角 0.99–1.06 mm 无差），不宣称。

侧面消融（同一份代码、同自由度、`--side-depth-weight 0`）证明侧面数据是必要的：去掉后
侧面嘴唇恶化 0.7–1.0 mm、下颌轮廓恶化 0.65–0.84 mm。完整数据见
`docs/JOINT_FUSION_REPORT.md`。

注意：联合模式下正面帧的姿态/表情也自由，`joint_anchor_update_mm` 可达 9–11 mm，**不能**
当作形状改动；判断纯身份变化要看 `joint_shape_delta_from_anchor`。

> `tools/import_multiview_capture.py` 可把 `airway3d.rgbd.multiview.v1` 采集转成本流程的
> capture 目录（每个视角取 1 张参考帧）。

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

## 9.1 实测耗时（同一台机器，RTX 3060 Laptop 6 GB，20 帧真实采集）

三列是同一采集、同一台机器、同一份 `prepared` 序列的实测；单位秒。

| 阶段 | 0.2.0 | 0.3.0 默认 | 0.3.0 `--fast --mid-rounds 1` |
|---|---|---|---|
| 准备（TUFA/YuNet/UniFace，CPU 推理） | 33.5 | 36.8 | 36.2 |
| 正面锚 `fit_c --front-only` | 101.6 | 29.9 | 28.6 |
| 共享身份 `fit_c --front-anchor` | 241.0 | 38.8 | 38.3 |
| 固定身份运动 `motion_v2` | 139.2 | 21.8 | 21.8 |
| 低频身份 `identity_v2` | 137.3 | 19.7 | 19.8 |
| 运动重估 `motion_v2 --identity-field` | 154.8 | 22.2 | 22.0 |
| 密集 Loop + 残差 `fit_d`（CPU 求解） | 1131.6 | 689.0 | 124.2 |
| **合计（含导出两个网格）** | **1945.7 s ≈ 32.4 min** | **865.4 s ≈ 14.4 min** | **298.3 s ≈ 5.0 min** |
| 加速比 | 1.00× | **2.24×** | **6.52×** |

两段收益分开报，不要相乘也不要相加：

- 只替换射线内核、不跳过任何阶段：1945.7 → 865.4 s（**2.24×**）；
- 在此之上再加 `--fast --mid-rounds 1`：865.4 → 298.3 s（**2.90×**）。

`export`（只再导出，不重跑拟合）仍约 2 s。

口径：`run_report.json` 中每个阶段的墙钟时间，含进程启动与导出。0.3.0 的密集段还有
`report.json:timing_sections` 给出阶段内分区耗时。本表取代此前 README 中"约 44 min"的
记录——那是更早一次不同条件的运行，与这里的可复现配置不是同一组数字。

剩余热点：`fit_d` 的中频 QP（936 个自由变量、约 22 万约束行、单轮 9 800–16 600 次
OSQP 迭代）。要再压缩需要改 QP 的缩放/条件数，本轮未做，因为会改变可行域上的数值行为。

逐阶段耗时表、消融实验、质量对比口径与"哪些早期结论已被推翻"，见
`docs/FAST_PATH_REPORT.md`。

---

## 10. 版本

`0.4.0`。`VERSION`、`--version`、`README`、`manifest.json` 四处一致。

本版本新增**正侧面联合形状融合**（`--shape-fusion joint`）：正面、左侧、右侧观测共同决定
同一个 FLAME 身份与局部修正场，替代"冻结身份 + 硬锁正面"的旧规则。默认仍为 `guarded`，
需要显式开启。**算法名未变**（`facecap-inspired-v2-strict`），变的是形状证据的用法。

0.3.0 是性能版本（射线内核替换 + 快速流程，32 分钟 → 5 分钟以内），并已知交付网格与
0.2.0 不是逐位一致——流水线在重心阈值 `bc > .05` 与 QP active set 处对 1e-7 量级扰动
敏感，0.2.0 的验收结论不沿用。两版的质量对比基准都是"同代码、同输入、同口径"的同机
运行，见 `CHANGELOG.md` 对应段落。

> 0.2.0 发布时默认只导出闭口网格；同版本的后续修订按要求改为**闭口 + 张口各一份**，
> 版本号不变、不新增研究模式，算法与参数完全未改。
