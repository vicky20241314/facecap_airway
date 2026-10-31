# facecap_airway / withB — 单台 RGB-D 人头重建 0.1.0

使用 **Orbbec Gemini 215** 采集真实 RGB-D，采集结束后离线拟合人头几何，导出
米制网格、逐帧参数、参考帧标点和观测支持度。项目采用全序列共享的 FLAME
身份，以及逐帧独立的头部姿态、表情和下颌开合。

这是 **FaceCap 启发的 FLAME 离线原型**，不是完整 FaceCap 复现，也不是实时
重建系统。纹理、模板补全或训练损失下降不构成真实几何精度证据。

- 仓库：<https://github.com/vicky20241314/facecap_airway>
- 源码目录：`D:\proiect\withB`
- 独立运行目录：`D:\proiect\withB-runtime`
- 版本唯一来源：根目录 `VERSION`；`--version`、Python 包和新运行报告读取此文件。
- 默认算法：C 的 **guarded 正面保护模式**。自由身份模式仅供研究对照。

## 1. 当前能做到什么

| 验证项目 | 状态及含义 |
|---|---|
| 迁移实现 | 已完成：相机、准备、拟合、细化、颈部、导出和审计均有独立入口 |
| 既有真实数据离线回归 | 历史审计记录为流程通过；几何效果验收失败，见第 10 节 |
| 相机连接和原始保存 | 已验证 Gemini 215、HW D2C、内参与时间戳保存 |
| 新真人采集到重建 | 采集/拟合产物已完成补充契约验证；原一键运行失败记录保留，本次未整链重跑，见第 10 节 |
| 真实侧面脸型、下颌角、前后厚度改善 | **未通过验收，不作改善声明** |
| 独立真实精度 | **未验证**：没有独立外参、人工轮廓或参考扫描 |

程序返回成功只说明对应执行/契约检查通过。它不等于几何效果验收通过。
`run` 的 `geometry_effect_acceptance` 默认是 `not_evaluated`；`verify` 检查的是
导出与建模契约；`regression` 才额外比较训练帧诊断。三者都不是独立精度验收。

## 2. 首次使用与一条启动命令

### 2.1 硬件与已验证环境

| 项目 | 本机配置 |
|---|---|
| CPU | Ryzen 7 5800H |
| GPU | RTX 3060 Laptop，6144 MiB（6 GiB）显存 |
| 相机 | Orbbec Gemini 215，USB 3.0 |
| 操作系统 / shell | Windows / PowerShell |
| Python | 3.10.22 |
| PyTorch / CUDA build | 2.0.1 / 11.7 |
| Orbbec Python SDK | `pyorbbecsdk==2.2.8` |

完整固定依赖见 [requirements.txt](requirements.txt)。已准备的解释器为
`D:\proiect\withB-runtime\env\python.exe`，不需要激活环境，也不依赖旧研究环境。

### 2.2 首先检查版本、模型和环境

在任意当前目录执行：

```powershell
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' --version
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' assets
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' status
```

期望版本输出 `headrgbd 0.1.0`。`assets` 核验 7 个资产的存在性和 SHA256；
`status` 保存解释器、依赖、CUDA、GPU 快照和路径配置。资产不符时应修复资产，
不能随意改清单哈希绕过检查。

### 2.3 连接相机并启动完整流程

确认相机朝向真人后执行：

```powershell
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' capture-run --session-name head --device cuda
```

该入口按以下顺序执行，GPU 拟合阶段串行运行：

```text
相机连接 / HW D2C / 内参检查
    ↓
四段引导采集 → 原始 RGB、毫米深度、时间戳、分段状态保存
    ↓
prepare → 人脸检测、68 点、语义、有效深度、头颈独立重叠跟踪
    ↓
C-front → 真实正面锚：共享身份 + 逐帧姿态/表情/下颌 + 有界细化
    ↓
C-guarded → 固定正面锚，估计斜侧运动，细化新增可见区域
    ↓
独立颈部开放表面 + 米制 OBJ / PLY / NPZ / JSON
    ↓
几何契约检查 + 参考标点复核 → run_report.json
```

仅测试相机连接和帧保存：

```powershell
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' probe --samples 3
```

`probe` 成功也不代表画面有人脸或能够重建，仍需检查实际 RGB、有效深度和人脸区域。

## 3. 目录、环境与模型资产

### 3.1 源码与运行数据分离

```text
D:\proiect\withB\
  VERSION                    唯一版本来源
  headrgbd.ps1                PowerShell 统一启动器
  requirements.txt           固定 Python 依赖
  config\defaults.json       CLI 默认采集/拟合参数
  config\assets.json         资产相对路径、用途、SHA256
  head_rgbd\                 准备、B/C 算法、验证、审计与 CLI
  core\                      相机、采集、检测与必要拟合工具
  third_party\flame\         模型加载/变形代码及许可
  third_party\tufa\          标点网络代码、小型 prompt 及许可
  tests\                     小型契约测试
  MIGRATION.md               迁移、历史验证与清理盘查记录
  MIGRATION_MANIFEST.txt     交付文件字节数与 SHA256（不含清单自身）

D:\proiect\withB-runtime\
  env\python.exe             独立 Python 环境
  assets\                    下列 7 个模型/嵌入资产
  captures\                  新采集原始数据
  outputs\                   每次独立重建/回归输出
  logs\                      环境、资产与发布检查
```

旧研究 worktree 和旧回放目录不是算法运行依赖。历史文档里保留的旧路径仅用于
追溯研究来源；新采集流程所需代码、解释器和模型均在上述两个目录中。
外部 `sequence.json` 必须连同其相对引用的逐帧 NPZ 一起提供，单独复制 JSON 不够。

所有新输出使用新目录。统一流水线以独占方式创建运行根，拒绝复用已有目录；
不要为了重试覆盖原始采集或旧失败报告。

### 3.2 改用另一个独立运行目录

```powershell
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' -Runtime 'E:\head-runtime' status
```

启动器要求该目录有 `env\python.exe`，并把 Python 工作目录切到源码根，结束后
恢复调用者目录。直接调用 Python 时，从源码根运行：

```powershell
Set-Location -LiteralPath 'D:\proiect\withB'
& 'D:\proiect\withB-runtime\env\python.exe' -m head_rgbd --runtime 'D:\proiect\withB-runtime' --help
```

Python 的路径优先级为 `--runtime` → `HEADRGBD_RUNTIME_DIR` → 源码父目录下
的 `withB-runtime`。PowerShell 启动器以自己的 `-Runtime` 参数设置运行目录，
默认使用 `D:\proiect\withB-runtime`；改路径时请显式传参。

### 3.3 在新机器上创建环境

已有本机环境无需重装。若要重新部署，请创建**新的**运行根，避免修改共享环境。
当前启动器采用 prefix 环境的平铺 `env\python.exe` 布局；普通 Windows venv
使用 `Scripts\python.exe`，不能直接当成这个布局。

下面是新建 Conda/Miniforge prefix 环境的安装命令，不能对已有共享环境照抄覆盖。
这是重建环境步骤，不是“仓库自带 Python/模型”的承诺：

```powershell
conda create --prefix 'D:\proiect\withB-runtime-new\env' python=3.10.22 pip -y
& 'D:\proiect\withB-runtime-new\env\python.exe' -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
& 'D:\proiect\withB-runtime-new\env\python.exe' -m pip install -r 'D:\proiect\withB\requirements.txt'
```

安装完成后放置合法获得的模型资产，执行 `-Runtime ... assets` 和 `status`。
新环境仍需实际验证，不能将本机既有环境的验证结论自动套用到新环境。

### 3.4 必要资产与许可

以下路径均相对于 `<runtime>\assets`：

| 路径 | 用途 |
|---|---|
| `flame\flame2023.pkl` | FLAME 身份、表情、下颌模型 |
| `flame\flame_static_embedding.pkl` | 静态标点嵌入 |
| `flame\flame_dynamic_embedding.npy` | 动态轮廓标点嵌入 |
| `models\yunet\face_detection_yunet_2023mar.onnx` | YuNet 人脸框检测 |
| `uniface\parsing_resnet34.onnx` | UniFace 面部/颈部语义 |
| `tufa\Face_prompt_base.pth` | TUFA 68 点网络 |
| `tufa\Prompt\shape_68.npz` | TUFA prompt |

准确哈希见 [config/assets.json](config/assets.json)。不自动从旧 worktree 寻找模型，
也不能把缺失模型当作成功降级。相应本地加载器在缺少资产时明确报错。

- `third_party/flame` 的代码许可见其 [LICENSE](third_party/flame/LICENSE)；
  **代码许可不代替 FLAME 模型许可**，模型需按原许可取得和使用。
- TUFA 代码附带 [GPL-2.0 许可](third_party/tufa/LICENSE)，复用和分发须遵守其要求。
- YuNet、UniFace、TUFA 权重等资产继续受各自上游许可约束。
- Git 不含 Python 环境、模型权重、采集、重建网格或旧研究归档。
  `third_party/tufa/Prompt/shape_*.npz` 是随源码保留的小型必要 prompt，已明确放行。

## 4. 真人采集操作

### 4.1 四段默认采集

| 顺序 | 参数标签 | 操作 |
|---|---|---|
| 1 | `closed:front` | 正对相机，自然闭唇，避免抿唇和咬紧 |
| 2 | `closed:left` | 保持闭口，头向画面左侧转约 25–35° |
| 3 | `closed:right` | 保持闭口，头向画面右侧转约 25–35° |
| 4 | `open:front` | 回到正面，明显张口，身体尽量稳定 |

这里的 left/right 是引导标签，不是独立姿态测量。实际 yaw 在离线拟合后记录。
现有研究数据约为 ±29° 斜侧，不是 90° 完全侧脸，也不是完整连续转头视频。

默认每段 5 帧、保存间隔 0.12 秒、相机预热 0.8 秒、段前倒计时 2 秒。
实际时间戳以采集记录为准，不能用期望间隔代替真实间隔。

### 4.2 每段如何开始

1. 连接 USB 3.0，相机朝向脸部。脸和下巴完整出现在画面中，尽量保留颈部。
2. 终端显示本段动作后，准备好姿势，先在**终端按 Enter**。
3. 倒计时结束，出现 RGB 预览；把焦点切到预览窗口，确认脸的位置后按 **SPACE**。
4. 保持姿势直到本段保存完成，再根据终端提示准备下一段。
5. 等待预览开始阶段可按 **ESC** 取消。异常/取消会尽可能留下 partial manifest。

当前预览窗口显示 RGB，不显示完整的深度叠加验收视图；不要把看到 RGB 当作深度
有效。可先执行 `probe` 检查保存帧，再查看 `session.json` 的有效深度统计和离线
`sequence.json` 的人脸深度像素、排除/降级原因。

采集阶段主要检查帧尺寸、配置和元数据；重叠跟踪及角度估计分别在准备/拟合阶段完成，
当前没有在实时预览里实现完整的质量反馈闭环。

`--auto` 跳过 Enter/SPACE 等待；`--no-preview` 关闭预览，但不自动跳过终端等待。
无人配合时自动采集可能只得到空场景，不能据此宣称真人链路验证成功。

### 4.3 采集约束与原始记录

必须包含闭口正面、张口正面和至少一个真实闭口斜侧段；默认同时采左右斜侧。
不能复制正面、旋转正面图或使用模板补全冒充真实侧面。

每次采集写入 `captures\<时间戳>_<session-name>\`：

- `session.json`：设备、SDK、HW D2C、相机参数、单位、分段、质量摘要。
- `segment_00_closed_front.npz` 等：原始 RGB、毫米深度、逐帧内参、时间戳。
- `segment_00.json` 等：逐帧检测、有效深度、同步差等元数据。
- 中断时可能只有 `session.partial.json`；不将它当作完整采集直接拟合。

记录 host monotonic 时间及可用的硬件/全局/系统时间戳、颜色/深度帧号和同步差。
没有的硬件字段保留缺失标记。相同 RGB/深度尺寸与 SDK HW D2C 配置只是对齐证据，
不等于独立物理标定或外参验收。

## 5. 输入数据契约与坐标

### 5.1 三种输入层次

| 输入 | 契约 | 适用入口 |
|---|---|---|
| 本程序原始采集 | `head_rgbd.capture.v1`，`session.json` + 同目录分段 NPZ | `prepare --capture` / `run --capture` |
| 外部逐帧块 | `head_rgbd.raw.v1`，JSON 的 `chunks` 引用同目录 NPZ | `prepare --frame-manifest` |
| 已准备观测 | `head_rgbd.sequence.v1`，`sequence.json` + 逐帧 NPZ | `run --sequence` / `reconstruct` |

原始分段 NPZ 必须至少包含：`rgb`（N×H×W×3、uint8 RGB）、`depth`
（N×H×W、毫米深度）、`intrinsics`（N×4，顺序 fx/fy/cx/cy）、
`timestamps_s`（N 个真实时间戳）。外部 manifest 需要声明毫米深度、RGB-D 已对齐、
坐标约定，并为各块提供 `state`、`view`、`segment`、`view_id` 和 `file`。
路径必须满足本地加载器的同目录限制；字段细节以 `prepare.py` 为准。

准备结果逐帧保存 RGB、深度、内参、二维标点 `uv`、米制三维标点 `xyz`、
逐点 `valid`、面部/颈部掩码、时间戳和来源帧号。`sequence.json` 对所有帧记录
候选/排除状态、原因、重叠、失跟、重定位和降级，不能只保留好帧而丢弃审计。

### 5.2 毫米深度到米制几何

RGB 与深度通过 HW D2C 对齐到颜色坐标，当前采集路径为 1280×720。
坐标约定统一为 **X 向右、Y 向上、Z 向后**（`X_right_Y_up_Z_backward`）。
像素 `(u,v)` 的正深度先转为米 `z = depth_mm / 1000`，再反投影：

```text
X =  (u - cx) * z / fx
Y = -(v - cy) * z / fy
Z = -z
```

因此镜头前的点通常具有负 Z。OBJ/PLY 顶点、三维标点和参数平移均以米表示；
报告中的残差/唇距在标为 `mm` 时才乘 1000。导入三维软件时选择米，不能再次缩放。

深度 0、非有限值和不在有效距离范围内的深度不会作为几何约束。
有效距离基本门槛是 `0.1 < z < 1.5 m`，还需语义、边界、遮挡等检查。
无效标点 XYZ 保留缺失语义，JSON 用 `null`，不能变成零坐标约束。

## 6. 算法流程详解

### 6.1 准备：检测、语义与独立头颈跟踪

1. YuNet 检测人脸框，TUFA 检测 68 个二维标点，UniFace 生成面部与颈部语义。
2. 根据对齐深度和逐帧内参反投影标点，逐点检查有效性。人脸框、有效面部深度等
   不满足契约时排除整帧；局部标点或颈部无效可降级，仍保留可靠头部观测。
3. 在相邻帧上半脸区域做 KLT 重叠跟踪：最多 160 个角点，至少 8 对有效点，
   RGB-D 刚体重叠中位误差小于约 8 mm 才记为 `tracked`。
4. 失跟允许使用标点重新初始化。排除帧、非递增时间戳、明显时间缺口建立 barrier。
5. 颈部用自己的语义和重叠变换；颈部失跟不终止头部拟合。

时间约束边必须同时满足：同 segment、`0 < Δt < 0.2 s`、当前相邻跟踪为
`tracked`、两帧可用且无 barrier。不会把四段不连续采集串成一条平滑运动。
长时跟踪和完全侧脸跟踪仍未验证。

### 6.2 帧的角色：共享身份与独立运动

每个采集段按质量、角度/时间分布选关键帧，默认每段最多 3 帧。
实际选择及原因写入 `report.json.frames`，不能用配置数代替实际参与数。

- **全序列共享**：50 维 FLAME 身份，以及共享的有界局部法向细化场。
- **逐帧独立**：30 维表情、3 轴下颌旋转、刚性旋转和平移。
- **身份更新**：仅允许闭口身份关键帧；身份阶段排除口部观测，并固定表情/下颌。
- **运动/表情更新**：所有可用帧都参与，包括非关键帧和张口帧；张口帧不更新身份。

闭口和张口是同一身份的不同动画状态，不是各拟合一个互不相关的人头。
原始深度不会在不同表情之间直接平均。

### 6.3 大尺度拟合：五阶段交替优化

默认每个 C 拟合预算为 240 步，约按 `[0,40,110,150,200,240]` 切分。
默认 guarded 冷启动包含一次 C-front 和一次 C-guarded，不能把 240 步误解成两者合计。

| 阶段 | 主要任务 | 身份/表情处理 |
|---|---|---|
| `motion_initial` | 标点驱动的初始相对姿态与口型 | 固定身份，逐帧估计运动/表情/下颌 |
| `identity_large` | 先恢复大尺度身份与配准 | 闭口关键帧更新共享身份，固定表情/下颌 |
| `expression_reestimate` | 身份稳定后重新解释表情 | 固定身份，更新逐帧状态 |
| `identity_surface` | 使用连续三角形表面对应细化身份拟合 | 闭口关键帧更新共享身份，仍排除口部 |
| `motion_final` | 最终逐帧姿态、表情和下颌调整 | 固定身份 |

损失包括有效三维/二维标点、鲁棒表面点到面残差、身份/表情/下颌先验，以及仅对
可靠相邻帧启用的时间项。身份阶段的正面权重为 3、斜侧为 1，避免斜侧帧数
压过正面观测。权重不改变几何接收门槛。优化还限制参数幅度并检查非有限目标。

### 6.4 三角形对应、可见性与鲁棒残差

原 B 的近邻顶点对应存在离散偏差。C 从相机原点沿真实深度像素射线查询网格
**第一个三角形交点**，使用面索引和重心坐标表示连续表面位置。

拟合观测先经过语义、有效距离、掩码腐蚀和深度跳变检查，再拒绝无首交、背面、
掠射和深度不一致区域。主要门槛包括入射因子 `>0.15`、深度差约 8 mm、
点面差约 5 mm。具体检查顺序和统计见 `geometry_c.py` 与 `correspondence_log`。

ICP 对应定期更新；一次更新之后固定重心坐标与面法向，用 Smooth-L1 点到面项
优化，而不是强迫真实点贴到最近顶点。首交查询处理当前网格自遮挡，不能解决
所有相机外遮挡或错误语义。斜侧运动还检查标点首交可见性（约 2 mm 窗口）和
三维外点（约 12 mm）；表面项启动后降低不确定标点的权重。

### 6.5 口部：避免身份吸收张口表情

- 内唇深度逐点判断；同一唇部深度跳变超过约 6 mm 的标点单独拒绝。
- 可靠内唇三维项降权；内唇深度无效时不进入三维损失，可靠二维检测仍可解释口型。
- 已知闭口状态使用约 `0–2 mm` 的宽区间软约束，不把自动深度标点当作精确真值。
- 身份阶段和共享局部细化排除口部，口部身份法向偏移固定为零。
- 每帧表情/下颌负责张口差异，张口口部残差不能反向改变闭口身份。

张口深度可能测到口腔边界或没有有效回波。零填充只是数值占位时，也必须有有效性
掩码阻止其进入损失；最终侧车明确保留 `null`，不能伪造三维观测。

### 6.6 局部身份细化与防翻转

大尺度模型稳定后，只从可靠闭口关键帧建立共享法向位移方程。先解释逐帧姿态与
表情，再通过三角形重心系数和当前动画法向把观测联系到公共身份细化场。
采用鲁棒重加权和相邻顶点平滑，不直接平均不同姿态/表情的深度。

约束保持保守：至少 2 帧真实支持才允许改变顶点，法向位移限于 **±3 mm**，
口部不做身份位移。位移限幅本身不能防止小三角形翻转，因此另外检查每个姿态下
的有向三角形面积：至少保留原来的 5%；必要时局部衰减共享偏移，最多 80 轮。
无法满足保护时明确失败，不通过放松阈值导出。实际逐帧安全偏移及调整记录分别
写入 `effective_deviation_m` 和 `safety_log`。

### 6.7 默认 guarded 正面保护模式

自由共享身份试验仍出现正面 P95 回归，因此默认采用保守的两遍流程：

1. **真实正面锚**：只用同批正面观测拟合身份及逐帧运动/表情/下颌，并完成有界细化。
2. **固定锚状态**：guarded 阶段继承粗身份、正面逐帧状态和锚细化结果，不再用
   斜侧重估全局粗身份。
3. **建立保护区域**：对真实正面有效语义深度像素每 2 像素采样，查询首交三角形，
   保护交点涉及顶点及一圈拓扑邻域。
4. **吸收新增可见区域**：斜侧仍独立估计运动/表情；只有保护区外、至少两帧真实
   斜侧支持的区域可以更新局部身份细节，总位移继续受 ±3 mm 和防翻转保护。
5. **重新验证正面**：检查正面渲染覆盖和逐射线深度与锚是否保持数值一致。
   `0.0001 mm` 只用于浮点一致性判断，不是实际精度承诺。

这一模式保护已有正面几何，但限制了斜侧对全局脸型的修改能力。它不是“侧面脸型
已经更准确”的结论；是否有有效侧面增量必须结合支持、覆盖和实际残差判断。

### 6.8 独立颈部

颈部按自己的 RGB-D 重叠运动分组；采集缺口或失跟产生新分量。在可靠段内把观测
变换到段锚帧，以真实深度栅格中位数生成开放三角表面，使用约 5 mm 深度一致性
和约 12 mm 最大边长保护。无可靠深度格点丢弃。

颈部不反馈头部身份，也不因头颈运动不一致而中止整次重建。当前输出是分段开放
表面，**没有跨段统一的颈部身份模型**，不是水密完整颈部。

### 6.9 与原 B、FaceCap 的区别

| 项目 | 原 B | 当前 C / guarded |
|---|---|---|
| 身份与表情 | FLAME 离线共享身份原型 | 分阶段交替，闭口身份、逐帧表情/下颌 |
| 深度对应 | 近邻顶点 | 首交三角形、重心坐标、鲁棒点到面 |
| 口部 | 有身份/表情耦合风险 | 身份排除口部、闭口软约束、无效深度显式缺失 |
| 正面回归 | 张口 P95 曾劣于仅正面对照 | 默认固定正面锚；相对旧 B-front 的张口问题仍未解决 |
| 局部细化 | 顶点法向位移 | 共享三角形方程、±3 mm、重复支持、防翻转 |
| 颈部 | 独立运动跟踪 | 增加按独立轨迹的实测开放表面 |

本项目没有复现 FaceCap 的 augmented blendshapes、UV bump/deviation 图、
Kinect/OpenCL 实时系统或同等精度保证，也没有复制 FaceCap 原实现。
本地新增内容是 FLAME 批量 PyTorch 优化、三角形射线 ICP、有界法向细化和审计。

## 7. 分步命令和研究入口

以下路径中的 `<采集目录>`、`<运行目录>` 是占位说明，使用时替换为实际新目录。
每次 `--output` 使用未存在的目录名。

### 7.1 只采集，稍后离线运行

```powershell
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' capture --session-name head
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' run --capture 'D:\proiect\withB-runtime\captures\<采集目录>' --output 'D:\proiect\withB-runtime\outputs\<运行目录>' --device cuda
```

### 7.2 准备、重建、验证分别执行

```powershell
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' prepare --capture 'D:\proiect\withB-runtime\captures\<采集目录>' --output 'D:\proiect\withB-runtime\outputs\prepare-01'
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' reconstruct --sequence 'D:\proiect\withB-runtime\outputs\prepare-01\prepared\sequence.json' --output 'D:\proiect\withB-runtime\outputs\fit-01' --device cuda
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' verify --sequence 'D:\proiect\withB-runtime\outputs\prepare-01\prepared\sequence.json' --fit 'D:\proiect\withB-runtime\outputs\fit-01\guarded' --output 'D:\proiect\withB-runtime\outputs\verify-01'
```

`reconstruct` 只做拟合和导出；需要完整验证时使用 `run`，或补充 `verify`。
已有 `sequence.json` 可以直接 `run --sequence ... --output ...`。
外部原始块使用 `prepare --frame-manifest ...`，然后使用其准备结果。

### 7.3 自由共享身份研究模式

```powershell
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' reconstruct --sequence 'D:\proiect\withB-runtime\outputs\prepare-01\prepared\sequence.json' --output 'D:\proiect\withB-runtime\outputs\free-study-01' --free-identity --research-only --device cuda
```

两个开关必须同时出现，输出为 `free_identity_research`。此模式在现有数据上发生过
正面回归，不能作为“效果更好”的产品默认值。

### 7.4 B / B-front / C-front / guarded C 回归

```powershell
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB\headrgbd.ps1' regression --sequence 'D:\proiect\withB-runtime\captures\sequence-v2\sequence.json' --original-comparison 'E:\research-archive\comparison-final\comparison.json' --output 'D:\proiect\withB-runtime\outputs\regression-new-01' --device cuda
```

`--original-comparison` 是需要自行提供的**固定原项目基线比较 JSON**，上面的 E 盘
只是路径示例，不是程序默认依赖。此入口串行重跑 B、B-front、C-front、guarded C，
再与固定基线诊断合并；它不会在本仓库重新运行原主项目。
原基线只有四个代表帧，且使用融合 burst 深度，B/C 使用逐帧输入；不能伪称全部
20 帧拥有完全一致输入和独立外参的原基线。

### 7.5 常用参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--device` | `cuda` | 拟合设备；可选 `cpu`，本机性能结论只针对已测 GPU |
| `--steps` | 240 | 每个 C 拟合阶段预算，不是靠增加次数保证效果 |
| `--points` | 1200 | 每帧用于拟合的深度采样预算，不是评估覆盖率 |
| `--keyframes-per-segment` | 3 | 每段关键帧目标数，实际角色以报告为准 |
| `--landmark-device` | `cpu` | 准备阶段 TUFA 设备，支持 cpu/cuda/auto |
| `--frames-per-view` | 5 | 每段保存帧数，最少 2 |
| `--interval-s` | 0.12 | 请求保存间隔；实际跟踪依据真实时间戳 |
| `--warmup-s` / `--countdown-s` | 0.8 / 2 | 相机预热秒数 / 段前倒计时 |
| `--views` | 四段默认值 | 可重复指定 `state:view`；仍须满足真实观测契约 |
| `--auto` / `--no-preview` | 关闭 | 自动段启动 / 不显示 RGB 预览 |

所有命令及各子命令支持 `--help`。CLI 默认值来自 `config/defaults.json`；
其中 `evidence` 项也记录保护契约，但几何阈值的实际实现位于 `geometry_c.py` 等模块，
不能假设改 JSON 就改变全部保护逻辑。改变阈值应作为算法变更独立验证。

## 8. 输出在哪里，如何使用

`capture-run` 默认输出根是
`outputs\<采集目录名>_reconstruction`。以下路径相对于该输出根：

| 路径 | 内容 |
|---|---|
| `run_report.json` | 版本、阶段命令/退出码、环境、GPU 快照、耗时、总状态和摘要 |
| `logs\*.log` | 各阶段 stdout/stderr；失败时首先查看对应日志 |
| `prepared\sequence.json` | 所有帧的候选/排除、来源、分段、重叠、降级记录 |
| `prepared\<frame-id>.npz` | 准备后的逐帧 RGB-D、标点、掩码、内参 |
| `reconstruction\front_anchor\` | 正面对照锚完整结果；验证正面保护时仍需要它 |
| `reconstruction\guarded\` | 默认最终重建，内部文件见下表 |
| `validation\verification.json` | 几何/标点/支持度/正面保护契约检查 |
| `validation\reference_geometry.json` | 参考几何诊断，不要与 fit 内同名侧车混淆 |

`reconstruction\guarded\` 中：

| 文件 | 用途 |
|---|---|
| `closed_coarse_m.obj` / `closed_refined_m.obj` | 闭口参考帧相机坐标中的粗/细网格，米 |
| `open_coarse_m.obj` / `open_refined_m.obj` | 张口参考帧相机坐标中的粗/细网格，米 |
| `closed_support_m.ply` / `open_support_m.ply` | 与对应参考状态绑定的支持度着色网格，颜色不是纹理 |
| `reference_geometry.json` | 粗/细标点、重心坐标、实际观测有效性、闭/张口参考与唇距 |
| `report.json` | 参考帧来源、时间戳、帧角色、角度、对应拒绝、细化与安全日志 |
| `parameters.npz` | 共享身份、逐帧表情/下颌/旋转/平移、粗/细网格和实际安全偏移 |
| `support.npz` | 逐帧顶点观测、次数、细化帧映射、支持标签、分数、位移 |
| `neck_track_XX_observed_m.ply` | 某一独立颈部轨迹的米制开放表面；可靠数据不足时可能不生成 |
| `neck_track_XX_support.npz` | 该颈部表面的顶点、面和观测支持 |

闭口/张口参考分别从对应状态的正面关键帧按清晰度选择。两个状态可以具有不同的
参考相机姿态，不能直接逐顶点相减解释成同坐标下的开口形变。
应读取 `report.json.references` 和 `parameters.npz` 的逐帧状态进行对齐。

粗网格标点必须配粗网格，细标点配细网格。标点为动态轮廓与静态 FLAME 68 点、
零起始索引，绑定实际参考帧、三角形与重心坐标；验证会从实际 OBJ 重新插值，
而非只检查内存里的标点数组。

`verify` 需要最终 fit、同次 `sequence.json` 和对应正面锚，不能只留下 OBJ 就期望
完整审计复现。移动运行目录后需明确处理报告中的来源/锚路径；路径可配置不等于
旧报告中的所有绝对路径会自动重写。

## 9. 支持度如何解读

`support.npz` 的顶点顺序与网格一致：

| 字段 | 含义 |
|---|---|
| `observed_frame_mask` | F×V 布尔矩阵；F 的顺序同 `report.json.frames` |
| `observed_frames` | 每顶点可靠观测帧数，为上表矩阵按帧求和 |
| `refinement_frame_mask` | 身份细化观测的逐帧支持矩阵 |
| `refinement_frame_indices` | 细化矩阵行对应的报告帧索引 |
| `refinement_frames` | 每顶点细化支持次数 |
| `deviation_m` | 共享法向位移，米；实际安全偏移另见 parameters |
| `region_label` / `label_names` | 下列支持类别 |
| `confidence` | `min(observed_frames / 3, 1)`，重复观测计数分数 |

| 标签 | PLY 颜色 | 解读 |
|---:|---|---|
| 2 | 蓝色 | 至少两帧可靠观测支持 |
| 1 | 黄色 | 单帧/不足观测，保持模型先验 |
| 0 | 灰色 | 未获得可靠深度支持，属于未观测补全 |

支持分数不是校准概率，不是独立覆盖率。相邻短序列帧高度相关；有重复支持也不
保证实际表面精度合格。后脑、遮挡耳部、未观测侧面仍可能主要由 FLAME 先验决定。
观察 ±90° 网格视图只能检查几何异常，不能证明那里有真实观测。

## 10. 实际验证记录与效果边界

### 10.1 既有 20 帧回归：审计记录

数据是四段真实短序列共 20 帧：闭口正面、左右约 29° 斜侧、张口正面。
没有张口侧面、完整连续转头、独立扫描或人工真值。

以下数值来自迁移时保存的审计
`D:\proiect\withB-runtime\logs\withb_migration_validation_final_20261031.json`。
它记录了历史回归报告及比较文件 SHA256。**本次文档更新时，审计中引用的
`outputs\regression-20261031-final` 原目录已不在该位置**，所以以下是已保存审计
的历史结果，不是本次重新运行或重新校验原输出的声明。来源研究记录另见
[C_RESULTS.md](head_rgbd/C_RESULTS.md) 和 [MIGRATION.md](MIGRATION.md)。

| 计时项 | 历史记录 |
|---|---:|
| 全回归（B、B-front、C-front、guarded、验证与比较） | 388.890 s |
| guarded C 子进程阶段（含进程开销） | 167.722 s |
| guarded C 内部 `elapsed_s`（对应、细化、证据与导出） | 163.275 s |
| guarded C 内部 `fitting_s` | 81.134 s |
| PyTorch allocator peak allocated / reserved | 65.767 / 88.000 MiB |

这些显存数值只统计拟合进程的 PyTorch allocator，不是整卡总显存；CPU 射线查询
也包含在阶段耗时中。冷启动 guarded 还需要额外的 C-front，不能只报侧面阶段。
GPU 阶段当时串行执行，没有终止其他 agent 进程。

实际身份关键帧为 `closed_front_0/2/4`、`closed_left_0/2/4`、
`closed_right_0/2/3`；其中六个斜侧帧参与侧面身份细化。
其余闭口帧及全部 `open_front_0..4` 仅估计运动/表情。整帧排除和降级均为 0。
模型估计 yaw：左 -29.14 至 -28.66°、右 29.75 至 30.80°；没有独立外参确认。

代表帧整脸相机轴深度残差，中位数 / P95，单位 mm：

| 帧 | 固定原项目基线 v0.12.0 | 原 B | guarded C |
|---|---:|---:|---:|
| `closed_front_1` | 1.744 / 8.093 | 1.128 / 6.037 | 0.453 / 2.409 |
| `closed_left_0` | 1.717 / 8.168 | 0.532 / 2.829 | 0.742 / 3.339 |
| `closed_right_3` | 1.882 / 6.424 | 0.533 / 2.773 | 0.735 / 3.416 |
| `open_front_2` | 1.937 / 8.488 | 1.051 / 4.312 | 0.706 / 4.319 |

固定基线提交是 `3e409af77a4686b0cf435fff2230b90a8d26e11d`，仅 stage2；
四帧基线使用融合 burst 深度，不能忽略与逐帧 B/C 的输入差异。

审计中记录的几何契约：5023 顶点、9976 三角面，有限、米制、OBJ 往返通过，
非水密；细化翻转 0，最大法向偏移 2.999345 mm，16 条时间边无跨段约束。
支持标签为 2：2179、1：126、0：2718 个顶点。

| 参考 | 粗 / 细内唇距 | 原始自动深度标点 |
|---|---:|---|
| 闭口 `closed_front_2` | 1.689601 / 1.689598 mm | 1.035963 mm，不是人工真值 |
| 张口 `open_front_1` | 28.531110 / 28.531118 mm | 无效，JSON `null` |

结论必须保留：六个斜侧身份帧的整脸 P95 或中位数仍未优于原 B；五个张口帧的
P95 仍差于旧 B-front。正面与自身 C-front 锚基本数值一致，但曾记录最大
`0.000001043 mm` 的正差，不能写成绝对零变化。
**历史回归执行通过，几何效果验收失败。**

### 10.2 新真人采集与验证失败定位

本次检查发现的实际新采集：

```text
D:\proiect\withB-runtime\captures\20261031_091957_065754_head
D:\proiect\withB-runtime\outputs\20261031_091957_065754_head_reconstruction
```

这次采集包含人脸和有效深度，不同于 03:09 的无人脸最小测试。原运行报告显示：

| 阶段 | 退出码 | 阶段耗时 |
|---|---:|---:|
| prepare | 0 | 33.650 s |
| C-front | 0 | 71.718 s |
| guarded C | 0 | 175.513 s |
| 几何契约验证 | 1 | 15.295 s |

失败日志为 `validation\verification.json` 的父目录不存在导致
`FileNotFoundError`，不是几何断言失败。0.1.0 已修复统一验证入口的目录创建，
并保留原始失败报告，不回写为成功。

0.1.0 发布检查对这些现存真人产物执行了 `verify`，两项验证子阶段均退出 0，
新报告位于：

```text
D:\proiect\withB-runtime\outputs\v010-live-artifact-verification-20261031\run_report.json
```

补充检查结果：有限米制网格、参考标点与实际 OBJ 一致、无跨段时间边、三角形
翻转 0、最大法向偏移 2.970078 mm；10 个正面帧相对锚的渲染覆盖一致、最大
深度差 0。支持标签 2/1/0 的顶点数分别为 2019/113/2891。

本批闭口参考 `closed_front_2` 的粗/细内唇距为 1.857772/1.857772 mm，
原始内唇深度无效；张口参考 `open_front_4` 为 38.256865/38.256873 mm，
原始自动标点距为 43.394176 mm。有效性按实际帧逐点判断，不能把上一批
“张口内唇无效”的状态硬编码到所有采集，也不能把自动标点当人工真值。

因此，已有真人采集、离线拟合和本次补充导出验证的**分段证据已齐备**；原始
一键运行仍记录失败。本次没有重新采集，也没有重跑整条 `capture-run`，不报告
“0.1.0 一键真人全链路已重新运行通过”。新真人数据还没有 B/B-front 对照或
独立精度验收，不能把契约通过当作效果改善。

### 10.3 0.1.0 软件检查

编译检查通过，20 项测试全部通过（包含 PowerShell `-File ... --version`
实际调用和新嵌套验证目录写出）；`--help`、`status`、`assets` 实际启动成功，
7 个资产 SHA256 全部一致。新报告包含 `project_version: "0.1.0"`；旧报告不回填版本。

### 10.4 哪些指标仍未验证

训练数据诊断使用原语义面部区域的有效深度，报告整脸、口部、下半脸、面颊的
残差、覆盖和异常比例；评估不沿用拟合时的外点剔除来掩盖坏区域。
这些依赖拟合姿态和训练深度，不能当作独立真实误差。

以下全部仍为未验证：独立表面误差、人工轮廓误差、独立有效覆盖率、完全侧脸、
长期跟踪、真实侧面脸型/下颌角/前后厚度改善和跨段统一颈部身份。
几何查看应关闭纹理，检查正面、斜侧及侧面；未观测补全不能冒充扫描证据。

## 11. 代码检查与故障排查

从源码根执行快速检查，不需要相机或完整 GPU 重建：

```powershell
Set-Location -LiteralPath 'D:\proiect\withB'
& 'D:\proiect\withB-runtime\env\python.exe' -m compileall -q head_rgbd core third_party tests
& 'D:\proiect\withB-runtime\env\python.exe' -m pytest -q tests head_rgbd/test_geometry_c.py
```

测试覆盖版本一致性、资产哈希、独占输出、时间缺口、真实斜侧输入要求、研究模式
开关、无效内唇空值、单位换算、排除/降级透传、失败采集审计与几何小型契约。
这些测试不能替代真实相机数据或独立精度验收。

| 现象 | 处理 |
|---|---|
| `Runtime Python not found` | 检查 `-Runtime` 和 `env\python.exe`；不要借用旧共享环境掩盖路径错误 |
| `missing` / `sha256_mismatch` | 按资产清单取得正确且获许可的模型；不要跳过核验 |
| 相机无法打开/被占用 | 检查 USB 3.0、SDK 和相机使用者；不终止其他 agent 进程 |
| `face_bbox_not_detected` | 检查实际相机朝向、脸的完整性和曝光；查看原始 RGB，而非拿旧回放替代 |
| 有效深度为零或口部深度缺失 | 查看 D2C、距离、深度原图和遮挡；缺失不能写为零约束 |
| `guarded ... requires ... oblique` | 需要真正闭口斜侧观测；不要复制正面凑帧 |
| 失跟/时间缺口 | 查看 sequence 的 overlap/barrier；允许重定位，不强行跨段加时间项 |
| 颈部深度不足 | 保留降级记录，继续可靠头部重建；不能伪造颈部表面 |
| 输出目录已存在 | 选择新的运行名；保留之前结果和失败审计 |
| `verify` 失败 | 先看阶段日志，区分路径/环境错误和几何断言；不能改阈值绕过 |
| CUDA 内存不足/GPU 忙 | 串行运行，等待占用释放或明确改为 CPU；不杀其他进程，重新记录计时口径 |
| 网格有后脑但没有观测 | 查看 support；模板补全不是真实扫描，不能据此宣称完整头部精度 |

## 12. 来源、迁移清单与磁盘边界

算法来自 `D:\proiect\rgbd-head-reconstruction-20261030`，来源分支
`codex/rgbd-head-reconstruction-20261030`，来源提交
`3d53af24b0b245198e9108534dd02cdc695c2a81`。当前项目独立维护，运行时不导入
旧 worktree，不替换原主项目默认算法。

[C_METHOD.md](head_rgbd/C_METHOD.md) 和 [C_RESULTS.md](head_rgbd/C_RESULTS.md)
保留来源研究语境，其中旧路径、旧版本表述和 `reproduce_c.ps1` 不是当前项目入口。
当前操作以本 README、`headrgbd.ps1` 和 CLI 帮助为准。

逐文件迁移清单、字节和 SHA256 见 [MIGRATION_MANIFEST.txt](MIGRATION_MANIFEST.txt)。
迁移前后磁盘测量口径、历史验证以及旧运行目录候选清理项见
[MIGRATION.md](MIGRATION.md)。清理候选不等于已删除。

本次版本化与仓库关联不删除旧数据、不合并 main、不创建/移动标签。
原始采集、仍在使用的环境/模型/缓存、源码/worktree 以及必要研究归档继续保留在
各自独立位置，不迁入程序目录；未经明确批准的路径不删除。
