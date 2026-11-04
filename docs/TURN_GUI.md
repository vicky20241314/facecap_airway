# 连续转头 RGB-D：GUI 与一键启动

新增入口服务于“正面锚点 + 连续闭口转头”实验算法。使用项目自带的 Python 和模型资产。

## 启动

**双击项目根目录的 `headrgbd_turn.cmd` 打开中文 GUI。**

PowerShell 也可以直接运行：

```powershell
& 'D:\proiect\facecap_airway_delivery\headrgbd_turn.cmd'
```

GUI 使用 `pythonw.exe` 启动，默认不会留下终端窗口。启动失败的详情写入
`runtime/logs/turn_gui_startup_error.log`；缺少运行环境时，CMD 启动器会保留错误提示。

备用 PowerShell 启动器：`headrgbd_turn.ps1`。原 `headrgbd.cmd` 继续服务现有四姿态算法。

## 现场采集并重建

1. 在“任务模式”选择“现场采集并重建”，确认计算设备和结果根目录。
2. 可先点击“环境检查”，检查模型资产、Python、PyQt6 和 CUDA；该按钮不打开相机。
3. 点击“开始采集并重建”，相机画面直接显示在 GUI 的“相机预览”页。
4. 正对镜头、自然闭口，点击“采集正面锚点”，倒计时后保持不动，默认记录 5 帧。
5. 第二段点击“开始转头采集”，按屏幕提示缓慢转向画面左侧、右侧，最后回到正面。
6. 采集结束后自动执行完整 8 阶段重建和产物验收；在“运行日志”中可查看实时输出。
7. 完成后切换到“重建结果”，查看关键帧、转头跨度、网格数量、耗时，或打开结果目录和网格文件。

相机保持固定，距离约 0.3–0.7 米，闭口并维持自然表情。“显示深度图”可切换深度预览，
零深度显示为黑色；深度伪彩范围为约 0.25–1 米。“中心深度”是画面中央区域的深度统计，用于调整距离。
转头帧数默认 100，可在 GUI 中选择 30–200 帧；默认采样间隔 0.12 秒，实际采集时长取决于相机帧到达时间。

“停止任务”会释放采集设备，或停止本次重建进程及其子进程。关闭正在工作的 GUI 也会先停止任务。
已写入的数据、失败阶段日志和 `session.partial.json` 会保留，不覆盖以前结果。

## 重建已有采集

选择“使用已有 RGB-D 数据”，再选择包含以下文件的连续转头采集目录：

```text
anchor_session.json
turn_session.json
session.json
segment_00_closed_front.npz
segment_01_closed_turn.npz
```

点击“开始重建”即可。普通 MP4 视频或缺少深度/标定的信息不能用作同步 RGB-D 输入。
两份角色清单由本实验的 GUI 或 `turn_capture` 生成，分别限制锚点和转头的输入范围。
其他受支持的独立 RGB-D 来源可通过下面的命令行 `--anchor` 与 `--turn` 传入。

## 命令行一键流程

以下命令使用启动器定位项目环境，从任意工作目录执行均可。输出目录会自动创建新时间戳名称。

```powershell
# 一次命令：现场采集 → 重建 → 验收
& 'D:\proiect\facecap_airway_delivery\headrgbd_turn.cmd' run

# 只采集两个闭口片段
& 'D:\proiect\facecap_airway_delivery\headrgbd_turn.cmd' capture

# 使用已经采集的目录
& 'D:\proiect\facecap_airway_delivery\headrgbd_turn.cmd' reconstruct `
    --capture 'D:\proiect\facecap_airway_delivery\runtime\captures\turn_e2e_live_20261102' `
    --device cuda

# 指定独立的正面和转头 RGB-D 来源
& 'D:\proiect\facecap_airway_delivery\headrgbd_turn.cmd' reconstruct `
    --anchor 'D:\data\anchor\session.json' `
    --turn 'D:\data\turn\session.json' `
    --config 'D:\proiect\facecap_airway_delivery\config\turn_experiment.json'

# 检查环境，或查看完整参数
& 'D:\proiect\facecap_airway_delivery\headrgbd_turn.cmd' check --device cuda
& 'D:\proiect\facecap_airway_delivery\headrgbd_turn.cmd' --help
```

命令行 `run` 和 `capture` 复用原来的采集预览，在预览窗口中每段按一次空格；GUI 中使用中文按钮。
`run/reconstruct` 支持 `--runtime`、`--config`、`--device`、`--output`、`--max-keyframes`、`--detect-stride`。
`run` 另支持 `--capture-output`、`--turn-frames`、`--interval-s`。
所有显式输出目录必须尚不存在，以保留旧数据和结果。

## 文件与进度

- 现场数据：`runtime/captures/gui_turn_<时间戳>/`。
- GUI 重建：默认 `runtime/turn_experiment/gui_turn_<时间戳>/`，可在 GUI 中更改根目录。
- 完整 GUI 日志：`runtime/logs/turn_gui_<时间戳>.log`。
- 阶段报告：本次运行目录的 `audit/run_report.json` 和 `audit/logs/`。
- 独立验收：本次运行目录的 `verification.json`。
- 最终交付：`delivery/` 中仍为 6 个文件，即三张 OBJ、选帧清单、配置和阶段统计。

GUI 的进度条表示已成功完成的阶段数，阶段状态直接读取实际审计报告。
失败时显示失败退出码和完整日志，不会把部分产物标记为成功。
网格文件通过系统关联的三维查看器打开；也可打开结果目录后自行选择软件。

GUI 依赖 `PyQt6==6.6.1`、`PyQt6-Qt6==6.6.1`，已写入 `requirements.txt`。
当前交付运行环境已包含这两项，启动器不会自动安装依赖或下载模型。

算法和输入契约见 [算法说明](TURN_ANCHOR_EXPERIMENT.md)。本次界面和运行测试见
[GUI 更新验证报告](TURN_GUI_VALIDATION_20261103.md)。
