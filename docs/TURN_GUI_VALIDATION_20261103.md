# 连续转头 GUI 与一键入口验证 — 2026-11-03

状态：**已完成；GUI、启动器、相机预览及真实数据重建验证通过。**

## 更新范围

本次更新集中在连续转头算法的应用入口。重建使用当前 `turn_run` 的 8 阶段流程和独立实验配置。

| 文件 | 内容 |
| --- | --- |
| `headrgbd_turn.cmd` | Windows 双击启动 GUI；显式参数转发到命令行流程；使用项目环境与 UTF-8 |
| `headrgbd_turn.ps1` | PowerShell 等价入口，支持运行环境路径与参数转发 |
| `head_rgbd/experiment/turn_app.py` | GUI / 一键采集重建 / 已有数据重建 / 仅采集 / 环境检查；重建成功后自动验收 |
| `head_rgbd/experiment/turn_gui.py` | 中文桌面 GUI、嵌入式 RGB/深度预览、采集控制、实时日志、实际阶段进度、停止进程树、结果入口 |
| `head_rgbd/experiment/turn_capture.py` | 增加可注入的预览适配器，同一采集与保存逻辑供 GUI 和原预览调用 |
| `requirements.txt` | 记录当前已安装的 PyQt6 6.6.1 / Qt6 6.6.1 依赖 |
| `tests/test_turn_app.py`、`test_turn_gui.py`、`test_turn_launchers.py` | 21 项新增回归测试 |
| `tools/verify_turn_gui.py`、`verify_turn_gui_preview.py` | 实际 GUI 重建与相机预览验证脚本，保存截图和 JSON |
| `docs/TURN_GUI.md`、README §11 | 启动命令、操作步骤、日志和输出说明 |

原 `headrgbd.cmd`、`headrgbd.ps1` 和生产算法入口未被本次更新编辑。
当前环境已包含 GUI 依赖，本次没有安装新软件或下载模型。
修改前相关文件备份在 `runtime/validation/turn_gui_20261103/source_before/`。

## 自动测试

- 基线：138 passed / 1 skipped，2 subtests passed，退出码 0。
- 最终：**159 passed / 1 skipped，2 subtests passed，35.99 秒，退出码 0**。
- 唯一跳过项仍为依赖旧重建样例的既有同帧测试；本次 GUI 的真实数据运行已单独执行。
- 记录：`runtime/validation/turn_gui_20261103/final_tests.xml`、`final_tests.log`。

新增覆盖包括：

1. 两份角色清单必须同时存在；一键 `run` 将新采集的锚点/转头清单正确传入重建。
2. GUI 取消采集时关闭预览、停止相机，并保留 partial 清单。
3. 图像跨线程传递后仍拥有独立像素内存；开始/取消事件正确传递。
4. 延迟预览消息不会重新启用已启动的采集按钮，避免重复启动片段。
5. 输入目录无效或进程无法启动时，显示错误并恢复 GUI 控件。
6. 实际启动一个有子进程的测试任务后，停止操作会结束父进程和子进程。
7. 小窗口中的结果信息保持可读，结果页可滚动，不会压缩掉耗时等文字。
8. CMD / PowerShell 启动器从含空格的目录、无关工作目录运行；中文和空格参数保持完整；显式失败保留退出码。
9. 两个启动器无参数时路由到 GUI；缺少运行环境时保留明确错误提示。

## 真实相机 GUI 预览

设备：Orbbec Gemini 215。实际打开嵌入式预览，GUI 收到 **1280×720** 的 RGB 和深度画面，
“采集正面锚点”按钮进入可用状态；切换深度预览后自动停止。

验证结果：**passed，退出码 0**。相机释放、GUI 恢复可用；此次预览检查没有开始片段录制。
部分采集清单按设计保留，用于记录取消状态。

- 机器记录：`runtime/validation/turn_gui_20261103/gui_preview_verification.json`。
- 本地深度预览截图：`runtime/validation/turn_gui_20261103/gui_depth_preview.png`。

GUI 的两段现场录制复用此前已完成现场采集的后端；本次新录制数据为空，完整重建使用下述已有真实数据。

## 真实 RGB-D 数据的 GUI 全流程

数据：`runtime/captures/turn_e2e_live_20261102/`，此前现场录制的正面锚点 5 帧 + 连续闭口转头 100 帧。
设备选择 CUDA，使用默认 `config/turn_experiment.json`。

验证程序实际打开 Windows GUI，选择“使用已有 RGB-D 数据”，点击“开始重建”，等待子进程结束。
进度条依次到达 8/8，成功后检查结果按钮可用、开始按钮恢复、停止按钮禁用，并重新检查验收 JSON。

**最终结果：passed，errors=[]，主进程退出码 0；完整重建 150.902 秒。**

阶段全部实际执行并通过：

`tracking → keyframe_selection → prepared_keyframes → front_anchor → joint_shared_identity → pose_correction → control_field → delivery`

- 入选 7 帧，覆盖左右两侧，角度跨度约 57.2°。
- 初始化/联合网格各 5,023 顶点、9,976 面；最终网格 **20,018 顶点、39,904 面**。
- 交付目录恰有 6 文件，三张网格重新加载、有限坐标及面索引检查通过。
- 输入/配置/实验源码/序列哈希绑定检查通过；最终 OBJ 与对应参考帧表面一致。
- 运行成功后自动写入 `verification.json`，GUI 显示“重建与验收通过”。

正式通过目录：

```text
D:\proiect\facecap_airway_delivery\runtime\validation\turn_gui_20261103\final\runs\gui_turn_20261103_162038_964299
```

GUI 验证报告和截图：

```text
runtime/validation/turn_gui_20261103/final/gui_verification.json
runtime/validation/turn_gui_20261103/final/gui_existing.png
runtime/validation/turn_gui_20261103/final/gui_result.png
```

本次检查证明当前 Windows 环境、相机预览与这份真实数据的 GUI 重建路径能够完整执行。
独立几何精度、不同人的数据和 CPU 全流程不属于本次通过结论。

## 复现

```powershell
# 日常打开界面
& 'D:\proiect\facecap_airway_delivery\headrgbd_turn.cmd'

# 一键现场采集 → 重建 → 验收
& 'D:\proiect\facecap_airway_delivery\headrgbd_turn.cmd' run

# 测试已有数据的真实 GUI 操作路径；自动保存报告和截图
& 'D:\proiect\facecap_airway_delivery\runtime\env\python.exe' `
    'D:\proiect\facecap_airway_delivery\tools\verify_turn_gui.py' `
    --capture 'D:\proiect\facecap_airway_delivery\runtime\captures\turn_e2e_live_20261102' `
    --output 'D:\proiect\facecap_airway_delivery\runtime\validation\turn_gui_recheck' `
    --device cuda
```
