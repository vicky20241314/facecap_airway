# 交付验证记录（2026-11-01）

本文件记录 0.2.0 交付包在本机的一次**实际**验证。所有数值来自真实运行，
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

- `delivery/` 中只有 1 个网格文件 `head_m.obj` + 3 个侧车；
- 顶点有限、面索引合法、包围盒为米级、`unit = m`；
- 参考帧、68 个标点（由网格 + 重心坐标重算）、支持度计数与最终密集阶段一致；
- `manifest.json` 版本与 `VERSION` 一致，算法名唯一。

## 6. 未验证项

- **几何精度未验证**：没有扫描配准、没有人工轮廓、没有独立外参。
- **真实相机新采集未在本轮执行**：相机已成功打开并读到对齐 RGB-D 与内参，
  但完整采集需要人员配合摆姿势，本轮没有用旧数据冒充新采集。
- 颈部补片来自真实深度观测，但不做跨段颈部身份配准（与算法原有限制一致）。
