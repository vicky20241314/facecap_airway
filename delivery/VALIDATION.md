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
