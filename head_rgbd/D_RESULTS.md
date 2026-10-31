# D 实验04：层级密集几何的实际运行与对照

实验03完整历史数值及视觉失败保留在 [D_RESULTS_03.md](D_RESULTS_03.md)。
本页最终数表对应显式 `hierarchical` 候选，不覆盖其失败输出。

本次基于 `D:\proiect\withB` 的 `ba20b9747792fd39785288ce5bcfe8ebe0d5cb94`，
在 `D:\proiect\withB-depth-hires-20261031`、分支
`codex/withb-depth-hires-20261031` 中实施，项目版本仍为 **0.1.0**。
不改原工作树、不推送、不发布、不合并 main、不创建或移动标签、不删除数据。

方法、论文和许可核对见 [D_METHOD.md](D_METHOD.md)。本次是 FaceCap 启发的
FLAME 离线研究入口，不是完整 FaceCap 或 DECA 复现，也不重新开展方案 A。

## 1. 输入与运行隔离

独立运行根：`D:\proiect\withB-depth-hires-runtime-20261031`。

| 内容 | 本次位置与核验 |
|---|---|
| 原始真人采集 | `captures/live`，来自指定的 `20261031_091957_065754_head`，9 个原始文件逐个 SHA256 一致 |
| 0.1.0 对照 | `inputs/v010`，原整条运行产物的独立副本；不改原结果 |
| 环境 | `env/python.exe`，Python 3.10.22、Torch 2.0.1 / CUDA 11.7，环境前缀和导入路径不使用旧运行根 |
| 新依赖 | 仅本环境新增 OSQP 1.0.5、joblib 1.5.2；实际冻结见 `logs/requirements-actual.txt` |
| 模型 | `assets`，原配置的 7 项模型资产全部通过指定哈希；没有新增神经网络权重 |
| 重新准备/原 C 初始化 | `outputs/initialization-utf8` |
| 显式全视角粗身份候选 | `outputs/experiment-01/joint_initialization` |
| 最终 D 实验 | `outputs/experiment-04-hierarchical` |
| 复用的稳定 D 控制场 | `outputs/experiment-03-coupled/guarded` 与 `joint`，严格核对模式/数据/几何/身份帧/参数和 SHA |
| 输入审计 | `logs/input-and-initialization-audit.json` |

重新初始化的 shape、expression、jaw、R、t、posed_vertices、refined_vertices
与原 0.1.0 对照数组逐项最大差为 0，因此没有把初始化变化混入 guarded 的比较。
全视角候选另跑粗身份拟合，其基面允许变化，必须和 guarded 分开解释。
joint 的局部场仍限于自己的粗基面 ±3 mm；这不是“相对 0.1.0 的全部几何变化
均小于 3 mm”的保证。`joint_coarse` 本身作为独立研究变量纳入实际对照。

输入仍是四段不连续真实短序列、20 帧，不是连续完整转头视频；没有张口斜侧或
90° 真实侧脸。没有独立扫描、人工轮廓、独立外参。以下残差均为同批训练观测
诊断，运动估计帧也不能冒充独立测试集。

## 2. 实际改动和没有解决的偏差

0.1.0 细网格与控制网格同为 5023 顶点/9976 面，原细化只移动顶点，没有加密。
原正面锚已经包含噪声位移，后续 guarded 会保护这些位移；只改斜侧不能修正锚。

本次从粗网格重新求锚，加入表面积投票、跨帧不一致性降权、毫米尺度梯度/弯曲
正则、重新计算的三角对应，以及参与求解的坡度/面积/法向约束。全部实际姿态
用同一个场检查和统一增量回溯，不再在 D 中逐顶点事后折半。没有做全脸后处理
平滑或修改着色替代几何修复。口部原排除、无效深度、3 mm 总位移、支持要求和
原有向面积保护继续保留；眼腔边界另加更保守排除。

控制拓扑结果叫 `regularized`；真正加密结果叫 `dense`，20018 顶点/39904 面。
层级密集层精确插值已变形的稳定控制表面，固定原控制顶点，仅在新增顶点拟合
共享小残差，重新接受原始深度方程，并重新计算观测支持和参考标点；
`dense_interpolation` 只是消融，不能作为实测细节结果。总位移相对于各候选基面
限制为 3 mm，没有粗细阶段叠加预算。残差另限于 ±0.5 mm，用各实际姿态的球内
剩余预算取交集。该值是保守研究信赖域，不是传感器精度标定。

实验03使用 `P(V)+normalize(P(N))*P(d)`，它不等于精确的 `P(V+N*d)`；
实际插值安全检查达到 37°–43° 法向变化。实验03的密集场本身从零独立求解，
其波纹主要暴露高自由度场吸收残留噪声/配准偏差；插值偏差则另说明不能简单
插值标量作为层级基面或公平消融。本次改成精确表面基面加残差，不做全脸事后平滑。残差求解使用
带常量偏移的八面仿射法向锥；实际 35°/5%/3mm 安全检查不变，仍是显式研究
选项，不修改 C 默认保护。实测新增细节统计排除从控制场继承的位移。

D 仍固定继承 C 的每帧姿态/表情/jaw，剩余偏差可能进入共享场；法向标量场也
不能解决较大的切向身份误差。guarded 阻止斜侧更新与正面重叠区域；显式
`joint-research` 允许闭口多角度共同改变粗身份，但必须另查正面回归和先验区域。
颈部继续独立运动和开放表面处理，没有独立统一颈部身份模型。

## 3. 消融和运行记录

| 运行目录 | 实际情况与用途 |
|---|---|
| `outputs/initialization` | Windows GBK 日志编码失败；保留失败记录，修复 UTF-8 启动后重跑，没有修改几何参数 |
| `outputs/initialization-utf8` | 从原始采集副本重新准备、C 正面、guarded、验证与参考几何检查成功 |
| `outputs/experiment-01` | 有界最小二乘与统一回溯版本；部分更新被安全检查压得很小，综合质量未通过，保留作消融 |
| `outputs/experiment-02-qp` | 约束 QP 初版；两姿态/独立切向近似仍导致小步长，综合质量未通过 |
| `outputs/experiment-03-coupled` | 全部 20 姿态约束；P95 平顺改善，但实际密集视图仍波纹，插值表示亦有离散偏差，保留失败 |
| `outputs/experiment-04-hierarchical` | 精确控制表面、共享小残差、剩余总预算与仿射安全约束；重新查询深度与支持 |

首轮物理邻域是旧欧氏口径；最终采用固定物质点、原基面双图 6 mm 最短路径邻域
`geodesic_dual_graph_v2`。它避免把空间相近却不相连的眼层/上下唇混合。不同口径
不拼接比较。所有失败和旧输出保留。

## 4. 复现与输出

完整冷启动命令见 [D_METHOD.md 第 10 节](D_METHOD.md#10-依赖与复现)。本次最终
实验复用自己已经完成且 sequence SHA256 一致的初始化，实际命令为：

```powershell
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\proiect\withB-depth-hires-20261031\headrgbd.ps1' -Runtime 'D:\proiect\withB-depth-hires-runtime-20261031' depth-hires --sequence 'D:\proiect\withB-depth-hires-runtime-20261031\outputs\initialization-utf8\prepared\sequence.json' --initial-fit 'D:\proiect\withB-depth-hires-runtime-20261031\outputs\initialization-utf8\reconstruction\guarded' --joint-initial-fit 'D:\proiect\withB-depth-hires-runtime-20261031\outputs\experiment-01\joint_initialization' --regularized-fit 'D:\proiect\withB-depth-hires-runtime-20261031\outputs\experiment-03-coupled\guarded' --joint-regularized-fit 'D:\proiect\withB-depth-hires-runtime-20261031\outputs\experiment-03-coupled\joint' --baseline 'D:\proiect\withB-depth-hires-runtime-20261031\inputs\v010' --output 'D:\proiect\withB-depth-hires-runtime-20261031\outputs\experiment-04-hierarchical' --research-only --include-joint --solver qp --dense-mode hierarchical --device cuda
```

重跑须更换不存在的输出目录。源代码快照位于最终运行的 `source_snapshot`，
流程命令/退出状态/环境记录在 `run_report.json`；拟合逐轮记录在各模式
`report.json`。最终 `comparison/comparison.json` 保留每帧、每方法、每个局部区
的完整统计，不只展示选取较好的帧。

两个模式各导出闭口/张口的 `*_coarse_m.obj`、`*_regularized_m.obj`、
`*_dense_m.obj` 和 `*_dense_interpolation_m.obj`。单位米，坐标为真实参考相机
`X右/Y上/Z向后`；没有把文件坐标改成展示用中心化坐标。

`mapping.npz` 保存父面/子面和顶点映射；`reference_geometry.json` 为每个实际
OBJ 阶段重算和往返校验重心标点，缺失观测仍为 null。支持度在最终各拓扑重查，
逐帧位存于 `support_*.npz`，灰/黄/蓝 PLY 为未观测、单帧、重复帧支持。
“重复支持”不是独立扫描认证，固定口眼区域可以有观测但仍由粗模型控制。

`geometry_flat.png` 与 `geometry_smooth.png` 采用统一相机、光照、尺度且无纹理。
90° 仅是模型检查视图；后脑、遮挡耳部和未观测侧面仍是模型补全。

## 5. 验收口径

实际导出、有限性、米制范围、父子标点一致性、空值、总位移和全部姿态安全检查
与几何质量验收分开。固定 6 mm 邻域的平面 RMS/法向分散用来检查几何平顺性，
不以细分后邻面夹角自然变小证明降噪。残差同时记录共同覆盖和逐方法覆盖。

非相邻三角形穿越/共面重叠是筛查，不是完整自交认证；相邻面的重叠和切触仍未
全面验证。原模板已有相交风险，必须使用相同拓扑的原粗基面作参考，不能直接
比较粗网格与细网格的相交三角形对数量。

严格综合门槛包含：物理平顺性改善、相对 0.1.0 的正面 P95 无回归、相对粗网格
的斜侧中位数/P95 无回归、覆盖不下降、同拓扑相交筛查风险不增加，以及各阶段
保护与新增深度约束成立。任意一项未通过，候选保留研究入口，不提升为默认算法。

独立表面误差、人工轮廓误差、独立覆盖率、真实侧面脸型/下颌角/厚度改善均
**未验证**。本次使用既有真实采集，不重新连接相机，也不拿本次运行声称新采集成功。

## 6. 源码与环境检查

```powershell
Set-Location -LiteralPath 'D:\proiect\withB-depth-hires-20261031'
& 'D:\proiect\withB-depth-hires-runtime-20261031\env\python.exe' -m pytest -q
git diff --check
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File .\headrgbd.ps1 -Runtime 'D:\proiect\withB-depth-hires-runtime-20261031' depth-hires --help
```

实际核心契约测试 **33 项通过**；差异格式、Python 语法与独立启动帮助检查通过。
实际环境冻结、模型 SHA256、原始采集哈希和源代码快照分别保存在运行目录，
模型/环境/数据/输出不加入 Git。小型契约测试不充当真实效果证据。

GPU 初始化与各重建阶段顺序运行，没有终止其他进程。D 使用 CPU，不申请
CUDA 张量；报告的 D CUDA 分配 0 不等于采样到了全设备的峰值显存。
C 的 allocated/reserved 是该进程 PyTorch allocator 峰值，包含同步计时，
不包含驱动、上下文、其他进程，也不是 `nvidia-smi` 全设备连续采样峰值。
阶段前后设备快照另存于 run report，不能拿它代替整段峰值。

CPU 阶段耗时包括查询、求解、支持度、独立颈部和导出，属于正常桌面负载下
的实测壁钟时间，不是空载基准。实验03期间做过文件读取、帮助检查和一次约
9 秒的 CPU 契约测试；实验04的33项测试在拟合前完成，拟合中只做轻量读写和
文档编辑。没有并行运行其他重建候选或 GPU 重建。

当前 D 研究入口要求导入 C 的有效 candidate 帧集合；遇到初始化已排除姿态的
行会明确拒绝并要求重新准备/初始化，不会补造该帧。本批数据没有此类排除。
更复杂坏帧序列、长期跟踪、完全侧脸和新增真人采集链路没有在本轮得到验收。

## 7. 最终实测状态

| 项目 | 结果 |
| --- | --- |
| 代码实现与核心测试 | 完成；33 项通过 |
| 指定真人采集离线流程 | passed |
| guarded 综合几何门槛 | not_passed |
| joint 综合几何门槛 | not_passed |
| 独立真实精度 | 未验证 |
| 默认算法/版本 | 仍为 C guarded / 0.1.0；D 未升级为默认 |
| 本次新相机采集 | 未进行；输入为用户指定真实采集副本 |
| 删除/推送/标签 | 0 文件 / 未推送 / 未创建或移动 |

## 8. 实际帧角色与角度

D 身份方程使用 9 个闭口关键帧；11 帧仅用于逐帧运动/表情、支持度和诊断。正面锚只使用其中的闭口正面关键帧。guarded 的粗身份来自正面锚，全视角粗身份只属于 joint 候选。

guarded 的斜侧方程仍受前向保护限制：大多数重叠顶点不能更新。把帧列为身份关键帧，并不表示该帧能改变全脸。每轮实际自由顶点数见下一表和 report.json。

| 帧 ID | 分段/状态 | D 角色 | guarded yaw ° | joint yaw ° | 初始化状态 |
| --- | --- | --- | --- | --- | --- |
| closed_front_0 | closed_front / closed | 身份关键帧 | -0.030 | -0.129 | candidate |
| closed_front_1 | closed_front / closed | 仅运动/表情 | 0.132 | 0.094 | candidate |
| closed_front_2 | closed_front / closed | 身份关键帧 | -0.457 | -0.517 | candidate |
| closed_front_3 | closed_front / closed | 仅运动/表情 | -0.361 | -0.433 | candidate |
| closed_front_4 | closed_front / closed | 身份关键帧 | -0.199 | -0.259 | candidate |
| closed_left_0 | closed_left / closed | 身份关键帧 | -27.038 | -26.679 | candidate |
| closed_left_1 | closed_left / closed | 仅运动/表情 | -27.804 | -27.689 | candidate |
| closed_left_2 | closed_left / closed | 身份关键帧 | -27.223 | -27.023 | candidate |
| closed_left_3 | closed_left / closed | 仅运动/表情 | -27.784 | -27.589 | candidate |
| closed_left_4 | closed_left / closed | 身份关键帧 | -27.758 | -27.503 | candidate |
| closed_right_0 | closed_right / closed | 身份关键帧 | 39.682 | 39.073 | candidate |
| closed_right_1 | closed_right / closed | 仅运动/表情 | 39.501 | 38.532 | candidate |
| closed_right_2 | closed_right / closed | 身份关键帧 | 39.717 | 39.040 | candidate |
| closed_right_3 | closed_right / closed | 仅运动/表情 | 39.403 | 38.844 | candidate |
| closed_right_4 | closed_right / closed | 身份关键帧 | 39.502 | 38.793 | candidate |
| open_front_0 | open_front / open | 仅运动/表情 | -0.994 | -1.066 | candidate |
| open_front_1 | open_front / open | 仅运动/表情 | -2.014 | -2.038 | candidate |
| open_front_2 | open_front / open | 仅运动/表情 | -1.401 | -1.462 | candidate |
| open_front_3 | open_front / open | 仅运动/表情 | -1.153 | -1.220 | candidate |
| open_front_4 | open_front / open | 仅运动/表情 | -1.168 | -1.163 | candidate |

实际保留 16 条段内时间边，已逐条验证不跨缺口。prepare 排除记录为 `[]`；本批导入帧均为 candidate。对应、深度跳变、语义、眼口排除和可见性拒绝数量保存在 field/evidence logs，而不是补造深度。

## 9. 实际导出、约束与支持

| 模式 | 控制 V/F | 密集 V/F | 同拓扑最大总位移 mm | 密集最大总位移 mm |
| --- | --- | --- | --- | --- |
| guarded | 5023 / 9976 | 20018 / 39904 | 3.000 | 3.000 |
| joint | 5023 / 9976 | 20018 / 39904 | 3.000 | 3.000 |

| 阶段 | 逐轮自由顶点 | 逐轮全局步长 | 最大坡度 | 最小有向面积比 | 最大法向变化 ° |
| --- | --- | --- | --- | --- | --- |
| guarded/front_anchor_regularized | 1038,1016,1016 | 1.000,1.000,1.000 | 0.429 | 0.106 | 27.349 |
| guarded/regularized | 5,5,5 | 1.000,1.000,1.000 | 0.429 | 0.106 | 27.349 |
| guarded/front_anchor_dense | 2300,2294,2294 | 1.000,1.000,1.000 | 0.450 | 0.100 | 33.656 |
| guarded/dense | 18,18,18 | 1.000,1.000,1.000 | 0.450 | 0.100 | 33.656 |
| joint/front_anchor_regularized | 1033,1002,1001 | 1.000,1.000,1.000 | 0.405 | 0.228 | 26.522 |
| joint/regularized | 1195,1201,1201 | 1.000,1.000,1.000 | 0.405 | 0.114 | 26.279 |
| joint/front_anchor_dense | 2307,2297,2298 | 1.000,1.000,1.000 | 0.450 | 0.114 | 33.467 |
| joint/dense | 3076,3075,3076 | 1.000,1.000,1.000 | 0.450 | 0.074 | 33.951 |

| 模式 | 无观测 | 单帧 | 至少两帧 | 新增顶点中闭口重复支持且非零 | 其位移中位数 mm | 与纯插值最大差 mm |
| --- | --- | --- | --- | --- | --- | --- |
| guarded | 12313 | 753 | 6952 | 1795 | 0.056 | 0.500 |
| joint | 12363 | 726 | 6929 | 2188 | 0.046 | 0.500 |

新增顶点总计 14995 个。上表非零量是最终几何相对精确控制表面插值的残差，而非继承的粗位移；0.01 mm 仅用于排除数值零，不是传感器精度或细节分辨率。标签在最终密集表面重新查询，未插值；置信度是计数启发式。所有模式的闭口/张口粗细 OBJ 均实际重载检查有限性、顶点/面数、米制范围和参考重心标点。

## 10. 相同物理尺度的平顺性

以下均为闭口参考、原有重复观测支持面、相同物质点和固定 6 mm 表面邻域；法向 RMS 度与局部平面 RMS 毫米分别统计。它们会包含真实曲率，下降不能单独证明真实形状更准确。

| 方法 | 邻域数 | 法向 RMS 中位数 ° | 法向 RMS P95 ° | 平面 RMS 中位数 mm | 平面 RMS P95 mm |
| --- | --- | --- | --- | --- | --- |
| coarse | 2820 | 14.211 | 38.857 | 0.226 | 0.607 |
| v010 | 2820 | 26.944 | 54.921 | 0.295 | 0.633 |
| D_guarded_regularized | 2820 | 14.678 | 38.335 | 0.233 | 0.616 |
| D_guarded_dense | 2820 | 14.791 | 38.360 | 0.234 | 0.615 |
| D_guarded_interpolation | 2820 | 14.678 | 38.335 | 0.233 | 0.616 |
| joint_coarse | 2820 | 14.180 | 38.559 | 0.224 | 0.607 |
| D_joint_regularized | 2820 | 14.732 | 38.578 | 0.232 | 0.625 |
| D_joint_dense | 2820 | 14.805 | 38.548 | 0.235 | 0.623 |

## 11. 同一批原始深度的残差与覆盖

单位毫米，按所有候选共同覆盖像素统计；coverage 是各方法相对有效语义深度的覆盖率。另有逐方法与 0.1.0 两两共同覆盖结果用于正面回归判定。自动局部掩码不是真实人工轮廓。所有 20 帧 × 8 方法 × 5 区域见 `comparison/all_frame_region_metrics.csv` 和完整 JSON；下表不是独立精度验收。

| 帧 | 方法 | 共同像素 | 整脸中位数 mm | 整脸 P95 mm | 覆盖率 |
| --- | --- | --- | --- | --- | --- |
| closed_front_2 | coarse | 4746 | 1.038 | 6.551 | 99.85% |
| closed_front_2 | v010 | 4746 | 0.471 | 3.072 | 99.85% |
| closed_front_2 | D_guarded_regularized | 4746 | 0.469 | 2.780 | 99.85% |
| closed_front_2 | D_guarded_dense | 4746 | 0.459 | 2.754 | 99.85% |
| closed_front_2 | D_guarded_interpolation | 4746 | 0.469 | 2.780 | 99.85% |
| closed_front_2 | joint_coarse | 4746 | 1.097 | 6.266 | 99.85% |
| closed_front_2 | D_joint_regularized | 4746 | 0.585 | 3.951 | 99.85% |
| closed_front_2 | D_joint_dense | 4746 | 0.571 | 3.952 | 99.85% |
| closed_left_0 | coarse | 4553 | 0.827 | 4.578 | 99.54% |
| closed_left_0 | v010 | 4553 | 0.841 | 4.163 | 99.83% |
| closed_left_0 | D_guarded_regularized | 4553 | 0.944 | 4.611 | 99.91% |
| closed_left_0 | D_guarded_dense | 4553 | 0.947 | 4.643 | 99.91% |
| closed_left_0 | D_guarded_interpolation | 4553 | 0.944 | 4.611 | 99.91% |
| closed_left_0 | joint_coarse | 4553 | 0.849 | 4.748 | 99.52% |
| closed_left_0 | D_joint_regularized | 4553 | 0.626 | 4.098 | 99.74% |
| closed_left_0 | D_joint_dense | 4553 | 0.607 | 4.157 | 99.74% |
| closed_right_0 | coarse | 6747 | 0.802 | 2.728 | 99.81% |
| closed_right_0 | v010 | 6747 | 0.708 | 2.455 | 99.90% |
| closed_right_0 | D_guarded_regularized | 6747 | 0.806 | 2.769 | 99.88% |
| closed_right_0 | D_guarded_dense | 6747 | 0.808 | 2.772 | 99.88% |
| closed_right_0 | D_guarded_interpolation | 6747 | 0.806 | 2.769 | 99.88% |
| closed_right_0 | joint_coarse | 6747 | 0.807 | 2.745 | 99.72% |
| closed_right_0 | D_joint_regularized | 6747 | 0.450 | 2.081 | 99.85% |
| closed_right_0 | D_joint_dense | 6747 | 0.445 | 2.039 | 99.85% |
| open_front_4 | coarse | 5001 | 1.338 | 14.291 | 99.78% |
| open_front_4 | v010 | 5001 | 0.922 | 8.483 | 99.78% |
| open_front_4 | D_guarded_regularized | 5001 | 0.972 | 7.994 | 99.78% |
| open_front_4 | D_guarded_dense | 5001 | 0.969 | 8.000 | 99.78% |
| open_front_4 | D_guarded_interpolation | 5001 | 0.972 | 7.994 | 99.78% |
| open_front_4 | joint_coarse | 5001 | 1.344 | 13.170 | 99.78% |
| open_front_4 | D_joint_regularized | 5001 | 1.033 | 8.520 | 99.78% |
| open_front_4 | D_joint_dense | 5001 | 1.038 | 8.580 | 99.78% |

| 帧 | 方法 | 口部 P95 mm | 下半脸/下颌代理 P95 mm | 左面颊 P95 mm | 右面颊 P95 mm |
| --- | --- | --- | --- | --- | --- |
| closed_front_2 | coarse | 2.982 | 9.258 | 7.217 | 5.566 |
| closed_front_2 | v010 | 2.982 | 3.764 | 2.835 | 2.870 |
| closed_front_2 | D_guarded_regularized | 2.982 | 2.675 | 2.654 | 2.760 |
| closed_front_2 | D_guarded_dense | 2.982 | 2.658 | 2.594 | 2.710 |
| closed_front_2 | D_guarded_interpolation | 2.982 | 2.675 | 2.654 | 2.760 |
| closed_front_2 | joint_coarse | 2.975 | 8.322 | 6.675 | 4.980 |
| closed_front_2 | D_joint_regularized | 2.975 | 4.335 | 3.461 | 3.811 |
| closed_front_2 | D_joint_dense | 2.975 | 4.243 | 3.592 | 3.803 |
| closed_left_0 | coarse | 7.167 | 7.117 | 9.040 | 2.800 |
| closed_left_0 | v010 | 6.773 | 6.876 | 10.071 | 2.494 |
| closed_left_0 | D_guarded_regularized | 6.773 | 6.876 | 9.766 | 2.965 |
| closed_left_0 | D_guarded_dense | 6.773 | 6.876 | 9.820 | 2.982 |
| closed_left_0 | D_guarded_interpolation | 6.773 | 6.876 | 9.766 | 2.965 |
| closed_left_0 | joint_coarse | 7.455 | 8.171 | 11.015 | 2.637 |
| closed_left_0 | D_joint_regularized | 6.927 | 6.737 | 13.335 | 1.946 |
| closed_left_0 | D_joint_dense | 6.927 | 6.737 | 13.330 | 1.907 |
| closed_right_0 | coarse | 5.626 | 2.792 | 2.511 | 4.162 |
| closed_right_0 | v010 | 5.571 | 2.679 | 2.412 | 4.514 |
| closed_right_0 | D_guarded_regularized | 5.591 | 3.402 | 2.683 | 3.585 |
| closed_right_0 | D_guarded_dense | 5.591 | 3.402 | 2.699 | 3.585 |
| closed_right_0 | D_guarded_interpolation | 5.591 | 3.402 | 2.683 | 3.585 |
| closed_right_0 | joint_coarse | 4.951 | 2.752 | 2.501 | 32.570 |
| closed_right_0 | D_joint_regularized | 4.899 | 2.292 | 1.435 | 34.669 |
| closed_right_0 | D_joint_dense | 4.899 | 2.291 | 1.399 | 34.523 |
| open_front_4 | coarse | 9.048 | 38.082 | 5.666 | 7.269 |
| open_front_4 | v010 | 8.194 | 37.746 | 3.352 | 5.306 |
| open_front_4 | D_guarded_regularized | 8.137 | 37.372 | 3.243 | 5.025 |
| open_front_4 | D_guarded_dense | 8.127 | 37.372 | 3.238 | 5.101 |
| open_front_4 | D_guarded_interpolation | 8.137 | 37.372 | 3.243 | 5.025 |
| open_front_4 | joint_coarse | 8.954 | 37.477 | 5.362 | 6.665 |
| open_front_4 | D_joint_regularized | 8.330 | 36.237 | 3.948 | 5.547 |
| open_front_4 | D_joint_dense | 8.330 | 36.221 | 3.970 | 5.741 |

## 12. 回归与综合质量判定

| 方法 | 物理平顺改善 | 正面 P95 回归帧数/10 | 最大回归 mm | 斜侧回归帧数/10 | 覆盖下降帧数/20 | 相交筛查不增加 | 综合 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D_guarded_regularized | 通过 | 0 | 0.000 | 8 | 2 | 通过 | not_passed |
| D_guarded_dense | 通过 | 0 | 0.000 | 9 | 1 | 通过 | not_passed |
| D_joint_regularized | 通过 | 9 | 0.934 | 0 | 8 | 未通过 | not_passed |
| D_joint_dense | 通过 | 9 | 0.982 | 0 | 8 | 未通过 | not_passed |

guarded 相对自己新锚的前向检查：20/20 通过（两拓扑 × 10 个正面帧）。这只证明后续斜侧没有破坏新锚，不等于新锚优于 0.1.0。正面回归仍按上表逐项保留，不以保护通过替代。

综合未通过的候选保持研究模式；没有默认升级。不能宣称真实侧面脸型、下颌角或前后厚度已经改善。下一步仍需同时解决平顺与真实观测拟合、姿态/表情残差，并以独立数据/标注验收；不能靠放宽本轮保护或只增加迭代掩盖。

## 13. 几何风险筛查

| 方法/同拓扑参考 | 状态 | 非相邻穿越对数 | 共面面积重叠对数 | 双面积 <1e-14 m² |
| --- | --- | --- | --- | --- |
| 原粗基面(coarse拓扑) | closed | 310 | 0 | 0 |
| 原粗基面(coarse拓扑) | open | 293 | 0 | 0 |
| 原粗基面(dense拓扑) | closed | 640 | 0 | 0 |
| 原粗基面(dense拓扑) | open | 591 | 0 | 0 |
| coarse | closed | 310 | 0 | 0 |
| coarse | open | 293 | 0 | 0 |
| v010 | closed | 336 | 0 | 0 |
| v010 | open | 359 | 0 | 0 |
| D_guarded_regularized | closed | 310 | 0 | 0 |
| D_guarded_regularized | open | 293 | 0 | 0 |
| D_guarded_dense | closed | 640 | 0 | 0 |
| D_guarded_dense | open | 591 | 0 | 0 |
| D_guarded_interpolation | closed | 640 | 0 | 0 |
| D_guarded_interpolation | open | 591 | 0 | 0 |
| joint_coarse | closed | 317 | 0 | 0 |
| joint_coarse | open | 295 | 0 | 0 |
| D_joint_regularized | closed | 317 | 0 | 0 |
| D_joint_regularized | open | 295 | 0 | 0 |
| D_joint_dense | closed | 661 | 0 | 0 |
| D_joint_dense | open | 597 | 0 | 0 |

相交对数不能跨分辨率直接比较；同一原有相交细分后会对应更多三角形对。joint 的总风险包含其粗身份阶段，表内另列 joint_coarse 以区分来源。以上不是相邻重叠/切触的完备认证，未声称网格无任何自交。全部实际姿态检查的是相对本候选粗基面的有向面积、法向变化、坡度和位移。

## 14. 参考帧、口部与坐标

| 状态 | 真实参考帧 | 两内唇深度均有效 | 观测间距 mm | 0.1.0 refined mm | guarded coarse mm | regularized mm | dense mm | joint dense mm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed | closed_front_2 | False | 缺失 | 1.858 | 1.858 | 1.858 | 1.858 | 1.859 |
| open | open_front_4 | True | 43.394 | 38.257 | 38.257 | 38.257 | 38.257 | 38.222 |

当前新采集与旧研究序列不是同一参考：本批闭口内唇缺失，张口两点有效。有效观测也是自动二维标点投影得到的原始深度取样，仍非人工真值。缺失为 null，未强制拟合成零；口部身份场固定。密集与控制标点使用各自三角形重心映射，实际 OBJ 重载后容差校验通过；密集映射的数值微差不当作新几何效果。

## 15. 实际耗时、显存与求解状态

| 阶段 | 内部实测秒 | PyTorch allocated MiB | PyTorch reserved MiB |
| --- | --- | --- | --- |
| C 正面锚 | 67.707 | 53.385 | 76.000 |
| C guarded | 164.830 | 65.781 | 88.000 |
| C 全视角粗身份候选 | 154.265 | 66.453 | 88.000 |
| D guarded | 526.086 | 未使用 CUDA | 不适用 |
| D joint | 702.071 | 未使用 CUDA | 不适用 |
| 最终比较 | 310.244 | 未使用 CUDA | 不适用 |

最终暖启动 run report 总壁钟为 1555.271 秒，不包含先前 C 初始化。原始数据准备另耗约 25.053 秒；冷启动还需顺序加上初始化与环境/子进程开销。不能把这些独立时间简单解释为实时帧率。

实验04还复用了实验03已验证且哈希绑定的同拓扑场；其 D 壁钟包括密集残差、两拓扑观测重查、颈部与导出，不含先前控制场求解。控制求解没有另记独立壁钟，不能把实验03整段密集耗时当成控制耗时。完整冷启动入口已接好，本轮采用分段初始化加暖启动闭环，不声称一条冷启动命令再次从头执行。

| 阶段 | 近似求解轮数 | 最多内部迭代 | 最多约束行 | 最大原始残差（求解器单位） |
| --- | --- | --- | --- | --- |
| guarded/front_anchor_regularized | 1 | 20000 | 243579 | 6.68917e-06 |
| guarded/regularized | 0 | 75 | 2432 | 1.11022e-16 |
| guarded/front_anchor_dense | 0 | 3525 | 1259247 | 6.81116e-17 |
| guarded/dense | 0 | 75 | 14510 | 1.19735e-16 |
| joint/front_anchor_regularized | 0 | 14750 | 240539 | 7.55566e-16 |
| joint/regularized | 0 | 7425 | 272579 | 2.23682e-16 |
| joint/front_anchor_dense | 0 | 2725 | 1254748 | 7.7981e-17 |
| joint/dense | 0 | 3200 | 1608264 | 6.56623e-17 |

状态 2 为 OSQP solved inaccurate，不能写成全部精确收敛。所有接受增量另外通过实际几何安全和目标值检查，但最优性仍未保证。当前全部姿态约束装配和 CPU 稀疏求解明显较慢，是本实现的实际限制，后续性能优化也须保持相同几何安全。

## 16. 审计与几何视图

`logs/delivery-summary-experiment-04-hierarchical.json` 汇总实际表格源数据；`logs/delivery-final-audit.json` 保存源快照、原输入/网格未变、资产、实际 OBJ 和逐文件清单核验。几何图包含原粗、0.1.0 refined、新同拓扑、guarded dense 和 joint dense，统一光照/尺度且无纹理；补充闭口/张口双侧图位于 `comparison/geometry_bilateral_*_flat.png`。90° 是模型视图，不是实测支持。

## 17. 实际几何查看与剩余问题

已实际查看闭口/张口、正面、左右30°、左右90°的统一光照无纹理图，以及面法向
和重算顶点法向两种显示。0.1.0 的眼周等局部噪声起伏减轻，但新密集结果在
额头、面颊仍有可见折面/波纹。**完整平顺外观验收未通过**，不能以 P95 指标
改善取代此结论，也不能声称细化不平顺已彻底解决。

补充 `comparison/geometry_interpolation_counterfactual.png` 把同拓扑控制表面、
精确几何插值和实际深度残差放在同一光照下比较。控制表面与精确插值在相同
物质点的最大差为约 `2.3e-13 mm`、面法向最大差约 `1.7e-6°`，即同一个分片
平面表面；但加密后重新平均顶点法向，显示已经明显不同。纯插值图也有类似
波纹，新残差的视觉变化相对小。因此不能把所有显示变化归因于新增深度噪声，
也不能通过沿用粗显示法线掩盖控制表面本身的折面。本次没有用换着色作修复。

当前层级基面仍是 C0 分片线性控制表面，不是 C1/C2 连续曲面。实现了增加顶点、
实测约束和稳定有界的层级场；没有完成连续曲面级的平顺重建验收。更高阶表面
表示及其总预算/动画/解剖边界验证仍缺失，未把该未实现内容写成成果。

guarded 密集结果的新增重复支持残差：1795 个顶点，中位数约 0.056 mm、P95
约 0.240 mm。这些量小于/接近本研究的噪声尺度下限，且来自相关短序列。
它们证明不是单纯切面，但不证明恢复了相应尺度的真实个体细节。独立细节可
分辨性、侧面脸型/下颌角/厚度和真实精度仍未验证。

最终状态：实现和真实数据分段闭环完成；物理平顺性诊断与正面保护通过；
guarded 无正面 P95 回归，但斜侧和覆盖仍未通过综合门槛；joint 仍有正面回归及
相交筛查计数增加。**整体质量和完整视觉验收未通过，默认 C 保持不变。**
