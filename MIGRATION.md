# withB 迁移与验证记录（2026-10-31）

## 来源与目标

| 项目 | 值 |
|---|---|
| 来源 worktree | `D:\proiect\rgbd-head-reconstruction-20261030` |
| 来源分支 | `codex/rgbd-head-reconstruction-20261030` |
| 来源提交 | `3d53af24b0b245198e9108534dd02cdc695c2a81` |
| 目标源码 | `D:\proiect\withB` |
| 目标分支 | `codex/withb-rgbd-head-20261030` |
| 独立运行目录 | `D:\proiect\withB-runtime` |
| 本机环境 | Windows / PowerShell / Python 3.10.22 |
| 项目版本 | `0.1.0`，以仓库根目录 `VERSION` 为唯一版本源 |
| 目标 remote | `origin` = `https://github.com/vicky20241314/facecap_airway.git` |
| 发布边界 | 推送到独立 `codex/` 分支；不合并 main、不移动或创建标签 |

迁移前已读取来源 `AGENTS.md`、`research/head_rgbd/C_METHOD.md`、
`C_RESULTS.md` 和 `experiment-c-20261030.json`，并确认来源 worktree 干净。
目标目录原有内容先检查后以不覆盖方式加入；没有修改来源 worktree、共享环境或
其他 agent 的输出。

## 迁移范围

已迁移实际运行依赖：

- `core/`：相机、采集、配置、检测、语义、联合拟合和 LM 等必要模块。
- `head_rgbd/`：准备、B/C 拟合、评估、几何契约、参考标点、支持度、颈部、审计、
  统一 CLI 和来源方法/结果文档。
- `third_party/flame/`：FLAME 代码及 MIT 许可证，不含模型资产。
- `third_party/tufa/`：TUFA 代码及 GPL-2.0 许可证，不含权重。
- `VERSION`：项目版本 `0.1.0` 的唯一来源。
- `config/assets.json`、`config/defaults.json`：资产 SHA256 与默认重建参数。
- `tests/test_contracts.py`：资产、时间、身份、口部空值、失败审计和采集契约测试。
- `headrgbd.ps1`、`requirements.txt`、`.gitignore`：独立启动和环境入口。

没有迁移方案 A、演示数据、旧采集样例、旧回放、试验网格、重复压缩包、Python
环境、模型、采集或重建输出。它们继续位于独立运行/研究目录，不进入 Git。
`.gitignore` 保留模型和运行资产的全局排除规则，同时明确放行
`third_party/tufa/Prompt/shape_*.npz` 这 4 个由源码清单固定的 TUFA prompt；
它们不是运行目录中的模型权重或采集产物。

相对来源 C，迁移中做了五项运行性修正：

1. `fit_b.py` / `fit_c.py` 的候选帧保留 `degraded` 标记。
2. 拟合报告保留准备阶段 `prepared_excluded_frames`。
3. 统一 CLI 汇总口正确透传排除帧和降级帧，并增加契约回归测试。
4. PowerShell 启动器固定从源码根执行后恢复调用者目录，保证任意当前目录可启动。
5. PowerShell 启动器支持 `--help`/`-h` 和无参数帮助，避免入口参数契约阻断 CLI。

## 运行与验证

环境：

- Python：`D:\proiect\withB-runtime\env\python.exe`
- PyTorch：2.0.1，CUDA build 11.7
- 相机：Orbbec Gemini 215，序列号 `AYZ81630069`，USB 3.0，SDK 2.2.8
- 资产：7 个，全部通过 SHA256
- 模型资产不入 Git，路径和哈希由 `config/assets.json` 固定

最新真实数据回归：

`D:\proiect\withB-runtime\outputs\regression-20261031-final\run_report.json`

| 阶段 | 退出码 | 耗时（秒） |
|---|---:|---:|
| `fit_b_all` | 0 | 33.390 |
| `fit_b_front` | 0 | 20.076 |
| `fit_c_front` | 0 | 73.173 |
| `fit_c_guarded` | 0 | 167.722 |
| `verify_guarded` | 0 | 15.280 |
| `reference_metrics` | 0 | 4.149 |
| `compare_all_methods` | 0 | 71.927 |
| 全流水线 | 0 | 388.890 |

guarded C 内部总耗时 163.275 秒（纯拟合字段 `fitting_s` 为 81.134 秒），PyTorch allocator 峰值 allocated/reserved
为 65.767/88.000 MiB。各 GPU 阶段串行运行，没有终止其他 agent 进程；reserved
只是本进程 allocator 峰值，不是整卡总显存。

身份关键帧：

`closed_front_0/2/4`、`closed_left_0/2/4`、`closed_right_0/2/3`。

侧面身份细化帧：

`closed_left_0/2/4`、`closed_right_0/2/3`。

仅运动/表情帧：

`closed_front_1/3`、`closed_left_1/3`、`closed_right_1/4`、
`open_front_0..4`。

整帧排除 0，降级 0。模型估计 yaw：左斜侧 -29.14 至 -28.66 度，右斜侧
29.75 至 30.80 度；没有独立外参验证。

几何契约：

- 网格 5023 顶点、9976 面，有限、OBJ 往返成功、单位为米。
- 网格不是水密真实扫描，后脑和未观测区域仍含 FLAME 先验补全。
- 时间边 16 条，全部在同一采集段内，无跨缺口约束。
- 最大有界法向细化 2.999345 mm，细化三角形翻转 0。
- 10 个正面帧相对 C-front 的 P95 数值一致；`open_front_3/4` 存在
  `0.000001043 mm` 上限的正差，仍按正差记录。
- 支持标签计数：重复可靠观测 2179、观测不足/先验 126、未观测补全 2718。
- 闭口参考 `closed_front_2`：粗/细内唇距 1.689601/1.689598 mm；
  原始自动深度标点 1.035963 mm，不是人工真值。
- 张口参考 `open_front_1`：粗/细内唇距 28.531110/28.531118 mm；
  原始内唇深度为 `null`，未以零代替三维约束。

## 效果验收

流水线执行通过，几何效果验收失败。

| 代表帧 | 原 B 中位数/P95 mm | guarded C 中位数/P95 mm |
|---|---:|---:|
| `closed_front_1` | 1.128 / 6.037 | 0.453 / 2.409 |
| `closed_left_0` | 0.532 / 2.829 | 0.742 / 3.339 |
| `closed_right_3` | 0.533 / 2.773 | 0.735 / 3.416 |
| `open_front_2` | 1.051 / 4.312 | 0.706 / 4.319 |

6 个斜侧身份帧
`closed_left_0/2/4`、`closed_right_0/2/3` 的整脸 P95 或中位数未优于 B。
张口 P95 相对 C 自身正面锚没有回归，但未超过旧仅正面对照。此前的 B 张口
P95 回归问题不能改名成改善。

独立表面误差、人工轮廓误差、独立有效覆盖率和真实侧面精度均未验证。训练帧
相机轴残差不能替代独立扫描验收。

## 真实相机状态

已实际连接相机、读取 HW D2C/内参并保存 3 段共 6 帧原始数据：

`D:\proiect\withB-runtime\captures\20261031_030936_371066_live-minimal`

失败审计：

`D:\proiect\withB-runtime\outputs\live-minimal-audit-failure\run_report.json`

6/6 帧因 `face_bbox_not_detected` 被排除，三段有效深度中位数均为 0，画面为
天花板且无人脸。原始数据和 session manifest SHA256 已保留，prepare 退出 1，
run report 退出语义为 2。结论是相机连接和失败审计已验证，但真人采集到网格
链路未跑通，不能用旧回放替代真实相机验证。

## 磁盘与清理盘查

固定测量点（2026-10-31 最终复验期间；后续文档或日志变化会造成少量差异）：

| 路径 | 文件数 | 字节 | GiB | 重解析点 |
|---|---:|---:|---:|---:|
| 来源研究 worktree | 315 | 446666208 | 0.416 | 0 |
| 旧研究运行根 | 101531 | 25166766101 | 23.438 | 2778 |
| 旧重建运行根 | 268 | 110445336 | 0.103 | 0 |
| 新运行根 | 50416 | 6828432072 | 6.360 | 0 |

目标源码口径：

| 项目 | 文件数 | 说明 |
|---|---:|---|
| `MIGRATION_MANIFEST.txt` 所列文件 | 71 | 可交付源码；逐文件字节和 SHA256 见清单 |
| `MIGRATION_MANIFEST.txt` | 1 | 固定格式的文件级 SHA256 清单 |
| `D:\proiect\withB`（含本地缓存，不含 `.git`） | 127 | 其中 55 个本地缓存文件不属于交付清单 |

提交后的精确整树、`.git`、缓存、清单和运行目录占用记录在独立审计 JSON
`D:\proiect\withB-runtime\logs\withb_migration_validation_final_20261031.json`
的 `disk_measurement` 中，避免把会随提交变化的 `.git` 字节数写进被提交文档
造成自引用漂移。

实际删除：0 个文件，释放 0 字节。以下全部只是待批准候选，未执行移动或删除。

最终机器可读审计：

`D:\proiect\withB-runtime\logs\withb_migration_validation_final_20261031.json`

### 建议优先确认的候选

| 绝对路径 | 字节 | 用途 | 判定理由 | 仍被引用 |
|---|---:|---|---|---|
| `D:\proiect\rgbd-head-reconstruction-runtime-20261030\delivery` | 19780467 | 解压后的研究交付 | 与保留 zip 的 69/69 文件 SHA256 全等 | 未发现 |
| `D:\proiect\rgbd-head-research-runtime-20261030\b-full` | 3938849 | B 早期全视角运行 | 与 `b-final` 的 OBJ/NPZ 全同，仅 report/reference 较旧 | 未发现 |
| `D:\proiect\rgbd-head-research-runtime-20261030\b-front-only` | 2706366 | B 早期仅正面运行 | 与 `b-front-final` 的 OBJ/NPZ 全同，仅 report 较旧 | 未发现 |
| `D:\proiect\rgbd-head-research-runtime-20261030\comparison` | 9571 | 旧 B 比较 | 已被 `comparison-final` 取代 | `record_results.py` 使用 final，不使用此目录 |
| `D:\proiect\rgbd-head-research-runtime-20261030\b-smoke` | 3863499 | B 小样本烟测 | 源文档/脚本未引用；保留价值仅为额外烟测审计 | 未发现 |

若只批准前四项，预计释放 26435253 字节（约 25.21 MiB）。若同时批准
`b-smoke`，预计释放 30298752 字节（约 28.90 MiB）。

### C 历史试验候选

这些目录仍被 `experiment-c-20261030.json` 作为失败/中间试验原始记录引用。
只有接受小型实验 JSON 和 `C_RESULTS.md` 作为审计归档后，才可讨论删除其原始输出。

| 绝对路径 | 字节 | 用途 | 仍被引用 |
|---|---:|---|---|
| `D:\proiect\rgbd-head-reconstruction-runtime-20261030\c-front-final` | 0 | 主机内存失败的空尝试 | 是，失败记录 |
| `D:\proiect\rgbd-head-reconstruction-runtime-20261030\c-front-trial-01` | 4309317 | 早期正面锚 | 是，trial 摘要 |
| `D:\proiect\rgbd-head-reconstruction-runtime-20261030\c-front-v2` | 4310447 | 第二版正面锚 | 是，trial 摘要 |
| `D:\proiect\rgbd-head-reconstruction-runtime-20261030\c-trial-01` | 6699870 | 早期 C | 是，trial 摘要 |
| `D:\proiect\rgbd-head-reconstruction-runtime-20261030\c-v2` | 6692329 | 第二版 C | 是，trial 摘要 |
| `D:\proiect\rgbd-head-reconstruction-runtime-20261030\comparison-c-trial-01` | 1245517 | 早期 C 比较 | 是，trial 摘要 |
| `D:\proiect\rgbd-head-reconstruction-runtime-20261030\comparison-c-v2` | 1398639 | 第二版 C 比较 | 是，trial 摘要 |

七项合计 24656119 字节；空目录释放 0。当前不建议把“JSON 有摘要”自动视为
足以删除原始试验，需明确批准。

### 高影响但高风险的 A 历史项

下列 A 输出仍被 `experiment-final-20261030.json` 明确引用，当前不推荐删除：

| 绝对路径 | 字节 | 状态/用途 |
|---|---:|---|
| `D:\proiect\rgbd-head-research-runtime-20261030\a-author-sample` | 84175355 | 作者样例失败 |
| `D:\proiect\rgbd-head-research-runtime-20261030\a-author-sample-v2` | 989235826 | 失败，v3 的恢复输入 |
| `D:\proiect\rgbd-head-research-runtime-20261030\a-author-sample-v3` | 1000484046 | 失败，v4 的恢复输入 |
| `D:\proiect\rgbd-head-research-runtime-20261030\a-author-sample-v4` | 1014217823 | 被终止，v5 的恢复输入 |
| `D:\proiect\rgbd-head-research-runtime-20261030\a-author-sample-v5` | 1072005052 | A 几何脚本完成，研究审计终态 |

前四项理论合计 3088113050 字节（约 2.876 GiB），但它们是已记录恢复链，
必须先确定 A 审计归档策略后才能考虑，不属于默认清理范围。

### 默认保留

- 原始采集、`sequence` / `sequence-v2` / `sequence-generic` 及相关输入块。
- `env-a`、`env-b`、`withB-runtime\env`、模型、缓存和下载资产。
- 源码仓库、worktree、Git 对象和 `withB-runtime\assets`。
- B 最终核验项：`b-final`、`b-front-final`、`baseline-v0120`、
  `comparison-final`、`sequence-v2`。
- C 最终核验项：`final-run`、`final-visibility`、
  `head-rgbd-c-20261030.zip`、环境记录和实验 JSON。
- `replay-b-original`、`replay-b-front-original` 和
  `comparison-replay-original`；它们仍是 `reproduce_c.ps1` 的默认输入。
- 用途不明、仍被程序引用或可作为最终研究归档的文件。

未明确批准的路径不会删除。后续清理必须先核验解析后的绝对路径、目录联接/
符号链接和引用关系，按批准的具体路径逐项执行，不递归删除旧运行根，不使用
通配符，不跟随链接。

## 结论边界

- “迁移实现完成”：是。
- “真实数据离线重建流程跑通”：是，最新回归退出 0。
- “真实相机连接和原始保存验证”：是，但采集内容无有效人脸/深度。
- “真实人脸采集到重建链路跑通”：否。
- “几何效果验收通过”：否。
- “独立真实精度验收通过”：未验证。

文件级清单与 SHA256 见 `MIGRATION_MANIFEST.txt`。

## 0.1.0 版本化与仓库关联补充（2026-10-31）

上文运行结果和磁盘测量保留迁移时的历史口径；以下是本次现场检查的变化：

- 已将项目版本定义为 `0.1.0`，唯一版本源为 `VERSION`；CLI、Python 包、
  新 status/run report 读取同一版本。不修改原项目版本或标签。
- README 扩展为完整中文使用和算法文档，区分准备、身份/表情、三角形对应、
  guarded 保护、细化、防翻转、颈部、支持度和真实精度边界。
- 本次发现 `outputs\regression-20261031-final` 已不在原位置；其结果仍由
  `logs\withb_migration_validation_final_20261031.json` 保存的历史审计支持。
  不能把历史路径写成当前可现场复验的输出，也没有在本次删除该目录。
- 新真人采集 `captures\20261031_091957_065754_head` 有人脸和有效深度。
  对应原运行 prepare、C-front、guarded 均退出 0；验证输出因未创建
  `validation` 父目录而失败。0.1.0 已修复该入口目录创建，未放宽几何保护。
- 对上述现存产物的补充验证位于
  `D:\proiect\withB-runtime\outputs\v010-live-artifact-verification-20261031`，
  验证及参考几何子阶段均退出 0；有限、米制、标点坐标/阶段、支持度、
  16 条段内时间边、防翻转与 10 个正面锚一致性检查通过。
- 原真人失败报告保留原样；本次未重采、未重跑整条 `capture-run`。
  分段证据已齐备，不能写成一键全链路重新运行通过，更不代表几何效果通过。
- 本次 compileall 通过，20 项测试通过，PowerShell 版本/帮助入口通过，
  status 和 7 项资产校验通过。修复了 `-File ... --version` 未透传的问题。
- 迁移清单重新生成并校验；清单记录当前工作树的文件字节，跨系统 checkout
  的换行转换可能改变字节哈希，比较时须采用相同换行口径。
- 按本次授权将独立分支关联并推送至 `facecap_airway`；不合并 main、不创建标签。
- 本次删除 0 文件、释放 0 字节；未对旧研究运行目录执行任何清理。

## D 深度高分辨率独立研究（2026-10-31）

本节只记录新研究分支，不改写以上历史迁移结论。

- 基点为 `D:\proiect\withB` 的 `ba20b9747792fd39785288ce5bcfe8ebe0d5cb94`；
  实施 worktree 为 `D:\proiect\withB-depth-hires-20261031`，分支为
  `codex/withb-depth-hires-20261031`。原工作树、研究 worktree 和旧结果保持不变。
- 独立运行根为 `D:\proiect\withB-depth-hires-runtime-20261031`，其中 `env`、
  `assets`、`captures/live` 和 `inputs/v010` 均为本次独立副本。新增 OSQP/joblib
  只安装于该环境。模型/数据/结果不加入 Git，没有新增模型权重。
- 新增 `depth-hires --research-only`、共享物理位移场、稀疏耦合几何约束、
  20018 顶点密集表面、重新查询的支持度、逐阶段参考标点和几何对照评估。
  保留 C guarded 默认入口、3 mm 总位移和原几何保护，项目版本仍为 0.1.0。
- 实际检查实验03几何图后，进一步增加显式 `hierarchical` 密集研究模式：
  精确插值变形后的控制表面，固定控制顶点、拟合新增顶点的小残差，并以全部
  姿态的剩余球内区间保证总位移仍不超过 3 mm。实验03完整归档于
  `head_rgbd/D_RESULTS_03.md`，原结果不覆盖；最终数表见 `D_RESULTS.md`。
- 修改的普通入口仅为 CLI 注册和 Windows UTF-8/帮助透传；未改写相机协议、
  原粗拟合模型、默认保护规则或来源 B/C 的历史研究结果。
- 论文、作者公开代码、许可证和实施差异见 `head_rgbd/D_METHOD.md`；
  最终运行路径、实际对照、耗时/显存口径、验收及限制见 `head_rgbd/D_RESULTS.md`。
- 失败编码尝试、初轮最小二乘、第二轮约束近似和最终耦合约束的输出均留在独立
  运行根；没有清理旧输入或用成功结果覆盖失败记录。
- 本次只在独立分支做本地提交，不推送、不发布、不合并 main、不创建/移动标签。
  删除 0 文件、释放 0 字节。逐文件源码清单重新生成于 `MIGRATION_MANIFEST.txt`。
