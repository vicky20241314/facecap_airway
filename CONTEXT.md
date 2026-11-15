# CONTEXT

本项目的规范术语。只放词义，不放实现细节或规格。

- **喉结标记中心（marker centre）**：贴在受试者喉结体表、由操作者按可触及的喉结隆起放置的 ArUco 贴纸中心；是体表观测点，不是甲状软骨内部测量点。
- **L_obs**：标记贴纸表面的原始观测三维点，标注帧坐标。
- **L_skin_est**：贴纸表面点沿局部法向减去标称贴纸厚度后的皮肤位置估计；不是直接皮肤测量。
- **L_anchor**：本次交付选定的喉结锚点；ArUco 来源时等于 L_skin_est。
- **C / 默认 C**：当前一键重建的颈部方案 = `neck_direct_v2` + 受试者解剖定位的 V 形喉结先验。
- **标记驱动中心（marker-driven centre）**：C 的喉结先验中心只由 L_anchor 在受试者坐标下的高度 t 决定，左右角 θ 固定为 0；幅度仍由解剖/比例决定。标记无效时按缺失策略处理，不静默回退。
- **anatomical_landmarks.json**：放在最终头模同目录的喉结 sidecar；供下游 `head_windpipe_regis` 读取，不是纹理或网格的一部分。
- **状态配对**：closed 头模只配 closed 观测，open 只配 open；不按文件名猜测。
- **live run**：开摄像头的一键 `run`；默认要求 closed/open 两态都检出有效标记，否则本次不出 final/measured 交付。
- **offline run**：`reconstruct`、`rebuild --capture`、`rebuild --dense` 等不经过实时采集门禁的入口；无标记时可回退 menton/C，但结果标 not_usable，不写可配对 sidecar。
- **自动标记识别**：不要求操作者输入 ID 或尺寸；检测颈部 ROI 内的 ArUco，自动读取 ID，并根据深度与四角三维边长估计黑方块尺寸及不确定度。多个 ID 或尺寸异常则拒绝。
- **最终交付（final/）**：C 重塑后的 `head_closed.obj` / `head_open.obj` 及其自包含 sidecar/manifest。
- **测量交付（measured/）**：未做颈部 C 重塑的 `base_head_closed.obj` / `base_head_open.obj` 及其自包含 sidecar/manifest；下游可分别拿两套网格做配准。
- **usable / not_usable**：usable 表示 sidecar 与 manifest 能被下游自动配对并通过字段/状态/帧/单位校验；not_usable 表示缺标记或校验失败，保留诊断但不冒充可用输入。
- **分层终止目标（tiered terminal target）**：按顺序比较的终点指标：基重建固定射线/接受状态 → 颈部连接折痕 → 颈部目标场残差 → 结构与密度。层间不可相加，低层不能补偿高层。
- **硬约束（hard constraint）**：标记语义、保护集、拓扑与两态面表、上颈密度、交付自包含；任一 fail/unverified 的候选不参与排名。
- **字典序排名（lexicographic ranking）**：先比第一层，只有相等才进入下一层；不是加权和。
- **候选白名单（candidate whitelist）**：允许离线试跑的少量高杠杆参数，每个候选只改一个参数；默认值、接受条件、标记语义与下游算法不在其中。
- **只加记录（record-only instrumentation）**：为分析而持久化求解器已有的真实能量/残差/迭代，不改变求解、参数、采样或接受规则。
- **单文件 HTML 报告**：每次分析或优化生成的可离线打开的 `.html`，样式与图内嵌、无外部资源、缺失值写 `null`。