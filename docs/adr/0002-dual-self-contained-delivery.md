# 双套自包含交付：final/ 与 measured/

背景：下游 `head_windpipe_regis` 会在同目录自动配对头模、`anatomical_landmarks.json` 和 `manifest.json`；用户也会分别拿 C 重塑网格和基础测量网格做配准。我们决定：一键结果同时生成 `result/final/`（C 网格）和 `result/measured/`（未做 C 的基础网格），两边各自包含 `head_closed.obj` / `head_open.obj`、`anatomical_landmarks.json`、最小自包含 `manifest.json`；根 `result/` 保留报告、快速验收和诊断文件，不承担自动配对。

**状态**：accepted

**备选方案**：只交付最终 C 网格（下游无法直接拿到测量网格）；两套平铺在 result/（同一状态两个候选头模，自动配对歧义）；只写 sidecar 不写 manifest（下游必须显式传路径）。

**后果**：每个子目录可独立送下游，不会把 C 与测量网格混用；sidecar 复制两份但内容一致；C 与 measured 的来源、参考帧和顶点面数在各自 manifest 中明确记录。
