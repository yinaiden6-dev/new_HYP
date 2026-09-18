# 真实 reference 配对可见性图册

共 18 张 3200×1800 高清图（PNG / SVG / 单页 PDF），以及 18 页合订 PDF、离线 HTML、总览 JPG、逐例 CSV / JSON / NPZ。

- 药盒：历史 ORIGINAL7 / EVAL128 / RAW 自然 C128 / HOLD-SWITCH。包含全部 12 次 RAW→ORIGINAL7 纠错；另展示 OUTCOME-0337 误改、OUTCOME-0446 未纠正。原总体成绩 RAW 88/128→99/128 不变。
- RPC：固定 full-H593 COST1 / RPC600 外部确认 / RAW 自然 C128 / HOLD-SWITCH。选择原已展示 RPC-Q-0036 之后按 query_id 排序的前三个纠错和第一个误改。原总体成绩 RAW 192/600→207/600 不变。
- 此处“错误 reference”取 RAW 错误首选；误改案例取最终错误首选，图中显式改为“最终误选 reference”。正确 reference 来自原评测身份连接，且在原 C128 内。
- 图内 S 是候选的实际加权 MaxSim 分数，M 是 sqrt(mean(wq) × mean(wr))，都不是最终头的动作分数。最终决策使用多项证据。
- 热图使用缓存的 FP64 数组，native grid + nearest 像素 + 固定 [0,1]，没有逐例归一化。照片保持完整画面和编码坐标框架，只作显示缩放。没有绘制物体 mask。
- 图片、缓存、原结果、程序哈希见 source_manifest.json。导出的 NPZ 与缓存逐字节核对；药盒 S/M 等 C4 标量还与封存 prejoin 的 FP64 hex 值核对。
- 事后案例选择用于讲解，不代表总体成功率或新增科学试验；不训练、不推理、不调整模型、阈值或候选。ownership 保留为未来方向。

打开 index.html 浏览；visibility_cases_18pages.pdf 可直接展示；assets 保存各输入的便携缩略图。
复现：OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-colpali/bin/python 'ICLR/new ROUTEA/RC/programs/render_rc_new_hyp_visibility_cases_v1.py'（从 benchmark 根目录执行）。
