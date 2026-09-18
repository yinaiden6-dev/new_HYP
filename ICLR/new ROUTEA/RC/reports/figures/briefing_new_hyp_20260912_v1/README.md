# Route A / new HYP 汇报图表（2026-09-12）

数据截至2026-09-11已完成结果。六张16:9图，PNG为3840×2160；另有矢量PDF/SVG和六页合并PDF。
这只是结果可视化，不是新实验，没有修改模型、训练输入、阈值或调度。

## 推荐汇报顺序与讲解

1. `06_new_hyp_mechanism_and_scope`：先交代RAW、reference证据和共享纠错头；new HYP不是空间分割。
2. `01_performance_separate_panels`：展示各自RAW对照下的内部收益，不连成数据规模曲线。
3. `02_rescue_break_balance`：净增由救回减破坏组成，零破坏仅是样本观察。
4. `03_candidate_binding_controls`：错绑消除原救回，支持reference相关证据的作用；不是像素因果。
5. `04_training_controls_593`：同ALL行改组权重440→447，保留所有强对照与条件门负结果。
6. `05_remaining_bottlenecks`：23缺席、123在场仍错，后者不能简单归于缺乏图像或同一个根因。

## 固定口径

- RAW=原始ColNomic检索；所有图的候选源是RAW自然C128，没有把D1收益混入。
- FROZEN_C与ORIGINAL7不是同一参数头。旧FROZEN_C EVAL32为25→27；图01的EVAL32使用ORIGINAL7，因此25→28。
- 32/90/128/593是query图片数；C128是每张query的候选数。不同面板可能有历史重用，绝不相加。
- 593是开发复用的分组五折，每折重新训练，不是一个最终全训练部署头；不是独立外部确认。
- 图04点图使用非零横轴并明确标注；图01准确率柱状图从0开始。没有构造未经验证的误差棒。
- 19/0、22/1分别属于ALL_COND与GROUP_BASE，不是一个模型的两种计数。
- 所有原始negative/control结果保留；这些图没有宣称严格无损、空间ownership、普遍最优或未经测试的定位能力。

## 可追溯与复现

`metrics.csv`提供性能数值，`chart_data_audit.json`列出来源文件SHA256及计数复核。
`build_charts.py`直接读取原JSON，重新核对593逐图正确数、救回/破坏、候选缺席、组划分及冻结32图。
此审计不替代原实验的独立validator；bootstrap未重新计算；图中p值引用已封存报告。
中文字体使用系统Droid Sans Fallback，程序将Matplotlib缓存留在本输出目录，不修改home配置。

复现：`python3 build_charts.py`。合并PDF：`RouteA_newHYP_briefing_6slides.pdf`。
