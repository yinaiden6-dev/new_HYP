# M视觉来源定位：实验已启动

2026-09-22，接用户“对。开始定位”。本分支干预的是RoMa读取的视觉线索，ColNomic tokens和自由内容分数保持原样。

## 当前运行记录

- 启动器 `5157660`：dev_cpuonly，COMPLETED 0:0，34秒。
- 首个完整C128试运行 `5157662_0`：dev_accelerated，15分钟，已实际初始化RoMa并开始前向。首对无干预的全稠密输出逐位一致回执已生成（`query000/instrumentation_parity.json`）；完整128资格仍在运行，不能用首对通过代替。
- 自动续接 `5157663`：cpuonly，10分钟，afterany:5157662，等待依赖。首个query部分完成且正常退出则恢复同一query；完整128独立验证通过后才向其余592个query扩展。最多46并发，15分钟分片，正常部分结果按8候选文件继续。
- 两个启动脚本的Slurm spool已与本地脚本逐字节核对；无旧任务迁移、取消或在运行源码改动。

## 这一轮测什么

7臂：NATIVE；query/reference各自GRAY、LOWPASS、SHUFFLE。先去颜色、削弱细节、置换大块排列。全部原593 query、每图自然C128；第一张自然query仅作工程资格，不按标签挑选。

每臂记录COARSE、LR4、LR2、LR1、HR4、HR2、HR1共7阶段。第一对native须与未挂观察器/缓存的原RoMa输出逐位相同，所有native最终token权重及M须与历史封存逐位相同。最终每query还独立用NumPy核算6272个M值。

判断依据是正确candidate相对错误candidate的质量优势、排名及归一化差值是否减弱。单纯全部M变低不能作为身份归因；全593、426张原正确、144张候选内原错误共同报告。仅按M排名是诊断读数，不将其当作新检索模型成绩。

颜色/细节/排列干预尚不能区分具体药名、logo、包装边界或背景类别。低通不等于专门去文字，块置换有接缝与分布偏移。后续局部干预将依据本轮证据再冻结，图像边缘不自动命名为背景。当前未增加SAM、OCR、检测/分割或新的训练标签。

## 已完成的工程检查

- 原输入恒等；灰度通道一致；块置换像素集合逐位保留；常量低通不变。
- 小型可执行模型七臂观察器检查：7阶段完整、native输出逐位相同、特征缓存复用有效，移除观察器后原输出不变。
- 两份shell启动脚本语法检查和真实shell参数展开检查通过。
- 本轮新增来源文件SHA校验通过；authority已冻结。真实GPU的完整C128资格仍待试运行回执。

## 文件入口

- 冻结计划：`plan/RC_H593_M_VISUAL_ORIGIN_V1_20260922.md`
- 授权：`registry/rc_h593_m_visual_origin_authority_v1_20260922.json`，SHA256 `e6bd0b2cd467d646b58021db27f1ae402465f1964eaaa3ac5d8b4da94ecb74fb`
- 程序：`programs/run_rc_h593_m_visual_origin_v1.py`、`programs/rc_roma_visual_origin_v1.py`
- 结果：`results/rc_h593_m_visual_origin_v1/`；各query的part文件包含完整FP64 token权重、16×16 overlap/warp采样、全部候选和来源。inputs目录保留变换后的64×64缩略tensor、原图SHA及变换标识；原图加冻结程序可重建完整输入，缩略tensor不是完整像素档案。
- 调度：同目录`dispatch/wave*.json`；query通过标志为`M_VISUAL_ORIGIN_QUERY_PASS`；全593汇总为`M_ORIGIN_JOIN_COUNTS_PASS`。

这次只启动了新的机制定位实验，没有新的准确率或视觉因果结论。
