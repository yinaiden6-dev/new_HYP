# 最终冻结复现入口

主入口 `../REPRODUCE.sh` 连接已全量验收的核心与新运行时Reference干预。核心Python实现及原 `final_full_verification.json` / `source_ready.json` 保持原样；原shell入口逐字保存在 `../REPRODUCE_CORE.sh`。历史 `source_ready.json` 的主shell SHA现在对应保留的CORE副本，新主入口的SHA由 `final_dispatcher_source_ready.json` 记录，避免把两份验收范围混在一起。

从归档工作区根目录执行：

```bash
results/colnomic_full_audit/20261013_resume/REPRODUCE.sh --verify-only
results/colnomic_full_audit/20261013_resume/REPRODUCE.sh --reproduce --output results/fresh_final_replay
results/colnomic_full_audit/20261013_resume/REPRODUCE.sh --verify-only --quick --output results/fresh_final_quick
results/colnomic_full_audit/20261013_resume/REPRODUCE.sh --plan-only --output results/fresh_final_plan
```

输出必须是新的空目录，位于工作区 `results/` 或 `/tmp/`；省略输出时生成独立的 `reproduction_support/final_runs/<UTC>_<PID>`。可用 `COLNOMIC_PYTHON` 选择相同依赖环境。CPU两线程，CUDA不可见，不读取编码权重，不启动编码器，不覆盖历史数据。

完整模式首先执行未修改的核心：1117×5405的三个冻结基线、八个关系/数量头和十个既有独立验证模块。`--reproduce` 导出评分、排名与权重，并重新从已保存embedding计算新Reference干预输出；不会重新编码图像。核心源码与十个验证器的实际SHA必须仍符合全量验收记录，否则入口拒绝借用旧验收。

补充分支在所有模式中均保留全部12个Query身份、60编辑臂、28个新Reference及40个新编码记录，逐点重放145个新与145个旧完整矩阵，检查稳定top2、所有127150个逐Token对应/Delta、ROI、交互与edit-minus-sham；另独立用NumPy复算63575个已桥接的Query特征L2/cosine记录，并重新计算15行统计及按12个身份的bootstrap区间。

`--quick` 仅将核心缩为固定8个分层查询，仍有每图5405个Reference和全部八个头，不重跑核心十个较长模块；新增干预验证完整执行。receipt明确区分 `core_quick_plus_complete_fresh_reference_verification` 与完整模式，不将quick冒充全量重验。此前核心全量验收SHA另行保留。

路径、安全审计钩子、临时文件与历史写保护沿用核心实现。所有旧 `/mnt/lsdf_zyin` 前缀都必须明确映射到当前解包工作区，缺失材料直接失败，不能回读原工作区。计划模式只写计划，不产生数值通过。

这些是冻结数值复现：28张新Reference均未逐位复现旧图库缓存；旧worker快照、旧图库处理器及中间层信息不足，未确定旧图库编码漂移的唯一来源。同有效Token序号轴不等于同物理patch格点。预选干预响应不构成新运行时完整图库成绩或编辑后的身份准确率，数值通过也不构成唯一失败机制证明。
