# 冻结审计的一键复现与只验

从解包后的工作区运行：

```bash
results/colnomic_full_audit/20261013_resume/REPRODUCE.sh --verify-only
results/colnomic_full_audit/20261013_resume/REPRODUCE.sh --reproduce --output results/fresh_frozen_replay
results/colnomic_full_audit/20261013_resume/REPRODUCE.sh --reproduce --quick --output results/fresh_quick_replay
results/colnomic_full_audit/20261013_resume/REPRODUCE.sh --plan-only --output results/fresh_plan
```

省略参数时默认只验，自动创建 `reproduction_support/runs/<UTC时间>_<PID>` 新目录。指定输出必须是新的空目录，位于当前工作区的 `results/` 或 `/tmp/`；历史审计分支不能作为输出。默认环境是 rebut Python，可通过 `COLNOMIC_PYTHON` 指定同依赖环境。CPU两线程，CUDA_VISIBLE_DEVICES为空，不加载编码模型。

完整模式重算全部1117×5405的BM25和原融合，核对原图像冻结标量的完整轴与稳定SHA排名；从已保存关系特征逐查询重放原mean MaxSim、Token数量、两主头及六null的权重/验证选参/预测和全库排名。再运行现有独立Token、set Probe、受控signed layout、人工框位置、位置文字内容控制、局部共同短语、crop、weighted bounds、cross-view及辅助统计验证器，共10个模块分支。weighted分支核验全部四目录，包含独立γ=0的人工框纯空间λ方法。其所有新receipt、CSV及矩阵导出都重定向到独立输出；审计钩子拒绝历史写入。

`--reproduce` 在上述数值验证之外导出三基线和八个关系/数量头的完整评分、稳定Reference排序、SHA设计排名、关系特征中的原mean MaxSim及对应模型/选择参数。它复放冻结材料，不重新编码图像。原图像评分从冻结cache导出，同时逐组合核对保存关系特征中的原mean MaxSim；没有声称CPU从头计算所有原始Query×Reference Token矩阵。

`--quick` 固定种子分层选择8个查询，仍包含每查询全部5405 Reference及八个头；主/null权重仍在完整训练样本上独立refit并重放val选择。后续较长模块验证不运行，receipt明确标为 `quick_partial_numerical_verification`。`--plan-only` 只测试计划入口，明确标为 `planned_not_numerically_verified`，不计数值通过。

路径由 `common.workspace_path` 处理：当前ROOT中的相对/绝对路径，或明确历史 `/mnt/lsdf_zyin` 前缀映射到当前ROOT；其它绝对路径、URI、越界 `..` 以及越界symlink被拒绝。映射后不存在的归档材料会失败，绝不回读外部原工作区。已知图像payload逐个验证存在性和SHA；新的路径映射表保存在独立receipt。`test_path_security.py` 真实测试移植、未知absolute、URI、traversal、symlink、缺失解包材料、外部旧workspace读和历史写保护。

各模块的数值、模板/Token范围及未计算scalar/bounds限制写入各自receipt和总 `validation.json`。数值通过不表示语义机制、编码器层因果、自然同文字布局、完整集合信息或整体goal已被证明。旧图库编码调用源码缺失、原模型权重未纳入本冻结入口，单独列为无法等价重开原encoder的限制；不会偷偷从归档外加载权重。

本次最终全量只验的固定receipt是 `final_full_verification.json`；10个模块全通过，明细保存在 `final_receipts/`。`final_input_source_summary.json` 汇总4688个payload路径和2282个独立文件的严格映射/SHA验证。`FINAL_SOURCES.sha256` 固定执行源码及入口；`source_ready.json` 记录本复现入口的冻结交付范围。主目标的最终验收和归档由总交付程序完成。
