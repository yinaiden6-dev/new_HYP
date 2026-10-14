# 复现

在工作区运行 `results/colnomic_full_audit/20261013_resume/cross_token_relation/REPRODUCE.sh verify`。默认仅使用 CPU 两线程，重放已保存模型的选参、权重、全部预测、图库排序、SHA设计排名、损益、identity bootstrap及训练输入连接；不加载编码模型。

`REPRODUCE.sh train` 从已存全图库关系特征重新训练两个主头和六个null，并重算八套完整图库标量和排名。`REPRODUCE.sh features` 用实际执行的冻结源码快照在物理GPU 0/1重新检查或续跑奇偶查询分区，再聚合；有逐Query提交和分区锁，已有分片按SHA复用。GPU分配器上限为总显存的0.06，batch32，FP32且禁TF32；入口会检查可用显存。

环境默认 `/home/jwei/miniconda3/envs/rebut/bin/python`；可用 `COLNOMIC_PYTHON` 指定同依赖环境。源码、Python/Torch/NumPy版本、执行阶段和缓存SHA见 `execution_provenance.json`、`CPU_runtime.json`、`CPU_verify_runtime.json`、worker runtime与`input_manifest.json`。实际矩阵计算、CPU训练与最终验证的源码快照分别保留，失败尝试的日志也保留。

验收见 `final_acceptance.json`：主验证20,333项、额外训练输入连接13项均通过。这是关系/数量读出组件的验收，整体full-goal验收由统一报告处理。关系特征是有限38维摘要；32对完整原始Token矩阵做了独立CPU摘要重算，全部6,037,385个候选组合的原始mean MaxSim逐项核对。没有声称逐项独立CPU重算全部38维关系特征，也没有声称穷尽Token集合信息。
