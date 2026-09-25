# H593 后 LLM 固定模型：共同方向、patch 剩余和匹配位置

用户2026-09-25授权「继续」，承接归因汇总末项。目标是解释全量POST_REAL纠错的计算路径；不是训练新模型或追求更高正确数。

## 来源与固定范围

- H593原分组五折，每张只使用其留出折的POST_REAL最终8遍端点。骨干、投影、适配器和INTERNAL3头全部固定。
- 原ColNomic自然C128；全部593张，含23张目标不在池中的图片。原GPU端点478/593，54救2损。
- 复用已验收的593张LLM hidden、原reference tokens、M、五折快照。不重新采集，不运行RoMa/视觉编码器/LLM，不用GPU。
- 行顺序来自原无标签manifest；不按成功/失败筛选。54次纠错仅在全部评分后作为解释子集。

## 与旧24张完全相同的七条回放

用实际BF16适配器计算恒定M下h0和真实M下hM，FP32差delta=hM−h0；common为patch均值，remainder=delta−common。

|回放|改变|
|---|---|
|CONSTANT|同一最终适配器的M固定为TRAIN均值|
|NATIVE|真实M完整响应|
|COMMON_ONLY|h0加共同响应|
|SPATIAL_ONLY|h0加去均值的patch剩余响应；名称沿用旧程序，不指ownership|
|RECOMPOSED|h0+common+remainder，独立投影和评分以验算重组|
|NATIVE_FIXED_ARGMAX|完整响应，但固定CONSTANT的每个patch参考命中|
|COMMON_FIXED_ARGMAX|共同响应，并固定同一参考命中|

每条均通过原冻结BF16检索投影与FP64归一化/自由MaxSim（固定命中臂例外），采用同一个本折INTERNAL3头和HOLD=0。固定位置是ColNomic内容命中，不是RoMa坐标。

保存每对的逐patch分数、reference命中索引、共同向量、delta/剩余范数、分解能量、完整C128分数和127挑战者logit。共同能量比例不是因果收益百分比。

## 验证与解释

先做manifest首图全部128候选的工程回放及独立patch→L→logit核算；原生/恒定CPU动作需与封存GPU相同才自动释放全量。该图不按标签选择。

全量各图保存CPU/GPU数值与动作差异。任何差异必须单独报告，不放宽标准冒充GPU位级回放或直接归因全部478。各模型的数值与动作先验收，再结合原已打开标签汇总：总正确/救回/误伤、各折、原54救回保留、与原生逐图决策差异。重组臂必须恢复原生，阈值沿用旧已验收方案。

共同响应能够保留纠错，仅支持该模型存在这条充分路径；不证明patch细节普遍无用。固定命中保留行为，仅支持此处不必重新选点。分数改变但决策不变也报告，不能只看总正确数。

## 执行

首图dev_cpuonly/cpuonly共同排队，4CPU、16GB、10分钟。通过后自动提交50片、最大并行50、cpuonly，同资源时限；每片按原顺序模50分配11或12张。候选边界断点，正常预算退出同Job续跑，上限80次；失败不自动当作有效结果。

单独CPU依赖汇总任务逐图独立验收并可断点续跑。worker不读取held标签、不并发写总汇；最终汇总串行生成报告。提交回执保存脚本与Slurm实际脚本字节一致性及实际资源。

输出：`results/rc_postllm_h593_decomposition_v1/`。最终报告：`reports/REPORT_POSTLLM_H593_DECOMPOSITION_V1_20260925.md`。
