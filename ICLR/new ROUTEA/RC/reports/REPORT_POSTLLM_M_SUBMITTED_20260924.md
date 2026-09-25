# LLM后注入：已启动

用户授权：现在做后注入。位置为LLM之后、检索投影之前。末端不直接读取M。

## 工程已验证

- 24/24图的hidden缓存完整；真实GPU零适配器回放，48次条件检查token最大误差0。
- 原投影含冻结ColNomic LoRA，独立PEFT输出及输入梯度误差0。只加载577664个冻结投影元素，不加载7B主干。
- 首次真实C128更新通过：梯度有限非零、适配器/头均更新、token变化可见、断点参数和优化器精确恢复。
- 4项单元测试通过，含逐query完整C128梯度验证和6种launcher退出/续跑情形。提交脚本与冻结源码逐字节一致。

## 任务

| Job | 内容 | 资源与时限 |
|---|---|---|
| 5162554 | 工程pilot已通过；结束时后续CPU提交触发dev数量上限，故调度器显示FAILED | 单GPU/8CPU/32GB/10分钟 |
| 5162555 | POST_REAL，从pilot第1步续到128 | dev_accelerated/accelerated，单GPU/8CPU/32GB/10分钟 |
| 5162556 | POST_CONSTANT，同初始化/同128步 | 同上 |
| 5162557 | POST_SHUFFLED，同初始化/同128步 | 同上 |
| 5162562 | 三臂全部成功后CPU固定读出refit与独立分数回放 | cpuonly,dev_cpuonly，8CPU/16GB/10分钟 |
| 5162570 | 预测封存后标签join与PRE/POST汇总 | cpuonly，8CPU/16GB/10分钟 |

CPU提交限制已经恢复，依赖逐项核实。没有取消或重复原训练。所有worker均可按候选/完整更新断点在同Job requeue。

## 评价

沿用V4的TRAIN16、ColNomic自然C128、相同初始化及128次fullC128 COST1。真实M臂固定终点做恒定/错绑推理。原PRE封存结果复用，POST三个条件均新训。固定终点后的2000步TRAIN-only refit为预定次级诊断。PROBE8已被历史开发打开，不构成新独立确认；本报告仅表示工程就绪和执行开始，不是科学GO。

执行文件：`programs/run_rc_postllm_m_v1.py`。
计划：`plan/RC_POSTLLM_M_V1_EXECUTION_20260924.md`。
权威：`registry/rc_postllm_m_v1_authority_20260924.json`。
结果与中间量：`results/rc_postllm_m_v1/`。
自动最终报告：`reports/REPORT_POSTLLM_M_V1_20260924.md`。
