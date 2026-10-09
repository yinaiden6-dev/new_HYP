# M593 任务检查与索引修复（2026-10-09）

## 核实状态

| Job | 任务 | sacct状态 | 用时 | 结果验收 |
|---|---|---|---|---|
| 5173348 | 结构/绑定隔离汇总 | COMPLETED 0:0 | 1分钟 | STRUCTURE_BINDING_INDEPENDENT_M_AND_PATCH_PASS，77个candidate-pair输入，M重算最大误差1.11e-16 |
| 5175001 | 593张inside汇总 | COMPLETED 0:0 | 6分38秒 | M_INSIDE_ALL593_COMPLETE / M_ORIGIN_JOIN_COUNTS_PASS |
| 5175002 | 593张visual汇总 | COMPLETED 0:0 | 41秒 | M_VISUAL_ORIGIN_ALL593_COMPLETE / M_ORIGIN_JOIN_COUNTS_PASS |
| 5175003 | 593张机制推广分析 | FAILED 1:0 | 31秒 | 参数列表使用字符串fold索引；未保存任何query结果 |

以上旧任务在10月2日结束。10月9日检查时没有仍在运行的旧任务；不能将汇总完成当作机制推广分析完成。

## 修复与续跑

根因：native_parameters和prior_product_parameters均为按fold排列的列表；原程序将fold转换为str，导致TypeError。仅改为整数fold索引，保留公式、五折参数、候选、对照、指标和面板。原冻结脚本、protocol、失败日志均保留。

隔离修复：programs/run_rc_m593_cached_replication_fix_v1.py；results/rc_m593_cached_replication_fix_v1/。

验证：实际第一张完整读取、19个世界评分及原128回放检查通过，anchor误差0、旧指标误差0；两种头全部5折的7参数读取通过。测试只覆盖首张端到端及参数读取，不能代替全量验收。

已重提5188460：dev_cpuonly,cpuonly共同排队，8 CPU、16GB、30分钟，无GPU。提交spool与源脚本逐字节相同，资源核对通过并已release。仅续跑失败分析，无需重新汇总或采集。提交记录在修复目录submission.json。

## 科学边界

结构隔离报告已完成，但77个candidate-pair输入不等于593张新验证，也不支持唯一因果分解。当前593推广分析仍无正式结果。修复任务将比较原128、余下465及全593的入口通路效应，以及冻结外部原折头的响应；不扩展POST或相位/Gram/局部绑定干预。
