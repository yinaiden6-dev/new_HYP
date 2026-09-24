# ColQwen base 最后一条实验链修复

用户要求：改、重提；当前保留实验是最后任务，不恢复旧分支。

旧任务5161030失败于 reference token 缓存一致性检查，权重完整加载。旧缓存与当前新编码不兼容已证实，造成差异的具体历史实现因素尚未证实。V1和其失败记录原样保留。

V2先对3个固定、无标签reference进行slow/PIL processor的SDPA/eager兼容检查，沿用cosine>=.999、relative L2<=.03。只有全部通过才复用旧图库；否则固定slow/SDPA，对5413张reference重新统一编码。query始终使用与图库同一编码配置，不混用。保存全部锚点token与配置信息。

保持H593原五折、自有自然C128、CONTENT7/MASS5 COST1与CE及绑定对照。RoMa继续复用已完成的相同配对，只补缺失部分。后续质量及CPU头的自动链保留，补齐CPU超时75/124后的有界续跑。

## 重提

- 首任务：5161044，accelerated与dev_accelerated共同排队，1 GPU、8 CPU、64GB、15分钟。
- 初次核验：PENDING，QOSMaxJobsPerUserLimit。此为提交后瞬时状态，后续以实时队列为准。
- 自动恢复链按阶段首片通过后释放后续，最大并行50；不恢复已取消的144项旧任务。
- 两条M蒸馏沿原协议和原预算继续，不重启或更改其正在运行的程序。

## 验证

AST及两个Slurm脚本bash语法通过；token一致/不一致/维度不同检测通过；旧缓存可复用与不可复用两条依赖路径、图库汇总依赖经模拟提交验证；CPU/GPU正常退出、75、124、异常1的推进或续排行为已执行检查。实际提交spool与源脚本字节一致。GPU编码修复效果仍待5161044真实验证，尚不能宣布兼容性通过。

计划：plan/RC_COLQWEN_BASE_NATIVE_V2_20260924.md
程序：programs/run_rc_colqwen_base_native_v2.py
authority：registry/rc_colqwen_base_native_authority_v2_20260924.json
结果：results/rc_colqwen_base_native_v2/
提交记录、实际脚本与模拟测试：results/rc_colqwen_base_native_v2/submission/
