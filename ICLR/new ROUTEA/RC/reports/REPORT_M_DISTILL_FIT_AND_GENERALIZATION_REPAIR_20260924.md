# 最后蒸馏任务：充分拟合与泛化一并修复

用户明确要求继续修复原457 TRAIN短预算失败，以及16/32训练内通过却未跨组泛化的问题。

本次只提交统一链5161355（初次核验PENDING/Priority，cpuonly/dev_cpuonly，8CPU、96GB、10分钟）；没有另行提交较早full_repair草案链，没有修改Qwen队列。实际提交spool与源脚本字节一致。

1. 64张延续原LBFGS迭代40，最多160。只以原四TRAIN门放行。
2. 通过后复用72张关系缓存，8片CPU补齐其余385 TRAIN，保存134维FP64context，可精确恢复原520维；每pair torch.equal验证，完整token和C128不变。
3. 共享缓存进行原457模型的全批优化对照（最多80次），原TRAIN门通过才评价held119。
4. 同时进行泛化修复：TRAIN435/43组拟合、22/8组验证，从未见过验证8组的64张模型初始化；正则0/.001与固定0/10/20/40/80快照，按组件等权验证log-M误差选择。不能使用已学习全457教师的模型初始化inner验证。
5. 按选定正则和精确迭代数，在全457从同一64张初始化重训，TRAIN校准后冻结并封存held119预测，再独立回放RAW/原RoMa/学生及绑定对照。泛化路径的TRAIN拟合门另行报告，未过门禁止推出token缺信息。

该方案针对过拟合信号增加了TRAIN内分组选择，不保证泛化提升。原8组已作为开发probe查看过，且原fold0本已打开，不宣称新独立确认。整个蒸馏具有RoMa教师监督，不称retrieval-only原头训练。

验证：缓存重建前向/梯度逐值相同；流式评测与原公式一致；分组过滤在读取缓存前生效；选优和依赖链经模拟验证；TRAIN role表中不在有效457的记录先排除，防止把原折全部角色表与有效训练集合混用。

计划：plan/RC_M_DISTILL_GENERALIZATION_V1_20260924.md
程序：programs/run_rc_m_distill_generalization_v1.py
authority：registry/rc_m_distill_generalization_authority_v1_20260924.json
结果与提交记录：results/rc_m_distill_generalization_v1/


## 2026-09-24 完成核对与续提修复

**完成的是64张拟合关卡，不是整条跨组蒸馏实验。** 原作业5161355计算运行6分53秒，n64在迭代80达到FIT_GATE_PASS；之后自动提交8片缓存时sbatch失败，作业因此为FAILED/1:0。未产生full457及新held119最终结果。不得把调度失败当作科学失败，也不得把小面板拟合通过当作held改善。

|面板|图数|归一化中心误差|候选对排序一致率|绝对log-M RMSE|原四门|
|---|---:|---:|---:|---:|---|
|训练拟合|64|0.239601|83.5688%|0.217077|4/4通过|
|开发probe|8|1.503017|55.0359%|0.663413|0/4通过|

当前模型具备在这64张TRAIN上达到原拟合标准的能力；8张开发probe仍显示明显泛化不足。它不证明所有ColNomic tokens缺信息，不证明所有读出无法泛化。probe为开发观察，未参与n64快照选择。

本轮重新校验authority、快照和模型SHA，从逐query指标重算均值及四项原门，均一致。

**执行修复已提交：**

- 5161375_[0-7%8]：CPU关系缓存，复用原72张，补385张的压缩关系；不重跑RoMa/编码器，不重训已通过的64张。
- 5161376：afterok整数组验收，然后按原计划自动提交457拟合对照与两条TRAIN内正则选择分支。
- 全部续提改走cpuonly；每次8CPU、32GB、10分钟，原超时续跑预算保留。两份实际spool与新launcher逐字节一致。
- 新execution wrapper只改变sbatch路由，并修复full457没有probe时的一处进度日志KeyError；科学程序原文件、参数、标签、划分、损失、门限及产物authority保持原样。
- 原5161355已失败，恢复提交不挂其afterok；依据已验收的n64结果继续，避免DependencyNeverSatisfied。

执行补充：registry/rc_m_distill_cpu_dispatch_v1_20260924.json；启动记录：results/rc_m_distill_generalization_v1/submission/cpu_dispatch_resume.json。
