# 真正EVAL结果：GAP校准未改善原99，原模型保留

本次确实评价同一批旧EVAL32与新EVAL128，不是TRAIN/OOF。固定ORIGINAL7/NATIVE7原ec7六权重与bias，RAW自然C128、完整127 challenger、原最高候选与已SWITCH动作不变。仅一个共享GAP系数由全TRAIN128检索身份标签精确求得，封存后同时用于两套EVAL；没有EVAL回调系数。

| 实际评价 | RAW | 原ORIGINAL7 | 加GAP | 相对原模型新增/损失 | 原正确保留 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EVAL128 | 88/128 | 99/128 | 99/128 | 1/1 | 98/99 |
| 旧EVAL32 | 25/32 | 28/32 | 27/32 | 0/1 | 27/28 |

新EVAL128救回DIFFICULT-0028，损失OUTCOME-0130；旧EVAL32损失OUTCOME-0618。预定三个性能目标全部未满足，因此不采用该校准，原ec7参数及28/32、99/128结果保留。TRAIN OOF的GAP110/128只是此前不同基头、不同图片上的控制结果，没有变成EVAL110。

## TRAIN保护为何没有推广到EVAL

全TRAIN128原ec7为108正确。精确求得alpha=0.5421478288984741，在不损失任何这108张训练正确图的约束内，最多救回5张，且该系数是实现5救回的最小可表示FP64值；TRAIN保护上界为0.5789502203098685。这是有限模型类的经验最优，不是泛化保证。

在EVAL上，相同alpha改变了三个正确性结局：

| 图 | 原最高logit | 加GAP后 | 实际变化 |
| --- | ---: | ---: | --- |
| DIFFICULT-0028（EVAL128） | -0.5244909911132738 | +0.01765683778520033 | target从RAW第2到第1，救回 |
| OUTCOME-0130（EVAL128） | -0.21092452188149946 | +0.33122330701697467 | wrong被放行，target第1到第2，损失 |
| OUTCOME-0618（旧EVAL32） | -0.010972945716349658 | +0.5311748831821245 | wrong被放行，target第1到第2，损失 |

所以不是原模型没有身份信息，也不是本轮solver未收敛。统一放宽HOLD遇到了跨query的置信度次序冲突：某些错误提议距离过零比需要救回的正确提议更近。训练中安全的全局增量，到了这些EVAL图上会先放行错误候选。

新128的full5412 MRR仍为0.8240132107292264：一张2→1与一张1→2的倒数排名变化恰好抵消，正确集合实际上改变了。相对原模型21组等权增益为0，组bootstrap95%[-2.381,+2.381]个百分点，精确双侧sign-flip p=1。旧32的C128 MRR从0.9014136904761905降至0.8857886904761905；11组等权差-4.545个百分点。两套rank_scope不同，分别报告。

## 这条全局校准路线的明确边界

对所有已封存原logits做隔离的EVAL后验容量核验，并由独立NumPy和解析首过零边界重算：

| 独立面板 | 保住全部原正确的最大安全GAP | 任意新增正确的最早GAP首点 |
| --- | ---: | ---: |
| EVAL128，保原99 | 0.21092452188149946（0130） | 0.5244909911132739（0028） |
| 旧EVAL32，保原28 | 0.010972945716349658（0618） | 3.218998338140889（0050） |

两个面板均安全上界低于最早救回点。新128有11张原错误HOLD且原top已为target的潜在救回，全部在保护上界之外。因此固定原候选排序、锁原SWITCH、只使用共享非负GAP系数的这一模型类，无论选哪个有限FP64系数，都不能在这些有限样本上完整保留原正确并新增正确。继续换同一个全局系数不能满足本轮严格保持目标。

这不证明所有系数的最高准确率都只能99，也不排除允许正确集合取舍、条件校准、不同候选排序或更丰富表示。容量材料隔离在rc_opened_eval_gap_hold_retention_capacity_v1，不给后续训练读取、不作为新模型或新成绩。

## 证据级别与执行

本轮是看到TRAIN OOF的GAP控制较好后单独冻结的探索性跟进，两套EVAL均已打开，不是原CONTENT主臂通过后的晋级或未触碰外部确认；父CONTENT NO-GO原样保留。任务监督只有检索身份关系，没有新框、mask、点或分割teacher，基础模型预训练照实披露。

5139362已COMPLETED0:0，用时56秒。全TRAIN精确拟合及独立solver、两套EVAL共40640个双臂logit封存与fresh回放、旧28/99正确集合及RAW25/88回归、MRR和组统计复核均通过。没有新编码器/RoMa、新特征、原头重训、EVAL系数调整或部署替换。

- [机器结果](../results/rc_original7_gap_hold_followup_v1/result.json)
- [独立结果复核](../results/rc_original7_gap_hold_followup_v1/validation.json)
- [唯一TRAIN参数与证书](../results/rc_original7_gap_hold_followup_v1/parameters.json)
- [隔离的EVAL容量证书](../results/rc_opened_eval_gap_hold_retention_capacity_v1/result.json)
- [容量独立复核](../results/rc_opened_eval_gap_hold_retention_capacity_v1/validation.json)
- [预先冻结的探索计划](../plan/RC_ORIGINAL7_GAP_HOLD_EXPLORATORY_FOLLOWUP_V1_20260910.md)
