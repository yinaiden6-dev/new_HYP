# Retrieval-only 主线：论文证据与最后一次 H 检验

> 命名与状态更新：当前框架统一称 **new HYP**，候选证据状态记E_g；
> 旧 **Reference HYP** 保留其空间命题含义。历史结果不变。此前停止
> 安排已被用户延长至北京时间2026-09-11 24:00。

本稿只记录已验证结果及正在固定的最终机制检验。当前主线为 RAW C128、
RoMa soft visibility、ColNomic weighted full-reference MaxSim 与冻结
HOLD/SWITCH head；它没有 superregion，也没有独立 H 网络。SAM 路径
已停止，相关区域成绩不进入 retrieval-only 主线证据。本文不声称旧
空间 HYP/P-only/P0 已通过，不声称 ownership 或未打开集确认。

## 模型和成绩必须逐条对应

| 模型/读出 | 数据与候选来源 | 基线 → 结果 | 已有证据层级 |
|---|---|---|---|
| FROZEN_C，原6特征7参数action | current-runtime EVAL32，原RAW C128 | RAW25 → 27，2救0损 | 已打开内部结果；原标量、action与P/V接口逐bit重放 |
| 同一FROZEN_C | opened difficult90，原RAW C128 | RAW61 → 69，8救0损 | 已打开回归集；已有候选绑定对照不保留8次救回 |
| NATIVE7/C_PAIRED，另一组冻结7参数 | matched EVAL32，原RAW C128 | RAW25 → 28，3救0损 | 当前该bundle最好内部结果；不能与69/90拼为同一head成绩 |
| 同一NATIVE7/C_PAIRED | matched TRAIN32，原RAW C128 | RAW27 → 28 | 已用于训练的描述，不作外部确认 |
| 无权重 ALL，query-image-only、FP64 normalize/mean-MaxSim | 同一FULL64的EVAL32，原RAW C128 | R_IMAGE23 → 同源R_FULL25，2救0损 | reference接口恢复；不是新H成绩，也不是旧RAW GPU32全query SUM精确重放 |
| 同一无权重 ALL 两R读出 | 同一FULL64的TRAIN32，原RAW C128 | 27 → 27 | 同源接口对照 |

最后两行来源：

- `results/rc_full64_all_reference_capacity_v1/result.json`
  SHA `b60bafe58a129bb749fc40bea08c31fe49faae2fd6200f86075d999c6bee317c`。
- `independent_arithmetic_validation.json`
  SHA `9463ead2fca325a78049c6a93139b29d7b61e7107d173ac6c3a62e3fba250f13`。
- FULL64输入缓存的8192对均与旧image-only a逐bit相同，最大差异0；
  同源RFULL仅增加原完整passage的有效reference tokens。来源覆盖2818个
  唯一reference，未新增encoder、RoMa或processor forward。

## 已有机制解释

原评分由全局 co-visibility 质量与加权内容匹配共同构成：

    M = sqrt(mean(wq) * mean(wr))
    L = sum_p wq[p] * max_j(wr[j] * cosine(q[p],r[j])) / sum_p wq[p]
    S = M * L

它保留完整candidate reference内的内容搜索，不将ColNomic身份读取
硬绑到单个RoMa对应token。原action还消费RAW gap、S、M、S/M以及Q/R
辅助差异特征。关系解释须以这个实际计算图为对象。

在原FROZEN_C上，中和显式M项或归一化内容项均丢失原32/90两组各自
全部救回。进一步固定L/Lq/Lr、统一M为1的因子一致干预，使difficult90
从69回到RAW61；11520个候选的相关输入完整保留、所有90条HOLD。
这支持旧成绩依赖质量与内容的组合，不能自动升级为高维关系结构的证明。
来源：`reports/REPORT_RC_ORIGINAL_ROMA_SUCCESS_AND_P_INFORMATION_LOSS_20260909.md`。

后来的P atom压缩未保留原完整reference visibility均值。无损P/V接口
修复已对8192对输入、32768个旧标量和8128个action logits逐bit回归，
保留旧FROZEN_C成绩。它是已完成的接口修复，不是新H或新识别增益。

原表示存在上下文混合：OUTCOME-0212的已打开像素诊断中，650个实际
局部输入block字节未变，其650个对应encoded tokens仍全部变化。这个
单例证据限制了“token所在位置就是证据完整物理来源”的解释；不证明
任意高维H已有效，也不将人工诊断区域用作训练输入。

## 空间 H 与关系 H 是两个命题

空间 H 要求reference生成连通多patch区域，并由独立通道验证。相关
历史失败和工程资格不因本次重新立题而变化。

当前可检验的关系命题是：候选条件visibility与内容证据的正确绑定，
是否提供超出同一全局M和相同visibility值多重集的识别贡献。这里不
新增H网络，不把既有scorer换名当作新模型创新。

最后检验冻结在
`plan/RC_REFERENCE_RELATION_FINAL_V1_20260909.md`：内容token/RAW/C128/
两组头/各通道质量与分母固定，只做预定query/reference visibility
绑定置换，重新计算所有派生标量和127-challenger action。唯一主比较
为NATIVE7/REAL与QR_BIND_DERANGED；Q-only/R-only仅分解，不择优。

主结果已完成并独立验证：EVAL32/NATIVE7 REAL28、QR27，净增+1，
正margin降幅13/32，等权supergroup平均降幅−0.00591804。预设的细粒度
绑定贡献门未通过；原三次RAW纠错全部在QR下保留，多出的一个正确
仅为避免破坏原RAW已正确的OUTCOME-0618。

这不否定候选级质量—内容联合校准。联合校准本身是多维关系的一种
形式；最后检验没有测试或否定所有这样的关系。保留该较有证据支持
的reference机制解释，不把当前结果概括为“reference方向失败”，
也不将其转写为旧空间P-only门通过。
按用户要求停止追加方法，转入论文总结。完整收口见
`REPORT_RC_FINAL_REFERENCE_MECHANISM_AND_PAPER_CLOSURE_20260909.md`。


## 延长开发期：固定校准改动的完整负结果

以下均为已打开matched EVAL32，原RAW full-gallery C128、127 challenger
HOLD/SWITCH；原NATIVE7/C基线RAW25→28。每项已独立重训/复现，均未
替换原模型，也不与旧FROZEN_C difficult90结果混合。

| 作业 | 固定改动 | EVAL32 | 相对原NATIVE7 | 证据入口 |
| --- | --- | --- | --- | --- |
| 5138654 | ABS12绝对尺度／REL12相对信息同容量对照 | 26／26 | 均0新增、2丢失 | results/rc_absolute_evidence_scale_calibration_v1 |
| 5138792 | PAIR/FULL监督 × SIGN/RANK目标 | 原PAIR_SIGN28，其余三格27 | 均未超越原28 | results/rc_native7_training_objective_factorial_v1 |
| 5138844 | 仅统一PAIR V1的Q/R位移为half | 原28，新26 | 0新增、2丢失，0220与0618 | results/rc_pair_control_scale_harmonization_v1 |
| 5138846 | D自由内容的reference轴恢复原上下文tokens | C28、D_IMAGE27、D_FULL25 | 新臂对C0新增、3丢失；对D_IMAGE0新增、2丢失 | results/rc_full_reference_content_bridge_v1 |

这些固定改法未提高当前最好内部结果。原PAIR混合位移一直正确实现；
5138844没有支持将其解释为可通过统一定义修复的性能瓶颈。它们不能
证明一切校准都无效、所有新表示都失败或原已验证增益消失。
已打开EVAL反复参与开发的事实须披露，不能把后续同集优选结果当成
未触碰外部确认。单项计划、结果、验证与未采用决定均保留。

完整reference表征读出D_FULL已完成训练与独立验证，无增益、不采用。
这也限定了先前ALL均值读出R_FULL23→25的外推范围：局部MaxSim值
增大并不保证校准后的目标竞争更好。


重训后的M/L证据充分性实验5138847已经独立验证。RAW2为25，RAW+M3、
RAW+L3、JOINT4均26，原NATIVE7仍28。JOINT4主简化判定未成立，但原完整
模型分别胜过两个重训单通道2救0损，group平均差为正；这项预定次比较
作为正向内部证据单列保留，不能被主比较未过的状态覆盖。
结果：results/rc_retrained_evidence_sufficiency_v1/result.json。
独立验证：同目录independent_validation.json。

S与Q/R响应的固定2×2已经冻结并提交5138852（提交时PENDING），尚无
科学结果。此试验复用原JOINT4与ORIGINAL7，仅新增PRODUCT5与RESPONSE6，
用于区分当前缺少的原列贡献。代数上原dS可作为dM/dL的非线性基函数，
但训练输入的代数检查不作为新增准确率或普遍理论证据。
