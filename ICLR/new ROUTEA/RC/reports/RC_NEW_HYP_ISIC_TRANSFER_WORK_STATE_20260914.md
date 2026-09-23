# 当前范围：new HYP论文主线收口，ownership及新编码器作为未来工作

2026-09-15。用户明确决定将当前已验证的new HYP机制、训练改进和跨数据集收益作为论文贡献收口；ownership保留为独立扩展，完整过目不忘系统及新的可训练token编码器列入未来目标。ColNomic作为本轮已验证实现保留，冻结表示的适配和表达局限写入limitations，不宣称其是所有剩余错误的已证实唯一根因。

当前写作范围以 [收口文件](../plan/RC_NEW_HYP_PAPER_SCOPE_FREEZE_V1_20260915.md) 为准。GroZi/RPC保持已封存外部确认，ISIC保持已打开队列跨域探索；旧panel和历史NO-GO不改写。COST1主模型、CE次模型角色不变。下一阶段为论文与复现材料整理；未来方向不自动启动新训练或任务。本范围决定不表示投稿材料已全部完成。

---

# 最新完成：ISIC537跨域探索通过，独立复核PASS

2026-09-14 Europe/Berlin。恢复汇总5145565已COMPLETED/0:0（35秒）。537张query、390-reference独立gallery、346患者；复用冻结H593商品头、ColNomic/RoMa、自然RAW C128、全部127 challenger及原HOLD/SWITCH。零ISIC训练、微调或阈值选择。

RAW 466/537；原成本COST4 469/537；GROUP_COST4 470/537；主COST1 500/537（相对RAW 34救0损）；次CE 513/537（47救0损）。COST1预定五项探索比较全部通过；相对RAW的等患者平均增益+5.21个百分点，患者bootstrap95%CI [3.18,7.48]个百分点。COST1_CBIND仅414/537。COST1相对COST4新增31张均已是COST4最高challenger，却被原头HOLD；此机制定位已逐行核实。

证据级别：positive_exploratory_transfer_signal=true；untouched_external_GO_claimed=false。这是已打开、历史选择过的ISIC/IMA++队列，不是全新未触碰医学确认，也不声称疾病诊断、ownership或无文字因果结论。所有模型与数据分母固定；旧28/32、99/128及GroZi/RPC结果不改写。

结果与完整报告：reports/REPORT_NEW_HYP_ISIC_FROZEN_TRANSFER_V1_20260914.md；results/rc_new_hyp_isic_transfer_v1/result.json（SHA256 6af5bfbafd36f9438372b2c5c1fff3e2f154ffa9c9af84000278809b14ea0d46）。result_validation.json与independent_final_audit_v1.json均PASS；completion_receipt_v1.json绑定结果、复核、调度与报告。

原链136任务COMPLETED、5144694_23 TIMEOUT、5144695汇总CANCELLED保留原状；超时片完整产物及额外CPU重放已通过，仅恢复不变汇总。没有待运行/待修复的ISIC任务，不重提推理、不继续监控。下方均为历史过程记录。

---

# 最新：全部537预测已验证，仅补跑汇总5145565

2026-09-14。RAW68片、RoMa68片、head预测68片全部产物齐全。原任务统计136个COMPLETED/0:0、5144694_23为TIMEOUT、5144695汇总被自动取消；不得称原链全部COMPLETED。

第23片日志显示8query的全部128候选完成并PREDICTIONS_SEALED，预测验证文件mtime为13:09:17.621UTC，Slurm结束13:09:18UTC；batch为COMPLETED/0:0，但array allocation仍TIMEOUT。其余所有片完成。新增只读恢复检验重新哈希全部RAW/RoMa/head payload与authority，68片覆盖537且无标签join；独立重复第23片4096个C4标量逐bit、80个head/CBIND决策逐bit与NumPy核算，通过。原生产验证共274944个C4标量。

只提交原始不变汇总脚本为5145565，10分钟，dev_accelerated,accelerated，初始Priority。真实spool与原launcher逐字节一致。新任务不依赖已TIMEOUT的父job，准入依据是完整产物及额外CPU重放；原超时/取消状态保留。模型、数据、阈值、统计和分母未改，没有重新训练/推理，也没有丢弃第23片。

恢复入口：先查5145565和 results/rc_new_hyp_isic_transfer_v1/{result.json,result_validation.json,independent_final_audit_v1.json}。此记录写入时尚无准确率与最终探索信号。不要重提原推理链，不重跑完整不可覆盖产物。

物证：registry/rc_new_hyp_isic_join_recovery_authority_v1_20260914.json；results/rc_new_hyp_isic_transfer_v1/submission/join_recovery_v1/{prejoin_validation.json,original_scheduler.json,submission.json}。

---

# 最新检查：ISIC首片全部完成并验证，剩余RAW因Priority排队

2026-09-13T23:16:30Z（2026-09-14 Europe/Berlin）。5144689 reference390完成0:0/10分42秒；5144690_0 RAW首片8query完成0:0/1分59秒；5144691_0 RoMa+五头及CBIND完成0:0/14分01秒。reference/RAW/RoMa/head验证均PASS；本轮重哈希首片三个payload和authority绑定一致，4096个C4标量CPU逐bit复算、NumPy最大误差7.11e-15、80个head/control决策均验证。已封存8/537 query，未join身份标签，暂无准确率。

5144693_[1-67%46]无未满足依赖，accelerated/Priority排队；5144694等待其全部完成；5144695等待剩余RoMa。未出现失败、超时或需要重提的任务。RoMa首片接近15分钟申请上限，不能保证其余分片都适用；本次未更改时限、程序、头或清单。完成的reference任务已从scontrol短期记录移除，sacct仍明确COMPLETED，非任务丢失。

物证 results/rc_new_hyp_isic_transfer_v1/submission/status_check_20260913T231630Z.json。后续检查已有链及result.json，不重复提交。GroZi/RPC结论保持。

---

# ISIC new HYP 固定头跨域探索：已提交并开始 reference 编码

2026-09-14 Europe/Berlin。用户“可以做个实验，试试”授权独立ISIC分支；不补RPC reference。

## 当前任务与恢复入口

- 5144689：390 reference编码，dev_accelerated，15分钟；已实际编码，最新可查日志。
- 5144690_[0]：afterok5144689；先验证reference产物，自动冻结inference authority，再运行RAW首片8query，10分钟。
- 5144691_[0]：afterok5144690，RoMa首片及五头/CBIND，15分钟。
- 5144693_[1-67%46]：afterok首片RoMa；剩余RAW，accelerated，每片10分钟。
- 5144694_[1-67%46]：afterok全部剩余RAW；剩余RoMa，accelerated，每片15分钟。
- 5144695：afterok首片RoMa和其余RoMa；537预测全部封存后统一join及独立复核，10分钟。

全部实际spool已与源launcher逐字节核对；调度依赖与时限保存submission/scheduler_confirmation_v1.json。
两大数组最初用dev_accelerated,accelerated，调度器拒绝为QOSMaxSubmitJobPerUserLimit；没有生成失败数组jobID。保留已提交首片，V2仅将大数组改为accelerated。5144692是sbatch --test-only输出，不是实际任务，不加入结果账本。

首个执行确认：reference编码正常；query自然推理尚未确认开始。恢复时先查这些现有job和产物，不重新提交。
最终结果应出现 result.json、result_validation.json、independent_final_audit_v1.json。目前无ISIC准确率、无正向信号判断。

## 固定科学范围

全部已下载989张IMA++图，418病灶。患者编号缺失的28病灶/62图在模型结果前排除；保留927图、390病灶、346患者。一病灶一reference按固定hash选择，其余537张为query。字节及EXIF-RGB均927唯一，源字节匿名复制未改动。分68片，最后1query。不同hash不证明独立拍摄会话或没有近重复；旧库存有历史mask/class可用性筛选，不能称全新未触碰外部集。

复用已封存H593全开发训练的五头，主COST1/次CE/COST4/GROUP_COST4/RAW2_CE，RAW单列。无ISIC训练/校准，不读取mask、dx或历史ISIC模型；患者ID只用于curator分组。全390-gallery自然C128、全部127challenger、原soft-visibility/full-reference MaxSim、原六统计/七参数、阈值0。CBIND循环移位64保留RAW。计算算子与RPC一致；不是旧DINO/区域分类实验重跑，也不是域内五折重训。

原head预测函数除数据集标记/末片报告外AST一致；390排序/ties、68片覆盖537、末片1query、患者不等大小bootstrap、两套runtime、bootstrap先冻结后RAW顺序均验证通过。部署代码/heads/计划清单在execution_preparation_v1.json和authorities中绑定；future inference authority由5144690在reference验证后、任何query forward前创建。

统计：346患者为组，所有同患者query一起，等患者均值差的100000次bootstrap/seed20260914；query微平均单报。预定探索正向信号要求COST1同时超过RAW/COST4/GROUP_COST4/RAW2_CE/自身CBIND，query净增、等患者差、95%组区间下界均正。CE不按结果晋级主模型。不称untouched external GO、不声明临床诊断、ownership或纯无文字因果。

GroZi355/480、RPC207/600两个固定头外部结果保持，SHA本轮已复核不变。旧28/32、99/128及H593 OOF分开报告。

## 文件

- 方案 plan/RC_NEW_HYP_ISIC_FROZEN_TRANSFER_V1_20260914.md
- 数据与结果 results/rc_new_hyp_isic_transfer_v1/
- reference authority registry/rc_new_hyp_isic_reference_encoding_authority_v1_20260914.json
- 自动生成的inference authority registry/rc_new_hyp_isic_inference_authority_v1_20260914.json
- submit V2只修复scheduler partition，V1源码与已封存preflight仍保留不变。
