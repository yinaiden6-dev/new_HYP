# 2026-09-13：六项本轮检验全部完成，主要可修瓶颈是保守动作成本

全部84子任务COMPLETED/0:0，16缓存、24桥接拟合、35覆盖拟合、5成本/绑定拟合及三个汇总均通过对应独立验证。本轮无待运行任务。

同H593原五折/冻结RAW C128/七参数、完整127 challenger且阈值仍0：RAW426，旧COST4 440，旧强GROUP_BASE447，新COST1 481，新CE486。COST4→COST1为44救3损、净+41；COST1→CE为14救9损、净+5，后一步等组区间跨0。相对447强基线，COST1是36救2损，CE是49救10损。

CE救回旧COST4的57张全部已经是旧头最强challenger，只是被HOLD拒绝；11个损失全部是正确RAW/HOLD被错误切换。RAW两参数CE仍426，冻结CE在CBIND上降到281。因此本轮提升主要支持任务成本与动作选择失配，并且仍依赖正确reference绑定，不是先加大头或换token才能突破。

TRAIN128原四折桥接：原六统计线性CE114；FREE113、POOL112、MATCH110、CHANNEL74。真实/错配CHANNEL训练都全对、留出却下降，不能将拟合成功当作身份机制成功。H593等128训练预算CE达到482/484/486，数量本身不解释旧440停滞；同身份增加视角存在收益但非持续单调。

当前CE107错中23个target缺C128、84个候选内错误。全部593已开发，无未触碰确认库存；六项已完成当前授权数据上的有限检验，不等于所有照片/token充分性或外部泛化都已证明。原固定28/32、99/128没有重测、没有拼账或部署替换，也没有new HYP external GO。

完整报告：[REPORT_SIX_CAUSE_ISOLATION_V1_20260913.md](REPORT_SIX_CAUSE_ISOLATION_V1_20260913.md)。机器分析results/rc_six_cause_isolation_v1/analysis.json及analysis_validation.json；可分享图six_cause_summary.png/pdf；最终调度物证completion_scheduler_receipt.json。

---

# 六项原因隔离：恢复入口

用户9月13日要求六项按优先度全部推进，随后再次要求继续六项分析。计划：`plan/RC_SIX_CAUSE_ISOLATION_V1_20260913.md`。所有新产物：`results/rc_six_cause_isolation_v1/`。无原EVAL重测、无正式392私有标签、无部署替换。

## 已完成

- inventory：5143531，3分19秒，593图像字节/解码像素/token精确碰撞均无不同身份冲突；987 ledger排除392后595记录、593唯一图，全部已开放，无未触碰确认库存。结果`inventory/result.json`、`inventory/validation.json`。
- visual：原TRAIN128 BASE7全部20错误完成非盲目视记录，`visual/manifest.json`、`visual/visual_review.json`、5个图页。至少0538的250mL、0150的28片、D0100的1粒/6粒有可见区分文字；不能推导每图可识别或token必然读取了这些字。
- bridge：首片5143534，其余缓存5143551；16片全部验证通过。首折5143552、其余23项5143567，5143570完成汇总。TRAIN128原四折/RAW C128：RAW86、BASE108、GLOBAL114；S6线性CE114、FREE113、POOL112、MATCH110、CHANNEL74、CHANNEL_PERM84。高维真实/错配分量均能拟合各折全部训练query，未转为跨身份优势。全部24项独立重训/NumPy核算通过，详见`bridge/result.json`、`bridge/validation.json`。
- coverage：首片5143568、其余34项5143569、汇总5143571全完成。相同H593五折、原六统计/七参数、全C128协议：RAW426、原COST4 440、新CE486；57救11损，对旧GROUP_BASE447为49救10损。全五折对COST4均净增。三次等128预算的CE broad为482/484/486、narrow482/483/482，不能说身份覆盖是唯一主因；每身份1→2照片CE净+1/+10/+18，2→4净+2/+4/−2，存在递减和采样敏感性。`coverage/result.json`、`coverage/validation.json`。

## 第4项必要补充（486出现后明确追加）

COST4→CE同时改变成本比和损失形式，不能就此把46净增全归因完整候选竞争。保持486头及其预测不变：同五折训练COST1原风险与RAW差+bias两参数CE；冻结CE与新COST1在既有CBIND输入零更新推理。补分57次救回的HOLD障碍/候选重排类型。

- 5143639_[0-4%5]：`slurm/rc_six_cause_loss_binding_v1.sbatch fit`，每项10分钟；
- 5143640：完整数组afterok后汇总，dev_accelerated,accelerated/10分钟；
- authority：`registry/rc_six_cause_loss_binding_authority_v1_20260913.json`；
- 完成物证：`loss_binding/fold0..4/validation.json`及`loss_binding/result.json`、`loss_binding/validation.json`。

这不是外部确认或原28/32、99/128提升；486是已开放H593分组OOF结果。CE方法以前已在小型mixed PAIR+FULL协议试过，不得声称从未做过交叉熵。当前发现是相同H593协议的因素隔离与更大内部效果。

## 尚不可写成彻底解决的命题

精确无碰撞不证明照片都可识别；高维拟合成功不证明跨身份可读性；当前分布探针失败不排除所有压缩损失；本轮不覆盖旧EVAL天花板的唯一根因；重分593不能获得新的外部确认。最终报告需保留这些边界。
