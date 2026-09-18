# Reference HYP收口范围与两处已分开的瓶颈

2026-09-09。本轮用户明确：reference HYP证明通过即结束，ownership不
需要。已记录在`registry/rc_reference_hyp_completion_scope_addendum_v1_20260909.json`。
不新增ownership门；原reference绑定、连通假设、完整C128和适用对照
保持，工程PASS和小子集正例不能自动称为HYP GO。

## 两处失败已直接区分

原SAM缓存9query的246mask产生234个合法四连通components。完整bank
逐component重算29,952个C128分数并独立复核：

- TRAIN56/66分别有2/8个component能让target严格胜全127 rivals；
  但原argmax(alpha)在全部mask之外，导致H0。实际图像确认主体mask已
  存在，seed分别落在桌面/阴影与玩具支架处，不是mask生成器没提主体。
- EVAL6的45个component在固定a+uniform-mean V下，target最好rank6。
  原选147格已覆盖可见主体、logo与条码，含可见35024系列编号。该例
  不能靠同一bank内换seed或mask排序变成top1。

能力定位：`results/rc_cached_sam3_region_bank_capacity_v1/result.json`，
SHA b7c6fe81c13d5290521521c990a8b7327e608a4a5d13a0f39bfb687ea489c947。
视觉依据：`results/rc_cached_sam3_three_failure_visual_audit_v1/result.json`。

## 已完成的唯一选区修复

旧规则取alpha的一个最高点，再选包含它的最高confidence合法mask。
新规则在同一234个合法component中最大化：

    SAM_confidence × sum(component内density)

QUERY density=alpha，REF density=alpha*wq_g；原两个shift也按固定
定义执行。V仍均匀mean_a，不让alpha/wq再次压低区域内身份token。
零训练、零新encoder/RoMa/SAM forward；规则与代码先冻结再读出。
新规则同时偏好累计质量/面积，不能将全部作用只归因于取消单点选择。

| split | ALL | QUERY | REF | query half-roll | REF geometry shift |
|---|---:|---:|---:|---:|---:|
| TRAIN2 | 2 | 1 | 2 | 1 | 1 |
| EVAL7 | 4 | 6 | 6 | 4 | 6 |

H0问题已修复：两侧各臂全部候选均可读。TRAIN56出现一个reference
条件带来的新增正确，同时错绑geometry后失去；EVAL7的REF、QUERY与
REF geometry shift最终预测7/7完全相同，reference-conditioning增量
仍为0。因此尚不能宣布reference HYP通过。

结果：`results/rc_cached_sam3_region_mass_closure_v1/result.json`，SHA
7ecf68bb72d67cf354bc3d501481cca876aa50c38a3f4b59f89279e8f79d0898。
独立验证SHA df54b263e2b14540d389bfb0ad89e060b0fdcba2382157fc79cfea5f56b9fdf0。
完整reference对照另见同目录`reference_conditioning_comparison.json`。
TRAIN2/EVAL7分开；EVAL7只含两个identity，不能视为七个独立身份结论。
原SAM历史像素/模型配置未封存的限制不变，未拼接缺缓存的25个EVAL。

## 已提交的固定H参考识别检验

job **5138508** 已提交，唯一启动查询为PENDING（Priority）。固定旧
QUERY_REGION选出的0006/box mask13/component0/147格H，不使用新
mass规则改变H，也不使用GT或target分数挑crop。

按用户最新范围只保留三输入：ORIGINAL、KEEP_H_ON_GRAY、
TIGHT_CROP_OF_KEEP_H。H原图像素并集为2,333,772像素，紧bbox为
[252,1386,3906,2142]，crop3654×756。原图token、完整a、旧147格V
须逐bit复现，才能继续后两输入。crop主读出固定为新native中心映回
原物理H的集合；all-crop仅作辅助，不能事后挑读法。

同一冻结ColNomic，3次forward，0训练、0RoMa/SAM新forward；输出
额外保留同次template tokens及原RAW拼接序列，但不改变本次主P/V。
裁剪同时改变有效分辨率、位置、边界和encoder上下文，正结果只能
归因于这次固定重编码操作，不独断为分辨率唯一原因。0006原RAW与
NativeC本已正确，局部V改进仍不能直接计为旧系统新增救回。

authority：`registry/rc_difficult0006_fixed_H_pixel_recovery_authority_v1_20260909.json`，
SHA 4d3ef1004f6dd8578795ded73d27c50149c5251e4bef14929b7b8f6f1abd05bd。
下一次读取`results/rc_difficult0006_fixed_H_pixel_recovery_v1/result.json`
和`independent_validation.json`；本轮不持续监控，无ownership后续任务。
