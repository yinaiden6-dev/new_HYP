# REBUT QR / QR-vec / QRR：已实施并提交

2026-09-27。原全593 M×481/492逐例排查已完成，独立重放全部75,904候选及两个头答案。新实验在独立目录实施，原主模型和旧任务不变。

## 已完成

- [全593对照报告](REPORT_H593_M_481_492_FULL_COMPARISON_20260927.md)：481→492为16救5损；M第一仍错29例＝18 HOLD＋10排序＋1新增误伤。全部593表与可筛选页面均已交付。
- [执行方案](../plan/RC_REBUT_QR_QRR_EXECUTION_V1_20260927.md)：已评估用户新计划、去除重复步骤，并明确M增量/替代解释的检验边界。
- 已重建历史496全部18维特征与预测，最大误差1.07e-14；保留481、492、496的历史强对照。
- 71个已有完整warp缓存已转成共享F71输入；没有新增encoder或RoMa前向，没有提交GPU。
- 独立几何验收通过，包括双向采样、source/mapping mask区分、低certainty保留、原M0/native_X不变、全71来源/形状/轴核对。
- 26项模块工程检查通过；真实query000三步QRR梯度与loss检查通过。工程通过不是准确率提升。

## 任务链

|Job|用途|已核实状态|
|---|---|---|
|5165781|CPU整理71共享F输入|71/71完成|
|5165782|共享F独立核算|PASS|
|5165783|fold0，4臂×3seed|dev_cpuonly RUNNING|
|5165784_[1-4%4]|fold1–4，各12项拟合|cpuonly排队Priority|
|5165786|全部60项完成后重放模型、阈值、汇总|等待上述两组afterok|

每折8 CPU/16GB/10分钟，内部4个独立worker各2 CPU；超时保留完整optimizer/RNG/cursor并同Job续跑。总计60项拟合打包成5份CPU申请，不生成60个独立大任务。调度状态为本报告写入时快照，以后变化以实时查询为准。

## 本轮回答什么

B_CAL只重拟合共同决策；QR独立标量；QR_VEC独立16维向量后比较；QRR在共同query局部先比较再池化。三种读出活跃参数分别25,202、28,290、25,266，均保留原M0和内容末端路径。首要结构比较是QRR对QR_VEC，不能只因超过标量QR就说三元结构必要。

标准化、epoch、阈值仅用TRAIN内部；最终重新同seed拟合，再封存held完整候选分数。独立汇总要求60/60齐全后才join held标签；按seed分别报告，不能把71×3计为213新样本。

当前只有71张完整F输入，且是已打开development面板。这是方案探索，不是全593新模型结果。原五折query身份/组件/图像隔离，但held身份reference作为TRAIN负候选出现过，是固定图库评测。旧头训练用过更大的原外层TRAIN，同71比较也须披露训练规模差异。

本轮能检验所测M以外信息与读出方式是否有增量，不能证明M为唯一因果解释。后续有效分支再补同容量汇总量读出、固定M0下破坏新增证据等控制。目标缺失的23张仍不能靠固定C128重排救回；593全对需要另行解决候选召回，不注入真实答案。

## 文件

- [baseline审计](../results/rc_rebut_qr_qrr_v1/baseline_audit.md)
- [71输入schema](../results/rc_rebut_qr_qrr_v1/evidence_schema.md)
- [独立输入验收](../results/rc_rebut_qr_qrr_v1/evidence_validation.json)
- [协议](../results/rc_rebut_qr_qrr_v1/protocol.json)
- [提交记录](../registry/rc_rebut_qr_qrr_submission_v1_20260927.json)

60项齐全后，results/rc_rebut_qr_qrr_v1将生成main_results.csv、paired_comparisons.csv、joined_predictions.json、report.md、validation.json。当前未声称新增准确率或科学GO。
