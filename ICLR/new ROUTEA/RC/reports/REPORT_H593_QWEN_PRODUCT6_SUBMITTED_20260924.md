# Qwen 增加 M×L：已实现并提交

## 完成更新（2026-09-24）

5161651 五折及 5161652 汇总均已 COMPLETED、exit 0:0；10/10拟合完成。五折每次45–49秒，汇总36秒。原提交时的PENDING记录保留在下方，已不是当前状态。

| H593原五折、ColNomic自然C128 | 旧加性ML5 | 新乘积PRODUCT6 | 对旧ML5救回／误伤／净增 |
|---|---:|---:|---:|
| CE（主比较） | 536/593 | 537/593 | 2／1／+1 |
| COST1（次比较） | 504/593 | 504/593 | 0／0／0 |

CE两种分组bootstrap区间均跨零，当前不支持乘积项有稳定额外收益；COST1正确集合不变，仅一张错误图改选了另一错误候选。独立复核50条SHA绑定、150,622个新增挑战者分数及汇总，最大数值差1.42e−14。

[完整结果](../results/rc_h593_qwen_quality_product_v1/report_zh.md) · [逐query及分组统计](../results/rc_h593_qwen_quality_product_v1/result.json) · [最终验收](../results/rc_h593_qwen_quality_product_v1/validation.json)

## 原提交记录

2026-09-24。根据用户要求，在原 QWEN_ML5 的 RAW、Qwen、M、L 四列差分上增加乘积差分，新增 **QWEN_ML_PRODUCT6**（五列输入＋偏置）。运算是 `sym(M_g×L_g, M_w×L_w)`，不是 `sym(M)×sym(L)`。

- **5161651_[0-4%5]**：五折CPU训练，每折拟合CE和COST1两种损失，共10次新增拟合。
- **5161652**：依赖全部五折成功，自动封存、独立核算与汇总。
- 两项均为 `cpuonly`，每次8CPU、24GB、10分钟；保存优化器断点，可同Job ID续跑。提交后核查时训练为PENDING，汇总为PENDING/Dependency；尚无新科学结果。
- 调度器脚本与冻结源文件逐字节一致；数组并行上限5，汇总依赖为 `afterok:5161651_*`。

数据沿用H593原五折与ColNomic自然C128，593张全部计入、570张目标在候选内。原模型和全部旧结果保留，旧头不重训，不新增GPU前向，也不修改内部M V3任务。

主比较固定为CE乘积头对CE加性ML5，COST1为对应次比较；将报告逐折及总体正确数、救回／误伤／净增、MRR与分组区间。原加性头的536/593、504/593不是新六参数头的成绩。

提交前检查：75,311个乘积特征与独立标量公式完全一致；新增系数为零时，原加性头两损失共1,186个决策全部复现。损失值／梯度、HOLD并列规则、零质量数值、两种损失的优化器断点恢复以及launcher六种控制路径检查均通过。源码经独立审查后冻结。

入口：[方案](../plan/RC_H593_QWEN_QUALITY_PRODUCT_V1_20260924.md)、[程序](../programs/run_rc_h593_qwen_quality_product_v1.py)、[任务记录](../results/rc_h593_qwen_quality_product_v1/jobs.json)、[调度核查](../results/rc_h593_qwen_quality_product_v1/submission_verified.json)、[独立特征检查](../results/rc_h593_qwen_quality_product_v1/independent_feature_preflight.json)、[自测记录](../results/rc_h593_qwen_quality_product_v1/self_test_receipt.json)。最终结果将写入 `results/rc_h593_qwen_quality_product_v1/result.json` 与 `report_zh.md`。
