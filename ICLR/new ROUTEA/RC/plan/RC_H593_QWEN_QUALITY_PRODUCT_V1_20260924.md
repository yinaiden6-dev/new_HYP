# H593：Qwen 加性校准增加一个质量–内容乘积项

2026-09-24。用户明确要求在 QWEN_ML5 上增加乘积项。新建实验，保留原程序、authority、结果与主次结论。本轮只增加这一项，不修改正在运行的内部 M 实验。

## 问题与固定比较

原 QWEN_ML5 读取 RAW、Qwen、M、L 的四列差分加偏置，没有 M×L。新增 **QWEN_ML_PRODUCT6**：保留原四列及定义，追加一列对称乘积差分，共六个参数。

令 w 为 RAW 第一名、g 为挑战者，新增列为：

\[
d_{ML}(g)=\frac{M_gL_g-M_wL_w}{|M_gL_g|+|M_wL_w|+10^{-12}}.
\]

这是先计算各候选 M×L 再做差分，不是对称 M 差分乘以对称 L 差分。原四列依次为 RAW 标准化差、Qwen 标准化差、M 对称差、自由 L 对称差，均保持不变；第六个参数为偏置。新增系数置零可嵌入原加性头。

- 预先固定主比较：**CE_QWEN_ML_PRODUCT6 对 CE_QWEN_ML5**。
- 次比较：**COST1_QWEN_ML_PRODUCT6 对 COST1_QWEN_ML5**。
- 同时描述对相同损失 QWEN_M4、QWEN3、RAW 及直接 Qwen 排序的救回／误伤，不在结果出来后交换主次或选择损失。
- 这是一项追加特征对照。正结果可支持本配方中乘积特征具有增量，不能单独排除额外参数／其他非线性特征也有效；本轮不追加网格搜索。

## 数据和训练完全沿用

使用已有 H593 缓存、ColNomic 自然 C128、原五折身份／来源 component 隔离；全部593张均评估，目标缺席23张仍计入分母。M 为配对整体质量，L 为完整 reference 的自由内容 MaxSim。只读取已有缓存，不增加编码器或 RoMa 前向。

旧四臂、两损失、五折结果直接复用：逐折核对四列原特征、旧参数及全部127个挑战者分数／决策，不重复训练旧基线。只有新六参数臂的 **2种损失×5折=10次拟合**。

新增头沿用零初始化、FP64、AdamW、学习率0.03、weight_decay 0.001、2000步、seed17、8CPU线程、最终步模型。训练仅使用各折原TRAIN中目标在C128内的query；禁止读取held身份、curator、其他折训练结果或报告。已有旧fold payload只用于无标签基线重算，不作为训练特征或监督。

动作规则沿用 RAW winner 的 HOLD=0，最高挑战分数严格大于0才 SWITCH；保留原候选轴与并列规则。无held阈值选择、早停选择、外部数据选择或新增损失。

## 验收和报告

提交前冻结源码、协议、launcher、父authority、缓存与原折结果的哈希。验证乘积列与独立标量公式一致、原四列不变、乘积系数置零恢复加性预测。训练保留参数、优化器和步数断点；每100步可保存，超时以同Job ID续跑。

五折预测与验证文件先封存，再打开curator核算。独立逐项点积复核全部新增候选logit，保留127个挑战分数、HOLD=0、逐query物理行号、训练拟合统计。

报告全体和逐折正确数、MRR@C128、救回／误伤／净增／动作改变、旧正确损失率，以及原component的等组与query加权cluster bootstrap 95%区间。检验的是已多次使用H593上的开发性分组效果，不是独立外部确认。仅正确数增加但区间跨零时须保留不确定性；不预先声称乘积有效。

## 执行

- 程序：`programs/run_rc_h593_qwen_quality_product_v1.py`。
- 输出：`results/rc_h593_qwen_quality_product_v1/`。
- Authority：`registry/rc_h593_qwen_quality_product_authority_v1_20260924.json`。
- Launcher：`slurm/rc_h593_qwen_quality_product_v1.sbatch`。
- 五折CPU数组最大并行5；每次8CPU、24GB、10分钟、预算450秒，断点续跑最多16次；全部成功后自动汇总。
- 不取消、不迁移其他任务；不修改原Qwen、简化外部头或内部M V3的封存状态。
