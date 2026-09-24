# 论文命名澄清：QWEN_ML5 不包含 M×L

2026-09-24。本文补充论文与结果表的术语定义；实验键名、已封存结果及运行程序保持原样。

**QWEN_ML5 同时读取 M、L 两列独立特征，没有计算 M×L。** 名字中的“ML”表示两种输入同时存在；“5”表示四个系数加一个偏置，共五个参数。不能因名字相近，将它与含显式乘积项的 MASS5／当前 PRODUCT5 视为同一种头。

## 推荐的论文表名与精确特征

下表适用于本轮 2026-09-24 Qwen 联合校准、简化外部迁移及内部 M V3 实验。M 是候选配对整体质量，L 是自由内容分数。

| 实验键名 | 推荐展示名称 | 输入特征（另加一个偏置） | 显式 M×L 项 |
|---|---|---|---|
| QWEN3 | Qwen 基础校准，3 参数 | RAW 标准化差分、Qwen 标准化差分 | 无 |
| QWEN_L4 | Qwen＋L，加性，4 参数 | 上述两列、L 对称差分 | 无 |
| QWEN_M4 | Qwen＋M，加性，4 参数 | 上述两列、M 对称差分 | 无 |
| QWEN_ML5 | **Qwen＋M＋L，加性，5 参数** | RAW 标准化差分、Qwen 标准化差分、M 对称差分、L 对称差分 | **无** |
| QWEN_ML_PRODUCT6（追加实验） | **Qwen＋M＋L＋M×L，6 参数** | QWEN_ML5 原四列，再增加 M×L 对称差分 | **有；结果须单独报告** |
| ADDITIVE4 | RAW＋M＋L，加性，4 参数 | RAW 标准化差分、M 对称差分、L 对称差分 | 无 |
| MASS5／当前 PRODUCT5 | **RAW＋M＋L＋M×L，含乘积项，5 参数** | RAW 标准化差分、M×L 对称差分、M 对称差分、L 对称差分 | **有** |
| INTERNAL3（V3） | 内部 M 条件化内容，3 参数读出 | RAW 标准化差分、条件化内容 Lθ 的对称差分 | 末端无显式乘积项；真实 M 由 PRE_REAL 内部适配器读取 |

不同头即使参数总数相同，也可能读取不同信息。历史档案中的 PRODUCT5 还须核对具体 L／S 定义，不能跨实验仅按名称合并。

## 运算定义

令 w 为原 RAW 第一名，g 为挑战者。定义：

\[
\delta_T(g)=\frac{T_g-T_w}{\max(\operatorname{std}_{C128}(T),10^{-12})},\qquad
d_T(g)=\frac{T_g-T_w}{|T_g|+|T_w|+10^{-12}}.
\]

QWEN_ML5 的挑战分数为：

\[
z_g=a\delta_{\mathrm{RAW}}(g)+b\delta_{\mathrm{Qwen}}(g)+c\,d_M(g)+d\,d_L(g)+e.
\]

这里没有乘积项。当前 MASS5／PRODUCT5 则为：

\[
z_g=a\delta_{\mathrm{RAW}}(g)+b\,d_{ML}(g)+c\,d_M(g)+d\,d_L(g)+e,
\qquad
d_{ML}(g)=\frac{M_gL_g-M_wL_w}{|M_gL_g|+|M_wL_w|+10^{-12}}.
\]

乘积头是先计算各候选的 M×L，再做对称差分；通常不等于先差分再相乘的 d_M×d_L。两者也不能混写。

用户要求追加的 QWEN_ML_PRODUCT6 在原 QWEN_ML5 分数上新增一个可学习的 `f × d_ML(g)` 项。沿用原五折、候选池和两种损失，直接与各自 QWEN_ML5 比较；见[追加乘积项方案](../plan/RC_H593_QWEN_QUALITY_PRODUCT_V1_20260924.md)。原 536／504 数字仍属于不含乘积的 QWEN_ML5，不能挪作新臂成绩。

## 结果可以支持什么

H593 原五折、ColNomic 自然 C128 上，QWEN_ML5 的 CE 为 536/593、COST1 为 504/593；各自 QWEN3 基线为 523/593、477/593。**这些结果属于加性 Qwen＋M＋L 校准，不能用作 M×L 交互有效的证据。** 同协议 QWEN_M4 也得到 536/504，正确集合与 QWEN_ML5 一致，因此本次观察到的新增纠错不需要新增 L 列来实现。CE 主臂的分组增益区间仍跨零，COST1 为预声明次臂，不能事后交换主次。

MASS5 与 ADDITIVE4 的比较才直接检验本轮外部简化头是否需要显式乘积列。内部 V3 另行检验真实 M 能否影响内容表示；其末端不直接读 M，不应写成一个普通的 L＋M 加性头。以上都是不同问题。

## 源码与结果依据

- [Qwen 联合头源码](../programs/run_rc_h593_qwen_quality_joint_v1.py)：ARMS 与 cache 的四列定义；变量 `ml` 是 `column_stack` 的两列矩阵，并非乘法。
- [简化外部头冻结源码](../programs/run_rc_simple_external_source_freeze_v1.py)：MASS5 乘积列定义。
- [内部 V3 源码](../programs/run_rc_internal_m_learned_use_v3.py)：INTERNAL3、ADDITIVE4、PRODUCT5 的 FEATURES 与 features。
- [三项工作完整结果](REPORT_NEW_HYP_THREE_REMAINING_CLAIMS_COMPLETED_20260924.md)：面板、主次臂、救回／误伤与统计边界。
- [Qwen 完整结果与核算](../results/rc_h593_qwen_quality_joint_v1/report_zh.md)、[固定简化头外部迁移](REPORT_SIMPLE_EXTERNAL_TRANSFER_V1_20260924.md)。
