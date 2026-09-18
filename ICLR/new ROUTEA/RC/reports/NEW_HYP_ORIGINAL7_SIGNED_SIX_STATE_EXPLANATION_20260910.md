# new HYP：原模型已经在学习完整证据的有符号方向

原ORIGINAL7/NATIVE7用检索监督学习六个系数和一个偏置：

\[
z_g=w^\top\phi(q,g;g_0)+b,\qquad
\phi=(RAWgap,dS,dM,d\ell,dQ,dR).
\]

这是**有符号六维仿射读出**：权重学习方向与尺度，偏置参与是否换掉RAW winner的判断。它保留正负号，不是投影点到原点的欧氏距离。对完整C128的127个challenger取最大logit；只有最大值>0才SWITCH，否则HOLD。RAW winner的策略分数固定为0，不能写成模型bias。

非线性已进入φ：候选可见性量M与加权完整reference内容匹配ℓ共同形成S；忽略舍入且未触发质量分母下限时，S=Mℓ。头中的dS、dM、dℓ是相对RAW winner的对比列，并非三个独立物理因素。Q/R列记录扰动响应。

当前证据是同一RAW C128、已打开matched EVAL32上的**25→28，3救0损**。固定参数的完整六列账本给出了具体解释：

| 决策环节 | 已核实的计算事实 |
| --- | --- |
| 救回正确reference | M或ℓ对比列置零，3次救回全部消失。 |
| 选择正确challenger | 0212的target−最强wrong margin为+0.659635，其中RAW项贡献+0.867069。 |
| 允许SWITCH | 0220的target胜wrong达+1.518921，但过零余量仅+0.012491。 |
| 保持原正确 | 25条均由HOLD保留。0618最强wrong为−0.010973，去Q/R变+0.306137而错误SWITCH；去S另会破坏0419。 |

所以，解释原成功需要同时看“身份竞争”和“换不换”的保持门，不能只看J/T二维坐标。以上是**固定模型的计算依赖**；列置零可能破坏自然取值关系，并未证明像素身份特异性或群体泛化。线性读出及这些代数也不是新理论本身。

Retrieval-only指本任务使用身份/正负配对标签，没有新增框、掩膜或点标注；冻结编码器和匹配器的既有预训练监督仍须披露。

扩展清单已冻结128图、24身份、21来源组，目前推进执行资格桥接；此处没有新128图成绩，也不自动宣布HYP成立。

依据：[完整六列账本](../results/rc_original7_six_feature_decisions_v1/result.json)、[回放验证](../results/rc_original7_six_feature_decisions_v1/validation.json)、[128图冻结计划](../plan/RC_ORIGINAL7_EVAL128_FULL_EVIDENCE_V1_20260910.md)。
