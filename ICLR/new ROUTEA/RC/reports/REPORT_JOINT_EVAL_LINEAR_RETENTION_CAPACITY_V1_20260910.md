# 联合保持容量：原六特征有局部增量空间，不是新模型成绩

固定原ORIGINAL7/NATIVE7六维特征和RAW自然C128，允许一个共享实数七参数线性头变化；同时要求严格保住旧EVAL32原28正确、新EVAL128原99正确。对新128候选在场的22条原错误逐条要求新增正确，完整检查每query全部127 challenger；每系统16256条不等式。7条召回缺席仍保留在人口账本，不能由本次reranking变量救回。

**结果为5个分别精确可行、16个精确不可行、1个数值未决。原实际EVAL成绩仍是旧28/32、新99/128。**

五个可行病例为：DIFFICULT-0028, OUTCOME-0270, OUTCOME-0133, OUTCOME-0809, OUTCOME-0818。每个有独立的精确有理witness，分别在保持原127张正确的同时满足该病例的严格正确约束；不是同一组参数同时修好五张，也没有计算或报告witness的新模型准确率。

这使“原六特征完全没有保留旧正确再增加新正确的表达空间”不再成立。16个不可行病例则有精确Farkas证书，表明它们各自与原127保持要求在此严格线性类中存在冲突。0337也属于这个不可行集合，不能期待任意重调原七个系数就同时修复它、严格保住全部原正确。

## 全22条结果

| 新128原错误 | 严格线性保持容量 |
| --- | --- |
| DIFFICULT-0099 | 精确不可行 |
| OUTCOME-0106 | 精确不可行 |
| OUTCOME-0809 | 分别存在精确可行解 |
| DIFFICULT-0028 | 分别存在精确可行解 |
| OUTCOME-0446 | 精确不可行 |
| OUTCOME-0819 | 精确不可行 |
| OUTCOME-0278 | 精确不可行 |
| OUTCOME-0821 | 精确不可行 |
| OUTCOME-0337 | 精确不可行 |
| DIFFICULT-0098 | 精确不可行 |
| OUTCOME-0270 | 分别存在精确可行解 |
| OUTCOME-0096 | 精确不可行 |
| OUTCOME-0815 | 精确不可行 |
| OUTCOME-0133 | 分别存在精确可行解 |
| OUTCOME-0769 | 精确不可行 |
| OUTCOME-0435 | 精确不可行 |
| DIFFICULT-0011 | 数值未决 |
| OUTCOME-0105 | 精确不可行 |
| OUTCOME-0138 | 精确不可行 |
| OUTCOME-0818 | 分别存在精确可行解 |
| OUTCOME-0407 | 精确不可行 |
| OUTCOME-0141 | 精确不可行 |

## 独立验证与范围

160条原ec7动作、20320个原logit逐bit重放，原28/99正确集合、target身份唯一性、C128轴和7个缺席均复核。五份primal的全部16256条有理margin精确>=1，16份dual的非负权重和为1、七列加权和精确为0；独立validator没有调用solver。

由于仍有1个数值未决，总表验证状态为INDEPENDENT_VALIDATION_UNRESOLVED；这不使已逐份复核的21份证书失效。主问题“至少存在一个严格保留增一解”已由五份witness证明，完整22病例分类仍有一个空缺。未决不是不可行，不增加求解预算追求齐整结论。

证明针对原binary64端点作为精确有理数的严格正margin实数线性类。HOLD=0、physical-row tie、特殊FP64舍入、非线性读出或允许部分原正确损失不在严格不可行结论内。数学可行不保证TRAIN能学到或泛化。

所有证书与端点在rc_opened_*目录隔离，不读取到任何后续训练，也不据此挑训练子集、复制参数或调整阈值。没有新模型checkpoint、encoder/RoMa或部署修改。

## 实际下一步

只用既定TRAIN128标签，检验围绕各折原BASE7的最小L1改动训练：将原TRAIN正确集作为必须保留的动作验收条件，再寻找能够修复TRAIN错误的候选参数；全部选择只发生在该折TRAIN，四折封存后才评价留出组。新学习规则不使用上述EVAL系数/方向，不新增特征。它检验训练保护能否泛化，而不把本页数学证书当作性能突破。

5139464已COMPLETED0:0，运行62秒；22系统LP搜索合计约8.02秒，剩余为输入重建与精确证书复核。

- [隔离结果](../results/rc_opened_joint_eval_linear_retention_capacity_v1/result.json)
- [独立验证](../results/rc_opened_joint_eval_linear_retention_capacity_v1/independent_validation.json)
- [冻结容量计划](../plan/RC_OPENED_JOINT_EVAL_LINEAR_RETENTION_CAPACITY_V1_20260910.md)
- [只用TRAIN的新学习计划](../plan/RC_TRAIN128_PROTECTED_PROJECTION_OOF4_V1_20260910.md)
