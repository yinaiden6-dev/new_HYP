# H593六项特征：候选排序容量定位

本轮没有训练或评估新模型。原COST1=481、CE=486、GAP_BIAS2=492均保持；下表是已打开标签下的数学容量诊断，不是新准确率。

全部593：426个RAW正确，23个RAW错误且target缺失，144个RAW错误且target在C128。对全部144检查完整127挑战者；其中104个原最高挑战者正确，40个错误。

| 范围 | 精确判定 | 数量 |
|---|---|---:|
| all144 | EXACT_CONVEX_HULL | 1 |
| all144 | LINEAR_SEPARABLE | 143 |
| blocked40 | EXACT_CONVEX_HULL | 1 |
| blocked40 | LINEAR_SEPARABLE | 39 |
| shared40 | EXACT_INFEASIBLE | 19 |
| shared40 | STRICT_FEASIBLE | 20 |
| shared40 | UNRESOLVED | 1 |

## 40个排序障碍的两层交叉

| 单query与同折保留后扩展 | 数量 |
|---|---:|
| EXACT_CONVEX_HULL__EXACT_INFEASIBLE | 1 |
| LINEAR_SEPARABLE__EXACT_INFEASIBLE | 18 |
| LINEAR_SEPARABLE__STRICT_FEASIBLE | 20 |
| LINEAR_SEPARABLE__UNRESOLVED | 1 |

LINEAR_SEPARABLE表示存在query专用六维线性方向使target严格第一；EXACT_CONVEX_HULL表示不能得到唯一线性第一，不排除平分，也不证明非线性无用。EXACT_FEATURE_COLLISION只限制六维pointwise评分。UNRESOLVED不作原因结论。

共享扩展按原头所属折分别进行：保留本折原正确最高挑战者，再逐个添加一张当前错误。STRICT_FEASIBLE只证明该组排名可以由一个共享六维方向同时实现；EXACT_INFEASIBLE只证明保留该集合时有冲突。允许损失后的群体净增仍可能，保留条件不是晋级门。多个分别可行的扩展不能合并理解为可同时救回。

所有结论由原FP64端点的Fraction精确运算复核，浮点solver仅找witness或凸组合。独立验证器在新进程重读全部输入SHA与证书，没有调用优化器。证书参数不得用于训练、初始化、阈值选择或预测成绩。

现有证据仍不能仅凭“单例可分”断言统计接口信息充分，更不能据此证明跨组可学习性。需结合共享冲突与后续TRAIN-only学习读出判断。

证据：`results/rc_opened_h593_rank_capacity_v1/result.json`、`validation.json`、`per_query.csv`。协议：`plan/RC_OPENED_H593_RANK_CAPACITY_V1_20260921.md`。
