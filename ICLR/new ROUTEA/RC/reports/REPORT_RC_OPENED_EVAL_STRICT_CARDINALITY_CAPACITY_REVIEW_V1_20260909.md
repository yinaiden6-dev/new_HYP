# new HYP：总体严格正确数容量的独立数学与源码审查

状态：`OPENED_EVAL_STRICT_CARDINALITY_SOURCE_REVIEW_PASS`。未发现妨碍按既定上限执行的数学或源码阻塞。本报告未运行新 producer、solver、validator 或调度查询；新结果仍须 companion validator 独立通过后才能解释。

## 绑定源码

- `plan/RC_OPENED_EVAL_STRICT_CARDINALITY_CAPACITY_V1_20260909.md`：`290a7c2eee5870ac97957db6242d5d3cfa9dc42ede005886e237424d05e34bc0`
- `programs/analyze_rc_opened_eval_strict_cardinality_capacity_v1.py`：`9a744e57818932c7a8474a54302c92d171d8f1df72b9f2e2bb82ba35ad67cf22`
- `programs/validate_rc_opened_eval_strict_cardinality_capacity_v1.py`：`a100b0fbbe08a7d646f838817c9129d1a64ebfc626cc1f815ac3fdac4e3dad16`

## 数学范围

固定原 EVAL32 的 native6 binary64 端点，加自由实数 bias，共七维系数。RAW 本已正确的 query 要求全部127个 challenger 严格低于0；RAW 原错误的 query 要求 target 严格胜过 HOLD0 和其余126个 challenger。每个 query 因而产生127条同质线性不等式；29条 query 是3683条。有限组严格正margin可共同缩放为单位margin，所以这组精确实数可行性定义自洽。

C(32,29)=4960，允许正确集合改变，不能误写496。保留 RAW25 全部正确的子层是 C(7,4)=35。按 execution 排序的 required29 tuple 枚举，每层内部保持词典顺序；35个 RAW 层先行，另4925个随后。

精确 Farkas 证明要求非负有理数权重之和为1、加权七维不等式向量之和严格为零。其单位右端之和为1，得到矛盾。只要另一个29集合包含证明使用的全部语义行，且原端点重建后的精确向量相同，该证明即可复用。不能从浮点solver状态或仅仅query数推断覆盖。

只有所有4960集合都有精确不可行证明，才排除这一固定实数线性类达到至少29个严格正确；结合已独立检查的原28严格正margin witness，才有严格类最大正确数为28的结论。第一次找到精确可行29集合仅证明存在性，不给全局最大值、不构成训练出的29/32，也不能将其系数投入训练或部署。

## 已独立完成的旧证书核对

本审查直接读取旧 `source_feature_endpoints.json`，按语义三元组重建 Fraction 行，没有运行旧或新solver。四个既有证书均为8条非零行；逐一验证权重非负、总和1及七列加权和全为零。随后独立枚举4960个29集合，确认旧证明覆盖4365、尚缺595；RAW35层覆盖19、尚缺16。这是已有精确证明的组合计数，不是新求解结果。

## 源码检查

- producer 构建全部4064条原语义约束，候选轴与端点经原冻结基线回放；只有精确双证书校验通过才加入池。
- validator 另写 `exact_equations` 从原端点重建 Fraction 系数，检查语义行标识、下标范围、精确双等式、初始证书来源与旧28正margin。它独立枚举全部4960集合，逐项检查实际引用证书覆盖该集合，未把集合个数当证明。
- trace 验证保证只能使用此前发现的证书，已覆盖集合不重复求解；RAW层和总体层的结论独立推导。
- 最多64个未覆盖集合的求解尝试；调用前累计solver工作时间须小于300秒，最后一次允许跨过软预算。原内层 primal/dual 各30秒限制未修改。失败或没有精确证书的数值状态保留 UNRESOLVED。
- 第一精确可行 witness 后不得再调用solver。结束后的全ledger可继续用已验证证书整理覆盖，这不构成新增求解。validator 同时检查停止原因、调用计数、时间累计与全部新证书来源。
- witness 只验证所选29条约束，不预测另外3条；源码明确保留 `new_model_accuracy=None`，没有checkpoint、训练、阈值选择或模型采用。
- 两个新 Python 文件均通过 AST 解析。以上是静态执行路径审查，不能替代新结果的独立验证。

## 解释边界

严格正margin类排除精确平局、零边界和依赖浮点舍入的特殊决策，也不包含其它特征、非线性模型或新数据。若 RAW35 全不可行、总体却可行，只能先说明某些 RAW query 的严格正确保证不能同时保留；没有计算其余3条预测时，不能直接宣称出现实际 RAW break。预算未闭合时必须报告未决，不能把它写成总体上限。既有四个保留原28集合的不可行证书，单独不足以支持总体最大28。
