# P/V职责恢复与开发桥接草案

2026-09-09。依据用户最新澄清：在原RoMa系统收益基础上建立理论并改善模型，
识别与HYP两项都要。该草案校正模块职责和同名arm混淆，不改写已有NO-GO，
不作为自然V、正式392、action或ownership的执行authority。

## 不能混用的两组三臂

旧retrieval A/B/C：

| 旧名字 | query权重 | reference权重 | 内容匹配 |
|---|---|---|---|
| A_ALL | 1 | 1 | full-reference MaxSim |
| B_QUERY | RoMa w_q(q,g) | 1 | full-reference MaxSim |
| C_PAIRED | RoMa w_q(q,g) | RoMa w_r(q,g) | full-reference weighted MaxSim |

旧B的query map仍依赖candidate reference，不能等同P研究里只读query的
TRUE_QUERY_ONLY。旧C不使用RoMa单点硬指派来限制内容匹配；它不是后来的
P assigned-cosine，也不是下游V的paired-region arm。

现有frozen-P RAW V的三臂：

| V名字 | 定义 |
|---|---|
| ALL_PATCH_RAW_COLNOMIC | 全query × 完整reference cosine MaxSim |
| FROZEN_QUERY_REGION_FULL_REFERENCE_RAW_COLNOMIC | 冻结candidate query支持 × 完整reference cosine MaxSim |
| FROZEN_PAIRED_REGION_RAW_COLNOMIC | 同一pair query支持并集上读取P配对reference atoms的mean cosine |

V后两臂共享pair query union分母，不支持的query点为0；没有P confidence或
RoMa可见性权重输入。若另加w_r，要另行冻结权重来源、对照及独立性边界，
不能称现有零参数RAW V未改变。保留旧w_q/w_r的恢复路径在旧命名下接近C，
不能以“B主臂”名称为由误删reference visibility。

## 新职责的正确含义

P输出目标无关、candidate-conditioned、连通、可重放的区域与对应并封存；
V在固定几何上检验身份内容。完整系统再检验识别收益及HYP作用。将P本身
排序超过强对照设为资格，是已有P-only研究合同的额外要求，不是所有P/V
理论的逻辑必然。该合同下的失败保持不变；新联合开发须显式前向立约。

可以把V_QUERY_FULL作为新桥接的主要身份读出，V_PAIRED用于定位硬对应
约束的作用，同时保留ALL_PATCH。这是预先指定的设计选择，不能从某个
已打开D1面板B/C正确数相同、MRR微小差异，推导B普遍更好或已获替换资格。
旧27/32和69/90对应的冻结完整系统保持基准，新的RAW/C_PAIRED28/32也未
自动替换它。TRAIN32开发结果不能和旧EVAL32直接拼成性能曲线。

## 本轮可交付的输入

从原V8的完整32×C128×REAL/C_BIND/P_COORD保存决策，原样导出FrozenPView：
只保留state、资源keys、grid shapes、query/reference indices和来源hash。
P score、pooling witness数值、head状态、target、base/rank均不得进入V view。
按原generation封存文件核对selected component及所有H0，调用现有
validate_frozen_p_view验证连通性、最小支持、grid范围与hash。

这批导出产物只声称ENGINEERING_STRUCTURAL_P_VIEW_COMPATIBILITY。
原V8的独立分类gate_go=false必须写入manifest；不将结构兼容PASS冒充原
P-only GO。来源是已打开TRAIN32上的已训练P，因此后续只能作为开发桥接，
不能声称独立验证或untouched HYP。PC原P全部H0的退化性质也要保留。

预期交付根：`results/rc_v8_frozen_p_views_v_bridge_v1`。
导出不计算V内容分数、不打开raw tokens或labels、不重选区域、不训练。

## 自然V仍需单独绑定的内容

现有V合同只有E0，未规定可自动执行的自然科学门。接收端必须另外明确：

- 本次是否仅opened TRAIN32开发、准确32×C128和数据来源；
- 明确接受原分类NO-GO但结构可验证的P几何作机制比较，而非伪造P GO；
- RAW token的资源key、坐标框架、编码器/token-space SHA与P view完全一致；
- 使用已有三臂原公式、无P分数、固定几何，先封存全分数再读roles；
- primary身份臂及比较基线在分数读取之前定清，不能看结果再挑主臂；
- B/C差异仅解释这个冻结支持上的内容读取限制；不能直接称HYP/ownership；
- 若接回完整系统，另绑定同一action路径与基线，再联合检验识别和空间依赖。

旧P停止门不会因为本草案而被宣告通过。本草案也不会自动开放正式392，
不会恢复V9或替换旧完整系统。

来源：
`plan/ROUTEA_MATCHED_THREE_ARM_SAME_CAPACITY_7PARAM_CROSSFIT_V1_20260902.md`；
`plan/ROUTEA_FROZEN_P_RAW_COLNOMIC_THREE_ARM_V_ONLY_E0_V1_CONTRACT_20260907.md`；
`src/rc_aslo_xf/frozen_p_raw_colnomic_three_arm_v1.py`；
`plan/RouteA_Latent_Target_Hypothesis_最终方案.md`。
