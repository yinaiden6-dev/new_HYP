# new HYP：当前校准机制的联合证据收口

本报告在5138847、5138852以及两组固定头组干预全部独立验证后写成。
它保留正向发现，同时区分不同head、不同数据、固定参数干预和重新
优化。当前研究仍为retrieval-only任务监督；不要求旧空间Reference
HYP或ownership通过，也不以改名宣称普遍识别定律。

## 1. 已测简单替代没有复现完整模型

同一原RAW C128、完整127 challenger HOLD/SWITCH、已打开matched
EVAL32，PAIR64+TRAIN32同一固定训练协议：

| 重训头 | 输入概要 | 正确数 | 相对RAW25救/损 |
| --- | --- | ---: | ---: |
| RAW2 | RAW gap | 25 | 0/0 |
| RAW_PLUS_M3 | RAW+质量M | 26 | 2/1 |
| RAW_PLUS_L3 | RAW+normalized内容L | 26 | 1/0 |
| JOINT4 | RAW+M+L | 26 | 2/1 |
| ORIGINAL7 | 原全部六特征 | 28 | 3/0 |

原完整模型分别胜两个重训单通道2救0损，group平均差为正。这项预定
次比较是正向内部证据，不应被JOINT4主简化门未过的状态盖住。它仅
排除了这些明确训练条件下的简单替代，未排除所有函数类最优解。
L仍包含visibility，M臂也仍保留RAW内容先验。

证据：results/rc_retrained_evidence_sufficiency_v1/result.json，SHA
20102901242c734c2b5c0404e0e5571999ab1dc25ceefa5cee2f37f884a12601。

## 2. 重训2×2给出的条件贡献

| S组 | Q/R组 | 头 | EVAL32正确数 |
| --- | --- | --- | ---: |
| 无 | 无 | JOINT4 | 26 |
| 有 | 无 | PRODUCT5 | 27 |
| 无 | 有 | RESPONSE6 | 26 |
| 有 | 有 | ORIGINAL7 | 28 |

补S的重训路径恢复0212的目标候选选择；在S已保留时补Q/R的重训路径
保护RAW原本正确的0618。四项条件比较全部报告，没有只挑一种背景。
两个较小头都没保留原全部28个正确，但各组的正向条件增量仍成立。

证据：results/rc_product_response_factorial_v1/result.json，SHA
 e9dae1e5837e72d3f38ba3e4473d0ff555b6ee90fadfbe3d3dbf64376759c0bf。

## 3. 直接输入效应与重新优化必须分开

固定同一NATIVE7权重，再将派生特征组置零，仍以原六维X@w+b和完整
127 challenger重选，结果为：

| 固定头条件 | EVAL32正确数 | 相对原28损失 | 原3个RAW rescue保留 |
| --- | ---: | --- | ---: |
| ORIGINAL7 | 28 | 无 | 3 |
| DROP_S | 26 | 0419、0618 | 3 |
| DROP_QR | 27 | 0618 | 3 |
| DROP_S_QR | 26 | 0419、0618 | 3 |

因此不能把重训后0212失误说成“S原直接项独自救回0212”：固定去S时
0212仍正确，失误出现在共享权重重新优化后的候选排序。对0618则有
直接证据：原最强wrong logit为−0.010973，去Q/R后为+0.306137；删列
后重训PRODUCT5降至+0.053113，仍错误SWITCH。

完整每challenger分解将原头与删列重训头差分为原删除组项、共享权重
改变量、bias改变量，最大FP64重组残差7.88e-15。实际动作按原算术
重新计算，而非依赖近似分解值。

证据：results/rc_frozen_group_effect_native64_v1/result.json，SHA
 eebbd6e9b15be6181ed375f04f6c8cbc7a985f5f871073c41b29c1f03a5975fe。
独立验证：同目录independent_validation.json，SHA
 c0d4fd6e729d3e23e36fc3b2533cb92da823c33931569a740c88539f1e2445cd。

## 4. 旧69/90说明响应作用不普遍同向

以下是另一组旧FROZEN_C参数、原RAW C128、opened difficult90，原
RAW61→69，逐候选(weight*features).sum()+bias；不能与NATIVE7的
28/32拼成同一模型或合并分母。

| 固定旧头条件 | 正确数 | 相对原69新增/损失 | 原8个rescue保留 |
| --- | ---: | ---: | ---: |
| ORIGINAL | 69 | 0/0 | 8 |
| DROP_S | 68 | 3/4 | 8 |
| DROP_QR | 70 | 1/0 | 8 |
| DROP_S_QR | 65 | 3/7 | 8 |

DROP_QR在旧90上确实多纠正一条（ordinal67），而在当前NATIVE7的
32条上损失0618。两边已独立验证，这限定了理论措辞：不能称Q/R响应
天然处处有益或普遍必要。70是已打开数据上的固定头诊断结果，尚未
作为部署模型；同旧head的matched32桥接也已独立验证：原27→DROP_QR27，但新增0220、
损失0618，1救1损。因此该修改没有在两组旧head结果上同时保持无损。

两组的原始rescue在固定移除这些显式S/Q/R项后都保留；这些项还影响
新救回、误切换和重新优化。不能把“完整模型效果更好”简单等同于
每个rescue直接依赖每个输入项。

证据：results/rc_frozen_group_effect_difficult90_v1/result.json，SHA
 c27f7c3aa543ba73130be059e1047415d86a35dc2fb17ffe5c00febfbbdb3356。
独立验证：同目录independent_validation.json，SHA
 13ce0b1d7d0bce7a0264f03f010466230b6def55236198bee5375166f7349b76。

## 5. 现在能写成什么理论与发现

可写的计算框架是：候选reference诱导质量统计和加权内容相容性；
共享校准使用其相对比较、乘积比较和响应统计，在基础检索先验上
形成保守的完整候选决策。训练监督只有身份/正负检索关系，本任务
无空间标签；冻结基础模型的预训练来源另行披露。

乘积比较的数学关系dS=(dM+dL)/(1+dM*dL)在正数、无epsilon时成立，
解释为何线性head中仍有M/L的非线性基函数。完整epsilon/FP64条件
和TRAIN数值重放已说明；这不是新代数定律，也不单靠公式证明泛化。

当前实验支持“完整校准在既有协议下优于已测简单替代”，同时给出
其作用有赖于具体参数、数据和训练路径的证据。不能写成“精确空间
绑定必然带来增益”“所有响应项普遍必要”或“所有相似物体都过目不忘”。
未满足的部分应保留：跨TRAIN group的稳定性、未触碰的外部确认、
新reference登记后的长期保持以及普遍最优性。目前没有发现保证任意
改法提升的瓶颈开关；已有受验证的增益也没有因简化试验失败而消失。

## 6. 继续工作

进行全12个TRAIN supergroup逐组删除的预定稳定性分析，PAIR64与
EVAL32保持完整，全部四头、全部删除条件报告，不挑最佳子集或改阈值。
这检验内部训练敏感性，不生成12个独立外部实验。旧FROZEN_C
DROP_QR的matched32桥接已完成，确认70/90的提升伴随原27/32正确集合
中的一条损失。桥接来源：results/rc_frozen_qr_removal_matched32_v1，
result SHA e06a01ed4d820df8e15776a1fbc130d359814fa1f07f1683656e8fe272553e2e，
独立验证SHA 3fb7fda90c00f874fb27a9fe08eca8d29e3a42b1274de1f6678fde7b5fb7ae79。

文献范围已核对：ColBERT、PFE、MagFace、AdaFace已有late interaction
和质量/不确定性识别先例，new HYP不能声称首次提出通用联合思想。
实际贡献应围绕具体机制、任务监督和可复核对照，而非去掉模型名称。
见reports/NEW_HYP_PRIMARY_LITERATURE_SCOPE_V1_20260909.md。


## 7. 已完成的训练稳定性与严格保留集合容量

5138859完整52+52重训验证通过。在12个既有TRAIN组逐个删除时，
PRODUCT5对JOINT4为8次净增、4次持平，ORIGINAL7对RESPONSE6为11次
净增、1次持平，S组两种背景均无负向净差。ORIGINAL7对PRODUCT5的
Q/R增量为5正6平1负；完整头正确数26–28、median27。
0044/0212正确12/12，0220正确3/12，0618正确9/12。前两次纠错稳健，
第三次纠错和部分误切换保护有训练敏感性。相关删组不是12个独立
外部实验；不能用它生成伪独立置信结论。

原六特征的严格保留28+指定一错容量诊断也已独立精确验证：四个
3683约束系统全部不可行，每个由8条语义约束的非负组合推出0≥1。
它说明保留这个完整正确集合的严格线性扩展不可能，但不排除牺牲
个别旧正确后总体达到29；不能把它写为无条件总体28上限。更宽的
总体29检查也已完成：允许集合改变时精确可行；保留RAW25的35种
目标集合均严格不可行。因此应报告识别数量与严格保持要求的冲突，
不能称总体正确数最多28。所有证书都使用已打开EVAL
标签，只作诊断，不是新模型或29/32成绩。

容量源：results/rc_opened_eval_strict_no_break_capacity_v1，结果SHA
3194c5fa41c8d49d37adfb3f262446f84fbf7ebacaba6aec7b88e493b3536870，
独立验证SHA bd803ccf5bce7edce82c4bc5a5d3f25dc6b44ced76c9197f8a9350c72a3ee8d9。
稳定性源：results/rc_train_group_jackknife_stability_v1，结果SHA
ce7eaeb9a3407e4626bb2bf00d8f8c2c0c328ce20ae28215f17069db1ec21536，
独立验证SHA 9d191cb11e24b4b3f98e500f104555e680361a9ade64d7300c39390410d5db88。


总体容量最终报告：reports/REPORT_NEW_HYP_LINEAR_CAPACITY_AND_RETENTION_CONFLICT_V1_20260909.md。
该结论允许个体取舍并保留严格margin边界；没有新的29/32模型成绩。
