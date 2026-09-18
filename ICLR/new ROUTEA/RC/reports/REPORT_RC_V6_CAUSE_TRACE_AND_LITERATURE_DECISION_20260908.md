# V6原因追踪、两簇复算及文献后的执行决定

2026-09-08。用户要求今日全力突破；GO仍以原冻结门为准。

## 已确认的两处机制缺口

5个target所选手/桌面支持（case01/02/06/07/08）的平均RoMa往返误差为
20.086、15.246、27.562、6.155、7.444个query patch cell；四点全部未回到
所选支持。生成器只检查前向邻接，所以它们仍然合法。坐标和序列化已复核。

同品牌错误则是另一种情况：case04 wrong往返误差仅0.0067格，仍是错误规格。
可靠对应对身份判别不充分，不能期望一个cycle阈值同时解决这两种问题。

轨迹图及完整数值：
`results/rc_competitive_witness_v6_cycle_two_support_v1/result.json`。
对应图位于同目录figures/case01_roundtrip.png至case09_roundtrip.png。

## 固定两簇并集没有构成解法

只把已固定Ht和Hw并为8个query位置，对完整C128的同一reference贯穿8格
求c均值。没有重选支持或对手。9例的target-minus-all-wrong-upper差值：

| case | 固定8格c差 |
|---|---:|
| 01 | -0.179374 |
| 02 | -0.124415 |
| 03 | +0.002137 |
| 04 | -0.034046 |
| 05 | -0.131492 |
| 06 | +0.090887 |
| 07 | -0.072333 |
| 08 | -0.137883 |
| 09 | -0.144887 |

只有两例为正，且case06两个query支持都是桌面。各候选这8格均无missing，
所以这一现象不能归因于missing上界。8格并集可能不连通且由标签辅助定义，
它只是描述性算术，不是rescue、retrieval或新的H1。本结果不测试包含未选
规格字段的更大region，也不能否定真正的品牌＋规格联合证据。

## 一跳预测检验

结果：`results/rc_v6_anchor_one_ring_prediction_v1/result.json`。
对Ht/Hw分别用四个已保存forward点拟合局部affine，测Manhattan一跳外邻域。
target/wrong各自对同一anchor取其已有对应，不重选点；reference token重用
原输入。36个cell中6个因query四点共线记UNDETERMINED，30个可计算。

几个关键反例：

- case03 target anchor预测外圈误差约0.062ref-cell，已有ring c对target较好；
  该种子确有可延伸局部信息，但不等于自动选择或完整C128GO。
- case04 target/wrong自身anchor预测外圈误差约0.044/0.028ref-cell，两者均
  稳定。这再次说明同品牌错误无法靠更强局部geometry单独排除。
- case08 wrong的木纹→结构示意图支持，预测其周边已有RoMa field误差仅
  0.045ref-cell。这是错误field也可局部平滑的反例；其仿射cell determinant
  约0.0135，结合大roundtrip误差，不能把小外圈拟合误差当成真实配对证明。

因此需要相互制约的几何解释，单向平滑和单向可延伸都不充分。

## 原始文献对照及可用理论结论

RoMa v2 §3.3说明precision只对共同可见且小误差对应学习。因此把其数值
作为任意背景/品牌配对的可靠置信没有依据。overlap也不等于具体产品身份。
来源：[RoMa v2](https://arxiv.org/html/2511.15706v1)。

NCNet在完整对应空间中使用邻域一致性来缓解重复纹理歧义，启发我们把
约束放在形成对应/region的阶段，而不是只重加权错配之后的分数。
来源：[NCNet](https://arxiv.org/abs/1810.10510)。

SuperGlue通过partial assignment和真实unmatched监督处理匹配与未匹配，
不支持把任意score的零点当null；仅过滤固定初始matches也无法新增初始
匹配中不存在的正确对应。来源：[SuperGlue](https://arxiv.org/html/1911.11763v2)。

本项目可严格成立的有限命题是：相同的保留观测无法被确定性readout区分；
增加块只有带来新的可辨信息才可能改变这一点。另一方面，连通性、精确
局部几何和局部竞争获胜都不是具体身份正确的充分条件。上述反例分别展示
它们缺少什么，不能被打包成“所有HYP不可能”的结论。

## 执行决定

下一步只开一个generation feasibility：native-cell cycle closure上的
maximal connected regions，取消按固定四格封顶，但不扫描区域尺度。
目的：把已实证不闭合的背景支持排除在legal family之外，再检查target
coverage是否仍达到26/32。该门通过也不宣称HYP已证明，之后共享selector
和身份counterevidence仍要过完整原控制/泛化门。
合同：`plan/RC_CYCLE_CLOSED_REGION_GENERATION_QUALIFICATION_V1_20260908.md`。

这些诊断没有新backbone forward、head训练或正式panel消费。已知NO-GO均保留。
