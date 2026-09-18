# 原RAW+RoMa成功的实际机制，以及P接口丢失的信息

2026-09-09。当前已定位一条旧成功实际依赖、后续P接口未保留的信息通道。
这比“区域太小”“应该换OT”“换一种pooling”具体，但尚不足以声称解释了
所有HYP失败或取得空间HYP GO。

## 对实际原系统的完整重放

使用产生27/32与69/90的原RAW C128、原缓存标量和冻结六特征七参数head。
每种干预均重算全部127个challenger，保持原零阈值、tie和身份去重语义。
首先逐条复现原预测和FP64 logits，再进行固定的输入项中和；没有训练。

| 原冻结计算图或指定干预 | current-runtime EVAL32 | opened difficult90 |
|---|---:|---:|
| RAW base | 25 | 61 |
| 原完整head | 27，2救0损 | 69，8救0损 |
| 仅中和显式visibility-mass差异项 | 25，两原救回均失去 | 61，八原救回均失去 |
| 仅中和normalized-local-similarity差异项 | 24，两原救回均失去、另1破坏 | 61，八原救回均失去 |
| 仅中和RAW gap项 | 25，1原救回失去、2个base正确被破坏 | 52，19个base正确被破坏 |
| 仅中和Q差异项 | 27，3救1损 | 70，9救0损 |
| 仅中和R差异项 | 27，3救1损 | 70，9救0损 |

两个数据集分别解释，不能相加成独立样本。EVAL是adaptive internal
compatibility，difficult90是已打开回归集；这些操作不能用于挑选新的部署
参数或继承未经验证的外部结论。70/90是干预后的观察，不是新模型GO。

两组各自全部原救回的实际选中challenger上，mass项与normalized-content项
提供正贡献；RAW gap、local-score项、Q/R项与bias提供负贡献。这里讨论
具体特征值乘固定权重后的作用，而不是凭权重大小猜测重要性。

例如EVAL的DIFFICULT-0044：mass贡献+2.73378，归一化内容+3.89308，其余
项及bias抵消后最终logit为+1.34445。OUTCOME-0212的两个正项分别为
+5.32599与+5.73137，最终logit为+1.68520。

Q/R显式差异项在这些救回上起抑制作用，解释了“破坏坐标后仍保留救回”
为何与旧成功不矛盾。不过原wq/wr仍参与内容评分，不能据此推出全部几何
无用或整个模型反空间。

## 排除只删一项造成的误解

旧local score可写成S=M L；Q/R控制分数同样有全局M因子。直接把head的
mass特征设为零，并未删除其他特征中隐含的M。

因此另做唯一的因子一致干预：逐candidate保留原
L=S/M、Lq=Sq/M、Lr=Sr/M，统一M'=1，重建四个旧输入标量，再走原head。
全部34560个归一化内容比值逐bit保持；11520个候选中没有M落入原1e-12
分母保护区间。

结果为difficult90 **69→61，八次原救回全失，90条全部HOLD**；无新增正确
也未破坏原61个正确。此干预只消除了全局标量M差异，L内部仍保留原
visibility加权内容，不能冒称删除了全部RoMa或全部几何。

## P接口究竟丢了什么

旧scorer先对dense overlap做完整query/reference cell区域平均，再计算

    M = sqrt(mean(full query visibility) * mean(full reference visibility))

见`programs/run_romav2_colnomic_visibility_xf_six_case_v1.py:28–35`。

P atom builder改成query中心采样oq，以及在warp(query)处采样orev；逐atom
reciprocal为sqrt(oq*orev)。完整reference均值并未进入九维特征、atom
字段或额外metadata。V3/GISC/V6使用这些特征，V8 REAL评分进一步只使用
选中对应的残差、reference分组和合法component。

证据位置：

- `src/rc_aslo_xf/reference_conditioned_latent_proposal_p_only_v1.py:477–513`
- `src/rc_aslo_xf/reference_conditioned_latent_proposal_gisc_v4.py:76–93`
- `src/rc_aslo_xf/reference_conditioned_coherent_residual_p_only_v8.py:56–91`
- `src/rc_aslo_xf/reference_conditioned_coherent_residual_p_only_v8.py:303–307`

这不仅是两个均值公式不同。在存在未被这些插值采样覆盖的reference区域
时，改变其overlap可以保持所有保留atom不变，却改变旧M。因此atom压缩
一般不是旧统计量的充分表示；单纯改这些head的权重不能普遍恢复它。

已证明的是“丢失统计量＋原成功依赖该统计量”的接口问题。尚未证明的是：
把M补回某个已失败P head后就一定通过，或其他改动如硬对应、连通约束、
优化目标变化没有额外影响。这些结论不能越界。

## 据此实施的修复

新`reference_visibility_pv_lossless_v1.py`把原融合scorer无损拆成P/V：
P封存完整双侧visibility、完整均值/M与canonical轴，V执行原full-reference
weighted MaxSim，原七参数action消费相同字段。原RAW score和候选轴保留。
P不被要求自行完成身份分类，也不再把局部采样当作全局质量。

8项独立原score-oracle测试通过；RAW-only导出完整64×128，两侧map SHA与
全部8192个M已逐bit对齐旧缓存。第一个真实候选的REAL/M/Q/R均逐bit相同。
完整自然P/V重算job5138124已完成：8192对原RAW tokens重新经过新P/V，
32768个REAL/M/Q/R标量与8128个action logits全部逐bit匹配。独立验证进程
再对照原缓存标量、重算全部action；EVAL25→27、2救0损，TRAIN27→28、
1救0损，所有原预测保持。见`results/rc_original_raw_visibility_pv_replay_v1/result.json`。

这是必要的接口修复，不是空间HYP证明。旧机制依赖质量与内容的合作，
以及RAW先验提供的保守决策；区域独立贡献仍须依据明确命题另行验证，
不能把恢复27/32命名为新检索增益。

## 旧最好系统尚未救回的具体位置

按同一冻结difficult90结果，剩余21条错误中，6条target不在原C128；其余
15条target都在候选内，但target对原RAW winner的原action logit全部不大于
零。15条中有11条的mass差和normalized-content差同时为正，仍不足以越过
原完整head的保守决策。这不构成调阈值的理由。

另外，原3次wrong-to-wrong SWITCH的错误challenger，normalized-content差
也都为正。因此“仅要求完整MaxSim内容差为正”不会排除它们。ordinal25与64
的实际错误challenger，在RAW分数、mass和normalized-content三项上都优于
target；这只是三个既有统计量上的观察，不能推成任意模型不可能区分。
主agent与独立agent随后都实际查看了以下两例的query、target和原selected
wrong三张图。未重新评分、未改标签，也未把个例外推为全部失败的原因。

**ordinal25：匹配到了另一个真实物体。** Query为
`1/difficult/cet03-0003-14_0000002.JPG`。原target为gallery row5049，
`prescription/cet03-0003-14.jpg`；原selected wrong为row395，
`animal/int08-0008-17.jpg`。前景手持细长白盒的可见说明侧面与target的
单次使用、重配说明、红色冷藏句和FDA电话排版对应。右下背景同时真实
出现了wrong reference的Interceptor PLUS犬图、50.1–100lbs、Elanco及
BROAD-SPECTRUM PARASITICIDE面板。因而该wrong reference的正身份匹配
不能简单解释成虚假纹理支持；原单目标基准仍把它计错，因为它不是本次
标注的目标物体。这里必须区分“图中存在哪个reference”和“本次要识别
哪个物体”。单看此query侧面不能独立确认前景药盒的药名和剂量。

**ordinal64：可见的细粒度差别未被原总体评分正确利用。** Query为
`1/difficult/692R-Albertsons-Cetrizine HCl-45s-Carton_00000003.jpg`。
Target row1113为Signature SELECT的45片包装，reference文件
`otc/692R-Albertsons-Cetrizine HCl-45s-Carton.jpg`；原wrong row3825为
CAMBER的100片包装，`otc/cetirizine-hci-10-mg-tablets-grn-1.jpg`。照片只
露出Drug Facts侧面，品牌与45/100片没有露出，不能拿不可见数量解释
模型应该如何判断。实际可见的是target相符的红色READ AND KEEP CARTON
开头及印字瓶口铝箔破损警告；wrong reference的对应红句写的是瓶盖下
安全封条。两者共享大量Cetirizine 10mg说明，但并非没有可见区别。

这两个已核实个例支持不同的故障类型：目标物体选择，以及相似包装间
细节验证。它们不支持把所有wrong-positive统称为几何错误加分。尤其当
背景reference确实在图中时，身份匹配本身可以正确，而单目标选择仍错；
不能通过改名或改标签将原错误计为成功。

进一步只读两例的原query/reference tokens，计算无权重FP64 full-reference
MaxSim的逐query-cell目标减wrong差：ordinal25有362个目标优势格与406个
wrong优势格，ordinal64为370与398；两例均非wrong逐格全支配。这说明现有
表示下仍存在局部正差，不能直接断言必须更换编码器。它也不证明这些格
在旧wr加权后的L里仍保持优势，或已构成可自动识别的ROI。
见`results/rc_old_roma_two_failure_maxsim_difference_v1/result.json`及同目录
`unweighted_difference_overlay.png`。数组保留DECODED_RAW_BEFORE_EXIF的
24×32原网格；仅绘图时按EXIF6顺时针转90度与显示原图对齐。

## 文献给出的边界

RoMa v2将overlap作为共可见性预测，训练监督来自深度一致性或warp cycle
consistency；它不是药品规格身份标签。因此将其与内容验证区分有原模型
依据，但不能把overlap直接当作exact-instance身份概率。
[RoMa v2原文](https://arxiv.org/html/2511.15706v3)

ColBERT的MaxSim保留每个query token对完整候选token集合的最佳匹配。
这支持保留原自由内容搜索；论文不替本项目证明硬对应损失了多少准确率。
[ColBERT原文](https://arxiv.org/abs/2004.12832)

SIS与视觉反事实解释针对固定模型的决定，错误决定也可能获得充分或必要
证据；这不能消除本工程TK1V3的错误candidate也有functional region反例。
[SIS](https://proceedings.mlr.press/v89/carter19a.html)、
[Counterfactual Visual Explanations](https://proceedings.mlr.press/v97/goyal19a.html)

## 结果绑定

- EVAL/TRAIN项级重放：`results/rc_frozen_roma_action_terms_v1/result.json`，
  SHA `9af91982d06630da09e84120dce9cdd673d06f0c11cb019750e818dfa59300a6`
- difficult90项级重放：`results/rc_frozen_roma_difficult90_action_terms_v1/result.json`，
  SHA `53bd8b545d2b51e97b9917c02e779b5c346aa613d4655413fe24fca17dcddd89`
- 因子一致干预：`results/rc_frozen_roma_visibility_mass_factor_v1/result.json`，
  SHA `79c95466ed0edfcd0fa34024baae71e16362e5595570456a7c5c497745300fb1`
- RAW-only完整输入：`results/rc_original_raw_visibility_pv_inputs_v1/manifest.json`，
  SHA `2777f72c1eacc3dd46f1ab4ba7f18c0dda23ddcb95a51f4b87c9fecd67dec2d4`
- 完整回放authority：`registry/rc_original_raw_visibility_pv_replay_authority_v1_20260909.json`，
  SHA `fa2a140b22ad8b5165ff4cd3fd3f4f281f8949758b852b83e05e5ecf80b5bcf4`
