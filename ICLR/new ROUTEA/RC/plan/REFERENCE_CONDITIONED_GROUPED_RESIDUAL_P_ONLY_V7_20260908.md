# V7：固定连通区域内的分组残差度量

2026-09-08。唯一一次post-hoc TRAIN32 development successor。
用户要求今晚推进HYP；这个合同冻结一个可失败的机制，不保证正向结果。
本合同不改V3/GISC/V6失败、既有generation定义或正式392的预冻结顺序。

## 问题与唯一改动

已封存generation coverage29/32，但整region mean-c仅19query有正向竞争证据。
三个旧正向四格仍在新region内，整体均值却转负；19个wrong-positive regions
中16个对固定competitor的raw cosine差也为正，不能只靠删除baseline解释。
现九标量丢失了128D配对向量的方向信息。

本次保留现有cycle-closed maximal connected family不变，改用原封存
ColNomic 128D tokens形成多通道残差摘要；没有新backbone或identity参数。
它检验不同通道、同reference cell的重复对应、区域内最大残差是否能提供
匹配对照之外的身份判别。它不是已验证的语义冲突检测器。

## 输入与来源

完整原TRAIN32、自然ColNomic C128、8个原atom shards及原sanitized输入。
输入adapter恢复生产时EXIF方向，必须与原q/r tensor SHA和source binding
精确一致。q/r在原source_query_indices/source_reference_indices取内容，
不能用P_COORD置换后的current索引重新取token。

所有有效原token重新以FP64单位归一化，后续度量、loss、optimizer均FP64。
原stored cosine由GPU FP32生成；CPU FP32/FP64复核使用预声明解析前向误差界
并报告非bit-exact，而不是用自然误差临时调容差。输入字节与绑定仍严格SHA。
无效atom的原feature0必须为0，复核时保留原valid mask。

P_COORD保留source内容，重算current坐标图；C_BIND使用before-RoMa/assignment
的真实donor bank。完整32个target-free contexts封闭后才开opaque postjoin。
生成器来源：rc_cycle_closed_region_generation_authority_v1_20260908.json。
真实reference资产为空、轴错、源漂移、q网格不完整均ABORT，不替换query。

## REAL：一个256参数共享度量

D=128。对H内每个原配对atom i，令单位向量x_i、y_j(i)，
`e_i,d = (x_i,d - y_j(i),d)^2`。

同一source reference cell j内，形成
`z_j,d = max_{i in H: j(i)=j} e_i,d`。
按source reference index canonical排序，对全部非空cell组成
`v_H = concat(mean_j z_j, max_j z_j)`，共256维。

仅一个共享theta∈R256，精确零初始化。
`w = 128 * softmax(theta)`；`S(H) = 1 - 0.5 * dot(w,v_H)`。
没有bias、纯geometry加分、reference-specific baseline或identity embedding。

遍历该candidate的全部固定maximal components，S(k)=max_H S(H)，
同分按生成器canonical component ordinal，选择与分数正负无关。
保存完整每个candidate的region utility vector及selectedH。

单位向量给出每维e∈[0,4]，因此非空S∈[-255,1]（实现报告FP64误差边界）。
空family用固定-256作为严格低于非空范围的结构性排序sentinel，状态H0。
这不是semantic no-match概率或由数据调出的阈值；不能用H0证明身份缺席。

## 可证性质与明确不能声称的性质

- 同一ref cell内增加一个完全相同的残差观测，描述符不变。
- 固定cell集合、固定w，逐元素增大已有最大残差不会提高score。
- 单atom、零theta在精确算术下恢复原cosine。
- max摘要丢弃次大残差；softmax权重也能削弱部分通道。因此不能声称完整
  信息保留、所有负identity证据不衰减或可见语义冲突已经被识别。
- 加入新的低残差ref cell可能降低mean项、改善总分，尽管max项不变。
  不声明全局单调反稀释；E0必须保留这一反例。
- contextual ColNomic tokens可携带region外信息。region-only token读取
  不等于pixel-local causal evidence，不能声称ownership或物体mask。

## 两个同容量对照与一个额外冻结强对照

三个新arm各自256参数、相同初始化/optimizer/训练样本mask，没有共享训练参数。

ALL_PATCH_NO_HYP：在该candidate全部原valid atoms上用同样的source-ref
分组max/mean描述符和同样的metric，无空间选区。P_COORD必须逐位不变。

TRUE QUERY_ONLY_REGION：先仅从query产生一个区域，所有candidate共用。
原完整native query网格上枚举全部四格连通sets，canonical current-index排序。
令g_i=(x_i-mean_source_query x)^2，query-only seed描述符为concat(mean_i g_i,max_i g_i)。
用该arm自己的w最大化dot(w,query_descriptor)，固定canonical tie。
这是query-only variation selector，不声称已有产品分割/语义saliency。

选区后，对每个candidate、每个固定q atom，遍历该reference完整有效token轴，
以a_d=w_d+w_(D+d)最小化sum_d a_d(x_i,d-y_j,d)^2；平局按source reference index。
允许多个q atoms指向同一个reference cell，不施加REAL的ref几何资格。
在这四个已选配对上使用同一个分组残差描述符及score。不得按candidate重选q区。
这是逐atom的固定hard assignment，不是对最终region score的联合最优匹配。
整个过程不依赖RoMa在该固定query区是否存在有效对应，避免missing削弱对照。

C_BIND换donor内容后重新算外观匹配。P_COORD在置换后的CURRENT query网格
重新选q区，通过inverse permutation取source tokens再匹配；C128必须共享
同一query permutation，否则ABORT。QUERY_ONLY不强制P_COORD不变。
保存其完整query selector vector（REAL/C_BIND/P_COORD各一份）及current/source H。
硬argmax/argmin只通过最终选中摘要和metric传梯度；不声称可微匹配或可微分割。

额外保留已独立验证的旧V6 QUERY_ONLY common-axis comparator，在TRAIN32上
22/32成功。它维持原checkpoint/输入/原512训练结果，禁止重训或改名后丢弃。
V7 REAL相对它也必须paired net>=4且rescue>break。这是额外保守门；新的
AP/query-only仍是主要同容量对照，不能用旧10参数对照替代同容量评估。

## 训练与checkpoint

用原V6实际512位置query顺序，按原contract SHA重建并匹配原query_order SHA；
seed17，AdamW lr0.03、betas(0.9,0.999)、eps1e-8、weight_decay0。

根据已封存REAL target-H1 coverage，预冻结共同结构可训练mask：29query可训练，
3query无target component；所有新arm使用同一mask。全32继续评估，不插target，
不把3个结构失败从coverage/paired分母删掉。这是显式训练协议后继，不宣称
与旧V6使用完全相同的学习目标或训练样本。旧强对照保持其原32条训练。

所有512位置均执行AdamW。mask为false时用连接该head计算图的严格零loss，
产生零当前梯度tensor（不是grad=None）；Adam动量仍可能移动参数，必须记录，
不能称为无更新。记录完整mask/queryorder/每armstep与梯度计数。

可训练位置继承relative P objective：full-C128 target CE、softplus(-margin)、
softplus(-(margin-C_BIND_margin))、softplus(-(margin-P_COORD_margin))。
ALLPATCH coordinate term固定softplus(0)，QUERY_ONLY坐标项按真实重选区计算。
无绝对score0的match/no-match监督，不新增温度/阈值/K/尺度搜索。

fresh完整512；resume在256强制保存/重载后到512。模型和全部AdamW状态在256、
512逐字节相同才允许reduce。所有train/checkpoint/reload/deploy用唯一同一函数。

## Gate和独立验证

完整32 query、各C128不变：target合法region>=26/32；REAL target胜strongest
wrong>=21/32；group-balanced REAL margin>0；相对两个新同容量对照分别net>=4、
rescue>break；额外相对旧强对照net>=4、rescue>break；C_BIND/P_COORD各>=21/32
降低margin且group-balanced drop>0。32分母不变，报告每query与所有同时失败门。

本generator的P_COORD全部family为空，所以该margin恒为0，坐标下降计数
退化为REAL正margin计数。保留原门并显式标注退化；它说明拓扑依赖坐标，
不能单独证明identity-specific spatial relationships。

独立validator重新加载输入/最终checkpoint，重建全部context，重算全部输出、
完整region vectors与query selector vectors逐位比较。只有全部独立工程检查
通过且所有开发门通过，才能称development GO；仍不是正式HYP证明。

## 正式阶段与退出

正式392 prerecall universe的P/RoMa结果仍封闭。仅在完整development GO且
独立验证后，按已有预冻结原ColNomic C128及target-presence选样规则取
fold2/3/4的11/11/10，identity/supergroup-disjoint。新方法、两个同容量对照、
旧强对照checkpoint和全部自然门在开正式结果前冻结；不能用新面板选模型。
正式adapter仍需独立E0/来源验收，不能因本合同写明就当作已可执行。

保持P-only；不进入V、HOLD/SWITCH、ownership，不读/改/依赖D1-MI或GroZi。
使用隔离V7 roots；现有src、authority、结果与其它任务不改。提交前核对source
及单writer，重复任务不得继续。用户此前要求不监控，提交时一次核对activation
后退出，不建立循环poll或自动科学阶段推进。
