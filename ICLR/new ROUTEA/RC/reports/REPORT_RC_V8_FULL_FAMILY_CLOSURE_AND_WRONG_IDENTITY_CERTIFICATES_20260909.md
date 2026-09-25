# 完整区域诊断收口：局部相容性仍不足以证明完整身份

2026-09-09。Job5137873完成，用时11分18秒，原artifact validator通过；本轮
独立实现重算全部344448个有理数单元、2691个区域状态及完整32条汇总，
0处差异。该作业是诊断，不是另一次模型拟合或新的自然GO/NO-GO。
原V8识别9/32与NO-GO保留。

## 本次真正确定的结果

| 当前冻结参数和完整合法family下的性质 | Query数 |
|---|---:|
| target至少有一个CERTIFIED区域 | 20/32 |
| target是严格唯一、其他candidate所有区域均被否证或H0 | 11/32 |
| 至少一个wrong candidate也有CERTIFIED区域 | 14/32 |
| target所有合法H1均被已观测竞争者否证 | 7/32 |
| target仅有UNRESOLVED可能性 | 2/32 |
| target没有合法H1 | 3/32 |

关键收口：20个有target证书的query，**原选中区域已全部有证书**。
从1672个已选H扩展到2691个完整合法H，没有新增任何target-certificate query。
同时所有2691个区域中只有2个仍UNRESOLVED；缺失已不是这条证书机制的主要
未决因素。不能继续仅靠扩大搜索或解决最后两个UNKNOWN预计突破。

即使两个UNRESOLVED都变成有利证书，当前冻结head/family的目标证书最多
22/32。这约束该特定正向证书机制，**不是新排序器或全部身份方法的准确率
上限**。不允许据此断言所有HYP不可能，也不修改原自然门。

原同域诊断确认的7个固定配对反转仍然成立。这些正向信息没有被否定，
但完整family告诉我们：多个reference可以在不同局部区域各自严格胜出。
“存在一块最支持自己的区域”与“正确识别整件相似物体”仍是不同命题。

## 错误证书不是由缺失目标对应制造

43个CERTIFIED区域中，22个属于target、21个属于wrong；分别涉及20和14个
query。wrong区域大小中位数20格，范围5—162；target为75.5格，范围4—420。

21个wrong证书中，16个对全部C128的对应完整，**21个对target的对应全部
完整**。6个是某目标合法区域的严格子集，8个与所有目标合法区域完全不交，
7个部分重叠。不能再把它们全称为嵌套的小碎片或UNKNOWN导致的假阳性。
9个query同时存在target与wrong证书，涉及13个wrong区域。

21个wrong区域的grouped-mean成本全部低于同域target；其中11个global-max
成本也更低。固定这两个既有标量，仅改其非负权重不能逆转那11个比较。
该结论不排除改变逐atom度量或平均测度的影响。

## 原图实例：可见72片，却能认证12片包装的共享品牌区域

query `q_635c6741b64b…` 的原图清楚写着72片；错误reference
`c_1ddcf3723ec2…` 的纸盒展开图写着12片。本轮两个审阅者分别看过原图，
并与已保存V6图像来源和当前V8区域对应核对。以下是图像诊断，未将这些
字段或坐标交给模型。

当前wrong的36格CERTIFIED区域覆盖query的品牌/ORIGINAL及上方附近，
source query行13—19（36×20网格），不含约行22的72字样。复核纠正：wrong
reference的上折页另有竖排12，约原图像素(775,120)，对应19×39网格的
(row2,col13)，确实包含在该wrong支持中；不能说reference支持排除了12。
这个reference格在保存的对应中来自query(row18,col11)。该区域的target对应完整，但wrong
对target的能量优势约0.261612，对全C128的最小优势约0.121864。

target另有56格证书区域，source query行14—24，延伸至下部规格文字。
因此这里的错误证书证明了该局部在当前读出下的相容性；它遗漏了query端
可见的数量区别，未证明解释了完整身份。不能把一个reference token覆盖12
直接等同于该token只编码12。我们尚未证明ColNomic不能读取72，也未证明该数字单独足够
区分全C128。空间连通和往返闭合都不能自动补上这个必要观测条件。

原图及哈希（文件名中的36不是本次人工读出的数量）：

- query：`1/outcome/AlkaSeltzer_36_00000000000000000001.png`，相对workspace，
  SHA `60a129173d7c77942021ed9f505798fe31cf5e2f39653f189a6dadf231955321`。
- wrong：`dailymed/data/box_flat_20000_images/data/raw_images/otc/AS Original 12 count carton.jpg`，
  相对workspace，SHA `12bae703d734209a4838d72e2891ed0793191dc1970a67354ada335169b0d8a5`。
- 已有来源记录：`results/rc_competitive_witness_v6_visual_forensics_v1/evidence.json`，case03。

## 仍须排除的聚合混杂

21个wrong证书中14个拥有比同域target更多的reference groups。当前做法虽
使用相同query cells，却依然采用各candidate自己的group分母及group内max。
因此“同一区域”还没有做到“同一批观测按同样权重比较”。

一个先验反例：四格query上target成本a=b为(.1,.1,.1,.9)，其ref分组
(0,0,0,1)；wrong成本为(.12,.12,.12,.95)，其ref分组(0,1,2,3)。
target每个点都更好，但原分组能量1.4高于wrong的1.2775；若对query点统一
平均，则target1.2低于wrong1.2775。该例可用uniform权重和单位token实现。
这是数学反例；自然21个wrong证书中发生多少次仍需计算。

下一次唯一检查见
[共同query平均诊断计划](../plan/RC_V8_COMMON_QUERY_MEASURE_DIAGNOSTIC_V1_20260909.md)：
保留全部2691个区域、C128、参数及RoMa对应，只将第一项改为query-cell均值，
保留global-max。逐项重放旧聚合，保存每个atom成本和完整缺失轴，以精确
有理数比较；缺失只给非负补全下界，不改成实际观测。此时逐atom支配必须
得到保留，可以查明是否存在由分组制造的实际逆转。

这只是剩余原因隔离，不是预告共同均值会成功。如果wrong在共同权重下仍
占优，就必须保留wrong-positive事实，后继需要处理身份区分内容的观测与
证据读出；不能再把局部存在性换一个名字继续拟合。任何后继仍须独立冻结
全部对照，并在未打开面板上确认，才能支持更强论文结论。

## 文献与论文表述边界

[Doppelgangers（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Cai_Doppelgangers_Learning_to_Disambiguate_Images_of_Similar_Structures_ICCV_2023_paper.html)
研究不同但相似表面之间的错误匹配；匹配与身份区分的差别已有明确先例。
[Counterfactual Visual Explanations（ICML 2019）](https://proceedings.mlr.press/v97/goyal19a.html)
通过对比区域替换解释类别改变，强调哪些差异影响分类；它不是本项目的
现成识别方法，也不能代替当前必要观测条件的定义。
[Triangulation embedding and democratic aggregation（CVPR 2014）](https://www.robots.ox.ac.uk/~vgg/publications/2014/Jegou14/jegou14.pdf)
研究局部描述子在图像聚合中的贡献分配，表明聚合测度不是无关细节。
这里不能把“共同权重”或“比较差异”本身包装成新颖贡献。

当前最稳妥的新发现是这个冻结系统中已经出现的反例链：不同支持的直接
比较会掩盖局部身份优势；改为同支持也可能让不同错误身份各有局部证书；
完整身份需要解释区分所需的观测，且仍要排除候选相关聚合的混杂。
这不等于HYP通过或RAW+RoMa必然进一步提升。

## 验证与来源

- 主结果：`results/rc_v8_full_family_completion_certificate_v1/result.json`，
  SHA `47668e24854a608974bc7c585b979ae980e361a8d70338b9d2efadd40efc35d6`。
- artifact验证：同根`artifact_validation.json`，
  SHA `a9895dea6a645ddbfbf84bbb8d210e06cd723a5714870183d38eda996b78daea`。
- 本轮独立复核：同根`independent_result_audit_20260909.json`，
  SHA `1348937844fa2a64962e96798d7aa26120465f33df04b5a676e17d841f1ab27a`。
  独立重算未导入生产或验证程序，不包含raw模型重跑。
- 本轮无新训练，正式392、V、action、HOLD/SWITCH、ownership保持封闭。
- 全部21个wrong与22个target区域的机制审计：
  `results/rc_v8_wrong_certificate_mechanism_audit_v1/result.json`，SHA
  `1aa8b871e64a0e60528a1e18b52b356fa063b8510ad2af479571ad660588395e`。
  该不可变原件中的“wrong reference支持不含12”视觉描述已撤回，须连同
  同根`visual_amendment_q635_20260909.json`读取。数字统计及原scalar证书不变。
  更正文件SHA：`70c5f26c4fa126c0c96af272f876742663e0cd2d8d974bf55b593985634f96b6`。

## 本轮提交与退出

共同query测度诊断 **Job5137920** 已提交。2026-09-09 02:06:23 Europe/Berlin
（00:06:23 UTC）的唯一激活核查为PENDING，无依赖，dev_cpuonly、4CPU、
64GiB、59分钟；脚本与日志路径核对通过。本轮不再查询或等待结果。

- authority：`registry/rc_v8_common_query_measure_diagnostic_v1_authority_20260909.json`，
  SHA `ad3eef6030b3dd83145384fc44766b07ac0860dcc2a630c70618cf651552d81a`。
- worker：`programs/run_rc_v8_common_query_measure_diagnostic_v1.py`，
  SHA `b913744823fedb65ddbee471e065244bbd219eb4b23358af55cb8c530a8b11a9`。
- 提交前3177种固定query分母补全、4368种逐atom支配情形、继承5328种
  分组补全，以及NaN占位、平局/非max严格改善、源轴、UNKNOWN、篡改检查均
  通过；实际空环境脚本文件启动验证通过，未读取自然数据。
- 独立源码审查确认完整旧聚合重放、新逐atom保存、统一query分母、严格
  唯一性和全量证书迁移规则；未声称当前已有新自然结果或新识别模型。
- 接手读取`results/rc_v8_common_query_measure_diagnostic_v1/result.json`
  与`artifact_validation.json`；后者由同一job在完整run之后自动另起进程生成。
  若不完整，只记工程状态，不据部分结果换公式；不要重复提交。
