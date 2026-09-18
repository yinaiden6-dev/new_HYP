# 固定七参数配方：完整TRAIN96与TRAIN128

本计划承接retrieval-only持续研究授权及已冻结TRAIN128输入计划。只检验训练覆盖；
不新增特征、网络、归一化、阈值、种子、训练步数搜索。当前ORIGINAL7是baseline，
旧EVAL32为25→28、新EVAL128为88→99（12救/1损）。两套EVAL都已打开用于开发，
本轮不称新的未触碰外部确认。研究截止北京时间2026-09-11 24:00保持不变。

## 三个预定条件

| 名称 | FULL训练上下文 | 辅助训练 | 参数 |
| --- | --- | --- | --- |
| ORIGINAL7 | 原FULL32 | 原PAIR64 | 原六列七参数 |
| FULL96 | TRAIN128清单前96张：原FULL32及原PAIR64图片的完整自然C128 | 原PAIR64原特征/标签不变 | 同六列七参数 |
| FULL128（主条件） | 以上96张再加预先固定的32张新视图 | 原PAIR64原特征/标签不变 | 同六列七参数 |

主比较固定为FULL128对ORIGINAL7；FULL96用于区分补足完整负候选上下文与新增视图，
不能看EVAL后在两者中挑主条件。原PAIR64含旧V1的roll-one控制及V2的half-roll，
其辅助列保持原样；新FULL输入统一当前half-roll。这是完整候选及其控制上下文的
扩充，不能把FULL96的任何变化只归因于图片数，亦不声称原PAIR64每列新FULL值不变。

TRAIN128清单维持原32训练身份/32组；旧EVAL32和新EVAL128身份、组、图片均排除。
选择规则、源SHA、匿名worker及TRAIN curator由独立metadata验证冻结。训练阶段不读
任一EVAL feature/target/result内容。可仅哈希预定来源以验证绑定。

## 原输入回归与训练定义域

原FULL32新tokens、RAW C128、RoMa maps/C4与原六列必须逐bit复现原输入。回归
original32+原PAIR64须恢复parameter SHA ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263。
不一致是工程阻断，不能当作扩充的科学结果；不能在出现差异后重调基线。

全部128张输入保留自然C128，不插入target、不按召回替换图片。原FULL loss只在
target属于自然C128时有定义：target缺席者保留在清单/召回台账，但不进入这个未定义的
FULL正例loss，不伪装为RAW正确，不给winner错误身份标签。分别报告96/128预选图片数、
有效FULL loss数、缺席数及原PAIR64覆盖；原PAIR64辅助loss继续原样。该条件仅用于
定义训练损失域，EVAL一律保留完整32/128分母。

## 冻结学习规则

复用原train_head函数及其依赖的原AST。六列依次为标准化RAW gap、S/M/L对比、Q/R
响应对比；七参数包括bias。每个条件seed17、全部权重/bias置零，AdamW(lr=0.03,
weight_decay=0.001)，2000步，只保留末步，不搜索checkpoint。loss为原PAIR64带4倍
保护权重的BCE均值，加有效FULL query loss均值：RAW正确用4*softplus(maxwrong)；
RAW错误且target在C128用softplus(-target)+4*softplus(maxotherwrong)。训练顺序沿
冻结worker顺序，PAIR64原序；FP64不变。无EVAL调参、阈值变化、身份专属参数或空间监督。

先完成三条件参数封存，再由独立新进程从零重复拟合并逐bit确认参数。独立过程不能
加载结果权重当作初值。head训练只使用TRAIN/PAIR检索身份标签；encoder/RoMa不训练。

## 评价封存与解释

独立拟合验证完成后，才读取旧EVAL32及新EVAL128的已资格化、无target映射的特征。
对每个条件和两套EVAL，计算完整127个challenger的REAL和整份candidate-evidence
half-roll C_BIND；固定最大logit>0才SWITCH，否则HOLD，保留physical-row平局规则。
所有预测封存并独立重放后才进行target join。不能在join时改图、重排或丢弃召回失败。

各模型/数据集分开报告：RAW与最终正确数、rescue/break、原模型正确集合保持、RAW
正确集合保持、MRR、候选召回及group统计。新EVAL128沿完整5412排名move-to-front
计算MRR，保留7个及任何已有C128缺席样本。旧EVAL32按既有同源排序定义单独报告。

必须区分：相对RAW纠错且零新增损失；相对原最佳模型新增且保留原正确；仅正净增但
发生取舍。旧28与新99正确集合分别检查；目标是提高或更稳健保持，不将任一单独PASS
自动改名为new HYP全部成立。组级效应及不确定性照实报告，不按p值停止或改方法。
本轮只是开发验证，不自动部署，不进入ownership，也不修改原已合格结果。

本计划拟合/评价需单独readout authority绑定完成后的输入、程序、preflight和上述合同。
输入物化authority不授权直接越过此封存过程。不得使用D1-MI、GroZi、SAM/空间teacher、
受保护formal392推理结果；基础模型预训练来源照实披露。
