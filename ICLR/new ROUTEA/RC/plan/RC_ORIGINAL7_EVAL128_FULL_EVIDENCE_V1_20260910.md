# new HYP：冻结原七参数头的128图完整证据验证

用户要求将32张评估图扩到128张或更多，并指出原模型是retrieval训练
的小头，J/T二维解释可能漏掉其完整机制。本计划以原模型的六维状态、
正确challenger竞争和HOLD/SWITCH为主线；不训练新头，不要求J/T或LP。

## 当前依据与模型身份

当前ORIGINAL7/NATIVE7在已打开matched EVAL32上RAW25→28、3救0损。
它有6个输入系数加bias：RAW gap、S对比、M对比、ell=S/max(M,eps)对比、
query扰动响应对比、reference扰动响应对比。候选内容评分仍为RoMa
soft visibility乘full-reference weighted MaxSim。

模型参数来自registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json，
参数SHA ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263。
沿用candidate_feature函数及全部127行X@w+b、最大logit>0才SWITCH、
physical-row平局规则。romav2_colnomic_frozen_gate_v1.py里的默认WEIGHT、
BIAS、logit属于另一旧FROZEN_C，不得误当当前ORIGINAL7使用。

## 选样仅用身份分组元数据

用户扩展授权：rc_retrieval_only_eval128_expansion_authority_addendum_v1_20260910.json。
资格池按source inventory固定：从RAW987目录排除当前PAIR64的20组、
FULL TRAIN32的12组及旧EVAL32的11组，合计43个已用identity/group。
另仅以公共formal392的query ID和image SHA作为负名单，不读取或消费
其P/RoMa结果。剩余267记录、266不同图片、24身份、21来源组。

完全不使用candidate存在性、排名、准确率或难易度作筛选。exact image
SHA重复先核对身份/组一致，再只保留一个query；若冲突，保留问题并
单独解决，不随意挑有利标签。选择128张，字面seed固定为
RC_ORIGINAL7_EVAL128_20260910：按group、组内identity、identity内image
的固定hash顺序作嵌套round-robin，覆盖全部可用组/身份并限制大组主导。
算法及源绑定由curator独立回放；冻结worker清单与单独role清单。

worker只获得匿名query索引、图像/特征来源和必要预处理字段，不获得
target、identity、group。旧EVAL32单独保留作回归，不混为新128。
本集合与当前训练和旧EVAL身份/组独立，但来源目录历史上已打开；没有
依据称它未触碰外部确认。新增128图仍须报告21组这一有限独立单位数。

## 当前运行接口的资格与计算

沿用当前runtime的冻结ColNomic-7B、processor、gallery与RoMaV2权重。
存在旧redacted tokens不等于它们已与当前runtime逐bit相同。先在已用
TRAIN源上检验token/RAW接口；无法证明旧token同源时按当前runtime
同一encoder重算，不把历史输入差异带进原头解释。

RAW排序使用既有full_gallery_scores：image+template、FP32 sum-MaxSim、
batch16、最高matmul precision、全5413物理reference。沿用既有corrected
identity去重及stable top128规则，保存完整RAW排名/分数和物理排序轴。
不插入target，也不删除后来发现target未进C128的query。

RoMa只在冻结RAW C128的全部reference上执行。沿用原EXIF/frame映射、
canonical geometry、FP64 cell means及score函数；query/reference控制
仍为各自cell轴一半的roll。保存128个candidate的S/S_Q/S_R/M、源hash
及wq/wr，便于独立重算。保持旧full-reference MaxSim，不硬绑定单token。

先用已用TRAIN数据完成当前runtime兼容试运行，再冻结扩展执行源码、
依赖、模型参数和launcher。运行采用独立新文件和append-only分片，
token/RAW、RoMa/C4及CPU头阶段分开资格；完好的分片不因汇总失败重算。
GPU阶段沿用已验证accelerated分区的1GPU/8CPU/96G/30分钟作业模式，
8条query/分片，16片，最多4片并发；所有阶段受研究截止时间限制。
CPU RAW汇总为8CPU/16G/15分钟；完整C4独立重算与预测封存为
8CPU/16G/30分钟；封存后的role join和统计为8CPU/16G/15分钟。

## 预定输出及完整六维解释

所有128图的RAW和ORIGINAL7 REAL/C_BIND预测先封存，才开放role join。
C_BIND保持RAW，按原half-roll转移整份candidate evidence。单独保存
冻结原参数下DROP_RAW/S/M/L/Q/R、DROP_QR、DROP_S_QR的全部127个logits
及动作；这些是计算依赖诊断，不是重训模型、物理删除或像素因果证明。

对每图记录RAW是否已正确、是否SWITCH、选中哪个challenger、六项贡献
和bias。若target在C128，分开报告target越过0阈值的余量，以及它与
最强wrong challenger的竞争余量；RAW winner的策略分数为0。若target
缺席，记录召回失败，不伪造target logit或跳过该图。

报告全部128图的candidate recall@128、RAW/ORIGINAL top1、rescue/break、
完整gallery MRR、各来源/身份组数量和组等权差异。完整gallery MRR遵循
一次SWITCH把所选candidate移到RAW排名首位，其他相对顺序不变。
同时单列target进入C128后的条件指标，不能以该条件子集替代全128。

对主要ORIGINAL7 vs RAW报告来源组级paired差异、固定seed20260910的
10,000次组bootstrap区间及两侧组sign-flip比较（至多21个非零组，
可完整枚举）；描述组相关及已打开开发背景，不把128张视为128独立身份。
置零和C_BIND对照完整报告，不根据本轮结果选择新的阈值、方向或参数。
若有净增但损失原正确，分别报出，不称满足“新增且保持”。

不访问D1-MI/GroZi/预留formal392推理结果，不引入SAM或本任务空间标注。
任务只有原检索身份监督，encoder预训练来源照实披露。原32、旧90及
新128的模型/候选/数据谱系分别保留；不混拼准确率，不自动部署或宣布
普遍HYP成立。执行截止北京时间2026-09-11 24:00，提交后继续推进。
