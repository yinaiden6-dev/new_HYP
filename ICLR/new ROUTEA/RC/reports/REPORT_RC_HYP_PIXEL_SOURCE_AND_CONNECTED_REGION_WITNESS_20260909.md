# 像素干预后的连通区域正例与旧action解释边界

2026-09-09。Job5138459完成并通过产物复算；同一原图、冻结ColNomic
权重与processor、原RAW C128 reference tokens，ORIGINAL token与完整
MaxSim矩阵均逐bit复现。总计4次单图encoder forward，零训练、零RoMa
新forward。对象是已打开EVAL32中的OUTCOME-0212，而不是新评测集。

## 一个可复核的连通区域正例

原人工target polygon对应的47个native token中心构成单一四邻接连通
component。固定该H，不按新得分挑位置；V仅对H内原full-reference
unweighted MaxSim作FP64均值，比较完整128个reference：

| query像素条件 | 同一GT47上的target排名 | 对全127的target margin | 全720均值排名 |
|---|---:|---:|---:|
| ORIGINAL | 1 | +0.0255868682 | 25 |
| KEEP_TARGET | 1 | +0.0241995908 | 2 |
| ERASE_TARGET | 95 | −0.0978252299 | 95 |
| ALL_GRAY | 80 | −0.1100633773 | 96 |

这给出一个明确的、由现成人工区域指定的见证：连通目标区域的内容足以
让固定MaxSim读出在该C128中唯一选出target，保留该区域像素后仍成立，
擦除后失效。无需硬reference坐标指派或新身份网络。
尚未证明自动P能选出H；这不是新32-query准确率或原P-only GO。此query
在原NATIVE7/C中已经正确，不能把25→1称为对原28/32的新增救回。

固定读出结果：`results/rc_outcome0212_fixed_region_readout_v1/result.json`，
SHA f3ab4e99b65d40fedf8f848130e414e8c05a9a66b3945d6f6d80739d02345f38。
四种像素结果：`results/rc_outcome0212_pixel_source_diagnostic_v1/result.json`，
SHA 81bbaaafb37e898524ac42b959e07bc322baf00ce0aaad4ed46e384fdd949388。

## token坐标不能直接作为独立像素证据

逐token实际processor输入块也已核验：native token p对应连续4个
vision patch，每块4×1176个FP32值。四图CPU重处理的完整pixel_values
SHA全部与GPU receipt相同。

ERASE_TARGET下，650/720个块的实际输入bytes完全未变，650个对应编码
全部改变；KEEP_TARGET下有37个输入块不变，37个编码全部改变。因此
本次干预直接显示编码依赖自身输入块以外的信息，不能把token地址等同
独立感受野。它没有定位影响来自具体哪一层，也不是ownership证明。

先前KEEP仍有优势的中心在GT外token346/445，cell实际上分别包含约
19.57%/24.35%的GT像素，不能拿它们证明完全背景cell的独立目标证据。
精确不变块证据来自实际processor数值，不依赖这个中心近似。
见`results/rc_outcome0212_packed_pixel_block_accounting_v1/result.json`，
SHA 026cca92850d2a8d6e518eb1fdb420b08d000df76534c62fb262d105d452ca28。

## 为什么旧action成功不能直接替V作证明

使用上述四种q_tokens重新计算C的全部b/br与四scalar，保持原RoMa
wq/wr、原RAW C128分数和原NATIVE7/C head固定，全部127 challenger
重算。ORIGINAL与原正确救回逐bit相同，但四条件均仍SWITCH到target：

| 条件 | 正确target的switch logit |
|---|---:|
| ORIGINAL | +1.652750 |
| KEEP_TARGET | +1.661150 |
| ERASE_TARGET | +2.367552 |
| ALL_GRAY | +1.888216 |

擦除后的内容score下降，冻结mass优势却保留；Q/R负权又奖励了翻转后的
roll关系，使action反而更强。这否定“该次条件救回必然需要query目标
内容通过ColNomic读出”这个解释，不否定旧成绩真实存在。原RoMa与RAW
仍携带原图信息；本实验没有把完整pipeline从灰图重新执行一遍。
不能说完整系统在看不到目标时也必然正确。

见`results/rc_outcome0212_frozen_roma_action_pixel_readout_v1/result.json`，
SHA c7a4aab092d4802f1e959bf851fcd460738cb1d99a1fbdf41b782fd97f3b6c48。

## 下一有限自动P检验

既有SAM3自动mask缓存与FULL64共有9条当前原图路径命中（TRAIN2/EVAL7），
包括原系统失败0050。复用该缓存与已冻结alpha：alpha只定位seed，取
最高confidence的合法seed连通mask component，V均匀读完整component；
不再让alpha削减区域内部每个身份token的权重。

固定query-only seed、reference-conditioned seed及既定位置/geometry
对照；不训练、不新运行任何encoder，先封存区域和C128分数再join标签。
这9条由可用性确定，不能拼接未覆盖query来声称32-query提升。旧SAM PT
未保存历史input image SHA，来源限制必须保留。执行定义见
`plan/RC_CACHED_SAM3_REGION_CLOSURE_V1_20260909.md`。
