# LISTWISE 正确性翻转的逐图检查与条件分流问题

已检查全部14张翻转query及26张不同reference原图。结果支持从具体可见证据和候选混淆类型入手；它没有证明按照片类型分流就一定有净增。

比较对象：相同冻结RAW full-gallery C128、127 challenger HOLD/SWITCH；ORIGINAL7/ec7与LISTWISE_UNIT1。二者均原PAIR64＋FULL32训练、六特征七参数。LISTWISE相对ORIGINAL7同时改变错误成本及FULL loss；相对TRAIN_UNIT_COST7才是仅改FULL loss的对照。

| 固定开发面板 | RAW | ORIGINAL7 | LISTWISE_UNIT1 | 对原头救/损 |
|---|---:|---:|---:|---:|
| EVAL32 | 25/32 | 28/32 | 26/32 | 0/2 |
| EVAL128 | 88/128 | 99/128 | 101/128 | 7/5 |

14张只代表上述两面板全部正确性翻转，不代表全部错误或数据集构成。这里不估计某类型在32/128/593中的发生率，不把两面板拼成一个新成绩。目视审阅已知结果，非盲审；源绑定及计数验证不等于视觉解释已独立验证。

## 照片观察与可计算的失败环节

**7张救回：旧头全部已将target排为最强challenger，但logit小于0，最后HOLD。** 它们不是本次才首次找到target；新头使其相对RAW的分数越过0。7张来自5个来源组，其中0809、0818、0821同组。

**7张损失：6张由正确RAW/HOLD变成错误SWITCH；0140则由正确SWITCH变成错误SWITCH。** 0140属于候选间排序反转，其余6例首先表现为错误挑战者越过RAW。两种失败不能用同一“调低HOLD门”解释。

原头与新头读取同一六维候选证据；这些动作事实由封存logit直接复核。人眼可见的文字、颜色差异是否被token保留、被加权或被头使用，仍是另一个问题。

| 照片中主要展示的面（仅14张） | 救回 | 损失 |
|---|---:|---:|
| 主品名/规格展示面 | 2 | 2 |
| 说明/表格/厂商/条码面 | 5 | 4 |
| 通用品类文字与色块面 | 0 | 1 |

同一粗类型里同时存在收益和损失：0818的120 mg可见且被救回，P03的24 mg也可见却被换成160 mg。0278与0280是同一目标盒的不同面，前者救回、后者损失。不能据此制定“说明面统一用旧头、正面统一用新头”或按产品身份硬分流的规则。

## 完整逐例记录

| Query | 面板/变化 | target / 原选择 / 新选择（physical row） | 照片与竞争reference的差异 |
|---|---|---|---|
| OUTCOME-0106 | EVAL128 / 救回 | 930 / 4335 / 930 | 绿色药盒背面与侧面；Drug Facts、CVS Health Allergy Nasal Spray、#797907及条码数字可见，手遮住部分左缘。 target 930 是 CVS Allergy Nasal Spray；旧错 4335 是 CVS Children's Fluticasone Propionate，绿色布局及大量说明相似。query 侧面产品名称、#797907和条码支持930，4335侧面文字与编号不同。 |
| OUTCOME-0809 | EVAL128 / 救回 | 512 / 508 / 512 | 单盒放在木桌上，盒体只占画面一部分；VETORYL CAPSULES及紫红规格角标120 mg清楚可见，另有警示文字面。 target 512 为120 mg；旧错508为20 mg，同系列版式接近，但角标数字和颜色不同。 |
| OUTCOME-0130 | EVAL128 / 损失 | 935 / 935 / 2732 | 小盒侧面和底部；上方是Drug Facts续页及非活性成分，下方可见经销商段落、CONTAINS NO ASPIRIN及货号/条码200000002710，未见正面主品名。 target 935 展开图包含这块侧面与货号。新错2732是Marc's Aspirin 325 mg，亦有白底黑字续页，但经销商、成分文字及布局不同。 |
| DIFFICULT-0028 | EVAL128 / 救回 | 1013 / 2224 / 1013 | 手持蓝边黄顶说明面置于多个药盒前；Docusate Sodium 100 mg和顶部封口警示清楚可见。 target 1013为H-E-B 100 mg，黄顶封口警示和说明布局匹配；旧错2224为CVS 50 mg，虽然同为蓝色Stool Softener，规格、顶部警示和布局不同。 |
| OUTCOME-0140 | EVAL128 / 损失 | 935 / 935 / 3663 | 红色盒顶；8 Hour Arthritis Pain Relief、Acetaminophen Extended-Release Tablets USP 650 mg及黄底功效文字可见。 target 935有对应红色顶盖。RAW错4358是黄色CVS版本；新错3663为蓝色CareOne版本。三者共享650 mg及相似功效用语，但颜色、字块组织及包装不同。 |
| OUTCOME-0795 | EVAL128 / 损失 | 499 / 499 / 325 | 白色侧面主要是Swine说明和Table 2. Tulieve Swine Dosing Guide两列表格，底面条码可见；表内数字不同于候选325的表格。 target 499为Tulieve，包含对应表格和底部条码；新错325为Enrofloxacin 100，亦有动物剂量表但标题、数值与列组织不同；中间模型选91 Doraject，为另一种动物药说明。 |
| OUTCOME-0278 | EVAL128 / 救回 | 4776 / 4794 / 4776 | 只有白色窄侧面，文字为US Govt License #1725、Manufactured by Sanofi Pasteur Inc.及Swiftwater地址；文字相对画面转向，主产品名不在该面。 target 4776的窄侧面与此一致。旧错4794为Flublok，其厂商面同样出现Sanofi Pasteur/Swiftwater，但制造商段落、license #1795及布局不同。 |
| OUTCOME-0821 | EVAL128 / 救回 | 512 / 507 / 512 | VETORYL盒的紫色厂商/条码面；Dechra地址、Rev. February 2022和条码17033-112-30可见；该面没有120 mg字样。 target 512的条码为17033-112-30；旧错507是5 mg版本，其同版面条码为17033-105-30，其他大段文字及配色相近。 |
| OUTCOME-0769 | EVAL128 / 救回 | 409 / 411 / 409 | 倒置的白色说明面，含狗用警示、两支给药注射器图示和条码5023534022572；主品名不在该面。 target 409 Loxicom含相同白色说明/图示/条码组合；旧错411 Meloxidyl含相似警示和注射器图示，但文字组织、图示所在底色和条码不同。 |
| OUTCOME-0280 | EVAL128 / 损失 | 4776 / 4776 / 4777 | 倒置说明面，2025–2026 season、45 micrograms、per 0.5 mL和15 mcg等文字可辨；窄侧面Sanofi文字也部分可见。 target 4776为对应Fluzone 2025–2026；新错4777为Fluzone High-Dose 2025 Southern Hemisphere，相似大段说明中写240 micrograms、per 0.7 mL和60 mcg，另有菌株差异。 |
| OUTCOME-0818 | EVAL128 / 救回 | 512 / 511 / 512 | 较清楚的正面，VETORYL CAPSULES、120 mg紫红角标、Dechra和30 Capsules均可见。 target 512为120 mg；旧错511为10 mg，绿色规格角标，主体设计与大段文字相同。 |
| NDV2-005-P03 | EVAL128 / 损失 | 32 / 32 / 16 | 黑色物体遮住盒正面下部及部分文字；VETone、Comfor-trate、NDC 86136-102-35与24 mg仍清楚可见。 target 32为24 mg紫色版本；新错16为160 mg红色版本，主版式接近但规格数字、NDC及颜色不同。 |
| OUTCOME-0618 | EVAL32 / 损失 | 3234 / 3234 / 4659 | 主要为Drug Facts续页警示；没有正面品牌/规格。可见aspirin相关语句、ringing in the ears和妊娠段落等具体文字，细字整体清晰度尚可。 target 3234为Best Choice Aspirin 325 mg，对应续页文字与布局；新错4659为Naproxen Sodium 220 mg，亦有高度相似的白底黑框警示面，但多处语句与段落不同。 |
| OUTCOME-0669 | EVAL32 / 损失 | 3334 / 3334 / 3332 | 肤色窄侧面只见BB CREAM SPF 20，未见正面品牌或BEIGE/FAIR色号名称；拍照亮度与色偏未知。 target 3334为e.l.f. BEIGE；新错3332为FAIR。两者该侧面文字与版式相同，参考侧面主要差在底色，款式名称在未展示的正面。 |

## 照片类型小头：有检验价值，但需明确其输入和职责

用户提出“不同类型图片使用不同突破方法”。可细化为三层：当前query显示哪些线索；这些线索是否区分当前候选；现有证据是否足以支持SWITCH。照片类型是第一层，不能直接替代后两层。

只看query的小头可以在检索前形成软权重；但同一张query面对不同reference，具有区分力的区域会变。比如同为VETORYL，规格或条码数字比大段共同文字更关键；跨不同药品时，主品名或表格内容可能已够。若门只读query，就没有直接观察候选差异；这是一项明确的输入限制，不是已经证明它必败。

因此值得对照的是“固定权重”“仅query条件”“query＋候选竞争条件”，让检索标签学习共享分配规则。路由采用软权重可以表达一图多种证据，不必强制命名为正面/背面等离散类型；但隐变量不会仅因retrieval训练就自动具有这些人类语义。

保持retrieval-only：训练监督仍为query对应哪个reference；本文件的图片类型、手读数字及EVAL救损标签禁止进入训练。复用已有ColNomic/RoMa输入，不增加SAM、OCR或人工区域标注。具体结构尚未冻结或运行，不能把这一设计建议写成已验证方案。

先在身份/来源组隔离的TRAIN划分检验；任何模型选择与特征选择也只用TRAIN内部分组。比较时保持训练图、候选、损失、专家/残差容量及预算匹配，逐臂报告rescue/loss/net。是否超过原强基线与是否支持条件信息的必要性分别报告；允许少量损失，不恢复零损失门。

## 与旧条件化工作的区别和重叠

已经做过条件化，不能把“再加一个小头”本身作为新方法。已核对run_rc_train128_disagreement_oof4_v1.py与run_rc_new_hyp593_oof5_v1.py：旧CONDITIONAL4根据RAW差、自由/加权选点分歧差、reference平均权重差及截距，仅调节自由内容差dF的非负补偿强度。它不是从照片token学习可见面/线索的前置路由器。

TRAIN128分组OOF中，BASE7与统一补偿均108/128、CONDITIONAL4为107/128；这是TRAIN留出，不是固定EVAL128。593五折强GROUP_BASE与GROUP_CONST均447/593、GROUP_COND446/593。旧四参数条件门没有证明优于匹配强对照。

本轮有意义的区别应是增加此前门未观察到的、由query内容或候选细粒度差异产生的条件信息，并隔离它是否有用；不能只换门名字或扩参数。当前没有断言所有历史分支都从未使用图像条件，也没有找到这个新输入组合的已完成匹配实验。

下一步尚待验证的是：这些细粒度差异在既有token和匹配过程中是否保留下来，以及TRAIN监督能否利用它们区分“正确挑战者被压住”与“错误挑战者被放行”。若输入丢失区分信息，只有读旧六个统计量的小头无法凭空恢复它；若信息在而读出失配，条件路由才有可利用的空间。

证据边界：本轮完成的是全部翻转样本的照片与动作联合诊断，未训练新小头、未增加准确率、未完成像素因果验证或new HYP GO。

[逐图浏览器](../results/rc_opened_listwise_flip_visual_audit_v1/index.html) · [逐例观察与不确定性](../results/rc_opened_listwise_flip_visual_audit_v1/visual_review.json) · [原图路径、SHA与logit](../results/rc_opened_listwise_flip_visual_audit_v1/manifest.json) · [覆盖及计数复核](../results/rc_opened_listwise_flip_visual_audit_v1/validation_and_summary.json)

[本轮模型比较](REPORT_FULL_CANDIDATE_IDENTITY_LOSS_V1_20260912.md) · [旧TRAIN128条件门](REPORT_NEW_HYP_TRAIN128_DISAGREEMENT_OOF4_RESULT_V2_LAYOUT_20260910.md) · [593强基线条件门](REPORT_H593_UNIFIED_HYPOTHESIS_PREDICTION_READOUT_V1_20260911.md)
