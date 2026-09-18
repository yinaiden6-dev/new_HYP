# OUTCOME-0337：reference权重在何处反转身份证据

本报告只解释已经打开的新EVAL128唯一新增损失，不是新模型或新准确率。原完整结果仍为RAW88→ORIGINAL7 99，12救/1损；没有修改评分、阈值或训练。

## 原图与来源

Query清楚可见LEADER的Infants Pain & Fever顶部及一面Drug Facts。target1134为同一LEADER婴幼儿款的展开包装；最终wrong1436为SoundBody儿童款包装，也有大片Drug Facts排版。原始RAW将target排第1、wrong排第17。

独立检查了query、两reference及同组两个救回0333/0345的实际文件SHA、TOKEN/RoMa元数据、身份来源记录、EXIF/frame与原图库物理行序；没有发现这些接口或精确重复问题。视觉上存在品牌/款式差别，但本次没有用人工框、mask或OCR信号训练模型。

## 固定tokens和原权重的分解

| 读取方式 | target1134 | wrong1436 | 本对比较 |
| --- | ---: | ---: | --- |
| 原RAW分数（image+template） | 509.999023 | 470.385681 | target高 |
| 全图、全reference、不加权cosine MaxSim均值 | 0.690415 | 0.636588 | target高 |
| 保留各自query权重，reference自由cosine MaxSim | 0.707504 | 0.594628 | target高 |
| 用原wr×cosine选位置后，只读该位置cosine | 0.443245 | 0.475781 | **首次反转** |
| 原L（含所选位置wr乘法） | 0.045377 | 0.328378 | wrong约7.24倍 |
| 原M | 0.004686 | 0.046676 | wrong约9.96倍 |

两侧8个原C4值均从保存tokens/maps逐bit复现；上表后三种读取是这个target/wrong对的后验诊断，没有声称重跑全部127个challenger后会得到何种新准确率。

保留query权重后的自由内容优势仍在。因此本例的反转可以明确定位到reference门控：它先改变reference内的argmax位置，让正确reference所读到的内容更差；随后所选位置的权重幅度继续放大错误优势。并非仅仅是最后线性头或二维投影旋转造成。

具体定义j*(i)=argmax_j wr(j)cos(q_i,r_j)、p_i=wq(i)/sum(wq)，则

    L = E_p[wr(j*) cos(q_i,r_j*)]
      = E_p[wr(j*)] E_p[cos(q_i,r_j*)] + Cov_p。

所选位置的平均wr为target0.108458、wrong0.711082，相差6.56倍；完整reference的平均wr相差32.15倍。L=S/M仍保留这部分权重幅度，不能称为不含geometry的纯身份分数。除以max(wr)后这一对仍是wrong更高；除以mean(wr)则反转，但后者还涉及权重集中程度，不等于证明了更好的identity评分。

最终错误logit为+1.273408；M/L贡献合计+9.150529超过RAW、S、响应与bias抑制。这解释了模型为何错误SWITCH。共享Drug Facts排版与视觉相似性是所见现象；尚未做像素干预，不能进一步断言它们是RoMa高权重的唯一原因，也不能推广为全部训练失败的共同根因。

## 为什么不能直接宣布修复

自由内容补列已完整试过：job5139024的FREE8保留原6列，追加sym(Fc,Fw)，F正是上述保留query权重的自由MaxSim；8参数全部重新训练后旧EVAL为27/32，原头28/32，损失0220。D_IMAGE替换证据后重训也为27/32。这些有效既有结果必须保留，不能把同一方案换名重跑。

当前FULL TRAIN32的四个错误0475/0477/0533/0538，也并非wrong的M/L同时高于target。因此“更多同类训练视图能学会避免该反转”仍是待检验假设，不能直接称已找到所有失败的根因。本轮先核查在原TRAIN/PAIR身份内扩充完整C128训练的可用数据，同时保持旧EVAL32及新EVAL128身份隔离。

来源：[逐项分解](../results/rc_eval128_0337_visibility_content_v1/result.json)、[原完整结果](REPORT_NEW_HYP_ORIGINAL7_EVAL128_MECHANISM_RESULT_V1_20260910.md)、[FREE8原结果](../results/rc_same_support_specificity_v1/result.json)、[D_IMAGE原结果](../results/rc_full_mass_free_identity_development_v1/result.json)。
