# new HYP：原七参数头为何救回3条并保住RAW25

对象为原RAW C128、RoMa soft visibility × full-reference image-token MaxSim的ORIGINAL7/NATIVE7。已打开matched EVAL32（11组），RAW25→28；本轮没有训练、换评分或读取扩展样本。

原头实际使用六维（下式只显示六位小数，执行使用完整FP64参数）：

    z = 1.007810 RAW − 3.780124 S + 5.716638 M + 5.912594 L
        − 0.148723 Q − 0.238728 R − 1.477875。

头中的S、M、L列均为原候选标量的对比，不能把头中的L列写成S列除M列。准确地说，候选标量 ell_g=S_g/max(M_g,epsilon)，L列=symmetric(ell_c,ell_w)；S列=symmetric(S_c,S_w)，M列=symmetric(M_c,M_w)。Q列=symmetric(S_c−S_Qc,S_w−S_Qw)，R列=symmetric(S_c−S_Rc,S_w−S_Rw)。RAW列=(RAW_c−RAW_w)/std_C128。取127个challenger最大logit，严格>0才SWITCH，否则HOLD。RAW winner策略分数为0，不是bias。

## 三次救回过了阈值门和竞争门

| Query | target row | target logit | 最强wrong logit | target−wrong |
| --- | ---: | ---: | ---: | ---: |
| DIFFICULT-0044 | 1130 | 1.436240263 | -0.410405673 | 1.846645936 |
| OUTCOME-0212 | 1024 | 1.652750227 | 0.993115362 | 0.659634864 |
| OUTCOME-0220 | 1024 | 0.012491441 | -1.506429622 | 1.518921063 |

target的六项有符号贡献（最后仍加bias −1.477874513）：

| Query | RAW | S | M | L | Q | R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DIFFICULT-0044 | -0.055777 | -3.266511 | +2.682311 | +3.927368 | -0.145526 | -0.227750 |
| OUTCOME-0212 | -3.713257 | -3.776334 | +5.225728 | +5.781851 | -0.148635 | -0.238728 |
| OUTCOME-0220 | -0.380490 | -2.666092 | +2.658396 | +2.113904 | -0.083316 | -0.152036 |

三条target均靠M与L的正贡献，越过RAW、S、Q/R及bias的负贡献；同时必须压过其它完整reference。0212中，target相对最强wrong的RAW项贡献约+0.867069，最终竞争margin只有+0.659635，RAW先验直接参与了身份竞争。
0220的竞争margin约1.518921，但SWITCH余量仅0.012491441；它的脆弱点是是否换掉RAW winner。

## 原正确的保持机制

25个RAW正确例全部通过HOLD保留。每条完整127个wrong中的最强者、六项和保护margin均已记录。bias在challenger间比较时抵消，在HOLD门中仍保留。最紧的保护余量：

- OUTCOME-0618：0.010972946；最强wrong row=4659。
- OUTCOME-0669：1.625713788；最强wrong row=3332。
- OUTCOME-0670：1.641465664；最强wrong row=3333。
- OUTCOME-0569：1.871156556；最强wrong row=4787。
- OUTCOME-0683：2.072088166；最强wrong row=220。

## 固定参数置零

| 条件 | 正确/32 | 原3个rescue保留 | 损失的原正确 |
| --- | ---: | ---: | --- |
| ORIGINAL | 28 | 3/3 | 无 |
| DROP_RAW | 25 | 2/3 | OUTCOME-0212, OUTCOME-0419, OUTCOME-0618 |
| DROP_S | 26 | 3/3 | OUTCOME-0419, OUTCOME-0618 |
| DROP_M | 25 | 0/3 | DIFFICULT-0044, OUTCOME-0212, OUTCOME-0220 |
| DROP_L | 24 | 0/3 | DIFFICULT-0044, OUTCOME-0211, OUTCOME-0212, OUTCOME-0220 |
| DROP_Q | 27 | 3/3 | OUTCOME-0618 |
| DROP_R | 27 | 3/3 | OUTCOME-0618 |
| DROP_QR | 27 | 3/3 | OUTCOME-0618 |
| DROP_S_QR | 26 | 3/3 | OUTCOME-0419, OUTCOME-0618 |

去M丢掉3次rescue而保持RAW25；去L还丢掉RAW正确0211。去RAW丢0212救回及0419/0618的HOLD。在所列固定参数置零条件中，去S或Q/R后本批3次rescue仍成立；这些项却保护某些原正确决策：去S损失0419/0618，单去Q、单去R或去QR均损失0618。
S/QR及S+QR联合置零复用已独立验证的完整logits。RAW、M、L、S块与各自单列相同；QR是两列响应块，不能把这些输入块当成独立的图像信息。
既有DROP_S_QR已证明原RAW/M/L系数能保留3次rescue，但只得26/32，无法替代完整头。因此没有枚举64个子集，也没有宣称最小或唯一组合。

## 原4个失败仍保留

| Query | target logit | 最强wrong logit | 原动作 | 计算层面失败位置 |
| --- | ---: | ---: | --- | --- |
| DIFFICULT-0050 | -3.218998338 | -5.894026421 | HOLD | TARGET_CANNOT_TRIGGER_SWITCH |
| OUTCOME-0213 | -1.723753118 | -0.191321167 | HOLD | TARGET_CANNOT_TRIGGER_SWITCH |
| OUTCOME-0373 | 0.407419904 | 1.924782839 | SWITCH | TARGET_LOSES_CHALLENGER_COMPETITION |
| OUTCOME-0676 | -1.629629041 | -1.451346752 | HOLD | TARGET_CANNOT_TRIGGER_SWITCH |

这些是固定模型的计算解释：六列来源和代数耦合，置零可能离开自然数据关系，不能解释为删除真实空间/内容信息或证明像素因果。局部条件必要性不是跨数据必要性。
仍未知：这些分量的像素级身份特异性，以及冻结原头在更大群体上的保持/救回稳定性。本账本可直接用于后续经授权的更大样本，此轮没有读取扩展universe。
全部4,064原logits逐bit回放；分项求和与矩阵运算最大舍入残差7.11e-15，动作使用完整原运算。全部32条、所有127 logits、六项和反事实动作均在result.json。
结果SHA：417259bf70cb9b6a1c9d0a993d265992201b0252d9f1059a0a5213c9cadfb442。

本V2仅澄清原标量与头输入对比列的记号，并收紧条件性措辞；V1及全部数值产物保留，未改变任何计算或结果。
