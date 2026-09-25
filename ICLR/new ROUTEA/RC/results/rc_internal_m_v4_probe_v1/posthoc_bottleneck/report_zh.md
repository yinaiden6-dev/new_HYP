# V4 PROBE8：内部内容变化与最终纠错的分界

本次只读封存预测，未训练、未调整阈值、未新增推理。原任务5162476正常完成，NumPy核算误差约3.55e-15。
8张来自6个component；目标在自然C128内7张。RAW与内部REAL均4/8、0救0损；外部ADDITIVE4和PRODUCT5均6/8、2救0损。

| RAW错误query | 身份 | REAL内容目标排名 | 同模型恒定M排名 | 同模型错绑M排名 | REAL动作目标排名 | 定位 |
|---|---|---:|---:|---:|---:|---|
| H593-f522efe671aecfa850b10989 | 766-e9-infants-pain-and-fever | 2 | 39 | 70 | 6 | Competing wrong challenger outranks target |
| H593-9bea68f2367dffd8339109ca | 964901_LR | — | — | — | — | Target absent from natural C128 |
| H593-90acde9b0567a472f4232120 | cefdinir-fig2 | 1 | 3 | 3 | 2 | Target is best challenger but not accepted over HOLD |
| H593-7a2a329a01beb7a806b52b3e | alp0b-0002-29 | 2 | 2 | 2 | 2 | Target is best challenger but not accepted over HOLD |

## 解释

- 766-e9-infants-pain-and-fever：真实M使内容目标排名达到2，恒定/错绑分别39/70；但联合头动作目标仅第6。TRAIN重训头后目标分数虽为正，错误挑战者仍更高。因此不是单独降低接受阈值能修好的样本。
- cefdinir-fig2：真实M下内容目标第1，恒定/错绑均第3；正确目标也是最佳挑战者，但动作分数仍为负，HOLD保留了错误RAW。联合头目标分数约-0.385，TRAIN重训头约-0.0765。这里表现为接受校准未将较好排序转为纠错。
- alp0b-0002-29：内容目标仍第2，内部及外部均未纠错。另1张目标缺席C128，与内部读出是否有效是不同限制。

因此并非内部M在probe上完全没有作用：至少两例的目标内容排名改善；但这次没有增加最终正确数。
这种候选条件内容分数的变化，不自动证明patch语义表示更优或局部注意力学到了身份线索。
这里只能定位本小面板的竞争排序与接受校准两个环节；不能据此手调probe阈值，也不能称为已经找到所有泛化失败的统一根因。
若继续模型开发，应仅在TRAIN内研究完整候选竞争与接受校准，在单独冻结的评估上验证；本轮不自动提交新的训练。
