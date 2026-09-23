# new HYP 结果汇总包：2026-09-20 消融补充版

[H593六项消融](H593_ablation_zh.md) · [Excel](all_results.xlsx) · [原始证据](H593_ablation_evidence/)

本版保留2026-09-19汇总包的全部非ZIP内容，并增加H593的60个删项重训头及冻结置零对照；旧统计和图片不改写。


从[主总结](complete_results_zh.md)、[HTML](complete_results_zh.html)或[Word](complete_results_zh.docx)开始阅读。

- 32／128／593、GroZi、ISIC主要结果、历史负结果及RPC诊断：主总结与all_results.xlsx。
- CRISP、固定手填免训练对照、COST4/COST1/CE训练耗时：主总结第3.1／3.2节及原有CSV、图表。
- 完整理论：theory_zh.md／HTML／Word；theory_sources内7份原文按SHA保留，不再只有仓库链接。历史文中的实验状态按原日期理解，当前范围以主总结为准。
- processed128：主总结第3.3节、processed128_zh.md／HTML／Word、图表、6份CSV、Excel分组页及processed128_evidence原始结果和验证记录。
- 验证：archive_manifest.json记录本轮所有文件SHA和新增来源；validation.json记录本轮复算。historical_validation保存更新前快照；训练耗时本身没有重跑或改写。

processed128是合成处理回归，RAW121／COST1 122／CE123（分母128），不是旧EVAL128；净增区间包含0。RPC仍仅作参考图不足诊断。Ownership、完整过目不忘和新编码器训练属于未来工作。

本次补充不包含历史报告集合的其余原件、PPT/可见性图片集、原始输入、模型参数或可执行完整流水线。它们仍有仓库入口；本包不自称完整模型复现包。历史理论原文中的证据链接也可能需要原工作区。