# F128 correction and rival-aggregation cause analysis

同一F128固定0阈值下，B_CAL103→QR108/ANCHOR108/MULTI106。复算15fits和247个epoch排除了已检查的选模/梯度/计数问题；不同对手的比较分数被直接平均，可在固定MULTI权重下解释一个丢失纠错。另一个样本和新增误伤尚未被同一算子修复。

The positive development signal and failed replication trigger are both retained. Fixed-weight saved-edge diagnostics (106→107) are posthoc mechanism evidence, not a newly trained or validated model result. Native M and candidate axes remain unchanged. The 30-fit seed confirmation was not submitted.

- [Full cause report](../../ICLR/new%20ROUTEA/RC/reports/REPORT_TOKEN_COMPETITION_F128_V2_CAUSE_ANALYSIS_20260927.md)
- [Analysis code, data and reproduction commands](../../ICLR/new%20ROUTEA/RC/results/rc_token_competition_f128_v2/cause_analysis/README.md)
- [Original completed experiment](../token_competition_f128_v2_20260927/README.md)
- [File inventory](files.json) · [Export checks](validation.json)

This addition includes all four analysis programs, complete saved-score evidence, the original-result source dependencies already in the preceding archive, and prepared seed-confirmation programs for later reuse. No images, weights, token/embedding caches or runtime err/out logs are uploaded. The mechanism input snapshot contains scalar candidate predictions and labels only. The code/data relocation checks establish saved-result replay, not backbone/model training without the excluded assets.
