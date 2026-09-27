# F128 V2：逐例分数原因审计

范围固定为 F128 已打开开发面板、原始 grouped five folds、seed0、原始自然 ColNomic C128。基线是同一 F128 phase-correct frozen 18-dimensional B_CAL；三个 arm 都是 B_CAL 加 native-token residual，主操作点 fixed0。原始最终验证为 `TOKEN_COMPETITION_F128_ALL15_INDEPENDENT_JOIN_PASS`。这里只读取已完成输出，没有训练、阈值调整、重选 checkpoint 或改写原始结果。

逐候选证据字段是 `zero_base`、`residual`、`logit`、`D`、`edges[].pair_difference`；分数满足 `logit = zero_base + residual`，RAW 分数严格为 0。最终动作选择分数最高且大于 0 的 challenger，否则 HOLD RAW。本审计重算了 384 个 arm-query 的基线和最终动作，核对原始 joined predictions；49,152 个候选分数均核对加法，读取的六个输入 SHA 在分析前后完全一致。

这些是已观察决策的算术解释。不能由 identity 名称推断文字、纹理、遮挡等视觉原因，也不能将分别训练的 ANCHOR/MULTI checkpoint 分数差当成同一权重下改 rival graph 的干预结果。

## 真正改变了哪些决定

| arm | 正确 /128 | 相对 B_CAL 改变 | rescue | break | 错→另一错 |
|---|---:|---:|---:|---:|---:|
| B_CAL | 103 | — | — | — | — |
| TOKEN_QR | 108 | 20 | 9 | 4 | 7 |
| TOKEN_QRR_ANCHOR | 108 | 20 | 9 | 4 | 7 |
| TOKEN_QRR_MULTI | 106 | 17 | 7 | 4 | 6 |

三者均损失基线已正确的 4/103 = 3.8835%。QR 与 ANCHOR 的 9 个 rescue 完全相同；但各 4 个 break 只有 3 个重合。ANCHOR 相对 QR 改 5 个决定：1 个改善、1 个退化、3 个错→另一错，所以准确率相同不表示逐例行为相同。

MULTI 相对 ANCHOR 改 6 个决定：丢掉 2 个 rescue，另有 4 个错→另一错，没有新救回，也没有消除 ANCHOR 的四个 break。MULTI 相对 QR 改 7 个决定：1 个改善、3 个退化、3 个错→另一错。完整列表见 `between_arm_decision_changes.csv`。

## 九个 rescue：不是只把分数推过 RAW

下表的 margin 定义为 target 分数减去该模型全部 C128 中 strongest wrong 分数；与单独的 target logit 不同。只有相对 strongest wrong 的竞争优势才足以选中 target。

| query ID（去掉 H593-） | B_CAL target rank | B_CAL margin | QR margin | ANCHOR margin | MULTI margin |
|---|---:|---:|---:|---:|---:|
| 08297e272b0164e6fdd42e9f | 2 | -1.880329 | 9.103937 | 6.317728 | 18.141328 |
| 1533c32a07d745928239de77 | 2 | -0.540170 | 0.235147 | 1.909236 | 1.433673 |
| 17e5724183278e475c4a304d | 2 | -0.439839 | 0.138240 | 0.002278 | 0.186636 |
| 1af45e7f0d56a52adeb60b07 | 6 | -1.812011 | 0.284465 | 1.051919 | -0.743059 |
| 1e203e93f22affa5c8c93e7b | 23 | -2.483570 | 4.696284 | 1.233089 | 1.148485 |
| 1e69202f59c1e38274747acc | 2 | -1.325391 | 4.810187 | 2.430122 | 2.657263 |
| 297ba148574e09f5723aa6b7 | 3 | -2.010613 | 4.188791 | 4.691465 | 2.528457 |
| 2e5153cd0e9bd1dddbb75dea | 6 | -1.425050 | 0.894400 | 0.004075 | -1.584615 |
| 30656f17b4be13cde8124a15 | 4 | -1.619245 | 3.646414 | 3.232404 | 0.360819 |

准确未四舍五入数值见 `primary_fixed0_cases.csv`。原始 B_CAL rank1 有 103 例，三个 arm 都保住 99 例；rank2 有 5 例，各救回 4 例；rank3–5 有 4 例，各救回 2 例；rank6–128 有 8 例，QR/ANCHOR 救回 3 例，MULTI 只救回 1 例（rank23）。所以 MULTI 的净退化具体集中在两个 rank6 case，而非缺少 target。

## MULTI 专有丢失：两个都是选错 challenger

`H593-1af45e7f0d56a52adeb60b07`：target row1183，主要错误对手 row1980。共同基线 target−wrong = -1.812011。ANCHOR 的 residual 差为 +2.863930，最终 margin +1.051919；MULTI residual 差降至 +1.068952，最终 margin -0.743059。MULTI target 最终分数 7.536341 已明显超过 RAW=0，但 row1980 得 8.279400；动作仍是 SWITCH，选择了基线原先的错误赢家。这不是 HOLD 阈值阻止纠正。

`H593-2e5153cd0e9bd1dddbb75dea`：target row935，错误对手 row1966。共同基线 margin -1.425050。ANCHOR residual 差 +1.429124，仅以 +0.004075 的极小 margin 救回；MULTI residual 差变成 -0.159566，margin -1.584615。MULTI target 7.784274、wrong 9.368890，也都是正分，仍是 SWITCH 选错 challenger。

MULTI 中 target 的 RAW edge/额外 rival edge 分别为 5.155611/2.072793 和 5.155725/1.687169；其 `D` 是两 edge 的均值，再减 `D_RAW` 得 residual。两个例子的错误赢家还分别对 row3975 形成额外 edge。因此正的 target-versus-rival 单边分数并不保证最终 C128 排名领先：最终比较是各候选分别聚合后的 D 差再加共同 B_CAL 差。完整 edge 和对应三臂候选记录见 `multi_specific_losses.json`。这是已训练解的分解，不能独立断言 graph 是唯一原因。

## 四个 break 与剩余失败

三臂共同 break：`H593-02ce6b371f2907428fc96ffe`（athletes-01）、`H593-04bd54f85fedc55966f9ee04`（vetoryl-09）、`H593-272a6668246b45626dea5ed4`（350 label）。三例原始 RAW 与 B_CAL 都已正确，target residual 固定为 0；错误 challenger 被 residual 推过 0，导致错误 SWITCH。

QR 第四个 break 是 `H593-0745854000292039c9c10f33`：B_CAL 已从错 RAW 切到正确 row1130，QR 将其对 row4599 的 margin 从 +0.926506 降到 -0.090661，导致改选错 challenger；ANCHOR/MULTI 此例仍正确。

ANCHOR/MULTI 第四个 break 是 `H593-29d505a42ed3f7b9bb256b2b`：正确 RAW row3334 的分数 0，错误 row3333 的分数从基线 -0.881234 升为 +1.198309/+0.148228；QR 下仍为 -0.479902，保住 HOLD。

QR/ANCHOR 最终各 20 错 = 8 个 target 不在 C128 + 4 个 target 在 C128 但 HOLD 错 RAW + 8 个 target 在 C128 却 SWITCH 到错误候选。MULTI 的 22 错 = 同样 8 个缺候选 + 同样 4 个可召回但 HOLD 的 query + 10 个可召回但错误 SWITCH。四个有 target 的错误 HOLD 共享 query IDs：`H593-174dc7ecf520dfa933141c09`、`H593-19cc2c81ca081e39a7d72b99`、`H593-1bef63e477961c10e0077a91`、`H593-2acfe07e77226545b2ae0e10`；target 最终分数均未越过 0。缺候选案例的 HOLD/SWITCH 变化都不能修正身份。

`failure_class=SWITCH_away_from_correct_RAW` 是相对 RAW 的错误类型，不能直接当作相对 B_CAL 的 break 数；以 `transition_vs_B_CAL` 统计基线回归。具体旧错是 `H593-20fdadbf18ce1e0e1a0746a9`：正确 RAW 为 row1134，但 B_CAL 已选错 row3887，三个 token arm 仍选 row3887。因此 QR 的这一类 4 例 = 3 个新 break + 1 个基线旧错；ANCHOR/MULTI 的这一类 5 例 = 4 个新 break + 同一个旧错。QR 剩下的新 break `H593-0745854000292039c9c10f33` 属于正确 challenger 被另一错误 challenger 超过。机器表明确区分两种定义。

## 文件

- `case_summary.json`：来源 SHA、计数、九个 rescue/四个 break 的完整 query IDs、模型间变化。
- `primary_fixed0_cases.csv`：384 行，全部 query×arm，含 target 深度、strongest wrong、固定对手分数分解与动作。
- `rescues_and_breaks.csv`：37 行，相对 B_CAL 改变正确性的案例。
- `errors.csv`：62 行，所有最终错误；`target_rank_depth.csv`：按 B_CAL rank 分层。
- `between_arm_decision_changes.csv`：18 行，三个模型比较的全部改变决定。
- `candidate_score_support.json`：每例 target、RAW、基线/最终赢家与 strongest wrong 的原始候选记录。
- `multi_specific_losses.json`：两例 MULTI 丢失的三臂分数与 edges。
- `build_case_analysis.py`：可重跑生成机器表；默认结果根从脚本所在目录解析，显式 `--root` 也可指定。

移植运行仅需 Python 标准库及归档结果根中的六个输入：`protocol.json`、`validation.json`、`joined_predictions.json`、`candidate_predictions.json`、`analysis/analysis_summary.json`、`analysis/per_query_diagnostics.csv`。例如在 Git clone 中运行 `python path/to/cause_analysis/cases/build_case_analysis.py --root path/to/results/rc_token_competition_f128_v2`。脚本不打开这些 JSON 内记录的原 HPC 绝对路径，而是核对归档输入的 SHA；无需模型权重、GPU 或原 HPC 文件系统。`relocation_check.json` 记录了临时迁移目录重跑通过，七个机器数据产物逐字节一致；`case_summary.json` 内的路径字段会按新根重写，数值及数据 SHA 保持一致。
