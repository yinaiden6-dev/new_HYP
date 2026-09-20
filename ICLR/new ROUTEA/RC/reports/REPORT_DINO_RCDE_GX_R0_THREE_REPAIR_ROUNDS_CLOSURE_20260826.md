# DINO-RCDE GX R0 三轮修复实验结项报告

日期：2026-08-26  
范围：GX-CBNR / H0 分支，不覆盖并行推进的 RGH 主线  
最终状态：`GX_CBNR_R0_NO_DEPLOYABLE_GEOMETRIC_EXCLUSIVITY` 保持不变

## 1. 结论摘要

正式 GX R0 已经给出有效的窄范围科学 NO-GO：当前
matched-null calibrated connected DINO relational energy 能保持全部 16 个
RAW-correct 样本，但只能使 16 个 RAW-wrong 样本中的 1 个转正；空间坐标
null `P_COORD` 甚至显著强于 REAL 路径。该结果通过独立验证，不是工程失败。

在 R0 之后，依次完成了三轮只改变一个核心因素的修复：

1. **R1Q：优化充分性修复**——不改表示、能量和 loss，只把训练延长到
   1,024 updates；
2. **R1P：native direct spatial-pair 修复**——不用 unary-vs-zero 读出，直接
   学习 `REAL` 相对 `P_COORD` 的 candidate pair score；
3. **R2A：live 4D consensus 修复**——不再冻结 post-consensus cache，使 DINO
   4D consensus 与 relational head 联合学习，并完成 16、32、72 updates 的
   同轨迹训练预算诊断。

三轮均未达到预注册门：

| 阶段 | 改变的唯一核心因素 | RAW-wrong 正方向 | 固定门 | 状态 |
|---|---|---:|---|---|
| Formal R0 | 固定 post-consensus GX-CBNR | 1/16 | 至少 11/16 | 科学 NO-GO |
| R1Q | 仅增加固定机制的优化预算 | 6/16 | 至少 11/16，且每折全部通过 | INSUFFICIENT |
| R1P | native `REAL` vs `P_COORD` pair loss | 7/16 | 至少 11/16，且每折全部通过 | INSUFFICIENT |
| R2A-16 | live 4D consensus，fold-1 | 1/4 | 至少 3/4 + 均值为正 | INSUFFICIENT |
| R2A-32 | 同一轨迹继续训练 | 0/4 | 同上 | INSUFFICIENT |
| R2A-72 | 同一轨迹继续训练 | 0/4 | 同上 | INSUFFICIENT |

最终判断是：**失败不能再主要归因于训练没收敛、unary-zero 读出错误，或
4D consensus 被冻结。当前 DINO relational representation/objective 能让平均
分数缓慢朝正确方向移动，却不能稳定产生可部署的 target-versus-rival 正方向，
也不具备 no-regret 性。继续扫描 update、LR、scale、threshold、pooling 或
HOLD 门没有依据。**

## 2. 正式母结果：GX R0

### 2.1 研究问题

给定 target-free 的自然 query、specific candidate reference、连通空间 root，
以及 matched candidate/spatial/region null，检查 candidate-specific relational
energy 是否同时满足：

- target 比 strongest rival 更强；
- REAL 比错绑 candidate 的 `C_BIND` 更强；
- REAL 比空间坐标破坏 `P_COORD` 更强；
- REAL 比 matched region resample 更强；
- 允许安全 HOLD/SWITCH，并使 rescue 多于 break。

### 2.2 正式结果

- 评价规模：4 folds，32 queries；其中 RAW-correct 16，RAW-wrong 16；
- RAW-correct retention：16/16；
- rescue / break：1 / 0；
- RAW-wrong final positive direction：1/16，低于 11/16 门；
- group-balanced `target Z - rival Z`：`+0.19636`，95% CI
  `[-0.15980, +0.54825]`，下界不为正；
- `REAL - C_BIND`：`-0.15281`，95% CI
  `[-0.50379, +0.17050]`；
- `REAL - N_REGION_RESAMPLE`：`-0.18449`，95% CI
  `[-0.51667, +0.16721]`；
- `REAL - P_COORD`：`-0.82115`，95% CI
  `[-1.17104, -0.47699]`，方向显著反转；
- 只有 fold-2 出现 1 次 rescue；fold-1、3、4 均为 0；
- 所有工程不变量和 access audit 通过，opened/sealed read 为 0。

因此正式状态为：

```text
GX_CBNR_R0_NO_DEPLOYABLE_GEOMETRIC_EXCLUSIVITY
```

该结论只否定当前 GX-CBNR matched-null relational energy 的可部署几何排他性，
不否定 DINO token 含信息、不否定 reference-conditioned hypothesis 总方向，也
不是 full-gallery retrieval 或 ownership 的总 NO-GO。

## 3. 第一轮修复：R1Q 优化充分性

### 3.1 修复假设

R0 可能只是 848 参数的 `ell/rho/F` head 没有被充分优化。为隔离这一因素，
R1Q 保留：

- R0 post-consensus relational cache；
- 848 个 trainable head 参数；
- C/P/N branches；
- exact loss；
- correct/wrong 0.5/0.5 采样；
- fold split、seed、weight decay 和 gradient clipping。

只改变训练预算：四折 fresh INIT、固定 LR `3e-4`、1,024 updates，将冻结的
128-record schedule 完整重复 8 次，不进行 checkpoint、threshold、scale 或
action 选择。

### 3.2 结果

- 32-query、4-fold 全部完成；
- 16 个 RAW-wrong 中，几何方向为正的只有 **6/16**，低于 11/16；
- 四折均未通过全部 optimization + decodability gate；
- fold-1 的 mean `target Z-rival Z` 仍为 `-0.133996`；
- fold-2/3/4 的均值虽然为正，但每折的 `REAL-P_COORD` 仍为负；
- RAW-wrong loss 没有任何一折达到预注册的 10% 下降门；
- fold-1、4 的 combined loss 也未达到 5% 下降门。

状态：

```text
GX_R1Q_OPTIMIZATION_REPAIR_INSUFFICIENT
```

独立验证：PASS。

### 3.3 含义

增加 head 优化预算确实把正方向从 R0 的 1/16 提高到 6/16，但仍远低于部署
门，且空间 null 方向没有被修正。因此“原机制只是没训练够”不是充分根因；
固定 post-consensus representation 和/或能量定义仍有限制。

## 4. 第二轮修复：R1P native direct spatial pair

### 4.1 修复假设

R0/R1Q 把 relational output 构造成相对 exact-zero 的 unary energy，可能与
RCDE comparator 的原生 pairwise 语义不一致。R1P 改为：

```text
S_P(q,g) = PairComparator(REAL(q,g), P_COORD(q,g))
L = softplus(-(S_target-S_rival)) + softplus(-S_target)
```

固定 denominator、roots、directions、caches、candidate binding、848 个参数、
四折、1,024 updates 和 LR `3e-4` 均不改变。RAW fusion、SWITCH/HOLD、
`C_BIND` 与 `N_REGION_RESAMPLE` 不进入该窄资格门。

### 4.2 结果

- 所有 fold 的训练 loss 均下降至少 10%；说明 native pair loss 可优化；
- 16 个 RAW-wrong 中，target-rival 正方向为 **7/16**，仍低于 11/16；
- fold-1 final mean target-rival 为 `-0.03903`；
- fold-2/3/4 target-rival 均值为正，但四折的 mean target spatial-pair score
  仍全部为负；
- 没有任何一折通过完整 direct-pair gate。

状态：

```text
GX_R1P_DIRECT_SPATIAL_PAIR_INSUFFICIENT
```

独立验证：PASS。

### 4.3 含义

direct pair 形式比 R1Q 多得到 1 个正方向（7/16 vs 6/16），并显著改善训练
loss；但 held-out spatial-pair 绝对方向仍为负。由此可排除“主要只是
unary-zero 实现把信号读坏”的解释。问题开始收窄到 representation、空间证据
本身以及训练目标的泛化语义。

## 5. 第三轮修复：R2A live 4D consensus

### 5.1 修复假设

R1Q/R1P 都使用冻结的 post-consensus cache。如果真正缺口在 4D
query-reference correspondence 没被自然 hard-negative 监督塑形，那么只训练
后端 head 无法修复。因此 R2A：

- 从原 fold-local RCDE checkpoint 初始化；
- 使 `4D_consensus + ell + rho + F` 联合训练；
- trainable scalars 增加到 2,656；
- 每次从 live query/reference token 重新执行 REAL 与 `P_COORD`；
- 使用同一 target-rival loss：
  `softplus(-(target-rival)) + softplus(-target)`；
- fold-1 使用 16 train episodes 和 8 identity-disjoint evaluation records；
- 评价集含 4 个 RAW-correct 和 4 个 RAW-wrong。

为检查是否仍是训练不足，使用同一模型、AdamW moments、数据顺序和初始评价，
精确连续执行 16、32、72 updates。24 个 raw contexts 通过 immutable hardlink
复用；32 从 update-16 续训，72 从 update-32 续训，没有重新初始化。

### 5.2 训练预算曲线

| Updates | last-8 loss | mean target score | mean target-rival | RAW-wrong 正方向 |
|---:|---:|---:|---:|---:|
| 0 | — | -1.52951 | -0.47501 | 1/4 |
| 16 | 1.79570 | -1.42474 | -0.41386 | 1/4 |
| 32 | 1.69235 | -1.33179 | -0.34660 | 0/4 |
| 72 | 1.46125 | -1.17472 | -0.26726 | 0/4 |

从 16 到 72：

- mean target score 改善 `+0.25003`；
- mean target-rival 改善 `+0.14660`；
- 但 RAW-wrong 正方向从 1/4 降为 0/4；
- 72 步仍不满足 mean target score > 0；
- 72 步仍不满足 mean target-rival > 0；
- 72 步仍不满足至少 3/4 RAW-wrong 正方向。

四个 RAW-wrong 的逐例轨迹：

| Query | 16步 margin | 32步 margin | 72步 margin | 判断 |
|---|---:|---:|---:|---|
| OUTCOME-0307 | -0.75234 | -0.57845 | -0.22337 | 持续改善但未转正 |
| OUTCOME-0532 | -0.76917 | -0.54934 | -0.15571 | 持续改善但未转正 |
| OUTCOME-0693 | -0.43211 | -0.41485 | -0.31837 | 小幅改善但未转正 |
| OUTCOME-0755 | +0.01951 | -0.00595 | -0.18367 | 随训练被破坏 |

三个预算点均为：

```text
GX_R2A_LIVE_CONSENSUS_FOLD1_INSUFFICIENT
```

32、72 结果均通过修正后的独立 validator。预算曲线状态为：

```text
GX_R2A_BUDGET_CURVE_OPTIMIZATION_NOT_REPAIR
```

### 5.3 含义

R2A 是三轮中最关键的定位结果：loss、mean target score 和 mean margin 都随
更新数平滑改善，证明梯度、参数更新和 live 4D 链路正常；但决策级正方向没有
改善，甚至把唯一的弱正例 OUTCOME-0755 推成负例。这说明：

1. 当前 objective 能学习一个总体平滑的相对趋势；
2. 该趋势不是稳定的 exact-reference decision evidence；
3. 更多 updates 会放大平均趋势，但不能保证跨过 strongest-rival 边界；
4. 当前机制缺乏 no-regret 约束或缺乏能区分近重复 target/rival 的观测证据；
5. “再多训练一些”不能作为下一步修复。

## 6. 三轮之后收窄的根因

### 6.1 已排除的主要解释

- **不是工程没跑通**：formal R0、R1Q、R1P、R2A 结果均完整写盘；正式结果和
  开发结果均经过逻辑 hash、population、checkpoint 与 gate 重算。
- **不是完全没有梯度**：R1Q/R1P 的参数显著更新，R2A 训练 loss 从首 8 步
  均值 `1.86547` 降至 72 步末 8 步 `1.46125`。
- **不只是训练步数不足**：16→32→72 平均值改善而 decision direction 变差。
- **不只是 unary-zero 读出错误**：R1P 使用 native direct pair 后仍只有 7/16。
- **不只是冻结 4D consensus**：R2A 解冻 4D consensus 后仍未得到 3/4 正方向。
- **不是 HOLD 门单纯过严**：R2A 在 action/HOLD 之前的 target-rival score 本身
  仍为负。

### 6.2 当前最可信的剩余瓶颈

当前 DINO dense correspondence 与 connected-root 条件中存在可优化的统计相关性，
但它仍不足以稳定回答：

```text
在自然近重复竞争下，这组连通 correspondence 是否只可由 candidate g 解释，
而不能由 strongest rival c 或 matched spatial null 同样解释？
```

具体表现为：

- candidate-common 纹理、文字、颜色和矩形结构仍能同时支持 target 与 rival；
- spatial null 不仅没有稳定销毁证据，formal R0 的 `P_COORD` 反而系统性更强；
- 平均分数可被优化，但每个 query 的符号和安全性不可预测；
- 当前 objective 没有提供足够强的自然 reference-specific exclusivity，也没有
  保护原本弱正的 hard case。

因此瓶颈位于**可观测的 reference-specific evidence 与 no-regret 泛化**，不是
继续增加相同表征上的优化强度。

## 7. 科学结论边界

### 可以声称

1. formal GX R0 是有效的窄范围科学 NO-GO；
2. 当前 fixed/live DINO relational mechanism 在 matched null 下没有达到可部署
   geometric exclusivity；
3. 更长 head 优化、native direct-pair readout、解冻 live 4D consensus 和增加
   update budget 均未突破该瓶颈；
4. 当前结果把失败从“工程/优化/读出可能有错”收窄到“自然 exact-reference
   evidence 和 no-regret 泛化不足”。

### 不能声称

1. 不能说 DINO token 没有信息；
2. 不能否定所有 reference-conditioned target hypothesis；
3. 不能把 R1Q/R1P/R2A 当独立科学 endpoint；它们是 formal R0 后的 adaptive
   development diagnostics；
4. 不能推出 full-gallery retrieval、target discovery 或 ownership 的总 NO-GO；
5. 不能因为平均 margin 变好就声称有 deployable rescue；
6. 不能在当前 fold 继续扫描 update、LR、scale、threshold、pooling 或 action
   gate 并把最优结果当科学证据。

## 8. 工程审计说明

- R2A 首次 16-step job `5110586` 正常完成，24 contexts、16 checkpoints 和结果
  均写盘；
- 32-step job `5110756` 与 72-step job `5110757` 正常完成；
- 自动 finalizer job `5110760` 首次因 validator 错误假定续训目录必须重复保存
  `1..N` 全部 checkpoints 而失败；
- 实际续训目录按合同只包含 parent checkpoint 与新增 checkpoints，模型结果不受
  影响；
- validator 已修正为核对连续的 `parent + (parent+1..N)` 序列；
- 32/72 结果随后均通过独立重算，预算 reducer 成功写出；
- 修复只改变只读验证逻辑，没有改变模型、训练、数据、checkpoint 或结果数字。

## 9. 最终决策

GX/H0 当前分支到此结束：

```text
formal scientific status:
GX_CBNR_R0_NO_DEPLOYABLE_GEOMETRIC_EXCLUSIVITY

adaptive repair closure:
R1Q INSUFFICIENT
R1P INSUFFICIENT
R2A OPTIMIZATION_NOT_REPAIR

next_authorized_stage: null
```

不再推进同一 GX family 的训练预算或 readout 扫描。该结论不修改、不取消、也
不干扰并行 RGH 主线；RGH 必须按其自己的 authority、完整 C128 strongest-rival、
`C_BIND`、`P_COORD` 和 no-regret gate 独立得出结论。

## 10. 证据入口

### Formal R0

- 合同：`plan/DINO_RCDE_GX_CBNR_R0_CONTRACT_V1_20260824.md`
- 正式结果：`results/dino_rcde_gx_formal_r0_science_v1/result.json`
- 独立验证：`results/dino_rcde_gx_formal_r0_science_validation_v1/result.json`
- science authority：`registry/h0/gx_cbnr_formal_r0_science_authority_v68_20260825.json`

### R1Q

- 合同：`plan/DINO_RCDE_GX_R1Q_OPTIMIZATION_REPAIR_V1_20260826.md`
- reduction：`results/dino_rcde_gx_r1q_optimization_reduction_v1/result.json`
- validation：`results/dino_rcde_gx_r1q_optimization_validation_v1/result.json`
- jobs：`5110134_[1-4]`，reducer `5110135`

### R1P

- 合同：`plan/DINO_RCDE_GX_R1P_DIRECT_SPATIAL_PAIR_V1_20260826.md`
- reduction：`results/dino_rcde_gx_r1p_direct_pair_reduction_v1/result.json`
- validation：`results/dino_rcde_gx_r1p_direct_pair_validation_v1/result.json`
- jobs：`5110150_[1-4]`，reducer `5110151`

### R2A

- 16-step 结果：`results/dino_rcde_gx_r2a_live_consensus_fold1_v1/result.json`
- 16-step validation：
  `results/dino_rcde_gx_r2a_live_consensus_fold1_validation_v1/result.json`
- 32-step 结果：
  `results/dino_rcde_gx_r2a_live_consensus_fold1_32_v1/result.json`
- 32-step validation：
  `results/dino_rcde_gx_r2a_live_consensus_fold1_32_validation_v1/result.json`
- 72-step 结果：
  `results/dino_rcde_gx_r2a_live_consensus_fold1_72_v1/result.json`
- 72-step validation：
  `results/dino_rcde_gx_r2a_live_consensus_fold1_72_validation_v1/result.json`
- 16/32/72 curve：
  `results/dino_rcde_gx_r2a_live_consensus_fold1_budget_curve_v1/result.json`
- jobs：16-step `5110586`，32-step `5110756`，72-step `5110757`

