# F128 原 token 多候选比较 V2：完整任务链已提交

2026-09-27。用户授权：“启动，提交完整任务链。”

本轮检验：保留已有效的 M0 与同面板冻结基线后，读取未经过空间平均的局部内容，并比较多个竞争候选，能否提供原 QR/QRR 未取得的判别收益。已提交全链；当前尚无新科学结果。

## 最新调度修订：训练 2–14 改为 accelerated

用户追加要求“5167554的4-14任务加一个空载GPU，去accelerated分区”。11 项已逐一迁移并核验，最新核查时 **5167554_4–14 全部 RUNNING**。

用户随后要求“2 3也移过去”，5167554_2、_3 也已逐项核验迁入 accelerated，各加 1 张空载 GPU；累计迁移范围为 **2–14，共 13 项**。第二次迁移记录位于 `accelerated_train_2_3/migration.json`。

- 每项请求变为 8 CPU、32GB、1 GPU、10 分钟；原数组标识、依赖、断点与同 ID requeue 保留。
- GPU 不参与训练：封存 launcher 仍设置 `CUDA_VISIBLE_DEVICES=''`，模型与数据继续使用 CPU。实验源码、模型定义与科学协议均未改变。
- 2–14 已从自动 CPU/dev 补位范围排除；0–1 及汇总 5167555 继续 CPU 主队列、dev 自动补位。补位程序按新增范围重启，修订保存在 `dev_chain_placement/accelerated_2_3_amendment.json`。
- 逐项迁移前后调度字段和最新运行记录位于 `results/rc_token_competition_f128_v2/accelerated_train_4_14/`；补位范围修订见 `dev_chain_placement/accelerated_amendment.json`。

## 此前调度修订：CPU 主队列，dev 自动补位

用户随后要求“该链所有任务都是cpuonly排队，dev补位”。已为整个剩余链生效，下面提交表保留最初提交记录。

- 精确范围为 5167552 的 16 片、5167553 验收、5167554 的 15 项训练和 5167555 汇总。保留原 Job ID、数组上限、依赖、资源和同 ID requeue。
- 普通任务以 `cpuonly` 排队；有 dev 名额时，将可运行的待排任务改为 `cpuonly,dev_cpuonly`，两个入口都保留。依赖等待阶段的验收/汇总回到 `cpuonly`，不提前占用 dev 名额。
- 实测 dev QOS 为每用户最多 4 个提交、1 个运行，因此整组 16 片直接迁入被调度器拒绝。采用自动补位，不宣称 dev 可并行 16 个。
- 独立的调度程序每 30 秒检查名额，完成后补入后续任务，覆盖后续训练与汇总；只写上述待排 Job 的 Partition，不提交、取消或重启实验。最长运行 7 天。
- 当前程序已在用户环境独立验证存活并实际完成 6 次分区变更；其中 5167552_1–4 已进入双分区队列，其他缓存片仍在普通 CPU 排队。
- 状态与前后资源核验记录位于 `results/rc_token_competition_f128_v2/dev_chain_placement/`，包含 `authorization.json`、`status.json`、`events.jsonl`、`live_verification.json`。科学协议与训练程序未变更。

## 范围与比较

- 数据为自然采集 execution ordinal 0–127 的 F128，原 grouped 五折；不是历史 EVAL128，也不是 H593 全量训练。
- 每张沿用原 ColNomic 自然 C128，所有候选都可以成为输出。不插入正确答案，不限于前两名。
- 直接复用已独立验收的五个 seed0 B_CAL，分别绑定内层拟合与外层重训的原头、归一化和划分。原 M0、基线参数不更新。
- 新增三臂 × 五折 × seed0，共 15 项 reader 训练。采用原完整候选身份损失、AdamW 和 TRAIN 内选择规则；若内层没有符合规则的改进，epoch0 精确回退基线。

| 臂 | 读取内容 | 比较方式 |
|---|---|---|
| TOKEN_QR | 原 query token、完整 reference 自由 MaxSim 对应内容、逐位置支持和匹配歧义 | 每个候选独立读出，再相对 RAW anchor |
| TOKEN_QRR_ANCHOR | 与 TOKEN_QR 同源局部输入 | 所有候选分别与 RAW anchor 比较 |
| TOKEN_QRR_MULTI | 与 ANCHOR 完全相同的输入与参数结构 | RAW anchor，加基线最高分的非自身、非 anchor 竞争者 |

MULTI 的第二竞争者明确排除自身和 RAW，避免 RAW 排名最高时退化回单一 anchor。边去重并按有效边数平均；分数并列按固定物理图库 ID 处理。这是通过同一个 query token 轴比较候选解释，不是重新计算原生 RoMa R–R 匹配。

最终分数为 `s_g = s0_g + D_g - D_RAW`。D 的末层零初始化，原答案保持 HOLD=0；继续沿用原切换规则。新增判别量 D 不重新命名成 M，不预设它是 M 的唯一因果解释。

## 已提交任务

| 阶段 | Job ID | 分区与并行 | 前置依赖 |
|---|---|---|---|
| 首图真实工程验证：缓存、独立核验、三臂梯度 | 5167551 | dev_cpuonly，已完成，退出码 0:0，耗时 1分31秒 | 无 |
| 128 张局部证据构建 | 5167552_[0-15%16] | cpuonly，16 片，每片 8 张 | 5167551 成功 |
| 全缓存独立验收 | 5167553 | cpuonly / dev_cpuonly | 全部 5167552 成功 |
| 三臂五折训练 | 5167554_[0-14%15] | cpuonly，15 并行上限 | 5167553 成功 |
| 逐候选独立重放与汇总 | 5167555 | cpuonly / dev_cpuonly | 全部 5167554 成功 |

每项请求 8 CPU、32GB、10 分钟、0 GPU。程序工作预算为 480 秒，预留保存时间；需要续跑时返回 75，由相同 Job ID requeue，最多 96 次。10 分钟是每次调度运行上限，不是全链预计总耗时。验证与最终预测也保存逐图断点。

全链复用现有编码器 tokens 与 RoMa u/v；没有新增编码器或 RoMa 前向。128 张是本轮 CPU 证据重组规模，不是新 GPU 采样规模。既有 H593 采集链继续独立运行，未修改。

实际 Slurm 字段已核对：资源、时限、数组范围、四条 afterok 依赖均正确；五份提交到调度器的 batch 脚本与本地文件逐字节相同。核查时首图为 RUNNING，其余均为正常 Dependency。

随后首图三臂已全部通过，`pilot_validation.json` 状态为 `TOKEN_COMPETITION_REAL_PILOT_ALL3_PASS`。Slurm 确认 5167551 为 COMPLETED / 0:0，stderr 为空。缓存数组 5167552 已自动解除依赖，目前因 Priority 排队；其余三段继续等待各自前置成功。最新快照见 `startup_validation.json`。

## 提交前验证

- 模型 38 项工程检查通过：零残差精确复现、完整 C128 可选、候选置换、反对称消息、有效 mask、实际梯度、优化器与 RNG 精确恢复等。
- 实际 launcher 的 12 项分支检查通过，包括超时/75 续跑、数组元素 ID、最大恢复次数及非恢复错误传播。
- 18 个 Python 源码编译和 6 个核心模块导入通过。
- 训练—中断—恢复—独立核算闭环通过：内层验证完成第 1 张后强制退出 75，恢复后训练更新数仍为 71，没有重复训练；独立重放误差为 0。该检查使用合成局部输入，只是工程证据。
- 真实源缓存预检通过：协议、代码、图像/配对缓存、自然 C128 与 common 轴的绑定一致。正式训练必须同时通过 manifest 和独立验收回执。
- 首张真实局部缓存已经构建完毕：720 个原 query tokens × 128 个候选。日志记录该构建步骤 6.78 秒，不包含整个 pilot 的初始化、独立验证或训练时间。
- 首图真实三臂均验证零初始化分数精确等于原 B_CAL、头参数不变、三步训练梯度有效。QR 每步约 0.19–0.20 秒、ANCHOR 约 0.35–0.37 秒、MULTI 约 0.63–0.65 秒；这只是该图工程计时，不外推完整训练总耗时。

## 数据与后续结果位置

统一结果目录：`results/rc_token_competition_f128_v2/`。

- `protocol.json`：冻结协议；SHA256 `f7b9349ca2e268de497a5a3693beefd3ab3c7e57a51478e384da952761987143`。
- `submission.json`、`scheduler_verification.json`、`submitted_spool/`：任务链与实际调度核查。
- `candidate_parts/`、`evidence/`：每候选原 token 匹配内容、位置索引、u/v 派生量、有效标记、原 FP64 M0 和来源 SHA；可复用与恢复。
- `fold*/seed0/*/`：模型、优化器/RNG 断点、TRAIN 内选择、逐候选分数。
- 完成后汇总器生成 `metrics.csv`、`paired_vs_B_CAL.csv`、`paired_structural.csv`、`candidate_predictions.json`、`trace_manifest.json` 和 `validation.json`。

必须等 15 项训练及独立重放全部通过后，才能报告新臂的 F128 正确数、救回/误伤和多竞争者相对单 anchor 的效果。首图、合成检查与成功提交都不等于科学验证成功。
