# 最新交付：new HYP展示材料完成

2026-09-15。已生成15页PPTX/PDF、中文逐页讲义与完整Word/HTML讲义、Excel/CSV统计表、PNG/PDF/SVG图及真实图片案例。已核对数据与图页；未新增训练或模型推理。交付入口：[展示材料说明](REPORT_NEW_HYP_SHOWCASE_DELIVERY_V1_20260915.md)。当前论文范围仍按2026-09-15收口文件，ownership、完整过目不忘和新编码器列为未来工作。

---

# 当前范围：new HYP论文主线收口，ownership及新编码器作为未来工作

2026-09-15。用户明确决定将当前已验证的new HYP机制、训练改进和跨数据集收益作为论文贡献收口；ownership保留为独立扩展，完整过目不忘系统及新的可训练token编码器列入未来目标。ColNomic作为本轮已验证实现保留，冻结表示的适配和表达局限写入limitations，不宣称其是所有剩余错误的已证实唯一根因。

当前写作范围以 [收口文件](../plan/RC_NEW_HYP_PAPER_SCOPE_FREEZE_V1_20260915.md) 为准。GroZi/RPC保持已封存外部确认，ISIC保持已打开队列跨域探索；旧panel和历史NO-GO不改写。COST1主模型、CE次模型角色不变。下一阶段为论文与复现材料整理；未来方向不自动启动新训练或任务。本范围决定不表示投稿材料已全部完成。

---

# 最新完成：ISIC537跨域探索通过，独立复核PASS

2026-09-14 Europe/Berlin。恢复汇总5145565已COMPLETED/0:0（35秒）。537张query、390-reference独立gallery、346患者；复用冻结H593商品头、ColNomic/RoMa、自然RAW C128、全部127 challenger及原HOLD/SWITCH。零ISIC训练、微调或阈值选择。

RAW 466/537；原成本COST4 469/537；GROUP_COST4 470/537；主COST1 500/537（相对RAW 34救0损）；次CE 513/537（47救0损）。COST1预定五项探索比较全部通过；相对RAW的等患者平均增益+5.21个百分点，患者bootstrap95%CI [3.18,7.48]个百分点。COST1_CBIND仅414/537。COST1相对COST4新增31张均已是COST4最高challenger，却被原头HOLD；此机制定位已逐行核实。

证据级别：positive_exploratory_transfer_signal=true；untouched_external_GO_claimed=false。这是已打开、历史选择过的ISIC/IMA++队列，不是全新未触碰医学确认，也不声称疾病诊断、ownership或无文字因果结论。所有模型与数据分母固定；旧28/32、99/128及GroZi/RPC结果不改写。

结果与完整报告：reports/REPORT_NEW_HYP_ISIC_FROZEN_TRANSFER_V1_20260914.md；results/rc_new_hyp_isic_transfer_v1/result.json（SHA256 6af5bfbafd36f9438372b2c5c1fff3e2f154ffa9c9af84000278809b14ea0d46）。result_validation.json与independent_final_audit_v1.json均PASS；completion_receipt_v1.json绑定结果、复核、调度与报告。

原链136任务COMPLETED、5144694_23 TIMEOUT、5144695汇总CANCELLED保留原状；超时片完整产物及额外CPU重放已通过，仅恢复不变汇总。没有待运行/待修复的ISIC任务，不重提推理、不继续监控。下方均为历史过程记录。

---

# 最新ISIC：537预测全部验证完成，恢复汇总5145565

2026-09-14。唯一TIMEOUT分片5144694_23的完整预测及额外CPU重放均通过；原汇总5144695因依赖取消。仅补跑原汇总/独立复核5145565，模型数据统计不改，无重新推理。恢复时查该job及ISIC result.json，不重复启动旧链。详细见 reports/RC_NEW_HYP_ISIC_TRANSFER_WORK_STATE_20260914.md。GroZi/RPC已完成结论保持。

---

# 最新ISIC：首片8/537已验证，剩余67片RAW因Priority排队

2026-09-14。reference390、RAW首片、RoMa首片及固定头全部COMPLETED/0:0且独立验证PASS。剩余5144693数组Priority，5144694/5144695等待正常依赖；无失败，无最终准确率。详细状态见 reports/RC_NEW_HYP_ISIC_TRANSFER_WORK_STATE_20260914.md。原GroZi/RPC完成结论保持。

---

# 当前新增分支：ISIC 固定头探索已提交；GroZi/RPC已完成结论保持

2026-09-14。用户授权“可以做个实验，试试”。ISIC927图/390病灶/346患者，390reference+537query；固定H593商品头，零ISIC训练，原RAW自然C128与127challenger。已打开历史库存，非全新外部医学确认。

任务5144689 reference正常编码；5144690 RAW首片/authority封存，5144691 RoMa首片，5144693剩余RAW，5144694剩余RoMa，5144695统一join与独立复核。所有依赖已验证；大数组accelerated最多46并行，RAW10min/RoMa15min。未有ISIC结果。完整恢复入口 reports/RC_NEW_HYP_ISIC_TRANSFER_WORK_STATE_20260914.md。不要重新启动已完成GroZi/RPC任务，不重复提交ISIC现有链。

---

# 最新：RPC600全部完成并独立复核，第二套外部new HYP联合门通过

2026-09-14 Europe/Berlin。152个有效任务全部COMPLETED/0:0，5143940汇总完成；75片RAW/RoMa/head验证齐全。独立从sealed logits重建6000个动作，复算MRR/计数/所有分组及两类bootstrap，全部通过。最终状态 RPC600_SCOPED_EXTERNAL_NEW_HYP_GO；没有待修复任务，本轮未提交新训练/推理。

同一固定H593头、RPC600张单商品整图、独立200-reference gallery、自然C128、全部127 challenger / 阈值0：RAW192/600，COST4 194，GROUP_COST4 196，主COST1 207，次CE217。主模型相对RAW18救3损、净增15（2.5pp；主95%区间[1.17,3.83]pp），相对GROUP净增11，相对COST4净增13。COST1_CBIND174，正确绑定相对错配净增33；RAW2_CE192。五项主分层SKU门及17品类敏感性区间均正。CE比COST1净增10但区间跨0，继续保持次模型。

RPC绝对准确率34.5%，并不高。C128召回575/600，25召回缺失全留分母；主模型另有368候选内错误（其中80为target最高challenger但被HOLD），没有根据外部错误调阈值/头。camera1 RAW66→75，camera2 31→35，camera3 95→97（各200）。3个RAW原正确损失完整报告，沿用可靠净增标准。

GroZi先前355/480 vs RAW321/480（34救0损）保持且结果SHA复核未变；两套分别通过各自冻结门，竞争空间/抽样/统计不同，不合并分母。共同结论为限定这两套实验的无外部微调、reference注册后的可靠纠错及候选绑定证据；不扩展到ownership、开放集或普遍高识别率。旧32/128/H593 OOF账本未改写。

正式报告 reports/REPORT_NEW_HYP_RPC_EXTERNAL_CONFIRMATION_V1_20260914.md；关键物证 results/rc_new_hyp_rpc_transfer_v1/{result.json,result_validation.json,independent_final_audit_v1.json,final_scheduler_receipt_v1.json,completion_receipt_v1.json}。恢复时勿继续监控或重提已完成5143936–5143940。

---

# 已确认RPC自然推理正常，按用户偏好退出

5143928 reference200完成并验证；5143936 RAW首片8图COMPLETED/0:0，1分52秒，独立验证PASS。5143937 RoMa已初始化并实际完成1张RPC图片的每张128候选计算，无Traceback。5143938/5143939/5143940依赖已逐项核对，剩余74片RAW→RoMa最多并行46，全部75片预测封存后自动汇总600。物证 results/rc_new_hyp_rpc_transfer_v1/submission/startup_confirmation_v1.json。

当前只是运行确认，RPC无准确率/GO；恢复时检查5143937–5143940与result.json/result_validation.json，不重启完成的reference或RAW首片。用户此前要求正常运行后退出，因此停止监控，让整批自动继续。

---

# 最新：RPC600固定模型外部评测链已提交

用户“开始”后已完成独立200-reference gallery适配与全部执行准备。固定H593五个头（主COST1、次CE、COST4/GROUP_COST4/RAW2_CE）不变，RPC600-query/200-reference原清单不变；本轮无训练、阈值或模型选择。

5143928 reference编码已COMPLETED/0:0，4分33秒。原4张TRAIN的image/template tokens、全5413 RAW分数及自然C128逐bit重放通过；200张RPC reference编码和独立CPU验证通过。产物 encoded_references/{legacy_bridge.json,payload.pt,receipt.json,validation.json}。

提交链（全部真实spool逐字节一致）：
- 5143936_[0]：RAW首片8图，10分钟，dev_accelerated/accelerated。
- 5143937_[0]：RoMa+五头及CBIND首片，15分钟，afterok5143936。
- 5143938_[1-74%46]：剩余RAW，每片8图、10分钟，afterok5143937。
- 5143939_[1-74%46]：剩余RoMa+head，每片15分钟，afterok全部5143938。
- 5143940：全部75片预测封存后一次join600标签和固定分层统计，10分钟，afterok5143937:5143939。

流水线共75片，两个大阶段各并行46、不叠加成92。独立gallery物理行0..199，自然C128，不追加旧gallery或插target。RAW与RoMa分别使用.venv-colpali与.venv-romav2，TORCH_HOME与数值循环固定。合成候选/tie/全200排序、head评分AST不变、独立分层bootstrap、三个shell分支和双runtime检查全部通过。最初本地preflight未初始化torch线程而报1≠8，补上真实入口相同初始化后完全通过；不是自然任务失败，未修改模型/推理数值。

统计：200SKU每个3相机query一起重采样，在17 supercategory内保持每层SKU数量，100000次、seed20260913；主COST1对RAW/COST4/GROUP_COST4/RAW2_CE/自身CBIND五项可靠净增联合门。另报17品类等组bootstrap敏感性、相机/品类/SKU、MRR、C128召回和动作组成。全600预测封存前不查看首片准确率。RPC尚无科学结果；GroZi已通过结论保持。

执行方案 plan/RC_NEW_HYP_RPC_INFERENCE_EXECUTION_V1_20260913.md；authority registry/rc_new_hyp_rpc_inference_authority_v1_20260913.json；源码 programs/{encode_rc_new_hyp_rpc_references_v1.py,run_rc_new_hyp_rpc_inference_v1.py,analyze_rc_new_hyp_rpc_external_v1.py,preflight_rc_new_hyp_rpc_inference_v1.py}。提交物证 results/rc_new_hyp_rpc_transfer_v1/submission/inference_chain_receipt_v1.json；最新运行确认应查 startup_confirmation_v1.json（尚未生成时不要声称已确认正常）。后续RPC结果入口 result.json/result_validation.json。继续保护D1-MI/formal392；不读取GroZi结果指导RPC头。

沿用用户“任务运行正常后退出”偏好：确认实际自然计算与依赖后退出，整批自动执行。

---

# 最新补充：RPC800数据资格完成，GroZi外部结论已归档

RPC全部800张静态图片（600 query、200 reference）已下载并独立验证通过：byte与EXIF-RGB指纹匹配、内部无精确重复，与原987-query ledger、5413 gallery及GroZi600无所检验的精确重合。物证 results/rc_new_hyp_rpc_transfer_v1/image_intake_validation_v1.json，状态 RPC800_IMAGE_INTAKE_PASS。下载与验证进程均已结束；RPC模型推理尚未开始，下一步为其独立200-reference gallery适配与固定头评测，保持原600-query协议。

GroZi120主COST1的外部new HYP联合门已完成且独立复核通过（355/480，RAW321/480，原配方COST4 326/480、GROUP_COST4 329/480）。正式报告 reports/REPORT_NEW_HYP_GROZI120_EXTERNAL_CONFIRMATION_V1_20260913.md；结果图及PDF/SVG/CSV在 reports/figures/new_hyp_grozi120_external_v1/。本轮所有执行均已结束；不要将旧的“首片运行中”历史段落当作当前状态。

---

# 最新：GroZi480 外部 new HYP 联合门通过；RPC800 已全部收齐

GroZi有效121个任务全部COMPLETED/0:0；60片RAW、60片RoMa、60片固定head预测及一次完整480标签join全部完成。新独立验证器从sealed logits重建4800个动作，独立重排MRR、计数及27视频组bootstrap，均通过。最终状态：`GROZI120_SCOPED_EXTERNAL_NEW_HYP_GO`。

同一固定H593训练头、同一GroZi480、120新reference追加到旧5413 gallery物理行、自然C128、全部127 challenger / 阈值0 HOLD-SWITCH：RAW321/480，COST4 326，GROUP_COST4 329，主COST1 355，次CE367。COST1相对RAW为34救0损，相对GROUP为26救0损，相对COST4为29救0损。COST1_CBIND313，错配后原34次救回全部消失并破坏8张；RAW2_CE321。五项预定主联合比较的query净增、等视频组差与95%视频组区间下界全部为正。CE比COST1净增12，但视频区间跨0，仍保留COST1为唯一主模型。

结论范围：固定模型只经reference注册，在这套GroZi商品裁图身份检索上取得可迁移的可靠净增，支持候选绑定的联合证据与检索训练成本校准。不是整货架定位、ownership、跨商店或完整过目不忘的证明。旧28/32、99/128与H593 OOF账本未改。C128召回410/480：70张召回缺失保留分母，COST1另有55张候选内错误。

正式报告：reports/REPORT_NEW_HYP_GROZI120_EXTERNAL_CONFIRMATION_V1_20260913.md。
关键物证：results/rc_new_hyp_grozi120_external_v1/{result.json,result_validation.json,independent_final_audit_v1.json,final_scheduler_receipt_v1.json,completion_receipt_v1.json}。原失败任务5143668与V2启动环境修复记录保留，勿重启完成链。

RPC：2026-09-13T09:37UTC单次端点探测恢复HTTP200；使用原下载程序和不变800图清单续传成功（600 query＋200 reference），零下载错误。download_receipt.json已完整封存。当前正在运行 programs/verify_rc_new_hyp_rpc_image_intake_v1.py 做全图byte/EXIF-RGB核对、内部重复、原987-query/5413-gallery与GroZi600重合检查；最终看 image_intake_validation_v1.json。RPC尚无模型推理或准确率，不因GroZi成功省略RPC检验。

RPC下一执行缺口：适配独立200-reference gallery（计划明确不追加旧gallery），复用已固定的五个H593头、原编码/RoMa数值算子与正确的双Python环境；600条query自然C128，按200SKU、17 supercategory分层bootstrap，与GroZi27视频分组不同。不得重新训练、选择CE替代主头、改抽样或根据GroZi错误修改头。

---

# 运行已确认：修复任务正在实际处理图片，按用户要求退出

5143672 已在 dev_accelerated / hkn0403 运行；RoMa 初始化成功，日志确认 GZ120-Q-0000 的全部128候选已计算完成，无 Traceback。5143673–5143675 的 afterok 依赖已逐项核对。证据：results/rc_new_hyp_grozi120_external_v1/submission/runtime_repair_v2/startup_confirmation.json。

这只是实际运行进度确认，首片尚未完成全部8图和CPU重放，整批仍无准确率或外部GO。用户要求“任务运行正常后退出”，因此停止监控，后续链自动继续。恢复时先读该 startup_confirmation，再检查新任务链，不重提旧失败任务。

---

# 最新：GroZi 启动环境修复已重提，等待首片运行确认

5143667 RAW 首片完成，8 query 自然 C128 及 CPU 验证通过。5143668 在模型加载前报 ROMA_RUNTIME_PREFIX：旧启动器给 RoMa 使用了 .venv-colpali，原冻结 profile 要求 .venv-romav2。5143669–5143671 因依赖失败自动取消；尚未产生 GroZi RoMa 预测或准确率。

已新增 V2 启动器与独立修复 authority，只按阶段选择原有 Python 环境。原 V1 worker、原 RoMa 数值循环、head、数据、评分及 join 程序均未修改。RoMa 环境的 Python/torch/torchvision/Pillow/numpy/TORCH_HOME 精确检查通过；两个 shell 分支使用外部 timeout stub 验证解释器与参数；实际提交 spool 逐字节一致。

修复链：5143672 RoMa 首片（15分钟，dev_accelerated/accelerated）；5143673 RAW 1–59（10分钟，并行46，afterok首片）；5143674 RoMa 1–59（15分钟，并行46，afterok剩余RAW）；5143675 全480封存后汇总。复用已完成的 RAW 首片。证据在 results/rc_new_hyp_grozi120_external_v1/submission/runtime_repair_v2/chain_receipt.json，方案为 plan/RC_NEW_HYP_GROZI120_RUNTIME_LAUNCH_REPAIR_V2_20260913.md。

用户最新要求：确认任务运行正常后退出，不等待整批结果。GroZi 输入为480张静态商品裁图＋120张reference；27个视频编号只用于来源分组。RPC仍为固定600 query＋200 reference静态图；下载325/800后遇HTTP429，暂无后台下载。当前无外部GO。

---

# 最新：GroZi-120已获准，600张图片就绪，完整480评测链已提交

用户已明确允许GroZi只读外部确认，上一节历史记录中的“待解除封存”状态已结束。模型输入都是静态图片：GroZi店内商品PNG原本来自视频；视频编号仅作来源分组。RPC同样是静态照片。

GroZi两个官方原档SHA与2026-09-05下载记录一致。商品82的坐标文件和info精确重复了两遍，派生时只消除重复块，原档未动。新协议与清单已在图像解码/推理前固定：每个商品一个reference、一个来源视频中的4张分散帧图片，共120身份、480 query、120 reference、27来源视频组。600图均已提取，内部无字节或RGB像素精确重复，与原987-query公开ledger及5413 gallery无字节重合；语义SKU同一性和预训练暴露不据此宣称完全排除。

数据与程序：`plan/RC_NEW_HYP_GROZI120_FROZEN_DATA_V1_20260913.md`，`programs/prepare_rc_new_hyp_grozi120_external_v1.py`，`results/rc_new_hyp_grozi120_external_v1/{metadata_validation.json,image_receipt.json,historical_byte_overlap_validation.json}`。

**5143664已COMPLETED/0:0，5分41秒**：4张旧TRAIN的image/template token、所有5413 RAW分数与自然C128全bit一致；120张GroZi reference编码及CPU完整性核算通过。入口`encoded_references/validation.json`。原固定头不需再次训练。

**完整推理链已经提交（工程片通过后自动展开，不在片0查看准确率）**：
- 5143667_[0]：RAW片0，10分钟，dev_accelerated/accelerated。
- 5143668_[0]：afterok前者，RoMa+全部固定head片0，15分钟。
- 5143669_[1-59%46]：afterok完整工程片，剩余RAW，10分钟。
- 5143670_[1-59%46]：afterok剩余全部RAW，剩余RoMa+head，15分钟。
- 5143671：afterok两个RoMa组，60片预测先封存后一次join标签并核算视频组区间，10分钟。

每片8 query，两大阶段不叠加成92并行。依赖失效自动停止后续。真实脚本spool都逐字节核对，物证`submission/inference_chain_receipt.json`及各任务initial_scheduler.txt。执行方案`plan/RC_NEW_HYP_GROZI120_INFERENCE_EXECUTION_V1_20260913.md`；authority `registry/rc_new_hyp_grozi120_inference_authority_v1_20260913.json`；程序`run_rc_new_hyp_grozi120_inference_v1.py`、`analyze_rc_new_hyp_grozi120_external_v1.py`。

当前还没有GroZi准确率或外部GO。最终消费者应生成`results/rc_new_hyp_grozi120_external_v1/result.json`和`result_validation.json`。主COST1、次CE与同协议基线及CBIND；GroZi零训练零调参，自然C128。旧gallery5413物理行保留并追加120 reference，不将120商品冒充128候选。

RPC原固定800图清单与成功下载325张保留，475张因官方HTTP429尚缺，当前无下载进程。不要根据GroZi未来效果隐去RPC。用户明确没有额外药盒照片，不再安排补拍。当前GroZi以既有单商品裁图做身份检索，不能写成整张货架无框定位或跨商店证明。

---

# new HYP 外部确认恢复入口：2026-09-13

## 用户最新选择

先用公开商品数据验证跨数据集效果。用户随后明确“没有药盒照片了，就仓库里这么多”，因此取消未来补拍安排，不以不存在的新药盒照片阻塞公开数据路线。随后问“Grozi不能用吗”；已完成公开文献和原保护协议核对，准备独立GroZi使用方案，询问是否解除此前对本地GroZi-120的专门封存。未收到该明确回复前，不访问该受保护目录。

## 已完成：五个真正固定的全开发集头

`results/rc_new_hyp_external_head_freeze_v1/bundle.json`；`completion_receipt.json`独立确认全部头文件SHA、验证SHA、570训练/23召回缺失记录边界和实际Slurm完成状态。

- COST1主模型，CE次模型；基线COST4、GROUP_COST4、RAW2_CE及无头RAW。
- 全部采用已经开放H593的原顺序和合格缓存，仅570条recall-present记录训练，全部2000步、零初始化、AdamW .03/.001，FP64原六统计/七参数（RAW2为两参数）。没有RPC训练、没有部署替换。
- 五头均独立进程从零重训到参数/所有REAL与CBIND logits逐bit一致；逐头NumPy检查150622个logit及全部动作，最大误差1.07e-14，全部通过。
- OOF的481/593、486/593不是这些全数据头的外部成绩。没有重新评分旧28/32、99/128来选头；外部推理次数仍0。
- 有效作业：5143647_1..4、5143652_0、5143653汇总，全部COMPLETED/0:0，拟合每项约一分钟、汇总36秒。首个5143647_0因启动器模拟检查意外产生的本地文件占用而FAILED；本地产物原样隔离到incidents，原配方重新提交5143652_0通过。无头或训练方法改动。旧受阻汇总5143648取消；提交spool逐字节相同，修复物证保留。

固定规则：[RC_NEW_HYP_EXTERNAL_HEAD_FREEZE_V1_20260913.md](../plan/RC_NEW_HYP_EXTERNAL_HEAD_FREEZE_V1_20260913.md)。主模型要求可靠净增、完整报告救回/损失，不恢复零损失门；CE不得看外部结果后替换主模型。new HYP另外检验正确绑定相对CBIND和RAW2对照。

## RPC：清单已固定，下载被服务端限流

计划：[RC_NEW_HYP_RPC_TRANSFER_V1_20260913.md](../plan/RC_NEW_HYP_RPC_TRANSFER_V1_20260913.md)。官方目录及单商品原始metadata已收到并记录SHA；源JSON空间字段不导出给检索worker。

`results/rc_new_hyp_rpc_transfer_v1/metadata_validation.json`：200 SKU、17 supercategory；camera0每SKU一张reference，camera1/2/3各一张query，共200 reference+600 query。只按元数据/固定hash选择，既没有观察图片挑样，也没有看模型成绩。worker、gallery、curator、download四份清单均固定。

运行`programs/download_rc_new_hyp_rpc_transfer_v1.py --workers 8`收取原始公开整图。已成功接收325/800张，字节SHA和EXIF转正RGB SHA逐张保存在downloads。之后出现连续失败，一次独立探测确认HTTP429 Too Many Requests；下载进程已停止，未继续重试轰击服务端。`download_interruption.json`记录325成功、475缺失。原清单不变，不能把已下载子集拿来替代600-query确认集。当前无下载后台任务。

可待服务限流解除后对原清单续传，或采用具有可验证原始文件同一性的公开原始归档；不能因下载中断换抽样。当前没有读取图片做人工目视、编码、RoMa或评分。全部800图接收完后仍需像素/跨集精确重复核对。

## GroZi：可以设计外部检验，专门封存尚待解除

方案：[RC_NEW_HYP_GROZI_EXTERNAL_INTAKE_PROPOSAL_V1_20260913.md](../plan/RC_NEW_HYP_GROZI_EXTERNAL_INTAKE_PROPOSAL_V1_20260913.md)。已读原作者论文，GroZi-120有web reference与商店视频商品裁图，适合跨域身份识别。现成裁图评测不否定retrieval-only训练，但不能说成整张货架无框定位。120身份不足单独组成去重C128，拟保留原gallery干扰库再追加新reference；按商品/视频处理帧相关性。不得从多个数据集成绩中挑好的发布。

只查过已有RC报告、协议与公开文献；未列举或读取`external/_staging/grozi120_v1`、DO_NOT_CONSUME或其数据。此前用户明确保护该路径，原合同第14节也记录禁止列举/读取/哈希/解压/使用；已通过异步问题申请仅用于只读外部确认的明确范围变更，不涉及D1-MI和formal392。

## 后续执行的确切缺口

1. 用户允许GroZi后先只读封存说明、版本与既往使用清单，确认是未触碰外部数据还是已开发数据，再在任何图片推理前固定抽样与分组。无需再重训五个最终头。
2. RPC限流解除时可以对原800图清单续传，不必重做元数据或改选样本。
3. 旧RAW入口硬编码5413物理行、5412身份，新gallery适配器尚未完成。可复用`materialize_rc_original7_train128_token_raw_v1.py`的encode_query与`run_romav2_colnomic_sealed_source_e0_v2.py`的full_gallery_scores/top128_rows，及`materialize_rc_new_hyp593_inputs_v1.py`编译的原RoMa数值循环。注意RAW原算子为FP32 MaxSim求和后转FP64分数；必须保持此行为，不把“head/C4 FP64”误改成RAW全FP64。参考图的完整sequence embedding与local image tokens需分别保留。
4. 新feature接口继续调用原FC.candidate_feature，CBIND是物理位置轴固定循环移位64并保留RAW差，不读query target选择错配。五个head/全部外部预测先封存，再join标签与计算组置信区间。原输入重放及新gallery全索引验证须完成后才提交自然推理。

现在完成的是模型冻结和外部数据准备的一部分，没有外部new HYP GO。
