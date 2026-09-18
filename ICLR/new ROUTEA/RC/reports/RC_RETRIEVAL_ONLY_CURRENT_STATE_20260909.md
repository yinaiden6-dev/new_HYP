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

# 最新2026-09-13：GroZi480外部图片检验已提交

用户已明确解除GroZi-120用于本次只读外部确认的封存。480张query图片+120张reference已就绪，按27来源视频分组；输入为静态商品图，不运行视频模型。五个全H593固定头全部完成；5143664 reference编码/旧TRAIN bit重放通过。完整推理链5143667→5143668→5143669→5143670→5143671已提交，最多46并行，RAW10分钟/RoMa15分钟。尚无外部准确率/GO。恢复入口：[RC_NEW_HYP_EXTERNAL_CONFIRMATION_WORK_STATE_20260913.md](RC_NEW_HYP_EXTERNAL_CONFIRMATION_WORK_STATE_20260913.md)。RPC已下325/800图后HTTP429停止，清单不变；不再安排补拍药盒。

---

# 2026-09-13：外部确认前五个固定头完成，公开数据准备中

恢复入口：[RC_NEW_HYP_EXTERNAL_CONFIRMATION_WORK_STATE_20260913.md](RC_NEW_HYP_EXTERNAL_CONFIRMATION_WORK_STATE_20260913.md)。COST1/CE/COST4/GROUP_COST4/RAW2_CE全H593固定头均通过Slurm从零重训及NumPy验证；有效拟合与汇总全部完成。主COST1、次CE，外部得分仍为空。RPC已冻结200 reference+600 query清单，收到325/800图后官方HTTP429，下载已停止并保留全部成功文件。用户不再补拍药盒，先走公开商品；询问GroZi可否使用，独立方案已准备，原专门封存的数据读取范围仍等待明确回复。D1-MI及formal392未开放。旧28/32、99/128未重测，H593 OOF481/486不冒充固定头的外部结果。

---

# 2026-09-13：六项本轮检验全部完成，主要可修瓶颈是保守动作成本

全部84子任务COMPLETED/0:0，16缓存、24桥接拟合、35覆盖拟合、5成本/绑定拟合及三个汇总均通过对应独立验证。本轮无待运行任务。

同H593原五折/冻结RAW C128/七参数、完整127 challenger且阈值仍0：RAW426，旧COST4 440，旧强GROUP_BASE447，新COST1 481，新CE486。COST4→COST1为44救3损、净+41；COST1→CE为14救9损、净+5，后一步等组区间跨0。相对447强基线，COST1是36救2损，CE是49救10损。

CE救回旧COST4的57张全部已经是旧头最强challenger，只是被HOLD拒绝；11个损失全部是正确RAW/HOLD被错误切换。RAW两参数CE仍426，冻结CE在CBIND上降到281。因此本轮提升主要支持任务成本与动作选择失配，并且仍依赖正确reference绑定，不是先加大头或换token才能突破。

TRAIN128原四折桥接：原六统计线性CE114；FREE113、POOL112、MATCH110、CHANNEL74。真实/错配CHANNEL训练都全对、留出却下降，不能将拟合成功当作身份机制成功。H593等128训练预算CE达到482/484/486，数量本身不解释旧440停滞；同身份增加视角存在收益但非持续单调。

当前CE107错中23个target缺C128、84个候选内错误。全部593已开发，无未触碰确认库存；六项已完成当前授权数据上的有限检验，不等于所有照片/token充分性或外部泛化都已证明。原固定28/32、99/128没有重测、没有拼账或部署替换，也没有new HYP external GO。

完整报告：[REPORT_SIX_CAUSE_ISOLATION_V1_20260913.md](REPORT_SIX_CAUSE_ISOLATION_V1_20260913.md)。机器分析results/rc_six_cause_isolation_v1/analysis.json及analysis_validation.json；可分享图six_cause_summary.png/pdf；最终调度物证completion_scheduler_receipt.json。

---

# 2026-09-13：六项隔离首轮完成，H593同七参数CE达到486/593

用户要求六项全部推进并再次继续。inventory、20错例视觉记录、16片信息桥接、24项TRAIN128输入/读取器/目标对照、35项H593数量/覆盖/目标对照均完成并验证。H593固定五折/RAW C128：RAW426→原COST4 440→CE486（57救11损）；相对GROUP_BASE447为49救10损。TRAIN128桥接：S6线性CE114、FREE113、POOL112、MATCH110、CHANNEL74；高维头训练全对但留出恶化。原28/32、99/128本轮没有重测，不能拼账。

第4项尚需拆开成本比与损失形式，已提交最小补充5143639五折、5143640自动汇总：同数据COST1、RAW两参数CE，以及冻结486头CBIND零更新推理。恢复入口reports/RC_SIX_CAUSE_ISOLATION_WORK_STATE_20260913.md；全部新产物results/rc_six_cause_isolation_v1/。全部593已开发，无未触碰确认库存；不能宣称new HYP外部GO。

---

# 2026-09-12：同CE进一步优化完成，三头逐图决策全部不变

5142672四折及5142673汇总全部COMPLETED/0:0。原TRAIN128四折、RAW C128/127 challenger：MEAN109→109、CURVE111→111、JOINT110→110，三头所有128张图selected reference与target rank逐项不变。97,536 logits独立核算及精确概率/区间损失界重放通过。原RAW86、BASE108、GLOBAL114保持。

12项中11项达到预设盒内剩余界≤1e-6，JOINT第1折为3.219e-6，不能写成12项全过。旧参数相对固定[-64,64]盒最优CE的差距上界最大6.7053e-5；继续优化相同输入/函数类/数据损失没有带来新决策。该结论不是准确率上限，也不等同继续原AdamW衰减目标。JOINT仍对同协议CURVE0救1损；两折虽训练CE更低，留出CE反而更差，未显示稳定跨组优势。

当前全部任务完成，无待运行任务。没有旧EVAL重测：原ec7仍28/32、99/128。完整分析reports/REPORT_PAIRED_CE_OPTIMIZATION_ISOLATION_V1_20260912.md；产物results/rc_paired_ce_optimization_isolation_v1/{result.json,result_validation.json,completion_analysis.json}。尚不能将输入缺信息、函数类/读取方式、目标或组分布指定为唯一根因，本轮不自动追加新模型。

---

# 2026-09-12：5142672四折＋5142673汇总，隔离新小头的拟合与泛化

启动复核：5142672四折已在accelerated同时RUNNING，错误日志为空；5142673正常等待完整四折成功。不存在成功的分区迁移或训练程序改动。

依据用户jiu/继续，保留上一轮全部输入、原TRAIN128四折、BASE7、MEAN2/CURVE3/JOINT3函数类，只对无额外正则FULL-C128 CE数据损失做预先固定[-64,64]参数盒内优化。不是新特征或旧EVAL晋级，也不等同继续原AdamW权重衰减过程。L-BFGS-B只提供搜索点，Fraction＋50位区间运算重建全盒损失界，fresh重放证书并独立核对全部损失/梯度/动作。

合成验证通过；5142672_[0-3%4]已提交accelerated，每折10分钟；5142673已提交dev_accelerated,accelerated/10分钟，完整数组afterok依赖。实际spool逐字节一致。当前没有新准确率。目标是分清“数据损失还有多少可拟合空间”和“降低它能否换来跨组收益”，不把solver成功当成模型成功。

恢复入口reports/RC_PAIRED_CE_OPTIMIZATION_ISOLATION_WORK_STATE_20260912.md；产物results/rc_paired_ce_optimization_isolation_v1/；authority registry/rc_paired_ce_optimization_isolation_authority_v1_20260912.json。上一轮TRAIN仍JOINT110、CURVE111、GLOBAL114；旧原ec7 EVAL仍28/32、99/128。

---

# 2026-09-12：同位置候选内容差完成110/128，未超过同容量111和既有114

5142654完成0:0/41秒，仅修复5412身份排序断言，原训练和预测完整保留。原TRAIN128/32身份/32组四折、同RAW C128与127-challenger动作：RAW86、BASE108、BIAS109、MEAN109、CURVE111、JOINT110、GLOBAL114。JOINT对BASE6救4损净+2，对同容量CURVE0救1损，对GLOBAL1救5损；未通过预定全部对照screen，不能称new HYP突破。

全部16缓存、四折fresh拟合重放、81,280 logits独立核算及RAW/BASE/GLOBAL逐图旧结果一致性检查完成。数学信息缺口并未转为本统计的独立检索优势；TRAIN内m2由平均差及其曲率可线性解释97.6%—98.6%方差，这不是身份信息比例或统一失败根因。主臂对BASE的组区间跨0。旧EVAL不重测、不替换：原ec7仍28/32、99/128，LISTWISE26/32、101/128，GLOBAL_T12827/32、98/128。

本轮全部完成，无待运行任务。报告reports/REPORT_PAIRED_LOCAL_EVIDENCE_OOF4_V1_20260912.md；结果results/rc_paired_local_evidence_oof4_v1/result.json、result_validation.json、join_repair_receipt.json、completion_readout.json。当前不追加同类标量参数变种。

---

# 2026-09-12：同位置候选内容差训练全部完成，5142654修复汇总身份轴断言

16缓存和5142612四折均完成0:0，四折fresh重放及81,280 logits独立核算通过。5142613汇总把5412个去重身份误写成5413物理图，止于FULL_GALLERY_ACTION_RANKING；属于汇总代码错误，还不能据此判科学成败。

已新增join-only修复作业5142654，dev_accelerated,accelerated/10分钟。保持原训练、所有预测、程序和authority，增加完整5412身份覆盖核对，仅重跑汇总。恢复入口reports/RC_PAIRED_LOCAL_EVIDENCE_OOF4_WORK_STATE_20260912.md；完成标志results/rc_paired_local_evidence_oof4_v1/result_validation.json与join_repair_receipt.json。

---

# 2026-09-12：同位置候选内容差检验已提交5142610—5142613

提交后复核：首片5142610_0已COMPLETED/0:0，Slurm运行1分23秒，独立验证误差≤9.99e-16；其余15缓存分片全部RUNNING，四折及汇总正常等待完整上游数组成功。

新授权有界检验已落到任务：首片5142610、其余15片5142611、四折5142612、自动汇总5142613，每项10分钟。完整RAW/reference tokens与RoMa visibility均存在，纠正此前缺少reference tokens的误判。首片日志已完成8图/1024候选的计算及fresh NumPy重算，约47秒阶段计时；其余阶段按完整数组afterok依赖自动推进。

原TRAIN128/32身份/32组原四折；旧BASE7不重训。主臂JOINT3读取同query位置candidate−RAW winner内容差的有符号二阶量，与同参数量CURVE3、MEAN2、BIAS1对照，并复用既有GLOBAL7 114/128为强对照。全部新臂同FULL-C128 CE、FP64、2000步、物理127 challenger、零阈值HOLD/SWITCH，允许损失要求净增。无自然新准确率或HYP GO；原EVAL32/128不在本轮读取范围。

完整恢复入口：reports/RC_PAIRED_LOCAL_EVIDENCE_OOF4_WORK_STATE_20260912.md。有效authority为registry/rc_paired_local_evidence_oof4_authority_v2_20260912.json；结果目录results/rc_paired_local_evidence_oof4_v1/。四折验证后自动产生result.json/result_validation.json/report.md；提交状态不能替代该结果。

---

# 2026-09-12：GLOBAL7统一微调原面板结果27/32、98/128，TRAIN净增未迁移

5142580在dev_accelerated完成0:0/65秒，原ec7冻结＋全TRAIN128七维残差，沿用前轮GLOBAL7配方。fresh参数重放、121920 logits/动作/完整排名、训练评价六项隔离检查均通过，最大NumPy误差7.11e-15；原模型参数/预测/逐图结果保持一致。

同RAW C128/完整127 challenger：EVAL32 RAW25→原ec7 28→GLOBAL7_T128 27（0救1损）；EVAL128 RAW88→原ec7 99→GLOBAL7_T128 98（3救4损）。对LISTWISE26/101，新模型分别1救0损和2救5损。本轮两原面板均未超过原头，不通过观察净增要求。

32唯一原正确损失0618。128救回0809、DIFFICULT-0028、0769；损失0130、0140、0403、0280。前轮TRAIN四折108→114保留原口径，本轮表明该配方在两套不同身份/组的固定面板没有重现收益，不能据此单独归因于分布或特征缺失。

原ec7继续单列28/32、99/128；LISTWISE为26/32、101/128；新GLOBAL27/32、98/128，无部署或HYP GO。当前任务全部完成，不自动追加试验，无需监控。

报告reports/REPORT_GLOBAL7_TRAIN128_FIXED_PANELS_V1_20260912.md；结果results/rc_global7_train128_fixed_panels_v1/{result.json,result_validation.json,completion_readout.json}。

---

# 2026-09-12：照片内容小头无净增；统一微调TRAIN留出108→114

完整任务链5142573、5142574四折、5142577均完成0:0；新参数fresh重放及81,280 logits/动作/完整排名/计数独立核算通过。该128是原TRAIN128分组四折（32身份/32组），不是固定EVAL128；原各折BASE直接复用，原头无训练更新。

同RAW C128/127 challenger HOLD-SWITCH：RAW86、BASE7 108、GLOBAL7 114、STATS28 107、QUERY28 108、QUERY_PERM28 107。QUERY主臂8救8损，对GLOBAL2救8损；当前全图token均值PCA3条件化未证明分流有效。仅此粗摘要/配方结论，不排除全部照片条件化。

GLOBAL统一微调副对照有7救1损、净+6，6正组1负组；等组差+4.95pp，bootstrap95%约[+0.68,+9.90]pp，精确双侧组p=.0625。按用户观察净增标准是积极结果，但不能改称照片小头成功或新HYP GO。GLOBAL仍为六特征加bias有效线性模型。相对BASE同时改变FULL监督范围、损失及第二阶段拟合，不能单独归因于CE；四个新臂之间条件匹配。

没有原EVAL重测：原ec7仍28/32、99/128；LISTWISE仍26/32、101/128；不把TRAIN114拼入该账本。GLOBAL配方是下一项同源固定面板检验的候选，但本轮未自动启动。全部当前任务完成，无需继续监控。

结果results/rc_query_content_routing_oof4_v1/{result.json,result_validation.json}；报告reports/REPORT_QUERY_CONTENT_ROUTING_OOF4_V1_20260912.md；完整执行记录reports/RC_QUERY_CONTENT_ROUTING_OOF4_WORK_STATE_20260912.md。

---

# 2026-09-12：照片内容小头检验已提交5142574四折，缓存完成

追加状态：0、1折已完成0:0（69/71秒），fresh重放和独立NumPy核对通过；2、3运行。5142577已迁移至dev_accelerated,accelerated，1GPU队列资源、CPU计算，同ID/10分钟/完整数组afterok不变，避免汇总继续等普通CPU队列。

按用户新继续指令执行TRAIN128/32身份/32组、原四折的有界检验。复用已验证各折BASE7，原头不重训。四个新残差臂GLOBAL7、STATS28、QUERY28、QUERY_PERM28共享FULL-C128交叉熵、FP64和2000步；QUERY主臂用图像token均值的TRAIN内PCA3调节原六证据权重。与统一权重、旧统计输入及同容量打乱query内容比较，不使用人工照片类型标注。

缓存5142573已在dev_accelerated完成0:0/52秒，源token/grid/SHA与新进程NumPy均值核对通过。四折5142574_[0-3%4]已提交dev_accelerated,accelerated，每折10分钟；提交核对时0、1运行，2、3待排程。汇总5142577在cpuonly，10分钟，完整数组afterok:5142574_*。开发队列四个提交名额已被四折占用，故汇总暂用普通CPU。三个spool逐字节一致。

尚无新准确率。此128为TRAIN分组留出，不是旧99/128的EVAL；不自动进入旧EVAL、其他593或部署。观察净增允许损失，组不确定性另报。粗内容摘要不等于读懂规格数字；候选细粒度内容读取仍未实现。

恢复入口reports/RC_QUERY_CONTENT_ROUTING_OOF4_WORK_STATE_20260912.md；输出results/rc_query_content_routing_oof4_v1/，后续看fold00..03独立验证及result.json/result_validation.json/report.md。

---

# 2026-09-12：全部14张翻转图完成目视检查；照片类型与候选区分线索需分开

用户提出按具体照片错误类型定位，并考虑检索前照片类型小头。已检查ORIGINAL7→LISTWISE_UNIT1全部14张正确性翻转query及26张不同reference原图；原图40个SHA、完整翻转覆盖和动作计数通过核对。结果已知的事后诊断，非盲审；视觉解释不是独立因果验证，禁止作为训练标签。

7张救回均已在原头中成为最强challenger，但原logit小于0而HOLD；7张损失中6张为正确RAW/HOLD变错误SWITCH，0140为正确SWITCH变错误SWITCH。照片展示面：主品名/规格面2救2损，说明/表格/厂商/条码面5救4损，只有通用品类文字与色块面0救1损。这只是14张翻转子集，不能估计整个32/128/593的类型分布。

具体反例：0818的120 mg清楚可见并救回，P03的24 mg也清楚可见却换成160 mg；0278与0280同一目标不同侧面，分别救回和损失。0669只有BB CREAM SPF20侧面，BEIGE/FAIR名称未展示，参考主要差在颜色。人眼看到线索不等于模型token已保留或实际使用。

旧CONDITIONAL4确实做过，只由RAW差、选点分歧差、reference平均权重差调节dF补偿；其失败不能省略。当前拟讨论的新增条件是照片内容与候选细粒度差异；结构未冻结、未训练、未提交新任务。仅query小头可作前置软路由，但它没有直接观察当前reference差异；与候选条件路由的对照仍待TRAIN分组检验，保持retrieval-only、允许损失但要求净增。当前仍原头28/32、99/128，新头26/32、101/128，无新593结果或HYP GO。

报告reports/REPORT_LISTWISE_FLIP_VISUAL_TYPES_V1_20260912.md；逐图浏览器results/rc_opened_listwise_flip_visual_audit_v1/index.html；原图/分数manifest.json、visual_review.json及validation_and_summary.json同目录。

---

# 2026-09-12：LISTWISE_UNIT1完成；原EVAL128有真实净增99→101，EVAL32下降28→26

Job5142537在dev_accelerated完成0:0，用时1分28秒；新头2000步及fresh参数重放通过，160图三头REAL/CBIND共121920logits、完整图库排名和原对照结果独立验证通过。数据仍原mixed96、六特征七参数、RAW C128。旧原头与unit-cost头未重训，预测直接复用。

| 模型 | EVAL32 | EVAL128 |
| --- | ---: | ---: |
| RAW | 25 | 88 |
| 原ec7 | 28 | 99 |
| 已测unit-cost | 27 | 99 |
| LISTWISE_UNIT1 | 26 | 101 |

对原ec7：32图0救2损；128图7救5损、净增2。对单因素匹配的unit-cost：32图0救1损；128图2救0损、净增2。因此按当前允许损失要求净增标准，EVAL128达到了观察净增；EVAL32未达标，不能概括成全线失败，也不能称跨面板一致突破。

仅由FULL损失替换新增的两例为OUTCOME-0821、OUTCOME-0818，属于同一来源组SOURCE_XML:animal/d34b2325-34e6-4eac-b4ad-d652315c6291.xml；原unit-cost已把target排第一challenger，但logit为−0.109256/−0.054111，LISTWISE变为+0.013473/+0.019323，固定阈值0下从HOLD转正确SWITCH。边际较小、增量集中一组，不据此宣称可靠跨组增益或new HYP独立理论成立。

新模型为26/32、101/128；旧模型为28/32、99/128，两者均保留且不自动替换部署。不能拼旧28和新101为同一头成绩，不能与593五折447直接拼接。来源均历史已打开开发面板；本次没有新593结果或外部确认。

结果results/rc_full_candidate_identity_loss_v1/{result.json,independent_validation.json,completion_readout.json}；报告reports/REPORT_FULL_CANDIDATE_IDENTITY_LOSS_V1_20260912.md。

---

# 2026-09-12：完整候选身份损失单因素检验已提交5142537

用户明确继续/go on，执行已固定LISTWISE_UNIT1。只训练一个新七参数头，保留原PAIR64+FULL32、完整RAW C128、FP64、seed17、AdamW .03/.001及2000步。相对已测TRAIN_UNIT_COST7，仅将FULL损失改为logsumexp([RAW0,127 logits])−target；PAIR仍cost1。没有温度/阈值/epoch选择，旧两头和160图预测直接复用。

合成验证已检查RAW0参与分母、全部127候选梯度、非最大wrong影响loss、target索引以及独立CE的loss/gradient/AdamW更新一致性；独立动作/rank/Fraction/net验证也通过。5142537已提交dev_accelerated,accelerated，CPU8/16G/1GPU分配（实际CPU计算），10分钟。提交spool逐字节一致。任务内自动fit、fresh replay、predict、independent readout；当前没有新准确率。

原强基线28/32、99/128；已测unit-cost为27/32、99/128（128中5救5损）。新目标须同时报告对这两个对照的救回/损失/净增，回到原数不叫突破。593不重跑，不将标准交叉熵冒称新理论。

程序programs/run_rc_full_candidate_identity_loss_v1.py；验证programs/validate_rc_full_candidate_identity_loss_v1.py；权威registry/rc_full_candidate_identity_loss_authority_v1_20260912.json；输出results/rc_full_candidate_identity_loss_v1/。预计报告reports/REPORT_FULL_CANDIDATE_IDENTITY_LOSS_V1_20260912.md。

---

# 2026-09-12：原因隔离完成，原TRAIN目标与六类已知改进动作存在已证冲突

5142472/5142473/5142474全部完成0:0；三模型121920logits、完整排名及数学风险界独立验证通过。原ORIGINAL7仍28/32和99/128；原成本4目标更充分优化后也是28/99，正确集合不变；仅成本4→1、原AdamW配方不变的新头为27/99，旧32为0救1损，新128为5救5损。

原保存头的记录损失0.9343481006454986；在预定[-64,64]^7内的最优值位于[0.9311129960982376,0.9311227458347325]，剩余下降空间约0.35%。六个预先固定的可行EVAL动作区域，其最小原TRAIN损失都被证明高于原头：五个联合保28+99再救一例，以及一个允许旧正确取舍的旧32至少29条件。由此，至少这些已知改进与当前经验风险偏好冲突；不能再单用“多跑训练”“统一放宽4倍成本”解释/解决当前平台期。

严格边界：只覆盖该参数盒、旧mixed96/原六特征/原cost4目标和六个固定区域；不是全部净增区域、无界全局最优、数据总体或信息论不可识别证明。成本1仍为5救5损，单一成本系数不是现成修复；损失形式、训练分布与区分条件之间尚未完全隔离。更低TRAIN风险未在本次改变识别正确集合，旧头保留。593未重跑，其447为此前五折开发结果，不混入本次面板。

结果results/rc_opened_convex_cause_readout_v1/result.json；独立验证同目录independent_validation.json；报告reports/REPORT_CONVEX_CAUSE_ISOLATION_V1_20260912.md。没有新试验或待监控作业。

---

# 2026-09-12：已实际增加开发分区，训练开始完成

此前仅核对了10分钟时限，没有改分区；本次按用户要求实际更新调度器。5142472及5142473_0/_1/_2改为dev_accelerated,accelerated，保持原ID/依赖/10分钟/CPU8/16G/1GPU。训练_0已在dev_accelerated完成56秒，界验证与160预测封存已通过。

dev QoS每用户最多1运行、4提交，六项诊断无法整批迁移（已收到并记录原始QoS拒绝）。诊断_3/_4/_5与collector暂留accelerated。dev_cpuonly/cpuonly的test-only验证通过，但当前预计排队更久；普通cpuonly独占整节点152CPU，不适合本次小CPU请求。没有新提交替代任务、没有取消或修改程序。

资源变更记录：registry/rc_convex_cause_partition_amendment_v1_20260912.json。诊断结果尚待全部完成，不将第0项单独当原因结论。

---

# 2026-09-12：原因隔离已提交，两项TRAIN对照＋六项数值诊断

用户明确go继续原因隔离。5142472数组两项：在固定参数盒[-64,64]^7内求原mixed96成本4经验损失的可验证数值界；以及保留原AdamW配方仅将错误成本4改1。旧ec7参数和预测复用，不重训旧头。原损失重算0.9343481006454986。

5142473数组六项：五个联合保旧28+99增一条件和一个旧32至少29正确（允许取舍）条件，全部使用既有严格可行证书，系数只在rc_opened隔离区，不当模型成绩。5142474依赖两数组成功后独立核算三头160图和原成本4的风险差；成本1的损失不能与成本4直接比。

两份preflight通过，核心12组合成数值界资格通过，spool逐字节核对。当前只表示已提交，尚无新原因结论或性能结果。完整恢复记录reports/RC_CONVEX_CAUSE_ISOLATION_WORK_STATE_20260912.md；数据及提交results/rc_convex_train_loss_cause_v1/；最终汇总results/rc_opened_convex_cause_readout_v1/。

本次新go只恢复这项有界任务，不无限延长旧试验。所有优化界限于参数盒，AdamW decoupled decay不当作L2；区间重叠必须报告未决，不凭solver状态宣称找到根因。

---

# 2026-09-11：重复计票与逐token可靠性方向并非全新，历史口径已更正

用户记得对。9月8日方案已写重复reference质量预算及DeepEMD；V7/V8已经实现并训练reference去重聚合，原V8 P-only TRAIN32为9/32 NO-GO。8月2日C6d-F1C已有同winner/challenger逐patch正负证据、可靠性门和全patch读出，job5041811资格训练0救0损，未进入外层正式评价。更早RCDE已有inverse column crowding。

撤回将上述概念作为此前未做的新方向的判断。当前RAW+RoMa/fullreference软容量的精确组合未找到完整实测，只能视作旧机制的接口复验候选；不能把不同输入自动当新理论，也不能把旧P-only或D1/C64失败直接等同于当前RAW/C128的失败。本次没有新增任务。

更正报告：reports/NOTE_NEW_HYP_MATCH_DIRECTIONS_HISTORY_CORRECTION_20260911.md。原候选方向文件已加更正提示，保留旧内容便于追溯。

---

# 2026-09-11：5141529完成；原PAIR＋FULL组均衡未提高准确率

作业COMPLETED 0:0，1分33秒；新头fresh replay、独立NumPy logits/动作/完整排名/计数复核通过，结果与源hash再次核对。原头训练更新为0，160图原预测直接复用。

原32图：RAW25→原ec7 28→GROUP_MIXED96 27，0救1损。唯一损失OUTCOME-0220：原ec7正确选择physical1024，新头退回错误RAW physical1311，target仍在C128。原128图：RAW88→原ec7 99→GROUP_MIXED96 99，0救0损；MRR .824013→.824274不构成准确率增益。

本轮保留原PAIR64+FULL32、特征/顺序/预算/损失1:1，仅改两个池内部组权重，仍没有超过原头。它不支持“只靠组均衡统一提高旧32/128”的修复；不再以零损失门或上轮FULL-only协议改动解释此次不增。原28/32、99/128保持，593五折447是此前不同协议结果，本轮未重跑。没有自动启动新方法或替换部署。

结果与完成记录：results/rc_original_mixed96_group_risk_v1/{result.json,result_validation.json,completion_readout.json}；报告reports/REPORT_ORIGINAL_MIXED96_GROUP_RISK_V1_20260911.md。

---

# 2026-09-11：直接复用原头；原PAIR＋FULL单因素组均衡已提交5141529

用户明确不再重训复现原头。ec7参数和160图原预测直接复用fixed269的已验证封存，原头训练更新为0。本轮唯一新头GROUP_MIXED96保留原PAIR64+FULL32、原训练顺序、4128x6训练特征SHA、原SIGN/BCE成本、两个损失1:1、七参数和2000步；仅将两个训练池内部mean改为各自20组/12组等权。不是269图FULL重训，也不删PAIR。

已检查已有2x2损失、product response和group jackknife；没有重复本条件。新输入与原特征SHA完全一致，源角色仅96条原训练范围，预检完成且没有自然训练更新。5141529已提交，accelerated/1GPU队列分配、实际CPU8、16G、10分钟，自动新头fit/replay→join→review→report，批处理spool逐字节一致。当前没有新科学结果，不替换28/32、99/128或593五折447。

唯一输出results/rc_original_mixed96_group_risk_v1/；权威registry/rc_original_mixed96_group_risk_authority_v1_20260911.json；提交快照submission.json。评价仍是已查看开发面板，以救回减损失>0报告观察净增，统计不确定性单列。当前UTC16截止安排不变。

---

# 2026-09-11：32/128/593三段成绩已可比较；旧面板未获增益

5141488的固定面板结果已通过独立复核：原EVAL32 RAW25→原ec7 28→GROUP269 26（0救2损）；原EVAL128 RAW88→原ec7 99→GROUP269 94（0救5损）。IMAGE269分别26和93；相同269训练行上组均衡分别净0和净1。因此不能把新头对旧头的下降全部归因于组权重，更不能称本轮改善了旧28/99。

593五折：RAW426→原结构较强SMALL444→GROUP_BASE447，净增3；同ALL训练行匹配对照为440→447，8救1损。三个分母对应不同身份构成和评价协议，32/128是同ec7固定头，593是每折重训原结构；不能解释成增加训练数据使准确率按32→128→593单调变化。原组合对RAW三套均有增益，当前分组方法只在593协议中超过较强原结构基线，尚无跨三套一致模型优势。

完整结果：results/rc_fixed_panels_train269_group_risk_v1/result.json；独立验证：result_validation.json；报告：reports/REPORT_FIXED_PANELS_TRAIN269_GROUP_RISK_V1_20260911.md。原基线逐图动作、C128和完整排名均核对，部署未改。

---

# 2026-09-11：原32/128三段性能链复测已启动（5141488）

用户要求分别列出同一评价集上的RAW → 原组合模型 → 理论指导优化。593五折已验证RAW426、原结构较强SMALL444（同ALL行440）、GROUP_BASE447，不能代替旧25→28/32与88→99/128的提升。

本轮固定原训练32身份/32组内全部269张唯一图，原EVAL32与原EVAL128的图像/身份/组全部排除。新训IMAGE269与GROUP269共享七参数、全部269图、2000步和FULL-C128 SIGN目标，仅图像等权与组等权不同；原ec7冻结。新头从零训练，不使用593五折参数。

5141488已开始运行，accelerated/1GPU队列占位、实际CPU8、16G、10分钟；任务内自动完成训练与fresh replay、历史输入一致性、评价、独立复核、报告。提交spool逐字节一致。当前没有新准确率，29/32仍不是已得到的训练结果。每面板按救回>损失记观察净增；历史已查看开发面板，不是外部确认；未替换旧部署。

唯一输出：results/rc_fixed_panels_train269_group_risk_v1/；权威：registry/rc_fixed_panels_train269_group_risk_authority_v1_20260911.json。近似名group269草稿没有提交，不作当前入口。

---

# 2026-09-11：按用户净增目标，本轮已达标，GROUP_BASE447作为当前开发对照

用户强调当前只要求净增。按同一593图五折指标，GROUP_BASE447对上一版ALL_COND445是5救3损、净增2；对补齐后的SMALL_CONST446是4救3损、净增1。该开发净增目标通过，不能再因条件门必要性/跨组显著性未充分证明而统称本轮失败。

当前后续开发对照方法为GROUP_BASE（7参数、MRR .813432），447是五折方法成绩，不是一个最终全训练部署头。旧预定主比较、统计结果与原99/128账本保持原样；理论原创性和外部泛化继续单独报告。

已修复的可操作因素是训练损失组权重；剩余593-447=146个错误中23个target缺席C128、123个在场仍错。额外条件门在该强基头上447→446，证据取舍仍有冲突，不能把123个错误全部归为同一个原因。

接受记录：registry/rc_h593_observed_net_gain_acceptance_v1_20260911.json。

---

# 2026-09-11：统一假说预测检验已完成，七参数组风险基头达到447

同593五折、同ALL输入，仅改损失组权重：BASE440→GROUP_BASE447（8救1损，精确组p=.0078125）。GROUP_CONST447、GROUP_COND446；预定GROUP_COND对ALL_COND445主比较没有稳定优势。强小基头补齐后SMALL_CONST446、SMALL_COND445，额外条件门仍未证明必要。

447对原条件445为5救3损，组差略负；对补齐后强对照446为4救3损，可靠优势仍未建立。不能把副比较成功改写为主比较通过，也不能与旧99/128比较。权重改变是可验证的训练解释与技术修正，组均衡本身不当新理论。

理论稿：reports/NEW_HYP_UNIFIED_DECISION_HYPOTHESIS_V3_20260911.md；预测结果：reports/REPORT_H593_UNIFIED_HYPOTHESIS_PREDICTION_READOUT_V1_20260911.md。5141412/5141413/5141414全部完成并独立复核，无待监控任务。旧所有结果及部署不变。

---

# 2026-09-11：新颖性表述校正

整体候选绑定有助识别是旧69/90、原99/128已经支持的结论。593实验只新增扩大群体验证与“冻结BASE、单独干预新增F补偿”的归因，不能包装成基础机制首次证明或新理论突破。强SMALL128对照仍只被3救2损、净增1地超过，可靠优势未建立。整体C_BIND保留为标准对照，不再单独作为新发现。

详见reports/NOTE_H593_BINDING_CLAIM_NOVELTY_CORRECTION_20260911.md。已完成实验及原报告保持冻结。

---

# 2026-09-11：593图识别与增量归因已完成，有明确正向证据

同一593五折：RAW426→BASE440（14救0损）→条件自由内容补偿445（19救0损对RAW；5救0损对BASE）。19次RAW纠错跨15组，精确组p=0.000061；整包候选绑定打乱后19次全部消失。

针对性跟进已全部完成：纯全局BIAS1仍440；保持BASE，仅打乱新增补偿绑定，COND445→441，原5次新增全部消失。支持候选绑定内容补偿的选择性纠错作用。条件化相对统一补偿444只多1，相对SMALL128基头444为3救2损，尚未证明这两个更强比较的可靠优势。J端点ENDPOINT2为439，不继续作为本轮性能候选。

5141332/5141333/5141334全部COMPLETED 0:0，独立复核通过；当前593相关工作已完成，无需继续监控。没有替换旧99/128或部署。

完整正向与限制收口：reports/REPORT_H593_POSITIVE_RETRIEVAL_AND_INCREMENT_ATTRIBUTION_CLOSURE_V1_20260911.md。

---

# 2026-09-11：593图结果已出，保留内容补偿正向主线

同一593五折：RAW426、SMALL128基头444、ALL基头440、CONST1 444、COND4 445。COND4对ALL是5救0损，对SMALL是3救2损，对CONST仅1救0损。对RAW19救0损/15个正向组；精确组检验p=0.000061。原99/128不可直接比较。

J端点ENDPOINT2为439，4救5损；RELATIVE2为445，7救2损，但J_BIND为449，不能把449当新模型或将+5归为正确J绑定。端点方案本轮不再作为性能候选。

五次COND救回全是原top challenger已为target但未过HOLD。新增针对性跟进：5141332仅拟合BIAS1并做冻结增量绑定控制 → 5141333汇总 → 5141334独立复核。accelerated/1GPU占位、CPU8/16G、10分钟。不改任何已有参数/结果。

综合结果：reports/REPORT_H593_MAIN_AND_ENDPOINT_RESULT_SYNTHESIS_V1_20260911.md；精确统计：results/rc_h593_completed_result_interpretation_v1/result.json。

---

# 2026-09-11：两组五折拟合均完成，等待最终汇总复核

5140747五折全部COMPLETED 0:0，每折88–89秒；5141118五折全部COMPLETED 0:0，每折81–84秒。全部10份参数/预测新进程重放验证及payload/receipt哈希核对通过。30分钟申请上限足够；原1小时59分钟申请过于保守。

最终指标仍待join/review。按用户刚才授权的同一资源方案，5140748/5140750/5141119/5141121已从cpuonly改accelerated、1GPU占位、8CPU/16G、10分钟，Job ID与依赖保留。没有读取新科学成绩，更没有把训练完成写成HYP GO。

完成记录：registry/rc_h593_both_fit_arrays_completed_v1_20260911.json；汇总资源覆盖：registry/rc_h593_readout_accelerated_resource_override_v1_20260911.json。

---

# 2026-09-11：拟合数组转accelerated并缩短申请

按用户要求，5140747（原1小时59分钟）与5141118（原30分钟）均改为accelerated、1GPU/8CPU/16G、30分钟/折、5折并行。CPU-only脚本与源hash未改，GPU只占位；Job ID及依赖不变。迁移已通过scontrol核实，检查时仍为PENDING，不能承诺立即开始。

资源覆盖记录：registry/rc_h593_accelerated_fit_resource_override_v1_20260911.json；完整前后调度字段：results/rc_h593_accelerated_fit_migration_v1/attempt.json。原GPU输入RAW10/RoMa15与科研截止不变。未来重提这两个拟合需显式沿用资源覆盖，不能直接恢复旧SBATCH头的cpuonly/1小时59配置。

---

# 2026-09-11：J端点符号独立分支已提交

新分支使用同一593五折及各fold冻结BASE7_ALL，比较RELATIVE1、RELATIVE2(r,1)与ENDPOINT2(r,s)，主比较端点状态是否优于全局偏置及原BASE。只用检索身份监督，每折只训练1/2新增参数；旧593实验不变、旧99不作直接比较。

已提交J缓存5141116/5141117 → 新五折5141118（另依赖原基头5140747）→ join5141119 → 独立复核5141121。无额外GPU，仅从资格化tokens/maps算J。原593最近检查RAW全部完成、RoMa9完成/10运行/24排队，无失败；新旧模型尚无新的本轮结果读出。

完整交接：reports/REPORT_H593_ENDPOINT_SIGN_EXECUTION_HANDOFF_V1_20260911.md。诊断与其它fold读取屏障已动态验证；不读取rc_opened_缓存作为模型输入。

---

# 2026-09-11：等待期间两项256图诊断已完成

5140966/5140967/5140988全部完成并独立复核。原ec7、RAW自然C128、EVAL128：原99；自由选点98（0救1损）；仅移除reference幅度93（1救7损）；两者移除89（1救11损）。0337自由选点仍错、移除幅度能救，不能把选点错误当全部根因。

22个target在场错误中，3个固定wrong对F→A反转；9个target在原S已是C128第一但最终头选错。J对target有更强选择性，但J单独排序93（6救12损）；candidate级wrong J>0仅48/16263，却涉及39/128张图。无自然原六特征＋winner标记精确摘要碰撞。旧SPEC8只给sym(Jc,Jw)，确会折叠部分绝对符号信息；未验证这就是其失败原因。

诊断全部隔离，不改593训练。解释修正：reports/REPORT_OPENED_EVIDENCE256_INTERPRETATION_CORRECTION_V1_20260911.md；完整结果：reports/REPORT_OPENED_EVIDENCE_LOSS_AND_SUMMARY256_V1_20260911.md。

---

# 2026-09-11：593图RAW保存错误已修复，新的输入数组5140924/5140925

旧5140736的43片均因PyTorch拒绝临时路径名在落盘处失败，不是超时或模型NO-GO。256张复用特征全部验证完成并保留。新增仅修复savepayload的V2 wrapper，两套环境实际写读及逐位检查通过。

新链：RAW5140924（10分钟/片，%46）→ RoMa5140925（15分钟/片，%46）→ 原新特征5140739 → 五折拟合5140747 → join5140748 → 独立复核5140750。旧死依赖RoMa5140737已取消。所有下游Job ID保留且依赖已核对。当前没有新识别结果；不会继续空等监控。

详细记录：reports/REPORT_NEW_HYP593_SERIALIZATION_REPAIR_V2_20260911.md。下面早前提交状态均为历史。

---

# 2026-09-11：593图五折分组检索训练已接通执行链

用户批准593张分组交叉验证；并行上限改46，RAW10分钟/片、RoMa15分钟/片。元数据593图/68身份/64组、五折119/118/119/119/118已独立验证。256输入复用、337补算。

已提交：RAW5140736 → RoMa5140737 → 新特征5140739；复用特征5140738独立；两特征链成功后 → 五折拟合5140747 → join5140748 → 独立结果复核5140750。主比较CONDITIONAL4对BASE7_ALL，另有SMALL128与CONSTANT1对照。当前没有新科学结果；旧原模型EVAL128仍保留99的历史账本，新593 OOF不得直接与99比较。用户要求没事干就退出，本轮不持续监控。

完整交接：reports/REPORT_NEW_HYP593_OOF5_EXECUTION_HANDOFF_V1_20260911.md。下文此前CE选择草案未执行，旧实验保留为历史，不代表当前优先分支。

---

# 当前状态：new HYP，提交后继续推进

## 2026-09-11：用户更新未来目标，净增优先、允许少量旧正确损失

用户在选择“原正确全保”后补充：“其实，如果净增可靠，损失一两张也行，毕竟现在要的是提高准确率。”以此最新要求为准：后续主线EVAL128原正确损失预算最多2张，必须有正净增，并完整报告组效应与不确定性；不能单凭净增即称可靠。历史门和结果不改写，原EVAL仍28/32、99/128。

新准备的TRAIN概率损失选择仅复用已封存的16FIT候选池；不新LP、不重新切组，以SELECT上完整128类CE提供比二值正确率更连续的选择信号。固定对照ACC_BUDGET/CE_FREE，主臂CE_BUDGET，SELECT预算ceil(2*n/128)。当前未自然执行，真正EVAL不开放。
目标更新：registry/rc_retrieval_only_net_gain_objective_update_v1_20260911.json
计划：plan/RC_TRAIN128_PROBABILITY_SELECTION_V1_20260911.md

下方保留之前的时间点记录；其中“尚未回答”已被本段更新。

## 2026-09-11 北京时间06时后：选择隔离试验完成，真实EVAL仍28/32、99/128

5140476已完成并fresh验证。以下都是TRAIN outer留组结果：BASE108，FIT_SELECT111
（6救3损），SELECT_RANK108，SELECT_PROTECT108（两者0救0损，原BASE正确全保）。
主门因没有增量未过，不进入真正EVAL；原模型没有改变。

前三折SELECT原正确29/31、32/35、34/35，全部候选在SELECT无新增正确，所以两个
SELECT臂都回BASE；第四折选择一份SELECT1救0损更新，但outer无新增。因此额外
保护过滤不是本次失败的唯一解释。基头已可能见过SELECT，只有增量拟合与之隔离；
outer对基头和增量均隔离，但作为已打开的开发数据复用，不称独立外部确认。

最新结果：reports/REPORT_TRAIN128_GROUP_HELD_SELECTION_RESULT_V1_20260911.md
完成回执：registry/rc_train128_group_held_selection_completion_v1_20260911.json
来源结果/验证：results/rc_train128_group_held_selection_v1/{result,validation}.json
本分支目前没有待运行的自然任务。截止仍为北京时间9月11日24:00。

用户问是否根本无解：已说明不能保证严格保全旧正确再新增一定能学到，16个严格
线性修复被排除不等于所有模型无解；原99的机制证据仍保留，未宣称新增HYP GO。
已向用户询问后续‘原正确全保’是硬门还是允许少量损失但可靠净增优先，尚未收到
回答；当前标准不变，不将已完成对照事后改成primary或GO。

下方为历史时间点记录。

## 2026-09-11 北京时间06时：恢复研究，原EVAL仍28/32与99/128

用户已恢复推进；截止仍为北京时间9月11日24:00（UTC16:00）。

暂停前5139521实际已完成，原状态段落中的RUNNING过时。TRAIN OOF：BASE108，
PROTECTED110（对BASE6救4损），REPAIR_ONLY108（6救6损）。四折TRAIN保护全部成立，
新组却有4张RAW正确HOLD被错误SWITCH；原23次正确纠错保留。主门未过，不进入EVAL。
当前已完成只读复核：97536个特征标量、48768个OOF logits、114项候选训练评分。
失败记录：reports/REPORT_TRAIN128_PROTECTED_PROJECTION_RESULT_AND_GENERALIZATION_V1_20260911.md
完成回执：registry/rc_train128_protected_projection_completion_v1_20260911.json

具体发现：旧配方在同一24组产生/保护/筛选候选，选择首先最大化TRAIN组准确率，
最后才以L1破平局；fold0/1选中L1位移18.89/29.31。损失还涉及较小位移fold2，
不能简单断言只限小步长便能解决。现在准备同池FIT/SELECT隔离检验：每outer折的
24个训练组按固定hash拆16 FIT/8 SELECT，FIT-only候选池封存后再用SELECT标签筛选。
原基头可能已见过SELECT的原FULL/PAIR标签，只声称增量拟合隔离；outer组仍同时
排除在基头与增量学习之外。旧全24组候选池不作为‘未见SELECT’复用。
新实验5140476已提交dev_cpuonly，8CPU/59分钟；源码124b03fd...、authority e78dbd23...，
spool逐byte一致。metadata 16FIT/8SELECT/8outer组和NIL合成检查、独立设计复核通过，
当前尚无自然结果。新输出：results/rc_train128_group_held_selection_v1；
计划：plan/RC_TRAIN128_GROUP_HELD_SELECTION_V1_20260911.md。

原六特征联合容量的5个分别可行、16不可行、1未决结论保留；不能说全部模型无解，
也不能把数学witness说成新成绩或交给训练。新方法是否能泛化无损尚未证明。

以下是历史记录，最新状态以上述时间点为准。

## 2026-09-10：联合容量检查完成，TRAIN保护式学习已启动

原实际EVAL仍旧28/32、新99/128，没有新增检索成绩。

5139464容量检查已COMPLETED0:0（62秒）：共享7参数、原六特征、同时严格保护旧28
和新99后，22条候选在场原错中5个分别精确可行、16个精确不可行、1个数值未决。
21份证书独立复核；未决不影响至少一个可行解的存在性证明，但全22分类未完全闭合。
五个各自可修不等于同一模型修五个，也不等于TRAIN已学出新100/128。所有系数隔离。
容量报告：reports/REPORT_JOINT_EVAL_LINEAR_RETENTION_CAPACITY_V1_20260910.md
完成回执：registry/rc_opened_joint_eval_linear_retention_capacity_completion_v1_20260910.json

因此现在转向仅TRAIN的保护式最小改动学习，而不直接扩大特征。5139521已提交并
RUNNING：固定旧四折BASE7和原六特征，PROTECTED7加入TRAIN旧正确动作保护，
REPAIR_ONLY7为同预算控制；每个TRAIN错误只求一次L1最小改动LP、按TRAIN组均值/
正确数/参数L1固定择一，原头始终入池。数值LP后用精确正margin和实际FP64动作
验收，四折fresh重求参数与预测再join。当前尚无完整OOF结果，禁止EVAL。
整个程序硬拒绝任何rc_opened_*读取，包括hash，EVAL证书参数/方向/病例不进入训练。
新计划：plan/RC_TRAIN128_PROTECTED_PROJECTION_OOF4_V1_20260910.md
输出：results/rc_train128_protected_projection_oof4_v1
新authority：registry/rc_train128_protected_projection_oof4_authority_v1_20260910.json
新提交：registry/rc_train128_protected_projection_oof4_submission_v1_20260910/submission.json

以下保留已完成的上一轮实际EVAL记录；最新进度以本段为准。

## 2026-09-10 08:23 UTC / 北京时间16:23：实际EVAL跟进完成，未突破原99

**主线最优仍是冻结ORIGINAL7/NATIVE7 ec7：旧EVAL32为28/32，新EVAL128为99/128。**
TRAIN或OOF的108、110都不是这批EVAL的成绩，不作99→108/110的前后比较。

最新探索性GAP跟进5139362已COMPLETED0:0（56秒），参数、40640个双臂EVAL logits、
两个面板原28/99正确集合、RAW25/88、MRR与组统计的fresh复核均通过。
唯一alpha=0.5421478288984741由全TRAIN128精确拟合（原TRAIN108正确保留，新增5），
EVAL未调参。实际结果：

| 同一EVAL面板 | 原ec7 | ec7+GAP | 对原模型新增/损失 | 原正确保留 |
| --- | ---: | ---: | ---: | ---: |
| 新EVAL128 | 99/128 | 99/128 | 1/1 | 98/99 |
| 旧EVAL32 | 28/32 | 27/32 | 0/1 | 27/28 |

新128救DIFFICULT-0028、损OUTCOME-0130；旧32损OUTCOME-0618。三个预设性能目标
全未满足，不采用、不改原头。新128 MRR仍0.8240132107292264，是2→1与1→2恰好
抵消，并非正确集合保住。旧32 C128 MRR降0.015625；不与新128 full5412口径混拼。

已隔离的精确EVAL后验容量证明进一步排除这条固定rank共享非负GAP路线：
- 新128保原99需alpha<=0.21092452188149946（0130）；任何新增的最早首点为
  0.5244909911132739（0028），11个target已为原top的潜在救回全在保护上界之外。
- 旧32保原28需alpha<=0.010972945716349658（0618）；最早救回首点
  3.218998338140889（0050）。
因此该类任何有限FP64 alpha都不能保全原正确再新增，不能继续调全局系数。
这不否定允许取舍的更高准确率、条件化/改排序模型，或既有原模型的正净增。
容量材料放在rc_opened_eval_*隔离目录，不给后续训练读取。

本轮三条自然计算链均完成，无本分支待运行任务：
- 5139328缓存；5139336四参数条件补偿：TRAIN OOF BASE108、常数108、条件107，NO-GO。
- 5139348锁SWITCH/排序的HOLD校准：CONTENT主臂108（2救2损），GAP控制110（2救0损）；
  CONTENT原门失败，未因控制较好改成GO。
- 5139362是上述GAP控制的独立、事后探索性EVAL跟进；它不是父CONTENT过门晋级，
  已打开EVAL及对照选择偏差明确保留，结果也未通过。不再启动同类全局补偿试验。

最新真实EVAL报告：reports/REPORT_ORIGINAL7_GAP_HOLD_EXPLORATORY_EVAL_RESULT_V1_20260910.md
最新完成回执：registry/rc_original7_gap_hold_followup_completion_receipt_v1_20260910.json
实际结果/复核：results/rc_original7_gap_hold_followup_v1/{result,validation}.json
隔离容量证书：results/rc_opened_eval_gap_hold_retention_capacity_v1/{result,validation}.json
原最优机制证据：reports/REPORT_NEW_HYP_ORIGINAL7_EVAL128_MECHANISM_RESULT_V1_20260910.md
new HYP理论定义/性质更新快照：reports/NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V2_20260910.md
（其中运行中段落为写作时快照，最新试验状态以本文件和最新报告为准。）

用户研究授权仍至北京时间9月11日24:00；ownership不在目标内。独立reference-first
S0由另一对话维护，不在上述“本分支无待运行任务”的范围，也不覆盖其validator。

以下保留旧时间点记录；未开始、运行中等描述均是历史，不代表最新状态。

## 2026-09-10 07:23 UTC / 北京时间15:23：条件化OOF已完成，未通过门

当前最优仍是ORIGINAL7/NATIVE7：旧EVAL32为28/32（RAW25，3救0损），
新EVAL128为99/128（RAW88，12救1损）。原ec7 seal SHA42c8e503...已核对未改。

新完成的TRAIN128、32正例身份/组、四折留组试验：RAW86/128；每折重新训练的
BASE7为108/128（23救1损），CONSTANT1为108/128（23救1损），CONDITIONAL4为
107/128（22救1损）。这些OOF基头不是ec7，不与旧/新EVAL成绩混拼。
等权组准确率依次83.125%、82.604%、81.979%。条件臂相对BASE救0389、损0087/0102；
相对CONSTANT仅额外损0102、没有新增正确。五个门中仅RAW break不增加通过，
正确数与等权组均未胜过两个对照。因此停止这项4参数配方，不打开EVAL、不替换原模型。
本轮负结果不等同于new HYP理论不存在，也不撤销原99/128的内部净增。

- 缓存5139328：COMPLETED0:0，2分38秒，16384候选、65536旧C4及81920新特征标量复核。
- 四折拟合/独立重训/join5139336：COMPLETED0:0，9分54秒；全部48768 heldout logits
  fresh重放及统计重算通过，EVAL读取0。该链无待运行任务。
- 三例已分账并1143个logit逐bit回放：0389由负logit跨0救回；0087被压回HOLD；
  0102的target/wrong竞争margin由+.02287翻为−.00898。后两例均为BASE已有rescue，
  所以仅看RAW break不变会漏掉这些损失。报告：
  reports/REPORT_TRAIN128_DISAGREEMENT_OOF4_THREE_CHANGED_DECISIONS_V1_20260910.md
- 用户要求继续最后推进。新增有限HOLD校准计划已冻结设计，prototype正在实现：
  原BASE已SWITCH全部不改、原top challenger排序不改，只比较GAP与自由内容辅助的
  原HOLD放行；每折用TRAIN身份标签精确拟合一个共享非负FP64系数，硬约束TRAIN
  原BASE正确集合保留，并以最小系数达到最大可救数。OOF需完整保留BASE正确且
  CONTENT胜BASE/GAP；未通过不进EVAL。该机制明确是HOLD校准，不称新身份排序。
  5139348已提交dev_cpuonly（8CPU、20分钟），spool与冻结launcher一致；原5139336
  已从Slurm实时表清理，故核对COMPLETED0:0及artifact资格后直接提交，运行时仍绑定
  全部前驱hash。此时尚无完整新OOF结果，更没有超过EVAL99的成绩。
  新authority：registry/rc_train128_hold_lift_exact_oof4_authority_v1_20260910.json
  新提交：registry/rc_train128_hold_lift_exact_oof4_submission_v1_20260910/submission.json
  计划：plan/RC_TRAIN128_HOLD_LIFT_EXACT_OOF4_V1_20260910.md
- 独立reference-first S0支线由另一对话推进，本对话不覆盖其validator、不重复提交；
  它不等同于此处99/128主线。见避免重复记录。

结果报告：reports/REPORT_NEW_HYP_TRAIN128_DISAGREEMENT_OOF4_RESULT_V2_LAYOUT_20260910.md
全32组图：results/rc_train128_disagreement_oof4_display_v2_layout/all32_group_results.png
完成回执：registry/rc_train128_disagreement_oof4_completion_receipt_v1_20260910.json
机器结果/复核：results/rc_train128_disagreement_oof4_v1/{result,validation}.json
冻结计划：plan/RC_TRAIN128_DISAGREEMENT_OOF4_V1_20260910.md
拟合authority：registry/rc_train128_disagreement_oof4_execution_authority_v1_20260910.json
提交记录：registry/rc_train128_disagreement_oof4_submission_v1_20260910/submission.json
缓存完成回执：registry/rc_train128_disagreement_cache_completion_receipt_v1_20260910.json
理论更新快照（07:17的OOF运行时状态，以本文件为最新进度）：
reports/NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V2_20260910.md
机制分解：reports/REPORT_NEW_HYP_REFERENCE_SELECTION_AND_CALIBRATION_V1_20260910.md
S0避免重复记录：registry/rc_retrieval_only_s0_replay_recovery_note_20260910.json

以下保留已完成链条的历史记录；其中“尚未启动条件模型”等描述是旧时间点状态。

## 当前目的与执行

本轮继续工作已全部完成（约2026-09-10 05:25 UTC，北京时间13:25），没有该链待运行任务。
TRAIN128输入链5139250→5139251→5139252→5139253、三条件读出5139291和外部复核
5139297均COMPLETED0:0。全部128图/65536 C4校验通过，原FULL32 tokens/RAW/maps/C4
逐bit一致；原参数ec7回归、三头fresh重训及121920 logits/960动作的外部复核均PASS。

科学结果：ORIGINAL7旧28/32、新99/128；FULL96与FULL128均旧27/32、新97/128。
两新头均未新增正确：损旧0220、新0140/0439，0337仍损；原25/88 RAW正确保持分别25/87。
没有采用新模型，原头的88→99（12救/1损）结果保留。本次补完整负候选/增加32视图，
在固定七参数配方下没有改善；不能据此声称任何扩容均无效。96/128 target全部自然
在C128内，原损失域无需实际剔除任何新TRAIN样本。

关键新定位：0337保留query权重的自由内容target0.707504>wrong0.594628；wr改选
reference-token位置后，裸cosine反转0.443245<0.475781，再由幅度放大。这是本例
reference门控造成身份证据反转的直接计算证据，不是所有失败的共同根因。
FREE8也已拆清：0220旧六项重拟合先使logit为−0.231542，第7列再负推−0.401337；
不能把整个失败只归于F或bias。冻结原7加同一alpha*dF的必要条件精确不相容：
保0220需alpha<0.3604843，修0337需alpha>=14.6900121。证书隔离在rc_opened_eval_strict_*
目录，不得用于训练参数/阈值，只排除该一维残差类。

FULL128丢失的0220/0140/0439仍各自胜最强wrong，却target logit<0而HOLD；0337的
wrong logit仍+0.844524。当前剩余问题是条件化处理不同证据冲突，不是再找统一阈值。
尚未启动或验证此类新模型。研究可继续到用户给定截止，不能把当前局部定位写成全问题已解决。

- 本轮结果和失败机制：reports/REPORT_NEW_HYP_TRAIN128_RESULT_AND_FAILURE_MECHANISM_V1_20260910.md
- 完成回执：registry/rc_original7_train128_completion_receipt_v1_20260910.json
- 三条件结果/独立复核：results/rc_original7_train128_readout_v1
- 输入cache：results/rc_original7_train128_inputs_v1（全部已资格化，可复用）
- TRAIN128清单：results/rc_original7_train128_manifest_v1（32训练身份/32组，原96+32视图）
- 输入authority：registry/rc_original7_train128_execution_authority_v1_20260910.json
- readout authority：registry/rc_original7_train128_readout_authority_v1_20260910.json
- 提交记录：registry/rc_original7_train128_dag_submission_v1_20260910
- 0337分解与review：results/rc_eval128_0337_visibility_content_v1
- FREE8分账：results/rc_free8_component_accounting_v1
- 隔离证书：results/rc_opened_eval_strict_free_residual_capacity_v1

所有隔离指query图片/正例身份/来源组，gallery固定共享，不称测试reference从未
作为TRAIN负候选出现。无新空间标注、D1-MI或受保护formal392推理消费；旧EVAL32和
新EVAL128均按已打开内部开发评价分别报告。

以下是已经完成的EVAL128结果，仍为当前最好模型的正式内部结果：

新增128图主线已完成并独立复核（2026-09-10约03:28 UTC，北京时间11:28）。
固定当前ORIGINAL7/NATIVE7原六特征七参数，无新训练，完整RAW5412→自然C128→
127-challenger HOLD/SWITCH：**RAW88/128→REAL99/128，12救/1损、净+11**，
MRR 0.770066940→0.824013211；candidate recall121/128，7个缺席样本保留。

本轮覆盖24身份、21来源组，与当前TRAIN/PAIR及旧EVAL身份/组分离；来源历史
已打开，不称未触碰外部确认。增益分布5组；21组等权增益+8.390个百分点，
组bootstrap95%[+2.041,+15.986]个百分点，精确双侧组sign-flip p=0.0625。
结论为扩大样本上的内部正净增，**严格零损保持未满足，未达到双侧p<0.05**。

主要固定参数对照：REAL99、C_BIND67、DROP_M88（0救/0损）、DROP_L85
（0救/3损）、DROP_RAW94（13救/7损）、DROP_S98（14救/4损）、DROP_Q99、
DROP_R97、DROP_QR98。原12救在DROP_M/L下全部消失，在C_BIND下仅保留1个。
支持候选绑定、质量/内容联合读出及RAW竞争/保持的条件性作用，不能宣称每列
都必需或空间ownership成立。唯一break是OUTCOME-0337：wrong的M/L合计
+9.150529超过RAW/S/响应/bias抑制，最终logit+1.273408；未据此调阈值。

5139177是在RoMa构造前因版本校验API口径错误失败，实际环境未漂移；已追加
V2修复并复用有效RAW桥接。完整链5139195→5139196→5139197→5139198→
5139199→5139200，以及独立复核5139233、绘图5139234全部完成。
全部GPU/C4为128图、16384候选、65536评分核对；所有预测先封存后join，
独立复核重放162560 logits、1280动作、完整5412排名及组统计，均PASS。
当前这些作业没有等待项；未自动替换模型或启动新阈值/训练试验。

- 机制结论：reports/REPORT_NEW_HYP_ORIGINAL7_EVAL128_MECHANISM_RESULT_V1_20260910.md
- 主结果：reports/REPORT_ORIGINAL7_EVAL128_FULL_EVIDENCE_V1_20260910.md
- 独立复核：reports/REPORT_ORIGINAL7_EVAL128_INDEPENDENT_REVIEW_V1_20260910.md
- 全部21组图/CSV：results/rc_original7_eval128_group_plot_v1
- 完成回执：registry/rc_original7_eval128_completion_receipt_v2_20260910.json
- 原执行计划：plan/RC_ORIGINAL7_EVAL128_FULL_EVIDENCE_V1_20260910.md
- V2权威：registry/rc_original7_eval128_full_evidence_execution_authority_v2_compat_20260910.json
- 提交/激活记录：registry/rc_original7_eval128_v2_dag_submission_20260910

目的：让相似物体识别更准，同时保持已有正确。此前32图开发门为保留
原28正确并新增；当前主线为冻结原完整六维模型，在新128图上验证其
相对RAW的纠错、保持及机制重复性，不混拼不同样本规模的准确率。
new HYP用来解释和支持这种reference识别能力，不要求空间框或ownership。

中断前的两个实验都已完成，不应重复提交：
- 5139021：ORIGINAL7=28，MOMENT8=28，CURVE8=27。离散度保住原28，
  但无新增；非线性控制丢0220。独立重训与额外动作核对通过。
- 5139024：ORIGINAL7=28，SPECIFIC8=27，FREE8=27；两新头均无新增，
  都丢0220。COMPLETED 0:0，1分57秒；独立重训与73,152个logits、
  576个动作核对通过。两轮均未达到用户目标，未采用新模型。

支持优化pilot（5139121）现已完成，COMPLETED 0:0，2分18秒。
固定前4条TRAIN、每条完整C128，512项全部通过独立精确原/对偶证书。
正确目标中1条从原支持负margin变为优化支持正margin，说明原权重确实
漏读可区分证据。但508个错误候选中502个也有正margin，支持存在性
不能代替身份判定。4条TRAIN属于3个identity/group，不称4个独立确认。

当前进入优化支持与固定支持的同容量识别比较：保留原六特征、原
PAIR64+FULL TRAIN32检索监督和127-challenger action，仅比较第七列。
ORIGINAL7复现原28，FIXED8复现5139024的SPECIFIC8，MAXMIN8采用优化
支持的可达margin。按用户在TRAIN图上提出的对角线投影，再加同为8参数
的PROJECTION8次比较：用户明确要投影点距原点距离，按candidate
计算abs(J+T)/sqrt(2)，再用原symmetric构造第七列，不调角度或权重。
早先有符号SUM8解释已被用户更正，未作自然执行。原MAXMIN8主比较保留，不按EVAL选一个arm
冒充唯一预设方法。retrieval-only负责学习身份判断，不要求支持生成器
单独分类。源码与启动预检通过，完整缓存5139143已完成（6分10秒），8,320项全部通过独立资格，
含512项原pilot复用、7,808项新求解，工程失败0。5139151四头训练及独立重训已完成（2分31秒）：
原ORIGINAL7=28，FIXED8=27，MAXMIN8=27，PROJECTION8=26。MAXMIN8较原
0救1损（0618）；PROJECTION8较原0救2损（0220、0618）。MAXMIN8的
TRAIN为29，未转化为EVAL增益。没有新模型被采用。

直接距离最大规则也已单独验证：零新训练、全部C128 argmax投影距离，
TRAIN25/32、EVAL22/32；它不执行原HOLD/SWITCH，不能与训练头混称。
用户的多数target沿该方向较远，在本批数据有有限支持，但不足以胜过
RAW25或完整模型28。精确TRAIN反例0101说明任意绕原点线性变换加距离
都不能令该点严格第一；这只限制该二维距离类，不否决群体/shared head。

- 直接距离结果：results/rc_direct_projection_candidate_diagnostic_v1/result.json
- 旋转容量反例：reports/REPORT_NEW_HYP_TRAIN0101_LINEAR_PROJECTION_CAPACITY_V1_20260910.md
- 四头结果复核：reports/REPORT_NEW_HYP_MAXMIN_READOUT_POST_RESULT_REVIEW_V1_20260910.md

用户进一步提出学习坐标方向；已在读取本轮EVAL前固定DIRECTION9计划，
只增加一个全局theta，与原8参数一起用原TRAIN/PAIR监督训练；固定theta=0
须复现PROJECTION8。5139157已完成并通过独立重训与48,768个logits/384个动作复核。
共享theta=-0.31862845 rad，对应投影轴26.743934度；2000次梯度中1999次
非零。固定方向EVAL26，学习方向27，救回0618、0新增损失；但相对
原ORIGINAL28仍丢0220且无新增。TRAIN两头均28。原模型未替换。

- 后续计划：plan/RC_REFERENCE_SUPPORT_MAXMIN_READOUT_V1_20260910.md
- 用户投影补充：plan/RC_REFERENCE_SUPPORT_PROJECTION_DISTANCE_ADDENDUM_V2_20260910.md
- TRAIN投影核对：results/rc_reference_support_projection_distance_train_description_v2/description.json；
  固定/优化/相加的target第一名分别3/2/3，相加尚未超过固定支持。
- 共享角度完成回执：registry/rc_shared_projection_direction9_completion_receipt_v1_20260910.json
- 共享角度结果复核：reports/REPORT_NEW_HYP_DIRECTION9_POST_RESULT_REVIEW_V1_20260910.md
- 共享角度提交回执：registry/rc_shared_projection_direction9_submission_receipt_v1_20260910.json
- 共享角度执行权威：registry/rc_shared_projection_direction9_authority_v1_20260910.json
- 共享角度计划：plan/RC_SHARED_PROJECTION_DIRECTION9_TRAIN_ONLY_V1_20260910.md
- 四头完成回执：registry/rc_reference_support_maxmin_readout_completion_receipt_v1_20260910.json
- 四头执行权威：registry/rc_reference_support_maxmin_readout_authority_v1_20260910.json
- 四头提交回执：registry/rc_reference_support_maxmin_readout_submission_receipt_v1_20260910.json
- 全量缓存权威：registry/rc_reference_support_maxmin_cache_authority_v1_20260910.json
- 全量缓存完成回执：registry/rc_reference_support_maxmin_cache_completion_receipt_v1_20260910.json
- 全量缓存提交回执：registry/rc_reference_support_maxmin_cache_submission_receipt_v1_20260910.json
- 全量缓存：FULL64×128 + PAIR64×2，共8,320项；每项比较全部127对手。
  其中512项可复用pilot，新增7,808项；PAIR完整内容profiles需要补存。
- pilot结果：results/rc_reference_support_maxmin_train_pilot_v1
- pilot说明：reports/REPORT_RC_REFERENCE_SUPPORT_MAXMIN_TRAIN_PILOT_RESULT_V1_20260910.md
- pilot回执：registry/rc_reference_support_maxmin_train_pilot_completion_receipt_v1_20260910.json
- 前两轮结果：results/rc_evidence_dispersion_vs_curve_v1；results/rc_same_support_specificity_v1
- 信息区别见证：results/rc_same_support_information_witness_v1（标签自由数学输入，非准确率）

保留“新增且不丢原正确”的严格用户目标，同时把有取舍的正净增单独
记录。所有模型只从原TRAIN/PAIR检索标签学习，EVAL容量证书不能作为
模型参数。提交后继续独立工作；不反复监控调度器，不改运行中代码。

## 已完成的瓶颈与稳定性证据

**5138859已完成并通过完整独立重训验证。** COMPLETED 0:0，20分08秒，
52次拟合加52次独立重训全部完成；另独立核对845,312个封存logits及
6,656个动作/指标。没有挑选删除组或采用新模型。

- 加S的效果在12次TRAIN删组中均不为负：无Q/R背景8正4平，有Q/R
  背景11正1平。这里是相关的训练扰动，不是12次独立外部实验。
- 有S时再加Q/R的query净增为5正6平1负，作用较不稳定。
- ORIGINAL7在删组后为26–28，median27；原28只在1/12条件完整保留。
  0044、0212始终正确（12/12），0220为3/12，0618为9/12。
- 四个原错误在全部删除条件中仍未被修正。

当前还完成了一个精确容量结论：原六特征+七参数实数线性读出，
若严格保留原28正确，无法再修复任一剩余错误；四个系统均有独立
验证的Farkas证书。这不是总体最多28的结论，因为用户允许个体取舍。
更宽的容量检查也已独立验证：允许正确集合改变时存在至少29严格
正确的数学解，但保留RAW25的全部35个目标集合都严格不可行。因此
28不是总体上限；定位的是当前特征线性类中的识别—保持冲突。
所有内容仅为已打开EVAL标签感知数学诊断，不是模型、
新准确率、部署权重或外部确认；不影响已冻结训练。

- 稳定性结果：results/rc_train_group_jackknife_stability_v1
- 完成回执：registry/rc_train_group_jackknife_stability_completion_receipt_v1_20260909.json
- 稳定性报告：reports/REPORT_RC_TRAIN_GROUP_JACKKNIFE_STABILITY_RESULT_V1_20260909.md
- 保留28的精确容量：results/rc_opened_eval_strict_no_break_capacity_v1
- 总体29容量结果：results/rc_opened_eval_strict_cardinality_capacity_v1
- 容量与保持冲突：reports/REPORT_NEW_HYP_LINEAR_CAPACITY_AND_RETENTION_CONFLICT_V1_20260909.md

本轮提交后持续做了证据图、文献/数学复核、两头固定组分析、同头
matched32桥接、完整稳定性与精确容量诊断；没有停在提交。用户要求
继续至北京时间2026-09-11 24:00，即2026-09-11 16:00 UTC；不持续
监控调度器的要求保留。此前本轮Slurm任务均已完成；支持pilot已验证，全量缓存5139143已完成且独立资格闭合，四头5139151已完成且未达增益保持目标；共享角度5139157已完成；当前主线正在执行冻结原头的新128图评估，最新进度见本文顶部。

## 目标、主线和边界

目标是解释或改善相似物体的reference检索识别。当前主线：原RAW C128
→ RoMa匹配质量/visibility与ColNomic full-reference内容匹配 → 共享
校准 → 全部127个challenger的HOLD/SWITCH。当前理论名称统一为
**new HYP**，证据状态记E_g；旧Reference HYP指原空间/superregion命题。

任务训练只用query–reference身份/正负检索配对，不用本任务框、mask、
point、人工crop或分割teacher；基础模型预训练来源单独披露。当前
不使用superregion/SAM，不要求ownership，不启动旧V9，不访问D1-MI、
GroZi或未授权正式392。HOLD表示保留RAW winner，不是未知类别拒绝。

## 保留的原模型成绩

- NATIVE7/C_PAIRED，已打开matched EVAL32，原RAW C128：25→28，
  3救0损；同一头TRAIN32为27→28。原28尚未被新模型替换。
- 另一旧FROZEN_C头：原matched EVAL32为25→27，2救0损；opened
  difficult90为61→69，8救0损。不能把NATIVE7的28与旧69/70拼接。
- 所有数字按已打开内部/回归证据报告，不冒充未触碰外部确认。

## 本轮确立的正向信息与限制

**重训简单替代未复现完整模型（5138847，独立验证完成）。**
RAW2=25，RAW+M3=26，RAW+L3=26，JOINT4=26，原NATIVE7=28。
原完整头分别胜过两个重训单通道2救0损，等权group平均差均为正。
JOINT4主简化门未过，但正向预定次比较单独保留。M臂仍有RAW内容先验，
L本身仍包含visibility；不同头容量明确不同。

**S/Q/R固定2×2（5138852，独立验证完成）。**
JOINT4=26、PRODUCT5=27、RESPONSE6=26、ORIGINAL7=28。补S的重训条件
恢复0212目标选择；有S时补Q/R的重训条件保护0618。两个较小头都未
保留原28全部正确，不能只说简化成功，也不能抹去条件正向增量。

**固定头和重训的作用不同（两组局部诊断均独立验证）。**
当前NATIVE7固定去S为26（丢0419、0618），去Q/R为27（丢0618），
去S+Q/R为26；原3个RAW rescue均保留。因此0212的重训失误不能归因
于原S的直接项。0618则有直接Q/R保护：wrong logit原−0.010973，去
Q/R后+0.306137，删除后重训PRODUCT5为+0.053113，仍错误SWITCH。

**旧FROZEN_C的70/90没有跨两组无损（旧matched32桥接已独立验证）。**
旧头difficult90固定去Q/R得70，较原69多1且0损，原8个rescue保留。
但同一旧头matched32仍27，新增0220、损失0618，1救1损。因此该修改
不直接采用；Q/R作用不能写成跨head、跨数据普遍有益。

## 理论与论文当前写到哪里

正数无epsilon时，原乘积对比dS=(dM+dL)/(1+dM*dL)，能为线性头提供
非线性基函数。原4128个TRAIN/PAIR比较无M floor，含epsilon公式对
原值最大残差4.44e-16，独立重执行一致；生产特征仍直接读取原FP64列。
这是函数表达性质，不是新代数定律或普遍识别保证。

文献已核对ColBERT、PFE、MagFace、AdaFace：通用质量/内容联合和
MaxSim已有先例。贡献应落在具体候选条件机制、retrieval-only任务
监督和可复核解释，不能通过删去RoMa/ColNomic名称制造一般性证明。

近期ABS12/REL12（均26）、目标2×2（三新格27）、PAIR控制统一（26）、
D参考轴扩展（25）均无新模型增益，完整负结果仍保留。新发现与限制
都写入证据表，不按结果择优隐藏。

## 关键入口

- 总机制收口：reports/REPORT_NEW_HYP_CALIBRATION_MECHANISM_SYNTHESIS_V1_20260909.md
- 全证据表：reports/REPORT_RC_RETRIEVAL_ONLY_FINAL_EVIDENCE_LEDGER_20260909.md
- 机器可读索引：results/rc_new_hyp_verified_evidence_index_v1/README.md
- 全query图：reports/figures/new_hyp_verified_outcome_matrices_v1
- 严格容量诊断计划：plan/RC_OPENED_EVAL_STRICT_NO_BREAK_CAPACITY_DIAGNOSTIC_V1_20260909.md
- 论文稿：reports/RC_RETRIEVAL_ONLY_PAPER_WORKING_DRAFT_20260909.md
- 定义与性质：reports/NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V1_20260909.md
- 数学复核：reports/NEW_HYP_PRODUCT_CONTRAST_MATHEMATICAL_REVIEW_V1_20260909.md
- 文献范围：reports/NEW_HYP_PRIMARY_LITERATURE_SCOPE_V1_20260909.md
- 重训充分性结果：results/rc_retrained_evidence_sufficiency_v1
- S/Q/R重训结果：results/rc_product_response_factorial_v1
- NATIVE7固定组：results/rc_frozen_group_effect_native64_v1
- 旧difficult90固定组：results/rc_frozen_group_effect_difficult90_v1
- 同旧head matched32桥接：results/rc_frozen_qr_removal_matched32_v1
- 研究延长授权：registry/rc_retrieval_only_research_extension_authority_v1_20260909.json
- 提交后继续：registry/rc_retrieval_only_post_submission_continuation_addendum_v1_20260909.json

不修改运行中或已冻结代码；工程验证、固定头诊断、内部开发结果、
训练敏感性和外部确认分别报告。此前训练、稳定性与两级容量诊断均已完成并独立验证。当前新三头试验已完成，支持优化TRAIN pilot待实现/预检/冻结。无训练子集选择、新模型自动采用或oracle参数使用。
后续不能把总体增益与零损失混成一个门：应分别报告，并在需要严格
保持时考虑超出原特征线性类的表达/信息。原冻结实验门和结果不改写。
