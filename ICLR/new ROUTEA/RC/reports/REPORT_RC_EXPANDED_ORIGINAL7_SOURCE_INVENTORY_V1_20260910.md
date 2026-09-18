# 扩展 ORIGINAL7 评价：已有 RAW 输入盘点

主线可以扩到至少128张评价图片，无需先构建J/T或运行LP。本次只做
元数据盘点；没有选择最终cohort、读取candidate成绩/存在性、解码图片、
读取tensor、执行训练或评价。

## 满足旧组排除后的实际数量

| 阶段 | query记录 | 独立图片SHA | identity | group |
|---|---:|---:|---:|---:|
| 原RAW catalog | 987 | 982 | 81 | 77 |
| 排除TRAIN32+PAIR64及旧EVAL32的全部43 identities/groups | 416 | 414 | 37 | 34 |
| 再排除formal392的全部query IDs和图片SHA | **267** | **266** | **24** | **21** |
| 上述剩余中已有历史RAW600 C128的部分 | 30 | 30 | 3 | 3 |

先做身份/group排除，再做formal392负成员排除，全程没有按target是否
进入C128或模型是否识别正确筛选。formal392的392个ID对应389张图片；
两种键都已排除。267条中仍有1个图片重复记录，最终图片cohort必须先
按已冻结规则处理该重复，不能把它算成第267张独立图片。

267条合格候选元数据的图片路径和redacted token文件均存在。它们只是
可供后续冻结的未选池，旧EVAL32应另列回归集。历史曝光不等于当前训练
重叠，本池也不被称为未触碰外部确认。

机器盘点：[result.json](../results/rc_expanded_original7_source_inventory_v1/result.json)，
SHA `654e23ff7b83bb301e8ac2afe0fc74d2b18e05ef473f89285a1d15d281eb538f`。
其中包含所有源文件hash及不含身份label的未选图片/token元数据清单。

## RAW query与gallery来源

- 987条target-free query catalog：
  `cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json`，
  SHA `df7af8a116d25881b9dcf274fdd389d268b1811b8e17c9c412ab064925e290ec`。
- 全987条redacted token index：
  `cache/l0_natural_hardneg_v2_redacted_query_tokens_v1/index_v1.json`，
  SHA `2a78dbde7d375f5f3640ad523b08c813a436ef0b0549dcea92960f73b1f0f26d`。
  其`rows/*.pt`全部存在；schema为FP16 image tokens及template tokens，
  token维度128，image网格来自catalog。完整源验证为同目录
  `validation/validation_complete.json`，状态`L0_V2_REDACTED_QUERY_TOKEN_CACHE_READY`。
- 原gallery：
  `/hkfs/work/workspace/scratch/ap7811-benchmark/colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt`，
  注册SHA `11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc`。
  原物理5413行、修复后5412 identities的谱系不变；本次没有读取该二进制文件。

## 历史RAW600候选确实存在，但不能当作新cohort的完整覆盖

历史RAW full-gallery分数/C128：
`results/dino_rcde_p0_v1_1/formal_job5068254/base_prejoin.pt`，注册SHA
`ba1df745e5986abfcad3e82924ec9b745c076126c4666de907495642c5980691`。
其receipt明确为600条RAW、label reads0、无target插入。

`programs/run_dino_rcde_p0_v1_1.py`显示它将原image+template tokens对
完整原gallery评分；`run_dino_rcde_p0_data_v1_2.py`独立重建C128。
对应匿名轴文件为
`results/dino_rcde_p0_v1_2/formal_job5069436/data/model_visible_c128.json`，
SHA `784d196f4f05abddbca13eab8aaa88f1c2cbc8cf48cd9db3e929bf68c5803a81`；
完整token/坐标谱系另见`cache/dino_rcde_colnomic_sr_full600_geometry_v2/`。
这个geometry缓存本身不是600条完整RoMa visibility缓存。

经过全部排除后，这个历史池只覆盖30张。其它237条记录需要新的
result-blind RAW ranking/C128步骤，或先证明另一个允许的RAW候选源。
历史缓存与当前fresh-runtime处理的token/候选parity也不能仅凭相同
模型名字假定；应在新输入方案中固定复用或桥接方式。

不要把`conditional_colnomic_p_spatial_v1`的775条token可用性误当作
775条当前RAW C128资格：其candidate prejoin走的是另一个D1绑定谱系。
987条redacted原始token可以独立使用，候选来源须另行闭合。

## 90+31的旧C4标量可用，visibility tensors缺失不阻止原七参数头

另外两个已打开旧panel与当前128张、RAW987图片hash均不重叠：

| panel | 图片 | 已验证的RAW token/C128 | 已验证的full128 C4 | 本次旧43组资格 |
|---|---:|---|---|---|
| difficult90 | 90 | `romav2_colnomic_difficult90_token_fullrank_prejoin_v1` | `romav2_colnomic_difficult90_real_prejoin_v1` | 待单独身份/group桥接 |
| new_difficult test primary | 31 | `romav2_colnomic_new_difficult_sealed_token_prejoin_v1`及`...fullrank_prejoin_v1` | `romav2_colnomic_new_difficult_sealed_roma_prejoin_v1` | 待单独身份/group桥接 |

各目录`validation.json`均存在且相应checks全部通过。原RoMa writer只
保存S、M、SQ、SR及query/reference map SHA，不保存wq/wr数组。
这些标量经来源与公式parity闭合即可服务**当前ORIGINAL7**，不需要
为它们重做J/T/LP。只有再评价新J/T等机制时才需要visibility数组补全。
这121张不能在身份/group资格完成前直接算入新评价人数。

## 可改编的程序/launcher与head边界

- 当前RAW token/full-gallery/C128步骤：
  `programs/materialize_romav2_colnomic_current_runtime_bridge_shard_v1.py`；
  launcher为`slurm/romav2_colnomic_current_runtime_bridge_prejoin_remaining_v1_30m.sbatch`。
  它原先固定64条选择，扩展时应新建文件，用新冻结cohort替换该选择。
- RoMa+C4步骤：
  `programs/materialize_romav2_colnomic_current_runtime_bridge_roma_shard_v1.py`；
  launcher为`slurm/romav2_colnomic_current_runtime_bridge_roma_remaining_v1_30m.sbatch`。
  difficult90对应的token/fullrank和REAL shard程序/30分钟launcher也可作为
  任意新cohort分片实现的参照；全部确切hash已写机器盘点。
- C4 scorer：`programs/run_romav2_colnomic_visibility_xf_six_case_v1.py`中的
  FP64 weighted full-reference MaxSim；原RAW full-gallery辅助函数在
  `programs/run_romav2_colnomic_sealed_source_e0_v2.py`。
- 主head必须用
  `registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json`
  或已验证5139151的**ORIGINAL7**权重/bias，再走完整127个`X@w+b`。
  `romav2_colnomic_frozen_gate_v1.py`的`candidate_feature`可复用；该模块
  自带的`WEIGHT/BIAS/logit`属于历史FROZEN_C、不是当前28/32的ORIGINAL7。

新cohort必须在任何C128结果产生前冻结；不能因target不在C128而移除
query。旧32回归、新≥128识别总体、candidate recall、各预定控制应
分别记录。当前这一步没有放行或执行扩大评价。
