# ColPali / ColQwen 跨编码器机制检验：缓存可行性

日期：2026-09-23。状态：完成只读缓存核查；尚未提交跨编码器训练或编码作业，没有新增准确率结果。

## 要回答的问题

固定 query–reference 图像对、候选集合和分组划分，将内容编码器由 ColNomic 换成 ColPali，检验 RoMa 配对质量与内容联合校准是否仍能纠错。该问题检验机制对内容编码器的可迁移性；不替代当前 ColNomic 表示与读出能力的六臂及 M 教师诊断。

## 实际核查

1. 本地存在 ColPali 模型快照：`/hkfs/home/project/hk-project-pai00054/ap7811/.cache/huggingface/hub/models--vidore--colpali-v1.3-hf/snapshots/7d3c8ab1c1908b32d701308fb1dfb2968d150c67`。历史检索元数据也指向该快照。配置是 `ColPaliForRetrieval`，1024 个图像 token，128 维输出，448×448 输入、14×14 patch。
2. `colpali/result/difficult/raw_gallery/cache/colpali_gallery_emb.npz` 含 `passage_emb` 与 `setids`。嵌入形状为 **5413×1030×128，FP16**。全部 5413 个 setid 与 RC `results/rc_new_hyp_processed128_regression_v1/gallery_manifest.json` 中图片文件名逐行相同，零错位。该检查是名称及顺序一致性，不是历史图像字节 SHA 一致性证明：旧 NPZ 没有保存原图 SHA。
3. `colpali/result/raw_gallery/cache/colpali_gallery_emb.npz` 是较早的 **5412** 行版本，不能直接用当前物理行号索引。优先使用上述 5413 行版本；不能混用两者。
4. 在 `colpali/` 的持久化嵌入文件和原检索程序中，找到的是 gallery 缓存。程序在 query 前向后直接评分，未保存 query tokens；历史 JSONL 保存 Top-10 身份和耗时，不能恢复完整 C128 分数或 token 匹配矩阵。因此尚不能只依靠这些文件完成替换实验。
5. `colqwen/difficult/raw_gallery/run_colqwen_retrieval_detail_name_metrics.json` 指向 `models/downloaded_models/colqwen2.5-base`。该模型本地 README 明确是未做检索训练的 base，用于确定性初始化投影层；权重索引包含 `custom_text_proj`。其视觉语言底座已有预训练，不能描述成整个 Qwen 从未训练。
6. 本地另有 `models/downloaded_models/colqwen2.5-v0.2` 检索训练 LoRA 及 `colqwen/trained_v0.2/` 隔离目录。本次该目录未发现完成的结果 JSON/JSONL 或特征缓存；不能把 base 缓存归到 trained-v0.2 名下。

## 复用范围和缺口

| 数据 | 可复用范围 | 缺口/边界 |
|---|---|---|
| 原 H593 RoMa 配对质量 M | 相同图片、相同候选图像对 | 保留原计算定义及源 SHA，不从 ColPali tokens 推算 M |
| query/reference 局部质量网格 u、v | 已有二维场可用于映射 | ColPali 是 32×32 图像 patch；需按图像坐标映射，不能直接复制 ColNomic 数组 |
| ColPali reference tokens | 5413 行历史缓存 | 核查源图像版本及 processor；1030 token 不等于 1030 图像 patch，需明确额外 6 个 token 的处理 |
| ColPali query tokens | 本次未找到可用持久化缓存 | 需要对预先固定的小面板补编码一次，并保存 token、input IDs、图像 token mask、网格、源图 SHA、模型与 processor 标识 |
| ColNomic query tokens | 不能充当 ColPali tokens | 替换编码器必须重新取得内容表示 |
| ColQwen base tokens | 仅限 base 附加诊断 | 不能充当 trained-v0.2；两侧必须使用同一 checkpoint、processor 与投影 |

若主张严格保留原七参数机制，需完成局部权重到新网格的映射及控制项重算。先做整体 M×自由内容的实验更容易复用，但应标为简化机制迁移，不能直接叫原 COST1 跨编码器结果。

## 建议锁定的检验

- 优先 **ColPali v1.3 检索训练模型**，ColQwen base 作为后续诊断，不按基线低或预期增幅大挑选主模型。
- 先固定一个小面板及其既有分组 TRAIN/held-out 划分，选样不读取结果；若使用已有第一折，必须明确是 TRAIN474 / held119，而不是只拿119张重新划分后沿用原指标。
- 为直接复用 RoMa，先固定原 ColNomic C128，在这同一候选集合内计算 ColPali 内容基线、ColPali-only learned correction、ColPali+RoMa 同结构 COST1 头，以及质量绑定打乱控制。训练臂使用相同检索标签、更新预算及固定规则。
- 原 ColNomic 头直接迁移可以另外报告；它检验参数迁移，与重训七参数小头检验机制迁移不同，不能混为一项。
- 主要比较每个编码器自己的基线、同监督内容头、联合质量头，报告救回、误伤、净增、分组不确定性以及 C128 缺失数。跨编码器直接比较绝对正确数不能隔离 RoMa 贡献。
- 这首先是 **固定 ColNomic 候选集合内的机制迁移**。若要声称完整 ColPali 检索管线泛化，需要它自己的自然 C128；未覆盖的 RoMa 图像对不能假称已有缓存。
- 本次没有新的跨编码器 GPU 作业；当前 M 教师诊断不因本核查被暂停或修改。

## 补充核查：用户指出 outcome / difficult 有部分缓存

已按这两个旧子集重新读取缓存容器及预测文件，不要求 new difficult 覆盖：

| 子集/路径 | 已有内容 |
|---|---|
| `colpali/result/raw_gallery/`（outcome） | 1132 条 query 检索记录；5412 行 reference token 缓存 |
| `colpali/result/difficult/raw_gallery/` | 224 条 query 检索记录；5413 行 reference token 缓存 |
| `colpali/more gallery/outcome/` | 1132 条 query 检索记录；15050 行 reference token 缓存 |

三个 NPZ 均只有 `passage_emb.npy`、`setids.npy`。上述 JSONL 字段均为 `query_index`、`query_setid`、`query_path`、`preds`、`time_ms`，不含 token 或逐候选数值分数。因此“已有部分缓存”成立；但在此次实际核查的这些文件中，尚未定位可复用的 query token 缓存。不能把目录名 outcome/difficult 当作容器包含 query embedding 的依据，也不能将未定位写成全 workspace 不存在。没有因此启动全量重编码。

## 扩大搜索记录

用户要求继续查 query tokens 后，已额外检查忽略文件、`new/` 通用 query embedding 命名和 `release/eval/workspace/colpali/` 备份。文件级清单与实际容器键记录见 `reports/colpali_query_cache_search_20260923.json`。`new/cache/query_target_emb_v1.pt` 的内部 `model_name` 为 ColNomic 3B；`new/cache/clean_7b/query_target_emb_7b_clean.pt`、`new/V2.5/cache/query_target_emb_7b_v2_5.pt` 等内部记录 ColNomic 7B。它们证实 workspace 存在 query embeddings，但不是已证实的 ColPali query tokens。没有为此提交 GPU 补编码。

## ColQwen base 的解释边界

base 上可能增益更大，也可能更小。更大的错误集合不等于更大的可纠正集合；质量证据不能保证恢复内容表示未区分的身份。若只有 base 上显著增益，可以说明该机制能辅助这个较弱检索读出，不能据此证明更强的跨编码器普适性。经过检索训练的 ColPali 上的同协议增益，是当前更直接的机制迁移证据。
