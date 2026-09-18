# ORIGINAL7扩展EVAL128：仅元数据的固定选样与隔离

本文件在新面板的任何RAW排名、RoMa或识别结果之前固定选样。依据用户
“128或者更多”及registry/rc_retrieval_only_eval128_expansion_authority_addendum_v1_20260910.json，
主比较为冻结ORIGINAL7对RAW；旧EVAL32单列回归，不混入这128张。
本文件只冻结输入和元数据，不授权自动启动自然评分。

输入为已登记inventory的267条、266个独立image SHA、24个identity、21个
supergroup。排除当前PAIR64+FULL TRAIN32的32个identity/group，以及旧
EVAL32的11个identity/group。正式392只作为公开query ID/image SHA负名单
排除；不读取其图像、embedding、私有角色或P/RoMa结果。

固定seed字符串：RC_ORIGINAL7_EVAL128_20260910。
排序key为SHA256(UTF8紧凑JSON数组[seed,domain,*parts])，ensure_ascii=False；
hash相同时再用相应原字符串排序。先按image SHA去重，若同一SHA的
identity或group冲突则整体中止。标签一致的重复记录按domain='record'
及query_id的key取首条，不使用缓存是否已有C128、任何分数或target presence。

group按domain='group',group_id排序；每个group内identity按
 domain='identity',group_id,identity排序；每个identity内image按
 domain='image',image_sha排序。依次轮转所有group；每次取该group的下一个
尚有图片的identity的一张图片；组内identity同样轮转。耗尽的identity/
group退出其队列，持续至128张。不得因后续候选召回失败替换或删除。

必须验证128个独立image SHA、覆盖全部21个group和24个identity，且与43
个旧identity/group及正式392的ID/SHA均无交集。完整名单按选择顺序冻结。
这属于保守标注历史曝光的扩展held-out development，不称untouched external。

worker manifest使用opaque query IDs及匿名image/token文件别名，不含target、
identity、group或原图片文件名。别名仅创建symlink，不读取或复制图像/token
payload；原文件SHA按已有元数据声明，执行前还须实际校验。原source路径、
query IDs、target identity、supergroup和标签metadata来源只在单独curator
ledger中。scorer不得读取该ledger；所有预测封存后才进行标签join。

独立新进程从RAW987元数据重新核验最终eligible集合及排除条件，检查重复
标签一致性，用另一种按轮次数展开的实现复现选样，再核验worker/curator
对应和别名。验证不读取图像、embedding、ranking、candidate presence或
任何新评估结果。原模型/旧结果不改写，不自动提交作业。
