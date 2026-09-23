# ColPali缓存v3：修复视觉参数核对

v2首片5160327_0的Transformers加载报告已没有missing/unexpected/mismatched/error；程序自身没有把SigLIP视觉模型自动去掉`vision_model`包装层的改名纳入参数集合核对，误报437项未映射。这是新增检查器的错误，不是额外437项权重未加载的证据。

v3保留语言模型键名修复，补充检查器中的视觉名称转换；新增对全部safetensors头部shape与实际模型shape的核对，不修改模型权重、不删除加载检查、不放宽三个旧reference锚点阈值。

提交前回放v2保存的失败参数集合，v3转换后的605项与上次实际模型605项完全一致；437项视觉转换无重复、无遗漏。checkpoint索引与两个safetensors文件头部键集合一致。这是检查器回归验证，不能替代GPU数值兼容性验证。

输出隔离在`results/rc_colpali_h593_query_tokens_v3_compat/`，原v1/v2失败产物保留。输入manifest仍与v1逐字节相同。

- 5160390：dev_accelerated，首片0，15分钟。
- 5160391：accelerated，1–49片，%50，每片15分钟，afterok首片。
- 5160392：全量成功后CPU汇总。

新作业spool与脚本逐字节核对通过；旧DependencyNeverSatisfied作业5160328/5160329已取消。仅在首片权重、reference锚点和12张query缓存完整验收后，才会自动推进其余49片。

截至提交后核对：首片尚在Priority排队，后续为Dependency；尚无GPU PASS。此链完成的是跨检索器实验所需的593张query token采集，不是跨检索器准确率结果。

## 首片执行后更新

5160390_0已完成权重加载与全部参数shape核对：605/605、所有issues为空。但三个旧reference锚点数值对比仍失败，relative L2分别0.31009、0.33790、0.22550，远超原0.03阈值。query仍无产物，后续未被放行；这次定位到完整加载后的推理环境兼容性问题，不再是缺失权重或检查器漏算。

根据历史`colpali/run_colpali_retrieval_detail_raw_gallery_horeka.sbatch`的`conda activate colpali`记录，下一验证隔离使用历史Conda环境（本地torch2.5.1、Transformers4.57.3），保持checkpoint、原图、BF16计算与旧锚点阈值不变。新程序`cache_colpali_h593_queries_v4_legacy.py`，输出`results/rc_colpali_h593_query_tokens_v4_legacy/`。

替代链：5160396首片dev_accelerated、5160397后续49片accelerated/%50、5160398全量汇总。旧失效依赖5160391/5160392已取消；旧验证产物保留。该环境验证尚待实际GPU结果，不能从代码修改推断兼容通过。
