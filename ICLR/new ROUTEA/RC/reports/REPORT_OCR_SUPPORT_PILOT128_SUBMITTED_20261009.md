# OCR双模态空间支持首轮：已提交

2026-10-09。首轮是H593 execution ordinal 0–127，沿用自然C128与原分组五折；不是旧EVAL128，也不是全593结论。

已核对真实query/reference：29／21文字框，识别分数、EXIF坐标、源图SHA、RoMa框内支持和M重算均通过；这是接口资格，不是识别增益结果。首图OCR推理12.18秒，首reference5.34秒，不含初始化，不能直接外推全部图片时长。

统一使用本地PP-OCRv5_server_det＋英文en_PP-OCRv5_mobile_rec；PaddleOCR3.4.0、PaddlePaddle3.3.1。涉及3885张去重图、3793个reference物理行。全部只做一次CPU OCR，保留文本、多边形、识别置信度、检测输出与SHA。复用原RoMa u/v与ColNomic评分，不重跑两个主干。

五个学习臂：仅再校准；TEXT；TEXT_M；TEXT_UV；TEXT_UV_SHUFFLED。每折TRAIN与held都从同一原外折COST1评分起步，再只用128面板TRAIN组训练残差头。固定COST1损失／2000步／0阈值，保存完整127挑战分数。主要比较真实空间支持对无空间支持、打乱绑定的增量，不能把单纯OCR涨点称为空间机制成功。

| 阶段 | Job ID |
|---|---|
| smoke | 5188667 |
| ocr | 5188668 |
| index | 5188669 |
| features | 5188670 |
| fit | 5188671 |
| join | 5188672 |

OCR32片，特征回放16片，训练5折；4CPU、24GB、每次10分钟、断点续跑。首个资格任务dev_cpuonly/cpuonly，其余cpuonly。所有依赖与实际资源核对通过，Slurm保存脚本逐字节复核通过。本次只提交和归档，不持续监控。

准备阶段修复：Paddle临时缓存改到独立workspace目录；图像路径为软链接，源校验改为比较实际文件内容SHA而非要求字符串等于resolve路径。原数据和科学对照未改变。

新增结果待五折及汇总验收。已有593上游归因链未修改。首次实验的文字模型、精确词匹配和960边长预处理均是本轮实现边界；负结果不等于OCR整体无效。
