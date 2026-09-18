# new HYP 展示材料已完成

2026-09-15。15页16:9演示稿、逐页中文讲义、Excel统计表和45张PNG/PDF/SVG独立图已完成；数据来自封存结果，未训练、推理或改变阈值。

- [离线演示入口](figures/new_hyp_showcase_20260915_v1/index.html)：方向键翻页，N切换讲解词，F全屏。
- [PowerPoint](figures/new_hyp_showcase_20260915_v1/new_HYP_presentation_15slides.pptx)：高清图页，15页可编辑演讲者备注。
- [演示PDF](figures/new_hyp_showcase_20260915_v1/new_HYP_presentation_15slides.pdf)。
- [逐页讲义PDF](figures/new_hyp_showcase_20260915_v1/new_HYP_lecture_zh.pdf)；[完整Word讲义](figures/new_hyp_showcase_20260915_v1/new_HYP_lecture_zh.docx)含术语、统计表与FAQ。
- [Excel统计表](figures/new_hyp_showcase_20260915_v1/new_HYP_statistics.xlsx)：33行外部模型结果、配对比较、案例清单、ISIC31救回分数和历史谱系；另附CSV。
- [全部页面缩略图](figures/new_hyp_showcase_20260915_v1/all_slides_contact_sheet.jpg)。
- [完整ZIP](figures/new_hyp_showcase_20260915_v1/new_HYP_showcase_complete.zip)。

内容包括reference真实token编码、完整推理流程、retrieval-only训练、准确率柱状图、救回/损失、候选绑定、分组区间、ISIC31例HOLD到正确SWITCH、真实成功与失败图片，以及实际缓存的query可见性网格。可见性网格不解释为空间ownership。

主COST1、次CE；GroZi/RPC为范围内外部确认，ISIC为已打开队列探索。旧32/128/90固定头与H593 OOF单列。Ownership、完整过目不忘和新编码器训练继续作为未来工作。

核验：原始计数、MRR、配对救损和分组均值重新计算，区间沿用封存值；原来源哈希不变。已检查PNG和PDF样页、PDF解析、PPTX图页及15页备注、Office压缩包与XML结构、离线HTML翻页/备注按钮逻辑。没有安装Office或浏览器渲染器，不宣称完成真实Office/浏览器端运行测试。核验收据见 [delivery_validation.json](figures/new_hyp_showcase_20260915_v1/delivery_validation.json)。
