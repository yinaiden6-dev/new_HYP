# 最终逐图视觉证据整合



稳定ledger1117例；原图SHA逐例验证；原图与关键候选overview实际审核1117/1117。

场景计数：{'background': {'single_object': 858, 'multi_object': 253, 'unclear': 6}, 'placement': {'tabletop': 601, 'handheld': 516}, 'occlusion': {'none': 1072, 'foreground': 44, 'unclear': 1}}；每维1117已实际审看，6例背景/1例前景仍不确定。

鼠标、白色设备与独立白纸统一按额外独立实体计入multi_object；实际原图补查8例修订保留于scene_supplements/foreground_object_definition_recheck.json。原标注及原审核history均保留。

scene_revisions.json保存首次完整35评分快照的原字节；scene_revisions_final.json供最终审后场景统计读取。

8例前景遮挡融合损害追加实际局部与正负候选规格审核：区分规格被挡yes3、none4、unknown1。foreground与关键区域/区分规格遮挡独立，关键区域未挡不能转成无前景。

明确细节核验13例、选定细节观察85例；只认可每条细节声明的实际范围，overview不等于全文验证。

同面控制逐失败配置保存B03/F03列表，37失败Query有经实际配对确认控制；质量、角度、尺度及复用混杂保留。未确认同面不代表同面成功图不存在。

quality字典先解析值的否定/未知/零值再映射类别；原粗标签/quality原字段不删改，归一化仍是未校准的描述性多标签。

records.json保留全部实际观察、修订来源SHA/图片SHA及候选ledger。机制结论须另接评分、Token与干预证据。
