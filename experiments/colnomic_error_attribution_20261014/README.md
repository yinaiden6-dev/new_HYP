# 本轮完整错误分析与归因实验（2026-10-14）

封存验收已完成。本目录提供按原始 SHA 复制的报告、代码和数值回执；同仓库 Release 保存完整 33,289,484,786 字节归档，包括 6,522 张原图、原始 embedding、全部矩阵、历史与新增对照材料。

- [最终报告](workspace/results/colnomic_full_audit/20261013_resume/REPORT.md)
- [最终协议](workspace/results/colnomic_full_audit/20261013_resume/FINAL_PROTOCOL.json)
- [实验代码](workspace/scripts/colnomic_mechanism/)
- [最终封存验收](acceptance/FINAL_GOAL_ACCEPTANCE.json)
- [完整成员 SHA 清单](acceptance/archive_manifest.json)
- [完整数据 Release](https://github.com/yinaiden6-dev/new_HYP/releases/tag/colnomic-error-attribution-20261014)

验收结果：55,142 个源文件无漂移；实际解包核验 55,149 个文件（含 7 个归档生成成员），另有 1 个成员清单；gzip 完整性通过。1,117 张 Query × 35 配置的 39,095 条排名全部回放一致，5,405 张图库轴一致；34,483 条包内相对链接有效。主交付 12,358 条链接与 30 份数值回执已核对。

归档 SHA256：`de78b51b70ab25d619cdb23c92dd6022f6cd9ca7fbccf3e590119f66cbb0e279`。

## 下载、恢复与复核

恢复脚本需要 Python 3.12 或更新版本。完整归档按原始字节切成 32 片，每片至多 1 GiB；分片不是独立压缩包，按清单顺序拼接可恢复完全相同的 tar.gz。

从本目录运行：

```bash
python restore_archive.py --download --extract-to restored_workspace
cd restored_workspace
python -m pip install -r requirements.txt
python scripts/colnomic_mechanism/package_full_audit.py --verify-extracted --axes --links
bash results/colnomic_full_audit/20261013_resume/REPRODUCE.sh --verify-only --quick
```

恢复脚本核验每片大小和 SHA256，再核验拼接后的整包 SHA256，并只解压到空目录。首次下载和恢复需要为分片、压缩包及解包后的文件准备足够磁盘空间。完整报告的图片、矩阵与案例页面链接应在恢复后的工作目录中打开；`workspace/` 是便于 GitHub 浏览的选择性副本，其逐文件来源见 `publication_manifest.json`。`--quick` 入口的范围是核心快速核验加完整新增参考对照，原完整核心数值复核回执也封存在归档中。

## 保留的解释边界

语义 ROI 只对 4 张重点图精确定位；全量区域分区守恒不等于全部身份关键 ROI 已定位。旧图库冻结缓存和排序可复核，但缺少旧 processor、中间层或上游文件，历史图库编码来源及漂移原因未完全复现。跨视角布局 pilot 只有 3 个身份，没有 held-test；新 Query 对旧图库条件排名不作为新编码器全图库准确率。完整审计不表示全部错误的唯一因果机制已经确定。

模型大权重和私密配置未封存。原报告、代码、预测和回执按字节保留；封存后的最终状态来自 `acceptance/FINAL_GOAL_ACCEPTANCE.json`，归档内的封存前状态不会为上传而改写。完整限制见最终报告和验收回执。
