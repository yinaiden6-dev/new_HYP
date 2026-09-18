# 模型与原始数据恢复

## 大模型权重

权重放在同仓库 Release `mainline-backup-20260919`，不是 Git LFS。每个附件小于 2 GiB；当前采用 1 GiB 原始字节分片，不改变精度、格式或模型参数。

完成上传后，Release 包含：

- ColNomic 7B 的 LoRA adapter；
- 原实际使用的 `colqwen2.5-7B-base` 七个权重分片；
- RoMaV2 2.0.1 checkpoint，其默认 DINOv3 描述子权重随 RoMa checkpoint 恢复。

精确文件列表以 `model_assets_manifest.json` 为准。模型配置、tokenizer 与小型头参数直接保存在 Git 文件中。原始 `adapter_config.json` 内仍含旧的绝对 base 路径：在新机器恢复时需要映射旧目录或在单独运行副本中修改为本地 base 目录，保留归档原件。

GitHub CLI 登录有权访问本私有仓库的账号后，在仓库根目录运行：

```bash
python tools/restore_models.py --download
```

脚本会下载分片，逐片校验 SHA256，合并并校验整个文件，然后放回原 workspace 的相对路径；默认清理已成功恢复的临时分片。已存在且校验正确的文件直接跳过，不覆盖不同内容的模型。需要约模型总量加最大一个模型原文件的临时空间。

GitHub 的 “Download ZIP” 仅下载仓库文件，不自动包含这里的 Release 权重。

## 小头参数

五套全 H593 冻结头在 `ICLR/new ROUTEA/RC/results/rc_new_hyp_external_head_freeze_v1/{COST1,CE,COST4,GROUP_COST4,RAW2_CE}/head.json`。其中 `theta_hex` 保留 FP64 参数的精确十六进制值；对应验证记录一并保存。当前可定位的历史 JSON 参数文件也随各实验结果保留，不能与外部固定头混用。

`heads/oof_h593/` 另存三组各五折的参数记录：原 H593 头、GROUP 基线、COST1/CE 等损失对照。它们从原封存载荷抽取参数，记录源文件 SHA；不携带混在载荷中的历史预测缓存，也没有重新训练。张量以 dtype、shape 和精确十六进制浮点值表示。

## 原始图片另存

根据用户选择，本仓库不复制原始 query/reference 图库。展示图中的示例照片属于展示材料，不等于完整数据备份。

`backup/raw_image_manifests.csv` 指向已保存的原始图片清单，包括 H593、processed128 以及相应外部面板的 worker/gallery/asset manifests。清单包含旧路径、身份或来源信息及已有图片哈希；它不替代图像字节本身。

原 workspace 根目录部分软链接指向已经过期的旧 `ap7811-dailymed`。完整数据迁移时还应独立核对这些链接及其可恢复的数据，不能只复制链接就认定图片已经保存。

## 仍未保存的内容

历史 token/特征/RoMa 中间缓存、历史训练载荷中的缓存预测数组、err/out/log、虚拟环境、原始大数据集和不属于当前路线的模型未上传。恢复模型并不自动恢复这些缓存。受保护的 D1-MI/formal392 科学结果未打开或打包。
