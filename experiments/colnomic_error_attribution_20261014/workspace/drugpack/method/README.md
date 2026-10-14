# 参考条件重排

这是原 rebut 老方法的正式实现。冻结 ColNomic 从全图库取候选，RoMa 提供配对覆盖支持；参考条件适配器或外部评分头对候选重排。药盒验证集选择模型，选定权重直接迁移到 GroZi、MILK10k 和 FORB。

- `post.py`：参考条件残差适配器。
- `reader.py`：关系读取器；`heads.py` / `scalar_head.py`：外部评分头。
- `prepare.py`、`hidden.py`、`roma.py`、`features.py`：候选与特征准备。
- `select.py`、`evaluate.py`、`verify.py`：模型选择、评估与核验。
- `legacy/`：运行及原实现数值对照必需的六个原始文件，来源摘要在 `provenance.json`。

配置在 `configs/method/`，缓存在 `cache/method/`，训练检查点及对应训练记录在 `weights/method/`，评估结果在 `results/method/`。路径均相对于项目根目录。

```bash
python -B -m drugpack method --help
python -B -m drugpack method prepare --help
python -B -m drugpack method post --help
python -B -m drugpack method evaluate --help
python -B -m drugpack verify method --help
```

冻结任务文件保留历史模块名与哈希，队列启动时转换到当前模块。历史实验清单中的 `rebut/...` 路径在读取时转换，不需要旧目录或兼容软链接。
