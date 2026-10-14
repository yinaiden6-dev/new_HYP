# 可观察困难与35配置交集

1117张独立图，39095次重复评测；原图image对照按已核实的输入合同选择：B01, B02, B03, B04, B21, B22, B24。

```json
{
  "all35_fail": {
    "queries": 25,
    "identities": 18
  },
  "any_method_oracle_correct": {
    "queries": 1092,
    "identities": 130
  },
  "fusion_introduced": {
    "queries": 41,
    "identities": 23
  },
  "B03_fail_others_all_original_image_correct": {
    "queries": 0,
    "identities": 0
  },
  "B03_fail_at_least_one_original_image_correct": {
    "queries": 125,
    "identities": 56
  }
}
```

all35_fail 表示35种设置均Top1错误；有一个方法正确的oracle需要真值，不能当作系统成绩。B03特有的两个字段严格限定为相对于列出的原图image设置，并不证明ColNomic某层缺陷。

逐配置/困难表分别保存类内失败率、该配置错误构成、困难发生率；多标签允许重叠。source-group配对bootstrap对查询聚类，但未消除身份、可见面、清晰度和尺度混杂。

同一场景中的不同差额来源、跨场景重复的融合错误及匹配/汇总响应，沿每图Token链接和总主档案追溯；无法识别机制的照片保留未知。
