#!/usr/bin/env python3
"""Join the fixed 71 panel; omit an optional preexisting metric missing four caches.

The prespecified new pooled readouts all have complete 71x128 scores. Original
full-token Col scores are auxiliary, available in only 67 original snapshots.
Preserve those values in the sealed per-query files, and omit this incomplete
auxiliary metric from the all-71 comparison; do not impute or shrink the panel.
"""
import json
from pathlib import Path
import analyze_rc_pair_relation_feature_readout_v1 as original

original_read=original.read
def complete_metric_read(path):
    value=original_read(path)
    if isinstance(value,dict) and 'scores' in value and 'query_id' in value:
        value['scores'].pop('COL_FULL_FORWARD_MAXSIM',None)
    return value

if __name__=='__main__':
    original.checked(original_read(original.OUT/'spec.json')['program'])
    source=original_read(original.OUT/'score_seal.json')
    available=sum('COL_FULL_FORWARD_MAXSIM' in original_read(b['path'])['scores'] for b in source['predictions'])
    original.write(original.OUT/'join_spec.json',dict(
        status='COMPLETE71_PRESPECIFIED_METRICS_ONLY',program=original.bind(__file__),
        source_score_seal=original.bind(original.OUT/'score_seal.json'),
        excluded_auxiliary_metric='COL_FULL_FORWARD_MAXSIM',available_query_count=available,
        reason='Original snapshot has initial full-token scores for67 queries only; new prespecified pooled methods have complete71 and are all retained',
        new_labels_read_before_this_join=False))
    original.read=complete_metric_read
    original.summarize()
    report=original.OUT/'report_zh.md'
    report.write_text(report.read_text()+'\n原full-token Col自由分数仅在67张原缓存存在，仍保留于逐query封存文件；为统一71张比较，表中不纳入该辅助列，未删除任何query。\n')
