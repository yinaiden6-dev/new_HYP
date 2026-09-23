#!/usr/bin/env python3
"""Post-join probability/ranking readout; do not fit or select any model."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_h593_conditional_rank_v1'

def read(path):return json.loads(Path(path).read_text())
def bind(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def checked(source):
    assert bind(source['path'])==source
    return Path(source['path'])

def main():
    validation=read(OUT/'validation.json');assert validation['status']=='CONDITIONAL_RANK_ALL_COUNTS_PASS'
    result=read(checked(validation['result']));predictions={}
    for f in range(5):
        v=read(checked(validation['fold_validations'][f]))
        p=read(checked(v['payload']))
        predictions.update({r['query_id']:r for r in p['predictions']})
    models=('COST1_FULL','CE_FULL','RANK_ONLY6')
    losses={m:defaultdict(float) for m in models};mrr={m:[] for m in models}
    present=defaultdict(int);wrong40=[]
    for row in result['rows']:
        if row['target_present']:present[row['fold']]+=1
        if not row['ranking_eligible']:continue
        if not row['target_is_top']['COST1_FULL']:wrong40.append(row)
        for m in models:
            z=[float.fromhex(v) for v in predictions[row['query_id']]['models'][m]['logits_hex']]
            rank=row['target_challenger_rank'][m]
            target_score=sorted(z,reverse=True)[rank-1];peak=max(z)
            losses[m][row['fold']]+=math.log(math.fsum(math.exp(v-peak) for v in z))+peak-target_score
            mrr[m].append(1/rank)
    assert sum(present.values())==570 and len(wrong40)==40
    readout={m:dict(OOF_conditional_CE_per570=math.fsum(losses[m].values())/570,
                   challenger_MRR_per144=math.fsum(mrr[m])/144,
                   OOF_conditional_CE_by_fold={str(f):losses[m][f]/present[f] for f in range(5)}) for m in models}
    shifts=dict(improved=sum(r['target_challenger_rank']['RANK_ONLY6']<r['target_challenger_rank']['COST1_FULL'] for r in wrong40),
                same=sum(r['target_challenger_rank']['RANK_ONLY6']==r['target_challenger_rank']['COST1_FULL'] for r in wrong40),
                worse=sum(r['target_challenger_rank']['RANK_ONLY6']>r['target_challenger_rank']['COST1_FULL'] for r in wrong40),
                new_top=sum(r['target_is_top']['RANK_ONLY6'] for r in wrong40))
    payload=dict(status='CONDITIONAL_RANK_POST_JOIN_RECOUNT_PASS',source=validation['result'],program=bind(__file__),
                 models=readout,old40_rank_shift_vs_COST1=shifts,model_fits=0,
                 interpretation='OOF conditional loss and challenger MRR improved; first-place identity count did not. This does not establish generic overfitting.')
    (OUT/'post_join_analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    report=ROOT/'reports/REPORT_H593_CONDITIONAL_RANK_V1_20260921.md'
    content=report.read_text().split('## 补充：损失与第一名的区别')[0].rstrip()
    lines=['','', '## 补充：损失与第一名的区别','',
           '| 排序头 | 外层条件CE／570个候选在场query | 挑战者MRR／144个RAW错误query |', '|---|---:|---:|']
    for m,s in readout.items():lines.append(f"| {m} | {s['OOF_conditional_CE_per570']:.8f} | {s['challenger_MRR_per144']:.8f} |")
    lines+=['','RANK_ONLY6的TRAIN条件CE五折均下降，TRAIN最高候选命中相对CE为+2/+2/0/+2/+2。外层条件CE也五折均下降，整体下降约6.87%，挑战者MRR略升，但top从104降到103。因此不能简单写成“只在训练集有效”或“没有任何跨组改善”；更准确的结论是已有概率及次级排序改善，尚未转为更多第一名。', '',
            '原40个COST1排序障碍中，12个排名改善、20个不变、8个下降，0个升到第一名。MRR是127挑战者内部诊断，不是完整检索流水线MRR。', '',
            '这支持下一步用同六维、纯target与最强错误margin的损失作单因素对照；并不保证该目标会成功，不新增任何最终准确率。补充证据：`post_join_analysis.json`。','']
    report.write_text(content+'\n'.join(lines))
    print(json.dumps(payload,ensure_ascii=False))

if __name__=='__main__':main()
