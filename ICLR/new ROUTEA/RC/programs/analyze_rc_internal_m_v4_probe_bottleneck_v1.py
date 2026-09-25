#!/usr/bin/env python3
"""Post-hoc explanation of sealed PROBE8 results; no fitting or new inference."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/rc_internal_m_v4_probe_v1'
OUT = SOURCE / 'posthoc_bottleneck'


def load(p):
    return json.loads(Path(p).read_text())


def binding(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def verify(b):
    assert binding(b['path']) == b
    return Path(b['path'])


def main():
    v = load(SOURCE / 'validation.json')
    result = load(verify(v['result'])); verify(v['report'])
    a = load(verify(v['authority']))
    m = load(verify(a['manifest']))
    manifest = {r['query_id']: r for r in m['probe_rows']}
    seal = load(verify(result['prelabel_seal']))
    for b in seal['predictions'] + seal['decisions']: verify(b)
    rows = []
    for old in result['rows']:
        q = old['query_id']; r = manifest[q]
        targets = [i for i, identity in enumerate(r['candidate_identities']) if identity == old['identity']]
        item = dict(query_id=q, identity=old['identity'], component=old['component'],
                    target_present=bool(targets), raw_correct=old['correct']['RAW'], content={}, action={})
        audit = load(SOURCE / 'audited_decisions' / (q + '.json'))['predictions']
        if targets:
            assert len(targets) == 1
            t = targets[0]
            for mode in ('REAL', 'REAL_CONSTANT', 'REAL_SHUFFLED', 'TRAIN_CONSTANT', 'TRAIN_SHUFFLED'):
                p = load(SOURCE / 'predictions' / mode / (q + '.json'))
                L = p['L']; assert len(L) == 128
                item['content'][mode] = dict(target_score=L[t], target_rank=1 + sum(x > L[t] for x in L))
            for mode, pred in audit.items():
                s = pred['scores128']; assert len(s) == 128
                wrong = max((i for i in range(128) if i != t), key=lambda i: s[i])
                best_challenger = max(r['challenger_positions'], key=lambda i: s[i])
                item['action'][mode] = dict(target_score=s[t], strongest_wrong_score=s[wrong],
                    target_margin=s[t]-s[wrong], target_rank=1+sum(x>s[t] for x in s),
                    target_is_best_challenger=t==best_challenger,
                    selected_identity=pred['prediction_identity'], correct=old['correct'][mode])
                assert old['correct'][mode] == (pred['prediction_identity'] == old['identity'])
            if old['correct']['RAW']:
                item['failure_type'] = 'RAW correct retained'
            elif old['correct']['REAL']:
                item['failure_type'] = 'Internal rescue'
            elif item['action']['REAL']['target_is_best_challenger']:
                item['failure_type'] = 'Target is best challenger but not accepted over HOLD'
            else:
                item['failure_type'] = 'Competing wrong challenger outranks target'
        else:
            item['failure_type'] = 'Target absent from natural C128'
        rows.append(item)
    out = dict(status='SEALED_PROBE8_POSTHOC_BOTTLENECK_AUDIT_PASS', source_result=v['result'],
               source_prelabel_seal=result['prelabel_seal'], program=binding(__file__), rows=rows,
               fitting_updates=0, new_inference=0, new_thresholds=0,
               evidence='Post-hoc explanation of eight previously opened probe images across six components; not model selection')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    lines=['# V4 PROBE8：内部内容变化与最终纠错的分界','',
           '本次只读封存预测，未训练、未调整阈值、未新增推理。原任务5162476正常完成，NumPy核算误差约3.55e-15。',
           '8张来自6个component；目标在自然C128内7张。RAW与内部REAL均4/8、0救0损；外部ADDITIVE4和PRODUCT5均6/8、2救0损。','',
           '| RAW错误query | 身份 | REAL内容目标排名 | 同模型恒定M排名 | 同模型错绑M排名 | REAL动作目标排名 | 定位 |',
           '|---|---|---:|---:|---:|---:|---|']
    for row in rows:
        if row['raw_correct']: continue
        c, ac = row['content'], row['action']
        if not row['target_present']:
            vals=['—']*4
        else:
            vals=[str(c[k]['target_rank']) for k in ('REAL','REAL_CONSTANT','REAL_SHUFFLED')]+[str(ac['REAL']['target_rank'])]
        lines.append('| '+ ' | '.join([row['query_id'],row['identity'],*vals,row['failure_type']])+' |')
    lines += ['', '## 解释', '',
      '- 766-e9-infants-pain-and-fever：真实M使内容目标排名达到2，恒定/错绑分别39/70；但联合头动作目标仅第6。TRAIN重训头后目标分数虽为正，错误挑战者仍更高。因此不是单独降低接受阈值能修好的样本。',
      '- cefdinir-fig2：真实M下内容目标第1，恒定/错绑均第3；正确目标也是最佳挑战者，但动作分数仍为负，HOLD保留了错误RAW。联合头目标分数约-0.385，TRAIN重训头约-0.0765。这里表现为接受校准未将较好排序转为纠错。',
      '- alp0b-0002-29：内容目标仍第2，内部及外部均未纠错。另1张目标缺席C128，与内部读出是否有效是不同限制。',
      '', '因此并非内部M在probe上完全没有作用：至少两例的目标内容排名改善；但这次没有增加最终正确数。',
      '这种候选条件内容分数的变化，不自动证明patch语义表示更优或局部注意力学到了身份线索。',
      '这里只能定位本小面板的竞争排序与接受校准两个环节；不能据此手调probe阈值，也不能称为已经找到所有泛化失败的统一根因。',
      '若继续模型开发，应仅在TRAIN内研究完整候选竞争与接受校准，在单独冻结的评估上验证；本轮不自动提交新的训练。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=out['status'],report=str(OUT/'report_zh.md'),raw_errors=sum(not r['raw_correct'] for r in rows)),ensure_ascii=False))


if __name__=='__main__':main()
