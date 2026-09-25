#!/usr/bin/env python3
"""Independent saved patch means, logits and actions; no model execution.

Full join opens the already-used H593 labels only after all query audits.
Partial audit checkpoints support short CPU jobs without concurrent summaries.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rc_postllm_h593_decomposition_v1'


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8 << 20), b''):
            h.update(b)
    return dict(path=str(p), sha256=h.hexdigest())


def checked(b):
    assert bind(b['path']) == b, b['path']
    return Path(b['path'])


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, p)


def action(row, values, theta):
    raw, w = row['raw_scores'], row['winner_index']
    idx = [i for i in range(128) if i != w]
    mu = math.fsum(raw)/128
    sd = math.sqrt(math.fsum((x-mu)**2 for x in raw)/128)
    logits = [math.fsum([theta[0]*(raw[i]-raw[w])/max(sd, 1e-12),
        theta[1]*(values[i]-values[w])/(abs(values[i])+abs(values[w])+1e-12), theta[2]]) for i in idx]
    j = max(range(127), key=lambda i:logits[i])
    pos = idx[j] if logits[j] > 0 else w
    return dict(prediction_position=pos, prediction_identity=row['candidate_identities'][pos],
        switched=pos != w, logits=logits)


def audit_query(record, row, protocol, pb):
    path = OUT/'queries'/record['query_id']/'result.json'
    sealpath = OUT/'audit_rows'/f"{record['index']:04d}.json"
    if sealpath.exists():
        old = read(sealpath)
        assert old['protocol'] == pb and old['source'] == bind(path)
        return old
    result = read(path)
    assert result['protocol'] == pb and result['query_id'] == row['query_id'] == record['query_id']
    assert result['fold'] == record['fold'] and result['candidate_ids'] == row['candidate_ids']
    assert result['M'] == row['M'] and result['raw_scores'] == row['raw_scores']
    assert result['winner_index'] == row['winner_index'] and len(result['candidate_results']) == 128
    values = {arm:[] for arm in protocol['score_arms']}
    max_patch_error, max_logit_error = 0., 0.
    for i,b in enumerate(result['candidate_results']):
        candidate = read(checked(b))
        assert candidate['protocol'] == pb and candidate['candidate_position'] == i
        assert candidate['candidate_id'] == row['candidate_ids'][i] and candidate['query_id'] == row['query_id']
        assert candidate['M'] == row['M'][i] and candidate['reference_binding'] == row['reference_tokens'][i]
        with np.load(checked(candidate['patch_evidence']), allow_pickle=False) as z:
            assert z['token_arms'].tolist() == protocol['token_arms']
            scores = z['maxsim_by_patch']
            assert scores.shape == z['best_reference_token'].shape
            assert scores.shape == (5, result['query_patch_count']) and np.isfinite(scores).all()
            assert np.max(np.abs(scores[1]-scores[4])) <= 1e-6
            per = {arm:math.fsum(scores[j])/scores.shape[1] for j,arm in enumerate(protocol['token_arms'])}
            per['NATIVE_FIXED_ARGMAX'] = math.fsum(z['fixed_native_by_patch'])/scores.shape[1]
            per['COMMON_FIXED_ARGMAX'] = math.fsum(z['fixed_common_by_patch'])/scores.shape[1]
            assert np.min(scores[1]-z['fixed_native_by_patch']) >= -1e-12
            assert np.min(scores[2]-z['fixed_common_by_patch']) >= -1e-12
            energy = float(np.mean(z['M_specific_hidden_delta_norm']**2))
            spatial = float(np.mean(z['M_specific_hidden_spatial_norm']**2))
            common = float(np.sum(z['common_hidden_vector'].astype(np.float64)**2))
            assert abs(energy-common-spatial) < max(1e-9, energy*1e-5)
            for arm,value in per.items():
                error = abs(value-candidate['scores'][arm])
                assert error < 1e-10
                max_patch_error = max(max_patch_error,error)
                values[arm].append(value)
    actual = {}
    original = {}
    for name,anchor in result['cpu_gpu_anchors'].items():
        original[name] = read(checked(anchor['source']))
        assert original[name]['candidate_ids'] == row['candidate_ids']
    for arm in values:
        saved = result['arms'][arm]['decision']
        assert saved['theta'] == original['NATIVE']['decision']['theta']
        assert saved['challenger_positions'] == row['challenger_positions'] and saved['hold_logit'] == 0
        current = action(row,values[arm],saved['theta'])
        error = float(np.max(np.abs(np.asarray(current['logits'])-saved['logits'])))
        assert error < 2e-10
        max_logit_error = max(max_logit_error,error)
        for k in ['prediction_position','prediction_identity','switched']:
            assert current[k] == saved[k]
        actual[arm] = current
    assert actual['NATIVE']['prediction_position'] == actual['RECOMPOSED']['prediction_position']
    anchors = {}
    for arm in original:
        orig = original[arm]
        anchors[arm] = dict(same_decision=actual[arm]['prediction_position'] == orig['decision']['prediction_position'],
            max_L_error=float(np.max(np.abs(np.asarray(values[arm])-orig['L']))),
            max_logit_error=float(np.max(np.abs(np.asarray(actual[arm]['logits'])-orig['decision']['logits']))))
        assert anchors[arm]['same_decision'] == result['cpu_gpu_anchors'][arm]['same_decision']
    audited = dict(status='SAVED_PATCH_TO_ACTION_INDEPENDENT_PASS',protocol=pb,source=bind(path),
        query_id=record['query_id'],fold=record['fold'],index=record['index'],arms=actual,
        max_patch_score_error=max_patch_error,max_logit_error=max_logit_error,
        anchors=anchors, energy_weighted_common_share=result['energy_weighted_common_share'])
    write(sealpath,audited)
    return audited


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--budget',type=float,default=460)
    args = ap.parse_args()
    started = time.monotonic()
    protocol = read(OUT/'protocol.json'); pb = bind(OUT/'protocol.json')
    for b in protocol['code_sources']:
        checked(b)
    manifest = read(checked(protocol['manifest']))
    audits = []
    records = protocol['rows'][:1] if args.pilot else protocol['rows']
    for record in records:
        if time.monotonic()-started > args.budget-30:
            print(json.dumps(dict(status='PARTIAL_AUDIT',completed=len(audits))),flush=True)
            sys.exit(75)
        row = manifest['rows'][record['index']]
        audits.append(audit_query(record,row,protocol,pb))
    if args.pilot:
        assert all(x['same_decision'] for x in audits[0]['anchors'].values()), 'CPU_GPU_PILOT_ACTION_DRIFT'
        write(OUT/'pilot_validation.json',dict(status='FIRST_FULL_C128_PATCH_SCORE_AND_ACTION_PASS',
            protocol=pb,audit=bind(OUT/'audit_rows/0000.json'),queries=1,held_label_reads=0))
        print(json.dumps(dict(status='PILOT_VALIDATED',query=records[0]['query_id'])),flush=True)
        return
    assert len(audits) == len({r['query_id'] for r in audits}) == 593
    write(OUT/'independent_prelabel_seal.json',dict(status='ALL593_PATCH_SCORES_ACTIONS_AUDITED',protocol=pb,
        rows=[bind(OUT/'audit_rows'/f"{r['index']:04d}.json") for r in audits],held_label_reads=0))
    original = read(checked(protocol['source_result']))
    labels = {r['query_id']:r for r in original['rows']}
    assert set(labels) == {r['query_id'] for r in audits}
    source_rescues = {q for q,r in labels.items() if r['correct']['POST_REAL'] and not r['correct']['RAW']}
    assert len(source_rescues) == 54
    totals = {}
    for arm in protocol['score_arms']:
        correct,rescues,broken,changed,kept = [],[],[],[],[]
        fold_correct = [0]*5
        for r in audits:
            q = r['query_id']; y = labels[q]; p = r['arms'][arm]['prediction_identity']
            good = p == y['identity']
            if good:
                correct.append(q);fold_correct[r['fold']] += 1
            if good and not y['correct']['RAW']:rescues.append(q)
            if not good and y['correct']['RAW']:broken.append(q)
            if p != r['arms']['NATIVE']['prediction_identity']:changed.append(q)
            if good and q in source_rescues:kept.append(q)
        totals[arm] = dict(correct=len(correct),rescues=len(rescues),breaks=len(broken),
            changes_vs_CPU_NATIVE=len(changed),original_54_rescues_kept=len(kept),fold_correct=fold_correct,
            rescue_ids=rescues,break_ids=broken,changed_ids=changed,kept_original_rescue_ids=kept)
    drift = {a:sum(not r['anchors'][a]['same_decision'] for r in audits) for a in ['NATIVE','CONSTANT']}
    shares = [r['energy_weighted_common_share'] for r in audits if r['energy_weighted_common_share'] is not None]
    result = dict(status='ALL593_DECOMPOSITION_AUDITED' if not any(drift.values()) else 'ALL593_AUDITED_CPU_GPU_ACTION_DIFFERENCES',
        protocol=pb,prelabel_seal=bind(OUT/'independent_prelabel_seal.json'),n=593,original_POST_REAL=478,
        RAW=426,target_in_C128=570,totals=totals,CPU_GPU_decision_differences=drift,
        maximum_independent_logit_error=max(r['max_logit_error'] for r in audits),
        median_query_energy_weighted_common_share=float(np.median(shares)),
        scope=protocol['scope'], rows=audits,
        interpretation='Fixed-parameter computational pathway intervention on opened OOF; energy share alone is not causal contribution. No new fit or accuracy claim.')
    write(OUT/'result.json',result)
    lines=['# H593 后 LLM：共同响应、patch 剩余与匹配位置归因','',
        '原五折封存 POST_REAL、自然 ColNomic C128、固定 INTERNAL3。全593张；不训练、不运行编码器或RoMa。', '',
        '|回放|正确/593|救回RAW|误伤RAW|保留原54次救回|与CPU原生决策不同|',
        '|---|---:|---:|---:|---:|---:|']
    for arm,t in totals.items():
        lines.append(f"|{arm}|{t['correct']}|{t['rescues']}|{t['breaks']}|{t['original_54_rescues_kept']}|{t['changes_vs_CPU_NATIVE']}|")
    lines += ['',f'CPU与封存GPU的决策差异：{drift}。存在差异时，CPU原生是干预基线，不能宣称精确解释原GPU478。',
        f"逐patch独立重算头分数最大误差：{result['maximum_independent_logit_error']:.3g}。",
        'COMMON_ONLY保留跨patch均值；SPATIAL_ONLY仅指去均值后的patch差异，不代表空间ownership。固定命中由同一适配器的恒定M路径提供。',
        '本报告为已打开OOF上的固定参数机制诊断，正确数保留不能替代逐图决策与纠错集合比较。',
        '[完整结果及逐图分数](../results/rc_postllm_h593_decomposition_v1/result.json)']
    report = ROOT/'reports/REPORT_POSTLLM_H593_DECOMPOSITION_V1_20260925.md'
    report.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],totals={a:{k:v for k,v in t.items() if not k.endswith('_ids')} for a,t in totals.items()},CPU_GPU_decision_differences=drift)),flush=True)


if __name__ == '__main__':
    main()
