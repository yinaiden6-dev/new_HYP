#!/usr/bin/env python3
"""Full H593 descriptive M ranking versus both sealed decisions. No fitting."""
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_m_481_492_comparison_20260927_v1'
BASE = ROOT / 'results/rc_h593_m_scalar_headroom_audit_20260927_v1'
NAMES = ['RAW', 'S', 'M', 'L', 'Q', 'R', 'bias']
TIERS = ['M_FIRST', 'M_FIRST_TIED', 'M_RANK_2_5', 'M_RANK_6_10', 'M_RANK_11_128', 'TARGET_ABSENT']
TRANSITIONS = ['BOTH_CORRECT', 'RESCUED_BY_492', 'BROKEN_BY_492', 'BOTH_WRONG']


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def checked(b):
    assert bind(b['path']) == b
    return read(b['path'])


def write_csv(p, records):
    fields = list(dict.fromkeys(k for r in records for k in r))
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader(); w.writerows(records)


def main():
    iv = read(BASE / 'independent_validation.json')
    assert iv['status'] == 'H593_M_SCALAR_HEADROOM_INDEPENDENT_PASS'
    audit = checked(iv['result']); old = {r['query_id']: r for r in audit['records']}
    cache = checked(audit['sources']['cache'])
    gallery = checked(audit['sources']['gallery'])
    labels = {r['physical_row']: r['identity'] for r in gallery['records']}
    gaproot = ROOT / 'results/rc_h593_s_bias_competition_v1'
    gv = read(gaproot / 'validation.json'); gap_result = checked(gv['result'])
    gr = {r['query_id']: r for r in gap_result['rows']}
    fold_sources = []; preds = {}; params = {}; gates = {}
    for f in range(5):
        p = gaproot / f'fold{f}/payload.json'
        fv = read(p.with_name('validation.json'))
        assert fv['payload'] == bind(p)
        d = read(p); fold_sources.append(bind(p)); gates[f] = d['parameters']['GAP_BIAS2']
        native = read(ROOT / f'results/rc_six_cause_isolation_v1/loss_binding/fold{f}/payload.json')
        params[f] = np.array([float.fromhex(x) for x in native['parameters']['COST1']])
        train = set(d['train_query_ids'])
        for r in d['predictions']:
            assert r['query_id'] not in train and r['query_id'] not in preds
            preds[r['query_id']] = r
    records = []; candidates = []; contributions = []; matrix = {k:Counter() for k in TIERS}
    maxerr = 0.
    for r in cache['rows']:
        q = r['query_id']; a = old[q]; g = gr[q]; p = preds[q]; f = a['fold']
        axis = r['axis']; w = r['winner']; cs = r['challengers']; mass = np.asarray(r['mass'])
        assert axis == p['candidate_physical_rows'] and cs == p['challenger_positions']
        target = a['target_physical']; t = axis.index(target) if target is not None else None
        if target is not None:
            assert labels[target] == g['identity']
        z = np.array([float.fromhex(v) for v in p['models']['COST1_FULL']['logits_hex']])
        zg = np.array([float.fromhex(v) for v in p['models']['GAP_BIAS2']['logits_hex']])
        X = np.asarray(r['native_X']); theta = params[f]
        error = float(np.max(abs(X @ theta[:6] + theta[6] - z)))
        maxerr = max(maxerr, error); assert error < 2e-10
        top = int(np.argmax(z)); runner = max(v for j,v in enumerate(z) if j != top)
        h = gates[f]; assert h['alpha'] == 0
        gate_score = float(z[top] + h['beta'] * (z[top]-runner) + h['bias'])
        replay = z.copy()
        if z[top] <= 0 and gate_score > 0 and not h['disabled']:
            replay[top] = gate_score
        assert np.array_equal(replay, zg), q
        score481 = np.zeros(128); score492 = np.zeros(128)
        score481[cs] = z; score492[cs] = zg
        s481 = cs[top] if z[top] > 0 else w
        s492 = cs[int(np.argmax(zg))] if max(zg) > 0 else w
        assert axis[s481] == g['selected']['COST1_FULL'] == a['final_physical']
        assert axis[s492] == g['selected']['GAP_BIAS2']
        c481 = t == s481; c492 = t == s492
        assert c481 == a['cost1_correct'] and c492 == g['correct']['GAP_BIAS2']
        transition = {(True,True):'BOTH_CORRECT',(False,True):'RESCUED_BY_492',
                      (True,False):'BROKEN_BY_492',(False,False):'BOTH_WRONG'}[c481,c492]
        tier = ('TARGET_ABSENT' if t is None else 'M_FIRST' if a['M_unique_first'] else
                'M_FIRST_TIED' if a['M_strict_rank'] == 1 else 'M_RANK_2_5' if a['M_strict_rank'] <= 5 else
                'M_RANK_6_10' if a['M_strict_rank'] <= 10 else 'M_RANK_11_128')
        matrix[tier][transition] += 1
        def failure(correct):
            if correct: return 'CORRECT'
            if t is None: return 'TARGET_ABSENT'
            if t == w: return 'RAW_CORRECT_BROKEN'
            if t == cs[top]: return 'TARGET_TOP_BUT_HOLD'
            return 'CHALLENGER_RANKING_BLOCKED'
        mtop = int(np.argmax(mass))
        row = dict(query_id=q, display_id=a['display_id'], fold=f, component=a['component'],
                   target_identity=g['identity'],
                   target_in_C128=t is not None, target_physical=target,
                   M_tier=tier, target_M=a.get('target_M'), target_M_rank=a.get('M_strict_rank'),
                   M_ties_with_wrong=a.get('M_ties_with_wrong'), target_M_unique_first=a.get('M_unique_first',False),
                   target_minus_strongest_wrong_M=a.get('target_minus_strongest_wrong_M'),
                   max_M_physical=axis[mtop], max_M_identity=labels[axis[mtop]], max_M=float(mass[mtop]),
                   RAW_physical=axis[w], RAW_identity=labels[axis[w]], RAW_correct=t==w,
                   model481_physical=axis[s481], model481_identity=labels[axis[s481]],
                   model481_correct=c481, model481_action='HOLD' if s481==w else 'SWITCH',
                   model481_selected_M=float(mass[s481]), model481_selected_M_rank=1+int(np.sum(mass>mass[s481]+1e-12)),
                   model481_selected_score=float(score481[s481]), model481_failure=failure(c481),
                   model492_physical=axis[s492], model492_identity=labels[axis[s492]],
                   model492_correct=c492, model492_action='HOLD' if s492==w else 'SWITCH',
                   model492_selected_M=float(mass[s492]), model492_selected_M_rank=1+int(np.sum(mass>mass[s492]+1e-12)),
                   model492_selected_score=float(score492[s492]), model492_failure=failure(c492),
                   transition=transition, head_top_challenger_physical=axis[cs[top]],
                   head_top_challenger_identity=labels[axis[cs[top]]], head_top_challenger_correct=t==cs[top],
                   model481_max_challenger_score=float(z[top]), model492_gate_if_original_HOLD=gate_score,
                   gate_evaluated_for_HOLD=bool(z[top]<=0), challenger_gap=float(z[top]-runner),
                   model481_target_score=float(score481[t]) if t is not None else None,
                   model492_target_score=float(score492[t]) if t is not None else None,
                   head_target_challenger_rank=a.get('head_challenger_rank'))
        if t is not None:
            wrong = [j for j in range(128) if j != t]
            rival = max(wrong,key=lambda j:score481[j])
            terms = np.zeros((128,7)); terms[cs,:6]=X*theta[:6]; terms[cs,6]=theta[6]
            diff=terms[t]-terms[rival]
            assert abs(diff.sum()-(score481[t]-score481[rival]))<2e-10
            item=dict(query_id=q,display_id=a['display_id'],baseline481_rival=axis[rival],
                      rival_is_HOLD=rival==w,target_minus_rival_score=float(score481[t]-score481[rival]),
                      signed_terms=dict(zip(NAMES,diff.tolist())),
                      interpretation='Frozen computational score decomposition; correlated inputs, not independent causal effects')
            contributions.append(item)
            row['baseline481_strongest_wrong_physical']=axis[rival]
            row.update({f'target_minus_rival_term_{k}':float(v) for k,v in zip(NAMES,diff)})
        records.append(row)
        candidates.append(dict(query_id=q,display_id=a['display_id'],physical_axis=axis,
                               M0=mass.tolist(),L_free=r['free_content'],
                               COST1_481_scores=score481.tolist(),GAP_BIAS2_492_scores=score492.tolist(),
                               HOLD_physical=axis[w],target_physical=target))
    records.sort(key=lambda r:r['display_id'])
    assert len(records)==len({r['query_id'] for r in records})==593
    assert sum(r['model481_correct'] for r in records)==481
    assert sum(r['model492_correct'] for r in records)==492
    first_wrong=[r for r in records if r['target_M_unique_first'] and not r['model492_correct']]
    summary=dict(status='H593_M_481_492_FULL_COMPARISON_REPLAY_PASS',population=593,
                 model481='Original grouped OOF NATIVE7-COST1',model492='GAP_BIAS2 nested calibration on original COST1',
                 candidate_source='Frozen ColNomic natural C128',
                 transitions=dict(Counter(r['transition'] for r in records)),
                 M_rank_matrix={t:{k:matrix[t][k] for k in TRANSITIONS} for t in TIERS},
                 failure481=dict(Counter(r['model481_failure'] for r in records if not r['model481_correct'])),
                 failure492=dict(Counter(r['model492_failure'] for r in records if not r['model492_correct'])),
                 M_first_492_errors=len(first_wrong),M_first_492_error_types=dict(Counter(r['model492_failure'] for r in first_wrong)),
                 M_first_492_error_ids=[r['display_id'] for r in first_wrong],
                 M_first_new_492_breaks=[r['display_id'] for r in first_wrong if r['model481_correct']],
                 low_M_correct_492=sum(r['model492_correct'] and not r['target_M_unique_first'] for r in records),
                 maximum_native_logit_replay_error=maxerr,gate_all127_exact_replay=True,
                 high_low_definition='Within-query target M ranking; no absolute global M threshold. Ranks use1e-12 tolerance.',
                 labels_used='Opened evaluation labels for diagnosis only. Target rank is not an inference-time routing feature.',
                 new_training_updates=0,new_encoder_or_matcher_forwards=0,
                 sources=dict(M_audit=iv['result'],M_validation=bind(BASE/'independent_validation.json'),
                              cache=audit['sources']['cache'],gallery=audit['sources']['gallery'],
                              GAP_result=gv['result'],GAP_validation=bind(gaproot/'validation.json'),GAP_folds=fold_sources),
                 program=bind(__file__))
    OUT.mkdir(parents=True,exist_ok=True)
    for name,data in [('summary.json',summary),('cases593.json',records),('score_decomposition.json',contributions)]:
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'all_candidates.jsonl').open('w') as f:
        for r in candidates:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write_csv(OUT/'cases593.csv',records)
    write_csv(OUT/'M_first_but_492_wrong_29.csv',first_wrong)
    write_csv(OUT/'M_lower_and_492_wrong.csv',[r for r in records if not r['model492_correct'] and r['target_in_C128'] and not r['target_M_unique_first']])
    write_csv(OUT/'492_rescues_and_breaks_21.csv',[r for r in records if r['transition'] in ('RESCUED_BY_492','BROKEN_BY_492')])
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>H593: M versus 481 / 492</title>
<style>body{font:15px system-ui;margin:24px;color:#182838}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{position:sticky;top:0;background:#edf3f5}select,input{padding:8px;margin:8px}td small{color:#666}.bad{color:#b31b25}.good{color:#19734c}</style>
<h1>H593: M ranking and the 481 / 492 decisions</h1><p>Same593 queries and natural ColNomic C128. 481 = original COST1; 492 = GAP_BIAS2. “High / low M” means within-query target rank, not an absolute mass threshold. Ground-truth labels are used only for this diagnostic.</p>
<p><a href="cases593.csv">All593 CSV</a> · <a href="M_first_but_492_wrong_29.csv">29 M-first errors</a> · <a href="all_candidates.jsonl">Complete128-candidate scores</a> · <a href="summary.json">Summary</a></p>
<label>Search <input id="search" placeholder="Query or reference name"></label><label>M rank <select id="tier"><option value="">All</option>__TIERS__</select></label><label>Outcome <select id="outcome"><option value="">All</option>__OUTCOMES__</select></label><label>492 error <select id="error"><option value="">All</option><option>TARGET_TOP_BUT_HOLD</option><option>CHALLENGER_RANKING_BLOCKED</option><option>RAW_CORRECT_BROKEN</option><option>TARGET_ABSENT</option><option>CORRECT</option></select></label><p id="count"></p>
<table><thead><tr><th>Query</th><th>True reference</th><th>Target M / rank</th><th>481 answer</th><th>492 answer</th><th>Outcome</th><th>492 diagnosis</th></tr></thead><tbody id="rows"></tbody></table>
<script>const DATA=__DATA__;const esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function choice(r,n){return `<span class="${r['model'+n+'_correct']?'good':'bad'}">${esc(r['model'+n+'_identity'])}</span><br><small>${esc(r['model'+n+'_action'])}; score ${r['model'+n+'_selected_score'].toFixed(4)}</small>`}
function show(){const q=document.getElementById('search').value.toLowerCase(),t=document.getElementById('tier').value,o=document.getElementById('outcome').value,e=document.getElementById('error').value;const rs=DATA.filter(r=>(!q||[r.display_id,r.target_identity,r.model481_identity,r.model492_identity].join(' ').toLowerCase().includes(q))&&(!t||r.M_tier===t)&&(!o||r.transition===o)&&(!e||r.model492_failure===e));document.getElementById('count').textContent=rs.length+' /593 queries';document.getElementById('rows').innerHTML=rs.map(r=>`<tr><td>${esc(r.display_id)}</td><td>${esc(r.target_identity)}</td><td>${r.target_M==null?'Outside C128':r.target_M.toFixed(6)+' / '+r.target_M_rank}</td><td>${choice(r,481)}</td><td>${choice(r,492)}</td><td>${esc(r.transition)}</td><td>${esc(r.model492_failure)}</td></tr>`).join('')}
for(const id of ['search','tier','outcome','error'])document.getElementById(id).addEventListener('input',show);show();</script></html>'''
    page=page.replace('__TIERS__',''.join(f'<option>{html.escape(x)}</option>' for x in TIERS)).replace('__OUTCOMES__',''.join(f'<option>{x}</option>' for x in TRANSITIONS)).replace('__DATA__',json.dumps(records,ensure_ascii=False).replace('<','\\u003c'))
    (OUT/'index.html').write_text(page)
    print(json.dumps({k:summary[k] for k in ['status','transitions','M_rank_matrix','failure492','M_first_492_error_types','M_first_new_492_breaks','maximum_native_logit_replay_error']},indent=2))


if __name__=='__main__':main()
