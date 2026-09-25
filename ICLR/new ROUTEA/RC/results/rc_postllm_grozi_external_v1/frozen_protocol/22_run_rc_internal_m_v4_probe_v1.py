#!/usr/bin/env python3
"""Frozen V4 inference on the previously opened, group-disjoint PROBE8.

No training or checkpoint selection. GPU workers cannot open curator labels;
the CPU join verifies all predictions before opening them. Original V4 is sealed.
"""
import argparse
import copy
import os
from pathlib import Path
import sys
import time

import numpy as np
import run_rc_internal_m_condition_scale_v4 as V

ROOT = V.ROOT
OUT = ROOT / 'results/rc_internal_m_v4_probe_v1'
AUTH = ROOT / 'registry/rc_internal_m_v4_probe_v1_authority_20260924.json'
PLAN = ROOT / 'plan/RC_INTERNAL_M_V4_PROBE8_20260924.md'
LAUNCH = ROOT / 'slurm/rc_internal_m_v4_probe_v1.sbatch'
REPORT = ROOT / 'reports/REPORT_INTERNAL_M_V4_PROBE8_20260924.md'
read, write, bind, checked, need = V.read, V.write, V.bind, V.checked, V.need
MODES = {'REAL': [('REAL', 'native'), ('REAL_CONSTANT', 'constant'), ('REAL_SHUFFLED', 'shuffled')],
         'CONTROLS': [('TRAIN_SHUFFLED', 'shuffled'), ('TRAIN_CONSTANT', 'constant')]}


def numpy_decision(row, content, theta, kind='INTERNAL3'):
    """Independent, label-free replay of FP64 features and HOLD=0 action."""
    raw, mass, val = [np.asarray(x, dtype=np.float64) for x in (row['raw_scores'], row['M'], content)]
    need(all(x.shape == (128,) and np.isfinite(x).all() for x in (raw, mass, val)), 'FINITE_C128')
    need(len(row['candidate_ids']) == len(set(row['candidate_ids'])) == 128, 'CANDIDATE_AXIS')
    w = row['winner_index']; idx = [i for i in range(128) if i != w]
    need(idx == row['challenger_positions'] and w == int(raw.argmax()), 'RAW_AND_CHALLENGER_AXIS')
    def sym(x):
        return (x[idx] - x[w]) / (np.abs(x[idx]) + abs(x[w]) + 1e-12)
    columns = [(raw[idx] - raw[w]) / max(raw.std(), 1e-12)]
    if kind == 'PRODUCT5': columns.append(sym(mass * val))
    if kind != 'INTERNAL3': columns.append(sym(mass))
    columns += [sym(val), np.ones(127)]
    x = np.stack(columns, axis=1); theta = np.asarray(theta, dtype=np.float64)
    need(kind in V.FEATURES and theta.shape == (x.shape[1],) and np.isfinite(theta).all(), 'HEAD_SHAPE')
    z = x @ theta
    pos = idx[int(z.argmax())] if float(z.max()) > 0 else w
    scores = np.zeros(128); scores[idx] = z
    return dict(prediction_position=pos, prediction_id=row['candidate_ids'][pos],
                prediction_identity=row['candidate_identities'][pos], switched=pos != w,
                challenger_positions=idx, logits=z.tolist(), scores128=scores.tolist(),
                theta=theta.tolist(), features=V.FEATURES[kind])


def verify_prediction(rec, row, authority, snapshot):
    need(rec['authority'] == authority and rec['snapshot'] == snapshot, 'PREDICTION_SOURCE')
    need(rec['query_id'] == row['query_id'] and rec['candidate_ids'] == row['candidate_ids'], 'PREDICTION_AXIS')
    need(rec['M'] == row['M'] and rec['raw_scores'] == row['raw_scores'], 'PREDICTION_INPUTS')
    need(rec['training_updates'] == 0 and rec['held_label_reads'] == 0, 'INFERENCE_ONLY')
    fresh = numpy_decision(row, rec['L'], rec['decision']['theta'])
    for key in ('prediction_position', 'prediction_id', 'prediction_identity', 'switched', 'challenger_positions'):
        need(fresh[key] == rec['decision'][key], 'DECISION_REPLAY:' + key)
    error = float(np.max(np.abs(np.asarray(fresh['logits']) - rec['decision']['logits'])))
    need(error < 1e-10, 'NUMPY_LOGIT_REPLAY')
    return error


def no_labels():
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
        path = str(Path(os.fsdecode(args[0])).resolve()).lower()
        need(not any(x in path for x in ('curator_roles', '/target_join/', 'd1-mi', 'd1_mi', 'formal392')), 'FORBIDDEN_LABEL_INPUT')
        need(path != str(V.V2 / 'result.json').lower(), 'NO_PREVIOUS_PROBE_RESULT')
    sys.addaudithook(audit)


def prepare():
    need(not AUTH.exists(), 'AUTHORITY_ALREADY_FROZEN')
    no_labels()
    a = read(V.AUTH); V.check_sources(a)
    rv = read(V.OUT / 'validation.json'); checked(rv['result']); checked(rv['report'])
    need(rv['status'] == 'TRAIN16_RESULT_ARTIFACTS_PASS', 'V4_COMPLETE')
    m = read(checked(a['manifest']))
    need(len(m['probe_rows']) == 8 and not any('target' in k or k in ('identity', 'component', 'group')
         for r in m['probe_rows'] for k in r), 'LABEL_FREE_PROBE8')
    fold = read(checked(m['provenance']['split']))['folds'][0]
    tq = {r['query_id'] for r in m['train_rows']}; pq = {r['query_id'] for r in m['probe_rows']}
    need(tq <= set(fold['train_query_ids']) and pq <= set(fold['heldout_query_ids']) and not tq & pq, 'ORIGINAL_SPLIT')
    snapshots = {}
    for key, directory in [('REAL', V.OUT/'PRE_REAL'), ('TRAIN_SHUFFLED', V.OUT/'PRE_SHUFFLED'),
                           ('TRAIN_CONSTANT', V.BASELINE/'PRE_CONSTANT')]:
        fit = read(directory/'fit_validation.json')
        need(fit['steps'] == 128 and not fit['direct_M_in_head'], 'FIXED128_NO_DIRECT_M')
        b = bind(directory/'snapshots/0128.pt'); need(b in fit['snapshots'], 'PINNED_SNAPSHOT')
        snapshots[key] = b
    bindings = []
    for row in m['probe_rows']:
        c = V.V2/'encoder_cache'/row['query_id']
        cv = read(c/'validation.json'); checked(cv['payload']); checked(cv['parity'])
        need(cv['authority'] == bind(V.V2_AUTH) and cv['status'] == 'QUERY_ENCODER_CACHE_PASS', 'QUALIFIED_PROBE_CACHE')
        bindings += [bind(c/'validation.json'), cv['payload'], cv['parity']]
        bindings += [b.get('token_file', b) for b in row['reference_tokens']]
    bindings = list({b['path']: b for b in bindings}.values())
    for b in bindings: checked(b)
    heads = {n: bind(V.OUT/'cpu'/f'{n}.json') for n in ('warmstart_INTERNAL3', 'ADDITIVE4', 'PRODUCT5')}
    refits = {n: bind(V.OUT/'readout_refits'/f'{n}.json') for n in
              ('GAIN_REAL', 'GAIN_REAL_CONSTANT', 'GAIN_REAL_SHUFFLED', 'GAIN_TRAIN_SHUFFLED', 'V3_CONSTANT')}
    for b in refits.values():
        d = read(checked(b)); need(set(d['fit_queries']) == tq and d['held_label_reads'] == 0, 'TRAIN_ONLY_REFIT')
    code = [Path(__file__), LAUNCH, PLAN, ROOT/'tests/test_internal_m_v4_probe_v1.py']
    write(AUTH, dict(status='FIXED_V4_OPENED_PROBE8_INFERENCE_AUTHORIZED', user_authorization='2026-09-24 用户继续：冻结V4后检验跨组',
          parent=bind(V.AUTH), parent_result=rv['result'], manifest=a['manifest'], snapshots=snapshots,
          cache_bindings=bindings, heads=heads, refit_heads=refits, code_sources=[bind(p) for p in code],
          source_fold=0, probe_query_ids=[r['query_id'] for r in m['probe_rows']], train_query_ids=sorted(tq),
          primary='REAL joint terminal128 versus RAW; fixed-head constant/shuffled interventions',
          secondary='Frozen TRAIN-only refit heads; no refit on probe and no selection of winner',
          modes=MODES, candidate_count=128, training_updates=0, previous_probe_exposure=True,
          evidence='Exploratory group-disjoint PROBE8 on previously opened development data, not independent confirmation',
          continuation='No automatic training or expansion after result', max_requeues=8))
    write(OUT/'preflight.json', dict(status='FROZEN_CHECKPOINTS_AND_PROBE_CACHE_PASS', authority=bind(AUTH),
          caches=8, training_updates=0, held_label_reads=0, query_split_disjoint=True, identity_check_at_postseal_join=True))
    print('Prepared', AUTH, flush=True)


def guard():
    a = read(AUTH)
    for b in a['code_sources']: checked(b)
    va = read(checked(a['parent'])); V.check_sources(va)
    for b in a['snapshots'].values(): checked(b)
    for b in a['cache_bindings']: checked(b)
    for b in list(a['heads'].values()) + list(a['refit_heads'].values()): checked(b)
    m = read(checked(a['manifest']))
    return a, va, m


def worker(lane, budget):
    import torch
    need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    no_labels(); a, va, m = guard(); V.setup(va)
    need(lane in MODES, 'LANE')
    if (OUT/lane/'validation.json').exists():
        seal=read(OUT/lane/'validation.json'); need(seal['authority']==bind(AUTH), 'RESUME_SEAL')
        for b in seal['predictions']: checked(b)
        return 0
    # Reuse the qualified frozen-weight loader, but write runtime receipts here.
    V.OUT = OUT/'runtime'; model = V.load_model(va)
    started = time.monotonic(); all_records = []
    for name, intervention in MODES[lane]:
        key = 'REAL' if lane == 'REAL' else name
        b = a['snapshots'][key]
        checkpoint = torch.load(checked(b), map_location='cpu', weights_only=True)
        need(checkpoint['step'] == 128, 'FIXED_TERMINAL_STEP')
        parent_auth = bind(V.BASE_AUTH) if name == 'TRAIN_CONSTANT' else a['parent']
        need(checkpoint['authority'] == parent_auth, 'CHECKPOINT_AUTHORITY')
        small = V.create_adapter(va, m, 'PRE_CONSTANT' if name == 'TRAIN_CONSTANT' else 'PRE_REAL')
        small.load_state_dict(checkpoint['adapter']); small.eval(); model.eval()
        theta = checkpoint['head'].detach().cpu()
        for row in m['probe_rows']:
            dest = OUT/'predictions'/name/(row['query_id']+'.json')
            if dest.exists():
                verify_prediction(read(dest),row,bind(AUTH),b); all_records.append(bind(dest));continue
            if time.monotonic()-started > budget-25: return 75
            partial=dest.with_suffix('.partial.json'); vals=[]
            if partial.exists():
                d=read(partial)
                need(d['authority']==bind(AUTH) and d['snapshot']==b and d['query_id']==row['query_id']
                     and d['mode']==name and d['candidate_ids']==row['candidate_ids'], 'PARTIAL_SOURCE')
                vals=d['L']; V.validate_pending(vals)
            def persist(v):
                write(partial,dict(authority=bind(AUTH),snapshot=b,query_id=row['query_id'],mode=name,
                      candidate_ids=row['candidate_ids'],L=v))
            cache=V.cached_row(row,'cuda'); refs=V.OLD.references(row)
            done=V.score_all(model,cache,small,row,refs,vals,started,budget,persist,intervention)
            del cache,refs; torch.cuda.empty_cache()
            if not done:return 75
            rec=dict(authority=bind(AUTH),snapshot=b,query_id=row['query_id'],mode=name,intervention=intervention,
                     candidate_ids=row['candidate_ids'],M=row['M'],raw_scores=row['raw_scores'],L=vals,
                     decision=V.choose(row,vals,theta),direct_M_in_head=False,training_updates=0,held_label_reads=0)
            verify_prediction(rec,row,bind(AUTH),b);write(dest,rec);all_records.append(bind(dest))
            print('PROBE_PREDICTION_SEALED',name,row['query_id'],flush=True)
        del small; torch.cuda.empty_cache()
    write(OUT/lane/'validation.json',dict(status='LABEL_FREE_PROBE_PREDICTIONS_PASS',authority=bind(AUTH),
          predictions=all_records,training_updates=0,held_label_reads=0))
    return 0


def join():
    import torch
    a, va, m = guard()
    frozen_heads={k:torch.load(checked(b),map_location='cpu',weights_only=True)['head'].tolist()
                  for k,b in a['snapshots'].items()}
    for lane in MODES:
        seal=read(OUT/lane/'validation.json');need(seal['authority']==bind(AUTH), 'LANE_AUTHORITY')
        expected=[OUT/'predictions'/name/(r['query_id']+'.json') for name,_ in MODES[lane] for r in m['probe_rows']]
        need(seal['predictions']==[bind(p) for p in expected], 'EXACT_PREDICTION_SET')
    audited={}; errors=[]; saved=[]
    refit_names={'REAL':'GAIN_REAL','REAL_CONSTANT':'GAIN_REAL_CONSTANT','REAL_SHUFFLED':'GAIN_REAL_SHUFFLED',
                 'TRAIN_SHUFFLED':'GAIN_TRAIN_SHUFFLED','TRAIN_CONSTANT':'V3_CONSTANT'}
    for row in m['probe_rows']:
        predictions={}
        for lane, modes in MODES.items():
            for name,_ in modes:
                p=OUT/'predictions'/name/(row['query_id']+'.json'); rec=read(p)
                key='REAL' if lane=='REAL' else name
                b=a['snapshots'][key]
                need(rec['decision']['theta']==frozen_heads[key] and rec['mode']==name
                     and not rec['direct_M_in_head'], 'EXACT_FROZEN_HEAD_AND_MODE')
                errors.append(verify_prediction(rec,row,bind(AUTH),b));saved.append(bind(p))
                predictions[name]=rec['decision']
                trained=read(checked(a['refit_heads'][refit_names[name]]))
                predictions[name+'_REFIT']=numpy_decision(row,rec['L'],trained['theta'])
        contents=V.cached_row(row)['fresh_L0'].double().tolist()
        for key, kind in [('warmstart_INTERNAL3','INTERNAL3'),('ADDITIVE4','ADDITIVE4'),('PRODUCT5','PRODUCT5')]:
            head=read(checked(a['heads'][key]));predictions[key]=numpy_decision(row,contents,head['theta'],kind)
        # Same REAL-refitted head with perturbed M: separates inference sensitivity
        # from the distinct refitted heads above. No probe fitting occurs.
        real_head=read(checked(a['refit_heads']['GAIN_REAL']))['theta']
        for name in ('REAL_CONSTANT','REAL_SHUFFLED'):
            rec=read(OUT/'predictions'/name/(row['query_id']+'.json'))
            predictions[name+'_REAL_REFIT_HEAD']=numpy_decision(row,rec['L'],real_head)
        audited[row['query_id']]=predictions
        write(OUT/'audited_decisions'/(row['query_id']+'.json'),dict(authority=bind(AUTH),
              query_id=row['query_id'],candidate_ids=row['candidate_ids'],predictions=predictions,held_label_reads=0))
    sealpath=OUT/'prelabel_validation.json'
    write(sealpath,dict(status='ALL_PROBE_PREDICTIONS_AND_NUMPY_REPLAY_PASS',authority=bind(AUTH),
          predictions=saved,decisions=[bind(OUT/'audited_decisions'/(r['query_id']+'.json')) for r in m['probe_rows']],
          maximum_logit_error=max(errors),training_updates=0,held_label_reads=0))
    # Curator is first opened only after the complete prediction/replay seal.
    parent=read(checked(m['provenance']['parent']))
    curator_b=parent['join_sources']['curator'];roles={r['query_id']:r for r in read(checked(curator_b))['records']}
    train=read(checked(m['provenance']['train_roles']))['records']
    comps={r['component'] for r in train}; identities={r['identity'] for r in train}; groups={r.get('group') for r in train if r.get('group') is not None}
    rows=[]
    for r in m['probe_rows']:
        role=roles[r['query_id']]
        need(role['outer_fold']==0 and role['component'] not in comps and role['identity'] not in identities, 'GROUP_IDENTITY_DISJOINT')
        need(role.get('group') is None or role['group'] not in groups, 'GROUP_DISJOINT')
        selected={'RAW':r['candidate_identities'][r['winner_index']],**{k:v['prediction_identity'] for k,v in audited[r['query_id']].items()}}
        rows.append(dict(query_id=r['query_id'],component=role['component'],group=role.get('group'),identity=role['identity'],
                    target_in_C128=role['identity'] in r['candidate_identities'],selected=selected,
                    correct={k:v==role['identity'] for k,v in selected.items()}))
    summaries={}
    for k in rows[0]['correct']:
        rescues=[r['query_id'] for r in rows if not r['correct']['RAW'] and r['correct'][k]]
        breaks=[r['query_id'] for r in rows if r['correct']['RAW'] and not r['correct'][k]]
        summaries[k]=dict(correct=sum(r['correct'][k] for r in rows),n=8,rescues=rescues,breaks=breaks,
                          changed=sum(r['selected'][k]!=r['selected']['RAW'] for r in rows))
    result=dict(status='FROZEN_V4_OPENED_PROBE8_COMPLETE',authority=bind(AUTH),prelabel_seal=bind(sealpath),label_source=curator_b,
                rows=rows,summary=summaries,probe_components=len({r['component'] for r in rows}),
                target_in_C128=sum(r['target_in_C128'] for r in rows),training_updates=0,
                evidence=a['evidence'],generalization_confirmed=False,checkpoint_selection=False)
    write(OUT/'result.json',result)
    lines=['# 内部 M V4：冻结终点的跨组 PROBE8 检验','',
           'ColNomic 原自然 C128；全部 adapter、小头、M标准化和尺度固定于 TRAIN16。无新训练、无probe选参。',
           '这8张已在早期开发实验中打开；未参与V4训练，不属于全新独立确认。','',
           '| 路径 | 正确/8 | 救回/误伤 |','|---|---:|---:|']
    for k,s in summaries.items(): lines.append(f"| {k} | {s['correct']}/8 | {len(s['rescues'])}/{len(s['breaks'])} |")
    lines+=['',f"实际component数：{result['probe_components']}；目标在C128：{result['target_in_C128']}/8。",
            'REAL是128步联合训练终点；REFIT为仅在原TRAIN16完成的固定读出重训，二者分别报告，不能按probe择优。',
            'REAL_CONSTANT/SHUFFLED共用REAL原终点头；带REAL_REFIT_HEAD的干预共用真实M重训头。',
            'TRAIN_SHUFFLED与TRAIN_CONSTANT为训练条件对照。末端内部头只有RAW差、条件化内容和bias。',
            '八图只能检验方向和失败方式；不自动追加实验，不宣称普遍有效。']
    REPORT.write_text('\n'.join(lines)+'\n')
    write(OUT/'validation.json',dict(status='PROBE8_SOURCE_NUMPY_IDENTITY_JOIN_PASS',authority=bind(AUTH),
          result=bind(OUT/'result.json'),report=bind(REPORT),max_numpy_error=max(errors),training_updates=0))
    print(result['status'],summaries,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','join']);p.add_argument('--lane',default='REAL');p.add_argument('--budget',type=float,default=380)
    args=p.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='join':join()
    else:sys.exit(worker(args.lane,args.budget))


if __name__=='__main__':main()
