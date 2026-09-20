#!/usr/bin/env python3
"""Matched H593 OOF correction with content statistics and no RoMa inputs."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'programs'))
import run_rc_six_cause_loss_binding_v1 as L

OUT = ROOT / 'results/rc_h593_learned_colnomic_only_v1'
AUTH = ROOT / 'registry/rc_h593_learned_colnomic_only_authority_v1_20260920.json'
PLAN = ROOT / 'plan/RC_H593_LEARNED_COLNOMIC_ONLY_V1_20260920.md'
LAUNCH = ROOT / 'slurm/rc_h593_learned_colnomic_only_v1.sbatch'
FEATURES = ['RAW', 'F', 'B', 'Gq', 'Gr', 'U']
MODELS = {'COST1_CONTENT7': ('COST1', list(range(6))), 'CE_CONTENT7': ('CE', list(range(6))),
          'COST1_MAXSIM3': ('COST1', [0, 1]), 'CE_MAXSIM3': ('CE', [0, 1]), 'RAW2_CE': ('CE', [0])}
read, bind, checked, need = L.read, L.bind, L.checked, L.need


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if p.exists():
        need(p.read_text() == content, 'IMMUTABLE:' + str(p)); return
    tmp = p.with_name('.' + p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(content); os.replace(tmp, p)


def envelope(b):
    r = read(checked(b['receipt'])); v = read(checked(b['validation']))
    need(r['payload'] == v['payload'] == b['payload'], 'TOKEN_ENVELOPE')
    need(v.get('receipt', b['receipt']) == b['receipt'], 'TOKEN_RECEIPT')
    need(v['status'].endswith(('CPU_REPLAY_PASS', '_CPU_PASS')), 'TOKEN_VALIDATION')


def prepare():
    need(not AUTH.exists(), 'NEW_AUTHORITY')
    old = read(ROOT / 'registry/rc_h593_six_feature_ablation_authority_v1_20260920.json')
    manifest = ROOT / 'results/rc_crisp_manual_baseline_v1/H593_workers.json'
    workers = read(manifest)['records']; groups = defaultdict(list)
    for w in workers:
        # Deliberately strip all RoMa metadata before exposing workers to runtime.
        v = {k:w[k] for k in ('query_id','execution_ordinal','source_query_id','source_index','source_image_sha256','raw')}
        groups[w['raw']['payload']['path']].append(v)
    shards = []
    for path in sorted(groups):
        rows = sorted(groups[path], key=lambda r:r['execution_ordinal'])
        envelope(rows[0]['raw']); need(len(rows) <= 8, 'SHARD_BOUND')
        shards.append(rows)
    need(len(shards) == 75 and sum(map(len, shards)) == 593, 'ALL_SOURCES')
    worker_path = OUT / 'workers.json'
    write(worker_path, dict(shards=shards, source_manifest=bind(manifest), labels_included=False, roma_inputs=False))
    folds = {}
    for f, b in old['fold_sources'].items():
        p = read(checked(b['full_payload'])); ce = read(checked(b['ce_payload']))
        v = read(checked(b['full_validation'])); need(v['payload'] == b['full_payload'], 'OLD_VALIDATED')
        folds[f] = dict(train_roles=b['train_roles'], train_query_ids=p['train_query_ids'],
                        effective_train_query_ids=ce['models']['ALL_CE']['query_ids'],
                        old_payload=b['full_payload'], old_validation=b['full_validation'])
    sources = dict(old['code_sources'])
    sources.update(content_program=bind(__file__), content_plan=bind(PLAN), content_launcher=bind(LAUNCH))
    write(AUTH, dict(status='LEARNED_COLNOMIC_ONLY_AUTHORIZED', authorization_date='2026-09-20',
                    sources=sources, workers=bind(worker_path), split=old['public_sources']['split'], folds=folds,
                    gallery=bind(ROOT/'results/rc_new_hyp_processed128_regression_v1/gallery_manifest.json'),
                    curator=old['join_sources']['curator'], old_result=old['join_sources']['old_result'],
                    features=FEATURES, models={k:dict(loss=v[0], columns=v[1]) for k,v in MODELS.items()},
                    primary_comparison=['FULL_COST1','COST1_CONTENT7'], precision='float64', steps=2000,
                    seed=17, optimizer=dict(name='AdamW',lr=.03,weight_decay=.001),
                    new_encoder_forwards=0, roma_input_reads=0, total_training_updates_with_replays=100000,
                    evidence_level='Opened H593 grouped OOF5 development comparison; not external confirmation'))
    print(json.dumps(dict(status='PREPARED',authority=bind(AUTH))), flush=True)


def guard(stage, fold=None, shard=None):
    a = read(AUTH)
    need(a['status']=='LEARNED_COLNOMIC_ONLY_AUTHORIZED', 'AUTHORITY')
    for b in a['sources'].values(): checked(b)
    workers = read(checked(a['workers']))['shards']
    allow = {Path(a[k]['path']).resolve() for k in ('workers','split')}
    if stage in ('worker', 'worker-verify'):
        need(shard in range(75), 'SHARD')
        for row in workers[shard]:
            allow.update(Path(b['path']).resolve() for b in row['raw'].values())
    if stage in ('fit','verify','join','join-verify'):
        allow.add(Path(a['gallery']['path']).resolve())
        use = range(5) if stage.startswith('join') else [fold]
        for f in use:
            need(f in range(5), 'FOLD')
            allow.add(Path(a['folds'][str(f)]['train_roles']['path']).resolve())
            if stage.startswith('join'):
                allow.update(Path(a['folds'][str(f)][k]['path']).resolve() for k in ('old_payload','old_validation'))
        if stage.startswith('join'):
            allow.update(Path(a[k]['path']).resolve() for k in ('curator','old_result'))
    def audit(event, args):
        if event=='socket.connect': need(not isinstance(args[1],tuple), 'OFFLINE')
        if event!='open' or not args or not isinstance(args[0], (str,bytes,os.PathLike)): return
        p=Path(os.fsdecode(args[0])).resolve(); s=str(p).lower()
        need(not any(t in s for t in ('d1-mi','d1_mi','formal392','/target_join/','gisc_prerecall_universe','/grozi/','/reports/','rc_opened_')), 'PROTECTED_READ')
        if ROOT/'results' in p.parents:
            own = OUT in p.parents
            if own:
                first=p.relative_to(OUT).parts[0]
                if stage in ('worker','worker-verify'):
                    own=first in ('workers.json','preflight.json') or (OUT/'features'/f'shard{shard:02d}') in p.parents
                elif stage in ('fit','verify'):
                    own=first in ('workers.json','preflight.json','features',f'fold{fold}')
                elif stage=='preflight': own=first=='preflight.json' or first.startswith('.preflight.json.')
            need(own or p in allow, 'UNLISTED_RESULT:' + s)
    sys.addaudithook(audit)
    if stage!='preflight':
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
        p=read(OUT/'preflight.json'); need(p['status']=='CONTENT_PREFLIGHT_PASS' and p['authority']==bind(AUTH), 'PREFLIGHT')
    return a, workers


def active(x):
    x=x.to(dtype=torch.float64)
    x=x[x.abs().sum(1)>1e-6]
    need(len(x)>=2 and bool(torch.isfinite(x).all()), 'ACTIVE_TOKENS')
    return x


def statistics(q, r):
    sim=q @ r.T
    f=sim.topk(2,dim=1).values; b=sim.topk(2,dim=0).values
    return torch.stack([f[:,0].mean(),b[0].mean(),(f[:,0]-f[:,1]).mean(),(b[0]-b[1]).mean(),sim.mean()])


def numpy_statistics(q, r):
    q=np.asarray(q,dtype=np.float64);r=np.asarray(r,dtype=np.float64)
    q=q[np.abs(q).sum(1)>1e-6];r=r[np.abs(r).sum(1)>1e-6]
    need(len(q)>=2 and len(r)>=2, 'NUMPY_ACTIVE')
    sim=np.matmul(q,r.T)
    f=np.partition(sim,-2,axis=1)[:,-2:]; b=np.partition(sim,-2,axis=0)[-2:]
    return np.array([np.max(sim,axis=1).mean(),np.max(sim,axis=0).mean(),
                     np.abs(f[:,1]-f[:,0]).mean(),np.abs(b[1]-b[0]).mean(),sim.mean()])


def features(raw, stats, winner):
    raw=list(map(float,raw)); mean=sum(raw)/len(raw)
    std=(sum((v-mean)**2 for v in raw)/len(raw))**.5
    out=[]
    for c in range(len(raw)):
        if c==winner:continue
        out.append([(raw[c]-raw[winner])/max(std,1e-12)]+
                   [(float(a)-float(b))/(abs(float(a))+abs(float(b))+1e-12) for a,b in zip(stats[c],stats[winner])])
    return out


def choose(z, row):
    j=int(np.argmax(z)) if isinstance(z,np.ndarray) else int(torch.argmax(z))
    pos=row['challenger_positions'][j] if float(z[j])>0 else row['winner']
    return row['candidate_physical_rows'][pos]


def tensor_sha(x):
    x=x.detach().cpu().contiguous()
    return hashlib.sha256(str(x.dtype).encode('ascii') +
                          json.dumps(list(x.shape),separators=(',',':')).encode('ascii') +
                          x.view(torch.uint8).numpy().tobytes()).hexdigest()


def worker(a, workers, shard, replay=False):
    folder=OUT/'features'/f'shard{shard:02d}'
    if not replay and (folder/'payload.json').exists():
        subprocess.run([sys.executable,__file__,'worker-verify','--shard',str(shard)],check=True);return
    rows=workers[shard]; b=rows[0]['raw'];envelope(b)
    need(all(w['raw']==b for w in rows), 'ONE_TOKEN_SOURCE')
    raw=torch.load(checked(b['payload']),map_location='cpu',weights_only=True,mmap=True)
    old=read(folder/'payload.json') if replay else None
    if replay:need(old['authority']==bind(AUTH), 'FEATURE_AUTHORITY')
    device=torch.device('cpu' if replay else 'cuda')
    if not replay:need(torch.cuda.is_available(), 'GPU_REQUIRED')
    records=[];maximum=0.;start=time.perf_counter();token_checks=set()
    with torch.inference_mode():
        for i,w in enumerate(rows):
            q=raw['records'][w['source_index']]
            need(q['query_id']==w['source_query_id'] and q['query_source_sha256']==w['source_image_sha256'], 'QUERY_SOURCE')
            need(tensor_sha(q['query_tokens'])==q['query_tokens_sha256'], 'QUERY_TOKEN_HASH')
            axis=q['candidate_physical_rows']; need(axis==sorted(set(axis)) and len(axis)==128, 'NATURAL_C128')
            need(sorted(q['raw_ranked_physical_rows'][:128])==axis and
                 q['candidate_ranked_physical_rows'][0]==q['raw_ranked_physical_rows'][0], 'RAW_RANK_AXIS')
            need(len(q['candidate_raw_scores'])==128 and bool(torch.isfinite(q['candidate_raw_scores']).all()), 'FINITE_RAW')
            winner=axis.index(q['candidate_ranked_physical_rows'][0]); challengers=[p for p in range(128) if p!=winner]
            stats=[];qt=active(q['query_tokens']).to(device)
            for physical in axis:
                ref=raw['references'][physical]
                if physical not in token_checks:
                    need(tensor_sha(ref['tokens'])==ref['tokens_sha256'], 'REFERENCE_TOKEN_HASH');token_checks.add(physical)
                if replay:
                    values=numpy_statistics(q['query_tokens'].numpy(),ref['tokens'].numpy())
                else:
                    values=statistics(qt,active(ref['tokens']).to(device)).cpu().numpy()
                stats.append(values.tolist())
            xs=features(q['candidate_raw_scores'].tolist(),stats,winner)
            record=dict(query_id=w['query_id'],execution_ordinal=w['execution_ordinal'],
                        source_image_sha256=w['source_image_sha256'],candidate_physical_rows=axis,
                        raw_ranked_physical_rows=q['raw_ranked_physical_rows'],winner=winner,challenger_positions=challengers,
                        raw_scores=q['candidate_raw_scores'].tolist(),statistics=stats,X=xs)
            if replay:
                expected=old['records'][i]
                need(all(record[k]==expected[k] for k in record if k not in ('statistics','X')), 'EXACT_METADATA')
                err=float(np.max(np.abs(np.array(stats)-np.array(expected['statistics'])))); maximum=max(maximum,err)
                need(err<2e-10, 'ALL_PAIR_CPU_STATISTICS')
                # Independent statistics followed by feature construction; relative gaps can amplify near zero.
                np.testing.assert_allclose(xs,expected['X'],atol=2e-9,rtol=2e-9)
            records.append(record)
            print(json.dumps(dict(event='CONTENT_QUERY_CHECKED' if replay else 'CONTENT_QUERY_READY',shard=shard,done=i+1,total=len(rows),elapsed=time.perf_counter()-start)),flush=True)
    if replay:
        write(folder/'validation.json',dict(status='CONTENT_ALL_PAIRS_NUMPY_PASS',authority=bind(AUTH),payload=bind(folder/'payload.json'),
             queries=len(rows),candidate_checks=len(rows)*128,statistic_checks=len(rows)*128*5,max_abs_error=maximum,roma_input_reads=0,label_reads=0))
    else:
        write(folder/'payload.json',dict(authority=bind(AUTH),shard=shard,records=records,raw_source=b,roma_input_reads=0,label_reads=0))
        write(folder/'runtime.json',dict(job=os.environ['SLURM_JOB_ID'],seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name()))
        subprocess.run([sys.executable,__file__,'worker-verify','--shard',str(shard)],check=True)


def load_features():
    rows=[]; validations=[]
    for shard in range(75):
        folder=OUT/'features'/f'shard{shard:02d}';v=read(folder/'validation.json')
        need(v['status']=='CONTENT_ALL_PAIRS_NUMPY_PASS' and v['authority']==bind(AUTH), 'ALL75_VALIDATED')
        p=read(checked(v['payload']));need(p['authority']==bind(AUTH), 'FEATURE_AUTHORITY');rows+=p['records']
        validations.append(bind(folder/'validation.json'))
    rows.sort(key=lambda r:r['execution_ordinal'])
    need(len(rows)==len({r['query_id'] for r in rows})==593 and [r['execution_ordinal'] for r in rows]==list(range(593)), 'ALL593')
    return rows,validations


def gallery(a):
    g=read(checked(a['gallery'])); need(g['physical_count']==5413 and g['identity_count']==5412, 'GALLERY')
    return {r['physical_row']:r['identity'] for r in g['records']}


def compute(a, fold):
    b=a['folds'][str(fold)]; roles={r['query_id']:r for r in read(checked(b['train_roles']))['records']}
    split=read(checked(a['split']))['folds'][fold]; rows,validations=load_features(); labels=gallery(a)
    train=[r for r in rows if r['query_id'] in set(split['train_query_ids'])]
    held=[r for r in rows if r['query_id'] in set(split['heldout_query_ids'])]
    need([r['query_id'] for r in train]==b['train_query_ids'] and set(roles)==set(b['train_query_ids']), 'ORIGINAL_TRAIN_ORDER')
    need(not set(roles)&{r['query_id'] for r in held}, 'HELDOUT_LABEL_DISJOINT')
    ys=[L.H.target_position(r,roles[r['query_id']]['identity'],labels) for r in train]
    keep=[i for i,y in enumerate(ys) if y>=-1]
    need([train[i]['query_id'] for i in keep]==b['effective_train_query_ids'], 'ORIGINAL_EFFECTIVE_TRAIN_ORDER')
    x=torch.tensor([train[i]['X'] for i in keep],dtype=torch.float64);y=torch.tensor([ys[i] for i in keep])
    h=torch.tensor([r['X'] for r in held],dtype=torch.float64)
    need(x.shape[1:]==h.shape[1:]==(127,6) and bool(torch.isfinite(x).all()) and bool(torch.isfinite(h).all()), 'FEATURE_SCHEMA')
    params={}; allz={};timing=[]
    for name,(loss,cols) in MODELS.items():
        start=time.perf_counter();torch.manual_seed(17)
        theta=L.train(x[:,:,cols],y,loss);allz[name]=h[:,:,cols]@theta[:-1]+theta[-1]
        params[name]=[float(v).hex() for v in theta]
        timing.append(dict(model=name,seconds=time.perf_counter()-start))
        print(json.dumps(dict(event='CONTENT_FIT_DONE',fold=fold,**timing[-1])),flush=True)
    preds=[]
    for i,r in enumerate(held):
        pred={k:r[k] for k in ('query_id','execution_ordinal','candidate_physical_rows','winner','challenger_positions')}
        pred['models']={m:dict(selected=choose(z[i],r),logits_hex=[float(v).hex() for v in z[i]]) for m,z in allz.items()}
        preds.append(pred)
    return dict(status='CONTENT_FOLD_SEALED',authority=bind(AUTH),fold=fold,train_query_ids=b['train_query_ids'],
                effective_train_query_ids=b['effective_train_query_ids'],parameters=params,predictions=preds,
                feature_validations=validations,heldout_label_reads=0,roma_input_reads=0),rows,timing


def fit(a,fold,replay=False):
    folder=OUT/f'fold{fold}'
    if not replay and (folder/'payload.json').exists():
        subprocess.run([sys.executable,__file__,'verify','--fold',str(fold)],check=True);return
    p,rows,timing=compute(a,fold)
    if not replay:
        write(folder/'payload.json',p);write(folder/'runtime.json',dict(job=os.environ['SLURM_JOB_ID'],fits=timing))
        subprocess.run([sys.executable,__file__,'verify','--fold',str(fold)],check=True);return
    need(p==read(folder/'payload.json'), 'FRESH_TRAIN_ALL_PARAMETER_LOGIT_BITS')
    rs={r['query_id']:r for r in rows};maximum=0.
    for pred in p['predictions']:
        r=rs[pred['query_id']];x=np.asarray(r['X'])
        for name,v in pred['models'].items():
            t=np.array([float.fromhex(h) for h in p['parameters'][name]])
            z=np.sum(x[:,MODELS[name][1]]*t[:-1],1)+t[-1]
            err=float(np.max(np.abs(z-np.array([float.fromhex(h) for h in v['logits_hex']]))));maximum=max(maximum,err)
            need(err<2e-10 and choose(z,r)==v['selected'], 'NUMPY_LOGITS_ACTION')
    write(folder/'validation.json',dict(status='CONTENT_FRESH_FIT_NUMPY_PASS',authority=bind(AUTH),payload=bind(folder/'payload.json'),
          logit_checks=len(p['predictions'])*len(MODELS)*127,max_abs_error=maximum,heldout_label_reads=0,roma_input_reads=0))


def paired(rows,base,new):
    groups=defaultdict(list)
    for r in rows:groups[r['component']].append(int(r['correct'][new])-int(r['correct'][base]))
    d=np.array([np.mean(v) for _,v in sorted(groups.items())]);rng=np.random.default_rng(20260920)
    boot=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(1)
    return dict(rescue=sum(r['correct'][new] and not r['correct'][base] for r in rows),
                loss=sum(r['correct'][base] and not r['correct'][new] for r in rows),
                changed=sum(r['selected'][base]!=r['selected'][new] for r in rows),
                equal_component_difference=float(d.mean()),bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))))


def sealed(a):
    ps=[];vs=[]
    for f in range(5):
        v=read(OUT/f'fold{f}/validation.json');need(v['status']=='CONTENT_FRESH_FIT_NUMPY_PASS' and v['authority']==bind(AUTH), 'ALL5_SEALS')
        p=read(checked(v['payload']));need(p['fold']==f and p['authority']==bind(AUTH), 'FOLD_SEAL');ps.append(p);vs.append(bind(OUT/f'fold{f}/validation.json'))
    write(OUT/'all_predictions_prelabel_seal.json',dict(authority=bind(AUTH),fold_validations=vs))
    roles={r['query_id']:r for r in read(checked(a['curator']))['records']}
    return ps,roles


def join(a):
    ps,roles=sealed(a);labels=gallery(a);features,_=load_features();fr={r['query_id']:r for r in features}
    prior=read(checked(a['old_result']));pr={r['query_id']:r for r in prior['rows']};rows=[]
    for p in ps:
        b=a['folds'][str(p['fold'])];v=read(checked(b['old_validation']));need(v['payload']==b['old_payload'], 'OLD_SEAL')
        old=read(checked(b['old_payload']));oldp={r['query_id']:r for r in old['predictions']}
        need(p['parameters']['RAW2_CE']==old['parameters']['RAW2_CE'], 'RAW2_PARAMETERS_BIT_EXACT')
        tr=read(checked(b['train_roles']))['records'];ti={r['identity'] for r in tr};tc={r['component'] for r in tr}
        for pred in p['predictions']:
            q=pred['query_id'];r=fr[q];role=roles[q]
            need(role['outer_fold']==p['fold'] and role['identity'] not in ti and role['component'] not in tc, 'HELD_IDENTITY_COMPONENT_DISJOINT')
            need(pred['models']['RAW2_CE']==oldp[q]['models']['RAW2_CE'], 'RAW2_ALL_LOGIT_BITS')
            selected=dict(RAW=r['candidate_physical_rows'][r['winner']],PATCH_MAXSIM=r['candidate_physical_rows'][int(np.argmax(np.array(r['statistics'])[:,0]))],
                          FULL_COST1=oldp[q]['models']['COST1']['selected'],FULL_CE=oldp[q]['models']['ALL_CE']['selected'],
                          **{m:v['selected'] for m,v in pred['models'].items()})
            correct={m:labels[s]==role['identity'] for m,s in selected.items()};ranked=r['raw_ranked_physical_rows']
            rank=next(i+1 for i,g in enumerate(ranked) if labels[g]==role['identity'])
            ranks={m:1 if correct[m] else rank+int(ranked.index(s)+1>rank) for m,s in selected.items()}
            for m,oldm in [('RAW','RAW'),('FULL_COST1','COST1'),('FULL_CE','ALL_CE'),('RAW2_CE','RAW2_CE')]:need(correct[m]==pr[q]['correct'][oldm], 'HISTORICAL_PARITY')
            rows.append(dict(query_id=q,original_query_id=role['original_query_id'],identity=role['identity'],component=role['component'],
                             fold=p['fold'],target_in_C128=rank<=128,selected=selected,correct=correct,ranks=ranks))
    rows.sort(key=lambda r:r['query_id']);need(len(rows)==len({r['query_id'] for r in rows})==593, 'ALL593')
    need(sum(r['target_in_C128'] for r in rows)==570 and len({r['component'] for r in rows})==64 and len({r['identity'] for r in rows})==68, 'POPULATION')
    summary={m:dict(correct=sum(r['correct'][m] for r in rows),MRR=sum(1/r['ranks'][m] for r in rows)/593,
                    against_RAW=paired(rows,'RAW',m),fold_correct={str(f):sum(r['correct'][m] for r in rows if r['fold']==f) for f in range(5)}) for m in rows[0]['correct']}
    need([summary[m]['correct'] for m in ('RAW','RAW2_CE','FULL_COST1','FULL_CE')]==[426,426,481,486], 'BASELINE_COUNTS')
    comparisons={f'{kind}_{model}__to__FULL_{kind}':paired(rows,f'{kind}_{model}',f'FULL_{kind}') for kind in ('COST1','CE') for model in ('CONTENT7','MAXSIM3')}
    result=dict(status='H593_LEARNED_COLNOMIC_ONLY_COMPLETE',authority=bind(AUTH),population=593,recall_present=570,
                features=FEATURES,primary='COST1_CONTENT7__to__FULL_COST1',summary=summary,comparisons=comparisons,rows=rows,
                evidence_level=a['evidence_level'],new_encoder_forwards=0,automatic_deployment_change=False,
                limitations=['Fixed descriptor family, not all possible content-only models.','H593 is opened development, not new external confirmation.',
                             'PATCH_MAXSIM here uses FP64 tokens, unlike the previous FP32 untrained comparator.'])
    write(OUT/'result.json',result)
    subprocess.run([sys.executable,__file__,'join-verify'],check=True)
    report=['# H593 learned ColNomic-only correction','',
            '原 H593 grouped OOF5，自然 RAW C128，检索身份监督；原完整头封存预测配对比较。23个候选缺失仍计入593分母。',
            'CONTENT7使用RAW差、双向无权重MaxSim、双向top1−top2间隙和整体相似度均值；六输入七参数。MAXSIM3仅RAW差与前向MaxSim。全部不读取RoMa权重或坐标。', '',
            '| 模型 | 正确/593 | MRR | 对RAW救/损 |','|---|---:|---:|---:|']
    for m,s in summary.items():report.append(f"| {m} | {s['correct']} | {s['MRR']:.6f} | {s['against_RAW']['rescue']}/{s['against_RAW']['loss']} |")
    report+=['','主比较：COST1_CONTENT7 → FULL_COST1；下表正差表示完整头较好。','',
             '| 比较 | 完整头救/损 | 等组差pp | bootstrap95% pp |','|---|---:|---:|---|']
    for m,s in comparisons.items():report.append(f"| {m} | {s['rescue']}/{s['loss']} | {100*s['equal_component_difference']:.3f} | [{100*s['bootstrap95'][0]:.3f}, {100*s['bootstrap95'][1]:.3f}] |")
    report+=['','全部75片内容统计经过每候选NumPy独立重算；5折新进程训练重放及NumPy动作检查通过，RAW2_CE参数和全部分数复现旧值。',
             '这是已开放H593开发对照；不宣称所有纯ColNomic模型都不能替代，不自动修改部署模型。峰值间隙不是已校准身份置信度。',
             '本表PATCH_MAXSIM是FP64读出，旧CRISP对照的PATCH_MAXSIM为FP32，须按实际结果分别记录。',
             '原始统计：features/shard*/payload.json；每折参数及127分数：fold*/payload.json；逐查询结果：result.json。']
    path=OUT/'report_zh.md';text='\n'.join(report)+'\n'
    if path.exists():need(path.read_text()==text,'REPORT_IMMUTABLE')
    else:path.write_text(text)
    print(json.dumps({m:s['correct'] for m,s in summary.items()}),flush=True)


def verify_join(a):
    ps,roles=sealed(a);labels=gallery(a);result=read(OUT/'result.json');rs={r['query_id']:r for r in result['rows']}
    features,_=load_features();fr={r['query_id']:r for r in features};counts=defaultdict(int)
    for p in ps:
        old=read(checked(a['folds'][str(p['fold'])]['old_payload']));oldp={v['query_id']:v for v in old['predictions']}
        for pred in p['predictions']:
            q=pred['query_id'];r=fr[q];axis=r['candidate_physical_rows'];raw=axis[r['winner']]
            chosen={'RAW':raw,'PATCH_MAXSIM':axis[max(range(128),key=lambda i:r['statistics'][i][0])]}
            models=dict(pred['models'],FULL_COST1=oldp[q]['models']['COST1'],FULL_CE=oldp[q]['models']['ALL_CE'])
            for m,v in models.items():
                z=[float.fromhex(h) for h in v['logits_hex']];need(len(z)==127 and all(np.isfinite(z)), '127_FINITE')
                k=max(range(127),key=lambda i:z[i]);chosen[m]=axis[r['challenger_positions'][k]] if z[k]>0 else raw
                need(chosen[m]==v['selected'], 'ACTION_FROM_ALL_SCORES')
            correct={m:labels[s]==roles[q]['identity'] for m,s in chosen.items()}
            need(chosen==rs[q]['selected'] and correct==rs[q]['correct'], 'INDEPENDENT_DECISION_AND_LABEL')
            # Explicit move-to-front order independently validates MRR for every model.
            for m,s in chosen.items():
                ranked=[s]+[v for v in r['raw_ranked_physical_rows'] if v!=s]
                rank=next(i+1 for i,v in enumerate(ranked) if labels[v]==roles[q]['identity'])
                need(rank==rs[q]['ranks'][m], 'INDEPENDENT_RANK');counts[m]+=int(correct[m])
    for m,s in result['summary'].items():
        need(counts[m]==s['correct'], 'COUNTS')
        need(paired(result['rows'],'RAW',m)==s['against_RAW'], 'PAIRED_REPLAY')
    for name,s in result['comparisons'].items():
        base,new=name.split('__to__');need(paired(result['rows'],base,new)==s, 'GROUP_STATS_REPLAY')
    write(OUT/'validation.json',dict(status='CONTENT_INDEPENDENT_ACTION_RANK_COUNTS_PASS',authority=bind(AUTH),result=bind(OUT/'result.json'),
          queries=593,baseline_counts=dict(RAW=426,RAW2_CE=426,FULL_COST1=481,FULL_CE=486),all_raw2_parameters_and_scores_bit_exact=True,
          independently_recomputed=['all_feature_statistics','all_head_logits','all_actions','all_label_joins','all_ranks','counts'],
          group_statistics='deterministic replay of shared statistical helper'))


def preflight():
    torch.manual_seed(17);errs=[]
    for nq,nr in [(3,4),(7,5),(5,9)]:
        for kind in ('random','negative','ties'):
            q=torch.randn(nq,11,dtype=torch.float64);r=torch.randn(nr,11,dtype=torch.float64)
            if kind=='negative':q=q.abs();r=-r.abs()
            if kind=='ties':r[:]=r[0];q[:]=q[0]
            got=statistics(active(q),active(r)).numpy();ref=numpy_statistics(q.numpy(),r.numpy())
            errs.append(float(np.max(np.abs(got-ref))));need(errs[-1]<1e-12,'INDEPENDENT_STATS')
            np.testing.assert_allclose(statistics(q.flip(0),r.flip(0)).numpy(),ref,atol=1e-12,rtol=0)
            qp=torch.cat([q,torch.zeros(1,11,dtype=torch.float64)]);rp=torch.cat([r,torch.zeros(2,11,dtype=torch.float64)])
            np.testing.assert_allclose(statistics(active(qp),active(rp)).numpy(),ref,atol=1e-12,rtol=0)
    row=dict(candidate_physical_rows=[3,9,12],winner=1,challenger_positions=[0,2])
    need(choose(np.array([0.,0.]),row)==9 and choose(np.array([1.,1.]),row)==3,'HOLD_TIE')
    # Same RAW arithmetic as original FC, including operation order.
    raw=[1.,3.,2.];dummy={i:dict(real_score=1.,visibility_mass=1.,query_control_score=0.,reference_control_score=0.) for i in range(3)}
    original=torch.stack([L.N.P.FC.candidate_feature(raw,dummy,i,1) for i in [0,2]])[:,0]
    own=torch.tensor(features(raw,np.ones((3,5)),1),dtype=torch.float64)[:,0]
    need(torch.equal(original,own),'RAW_FEATURE_BIT_ARITHMETIC')
    write(OUT/'preflight.json',dict(status='CONTENT_PREFLIGHT_PASS',authority=bind(AUTH),natural_updates=0,natural_token_reads=0,
          checks=['NumPy_statistics','negative_scores','ties','variable_lengths','padding','token_permutation','HOLD_and_ties','RAW_feature_arithmetic'],max_error=max(errs)))
    print('CONTENT_PREFLIGHT_PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','preflight','worker','worker-verify','fit','verify','join','join-verify']);p.add_argument('--shard',type=int);p.add_argument('--fold',type=int)
    args=p.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        a,workers=guard(args.stage,args.fold,args.shard)
        if args.stage=='preflight':preflight()
        elif args.stage.startswith('worker'):worker(a,workers,args.shard,args.stage=='worker-verify')
        elif args.stage=='join':join(a)
        elif args.stage=='join-verify':verify_join(a)
        else:fit(a,args.fold,args.stage=='verify')
