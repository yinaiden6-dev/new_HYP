#!/usr/bin/env python3
"""Frozen current-cache panel: original COST1 replay and grouped visual trends."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
OUT = ROOT / 'results/rc_h593_cached_visual_trend_v1'
AUTH = ROOT / 'registry/rc_h593_cached_visual_trend_authority_v1_20260923.json'
VIS = ROOT / 'results/rc_h593_m_visual_origin_v1'
OLD = ROOT / 'results/rc_h593_subset41_frozen_paths_v1'
EVAL = ROOT / 'results/rc_h593_quality_operator_eval_v1'
ARMS = ('Q_GRAY', 'R_GRAY', 'Q_LOWPASS', 'R_LOWPASS', 'Q_SHUFFLE', 'R_SHUFFLE')
PATHS = ('native/HR1', 'native/COARSE') + tuple('visual/' + a + '/HR1' for a in ARMS)


def read(p): return json.loads(Path(p).read_text())
def bind(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(b):
    assert bind(b['path']) == b, b['path']
    return Path(b['path'])
def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if p.exists():
        assert p.read_text() == text, p
        return
    tmp = p.with_name('.' + p.name + '.tmp')
    tmp.write_text(text); os.replace(tmp, p)


def prepare():
    assert not AUTH.exists()
    olda = read(ROOT / 'registry/rc_h593_subset41_frozen_paths_authority_v1_20260923.json')
    records = []
    for f in sorted(VIS.glob('query[0-9][0-9][0-9]/validation.json')):
        v = read(f); assert v['status'] == 'M_VISUAL_ORIGIN_QUERY_PASS'
        p = read(checked(v['payload'])); i = int(f.parent.name[5:])
        records.append(dict(index=i, query_id=p['query_id'], visual_validation=bind(f), visual=v['payload'], operator=p['operator_source']))
    assert records and len({r['index'] for r in records}) == len(records)
    heads = {}; folds = []
    wanted = {r['query_id'] for r in records}
    for fold in range(5):
        v = read(EVAL / f'fold{fold}/validation.json'); p = read(checked(v['payload'])); folds.append(v['payload'])
        assert v['status'] == 'QUALITY_OPERATOR_EVAL_FRESH_NUMPY_PASS'
        train = set(p['train_query_ids'])
        for row in p['predictions']:
            qid = row['query_id']
            if qid not in wanted: continue
            assert qid not in train and qid not in heads
            heads[qid] = dict(fold=fold, theta_hex=p['parameters']['COST1_FROZEN_NATIVE']['theta_hex'], native=row['models']['COST1_FROZEN_NATIVE'])
    assert set(heads) == wanted
    for r in records:
        f = OLD / f"query{r['index']:03d}/validation.json"
        if f.exists():
            v = read(f); assert v['status'] == 'SUBSET41_COST1_FIXED_PATH_CPU_PASS'
            checked(v['payload']); r['reused_prediction'] = v['payload']; r['reused_validation'] = bind(f)
    write(OUT / 'snapshot.json', dict(records=records, heads=heads, fold_sources=folds,
        selection='All fully validated visual caches at freeze; opened development panel; not outcome-selected',
        new_gpu_forwards=0, new_training=False))
    sources = [Path(__file__), ROOT/'programs/run_rc_h593_subset41_frozen_paths_v1.py',
        ROOT/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py', ROOT/'slurm/rc_h593_cached_visual_trend_v1.sbatch']
    write(AUTH, dict(status='CACHED_VISUAL_TREND_AUTHORIZED', sources=[bind(p) for p in sources], snapshot=bind(OUT/'snapshot.json'),
        old_result=olda['old_result'], gallery=olda['gallery'], workers=olda['workers'],
        primary_arms=list(ARMS), head='Original held-out fold COST1', candidates='Natural RAW C128',
        supervision_changes=False, threshold=0, precision='FP64 original replay', new_gpu_forwards=0,
        inference='Exploratory paired group-equal effects; 20000 cluster bootstrap draws and grouped sign flips; Holm across six arms per outcome',
        expansion='Only if current evidence insufficient: fixed total128, no significance-based repeated stopping; no automatic GPU expansion here'))
    print(json.dumps(dict(queries=len(records), reused=sum('reused_prediction' in r for r in records), cpu_indices=[r['index'] for r in records if 'reused_prediction' not in r])), flush=True)


def guard():
    a = read(AUTH)
    for b in [*a['sources'], a['snapshot']]: checked(b)
    return a, read(a['snapshot']['path'])


def worker(index):
    import fcntl
    import numpy as np
    import torch
    from torch.nn import functional as F
    import run_rc_h593_subset41_frozen_paths_v1 as R
    from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import candidate_feature
    assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available()
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    a, snap = guard(); rec = next(r for r in snap['records'] if r['index'] == index)
    assert 'reused_prediction' not in rec
    d = OUT / f'query{index:03d}'; d.mkdir(parents=True, exist_ok=True)
    lk=(d/'worker.lock').open('a+'); fcntl.flock(lk, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (d/'validation.json').exists(): checked(read(d/'validation.json')['payload']); return
    p = read(checked(rec['visual'])); op = read(checked(rec['operator']))
    w = read(checked(a['workers']))['records'][index]
    q, prior, refs = R.C.load_input(w)
    assert p['query_id'] == op['query_id'] == w['query_id']
    assert p['candidate_physical_rows'] == op['candidate_physical_rows'] == q['candidate_physical_rows']
    pairs = []
    for b in p['parts']: pairs.extend(torch.load(checked(b), map_location='cpu', weights_only=True)['pairs'])
    assert [r['candidate_position'] for r in pairs] == list(range(128))
    qnorm = F.normalize(q['query_tokens'].to(torch.float64), dim=1)
    max_error = 0.0
    for pos, physical in enumerate(q['candidate_physical_rows']):
        dest=d/f'pair{pos:03d}.pt'
        if dest.exists(): continue
        ref=refs[physical]; native=prior['candidates'][pos]
        assert ref['tokens_sha256']==native['reference_tokens_sha256'] and R.C.M.token_sha(ref['tokens'])==ref['tokens_sha256']
        sim=qnorm@F.normalize(ref['tokens'].to(torch.float64),dim=1).T
        nu,nv=native['query_visibility'],native['reference_visibility']; loc=R.locals_torch(sim,nv)
        ns=sim.numpy(); un=nu.numpy(); vn=nv.numpy()
        nloc=(np.max(ns*vn[None],axis=1),np.max(ns*np.roll(vn,max(1,len(vn)//2))[None],axis=1))
        scores={}; error=0.0
        for path in PATHS:
            bits=path.split('/'); arm='NATIVE' if bits[0]=='native' else bits[1]; stage=bits[-1]
            z=pairs[pos]['arms'][arm]['stages'][stage]; u,v=z['AB']['weights'],z['BA']['weights']
            for mode in ('M_ONLY','FULL_UV'):
                key=mode+'/'+path; scores[key]=R.pair_scores(sim,nu,nv,u,v,mode,loc)
                independent=R.independent(ns,un,vn,u.numpy(),v.numpy(),mode,nloc)
                error=max(error,max(abs(scores[key][k]-independent[k]) for k in R.C.M.SCORE_KEYS))
                assert error<2e-10
                if path=='native/HR1':
                    assert all(float(scores[key][k]).hex()==float(native['old_scores'][k]).hex() for k in R.C.M.SCORE_KEYS)
        R.save(dest,dict(scores=scores,max_numpy_error=error,position=pos,authority=bind(AUTH)))
    evidence=[torch.load(d/f'pair{i:03d}.pt',map_location='cpu',weights_only=True) for i in range(128)]
    head=snap['heads'][w['query_id']]; theta=torch.tensor([float.fromhex(s) for s in head['theta_hex']],dtype=torch.float64)
    raw=list(map(float,q['candidate_raw_scores'])); row={k:op[k] for k in ('query_id','execution_ordinal','candidate_physical_rows','winner','challenger_positions')}
    models={}; features={}; max_logit_error=0.0
    for name in evidence[0]['scores']:
        ev={i:v['scores'][name] for i,v in enumerate(evidence)}
        x=torch.stack([candidate_feature(raw,ev,c,row['winner']) for c in row['challenger_positions']]); z=x@theta[:-1]+theta[-1]
        pred=dict(logits_hex=R.E.hx(z),selected=R.E.choose(z,row)); models[name]=pred; features[name]=x
        nz=np.sum(x.numpy()*theta[:-1].numpy(),axis=1)+float(theta[-1]); err=float(np.max(abs(nz-z.numpy())))
        assert err<2e-10 and R.E.choose(nz,row)==pred['selected']; max_logit_error=max(max_logit_error,err)
        if name.endswith('/native/HR1'):
            assert torch.equal(x,torch.tensor(op['modes']['NATIVE']['X'],dtype=torch.float64)) and pred==head['native']
    R.save(d/'features.pt',dict(X=features,theta=theta,row=row,authority=bind(AUTH)))
    write(d/'payload.json',dict(index=index,row=row,head=head,fold=head['fold'],models=models,features=bind(d/'features.pt'),visual=rec['visual'],label_reads=0,new_gpu_forwards=0))
    write(d/'validation.json',dict(status='CACHED_VISUAL_FROZEN_COST1_NUMPY_PASS',payload=bind(d/'payload.json'),native_all127_logits_bit_exact=True,
        max_numpy_logit_error=max_logit_error,max_numpy_scalar_error=max(e['max_numpy_error'] for e in evidence),new_gpu_forwards=0))
    print(dict(index=index,status='VALIDATED',models=len(models)),flush=True)


def grouped(items, value, seed=20260923):
    import numpy as np
    grouped_values={}
    for r in items:
        x=value(r)
        if x is not None: grouped_values.setdefault(r['component'],[]).append(float(x))
    keys=sorted(grouped_values); v=np.array([np.mean(grouped_values[k]) for k in keys]); n=len(v)
    assert n>0 and np.isfinite(v).all()
    rng=np.random.default_rng(seed)
    boots=v[rng.integers(0,n,size=(20000,n))].mean(1)
    active=v[v!=0]; obs=abs(float(v.sum()))
    if not len(active): p=1.0
    elif len(active)<=18:
        patterns=np.arange(1<<len(active),dtype=np.uint64)
        totals=np.zeros(len(patterns))
        for j,x in enumerate(active): totals+=(2*((patterns>>j)&1).astype(float)-1)*x
        p=float(np.mean(np.abs(totals)>=obs-1e-12))
    else:
        flips=rng.choice([-1.,1.],size=(20000,len(active)))@active
        p=float((1+np.sum(np.abs(flips)>=obs-1e-12))/20001)
    lodo=[float(np.delete(v,j).mean()) for j in range(n)] if n>1 else []
    return dict(groups=n,mean=float(v.mean()),ci95=list(map(float,np.quantile(boots,[.025,.975]))),
        grouped_signflip_p=p,negative_groups=int(sum(v<0)),zero_groups=int(sum(v==0)),positive_groups=int(sum(v>0)),
        leave_one_group_out_range=[min(lodo),max(lodo)] if lodo else None)


def holm(summaries, field):
    ordered=sorted(ARMS,key=lambda a:summaries[a][field]['grouped_signflip_p']); last=0.
    for j,a in enumerate(ordered):
        last=max(last,min(1.,(len(ordered)-j)*summaries[a][field]['grouped_signflip_p']))
        summaries[a][field]['holm_p']=last


def join():
    a,snap=guard(); sealed=[]; payloads={}
    for r in snap['records']:
        if 'reused_prediction' in r: b=r['reused_prediction']
        else:
            v=read(OUT/f"query{r['index']:03d}/validation.json"); assert v['status']=='CACHED_VISUAL_FROZEN_COST1_NUMPY_PASS'; b=v['payload']
        payloads[r['index']]=read(checked(b)); sealed.append(b)
    write(OUT/'prediction_seal.json',dict(authority=bind(AUTH),predictions=sealed,new_training=False,new_gpu_forwards=0))
    old={r['query_id']:r for r in read(checked(a['old_result']))['rows']}
    labels={r['physical_row']:r['identity'] for r in read(checked(a['gallery']))['records']}
    rows=[]; logits_checked=0
    for rec in snap['records']:
        p=payloads[rec['index']]; vis=read(checked(rec['visual'])); ref=old[rec['query_id']]; axis=p['row']['candidate_physical_rows']
        assert ref['fold']==p['fold'] and axis==vis['candidate_physical_rows']
        targets=[i for i,v in enumerate(axis) if labels[v]==ref['identity']]; assert len(targets)<=1
        r=dict(index=rec['index'],query_id=rec['query_id'],original_query_id=ref['original_query_id'],component=ref['component'],
            target_present=bool(targets),raw_correct=ref['correct']['RAW'],native_correct=ref['correct']['COST1_FROZEN_NATIVE'],models={},quality={})
        for path in PATHS:
            for mode in ('M_ONLY','FULL_UV'):
                name=mode+'/'+path; pred=p['models'][name]; z=[float.fromhex(s) for s in pred['logits_hex']]; assert len(z)==127 and all(math.isfinite(x) for x in z)
                j=max(range(127),key=z.__getitem__); selected=axis[p['row']['challenger_positions'][j]] if z[j]>0 else axis[p['row']['winner']]
                assert selected==pred['selected']; logits_checked+=127
                scores=[0.]*128
                for c,x in zip(p['row']['challenger_positions'],z):scores[c]=x
                t=targets[0] if targets else None
                r['models'][name]=dict(correct=labels[selected]==ref['identity'],selected=selected,
                    target_margin=(scores[t]-max(x for i,x in enumerate(scores) if i!=t)) if t is not None else None)
        assert r['models']['FULL_UV/native/HR1']['correct']==r['native_correct']
        for arm in ('NATIVE',)+ARMS:
            masses=vis['masses'][arm]['HR1']; assert len(masses)==128
            q=dict(mean_M=sum(masses)/128)
            if targets:
                t=targets[0]; wrong=max(m for i,m in enumerate(masses) if i!=t)
                q.update(target_M=masses[t],strongest_wrong_M=wrong,normalized_gap=(masses[t]-wrong)/(abs(masses[t])+abs(wrong)+1e-12))
            r['quality'][arm]=q
        rows.append(r)
    summaries={}
    for arm in ARMS:
        s={}
        for mode in ('M_ONLY','FULL_UV'):
            name=mode+'/visual/'+arm+'/HR1'
            s[mode]=dict(correct=sum(r['models'][name]['correct'] for r in rows),rescue=sum(r['models'][name]['correct'] and not r['native_correct'] for r in rows),
                break_count=sum(not r['models'][name]['correct'] and r['native_correct'] for r in rows),
                retained_rescues=sum(r['models'][name]['correct'] and r['native_correct'] and not r['raw_correct'] for r in rows),
                **grouped(rows,lambda r: int(r['models'][name]['correct'])-int(r['native_correct'])))
            s[mode+'_margin']=grouped(rows,lambda r: (r['models'][name]['target_margin']-r['models']['FULL_UV/native/HR1']['target_margin']) if r['target_present'] else None)
        s['quality_gap']=grouped(rows,lambda r: r['quality'][arm]['normalized_gap']-r['quality']['NATIVE']['normalized_gap'] if r['target_present'] else None)
        s['log_mean_M_ratio']=grouped(rows,lambda r: math.log(max(r['quality'][arm]['mean_M'],1e-300)/max(r['quality']['NATIVE']['mean_M'],1e-300)))
        summaries[arm]=s
    for field in ('M_ONLY','FULL_UV','M_ONLY_margin','FULL_UV_margin','quality_gap','log_mean_M_ratio'):holm(summaries,field)
    result=dict(status='CACHED_VISUAL_TREND_COMPLETE',authority=bind(AUTH),prediction_seal=bind(OUT/'prediction_seal.json'),queries=len(rows),groups=len({r['component'] for r in rows}),
        raw_correct=sum(r['raw_correct'] for r in rows),native_correct=sum(r['native_correct'] for r in rows),target_absent=sum(not r['target_present'] for r in rows),
        rescue_queries=[r['original_query_id'] for r in rows if r['native_correct'] and not r['raw_correct']],
        rescue_groups=len({r['component'] for r in rows if r['native_correct'] and not r['raw_correct']}),summary=summaries,rows=rows,
        uncertainty_boundary='Opened cache-available sample; grouped resampling is descriptive, not untouched confirmation. No difference is not equivalence. Image intervention strengths are not matched.',
        new_gpu_forwards=0,training=False)
    write(OUT/'result.json',result)
    write(OUT/'validation.json',dict(status='CACHED_VISUAL_TREND_RECOUNT_PASS',result=bind(OUT/'result.json'),queries=len(rows),logits_recounted=logits_checked,new_gpu_forwards=0))
    lines=['# 当前缓存的视觉干预趋势（冻结原 COST1）','',f"{result['queries']}张 / {result['groups']}组；RAW {result['raw_correct']}，COST1 {result['native_correct']}；原纠错{len(result['rescue_queries'])}张，来自{result['rescue_groups']}组。",'',
        '所有对照固定原折头、RAW C128、HOLD=0、原ColNomic内容。M_ONLY只替换质量；FULL_UV替换两侧权重。原生127分数必须逐位复现，新增回放有独立NumPy校验。','',
        '|干预|M_ONLY正确|FULL_UV正确|FULL_UV救/损|准确率分组等权差95%区间(pp)|Holm p|质量区分间隔变化95%区间|', '|---|---:|---:|---:|---|---:|---|']
    for arm,s in summaries.items():
        f=s['FULL_UV']; q=s['quality_gap']; ci=[100*x for x in f['ci95']]
        lines.append(f"|{arm}|{s['M_ONLY']['correct']}|{f['correct']}|{f['rescue']}/{f['break_count']}|[{ci[0]:.2f}, {ci[1]:.2f}]|{f['holm_p']:.4g}|[{q['ci95'][0]:+.4f}, {q['ci95'][1]:+.4f}]|")
    lines += ['', '区分间隔为target与最强wrong的M之差除以二者绝对值和；不同干预都与同query原图配对。按身份/来源component聚合，组等权bootstrap20000次；分组符号翻转检验，六干预内Holm校正。全部逐query结果、margin变化、分组正负方向及去掉任一组后的均值范围保存在result.json。', '',
        '本面板由缓存完成情况固定，已经打开过，不是独立确认。准确率不变不能证明某因素无用，区间也不支持等价声明。低通同时损坏文字/纹理/边缘；分块打乱引入新边缘；不得归因到单一语义因素。新增样本若需要，固定扩至128并单独报告新增组，不按显著性反复停止。']
    (ROOT/'reports/REPORT_H593_CACHED_VISUAL_TREND_20260923.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','summary')},ensure_ascii=False),flush=True)
    print(json.dumps(summaries,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','worker','join'));ap.add_argument('--index',type=int)
    args=ap.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='worker':worker(args.index)
    else:join()
