"""Independent POST replay from sealed content and checkpoint heads.

No production feature/decision imports. Torch is used only to read frozen
snapshot parameters, not to calculate or fit scores.
"""
from pathlib import Path
import hashlib
import json
import math
import torch

HERE=Path(__file__).resolve().parent
OUT=HERE.parent
ROOT=OUT.parents[1]
AUTH=ROOT/'registry/rc_postllm_m_v1_authority_20260924.json'
ARMS=('POST_REAL','POST_CONSTANT','POST_SHUFFLED')
sources={}

def binding(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def read(path):
    p=Path(path).resolve();sources[str(p)]=binding(p)['sha256']
    return json.loads(p.read_text())

def checked(b):
    assert binding(b['path'])==b,('SHA',b['path'])
    sources[b['path']]=b['sha256']
    return Path(b['path'])

def decision(row,content,theta,kind='INTERNAL3'):
    raw=list(map(float,row['raw_scores']));mass=list(map(float,row['M']));val=list(map(float,content))
    assert len(raw)==len(mass)==len(val)==len(row['candidate_ids'])==128
    assert all(math.isfinite(x) for seq in (raw,mass,val,theta) for x in seq)
    w=max(range(128),key=lambda i:raw[i]);idx=[i for i in range(128) if i!=w]
    assert w==row['winner_index'] and idx==row['challenger_positions']
    mean=math.fsum(raw)/128;std=max(math.sqrt(math.fsum((v-mean)**2 for v in raw)/128),1e-12)
    def sym(x,i):return (x[i]-x[w])/(abs(x[i])+abs(x[w])+1e-12)
    z=[];features=[]
    for i in idx:
        x=[(raw[i]-raw[w])/std]
        if kind=='PRODUCT5':x.append(sym([a*b for a,b in zip(mass,val)],i))
        if kind!='INTERNAL3':x.append(sym(mass,i))
        x += [sym(val,i),1.]
        assert len(x)==len(theta)
        z.append(math.fsum(a*b for a,b in zip(x,theta)));features.append(x)
    best=max(range(127),key=lambda j:z[j]);pos=idx[best] if z[best]>0 else w
    scores=[0.]*128
    for i,v in zip(idx,z):scores[i]=v
    return dict(prediction_position=pos,prediction_id=row['candidate_ids'][pos],
        prediction_identity=row['candidate_identities'][pos],switched=pos!=w,
        logits=z,scores128=scores,challenger_positions=idx,features=features)

def compare(expected,actual):
    for k in ('prediction_position','prediction_id','prediction_identity','switched','challenger_positions'):
        assert expected[k]==actual[k],k
    error=max(abs(x-y) for k in ('logits','scores128') for x,y in zip(expected[k],actual[k]))
    assert error<1e-10,error
    return error

def main():
    a=read(AUTH);m=read(checked(a['manifest']));ab=binding(AUTH)
    for b in a['code_sources']:checked(b)
    seal=read(OUT/'prelabel_validation.json')
    assert seal['status']=='POST_ALL_PREDICTIONS_NUMPY_REPLAY_PASS' and seal['authority']==ab
    assert len(seal['decisions'])==24 and len(seal['refits'])==3 and seal['held_label_reads']==0
    for b in seal['decisions']+seal['refits']:checked(b)
    decisions_by_path={b['path']:b for b in seal['decisions']}
    source_records={};heads={};refits={};train_ids=[r['query_id'] for r in m['train_rows']]
    assert len(set(train_ids))==16 and len(m['probe_rows'])==8
    assert not set(train_ids)&{r['query_id'] for r in m['probe_rows']}
    warm=read(checked(a['warm_head']))['theta']
    for arm in ARMS:
        fv=read(OUT/arm/'fit_validation.json')
        assert fv['authority']==ab and fv['steps']==128 and not fv['direct_M_in_head'] and fv['held_label_reads']==0
        endseal=read(checked(next(b for b in fv['endpoints'] if '/0128/' in b['path'])))
        assert endseal['authority']==ab and endseal['step']==128
        snapshot=next(b for b in fv['snapshots'] if b['path'].endswith('/0128.pt'))
        state=torch.load(checked(snapshot),map_location='cpu',weights_only=True)
        assert state['authority']==ab and state['step']==128 and state['arm']==arm
        heads[arm]=state['head'].tolist()
        ps=read(OUT/arm/'probe_validation.json')
        assert ps['authority']==ab and ps['snapshot']==snapshot
        assert len(ps['predictions'])==(24 if arm=='POST_REAL' else 8)
        assert len(endseal['predictions'])==(48 if arm=='POST_REAL' else 16)
        for split,sl in [('train',endseal),('probe',ps)]:
            for b in sl['predictions']:
                rec=read(checked(b))
                assert rec['authority']==ab and rec['snapshot']==snapshot
                assert rec['arm']==arm and rec['direct_M_in_head'] is False and rec['held_label_reads']==0
                source_records[(split,arm,rec['intervention'],rec['query_id'])]=rec
        path=OUT/'readout_refits'/(arm+'.json')
        assert binding(path) in seal['refits']
        refits[arm]=read(path)
        assert refits[arm]['authority']==ab and refits[arm]['steps']==2000
        assert refits[arm]['fit_queries']==train_ids and refits[arm]['held_label_reads']==0
        assert refits[arm]['initial_theta']==warm and refits[arm]['direct_M_in_head'] is False
    # All scalar predictions/heads and their parents have now passed the seal.
    labels=read(checked(a['probe_label_source']))
    roles={r['query_id']:r for r in labels['rows']}
    rows=[];maximum=0.;count=0
    for split in ('train','probe'):
        for row in m[split+'_rows']:
            q=row['query_id'];path=OUT/'audited_decisions'/split/(q+'.json')
            saved=read(checked(decisions_by_path[str(path)]))['decisions']
            rebuilt={}
            for arm in ARMS:
                modes=('native','constant','shuffled') if arm=='POST_REAL' else ('native',)
                for mode in modes:
                    rec=source_records[(split,arm,mode,q)]
                    assert rec['candidate_ids']==row['candidate_ids'] and rec['raw_scores']==row['raw_scores']
                    expected_mass=row['M'][1:]+row['M'][:1] if arm=='POST_SHUFFLED' else row['M']
                    assert rec['M']==expected_mass
                    name=arm if mode=='native' else arm+'_'+mode.upper()
                    for suffix,theta in [('',heads[arm]),('_REFIT',refits[arm]['theta'])]:
                        key=name+suffix
                        pred=decision(row,rec['L'],theta)
                        assert saved[key]['theta']==theta
                        maximum=max(maximum,compare(pred,saved[key]))
                        if suffix=='':
                            assert rec['decision']['theta']==theta
                            maximum=max(maximum,compare(pred,rec['decision']))
                        rebuilt[key]=pred;count+=127
            cv=read(ROOT/'results/rc_prellm_m_adapter_v2/encoder_cache'/q/'validation.json')
            parity=read(checked(cv['parity']))
            for kind,b in a['external_heads'].items():
                head=read(checked(b))['theta'];key='EXTERNAL_'+kind
                pred=decision(row,parity['fresh_L0'],head,kind)
                assert saved[key]['theta']==head
                maximum=max(maximum,compare(pred,saved[key]))
                rebuilt[key]=pred;count+=127
            assert set(rebuilt)==set(saved)
            identity=row['target_id'] if split=='train' else roles[q]['identity']
            selected={'RAW':row['candidate_identities'][row['winner_index']],**{k:v['prediction_identity'] for k,v in rebuilt.items()}}
            correct={k:v==identity for k,v in selected.items()}
            t=[i for i,x in enumerate(row['candidate_identities']) if x==identity]
            assert len(t)<=1
            diagnostic={}
            if t:
                for name,pred in rebuilt.items():
                    scores=pred['scores128'];w=row['winner_index'];rival=max(i for i in range(128) if i!=t[0])
                    rival=max((i for i in range(128) if i!=t[0]),key=lambda i:scores[i])
                    diagnostic[name]=dict(target_score=scores[t[0]],target_rank=1+sum(v>scores[t[0]] for v in scores),
                        strongest_wrong_score=scores[rival],strongest_wrong_identity=row['candidate_identities'][rival])
            rows.append(dict(query_id=q,split=split,identity=identity,target_in_C128=bool(t),selected=selected,correct=correct,diagnostic=diagnostic))
    summary={}
    for split in ('train','probe'):
        rr=[r for r in rows if r['split']==split]
        summary[split]={k:dict(correct=sum(r['correct'][k] for r in rr),n=len(rr),
            rescues=[r['query_id'] for r in rr if r['correct'][k] and not r['correct']['RAW']],
            breaks=[r['query_id'] for r in rr if not r['correct'][k] and r['correct']['RAW']]) for k in rr[0]['correct']}
    mapping={'POST_REAL':'REAL','POST_REAL_REFIT':'REAL_REFIT','POST_REAL_CONSTANT':'REAL_CONSTANT',
        'POST_REAL_CONSTANT_REFIT':'REAL_CONSTANT_REAL_REFIT_HEAD','POST_REAL_SHUFFLED':'REAL_SHUFFLED',
        'POST_REAL_SHUFFLED_REFIT':'REAL_SHUFFLED_REAL_REFIT_HEAD','POST_CONSTANT':'TRAIN_CONSTANT',
        'POST_CONSTANT_REFIT':'TRAIN_CONSTANT_REFIT','POST_SHUFFLED':'TRAIN_SHUFFLED','POST_SHUFFLED_REFIT':'TRAIN_SHUFFLED_REFIT'}
    position_comparison={k:dict(POST=summary['probe'][k],PRE=labels['summary'][pre],pre_name=pre) for k,pre in mapping.items()}
    official=OUT/'result.json'
    compared_official=False
    if official.exists():
        o=read(official);assert o['summary']==summary
        compared_official=True
    result=dict(status='INDEPENDENT_POST24_SOURCE_HEAD_FEATURE_SCORE_HOLD_AND_COUNTS_PASS',queries=24,
        challenger_logits_recomputed=count,maximum_absolute_error=maximum,summary=summary,rows=rows,
        matched_PRE_PROBE8_comparison=position_comparison,sources=sources,official_join_compared=compared_official,
        fitting_updates=0,new_inference=0,threshold_searches=0,production_scoring_imports=False,
        evidence='Opened development TRAIN16/PROBE8; source replay does not prove universal position superiority')
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','challenger_logits_recomputed','maximum_absolute_error','summary','official_join_compared')}))

if __name__=='__main__':main()
