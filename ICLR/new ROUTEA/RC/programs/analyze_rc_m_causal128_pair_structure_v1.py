#!/usr/bin/env python3
"""Label-free separability audit of pair support, without fitting a new model."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ARMS=('A1J1P1','A1J1P0','A1J0P1','A1J0P0')
def read(p):return json.loads(Path(p).read_text())
def bind(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def checked(b):
    assert bind(b['path'])=={k:b[k] for k in ['path','sha256']},'SOURCE_CHANGED'
    return Path(b['path'])
def put(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert read(p)==d,'REFUSE_OVERWRITE_DIFFERENT_RESULT';return
    t=p.with_name(p.name+f'.tmp.{os.getpid()}');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def stats(xs):
    if not xs:return dict(count=0)
    return dict(count=len(xs),mean=statistics.fmean(xs),median=statistics.median(xs),
                mean_abs=statistics.fmean(map(abs,xs)),max_abs=max(map(abs,xs)),
                rms=math.sqrt(statistics.fmean(x*x for x in xs)))
def contrast(a,b,r,s):return a[r]-a[s]-b[r]+b[s]
def analyze(rows):
    assert len(rows)==128 and [r['index'] for r in rows]==list(range(128))
    data={a:[] for a in ARMS};queryranges={a:[] for a in ARMS};refs={a:defaultdict(list) for a in ARMS}
    cycles=[];replay=0.
    for row in rows:
        assert len(row['axis'])==len(set(row['axis']))==len(row['pair_audit'])==128
        for arm in ARMS:
            logs={};qvalues=[]
            for pos,z in enumerate(row['pair_audit']):
                physical=row['axis'][pos];assert z['position']==pos and z['physical_row']==physical
                q,v=z['mean_weights'][arm];m=z['M'][arm]
                assert q>0 and v>0 and m>0 and all(math.isfinite(x) for x in [q,v,m])
                assert m==row['M'][arm][pos]
                replay=max(replay,abs(m-math.sqrt(q*v)))
                qvalues.append(q);refs[arm][physical].append(v);logs[physical]=math.log(m)
            data[arm].append(logs);queryranges[arm].append(max(qvalues)-min(qvalues))
    assert replay<2e-12
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            common=sorted(set(rows[i]['axis'])&set(rows[j]['axis']))
            if len(common)<2:continue
            r,s=common[0],common[-1]
            cycles.append(dict(query_indices=[i,j],reference_rows=[r,s],shared_references=len(common),
                contrast={arm:contrast(data[arm][i],data[arm][j],r,s) for arm in ARMS}))
    summary={}
    for arm in ARMS:
        rr=[max(v)-min(v) for v in refs[arm].values() if len(v)>1]
        summary[arm]=dict(query_side_candidate_range=stats(queryranges[arm]),
                         reference_side_query_range=stats(rr),
                         logM_cycle=stats([c['contrast'][arm] for c in cycles]))
    increments={arm:stats([c['contrast'][arm]-c['contrast']['A1J0P0'] for c in cycles]) for arm in ARMS[:-1]}
    return dict(arms=summary,cycle_increment_vs_A_ONLY=increments,cycles=cycles,
                cycle_count=len(cycles),maximum_mass_replay_error=replay)

def main_run(root):
    source=root/'paths';out=root/'pair_structure';v=read(source/'cache_validation.json')
    assert v['status']=='M_CAUSAL128_EXTRACT_ALL128_PASS'
    c=read(checked(v['cache']));result=analyze(c['rows'])
    result.update(status='M_CAUSAL128_PAIR_STRUCTURE_COMPLETE',queries=128,labels_read=False,new_fits=0,
        sources=dict(validation=bind(source/'cache_validation.json'),cache=bind(source/'cache.json'),program=bind(__file__)),
        definition='If M(q,r)=a(q)b(r), every four-cycle logM(q,r)-logM(q,s)-logM(t,r)+logM(t,s) is zero. Nonzero beyond the numerical control rules out this single-image multiplicative explanation on the observed entries; it does not establish identity usefulness by itself.',
        sampling='For every pair of query rows with >=2 common physical references, take smallest/largest common physical row. Selection is independent of M and labels.',
        caveats=['Overlapping cycles are dependent; no significance test or effective sample-size claim.',
                 'A-only empirical numerical variation is retained as control, never thresholded to manufacture exact separability.',
                 'Nonseparability measures pair interaction, not correctness or unique causality.',
                 'The native coarse matcher may provide interaction through J and through derived P; these are not independent information sources.'])
    put(out/'result.json',result)
    lines=['# H128: single-image scores versus pair-specific support','',result['definition'],'',
           '| Input | Query-side candidate range max | Ref-side query range max | Cycle RMS | Cycle max abs |',
           '|---|---:|---:|---:|---:|']
    for a,x in result['arms'].items():
        lines.append(f"| {a} | {x['query_side_candidate_range'].get('max_abs',0):.8g} | {x['reference_side_query_range'].get('max_abs',0):.8g} | {x['logM_cycle'].get('rms',0):.8g} | {x['logM_cycle'].get('max_abs',0):.8g} |")
    lines+=['',f"Deterministically retained query-pair cycles: {result['cycle_count']}.",result['sampling'],'',*result['caveats']]
    report='\n'.join(lines)+'\n';rp=out/'report.md'
    if rp.exists():assert rp.read_text()==report
    else:rp.write_text(report)
    put(out/'validation.json',dict(status='M_CAUSAL128_PAIR_STRUCTURE_PASS',result=bind(out/'result.json'),report=bind(rp),
          source_validation=bind(source/'cache_validation.json'),queries=128,labels_read=False))
    print(json.dumps({'status':'M_CAUSAL128_PAIR_STRUCTURE_PASS','cycles':result['cycle_count']}))

def self_test():
    a={1:math.log(2*3),2:math.log(2*7)};b={1:math.log(5*3),2:math.log(5*7)}
    assert abs(contrast(a,b,1,2))<1e-14
    b[1]+=.25;assert abs(contrast(a,b,1,2)+.25)<1e-14
    assert stats([2,-2])['mean']==0 and stats([2,-2])['rms']==2
    rows=[]
    for i in range(128):
        rec=[];mm={arm:[] for arm in ARMS}
        for j in range(128):
            means={};masses={}
            for arm in ARMS:
                q=(i+1)/256;v=(j+1)/256
                if arm=='A1J1P1':q*=1+.1*math.sin((i+1)*(j+1))
                means[arm]=[q,v];masses[arm]=math.sqrt(q*v);mm[arm].append(masses[arm])
            rec.append(dict(position=j,physical_row=j,mean_weights=means,M=masses))
        rows.append(dict(index=i,axis=list(range(128)),pair_audit=rec,M=mm))
    r=analyze(rows)
    assert r['cycle_count']==8128 and r['arms']['A1J0P0']['logM_cycle']['max_abs']<1e-13
    assert r['arms']['A1J1P1']['logM_cycle']['rms']>.01
    rows[0]['pair_audit'][0]['M']['A1J0P0']+=.01
    try:analyze(rows)
    except AssertionError:pass
    else:raise AssertionError('M_CORRUPTION_ACCEPTED')
    print('PAIR_STRUCTURE_ALGEBRA_TEST_PASS')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:
        try:main_run(a.root)
        except FileNotFoundError as e:print(json.dumps({'status':'WAITING_FOR_INPUT','error':str(e)}));raise SystemExit(75)
