#!/usr/bin/env python3
"""Frozen-score mechanism replication after all593 unified acquisition.

No model fitting, encoder, matcher or POST forward. Prior front128 results are
replayed as a mandatory compatibility check, not counted as new evidence.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np

RC = Path(__file__).resolve().parents[1]
OUT = RC/'results/rc_m593_cached_replication_fix_v1'
INSIDE = RC/'results/rc_h593_m_inside_v1'
VISUAL = RC/'results/rc_h593_m_visual_origin_v1'
SIMPLE = RC/'results/rc_h593_simple_explanations_v1'
AUTH = RC/'registry/rc_h593_simple_explanations_authority_v1_20260924.json'
OLD = RC/'results/rc_m_causal128_attribution_chain_v1/bridge/result.json'
ARMS = ['A1J1P1', 'A1J1P0', 'A1J0P1', 'A1J0P0']
CONTRASTS = {'P_with_J':(1,-1,0,0), 'P_without_J':(0,0,1,-1),
             'J_with_P':(1,0,-1,0), 'J_without_P':(0,1,0,-1),
             'J_P_interaction':(1,-1,-1,1)}


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return {'path':str(path), 'sha256':h.hexdigest()}


def checked(binding):
    assert bind(binding['path']) == binding, ('SOURCE_DRIFT',binding['path'])
    return Path(binding['path'])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def prepare():
    assert not (OUT/'protocol.json').exists(), 'Already prepared; do not overwrite protocol'
    final = RC/'results/rc_h593_unified_long4_v1/result.json'
    d = read(final)
    assert d['status']=='ALL593_THREE_FAMILIES_VALIDATED_PASS'
    assert d['complete_queries']==593 and d['validated_families']==1779
    assert [r['index'] for r in d['records']]==list(range(593)) and not d['missing']
    a=read(AUTH); cache=read(SIMPLE/'cache_validation.json')
    assert cache['status']=='CACHE_PASS' and cache['authority']==bind(AUTH)
    sources = [Path(__file__), RC/'programs/submit_rc_m593_cached_replication_v1.py', RC/'plan/RC_M593_CACHED_REPLICATION_V1_20261002.md',
        RC/'slurm/rc_m593_cached_replication_v1.sbatch',
        RC/'programs/run_rc_h593_m_inside_v1.py', RC/'programs/run_rc_h593_m_visual_origin_v1.py',
        RC/'programs/run_rc_m_phase_bridge_v2.py', RC/'programs/run_rc_m_causal128_frozen_bridge_v1.py',
        RC/'slurm/rc_h593_m_inside_control_v1.sbatch', RC/'slurm/rc_h593_m_visual_origin_control_v1.sbatch']
    p={'status':'M593_CACHED_REPLICATION_FROZEN','code_sources':[bind(x) for x in sources],
       'unified_validation':bind(final),'simple_authority':bind(AUTH),'cache':cache['payload'],
       'cache_validation':bind(SIMPLE/'cache_validation.json'), 'gallery':a['public']['gallery'],
       'labels':bind(RC/'results/rc_h593_quality_operator_eval_v1/result.json'),
       'old128':bind(OLD),'queries':593,'candidates':128,'arms':ARMS,'contrasts':CONTRASTS,
       'panels':['FRONT128','TAIL465','ALL593','TAIL_NEW_COMPONENTS','TAIL_SHARED_COMPONENTS'],
       'wrong':'Maximum frozen free ColNomic content among wrong candidates; physical-row tie break',
       'primary':['logM_gap','NATIVE7_action_gap','M_FREE_action_gap'],
       'uncertainty':'Component-equal, exploratory percentile bootstrap, 10000 draws, seed20261002; no multiple-test significance claim',
       'training':False,'GPU':False,'new_POST_forward':False,
       'scope':'Opened H593; tail465 is an acquisition extension, not untouched confirmation. Native COARSE is the intervention baseline; HR1 is a separate anchor.',
       'boundary':'Direct predictor input pathways, not unique causes or pure physical geometry. POST, phase/Gram/binding manipulations and same-family compensation are not newly extended here.'}
    write(OUT/'protocol.json',p)
    print(json.dumps({'status':p['status'],'protocol':bind(OUT/'protocol.json')}),flush=True)


def sym(x,y):
    return (x-y)/(np.abs(x)+np.abs(y)+1e-12)


def scores(row,masses,native_scores,native_theta,simple_theta):
    m=np.asarray(masses,np.float64);w=row['winner'];c=np.asarray(row['challengers'])
    assert m.shape==(128,) and np.isfinite(m).all() and (m>0).all() and (m<=1).all()
    s,q,r=(np.asarray(native_scores)*(m/np.asarray(row['mass']))[:,None]).T
    l=s/np.maximum(m,1e-12);free=np.asarray(row['free_content']);raw=np.asarray(row['native_X'])[:,0]
    x=np.stack([raw,sym(s[c],s[w]),sym(m[c],m[w]),sym(l[c],l[w]),
                sym((s-q)[c],(s-q)[w]),sym((s-r)[c],(s-r)[w])],1)
    y=np.stack([raw,sym((m*free)[c],(m*free)[w]),sym(m[c],m[w]),sym(free[c],free[w]),np.zeros(127),np.zeros(127)],1)
    answer={}
    for name,features,theta in [('NATIVE7',x,native_theta),('M_FREE',y,simple_theta)]:
        z=features@theta[:-1]+theta[-1];allz=np.zeros(128);allz[c]=z
        pick=int(c[int(z.argmax())]) if float(z.max())>0 else w
        answer[name]={'logits128':allz.tolist(),'prediction_position':pick}
    return answer,x


def stats(rows):
    grouped=defaultdict(list)
    for group,value in rows:
        if value is not None:
            assert math.isfinite(value)
            grouped[group].append(value)
    values=np.asarray([np.mean(grouped[g]) for g in sorted(grouped)],np.float64)
    if not len(values):return {'queries':0,'groups':0,'mean':None,'exploratory_group95':None}
    rng=np.random.default_rng(20261002)
    b=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return {'queries':sum(map(len,grouped.values())),'groups':len(values),'mean':float(values.mean()),
            'positive_groups':int((values>0).sum()),'negative_groups':int((values<0).sum()),
            'exploratory_group95':np.quantile(b,[.025,.975]).tolist()}


def analyze():
    p=read(OUT/'protocol.json');pb=bind(OUT/'protocol.json')
    for b in p['code_sources']:checked(b)
    final=read(checked(p['unified_validation']))
    dependencies=[]
    for root,status in [(INSIDE,'M_INSIDE_ALL593_COMPLETE'),(VISUAL,'M_VISUAL_ORIGIN_ALL593_COMPLETE')]:
        v=read(root/'validation.json');result=read(checked(v['result']))
        assert v['status']=='M_ORIGIN_JOIN_COUNTS_PASS' and result['status']==status
        assert len(result['rows'])==593
        dependencies.append(bind(root/'validation.json'))
    cache=read(checked(p['cache']));authority=read(checked(p['simple_authority']))
    labels={x['query_id']:x for x in read(checked(p['labels']))['rows']}
    gallery={x['physical_row']:x['identity'] for x in read(checked(p['gallery']))['records']}
    old={r['index']:r for r in read(checked(p['old128']))['rows']}
    assert len(cache['rows'])==593 and len(labels)==593 and len(old)==128
    query_rows=[];max_anchor=0.;max_old=0.;native_counts=defaultdict(int);seals=[]
    for i,row in enumerate(cache['rows']):
        qp=OUT/'queries'/f'{i:03d}.json'
        if qp.exists():
            stored=read(qp);assert stored['protocol']==pb
            query_rows.append(stored['row']);seals.append(bind(qp));continue
        assert row['execution_ordinal']==i and len(set(row['axis']))==128
        family=final['records'][i]['families'];assert final['records'][i]['query_id']==row['query_id']
        nv=read(checked(family['inside']['validation']));ivp=INSIDE/f'query{i:03d}/inside_validation.json'
        iv=read(ivp);im=read(checked(iv['payload']));native=read(checked(nv['payload']))
        vv=read(checked(family['visual']['validation']));vis=read(checked(vv['payload']))
        assert nv['payload']==family['inside']['payload'] and vv['payload']==family['visual']['payload']
        assert iv['status']=='M_INSIDE_PATHS_PASS' and len(im['records'])==128
        assert im['query_id']==native['query_id']==vis['query_id']==row['query_id']
        assert native['candidate_physical_rows']==vis['candidate_physical_rows']==row['axis']
        assert all(x['candidate_position']==j and x['physical_row']==row['axis'][j] for j,x in enumerate(im['records']))
        lab=labels[row['query_id']];fold=int(lab['fold'])
        target=[j for j,r in enumerate(row['axis']) if gallery[r]==lab['identity']]
        assert len(target)<=1 and bool(target)==lab['target_in_C128']
        t=target[0] if target else None
        w=max((j for j in range(128) if j!=t),key=lambda j:(row['free_content'][j],-row['axis'][j])) if target else None
        src=authority['sources'][i]['payload'];native_payload=read(checked(src))['modes']['NATIVE']
        ns=[[z[k] for k in ('real_score','query_control_score','reference_control_score')] for z in native_payload['scores']]
        nt=np.array([float.fromhex(x) for x in cache['native_parameters'][fold]['COST1']])
        st=np.array([float.fromhex(x) for x in cache['prior_product_parameters'][fold]['COST1']])
        anchor,features=scores(row,row['mass'],ns,nt,st)
        error=float(np.max(np.abs(features-np.asarray(row['native_X']))));assert error<2e-10
        masses={a:[r['branches'][a]['COARSE'] for r in im['records']] for a in ARMS}
        masses['ORIGINAL_HR1']=row['mass']
        for a,stages in vis['masses'].items():
            for stage in ['COARSE','HR1']:masses['VISUAL/'+a+'/'+stage]=stages[stage]
        assert np.max(np.abs(np.asarray(masses['VISUAL/NATIVE/COARSE'])-masses[ARMS[0]]))<2e-12
        assert np.max(np.abs(np.asarray(masses['VISUAL/NATIVE/HR1'])-row['mass']))<2e-12
        rec={'index':i,'query_id':row['query_id'],'component':lab['component'],'fold':int(fold),
             'axis':row['axis'],'target_position':t,'fixed_wrong_position':w,'raw_correct':lab['correct']['RAW'],
             'target_present':bool(target),'worlds':{},'anchor_error':error,'old128_max_error':0.,
             'sources':[bind(ivp),iv['payload'],family['inside']['validation'],family['visual']['validation'],vv['payload'],src]}
        for world,m in masses.items():
            pred,_=scores(row,m,ns,nt,st);metrics={}
            if target:
                metrics={'logM_target':math.log(m[t]),'logM_wrong':math.log(m[w]),'logM_gap':math.log(m[t])-math.log(m[w])}
                for name,z in pred.items():metrics[name+'_action_gap']=z['logits128'][t]-z['logits128'][w]
            for name,z in pred.items():metrics[name+'_correct']=z['prediction_position']==t
            rec['worlds'][world]={'M':m,'external':pred,'metrics':metrics}
            if i<128 and world in old[i]['arms']:
                assert old[i]['query_id']==row['query_id'] and old[i]['target_position']==t and old[i]['fixed_wrong_position']==w
                for k in ['logM_gap','NATIVE7_action_gap','M_FREE_action_gap']:
                    if target:rec['old128_max_error']=max(rec['old128_max_error'],abs(metrics[k]-old[i]['arms'][world][k]))
                for name,z in pred.items():
                    assert z['prediction_position']==old[i]['arms'][world][name+'_prediction_position']
        assert rec['old128_max_error']<2e-10,('OLD128_REPLAY',i,rec['old128_max_error'])
        write(qp,{'protocol':pb,'row':rec});query_rows.append(rec);seals.append(bind(qp))
        print(json.dumps({'stage':'QUERY_SAVED','index':i,'old128_error':rec['old128_max_error']}),flush=True)
    oldgroups={r['component'] for r in query_rows[:128]}
    panels={'FRONT128':query_rows[:128],'TAIL465':query_rows[128:],'ALL593':query_rows,
            'TAIL_NEW_COMPONENTS':[r for r in query_rows[128:] if r['component'] not in oldgroups],
            'TAIL_SHARED_COMPONENTS':[r for r in query_rows[128:] if r['component'] in oldgroups]}
    summaries={};csvrows=[]
    for panel,rs in panels.items():
        data={'queries':len(rs),'components':len({r['component'] for r in rs}),
              'target_present':sum(r['target_present'] for r in rs),'path_effects':{},'visual_effects':{}}
        for name,coefs in CONTRASTS.items():
            data['path_effects'][name]={}
            for metric in ['logM_gap','logM_target','logM_wrong','NATIVE7_action_gap','M_FREE_action_gap']:
                pairs=[(r['component'],sum(c*r['worlds'][a]['metrics'][metric] for c,a in zip(coefs,ARMS))) for r in rs if r['target_present']]
                result=stats(pairs);data['path_effects'][name][metric]=result
                csvrows.append({'panel':panel,'contrast':name,'metric':metric,**result})
        for world in query_rows[0]['worlds']:
            if not world.startswith('VISUAL/') or '/NATIVE/' in world:continue
            base='VISUAL/NATIVE/'+world.rsplit('/',1)[1]
            data['visual_effects'][world]={}
            for metric in ['logM_gap','logM_target','logM_wrong']:
                result=stats([(r['component'],r['worlds'][world]['metrics'][metric]-r['worlds'][base]['metrics'][metric]) for r in rs if r['target_present']])
                data['visual_effects'][world][metric]=result
        data['secondary_frozen_correct_counts']={a:{n:sum(r['worlds'][a]['metrics'][n+'_correct'] for r in rs) for n in ['NATIVE7','M_FREE']} for a in ARMS+['ORIGINAL_HR1']}
        summaries[panel]=data
    assert sum(r['target_present'] for r in query_rows)==570
    result={'status':'M593_CACHED_PATH_AND_EXTERNAL_REPLICATION_COMPLETE','protocol':pb,'panels':summaries,
            'queries':593,'candidate_scores_per_world':593*128,'query_receipts':seals,
            'consumer_validations':dependencies,'max_old128_replay_error':max(r['old128_max_error'] for r in query_rows),
            'max_native_feature_error':max(r['anchor_error'] for r in query_rows),
            'POST_expanded':False,'new_training':False,'unique_cause_claim':False,'untouched_confirmation':False}
    write(OUT/'result.json',result)
    with (OUT/'path_effects.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    lines=['# M机制：原128与新增465的缓存复核','',
        '同一H593自然C128、原组件分组、冻结外部原折头。没有重训、没有新RoMa或编码器前向。全593为已打开数据；新增465不等于独立未接触确认。',
        '本轮仅扩展入口通路及M到外部决策的路径；POST内部传播、相位/Gram/绑定性质与删除后补偿未扩展。',
        '', '| 面板 | 张数 / 在池内 | 组件数 | P有J时 logM差效应 | P无J时 logM差效应 | J×P交互 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        fields=[]
        for c in ['P_with_J','P_without_J','J_P_interaction']:
            x=s['path_effects'][c]['logM_gap'];fields.append(f"{x['mean']:.6f} {x['exploratory_group95']}" if x['mean'] is not None else 'NA')
        lines.append(f"| {name} | {s['queries']} / {s['target_present']} | {s['components']} | "+' | '.join(fields)+' |')
    lines+=['','每张图使用固定自由内容最强错误候选；先算干预前后target−wrong分差，再按component等权汇总。候选不是独立样本，多个对照未作多重检验校正。',
        'J=0只删除预测器的J入口，P仍由原双图过程产生；不能解释成彻底删除两图信息。视觉干预同时影响多种线索，不独立等同于文字、纹理或布局。',
        '完整逐候选M和两个冻结头的logits在queries/*.json；区分效应与target/wrong各自效应在result.json和path_effects.csv。',
        f"旧128所有匹配指标最大重放误差：{result['max_old128_replay_error']:.3g}。"]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'validation.json',{'status':'M593_SEALS_AND_OLD128_REPLAY_PASS','result':bind(OUT/'result.json'),
          'queries':593,'target_present':570,'old128_queries_replayed':128,'candidate_axis':128,
          'max_old128_replay_error':result['max_old128_replay_error'],'scope':'Saved-score and legacy endpoint replay; no new POST validation'})
    print(json.dumps({'status':result['status'],'panels':{k: {'queries':v['queries'],'P_with_J':v['path_effects']['P_with_J']['logM_gap']} for k,v in summaries.items()}}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','analyze']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else analyze()
