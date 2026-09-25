"""Independent source-to-decision replay; stdlib only, no production imports.

Run only after the producer's join stage. Rebuild every feature directly from
sealed RAW/L0/L inputs, then recompute strict HOLD and all TRAIN16/PROBE8 counts.
"""
from pathlib import Path
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
PROBE = ROOT/'results/rc_internal_m_v4_probe_v1'
V4 = ROOT/'results/rc_internal_m_condition_scale_v4'
V3 = ROOT/'results/rc_internal_m_learned_use_v3'
V2 = ROOT/'results/rc_prellm_m_adapter_v2'
AUTH = ROOT/'registry/rc_internal_m_content_preserving_readout_v1_authority_20260924.json'
ARMS = ('ORIGINAL3_CONTINUE', 'PRESERVE_REAL', 'PRESERVE_TRAIN_CONSTANT', 'PRESERVE_TRAIN_SHUFFLED')
SOURCES = {}
TOL = 1e-10


def bind(path):
    p = Path(path).resolve()
    return {'path':str(p), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def load(path):
    p = Path(path).resolve()
    SOURCES[str(p)] = bind(p)['sha256']
    return json.loads(p.read_text())


def verified(binding):
    assert bind(binding['path']) == binding, 'Source changed: '+binding['path']
    return Path(binding['path'])


def origin_path(split, mode, query):
    if split == 'probe':
        return PROBE/'predictions'/mode/(query+'.json')
    if mode == 'TRAIN_CONSTANT':
        return V3/'PRE_CONSTANT/endpoints/0128/native'/(query+'.json')
    where = {'REAL':('PRE_REAL','native'), 'REAL_CONSTANT':('PRE_REAL','constant'),
             'REAL_SHUFFLED':('PRE_REAL','shuffled'), 'TRAIN_SHUFFLED':('PRE_SHUFFLED','native')}
    arm, condition = where[mode]
    return V4/arm/'endpoints/0128'/condition/(query+'.json')


def contrast(values, candidate, winner):
    a, b = float(values[candidate]), float(values[winner])
    return (a-b)/(abs(a)+abs(b)+1e-12)


def reconstruct(row, original, adapted, head, kind):
    raw = [float(x) for x in row['raw_scores']]
    assert len(raw) == len(original) == len(adapted) == 128
    assert all(math.isfinite(x) for values in (raw, original, adapted) for x in values)
    ids = row['candidate_ids']; assert len(ids) == len(set(ids)) == 128
    winner = max(range(128), key=lambda k:raw[k])
    assert winner == row['winner_index']
    idx = [k for k in range(128) if k != winner]
    assert idx == row['challenger_positions']
    mean = math.fsum(raw)/128
    std = max(math.sqrt(math.fsum((x-mean)**2 for x in raw)/128), 1e-12)
    xs, logits = [], []
    for k in idx:
        delta_raw = (raw[k]-raw[winner])/std
        old = contrast(original,k,winner)
        new = contrast(adapted,k,winner)
        if kind == 'original':
            x = [delta_raw,old,1.]
        elif kind == 'adapted':
            x = [delta_raw,new,1.]
        elif kind == 'external':
            x = [delta_raw,contrast(row['M'],k,winner),old,1.]
        else:
            assert kind == 'preserve'
            x = [delta_raw,old,new-old,1.]
        assert len(x) == len(head)
        z = math.fsum(v*float(t) for v,t in zip(x,head))
        assert math.isfinite(z)
        xs.append(x);logits.append(z)
    best = max(range(127),key=lambda j:logits[j])
    pos = idx[best] if logits[best] > 0. else winner
    scores=[0.]*128
    for k,z in zip(idx,logits): scores[k]=z
    return dict(features=xs,logits=logits,scores128=scores,prediction_position=pos,
                prediction_id=ids[pos],prediction_identity=row['candidate_identities'][pos],
                switched=pos != winner)


def max_error(a,b):
    assert len(a)==len(b)
    if not a:return 0.
    if isinstance(a[0],list):return max(max_error(x,y) for x,y in zip(a,b))
    assert all(math.isfinite(float(x)) for x in a+b)
    return max(abs(float(x)-float(y)) for x,y in zip(a,b))


def main():
    # No partial results or pre-join execution accepted.
    done=load(OUT/'validation.json')
    assert done['status']=='CONTENT_PRESERVING_READOUT_SOURCE_NUMPY_JOIN_PASS'
    assert done['authority']==bind(AUTH)
    result=load(verified(done['result']));verified(done['report'])
    authority=load(AUTH);manifest=load(verified(authority['manifest']))
    for b in authority['code_sources']+authority['input_sources']:verified(b)
    verified(authority['parent_authority'])
    frozen=load(verified(result['prelabel_seal']))
    assert frozen['authority']==bind(AUTH)
    for b in frozen['heads']+frozen['predictions']:verified(b)
    assert len(frozen['heads'])==4 and len(frozen['predictions'])==24
    input_data=load(verified(authority['inputs']))
    assert input_data['held_labels_included'] is False
    declared_sources={b['path']:b for b in authority['input_sources']}
    heads={arm:load(OUT/'heads'/(arm+'.json')) for arm in ARMS}
    train_ids=[r['query_id'] for r in manifest['train_rows']]
    probe_ids=[r['query_id'] for r in manifest['probe_rows']]
    assert len(train_ids)==len(set(train_ids))==16
    assert len(probe_ids)==len(set(probe_ids))==8 and not set(train_ids)&set(probe_ids)
    warm=load(verified(authority['warm_head']))['theta']
    adapted_head=load(verified(authority['adapted_reference']))['theta']
    external_head=load(verified(authority['external_reference']))['theta']
    for arm,head in heads.items():
        assert head['authority']==bind(AUTH) and head['steps']==2000
        assert head['fit_queries']==train_ids
        assert head['held_labels_read']==head['adapter_updates']==0
        assert head['direct_M_input'] is False
        initial=warm if arm=='ORIGINAL3_CONTINUE' else [warm[0],warm[1],0.,warm[2]]
        assert head['initial_theta']==initial
    # Labels are used only to recount already frozen final predictions.
    previous_probe=load(verified(result['label_source']))
    probe_labels={r['query_id']:r for r in previous_probe['rows']}
    reported={(r['split'],r['query_id']):r for r in result['rows']}
    replay=[];errors={'features':0.,'logits':0.,'scores128':0.}
    for split,rows in [('train',manifest['train_rows']),('probe',manifest['probe_rows'])]:
        label='TRAIN16' if split=='train' else 'PROBE8'
        for row in rows:
            q=row['query_id']
            cv_path=V2/'encoder_cache'/q/'validation.json'
            cv=load(verified(declared_sources[str(cv_path)]))
            parity=load(verified(cv['parity']))
            original=parity['fresh_L0']
            assert original==input_data['rows'][q]['L0']
            contents={}
            for mode in ('REAL','REAL_CONSTANT','REAL_SHUFFLED','TRAIN_CONSTANT','TRAIN_SHUFFLED'):
                path=origin_path(split,mode,q)
                src=load(verified(declared_sources[str(path)]))
                assert src['query_id']==q and src['candidate_ids']==row['candidate_ids']
                contents[mode]=src['L']
                assert src['L']==input_data['rows'][q][mode]
            config={
                'ORIGINAL3_CONTINUE':('original','REAL',heads['ORIGINAL3_CONTINUE']['theta']),
                'PRESERVE_REAL':('preserve','REAL',heads['PRESERVE_REAL']['theta']),
                'PRESERVE_TRAIN_CONSTANT':('preserve','TRAIN_CONSTANT',heads['PRESERVE_TRAIN_CONSTANT']['theta']),
                'PRESERVE_TRAIN_SHUFFLED':('preserve','TRAIN_SHUFFLED',heads['PRESERVE_TRAIN_SHUFFLED']['theta']),
                'PRESERVE_REAL_INFER_REAL_CONSTANT':('preserve','REAL_CONSTANT',heads['PRESERVE_REAL']['theta']),
                'PRESERVE_REAL_INFER_REAL_SHUFFLED':('preserve','REAL_SHUFFLED',heads['PRESERVE_REAL']['theta']),
                'ADAPTED3_FROZEN_REFIT':('adapted','REAL',adapted_head),
                'EXTERNAL_ADDITIVE4':('external','REAL',external_head)}
            saved_path=OUT/'predictions'/split/(q+'.json')
            assert bind(saved_path) in frozen['predictions']
            saved=load(saved_path)
            assert saved['candidate_ids']==row['candidate_ids']
            assert set(saved['predictions'])==set(config)
            identity=row['target_id'] if split=='train' else probe_labels[q]['identity']
            if split=='train':
                assert [i for i,s in enumerate(row['candidate_identities']) if s==identity]==row['target_positions']
            selected={'RAW':row['candidate_identities'][row['winner_index']]}
            predictions={}
            for mode,(kind,content_mode,head) in config.items():
                reconstructed=reconstruct(row,original,contents[content_mode],head,kind)
                old=saved['predictions'][mode]
                assert list(head)==old['theta']
                for key in ('prediction_position','prediction_id','prediction_identity','switched'):
                    assert reconstructed[key]==old[key],(q,mode,key)
                for key in ('logits','scores128'):
                    e=max_error(reconstructed[key],old[key]);assert e<TOL,(q,mode,key,e)
                    errors[key]=max(errors[key],e)
                if kind in ('original','preserve'):
                    e=max_error(reconstructed['features'],old['features']);assert e<TOL,(q,mode,e)
                    errors['features']=max(errors['features'],e)
                selected[mode]=reconstructed['prediction_identity']
                predictions[mode]=reconstructed
            correct={k:v==identity for k,v in selected.items()}
            old=reported[(label,q)]
            assert old['selected']==selected and old['correct']==correct
            assert old['target_present']==(identity in row['candidate_identities'])
            replay.append(dict(query_id=q,split=label,identity=identity,selected=selected,correct=correct,predictions=predictions))
    summary={}
    for split in ('TRAIN16','PROBE8'):
        rows=[r for r in replay if r['split']==split]
        summary[split]={k:dict(correct=sum(r['correct'][k] for r in rows),n=len(rows),
            rescues=[r['query_id'] for r in rows if r['correct'][k] and not r['correct']['RAW']],
            breaks=[r['query_id'] for r in rows if not r['correct'][k] and r['correct']['RAW']]) for k in rows[0]['correct']}
    assert summary==result['summary']
    output=dict(status='INDEPENDENT_RAW_CONTENT_FEATURE_LOGIT_ACTION_AND_COUNTS_PASS',queries=24,
        models_per_query=8,challenger_logits_replayed=24*8*127,source_sha256=SOURCES,
        maximum_absolute_errors=errors,summary=summary,replay=replay,
        production_feature_or_decision_functions_imported=False,saved_X_used_as_input=False,
        fitting_updates=0,new_inference=0,threshold_searches=0,
        scope='Independent sealed-artifact replay; does not establish training convergence or untouched generalization')
    path=HERE/'validation.json';path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('status','queries','models_per_query','challenger_logits_replayed','maximum_absolute_errors','summary')}))


if __name__=='__main__':main()
