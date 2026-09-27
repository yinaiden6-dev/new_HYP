#!/usr/bin/env python3
"""Frozen external/POST M-only mediation of the all-front128 A/J/P deletions.

All image content, local weighting shape, frozen fold heads, and RAW candidates
remain fixed. Only M and its algebraic descendants change. This is deliberately
not an end-to-end intervention that reruns every downstream RoMa operation.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
from run_rc_m_phase_bridge_v2 import external
from run_rc_postllm_h593_v1 import Inputs
from rc_postllm_m_backend_v1 import load_projection
from rc_prellm_m_scaled_adapter_v4 import ScaledQualityResidualAdapter
from analyze_rc_postllm_signal_decomposition_v2 import projected, decision

DEFAULT = ROOT / 'results/rc_m_causal128_attribution_chain_v1/bridge'
POST = ROOT / 'results/rc_postllm_h593_decomposition_v1/protocol.json'
SIMPLE = ROOT / 'results/rc_h593_simple_explanations_v1/cache.json'
SIMPLE_AUTH = ROOT / 'registry/rc_h593_simple_explanations_authority_v1_20260924.json'
LABELS = ROOT / 'results/rc_h593_quality_operator_eval_v1/result.json'
RESTORE = ROOT / 'results/rc_m_property_restoration_factorial_v2'
HISTORICAL = ROOT / 'results/rc_mass_discrimination_source_v1/result.json'
PHASE = ROOT / 'results/rc_m_phase_external_internal_bridge_v2'
ARMS = ('A1J1P1', 'A1J1P0', 'A1J0P1', 'A1J0P0')
WORLDS = (*ARMS, 'ORIGINAL_HR1')
NEW_GROUP_INDICES = [78,82,93,96,100,101,104,107,108,112,114,118]
LIMIT = 2e-10


def read(path): return json.loads(Path(path).read_text())
def bind(path):
    path = Path(path).resolve(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return dict(path=str(path), sha256=h.hexdigest())
def checked(binding):
    actual = bind(binding['path'])
    assert actual['sha256'] == binding['sha256'], ('SOURCE_SHA_DRIFT', binding['path'])
    return Path(binding['path'])
def write(path, value, immutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if immutable and path.exists():
        assert path.read_text() == text, ('IMMUTABLE_JSON_DRIFT', str(path)); return
    temporary = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(text); os.replace(temporary, path)
def configure():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
def code_sources():
    return [bind(ROOT / 'programs' / name) for name in (
        Path(__file__).name, 'run_rc_m_phase_bridge_v2.py', 'run_rc_postllm_h593_v1.py',
        'rc_postllm_m_backend_v1.py', 'rc_prellm_m_scaled_adapter_v4.py',
        'rc_prellm_m_adapter_v1.py', 'analyze_rc_postllm_signal_decomposition_v2.py')]


def external_record(ext, fold, simple, authority):
    payload = authority['sources'][ext['execution_ordinal']]['payload']
    native = read(checked(payload))['modes']['NATIVE']
    record = dict(query_id=ext['query_id'], fold=fold, winner=ext['winner'],
        challengers=ext['challengers'], original_M=ext['mass'], free_content=ext['free_content'],
        native_X=ext['native_X'], native_scores=[[v[k] for k in
          ('real_score','query_control_score','reference_control_score')] for v in native['scores']],
        native_theta=[float.fromhex(v) for v in simple['native_parameters'][fold]['COST1']],
        simple_theta=[float.fromhex(v) for v in simple['prior_product_parameters'][fold]['COST1']],
        source=payload)
    error = float(np.max(np.abs(np.asarray(external(record, record['original_M'])['NATIVE7']['features'])
                               - np.asarray(record['native_X']))))
    assert error < LIMIT, ('EXTERNAL_FEATURE_ANCHOR', error)
    return record, error


def prepare(out, source):
    validation = read(source / 'cache_validation.json')
    assert 'PASS' in validation['status'], 'PATH_CACHE_NOT_VALIDATED'
    data = read(source / 'cache.json')
    if 'cache' in validation: assert validation['cache'] == bind(source / 'cache.json')
    rows = data['rows']; assert [r['index'] for r in rows] == list(range(128))
    pp = read(POST); manifest = read(checked(pp['manifest']))
    bypost = {r['query_id']: r for r in manifest['rows']}
    folds = {r['query_id']: r['fold'] for r in pp['rows']}
    simple = read(SIMPLE); byext = {r['query_id']: r for r in simple['rows']}
    authority = read(SIMPLE_AUTH)
    # Only identifiers/component are used for the locked subgroup, not outcomes.
    metadata = {r['query_id']: dict(component=r['component']) for r in read(LABELS)['rows']}
    oldgroups = {metadata[r['query_id']]['component'] for r in rows[:71]}
    subgroup = [r['index'] for r in rows if metadata[r['query_id']]['component'] not in oldgroups]
    assert subgroup == NEW_GROUP_INDICES
    assert len({metadata[rows[i]['query_id']]['component'] for i in subgroup}) == 9
    inputs = []; errors = []
    for row in rows:
        qid = row['query_id']; post = bypost[qid]; fold = folds[qid]; ext = byext[qid]
        assert row['axis'] == post['candidate_ids'] == ext['axis']
        assert row['winner'] == post['winner_index'] == ext['winner']
        assert row['fold'] == fold
        assert np.max(np.abs(np.asarray(post['M']) - row['original_M'])) < 1e-14
        for arm in ARMS:
            values = np.asarray(row['M'][arm], np.float64)
            assert values.shape == (128,) and np.isfinite(values).all() and (values > 0).all() and (values <= 1).all()
        er, err = external_record(ext, fold, simple, authority); errors.append(err)
        inputs.append(dict(index=row['index'], query_id=qid, fold=fold,
            component=metadata[qid]['component'], mechanism_group_replication=row['index'] in subgroup,
            post_row=post, external=er, masses={**{a: row['M'][a] for a in ARMS}, 'ORIGINAL_HR1':post['M']}))
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'inputs.json', dict(rows=inputs, labels_included=False), immutable=True)
    protocol = dict(status='FROZEN_CAUSAL128_M_ONLY_BRIDGE_PREPARED',
        inputs=bind(out/'inputs.json'), cache=bind(source/'cache.json'),
        cache_validation=bind(source/'cache_validation.json'), source_protocol=bind(source/'protocol.json'),
        post_protocol=bind(POST), authority=pp['authority'], manifest=pp['manifest'], endpoints=pp['endpoints'],
        code_sources=code_sources(), original_label_source=bind(LABELS), queries=128, candidates=128,
        worlds=list(WORLDS), native='A1J1P1', threads=4, max_anchor_error=LIMIT,
        external_native_feature_error=max(errors), group_replication_indices=subgroup,
        group_replication_n=12, group_replication_groups=9,
        definition='Replace only candidate M by each saved COARSE A/J/P-deletion output. For NATIVE7 all M-scaled score descendants are recomputed with original local shapes/content fixed. M_FREE uses the same M and fixed free content. POST_REAL changes its real M conditioning input; final INTERNAL3 does not directly receive M.',
        baseline='Each deletion contrasts with COARSE A1J1P1 from the same cache; ORIGINAL_HR1 is a separate replay anchor.',
        scope='Opened front128 of H593; original held-fold heads and natural ColNomic C128. Full128 primary; twelve queries/nine components absent from prior F71 are a locked separate mechanism replication subset, not untouched data.',
        training=False, gpu=False, new_encoder_or_roma_forwards=0,
        label_policy='Labels join only after all128 full-C128 score/patch artifacts are sealed. Query/component identifiers lock the subset; no result-based selection.')
    write(out/'protocol.json', protocol, immutable=True)
    write(out/'preparation_validation.json',dict(status='FROZEN_CAUSAL128_BRIDGE_PREPARATION_PASS',
        protocol=bind(out/'protocol.json'),queries=128,max_external_native_error=max(errors)), immutable=True)
    print(json.dumps(dict(status=protocol['status'], queries=128, worlds=len(WORLDS))), flush=True)


def protocol(out):
    p = read(out/'protocol.json')
    for b in p['code_sources']: checked(b)
    for key in ('inputs','cache','cache_validation','post_protocol','authority','manifest'):
        checked(p[key])
    return p, bind(out/'protocol.json')


class Context:
    def __init__(self, p):
        self.p = p; self.authority = read(checked(p['authority']))
        self.bank = Inputs(p['authority'], 'cpu', cpu_gb=2, gpu_gb=.5)
        self.projection = load_projection(self.authority, 'cpu')
        self.models = {}
    def load(self, item):
        fold = str(item['fold'])
        if fold not in self.models:
            cp = torch.load(checked(self.p['endpoints'][fold]['snapshot']), map_location='cpu', weights_only=True)
            assert cp['arm'] == 'POST_REAL' and cp['authority'] == self.p['authority']
            assert cp['input_scope']['manifest'] == self.p['manifest']
            a = self.authority
            model = ScaledQualityResidualAdapter(hidden_size=3584, bottleneck=a['bottleneck'],
                     condition_gain=a['condition_gain'], residual_scale=a['residual_scale'])
            model.load_state_dict(cp['adapter']); model.eval().requires_grad_(False); model.conditioning='real'
            self.models[fold] = (model, cp['head'].double().tolist())
        adapter, head = self.models[fold]
        cache = self.bank.cache_cpu(item['post_row']); refs = self.bank.refs(item['post_row'])
        return adapter, head, cache, refs
    @torch.no_grad()
    def candidate(self, adapter, cache, ref, masses):
        hidden = cache['hidden'][cache['image_mask']]; values=[]; locations=[]
        for mass in masses:
            tokens = projected(self.projection, cache, adapter(hidden, float(mass)))
            best, idx = (F.normalize(tokens.double(),dim=-1) @ ref.T).max(1)
            values.append(best.numpy()); locations.append(idx.numpy().astype(np.int32))
        return np.stack(values), np.stack(locations)


def npz_once(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), ('ORPHAN_OR_DUPLICATE_PATCH_ARTIFACT',str(path))
    tmp = path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream: np.savez_compressed(stream, **values)
    os.replace(tmp,path)


def row_worker(out, p, pb, item, context, deadline):
    folder=out/'queries'/f"{item['index']:03d}"; folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        done=folder/'result.json'
        if done.exists():
            assert read(done)['protocol']==pb; return 0
        adapter,head,cache,refs=context.load(item); records=[]
        for pos in range(128):
            path=folder/'candidates'/f'{pos:03d}.json'
            if path.exists():
                rec=read(path);assert rec['protocol']==pb and rec['position']==pos
                checked(rec['patch_evidence']);records.append(rec);continue
            if time.monotonic()>deadline-20:return 75
            masses=[item['masses'][arm][pos] for arm in WORLDS]
            tick=time.monotonic(); values, locations=context.candidate(adapter,cache,refs[pos],masses)
            patch=folder/'patch'/f'{pos:03d}.npz'
            # A complete NPZ without its JSON receipt is an interrupted publish;
            # reuse only after independently checking its masses/worlds and means.
            if patch.exists():
                with np.load(patch) as z:
                    assert list(z['worlds'])==list(WORLDS) and np.array_equal(z['M'],masses)
                    assert np.array_equal(z['maxsim'],values) and np.array_equal(z['argmax'],locations)
            else: npz_once(patch, worlds=np.asarray(WORLDS), M=np.asarray(masses), maxsim=values,argmax=locations,L=values.mean(1))
            rec=dict(protocol=pb,index=item['index'],query_id=item['query_id'],position=pos,
                candidate_id=item['post_row']['candidate_ids'][pos],patch_evidence=bind(patch),
                scores=values.mean(1).tolist(),seconds=time.monotonic()-tick)
            write(path,rec,immutable=True);records.append(rec)
        values=np.asarray([r['scores'] for r in records]).T; arms={}
        for i,arm in enumerate(WORLDS):
            post=decision(item['post_row'],values[i],head)
            logits=np.zeros(128);logits[item['post_row']['challenger_positions']]=post['logits']
            arms[arm]=dict(M=item['masses'][arm],L=values[i].tolist(),POST=dict(
                logits128=logits.tolist(),prediction_position=post['prediction_position'],switched=post['switched']),
                external=external(item['external'],item['masses'][arm]))
        oldpath=ROOT/'results/rc_postllm_h593_decomposition_v1/queries'/item['query_id']/'result.json'
        old=read(oldpath)['arms']['NATIVE']
        err=float(np.max(np.abs(np.asarray(old['L'])-values[-1])))
        assert err<LIMIT and old['decision']['prediction_position']==arms['ORIGINAL_HR1']['POST']['prediction_position'],('POST_REPLAY',item['index'],err)
        result=dict(status='CAUSAL128_FROZEN_BRIDGE_ROW_COMPLETE',protocol=pb,index=item['index'],
            query_id=item['query_id'],fold=item['fold'],component=item['component'],worlds=arms,head=head,
            original_anchor=dict(source=bind(oldpath),maximum_L_error=err,same_decision=True),
            candidate_results=[bind(folder/'candidates'/f'{pos:03d}.json') for pos in range(128)],
            query_hidden_source=context.bank.query_sources[item['query_id']],snapshot=p['endpoints'][str(item['fold'])]['snapshot'])
        write(done,result,immutable=True)
        print(json.dumps(dict(stage='FROZEN_BRIDGE_QUERY_DONE',index=item['index'])),flush=True)
        return 0


def worker(out, shard, shards, budget, index=None):
    p,pb=protocol(out); rows=read(p['inputs']['path'])['rows'];ctx=Context(p)
    assert 0<=shard<shards;deadline=time.monotonic()+budget
    selected = [rows[index]] if index is not None else rows[shard::shards]
    if index is not None: assert 0 <= index < 128
    for item in selected:
        if time.monotonic()>deadline-30:return 75
        code=row_worker(out,p,pb,item,ctx,deadline)
        if code:return code
    return 0


def target_wrong(row,item,label):
    targets=[i for i,identity in enumerate(item['post_row']['candidate_identities']) if identity==label['identity']]
    assert len(targets)<=1
    if not targets:return None,None
    target=targets[0];axis=item['post_row']['candidate_ids'];free=item['external']['free_content']
    wrong=max((i for i in range(128) if i!=target),key=lambda i:(free[i],-axis[i]))
    return target,wrong


def group_summary(rows,key):
    groups={}
    for row in rows:
        value=row.get(key)
        if value is not None:groups.setdefault(row['component'],[]).append(float(value))
    means=np.asarray([np.mean(v) for v in groups.values()]);n=len(means)
    if not n:return dict(groups=0,mean=None)
    rng=np.random.default_rng(20260927)
    boot=means[rng.integers(0,n,size=(10000,n))].mean(1)
    return dict(groups=n,mean=float(means.mean()),positive_groups=int((means>0).sum()),negative_groups=int((means<0).sum()),
                exploratory_bootstrap95=np.quantile(boot,[.025,.975]).tolist())


def property_contrasts(arms, metric):
    """Conditional path participation, not deletion of phase alone."""
    jp, j, p, neither = [arms[a][metric] for a in ARMS]
    return dict(P_with_J=jp-j, P_without_J=p-neither,
                J_with_P=jp-p, J_without_P=j-neither,
                J_P_interaction=jp-j-p+neither)


def independent_gap(item, masses, content, head, target, wrong):
    """Literal NumPy audit, independent of external() and decision()."""
    w=int(item['post_row']['winner_index']); raw=np.asarray(item['post_row']['raw_scores'],np.float64)
    m=np.asarray(masses,np.float64); L=np.asarray(content,np.float64)
    ext=item['external']; base=np.asarray(ext['native_scores'],np.float64)
    scores=base*(m/np.asarray(ext['original_M'],np.float64))[:,None]
    free=np.asarray(ext['free_content'],np.float64)
    challengers=ext['challengers']; raw_features={pos:ext['native_X'][j][0] for j,pos in enumerate(challengers)}
    def s(a,b):return (a-b)/(abs(a)+abs(b)+1e-12)
    def one(pos):
        if pos==w:return dict(POST_action=0.,NATIVE7_action=0.,M_FREE_action=0.)
        post=float(np.dot([(raw[pos]-raw[w])/max(float(raw.std()),1e-12),s(L[pos],L[w]),1.],head))
        sr,qr,rr=scores[pos]; sw,qw,rw=scores[w]
        x=[raw_features[pos],s(sr,sw),s(m[pos],m[w]),s(sr/max(m[pos],1e-12),sw/max(m[w],1e-12)),
           s(sr-qr,sw-qw),s(sr-rr,sw-rw),1.]
        y=[raw_features[pos],s(m[pos]*free[pos],m[w]*free[w]),s(m[pos],m[w]),s(free[pos],free[w]),0.,0.,1.]
        return dict(POST_action=post,NATIVE7_action=float(np.dot(x,ext['native_theta'])),
                    M_FREE_action=float(np.dot(y,ext['simple_theta'])))
    t,z=one(target),one(wrong)
    return dict(M_gap=float(m[target]-m[wrong]),logM_gap=float(np.log(m[target])-np.log(m[wrong])),
                POST_content_gap=float(L[target]-L[wrong]),
                **{k+'_gap':t[k]-z[k] for k in t})


def audit_joint_row(row,item,arms):
    target,wrong=row['target_position'],row['fixed_wrong_position']
    positions=sorted([target,wrong]);bypos={};maximum=0.
    assert len(row['patch_outputs'])==2 and len(row['saved_arm_inputs'])==2*len(arms)
    for index,(pos,patch) in enumerate(zip(positions,row['patch_outputs'])):
        with np.load(checked(patch)) as z:
            assert list(z['worlds'])==arms and np.isfinite(z['maxsim']).all()
            values=z['maxsim'].mean(1);assert np.max(np.abs(values-z['L']))<1e-14
            masses=[]
            for a,b in enumerate(row['saved_arm_inputs'][index*len(arms):(index+1)*len(arms)]):
                saved=torch.load(checked(b),map_location='cpu',weights_only=True)
                assert saved['position']==pos and saved['arm']==arms[a]
                sides=[side['weights'].numpy() for side in saved['sides']]
                mass=float(np.sqrt(np.mean(sides[0])*np.mean(sides[1])))
                assert abs(mass-z['M'][a])<LIMIT and abs(mass-saved['M'])<LIMIT
                masses.append(mass)
            bypos[pos]=dict(M=masses,L=values)
    original=read(checked(row['original_POST']))['arms']['NATIVE']['L']
    for a,arm in enumerate(arms):
        ms=np.asarray(item['masses']['ORIGINAL_HR1']).copy();ls=np.asarray(original).copy()
        for pos in positions:ms[pos]=bypos[pos]['M'][a];ls[pos]=bypos[pos]['L'][a]
        independent=independent_gap(item,ms,ls,row['head'],target,wrong)
        error=max(abs(independent[k]-row['arms'][arm][k]) for k in independent)
        assert error<LIMIT,('JOINT_NUMPY_ACTION_REPLAY',row['index'],arm,error)
        maximum=max(maximum,error)
    return maximum


def validate_scores(out,p,pb,item):
    folder=out/'queries'/f"{item['index']:03d}";row=read(folder/'result.json')
    assert row['protocol']==pb and row['query_id']==item['query_id']
    vals=[];maxerr=0.
    for pos,b in enumerate(row['candidate_results']):
        rec=read(checked(b));assert rec['position']==pos and rec['candidate_id']==item['post_row']['candidate_ids'][pos]
        with np.load(checked(rec['patch_evidence'])) as z:
            assert list(z['worlds'])==list(WORLDS)
            assert np.array_equal(z['M'],[item['masses'][a][pos] for a in WORLDS])
            means=z['maxsim'].mean(1);assert np.isfinite(means).all()
            error=float(np.max(np.abs(means-np.asarray(rec['scores']))));assert error<1e-14
            vals.append(means);maxerr=max(maxerr,error)
    values=np.asarray(vals).T
    for i,arm in enumerate(WORLDS):
        assert np.array_equal(values[i],row['worlds'][arm]['L'])
        post=decision(item['post_row'],values[i],row['head'])
        got=np.asarray(row['worlds'][arm]['POST']['logits128'])[item['post_row']['challenger_positions']]
        assert np.max(np.abs(got-post['logits']))<1e-14
        assert post['prediction_position']==row['worlds'][arm]['POST']['prediction_position']
        assert external(item['external'],item['masses'][arm])==row['worlds'][arm]['external']
    return row,maxerr


def join(out):
    p,pb=protocol(out);items=read(p['inputs']['path'])['rows'];sealed=[];maximum=0.
    for item in items:
        row,error=validate_scores(out,p,pb,item);maximum=max(maximum,error);sealed.append(row)
    labels={r['query_id']:r for r in read(checked(p['original_label_source']))['rows']}
    records=[];summaries={}
    for item,row in zip(items,sealed):
        lab=labels[item['query_id']];target,wrong=target_wrong(row,item,lab)
        native=row['worlds']['A1J1P1'];w=item['post_row']['winner_index'];identity=lab['identity']
        rec=dict(index=item['index'],query_id=item['query_id'],component=item['component'],fold=item['fold'],
            mechanism_group_replication=item['mechanism_group_replication'],target_present=target is not None,
            target_position=target,fixed_wrong_position=wrong,raw_correct=item['post_row']['candidate_identities'][w]==identity,arms={})
        for arm in WORLDS:
            world=row['worlds'][arm];metrics={}
            if target is not None:
                mt,mw=world['M'][target],world['M'][wrong]
                metrics.update(M_gap=mt-mw,logM_gap=float(np.log(mt)-np.log(mw)),
                    POST_content_gap=world['L'][target]-world['L'][wrong],
                    POST_action_gap=world['POST']['logits128'][target]-world['POST']['logits128'][wrong])
                for name in ('NATIVE7','M_FREE'):
                    z=world['external'][name]['logits128'];metrics[name+'_action_gap']=z[target]-z[wrong]
            for name,pred in [('POST',world['POST']),*world['external'].items()]:
                pick=pred['prediction_position'];metrics[name+'_correct']=item['post_row']['candidate_identities'][pick]==identity
                metrics[name+'_prediction_position']=pick;metrics[name+'_switched']=pick!=w
            rec['arms'][arm]=metrics
        records.append(rec)
    for panel,rs in [('FULL128',records),('GROUP_DISJOINT12',[r for r in records if r['mechanism_group_replication']])]:
        summary={}
        for arm in WORLDS:
            value={}
            for model in ('POST','NATIVE7','M_FREE'):
                key=model+'_correct';correct=sum(r['arms'][arm][key] for r in rs)
                value[model]=dict(correct=correct,total=len(rs),rescues=sum(not r['raw_correct'] and r['arms'][arm][key] for r in rs),
                    breaks=sum(r['raw_correct'] and not r['arms'][arm][key] for r in rs),
                    switches=sum(r['arms'][arm][model+'_switched'] for r in rs))
            effects=[]
            for r in rs:
                if r['target_present']:
                    effect=dict(component=r['component'])
                    for key in ('M_gap','logM_gap','POST_content_gap','POST_action_gap','NATIVE7_action_gap','M_FREE_action_gap'):
                        effect[key]=r['arms'][arm][key]-r['arms']['A1J1P1'][key]
                    effects.append(effect)
            value['delta_vs_same_COARSE_native']={key:group_summary(effects,key) for key in
                ('M_gap','logM_gap','POST_content_gap','POST_action_gap','NATIVE7_action_gap','M_FREE_action_gap')}
            summary[arm]=value
        property_effects={}
        for metric in ('M_gap','logM_gap','POST_content_gap','POST_action_gap','NATIVE7_action_gap','M_FREE_action_gap'):
            effects=[dict(component=r['component'],**property_contrasts(r['arms'],metric))
                     for r in rs if r['target_present']]
            property_effects[metric]={key:group_summary(effects,key) for key in
                ('P_with_J','P_without_J','J_with_P','J_without_P','J_P_interaction')}
        summaries[panel]=dict(worlds=summary,conditional_path_effects=property_effects)
    result=dict(status='CAUSAL128_FROZEN_BRIDGE_COMPLETE',protocol=pb,queries=128,summary=summaries,rows=records,
        scope=p['scope'],definition=p['definition'],accuracy_is_secondary=True,unique_causality_claim=False)
    write(out/'result.json',result,immutable=True)
    lines=['# Front128 frozen M-only causal bridge','',p['definition'],'',p['scope'],'',
        'Each intervention is compared with the same saved COARSE A1J1P1 anchor. ORIGINAL_HR1 is a separate original-model replay. No model was trained.',
        '', '| Panel | Endpoint | P with J | P without J | J/P interaction |', '|---|---|---:|---:|---:|']
    for panel,summary in summaries.items():
        for metric in ('logM_gap','NATIVE7_action_gap','M_FREE_action_gap','POST_content_gap','POST_action_gap'):
            effects=summary['conditional_path_effects'][metric]
            lines.append(f"| {panel} | {metric} | {effects['P_with_J']['mean']:.7g} | {effects['P_without_J']['mean']:.7g} | {effects['J_P_interaction']['mean']:.7g} |")
    lines.extend(['','Full candidate decisions are secondary mechanism endpoints:','',
        '| Panel | M source | POST correct | NATIVE7 correct | M_FREE correct |', '|---|---|---:|---:|---:|'])
    for panel,summary in summaries.items():
        for arm,v in summary['worlds'].items():lines.append(f"| {panel} | {arm} | {v['POST']['correct']}/{v['POST']['total']} | {v['NATIVE7']['correct']}/{v['NATIVE7']['total']} | {v['M_FREE']['correct']}/{v['M_FREE']['total']} |")
    lines.extend(['','Group-equal means, exploratory group-bootstrap intervals, fixed-target-versus-wrong M/logM and downstream content/action contrasts are recorded in result.json.',
        'This tests propagation through the existing M interface, not a unique cause or removal of all position information; A/J may retain other spatial signals.'])
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    write(out/'validation.json',dict(status='CAUSAL128_FROZEN_BRIDGE_INDEPENDENT_ARITHMETIC_PASS',protocol=pb,
        queries=128,candidates=16384,worlds=len(WORLDS),max_patch_mean_error=maximum,
        result=bind(out/'result.json'),report=bind(out/'report.md'),
        rows=[bind(out/'queries'/f'{i:03d}'/'result.json') for i in range(128)],
        original_HR1_anchor_all_same_decision=True,labels_joined_after_all_candidate_artifacts=True),immutable=True)
    print(json.dumps(dict(status=result['status'],queries=128)),flush=True)


def joint_pairs(out,budget):
    """Propagate existing 14 CPU pairs, without producing full-axis rankings.

Only the two sealed target/fixed-wrong M values change; the original HR1 M/L
at the RAW winner is fixed if that winner is neither pair. This explicit sparse
intervention permits action-gap propagation but not a full upstream C128 claim.
"""
    p,pb=protocol(out);allitems={r['index']:r for r in read(p['inputs']['path'])['rows']}
    oldv=read(RESTORE/'validation.json');assert oldv['status']=='PROPERTY_FACTORIAL_FIXED14_18_INDEPENDENT_PASS'
    rp=read(RESTORE/'protocol.json');historical={r['index']:r for r in read(HISTORICAL)['phase_rows']}
    phasev=read(PHASE/'validation.json');assert phasev['status']=='PHASE_BRIDGE_PANEL8_INDEPENDENT_PASS'
    for b in phasev['audits']:checked(b)
    sources=dict(restoration_protocol=bind(RESTORE/'protocol.json'),restoration_validation=bind(RESTORE/'validation.json'),
        historical_fixed_pairs=bind(HISTORICAL),prior_phase_bridge=bind(PHASE/'validation.json'),
        prior_phase_result=bind(PHASE/'result.json'))
    folder=out/'joint_pairs';folder.mkdir(parents=True,exist_ok=True);start=time.monotonic();ctx=Context(p)
    with (folder/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'validation.json').exists():
            assert read(folder/'validation.json')['protocol']==pb;return 0
        completed=[]
        for worker in rp['workers']:
            index=worker['index']
            if not worker['positions']:continue
            item=allitems[index];dest=folder/f'query{index:03d}.json'
            if dest.exists():
                result=read(dest);assert result['protocol']==pb and result['sources']==sources
                completed.append(bind(dest));continue
            if time.monotonic()-start>budget-30:return 75
            adapter,head,cache,refs=ctx.load(item); src=historical[index]
            target=int(src['target_position']);wrong=int(src['arm_metrics']['NATIVE']['strongest_position'])
            assert sorted([target,wrong])==worker['positions']
            original_path=ROOT/'results/rc_postllm_h593_decomposition_v1/queries'/item['query_id']/'result.json'
            original=read(original_path)['arms']['NATIVE']['L'];pairdata={};pairsources=[]
            arms=rp['arms'];metrics={};patchbindings=[]
            for pos in worker['positions']:
                pr=read(RESTORE/f'query{index:03d}/pair{pos:03d}/validation.json')
                assert pr['status']=='PROPERTY_RESTORE_CPU_PAIR_PASS' and pr['protocol']==sources['restoration_protocol']
                masses=[]
                for arm in arms:
                    saved=next(a for a in pr['arms'] if a['arm']==arm)
                    rec=torch.load(checked(saved['payload']),map_location='cpu',weights_only=True)
                    measured=float(torch.sqrt(rec['sides'][0]['weights'].mean()*rec['sides'][1]['weights'].mean()))
                    assert abs(measured-rec['M'])<1e-14 and abs(measured-saved['M'])<1e-14
                    masses.append(measured);pairsources.append(saved['payload'])
                values,locations=ctx.candidate(adapter,cache,refs[pos],masses)
                patch=folder/f'query{index:03d}_pair{pos:03d}.npz'
                if patch.exists():
                    with np.load(patch) as z:assert np.array_equal(z['maxsim'],values) and np.array_equal(z['M'],masses)
                else:npz_once(patch,worlds=np.asarray(arms),M=np.asarray(masses),maxsim=values,argmax=locations,L=values.mean(1))
                patchbindings.append(bind(patch));pairdata[pos]=dict(M=np.asarray(masses),L=values.mean(1))
            for a,arm in enumerate(arms):
                ms=np.asarray(item['masses']['ORIGINAL_HR1']).copy();ls=np.asarray(original).copy()
                for pos in worker['positions']:ms[pos]=pairdata[pos]['M'][a];ls[pos]=pairdata[pos]['L'][a]
                post=decision(item['post_row'],ls,head);z=np.zeros(128);z[item['post_row']['challenger_positions']]=post['logits']
                ext=external(item['external'],ms)
                metrics[arm]=dict(M_gap=float(ms[target]-ms[wrong]),logM_gap=float(np.log(ms[target])-np.log(ms[wrong])),
                    POST_content_gap=float(ls[target]-ls[wrong]),POST_action_gap=float(z[target]-z[wrong]),
                    NATIVE7_action_gap=ext['NATIVE7']['logits128'][target]-ext['NATIVE7']['logits128'][wrong],
                    M_FREE_action_gap=ext['M_FREE']['logits128'][target]-ext['M_FREE']['logits128'][wrong])
            result=dict(status='JOINT_RESTORATION_FIXED_PAIR_PROPAGATION_COMPLETE',protocol=pb,index=index,query_id=item['query_id'],
                component=item['component'],target_position=target,fixed_wrong_position=wrong,RAW_winner=item['post_row']['winner_index'],
                scope='Two presealed candidate M interventions; unchanged original HR1 M/L anchor if RAW winner is a third candidate. No full-C128 prediction/rank/accuracy output.',
                sources=sources,head=head,original_POST=bind(original_path),saved_arm_inputs=pairsources,patch_outputs=patchbindings,arms=metrics)
            write(dest,result,immutable=True);completed.append(bind(dest))
        assert len(completed)==7
        allrows=[read(checked(b)) for b in completed];phasearms=rp['phase_arms'];contrasts={}
        maximum_error=max(audit_joint_row(row,allitems[row['index']],rp['arms']) for row in allrows)
        for metric in ('M_gap','logM_gap','POST_content_gap','POST_action_gap','NATIVE7_action_gap','M_FREE_action_gap'):
            per=[]
            for row in allrows:
                def cell(prefix,scope):return float(np.mean([row['arms'][prefix+a][metric] for a in phasearms if a.startswith(scope)]))
                a0g,a0l,a1g,a1l=cell('','GLOBAL'),cell('','LOCAL'),cell('AMP_PERMUTE__','GLOBAL'),cell('AMP_PERMUTE__','LOCAL')
                per.append(dict(component=row['component'],phase_restore_after_amplitude_damage=a1g-a1l,
                    amplitude_restore_under_local_damage=a0l-a1l,interaction=(a0g-a0l)-(a1g-a1l)))
            contrasts[metric]={key:group_summary(per,key) for key in ('phase_restore_after_amplitude_damage','amplitude_restore_under_local_damage','interaction')}
        result=dict(status='JOINT_RESTORATION_FIXED14_FROZEN_BRIDGE_COMPLETE',protocol=pb,sources=sources,queries=7,pairs=14,
            arms=arms,contrasts=contrasts,rows=completed,full_C128_predictions_computed=False,training=False,
            definition=allrows[0]['scope'],prior_phase_bridge_reused_without_refitting=True,
            prior_phase_summary_LOCAL_minus_GLOBAL=read(checked(sources['prior_phase_result']))['summary_LOCAL_minus_GLOBAL'],
            sign_convention='Historical phase summary is LOCAL minus GLOBAL; current restoration contrast is GLOBAL minus LOCAL under amplitude permutation.')
        write(folder/'result.json',result,immutable=True)
        write(folder/'validation.json',dict(status='JOINT_RESTORATION_FIXED14_BRIDGE_PASS',protocol=pb,result=bind(folder/'result.json'),
            queries=7,pairs=14,arms=18,rows=completed,sources=sources,
            independent_numpy_M_patch_mean_and_action_gap_max_error=maximum_error,
            raw_winner_zero_logit_explicit=True),immutable=True)
        return 0


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','worker','join','joint-pairs'])
    ap.add_argument('--out','--root',dest='out',type=Path,default=DEFAULT);ap.add_argument('--source-root',type=Path,default=DEFAULT.parent/'paths')
    ap.add_argument('--index',type=int);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=16)
    ap.add_argument('--budget',type=float,default=430)
    args=ap.parse_args();configure()
    if args.command=='prepare':prepare(args.out,args.source_root);return 0
    if args.command=='worker':return worker(args.out,args.shard,args.shards,args.budget,args.index)
    if args.command=='join':join(args.out);return 0
    return joint_pairs(args.out,args.budget)

if __name__=='__main__':sys.exit(main())
