#!/usr/bin/env python3
"""Single-fold frozen ColNomic-to-RoMa-mass diagnostic, CPU only."""
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import run_rc_h593_pair_quality_cpu_v1 as Q
import rc_pair_quality_core_v1 as C
from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import candidate_feature
OUT=ROOT/'results/rc_fold0_mass_teacher_v1'
AUTH=ROOT/'registry/rc_fold0_mass_teacher_authority_v1_20260923.json'
PLAN=ROOT/'plan/RC_FOLD0_ROMA_M_TEACHER_DIAGNOSTIC_V1_20260923.md'
STEPS=2000
EPS=1e-8
read,write,save,bind=Q.read,Q.write,Q.save,Q.bind


def checked(b):
    assert bind(b['path'])==b,('SHA_DRIFT',b['path'])
    return Path(b['path'])


def prepare():
    assert not AUTH.exists()
    parent=read(Q.AUTH);ready=read(checked(parent['ready']));catalog=read(checked(parent['catalog']))
    oldfit=read(Q.OUT/'fit01/payload.json');oldval=read(Q.OUT/'fit01/validation.json')
    assert oldval['payload']==bind(Q.OUT/'fit01/payload.json') and oldfit['authority']==bind(Q.AUTH)
    train_ids=set(oldfit['train_query_ids']);split=read(checked(parent['public_sources']['split']))['folds'][0]
    held_ids=set(split['heldout_query_ids']);assert len(train_ids)==457 and len(held_ids)==119 and not train_ids&held_ids
    records={'train':[],'held':[]};input_sources=[]
    for meta in catalog['queries']:
        qid=meta['query_id']
        if qid not in train_ids|held_ids:continue
        p=torch.load(checked(meta['payload']),map_location='cpu',weights_only=True)
        assert p['query_id']==qid and len(p['pairs'])==128
        rec=dict(query_id=qid,execution_ordinal=meta['execution_ordinal'],query_image_key=p['query_image_key'],
                 candidate_physical_rows=p['candidate_physical_rows'],candidate_raw_scores=p['candidate_raw_scores'],
                 winner=p['winner'],challenger_positions=p['challenger_positions'],
                 reference_keys=[x['image_key'] for x in p['pairs']])
        group='train' if qid in train_ids else 'held'
        if group=='train':rec['teacher_M']=[float(x['native_c4'][1]) for x in p['pairs']]
        records[group].append(rec);input_sources.append(meta['payload'])
    for split_name,items in records.items():
        write(OUT/(split_name+'_inputs.json'),dict(records=items,teacher_included=split_name=='train',identity_labels_included=False))
    constant=float(np.mean([m for r in records['train'] for m in r['teacher_M']]))
    sources=[Path(__file__),Path(C.__file__),Path(Q.__file__),PLAN,
             ROOT/'programs/dispatch_rc_fold0_mass_teacher_v1.py',ROOT/'slurm/rc_fold0_mass_teacher_v1.sbatch',
             ROOT/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py']
    write(AUTH,dict(status='FOLD0_M_TEACHER_DIAGNOSTIC_AUTHORIZED',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources=[bind(p) for p in sources],parent=bind(Q.AUTH),ready=parent['ready'],
        train_inputs=bind(OUT/'train_inputs.json'),held_inputs=bind(OUT/'held_inputs.json'),
        original_input_sources=input_sources,previous_identity_fit=bind(Q.OUT/'fit01/payload.json'),
        frozen_original_head=parent['fold_sources']['0']['full_payload'],
        frozen_simple_model=bind(Q.OUT/'fit05/final_model.pt'),frozen_simple_payload=bind(Q.OUT/'fit05/payload.json'),
        operator_root=str(ROOT/'results/rc_h593_quality_operator_v1'),
        gallery=parent['public_sources']['gallery'],curator=parent['join_sources']['curator'],
        train_roles=parent['fold_sources']['0']['train_roles'],steps=STEPS,epsilon=EPS,
        constant_mass=constant,fold=0,train_queries=457,held_queries=119,
        student='COL_ONLY_PAIR',quality_parameters=8353,seed=17,lr=.001,weight_decay=.001,
        loss='Mean squared log mass error on all128; includes candidate-centered error and mean-log error',
        no_identity_training=True,teacher_diagnostic_not_original_retrieval_only=True,
        new_encoder_forwards=0,new_RoMa_forwards=0,max_chunks=128,chunk_seconds=450))
    # Streamed full-candidate gradient must match a joint graph, including near-zero teacher.
    import copy
    gen=torch.Generator().manual_seed(91)
    query=dict(z=torch.randn(4,128,generator=gen,dtype=torch.float64),xy=torch.rand(4,2,generator=gen,dtype=torch.float64))
    refs=[dict(z=torch.randn(5,128,generator=gen,dtype=torch.float64),xy=torch.rand(5,2,generator=gen,dtype=torch.float64)) for _ in range(3)]
    m1=C.Quality()
    with torch.no_grad():m1.up.weight.normal_(generator=gen,std=.05)
    m2=copy.deepcopy(m1);teacher=torch.tensor([0.,.01,.2],dtype=torch.float64)
    predictions=torch.stack([C.pair(m1,query,ref,'PAIR')[0] for ref in refs])
    loss=((torch.log(predictions+EPS)-torch.log(teacher+EPS))**2).mean();loss.backward()
    residual=[]
    for i,ref in enumerate(refs):
        v=C.pair(m2,query,ref,'PAIR')[0];delta=torch.log(v+EPS)-torch.log(teacher[i]+EPS)
        (delta.square()/3).backward();residual.append(float(delta.detach()))
    error=max(float((p.grad-q.grad).abs().max()) for p,q in zip(m1.parameters(),m2.parameters()))
    assert error<2e-12 and abs(float(loss.detach())-(np.var(residual)+np.mean(residual)**2))<2e-12
    scores=[dict(real_score=.02,visibility_mass=.1,query_control_score=.01,reference_control_score=.015),
            dict(real_score=0.,visibility_mass=0.,query_control_score=0.,reference_control_score=0.),
            dict(real_score=.03,visibility_mass=.2,query_control_score=.025,reference_control_score=.01)]
    changed=mass_only(scores,np.array([.3,.05,.12]));raw=[.2,.3,.1];ix=[0,2]
    tx=torch.stack([candidate_feature(raw,dict(enumerate(changed)),i,1) for i in ix])
    replay_error=float(abs(tx.numpy()-features_numpy(raw,changed,1,ix)).max());assert replay_error<2e-12
    assert mass_only(scores,np.array([.1,0.,.2]))==scores
    write(OUT/'preflight.json',dict(status='TEACHER_STREAM_GRADIENT_AND_LOG_DECOMPOSITION_PASS',authority=bind(AUTH),max_gradient_error=error,
                                   mass_only_feature_numpy_error=replay_error,zero_mass_and_identity_substitution_pass=True))
    print(dict(event='PREPARED',train=457,held=119,constant_mass=constant),flush=True)


def guard(stage):
    a=read(AUTH)
    for b in a['sources']+[a['parent'],a['ready'],a['train_inputs'],a['held_inputs']]:checked(b)
    assert read(OUT/'preflight.json')['authority']==bind(AUTH)
    assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available()
    allowed={Path(a[k]['path']).resolve() for k in ('ready','parent','train_inputs','held_inputs')}
    allowed.update(Path(b['path']).resolve() for b in read(a['ready']['path'])['features'].values())
    if stage=='join':
        allowed.update(Path(a[k]['path']).resolve() for k in ('frozen_original_head','frozen_simple_model','frozen_simple_payload','gallery','curator','train_roles'))
        simp=read(checked(a['frozen_simple_payload']));allowed.update(Path(p['path']).resolve() for p in simp['predictions'])
    cache=Q.CACHE.resolve();own=OUT.resolve();operator=Path(a['operator_root']).resolve()
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();s=str(path).lower()
        assert not any(x in s for x in ('d1-mi','d1_mi','formal392','/grozi/','/isic/')),s
        if stage!='join':assert 'curator_roles' not in s and '/reports/' not in s,s
        if ROOT/'results' in path.parents:
            assert own in path.parents or cache/'images' in path.parents or path in allowed or (stage=='join' and operator in path.parents),('UNLISTED_RESULT',s)
    sys.addaudithook(audit)
    return a


def estimate(rec,bank,model,traces=False):
    query=bank(rec['query_image_key']);values=[];details=[]
    for key in rec['reference_keys']:
        value,trace=C.pair(model,query,bank(key),'PAIR',traces);values.append(value)
        if traces:details.append(dict(u=trace['u'].cpu(),v=trace['v'].cpu(),
                                      query_relation_summary=trace['query_relation'][:,-8:].mean(0).cpu(),
                                      reference_relation_summary=trace['reference_relation'][:,-8:].mean(0).cpu()))
    return torch.stack(values),details


def fit(a):
    if (OUT/'fit_validation.json').exists():checked(read(OUT/'fit_validation.json')['payload']);return
    start=time.monotonic();train=read(a['train_inputs']['path'])['records'];held=read(a['held_inputs']['path'])['records']
    ready=read(a['ready']['path']);bank=Q.Bank(ready['features'],'COL_ONLY','cpu');model=C.Quality()
    opt=torch.optim.AdamW(model.parameters(),lr=a['lr'],weight_decay=a['weight_decay']);step=0;history=[]
    cp=OUT/'checkpoint.pt'
    if cp.exists():
        p=torch.load(cp,map_location='cpu',weights_only=True);assert p['authority']==bind(AUTH)
        model.load_state_dict(p['model']);opt.load_state_dict(p['optimizer']);step=p['step'];history=p['history']
    def checkpoint():save(cp,dict(authority=bind(AUTH),step=step,model=model.state_dict(),optimizer=opt.state_dict(),history=history),mutable=True)
    def receipt(phase):
        write(OUT/'chunks'/f"{os.environ['SLURM_JOB_ID']}.json",dict(status='M_TEACHER_NORMAL_CHUNK',phase=phase,step=step,authority=bind(AUTH),seconds=time.monotonic()-start))
    while step<STEPS:
        if time.monotonic()-start>a['chunk_seconds']:checkpoint();receipt('train');return
        epoch,offset=divmod(step,len(train));g=torch.Generator().manual_seed(17+1000003*epoch)
        rec=train[int(torch.randperm(len(train),generator=g)[offset])];query=bank(rec['query_image_key']);opt.zero_grad();residual=[]
        for key,target in zip(rec['reference_keys'],rec['teacher_M']):
            value,_=C.pair(model,query,bank(key),'PAIR');delta=torch.log(value+EPS)-np.log(target+EPS)
            (delta.square()/128).backward();residual.append(float(delta.detach()))
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step();step+=1
        if step%25==0:
            history.append(dict(step=step,query_id=rec['query_id'],log_mse=float(np.mean(np.square(residual))),
                                centered_log_mse=float(np.var(residual)),mean_log_error=float(np.mean(residual))))
            checkpoint();print(dict(event='TEACHER_UPDATE',**history[-1],seconds=time.monotonic()-start),flush=True)
    checkpoint();final=OUT/'final_model.pt'
    if not final.exists():save(final,dict(authority=bind(AUTH),model=model.state_dict(),step=step))
    frozen=torch.load(final,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,frozen['model'][k]) for k,v in model.state_dict().items())
    bindings=[]
    for split_name,records in (('train',train),('held',held)):
        for rec in records:
            dest=OUT/'predictions'/split_name/f"query{rec['execution_ordinal']:03d}.pt"
            if dest.exists():
                b=bind(dest);p=torch.load(dest,map_location='cpu',weights_only=True);assert p['model']==bind(final)
                bindings.append(dict(split=split_name,query_id=rec['query_id'],prediction=b));continue
            if time.monotonic()-start>a['chunk_seconds']:receipt('predict_'+split_name);return
            with torch.no_grad():m,traces=estimate(rec,bank,model,split_name=='held')
            assert m.shape==(128,) and torch.isfinite(m).all() and (m>=0).all() and (m<=1).all()
            if traces:
                rebuilt=np.array([np.sqrt(t['u'].numpy().mean()*t['v'].numpy().mean()) for t in traces]);assert np.max(abs(rebuilt-m.numpy()))<2e-12
            save(dest,dict(authority=bind(AUTH),model=bind(final),query_id=rec['query_id'],execution_ordinal=rec['execution_ordinal'],
                           candidate_physical_rows=rec['candidate_physical_rows'],M=m.cpu(),quality_traces=traces,teacher_at_inference=False,held_identity_reads=0))
            bindings.append(dict(split=split_name,query_id=rec['query_id'],prediction=bind(dest)))
    write(OUT/'prediction_seal.json',dict(status='M_TEACHER_PREDICTIONS_SEALED',authority=bind(AUTH),model=bind(final),predictions=bindings,train=457,held=119,held_teacher_reads=0,held_identity_reads=0))
    write(OUT/'fit_validation.json',dict(status='M_TEACHER_FIT_AND_PREDICTIONS_PASS',authority=bind(AUTH),payload=bind(OUT/'prediction_seal.json')))
    receipt('complete')


def features_numpy(raw,scores,w,ix):
    raw=np.array(raw);s=np.array([v['real_score'] for v in scores]);m=np.array([v['visibility_mass'] for v in scores]);l=s/np.maximum(m,1e-12)
    q=s-np.array([v['query_control_score'] for v in scores]);r=s-np.array([v['reference_control_score'] for v in scores])
    def sym(v):return (v[ix]-v[w])/(abs(v[ix])+abs(v[w])+1e-12)
    return np.stack([(raw[ix]-raw[w])/max(float(raw.std()),1e-12),sym(s),sym(m),sym(l),sym(q),sym(r)],1)


def mass_only(native,m):
    result=[]
    for old,new in zip(native,m):
        denom=old['visibility_mass'];ratio=float(new)/denom if denom>0 else 0.
        result.append({k:(float(new) if k=='visibility_mass' else float(v)*ratio) for k,v in old.items()})
    return result


def quality(pred,teacher):
    delta=np.log(pred+EPS)-np.log(teacher+EPS)
    tri=np.triu(np.ones((128,128),dtype=bool),1);td=teacher[:,None]-teacher[None,:];sd=pred[:,None]-pred[None,:];valid=tri&(abs(td)>1e-12)
    agreement=(np.sign(td[valid])==np.sign(sd[valid])).astype(float);agreement[sd[valid]==0]=.5
    return dict(log_mse=float(np.mean(delta**2)),centered_log_mse=float(np.var(delta)),mean_log_error=float(delta.mean()),
                rmse=float(np.sqrt(np.mean((pred-teacher)**2))),pair_order_agreement=float(agreement.mean()) if valid.any() else None)


def join(a):
    v=read(OUT/'fit_validation.json');seal=read(checked(v['payload']));assert seal['authority']==bind(AUTH)
    for b in seal['predictions']:checked(b['prediction'])
    train=read(a['train_inputs']['path'])['records'];held=read(a['held_inputs']['path'])['records'];records={r['query_id']:r for r in train+held}
    pred={b['query_id']:torch.load(b['prediction']['path'],map_location='cpu',weights_only=True) for b in seal['predictions']}
    original=read(checked(a['frozen_original_head']));theta=torch.tensor([float.fromhex(x) for x in original['parameters']['COST1']],dtype=torch.float64)
    old={p['query_id']:p['models']['COST1'] for p in original['predictions']}
    simp=read(checked(a['frozen_simple_payload']));simple={}
    for b in simp['predictions']:
        p=torch.load(checked(b),map_location='cpu',weights_only=True);simple[p['query_id']]=p
    theta5=torch.load(checked(a['frozen_simple_model']),map_location='cpu',weights_only=True)['theta']
    # No optimization after this point; held teacher and identities are analysis only.
    labels={r['physical_row']:r['identity'] for r in read(checked(a['gallery']))['records']}
    roles={r['query_id']:r for r in read(checked(a['curator']))['records']}
    trained=read(checked(a['train_roles']))['records'];train_groups={r['component'] for r in trained}
    qrows=[];rows=[];maximum=0.;checks=0
    for split_name,items in (('train',train),('held',held)):
        for rec in items:
            qid=rec['query_id'];student=pred[qid]['M'].numpy()
            if split_name=='train':teacher=np.array(rec['teacher_M'])
            else:
                folder=Path(a['operator_root'])/f"query{rec['execution_ordinal']:03d}";ov=read(folder/'validation.json');op=read(checked(ov['payload']))
                assert ov['status']=='QUALITY_OPERATOR_QUERY_PASS' and op['query_id']==qid and op['candidate_physical_rows']==rec['candidate_physical_rows']
                native=op['modes']['NATIVE']['scores'];teacher=np.array([s['visibility_mass'] for s in native])
                assert np.array_equal(teacher,simple[qid]['M'].numpy())
            qrows.append(dict(query_id=qid,split=split_name,student=quality(student,teacher),constant=quality(np.full(128,a['constant_mass']),teacher)))
            if split_name=='train':continue
            role=roles[qid];assert role['outer_fold']==0 and role['component'] not in train_groups
            w=rec['winner'];ix=rec['challenger_positions'];cand=rec['candidate_physical_rows'];raw=rec['candidate_raw_scores'];models={}
            def selected(z):
                j=int(np.argmax(z));return cand[ix[j] if z[j]>0 else w]
            variants={'NATIVE':teacher,'STUDENT':student,'STUDENT_CBIND':np.roll(student,1),'CONSTANT':np.full(128,a['constant_mass'])}
            for name,m in variants.items():
                scores=native if name=='NATIVE' else mass_only(native,m)
                x=torch.stack([candidate_feature(raw,dict(enumerate(scores)),j,w) for j in ix]);z=x@theta[:-1]+theta[-1]
                nz=(features_numpy(raw,scores,w,ix)*theta[:-1].numpy()).sum(1)+float(theta[-1]);error=float(np.max(abs(nz-z.numpy())));assert error<2e-10
                maximum=max(maximum,error);checks+=127
                if name=='NATIVE':
                    assert torch.equal(x,torch.tensor(op['modes']['NATIVE']['X'],dtype=torch.float64))
                    assert [float(y).hex() for y in z]==old[qid]['logits_hex'] and selected(z.numpy())==old[qid]['selected']
                models['ORIGINAL7_'+name]=dict(selected=selected(z.numpy()),logits=z.tolist())
                x5=C.features(raw,w,torch.tensor(m),simple[qid]['L0']);z5=x5@theta5[:-1]+theta5[-1]
                if name=='NATIVE':assert torch.equal(z5,simple[qid]['logits'])
                models['FREE5_'+name]=dict(selected=selected(z5.numpy()),logits=z5.tolist())
            models['RAW']=dict(selected=cand[w])
            for val in models.values():val['correct']=labels[val['selected']]==role['identity']
            target=[i for i,physical in enumerate(cand) if labels[physical]==role['identity']]
            row=dict(query_id=qid,original_query_id=role['original_query_id'],component=role['component'],target_in_C128=bool(target),models=models,
                     student_M=student.tolist(),teacher_M=teacher.tolist(),quality=qrows[-1]['student'])
            if target:
                t=target[0];mask=np.arange(128)!=t
                row['student_target_strict_first']=bool(student[t]>student[mask].max());row['teacher_target_strict_first']=bool(teacher[t]>teacher[mask].max())
            rows.append(row)
    assert len(rows)==119
    origres=[r for r in rows if not r['models']['RAW']['correct'] and r['models']['ORIGINAL7_NATIVE']['correct']]
    summary={}
    for name in rows[0]['models']:
        summary[name]=dict(correct=sum(r['models'][name]['correct'] for r in rows),n=119,
          rescue=sum(not r['models']['RAW']['correct'] and r['models'][name]['correct'] for r in rows),
          breaks=sum(r['models']['RAW']['correct'] and not r['models'][name]['correct'] for r in rows),
          original_rescues_retained=sum(r['models'][name]['correct'] for r in origres),
          original_decisions_same=sum(r['models'][name]['selected']==r['models']['ORIGINAL7_NATIVE']['selected'] for r in rows))
    assert summary['RAW']['correct']==98 and summary['ORIGINAL7_NATIVE']['correct']==105 and summary['FREE5_NATIVE']['correct']==104
    errors={s:{m:{k:float(np.mean([q[m][k] for q in qrows if q['split']==s and q[m][k] is not None])) for k in qrows[0][m]} for m in ('student','constant')} for s in ('train','held')}
    write(OUT/'result.json',dict(status='M_TEACHER_FIRST_FOLD_DIAGNOSTIC_COMPLETE',authority=bind(AUTH),summary=summary,quality_errors=errors,
          rows=rows,quality_rows=qrows,scope='Teacher-supervised mechanism diagnostic; frozen original7 M-only retains native local weights; frozen FREE5 has free content; no mainline replacement'))
    write(OUT/'validation.json',dict(status='M_TEACHER_NATIVE_PARITY_AND_NUMPY_PASS',result=bind(OUT/'result.json'),held=119,train=457,checked_logits=checks,max_logit_error=maximum))
    print(json.dumps(dict(summary=summary,quality_errors=errors)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','fit','join'));args=parser.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        a=guard(args.stage)
        fit(a) if args.stage=='fit' else join(a)
