#!/usr/bin/env python3
"""Fresh full-TRAIN fit only after the small-panel relative/absolute M gate."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import time
import fcntl
import numpy as np
import torch
import run_rc_fold0_mass_teacher_v1 as T
import run_rc_colnomic_m_scale_fit_v2 as S
import calibrate_rc_colnomic_m_scale_fit_v3 as G
import rc_m_logit_calibration_v3 as K

ROOT=T.ROOT
OUT=ROOT/'results/rc_fold0_m_scale_teacher_v2'
AUTH=ROOT/'registry/rc_fold0_m_scale_teacher_authority_v2_20260924.json'
PLAN=ROOT/'plan/RC_FOLD0_M_SCALE_TEACHER_V2_20260924.md'
LAUNCH=ROOT/'slurm/rc_fold0_m_scale_teacher_v2.sbatch'
OLD_AUTH=T.AUTH
T.OUT=OUT;T.AUTH=AUTH
read,write,save,bind,checked=T.read,T.write,T.save,T.bind,T.checked


def prepare():
    assert not AUTH.exists()
    gateval=read(G.OUT/'validation.json');res=read(checked(gateval['result']))
    assert res['status']=='SMALL_TRAIN_FIT_GATE_PASS' and res['selected_recipe'] in S.ARMS
    recipe=res['selected_recipe'];assert all(res['arms'][recipe]['gate'].values())
    assert recipe==next(arm for arm in S.ARMS if all(res['arms'][arm]['gate'].values()))
    a=read(OLD_AUTH);train=read(checked(a['train_inputs']))
    assert not train['identity_labels_included'] and train['teacher_included'] and len(train['records'])==457
    value=math.exp(float(np.log(np.array([r['teacher_M'] for r in train['records']])+T.EPS).mean()))-T.EPS
    assert 0<value<1
    a.update(status='M_SCALE_TEACHER_FOLD0_FROZEN',previous_teacher_authority=bind(OLD_AUTH),
        sources=[bind(p) for p in (Path(__file__),Path(T.__file__),Path(S.__file__),Path(S.D.__file__),
                 Path(T.C.__file__),Path(T.Q.__file__),Path(G.__file__),Path(K.__file__),
                 ROOT/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py',PLAN,LAUNCH)],
        small_panel_result=gateval['result'],small_panel_validation=bind(G.OUT/'validation.json'),
        selected_recipe=recipe,initial_mass=value,initial_bias=math.log(value/(1-value)),
        max_chunks=48,chunk_seconds=450,fresh_initialization=True,steps=2000,
        mass_calibration='One shared logit-M bias fitted on TRAIN after freezing network',
        extra_calibration_parameters=1,
        loss='query-centered log MSE (weight 1 or 4 after step256) plus mean-log residual squared',
        evidence='Reused development fold0: TRAIN457 effective, held119; not new external confirmation')
    write(AUTH,a)
    write(OUT/'preflight.json',dict(status='TRAIN_FIT_GATE_AND_EXISTING_PACKED_GRADIENT_PASS',authority=bind(AUTH),
                                  small_panel_validation=bind(G.OUT/'validation.json'),selected_recipe=recipe))
    print(dict(status='PREPARED',recipe=recipe,initial_mass=value,train=457,held=119),flush=True)


def fit(a):
    if (OUT/'fit_validation.json').exists():checked(read(OUT/'fit_validation.json')['payload']);return
    lock=(OUT/'fit.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();train=read(a['train_inputs']['path'])['records']
    bank=T.Q.Bank(read(a['ready']['path'])['features'],'COL_ONLY','cpu')
    T.Q.checked=checked
    model=T.C.Quality(seed=17)
    with torch.no_grad():model.up.bias.fill_(a['initial_bias'])
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    step=0;history=[];cp=OUT/'checkpoint.pt'
    if cp.exists():
        p=torch.load(cp,map_location='cpu',weights_only=True);assert p['authority']==bind(AUTH)
        model.load_state_dict(p['model']);opt.load_state_dict(p['optimizer']);step=p['step'];history=p['history']
    def checkpoint():save(cp,dict(authority=bind(AUTH),step=step,model=model.state_dict(),optimizer=opt.state_dict(),history=history),mutable=True)
    def receipt(phase):
        v=dict(status='NORMAL_CHUNK',phase=phase,step=step,seconds=time.monotonic()-start,authority=bind(AUTH))
        write(OUT/'chunks'/f"{os.environ['SLURM_JOB_ID']}.json",v);write(OUT/'status.json',v,mutable=True);print(v,flush=True)
    while step<2000:
        if time.monotonic()-start>a['chunk_seconds']:checkpoint();receipt('fit');return
        epoch,offset=divmod(step,len(train));g=torch.Generator().manual_seed(17+1000003*epoch)
        rec=train[int(torch.randperm(len(train),generator=g)[offset])];q=bank(rec['query_image_key'])
        rel=[]
        with torch.no_grad():
            for key in rec['reference_keys']:
                r=bank(key);rel.extend(T.C.relation(q['z'],r['z'],q['xy'],r['xy'],'PAIR'))
        packed=dict(x=torch.cat(rel),lengths=[len(x) for x in rel]);del rel
        opt.zero_grad();m,_=S.D.packed_mass(model,packed)
        target=torch.tensor(rec['teacher_M'],dtype=torch.float64)
        weight=4 if a['selected_recipe']=='BALANCED' and step>=256 else 1
        loss=S.objective(m,target,weight);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step();step+=1
        if step%25==0:
            history.append(dict(step=step,query_id=rec['query_id'],weight=weight,loss=float(loss.detach()),
                                metrics=S.D.metrics(m.detach().numpy(),target.numpy())))
            checkpoint();print(dict(event='TEACHER_UPDATE',step=step,seconds=time.monotonic()-start,**history[-1]['metrics']),flush=True)
        del m,loss,packed
    checkpoint();final=OUT/'final_model.pt'
    if not final.exists():save(final,dict(authority=bind(AUTH),model=model.state_dict(),step=step))
    frozen=torch.load(final,weights_only=True);assert all(torch.equal(v,frozen['model'][k]) for k,v in model.state_dict().items())
    # Absolute scale is fitted with ONE shared scalar on TRAIN only.
    train_mass=[];calibration_inputs=[]
    for rec in train:
        dest=OUT/'uncalibrated_train'/f"query{rec['execution_ordinal']:03d}.pt"
        if not dest.exists():
            if time.monotonic()-start>a['chunk_seconds']:receipt('calibration_train_predict');return
            with torch.no_grad():m,_=T.estimate(rec,bank,model,False)
            save(dest,dict(authority=bind(AUTH),model=bind(final),query_id=rec['query_id'],M=m))
        p=torch.load(dest,weights_only=True)
        assert p['authority']==bind(AUTH) and p['model']==bind(final) and p['query_id']==rec['query_id']
        train_mass.append(p['M'].numpy());calibration_inputs.append(bind(dest))
    calibration=OUT/'mass_calibration.json'
    target=np.asarray([rec['teacher_M'] for rec in train])
    if not calibration.exists():
        bias=K.fit(np.asarray(train_mass),target)
        write(calibration,dict(authority=bind(AUTH),model=bind(final),bias=bias,
            training_predictions=calibration_inputs,train=457,held_reads=0,parameters=1,
            residual_mean=float(np.mean(np.log(K.apply(train_mass,bias)+T.EPS)-np.log(target+T.EPS)))))
    cal=read(calibration)
    assert cal['authority']==bind(AUTH) and cal['model']==bind(final) and cal['training_predictions']==calibration_inputs
    assert abs(float(np.mean(np.log(K.apply(train_mass,cal['bias'])+T.EPS)-np.log(target+T.EPS))))<1e-12
    held=read(a['held_inputs']['path']);assert not held['teacher_included'] and not held['identity_labels_included']
    bindings=[]
    for split,records in (('train',train),('held',held['records'])):
        for rec in records:
            dest=OUT/'predictions'/split/f"query{rec['execution_ordinal']:03d}.pt"
            if dest.exists():
                p=torch.load(dest,weights_only=True);assert p['authority']==bind(AUTH) and p['model']==bind(final) and p['calibration']==bind(calibration)
            else:
                if time.monotonic()-start>a['chunk_seconds']:receipt('predict_'+split);return
                if split=='train':
                    m=torch.load(OUT/'uncalibrated_train'/f"query{rec['execution_ordinal']:03d}.pt",weights_only=True)['M'];traces=[]
                else:
                    with torch.no_grad():m,traces=T.estimate(rec,bank,model,True)
                assert m.shape==(128,) and torch.isfinite(m).all() and (m>=0).all() and (m<=1).all()
                if traces:assert max(abs(np.sqrt(t['u'].numpy().mean()*t['v'].numpy().mean())-float(m[i])) for i,t in enumerate(traces))<2e-12
                save(dest,dict(authority=bind(AUTH),model=bind(final),query_id=rec['query_id'],execution_ordinal=rec['execution_ordinal'],
                    candidate_physical_rows=rec['candidate_physical_rows'],M=torch.from_numpy(K.apply(m.numpy(),cal['bias'])),
                    M_uncalibrated=m,calibration=bind(calibration),quality_traces=traces,quality_traces_scope='before scalar M calibration',
                    teacher_at_inference=False,held_identity_reads=0))
            bindings.append(dict(split=split,query_id=rec['query_id'],prediction=bind(dest)))
    write(OUT/'prediction_seal.json',dict(status='M_TEACHER_PREDICTIONS_SEALED',authority=bind(AUTH),model=bind(final),
        predictions=bindings,train=457,held=119,held_teacher_reads=0,held_identity_reads=0))
    write(OUT/'fit_validation.json',dict(status='M_TEACHER_FIT_AND_PREDICTIONS_PASS',authority=bind(AUTH),payload=bind(OUT/'prediction_seal.json')))
    receipt('complete')


def advance():
    a=read(AUTH)
    for b in a['sources']:checked(b)
    if (OUT/'validation.json').exists():
        checked(read(OUT/'validation.json')['result']);write(OUT/'status.json',dict(status='COMPLETE'),mutable=True);return
    jid=os.environ['SLURM_JOB_ID'];c=read(OUT/'chunks'/f'{jid}.json')
    assert c['status']=='NORMAL_CHUNK' and c['authority']==bind(AUTH)
    count=len(list((OUT/'chunks').glob('*.json')))
    if count>=a['max_chunks']:
        write(OUT/'status.json',dict(status='STOPPED_CHUNK_BUDGET',chunks=count),mutable=True);return
    dest=OUT/'continuations'/f'{jid}.json'
    if dest.exists():return
    stage='join' if (OUT/'fit_validation.json').exists() else 'fit'
    p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+jid,str(LAUNCH),stage],capture_output=True,text=True,check=True,timeout=45)
    job=p.stdout.strip().split(';')[0];assert job.isdigit();write(dest,dict(job_id=job,stage=stage,previous=jid))
    print(dict(event='CONTINUATION_SUBMITTED',job_id=job,stage=stage),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','fit','join','advance'));stage=parser.parse_args().stage
    torch.set_num_threads(8)
    if stage=='prepare':prepare()
    elif stage=='advance':advance()
    else:
        a=read(AUTH);checked(a['small_panel_result']);checked(a['small_panel_validation'])
        a=T.guard(stage)
        if stage=='fit':fit(a)
        else:
            cal=read(OUT/'mass_calibration.json');seal=read(OUT/'prediction_seal.json')
            for b in seal['predictions']:
                p=torch.load(checked(b['prediction']),weights_only=True)
                assert p['calibration']==bind(OUT/'mass_calibration.json')
                assert np.array_equal(p['M'].numpy(),K.apply(p['M_uncalibrated'].numpy(),cal['bias']))
            write(OUT/'calibration_validation.json',dict(status='TRAIN_ONLY_CALIBRATION_REPLAY_PASS',
                calibration=bind(OUT/'mass_calibration.json'),predictions=len(seal['predictions']),held_parameters_fitted=0))
            T.join(a)
