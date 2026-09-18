#!/usr/bin/env python3
"""Fit exactly the screened GLOBAL7 residual on TRAIN128, then seal fixed panels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
sys.dont_write_bytecode=True
import run_rc_query_content_routing_oof4_v1 as R
import run_rc_new_hyp593_oof5_v1 as E
P=R.P
need,read,sha,bind,checked,write,write_bytes=R.NEED,R.read,R.sha,R.bind,R.checked,R.write,R.write_bytes
OUT=ROOT/'results/rc_global7_train128_fixed_panels_v1'
AUTH=ROOT/'registry/rc_global7_train128_fixed_panels_authority_v1_20260912.json'
HEAD=ROOT/'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json'
PREV=ROOT/'results/rc_full_candidate_identity_loss_v1'
FIXED=ROOT/'results/rc_fixed_panels_train269_group_risk_v1'
MODEL='GLOBAL7_T128'
BASELINES=('ORIGINAL7','LISTWISE_UNIT1')
LABELS_OPEN=False


def prepare():
    roles=read(P.META/'curator_roles.json')
    need(roles['training_labels_only'] is True and roles['role']=='TRAIN','TRAIN_ONLY_CURATOR')
    workers=read(P.META/'worker_manifest.json')['records']
    modern=read(FIXED/'train_roles.json')
    need(modern['evaluation_labels_included'] is False,'MODERN_TRAIN_ONLY')
    lookup={r['original_query_id']:r for r in modern['records']}
    ordered=sorted(roles['records'],key=lambda r:r['execution_ordinal'])
    records=[]
    for r,w in zip(ordered,workers,strict=True):
        m=lookup[r['original_query_id']]
        need(r['query_id']==w['query_id'] and r['execution_ordinal']==w['execution_ordinal'],'TRAIN_QUERY_ORDER')
        need(r['source_image_sha256']==m['source_image_sha256']==w['source_image_sha256'],'TRAIN_IMAGE_BRIDGE')
        need(r['identity']==m['identity'] and r['group']==m['group'],'TRAIN_IDENTITY_GROUP_BRIDGE')
        records.append({**{k:r[k] for k in ('query_id','execution_ordinal','original_query_id','source_image_sha256','identity','group')},'canonical_query_id':m['query_id'],'component':m['component']})
    need(len(records)==len({r['source_image_sha256'] for r in records})==128,'EXACT128')
    need(len({r['identity'] for r in records})==len({r['group'] for r in records})==32,'EXACT32_IDENTITIES_GROUPS')
    write(OUT/'train_roles.json',dict(status='GLOBAL7_TRAIN128_LABELS_ONLY',records=records,evaluation_labels_included=False,source_train_curator=bind(P.META/'curator_roles.json'),source_modern_train=bind(FIXED/'train_roles.json'),source_worker=bind(P.META/'worker_manifest.json')))
    print('GLOBAL7_TRAIN128_METADATA_BRIDGE_PASS',flush=True)


def guard(stage):
    a=read(AUTH)
    need(a['program']==bind(__file__) and a['screened_helper']==bind(R.__file__),'FROZEN_NEW_AND_PARENT_PROGRAMS')
    checked(a['native_helper']);checked(a['feature_core']);checked(a['eval_feature_loader'])
    if stage!='preflight':need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        need(not any(v in s for v in ('d1-mi','d1_mi','/grozi/','gisc_prerecall_universe','/target_join/','/role_shards/')),'PROTECTED_SOURCE')
        if not LABELS_OPEN:
            need('curator_roles' not in s and 'rc_opened_' not in s and '/reports/' not in s,'PREJOIN_LABEL_BOUNDARY')
        if stage in ('fit','replay','preflight') and ROOT/'results' in p.parents:
            public={Path(b['path']).resolve() for b in a['train_sources'].values()}
            need(OUT in p.parents or p in public,'TRAIN_ONLY_RESULT_SOURCE:'+s)
        if E.OUT/'fits' in p.parents or E.OUT/'roles' in p.parents:
            need(False,'NO593_PARAMETERS_OR_LABELS')
    sys.addaudithook(audit)
    if stage!='preflight':
        pf=read(OUT/'preflight.json')
        need(pf['authority']==bind(AUTH) and pf['status']=='GLOBAL7_TRAIN128_PREFLIGHT_PASS','FROZEN_PREFLIGHT')
    return a


def head():
    h=read(checked(read(AUTH)['original_head']))
    need(h['parameter_sha256']=='ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263','EXACT_EC7')
    return torch.tensor([float.fromhex(v) for v in h['weight_binary64']]+[float.fromhex(h['bias_binary64'])],dtype=torch.float64)


def compute():
    a=read(AUTH)
    for b in a['train_sources'].values():checked(b)
    rows=P.qualified_rows();roles=read(OUT/'train_roles.json')['records']
    labels,mapping=P.gallery_labels();targets=[]
    for r,role in zip(rows,roles,strict=True):
        need(r['query_id']==role['query_id'] and r['execution_ordinal']==role['execution_ordinal'] and r['source_image_sha256']==role['source_image_sha256'],'FROZEN_TRAIN128_LABEL_JOIN')
        pos=[i for i,p in enumerate(r['candidate_physical_rows']) if labels[p]==role['identity']]
        need(len(pos)==1,'FULL128_TARGET_PRESENT')
        targets.append(0 if pos[0]==r['base_winner_position'] else r['challenger_positions'].index(pos[0])+1)
    base_theta=head();x=torch.stack([r['X'] for r in rows])
    base=torch.stack([r['X']@base_theta[:6]+base_theta[6] for r in rows])
    design=R.design(x,torch.ones((128,127,1),dtype=torch.float64));y=torch.tensor(targets)
    delta,last_loss=R.optimize(design,base,y)
    after=base+design@delta
    return dict(status='GLOBAL7_TRAIN128_FIT_COMPLETE',model=MODEL,authority=bind(AUTH),base_theta_binary64=[float(v).hex() for v in base_theta],delta_binary64=[float(v).hex() for v in delta],train_query_ids=[r['query_id'] for r in rows],train_count=128,train_roles=bind(OUT/'train_roles.json'),native_features=a['train_sources']['native_features'],gallery_mapping_sha256=mapping,last_preupdate_loss=last_loss,postupdate_loss=float(R.objective(after,y)),original_head_training_updates=0,new_residual_training_updates=2000,heldout_label_reads=0,optimizer=dict(name='AdamW',lr=.03,weight_decay=.001,steps=2000,initialization='zero residual',dtype='float64'),function_source=bind(R.__file__),production_arithmetic='X@base_w+base_b plus cat(X,ones)@delta; no parameter-collapse rewrite')


def fit(replay=False,nonce=None):
    value=compute()
    if replay:
        need(nonce==os.environ.get('GLOBAL7_T128_NONCE'),'FRESH_PROCESS_NONCE')
        need(value==read(OUT/'fit.json'),'FRESH_GLOBAL7_PARAMETER_BITS')
        write(OUT/'fit_validation.json',dict(status='GLOBAL7_TRAIN128_FRESH_PARAMETER_REPLAY_PASS',fit=bind(OUT/'fit.json'),authority=bind(AUTH),fresh_nonce=nonce,original_head_training_updates=0,heldout_label_reads=0))
    else:
        write(OUT/'fit.json',value);nonce=uuid.uuid4().hex
        subprocess.run([sys.executable,__file__,'replay','--nonce',nonce],check=True,env=dict(os.environ,GLOBAL7_T128_NONCE=nonce))
    print('GLOBAL7_TRAIN128_FRESH_PARAMETER_REPLAY_PASS',flush=True)


def previous_predictions():
    a=read(AUTH);seal=read(checked(a['prediction_sources']['previous_prediction_seal']))
    need(seal['status']=='LISTWISE_UNIT1_ALL160_PREDICTIONS_SEALED' and seal['payload']==a['prediction_sources']['previous_payload'],'PREVIOUS_PRELABEL_PREDICTION_SEAL')
    value=torch.load(checked(seal['payload']),map_location='cpu',weights_only=True)
    return value,seal


def predict():
    a=read(AUTH);v=read(OUT/'fit_validation.json')
    need(v['status']=='GLOBAL7_TRAIN128_FRESH_PARAMETER_REPLAY_PASS' and v['fit']==bind(OUT/'fit.json') and v['authority']==bind(AUTH),'QUALIFIED_FIT')
    fit=read(OUT/'fit.json');theta=head();delta=torch.tensor([float.fromhex(x) for x in fit['delta_binary64']],dtype=torch.float64)
    old,oldseal=previous_predictions();prior={r['query_id']:r for r in old['predictions']}
    need(torch.equal(theta,old['parameters']['ORIGINAL7']['theta']),'OLD_EC7_PARAMETER_BITS')
    rows,sources=E.features();need(sources==old['feature_sources'],'EXACT_FROZEN_FEATURE_SOURCES')
    rows=[r for r in rows if r['query_id'] in prior];need(len(rows)==160,'EXACT_OLD160')
    predictions=[]
    for r in rows:
        p=prior[r['query_id']]
        keys=('query_id','execution_ordinal','candidate_physical_rows','raw_ranked_physical_rows','winner','challenger_positions')
        need(all(r[k]==p[k] for k in keys),'ALL_FIXED_AXES')
        record={k:r[k] for k in keys};record['models']={m:p['models'][m] for m in BASELINES};record['models'][MODEL]={}
        for mode in ('REAL','CBIND'):
            x=r['modes'][mode]['X'];base=x@theta[:6]+theta[6]
            need(torch.equal(base,p['models']['ORIGINAL7'][mode]['logits']),'BASE_LOGIT_BITS')
            z=base+R.design(x,torch.ones((127,1),dtype=torch.float64))@delta
            j=int(torch.argmax(z));pos=r['challenger_positions'][j] if float(z[j])>0 else r['winner']
            record['models'][MODEL][mode]=dict(logits=z,selected_position=pos)
        predictions.append(record)
    params={m:old['parameters'][m] for m in BASELINES};params[MODEL]=dict(base_theta=theta,delta=delta)
    payload=dict(parameters=params,predictions=predictions,feature_sources=sources,fit=bind(OUT/'fit.json'),fit_validation=bind(OUT/'fit_validation.json'),authority=bind(AUTH),previous_payload=oldseal['payload'],heldout_label_reads=0)
    import io
    stream=io.BytesIO();torch.save(payload,stream);write_bytes(OUT/'predictions.pt',stream.getvalue())
    write(OUT/'prediction_seal.json',dict(status='GLOBAL7_TRAIN128_ALL160_REAL_CBIND_PRELABEL_SEALED',payload=bind(OUT/'predictions.pt'),fit=bind(OUT/'fit.json'),fit_validation=bind(OUT/'fit_validation.json'),authority=bind(AUTH),heldout_label_reads=0))
    print('GLOBAL7_TRAIN128_ALL160_REAL_CBIND_PRELABEL_SEALED',flush=True)


def preflight():
    torch.manual_seed(17);x=torch.randn(4,3,6,dtype=torch.float64);base=torch.randn(4,3,dtype=torch.float64);y=torch.tensor([0,1,2,3])
    design=R.design(x,torch.ones((4,3,1),dtype=torch.float64))
    delta=torch.nn.Parameter(torch.zeros(7,dtype=torch.float64));opt=torch.optim.AdamW([delta],lr=.03,weight_decay=.001)
    allz=torch.cat((base.new_zeros((4,1)),base+torch.cat((x,x.new_ones((4,3,1))),dim=-1)@delta),dim=-1)
    direct=F.cross_entropy(allz,y);direct.backward();grad=delta.grad.clone();opt.step()
    observed,loss=R.optimize(design,base,y,steps=1)
    need(abs(loss-float(direct.detach()))<1e-12 and torch.allclose(observed,delta.detach(),atol=1e-12,rtol=0),'INDEPENDENT_CE_AND_ADAMW_STEP')
    test=torch.zeros(7,dtype=torch.float64,requires_grad=True);lg=R.objective(base+design@test,y);g=torch.autograd.grad(lg,test)[0]
    need(torch.allclose(g,grad,atol=1e-12,rtol=0),'INDEPENDENT_GRADIENT')
    need(torch.equal(base+design@torch.zeros(7,dtype=torch.float64),base),'ZERO_RESIDUAL_BASE_PARITY')
    need(R.STEPS==2000 and len(head())==7,'SCREENED_BUDGET_AND_BASE')
    write(OUT/'preflight.json',dict(status='GLOBAL7_TRAIN128_PREFLIGHT_PASS',authority=bind(AUTH),natural_training_updates=0,checks=['same_screened_GLOBAL7_design_loss_optimizer','independent_CE_gradient_AdamW','zero_residual_base_parity','frozen_ec7_registry']))
    print('GLOBAL7_TRAIN128_PREFLIGHT_PASS',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','preflight','fit','replay','predict']);ap.add_argument('--nonce');args=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        guard(args.stage)
        if args.stage=='preflight':preflight()
        elif args.stage=='predict':predict()
        else:fit(args.stage=='replay',args.nonce)
