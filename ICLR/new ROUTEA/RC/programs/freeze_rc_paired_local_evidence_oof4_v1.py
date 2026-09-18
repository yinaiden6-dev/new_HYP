#!/usr/bin/env python3
"""Freeze a bounded follow-up without reading natural heldout labels."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'programs'))
import run_rc_paired_local_evidence_oof4_v1 as N
import validate_rc_query_content_routing_oof4_v1 as V

old=N.read(N.R.AUTH)
source=N.read(N.checked(old['public_sources']['input_validation']))
for b in old['public_sources'].values():N.checked(b)
for bundle in old['fold_sources'].values():
    for b in bundle.values():N.checked(b)
global_sources={str(f):dict(payload=N.bind(N.R.OUT/f'fold{f:02d}'/'fit_predictions.json'),validation=N.bind(N.R.OUT/f'fold{f:02d}'/'independent_validation.json')) for f in range(4)}
codes={k:N.bind(p) for k,p in dict(program=N.__file__,validator=N.VALIDATOR,freezer=__file__,launcher=ROOT/'slurm/rc_paired_local_evidence_oof4_v1.sbatch',plan=ROOT/'plan/RC_PAIRED_LOCAL_EVIDENCE_OOF4_V1_20260912.md',optimizer_helper=N.R.__file__,native_helper=N.P.__file__,feature_core=N.P.FC.__file__,statistics_helper=V.__file__).items()}
a=dict(status='PAIRED_LOCAL_TRAIN128_OOF4_BOUNDED_AUTHORIZED',user_authorization='2026-09-12 explicit continued research and resume: 继续啊哥; one TRAIN-only comparison',code_sources=codes,public_sources=old['public_sources'],fold_sources=old['fold_sources'],global_sources=global_sources,raw_shards=source['raw_shards'],roma_shards=source['roma_shards'],query_count=128,identity_count=32,group_count=32,folds=4,primary='JOINT3',models=list(N.MODELS),new_parameters={'BIAS1':1,'MEAN2':2,'CURVE3':3,'JOINT3':3},steps=2000,optimizer=dict(name='AdamW',lr=.03,weight_decay=.001,dtype='float64',initialization='zero residual'),loss='FULL C128 CE including RAW zero and all127',candidate_source='unchanged RAW C128',acceptance='observed paired net with losses allowed; JOINT3 compared with every frozen control; group uncertainty separately',automatic_deployment=False,automatic_extra_trials=False,no_EVAL_access=True,no_other593_access=True,no_visual_type_training_labels=True,original_head_training_updates=0)
a['supersedes_prelaunch_authority']=N.bind(ROOT/'registry/rc_paired_local_evidence_oof4_authority_v1_20260912.json')
a['prelaunch_engineering_amendment']='Permit only atomic preflight.json temporary output under preflight stage. Initial synthetic assertions passed but receipt write was blocked. No natural jobs or fitting occurred under V1; scientific recipe unchanged.'
N.write(N.AUTH,a)
print(N.bind(N.AUTH))
