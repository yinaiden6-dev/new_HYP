#!/usr/bin/env python3
"""Pin existing TRAIN folds and completed cache, with no EVAL access."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'programs'))
import run_rc_paired_ce_optimization_isolation_v1 as C
import validate_rc_query_content_routing_oof4_v1 as V

parent=C.read(C.N.AUTH)
codes={'parent_'+k:b for k,b in parent['code_sources'].items()}
codes.update({k:C.bind(p) for k,p in dict(program=C.__file__,collector=ROOT/'programs/collect_rc_paired_ce_optimization_isolation_v1.py',freezer=__file__,interval_core=C.K.__file__,statistics_helper=V.__file__,launcher=ROOT/'slurm/rc_paired_ce_optimization_isolation_v1.sbatch',plan=ROOT/'plan/RC_PAIRED_CE_OPTIMIZATION_ISOLATION_V1_20260912.md').items()})
cache_sources=[]
for shard in range(16):
    folder=C.N.OUT/f'cache{shard:02d}'
    v=C.read(folder/'validation.json');r=C.read(C.checked(v['receipt']))
    C.need(v['status']=='PAIRED_LOCAL_FRESH_NUMPY_CACHE_PASS' and v['payload']==r['payload'],'QUALIFIED_CACHE')
    cache_sources.append(dict(validation=C.bind(folder/'validation.json'),receipt=v['receipt'],payload=r['payload']))
paired={str(f):dict(predictions=C.bind(C.N.OUT/f'fold{f:02d}'/'fit_predictions.json'),validation=C.bind(C.N.OUT/f'fold{f:02d}'/'independent_validation.json')) for f in range(4)}
a=dict(status='PAIRED_CE_OPTIMIZATION_ISOLATION_AUTHORIZED',user_authorization='2026-09-12 jiu then 继续; isolate current local-statistic failure before new mechanisms',code_sources=codes,parent_authority=C.bind(C.N.AUTH),public_sources=parent['public_sources'],cache_sources=cache_sources,fold_sources=parent['fold_sources'],paired_fold_sources=paired,join_sources=dict(previous_result=C.bind(C.N.OUT/'result.json'),previous_validation=C.bind(C.N.OUT/'result_validation.json')),models=list(C.MODELS),box_bound=64,loss='unregularized FULL C128 CE data loss',optimizer=dict(name='L-BFGS-B',initialization='previous AdamW final head',maxiter=2000,maxls=50,ftol=1e-14,gtol=1e-10),probability_denominator=C.DEN,interval_dps=50,certificate_gap_threshold='1/1000000',query_count=128,folds=4,training_labels='same fold TRAIN query-reference only',automatic_extra_trials=False,automatic_EVAL=False,HYP_GO_claimed=False)
C.write(C.AUTH,a)
print(C.bind(C.AUTH))
