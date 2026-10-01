#!/usr/bin/env python3
"""Post-hoc descriptive contrasts from sealed worlds; no new experiment."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'results/rc_m_structure_binding_isolation_v2'


def main():
    source=ROOT/'result.json';d=json.loads(source.read_text());summary={};descriptive={}
    for cohort in d['summaries']:
        rows=[x for x in d['rows'] if x['cohort']==cohort and x['opponent']=='C_STRONGEST']
        summary[cohort]={};descriptive[cohort]={}
        def grouped(fn):
            groups={}
            for r in rows:groups.setdefault(r['component'],[]).append(fn(r))
            return np.array([np.mean(x) for x in groups.values()])
        for ep in ['logM','POST_content']:
            summary[cohort][ep]={}
            for name,left,right in [('QP_minus_SP','Q_P','S_P'),('QALL_minus_QC','Q_ALL','Q_C')]:
                vals=grouped(lambda r:sum(r['metrics'][ep]['world_gaps'][left+'_'+s]-r['metrics'][ep]['world_gaps'][right+'_'+s] for s in ['PLUS','MINUS'])/2)
                rng=np.random.default_rng(20261002)
                ci=np.quantile(vals[rng.integers(len(vals),size=(10000,len(vals)))].mean(1),[.025,.975])
                summary[cohort][ep][name]=dict(group_equal_mean=float(vals.mean()),exploratory_group95=ci.tolist(),queries=len(rows),groups=len(vals))
            descriptive[cohort][ep]={'NATIVE_gap':float(grouped(lambda r:r['metrics'][ep]['world_gaps']['NATIVE']).mean())}
            for arm in ['Q_P','Q_C','Q_ALL','S_P','Q_AFTER_S','S_AFTER_Q']:
                value=float(grouped(lambda r:sum(r['metrics'][ep]['world_gaps'][arm+'_'+s] for s in ['PLUS','MINUS'])/2-r['metrics'][ep]['world_gaps']['NATIVE']).mean())
                descriptive[cohort][ep][arm+'_minus_NATIVE']=value
        descriptive[cohort]['absolute_logM_changes']={}
        for arm in ['Q_ALL','S_P']:
            for side in ['target','wrong']:
                descriptive[cohort]['absolute_logM_changes'][arm+'_'+side]=float(grouped(lambda r:
                    sum(r['metrics']['logM'][side][arm+'_'+s] for s in ['PLUS','MINUS'])/2-r['metrics']['logM'][side]['NATIVE']).mean())
    out=dict(status='POSTHOC_INTERPRETATION_FROM_SEALED_WORLDS',not_new_prespecified_primary=True,opponent='C_STRONGEST',
        summaries=summary,descriptive=descriptive,source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        program=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (ROOT/'interpretation_contrasts_20261002.json').write_text(json.dumps(out,indent=2)+'\n')
    print('POSTHOC_CONTRASTS_SAVED')


if __name__=='__main__':main()
