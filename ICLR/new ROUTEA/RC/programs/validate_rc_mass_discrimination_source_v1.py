#!/usr/bin/env python3
"""Independent primary contrast, saved tensor arithmetic and dose verification."""
import hashlib
import json
import math
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_mass_discrimination_source_v1'
PHASE=ROOT/'results/rc_roma_position_factor_v1'


def main():
    torch.set_num_threads(1)
    r=json.loads((OUT/'result.json').read_text())
    for b in r['sources']:
        assert hashlib.sha256(Path(b['path']).read_bytes()).hexdigest()==b['sha256']
    effects=[];maxdose=0.;masserror=0.;tensorpairs=0; per=[]
    for q in r['phase_rows']:
        if not q['target_present']: continue
        t=q['target_position'];w=q['arm_metrics']['NATIVE']['strongest_position']; m=q['mass_by_arm']
        g=math.fsum(math.log(m[a][t]/m[a][w]) for a in m if a.startswith('GLOBAL'))/4
        l=math.fsum(math.log(m[a][t]/m[a][w]) for a in m if a.startswith('LOCAL'))/4
        effect=l-g;effects.append(effect)
        assert abs(effect-q['contrasts']['strongest_content_logM_gap']['LOCAL_minus_GLOBAL'])<2e-14
        for pos in [t,w]:
            p=torch.load(PHASE/f"query{q['index']:03d}"/f'pair{pos:03d}.pt',weights_only=True,map_location='cpu',mmap=True)
            tensorpairs+=1
            for arm,sides in p['branches'].items():
                means=[math.fsum(float(v) for v in s['weights'].flatten())/s['weights'].numel() for s in sides]
                mm=math.sqrt(means[0]*means[1]);masserror=max(masserror,abs(mm-m[arm][pos]))
            for side in (0,1):
                for axis in ('X','Y'):
                    dose={scope:math.fsum(p['branches'][f'{scope}_{axis}_{sg}'][side]['invariants']['perturbation_rms']**2 for sg in ('PLUS','MINUS')) for scope in ('GLOBAL','LOCAL')}
                    maxdose=max(maxdose,abs(dose['GLOBAL']-dose['LOCAL'])/max(dose.values()))
        per.append(dict(index=q['index'],logM_contrast_LOCAL_minus_GLOBAL=effect))
    mean=math.fsum(effects)/len(effects)
    assert abs(mean-r['phase_contrasts']['strongest_content_logM_gap']['LOCAL_minus_GLOBAL']['group_equal_mean'])<2e-14
    assert masserror<2e-14 and maxdose<2e-6
    output=dict(status='INDEPENDENT_M_SOURCE_PRIMARY_CONTRAST_PASS',queries=len(effects),
        negative_effect_queries=sum(x<0 for x in effects),mean_LOCAL_minus_GLOBAL=mean,
        independently_recomputed_critical_tensor_pairs=tensorpairs,max_mass_recompute_error=masserror,
        max_antithetic_dose_relative_difference=maxdose,per_query=per,
        source_result_sha256=hashlib.sha256((OUT/'result.json').read_bytes()).hexdigest(),
        program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_accuracy_or_head_scoring=True,no_new_model_forward=True)
    (OUT/'independent_validation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
