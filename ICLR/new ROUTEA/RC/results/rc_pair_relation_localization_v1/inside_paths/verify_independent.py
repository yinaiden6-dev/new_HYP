#!/usr/bin/env python3
"""Independent rank implementation and artifact closure check; CPU only."""
import hashlib
import json
from pathlib import Path
import scipy
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    validation = json.loads((ROOT/'validation.json').read_text())
    for source in validation['artifacts']+[validation['script']]:
        assert bind(source['path']) == source
    rows = json.loads((ROOT/'rows41.json').read_text())
    maximum_error = 0.
    for row in rows:
        cc = row['component_diagnostics']
        concentration = [x['P_concentration_bidirectional_mean'] for x in cc['candidates']]
        for field in ('M_native_HR1','logM_P_effect_J1','logM_P_effect_J0','logM_JP_interaction'):
            expected = float(spearmanr(concentration,[x[field]for x in cc['candidates']]).statistic)
            maximum_error = max(maximum_error,abs(expected-cc['metrics']['rho_P_concentration_bidirectional_mean_vs_'+field]))
    assert maximum_error < 1e-12
    eligible = [x for x in rows if x['target_in_C128']]
    counts = {}
    for ct,mt,label in ((True,True,'both_C_and_M'),(True,False,'only_C'),
                        (False,True,'only_M'),(False,False,'neither')):
        counts[label] = sum((x['component_diagnostics']['metrics']['target_C_rank_desc']==1)==ct
            and (x['component_diagnostics']['metrics']['target_M_rank_desc']==1)==mt for x in eligible)
    result = dict(status='INDEPENDENT_SCIPY_RANK_AND_ARTIFACT_HASH_PASS',scipy_version=scipy.__version__,
        script=bind(__file__),validation=bind(ROOT/'validation.json'),
        maximum_spearman_error=maximum_error,queries=41,correlations=164,
        target_present_queries=38,C_M_candidate_top1_cross_counts=counts,
        scope='Independent rank calculation and output hash check; does not rerun upstream models or establish semantic correspondence.')
    (ROOT/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
