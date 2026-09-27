#!/usr/bin/env python3
"""Independent endpoint and finite-difference audit; no model forward."""
import argparse
import json
from pathlib import Path
import numpy as np

RC=Path(__file__).resolve().parents[1]
DEFAULT=RC/'results/rc_post_m_level_gap_v1'
CELLS=('LL','LG','GL','GG')
MATRIX=np.asarray([[-1,1,0,0],[0,0,-1,1],[-1,0,1,0],[0,-1,0,1],[1,-1,-1,1],[-1,0,0,1],[-.5,.5,-.5,.5],[-.5,-.5,.5,.5]])
KEYS=('delta_at_mu_L','delta_at_mu_G','mu_at_delta_L','mu_at_delta_G','interaction','total','symmetric_delta','symmetric_mu')


def read(p):return json.loads(Path(p).read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    protocol=read(a.root/'protocol.json');workers=read(protocol['workers']['path'])['workers'];errors=[];n=0;phase_ends=0;npz_count=0;incomplete=0
    for w in workers:
        path=a.root/'queries'/f"{w['index']:03d}"/'result.json'
        if not path.exists():
            if a.allow_partial:continue
            raise AssertionError(('MISSING_QUERY',w['index']))
        row=read(path);n+=1;phase=read(w['phase_row']['path']) if w['phase_row'] else None
        assert row['query_id']==w['query_id'] and row['fold']==w['fold']
        for out,design in zip(row['comparisons'],w['comparisons']):
            for cell,world in [('LL',design['low']),('GG',design['high'])]:
                if phase and out['panel']=='phase9':
                    arm=world.removeprefix('PHASE_')
                    for key,value in phase['arms'][arm].items():errors.append(abs(out['cells'][cell]['gaps'][key]-value))
                    for role in ('target','fixed_wrong'):
                        for key,value in phase['individual_endpoints'][arm][role].items():errors.append(abs(out['cells'][cell]['sides'][role][key]-value))
                    phase_ends+=1
            if not out['complete_legal_grid']:
                incomplete+=1;assert not out['effects'];assert any(not v['valid'] for v in out['cells'].values());continue
            for metric,values in out['effects'].items():
                v=np.asarray([out['cells'][c]['gaps'][metric] for c in CELLS]);want=MATRIX@v
                errors.extend(np.abs(want-np.asarray([values[k] for k in KEYS])).tolist())
                errors.append(abs(values['symmetric_mu']+values['symmetric_delta']-values['total']))
                errors.append(abs(values['delta_at_mu_L']+values['mu_at_delta_G']-values['total']))
        for item in row['patch_outputs']:
            npz_count+=1
            with np.load(item['patch']['path']) as z:
                assert np.max(np.abs(z['maxsim'].mean(1)-z['L']))<1e-14
                assert np.isfinite(z['M']).all() and ((z['M']>0)&(z['M']<=1)).all()
                assert [float(m).hex() for m in z['M']]==list(z['mass_hex'])
    maximum=max(errors,default=0.);assert maximum<2e-10
    if not a.allow_partial:assert n==120 and phase_ends==288 and incomplete==21 and npz_count==240
    result=dict(status='POST_M_LEVEL_GAP_SECOND_ARITHMETIC_PASS' if n==120 else 'POST_M_LEVEL_GAP_PARTIAL_SECOND_ARITHMETIC_PASS',
        queries=n,phase_action_and_side_endpoint_replays=phase_ends,explicit_illegal_grids=incomplete,query_pair_npz=npz_count,
        endpoint_and_matrix_max_error=maximum,new_forwards=0,full128_old_action_parity_claim=False)
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if not a.allow_partial:
        out=a.root/'second_arithmetic_validation.json'
        if out.exists():assert out.read_text()==text
        else:out.write_text(text)
    print(text,end='')

if __name__=='__main__':main()
