"""Independent direct materialization and actual adapter-hook verification."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys,json,hashlib
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'programs'))
from rc_prellm_m_scaled_adapter_v4 import ScaledQualityResidualAdapter
r=json.loads((OUT/'result.json').read_text())
m=json.loads((ROOT/'results/rc_prellm_m_adapter_v2/input_manifest.json').read_text())
a=json.loads((ROOT/'registry/rc_internal_m_condition_scale_v4_authority_20260924.json').read_text())
cp=torch.load(ROOT/'results/rc_internal_m_condition_scale_v4/PRE_REAL/snapshots/0128.pt',map_location='cpu',weights_only=True)
small=ScaledQualityResidualAdapter(hidden_size=3584,bottleneck=16,condition_gain=a['condition_gain'])
small.load_state_dict(cp['adapter']);small.eval()
query=m['probe_rows'][0]
saved=next(x for x in r['rows'] if x['query_id']==query['query_id'])
cache=torch.load(ROOT/'results/rc_prellm_m_adapter_v2/encoder_cache'/query['query_id']/'payload.pt',map_location='cpu',weights_only=True)
x=cache['merged']
order=sorted(range(128),key=lambda k:query['M'][k])
captured=[]
hook=small.up.register_forward_hook(lambda mod,args,out:captured.append(out.detach().clone()))
with torch.no_grad():
    small.conditioning='constant'
    base=small(x,query['M'][0])
    constant=captured.pop()*small.residual_scale
    small.conditioning='real'
    checks=[]
    for label,k in [('min',order[0]),('median',order[64]),('max',order[-1])]:
        actual=small(x,query['M'][k])
        real=captured.pop()*small.residual_scale
        delta=(real-constant).double()
        mean=delta.mean(0)
        centered=delta-mean
        e=float(delta.square().sum(1).mean())
        common=float(mean.square().sum())
        spatial=float(centered.square().sum(1).mean())
        inferred=saved['candidates'][k]
        rel=abs(e-inferred['M_specific_energy'])/max(e,1e-16)
        share_error=abs(common/e-inferred['M_specific_common_energy_share'])
        rounded=(actual.float()-base.float()).double()
        re=float(rounded.square().sum(1).mean())
        rc=float(rounded.mean(0).square().sum())
        checks.append({'mass_order':label,'candidate_position':k,'M':query['M'][k],
            'direct_energy':e,'gram_energy':inferred['M_specific_energy'],'energy_relative_error':rel,
            'orthogonal_energy_closure_abs_error':abs(e-common-spatial),
            'direct_common_share':common/e,'gram_common_share':inferred['M_specific_common_energy_share'],
            'common_share_abs_error':share_error,
            'actual_BF16_input_delta_common_share':rc/re})
        assert rel<1e-4 and share_error<1e-5
        assert abs(e-common-spatial)<max(1e-8,e*1e-10)
hook.remove()
report={'status':'DIRECT_ADAPTER_HOOK_VS_GRAM_PASS','query_id':query['query_id'],'patches':len(x),
    'checks':checks,'max_energy_relative_error':max(x['energy_relative_error'] for x in checks),
    'max_common_share_abs_error':max(x['common_share_abs_error'] for x in checks),
    'note':'Actual original adapter called on BF16 cached merger tokens on CPU. Hook captures up output before BF16 cast; standardized_zero condition is original constant path. No LLM forward.'}
(OUT/'direct_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
