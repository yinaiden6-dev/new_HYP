#!/usr/bin/env python3
"""Replay sealed POST small modules on cached hidden, checking actual M input.

No RoMa, visual encoder, LLM, training, or experiment mutation. Records a new
audit only. Attention/encoder forward is not needed for this CPU replay.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

RC=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RC/'programs'),str(RC/'src')]
import run_rc_m_causal128_frozen_bridge_v1 as B
from audit_rc_m_causal128_property_closure_v1 import binding, read, write_new


def main(root):
    torch.set_num_threads(4)
    bridge=root/'newgroup_property_bridge_v1'
    result=read(bridge/'result.json');protocol=read(bridge/'protocol.json')
    parent=read(root/'bridge/protocol.json');context=B.Context(parent)
    items={r['index']:r for r in read(root/'bridge/inputs.json')['rows']}
    checks=[];maximum=0.;conditions=0
    for row in result['rows']:
        item=items[row['index']];adapter,head,cache,refs=context.load(item)
        assert head==row['head']
        hidden=cache['hidden'][cache['image_mask']]
        normalized=F.layer_norm(hidden.float(),(hidden.shape[-1],))
        # Capture the actual concatenation delivered to the frozen down layer.
        captured=[]
        def inspect(module,args):
            xx=args[0]
            assert torch.equal(xx[:,:-1],normalized)
            assert torch.equal(xx[:,-1],xx[0,-1].expand(len(xx)))
            captured.append(float(xx[0,-1]))
        hook=adapter.down.register_forward_pre_hook(inspect)
        for pos,payload in zip(sorted([row['target_position'],row['fixed_wrong_position']]),row['patch_outputs']):
            with np.load(B.checked(payload)) as z:
                mm=z['M'].copy();values=z['maxsim'].copy();locations=z['argmax'].copy();worlds=z['worlds'].tolist();savedL=z['L'].copy()
            assert worlds==protocol['arms'];captured.clear()
            observed,actual_locations=context.candidate(adapter,cache,refs[pos],mm)
            err=float(np.max(abs(observed-values)));maximum=max(maximum,err)
            assert err<2e-10 and np.array_equal(actual_locations,locations)
            assert np.max(abs(savedL-observed.mean(1)))<2e-10
            assert len(captured)==len(mm)==18
            raw=torch.as_tensor(mm,dtype=torch.float32)
            expected=(raw.clamp_min(adapter.mass_epsilon).log()-adapter.mass_log_mean)/adapter.mass_log_std*adapter.condition_gain
            assert np.array_equal(np.asarray(captured,np.float32),expected.numpy())
            role='target' if pos==row['target_position'] else 'fixed_wrong'
            for i,arm in enumerate(worlds):
                assert mm[i]==row['individual_endpoints'][arm][role]['M']
                assert abs(savedL[i]-row['individual_endpoints'][arm][role]['POST_content'])<2e-10
            checks.append(dict(index=row['index'],query_id=row['query_id'],fold=row['fold'],role=role,position=pos,
                snapshot=row['snapshot'],patch_output=payload,worlds=worlds,M=mm.tolist(),
                actual_down_layer_condition=captured.copy(),patch_count=len(hidden),
                mass_log_mean=float(adapter.mass_log_mean),mass_log_std=float(adapter.mass_log_std),
                mass_epsilon=float(adapter.mass_epsilon),condition_gain=adapter.condition_gain,
                patch_replay_max_error=err,argmax_exact=True))
            conditions+=len(captured)
        hook.remove()
        print(json.dumps(dict(index=row['index'],candidate_arms_checked=conditions,maximum_patch_error=maximum)),flush=True)
    contrast='phase_restoration_native_amplitude'
    side={role:{metric:result['individual_summary'][role][metric][contrast]
        for metric in ['logM','POST_content','POST_action']} for role in ['target','fixed_wrong']}
    out=dict(status='POST_ABSOLUTE_M_ACTUAL_INPUT_AND_PATCH_REPLAY_PASS',
        source_code=binding(__file__),source_protocol=binding(bridge/'protocol.json'),
        source_validation=binding(bridge/'validation.json'),source_result=binding(bridge/'result.json'),
        queries=9,pairs=18,candidate_arms=conditions,maximum_patch_error=maximum,
        actual_input_condition_exact=True,hidden_content_constant_across_arms=True,
        argmax_replay_exact=True,checks=checks,native_amplitude_phase_separate_sides=side,
        definition='Per-candidate absolute FP32 log(M), standardized by frozen TRAIN buffers and fixed gain, is concatenated with per-patch layer-normalized query hidden. No pairwise relative gap enters the adapter.',
        inference='A larger target-minus-wrong logM gap does not imply a larger posterior content gap: absolute candidate levels and the candidate reference jointly determine normalized MaxSim. The two side changes, including the observed sign reversal, replay from the same frozen module.',
        limits='No universal monotonicity or unique nonlinear-source claim; this audit rules out mass/axis mismatch, missing conditioning, changing hidden inputs, and stale patch outputs in these 324 cached evaluations.',
        new_RoMa_visual_or_LLM_forwards=0,new_gpu_forwards=0,new_training=0,experiment_outputs_changed=False)
    write_new(root/'independent_property_review_v1/post_absolute_input_validation.json',out)
    print(out['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=RC/'results/rc_m_causal128_attribution_chain_v1')
    main(p.parse_args().root.resolve())
