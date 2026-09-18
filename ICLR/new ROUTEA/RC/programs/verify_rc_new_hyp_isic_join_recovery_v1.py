#!/usr/bin/env python3
"""Read-only recovery qualification of sealed inference after scheduler timeout."""
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'programs'), str(ROOT/'src')]
import run_rc_new_hyp_isic_inference_v1 as I
OUT = ROOT/'results/rc_new_hyp_isic_transfer_v1'
RECOVERY = OUT/'submission/join_recovery_v1'


def main():
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
        name = str(Path(os.fsdecode(args[0])).resolve()).lower()
        assert not any(x in name for x in ('curator_roles', '/target_join/', 'd1-mi', 'd1_mi', '/reports/')), 'NO_LABEL_READ'
    sys.addaudithook(audit)
    a = I.read(I.AUTH); ab = I.bind(I.AUTH)
    for b in a['sources'].values():
        if Path(b['path']).stat().st_size < 16*(1 << 20): I.checked(b)
    for sources in a['heads'].values():
        for b in sources.values(): I.checked(b)
    workers = I.read(I.checked(a['sources']['worker']))['records']
    seals, all_queries, all_C4, max_error = [], [], 0, 0.0
    for shard in range(68):
        count = 1 if shard == 67 else 8
        validations = {}
        for kind, status in [('raw','ISIC_RAW_CPU_PASS'), ('roma','ISIC_ROMA_CPU_PASS'),
                             ('predictions','ISIC_FROZEN_HEAD_NUMPY_ACTION_PASS')]:
            folder = OUT/kind/f'shard{shard:02d}'
            v = I.read(folder/'validation.json')
            assert v['status'] == status and v['authority'] == ab and v['target_reads'] == 0
            I.checked(v['payload']); validations[kind] = v
            if kind != 'predictions':
                receipt = I.read(folder/'receipt.json')
                assert receipt['authority'] == ab and receipt['payload'] == v['payload'] and receipt['query_count'] == count
        assert validations['roma']['C4_checks'] == count*128*4
        all_C4 += validations['roma']['C4_checks']
        max_error = max(max_error, validations['predictions']['max_abs_error'])
        p = I.read(validations['predictions']['payload']['path'])
        assert p['authority'] == ab and p['raw'] == validations['raw']['payload'] and p['roma'] == validations['roma']['payload']
        assert p['training_updates'] == p['target_reads'] == 0 and len(p['records']) == count
        for row, w in zip(p['records'], workers[shard*8:shard*8+count]):
            assert row['query_id'] == w['query_id'] and row['execution_ordinal'] == w['execution_ordinal']
            assert sorted(row['raw_ranked_physical_rows']) == list(range(390))
            assert len(row['models']) == 10
            for result in row['models'].values():
                z = [float.fromhex(h) for h in result['logits_hex']]
                assert len(z) == 127 and all(map(math.isfinite,z))
            all_queries.append(row['query_id'])
        seals.append(dict(shard=shard, queries=count, validations={k:I.bind(OUT/k/f'shard{shard:02d}'/'validation.json') for k in validations}))
        if shard % 16 == 0: print('hashed_complete_shards', shard+1, flush=True)
    assert all_queries == [w['query_id'] for w in workers] and len(all_queries) == 537
    # The only TIMEOUT member: repeat all C4 and head calculations on CPU from sealed inputs.
    shard = 23
    raw = torch.load(OUT/'raw/shard23/payload.pt', map_location='cpu', weights_only=True)
    roma = torch.load(OUT/'roma/shard23/payload.pt', map_location='cpu', weights_only=True)
    predictions = I.read(OUT/'predictions/shard23/payload.json')['records']
    module = I.M.loadmodule(I.M.ROMA_SOURCE, 'isic_recovery_cpu_replay')
    checks = actions = 0; error = 0.0
    heads = {name:torch.tensor([float.fromhex(x) for x in I.read(sources['head']['path'])['theta_hex']],dtype=torch.float64)
             for name,sources in a['heads'].items()}
    with torch.inference_mode():
        for q, r, prediction in zip(raw['records'], roma['records'], predictions, strict=True):
            assert q['query_id'] == r['query_id'] == prediction['query_id']
            axis=q['candidate_physical_rows']; winner=axis.index(q['candidate_ranked_physical_rows'][0])
            challengers=[i for i in range(128) if i!=winner]
            assert len(r['candidates']) == 128
            for j,c in enumerate(r['candidates']):
                ref=raw['references'][axis[j]]
                assert c['physical_row']==ref['physical_row'] and c['candidate_position']==j
                scores=module.replay_c4(q['query_tokens'],ref['tokens'],c['query_visibility'],c['reference_visibility'])
                for key,value in scores.items():
                    assert float(value).hex()==float(c['old_scores'][key]).hex();checks+=1
            scores=[c['old_scores'] for c in r['candidates']]
            for mode in ('REAL','CBIND'):
                evidence={i:scores[i if mode=='REAL' else (i+64)%128] for i in range(128)}
                x=torch.stack([I.FC.candidate_feature(q['candidate_raw_scores'].tolist(),evidence,c,winner) for c in challengers])
                for model,theta in heads.items():
                    features=x[:,:1] if model=='RAW2_CE' else x
                    z=features@theta[:-1]+theta[-1]
                    key=model if mode=='REAL' else model+'_CBIND'
                    saved=prediction['models'][key]
                    assert [float(t).hex() for t in z]==saved['logits_hex']
                    independent=np.sum(features.numpy()*theta[:-1].numpy(),axis=1)+float(theta[-1])
                    error=max(error,float(np.abs(independent-z.numpy()).max()))
                    k=int(np.argmax(independent));chosen=axis[challengers[k]] if independent[k]>0 else axis[winner]
                    assert chosen==saved['selected'];actions+=1
    assert checks==4096 and actions==80 and error<2e-10
    result=dict(status='ISIC_ALL537_SEALS_AND_TIMEOUT_SHARD_CPU_REPLAY_PASS',authority=ab,
        program=I.bind(__file__),query_count=537,shard_count=68,seals=seals,total_original_CPU_C4_checks=all_C4,
        maximum_original_numpy_error=max_error,timeout_job='5144694_23',scheduler_state_preserved='TIMEOUT',
        timeout_shard_repeated_C4_checks=checks,timeout_shard_repeated_head_actions=actions,
        repeated_numpy_max_error=error,training_updates=0,query_gpu_forwards=0,target_reads=0,
        disposition='All predictions complete and independently valid; run unchanged final join only.')
    I.write(RECOVERY/'prejoin_validation.json',result)
    print(result['status'],flush=True)


if __name__=='__main__':main()
