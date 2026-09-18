#!/usr/bin/env python3
"""After prediction sealing, compare fixed-panel candidate axes to historic inputs."""
import os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
import run_rc_new_hyp593_oof5_v1 as P
import cache_rc_new_hyp593_features_v1 as C
OUT=ROOT/'results/rc_fixed_panels_train269_group_risk_v1'
AUTH=ROOT/'registry/rc_fixed_panels_train269_group_risk_authority_v1_20260911.json'

def main():
 M.need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED');M.need(M.datetime.now(M.timezone.utc)<M.DEADLINE,'USER_CUTOFF')
 a=M.read(AUTH)
 for b in a['sources'].values():M.checked(b)
 v=M.read(OUT/'fits/validation.json');receipt=M.read(OUT/'fits/receipt.json');M.need(v['status']=='FIXED_PANEL_RETRAIN_AND_PREDICTION_REPLAY_PASS' and v['payload']==receipt['payload'] and v['receipt']==M.bind(OUT/'fits/receipt.json'),'PREDICTION_SEAL_BEFORE_METADATA')
 torch.load(M.checked(receipt['payload']),map_location='cpu',weights_only=True)
 rows,_=P.features();byid={r['query_id']:r for r in rows};roles={r['query_id']:r for r in M.read(OUT/'eval_curator_roles.json')['records']};manifest=M.read(OUT/'panel_manifest.json')
 old=M.read(M.checked(a['evaluation_sources']['original32_result']));oldrows={}
 for seal in old['bindings']['full_shards']:
  folder=ROOT/'results/routea_matched_three_arm_fullnegative_features_v1'/f"shard{seal['shard']:02d}"
  for fn,key in [('payload.pt','payload_sha256'),('receipt.json','receipt_sha256'),('validation.json','validation_sha256')]:M.need(M.sha(folder/fn)==seal[key],'HISTORIC_SHARD_BINDING')
  data=torch.load(folder/'payload.pt',map_location='cpu',weights_only=False,mmap=True)
  for r in data['records']:
   if r['data_split_role']=='EVAL':oldrows[r['query_id']]=r
 checks=[]
 for q in manifest['panels']['EVAL32']:
  r=byid[q];o=oldrows[roles[q]['original_query_id']];axis=list(map(int,o['candidate_physical_rows']));order=[axis[i] for i in sorted(range(128),key=lambda i:(-float(o['base_scores'][i]),axis[i]))]
  x=o['real_native_features']['C_PAIRED'];xr=r['modes']['REAL']['X'];same_shape=x.shape==xr.shape;delta=float(torch.max(torch.abs(x-xr))) if same_shape else None
  fields=dict(candidate_axis=axis==r['candidate_physical_rows'],raw_C128_order=order==r['raw_ranked_physical_rows'][:128],winner_position=int(o['base_winner_position'])==r['winner'],challenger_axis=list(o['challenger_positions'])==r['challenger_positions'],REAL_feature_numeric_parity=same_shape and torch.allclose(x,xr,atol=1e-12,rtol=1e-12))
  checks.append(dict(query_id=q,panel='EVAL32',checks=fields,feature_bit_equal=torch.equal(x,xr),feature_max_abs_difference=delta))
 workers={w['query_id']:w for w in M.read(P.OUT/'metadata/worker_manifest.json')['records']}
 for q in manifest['panels']['EVAL128']:
  w=workers[q];M.need('reuse' in w,'ORIGINAL128_SOURCE_REUSE');s=w['reuse'];raw=C.loadsource(s['token_raw']);o=raw['records'][s['record_index']];r=byid[q]
  fields=dict(source_query_id=o['query_id']==s['source_query_id'],source_image_sha=o['query_source_sha256']==roles[q]['source_image_sha256'],candidate_axis=o['candidate_physical_rows']==r['candidate_physical_rows'],raw_full_gallery_order=o['raw_ranked_physical_rows']==r['raw_ranked_physical_rows'],raw_C128_order=o['candidate_ranked_physical_rows']==r['raw_ranked_physical_rows'][:128])
  checks.append(dict(query_id=q,panel='EVAL128',checks=fields))
 passed=len(checks)==160 and all(all(r['checks'].values()) for r in checks)
 M.write(OUT/'historic_input_parity.json',dict(status='FIXED269_HISTORIC_INPUT_PARITY_PASS' if passed else 'FIXED269_HISTORIC_INPUT_PARITY_MISMATCH',all_pass=passed,program=M.bind(__file__),authority=M.bind(AUTH),rows=checks))
 M.need(passed,'HISTORIC_INPUT_PARITY');print('FIXED269_HISTORIC_INPUT_PARITY_PASS',flush=True)
if __name__=='__main__':torch.set_num_threads(8);torch.set_num_interop_threads(1);main()
