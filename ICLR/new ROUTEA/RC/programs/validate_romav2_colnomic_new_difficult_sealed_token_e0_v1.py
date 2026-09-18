#!/usr/bin/env python3
import hashlib,json,math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results/romav2_colnomic_new_difficult_sealed_token_e0_v1';MAN=ROOT/'results/romav2_colnomic_new_difficult_sealed_manifest_v1/prejoin_manifest.json';OUT=SRC/'independent_validation.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def tsha(x):
 x=x.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(json.dumps(list(x.shape),separators=(',',':')).encode());h.update(x.view(torch.uint8).numpy().tobytes());return h.hexdigest()
def main():
 if OUT.exists():raise RuntimeError('immutable E0 validation exists')
 r=json.load(open(SRC/'receipt.json'));m=json.load(open(MAN));p=torch.load(SRC/'payload.pt',map_location='cpu',weights_only=False,mmap=True);axis=list(map(int,p['candidate_physical_rows']));q=p['query_tokens'];refs=p['references'];checks={'receipt':r['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_TOKEN_E0_READY' and r['target_join_manifest_read_count']==r['target_label_read_count']==r['sealed_roma_scoring_count']==0 and r['sealed_model_scoring_count']==r['sealed_pixel_decode_count']==1,'payload_hash':sha(SRC/'payload.pt')==r['payload_sha256'],'query_binding':p['query_ordinal']==0 and p['query_id']==m['rows'][0]['query_id'] and p['query_sha256']==m['rows'][0]['query_sha256'],'query_tensor':q.dtype==torch.float16 and q.ndim==2 and q.shape[1]==128 and q.shape[0]==math.prod(p['query_grid_shape']) and tsha(q)==p['query_tokens_sha256'],'axis':len(axis)==len(set(axis))==128 and axis==sorted(axis) and len(p['candidate_ranked_physical_rows'])==128 and set(axis)==set(p['candidate_ranked_physical_rows']) and p['candidate_raw_scores'].dtype==torch.float64 and tuple(p['candidate_raw_scores'].shape)==(128,) and bool(torch.isfinite(p['candidate_raw_scores']).all()),'references':set(axis)==set(map(int,refs)) and all(x['tokens'].dtype==torch.float16 and x['tokens'].shape==(math.prod(x['grid_shape']),128) and tsha(x['tokens'])==x['tokens_sha256'] for x in refs.values()),'zero_target':p['target_or_label_read_count']==0};passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_new_difficult_sealed_token_e0_independent_validation_v1_20260901','status':'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_TOKEN_E0_INDEPENDENT_VALIDATION_PASS' if passed else 'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_TOKEN_E0_INDEPENDENT_VALIDATION_FAIL','checks':checks,'payload_sha256':sha(SRC/'payload.pt'),'target_join_manifest_read_count':0,'target_label_read_count':0,'sealed_pixel_decode_count':1,'sealed_model_scoring_count':1,'sealed_roma_scoring_count':0,'sealed_full_tokenization_authorized':passed,'sealed_roma_scoring_authorized':False,'next_authorized_stage':'NEW_DIFFICULT_SEALED_FULL_TOKENIZATION' if passed else None};OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'checks':checks},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
