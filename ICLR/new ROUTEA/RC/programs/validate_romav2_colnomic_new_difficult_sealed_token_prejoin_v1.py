#!/usr/bin/env python3
import hashlib,json,math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results/romav2_colnomic_new_difficult_sealed_token_prejoin_v1';MAN=ROOT/'results/romav2_colnomic_new_difficult_sealed_manifest_v1/prejoin_manifest.json';E0=ROOT/'results/romav2_colnomic_new_difficult_sealed_token_e0_v1/independent_validation.json';OUT=SRC/'validation.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def tsha(x):
 x=x.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(json.dumps(list(x.shape),separators=(',',':')).encode());h.update(x.view(torch.uint8).numpy().tobytes());return h.hexdigest()
def main():
 if OUT.exists():raise RuntimeError('immutable full token validation exists')
 manifest=json.load(open(MAN));rows_by={int(x['query_ordinal']):x for x in manifest['rows']};ordinals=[];kinds=[];new_reads=0;shards=[]
 for s in range(8):
  d=SRC/f'shard{s:02d}';r=json.load(open(d/'receipt.json'));p=torch.load(d/'payload.pt',map_location='cpu',weights_only=False,mmap=True);checks={'receipt':r['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_TOKEN_PREJOIN_SHARD_READY' and r['shard']==s and r['target_join_manifest_read_count']==r['target_label_read_count']==r['sealed_roma_scoring_count']==0,'hash':sha(d/'payload.pt')==r['payload_sha256'],'payload':p['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_TOKEN_PREJOIN_SHARD_READY' and p['shard']==s and p['target_or_label_read_count']==p['target_join_manifest_read_count']==0};ok=True;needed=set()
  for q in p['records']:
   o=int(q['query_ordinal']);ordinals.append(o);kinds.append(q['source_kind']);m=rows_by[o];axis=list(map(int,q['candidate_physical_rows']));needed.update(axis);t=q['query_tokens'];ok &= q['query_id']==m['query_id'] and q['query_sha256']==m['query_sha256'] and len(axis)==len(set(axis))==128 and axis==sorted(axis) and set(axis)==set(q['candidate_ranked_physical_rows']) and t.dtype==torch.float16 and t.shape==(math.prod(q['query_grid_shape']),128) and tsha(t)==q['query_tokens_sha256'] and q['candidate_raw_scores'].dtype==torch.float64 and tuple(q['candidate_raw_scores'].shape)==(128,) and bool(torch.isfinite(q['candidate_raw_scores']).all()) and q['target_or_label_read_count']==0
  checks['queries']=ok;checks['references']=needed<=set(map(int,p['references'])) and all(x['tokens'].dtype==torch.float16 and x['tokens'].shape==(math.prod(x['grid_shape']),128) and tsha(x['tokens'])==x['tokens_sha256'] for x in p['references'].values());new_reads+=r['new_sealed_model_scoring_count'];passed=all(checks.values());shards.append({'shard':s,'checks':checks,'pass':passed,'payload_sha256':r['payload_sha256'],'query_count':len(p['records'])})
 checks={'all_shards':all(x['pass'] for x in shards),'all_31_ordinals':sorted(ordinals)==list(range(31)),'unique_queries':len(set(ordinals))==31,'one_e0_reuse':kinds.count('REUSED_INDEPENDENTLY_VALIDATED_E0')==1,'thirty_fresh':kinds.count('FRESH_SEALED_TARGET_FREE_TOKENIZATION')==30,'fresh_scoring_count':new_reads==30,'manifest_hash':all(json.load(open(SRC/f'shard{s:02d}/receipt.json'))['prejoin_manifest_sha256']==sha(MAN) for s in range(8))};passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_new_difficult_sealed_token_prejoin_validation_v1_20260901','status':'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_TOKEN_PREJOIN_VALIDATION_PASS' if passed else 'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_TOKEN_PREJOIN_VALIDATION_FAIL','checks':checks,'shards':shards,'query_count':len(ordinals),'cumulative_sealed_pixel_decode_count':31,'cumulative_sealed_model_scoring_count':31,'target_join_manifest_read_count':0,'target_label_read_count':0,'sealed_roma_scoring_count':0,'sealed_roma_prejoin_authorized':passed,'sealed_target_join_authorized':False,'next_authorized_stage':'NEW_DIFFICULT_SEALED_ROMA_PREJOIN' if passed else None};OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'checks':checks},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
