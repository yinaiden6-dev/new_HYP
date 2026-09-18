#!/usr/bin/env python3
import hashlib,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rc_aslo_xf.conditional_rep_sources import build_gallery_source
SRC=ROOT/'results/romav2_colnomic_difficult90_token_fullrank_prejoin_v1';MAN=ROOT/'results/romav2_colnomic_difficult90_regression_manifest_v1/prejoin_manifest.json';OUT=SRC/'validation.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def tsha(x):
 x=x.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(json.dumps(list(x.shape),separators=(',',':')).encode());h.update(x.view(torch.uint8).numpy().tobytes());return h.hexdigest()
def main():
 if OUT.exists():raise RuntimeError('immutable difficult90 validation exists')
 manifest=json.load(open(MAN));m={int(x['query_ordinal']):x for x in manifest['rows']};gallery=build_gallery_source(verify_cache_file_sha256=True);labels=gallery.corrected_identities;ordinals=[];shards=[]
 for s in range(10):
  d=SRC/f'shard{s:02d}';r=json.load(open(d/'receipt.json'));p=torch.load(d/'payload.pt',map_location='cpu',weights_only=False,mmap=True);checks={'receipt':r['status']=='ROMAV2_COLNOMIC_DIFFICULT90_TOKEN_FULLRANK_PREJOIN_SHARD_READY' and r['shard']==s and r['query_count']==9 and r['target_label_read_count']==0,'hash':sha(d/'payload.pt')==r['payload_sha256'],'payload':p['status']=='ROMAV2_COLNOMIC_DIFFICULT90_TOKEN_FULLRANK_PREJOIN_SHARD_READY' and p['shard']==s and p['shard_count']==10 and p['target_or_label_read_count']==p['model_update_count']==0};ok=True;needed=set()
  for q in p['records']:
   o=int(q['query_ordinal']);ordinals.append(o);mm=m[o];axis=list(map(int,q['candidate_physical_rows']));rank128=list(map(int,q['candidate_ranked_physical_rows']));full=list(map(int,q['fullrank_physical_rows']));t=q['query_tokens'];needed.update(axis);full_labels=[labels[x] for x in full];ok &= q['query_id']==mm['query_id'] and q['query_sha256']==mm['query_sha256'] and q['opened_split']==mm['opened_split'] and len(axis)==len(set(axis))==128 and axis==sorted(axis) and set(axis)==set(rank128) and len(full)==len(set(full_labels))==5412 and full[:128]==rank128 and t.dtype==torch.float16 and t.shape==(math.prod(q['query_grid_shape']),128) and tsha(t)==q['query_tokens_sha256'] and q['candidate_raw_scores'].dtype==q['fullrank_score_values'].dtype==torch.float64 and tuple(q['candidate_raw_scores'].shape)==(128,) and tuple(q['fullrank_score_values'].shape)==(5412,) and bool(torch.isfinite(q['candidate_raw_scores']).all()) and bool(torch.isfinite(q['fullrank_score_values']).all()) and q['target_or_label_read_count']==0
  checks['records']=ok;checks['references']=needed<=set(map(int,p['references'])) and all(x['tokens'].dtype==torch.float16 and x['tokens'].shape==(math.prod(x['grid_shape']),128) and tsha(x['tokens'])==x['tokens_sha256'] for x in p['references'].values());passed=all(checks.values());shards.append({'shard':s,'checks':checks,'pass':passed,'payload_sha256':r['payload_sha256'],'query_count':9})
 checks={'all_shards':all(x['pass'] for x in shards),'all_90_ordinals':sorted(ordinals)==list(range(90)),'unique_queries':len(set(ordinals))==90,'manifest_population':len(m)==90,'gallery_population':len(labels)==5413 and len(set(labels))==5412};passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_difficult90_token_fullrank_prejoin_validation_v1_20260901','status':'ROMAV2_COLNOMIC_DIFFICULT90_TOKEN_FULLRANK_PREJOIN_VALIDATION_PASS' if passed else 'ROMAV2_COLNOMIC_DIFFICULT90_TOKEN_FULLRANK_PREJOIN_VALIDATION_FAIL','checks':checks,'shards':shards,'query_count':len(ordinals),'target_label_read_count':0,'real_prejoin_authorized':passed,'next_authorized_stage':'DIFFICULT90_REAL_PREJOIN' if passed else None};OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'checks':checks},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
