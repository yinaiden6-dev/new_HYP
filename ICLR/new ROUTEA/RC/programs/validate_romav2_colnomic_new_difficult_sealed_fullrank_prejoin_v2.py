#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rc_aslo_xf.conditional_rep_sources import build_gallery_source
SRC=ROOT/'results/romav2_colnomic_new_difficult_sealed_fullrank_prejoin_v1';TOK=ROOT/'results/romav2_colnomic_new_difficult_sealed_token_prejoin_v1';V1=SRC/'validation.json';OUT=SRC/'validation_v2.json'
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 if OUT.exists():raise RuntimeError('immutable fullrank v2 validation exists')
 assert json.load(open(V1))['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_FULLRANK_PREJOIN_VALIDATION_PASS';token={}
 for s in range(8):
  p=torch.load(TOK/f'shard{s:02d}/payload.pt',map_location='cpu',weights_only=False,mmap=True)
  for r in p['records']:token[int(r['query_ordinal'])]=(r['query_id'],r['query_sha256'])
 gallery=build_gallery_source(verify_cache_file_sha256=True);labels=gallery.corrected_identities;checks={'physical_rows':len(labels)==5413,'corrected_exact_labels':len(set(labels))==5412,'token_population':len(token)==31};rows=[];all_ok=True
 for s in range(8):
  x=json.load(open(SRC/f'shard{s:02d}/result.json'))
  for r in x['rows']:
   o=int(r['query_ordinal']);rank=list(map(int,r['ranked_physical_rows']));rank_labels=[labels[i] for i in rank];ok=(r['query_id'],r['query_sha256'])==token[o] and len(rank)==len(rank_labels)==5412 and len(set(rank_labels))==5412;all_ok &= ok;rows.append({'query_ordinal':o,'query_binding_exact':ok,'exact_label_rank_sequence_sha256':canonical(rank_labels),'exact_label_score_sequence_sha256':canonical(r['ranked_score_values'])})
 checks['all_query_id_sha_bindings']=all_ok;checks['all_31_ordinals']=sorted(x['query_ordinal'] for x in rows)==list(range(31));passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_new_difficult_sealed_fullrank_prejoin_validation_v2_20260901','status':'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_FULLRANK_PREJOIN_VALIDATION_V2_PASS' if passed else 'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_FULLRANK_PREJOIN_VALIDATION_V2_FAIL','checks':checks,'query_rows':rows,'gallery_physical_row_count':5413,'corrected_exact_label_count':5412,'corrected_mapping_sha256':gallery.corrected_mapping_sha256,'target_join_manifest_read_count':0,'target_label_read_count':0,'next_authorized_stage':'NEW_DIFFICULT_SEALED_CBIND_PREJOIN' if passed else None};OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'checks':checks},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
