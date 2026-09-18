#!/usr/bin/env python3
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'programs'))
import run_romav2_colnomic_sealed_source_e0_v1 as v1
import run_romav2_colnomic_sealed_source_e0_v2 as v2
AUTH=ROOT/'results/romav2_colnomic_new_difficult_sealed_roma_prejoin_v1/validation.json';MANIFEST=ROOT/'results/romav2_colnomic_new_difficult_sealed_manifest_v1/prejoin_manifest.json';OUTROOT=ROOT/'results/romav2_colnomic_new_difficult_sealed_fullrank_prejoin_v1';SHARDS=8
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--shard',type=int,required=True);z=a.parse_args();assert z.shard in range(SHARDS);out=OUTROOT/f'shard{z.shard:02d}/result.json'
 if out.exists():raise RuntimeError('immutable sealed fullrank shard exists')
 auth=json.load(open(AUTH));assert auth['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_ROMA_PREJOIN_VALIDATION_PASS' and auth['sealed_fullrank_prejoin_authorized'] is True and auth['sealed_result_reduction_authorized'] is False
 manifest=json.load(open(MANIFEST));work=[r for r in manifest['rows'] if int(r['query_ordinal'])%SHARDS==z.shard];assert len(work) in {3,4}
 from colpali_engine.models import ColQwen2_5,ColQwen2_5_Processor
 gallery=v1.build_gallery_source(verify_cache_file_sha256=True);payload=torch.load(v1.GALLERY_CACHE,map_location='cpu',weights_only=False,mmap=True);passages=payload['passage_emb'];labels=gallery.corrected_identities;device=torch.device('cuda');encoder=ColQwen2_5.from_pretrained(str(v1.MODEL),torch_dtype=torch.bfloat16).to(device).eval();encoder.requires_grad_(False);processor=ColQwen2_5_Processor.from_pretrained(str(v1.MODEL));rows=[]
 for row in work:
  path=Path(row['query_path']);assert sha(path)==row['query_sha256']
  with Image.open(path) as im:inputs=processor.process_images([ImageOps.exif_transpose(im).convert('RGB')]).to(device)
  encoded=encoder(**inputs)[0].float();mask=inputs['input_ids'][0]==processor.image_token_id;image=encoded[mask].detach().half().cpu().contiguous();template=encoded[~mask].detach().half().cpu().contiguous();scores=v2.full_gallery_scores(torch.cat((image,template)),passages,device);ranked=v2.top128_rows(scores,labels);seen=set();full=[]
  for physical in torch.argsort(scores,descending=True,stable=True).tolist():
   label=labels[physical]
   if label in seen:continue
   seen.add(label);full.append(int(physical))
  assert len(full)==len(set(labels)) and full[:128]==ranked;rows.append({'query_ordinal':int(row['query_ordinal']),'query_id':row['query_id'],'query_sha256':row['query_sha256'],'exact_label_count':len(full),'ranked_physical_rows':full,'ranked_score_values':[float(scores[x]) for x in full],'target_or_label_read_count':0});print(json.dumps({'event':'sealed_fullrank_ready','shard':z.shard,'ordinal':row['query_ordinal'],'exact_label_count':len(full)},sort_keys=True),flush=True)
 value={'schema_version':'rc_romav2_colnomic_new_difficult_sealed_fullrank_prejoin_shard_v1_20260901','status':'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_FULLRANK_PREJOIN_SHARD_READY','shard':z.shard,'rows':rows,'query_count':len(rows),'exact_label_count':len(set(labels)),'target_or_label_read_count':0,'target_join_manifest_read_count':0,'new_sealed_pixel_decode_count':len(rows),'new_sealed_model_scoring_count':len(rows),'logical_sha256':''};value['logical_sha256']=logical(value);out.parent.mkdir(parents=True,exist_ok=False);out.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':value['status'],'shard':z.shard,'query_count':len(rows)},sort_keys=True))
if __name__=='__main__':main()
