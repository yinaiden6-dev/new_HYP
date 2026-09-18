#!/usr/bin/env python3
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'programs'))
import run_romav2_colnomic_sealed_source_e0_v1 as v1
import run_romav2_colnomic_sealed_source_e0_v2 as v2
from rc_aslo_xf.cw1_sr0_s8_feature_runtime_v1 import HybridSpatialReferenceResolver
AUTH=ROOT/'results/romav2_colnomic_new_difficult_sealed_token_e0_v1/independent_validation.json';MANIFEST=ROOT/'results/romav2_colnomic_new_difficult_sealed_manifest_v1/prejoin_manifest.json';E0=ROOT/'results/romav2_colnomic_new_difficult_sealed_token_e0_v1/payload.pt';OUTROOT=ROOT/'results/romav2_colnomic_new_difficult_sealed_token_prejoin_v1';SHARDS=8
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def tsha(x):
 x=x.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(json.dumps(list(x.shape),separators=(',',':')).encode());h.update(x.view(torch.uint8).numpy().tobytes());return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--shard',type=int,required=True);z=a.parse_args();assert z.shard in range(SHARDS);out=OUTROOT/f'shard{z.shard:02d}/payload.pt';receipt_path=OUTROOT/f'shard{z.shard:02d}/receipt.json'
 if out.exists() or receipt_path.exists():raise RuntimeError('immutable sealed token shard exists')
 auth=json.load(open(AUTH));assert auth['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_TOKEN_E0_INDEPENDENT_VALIDATION_PASS' and auth['sealed_full_tokenization_authorized'] is True and auth['sealed_roma_scoring_authorized'] is False
 manifest=json.load(open(MANIFEST));work=[r for r in manifest['rows'] if int(r['query_ordinal'])%SHARDS==z.shard];assert len(work) in {3,4};e0=torch.load(E0,map_location='cpu',weights_only=False,mmap=True)
 from colpali_engine.models import ColQwen2_5,ColQwen2_5_Processor
 gallery_source=v1.build_gallery_source(verify_cache_file_sha256=True);gallery_payload=torch.load(v1.GALLERY_CACHE,map_location='cpu',weights_only=False,mmap=True);passages=gallery_payload['passage_emb'];labels=gallery_source.corrected_identities;device=torch.device('cuda');encoder=ColQwen2_5.from_pretrained(str(v1.MODEL),torch_dtype=torch.bfloat16).to(device).eval();encoder.requires_grad_(False);processor=ColQwen2_5_Processor.from_pretrained(str(v1.MODEL));resolver=HybridSpatialReferenceResolver(gallery_source=gallery_source,processor_factory=lambda:processor,gallery_embedding_loader=lambda:passages);records=[];refs={};new_reads=0
 for row in work:
  ordinal=int(row['query_ordinal']);path=Path(row['query_path']);assert sha(path)==row['query_sha256']
  if ordinal==0:
   assert e0['query_id']==row['query_id'] and e0['query_sha256']==row['query_sha256'];image=e0['query_tokens'];grid=tuple(e0['query_grid_shape']);axis=list(map(int,e0['candidate_physical_rows']));ranked=list(map(int,e0['candidate_ranked_physical_rows']));raw=e0['candidate_raw_scores'];orientation=int(e0['source_exif_orientation']);refs.update({int(k):v for k,v in e0['references'].items()});source_kind='REUSED_INDEPENDENTLY_VALIDATED_E0'
  else:
   with Image.open(path) as im:orientation=int(im.getexif().get(274,1));inputs=processor.process_images([ImageOps.exif_transpose(im).convert('RGB')]).to(device)
   encoded=encoder(**inputs)[0].float();mask=inputs['input_ids'][0]==processor.image_token_id;image=encoded[mask].detach().half().cpu().contiguous();template=encoded[~mask].detach().half().cpu().contiguous();_,h,w=[int(x) for x in inputs['image_grid_thw'][0].tolist()];merge=int(processor.image_processor.merge_size);grid=(h//merge,w//merge);scores=v2.full_gallery_scores(torch.cat((image,template)),passages,device);ranked=v2.top128_rows(scores,labels);axis=sorted(ranked);raw=torch.tensor([float(scores[r]) for r in axis],dtype=torch.float64);new_reads+=1;source_kind='FRESH_SEALED_TARGET_FREE_TOKENIZATION'
  records.append({'query_ordinal':ordinal,'query_id':row['query_id'],'query_path':str(path),'query_sha256':row['query_sha256'],'source_exif_orientation':orientation,'query_grid_shape':grid,'query_tokens':image,'query_tokens_sha256':tsha(image),'candidate_physical_rows':axis,'candidate_ranked_physical_rows':ranked,'candidate_raw_scores':raw,'source_kind':source_kind,'target_or_label_read_count':0});print(json.dumps({'event':'sealed_token_query_ready','shard':z.shard,'ordinal':ordinal,'source_kind':source_kind},sort_keys=True),flush=True)
 for row in sorted({r for q in records for r in q['candidate_physical_rows']}):
  if row not in refs:
   x=resolver.resolve(row);refs[row]={'physical_row':row,'source_path':str(gallery_source.raw_paths[row]),'grid_shape':x.grid_shape,'tokens':x.tokens,'tokens_sha256':x.tokens_sha256,'source_kind':x.source_kind,'source_logical_sha256':x.source_logical_sha256}
 payload={'schema_version':'rc_romav2_colnomic_new_difficult_sealed_token_prejoin_shard_v1_20260901','status':'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_TOKEN_PREJOIN_SHARD_READY','shard':z.shard,'shard_count':SHARDS,'records':records,'references':refs,'target_or_label_read_count':0,'target_join_manifest_read_count':0,'new_sealed_pixel_decode_count':new_reads,'new_sealed_model_scoring_count':new_reads};out.parent.mkdir(parents=True,exist_ok=False);torch.save(payload,out);receipt={'schema_version':'rc_romav2_colnomic_new_difficult_sealed_token_prejoin_receipt_v1_20260901','status':'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_TOKEN_PREJOIN_SHARD_READY','shard':z.shard,'query_count':len(records),'reference_count':len(refs),'payload_sha256':sha(out),'e0_validation_sha256':sha(AUTH),'prejoin_manifest_sha256':sha(MANIFEST),'target_join_manifest_read_count':0,'target_label_read_count':0,'new_sealed_pixel_decode_count':new_reads,'new_sealed_model_scoring_count':new_reads,'sealed_roma_scoring_count':0,'next_authorized_stage':'SEALED_FULL_TOKEN_PREJOIN_VALIDATION'};receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt,sort_keys=True))
if __name__=='__main__':main()
