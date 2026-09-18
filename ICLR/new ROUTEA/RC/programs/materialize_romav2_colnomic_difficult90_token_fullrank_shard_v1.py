#!/usr/bin/env python3
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'programs'))
import run_romav2_colnomic_sealed_source_e0_v1 as v1
import run_romav2_colnomic_sealed_source_e0_v2 as v2
from rc_aslo_xf.cw1_sr0_s8_feature_runtime_v1 import HybridSpatialReferenceResolver
MAN=ROOT/'results/romav2_colnomic_difficult90_regression_manifest_v1/prejoin_manifest.json';RECEIPT=ROOT/'results/romav2_colnomic_difficult90_regression_manifest_v1/receipt.json';AUTH=ROOT/'registry/romav2_colnomic_new_difficult_sealed_reduction_authority_v1_20260901.json';LINEAGE=ROOT/'results/romav2_colnomic_new_difficult_model_lineage_v1/result.json';OUTROOT=ROOT/'results/romav2_colnomic_difficult90_token_fullrank_prejoin_v1';SHARDS=10
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def tsha(x):
 x=x.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(json.dumps(list(x.shape),separators=(',',':')).encode());h.update(x.view(torch.uint8).numpy().tobytes());return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--shard',type=int,required=True);z=a.parse_args();assert z.shard in range(SHARDS);out=OUTROOT/f'shard{z.shard:02d}/payload.pt';rp=OUTROOT/f'shard{z.shard:02d}/receipt.json'
 if out.exists() or rp.exists():raise RuntimeError('immutable difficult90 shard exists')
 assert json.load(open(RECEIPT))['status']=='ROMAV2_COLNOMIC_DIFFICULT90_MANIFEST_BINDING_PASS';assert json.load(open(AUTH))['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_REDUCTION_AUTHORIZED';assert json.load(open(LINEAGE))['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_MODEL_LINEAGE_FROZEN';manifest=json.load(open(MAN));work=[r for r in manifest['rows'] if int(r['query_ordinal'])%SHARDS==z.shard];assert len(work)==9
 from colpali_engine.models import ColQwen2_5,ColQwen2_5_Processor
 gallery=v1.build_gallery_source(verify_cache_file_sha256=True);gp=torch.load(v1.GALLERY_CACHE,map_location='cpu',weights_only=False,mmap=True);passages=gp['passage_emb'];labels=gallery.corrected_identities;device=torch.device('cuda');encoder=ColQwen2_5.from_pretrained(str(v1.MODEL),torch_dtype=torch.bfloat16).to(device).eval();encoder.requires_grad_(False);processor=ColQwen2_5_Processor.from_pretrained(str(v1.MODEL));resolver=HybridSpatialReferenceResolver(gallery_source=gallery,processor_factory=lambda:processor,gallery_embedding_loader=lambda:passages);records=[];refs={}
 for row in work:
  path=Path(row['query_path']);assert sha(path)==row['query_sha256']
  with Image.open(path) as im:orientation=int(im.getexif().get(274,1));inputs=processor.process_images([im.convert('RGB')]).to(device)
  encoded=encoder(**inputs)[0].float();mask=inputs['input_ids'][0]==processor.image_token_id;image=encoded[mask].detach().half().cpu().contiguous();template=encoded[~mask].detach().half().cpu().contiguous();_,h,w=[int(x) for x in inputs['image_grid_thw'][0].tolist()];merge=int(processor.image_processor.merge_size);grid=(h//merge,w//merge);scores=v2.full_gallery_scores(torch.cat((image,template)),passages,device);ranked128=v2.top128_rows(scores,labels);axis=sorted(ranked128);seen=set();full=[]
  for physical in torch.argsort(scores,descending=True,stable=True).tolist():
   if labels[physical] in seen:continue
   seen.add(labels[physical]);full.append(int(physical))
  assert len(full)==5412 and full[:128]==ranked128;records.append({'query_ordinal':int(row['query_ordinal']),'query_id':row['query_id'],'opened_split':row['opened_split'],'query_path':str(path),'query_sha256':row['query_sha256'],'source_exif_orientation':orientation,'query_grid_shape':grid,'query_tokens':image,'query_tokens_sha256':tsha(image),'candidate_physical_rows':axis,'candidate_ranked_physical_rows':ranked128,'candidate_raw_scores':torch.tensor([float(scores[r]) for r in axis],dtype=torch.float64),'fullrank_physical_rows':full,'fullrank_score_values':torch.tensor([float(scores[r]) for r in full],dtype=torch.float64),'target_or_label_read_count':0});print(json.dumps({'event':'difficult90_query_ready','shard':z.shard,'ordinal':row['query_ordinal']},sort_keys=True),flush=True)
 for r in sorted({x for q in records for x in q['candidate_physical_rows']}):
  x=resolver.resolve(r);refs[r]={'physical_row':r,'source_path':str(gallery.raw_paths[r]),'grid_shape':x.grid_shape,'tokens':x.tokens,'tokens_sha256':x.tokens_sha256,'source_kind':x.source_kind,'source_logical_sha256':x.source_logical_sha256}
 payload={'schema_version':'rc_romav2_colnomic_difficult90_token_fullrank_prejoin_shard_v1_20260901','status':'ROMAV2_COLNOMIC_DIFFICULT90_TOKEN_FULLRANK_PREJOIN_SHARD_READY','shard':z.shard,'shard_count':SHARDS,'records':records,'references':refs,'target_or_label_read_count':0,'model_update_count':0};out.parent.mkdir(parents=True,exist_ok=False);torch.save(payload,out);receipt={'schema_version':'rc_romav2_colnomic_difficult90_token_fullrank_receipt_v1_20260901','status':'ROMAV2_COLNOMIC_DIFFICULT90_TOKEN_FULLRANK_PREJOIN_SHARD_READY','shard':z.shard,'query_count':len(records),'reference_count':len(refs),'payload_sha256':sha(out),'manifest_sha256':sha(MAN),'target_label_read_count':0,'next_authorized_stage':'DIFFICULT90_REAL_CBIND_PREJOIN'};rp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt,sort_keys=True))
if __name__=='__main__':main()
