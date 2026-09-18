#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
import torch
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'programs'))
import run_romav2_colnomic_sealed_source_e0_v1 as v1
import run_romav2_colnomic_sealed_source_e0_v2 as v2
from rc_aslo_xf.cw1_sr0_s8_feature_runtime_v1 import HybridSpatialReferenceResolver
AUTH=ROOT/'results/romav2_colnomic_current_runtime_frozen_gate_v1/independent_validation.json';MANIFEST=ROOT/'results/romav2_colnomic_new_difficult_sealed_manifest_v1/prejoin_manifest.json';OUT=ROOT/'results/romav2_colnomic_new_difficult_sealed_token_e0_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def tsha(x):
 x=x.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(json.dumps(list(x.shape),separators=(',',':')).encode());h.update(x.view(torch.uint8).numpy().tobytes());return h.hexdigest()
def main():
 if OUT.exists():raise RuntimeError('immutable sealed token E0 exists')
 auth=json.load(open(AUTH));assert auth['status']=='ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_INDEPENDENT_VALIDATION_PASS' and auth['sealed_query_tokenization_e0_authorized'] is True and auth['sealed_full_scoring_authorized'] is False
 manifest=json.load(open(MANIFEST));assert manifest['status']=='NEW_DIFFICULT_SEALED_PREJOIN_MANIFEST_BOUND' and manifest['query_count']==31 and manifest['target_field_count']==0;row=manifest['rows'][0];path=Path(row['query_path']);assert sha(path)==row['query_sha256']
 from colpali_engine.models import ColQwen2_5,ColQwen2_5_Processor
 gallery_source=v1.build_gallery_source(verify_cache_file_sha256=True);gallery_payload=torch.load(v1.GALLERY_CACHE,map_location='cpu',weights_only=False,mmap=True);passages=gallery_payload['passage_emb'];labels=gallery_source.corrected_identities;device=torch.device('cuda');encoder=ColQwen2_5.from_pretrained(str(v1.MODEL),torch_dtype=torch.bfloat16).to(device).eval();encoder.requires_grad_(False);processor=ColQwen2_5_Processor.from_pretrained(str(v1.MODEL))
 with Image.open(path) as raw:orientation=int(raw.getexif().get(274,1));inputs=processor.process_images([ImageOps.exif_transpose(raw).convert('RGB')]).to(device)
 encoded=encoder(**inputs)[0].float();mask=inputs['input_ids'][0]==processor.image_token_id;image=encoded[mask].detach().half().cpu().contiguous();template=encoded[~mask].detach().half().cpu().contiguous();_,h,w=[int(x) for x in inputs['image_grid_thw'][0].tolist()];merge=int(processor.image_processor.merge_size);grid=(h//merge,w//merge);assert image.shape==(grid[0]*grid[1],128)
 scores=v2.full_gallery_scores(torch.cat((image,template)),passages,device);ranked=v2.top128_rows(scores,labels);axis=sorted(ranked);resolver=HybridSpatialReferenceResolver(gallery_source=gallery_source,processor_factory=lambda:processor,gallery_embedding_loader=lambda:passages);refs={}
 for i,r in enumerate(axis):
  x=resolver.resolve(r);refs[r]={'physical_row':r,'source_path':str(gallery_source.raw_paths[r]),'grid_shape':x.grid_shape,'tokens':x.tokens,'tokens_sha256':x.tokens_sha256,'source_kind':x.source_kind,'source_logical_sha256':x.source_logical_sha256}
  if (i+1)%32==0:print(json.dumps({'event':'sealed_e0_reference_ready','done':i+1,'total':128}),flush=True)
 payload={'schema_version':'rc_romav2_colnomic_new_difficult_sealed_token_e0_payload_v1_20260901','status':'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_TOKEN_E0_READY','query_ordinal':0,'query_id':row['query_id'],'query_path':str(path),'query_sha256':row['query_sha256'],'source_exif_orientation':orientation,'query_grid_shape':grid,'query_tokens':image,'query_tokens_sha256':tsha(image),'candidate_physical_rows':axis,'candidate_ranked_physical_rows':ranked,'candidate_raw_scores':torch.tensor([float(scores[r]) for r in axis],dtype=torch.float64),'references':refs,'target_or_label_read_count':0,'sealed_pixel_decode_count':1,'sealed_model_scoring_count':1}
 OUT.mkdir(parents=True,exist_ok=False);pp=OUT/'payload.pt';torch.save(payload,pp);receipt={'schema_version':'rc_romav2_colnomic_new_difficult_sealed_token_e0_receipt_v1_20260901','status':'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_TOKEN_E0_READY','claim_level':'ONE_SEALED_QUERY_TARGET_FREE_TOKENIZATION_AND_C128_ENGINEERING_ONLY','query_count':1,'candidate_count':128,'reference_count':128,'payload_sha256':sha(pp),'authority_sha256':sha(AUTH),'prejoin_manifest_sha256':sha(MANIFEST),'target_join_manifest_read_count':0,'target_label_read_count':0,'sealed_pixel_decode_count':1,'sealed_model_scoring_count':1,'sealed_roma_scoring_count':0,'sealed_full_endpoint_authorized':False,'next_authorized_stage':'NEW_DIFFICULT_SEALED_TOKEN_E0_INDEPENDENT_VALIDATION'};(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt,sort_keys=True))
if __name__=='__main__':main()
