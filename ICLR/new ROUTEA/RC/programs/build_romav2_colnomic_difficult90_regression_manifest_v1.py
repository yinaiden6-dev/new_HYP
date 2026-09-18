#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));ROUTEA=ROOT.parent;VAL=ROUTEA/'results/b4_learned_ownership/difficult_val_cache_v1/manifest.json';TEST=ROUTEA/'results/b4_learned_ownership/difficult_test_cache_v1/manifest.json';SPLIT=ROUTEA/'data/route_a_b4_difficult_identity_split_v1.json';AUTH=ROOT/'registry/romav2_colnomic_new_difficult_sealed_reduction_authority_v1_20260901.json';OUT=ROOT/'results/romav2_colnomic_difficult90_regression_manifest_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):v['logical_sha256']=logical(v);p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def main():
 if OUT.exists():raise RuntimeError('immutable difficult90 manifest exists')
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 authority=json.load(open(AUTH));assert authority['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_REDUCTION_AUTHORIZED';split=json.load(open(SPLIT));val=json.load(open(VAL));test=json.load(open(TEST));assert val['query_count']==38 and val['identity_count']==4 and test['query_count']==52 and test['identity_count']==5;gallery=build_gallery_source(verify_cache_file_sha256=True);label_rows={label:i for i,label in enumerate(gallery.corrected_identities)};source=[]
 for role,doc in (('val',val),('test',test)):
  for r in doc['rows']:source.append((role,int(r['index']),r))
 source.sort(key=lambda x:(x[0],x[1]));assert len(source)==90 and len({x[2]['identity'] for x in source})==9;pre=[];join=[]
 for ordinal,(role,index,r) in enumerate(source):
  path=Path(r['path']);image_sha=sha(path);qid='DREG-'+hashlib.sha256(('difficult90-v1|'+image_sha).encode()).hexdigest()[:16]
  with Image.open(path) as im:raw=[int(im.width),int(im.height)];orientation=int(im.getexif().get(274,1))
  identity=str(r['identity']);target_row=label_rows[identity];pre.append({'query_ordinal':ordinal,'query_id':qid,'opened_split':role,'query_path':str(path),'query_sha256':image_sha,'raw_size_wh':raw,'exif_orientation':orientation});join.append({'query_ordinal':ordinal,'query_id':qid,'opened_split':role,'target_exact_label':identity,'target_gallery_physical_row':target_row,'target_reference_sha256':sha(gallery.raw_paths[target_row])})
 OUT.mkdir(parents=True,exist_ok=False);p={'schema_version':'rc_romav2_colnomic_difficult90_prejoin_manifest_v1_20260901','status':'ROMAV2_COLNOMIC_DIFFICULT90_PREJOIN_MANIFEST_BOUND','query_count':90,'identity_count_hidden_from_scoring_process':9,'target_field_count':0,'rows':pre,'logical_sha256':''};j={'schema_version':'rc_romav2_colnomic_difficult90_target_join_v1_20260901','status':'ROMAV2_COLNOMIC_DIFFICULT90_TARGET_JOIN_BOUND_REDUCER_ONLY','query_count':90,'identity_count':9,'rows':join,'logical_sha256':''};write(OUT/'prejoin_manifest.json',p);write(OUT/'target_join_manifest.json',j);receipt={'schema_version':'rc_romav2_colnomic_difficult90_manifest_receipt_v1_20260901','status':'ROMAV2_COLNOMIC_DIFFICULT90_MANIFEST_BINDING_PASS','role':'OPENED_REGRESSION_ONLY','query_count':90,'identity_count':9,'val_count':38,'test_count':52,'sources':{'val_sha256':sha(VAL),'test_sha256':sha(TEST),'split_sha256':sha(SPLIT),'sealed_authority_sha256':sha(AUTH)},'prejoin_sha256':sha(OUT/'prejoin_manifest.json'),'target_join_sha256':sha(OUT/'target_join_manifest.json'),'model_score_read_count':0,'next_authorized_stage':'DIFFICULT90_TARGET_FREE_TOKEN_FULLRANK'};write(OUT/'receipt.json',receipt);print(json.dumps({'status':receipt['status'],'query_count':90,'identity_count':9},sort_keys=True))
if __name__=='__main__':main()
