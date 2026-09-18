#!/usr/bin/env python3
import csv,hashlib,json,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));ROUTEA=ROOT.parent;CSV=ROUTEA/'data/video_pair_candidates_v1/human_confirmation_v1/human_confirmation.csv';BUNDLE=ROOT/'isolated/romav2_colnomic_candidate_bound_no_regret_v1_20260901/bundle_manifest.json';OUT=ROOT/'results/romav2_colnomic_b7_video_manifest_v1';VIDEOS=('DSCF0062.MOV','DSCF0063.MOV','DSCF0064.MOV','DSCF0065.MOV','DSCF0113.MOV','DSCF0144.MOV','DSCF0180.MOV','DSCF0181.MOV','DSCF0183.MOV','DSCF0188.MOV','DSCF0195.MOV')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):v['logical_sha256']=logical(v);p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def main():
 if OUT.exists():raise RuntimeError('immutable B7 manifest exists')
 assert json.load(open(BUNDLE))['status']=='ROMAV2_COLNOMIC_CANDIDATE_BOUND_NO_REGRET_MODEL_FROZEN';rows={r['video_name']:r for r in csv.DictReader(open(CSV,newline='',encoding='utf-8-sig'))};assert all(v in rows for v in VIDEOS);from rc_aslo_xf.conditional_rep_sources import build_gallery_source;gallery=build_gallery_source(verify_cache_file_sha256=True);label_rows={label:i for i,label in enumerate(gallery.corrected_identities)};pre=[];join=[];ordinal=0
 for video in VIDEOS:
  r=rows[video];assert r['confidence']=='high' and r['preliminary_corrected_identity'] and r['corrected_gallery_path'];identity=r['preliminary_corrected_identity'];target_row=label_rows[identity];assert Path(r['corrected_gallery_path']).resolve()==gallery.raw_paths[target_row].resolve();frames=r['query_frame_paths'].split('|');assert len(frames)==3
  for frame_index,name in enumerate(frames):
   path=Path(name);image_sha=sha(path);qid='B7R-'+hashlib.sha256((video+'|'+str(frame_index)+'|'+image_sha).encode()).hexdigest()[:16]
   with Image.open(path) as im:raw=[int(im.width),int(im.height)];orientation=int(im.getexif().get(274,1))
   pre.append({'query_ordinal':ordinal,'query_id':qid,'video_name':video,'frame_index':frame_index,'query_path':str(path),'query_sha256':image_sha,'raw_size_wh':raw,'exif_orientation':orientation});join.append({'query_ordinal':ordinal,'query_id':qid,'video_name':video,'frame_index':frame_index,'target_exact_label':identity,'target_gallery_physical_row':target_row,'target_reference_sha256':sha(gallery.raw_paths[target_row]),'label_status':'PRELIMINARY_HIGH_CONFIDENCE_HUMAN_CONFIRMATION_PENDING'});ordinal+=1
 assert ordinal==33 and len({x['target_exact_label'] for x in join})==9;OUT.mkdir(parents=True,exist_ok=False);p={'schema_version':'rc_romav2_colnomic_b7_video_prejoin_manifest_v1_20260901','status':'ROMAV2_COLNOMIC_B7_VIDEO_PREJOIN_MANIFEST_BOUND','claim_level':'OPENED_EXPLORATORY_PROVISIONAL_LABELS','query_count':33,'video_count':11,'identity_count_hidden_from_scoring_process':9,'target_field_count':0,'rows':pre,'logical_sha256':''};j={'schema_version':'rc_romav2_colnomic_b7_video_target_join_v1_20260901','status':'ROMAV2_COLNOMIC_B7_VIDEO_TARGET_JOIN_BOUND_PROVISIONAL','query_count':33,'video_count':11,'identity_count':9,'human_confirmation_complete':False,'rows':join,'logical_sha256':''};write(OUT/'prejoin_manifest.json',p);write(OUT/'target_join_manifest.json',j);receipt={'schema_version':'rc_romav2_colnomic_b7_video_manifest_receipt_v1_20260901','status':'ROMAV2_COLNOMIC_B7_VIDEO_MANIFEST_BINDING_PASS','query_count':33,'video_count':11,'identity_count':9,'source_csv_sha256':sha(CSV),'bundle_sha256':sha(BUNDLE),'prejoin_sha256':sha(OUT/'prejoin_manifest.json'),'target_join_sha256':sha(OUT/'target_join_manifest.json'),'model_score_read_count':0,'next_authorized_stage':'B7_VIDEO_TARGET_FREE_TOKEN_FULLRANK'};write(OUT/'receipt.json',receipt);print(json.dumps({'status':receipt['status'],'queries':33,'videos':11,'identities':9},sort_keys=True))
if __name__=='__main__':main()
