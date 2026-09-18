#!/usr/bin/env python3
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'programs'))
import run_romav2_colnomic_visibility_xf_six_case_v1 as core
from rc_aslo_xf.colnomic_dino_canonical_geometry_v2 import build_colnomic_canonical_geometry_v2,DECODED_RAW_BEFORE_EXIF,EXIF_ORIENTED_BEFORE_RESIZE
from romav2 import RoMaV2

SRC=ROOT/'results/romav2_colnomic_current_runtime_bridge_prejoin_v1';VALID=SRC/'validation.json';OUTROOT=ROOT/'results/romav2_colnomic_current_runtime_bridge_roma_prejoin_v1';PROCESSOR=ROOT.parents[2]/'models/downloaded_models/colnomic-embed-multimodal-7b/preprocessor_config.json'
def fsha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def geometry(path,source_sha,source_key,grid,frame):
 with Image.open(path) as im:
  raw_hw=(int(im.height),int(im.width));orientation=int(im.getexif().get(274,1))
 return build_colnomic_canonical_geometry_v2(source_image_sha256=source_sha,source_key=source_key,processor_config_sha256=fsha(PROCESSOR),raw_size_hw=raw_hw,exif_orientation=orientation,merged_grid_shape=tuple(grid),processor_input_frame=frame)
def main():
 a=argparse.ArgumentParser();a.add_argument('--shard',type=int,required=True);x=a.parse_args();assert x.shard in range(8)
 validation=json.load(open(VALID));assert validation['status']=='ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_PREJOIN_VALIDATION_PASS'
 inp=SRC/f'shard{x.shard:02d}/payload.pt';payload=torch.load(inp,map_location='cpu',weights_only=False,mmap=True);assert payload['status']=='ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_PREJOIN_SHARD_READY' and payload['shard']==x.shard
 out=OUTROOT/f'shard{x.shard:02d}/result.json'
 if out.exists():raise RuntimeError('immutable RoMa bridge shard exists')
 torch.set_float32_matmul_precision('highest');torch.manual_seed(17);model=RoMaV2();rows=[];ref_sha={}
 for record in payload['records']:
  qpath=Path(record['query_source_path']);qimg=core.oriented(qpath);qframe=EXIF_ORIENTED_BEFORE_RESIZE if record['track']=='new_difficult_train' else DECODED_RAW_BEFORE_EXIF;qgeom=geometry(qpath,record['query_source_sha256'],f"bridge-query:{record['execution_ordinal']}",record['query_grid_shape'],qframe);raw={int(row):float(score) for row,score in zip(record['candidate_physical_rows'],record['candidate_raw_scores'].tolist(),strict=True)};candidates=[]
  for pos,row in enumerate(record['candidate_physical_rows']):
   ref=payload['references'][int(row)];rpath=Path(ref['source_path']);source_sha=ref_sha.setdefault(int(row),fsha(rpath));rgeom=geometry(rpath,source_sha,f'gallery-row:{row}',ref['grid_shape'],DECODED_RAW_BEFORE_EXIF);pred=model.match(qimg,core.oriented(rpath));wq=core.cell_means(pred['overlap_AB'][0,...,0].detach().cpu(),qgeom);wr=core.cell_means(pred['overlap_BA'][0,...,0].detach().cpu(),rgeom);real,mass,_=core.score(record['query_tokens'],ref['tokens'],wq,wr);qc,_,_=core.score(record['query_tokens'],ref['tokens'],wq.roll(max(1,wq.numel()//2)),wr);rc,_,_=core.score(record['query_tokens'],ref['tokens'],wq,wr.roll(max(1,wr.numel()//2)));candidates.append({'candidate_position':pos,'physical_row':int(row),'raw_score':raw[int(row)],'real_score':float(real),'query_control_score':float(qc),'reference_control_score':float(rc),'visibility_mass':float(mass),'query_map_sha256':hashlib.sha256(wq.contiguous().numpy().tobytes()).hexdigest(),'reference_map_sha256':hashlib.sha256(wr.contiguous().numpy().tobytes()).hexdigest()})
  rows.append({'role':record['role'],'execution_ordinal':record['execution_ordinal'],'query_id':record['query_id'],'track':record['track'],'candidate_count':128,'candidates':candidates,'target_role_read_count':0,'target_insertion_count':0});print(json.dumps({'event':'roma_query_ready','shard':x.shard,'role':record['role'],'execution':record['execution_ordinal'],'candidate_count':128},sort_keys=True),flush=True)
 value={'schema_version':'rc_romav2_colnomic_current_runtime_bridge_roma_prejoin_shard_v1_20260901','status':'ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_ROMA_PREJOIN_SHARD_READY','claim_level':'TARGET_FREE_CURRENT_RUNTIME_FULL_C128_VISIBILITY_XF','shard':x.shard,'rows':rows,'bindings':{'source_payload_sha256':fsha(inp),'source_validation_sha256':fsha(VALID),'processor_config_sha256':fsha(PROCESSOR)},'target_role_read_count':0,'target_insertion_count':0,'sealed_read_count':0,'model_update_count':0,'logical_sha256':''};value['logical_sha256']=core.logical(value);core.atomic(out,value);print(json.dumps({'status':value['status'],'shard':x.shard},sort_keys=True))
if __name__=='__main__':main()
