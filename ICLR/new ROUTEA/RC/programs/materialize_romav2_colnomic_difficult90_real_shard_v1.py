#!/usr/bin/env python3
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'programs'))
import materialize_romav2_colnomic_current_runtime_bridge_roma_shard_v1 as bridge
import run_romav2_colnomic_visibility_xf_six_case_v1 as core
from rc_aslo_xf.colnomic_dino_canonical_geometry_v2 import DECODED_RAW_BEFORE_EXIF
from romav2 import RoMaV2
SRC=ROOT/'results/romav2_colnomic_difficult90_token_fullrank_prejoin_v1';AUTH=SRC/'validation.json';OUTROOT=ROOT/'results/romav2_colnomic_difficult90_real_prejoin_v1'
def main():
 a=argparse.ArgumentParser();a.add_argument('--shard',type=int,required=True);x=a.parse_args();assert x.shard in range(10);auth=json.load(open(AUTH));assert auth['status']=='ROMAV2_COLNOMIC_DIFFICULT90_TOKEN_FULLRANK_PREJOIN_VALIDATION_PASS' and auth['real_prejoin_authorized'] is True
 inp=SRC/f'shard{x.shard:02d}/payload.pt';payload=torch.load(inp,map_location='cpu',weights_only=False,mmap=True);out=OUTROOT/f'shard{x.shard:02d}/result.json'
 if out.exists():raise RuntimeError('immutable difficult90 REAL shard exists')
 torch.set_float32_matmul_precision('highest');torch.manual_seed(17);model=RoMaV2();rows=[];ref_sha={}
 for record in payload['records']:
  qpath=Path(record['query_path']);qimg=core.oriented(qpath);qgeom=bridge.geometry(qpath,record['query_sha256'],f"difficult90-query:{record['query_ordinal']}",record['query_grid_shape'],DECODED_RAW_BEFORE_EXIF);raw={int(row):float(v) for row,v in zip(record['candidate_physical_rows'],record['candidate_raw_scores'].tolist(),strict=True)};candidates=[]
  for pos,row in enumerate(record['candidate_physical_rows']):
   ref=payload['references'][int(row)];rpath=Path(ref['source_path']);source_sha=ref_sha.setdefault(int(row),bridge.fsha(rpath));rgeom=bridge.geometry(rpath,source_sha,f'gallery-row:{row}',ref['grid_shape'],DECODED_RAW_BEFORE_EXIF);pred=model.match(qimg,core.oriented(rpath));wq=core.cell_means(pred['overlap_AB'][0,...,0].detach().cpu(),qgeom);wr=core.cell_means(pred['overlap_BA'][0,...,0].detach().cpu(),rgeom);real,mass,_=core.score(record['query_tokens'],ref['tokens'],wq,wr);qc,_,_=core.score(record['query_tokens'],ref['tokens'],wq.roll(max(1,wq.numel()//2)),wr);rc,_,_=core.score(record['query_tokens'],ref['tokens'],wq,wr.roll(max(1,wr.numel()//2)));candidates.append({'candidate_position':pos,'physical_row':int(row),'raw_score':raw[int(row)],'real_score':float(real),'query_control_score':float(qc),'reference_control_score':float(rc),'visibility_mass':float(mass),'query_map_sha256':hashlib.sha256(wq.contiguous().numpy().tobytes()).hexdigest(),'reference_map_sha256':hashlib.sha256(wr.contiguous().numpy().tobytes()).hexdigest()})
  rows.append({'query_ordinal':record['query_ordinal'],'query_id':record['query_id'],'opened_split':record['opened_split'],'candidate_count':128,'candidates':candidates,'target_role_read_count':0,'target_insertion_count':0});print(json.dumps({'event':'difficult90_real_ready','shard':x.shard,'ordinal':record['query_ordinal']},sort_keys=True),flush=True)
 value={'schema_version':'rc_romav2_colnomic_difficult90_real_prejoin_shard_v1_20260901','status':'ROMAV2_COLNOMIC_DIFFICULT90_REAL_PREJOIN_SHARD_READY','claim_level':'OPENED_REGRESSION_TARGET_FREE_FULL_C128_VISIBILITY_XF','shard':x.shard,'rows':rows,'bindings':{'source_payload_sha256':bridge.fsha(inp),'source_validation_sha256':bridge.fsha(AUTH),'processor_config_sha256':bridge.fsha(bridge.PROCESSOR)},'target_role_read_count':0,'target_insertion_count':0,'model_update_count':0,'logical_sha256':''};value['logical_sha256']=core.logical(value);core.atomic(out,value);print(json.dumps({'status':value['status'],'shard':x.shard,'query_count':len(rows)},sort_keys=True))
if __name__=='__main__':main()
