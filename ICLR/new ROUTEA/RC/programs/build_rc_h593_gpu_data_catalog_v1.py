#!/usr/bin/env python3
"""Label-free index of reusable GPU acquisitions and their CPU consumers."""
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'cache/rc_h593_gpu_data_catalog_v1'
SPEC={
 'inside':('rc_h593_m_inside_v1','M_VISUAL_ORIGIN_QUERY_PASS','rc_h593_m_inside_authority_v1_20260922.json'),
 'visual':('rc_h593_m_visual_origin_v1','M_VISUAL_ORIGIN_QUERY_PASS','rc_h593_m_visual_origin_authority_v1_20260922.json'),
 'coordinate':('rc_h593_roma_coordinate_precision_v2','ROMA_COORDINATE_QUERY_PASS','rc_h593_roma_coordinate_precision_authority_v2_20260922.json')}
def read(p):return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(b):assert bind(b['path'])==b,b['path'];return Path(b['path'])
def build():
    rows=[];counts=dict.fromkeys(SPEC,0)
    for i in range(593):
        row=dict(index=i,sources={},missing=[])
        for family,(folder,status,authority) in SPEC.items():
            p=ROOT/'results'/folder/f'query{i:03d}'/'validation.json'
            if not p.exists():row['missing'].append(family);continue
            v=read(p);assert v['status']==status and v['authority']==bind(ROOT/'registry'/authority)
            checked(v['payload'])
            source=dict(validation=bind(p),payload=v['payload'])
            if family=='inside':
                ip=p.with_name('inside_validation.json')
                if not ip.exists():row['missing'].append(family);continue
                iv=read(ip);assert iv['status']=='M_INSIDE_PATHS_PASS' and iv['authority']==v['authority']
                checked(iv['payload']);source['paths_validation']=bind(ip);source['paths_payload']=iv['payload']
            row['sources'][family]=source;counts[family]+=1
        rows.append(row)
    fp=ROOT/'results/rc_h593_feature_fusion_cache_v1/ready.json';features=read(fp)
    assert features['status']=='FUSION_ALL593_FEATURE_CACHE_PASS' and features['queries']==593
    archive=dict(status='GPU_DATA_CATALOG_COMPLETE' if all(v==593 for v in counts.values()) else 'GPU_DATA_CATALOG_PARTIAL',
      time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),counts=counts,queries=593,candidates=128,
      scope='Existing registered GPU acquisition outputs; tensor parts retain original consumer hash verification',
      feature_bank=dict(manifest=bind(fp),unique_images=features['unique_images'],complete=True),
      old_pair_index=bind(ROOT/'results/rc_crisp_manual_baseline_v1/H593_workers.json'),rows=rows,
      cpu_consumers=dict(inside='run_rc_h593_m_inside_v1.py join; run_rc_h593_m_path_readout_v1.py',
        visual='run_rc_h593_m_visual_origin_v1.py join',coordinate='run_rc_h593_roma_coordinate_eval_v2.py'),
      excluded_from_fixed_output_cache='Trainable fusion adapter outputs and gradients change each update; fixed input bank is already reusable',
      labels_read=0,models_executed=0)
    DEST.mkdir(parents=True,exist_ok=True);p=DEST/'catalog.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(archive,indent=2)+'\n');tmp.replace(p)
    print(json.dumps(dict(status=archive['status'],counts=counts,feature_images=features['unique_images'],catalog=str(p))))
if __name__=='__main__':build()
