#!/usr/bin/env python3
"""Freeze a single preselected source model, before any external predictions."""
from pathlib import Path
import json
import shutil

import cache_rc_postllm_h593_v1 as C

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_postllm_grozi_external_v1'
AUTH=ROOT/'registry/rc_postllm_grozi_external_authority_v1_20260925.json'


def main():
    C.need(not AUTH.exists(),'AUTHORITY_ALREADY_FROZEN')
    parent_path=ROOT/'registry/rc_postllm_h593_authority_v1_20260924.json'
    parent=C.read(parent_path)
    for b in parent['code_sources']:C.checked(b)
    source=ROOT/'results/rc_postllm_h593_v1/fold0'
    fit=C.read(source/'POST_REAL/fit_validation.json')
    C.need(fit['status']=='H593_FIXED_EIGHT_PASSES_COMPLETE' and fit['steps']==3656
           and fit['train_queries']==457 and not fit['direct_M_in_head'],'FIXED_FOLD0_SOURCE')
    C.checked(fit['snapshot'])
    ext_path=ROOT/'registry/rc_new_hyp_grozi120_inference_authority_v1_20260913.json'
    ext=C.read(ext_path)
    wp=C.checked(ext['sources']['worker']); workers=C.read(wp)['records']
    C.need(len(workers)==480 and [w['execution_ordinal'] for w in workers]==list(range(480)), 'GROZI480_AXIS')
    C.need(all(set(w)=={'execution_ordinal','image_path','query_id'} for w in workers),'LABEL_FREE_WORKERS')
    legacy=C.read(ROOT/'results/rc_simple_external_transfer_v1/preflight.json')['legacy_gallery']
    C.checked(legacy);C.checked(ext['sources']['gallery_append'])
    sources=[]
    for relative in [
        'programs/run_rc_postllm_grozi_external_v1.py',
        'programs/join_rc_postllm_grozi_external_v1.py',
        'programs/prepare_rc_postllm_grozi_external_v1.py',
        'programs/submit_rc_postllm_grozi_external_v1.py',
        'slurm/rc_postllm_grozi_external_v1.sbatch',
        'plan/RC_POSTLLM_GROZI_EXTERNAL_V1_20260925.md',
        'tests/test_postllm_grozi_external_v1.py',
        'programs/run_rc_simple_external_replay_v1.py',
        'programs/join_rc_postllm_h593_v1.py']:
        sources.append(C.bind(ROOT/relative))
    sources+=parent['code_sources']
    sources=list({x['path']:x for x in sources}.values())
    a=dict(status='FROZEN_POSTLLM_GROZI480_AUTHORIZED',
        user_authorization='2026-09-25: 然后补一个内部模型的外部数据实验就收口',
        source_authority=C.bind(parent_path),source_fold=0,source_fit=C.bind(source/'POST_REAL/fit_validation.json'),
        source_snapshot=fit['snapshot'],source_heads={k:C.bind(source/'warm'/f'{k}.json')
            for k in ('INTERNAL3','ADDITIVE4','PRODUCT5')},
        model=parent['model'],model_source_validation=parent['model_source_validation'],
        external_authority=C.bind(ext_path),workers=ext['sources']['worker'],legacy_gallery=legacy,
        gallery_append=ext['sources']['gallery_append'],code_sources=sources,
        dataset='GroZi480',n=480,shards=60,workers_parallel=12,source_video_groups=27,
        primary='POST_REAL',controls=['RAW','NO_ADAPTER_INTERNAL3','POST_REAL_CONSTANT',
            'POST_REAL_SHUFFLED','EXTERNAL_ADDITIVE4','EXTERNAL_PRODUCT5'],
        training_updates=0,switch_threshold=0.,candidate_count=128,physical_gallery=5533,
        source_parity={'historical_fp16_exact':True,'projection_exact':True,'max_content_error':0.005},
        frame='EXIF_ORIENTED_BEFORE_RESIZE',processor_backend='torchvision',attention='sdpa',
        new_query_encoder_forwards=480,new_reference_encoder_forwards=0,new_roma_forwards=0,
        external_fitting=False,external_normalization=False,external_threshold_selection=False,
        fold_selection='Fixed fold0 by index before external predictions; no external-result selection',
        source_normalization='Saved fold0 TRAIN log-M mean/std; no external statistics',
        seal_policy='All480 predictions and controls before curator-label access',
        panel_scope='Previously opened external product-crop panel, not new untouched data or scene segmentation',
        continuation='Finish result and report; no automatic optimization, resampling or new experiments',
        time_limit='00:10:00',max_requeues=12)
    for b in a['source_heads'].values():C.checked(b)
    archive=OUT/'frozen_protocol';archive.mkdir(parents=True,exist_ok=True)
    for i,b in enumerate(sources):shutil.copyfile(b['path'],archive/f'{i:02d}_{Path(b["path"]).name}')
    C.write(AUTH,a);shutil.copyfile(AUTH,archive/AUTH.name)
    C.write(OUT/'preflight.json',dict(status='FROZEN_POSTLLM_GROZI_PROTOCOL_PASS',authority=C.bind(AUTH),
        queries=480,source_fold=0,source_snapshot=a['source_snapshot'],held_label_reads=0,
        external_training_updates=0,new_roma_forwards=0))
    print(json.dumps({'status':'FROZEN','authority':C.bind(AUTH)},ensure_ascii=False))


if __name__=='__main__':main()
