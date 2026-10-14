"""Register the frozen CONTENT_BOX_CE_POLISH18 head as configuration H01 (36).

This is a decision-only system. No full-gallery reranking is defined, so R@5,
R@10, MRR and error ranks remain unavailable rather than copying RAW ranks.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .data import Dataset, file_hash, read_csv, read_json, write_json

ROOT=Path(__file__).resolve().parents[1]
ID='H01'
FAMILY='frozen_decision_head'
MODE='content_box_ce_polish18_fold0'
RESULT=Path('results/content_box_ce_polish18/fold0_full1117')
RANK_METRICS=('recall_at_5','recall_at_10','mrr')


def write_csv(path,rows,fields):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def decision_summary(rows):
    packages=sorted({r['package_id'] for r in rows})
    photo=sum(r['correct'] for r in rows)/len(rows) if rows else None
    package=float(np.mean([np.mean([r['correct'] for r in rows if r['package_id']==p]) for p in packages])) if rows else None
    return dict(queries=len(rows),packages=len(packages),source_groups=len({r['group_id'] for r in rows}),
        failures=0,correct=sum(r['correct'] for r in rows),incorrect=sum(not r['correct'] for r in rows),
        photo_average=dict(recall_at_1=photo,**dict.fromkeys(RANK_METRICS)),
        package_average=dict(recall_at_1=package,**dict.fromkeys(RANK_METRICS)),
        ranking_scope='decision_only; full-gallery ranks are not defined')


def verify_result(root,dataset,folder):
    """Replay all saved X11 logits, fixed-head actions and current SHA labels."""
    config=read_json(folder/'config.json');deployment=root/config['deployment']
    manifest=read_json(deployment/'manifest.json');model_path=root/config['model_path']
    assert config['id']==ID and config['experiment_ordinal']==36 and config['family']==FAMILY
    assert config['fold']==0 and config['threshold']==0 and config['ranking_scope']=='decision_only'
    assert dataset.matches_full_inventory(config['dataset_sha256']) and config['annotation_sha256']==dataset.annotation_sha256
    assert config['deployment_manifest_sha256']==file_hash(deployment/'manifest.json')
    assert config['model_sha256']==file_hash(model_path) and config['infer_sha256']==file_hash(root/config['infer_path'])
    model=read_json(model_path);assert model['fold']==0
    audit=read_json(deployment/'verification.json');assert audit['status']=='passed' and audit['queries_verified']==len(dataset.photos)
    assert audit['logits_verified']==127*len(dataset.photos) and audit['threshold']==0 and audit['target_insertions']==0
    scores=torch.load(root/'cache/scores/colnomic7b/image.pt',map_location='cpu',weights_only=True,mmap=True)
    assert scores['query_ids']==[r['photo_id'] for r in dataset.photos]
    assert scores['reference_ids']==[r['reference_id'] for r in dataset.references]
    spec=importlib.util.spec_from_file_location('registered_polish18_infer',root/config['infer_path'])
    infer=importlib.util.module_from_spec(spec);spec.loader.exec_module(infer)
    predictions=[json.loads(s) for s in (folder/'predictions.jsonl').read_text().splitlines()]
    assert [r['photo_id'] for r in predictions]==scores['query_ids']
    ref_sha={r['reference_id']:r['sha256'] for r in dataset.references}
    for index,(q,row) in enumerate(zip(manifest['queries'],predictions,strict=True)):
        assert q['id']==row['photo_id'] and q['bank_index']==index
        case=deployment/'queries'/q['id'];record=read_json(case/'input.json');prediction=read_json(case/'prediction.json');receipt=read_json(case/'receipt.json')
        assert receipt['manifest_signature']==manifest['signature'] and receipt['model_sha256']==config['model_sha256']
        assert file_hash(case/'input.json')==receipt['input_sha256'] and file_hash(case/'prediction.json')==receipt['prediction_sha256']
        assert infer.predict(record,model)==prediction
        seen=set();ranked=[]
        for ri in np.argsort(-scores['scores'][index].numpy(),kind='stable'):
            ri=int(ri);sha=dataset.references[ri]['sha256']
            if sha in seen:continue
            seen.add(sha);ranked.append(ri)
            if len(ranked)==128:break
        assert q['ranked_physical_rows']==ranked and q['candidate_physical_rows']==sorted(ranked)
        assert record['candidate_physical_rows']==q['candidate_physical_rows'] and record['winner']==q['winner']
        assert record['challenger_positions']==q['challenger_positions']
        assert q['raw_scores']==[float(scores['scores'][index,i]) for i in q['candidate_physical_rows']]
        assert prediction['selected'] in q['candidate_physical_rows'] and prediction['fold']==0
        selected=dataset.references[prediction['selected']]['reference_id']
        positive=sorted(dataset.positives[row['package_id']]);correct=ref_sha[selected] in {ref_sha[r] for r in positive}
        assert row['selected_reference_id']==selected and row['correct']==correct and row['positive_reference_ids']==positive
        assert row['rank']==(1 if correct else None) and row['top_reference_ids']==[selected]
        assert row['failure'] is None and row['ranking_scope']=='decision_only' and row['action']==prediction['action']
        assert row['type']==dataset.labels[row['photo_id']]
    metrics=read_json(folder/'metrics.json');expected=decision_summary(predictions)
    assert all(metrics[k]==v for k,v in expected.items())
    return dict(status='passed',model=config['model'],mode=MODE,protocol=dataset.full_protocol,predictions=len(predictions),
        independently_verified_decisions=True,independently_verified_natural_C128=True,independently_verified_ranks=False,
        logits_replayed=127*len(predictions),experiment_ordinal=36,ranking_scope='decision_only',
        photo_average=metrics['photo_average'],package_average=metrics['package_average'],
        fold=0,threshold=0,encoder_forwards=0,training_updates=0)


def register(root=ROOT):
    torch.set_num_threads(2);dataset=Dataset(root);folder=root/RESULT
    active=read_json(root/'cache/content_box_ce_polish18/deployment/active.json');deployment=Path(active['path'])
    if not deployment.is_absolute():deployment=root/'cache/content_box_ce_polish18/deployment'/deployment
    manifest=read_json(deployment/'manifest.json');summary=read_json(deployment/'summary.json')
    assert summary['status']=='complete' and summary['queries_completed']==len(dataset.photos)==1117
    assert [q['id'] for q in manifest['queries']]==[q['photo_id'] for q in dataset.photos]
    for p in [root/'dataset/lists/photos.csv',root/'dataset/lists/references.csv']:
        assert manifest['recipe']['inputs'][str(p)]==file_hash(p)
    folder.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup=root/'cache/completions'/('register_H01_'+stamp);backup.mkdir(parents=True,exist_ok=False)
    for name in ['all_results.csv','cohort_results.csv','verification.json','README.md']:
        shutil.copy2(root/'results'/name,backup/name)
    config=dict(id=ID,experiment_ordinal=36,family=FAMILY,model='colnomic7b',head='CONTENT_BOX_CE_POLISH18',mode=MODE,
        protocol=dataset.full_protocol,queries=len(dataset.photos),references=len(dataset.references),
        dataset_sha256=dataset.hashes,annotation_sha256=dataset.annotation_sha256,active_difference=dataset.active_difference,
        fold=0,threshold=0,candidate_budget=128,ranking_scope='decision_only',labels_used_only_for_evaluation=True,
        deployment=str(deployment.relative_to(root)),deployment_manifest_sha256=file_hash(deployment/'manifest.json'),
        model_path='weights/content_box_ce_polish18/models/fold0.json',model_sha256=file_hash(root/'weights/content_box_ce_polish18/models/fold0.json'),
        infer_path='weights/content_box_ce_polish18/infer.py',infer_sha256=file_hash(root/'weights/content_box_ce_polish18/infer.py'),
        result=str(RESULT),encoder_forwards=0,training_updates=0,source_commit=manifest['recipe']['source_commit'],
        unavailable_metrics=list(RANK_METRICS),overlap={k:v for k,v in summary['overlap'].items() if k!='records'})
    rows=[]
    for q in manifest['queries']:
        prediction=read_json(deployment/'queries'/q['id']/'prediction.json');selected=dataset.references[prediction['selected']]['reference_id']
        positive=sorted(dataset.positives[q['package_id']])
        correct=dataset.references[prediction['selected']]['sha256'] in {r['sha256'] for r in dataset.references if r['reference_id'] in positive}
        rows.append(dict(photo_id=q['id'],package_id=q['package_id'],group_id=dataset.groups[q['package_id']],rank=1 if correct else None,
            correct=correct,selected_reference_id=selected,top_reference_ids=[selected],positive_reference_ids=positive,
            failure=None,type=dataset.labels[q['id']],ranking_scope='decision_only',action=prediction['action'],
            RAW_reference_id=dataset.references[q['ranked_physical_rows'][0]]['reference_id'],candidate_budget=128))
    metrics=decision_summary(rows);metrics.update(protocol=dataset.full_protocol,reference_rows=len(dataset.references),
        unique_reference_images=len({r['sha256'] for r in dataset.references}),rescued=summary['RESCUE'],harmed=summary['BREAK'],
        target_absent_C128=summary['target_absent_C128'],fold=0,threshold=0,overlap=config['overlap'])
    write_json(folder/'config.json',config);write_json(folder/'metrics.json',metrics)
    (folder/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    by_type={}
    for placement in [None,'tabletop','handheld']:
        for background in [None,'single_object','multi_object']:
            if placement is None and background is None:continue
            key='/'.join(v for v in [placement,background] if v)
            by_type[key]=decision_summary([r for r in rows if (placement is None or r['type']['placement']==placement) and (background is None or r['type']['background']==background)])
    for occlusion in ['none','foreground','unclear']:
        by_type['occlusion='+occlusion]=decision_summary([r for r in rows if r['type'].get('occlusion')==occlusion])
    write_json(folder/'by_type.json',by_type)
    verified=verify_result(root,dataset,folder);write_json(folder/'registration_verification.json',verified)
    with (root/'results/.summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        table=read_csv(root/'results/all_results.csv');fields=list(table[0]);old=[r for r in table if r['id']!=ID]
        row=dict(id=ID,family=FAMILY,model='colnomic7b',mode=MODE,queries=len(rows),references=len(dataset.references),result=str(RESULT),
            recall_at_1=metrics['photo_average']['recall_at_1'],**{k:'' for k in RANK_METRICS})
        insert=next((i for i,r in enumerate(old) if r['id'].startswith('L')),len(old));old.insert(insert,row)
        assert insert==35 and len({r['id'] for r in old})==len(old)
        write_csv(root/'results/all_results.csv',old,fields)
        cohort=read_csv(root/'results/cohort_results.csv');cfields=list(cohort[0]);cohort=[r for r in cohort if r['id']!=ID]
        for label,subset in [('all',rows),('original_1000',[r for r in rows if int(r['package_id'].split('_')[1])<=113]),('supplement_117',[r for r in rows if int(r['package_id'].split('_')[1])>113])]:
            cohort.append(dict(id=ID,cohort=label,queries=len(subset),recall_at_1=sum(r['correct'] for r in subset)/len(subset),**{k:'' for k in RANK_METRICS}))
        write_csv(root/'results/cohort_results.csv',cohort,cfields)
        verification=read_json(root/'results/verification.json')
        assert verification['status']=='passed' or (verification['status']=='partial' and verification.get('current_scene_annotation_check',{}).get('status')=='pending_refresh')
        assert set(verification['methods'])=={r['id'] for r in old if r['id']!=ID} or set(verification['methods'])=={r['id'] for r in old}
        verification['methods'][ID]=verified;verification['configurations']=len(old)
        verification['registered_decision_head_utc']=datetime.now(timezone.utc).isoformat()
        write_json(root/'results/verification.json',verification)
    print(json.dumps(dict(id=ID,experiment_ordinal=36,total_registered_configurations=len(old),correct=metrics['correct'],recall_at_1=metrics['photo_average']['recall_at_1'],
        unavailable_metrics=list(RANK_METRICS),verification='passed',backup=str(backup)),ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    if args.verify_only:
        torch.set_num_threads(2);print(json.dumps(verify_result(ROOT,Dataset(ROOT),ROOT/RESULT),ensure_ascii=False,indent=2))
    else:register()


if __name__=='__main__':main()
