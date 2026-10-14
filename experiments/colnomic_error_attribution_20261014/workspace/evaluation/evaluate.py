"""Run frozen retrieval configurations against the complete formal inventory."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil
import time

import torch

from .cache import Cache, save_tensor
from .data import Dataset, digest, file_hash, read_json, write_json
from .dense import DENSE_DIMENSIONS, IMAGE_ENCODERS
from .metrics import evaluate, paired_report
from .scoring import SCORING_VERSION, CRISP_SOURCE, bm25_scores, dot_scores, fuse, multivector_scores, tokenize

NAMES=('colpali','colnomic3b','colnomic7b','Qwen3-VL-Embedding-2B',*IMAGE_ENCODERS,'colqwen2.5-v0.2')
MODES=('image','ocr_to_image','fusion','image_crisp','ocr_to_image_crisp','ocr_to_ocr','ocr_fusion')
FUSIONS=('fusion','ocr_fusion')
IDS={}
for i,model in enumerate(NAMES[:4]):
    for mode,base in [('image',1),('ocr_to_image',6),('fusion',10)]:
        IDS[(model,mode)]=f'B{base+i:02d}'
    if i<3:
        IDS[(model,'image_crisp')]=f'B{14+i:02d}'
        IDS[(model,'ocr_to_image_crisp')]=f'B{17+i:02d}'
IDS[('bm25','bm25')]='B05'
IDS.update({('Qwen3-VL-Embedding-2B','ocr_to_ocr'):'B20',('dinov3','image'):'B21',
            ('siglip2','image'):'B22',('Qwen3-VL-Embedding-2B','ocr_fusion'):'B23'})
IDS.update({('colqwen2.5-v0.2',mode):f'B{24+i:02d}' for i,mode in enumerate(
    ('image','ocr_to_image','fusion','image_crisp','ocr_to_image_crisp'))})


def load_scores(path,signature):
    if path.exists():
        value=torch.load(path,map_location='cpu',weights_only=True)
        if value.get('signature')==signature and torch.isfinite(value['scores']).all():
            return value
    return None


def neural_scores(cache,model,view,args,reference_view='image'):
    query=cache.embeddings(model,view,'photos')
    gallery=cache.embeddings(model,reference_view,'references')
    cache.release()
    recipe=dict(version=SCORING_VERSION,model=model,view=view,query=query['_file_sha256'],
        reference=gallery['_file_sha256'],crisp_source=CRISP_SOURCE,
        scorer_sha256=file_hash(Path(__file__).with_name('scoring.py')),
        valid_tokens='remove padding and zero-norm vectors; preserve attended template tokens',
        maxsim='mean over valid query tokens',dtype='float32',tf32=False)
    if reference_view=='text':
        recipe.update(reference_view='text',empty_query='count as retrieval failure',empty_reference='rank after nonempty OCR')
    if model in IMAGE_ENCODERS:
        recipe.update(reference_view='image',pooling=cache.config['models'][model]['image_contract']['pooling'])
    signature=digest(recipe)
    directory=cache.root/'cache/scores'/model
    mode='image' if view=='image' else ('ocr_to_ocr' if reference_view=='text' else 'ocr_to_image')
    paths=[directory/f'{mode}.pt']
    if model not in DENSE_DIMENSIONS:
        paths.append(directory/f'{mode}_crisp.pt')
    stored=[load_scores(p,signature) for p in paths]
    start=time.monotonic()
    if all(v is not None for v in stored) and not args.refresh_scores:
        for value in stored:
            value['cache_hit']=True
        return {p.stem:v for p,v in zip(paths,stored)},query
    print(f'Scoring {model}/{view}: {len(query["ids"])} x {len(gallery["ids"])}',flush=True)
    if model in DENSE_DIMENSIONS:
        matrices=[dot_scores(query['embeddings'],gallery['embeddings'])]
        if reference_view=='text':
            # A missing OCR document must not outrank a negative cosine match.
            matrices[0][:,~torch.as_tensor(gallery['has_text'],dtype=torch.bool)]=-1.000001
    else:
        chunks=directory/(mode+'_intermediate');chunks.mkdir(parents=True,exist_ok=True)
        matrices=[torch.zeros(len(query['ids']),len(gallery['ids'])) for _ in range(2)]
        prior_seconds=0.
        for first in range(0,len(query['ids']),50):
            last=min(first+50,len(query['ids']))
            cp=chunks/f'{first:05d}.pt'
            previous=torch.load(cp,map_location='cpu',weights_only=True) if cp.exists() else None
            if previous and previous.get('signature')==signature and not args.refresh_scores:
                pair=previous['matrices'];prior_seconds+=previous['seconds']
            else:
                chunk_start=time.monotonic()
                pair=multivector_scores(query['embeddings'][first:last],gallery['embeddings'],
                    device=args.device,query_batch=args.query_batch,reference_batch=args.reference_batch,
                    token_block=args.token_block,
                    progress=lambda done,total,seconds:print(
                        f'{model}/{view}: {first+done}/{len(query["ids"])} ({seconds:.1f}s chunk)',flush=True))
                save_tensor(cp,dict(signature=signature,matrices=pair,seconds=time.monotonic()-chunk_start))
            for target,part in zip(matrices,pair,strict=True):
                target[first:last]=part
    elapsed=time.monotonic()-start+(prior_seconds if model not in DENSE_DIMENSIONS else 0.)
    result={}
    for path,matrix in zip(paths,matrices,strict=True):
        if matrix.shape!=(len(query['ids']),len(gallery['ids'])) or not torch.isfinite(matrix).all():
            raise ValueError('Invalid score matrix')
        value=dict(signature=signature,recipe=recipe,scores=matrix,compute_seconds=elapsed,
            timing_scope='joint MaxSim and CRISP scoring' if len(paths)>1 else 'dense dot product',
            query_ids=query['ids'],reference_ids=gallery['ids'],
            blocks=dict(query=args.query_batch,reference=args.reference_batch,token=args.token_block),
            device='cpu' if model in DENSE_DIMENSIONS else args.device,cache_hit=False)
        save_tensor(path,value);result[path.stem]=value
    if len(paths)>1:
        shutil.rmtree(chunks)
    return result,query


def write_result(root,dataset,model,mode,scores,valid,config,alpha):
    folder=mode if mode not in FUSIONS or alpha==0.5 else f'{mode}_alpha_{alpha:g}'
    output=root/'results'/model/folder
    predictions,metrics,by_type=evaluate(scores,dataset,valid)
    config.update(id=IDS[(model,mode)],protocol=dataset.full_protocol,model=model,mode=mode,
        dataset_sha256=dataset.hashes,queries=len(dataset.photos),references=len(dataset.references),
        annotation_sha256=dataset.annotation_sha256,
        active_difference=dataset.active_difference,labels_used_only_for_evaluation=True)
    write_json(output/'config.json',config)
    write_json(output/'metrics.json',metrics)
    write_json(output/'by_type.json',by_type)
    temp=output/'predictions.jsonl.tmp'
    temp.write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in predictions))
    temp.replace(output/'predictions.jsonl')
    print(f'{IDS[(model,mode)]} {model}/{folder}: R@1={metrics["photo_average"]["recall_at_1"]:.4f}',flush=True)
    return predictions


def summarize_results(root):
    directory=root/'results';directory.mkdir(exist_ok=True)
    current=Dataset(root)
    with (directory/'.summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        summaries=[];predictions={}
        for path in sorted(directory.glob('*/*/metrics.json')):
            if not path.with_name('predictions.jsonl').exists():
                continue
            config=read_json(path.with_name('config.json'));metrics=read_json(path)
            if config.get('family')=='frozen_decision_head':
                continue
            if not current.matches_full_inventory(config.get('dataset_sha256')) or config.get('annotation_sha256')!=current.annotation_sha256:
                continue
            if config['mode'] in FUSIONS and config['alpha']!=0.5:
                continue
            row=dict(id=config['id'],model=config['model'],mode=config['mode'],
                     queries=metrics['queries'],references=metrics['reference_rows'])
            row.update({'photo_'+k:v for k,v in metrics['photo_average'].items()})
            row.update({'package_'+k:v for k,v in metrics['package_average'].items()})
            summaries.append(row)
            predictions[config['id']]=[json.loads(line) for line in path.with_name('predictions.jsonl').read_text().splitlines()]
        if not summaries:
            return
        summaries.sort(key=lambda r:r['id'])
        temporary=directory/'summary.csv.tmp'
        with temporary.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
        temporary.replace(directory/'summary.csv')
        all_path=directory/'all_results.csv'
        previous=[]
        if all_path.exists():
            with all_path.open(newline='') as stream:
                previous=[r for r in csv.DictReader(stream) if r['family']!='baseline']
        combined=[dict(id=r['id'],family='baseline',model=r['model'],mode=r['mode'],
            queries=r['queries'],references=r['references'],
            **{key:r['photo_'+key] for key in ('recall_at_1','recall_at_5','recall_at_10','mrr')},
            result=f"results/{r['model']}/{r['mode']}") for r in summaries]
        for row in previous:
            config_path=root/row['result']/'config.json'
            if config_path.exists():
                spec=read_json(config_path)
                if (current.matches_full_inventory(spec.get('dataset_sha256')) and
                        spec.get('annotation_sha256')==current.annotation_sha256):
                    metrics=read_json(config_path.with_name('metrics.json'))
                    row.update(metrics['photo_average'],queries=metrics['queries'],references=metrics['reference_rows'])
                    combined.append(row)
        combined.sort(key=lambda r:r['id'])
        temporary=all_path.with_suffix('.csv.tmp')
        with temporary.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(combined[0]))
            writer.writeheader();writer.writerows(combined)
        temporary.replace(all_path)
        comparisons={}
        for i in range(4):
            pairs=[(f'B{6+i:02d}',f'B{1+i:02d}'),(f'B{10+i:02d}',f'B{1+i:02d}')]
            if i<3:
                pairs.extend([(f'B{14+i:02d}',f'B{1+i:02d}'),(f'B{17+i:02d}',f'B{6+i:02d}')])
            for first,second in pairs:
                if first in predictions and second in predictions:
                    comparisons[first+'_minus_'+second]=paired_report(predictions[first],predictions[second])
        for first,second in [('B20','B05'),('B20','B09'),('B21','B03'),('B22','B04'),
                             ('B23','B04'),('B23','B20'),('B23','B13'),
                             ('B25','B24'),('B26','B24'),('B27','B24'),('B28','B25')]:
            if first in predictions and second in predictions:
                comparisons[first+'_minus_'+second]=paired_report(predictions[first],predictions[second])
        write_json(directory/'paired_comparisons.json',comparisons)


def run_model(dataset,settings,model,args):
    cache=Cache(dataset,settings,args.device,args.cache_only,args.refresh_embeddings,
                image_devices=args.image_devices,image_batch_size=args.image_batch_size)
    if model in IMAGE_ENCODERS:
        cache.cost['image_encoding_execution']=dict(devices=args.image_devices or [args.device],
            total_batch_size=args.image_batch_size,encoder_sha256=file_hash(Path(__file__).with_name('dense.py')))
    if model=='bm25':
        queries=cache.ocr('photos');refs=cache.ocr('references')
        qt=[queries[r['photo_id']]['text'] for r in dataset.photos]
        rt=[refs[r['reference_id']]['text'] for r in dataset.references]
        recipe=dict(version=SCORING_VERSION,k1=1.2,b=.75,tokenization='Unicode letters/digits; lowercase; query frequency',
            scorer_sha256=file_hash(Path(__file__).with_name('scoring.py')),
            query_order=digest([r['photo_id'] for r in dataset.photos]),
            reference_order=digest([r['reference_id'] for r in dataset.references]),
            query_ocr=file_hash(dataset.root/'cache/ocr/photos.json'),
            reference_ocr=file_hash(dataset.root/'cache/ocr/references.json'))
        sig=digest(recipe);path=dataset.root/'cache/scores/bm25/bm25.pt'
        value=load_scores(path,sig);hit=value is not None and not args.refresh_scores
        if not hit:
            start=time.monotonic();scores=bm25_scores(qt,rt)
            value=dict(signature=sig,scores=scores,compute_seconds=time.monotonic()-start)
            save_tensor(path,value)
        write_result(dataset.root,dataset,model,'bm25',value['scores'],[bool(tokenize(t)) for t in qt],
            dict(scoring=recipe,score_cache_hit=hit,retrieval_seconds=0. if hit else value['compute_seconds'],
                 score_source_compute_seconds=value['compute_seconds'],timing_scope='BM25 scoring',
                 cost=cache.cost,cost_note='Cached OCR inference time is unavailable for the original run.'),args.alpha)
        return
    modes=[m for m in MODES if (model,m) in IDS] if args.mode=='all' else [args.mode]
    if any((model,m) not in IDS for m in modes):
        raise ValueError(f'Unsupported mode for {model}: {modes}')
    values={};query_banks={}
    for view in ('image','text'):
        required=any(m in (('image','image_crisp','fusion','ocr_fusion') if view=='image' else ('ocr_to_image','ocr_to_image_crisp','fusion')) for m in modes)
        if required:
            stored,query=neural_scores(cache,model,view,args);values.update(stored);query_banks[view]=query
    if any(m in ('ocr_to_ocr','ocr_fusion') for m in modes):
        stored,query=neural_scores(cache,model,'text',args,reference_view='text')
        values.update(stored);query_banks['text']=query
    for mode in [m for m in FUSIONS if m in modes]:
        text_mode='ocr_to_image' if mode=='fusion' else 'ocr_to_ocr'
        fusion_start=time.monotonic()
        scores=fuse(values['image']['scores'],values[text_mode]['scores'],query_banks['text']['has_text'],args.alpha)
        recipe=dict(image=values['image']['signature'],text=values[text_mode]['signature'],
                    alpha=args.alpha,normalization='per-query population z-score',empty_text='image fallback')
        values[mode]=dict(scores=scores,signature=digest(recipe),recipe=recipe,
            compute_seconds=time.monotonic()-fusion_start,timing_scope='cached base-score fusion',cache_hit=False,device='cpu')
        save_tensor(dataset.root/f'cache/scores/{model}/{mode}_alpha_{args.alpha:g}.pt',values[mode])
    for mode in modes:
        value=values[mode]
        valid=query_banks['text']['has_text'] if mode.startswith('ocr_to_') else None
        write_result(dataset.root,dataset,model,mode,value['scores'],valid,
            dict(model_spec=settings['models'][model],scoring=value.get('recipe'),
                 alpha=args.alpha if mode in FUSIONS else None,cost=cache.cost,
                 retrieval_seconds=0. if value.get('cache_hit') else value['compute_seconds'],
                 score_source_compute_seconds=value['compute_seconds'],timing_scope=value.get('timing_scope'),
                 scoring_device=value.get('device'),scoring_blocks=value.get('blocks'),score_signature=value['signature'],
                 score_cache_hit=value.get('cache_hit',True),
                 cost_note='Cache validation and current generation are measured separately; no historical inference time is inferred.'),args.alpha)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--model',choices=['all','bm25',*NAMES],default='all')
    p.add_argument('--mode',choices=['all',*MODES],default='all')
    p.add_argument('--device',default='cuda:0');p.add_argument('--alpha',type=float,default=.5)
    p.add_argument('--image-devices',nargs='+',help='DINOv3/SigLIP2 replica devices, e.g. cuda:0 cuda:1')
    p.add_argument('--image-batch-size',type=int,default=8,help='Total dense image batch across replica devices')
    p.add_argument('--cache-only',action='store_true');p.add_argument('--refresh-embeddings',action='store_true')
    p.add_argument('--refresh-scores',action='store_true');p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--query-batch',type=int,default=2);p.add_argument('--reference-batch',type=int,default=16)
    p.add_argument('--token-block',type=int,default=2048)
    args=p.parse_args();torch.set_num_threads(2)
    if args.image_batch_size<1:
        p.error('--image-batch-size must be positive')
    if args.image_devices and len(set(args.image_devices))!=len(args.image_devices):
        p.error('--image-devices must be distinct')
    if args.summarize_only:
        summarize_results(args.root);return
    if args.cache_only and args.refresh_embeddings:
        p.error('--cache-only and --refresh-embeddings cannot be combined')
    dataset=Dataset(args.root);settings=read_json(args.root/'configs/baselines.json')
    for model in [*NAMES,'bm25'] if args.model=='all' else [args.model]:
        run_model(dataset,settings,model,args)
        summarize_results(dataset.root)


if __name__=='__main__':
    main()
