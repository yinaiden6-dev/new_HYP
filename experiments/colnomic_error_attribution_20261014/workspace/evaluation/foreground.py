"""D01 human-box / D03 automatic SAM3 crops and full-gallery retrieval.

Run with the rebut Python environment. Manual regions are frozen at startup;
automatic inference receives only pixels and the fixed generic text prompt.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from .cache import Cache, save_tensor, valid_embedding
from .data import Dataset, digest, file_hash, read_json, write_json
from .metrics import evaluate, paired_report
from .regions import crop_region, open_photo
from .scoring import multivector_scores, SCORING_VERSION


def now():
    return datetime.now(timezone.utc).isoformat()


def save_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    image.save(temp, format='PNG'); temp.replace(path)


def checkpoint_spec(path):
    path = Path(path).resolve()
    files = [path] if path.is_file() else sorted(p for p in path.iterdir()
              if p.is_file() and p.suffix in ('.safetensors', '.json', '.txt', '.model'))
    if not files or not any(p.suffix in ('.pt', '.safetensors') for p in files):
        raise ValueError('Provide an authorized SAM3 .pt file or Transformers checkpoint directory')
    return dict(path=str(path), files={p.name:file_hash(p) for p in files})


def select_sam3_top1(outputs, threshold):
    """Choose the same stable top-1 before expanding masks to original resolution."""
    scores = outputs.pred_logits.sigmoid()
    if outputs.presence_logits is not None:
        scores = scores * outputs.presence_logits.sigmoid()
    if scores.ndim != 2 or scores.shape[0] != 1 or not torch.isfinite(scores).all():
        raise ValueError('Expected one image with finite SAM3 confidence scores')
    if scores.shape[1] == 0:
        return outputs, dict(candidates=0, selected_index=None)
    index = int(scores[0].argmax())
    keep = scores[0] > threshold
    selected = SimpleNamespace(pred_logits=outputs.pred_logits[:,index:index+1],
        pred_boxes=outputs.pred_boxes[:,index:index+1],
        pred_masks=outputs.pred_masks[:,index:index+1],presence_logits=outputs.presence_logits)
    return selected, dict(candidates=int(keep.sum()), selected_index=int(keep[:index].sum()))


class Sam3Segmenter:
    top1_before_postprocessing = True

    def __init__(self, checkpoint, config, device, repo=None):
        self.config, self.device = config, device
        self.official = Path(checkpoint).is_file()
        if self.official:
            if repo:
                sys.path.insert(0, str(Path(repo).resolve()))
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            self.model = build_sam3_image_model(bpe_path=None, checkpoint_path=str(checkpoint),
                                               device=device, eval_mode=True)
            self.processor = Sam3Processor(self.model, device=device,
                               confidence_threshold=config['sam_confidence_threshold'])
        else:
            from transformers import Sam3Model, Sam3Processor
            self.processor = Sam3Processor.from_pretrained(checkpoint, local_files_only=True)
            self.model = Sam3Model.from_pretrained(checkpoint, local_files_only=True,
                              dtype=torch.float32, attn_implementation='sdpa').to(device).eval()

    @torch.inference_mode()
    def __call__(self, image):
        # fp16 autocast works on the available pre-Ampere GPUs; do not assume bf16 support.
        context = torch.autocast('cuda', dtype=torch.float16) if self.device.startswith('cuda') else nullcontext()
        selection = None
        with context:
            if self.official:
                state = self.processor.set_image(image)
                result = self.processor.set_text_prompt(state=state, prompt=self.config['sam_prompt'])
            else:
                inputs = self.processor(images=image, text=self.config['sam_prompt'], return_tensors='pt').to(self.device)
                outputs = self.model(**inputs)
                selected, selection = select_sam3_top1(outputs,self.config['sam_confidence_threshold'])
                result = self.processor.post_process_instance_segmentation(selected,
                    threshold=self.config['sam_confidence_threshold'], mask_threshold=self.config['sam_mask_threshold'],
                    target_sizes=[(image.height,image.width)])[0]
        scores = torch.as_tensor(result['scores']).detach().float().cpu().flatten()
        if not torch.isfinite(scores).all():
            raise ValueError('Non-finite SAM confidence scores')
        if not len(scores):
            return None, dict(candidates=0, fallback_reason='no_detection')
        index = int(torch.argsort(scores, descending=True, stable=True)[0])
        raw = torch.as_tensor(result['masks'][index]).detach().float().cpu().squeeze()
        if raw.ndim != 2 or not torch.isfinite(raw).all():
            raise ValueError('Invalid SAM mask')
        resized = tuple(raw.shape) != (image.height,image.width)
        if resized:
            raw = torch.nn.functional.interpolate(raw[None,None], size=(image.height,image.width),
                                                   mode='bilinear', align_corners=False)[0,0]
        mask = raw > self.config['sam_mask_threshold']
        details = dict(candidates=len(scores), selected_index=index, confidence=float(scores[index]),
                       mask_resized_to_original=resized)
        if selection is not None:
            details.update(selection)
        if not mask.any():
            return None, dict(**details, fallback_reason='empty_mask')
        return Image.fromarray(mask.numpy().astype('uint8')*255), details


def manual_snapshot(dataset, allow_incomplete):
    from dataset.tools.subject_boxes import resolve_records
    path = dataset.root/'dataset/annotations/subject_boxes.json'
    prelabels_path = path.with_name('subject_box_prelabels.json')
    if not path.exists() and not prelabels_path.exists():
        raise ValueError('No saved boxes or prelabels are available')
    data = read_json(path) if path.exists() else dict(schema='release.subject_boxes.v1',
        photos_sha256=dataset.hashes['photos.csv'], records={})
    if data.get('schema') != 'release.subject_boxes.v1' or data.get('photos_sha256') != dataset.hashes['photos.csv']:
        raise ValueError('Human boxes belong to a different inventory')
    ids = {r['photo_id'] for r in dataset.photos}
    if set(data['records']) - ids:
        raise ValueError('Human annotations contain unknown photos')
    if prelabels_path.exists():
        prelabels = read_json(prelabels_path)
        if (prelabels.get('schema') == 'release.subject_box_prelabels.v1' and
                prelabels.get('photos_sha256') == dataset.hashes['photos.csv']):
            data['records'] = resolve_records(dataset.photos, data['records'], prelabels['records'])
    pending = [pid for pid in ids if data['records'].get(pid,{}).get('status','pending') == 'pending']
    if pending and not allow_incomplete:
        raise ValueError(f'{len(pending)} photos still need boxes. Finish annotation or explicitly use --allow-incomplete.')
    return data


def box_inputs(snapshot):
    return {pid: {k: v for k, v in record.items() if k not in ('package_id', 'image_path')}
            for pid, record in snapshot['records'].items()}


def verify_rankings(scores, predictions, dataset):
    """Check the complete output with NumPy sorting, independently of torch ranking."""
    reference_ids = [r['reference_id'] for r in dataset.references]
    sha = {r['reference_id']:r['sha256'] for r in dataset.references}
    top_count = min(10, len(set(sha.values())))
    if [p['photo_id'] for p in predictions] != [p['photo_id'] for p in dataset.photos]:
        raise ValueError('Predictions do not cover the ordered formal queries')
    order = np.argsort(-scores.numpy(), axis=1, kind='stable')
    for photo, prediction, ranking in zip(dataset.photos, predictions, order, strict=True):
        positives = {sha[r] for r in dataset.positives[photo['package_id']]}
        seen, top, rank = set(), [], None
        for index in ranking:
            rid = reference_ids[index]
            if sha[rid] in seen:
                continue
            seen.add(sha[rid])
            if len(top) < 10:
                top.append(rid)
            if rank is None and sha[rid] in positives:
                rank = len(seen)
            if rank is not None and len(top) == top_count:
                break
        if (prediction['rank'] != rank or prediction['correct'] != (rank == 1) or
                prediction['top_reference_ids'] != top or prediction['selected_reference_id'] != top[0]):
            raise ValueError(f"Independent ranking check failed: {photo['photo_id']}")
    return dict(status='passed',method='independent NumPy stable ranking and SHA deduplication',
                predictions=len(predictions),reference_rows=len(reference_ids),verified_utc=now())


def prepare(dataset, directory, spec, progress, segmenter_factory=Sam3Segmenter):
    manifest_path = directory/'manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get('signature') != digest(spec):
            raise ValueError('Stale foreground manifest')
        verify_manifest(dataset, manifest)
        return manifest
    cfg = spec['settings']; rows = []; segmenter = None
    item_dir = directory/'intermediate/regions'; item_dir.mkdir(parents=True,exist_ok=True)
    for index, photo in enumerate(dataset.photos):
        pid = photo['photo_id']; item = item_dir/f'{pid}.json'
        if item.exists():
            row = read_json(item)
            if row['source_sha256'] != photo['sha256']:
                raise ValueError(f'Stale source: {pid}')
            for path_key, sha_key in [('query_path','query_sha256'), ('mask_path','mask_sha256')]:
                if row.get(path_key) and file_hash(dataset.root/row[path_key]) != row[sha_key]:
                    raise ValueError(f'Cached crop/mask was modified: {pid}')
        else:
            image = open_photo(dataset.root/'dataset'/photo['path'])
            row = dict(photo_id=pid, source_path='dataset/'+photo['path'], source_sha256=photo['sha256'],
                       width=image.width, height=image.height, fallback_reason=None)
            mask = None; box = None
            if spec['route'] == 'manual':
                annotation = spec['annotations']['records'].get(pid,{})
                state = annotation.get('status','pending')
                if state not in ('accepted','confirmed','pending','unclear'):
                    raise ValueError(f'Invalid annotation status: {pid}')
                if annotation and (annotation.get('image_sha256') != photo['sha256'] or
                    (annotation.get('width'),annotation.get('height')) != image.size):
                    raise ValueError(f'Stale annotation: {pid}')
                row['annotation_status'] = state
                row['annotation_source'] = annotation.get('source')
                if state in ('accepted','confirmed'):
                    box = annotation['bbox']
                else:
                    row['fallback_reason'] = 'human_'+state
            else:
                if segmenter is None:
                    segmenter = segmenter_factory(spec['sam']['path'], cfg, spec['device'], spec.get('sam_repo'))
                mask, details = segmenter(image); row.update(details)
                if mask is not None:
                    box = list(mask.getbbox())
            if box is None:
                row.update(query_path=row['source_path'], query_sha256=photo['sha256'], fallback=True)
            else:
                cropped, bounds = crop_region(image, box, cfg['margin'], mask)
                path = directory/'photos'/f'{pid}.png'; save_png(path,cropped)
                row.update(query_path=str(path.relative_to(dataset.root)), query_sha256=file_hash(path),
                           bbox=list(box), crop_bounds=bounds, fallback=False)
                if mask is not None:
                    mask_path = directory/'intermediate/masks'/f'{pid}.png'; save_png(mask_path,mask)
                    row.update(mask_path=str(mask_path.relative_to(dataset.root)), mask_sha256=file_hash(mask_path))
            write_json(item,row)
        rows.append(row)
        progress('prepare',index+1,len(dataset.photos), fallback_count=sum(r['fallback'] for r in rows))
    del segmenter; gc.collect(); torch.cuda.empty_cache()
    manifest = dict(schema='release.foreground_inputs.v1', signature=digest(spec),
                    photos=rows, counts=dict(total=len(rows), cropped=sum(not r['fallback'] for r in rows),
                    fallback=sum(r['fallback'] for r in rows)),
                    fallback_reasons=dict(Counter(r['fallback_reason'] for r in rows if r['fallback'])))
    write_json(directory/'manifest.json',manifest)
    return manifest


def verify_manifest(dataset, manifest):
    rows = manifest['photos']
    if [r['photo_id'] for r in rows] != [p['photo_id'] for p in dataset.photos]:
        raise ValueError('Foreground manifest does not match the ordered photos')
    for row, photo in zip(rows, dataset.photos, strict=True):
        if row['source_sha256'] != photo['sha256']:
            raise ValueError(f"Stale source: {photo['photo_id']}")
        for path_key, sha_key in [('query_path', 'query_sha256'), ('mask_path', 'mask_sha256')]:
            path = dataset.root/row[path_key] if row.get(path_key) else None
            if path_key == 'query_path' and row.get('fallback'):
                path = dataset.root/'dataset'/photo['path']
            if path and file_hash(path) != row[sha_key]:
                raise ValueError(f"Cached crop/mask was modified: {photo['photo_id']}")


def load_bank(directory, spec, manifest):
    path = directory/'photo_embeddings.pt'
    value = torch.load(path, map_location='cpu', weights_only=True)
    model = spec['settings']['model']; config = spec['baselines']['models'][model]
    rows = manifest['photos']
    expected = dict(ids=[r['photo_id'] for r in rows],
        query_sha256=[r['query_sha256'] for r in rows], fallback=[r['fallback'] for r in rows],
        model=config['image_model'], contract=config['image_contract'], signature=digest(spec))
    if any(value.get(k) != v for k,v in expected.items()):
        raise ValueError('Stale foreground embedding bank')
    if len(value['embeddings']) != len(rows) or any(
            not valid_embedding(v, model, 'image') for v in value['embeddings']):
        raise ValueError('Invalid foreground embedding bank')
    if (directory/'bank.json').exists():
        if file_hash(path) != read_json(directory/'bank.json')['bank_sha256']:
            raise ValueError('Foreground embedding bank was modified')
    return value['embeddings']


def score_signature(spec, manifest, reference_hash):
    return digest(dict(spec=digest(spec), query=[r['query_sha256'] for r in manifest['photos']],
                       reference=reference_hash, scorer=SCORING_VERSION))


def retrieve(dataset, directory, spec, manifest, progress, embedding_only=False):
    from .encoders import Encoder
    baseline_config = spec['baselines']; model = spec['settings']['model']
    cache = Cache(dataset,baseline_config,device=spec['device'],cache_only=True)
    refs = cache.embeddings(model,'image','references')
    original = cache.embeddings(model,'image','photos')
    rows = manifest['photos']; ids = [r['photo_id'] for r in rows]
    if ids != original['ids']:
        raise ValueError('Foreground queries are not the complete ordered formal inventory')
    vectors = []; encoder = None; encoded = 0; started = time.monotonic()
    bank_path = directory/'photo_embeddings.pt'
    cached_vectors = load_bank(directory, spec, manifest) if bank_path.exists() else None
    for i,row in enumerate(rows):
        item = directory/'intermediate/embeddings'/f"{row['photo_id']}.pt"
        if cached_vectors is not None:
            value = cached_vectors[i]
        elif row['fallback']:
            value = original['embeddings'][i]
        elif item.exists():
            stored = torch.load(item,map_location='cpu',weights_only=True)
            if stored['query_sha256'] != row['query_sha256']:
                raise ValueError('Stale crop embedding')
            value = stored['embedding']
        else:
            if encoder is None:
                encoder = Encoder(model,baseline_config['models'][model],dataset.root,spec['device'])
            value = encoder.encode_image(dataset.root/row['query_path'])
            if not valid_embedding(value,model,'image'):
                raise ValueError(f"Invalid embedding: {row['photo_id']}")
            save_tensor(item,dict(query_sha256=row['query_sha256'],embedding=value)); encoded += 1
        if not valid_embedding(value,model,'image'):
            raise ValueError(f"Invalid cached embedding: {row['photo_id']}")
        vectors.append(value); progress('embedding',i+1,len(rows),new_embeddings=encoded)
    if encoder:
        encoder.release()
    encoding_seconds = time.monotonic()-started
    if cached_vectors is None:
        save_tensor(bank_path,dict(schema='release.foreground_embeddings.v1',ids=ids,
            embeddings=vectors, query_sha256=[r['query_sha256'] for r in rows],
            fallback=[r['fallback'] for r in rows], model=baseline_config['models'][model]['image_model'],
            contract=baseline_config['models'][model]['image_contract'],signature=digest(spec)))
    bank_info = dict(schema='release.foreground_bank.v1', signature=digest(spec),
        model=model, photos=len(rows), reference_rows=len(dataset.references), counts=manifest['counts'],
        bank=str(bank_path.relative_to(dataset.root)), bank_sha256=file_hash(bank_path),
        manifest=str((directory/'manifest.json').relative_to(dataset.root)),
        manifest_sha256=file_hash(directory/'manifest.json'), reference_cache_sha256=refs['_file_sha256'],
        reference_bank=f'cache/image_embeddings/{model}/references.pt',
        annotation_sources=dict(Counter(r.get('annotation_source','sam3') for r in rows)),
        dtype=str(vectors[0].dtype), dimensions=vectors[0].shape[-1], encoding_seconds_this_process=encoding_seconds,
        new_embeddings=encoded, completed_utc=now())
    write_json(directory/'bank.json',bank_info)
    write_json(directory.parent/'latest.json',bank_info)
    if embedding_only:
        progress('embedded',len(rows),len(rows),bank=str(bank_path),bank_sha256=bank_info['bank_sha256'])
        return
    scores = []; scoring_seconds=0.; cfg=spec['settings']
    final_path = directory/'scores.pt'
    final_signature = score_signature(spec, manifest, refs['_file_sha256'])
    if final_path.exists():
        saved = torch.load(final_path, map_location='cpu', weights_only=True)
        if saved['signature'] != final_signature:
            raise ValueError('Stale retrieval scores')
        scores.append(saved['scores'])
    for start in (() if scores else range(0,len(rows),50)):
        stop=min(start+50,len(rows)); path=directory/'intermediate/scores'/f'{start:04d}-{stop:04d}.pt'
        signature=digest(dict(spec=digest(spec),query=[r['query_sha256'] for r in rows[start:stop]],
                             reference=refs['_file_sha256'],scorer=SCORING_VERSION))
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['signature'] != signature:
                raise ValueError('Stale retrieval scores')
            score=saved['scores']
        else:
            tick=time.monotonic()
            score,_=multivector_scores(vectors[start:stop],refs['embeddings'],device=spec['device'],
                query_batch=cfg['query_batch'],reference_batch=cfg['reference_batch'],token_block=cfg['token_block'],
                progress=lambda done,total,seconds:progress('retrieval',start+done,len(rows)))
            elapsed=time.monotonic()-tick;scoring_seconds+=elapsed
            save_tensor(path,dict(signature=signature,scores=score,compute_seconds=elapsed))
        scores.append(score);progress('retrieval',stop,len(rows))
    scores=torch.cat(scores)
    if tuple(scores.shape) != (len(rows), len(dataset.references)) or not torch.isfinite(scores).all():
        raise ValueError('Invalid foreground score matrix')
    if not final_path.exists():
        save_tensor(final_path, dict(signature=final_signature, scores=scores))
    predictions,metrics,by_type=evaluate(scores,dataset)
    verification=verify_rankings(scores,predictions,dataset)
    control_path=dataset.root/'results/colnomic7b/image'
    control_spec=read_json(control_path/'config.json')
    if not dataset.matches_full_inventory(control_spec['dataset_sha256']) or control_spec['annotation_sha256'] != dataset.annotation_sha256:
        raise ValueError('B03 comparison uses different data or scene labels')
    control=[json.loads(line) for line in (control_path/'predictions.jsonl').read_text().splitlines()]
    for prediction,row in zip(predictions,rows,strict=True):
        prediction['foreground']=dict(fallback=row['fallback'],reason=row['fallback_reason'])
    result=dataset.root/'results/foreground'/directory.parent.name/directory.name
    result.mkdir(parents=True,exist_ok=True)
    write_json(result/'config.json',dict(spec,protocol=getattr(dataset,'full_protocol',f'Full-{len(rows)}'),
               queries=len(rows),references=len(dataset.references),counts=manifest['counts'],id='D01' if spec['route']=='manual' else 'D03',
               region_manifest=str((directory/'manifest.json').relative_to(dataset.root)),
               region_manifest_sha256=file_hash(directory/'manifest.json'),reference_cache_sha256=refs['_file_sha256'],
               cost=dict(encoding_and_cache_seconds=encoding_seconds,new_embeddings=encoded,
                         scoring_seconds_this_process=scoring_seconds),
               timing_scope='retrieval jointly computes MaxSim and CRISP; only MaxSim is reported'))
    write_json(result/'metrics.json',metrics);write_json(result/'by_type.json',by_type)
    paired=paired_report(predictions,control)
    write_json(result/'paired_vs_B03.json',paired)
    write_json(result/'verification.json',verification)
    write_json(result/'regions_summary.json',dict(counts=manifest['counts'],fallback_reasons=manifest['fallback_reasons']))
    output=result/'predictions.jsonl';tmp=output.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in predictions));tmp.replace(output)
    control_ranks=np.array([p['rank'] or np.inf for p in control])
    control_values=[float((control_ranks<=k).mean()) for k in (1,5,10)]+[float((1/control_ranks).mean())]
    current=metrics['photo_average'];ci=paired['intervals']['photo_average']['recall_at_1']
    identifier='D01' if spec['route']=='manual' else 'D03'
    description='人工框裁剪' if spec['route']=='manual' else 'SAM3 自动 Top-1 白底抠图'
    report=['# 主体检索实验结果','',
            f"全部 {len(rows):,} 张实拍检索 {len(dataset.references):,} 行 reference；固定 ColNomic 7B 和 MaxSim。",'',
            '| ID | 输入 | R@1 | R@5 | R@10 | MRR |','|---|---|---:|---:|---:|---:|',
            f'| B03 | 原始整图 | {control_values[0]:.2%} | {control_values[1]:.2%} | {control_values[2]:.2%} | {control_values[3]:.4f} |',
            f"| {identifier} | {description} | {current['recall_at_1']:.2%} | {current['recall_at_5']:.2%} | {current['recall_at_10']:.2%} | {current['mrr']:.4f} |",'',
            f"成功裁剪 {manifest['counts']['cropped']} 张，回退原图 {manifest['counts']['fallback']} 张；所有 query 均保留。",'',
            f"相对 B03：救回 {paired['rescued']} 张，带错 {paired['harmed']} 张。R@1 差值 {(current['recall_at_1']-control_values[0])*100:+.2f} 个百分点；配对 95% 区间 [{ci['low']*100:+.2f}, {ci['high']*100:+.2f}]。",'',
            '完整预测已通过独立 NumPy 稳定排序和图片 SHA 去重校验。',
            '推理不按检索结果挑主体；全量队列包含历史开发数据，并非独立留出测试。',
            '详细输入、阈值、权重/代码哈希见 config.json；回退原因见 regions_summary.json。','']
    (result/'REPORT.md').write_text('\n'.join(report))
    write_json(result.parent/'latest.json',dict(result=str(result.relative_to(dataset.root)),id=identifier,
               metrics=current,counts=manifest['counts'],verification='passed',completed_utc=now()))
    progress('complete',len(rows),len(rows),result=str(result),metrics=metrics['photo_average'])


def verify_completed(dataset, route, prepare_only=False, embedding_only=False):
    """Read the last completed run without inference or rewriting its provenance."""
    name = 'human_boxes' if route == 'manual' else 'sam3_auto'
    latest = read_json(dataset.root/'cache/foreground'/name/'latest.json')
    directory = (dataset.root/latest['bank']).parent
    spec = read_json(directory/'config.json')
    current = read_json(dataset.root/'configs/baselines.json')
    if (spec['route'] != route or not dataset.matches_encoding_inventory(spec['dataset_sha256']) or
            spec['annotation_sha256'] != dataset.annotation_sha256 or
            spec['settings'] != read_json(dataset.root/'configs/foreground.json')):
        raise ValueError('Completed foreground run uses different data, labels or settings')
    model = spec['settings']['model']
    identity = lambda c: {k:v for k,v in c.items() if k not in ('checkpoint', 'base_checkpoint')}
    if identity(spec['baselines']['models'][model]) != identity(current['models'][model]):
        raise ValueError('Completed foreground run uses a different model recipe')
    for name in ('evaluation/regions.py', 'evaluation/encoders.py', 'evaluation/scoring.py', 'evaluation/metrics.py'):
        if spec['code_sha256'][name] != file_hash(dataset.root/name):
            raise ValueError(f'Completed run uses a different implementation: {name}')
    if route == 'manual' and box_inputs(spec['annotations']) != box_inputs(manual_snapshot(dataset, spec['allow_incomplete'])):
        raise ValueError('Human boxes changed since the completed run')
    manifest = read_json(directory/'manifest.json')
    if manifest['signature'] != digest(spec) or file_hash(directory/'manifest.json') != latest['manifest_sha256']:
        raise ValueError('Completed foreground manifest was modified')
    verify_manifest(dataset, manifest)
    report = dict(status='passed', run=str(directory.relative_to(dataset.root)),
                  photos=len(dataset.photos), generated_embeddings=0)
    if not prepare_only:
        load_bank(directory, spec, manifest)
        reference = dataset.root/latest['reference_bank']
        if file_hash(reference) != latest['reference_cache_sha256']:
            raise ValueError('Reference bank changed since the completed run')
    if not prepare_only and not embedding_only:
        saved = torch.load(directory/'scores.pt', map_location='cpu', weights_only=True)
        if saved['signature'] != score_signature(spec, manifest, latest['reference_cache_sha256']):
            raise ValueError('Stale retrieval scores')
        scores = saved['scores']
        if tuple(scores.shape) != (len(dataset.photos), len(dataset.references)) or not torch.isfinite(scores).all():
            raise ValueError('Invalid foreground score matrix')
        result = dataset.root/'results/foreground'/directory.parent.name/directory.name
        predictions = [json.loads(s) for s in (result/'predictions.jsonl').read_text().splitlines()]
        report['ranking_verification'] = verify_rankings(scores, predictions, dataset)
        report['metrics'] = read_json(result/'metrics.json')['photo_average']
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('route',choices=['manual','sam3'])
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--sam-checkpoint',type=Path)
    parser.add_argument('--sam-repo',type=Path,help='Official SAM3 source tree when using sam3.pt')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--cache-only',action='store_true',help='Verify and report the last complete run; no inference or result changes')
    parser.add_argument('--allow-incomplete',action='store_true',help='Explicit partial annotation diagnostic; pending rows fall back to original embeddings')
    stages=parser.add_mutually_exclusive_group()
    stages.add_argument('--prepare-only',action='store_true')
    stages.add_argument('--embedding-only',action='store_true',help='Build crops and the embedding bank without retrieval scoring')
    args=parser.parse_args()
    torch.set_num_threads(4)
    dataset=Dataset(args.root);dataset.verify_images('photos')
    if args.cache_only:
        print(json.dumps(verify_completed(dataset, args.route, args.prepare_only, args.embedding_only), indent=2))
        return
    spec=dict(route=args.route,settings=read_json(dataset.root/'configs/foreground.json'),
        dataset_sha256=dataset.hashes,annotation_sha256=dataset.annotation_sha256,device=args.device,
        baselines=read_json(dataset.root/'configs/baselines.json'),
        code_sha256={name:file_hash(dataset.root/name) for name in ['evaluation/foreground.py','evaluation/regions.py',
             'evaluation/scoring.py','evaluation/encoders.py','evaluation/metrics.py']},
        environment={name:importlib.metadata.version(name) for name in ['torch','transformers','Pillow','numpy']})
    if args.route=='manual':
        spec['annotations']=manual_snapshot(dataset,args.allow_incomplete)
        spec['allow_incomplete']=args.allow_incomplete
    else:
        if args.sam_checkpoint is None:
            parser.error('sam3 requires --sam-checkpoint with authorized local weights')
        spec['sam']=checkpoint_spec(args.sam_checkpoint)
        if args.sam_repo:
            spec['sam_repo']=str(args.sam_repo.resolve())
            spec['sam_source_sha256']={str(p.relative_to(args.sam_repo)):file_hash(p)
                                      for p in sorted(args.sam_repo.rglob('*.py')) if '.git' not in p.parts}
    route='human_boxes' if args.route=='manual' else 'sam3_auto'
    directory=dataset.root/'cache/foreground'/route/digest(spec)[:16];directory.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    def progress(stage,done,total,**extra):
        state=dict(stage=stage,done=done,total=total,pid=os.getpid(),updated_at=now(),
                   elapsed_seconds=time.monotonic()-started,run_directory=str(directory),**extra)
        write_json(directory/'progress.json',state)
        if stage in ('complete','failed','prepared','embedded') or done%25==0 or done==total:
            print(json.dumps(state,ensure_ascii=False),flush=True)
    with (directory/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(directory/'config.json',spec)
        try:
            manifest=prepare(dataset,directory,spec,progress)
            if args.prepare_only:
                progress('prepared',len(dataset.photos),len(dataset.photos),counts=manifest['counts'])
            else:
                retrieve(dataset,directory,spec,manifest,progress,embedding_only=args.embedding_only)
        except Exception as error:
            previous=read_json(directory/'progress.json') if (directory/'progress.json').exists() else {}
            progress('failed',previous.get('done',0),len(dataset.photos),failed_stage=previous.get('stage'),
                     error=f'{type(error).__name__}: {error}')
            raise


if __name__=='__main__':
    main()
