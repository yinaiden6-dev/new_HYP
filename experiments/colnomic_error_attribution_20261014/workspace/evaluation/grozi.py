"""Frozen retrieval baselines for GroZi's fixed, annotated crop subset."""
import argparse
import copy
import csv
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .cache import save_tensor, valid_embedding
from .data import digest, file_hash, read_json, write_json
from .dense import DenseEncoder, DENSE_DIMENSIONS
from .encoders import Encoder
from .milk10k import GenericQwenEncoder
from .scoring import bm25_scores, fuse, multivector_scores, tokenize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'other_datasets/GroZi-3.2k'
MODELS = ('colpali', 'colnomic3b', 'colnomic7b', 'Qwen3-VL-Embedding-2B', 'dinov3', 'siglip2')
PROTOCOL = 'grozi.os2d.val-new-cl.nondifficult468.gallery1063.v1'
IDS = {('bm25', 'bm25'): 'B05'}
for i, model in enumerate(MODELS[:4]):
    for mode, offset in [('image', 1), ('ocr_to_image', 6), ('fusion', 10)]:
        IDS[model, mode] = f'B{offset+i:02d}'
    if i < 3:
        IDS[model, 'image_crisp'] = f'B{14+i:02d}'
        IDS[model, 'ocr_to_image_crisp'] = f'B{17+i:02d}'
        IDS[model, 'ocr_to_ocr'] = f'B{24+i:02d}'
        IDS[model, 'ocr_fusion'] = f'B{27+i:02d}'
IDS.update({('Qwen3-VL-Embedding-2B', 'ocr_to_ocr'): 'B20', ('dinov3', 'image'): 'B21',
            ('siglip2', 'image'): 'B22', ('Qwen3-VL-Embedding-2B', 'ocr_fusion'): 'B23'})
for i, model in enumerate(MODELS):
    IDS[model, 'bm25_image_fusion'] = f'F{i+1:02d}'


def sources(names):
    return {name: file_hash(ROOT / name) for name in names}


def inventory(data, verify_images=True):
    metadata = read_json(data / 'retrieval_subset/metadata.json')
    assert metadata['counts']['photos'] == 468 and metadata['counts']['references'] == 1063
    assert metadata['protocol']['query_filter'] == {'split': 'val-new-cl', 'difficult': 0}
    for side, key in [('photos', 'photo_id'), ('references', 'reference_id')]:
        rows = metadata[side]
        assert [r['index'] for r in rows] == list(range(len(rows)))
        assert len({r[key] for r in rows}) == len(rows)
        if verify_images:
            for row in rows:
                assert file_hash(data / 'retrieval_subset' / row['image_path']) == row['sha256']
    for row in metadata['photos']:
        reference = metadata['references'][row['positive_reference_index']]
        assert row['positive_reference_id'] == reference['reference_id'] and row['class_id'] == reference['class_id']
    return metadata


def prepare(data):
    metadata = inventory(data)
    config_path = data / 'baseline_config.json'
    if config_path.exists():
        raise FileExistsError(config_path)
    base = read_json(ROOT / 'other_datasets/MILK10K/baseline_config.json')['models']
    models = {name: copy.deepcopy(base[name]) for name in MODELS}
    for name, spec in models.items():
        spec['image_contract']['input'] = 'provided GroZi product crop or independent reference RGB image'
        if name == 'Qwen3-VL-Embedding-2B':
            spec['image_contract']['recipe'] = 'grozi.qwen_generic_image.v1'
    config = dict(protocol=PROTOCOL, models=models, alpha=0.5, bm25=dict(k1=1.2, b=0.75),
        training='none; frozen local checkpoints', parameter_selection='none on this validation subset',
        query_batch=1, reference_batch=8, token_block=256, dense_batch=8,
        ocr=read_json(ROOT / 'configs/ppocrv6.json'),
        planned_results=[dict(id=v, model=k[0], mode=k[1]) for k, v in sorted(IDS.items(), key=lambda x:x[1])],
        dataset_sha256=file_hash(data / 'retrieval_subset/metadata.json'),
        tie_break='ascending original reference class_id; no reference deduplication',
        score_empty_query='failure in OCR-only modes; image fallback in fusion',
        score_empty_reference='not a retrievable positive in OCR-to-OCR/BM25; neural empty reference score=-2',
        source_files=sources(['evaluation/grozi.py', 'evaluation/grozi_ocr.py', 'evaluation/encoders.py',
            'evaluation/dense.py', 'evaluation/milk10k.py', 'evaluation/scoring.py', 'evaluation/ocr.py']))
    write_json(config_path, config)
    snapshot = data / 'cache/intermediate/source_snapshot'
    for relative in config['source_files']:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    write_json(data / 'cache/manifest.json', dict(protocol=PROTOCOL, dataset_sha256=config['dataset_sha256'],
        photos=metadata['photos'], references=metadata['references'], configuration_sha256=file_hash(config_path)))
    print(f'Prepared {len(IDS)} baseline configurations', flush=True)


def load_ocr(data, side, metadata):
    path = data / 'cache/ocr' / f'{side}.json'
    if not path.exists():
        raise FileNotFoundError(f'OCR is not ready: {path}')
    bank = read_json(path)
    rows = metadata['photos' if side == 'photo' else 'references']
    key = 'photo_id' if side == 'photo' else 'reference_id'
    assert bank['complete'] and bank['ids'] == [r[key] for r in rows]
    assert bank['configuration']['config'] == read_json(data / 'baseline_config.json')['ocr']
    assert bank['configuration']['implementation'] == file_hash(ROOT / 'evaluation/ocr.py')
    for row in rows:
        assert bank['entries'][row[key]]['image_sha256'] == row['sha256']
        assert bank['entries'][row[key]]['image_path'] == 'retrieval_subset/' + row['image_path']
    return bank, [bank['entries'][r[key]]['text'] for r in rows]


def encode(data, model, device, metadata, config, cache_only=False):
    spec = config['models'][model]
    encoder = None
    banks = {}
    source = sources(['evaluation/encoders.py', 'evaluation/dense.py', 'evaluation/milk10k.py'])
    views = ('image', 'text') if model in MODELS[:4] else ('image',)
    try:
        for view in views:
            for side, population, key in [('photo', 'photos', 'photo_id'), ('ref', 'references', 'reference_id')]:
                rows = metadata[population]
                texts = load_ocr(data, side, metadata)[1] if view == 'text' else None
                recipe = dict(model=spec, view=view, source=source,
                    inputs=[(r[key], r['sha256']) for r in rows],
                    text_sha256=digest(texts) if texts is not None else None)
                signature = digest(recipe)
                folder = 'image_embeddings' if view == 'image' else 'ocr_embeddings'
                path = data / 'cache' / folder / model / f'{side}.pt'
                if path.exists():
                    bank = torch.load(path, map_location='cpu', weights_only=True)
                    assert bank['signature'] == signature
                    assert bank['ids'] == [r[key] for r in rows]
                    assert len(bank['embeddings']) == len(rows)
                    assert all(valid_embedding(v, model, view, has_text=h, tokens=t)
                        for v,h,t in zip(bank['embeddings'], bank['has_text'], bank['tokens'], strict=True))
                    banks[view, side] = bank
                    print(f'Cached {model}/{view}/{side}: {len(rows)}', flush=True)
                    continue
                if cache_only:
                    raise FileNotFoundError(path)
                partial_path = data / 'cache/intermediate' / model / f'{view}_{side}.partial.pt'
                partial = torch.load(partial_path, map_location='cpu', weights_only=True) if partial_path.exists() else {}
                assert not partial or partial['signature'] == signature
                values = partial.get('embeddings', [])
                token_counts = partial.get('tokens', [])
                elapsed = partial.get('encoding_seconds', 0.)
                has_text = [bool(t.strip()) for t in texts] if texts is not None else [True] * len(rows)
                assert len(values) == len(token_counts)
                assert all(valid_embedding(v, model, view, has_text=has_text[i], tokens=token_counts[i]) for i,v in enumerate(values))
                if encoder is None:
                    started = time.monotonic()
                    factory = DenseEncoder if model in ('dinov3', 'siglip2') else (
                        GenericQwenEncoder if model == 'Qwen3-VL-Embedding-2B' else Encoder)
                    encoder = factory(model, spec, ROOT, device)
                    loading = time.monotonic() - started
                    write_json(data / 'cache/intermediate' / model / 'runtime.json', dict(
                        model_loading_seconds=loading, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                        hardware=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                        packages={k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'numpy', 'Pillow')}))
                    print(f'Loaded {model}: {loading:.1f}s', flush=True)
                batch_size = config['dense_batch'] if model in ('dinov3', 'siglip2') and view == 'image' else 1
                while len(values) < len(rows):
                    first = len(values)
                    batch = rows[first:first+batch_size]
                    start = time.monotonic()
                    if view == 'text':
                        value, count = encoder.encode_text(texts[first])
                        if model == 'Qwen3-VL-Embedding-2B' and has_text[first]:
                            value = torch.nn.functional.normalize(value.float(), dim=-1)
                        new, counts = [value], [count]
                    else:
                        paths = [data / 'retrieval_subset' / r['image_path'] for r in batch]
                        new = encoder.encode_images(paths) if batch_size > 1 else [encoder.encode_image(paths[0])]
                        counts = [v.shape[0] if v.ndim == 2 else 1 for v in new]
                    elapsed += time.monotonic() - start
                    assert len(new) == len(batch)
                    for i,(value,count) in enumerate(zip(new, counts, strict=True)):
                        assert valid_embedding(value, model, view, has_text=has_text[first+i], tokens=count)
                    values.extend(new); token_counts.extend(counts)
                    if len(values) % 32 == 0 or len(values) == len(rows):
                        save_tensor(partial_path, dict(signature=signature, embeddings=values,
                            tokens=token_counts, encoding_seconds=elapsed))
                        print(f'{model}/{view}/{side}: {len(values)}/{len(rows)} ({elapsed:.1f}s)', flush=True)
                bank = dict(schema='grozi.embeddings.v1', signature=signature, recipe=recipe,
                    model=model, view=view, population=side, ids=[r[key] for r in rows],
                    image_sha256=[r['sha256'] for r in rows], image_paths=[r['image_path'] for r in rows],
                    class_ids=[r['class_id'] for r in rows], has_text=has_text, tokens=token_counts,
                    encoding_seconds=elapsed, embeddings=torch.stack(values) if model in DENSE_DIMENSIONS else values)
                save_tensor(path, bank)
                partial_path.unlink(missing_ok=True)
                banks[view, side] = bank
    finally:
        if encoder is not None:
            encoder.release()
        gc.collect()
        torch.cuda.empty_cache()
    return banks


def metrics(scores, metadata, query_valid=None, reference_valid=None):
    a = scores.detach().cpu().numpy()
    photos, references = metadata['photos'], metadata['references']
    assert a.shape == (len(photos), len(references)) and np.isfinite(a).all()
    query_valid = [True]*len(photos) if query_valid is None else query_valid
    reference_valid = [True]*len(references) if reference_valid is None else reference_valid
    predictions, ranks = [], []
    for i, row in enumerate(photos):
        correct = row['positive_reference_index']
        order = np.argsort(-a[i], kind='stable')
        valid = query_valid[i] and reference_valid[correct]
        rank = int(np.flatnonzero(order == correct)[0])+1 if valid else None
        ranks.append(rank if rank is not None else np.inf)
        predictions.append(dict(photo_id=row['photo_id'], class_id=row['class_id'], source_image_id=row['source_image_id'],
            positive_reference_id=row['positive_reference_id'], rank=rank,
            failure=None if valid else ('empty_query_ocr' if not query_valid[i] else 'empty_positive_reference_ocr'),
            top10_ids=[references[j]['reference_id'] for j in order[:10]] if query_valid[i] else [],
            top10_scores=[float(a[i,j]) for j in order[:10]] if query_valid[i] else []))
    ranks = np.array(ranks)
    values = np.column_stack([ranks<=1, ranks<=5, ranks<=10, 1/ranks])
    names = ['R@1', 'R@5', 'R@10', 'MRR']
    report = dict(queries=len(photos), references=len(references), **dict(zip(names, values.mean(0).tolist())),
        correct_at_1=int((ranks == 1).sum()), failures=int((~np.isfinite(ranks)).sum()))
    class_ids = np.array([p['class_id'] for p in photos])
    report['class_average'] = dict(zip(names, np.stack([values[class_ids == c].mean(0) for c in np.unique(class_ids)]).mean(0).tolist()))
    scenes = np.array([p['source_image_id'] for p in photos])
    groups = np.unique(scenes)
    totals = np.stack([values[scenes == s].sum(0) for s in groups])
    counts = np.array([(scenes == s).sum() for s in groups])
    weights = np.random.default_rng(20260924).multinomial(len(groups), np.ones(len(groups))/len(groups), size=10000)
    bounds = np.quantile((weights @ totals) / (weights @ counts)[:,None], [.025, .975], axis=0)
    report['confidence_intervals'] = dict(unit='source shelf scene', groups=len(groups), repetitions=10000,
        seed=20260924, percentile=95, metrics={n:dict(low=float(bounds[0,i]), high=float(bounds[1,i])) for i,n in enumerate(names)})
    return report, predictions


def write_result(data, model, mode, value, metadata, query_valid=None, reference_valid=None):
    report, predictions = metrics(value['scores'], metadata, query_valid, reference_valid)
    folder = data / 'results' / model / mode
    config = read_json(data / 'baseline_config.json')
    write_json(folder / 'config.json', dict(id=IDS[model,mode], protocol=PROTOCOL, model=model, mode=mode,
        dataset_sha256=config['dataset_sha256'], model_spec=config['models'].get(model),
        score_signature=value['signature'], recipe=value['recipe'], scoring_seconds=value['compute_seconds'],
        timing_scope=value['timing_scope'], alpha=0.5 if 'fusion' in mode else None,
        query_valid=list(map(bool, query_valid)) if query_valid is not None else [True]*len(metadata['photos']),
        reference_valid=list(map(bool, reference_valid)) if reference_valid is not None else [True]*len(metadata['references'])))
    write_json(folder / 'metrics.json', report)
    tmp = folder / 'predictions.jsonl.tmp'
    tmp.write_text(''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in predictions))
    tmp.replace(folder / 'predictions.jsonl')
    print(f'{IDS[model,mode]} {model}/{mode}: R@1={report["R@1"]:.4f}', flush=True)


def score_pair(data, model, mode, query, reference, device, config):
    recipe = dict(protocol=PROTOCOL, query=query['signature'], reference=reference['signature'],
        dtype='float32', tf32=False, scorer_sha256=file_hash(ROOT / 'evaluation/scoring.py'),
        scoring='normalized dot product' if model in DENSE_DIMENSIONS else 'mean MaxSim and CRISP',
        empty_reference='neural score=-2; unavailable positive counts as failure',
        row_order='fixed subset metadata order')
    signature = digest(recipe)
    directory = data / 'cache/scores' / model
    path = directory / f'{mode}.pt'
    modes = [mode] + ([mode+'_crisp'] if model not in DENSE_DIMENSIONS and mode != 'ocr_to_ocr' else [])
    if all((directory/f'{m}.pt').exists() for m in modes):
        values = {m: torch.load(directory/f'{m}.pt', map_location='cpu', weights_only=True) for m in modes}
        assert all(v['signature'] == signature for v in values.values())
        return values
    started = time.monotonic()
    q, r = query['embeddings'], reference['embeddings']
    if model in DENSE_DIMENSIONS:
        matrices = [q.float() @ r.float().T]
        matrices[0][:, ~torch.tensor(reference['has_text'])] = -2.
        seconds = time.monotonic()-started
    else:
        # The shared scorer requires nonempty references; scatter valid columns back.
        valid_indices = [i for i, v in enumerate(r) if v.shape[0] > 0]
        nonempty = [r[i] for i in valid_indices]
        matrices = [torch.full((len(q),len(r)), -2., dtype=torch.float32) for _ in range(2)]
        seconds = 0.
        for first in range(0,len(q),32):
            part_path = data / 'cache/intermediate' / model / f'{mode}_{first:05d}.scores.pt'
            if part_path.exists():
                part = torch.load(part_path, map_location='cpu', weights_only=True)
                assert part['signature'] == signature
            else:
                start = time.monotonic()
                pair = multivector_scores(q[first:first+32], nonempty, device=device,
                    query_batch=config['query_batch'], reference_batch=config['reference_batch'], token_block=config['token_block'])
                part = dict(signature=signature, matrices=pair, seconds=time.monotonic()-start)
                save_tensor(part_path, part)
            seconds += part['seconds']
            for target, block in zip(matrices,part['matrices'],strict=True):
                target[first:first+len(block),valid_indices] = block
            print(f'{model}/{mode} scored {min(first+32,len(q))}/{len(q)} ({seconds:.1f}s)', flush=True)
    results = {}
    for m, matrix in zip(modes, matrices):
        assert matrix.shape == (468,1063) and torch.isfinite(matrix).all()
        value = dict(signature=signature, recipe=recipe, scores=matrix, query_ids=query['ids'], reference_ids=reference['ids'],
            compute_seconds=seconds, timing_scope='joint MaxSim+CRISP full-gallery scoring' if model not in DENSE_DIMENSIONS else 'CPU dense full-gallery dot product')
        save_tensor(directory/f'{m}.pt',value); results[m]=value
    for part_path in (data/'cache/intermediate'/model).glob(f'{mode}_*.scores.pt'):
        part_path.unlink()
    return results


def fusion_result(data, model, mode, image, text, has_text, metadata):
    recipe = dict(image=image['signature'], text=text['signature'], alpha=0.5,
        normalization='per-query population z-score; fixed equal weights; empty query OCR uses image score',
        scorer_sha256=file_hash(ROOT/'evaluation/scoring.py'))
    start = time.monotonic()
    scores = fuse(image['scores'], text['scores'], has_text, .5)
    value = dict(signature=digest(recipe), recipe=recipe, scores=scores,
        compute_seconds=time.monotonic()-start, timing_scope='fusion of cached scores only; excludes all upstream stages',
        query_ids=[r['photo_id'] for r in metadata['photos']], reference_ids=[r['reference_id'] for r in metadata['references']])
    save_tensor(data/'cache/scores'/model/f'{mode}.pt',value)
    write_result(data,model,mode,value,metadata)


def run_model(data, model, device, cache_only=False):
    metadata = inventory(data)
    config = read_json(data/'baseline_config.json')
    assert config['dataset_sha256'] == file_hash(data/'retrieval_subset/metadata.json')
    banks = encode(data,model,device,metadata,config,cache_only)
    values = score_pair(data,model,'image',banks['image','photo'],banks['image','ref'],device,config)
    for mode,value in values.items():
        write_result(data,model,mode,value,metadata)
    if model in MODELS[:4]:
        for mode, reference_view in [('ocr_to_image','image'),('ocr_to_ocr','text')]:
            pair = score_pair(data,model,mode,banks['text','photo'],banks[reference_view,'ref'],device,config)
            values.update(pair)
            for kind,value in pair.items():
                write_result(data,model,kind,value,metadata,banks['text','photo']['has_text'],
                    banks['text','ref']['has_text'] if reference_view=='text' else None)
        fusion_result(data,model,'fusion',values['image'],values['ocr_to_image'],banks['text','photo']['has_text'],metadata)
        fusion_result(data,model,'ocr_fusion',values['image'],values['ocr_to_ocr'],banks['text','photo']['has_text'],metadata)


def run_bm25(data):
    metadata = inventory(data)
    qbank, qt = load_ocr(data,'photo',metadata)
    rbank, rt = load_ocr(data,'ref',metadata)
    recipe = dict(query_ocr=file_hash(data/'cache/ocr/photo.json'), reference_ocr=file_hash(data/'cache/ocr/ref.json'),
        k1=1.2,b=.75,tokenization='Unicode letters/digits; lowercase; query term frequency',
        scorer_sha256=file_hash(ROOT/'evaluation/scoring.py'),protocol=PROTOCOL)
    path = data/'cache/scores/bm25/bm25.pt'
    if path.exists():
        value = torch.load(path,map_location='cpu',weights_only=True)
        assert value['signature'] == digest(recipe)
    else:
        start=time.monotonic();scores=bm25_scores(qt,rt)
        value=dict(signature=digest(recipe),recipe=recipe,scores=scores,compute_seconds=time.monotonic()-start,
            timing_scope='CPU BM25 indexing and full-gallery scoring; excludes OCR',
            query_ids=qbank['ids'],reference_ids=rbank['ids'])
        save_tensor(path,value)
    write_result(data,'bm25','bm25',value,metadata,[bool(tokenize(t)) for t in qt],[bool(tokenize(t)) for t in rt])


def finalize(data):
    metadata=inventory(data)
    run_bm25(data)
    bm=torch.load(data/'cache/scores/bm25/bm25.pt',map_location='cpu',weights_only=True)
    has_text=[bool(tokenize(t)) for t in load_ocr(data,'photo',metadata)[1]]
    for model in MODELS:
        path=data/'cache/scores'/model/'image.pt'
        if path.exists():
            image=torch.load(path,map_location='cpu',weights_only=True)
            fusion_result(data,model,'bm25_image_fusion',image,bm,has_text,metadata)
    summarize(data)


def summarize(data):
    rows=[]
    for (model,mode),identifier in sorted(IDS.items(),key=lambda item:item[1]):
        path=data/'results'/model/mode/'metrics.json'
        if path.exists():
            result=read_json(path)
            rows.append(dict(id=identifier,model=model,mode=mode,**{k:result[k] for k in ['queries','references','R@1','R@5','R@10','MRR','correct_at_1','failures']}))
    directory=data/'results';directory.mkdir(exist_ok=True)
    if rows:
        with (directory/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    timing=[]
    for model in MODELS:
        for view,folder in [('image','image_embeddings'),('text','ocr_embeddings')]:
            for side in ('photo','ref'):
                path=data/'cache'/folder/model/f'{side}.pt'
                if path.exists():
                    bank=torch.load(path,map_location='cpu',weights_only=True)
                    timing.append(dict(model=model,input=view,population=side,images=len(bank['ids']),seconds=bank['encoding_seconds'],seconds_per_image=bank['encoding_seconds']/len(bank['ids'])))
    for side in ('photo','ref'):
        path=data/'cache/ocr'/f'{side}.json'
        if path.exists():
            bank=read_json(path)
            timing.append(dict(model='PP-OCRv6',input='ocr',population=side,images=len(bank['entries']),seconds=bank['inference_seconds'],seconds_per_image=bank['inference_seconds']/len(bank['entries'])))
    if timing:
        with (directory/'timing.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(timing[0]));writer.writeheader();writer.writerows(timing)
    lines=['# GroZi-3.2k 商品裁剪检索基础实验','',
        '468 个非难例查询（161 款商品、78 张货架场景）→ 全部 1,063 张参考图。',
        '使用 OS2D 官方 val-new-cl 划分的项目自建检索协议；不是官方检测指标或官方测试集成绩。',
        '冻结模型；参数不在本子集选择。融合固定为逐查询 z-score 后各占 0.5，空查询 OCR 回退到图像检索。','',
        '| ID | 模型 | 模式 | R@1 | R@5 | R@10 | MRR |','|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['id']} | {row['model']} | {row['mode']} | {100*row['R@1']:.2f} | {100*row['R@5']:.2f} | {100*row['R@10']:.2f} | {row['MRR']:.4f} |")
    lines+=['',f'已完成 {len(rows)}/{len(IDS)} 项配置。','',
        'R@K 单位为百分比；主表对查询取平均，逐项 metrics.json 同时给出商品类别平均和按货架场景分组的 95% bootstrap 区间。',
        'Qwen 图像与文字均使用官方通用提示，并对输出做 FP32 L2 归一化。DINOv3 使用 FP32。',
        '多向量检索保留编码器有效 token，剔除 padding/零向量；采用 FP32 mean MaxSim。CRISP 是既有评分对照。',
        'OCR-only 的空查询或缺失正确参考 OCR 计为失败，保留在 468 个查询分母中。同分按固定参考图顺序排序。',
        'OCR 使用同一 PP-OCRv6 medium 配方，四方向识别及去重；不读取类别名称、路径或框标签作为模型输入。','',
        'timing.csv 是特征提取/OCR 阶段墙钟耗时，包含图像预处理、排除权重加载。各配置的评分耗时保存在 config.json。',
        'MaxSim 与 CRISP 同时计算，记录的是两者联合耗时；融合耗时仅指已有分数的合并，不能当作端到端检索延迟。',
        '任务可能并行运行，速度表是本次实测阶段耗时，不是独占 GPU 的受控速度比较。','',
        '缓存：../cache/image_embeddings、../cache/ocr_embeddings、../cache/ocr、../cache/scores；均按 photo/ref 分开。',
        '复现配置：../baseline_config.json；代码快照与日志：../cache/intermediate/。']
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(f'Summary: {len(rows)}/{len(IDS)}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--device',default='cuda:0')
    for flag in ('prepare','bm25','finalize','summarize','cache-only'):
        parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.prepare: prepare(args.data)
    if args.model:
        path=args.data/'cache/intermediate'/f'{args.model}.lock';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run_model(args.data,args.model,args.device,args.cache_only)
    if args.bm25: run_bm25(args.data)
    if args.finalize: finalize(args.data)
    elif args.summarize: summarize(args.data)


if __name__=='__main__':
    main()
