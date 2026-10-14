"""Frozen image retrieval on the official MILK10k test pairs only."""
import argparse
import csv
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .cache import save_tensor
from .data import digest, file_hash, read_csv, read_json, write_json
from .encoders import Encoder
from .dense import DenseEncoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'other_datasets/MILK10K'
MODEL_ORDER = ('colpali', 'colnomic3b', 'colnomic7b', 'Qwen3-VL-Embedding-2B',
               'siglip2', 'dinov3', 'biomedclip')
DIMENSIONS = {'Qwen3-VL-Embedding-2B': 2048, 'siglip2': 1024,
              'dinov3': 1024, 'biomedclip': 512}
NAMES = {'colpali': 'ColPali', 'colnomic3b': 'ColNomic 3B', 'colnomic7b': 'ColNomic 7B',
         'Qwen3-VL-Embedding-2B': 'Qwen3-VL-Embedding-2B', 'siglip2': 'SigLIP2 Large/16-256',
         'dinov3': 'DINOv3 ViT-L/16', 'biomedclip': 'BiomedCLIP'}
PROTOCOL = 'MILK10k.test.clinical_to_dermoscopic.479.v1'


def test_pairs(data, expected_count=479):
    """Join by lesion ID, never by filename or a hidden diagnostic label."""
    metadata = read_csv(data / 'MILK10k_Test_Metadata.csv')
    by_image = {r['isic_id']: r for r in metadata}
    if len(by_image) != len(metadata):
        raise ValueError('Duplicate test image ID')
    rows = sorted((r for r in read_csv(data / 'retrieval_pairs.csv') if r['split'] == 'test'),
                  key=lambda r: r['lesion_id'])
    if len(rows) != expected_count or len({r['lesion_id'] for r in rows}) != len(rows):
        raise ValueError('Unexpected or duplicate test lesions')
    result = []
    used = set()
    pixels = set()
    for row in rows:
        item = {'lesion_id': row['lesion_id']}
        for side, kind in (('clinical', 'clinical: close-up'), ('dermoscopic', 'dermoscopic')):
            relative = Path(row[side + '_image'])
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'MILK10k_Test_Input':
                raise ValueError('A retrieval image is outside the official test directory')
            path = data / relative
            image_id = path.stem
            meta = by_image.get(image_id)
            if not meta or meta['lesion_id'] != row['lesion_id'] or meta['image_type'] != kind:
                raise ValueError('Test metadata and pair manifest disagree')
            if image_id in used:
                raise ValueError('Image occurs in multiple pairs or modalities')
            used.add(image_id)
            with Image.open(path) as im:
                rgb = im.convert('RGB')
                import hashlib
                pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                dimensions = list(rgb.size)
            if pixel_hash in pixels:
                raise ValueError('Duplicate decoded test image; inspect positives before scoring')
            pixels.add(pixel_hash)
            item[side] = dict(image_id=image_id, path=str(relative), sha256=file_hash(path),
                              pixel_sha256=pixel_hash, dimensions=dimensions)
        result.append(item)
    if used != set(by_image):
        raise ValueError('Pair list does not cover all test images')
    return result


def source_signature():
    return {name: file_hash(ROOT / name) for name in
            ('evaluation/milk10k.py', 'evaluation/encoders.py', 'evaluation/dense.py')}


def prepare(data):
    pairs = test_pairs(data)
    manifest = dict(schema='milk10k.test_manifest.v1', protocol=PROTOCOL, pairs=pairs,
                    query_count=len(pairs), gallery_count=len(pairs),
                    input='image pixels only; no OCR, diagnosis, patient metadata or identifiers',
                    direction='clinical close-up -> dermoscopic',
                    positive='same lesion_id; exactly one positive per query',
                    training='none; official training images excluded',
                    parameter_selection='none on this test set',
                    official_task='diagnosis; this image-retrieval protocol is a project adaptation',
                    patient_independence='not established; patient IDs are unavailable',
                    tie_break='ascending gallery lesion_id',
                    source_metadata_sha256=file_hash(data / 'MILK10k_Test_Metadata.csv'))
    manifest['signature'] = digest(manifest)
    write_json(data / 'cache/test_manifest.json', manifest)
    config_path = data / 'baseline_config.json'
    if not config_path.exists():
        base = read_json(ROOT / 'configs/baselines.json')['models']
        models = {k: base[k] for k in MODEL_ORDER[:6]}
        models['dinov3']['image_contract'].update(dtype='torch.float32', processor_backend='torchvision')
        models['Qwen3-VL-Embedding-2B']['image_instruction'] = "Represent the user's input."
        models['Qwen3-VL-Embedding-2B']['image_contract'].update(recipe='milk10k.qwen_generic_image.v1',
            instruction="Represent the user's input.", pooling='official last attended token; FP32 L2 renormalization',
            output_dtype='torch.float32')
        models['biomedclip'] = dict(family='biomedclip', checkpoint='weights/biomedclip',
            image_model=dict(repo='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                revision='9f341de24bfb00180f1b847274256e9b65a3a32e'),
            image_contract=dict(input='official OpenCLIP evaluation RGB image preprocessing',
                dtype='torch.float32', pooling='official visual projection; FP32 L2 normalization'))
        write_json(config_path, dict(protocol=PROTOCOL, models=models, score='FP32 MaxSim or normalized dot product',
                                    multivector_query_batch=1, reference_batch=8, dense_batch=8))
    return manifest


class GenericQwenEncoder(Encoder):
    @torch.inference_mode()
    def encode_image(self, path):
        from qwen_vl_utils import process_vision_info
        with Image.open(path) as source:
            image = source.convert('RGB')
        instruction = self.config['image_instruction']
        if instruction != self.embedder.default_instruction:
            raise ValueError('Unexpected Qwen generic instruction')
        conversation = [self.embedder.format_model_input(image=image, instruction=instruction)]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        images, videos, kwargs = process_vision_info(conversation, image_patch_size=16,
            return_video_metadata=True, return_video_kwargs=True)
        if images is None or videos is not None:
            raise ValueError('Invalid image input')
        inputs = self.processor(text=prompt, images=images, truncation=True,
            max_length=self.embedder.max_length, padding=True, do_resize=False, return_tensors='pt', **kwargs)
        return F.normalize(self.encode_prepared(inputs)[0][0].float(), dim=-1)


class AuxiliaryEncoder:
    def __init__(self, name, spec, device):
        self.name, self.device = name, device
        path = ROOT / spec['checkpoint']
        if name == 'biomedclip':
            import open_clip
            from open_clip.factory import _MODEL_CONFIGS
            config = read_json(path / 'open_clip_config.json')
            _MODEL_CONFIGS['milk10k_biomedclip'] = config['model_cfg']
            self.model, _, self.processor = open_clip.create_model_and_transforms(
                'milk10k_biomedclip', pretrained=str(path / 'open_clip_pytorch_model.bin'),
                **{f'image_{k}': v for k, v in config['preprocess_cfg'].items()})
        else:
            raise ValueError(name)
        self.model.to(device).eval()

    @torch.inference_mode()
    def encode_images(self, paths):
        images = []
        for path in paths:
            with Image.open(path) as im:
                images.append(im.convert('RGB'))
        inputs = torch.stack([self.processor(im) for im in images]).to(self.device)
        vectors = self.model.encode_image(inputs, normalize=False)
        return list(F.normalize(vectors.float(), dim=-1).cpu().unbind())

    def release(self):
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()


def valid(value, model):
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        return False
    if model in DIMENSIONS:
        return (value.shape == (DIMENSIONS[model],) and value.dtype == torch.float32
                and abs(value.norm().item() - 1) < .03)
    if value.ndim != 2 or value.shape[1] != 128 or value.dtype != torch.float16:
        return False
    norms = value.float().norm(dim=-1)
    norms = norms[norms > 1e-8]
    return bool(len(norms) and ((norms - 1).abs() < .03).all())


def encode_banks(data, model, spec, manifest, device, cache_only):
    cache = data / 'cache'
    source = source_signature()
    encoder = None
    banks = {}
    load_seconds = 0.
    wall_start = time.monotonic()
    try:
        for side in ('clinical', 'dermoscopic'):
            rows = [dict(p[side], lesion_id=p['lesion_id']) for p in manifest['pairs']]
            signature = digest(dict(rows=rows, model=spec, source=source))
            path = cache / 'image_embeddings' / model / (side + '.pt')
            if path.exists():
                bank = torch.load(path, map_location='cpu', weights_only=True)
                if bank.get('signature') != signature or len(bank['embeddings']) != len(rows):
                    raise ValueError(f'Stale image bank: {path}')
                if not all(valid(v, model) for v in bank['embeddings']):
                    raise ValueError(f'Invalid embeddings: {path}')
                banks[side] = bank
                print(f'{model}/{side}: verified cached {len(rows)} images', flush=True)
                continue
            if cache_only:
                raise FileNotFoundError(path)
            partial_path = cache / 'intermediate' / model / (side + '.partial.pt')
            partial = torch.load(partial_path, map_location='cpu', weights_only=True) if partial_path.exists() else {}
            if partial and partial.get('signature') != signature:
                raise ValueError(f'Stale partial bank: {partial_path}')
            values = partial.get('embeddings', [])
            seconds = partial.get('encoding_seconds', 0.)
            if not all(valid(v, model) for v in values):
                raise ValueError('Invalid partial embeddings')
            if encoder is None:
                start = time.monotonic()
                if model == 'biomedclip':
                    encoder = AuxiliaryEncoder(model, spec, device)
                elif model in ('siglip2', 'dinov3'):
                    encoder = DenseEncoder(model, spec, ROOT, device)
                else:
                    factory = GenericQwenEncoder if model == 'Qwen3-VL-Embedding-2B' else Encoder
                    encoder = factory(model, spec, ROOT, device)
                load_seconds += time.monotonic() - start
                print(f'{model}: loaded in {load_seconds:.1f}s', flush=True)
            batch_size = 8 if hasattr(encoder, 'encode_images') else 1
            while len(values) < len(rows):
                part = rows[len(values):len(values) + batch_size]
                started = time.monotonic()
                paths = [data / r['path'] for r in part]
                encoded = encoder.encode_images(paths) if batch_size > 1 else [encoder.encode_image(paths[0])]
                seconds += time.monotonic() - started
                if len(encoded) != len(part) or not all(valid(v, model) for v in encoded):
                    raise ValueError('Non-finite or malformed image embeddings')
                values.extend(encoded)
                if len(values) % 16 == 0 or len(values) == len(rows):
                    save_tensor(partial_path, dict(signature=signature, embeddings=values, encoding_seconds=seconds))
                    print(f'{model}/{side}: {len(values)}/{len(rows)}, encoding {seconds:.1f}s', flush=True)
            bank = dict(schema='milk10k.image_embeddings.v1', protocol=PROTOCOL, population=side,
                model=spec['image_model'], contract=spec, signature=signature, source=source,
                ids=[r['image_id'] for r in rows], lesion_ids=[r['lesion_id'] for r in rows],
                image_paths=[r['path'] for r in rows], image_sha256=[r['sha256'] for r in rows],
                embeddings=torch.stack(values) if model in DIMENSIONS else values,
                encoding_seconds=seconds)
            save_tensor(path, bank)
            partial_path.unlink(missing_ok=True)
            banks[side] = bank
    finally:
        if encoder is not None:
            encoder.release()
            del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return banks, dict(model_loading_seconds=load_seconds, bank_stage_wall_seconds=time.monotonic() - wall_start)


@torch.inference_mode()
def maxsim(queries, references, device='cuda:0', reference_batch=8, progress=None):
    """FP32 mean MaxSim, with zero/padding tokens excluded on both sides."""
    qs = [v[v.float().norm(dim=-1) > 1e-8].float() for v in queries]
    rs = [v[v.float().norm(dim=-1) > 1e-8].float() for v in references]
    if any(not len(v) for v in qs + rs):
        raise ValueError('No effective image tokens')
    output = torch.empty(len(qs), len(rs), dtype=torch.float32)
    for ri in range(0, len(rs), reference_batch):
        refs = rs[ri:ri + reference_batch]
        lengths = torch.tensor([len(r) for r in refs], device=device)
        ref = torch.nn.utils.rnn.pad_sequence(refs, batch_first=True).to(device)
        mask = torch.arange(ref.shape[1], device=device)[None] < lengths[:, None]
        flat = ref.flatten(0, 1).T.contiguous()
        for qi, query in enumerate(qs):
            # Blocking the query axis bounds memory even for large token grids.
            total = torch.zeros(len(refs), device=device)
            for offset in range(0, len(query), 256):
                block = query[offset:offset + 256].to(device)
                sim = (block @ flat).reshape(len(block), len(refs), ref.shape[1])
                total += sim.masked_fill(~mask[None], -torch.inf).amax(-1).sum(0)
            output[qi, ri:ri + len(refs)] = (total / len(query)).cpu()
        if progress:
            progress(min(ri + len(refs), len(rs)), len(rs))
    return output


def metrics_and_predictions(scores, query_ids, gallery_ids, query_lesions, gallery_lesions):
    a = np.asarray(scores, dtype=np.float32)
    if a.shape != (len(query_ids), len(gallery_ids)) or not np.isfinite(a).all():
        raise ValueError('Invalid score matrix')
    lookup = {v: i for i, v in enumerate(gallery_lesions)}
    if len(lookup) != len(gallery_lesions) or set(query_lesions) != set(gallery_lesions):
        raise ValueError('Expected one unique positive per query')
    predictions = []
    ranks = []
    for i, lesion in enumerate(query_lesions):
        order = np.argsort(-a[i], kind='stable')
        correct = lookup[lesion]
        rank = int(np.flatnonzero(order == correct)[0]) + 1
        ranks.append(rank)
        predictions.append(dict(query_id=query_ids[i], lesion_id=lesion,
            positive_id=gallery_ids[correct], rank=rank, top1_id=gallery_ids[order[0]],
            top10_ids=[gallery_ids[j] for j in order[:10]],
            top10_scores=[float(a[i, j]) for j in order[:10]]))
    r = np.asarray(ranks)
    metrics = dict(queries=len(r), gallery=len(gallery_ids),
        **{f'R@{k}': float((r <= k).mean()) for k in (1, 5, 10)},
        MRR=float((1. / r).mean()), correct_at_1=int((r == 1).sum()),
        mean_rank=float(r.mean()), median_rank=float(np.median(r)))
    return metrics, predictions


def evaluate(data, model, device, cache_only=False):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = read_json(data / 'baseline_config.json')
    manifest = read_json(data / 'cache/test_manifest.json')
    # Recheck the image inventory on every run before accepting any old cache.
    if test_pairs(data) != manifest['pairs']:
        raise ValueError('Test image inventory changed')
    spec = config['models'][model]
    banks, timing = encode_banks(data, model, spec, manifest, device, cache_only)
    q, r = banks['clinical'], banks['dermoscopic']
    score_path = data / 'cache/scores' / (model + '.pt')
    recipe = dict(query_signature=q['signature'], gallery_signature=r['signature'],
        scoring='normalized_dot_fp32' if model in DIMENSIONS else 'mean_maxsim_fp32_no_padding',
        source=source_signature())
    if score_path.exists():
        cached = torch.load(score_path, map_location='cpu', weights_only=True)
        if cached['recipe'] != recipe:
            raise ValueError('Stale score matrix')
        scores = cached['scores']
        score_seconds = cached['scoring_seconds']
    else:
        started = time.monotonic()
        if model in DIMENSIONS:
            scores = q['embeddings'].float() @ r['embeddings'].float().T
        else:
            scores = maxsim(q['embeddings'], r['embeddings'], device=device,
                progress=lambda done, total: print(f'{model}/scoring references: {done}/{total}', flush=True))
        score_seconds = time.monotonic() - started
        save_tensor(score_path, dict(schema='milk10k.image_scores.v1', recipe=recipe, scores=scores,
            query_ids=q['ids'], gallery_ids=r['ids'], scoring_seconds=score_seconds))
    metrics, predictions = metrics_and_predictions(scores.numpy(), q['ids'], r['ids'], q['lesion_ids'], r['lesion_ids'])
    result_dir = data / 'results' / model
    result_dir.mkdir(parents=True, exist_ok=True)
    config_result = dict(protocol=PROTOCOL, model=model, model_spec=spec, manifest_signature=manifest['signature'],
        recipe=recipe, training='none', query_population='official test clinical',
        gallery_population='official test dermoscopic', query_count=479, gallery_count=479,
        score_sha256=file_hash(score_path), image_inputs_only=True,
        python_packages={k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'numpy', 'Pillow')},
        hardware=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        timing=dict(**timing, clinical_encoding_seconds=q['encoding_seconds'],
                    dermoscopic_encoding_seconds=r['encoding_seconds'], scoring_seconds=score_seconds,
                    note='Measured stage wall times, including CPU image preprocessing; not a request latency benchmark'))
    if model == 'biomedclip':
        config_result['python_packages'].update({k: importlib.metadata.version(k) for k in ('open_clip_torch', 'timm')})
    write_json(result_dir / 'config.json', config_result)
    write_json(result_dir / 'metrics.json', metrics)
    temp = result_dir / f'predictions.{os.getpid()}.tmp'
    temp.write_text(''.join(json.dumps(p) + '\n' for p in predictions))
    temp.replace(result_dir / 'predictions.jsonl')
    print(json.dumps(dict(model=model, **metrics)), flush=True)


def summarize(data):
    rows = []
    for model in MODEL_ORDER:
        path = data / 'results' / model / 'metrics.json'
        if path.exists():
            rows.append(dict(model=model, **read_json(path)))
    if not rows:
        return
    directory = data / 'results'
    with (directory / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# MILK10k 测试集基础图像检索', '',
        '479 张临床近照 → 全部 479 张测试皮肤镜图；同一病灶为唯一正例。',
        '模型冻结；不使用训练图片、OCR、诊断标签、RoMa、病灶定位或新方法。',
        '这是本项目构建的图像检索评估，不是 ISIC 官方诊断比赛成绩。', '',
        '| 模型 | 评分 | R@1 | R@5 | R@10 | MRR | Top-1 正确数 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        scoring = '向量余弦' if row['model'] in DIMENSIONS else 'MaxSim'
        lines.append(f"| {NAMES[row['model']]} | {scoring} | {100*row['R@1']:.2f}% | {100*row['R@5']:.2f}% | "
                     f"{100*row['R@10']:.2f}% | {row['MRR']:.4f} | {row['correct_at_1']}/479 |")
    lines += ['', f'已完成 {len(rows)}/{len(MODEL_ORDER)} 项预设配置。', '',
        'Qwen 使用官方通用提示 `Represent the user\'s input.`，两侧一致。',
        'Qwen 的官方归一化输出再次进行 FP32 L2 归一化，消除 FP16 舍入导致的长度误差。',
        '其他模型采用官方图像处理；SigLIP2 缩放到 256 方图，DINOv3 缩放到 224 方图，BiomedCLIP 使用标准 224 图像处理，未做病灶定位。',
        '多向量分数为 FP32 mean MaxSim，排除零向量与 padding；单向量为归一化点积。',
        '同分按固定图库顺序（lesion_id）排序。训练集中的重复图不进入本实验。',
        '原始 metadata 不含患者 ID，因此不能证明患者完全独立。', '',
        '缓存：`../cache/image_embeddings/<model>/{clinical,dermoscopic}.pt`、`../cache/scores/<model>.pt`。',
        '每模型目录保留 config.json、metrics.json 和逐查询 predictions.jsonl。']
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--model', choices=MODEL_ORDER)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        manifest = prepare(args.data)
        print(f"Prepared {manifest['query_count']} test pairs", flush=True)
    if args.model:
        lock_path = args.data / 'cache/intermediate' / (args.model + '.lock')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            evaluate(args.data, args.model, args.device, args.cache_only)
    if args.summarize:
        summarize(args.data)
    if not (args.prepare or args.model or args.summarize):
        parser.error('Choose --prepare, --model or --summarize')


if __name__ == '__main__':
    main()
