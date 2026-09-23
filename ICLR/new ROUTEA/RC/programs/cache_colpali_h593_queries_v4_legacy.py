#!/usr/bin/env python3
"""Cache image-only ColPali query embeddings; no labels, scores, or training."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import struct
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[2]
OUT = ROOT / 'results/rc_colpali_h593_query_tokens_v4_legacy'
MODEL = Path('/hkfs/home/project/hk-project-pai00054/ap7811/.cache/huggingface/hub/models--vidore--colpali-v1.3-hf/snapshots/7d3c8ab1c1908b32d701308fb1dfb2968d150c67')
CATALOG = ROOT / 'results/rc_h593_feature_fusion_cache_v1/catalog.json'
GALLERY = WORK / 'colpali/result/difficult/raw_gallery/cache/colpali_gallery_emb.npz'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def publish_json(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(d, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def prepare():
    c = json.loads(CATALOG.read_text())
    records = []
    for q in c['queries']:
        item = c['images'][q['query_image_key']]['item']
        assert sha(item['path']) == item['image_sha256'], item['path']
        records.append({'query_id': q['query_id'], 'ordinal': q['execution_ordinal'],
                        'image_path': item['path'], 'image_sha256': item['image_sha256'],
                        'source_frame': item['frame'], 'source_image_key': q['query_image_key']})
    assert len(records) == 593 and len({r['query_id'] for r in records}) == 593
    gm = ROOT / 'results/rc_new_hyp_processed128_regression_v1/gallery_manifest.json'
    refs = json.loads(gm.read_text())['records']
    anchors = [{**refs[i], 'image_sha256': sha(refs[i]['image_path'])} for i in (0, 2706, 5412)]
    configs = {p.name: sha(p) for p in MODEL.glob('*.json')}
    weights = {p.name: {'resolved_path': str(p.resolve()), 'bytes': p.stat().st_size}
               for p in MODEL.glob('*.safetensors')}
    assert len(weights) == 2
    manifest = {'schema': 'colpali_h593_query_cache_v1', 'query_count': 593,
                'catalog': {'path': str(CATALOG), 'sha256': sha(CATALOG)},
                'model': str(MODEL), 'model_config_sha256': configs, 'model_weights': weights,
                'gallery_cache': str(GALLERY), 'gallery_manifest_sha256': sha(gm),
                'reference_anchors': anchors, 'records': records,
                'label_reads': False, 'training_updates': 0, 'candidate_changes': False,
                'decode': 'PIL.Image.open(path).convert(RGB); no added EXIF transform',
                'compute_dtype': 'bfloat16', 'storage_dtype': 'float16',
                'compatibility_gate': {'min_token_cosine': 0.999, 'relative_l2_max': 0.03,
                                       'sequence_tokens': 1030, 'image_tokens': 1024}}
    dest = OUT / 'manifest.json'
    if dest.exists():
        assert json.loads(dest.read_text()) == manifest
    else:
        publish_json(dest, manifest)
    print(json.dumps({'status': 'INPUTS_FROZEN', 'queries': 593, 'manifest': str(dest)}), flush=True)


def gallery_memmap(path):
    import numpy as np
    with zipfile.ZipFile(path) as z:
        info = z.getinfo('passage_emb.npy')
        assert info.compress_type == zipfile.ZIP_STORED
    with open(path, 'rb') as f:
        f.seek(info.header_offset)
        header = f.read(30)
        assert header[:4] == b'PK\x03\x04'
        name_len, extra_len = struct.unpack('<HH', header[26:30])
        f.seek(name_len + extra_len, 1)
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        offset = f.tell()
    assert shape == (5413, 1030, 128) and not fortran
    return np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape)


def selected_records(m, shard, shards):
    assert shards == 50 and 0 <= shard < shards
    return [r for r in sorted(m['records'], key=lambda r: r['ordinal'])
            if r['ordinal'] % shards == shard]


def checkpoint_to_model_key(key):
    """Mirror the built-in conversion in the historical Transformers 4.57.3."""
    key = re.sub(r'^vlm\.language_model\.model\.', 'vlm.model.language_model.', key)
    key = re.sub(r'^vlm\.vision_tower\.', 'vlm.model.vision_tower.', key)
    return re.sub(r'^vlm\.multi_modal_projector\.', 'vlm.model.multi_modal_projector.', key)


def extract(budget, shard, shards):
    started = time.monotonic()
    print(json.dumps({'status': 'IMPORTING_HISTORICAL_ENV', 'shard': shard}), flush=True)
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import ColPaliForRetrieval, ColPaliProcessor
    print(json.dumps({'status': 'HISTORICAL_ENV_IMPORTED', 'torch': torch.__version__,
                      'transformers': transformers.__version__}), flush=True)
    manifest_path = OUT / 'manifest.json'
    m = json.loads(manifest_path.read_text())
    manifest_sha = sha(manifest_path)
    records = selected_records(m, shard, shards)
    shard_out = OUT / 'shards' / f'{shard:02d}'
    shard_out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (shard_out / 'writer.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print(json.dumps({'status': 'SHARD_START', 'shard': shard, 'queries': len(records)}), flush=True)
    assert torch.cuda.is_available(), 'GPU required only for query encoding'
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name, digest in m['model_config_sha256'].items():
        assert sha(MODEL / name) == digest
    # Use the environment referenced by the historical gallery launcher.
    # v5 loaded correctly in v3 but failed the unchanged old-gallery anchors.
    assert transformers.__version__ == '4.57.3', transformers.__version__
    assert torch.__version__.split('+')[0] == '2.5.1', torch.__version__
    key_mapping = 'transformers_4.57.3_builtin'
    model, loading = ColPaliForRetrieval.from_pretrained(
        str(MODEL), torch_dtype=torch.bfloat16, local_files_only=True,
        output_loading_info=True)
    bad = {k: sorted(map(str, loading.get(k, []))) for k in
           ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')}
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    converted = [checkpoint_to_model_key(k) for k in index]
    state = model.state_dict()
    # The unused generation head in v4 shares the loaded input embedding.
    # Accept only verified storage aliases, never an independently initialized head.
    aliases = {}
    alias, target = 'vlm.lm_head.weight', 'vlm.model.language_model.embed_tokens.weight'
    if alias in state and target in state:
        assert state[alias].data_ptr() == state[target].data_ptr(), 'Untied language head'
        assert state[alias].shape == state[target].shape
        aliases[alias] = target
    actual = set(state) - set(aliases)
    bad['missing_keys'] = [k for k in bad['missing_keys'] if k not in aliases]
    bad['unmapped_model_keys'] = sorted(actual - set(converted))
    bad['unused_checkpoint_keys'] = sorted(set(converted) - actual)
    assert len(converted) == len(set(converted)), 'Checkpoint rename collision'
    # Read only safetensors metadata, checking every stored tensor's shape.
    expected_shapes = {}
    for filename in sorted(set(index.values())):
        with (MODEL / filename).open('rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_size))
        for key, metadata in header.items():
            if key != '__metadata__':
                expected_shapes[checkpoint_to_model_key(key)] = metadata['shape']
    bad['checkpoint_shape_mismatches'] = sorted(
        k for k in actual & set(expected_shapes)
        if list(state[k].shape) != expected_shapes[k])
    bad['index_header_key_difference'] = sorted(set(expected_shapes) ^ set(converted))
    clean_load = not any(bad.values())
    publish_json(shard_out / 'model_loading.json', {
        'status': 'COLPALI_ALL_WEIGHTS_LOADED' if clean_load else 'FAIL',
        'transformers': transformers.__version__, 'key_mapping': key_mapping,
        'checkpoint_keys': len(index), 'model_keys': len(actual), 'issues': bad,
        'verified_tied_aliases': aliases,
        'automatic_key_conversion_accounted_for': 'transformers_4.57.3_colpali_mapping',
        'program_sha256': sha(__file__), 'model': str(MODEL)})
    assert clean_load, f'Incomplete checkpoint load: {bad}'
    model = model.to('cuda').eval()
    processor = ColPaliProcessor.from_pretrained(str(MODEL), local_files_only=True, use_fast=True)
    provenance = {'model': str(MODEL), 'manifest_sha256': manifest_sha,
                  'torch': torch.__version__, 'transformers': transformers.__version__,
                  'image_processor': type(processor.image_processor).__name__,
                  'processor': type(processor).__name__, 'device': torch.cuda.get_device_name(),
                  'job_id': os.environ.get('SLURM_JOB_ID'), 'program_sha256': sha(__file__),
                  'key_mapping': key_mapping,
                  'model_loading_sha256': sha(shard_out / 'model_loading.json')}

    def encode(records):
        images, sizes, pixel_shas = [], [], []
        for r in records:
            assert sha(r['image_path']) == r['image_sha256'], r['image_path']
            with Image.open(r['image_path']) as f:
                im = f.convert('RGB')
            images.append(im); sizes.append(im.size)
            pixel_shas.append(hashlib.sha256(im.tobytes()).hexdigest())
        inputs = processor(images=images, return_tensors='pt')
        ids = inputs['input_ids'].clone()
        mask = ids == model.config.vlm_config.image_token_index
        assert torch.all(mask.sum(1) == 1024)
        assert ids.shape[1] == 1030
        with torch.inference_mode():
            embedding = model(**{k: v.to('cuda') for k, v in inputs.items()}).embeddings
        embedding = embedding.to(dtype=torch.float16, device='cpu')
        assert torch.isfinite(embedding).all() and embedding.shape[1:] == (1030, 128)
        return embedding, ids, inputs['attention_mask'].cpu(), mask, sizes, pixel_shas

    # Re-encode three references solely to validate old-gallery/new-query compatibility.
    # Their comparison is numerical, not a claim of bit-exact historical reproduction.
    old = gallery_memmap(GALLERY)
    emb, ids, attn, imask, sizes, pix = encode(m['reference_anchors'])
    comparison = []
    for i, r in enumerate(m['reference_anchors']):
        a = torch.from_numpy(np.array(old[r['physical_row']], copy=True)).double()
        b = emb[i].double()
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=1)
        relative = float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(a))
        comparison.append({'physical_row': r['physical_row'], 'min_token_cosine': float(cosine.min()),
                           'relative_l2': relative, 'max_abs_error': float((a-b).abs().max())})
    ok = all(x['min_token_cosine'] >= 0.999 and x['relative_l2'] <= 0.03 for x in comparison)
    publish_json(shard_out / 'compatibility.json', {'status': 'PASS' if ok else 'FAIL',
                 'comparison': comparison, 'provenance': provenance,
                 'scope': 'three reference anchors, numerical compatibility only'})
    assert ok, 'Historical gallery compatibility failed; no query encoding released'
    print(json.dumps({'status': 'REFERENCE_COMPATIBILITY_PASS', 'comparison': comparison}), flush=True)

    def accepted(r):
        p = OUT / 'queries' / (r['query_id'] + '.pt')
        v = p.with_suffix('.json')
        if not v.exists():
            return False
        receipt = json.loads(v.read_text())
        assert receipt['manifest_sha256'] == manifest_sha
        assert receipt['image_sha256'] == r['image_sha256']
        assert receipt['payload_sha256'] == sha(p)
        return True

    pending = [r for r in records if not accepted(r)]
    done = len(records) - len(pending)
    for begin in range(0, len(pending), 4):
        if time.monotonic() - started > budget:
            publish_json(shard_out / 'progress.json', {'status': 'RESUMABLE', 'complete': done,
                         'shard': shard, 'total': len(records), 'elapsed_seconds': time.monotonic()-started})
            return 75
        batch = pending[begin:begin+4]
        emb, ids, attn, masks, sizes, pixel_shas = encode(batch)
        for i, r in enumerate(batch):
            p = OUT / 'queries' / (r['query_id'] + '.pt')
            p.parent.mkdir(parents=True, exist_ok=True)
            # Only unreceipted interrupted output may be regenerated.
            assert not p.with_suffix('.json').exists()
            payload = {'schema': 'colpali_query_tokens_v1', **r, 'provenance': provenance,
                       'tokens': emb[i].contiguous(), 'input_ids': ids[i],
                       'attention_mask': attn[i], 'image_token_mask': masks[i],
                       'image_grid_hw': (32, 32), 'original_size_wh': sizes[i],
                       'rgb_pixels_sha256': pixel_shas[i],
                       'image_patch_order': 'row_major', 'resize_wh': (448, 448)}
            tmp = p.with_name(p.name + '.tmp')
            with tmp.open('wb') as f:
                torch.save(payload, f); f.flush(); os.fsync(f.fileno())
            check = torch.load(tmp, map_location='cpu', weights_only=False)
            assert torch.equal(check['tokens'], payload['tokens'])
            assert int(check['image_token_mask'].sum()) == 1024
            os.replace(tmp, p)
            publish_json(p.with_suffix('.json'), {'status': 'COLPALI_QUERY_CACHE_PASS',
                         'query_id': r['query_id'], 'image_sha256': r['image_sha256'],
                         'manifest_sha256': manifest_sha, 'payload_sha256': sha(p),
                         'token_shape': [1030, 128], 'image_tokens': 1024})
            done += 1
        status = {'status': 'RUNNING', 'complete': done, 'total': len(records), 'shard': shard,
                  'elapsed_seconds': time.monotonic()-started}
        publish_json(shard_out / 'progress.json', status)
        print(json.dumps(status), flush=True)
    assert all(accepted(r) for r in records)
    publish_json(shard_out / 'validation.json', {'status': 'COLPALI_QUERY_SHARD_PASS',
                 'complete': len(records), 'shard': shard, 'shards': shards,
                 'query_ids': [r['query_id'] for r in records], 'manifest_sha256': manifest_sha,
                 'compatibility_sha256': sha(shard_out / 'compatibility.json'),
                 'training_updates': 0, 'label_reads': False,
                 'elapsed_seconds_this_run': time.monotonic()-started})
    print(f'COLPALI_QUERY_SHARD_PASS {shard} {len(records)}/{len(records)}', flush=True)
    return 0


def join():
    m = json.loads((OUT / 'manifest.json').read_text())
    ms = sha(OUT / 'manifest.json')
    ids, evidence = [], []
    for shard in range(50):
        p = OUT / 'shards' / f'{shard:02d}'
        v = json.loads((p / 'validation.json').read_text())
        expected = [r['query_id'] for r in selected_records(m, shard, 50)]
        assert v['status'] == 'COLPALI_QUERY_SHARD_PASS'
        assert v['manifest_sha256'] == ms and v['query_ids'] == expected
        assert v['complete'] == len(expected) and v['shard'] == shard
        assert v['compatibility_sha256'] == sha(p / 'compatibility.json')
        assert json.loads((p / 'compatibility.json').read_text())['status'] == 'PASS'
        loading = json.loads((p / 'model_loading.json').read_text())
        assert loading['status'] == 'COLPALI_ALL_WEIGHTS_LOADED'
        assert not any(loading['issues'].values())
        assert json.loads((p / 'compatibility.json').read_text())['provenance']['model_loading_sha256'] == sha(p / 'model_loading.json')
        ids.extend(expected)
        evidence.append({'shard': shard, 'sha256': sha(p / 'validation.json')})
    assert len(ids) == len(set(ids)) == 593
    for r in m['records']:
        p = OUT / 'queries' / (r['query_id'] + '.pt')
        v = json.loads(p.with_suffix('.json').read_text())
        assert v['status'] == 'COLPALI_QUERY_CACHE_PASS' and v['query_id'] == r['query_id']
        assert v['manifest_sha256'] == ms and v['image_sha256'] == r['image_sha256']
        assert v['payload_sha256'] == sha(p)
    publish_json(OUT / 'validation.json', {'status': 'COLPALI_H593_QUERY_CACHE_PASS',
                 'complete': 593, 'shards': 50, 'manifest_sha256': ms,
                 'shard_validations': evidence, 'label_reads': False, 'training_updates': 0})
    print('COLPALI_H593_QUERY_CACHE_PASS 593/593; 50/50 shards', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'extract', 'join'])
    ap.add_argument('--budget', type=int, default=780)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=50)
    args = ap.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'extract':
        raise SystemExit(extract(args.budget, args.shard, args.shards))
    else:
        join()
