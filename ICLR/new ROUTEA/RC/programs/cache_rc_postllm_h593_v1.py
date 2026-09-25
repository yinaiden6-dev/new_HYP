#!/usr/bin/env python3
"""Collect only frozen H593 query hidden states before the retrieval projector.

The 128-D gallery tokens and existing 24 qualified query caches are read-only.
Each new query needs one native vision/LLM forward. A pre-hook captures the
actual full-sequence 3584-D input to the original frozen retrieval projection;
an independent projection replay checks it against the native output. No model
is trained, and no fold labels are opened by this worker.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_postllm_h593_v1'
AUTH = ROOT / 'registry/rc_postllm_h593_authority_v1_20260924.json'
STATUS = 'QUERY_ENCODER_CACHE_PASS'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def checked(binding):
    path = Path(binding['path'])
    need(sha(path) == binding['sha256'], 'SOURCE_SHA:' + str(path))
    return path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def emit(**record):
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def install_read_guard():
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = str(Path(os.fsdecode(args[0])).resolve()).lower()
        need(not any(x in path for x in ('formal392', 'd1-mi', 'd1_mi', '/target_join/',
                                        'curator_roles', '/fold_inputs/')),
             'CACHE_WORKER_PROTECTED_OR_LABEL_PATH:' + path)
    sys.addaudithook(audit)


def source_token_binding(row):
    entry = row['query_tokens']
    return entry.get('token_file', entry)


def legacy_validation(row):
    if row.get('hidden_validation') is not None:
        return row['hidden_validation']
    if row.get('hidden_cache') is not None:
        return row['hidden_cache']['validation']
    return None


def cache_directory(row):
    expected = OUT / 'encoder_cache' / row['query_id']
    supplied = Path(row.get('hidden_cache_dir', expected))
    need(supplied.resolve() == expected.resolve(), 'CACHE_DESTINATION_OUTSIDE_NEW_ROOT')
    return expected


def verify_model_sources(a, manifest):
    """Reuse full SHA receipt only with unchanged inode/size/mtime signatures."""
    receipt_binding = a['model_source_validation']
    receipt = read(checked(receipt_binding))
    need(receipt['status'] == 'MODEL_SOURCE_HASHES_PASS', 'UNQUALIFIED_MODEL_SOURCE_RECEIPT')
    expected = {x['path']: x for x in manifest['encoder']['model_sources'].values()}
    actual = {x['binding']['path']: x['binding'] for x in receipt['files']}
    need(expected == actual, 'MODEL_SOURCE_RECEIPT_SET_MISMATCH')
    for record in receipt['files']:
        path = Path(record['binding']['path']); stat = path.stat()
        need(record['stat'][1:] == [stat.st_ino, stat.st_size, stat.st_mtime_ns],
             'MODEL_SOURCE_STAT_DRIFT:' + str(path))
        if stat.st_size <= 2 << 20:
            checked(record['binding'])
    return receipt_binding


def load_contract():
    a = read(AUTH); authority = bind(AUTH)
    manifest = read(checked(a['manifest']))
    need(a['attention'] == 'sdpa' and a['processor_backend'] == 'torchvision', 'FROZEN_PROCESSOR_CONTRACT')
    need(Path(a['model']).resolve() == Path(manifest['encoder']['model_path']).resolve(), 'MODEL_LINEAGE')
    sources = a.get('code_sources', [])
    own = bind(__file__)
    need(own in sources, 'CACHE_WORKER_NOT_BOUND_BY_AUTHORITY')
    for source in sources:
        checked(source)
    verify_model_sources(a, manifest)
    rows = manifest['rows']
    need(len(rows) == 593 and len({r['query_id'] for r in rows}) == 593, 'H593_QUERY_AXIS')
    need([r['execution_ordinal'] for r in rows] == list(range(593)), 'EXECUTION_ORDINAL_AXIS')
    for row in rows:
        need(len(row['reference_tokens']) == len(row['L0']) == 128, 'FULL_C128_REFERENCE_AXIS')
        need(not any(k in row for k in ('target_id', 'target_positions', 'target_position',
                                        'raw_correct', 'target_identity')), 'MANIFEST_LABEL_FIELD')
        cache_directory(row)
    return a, manifest, authority


def tensor_contract(payload, row):
    import torch
    hidden, mask, native, batch = (payload[k] for k in ('hidden', 'image_mask', 'native_tokens', 'batch'))
    need(payload['query_id'] == row['query_id'], 'CACHE_QUERY_ID')
    need(hidden.ndim == 3 and hidden.shape[0] == 1 and hidden.shape[-1] == 3584,
         'MUST_CACHE_TRUE_FULL_SEQUENCE_3584D_HIDDEN')
    need(hidden.dtype == torch.bfloat16 and native.dtype == torch.bfloat16, 'BF16_CACHE_DTYPES')
    need(mask.dtype == torch.bool and tuple(mask.shape) == tuple(hidden.shape[:2]), 'IMAGE_MASK_SHAPE')
    need(tuple(native.shape) == tuple(hidden.shape[:2]) + (128,), 'NATIVE_FULL_SEQUENCE_SHAPE')
    need(tuple(batch['attention_mask'].shape) == tuple(mask.shape) and bool(mask.any()), 'ATTENTION_MASK')
    need(not bool((mask & ~batch['attention_mask'].bool()).any()), 'IMAGE_MASK_IN_PADDING')
    need('pixel_values' not in batch, 'PIXELS_MUST_NOT_BE_PERSISTED')
    need(not hidden.requires_grad and bool(torch.isfinite(hidden).all()) and bool(torch.isfinite(native).all()),
         'FINITE_DETACHED_CACHE')
    need(payload['fresh_L0'].shape == payload['original_L0'].shape == (128,), 'FRESH_CONTENT_AXIS')
    need(bool(torch.isfinite(payload['fresh_L0']).all()), 'FINITE_FRESH_CONTENT')


def verify_existing(row, authority, *, tensors=False):
    legacy = legacy_validation(row)
    path = checked(legacy) if legacy else cache_directory(row) / 'validation.json'
    if not path.exists():
        return None
    record = read(path)
    need(record['status'] == STATUS, 'CACHE_NOT_QUALIFIED')
    need(record['image_sha256'] == row['source_image_sha256'], 'CACHE_IMAGE_LINEAGE')
    if not legacy:
        need(record['authority'] == authority and record['query_id'] == row['query_id'], 'NEW_CACHE_AUTHORITY')
    payload_path = checked(record['payload']); checked(record['parity'])
    if row.get('hidden_cache') is not None:
        need(row['hidden_cache']['payload'] == record['payload'], 'LEGACY_PAYLOAD_BINDING')
        need(row['hidden_cache']['parity'] == record['parity'], 'LEGACY_PARITY_BINDING')
    if tensors:
        import torch
        payload = torch.load(payload_path, map_location='cpu', weights_only=True)
        tensor_contract(payload, row)
    return {'query_id': row['query_id'], 'execution_ordinal': row['execution_ordinal'],
            'source': 'legacy_v2' if legacy else 'new_h593', 'validation': bind(path),
            'payload': record['payload'], 'parity': record['parity']}


def load_model(a, authority):
    import torch
    from colpali_engine.models import ColQwen2_5_Processor
    from rc_prellm_m_adapter_v1 import _load_frozen_colnomic_weights
    need(torch.cuda.is_available(), 'GPU_REQUIRED')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    model, report = _load_frozen_colnomic_weights(a['model'], 'cuda', attention='sdpa',
                                                verify_weight_hashes=False)
    # Omitting use_fast is essential: explicit False selected a different backend.
    processor = ColQwen2_5_Processor.from_pretrained(a['model'], local_files_only=True)
    backend = str(getattr(processor.image_processor, 'backend', None))
    need(backend == 'torchvision', 'QUALIFIED_TORCHVISION_PROCESSOR_REQUIRED')
    need(not any(p.requires_grad for p in model.parameters()), 'BACKBONE_MUST_BE_FROZEN')
    need(any('lora_' in name for name, _ in model.named_parameters()), 'RETRIEVAL_LORA_MISSING')
    report.update(processor_backend=backend, processor_use_fast_keyword='omitted',
                  loading_seconds=time.monotonic()-started, source_validation=a['model_source_validation'])
    name = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}.json"
    write(OUT/'loading'/name, {'authority': authority, 'report': report})
    emit(stage='model_loaded', seconds=report['loading_seconds'], gpu=torch.cuda.get_device_name())
    return model, processor


def projection_parity(model, payload):
    import torch
    from rc_prellm_m_adapter_v1 import project_cached_hidden
    with torch.no_grad():
        actual = project_cached_hidden(model, payload['hidden'], payload['batch'], None, [0.0])
    expected = payload['native_tokens']
    return {'full_sequence_max_abs_error': float((actual.float()-expected.float()).abs().max()),
            'full_sequence_exact_equal': bool(torch.equal(actual, expected)),
            'image_tokens': int(payload['image_mask'].sum()),
            'sequence_tokens': int(payload['hidden'].shape[1])}


def cuda_payload(payload):
    import torch
    return {key: ({k: v.to('cuda') for k, v in value.items()} if key == 'batch'
                  else value.to('cuda') if isinstance(value, torch.Tensor) else value)
            for key, value in payload.items()}


@contextmanager
def query_lock(query_id):
    directory = OUT/'locks'; directory.mkdir(parents=True, exist_ok=True)
    with (directory/(query_id+'.lock')).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def collect(row, model, processor, a, authority):
    import torch
    from PIL import Image, ImageOps
    from torch.nn import functional as F
    from rc_prellm_m_adapter_v1 import find_colnomic_base
    with query_lock(row['query_id']):
        existing = verify_existing(row, authority)
        if existing is not None:
            return existing
        destination = cache_directory(row)
        need(not destination.exists(), 'INCOMPLETE_FINAL_CACHE_DIRECTORY_REQUIRES_INSPECTION')
        started = time.monotonic()
        need(sha(row['image_path']) == row['source_image_sha256'], 'QUERY_IMAGE_SHA')
        with Image.open(row['image_path']) as image:
            if row['frame'] in ('EXIF_TRANSPOSED', 'EXIF_NORMALIZED', 'EXIF_ORIENTED_BEFORE_RESIZE'):
                image = ImageOps.exif_transpose(image)
            else:
                need(row['frame'] == 'DECODED_RAW_BEFORE_EXIF', 'UNKNOWN_QUERY_IMAGE_FRAME')
            batch = processor.process_images([image.convert('RGB')]).to('cuda')
        base = find_colnomic_base(model)
        captured = []
        def hook(module, args):
            need(len(args) == 1, 'UNEXPECTED_RETRIEVAL_PROJECTOR_ARGUMENTS')
            captured.append(args[0].detach().clone())
        handle = base.custom_text_proj.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                native = model(**batch)
        finally:
            handle.remove()
        need(len(captured) == 1, 'EXACTLY_ONE_NATIVE_FULL_HIDDEN_CAPTURE')
        hidden = captured.pop()
        mask = batch['input_ids'].eq(base.config.image_token_id)
        gpu = {'hidden': hidden, 'native_tokens': native, 'image_mask': mask,
               'batch': {k: v for k, v in batch.items() if k != 'pixel_values'}}
        parity = projection_parity(model, gpu)
        need(parity['full_sequence_max_abs_error'] <= a.get('cache_parity_atol', 1e-6),
             'NATIVE_VS_INDEPENDENT_FULL_HIDDEN_PROJECTION')
        old_file = source_token_binding(row)
        original_payload = torch.load(checked(old_file), map_location='cpu', weights_only=True)
        original = original_payload['tokens'].to('cuda')
        current = native[mask]
        need(original.shape == current.shape, 'HISTORICAL_NATIVE_QUERY_TOKEN_SHAPE')
        historical_exact = bool(torch.equal(current.float().half(), original.half()))
        cosine = float(F.cosine_similarity(original.float(), current.float(), dim=-1).mean())
        old_unit = F.normalize(original.double(), dim=-1)
        fresh_unit = F.normalize(current.double(), dim=-1)
        original_L, fresh_L = [], []
        for reference_binding in row['reference_tokens']:
            source = reference_binding.get('token_file', reference_binding)
            reference_payload = torch.load(checked(source), map_location='cpu', weights_only=True)
            reference = F.normalize(reference_payload['tokens'].to(device='cuda', dtype=torch.float64), dim=-1)
            original_L.append((old_unit@reference.T).max(1).values.mean().cpu())
            fresh_L.append((fresh_unit@reference.T).max(1).values.mean().cpu())
        original_L = torch.stack(original_L); fresh_L = torch.stack(fresh_L)
        old_replay_error = float((original_L-torch.tensor(row['L0'], dtype=torch.float64)).abs().max())
        drift = float((fresh_L-original_L).abs().max())
        need(old_replay_error < 2e-10, 'HISTORICAL_CONTENT_REPLAY')
        gates = a.get('source_parity', {'mean_cosine_min': .999, 'max_content_error': .005})
        need(cosine >= gates['mean_cosine_min'], 'HISTORICAL_NATIVE_TOKEN_COSINE')
        need(drift <= gates['max_content_error'], 'HISTORICAL_CONTENT_DRIFT')
        need(historical_exact or not a.get('cache', {}).get('require_historical_fp16_exact', True),
             'HISTORICAL_FP16_TOKEN_EQUALITY')
        parity.update(query_id=row['query_id'], historical_fp16_exact=historical_exact,
                      historical_token_max_abs_error=float((original.float()-current.float()).abs().max()),
                      token_mean_cosine=cosine, original_content_replay_error=old_replay_error,
                      content_max_drift=drift, fresh_L0=fresh_L.tolist(), original_L0=original_L.tolist(),
                      query_token_source=old_file, reference_token_sources=row['reference_tokens'],
                      native_query_encoder_forwards=1, reference_encoder_forwards=0,
                      roma_forwards=0, held_labels_read=False)
        payload = dict(authority=authority, query_id=row['query_id'],
                       batch={k: v.detach().cpu() for k,v in gpu['batch'].items()},
                       hidden=hidden.cpu(), image_mask=mask.cpu(), native_tokens=native.cpu(),
                       fresh_L0=fresh_L, original_L0=original_L)
        tensor_contract(payload, row)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=row['query_id']+'.partial.', dir=destination.parent))
        # File objects avoid torch's platform-dependent hidden-temp-name rejection.
        with (temporary/'payload.pt').open('wb') as stream:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        write(temporary/'parity.json', parity)
        def final_binding(name):
            return {'path': str((destination/name).resolve()), 'sha256': sha(temporary/name)}
        validation = dict(status=STATUS, authority=authority, query_id=row['query_id'],
                          execution_ordinal=row['execution_ordinal'],
                          payload=final_binding('payload.pt'), parity=final_binding('parity.json'),
                          image_sha256=row['source_image_sha256'], held_labels_read=False,
                          true_hidden_width=3584, native_token_width=128,
                          encoder_forwards=1, reference_encoder_forwards=0, roma_forwards=0,
                          seconds=time.monotonic()-started,
                          peak_cuda_bytes=torch.cuda.max_memory_allocated())
        write(temporary/'validation.json', validation)
        os.rename(temporary, destination)
        emit(stage='query_cache', query_id=row['query_id'], seconds=validation['seconds'],
             historical_fp16_exact=historical_exact, projection_error=parity['full_sequence_max_abs_error'],
             content_drift=drift)
        return verify_existing(row, authority)


def run_pilot(a, manifest, authority, budget, started):
    import torch
    rows = manifest['rows']
    old = next((r for r in rows if legacy_validation(r)), None)
    fresh = next((r for r in rows if not legacy_validation(r)), None)
    need(old is not None and fresh is not None, 'PILOT_REQUIRES_LEGACY_AND_MISSING_QUERY')
    receipt = verify_existing(old, authority, tensors=True)
    model, processor = load_model(a, authority)
    old_cache = torch.load(checked(receipt['payload']), map_location='cpu', weights_only=True)
    old_gpu = cuda_payload(old_cache)
    old_parity = projection_parity(model, old_gpu)
    need(old_parity['full_sequence_exact_equal'], 'LEGACY_GPU_PROJECTOR_EXACT_PARITY')
    del old_gpu, old_cache; torch.cuda.empty_cache()
    need(time.monotonic()-started < budget-30, 'PILOT_BUDGET_BEFORE_FIRST_NEW_QUERY')
    first = collect(fresh, model, processor, a, authority)
    result = {'status': 'H593_QUERY_HIDDEN_PILOT_PASS', 'authority': authority,
              'legacy_query': receipt, 'legacy_GPU_projection': old_parity, 'first_new_query': first,
              'seconds': time.monotonic()-started, 'held_labels_read': False,
              'reference_encoder_forwards': 0, 'roma_forwards': 0}
    write(OUT/'encoder_cache/pilot_validation.json', result)
    emit(stage='pilot', status=result['status'], seconds=result['seconds'])


def run_cache(a, manifest, authority, shard, shards, budget, started):
    import torch
    pilot = read(OUT/'encoder_cache/pilot_validation.json')
    need(pilot['status'] == 'H593_QUERY_HIDDEN_PILOT_PASS' and pilot['authority'] == authority,
         'QUALIFIED_PILOT_REQUIRED_BEFORE_SHARDS')
    selected = [r for r in manifest['rows'] if r['execution_ordinal'] % shards == shard]
    completed = []; pending = []
    for row in selected:
        previous = verify_existing(row, authority)
        (completed if previous else pending).append(previous if previous else row)
    model = processor = None
    if pending:
        model, processor = load_model(a, authority)
    for row in pending:
        if time.monotonic()-started >= budget-30:
            write(OUT/'encoder_cache/shards'/f'{shard:02d}_progress.json',
                  {'status': 'PARTIAL_BUDGET', 'authority': authority, 'shard': shard, 'shards': shards,
                   'completed': completed, 'total': len(selected), 'seconds': time.monotonic()-started})
            return False
        completed.append(collect(row, model, processor, a, authority)); torch.cuda.empty_cache()
    completed.sort(key=lambda r: r['execution_ordinal'])
    write(OUT/'encoder_cache/shards'/f'{shard:02d}_validation.json',
          {'status': 'H593_QUERY_HIDDEN_SHARD_PASS', 'authority': authority, 'shard': shard,
           'shards': shards, 'count': len(completed), 'queries': completed,
           'seconds': time.monotonic()-started, 'held_labels_read': False})
    emit(stage='cache', status='H593_QUERY_HIDDEN_SHARD_PASS', shard=shard, count=len(completed))
    return True


def run_validate(manifest, authority):
    complete = []; missing = []
    for row in manifest['rows']:
        receipt = verify_existing(row, authority, tensors=True)
        if receipt is None:
            missing.append(row['query_id'])
        else:
            complete.append(receipt)
    status = 'H593_QUERY_HIDDEN_ALL593_PASS' if not missing else 'H593_QUERY_HIDDEN_INCOMPLETE'
    result = {'status': status, 'authority': authority, 'count': len(complete),
              'legacy_count': sum(x['source'] == 'legacy_v2' for x in complete),
              'new_count': sum(x['source'] == 'new_h593' for x in complete),
              'queries': complete, 'missing': missing, 'held_labels_read': False,
              'reference_encoder_forwards': 0, 'roma_forwards': 0}
    write(OUT/'encoder_cache'/('validation.json' if not missing else 'progress.json'), result)
    emit(stage='validate', status=status, count=len(complete), missing=len(missing))
    return not missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('pilot', 'cache', 'validate'))
    parser.add_argument('--shard', '--shard-index', dest='shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=50)
    parser.add_argument('--budget', type=float, default=400)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or args.budget <= 30 or args.threads < 1:
        parser.error('require 0 <= shard < shards, budget > 30, threads >= 1')
    install_read_guard()
    import torch
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    started = time.monotonic()
    a, manifest, authority = load_contract()
    if args.stage == 'pilot':
        run_pilot(a, manifest, authority, args.budget, started)
    elif args.stage == 'validate':
        if not run_validate(manifest, authority):
            raise SystemExit(75)
    elif not run_cache(a, manifest, authority, args.shard, args.shards, args.budget, started):
        raise SystemExit(75)


if __name__ == '__main__':
    main()
