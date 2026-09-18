#!/usr/bin/env python3
"""Frozen encoder bridge and reference-only enrollment; zero query evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
import materialize_rc_original7_train128_token_raw_v1 as R
import run_romav2_colnomic_sealed_source_e0_v2 as S
import run_rc_train128_disagreement_oof4_v1 as P

OUT = ROOT / 'results/rc_new_hyp_isic_transfer_v1/encoded_references'
AUTH = ROOT / 'registry/rc_new_hyp_isic_reference_encoding_authority_v1_20260914.json'


def read(p): return json.loads(Path(p).read_text())
def bind(p): return R.bind(Path(p))


def checked(b):
    p = Path(b['path'])
    assert R.sha(p) == b['sha256'], 'SOURCE_DRIFT:' + str(p)
    return p


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def guard():
    assert os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED'
    a = read(AUTH)
    allowed = set()
    for family in ('sources', 'operator_sources'):
        for b in a[family].values(): allowed.add(checked(b).resolve())
    def audit(event, args):
        if event == 'socket.connect':
            address = args[1]
            assert not isinstance(address, tuple), 'OFFLINE_MODEL_RUNTIME'
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
        p = Path(os.fsdecode(args[0])).resolve()
        name = str(p).lower()
        assert not any(s in name for s in ('curator_roles', '/target_join/', 'd1-mi', 'd1_mi', '/reports/', 'rc_opened_', 'img_metadata.csv', 'metadata_c3', 'seg_metadata', '/masks/')), 'PROTECTED_READ'
        if ROOT / 'results' in p.parents:
            own = OUT in p.parents
            reference_image = p.parent == OUT.parent / 'images' and p.name.startswith('ISIC-REF-')
            assert own or reference_image or p in allowed, 'UNLISTED_RESULT:' + name
    sys.addaudithook(audit)
    assert read(checked(a['sources']['overlap']))['status'] == 'ISIC927_IMAGE_INTAKE_PASS'
    assert read(checked(a['sources']['images']))['status'] == 'ISIC927_IMAGES_RECEIVED_NOT_SCORED'
    assert read(checked(a['sources']['heads']))['status'] == 'EXTERNAL_HEADS_READY_DATA_AND_INFERENCE_PENDING'
    return a


def compute(a):
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    pre = read(checked(a['sources']['old_preflight']))
    assert R.runtime_libraries() == pre['runtime_libraries'], 'FROZEN_RUNTIME'
    device = torch.device('cuda')
    gpu = R.gpu_runtime()
    encoder = ColQwen2_5.from_pretrained(str(R.MODEL), torch_dtype=torch.bfloat16).to(device).eval()
    encoder.requires_grad_(False)
    processor = ColQwen2_5_Processor.from_pretrained(str(R.MODEL))
    old = torch.load(checked(a['sources']['old_train_raw']), map_location='cpu', mmap=True, weights_only=True)
    v = read(checked(a['sources']['old_train_validation']))
    assert v['status'] == 'RC_ORIGINAL7_TRAIN128_TOKEN_RAW_CPU_REPLAY_PASS' and v['payload'] == a['sources']['old_train_raw']
    legacy = torch.load(R.GALLERY, map_location='cpu', mmap=True, weights_only=False)
    labels, _ = P.gallery_labels()
    bridge = []
    with torch.inference_mode():
        # Four already-opened TRAIN images; never read their target labels or fit a head.
        for original in old['records'][:4]:
            image, template, grid, _ = R.encode_query(original, processor, encoder, device)
            assert torch.equal(image, original['query_tokens']) and torch.equal(template, original['template_tokens'])
            assert grid == list(original['query_grid_shape']), 'OLD_QUERY_ENCODING_PARITY'
            scores = S.full_gallery_scores(torch.cat((image, template)), legacy['passage_emb'], device, batch_size=16)
            assert torch.equal(scores, original['raw_physical_scores']), 'OLD_RAW_ALL5413_SCORE_BITS'
            order = S.top128_rows(scores, labels)
            assert order == original['candidate_ranked_physical_rows'], 'OLD_NATURAL_C128_PARITY'
            bridge.append(dict(query_id=original['query_id'], all5413_scores_bit_exact=True, image_template_tokens_bit_exact=True, natural_C128_bit_exact=True))
        write(OUT / 'legacy_bridge.json', dict(status='ISIC_ENCODER_OLD_TRAIN4_RAW_BRIDGE_PASS',
              authority=bind(AUTH), checks=bridge, external_query_forwards=0, training_updates=0))
        del legacy
        gallery = read(checked(a['sources']['gallery']))['records']
        images = {r['asset_id']: r for r in read(checked(a['sources']['images']))['files'] if r['role'] == 'reference'}
        records = []
        for i, entry in enumerate(gallery):
            asset = images[entry['reference_id']]
            source = dict(query_source_path=entry['image_path'], query_source_sha256=asset['image_sha256'], track='outcome')
            image, template, grid, metadata = R.encode_query(source, processor, encoder, device)
            # Reconstruct the original full token sequence; RAW references retain template tokens.
            with Image.open(entry['image_path']) as raw:
                inputs = processor.process_images([raw.convert('RGB')])
            mask = inputs['input_ids'][0] == processor.image_token_id
            assert int(mask.sum()) == len(image) and int((~mask).sum()) == len(template)
            passage = torch.empty((len(mask), 128), dtype=torch.float16)
            passage[mask], passage[~mask] = image, template
            records.append(dict(physical_row=entry['physical_row'], reference_id=entry['reference_id'],
                  source_path=entry['image_path'], source_image_sha256=asset['image_sha256'], grid_shape=grid,
                  tokens=image, tokens_sha256=R.tensor_sha(image), template_tokens=template,
                  passage_emb=passage.contiguous(), passage_sha256=R.tensor_sha(passage),
                  processor_input_frame='DECODED_RAW_BEFORE_EXIF', image_metadata=metadata))
            if (i + 1) % 20 == 0: print(json.dumps(dict(encoded_references=i + 1, total=390)), flush=True)
    assert len(records) == 390 and [r['physical_row'] for r in records] == list(range(390))
    return dict(status='ISIC390_REFERENCE_TOKENS_SEALED', records=records, authority=bind(AUTH),
                legacy_bridge=bind(OUT / 'legacy_bridge.json'), GPU_runtime=gpu,
                runtime_libraries=R.runtime_libraries(), external_reference_forwards=390,
                external_query_forwards=0, head_predictions=0, training_updates=0)


def validate(a):
    receipt = read(OUT / 'receipt.json')
    assert receipt['authority'] == bind(AUTH)
    value = torch.load(checked(receipt['payload']), map_location='cpu', weights_only=True)
    expected = read(checked(a['sources']['gallery']))['records']
    assert len(expected) == len(value['records']) == 390
    for row, entry in zip(value['records'], expected):
        assert row['physical_row'] == entry['physical_row'] and row['source_path'] == entry['image_path']
        assert R.sha(Path(row['source_path'])) == row['source_image_sha256']
        assert row['tokens'].dtype == row['passage_emb'].dtype == torch.float16
        assert row['tokens'].shape == (row['grid_shape'][0] * row['grid_shape'][1], 128)
        assert R.tensor_sha(row['tokens']) == row['tokens_sha256'] and R.tensor_sha(row['passage_emb']) == row['passage_sha256']
        assert torch.isfinite(row['tokens']).all() and torch.isfinite(row['passage_emb']).all()
    write(OUT / 'validation.json', dict(status='ISIC390_REFERENCE_CPU_INTEGRITY_PASS', authority=bind(AUTH),
          payload=receipt['payload'], receipt=bind(OUT / 'receipt.json'), legacy_bridge=value['legacy_bridge'],
          reference_count=390, external_query_forwards=0, head_predictions=0))
    print('ISIC390_REFERENCE_CPU_INTEGRITY_PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['encode', 'validate'])
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(17)
    a = guard()
    if args.stage == 'validate': validate(a)
    else:
        value = compute(a)
        path = OUT / 'payload.pt'
        with path.open('xb') as f: torch.save(value, f)
        write(OUT / 'receipt.json', dict(status=value['status'], authority=bind(AUTH), payload=bind(path), legacy_bridge=value['legacy_bridge']))
        subprocess.run([sys.executable, __file__, 'validate'], check=True)
