from evaluation.paths import project_path
import argparse
import os
import socket
import time
from types import SimpleNamespace

from PIL import Image
import torch

from .common import ROOT, WORK, Projection, claim, key, load_bank, read, save, sha


def single_encoder(config):
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from peft import PeftModel
    from evaluation.encoders import stable_sdpa
    ALL_ATTENTION_FUNCTIONS.register('sdpa', stable_sdpa)
    base, info = ColQwen2_5.from_pretrained(ROOT/config['base_checkpoint'], torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
    assert not info.get('missing_keys') and not info.get('mismatched_keys')
    assert set(info.get('unexpected_keys',[])) <= {'lm_head.weight'}
    model = PeftModel.from_pretrained(base, ROOT/config['checkpoint'], local_files_only=True).to('cuda:0').eval()
    processor = ColQwen2_5_Processor.from_pretrained(ROOT/config['checkpoint'], local_files_only=True)
    return SimpleNamespace(model=model, processor=processor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['medicine', 'grozi', 'milk'])
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--max-queries', type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    from evaluation.encoders import Encoder
    config = read(ROOT / 'configs/baselines.json')['models']['colnomic7b']
    encoder = None
    captured = {}
    completed = 0
    for dataset in args.datasets:
        manifest = read(WORK / dataset / 'manifest.json')
        bank = load_bank(manifest['source']['query_bank'])
        assert bank['ids'] == [r['original_id'] for r in manifest['rows']]
        for row in manifest['rows']:
            path = WORK / dataset / 'hidden' / (key(row) + '.pt')
            if path.exists():
                continue
            with claim(path.with_suffix('.lock')) as acquired:
                if not acquired or path.exists():
                    continue
                if encoder is None:
                    encoder = single_encoder(config) if args.single else Encoder('colnomic7b', config, ROOT, 'cuda:0')
                    encoder.model.requires_grad_(False)
                    projection = encoder.model.base_model.model.custom_text_proj
                    projection.register_forward_pre_hook(lambda m, inp: captured.update(hidden=inp[0].detach()))
                    state = dict(weight=projection.base_layer.weight.detach().cpu(),
                                 bias=projection.base_layer.bias.detach().cpu(),
                                 a=projection.lora_A['default'].weight.detach().cpu(),
                                 b=projection.lora_B['default'].weight.detach().cpu(),
                                 scaling=float(projection.scaling['default']))
                    fixed = Projection(state).to('cuda:0').eval()
                    with claim(WORK / 'projection.lock') as held:
                        if held and not (WORK / 'projection.pt').exists():
                            save(WORK / 'projection.pt', state)
                    print('Loaded ColNomic 7B', flush=True)
                assert sha(row['path']) == row['sha256']
                start = time.monotonic()
                with Image.open(project_path(row['path'])) as im:
                    inputs = encoder.processor.process_images([im.convert('RGB')]).to('cuda:0')
                with torch.inference_mode():
                    z = encoder.model(**inputs, use_cache=False)[0]
                    h = captured.pop('hidden')[0]
                    mask = inputs['input_ids'][0].eq(encoder.model.config.image_token_id)
                    actual = fixed(h)
                    error = float((actual.float() - z.float()).abs().max())
                    if error != 0:
                        raise ValueError(f'Frozen projection parity failed: {error}')
                    old = bank['embeddings'][row['query_index']]
                    assert old.shape == z.shape
                    cache_error = float((old.float() - z.cpu().float()).abs().max())
                    cosine = float(torch.nn.functional.cosine_similarity(old.float(), z.cpu().float()).mean())
                    if not cache_error < 1e-6:
                        raise ValueError(f'Existing embedding parity failed for {row["id"]}: {cache_error}')
                save(path, dict(id=row['id'], signature=manifest['signature'], sha256=row['sha256'],
                                hidden=h.cpu(), mask=mask.cpu(), native=z.cpu(),
                                grid=inputs['image_grid_thw'].cpu(), projection_error=error,
                                cached_embedding_max_error=cache_error, cached_embedding_cosine=cosine,
                                seconds=time.monotonic()-start, host=socket.gethostname()))
                print(dataset, row['query_index'], row['id'], round(time.monotonic()-start, 3), cache_error, flush=True)
                completed += 1
                if args.max_queries and completed >= args.max_queries:
                    return
    print('HIDDEN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
