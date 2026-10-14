from evaluation.paths import project_path
import argparse
import os
import socket
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence

from .common import ROOT, WORK, RESULTS, key, load_bank, read, save
from .evaluate import Predictor
from .features import Banks, extract


def synchronize():
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


@torch.inference_mode()
def maxsim(query,references,device):
    query = query[query.float().norm(dim=-1)>1e-8].float().to(device)
    result = []
    for start in range(0,len(references),16):
        chunk = references[start:start+16]
        lengths = torch.tensor([len(r) for r in chunk],device=device)
        ref = pad_sequence(chunk,batch_first=True).to(device)
        valid = torch.arange(ref.shape[1],device=device)[None] < lengths[:,None]
        sim = torch.matmul(query,ref.transpose(1,2)).masked_fill(~valid[:,None],-torch.inf)
        result.append(sim.max(-1).values.mean(-1).cpu())
    return torch.cat(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run')
    parser.add_argument('--suite',action='store_true')
    parser.add_argument('--queries',type=int,default=10)
    parser.add_argument('--single-encoder',action='store_true')
    args = parser.parse_args()
    if args.suite:
        assert args.run is None
        selected = read(RESULTS/'selection.json')
        runs = [selected['primary']['strict'],selected['selected']['strict/external'],
                selected['selected']['strict/reader']]
        for name in runs:
            command = ['bash',str(ROOT/'scripts/forb_python.sh'),'-m','drugpack.method.benchmark',
                       '--run',name,'--queries',str(args.queries)]
            if args.single_encoder:
                command.append('--single-encoder')
            subprocess.run(command,cwd=ROOT,check=True)
        return
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    roma_device = 1 if args.single_encoder else 2
    assert torch.cuda.device_count() == roma_device+1, 'Expose only the GPUs reserved for this benchmark'
    selection = read(RESULTS/'selection.json')
    run_name = args.run or selection['primary']['strict']
    run = selection['runs'][run_name]
    manifest = read(WORK/'medicine'/'manifest.json')
    k = run['candidate_k']
    val = [r for r in manifest['rows'] if r['split']=='val']
    rng = np.random.default_rng(20260925)
    rows = [val[i] for i in rng.permutation(len(val))[:args.queries]]
    folder = RESULTS/'timing'/run_name
    save(folder/'selection.json',dict(query_ids=[r['id'] for r in rows],seed=20260925,
                                    selection='uniform validation query sample without labels'))
    bank = load_bank(manifest['source']['reference_bank'])
    refs = [bank['embeddings'][i] for i in range(len(manifest['references']))]
    refs = [r[r.float().norm(dim=-1)>1e-8].float() for r in refs]
    load_start = time.perf_counter()
    config = read(ROOT/'configs/baselines.json')['models']['colnomic7b']
    if args.single_encoder:
        from .hidden import single_encoder
        encoder = single_encoder(config)
    else:
        from evaluation.encoders import Encoder
        encoder = Encoder('colnomic7b',config,ROOT,'cuda:0')
    encoder.model.requires_grad_(False)
    captured = {}
    encoder.model.base_model.model.custom_text_proj.register_forward_pre_hook(
        lambda m,inp:captured.update(hidden=inp[0].detach()))
    predictor = Predictor(manifest,run,'cuda:0')
    needs_support = (predictor.kind=='post' and predictor.config['arm']!='constant') or (
        predictor.kind in ('external','scalar_head') and predictor.config['kind']!='INTERNAL3')
    torch.cuda.set_device(roma_device)
    matcher = None
    if needs_support:
        from .roma import Matcher
        matcher = Matcher(amp=read(ROOT/'configs/method/protocol.json')['roma']['precision']=='fp16')
    feature_bank = Banks(manifest,f'cuda:{roma_device}')
    synchronize()
    loading_seconds = time.perf_counter()-load_start

    @torch.inference_mode()
    def encode(row):
        with Image.open(project_path(row['path'])) as source:
            inputs = encoder.processor.process_images([source.convert('RGB')]).to('cuda:0')
        tokens = encoder.model(**inputs,use_cache=False)[0]
        hidden = dict(id=row['id'],signature=manifest['signature'],hidden=captured.pop('hidden')[0].cpu(),
                      mask=inputs['input_ids'][0].eq(encoder.model.config.image_token_id).cpu(),
                      native=tokens.cpu(),grid=inputs['image_grid_thw'].cpu())
        return tokens,hidden

    encode(rows[0])
    if matcher:
        matcher.pair(rows[0],manifest['references'][rows[0]['candidates'][0]])
    synchronize()
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device)
    records = []
    original = torch.load(project_path(manifest['source']['score_bank']),weights_only=True,mmap=True)['scores']
    for row in rows:
        if matcher:
            for module in (matcher.model.f,matcher.model.refiner_features):
                module.values.clear()
                module.used = 0
            matcher.images.clear()
        synchronize()
        started = time.perf_counter()
        tokens,hidden = encode(row)
        synchronize(); encoded = time.perf_counter()
        raw = maxsim(tokens,refs,f'cuda:{roma_device}')
        gallery = torch.tensor(manifest['gallery_indices'])
        candidates = gallery[torch.argsort(raw[gallery],descending=True,stable=True)[:k]].tolist()
        fresh_row = dict(row,candidates=candidates,raw_scores=raw[candidates].tolist())
        synchronize(); retrieved = time.perf_counter()
        pairs = [matcher.pair(row,manifest['references'][r]) for r in candidates] if matcher else []
        synchronize(); matched = time.perf_counter()
        if matcher:
            support = dict(candidates=candidates,pairs=pairs,signature='fresh-online-benchmark')
            value = extract(fresh_row,feature_bank,support,hidden)
        else:
            q = torch.nn.functional.normalize(hidden['native'][hidden['mask']].to(feature_bank.device).float(),dim=-1)
            values = torch.zeros(len(candidates),8)
            for i,r in enumerate(candidates):
                values[i,0] = (q @ feature_bank.visual(r,True).T).max(-1).values.mean().cpu()
            value = dict(raw=torch.tensor(fresh_row['raw_scores']),values=values)
        synchronize(); pooled = time.perf_counter()
        if predictor.inputs:
            predictor.inputs.hidden[key(row)] = hidden
            predictor.inputs.fresh_queries[row['id']] = torch.nn.functional.normalize(
                hidden['native'][hidden['mask']].to('cuda:0').float(),dim=-1)
            if pairs:
                predictor.inputs.supports[key(row)] = dict(candidates=candidates,values=torch.stack([p['u'] for p in pairs]))
        prediction = predictor.predict(dict(row=fresh_row,features=value),k,None)
        synchronize(); finished = time.perf_counter()
        error = float((raw-original[row['query_index']]).abs().max())
        assert error < 1e-5, f'Fresh first-stage scoring mismatch: {error}'
        record = dict(id=row['id'],k=k,encoder=encoded-started,full_gallery_search=retrieved-encoded,
             roma=matched-retrieved,feature_pooling=pooled-matched,learned_reranking=finished-pooled,
             end_to_end=finished-started,raw_end_to_end=retrieved-started,score_parity_error=error,
             candidate_set_differences=len(set(candidates)^set(row['candidates'][:k])),
             top_reference=candidates[prediction['permutation'][0]])
        records.append(record)
        save(folder/'progress.json',records)
        print(record,flush=True)
    stages = ('encoder','full_gallery_search','roma','feature_pooling','learned_reranking','end_to_end','raw_end_to_end')
    result = dict(run=run_name,checkpoint_sha256=run['checkpoint_sha256'],host=socket.gethostname(),
          visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,
          gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
          model_loading_seconds=loading_seconds,queries=len(rows),gallery_rows=len(refs),unique_gallery=len(manifest['gallery_indices']),
          scope='Fresh image decode and ColNomic encoding, exact full-gallery MaxSim, fresh RoMa pairs, feature pooling and trained reranking. Fixed gallery embeddings and initialized models are resident; no cached query embeddings or pair outputs.',
          uses_roma=needs_support,
          support_cache='DINO/VGG/image LRUs cleared before each query; query features reused between candidates within that query.' if needs_support else 'RoMa is not loaded or executed.',
          batch=dict(queries=1,reference_scoring=16,post=8),k=k,records=records,
          seconds={s:dict(mean=float(np.mean([r[s] for r in records])),median=float(np.median([r[s] for r in records])),
                         p95=float(np.quantile([r[s] for r in records],.95))) for s in stages},
          peak_memory_bytes=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
    save(folder/'complete.json',result)


if __name__=='__main__':
    main()
