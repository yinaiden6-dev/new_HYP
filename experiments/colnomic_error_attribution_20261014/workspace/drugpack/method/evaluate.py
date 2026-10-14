from evaluation.paths import project_path
import argparse
from pathlib import Path
import socket
import time

import torch

from .common import RESULTS, WORK, claim, evaluation_group, key, read, save, sha, signature
from .heads import features, intervene_mass, mass_columns
from .ranking import baseline_ranks, restored_rank, summarize


def item_for(manifest, row):
    value = torch.load(WORK/manifest['dataset']/'features'/(key(row)+'.pt'), weights_only=True)
    assert value['id'] == row['id'] and value['signature'] == manifest['signature']
    assert value['candidates'] == row['candidates']
    return dict(row=row,features=value)


class Predictor:
    def __init__(self, manifest, run, device):
        self.manifest, self.run = manifest, run
        self.model = self.inputs = None
        self.device = device
        self.state = torch.load(project_path(run['checkpoint']),weights_only=True,map_location='cpu')
        assert sha(run['checkpoint']) == run['checkpoint_sha256']
        self.config = self.state['config']
        self.theta = self.state['theta'].double()
        if 'adapter' in self.state:
            from .post import Post, Inputs
            self.model = Post(self.config,self.state['normalization']).to(device).eval()
            self.model.adapter.load_state_dict(self.state['adapter'])
            self.inputs = Inputs(manifest,device)
            self.kind = 'post'
        elif 'mlp' in self.state:
            from .scalar_head import ScalarHead
            self.model = ScalarHead(self.config['width'], self.config['seed'], self.state['normalization']).to(device).eval()
            self.model.load_state_dict(self.state['mlp'])
            self.kind = 'scalar_head'
        elif 'model' in self.state:
            from .reader import Reader, Inputs
            self.model = Reader(self.config['arm'],self.config['seed']).to(device).eval()
            self.model.load_state_dict(self.state['model'])
            self.inputs = Inputs(manifest,device)
            self.kind = 'reader'
        else:
            self.kind = 'external'

    @torch.inference_mode()
    def predict(self, item, k, intervention):
        n = min(k,len(item['row']['candidates']))
        row = dict(item['row'],candidates=item['row']['candidates'][:n])
        value = item['features']
        vals = value['values'][:n].clone()
        sliced = dict(item,row=row,features=dict(value,raw=value['raw'][:n],values=vals))
        if self.kind == 'post':
            from .post import scores
            source = item if self.config['arm'] == 'shuffled' or intervention == 'shuffled' else sliced
            content = scores(self.model,self.inputs,source,8,intervention).cpu()[:n]
            logits = features(sliced['features']['raw'],vals,content=content) @ self.theta
        elif self.kind == 'reader':
            from .reader import scores, logits as reader_logits
            relation = scores(self.model,self.inputs,sliced,intervention or 'native').cpu()
            logits = reader_logits(sliced,relation,self.theta)
        elif self.kind == 'scalar_head':
            vals = intervene_mass(vals, row['candidates'], intervention)
            logits = self.model(sliced['features']['raw'].to(self.device), vals.to(self.device), self.theta.to(self.device))
        else:
            vals = intervene_mass(vals, row['candidates'], intervention, mass_columns(self.config['kind']))
            logits = features(sliced['features']['raw'],vals,self.config['kind']) @ self.theta
        scores = torch.cat([torch.zeros_like(logits[:1]),logits])
        assert torch.isfinite(scores).all(), (item['row']['id'], self.run['checkpoint'],
                                              torch.where(~torch.isfinite(scores))[0].tolist())
        return dict(scores=scores.cpu(),permutation=torch.argsort(scores,descending=True,stable=True).tolist())


def controls(item, k, name):
    n = min(k,len(item['row']['candidates']))
    columns = {'L0':0,'M':1,'weighted_L':2,'S':3,'U':6,'V':7}
    scores = item['features']['raw'][:n] if name == 'RAW' else item['features']['values'][:n,columns[name]]
    return dict(scores=scores,permutation=torch.argsort(scores,descending=True,stable=True).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--split',choices=['val','test','transfer'],required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--k',type=int)
    parser.add_argument('--intervention',choices=['constant','shuffled','shuffle','rotate'])
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--query-start',type=int)
    parser.add_argument('--query-stop',type=int)
    parser.add_argument('--aggregate-only',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = read(WORK/args.dataset/'manifest.json')
    source = None
    is_control = args.run in ('RAW','L0','M','weighted_L','S','U','V')
    if not is_control or args.split != 'val':
        while not (RESULTS/'selection.json').exists():
            if not args.wait:
                raise RuntimeError('Freeze medicine validation selection before test or transfer')
            time.sleep(20)
        selection = read(RESULTS/'selection.json')
        if not is_control:
            source = selection['runs'][args.run]
    k = args.k or (source['candidate_k'] if source else 128)
    if args.dataset == 'forb32':
        k = 32
    assert 1 < k <= manifest['candidates']
    labels = read(WORK/args.dataset/'labels.json')
    rows = [r for r in manifest['rows'] if r['split'] == args.split]
    total_rows = len(rows)
    partial = args.query_start is not None or args.query_stop is not None
    if partial:
        assert args.query_start is not None and args.query_stop is not None and not args.aggregate_only
        assert 0 <= args.query_start < args.query_stop <= len(rows)
        rows = rows[args.query_start:args.query_stop]
    config = dict(dataset=args.dataset,split=args.split,run=args.run,k=k,intervention=args.intervention,
                  data_signature=manifest['signature'],labels_sha256=sha(WORK/args.dataset/'labels.json'),
                  checkpoint_sha256=source['checkpoint_sha256'] if source else None)
    sig = signature(config)
    name = f'{args.run}_k{k}' + ('_'+args.intervention if args.intervention else '')
    folder = RESULTS/'evaluation'/args.dataset/args.split/name
    def finish_part():
        save(folder/'shards'/f'{args.query_start}_{args.query_stop}.json',
             dict(signature=sig,query_start=args.query_start,query_stop=args.query_stop,
                  ids=[r['id'] for r in rows],total_queries=total_rows))
    if (folder/'complete.json').exists():
        assert read(folder/'complete.json')['signature'] == sig
        if partial:
            finish_part()
        return
    if args.aggregate_only:
        assert all((folder/'predictions'/(key(r)+'.pt')).exists() for r in rows), 'Cannot aggregate missing queries'
    raw_ranks = baseline_ranks(manifest)
    positive_ranks = read(WORK/args.dataset/'full_ranks.json')['positive_ranks']
    predictor = None
    while True:
        pending = 0
        for row in rows:
            path = folder/'predictions'/(key(row)+'.pt')
            if path.exists():
                continue
            pending += 1
            feature = WORK/args.dataset/'features'/(key(row)+'.pt')
            if not feature.exists() and args.run != 'RAW':
                if not args.wait:
                    raise FileNotFoundError(feature)
                continue
            with claim(path.with_suffix('.lock')) as held:
                if not held or path.exists():
                    continue
                if predictor is None and not is_control:
                    predictor = Predictor(manifest,source,args.device)
                item = (dict(row=row,features=dict(raw=torch.tensor(row['raw_scores'])))
                        if args.run == 'RAW' else item_for(manifest,row))
                if predictor and predictor.device != 'cpu':
                    torch.cuda.synchronize()
                started = time.perf_counter()
                pred = controls(item,k,args.run) if is_control else predictor.predict(item,k,args.intervention)
                if predictor and predictor.device != 'cpu':
                    torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                candidates = row['candidates'][:k]
                truth = set(labels[row['id']])
                rank = restored_rank(raw_ranks[row['id']],candidates,truth,pred['permutation'])
                reranked = {str(candidates[p]):i+1 for i,p in enumerate(pred['permutation'])}
                full_positive_ranks = {p:reranked.get(p,r) for p,r in positive_ranks[row['id']].items()}
                assert rank == min(full_positive_ranks.values())
                save(path,dict(signature=sig,id=row['id'],group=evaluation_group(args.dataset,row),identity=row['identity'],
                     domain=row.get('domain'),difficulty=row.get('difficulty'),rank=rank,
                     raw_rank=raw_ranks[row['id']],target_in_candidates=bool(truth.intersection(candidates)),
                     positive_ranks=full_positive_ranks,
                     top_reference_indices=([candidates[i] for i in pred['permutation']] + row['candidates'][len(candidates):])[:10],
                     permutation=pred['permutation'],scores=pred['scores'],seconds=elapsed,
                     worker=dict(host=socket.gethostname(),device=args.device,
                                 hardware=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU')))
        if all((folder/'predictions'/(key(r)+'.pt')).exists() for r in rows):
            break
        if not args.wait:
            raise RuntimeError(f'{pending} predictions incomplete')
        time.sleep(20)
    predictions = [torch.load(folder/'predictions'/(key(r)+'.pt'),weights_only=True) for r in rows]
    assert all(p['signature'] == sig and p['id'] == r['id'] for p,r in zip(predictions,rows,strict=True))
    if partial:
        finish_part()
        print('SHARD_COMPLETE',args.run,args.query_start,args.query_stop,flush=True)
        return
    metrics = summarize(predictions)
    if args.dataset.startswith('forb'):
        metrics['by_domain'] = {d:summarize([p for p in predictions if p['domain']==d],False)
                                for d in sorted({p['domain'] for p in predictions})}
    save(folder/'complete.json',dict(signature=sig,config=config,metrics=metrics,
         timing=dict(scope='cached feature reranking on the listed workers, including per-query token transfer/loading; excludes image encoder, first-stage retrieval and RoMa',
                     workers=list({signature(p.get('worker',{})):p.get('worker',{}) for p in predictions}.values()),
                     total_seconds=sum(p['seconds'] for p in predictions),
                     seconds_per_query=sum(p['seconds'] for p in predictions)/len(predictions))))
    save(folder/'predictions.json',[{k:v for k,v in p.items() if k not in ('scores','signature')} for p in predictions])
    print(args.dataset,args.run,k,metrics,flush=True)


if __name__ == '__main__':
    main()
