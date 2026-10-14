import argparse
from collections import OrderedDict
import math
import sys
import time

import torch
from torch.nn import functional as F

from .common import RUNS, ROOT, WORK, RESULTS, Projection, hidden_file, key, read, save, signature
from .features import Banks
from .heads import cost, evaluate, features, fit, load_rows, wait_rows

sys.path.insert(0, str(ROOT / 'drugpack/method/legacy'))
from rc_prellm_m_scaled_adapter_v4 import ScaledQualityResidualAdapter


class Post(torch.nn.Module):
    def __init__(self, config, normalization):
        super().__init__()
        torch.manual_seed(config['seed'])
        self.dimension = 128 if config.get('location') == 'embedding' else 3584
        self.adapter = ScaledQualityResidualAdapter(hidden_size=self.dimension, bottleneck=config['bottleneck'],
            conditioning='constant' if config['arm'] == 'constant' else 'real',
            condition_gain=math.sqrt(self.dimension), mass_log_mean=normalization['mean'],
            mass_log_std=normalization['std'], mass_epsilon=1e-8, residual_scale=config['scale'])
        self.projection = Projection(torch.load(WORK/'projection.pt', weights_only=True)) if self.dimension == 3584 else torch.nn.Identity()
        self.config = config

    def forward(self, hidden, mask, masses):
        h = hidden.unsqueeze(0).expand(len(masses), -1, -1).clone()
        values = h[:, mask].reshape(-1, self.dimension)
        condition = masses.reshape(-1) if masses.ndim == 2 else masses.repeat_interleave(int(mask.sum()))
        assert len(condition) == len(values)
        h[:, mask] = self.adapter(values, condition).reshape(len(masses), -1, self.dimension)
        return F.normalize(self.projection(h)[:, mask].float(), dim=-1)


class Inputs:
    def __init__(self, manifest, device):
        self.manifest, self.device = manifest, device
        self.bank = Banks(manifest, device)
        self.hidden, self.refs, self.supports = OrderedDict(), OrderedDict(), OrderedDict()
        self.fresh_queries = {}

    def get(self, row):
        token = key(row)
        if token not in self.hidden:
            value = torch.load(hidden_file(self.manifest,row), weights_only=True)
            assert value['signature'] == self.manifest.get('hidden_signature',self.manifest['signature'])
            self.hidden[token] = value
        result = self.hidden.pop(token)
        self.hidden[token] = result
        while len(self.hidden) > 800:
            self.hidden.popitem(last=False)
        return result['hidden'].to(self.device), result['mask'].to(self.device)

    def reference(self, index):
        if index not in self.refs:
            self.refs[index] = self.bank.visual(index, True)
        result = self.refs.pop(index)
        self.refs[index] = result
        while len(self.refs) > 1024:
            self.refs.popitem(last=False)
        return result

    def patch_support(self, row):
        token = key(row)
        if token not in self.supports:
            source = torch.load(WORK/self.manifest['dataset']/'roma512'/(token+'.pt'),weights_only=True)
            self.supports[token] = dict(candidates=source['candidates'],values=torch.stack([p['u'] for p in source['pairs']]))
        result = self.supports.pop(token)
        self.supports[token] = result
        while len(self.supports) > 800:
            self.supports.popitem(last=False)
        positions = {r:i for i,r in enumerate(result['candidates'])}
        return result['values'][[positions[r] for r in row['candidates']]].float().to(self.device)


def masses_for(item, arm, device, intervention=None, patch=None):
    masses = item['features']['values'][:, 1].to(device) if patch is None else patch
    if arm == 'shuffled' or intervention == 'shuffled':
        physical = torch.tensor(item['row']['candidates'], device=device).argsort()
        changed = masses.clone()
        changed[physical] = masses[physical].roll(-1,dims=0)
        masses = changed
    return masses


def score_positions(model, inputs, item, positions, intervention=None):
    if model.dimension == 128:
        h = inputs.fresh_queries.get(item['row']['id'])
        if h is None:
            h = inputs.bank.visual(item['row']['query_index'])
        mask = torch.ones(len(h),dtype=torch.bool,device=h.device)
    else:
        h, mask = inputs.get(item['row'])
    patch = inputs.patch_support(item['row']) if model.config.get('condition') == 'patch' else None
    masses = masses_for(item, model.config['arm'], inputs.device, intervention,patch)
    original = model.adapter.conditioning
    if intervention == 'constant':
        model.adapter.conditioning = 'constant'
    try:
        tokens = model(h, mask, masses[positions])
    finally:
        model.adapter.conditioning = original
    refs = [inputs.reference(item['row']['candidates'][p]) for p in positions]
    length = max(len(r) for r in refs)
    padded = torch.zeros((len(refs), length, 128), device=inputs.device)
    valid = torch.zeros((len(refs), length), dtype=torch.bool, device=inputs.device)
    for i, r in enumerate(refs):
        padded[i, :len(r)] = r
        valid[i, :len(r)] = True
    similarities = torch.bmm(tokens, padded.transpose(1, 2))
    similarities = similarities.masked_fill(~valid[:, None], -torch.inf)
    return similarities.max(-1).values.mean(-1)


@torch.no_grad()
def scores(model, inputs, item, batch=8, intervention=None):
    values = []
    n = len(item['row']['candidates'])
    for start in range(0, n, batch):
        values.append(score_positions(model, inputs, item, list(range(start,min(start+batch,n))), intervention))
    return torch.cat(values).double()


def update(model, inputs, item, theta, optimizer, loss, batch):
    start = time.monotonic()
    content = scores(model, inputs, item, batch).requires_grad_()
    raw = item['features']['raw'].to(inputs.device)
    vals = item['features']['values'].to(inputs.device)
    objective = cost(features(raw, vals, content=content) @ theta, item['target'], loss)
    dc, dh = torch.autograd.grad(objective, (content, theta))
    active = dc.nonzero().flatten().tolist()
    optimizer.zero_grad(set_to_none=True)
    theta.grad = dh.detach()
    replay_error = 0.
    for start_pos in sorted({p//batch*batch for p in active}):
        pos = list(range(start_pos,min(start_pos+batch,len(content))))
        score = score_positions(model, inputs, item, pos)
        replay_error = max(replay_error, float((score.detach().double()-content[pos].detach()).abs().max()))
        (score * dc[pos].detach()).sum().backward()
    if replay_error > 2e-5:
        raise ValueError(f'Gradient replay differs: {replay_error}')
    parameters = list(model.adapter.parameters())
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError('Non-finite adapter gradient')
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
    torch.nn.utils.clip_grad_norm_([theta], 1.)
    optimizer.step()
    return dict(loss=float(objective.detach()), gradient_norm=norm, replay_error=replay_error,
                active=len(active), replay_layout='original_forward_batches', seconds=time.monotonic()-start)


@torch.inference_mode()
def validate(model, inputs, rows, theta, batch, intervention=None):
    contents = [scores(model, inputs, r, batch, intervention).cpu() for r in rows]
    reports = {}
    for k in (8,16,32,64,128):
        report, predictions = evaluate(rows, theta.detach().cpu(), 'INTERNAL3', k, contents=contents)
        reports[str(k)] = dict(metrics=report, predictions=predictions)
    return reports, contents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['real','constant','shuffled'], default='real')
    parser.add_argument('--dataset', default='medicine')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--loss', choices=['cost','ce'], default='cost')
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--bottleneck', type=int, default=16)
    parser.add_argument('--location', choices=['hidden','embedding'], default='hidden')
    parser.add_argument('--condition', choices=['scalar','patch'], default='scalar')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    while True:
        try:
            if args.smoke:
                manifest = read(WORK/'medicine/manifest.json')
                labels = read(WORK/'medicine/labels.json')
                train = []
                for row in manifest['rows']:
                    path = WORK/'medicine/features'/(key(row)+'.pt')
                    if row['split'] == 'train' and path.exists():
                        f = torch.load(path,weights_only=True)
                        pos = [i for i,r in enumerate(row['candidates']) if r in labels[row['id']]]
                        train.append(dict(row=row,features=f,positions=pos,target=pos[0] if pos else -1))
                    if len(train) == 4:
                        break
                assert len(train) == 4
                val = []
            else:
                manifest, train = load_rows(args.dataset,'train',args.budget)
                val = []
            break
        except FileNotFoundError:
            if not args.wait:
                raise
            time.sleep(20)
    config = dict(arm=args.arm,seed=args.seed,loss=args.loss,budget=args.budget,epochs=args.epochs,
                  bottleneck=args.bottleneck,location=args.location,condition=args.condition,scale=.1,adapter_lr=3e-4,head_lr=.03,
                  dataset=args.dataset,manifest=manifest['signature'],feature_recipe=train[0]['features']['support_signature'],
                  batch=args.batch,train=[r['row']['id'] for r in train])
    if args.condition == 'patch':
        total_values,total_sum,total_squared = 0,0.,0.
        for item in train:
            source = torch.load(WORK/args.dataset/'roma512'/(key(item['row'])+'.pt'),weights_only=True)
            assert source['candidates'] == item['row']['candidates']
            logs = torch.stack([p['u'] for p in source['pairs']]).double().clamp_min(1e-8).log()
            total_values += logs.numel()
            total_sum += float(logs.sum())
            total_squared += float(logs.square().sum())
        mean = total_sum/total_values
        normalization = dict(mean=mean,std=math.sqrt(max(total_squared/total_values-mean*mean,1e-12)))
    else:
        logs = torch.cat([r['features']['values'][:,1] for r in train]).double().clamp_min(1e-8).log()
        normalization = dict(mean=float(logs.mean()),std=float(logs.std(unbiased=False)))
    run = f'post_{args.arm}_{args.loss}_s{args.seed}_b{args.budget}_d{args.bottleneck}'
    if args.location == 'embedding':
        run = 'embedding_' + run
    if args.condition == 'patch':
        run = 'patch_' + run
    if args.dataset != 'medicine':
        run = args.dataset + '_' + run
    folder = (RESULTS/'smoke' if args.smoke else RUNS)/run
    sig = signature(config)
    if (folder/'complete.json').exists():
        assert read(folder/'complete.json')['signature'] == sig
        return
    model = Post(config, normalization).to('cuda:0')
    inputs = Inputs(manifest, 'cuda:0')
    warm = fit(train, 'INTERNAL3', args.loss)
    theta = torch.nn.Parameter(warm.to('cuda:0'))
    optimizer = torch.optim.AdamW([dict(params=model.adapter.parameters(),lr=3e-4),
                                  dict(params=[theta],lr=.03)],weight_decay=.001)
    step, best, history = 0, (-1.,-1.), []
    checkpoint = folder/'resume.pt'
    def snapshot():
        return dict(signature=sig,config=config,normalization=normalization,
                    adapter={k:v.detach().cpu() for k,v in model.adapter.state_dict().items()},
                    theta=theta.detach().cpu(),step=step,best=best,history=history)
    if checkpoint.exists() and not args.smoke:
        previous = torch.load(checkpoint,weights_only=True)
        assert previous['signature'] == sig
        model.adapter.load_state_dict(previous['adapter'])
        theta.data.copy_(previous['theta'].to(theta.device))
        optimizer.load_state_dict(previous['optimizer'])
        step,best,history = previous['step'],tuple(previous['best']),previous['history']
    else:
        initial = scores(model,inputs,train[0],args.batch).cpu()
        error = float((initial-train[0]['features']['values'][:,0].double()).abs().max())
        if error > 2e-5:
            raise ValueError(f'Zero-adapter native score parity failed: {error}')
        save(folder/'initialization.json',dict(signature=sig,zero_score_error=error,
              parameters=sum(p.numel() for p in model.adapter.parameters()),normalization=normalization))
    total = 4 if args.smoke else len(train)*args.epochs
    while step < total:
        epoch,offset = divmod(step,len(train))
        order = torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        item = train[order[offset]]
        record = update(model,inputs,item,theta,optimizer,args.loss,args.batch)
        step += 1
        record.update(step=step,epoch=epoch+1,id=item['row']['id'])
        history.append(record)
        print(run,record,flush=True)
        if step % 25 == 0 or step == total:
            save(checkpoint,dict(**snapshot(),optimizer=optimizer.state_dict()))
            save(folder/'progress.json',dict(step=step,total=total,last=record,updated=time.time()))
        if not args.smoke and step % len(train) == 0 and (epoch+1) % 2 == 0:
            if not val:
                _, val = wait_rows(args.dataset, 'val')
            validations, contents = validate(model,inputs,val,theta,args.batch)
            metric = max((v['metrics']['macro']['R@1'],v['metrics']['macro']['MRR']) for v in validations.values())
            save(folder/f'val_epoch{epoch+1}.json',validations)
            if metric > best:
                best = metric
                save(folder/'best.pt',snapshot())
                save(folder/'best_validation.json',validations)
                save(folder/'best_validation_contents.pt',contents)
            save(checkpoint,dict(**snapshot(),optimizer=optimizer.state_dict()))
    save(folder/'final.pt',snapshot())
    save(folder/'complete.json',dict(signature=sig,config=config,best=best,step=step,
                                    mean_update_seconds=sum(x['seconds'] for x in history)/len(history)))


if __name__ == '__main__':
    main()
