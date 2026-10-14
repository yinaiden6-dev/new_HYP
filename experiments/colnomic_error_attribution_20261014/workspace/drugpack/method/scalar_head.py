import argparse
import time

import torch

from .common import RUNS, RESULTS, read, save, signature
from .heads import cost, features, fit, load_rows, summary, wait_rows


class ScalarHead(torch.nn.Module):
    def __init__(self, width, seed, normalization):
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(5, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 1))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.normalization = normalization

    def forward(self, raw, values, theta):
        base = features(raw, values, 'ADDITIVE4')
        mass = (values[:, 1].double().clamp_min(1e-8).log() - self.normalization['mean']) / self.normalization['std']
        inputs = torch.cat((base[:, :3], mass[1:, None], mass[0].expand(len(base), 1)), -1)
        return base @ theta + self.net(inputs.float()).flatten().double()


@torch.inference_mode()
def validate(model, rows, theta):
    reports = {}
    for k in (8, 16, 32, 64, 128):
        orders = []
        for item in rows:
            value = item['features']
            n = min(k, len(value['raw']))
            logits = model(value['raw'][:n], value['values'][:n], theta)
            score = torch.cat((torch.zeros(1, dtype=logits.dtype), logits))
            order = torch.argsort(score, descending=True, stable=True).tolist()
            orders.append(order + list(range(n, len(value['raw']))))
        metrics, predictions = summary(rows, orders)
        reports[str(k)] = dict(metrics=metrics, predictions=predictions)
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='medicine')
    parser.add_argument('--width', type=int, choices=[61, 340], required=True)
    parser.add_argument('--loss', choices=['cost', 'ce'], default='cost')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--epochs', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest, train = wait_rows(args.dataset, 'train')
    masses = torch.cat([r['features']['values'][:, 1] for r in train]).double().clamp_min(1e-8).log()
    normalization = dict(mean=float(masses.mean()), std=float(masses.std(unbiased=False)))
    config = dict(kind='SCALAR_MLP', dataset=args.dataset, width=args.width, seed=args.seed,
        loss=args.loss, epochs=args.epochs, lr=3e-4, head_lr=.03, weight_decay=.001,
        manifest=manifest['signature'], feature_recipe=train[0]['features']['support_signature'],
        training_rows=[r['row']['id'] for r in train], inputs='relative RAW/M/L0; train-standardized log M of candidate and RAW winner')
    sig = signature(config)
    name = f'external_mlp_w{args.width}_{args.loss}_s{args.seed}'
    if args.dataset != 'medicine':
        name = args.dataset + '_' + name
    folder = RUNS/name
    if (folder/'complete.json').exists():
        assert read(folder/'complete.json')['signature'] == sig
        return
    model = ScalarHead(args.width, args.seed, normalization)
    theta = torch.nn.Parameter(fit(train, 'ADDITIVE4', args.loss))
    optimizer = torch.optim.AdamW([dict(params=model.parameters(), lr=3e-4),
                                  dict(params=[theta], lr=.03)], weight_decay=.001)
    step, best, history, val = 0, (-1., -1.), [], []
    checkpoint = folder/'resume.pt'
    def snapshot():
        return dict(signature=sig, config=config, normalization=normalization,
                    mlp=model.state_dict(), theta=theta.detach(), step=step, best=best, history=history)
    if checkpoint.exists():
        old = torch.load(checkpoint, weights_only=True)
        assert old['signature'] == sig
        model.load_state_dict(old['mlp']); theta.data.copy_(old['theta'])
        optimizer.load_state_dict(old['optimizer'])
        step, best, history = old['step'], tuple(old['best']), old['history']
    while step < len(train)*args.epochs:
        epoch, offset = divmod(step, len(train))
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        item = train[order[offset]]
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        value = item['features']
        loss = cost(model(value['raw'], value['values'], theta), item['target'], args.loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in (*model.parameters(), theta))
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        torch.nn.utils.clip_grad_norm_([theta], 1.)
        optimizer.step()
        step += 1
        record = dict(step=step, loss=float(loss.detach()), gradient_norm=norm, seconds=time.monotonic()-start)
        history.append(record)
        if step % 25 == 0 or step == len(train)*args.epochs:
            save(checkpoint, dict(**snapshot(), optimizer=optimizer.state_dict()))
            save(folder/'progress.json', dict(step=step, total=len(train)*args.epochs, last=record, updated=time.time()))
            print(name, record, flush=True)
        if step % len(train) == 0 and (epoch+1) % 2 == 0:
            if not val:
                _, val = wait_rows(args.dataset, 'val')
            reports = validate(model, val, theta)
            metric = max((v['metrics']['macro']['R@1'], v['metrics']['macro']['MRR']) for v in reports.values())
            save(folder/f'val_epoch{epoch+1}.json', reports)
            if metric > best:
                best = metric
                save(folder/'best.pt', snapshot())
                save(folder/'best_validation.json', reports)
            save(checkpoint, dict(**snapshot(), optimizer=optimizer.state_dict()))
    save(folder/'final.pt', snapshot())
    save(folder/'complete.json', dict(signature=sig, config=config, best=best, step=step,
        parameters=sum(p.numel() for p in model.parameters()),
        mean_update_seconds=sum(r['seconds'] for r in history)/len(history)))


if __name__ == '__main__':
    main()
