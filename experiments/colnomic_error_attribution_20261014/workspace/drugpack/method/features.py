from evaluation.paths import project_path
import argparse
import os
import time

from PIL import Image
import torch
from torch.nn import functional as F

from .common import ROOT, WORK, claim, hidden_file, key, load_bank, read, save


class Banks:
    def __init__(self, manifest, device='cpu'):
        self.manifest, self.device = manifest, device
        self.q = load_bank(manifest['source']['query_bank'])
        self.r = load_bank(manifest['source']['reference_bank'])
        assert self.q['ids'] == [x['original_id'] for x in manifest['rows']]
        assert self.r['ids'] == [x['id'] for x in manifest['references']]
        from colpali_engine.models import ColQwen2_5_Processor
        self.processor = ColQwen2_5_Processor.from_pretrained(ROOT / 'weights/colnomic7b/adapter', local_files_only=True)
        self.layout = {}

    def visual(self, index, reference=False):
        bank = self.r if reference else self.q
        record = self.manifest['references' if reference else 'rows'][index]
        marker = (reference, index)
        if marker not in self.layout:
            from .token_layout import token_layout
            with Image.open(project_path(record['path'])) as im:
                self.layout[marker] = token_layout(self.processor, 'colnomic7b', *im.size)
        layout = self.layout[marker]
        z = bank['embeddings'][index]
        assert len(z) == layout['total']
        return F.normalize(z[layout['start']:layout['start'] + layout['count']].to(self.device).float(), dim=-1)


@torch.inference_mode()
def extract(row, bank, support, hidden=None):
    if hidden is None:
        hidden = torch.load(hidden_file(bank.manifest,row), weights_only=True)
    assert hidden['id'] == row['id'] and hidden['signature'] == bank.manifest.get('hidden_signature',bank.manifest['signature'])
    if 'cached_embedding_max_error' in hidden:
        assert hidden['cached_embedding_max_error'] < 1e-6, f'Query embedding parity failed: {row["id"]}'
    q = F.normalize(hidden['native'][hidden['mask']].to(bank.device).float(), dim=-1)
    values = []
    assert support['candidates'] == row['candidates']
    for j, reference in enumerate(row['candidates']):
        r = bank.visual(reference, True)
        pair = support['pairs'][j]
        u, v = pair['u'].to(bank.device).float(), pair['v'].to(bank.device).float()
        assert len(u) == len(q) and len(v) == len(r)
        c = q @ r.T
        l0 = c.max(1).values.mean()
        local = (c * v[None]).max(1).values
        weighted = (u * local).sum() / u.sum().clamp_min(1e-8)
        qcontrol = (u.roll(1) * local).sum() / u.sum().clamp_min(1e-8)
        rcontrol = (u * (c * v.roll(1)[None]).max(1).values).sum() / u.sum().clamp_min(1e-8)
        values.append([float(l0), pair['M'], float(weighted), float(weighted) * pair['M'],
                       float(weighted-qcontrol), float(weighted-rcontrol),
                       float(u.mean()), float(v.mean())])
    x = torch.tensor(values, dtype=torch.float32)
    assert torch.isfinite(x).all()
    return dict(id=row['id'], signature=bank.manifest['signature'], candidates=row['candidates'],
                columns=['L0', 'M', 'weighted_L', 'S', 'query_roll_delta', 'reference_roll_delta', 'mean_u', 'mean_v'],
                values=x, raw=torch.tensor(row['raw_scores']), support_signature=support['signature'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['medicine', 'grozi', 'milk'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    banks = {}
    while True:
        pending = 0
        for dataset in args.datasets:
            manifest = read(WORK / dataset / 'manifest.json')
            for row in manifest['rows']:
                path = WORK / dataset / 'features' / (key(row) + '.pt')
                if path.exists():
                    continue
                support_path = WORK / dataset / 'roma512' / (key(row) + '.pt')
                hidden_path = hidden_file(manifest,row)
                if not support_path.exists() or not hidden_path.exists():
                    pending += 1
                    continue
                with claim(path.with_suffix('.lock')) as acquired:
                    if not acquired or path.exists():
                        pending += int(not path.exists())
                        continue
                    if dataset not in banks:
                        banks[dataset] = Banks(manifest, args.device)
                    support = torch.load(support_path, weights_only=True)
                    start = time.monotonic()
                    value = extract(row, banks[dataset], support)
                    value['seconds'] = time.monotonic()-start
                    save(path, value)
                    print(dataset, row['query_index'], round(value['seconds'], 3), flush=True)
        if not args.watch or not pending:
            break
        time.sleep(10)


if __name__ == '__main__':
    main()
