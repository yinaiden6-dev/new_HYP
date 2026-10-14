from evaluation.paths import project_path
import fcntl
import hashlib
import json
import os
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'cache/method'
RESULTS = ROOT / 'results/method'
RUNS = ROOT / 'weights/method'


def read(path):
    return json.loads(project_path(path).read_text())


def sha(path):
    with project_path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    if path.suffix == '.json':
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    else:
        torch.save(value, temp)
    os.replace(temp, path)


def key(row):
    return hashlib.sha256(row['id'].encode()).hexdigest()[:24]


def hidden_file(manifest, row):
    return WORK/manifest.get('hidden_dataset',manifest['dataset'])/'hidden'/(key(row)+'.pt')


def evaluation_group(dataset, row):
    return row['sha256'] if dataset.startswith('forb') else row['group']


@contextmanager
def claim(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class Projection(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        for name in ('weight', 'bias', 'a', 'b'):
            self.register_buffer(name, state[name])
        self.scaling = state['scaling']

    def forward(self, hidden):
        f = torch.nn.functional.linear
        base = f(hidden.to(self.weight.dtype), self.weight, self.bias)
        update = f(f(hidden.to(self.a.dtype), self.a), self.b)
        result = (base + update * self.scaling).to(base.dtype)
        return result / result.norm(dim=-1, keepdim=True)


def load_bank(path):
    path = project_path(path)
    if path.is_file():
        return torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    metadata = read(path/'index.json')

    class Shards:
        def __init__(self):
            self.loaded = OrderedDict()
            self.locations = {}
            for i, shard in enumerate(metadata['shards']):
                for j in range(shard['start'], shard['stop']):
                    self.locations[j] = (i, j-shard['start'])

        def __getitem__(self, index):
            shard_index, offset = self.locations[index]
            shard = metadata['shards'][shard_index]
            if shard_index not in self.loaded:
                value = torch.load(path/shard['path'],weights_only=True,map_location='cpu',mmap=True)
                assert value['ids'] == metadata['ids'][shard['start']:shard['stop']]
                assert value['image_sha256'] == metadata['image_sha256'][shard['start']:shard['stop']]
                assert value['signature'] == metadata['signature']
                self.loaded[shard_index] = value
            value = self.loaded.pop(shard_index)
            self.loaded[shard_index] = value
            while len(self.loaded) > 8:
                self.loaded.popitem(last=False)
            return value['embeddings'][offset]

    return dict(ids=metadata['ids'],sha256=metadata['image_sha256'],embeddings=Shards())
