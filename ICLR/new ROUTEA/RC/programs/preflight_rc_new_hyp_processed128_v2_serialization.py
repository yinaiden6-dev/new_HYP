#!/usr/bin/env python3
"""Exercise actual repair writes, CPU mmap reads and overwrite rejection."""
import json
import os
from pathlib import Path
import sys
import tempfile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rc_new_hyp_processed128_v2_serialization as W
V = W.V

if __name__ == '__main__':
    runtime = sys.argv[1]
    actual = V.OUT
    with tempfile.TemporaryDirectory(prefix='proc128-serialize-') as d:
        V.OUT = Path(d)
        p = dict(records=[dict(x=torch.arange(12).reshape(3,4), y=1.25) for _ in range(8)])
        W.save('raw', 0, p)
        path = V.OUT / 'raw/shard00/payload.pt'
        before = V.bind(path)
        loaded = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        assert len(loaded['records']) == 8
        assert all(torch.equal(x['x'], y['x']) and x['y'] == y['y'] for x,y in zip(p['records'], loaded['records']))
        rejected = False
        try:
            W.save('raw', 0, dict(records=[]))
        except FileExistsError:
            rejected = True
        assert rejected and V.bind(path) == before
        receipt = V.read(path.parent / 'receipt.json')
        assert receipt['payload'] == before and receipt['authority'] == V.bind(V.AUTH)
    V.OUT = actual
    result = dict(status='PROCESSED128_FILE_OBJECT_SERIALIZATION_PASS', runtime=runtime,
                  torch_version=str(torch.__version__), python=sys.executable,
                  mmap_weights_only_roundtrip=True, overwrite_rejected=True,
                  natural_tensor_reads=0, wrapper=V.bind(Path(W.__file__)),
                  original_program=V.bind(Path(V.__file__)))
    V.write(actual / f'serialization_v2_{runtime}_preflight.json', result)
    print(json.dumps(result))
