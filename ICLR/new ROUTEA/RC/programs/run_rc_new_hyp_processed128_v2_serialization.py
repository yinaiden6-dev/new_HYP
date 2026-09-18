#!/usr/bin/env python3
"""Append-only recovery: replace only the failed torch.save file destination."""
import argparse
import os
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rc_new_hyp_processed128_v1 as V

REPAIR = V.ROOT / 'registry/rc_new_hyp_processed128_serialization_repair_v2_20260918.json'


def save(kind, shard, payload):
    folder = V.OUT / kind / f'shard{shard:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'payload.pt'
    V.R.plain_only(payload)
    tmp = folder / ('.partial-' + str(os.getpid()))
    # A file object avoids PyTorch's rejection of the original hidden basename.
    with tmp.open('xb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)
    tmp.unlink()
    V.write(folder / 'receipt.json', dict(authority=V.bind(V.AUTH), payload=V.bind(path),
            query_count=8, serialization_repair_source=V.bind(Path(__file__).resolve())))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['raw', 'roma'])
    ap.add_argument('--shard', type=int, required=True)
    args = ap.parse_args()
    repair = V.read(REPAIR)
    V.need(repair['status'] == 'PROCESSED128_SERIALIZATION_ONLY_RECOVERY', 'REPAIR_AUTHORITY')
    for b in repair['sources'].values():
        V.checked(b)
    V.need(repair['original_authority'] == V.bind(V.AUTH), 'ORIGINAL_AUTHORITY_UNCHANGED')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(17)
    authority = V.guard(args.stage)
    V.save = save
    getattr(V, args.stage)(args.shard, authority)
