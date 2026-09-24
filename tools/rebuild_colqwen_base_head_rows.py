#!/usr/bin/env python3
"""Restore the omitted >100MB head/rows.json using archived per-query JSON."""
import hashlib
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
rc = repo / 'ICLR/new ROUTEA/RC'
root = rc / 'results/rc_colqwen_base_native_v2'
authority = json.loads((rc / 'registry/rc_colqwen_base_head_authority_v2_20260924.json').read_text())
quality = {r['query_id']: r for r in json.loads((root / 'quality/ready_workers.json').read_text())['records']}
rows = []
for i in range(593):
    row = json.loads((root / f'native/queries/query{i:03d}/payload.json').read_text())
    assert row['execution_ordinal'] == i
    row['mass'] = quality[row['query_id']]['mass']
    rows.append(row)
data = (json.dumps({'records': rows}, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
assert hashlib.sha256(data).hexdigest() == authority['rows']['sha256'], 'Original source checksum mismatch'
output = root / 'head/rows.json'
if output.exists():
    assert output.read_bytes() == data
else:
    output.write_bytes(data)
print(f'Restored {output}; SHA256 verified against the original authority')
