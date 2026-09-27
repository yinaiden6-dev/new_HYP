#!/usr/bin/env python3
"""TRAIN-only labels and bounded, lazy native-token F128 evidence for V2."""
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
import sys
import torch

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC / 'src'))
from run_rebut_qr_qrr_v1 import FrozenInputs, checked, digest, read
from rebut_qr_qrr_f128_common_v1 import verify_sources
from rc_aslo_xf.token_competition_v2 import TokenEvidence

ARMS = ('TOKEN_QR', 'TOKEN_QRR_ANCHOR', 'TOKEN_QRR_MULTI')


def verify_panel(protocol):
    panel = protocol['panel_query_ids']
    if len(panel) != 128 or len(set(panel)) != 128 or protocol['seeds'] != [0]:
        raise ValueError('token V2 requires exact F128 panel and seed0')
    if tuple(protocol['arms']) != ARMS:
        raise ValueError('token V2 arm registry changed')
    seen = []
    for f in range(5):
        fold = protocol['folds'][str(f)]
        train, held = set(fold['train_query_ids']), set(fold['heldout_query_ids'])
        fit, val = set(fold['inner_fit_query_ids']), set(fold['inner_val_query_ids'])
        if train & held or train | held != set(panel) or fit & val or fit | val != train:
            raise ValueError('token V2 nested fold partition mismatch')
        seen.extend(fold['heldout_query_ids'])
    if len(seen) != 128 or set(seen) != set(panel):
        raise ValueError('token V2 held fold coverage mismatch')
    return set(panel)


def verify_evidence_gate(protocol):
    manifest_path = checked(protocol['evidence_manifest'])
    manifest = read(manifest_path)
    records = manifest.get('records', [])
    if (manifest.get('status') != 'TOKEN_COMPETITION_EVIDENCE128_PASS' or len(records) != 128 or
            {r['query_id'] for r in records} != set(protocol['panel_query_ids']) or
            sorted(int(r['execution_ordinal']) for r in records) != list(range(128))):
        raise ValueError('full native token evidence gate required')
    receipt_path = checked(protocol.get('evidence_validation', manifest_path.parent / 'evidence_validation.json'))
    receipt = read(receipt_path)
    manifest_binding = {'path': str(manifest_path.resolve()), 'sha256': digest(manifest_path)}
    verifier_path = RC / 'programs/verify_token_competition_evidence_v2.py'
    verifier_binding = {'path': str(verifier_path.resolve()), 'sha256': digest(verifier_path)}
    if (receipt.get('status') != 'TOKEN_COMPETITION_EVIDENCE128_PASS' or
            receipt.get('manifest') != manifest_binding or receipt.get('verifier') != verifier_binding or
            receipt.get('queries') != 128 or receipt.get('candidates') != 128*128 or
            receipt.get('labels_read') != 0 or read(checked(receipt['protocol'])) != protocol):
        raise ValueError('independent full-panel evidence validation receipt required')
    checked(receipt['manifest'])
    checked(receipt['verifier'])
    return {'path': str(receipt_path.resolve()), 'sha256': digest(receipt_path)}


class LazyEvidence(Mapping):
    """At most cache_queries materialized FP64 queries; never preload the panel."""
    def __init__(self, owner, cache_queries):
        self.owner = owner
        self.cache_queries = max(1, int(cache_queries))
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.owner.bindings)

    def __iter__(self):
        return iter(self.owner.bindings)

    def __getitem__(self, qid):
        if qid not in self.owner.bindings:
            raise KeyError(qid)
        if qid not in self.cache:
            # Bindings were SHA-checked during initialization; recheck each cache
            # miss to reject cache mutation throughout resumed fitting/replay.
            path = checked(self.owner.bindings[qid])
            payload = torch.load(path, map_location='cpu', weights_only=False)
            row = self.owner.common[qid]
            axis = payload.get('axis', payload.get('axis128'))
            if torch.is_tensor(axis):
                axis = axis.tolist()
            if (payload['query_id'] != qid or list(axis) != list(row['axis']) or
                    int(payload['winner']) != int(row['winner']) or
                    int(payload['execution_ordinal']) != int(row['execution_ordinal'])):
                raise ValueError('token evidence query/candidate binding mismatch')
            q, ref, local = payload['query_tokens'], payload['matched_reference'], payload['local_scalars']
            valid = payload['query_valid']
            if (q.ndim != 2 or q.shape[1] != 128 or ref.shape != (128, q.shape[0], 128) or
                    local.shape != (128, q.shape[0], 8) or valid.shape != (q.shape[0],) or
                    valid.dtype != torch.bool or not valid.any()):
                raise ValueError('native token dimensions or mask mismatch')
            if any(x.dtype != torch.float32 or not torch.isfinite(x).all() for x in (q, ref, local)):
                raise ValueError('token evidence must be finite FP32')
            m0 = torch.as_tensor(payload['M0'])
            if m0.shape != (128,) or m0.dtype != torch.float64 or not torch.isfinite(m0).all():
                raise ValueError('evidence M0 must retain 128 finite FP64 scores')
            if not torch.equal(m0, row['M0']):
                raise ValueError('evidence M0 differs from frozen common features')
            # Both normalized native token vectors and the eight declared local
            # scalar channels are kept; no spatial pooling or F standardization.
            joined = torch.cat((q[None].expand(128, -1, -1), ref, local), dim=-1).double()[None]
            matched_valid = payload['matched_valid']
            if matched_valid.shape != (128, q.shape[0]) or matched_valid.dtype != torch.bool:
                raise ValueError('matched reference validity mask mismatch')
            joint_valid = valid[None, :] & matched_valid
            ev = TokenEvidence(joined, None, joint_valid[None], None,
                               torch.tensor([list(axis)], dtype=torch.long))
            ev.validate()
            self.cache[qid] = ev
            while len(self.cache) > self.cache_queries:
                self.cache.popitem(last=False)
        self.cache.move_to_end(qid)
        return self.cache[qid]


class TokenInputs(FrozenInputs):
    def __init__(self, protocol, fold, arm, pilot_query_id=None):
        verify_sources(protocol)
        panel = verify_panel(protocol)
        if pilot_query_id is None:
            verify_evidence_gate(protocol)
        elif pilot_query_id not in fold['inner_fit_query_ids']:
            raise ValueError('real-query pilot is restricted to inner TRAIN')
        super().__init__(protocol, fold, 'B_CAL', pilot_query_id)
        self.arm = arm
        for qid in self.query_ids:
            if len({self.identity[int(p)] for p in self.common[qid]['axis']}) != 128:
                raise ValueError('token competition requires 128 unique candidate identities')
        if sorted(int(self.common[q]['execution_ordinal']) for q in panel) != list(range(128)):
            raise ValueError('token V2 requires original execution ordinals 0..127')
        manifest = read(checked(protocol['evidence_manifest']))
        records = manifest['records']
        entries = {r['query_id']: r for r in records}
        if len(entries) != len(records):
            raise ValueError('duplicate token evidence query')
        selected = [pilot_query_id] if pilot_query_id is not None else self.query_ids
        for qid in selected:
            entry = entries[qid]
            if int(entry['execution_ordinal']) != int(self.common[qid]['execution_ordinal']):
                raise ValueError('token evidence execution ordinal mismatch')
            ep = checked(entry['payload'])
            self.bindings[qid] = {'path': str(ep), 'sha256': entry['payload']['sha256']}
        self.evidence = LazyEvidence(self, protocol.get('evidence_cache_queries', 2))
