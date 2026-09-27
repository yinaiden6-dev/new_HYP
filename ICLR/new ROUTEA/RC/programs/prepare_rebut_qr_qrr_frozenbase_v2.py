#!/usr/bin/env python3
"""Seal the existing F71 inputs and phase-correct trained B_CAL checkpoints."""
import copy
import hashlib
import json
from pathlib import Path

import torch

RC = Path(__file__).resolve().parents[1]
OLD = RC / 'results/rc_rebut_qr_qrr_v1'
OUT = RC / 'results/rc_rebut_qr_qrr_frozenbase_v2'


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def immutable_json(p, value):
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if p.exists() and p.read_text() != data:
        raise ValueError(f'Refuse changing existing frozen preparation: {p}')
    p.write_text(data)


def main():
    torch.set_num_threads(2)
    old = read(OLD / 'protocol.json')
    validation = read(OLD / 'validation.json')
    assert validation['status'] == 'REBUT_QR_QRR_ALL60_F71_INDEPENDENT_JOIN_PASS'
    assert validation['protocol'] == bind(OLD / 'protocol.json')
    common_binding = old['common_features']
    assert bind(common_binding['path']) == common_binding
    common = {x['query_id']: x for x in torch.load(common_binding['path'], map_location='cpu', weights_only=False)['records']}
    baselines, checks = {}, []
    for fold in range(5):
        split = old['folds'][str(fold)]
        baselines[str(fold)] = {}
        for seed in old['seeds']:
            folder = OLD / f'fold{fold}/seed{seed}/B_CAL'
            result = read(folder / 'result.json')
            assert result['status'] == 'TRAINING_AND_LABEL_FREE_PREDICTIONS_COMPLETE'
            assert result['binding']['protocol'] == bind(OLD / 'protocol.json')
            for k in ['train_query_ids', 'inner_fit_query_ids', 'inner_val_query_ids', 'heldout_query_ids']:
                assert result[k] == split[k]
            model = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
            checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
            assert result['model_sha256'] == bind(folder / 'model.pt')['sha256']
            assert checkpoint['stage'] == 'done' and checkpoint['best_epoch'] == result['best_epoch']
            assert torch.equal(model['model']['head.weight'], checkpoint['model']['head.weight'])
            assert torch.equal(model['normalization'], checkpoint['normalization'])
            inner = read(folder / 'inner_validation.json')
            max_error = 0.0
            for phase, ids, weight, scale, records in [
                ('inner', split['inner_fit_query_ids'], checkpoint['best_model']['head.weight'], checkpoint['inner_normalization'], inner['records']),
                ('outer', split['train_query_ids'], model['model']['head.weight'], model['normalization'], result['predictions']),
            ]:
                chunks = [common[q]['X0'][torch.arange(128) != common[q]['winner']] for q in ids]
                rms = torch.cat(chunks).square().mean(0).sqrt()
                rms = torch.where(rms > 1e-12, rms, torch.ones_like(rms))
                assert torch.equal(rms, scale), (fold, seed, phase, 'RMS')
                for row in records:
                    q = row['query_id']; x = common[q]
                    assert row['axis'] == x['axis'] and row['winner'] == x['winner']
                    scores = torch.nn.functional.linear(x['X0'] / scale, weight).squeeze(-1)
                    scores[x['winner']] = 0
                    expected = torch.tensor(row['logits'], dtype=torch.float64)
                    error = float((scores - expected).abs().max())
                    max_error = max(max_error, error)
                    assert error == 0, (fold, seed, phase, error)
            baselines[str(fold)][str(seed)] = {
                'source_result': bind(folder / 'result.json'),
                'source_model': bind(folder / 'model.pt'),
                'source_checkpoint': bind(folder / 'checkpoint.pt'),
            }
            checks.append({'fold': fold, 'seed': seed, 'baseline_epoch': result['best_epoch'],
                           'phase_correct_RMS_pass': True, 'maximum_logit_replay_error': max_error})
    protocol = copy.deepcopy(old)
    protocol.update({
        'version': 'REBUT_QR_QRR_FROZENBASE_V2',
        'status': 'FROZEN_BEFORE_V2_TRAINING',
        'user_authorization': '2026-09-27 continue: trained baseline frozen, new residual learns on existing F71; old results preserved',
        'output_root': str(OUT), 'arms': ['QR', 'QR_VEC', 'QRR'],
        'baseline_bindings': baselines,
        'source_protocol': bind(OLD / 'protocol.json'),
        'source_validation': bind(OLD / 'validation.json'),
        'dependency_bindings': {
            'module_v1': bind(RC / 'src/rc_aslo_xf/rebut_qr_qrr_v1.py'),
            'runner_v1': bind(RC / 'programs/run_rebut_qr_qrr_v1.py'),
        },
        'head': 'phase-correct trained B_CAL weights and RMS are frozen; inner uses best_model trained on inner_fit, outer uses model trained on outer TRAIN',
        'checkpoint': 'resumable reader-only AdamW; include baseline epoch0; fresh same-seed reader refit at selected epoch with frozen outer baseline',
        'threshold': 'fixed zero primary; unchanged inherited B_CAL inner threshold secondary; no new threshold fitting',
        'selection': {
            'main': 'epochs with inner fixed-zero correct count strictly greater than baseline; among these minimum inner validation CE, ties earliest; if none epoch0',
            'early_stopping': 'inner validation CE, include epoch0; patience10; max100 epochs',
            'best_CE': 'archive for diagnosis; not separately selected using held results',
            'fallback': 'epoch0 exactly restores phase-specific B_CAL; TRAIN selection does not guarantee no held breaks',
        },
        'normalization': 'reuse phase-specific sealed B_CAL X0 RMS; F remains unchanged and unstandardized, to isolate training design',
        'source_baseline_fits_reused': 15, 'new_reader_fits': 45,
        'normalization_or_feature_ablation_added': False,
        'evidence_level': 'Repeatedly explored F71 grouped development; not untouched test or new H593 result',
        'secondary': ['new residual versus phase-matched frozen B_CAL', 'v1 from-zero context', 'inherited B_CAL tau sensitivity'],
    })
    OUT.mkdir(parents=True, exist_ok=True)
    immutable_json(OUT / 'protocol.json', protocol)
    immutable_json(OUT / 'baseline_preparation_validation.json', {
        'status': 'FROZENBASE_V2_PHASE_SPECIFIC_BASELINES_PASS',
        'protocol': bind(OUT / 'protocol.json'), 'baseline_fits': 15, 'checks': checks,
        'maximum_baseline_replay_error': max(x['maximum_logit_replay_error'] for x in checks),
        'held_labels_read': False, 'new_training_updates': 0,
        'preparation_program': bind(__file__),
    })
    print(json.dumps({'status': 'PREPARED', 'baseline_fits_reused': 15, 'new_reader_fits': 45,
                      'protocol': str(OUT / 'protocol.json'), 'baseline_replay_error': 0}))


if __name__ == '__main__':
    main()
