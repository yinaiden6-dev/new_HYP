#!/usr/bin/env python3
"""F128 entrypoint for immutable v1 B_CAL and v2 frozen-baseline readers.

Only the population adapter and artifact panel metadata differ. The delegated
backend is pinned in protocol.sources; runner_sha256 identifies THIS entrypoint.
No old source file or F71 experiment artifact is modified.
"""
import argparse
import fcntl
import sys
from pathlib import Path
import torch
import run_rebut_qr_qrr_v1 as baseline
import run_rebut_qr_qrr_frozenbase_v2 as residual
from rebut_qr_qrr_f128_common_v1 import F128Inputs, verify_panel, verify_sources, verify_evidence_gate


def seal_baseline_terminal(protocol, fold, seed):
    """Repair only v1's result-written/checkpoint-predict interruption window."""
    folder = Path(protocol['output_root']) / f'fold{fold}/seed{seed}/B_CAL'
    result_path, checkpoint_path = folder / 'result.json', folder / 'checkpoint.pt'
    if not result_path.exists():
        return
    with (folder / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = baseline.read(result_path)
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        saved = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
        if result['binding'] != state['binding'] or saved['binding'] != state['binding']:
            raise ValueError('completed baseline artifact bindings disagree')
        if baseline.digest(folder / 'model.pt') != result['model_sha256']:
            raise ValueError('completed baseline model hash mismatch')
        if state['stage'] == 'done':
            return
        if (state['stage'] != 'predict' or state['best_epoch'] != result['best_epoch'] or
                state['calibration'] != result['calibration'] or
                not torch.equal(state['normalization'], saved['normalization']) or
                set(state['model']) != set(saved['model']) or
                not all(torch.equal(state['model'][k], saved['model'][k]) for k in state['model'])):
            raise ValueError('unsafe baseline terminal checkpoint repair')
        state['stage'] = 'done'
        baseline.atomic_torch(checkpoint_path, state)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--phase', required=True, choices=['baseline', 'residual'])
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--fold', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--arm', required=True)
    args, _ = parser.parse_known_args()
    protocol = baseline.read(args.protocol)
    verify_panel(protocol)
    verify_sources(protocol)
    evidence_gate = None if '--pilot' in sys.argv else verify_evidence_gate(protocol)
    if args.phase == 'baseline':
        if args.arm != 'B_CAL' or protocol['arms'] != ['B_CAL']:
            raise ValueError('baseline phase is only the same-F128 B_CAL')
        backend = baseline
        if '--pilot' not in sys.argv:
            seal_baseline_terminal(protocol, args.fold, args.seed)
    else:
        if args.arm not in ('QR', 'QR_VEC', 'QRR') or protocol['arms'] != ['QR', 'QR_VEC', 'QRR']:
            raise ValueError('residual phase requires the three registered frozen-baseline readers')
        source_validation = baseline.read(baseline.checked(protocol['source_validation']))
        if (source_validation.get('status') != 'F128_BASELINE15_INDEPENDENT_PASS' or
                source_validation.get('protocol') != protocol['source_protocol']):
            raise ValueError('same-F128 baseline independent seal missing')
        backend = residual
    original_json_writer = backend.atomic_json
    def write_f128(path, payload):
        if isinstance(payload, dict) and 'panel' in payload:
            payload = {**payload, 'panel': 'F128 developmental original-fold intersection',
                       'population_queries': 128, 'f128_evidence_gate': evidence_gate,
                       'execution_backend': 'v1_B_CAL' if args.phase == 'baseline' else 'v2_frozen_base'}
        return original_json_writer(path, payload)
    backend.FrozenInputs = F128Inputs
    backend.atomic_json = write_f128
    backend.__file__ = str(Path(__file__).resolve())
    phase_index = sys.argv.index('--phase')
    del sys.argv[phase_index:phase_index + 2]
    backend.main()


if __name__ == '__main__':
    main()
