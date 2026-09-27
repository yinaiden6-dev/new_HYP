#!/usr/bin/env python3
"""Import the unchanged V2 numerical runner with a pinned confirmation adapter."""
import sys
from pathlib import Path
import run_token_competition_v2 as core
from token_competition_confirm_data_v1 import TokenInputs, verify_panel, verify_evidence_gate
from token_competition_confirm_common_v1 import bind, checked, read, verify_confirmation


def main():
    # The V2 core owns all argument parsing, RNG, training and resume state.
    if '--pilot' in sys.argv:
        raise ValueError('Additional seed confirmation does not rerun an engineering pilot')
    protocol_path = Path(sys.argv[sys.argv.index('--protocol') + 1]).resolve()
    protocol = read(protocol_path)
    verify_confirmation(protocol)
    core.TokenInputs = TokenInputs
    core.verify_panel = verify_panel
    core.verify_evidence_gate = verify_evidence_gate
    original_write = core.atomic_json
    def write(path, value):
        if value.get('status') == 'FROZEN_BASE_RESIDUAL_LABEL_FREE_COMPLETE':
            value = {**value, 'panel': 'F128 developmental original-fold intersection; native token competition confirmation seed' + str(value['binding']['seed'])}
        original_write(path, value)
    core.atomic_json = write
    core.main()
    fold = int(sys.argv[sys.argv.index('--fold') + 1])
    seed = int(sys.argv[sys.argv.index('--seed') + 1])
    arm = sys.argv[sys.argv.index('--arm') + 1]
    folder = Path(protocol['output_root']) / f'fold{fold}/seed{seed}/{arm}'
    result = read(folder / 'result.json')
    if result['binding']['protocol'] != bind(protocol_path):
        raise ValueError('Confirmation result does not bind its own protocol')
    sources = {name: protocol['code_sources'][name] for name in (
        'programs/run_token_competition_confirm_v1.py',
        'programs/token_competition_confirm_data_v1.py',
        'programs/run_token_competition_v2.py',
        'programs/token_competition_data_v2.py')}
    for value in sources.values():
        checked(value)
    original_write(folder / 'execution_adapter.json', {
        'status': 'TOKEN_CONFIRMATION_EXECUTION_ADAPTER_PASS',
        'protocol': bind(protocol_path), 'result': bind(folder / 'result.json'),
        'fold': fold, 'seed': seed, 'arm': arm, 'sources': sources,
        'origin_evidence_validation': protocol['origin_evidence_validation'],
        'held_labels_opened': False,
    })


if __name__ == '__main__':
    main()
