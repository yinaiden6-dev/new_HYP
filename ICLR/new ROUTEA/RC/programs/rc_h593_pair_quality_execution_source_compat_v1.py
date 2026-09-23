"""Narrow source-hash compatibility for the user-requested original dispatcher edit."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'registry/rc_h593_pair_quality_original_dev_execution_override_20260923.json'


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def approved_execution_source_change(original):
    allowed = {
        str(ROOT / 'programs/dispatch_rc_h593_pair_quality_cpu_v1.py'),
        str(ROOT / 'programs/run_rc_h593_pair_quality_cpu_v1.py'),
    }
    if original.get('path') not in allowed or not RECORD.is_file():
        return False
    record = json.loads(RECORD.read_text())
    if record.get('status') != 'ORIGINAL_DISPATCHER_DEV_CPU_EDIT_AUTHORIZED':
        return False
    if binding(__file__) != record['compatibility_helper']:
        return False
    if binding(record['scientific_authority']['path']) != record['scientific_authority']:
        return False
    change = record['approved_changes'].get(original['path'])
    return bool(change and change['before'] == original
                and binding(original['path']) == change['after'])
