"""Shared source authority for the frozen ORIGINAL7 expanded evaluation.

Only public bindings are read here. Curator target bindings remain in a
separate authority field and are opened only by the post-seal CPU join stage.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / 'registry/rc_original7_eval128_full_evidence_execution_authority_v1_20260910.json'
AUTHORITY_STATUS = 'ORIGINAL7_EVAL128_FULL_EVIDENCE_AUTHORIZED'
CUTOFF = datetime(2026, 9, 11, 16, tzinfo=timezone.utc)
ALLOWED_STAGES = (
    'token_raw_bridge', 'roma_bridge', 'token_raw_shard', 'raw_aggregate',
    'roma_shard', 'finalize_prejoin', 'join_validated',
)
PROGRAM_KEYS = {
    'token_raw_bridge': 'token_raw_program', 'token_raw_shard': 'token_raw_program',
    'roma_bridge': 'roma_program', 'roma_shard': 'roma_program',
    'raw_aggregate': 'cpu_program', 'finalize_prejoin': 'cpu_program',
    'join_validated': 'cpu_program',
}


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def bound_path(binding):
    path = Path(binding['path'])
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_public_bindings(bindings):
    """Validate a flat public path/SHA map, never follow private role pointers."""
    need(isinstance(bindings, dict) and bindings, 'PUBLIC_SOURCE_BINDINGS_REQUIRED')
    for name, binding in bindings.items():
        need(isinstance(binding, dict) and {'path', 'sha256'} <= binding.keys(),
             'FLAT_PUBLIC_BINDING:' + name)
        path = bound_path(binding)
        need(sha(path) == binding['sha256'], 'FROZEN_SOURCE_DRIFT:' + name)


def require_authority(stage, program_path):
    need(stage in ALLOWED_STAGES, 'UNKNOWN_EXECUTION_STAGE')
    need(datetime.now(timezone.utc) < CUTOFF, 'USER_RESEARCH_DEADLINE_REACHED')
    need(bool(os.environ.get('SLURM_JOB_ID')), 'NATURAL_STAGE_REQUIRES_SLURM_ALLOCATION')
    authority = read(AUTHORITY)
    need(authority['status'] == AUTHORITY_STATUS, 'AUTHORITY_STATUS')
    need(authority['cutoff_UTC'] == '2026-09-11T16:00:00+00:00', 'AUTHORITY_DEADLINE')
    need(stage in authority['allowed_stages'], 'STAGE_NOT_AUTHORIZED')
    need(authority['query_count'] == 128 and authority['primary_model'] == 'ORIGINAL7_NATIVE7',
         'FROZEN_COHORT_AND_HEAD')
    sources = authority['source_bindings']
    verify_public_bindings(sources)
    need(bound_path(sources[PROGRAM_KEYS[stage]]) == Path(program_path).resolve(),
         'WRONG_PROGRAM_FOR_STAGE')
    need(sources['execution_common_program'] == bind(Path(__file__).resolve()),
         'COMMON_AUTHORITY_PROGRAM_DRIFT')
    return authority


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode()
    if path.exists():
        need(path.read_bytes() == data, 'APPEND_ONLY_OUTPUT_DRIFT:' + str(path))
        return
    temporary = path.with_name('.' + path.name + '.partial.' + str(os.getpid()))
    try:
        with temporary.open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
