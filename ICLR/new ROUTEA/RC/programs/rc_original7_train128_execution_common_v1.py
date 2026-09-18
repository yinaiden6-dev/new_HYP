"""Shared source authority for retrieval-only TRAIN128 input materialization.

Only public bindings are read here. Curator target bindings remain in a
separate authority field and are opened only by the post-seal CPU join stage.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / 'registry/rc_original7_train128_execution_authority_v1_20260910.json'
AUTHORITY_STATUS = 'ORIGINAL7_TRAIN128_INPUTS_AUTHORIZED'
CUTOFF = datetime(2026, 9, 11, 16, tzinfo=timezone.utc)
ALLOWED_STAGES = (
    'token_raw_bridge', 'roma_bridge', 'token_raw_shard', 'raw_aggregate',
    'roma_shard', 'finalize_prejoin', 'join_validated', 'qualify_train',
)
PROGRAM_KEYS = {
    'token_raw_bridge': 'token_raw_program', 'token_raw_shard': 'token_raw_program',
    'roma_bridge': 'roma_program', 'roma_shard': 'roma_program',
    'raw_aggregate': 'cpu_program', 'finalize_prejoin': 'cpu_program',
    'join_validated': 'cpu_program', 'qualify_train': 'cpu_program',
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


PARENT_AUTHORITY = ROOT / 'registry/rc_original7_eval128_full_evidence_execution_authority_v2_compat_20260910.json'
PARENT_SHA = 'bc671f19d6b975b96e1e6541ac11356889bac9f8e9f138f8977520b9a888a519'
TOKEN_PARENT = ROOT / 'registry/rc_original7_eval128_full_evidence_execution_authority_v1_20260910.json'
TOKEN_PARENT_SHA = '95a15a6d83a53e70ca2e2750344f42c4c2483f0e9d454260835a2961f1996b6c'
TOKEN_BRIDGE = ROOT / 'results/rc_original7_eval128_token_raw_v1/bridge'
ROMA_BRIDGE = ROOT / 'results/rc_original7_eval128_roma_v1/bridge'
TOKEN_VALIDATION_SHA = '0f9bc42fecce900bf845b06ba1e21f17f09515217143c70d416da48d4459e557'
ROMA_VALIDATION_SHA = 'b9ef254ffd4b3919d9ac3764f80654abc8b98b5bf0dcf943784063ad7eaede6c'
WORKLOAD_TOKEN_KEYS = {
 'token_raw_bridge_launcher','token_raw_common_program','token_raw_metadata_validation',
 'token_raw_plan','token_raw_preflight','token_raw_program','token_raw_shard_launcher','token_raw_worker_manifest',
}
STABLE_OTHER_KEYS = {
 'cpu_current_ORIGINAL7_parameters','cpu_feature_core','cpu_gallery_identity_contract',
 'cpu_gallery_identity_core','cpu_gallery_identity_registry','cpu_legacy_C4_score_core',
 'roma_source_profile','parent_research_authority','post_submission_continuation',
}
def invariant_names(parent):
 return sorted(k for k in parent['source_bindings'] if
  (k.startswith('token_raw_') and k not in WORKLOAD_TOKEN_KEYS) or k in STABLE_OTHER_KEYS)

def require_reused_engineering_bridges():
 """Reuse proven input operators while keeping both completed bridge authorities intact."""
 a=read(AUTHORITY);need(a['status']==AUTHORITY_STATUS,'TRAIN128_AUTHORITY_REQUIRED')
 need(sha(PARENT_AUTHORITY)==PARENT_SHA and a['parent_execution_authority']==bind(PARENT_AUTHORITY),'EXACT_PARENT_EVAL128_V2')
 parent=read(PARENT_AUTHORITY);names=invariant_names(parent)
 need(a['source_invariant_names']==names,'COMPLETE_FROZEN_OPERATOR_SOURCE_SET')
 for name in names:need(a['source_bindings'][name]==parent['source_bindings'][name],'FROZEN_OPERATOR_CHANGED:'+name)
 need(sha(TOKEN_PARENT)==TOKEN_PARENT_SHA,'TOKEN_BRIDGE_PARENT95')
 answer={}
 for kind,folder,digest,status,parent_path in [
  ('token_raw',TOKEN_BRIDGE,TOKEN_VALIDATION_SHA,'RC_ORIGINAL7_EVAL128_TOKEN_RAW_BRIDGE_PASS',TOKEN_PARENT),
  ('roma',ROMA_BRIDGE,ROMA_VALIDATION_SHA,'RC_ORIGINAL7_EVAL128_ROMA_BRIDGE_PASS',PARENT_AUTHORITY)]:
  bindings={k:bind(folder/f) for k,f in [('payload','payload.pt'),('receipt','receipt.json'),('validation','validation.json')]}
  need(bindings['validation']['sha256']==digest,'QUALIFIED_BRIDGE_PIN:'+kind)
  v=read(folder/'validation.json');r=read(folder/'receipt.json')
  need(v['status']==status and v['authority']==r['authority']==bind(parent_path),'BRIDGE_PROVENANCE:'+kind)
  need(v['payload']==r['payload']==bindings['payload'] and v['receipt']==bindings['receipt'],'BRIDGE_ARTIFACT_BINDING:'+kind)
  if kind=='token_raw':need(all(x is True for x in r['bridge_checks'].values()) and r['query_count']==1,'TOKEN_ORIGINAL_PARITY')
  else:need(v['engineering_original_maps_C4_exact'] and v['all_C4_bits_exact'] and v['independently_recomputed_C4_scalars']==512,'ROMA_ORIGINAL_PARITY')
  need(a['reused_engineering_bridges'][kind]==bindings,'EXPLICIT_BRIDGE_REUSE:'+kind);answer[kind]=bindings
 need(read(ROMA_BRIDGE/'validation.json')['token_source']==answer['token_raw'],'ROMA_REUSED_TOKEN_CHAIN')
 return answer
