#!/usr/bin/env python3
"""Versioned preprocessing repair and CPU controls for the frozen internal-M pilot.

The v1 implementation, authority, and failed parity outputs remain immutable.
Only the processor backend is repaired; weights, adapter, gates, data and updates
are inherited unchanged. All supplemental predictions precede the held label join.
"""
import copy
import os
from pathlib import Path
import shutil
import time

import run_rc_prellm_m_pilot_v1 as P
import rc_prellm_m_join_v1 as J

ROOT = P.ROOT
PARENT_OUT = P.OUT
PARENT_AUTH = P.AUTH
DIAG = ROOT / 'results/rc_internal_m_loading_diagnostic_v1/5161422/report.json'
P.OUT = ROOT / 'results/rc_prellm_m_adapter_v2'
P.AUTH = ROOT / 'registry/rc_prellm_m_adapter_authority_v2_20260924.json'
P.PLAN = ROOT / 'plan/RC_COLNOMIC_INTERNAL_M_PILOT_V2_20260924.md'
P.LAUNCH = ROOT / 'slurm/rc_prellm_m_pilot_v2.sbatch'
J.OUT, J.AUTH = P.OUT, P.AUTH
J.REPORT = ROOT / 'reports/REPORT_COLNOMIC_INTERNAL_M_PILOT_V2_20260924.md'
ORIGINAL_JOIN, ORIGINAL_PUBLISH = J.join, J.publish


def prepare():
    P.need(not P.AUTH.exists(), 'V2_AUTHORITY_ALREADY_EXISTS')
    parent = P.read(PARENT_AUTH)
    for source in parent['code_sources']:
        P.checked(source)
    diagnostic = P.read(DIAG)
    P.need(diagnostic['status'] == 'DIAGNOSTIC_COMPLETE_NO_PILOT_ADVANCE', 'DIAGNOSTIC_COMPLETE')
    repaired = diagnostic['variants']['strict_bf16_original_default']
    P.need(repaired['original_gates_pass'] and repaired['tokens_half_exact'], 'DIAGNOSTIC_EXACT_ORIGINAL_TOKENS')
    P.need(not diagnostic['variants']['strict_bf16_explicit_false']['original_gates_pass'], 'FAILURE_REPRODUCED')
    P.need(diagnostic['authority'] == P.bind(PARENT_AUTH), 'DIAGNOSTIC_PARENT_AUTHORITY')
    P.OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(P.checked(parent['manifest']), P.OUT / 'input_manifest.json')
    a = copy.deepcopy(parent)
    a.update(parent_authority=P.bind(PARENT_AUTH), loading_diagnostic=P.bind(DIAG),
             manifest=P.bind(P.OUT / 'input_manifest.json'),
             use_fast='original_default_keyword_omitted', processor_backend='torchvision',
             repair='Restore historical default processor only; no weight/dtype/attention/gate changes',
             source_stat_policy='Pinned SHA plus inode/size/mtime; st_dev is host-local and recorded only',
             supplemental_controls=dict(module='rc_internal_m_cpu_controls_v1',
                 names=['ADDITIVE4_REPLAY', 'PURE_L', 'SCORE_UPDATE5'],
                 supervision='Same TRAIN16 only for the new five-parameter score residual',
                 budget='16 updates; matching updates does not imply equal optimization',
                 label_policy='All supplementary predictions sealed before original held label join'))
    additional = [Path(__file__), P.PLAN, P.LAUNCH,
                  ROOT / 'programs/rc_internal_m_cpu_controls_v1.py',
                  ROOT / 'tests/test_internal_m_cpu_controls_v1.py',
                  ROOT / 'plan/RC_COLNOMIC_INTERNAL_M_ADDITIONAL_CONTROLS_20260924.md']
    a['code_sources'] = parent['code_sources'] + [P.bind(p) for p in additional]
    # Reuse the completed source-file hash check, while independently checking
    # every binding and filesystem signature. No model file has been rewritten.
    receipt = P.read(PARENT_OUT / 'model_source_validation.json')
    P.need(receipt['authority'] == P.bind(PARENT_AUTH), 'PARENT_SOURCE_RECEIPT')
    P.need(receipt['status'] == 'MODEL_SOURCE_HASHES_PASS', 'PARENT_SOURCE_HASHES_PASS')
    P.need(diagnostic['source_validation'] == P.bind(PARENT_OUT / 'model_source_validation.json'),
           'DIAGNOSTIC_SOURCE_VALIDATION')
    sources = P.read(P.checked(a['manifest']))['encoder']['model_sources']
    P.need({x['binding']['path']: x['binding'] for x in receipt['files']} ==
           {b['path']: b for b in sources.values()}, 'SOURCE_FILE_SET')
    for item in receipt['files']:
        s = Path(item['binding']['path']).stat()
        P.need(item['stat'][1:] == [s.st_ino, s.st_size, s.st_mtime_ns], 'MODEL_SOURCE_STAT_DRIFT')
    P.write(P.AUTH, a)
    receipt.update(authority=P.bind(P.AUTH), inherited_receipt=P.bind(PARENT_OUT / 'model_source_validation.json'))
    P.write(P.OUT / 'model_source_validation.json', receipt)
    P.emit(stage='prepare', authority=P.bind(P.AUTH), processor_backend=a['processor_backend'])


def verify_model_sources(m):
    """GPFS device numbers differ between login and compute host mount tables."""
    receipt = P.read(P.OUT / 'model_source_validation.json')
    P.need(receipt['authority'] == P.bind(P.AUTH) and receipt['status'] == 'MODEL_SOURCE_HASHES_PASS',
           'MODEL_SOURCE_AUTH')
    expected = {b['path']: b for b in m['encoder']['model_sources'].values()}
    P.need({x['binding']['path']: x['binding'] for x in receipt['files']} == expected, 'MODEL_SOURCE_SET')
    for item in receipt['files']:
        s = Path(item['binding']['path']).stat()
        P.need(item['stat'][1:] == [s.st_ino, s.st_size, s.st_mtime_ns], 'MODEL_SOURCE_STAT_DRIFT')


def load_model(a):
    import torch
    from colpali_engine.models import ColQwen2_5_Processor
    from rc_prellm_m_adapter_v1 import _load_frozen_colnomic_weights
    P.need(torch.cuda.is_available(), 'GPU_REQUIRED')
    started = time.monotonic()
    model, report = _load_frozen_colnomic_weights(a['model'], 'cuda', attention=a['attention'],
                                                verify_weight_hashes=False)
    # Omit use_fast exactly as in the qualified historical materializer.
    processor = ColQwen2_5_Processor.from_pretrained(a['model'], local_files_only=True)
    backend = str(getattr(processor.image_processor, 'backend', None))
    P.need(backend == a['processor_backend'] == 'torchvision', 'HISTORICAL_PROCESSOR_BACKEND')
    report.update(processor_backend=backend, processor_use_fast_keyword='omitted',
                  image_processor_class=type(processor.image_processor).__name__)
    P.write(P.OUT / 'loading' / f"{os.environ['SLURM_JOB_ID']}_{os.environ.get('SLURM_RESTART_COUNT','0')}.json",
            dict(authority=P.bind(P.AUTH), report=report))
    P.need(not any(p.requires_grad for p in model.parameters()), 'FROZEN_BACKBONE')
    P.need(any('lora_' in n for n, _ in model.named_parameters()), 'RETRIEVAL_LORA_REQUIRED')
    P.emit(event='model_loaded', seconds=time.monotonic()-started, gpu=torch.cuda.get_device_name(),
           attention=a['attention'], processor_backend=backend)
    return model, processor


def publish(result, validation):
    ORIGINAL_PUBLISH(result, validation)
    text = J.REPORT.read_text().replace('pilot v1', 'pilot v2').replace('rc_prellm_m_adapter_v1/', 'rc_prellm_m_adapter_v2/')
    J.REPORT.write_text(text)


def join(a, m):
    from rc_internal_m_cpu_controls_v1 import run_controls, summarize_controls
    # Validate all four sealed arms before fitting a CPU-only TRAIN control or
    # reading any held labels. run_controls itself never reads the held curator.
    J.prelabel_audit(a, m)
    controls = run_controls(P.AUTH, m, P.OUT)
    result = ORIGINAL_JOIN(a, m)
    summary = summarize_controls(result, controls)
    with J.REPORT.open('a') as stream:
        stream.write('\n补充 CPU 对照已在 held 标签读取前封存：'
                     '[对照报告](../results/rc_prellm_m_adapter_v2/cpu_controls/report.md)。\n')
    P.emit(stage='cpu_controls', status=summary.get('status', 'COMPLETE'))


P.prepare, P.load_model, P.verify_model_sources = prepare, load_model, verify_model_sources
J.join, J.publish = join, publish

if __name__ == '__main__':
    P.main()
