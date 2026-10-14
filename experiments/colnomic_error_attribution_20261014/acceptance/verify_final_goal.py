#!/usr/bin/env python3
"""Assemble final acceptance only after actual archive/extracted replay receipts exist.

This external sidecar is deliberately excluded from the immutable archive.
The archive byte SHA is the completed full-byte packager result; this step
cross-checks its sidecar, byte size, member manifest, actual extracted checks,
and independently executed extracted dispatcher. It does not rehash 33 GB.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('results/colnomic_full_audit/20261013_resume')

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(2**20):
            h.update(data)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--extracted-workspace', type=Path, required=True)
    parser.add_argument('--dispatcher-output', type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    extracted = args.extracted_workspace.resolve()
    replay_output = args.dispatcher_output.resolve()
    support = workspace / BASE / 'delivery_support'
    analysis_path = workspace / BASE / 'delivery/acceptance.json'
    protocol_path = workspace / BASE / 'FINAL_PROTOCOL.json'
    archive_set_path = support / 'archive_set.json'
    extracted_path = support / 'final_extracted_validation.json'
    replay_path = replay_output / 'validation.json'
    analysis = read(analysis_path)
    archives = read(archive_set_path)
    decoded = read(extracted_path)
    replay = read(replay_path)
    manifest_path = extracted / BASE / 'delivery_support/archive_manifest.json'
    manifest = read(manifest_path)
    assert analysis['status'] == 'analysis_accepted'
    assert archives['status'] == decoded['status'] == replay['status'] == 'passed'
    assert archives['independent_decoded_workspace_validation'] == decoded
    assert len(archives['volumes']) == 1
    assert archives['actual_SHA256_all_logical_volume_members']
    assert archives['source_SHA_rechecked_after_all_volumes']
    volume = archives['volumes'][0]
    archive_path = workspace / volume['path']
    assert archive_path.is_file() and archive_path.stat().st_size == volume['bytes']
    sidecar = archive_path.with_name(archive_path.name + '.sha256')
    assert sidecar.read_text().split() == [volume['sha256'], archive_path.name]
    members = len(manifest['files']) + len(manifest['virtual_files'])
    verification = volume['verification']
    assert verification['status'] == 'passed'
    assert verification['logical_members_verified'] == members + 1
    assert verification['unsafe_or_duplicate_members'] is False
    assert verification['manifest_sha256'] == sha(manifest_path)
    assert decoded['files_SHA_verified'] == members
    assert decoded['member_manifest_sha256'] == archives['member_manifest_sha256'] == sha(manifest_path)
    assert decoded['links']['status'] == 'passed' and decoded['links']['broken_relative_links'] == 0
    axes = decoded['axes']
    assert axes['status'] == 'passed'
    assert (axes['queries'], axes['references'], axes['configurations']) == (1117, 5405, 35)
    assert axes['actual_prediction_rows_replayed'] + axes['empty_OCR_rows'] == 1117 * 35
    assert replay['plan']['workspace'] == str(extracted)
    assert replay['plan']['output'] == str(replay_output)
    assert replay['plan']['mode'] == 'verify-only' and replay['plan']['quick']
    assert replay['scope'] == 'core_quick_plus_complete_fresh_reference_verification'
    assert replay['core_validation_SHA256'] == sha(replay_output / 'core/validation.json')
    for key in ['fresh_reference', 'fresh_reference_feature_bridge', 'fresh_reference_descriptive_summary']:
        assert replay[key]['status'] == 'passed'
    assert replay['fresh_reference']['full_pointwise_matrices'] == 290
    assert replay['fresh_reference_feature_bridge']['replayed_feature_token_rows'] == 63575
    core_path = extracted / BASE / 'reproduction_support/final_full_verification.json'
    assert read(core_path)['status'] == 'passed'
    assert replay['previously_full_core_validation_SHA256'] == sha(core_path)
    assert replay['dispatcher_SHA256'] == sha(extracted / 'scripts/colnomic_mechanism/reproduce_final_audit.py')
    sources = [BASE / 'delivery/acceptance.json', BASE / 'FINAL_PROTOCOL.json', BASE / 'REPORT.md',
               BASE / 'REPRODUCE.sh', BASE / 'reproduction_support/final_full_verification.json']
    provenance = {}
    index = {row['path']: row for row in manifest['files']}
    for relative in sources:
        expected = index[str(relative)]['sha256']
        assert sha(workspace / relative) == sha(extracted / relative) == expected
        provenance[str(relative)] = expected
    checks = {str(p): sha(p) for p in [archive_set_path, extracted_path, replay_path, analysis_path, protocol_path]}
    result = dict(
        status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
        completion_scope='All executable analysis, independent numerical acceptance, immutable archive integrity and actual extracted verification; unresolved mechanisms retained explicitly',
        archive=dict(path=str(archive_path), bytes=volume['bytes'], SHA256=volume['sha256'], member_manifest_SHA256=sha(manifest_path), logical_members_verified=members+1),
        acceptance_receipts_SHA256=checks, frozen_source_SHA256=provenance,
        extracted_workspace=str(extracted), extracted_dispatcher_output=str(replay_output),
        extracted_rank_replay=dict(queries=axes['queries'], references=axes['references'], configurations=axes['configurations'], prediction_rows=axes['actual_prediction_rows_replayed'], empty_OCR_rows=axes['empty_OCR_rows']),
        extracted_dispatcher_scope=replay['scope'], previously_full_core_SHA256=sha(core_path),
        coverage=analysis['coverage'], limitations=analysis['limitations'],
        verifier_SHA256=sha(Path(__file__)),
        analysis_accepted=analysis['status'] == 'analysis_accepted',
        archive_and_actual_extraction_accepted=archives['status'] == decoded['status'] == replay['status'] == 'passed',
        all_queries_unique_causes_identified=False,
        full_fresh_gallery_accuracy_claimed=False,
        state_policy='Authoritative post-seal external completion receipt; frozen payload state remains analysis_accepted_archive_pending to avoid self-referential SHA or post-seal mutation')
    output = support / 'FINAL_GOAL_ACCEPTANCE.json'
    assert not output.exists(), 'Do not replace an existing final receipt'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    output.with_name(output.name + '.sha256').write_text(sha(output) + '  ' + output.name + '\n')
    print('FINAL_GOAL_ACCEPTANCE', output, result['status'], flush=True)

if __name__ == '__main__':
    main()
