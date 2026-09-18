#!/usr/bin/env python3
"""Pin the new ISIC execution using qualified unchanged RPC operator sources."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_new_hyp_isic_transfer_v1'
PLAN = ROOT / 'plan/RC_NEW_HYP_ISIC_FROZEN_TRANSFER_V1_20260914.md'


def read(p): return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p); h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return dict(path=str(p.resolve()), sha256=h.hexdigest())


def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['references', 'inference'])
    stage = parser.parse_args().stage
    assert read(OUT / 'image_intake_validation_v1.json')['status'] == 'ISIC927_IMAGE_INTAKE_PASS'
    assert read(OUT / 'submission/inference_preflight_v1.json')['status'] == 'ISIC_ADAPTER_PATIENT_STATISTICS_LAUNCH_RUNTIME_PASS'
    mappings = dict(data_plan=PLAN, plan=PLAN, gallery=OUT/'gallery_manifest.json',
                    metadata=OUT/'dataset_manifest.json', metadata_validation=OUT/'dataset_manifest.json',
                    overlap=OUT/'image_intake_validation_v1.json', overlap_validation=OUT/'image_intake_validation_v1.json',
                    images=OUT/'image_receipt.json', image_receipt=OUT/'image_receipt.json',
                    launcher_preflight=OUT/'submission/inference_preflight_v1.json',
                    inference_preflight=OUT/'submission/inference_preflight_v1.json', worker=OUT/'worker_manifest.json')
    if stage == 'references':
        a = read(ROOT / 'registry/rc_new_hyp_rpc_reference_encoding_authority_v1_20260913.json')
        mappings.update(program=ROOT/'programs/encode_rc_new_hyp_isic_references_v1.py',
                        launcher=ROOT/'slurm/rc_new_hyp_isic_reference_encoding_v1.sbatch')
        a.update(status='AUTHORIZED_OPENED_ISIC390_REFERENCE_ENCODING', external_reference_forwards=390,
                 untouched_external_claimed=False, cohort='390 lesions,346 known patients,537 query images')
        destination = ROOT / 'registry/rc_new_hyp_isic_reference_encoding_authority_v1_20260914.json'
    else:
        ref = OUT / 'encoded_references'
        v = read(ref/'validation.json')
        assert v['status'] == 'ISIC390_REFERENCE_CPU_INTEGRITY_PASS' and v['reference_count'] == 390
        assert v['payload'] == bind(ref/'payload.pt')
        a = read(ROOT / 'registry/rc_new_hyp_rpc_inference_authority_v1_20260913.json')
        mappings.update(reference_authority=ROOT/'registry/rc_new_hyp_isic_reference_encoding_authority_v1_20260914.json',
                        reference_payload=ref/'payload.pt', reference_validation=ref/'validation.json',
                        legacy_bridge=ref/'legacy_bridge.json',
                        program=ROOT/'programs/run_rc_new_hyp_isic_inference_v1.py',
                        preflight_program=ROOT/'programs/preflight_rc_new_hyp_isic_transfer_v1.py',
                        join_program=ROOT/'programs/analyze_rc_new_hyp_isic_transfer_v1.py',
                        launcher=ROOT/'slurm/rc_new_hyp_isic_inference_v1.sbatch',
                        join_launcher=ROOT/'slurm/rc_new_hyp_isic_join_v1.sbatch')
        a.update(status='ISIC537_OPENED_FROZEN_TRANSFER_AUTHORIZED', query_count=537, reference_count=390,
                 gallery_physical_rows=390, shards=68, last_shard_size=1, patient_count=346,
                 statistics='equal346-patient paired difference;all within-patient images together;100000 draws;seed20260914',
                 curator_after_all_seals=bind(OUT/'curator_roles.json'), untouched_external_claimed=False)
        destination = ROOT / 'registry/rc_new_hyp_isic_inference_authority_v1_20260914.json'
    for k in a['sources']:
        if k in mappings: a['sources'][k] = bind(mappings[k])
    a['freeze_program'] = bind(__file__)
    a['authorization'] = 'User: 可以做个实验，试试。 ISIC exploratory transfer; no spatial supervision or medical diagnosis.'
    write(destination, a)
    print(json.dumps(dict(authority=bind(destination), stage=stage)), flush=True)


if __name__ == '__main__': main()
