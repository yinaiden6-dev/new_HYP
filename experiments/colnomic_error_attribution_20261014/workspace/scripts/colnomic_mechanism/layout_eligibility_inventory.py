"""Inventory every identity meeting frozen layout quality rules, without encoding."""
import argparse
from collections import defaultdict
from pathlib import Path
import re
import numpy as np
import torch
import cv2
from PIL import Image, ImageOps

from scripts.colnomic_mechanism.common import ROOT, data, read, write, csvout, sha
from scripts.colnomic_mechanism.extended_layouts import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').exists():
        raise FileExistsError('Completed inventory exists')
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    d = data()
    selected = read(ROOT / 'results/token_mechanism_experiments/20261013_v2/stage4_extended_photos_v3/sources.json')
    selected_packages = {r['package_id'] for r in selected}
    packages = defaultdict(list)
    for i, qid in enumerate(d['qb']['ids']):
        packages[d['photos'][qid]['package_id']].append((qid, i))
    eligibility, failures = [], []
    write(output / 'protocol.json', dict(quality_rule='Same primary prepare() rules with max_tilt=.08; no train/val caps; all query source provenances inventoried separately',
          selection='first eligible word/photo per identity in fixed lexical photo order; no score, correctness or feature inputs',
          supplementary_tilt_source_reported_separately=True,
          purpose='Detect the previous train=20/val=8 sampling caps and quantify all qualifying independent identities before any new encoding'))
    for package in sorted(packages):
        if (output / (package + '.json')).exists():
            eligibility.append(read(output / (package + '.json')))
            continue
        chosen = None
        by_provenance = {}
        present_provenances = {d['qb']['provenance'][i]['kind'] for qid, i in packages[package]}
        for qid, qi in sorted(packages[package]):
            provenance = d['qb']['provenance'][qi]['kind']
            if provenance in by_provenance:
                continue
            record = d['qo'][qid]
            path = ROOT / d['qb']['paths'][qi]
            if not path.exists():
                failures.append(dict(photo_id=qid, reason='missing original'))
                continue
            assert sha(path) == d['qb']['sha256'][qi]
            with Image.open(path) as image:
                a = np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()
            spec = d['metadata']['photos'][qi]
            bbox = np.array(d['boxes'][qid]['bbox'])
            for li, line in enumerate(record['lines']):
                if not re.fullmatch(r'[A-Za-z]{4,20}[.!]?', line['text'].strip()):
                    continue
                try:
                    box, background, shifts = prepare(a, line, spec, bbox, record['lines'], .08)
                except ValueError as error:
                    failures.append(dict(photo_id=qid, package_id=package, OCR_line_index=li, reason=str(error)))
                    continue
                candidate = dict(photo_id=qid, package_id=package, split=d['splits'][qid], group_id=d['groups'][package],
                    cache_provenance=d['qb']['provenance'][qi]['kind'], eligible=True, OCR_word=line['text'], OCR_line_index=li,
                    original_path=str(path), original_SHA256=d['qb']['sha256'][qi], bbox_display=list(map(int, box)), shifts=shifts,
                    background_RGB=background.tolist(), previously_encoded_identity=package in selected_packages)
                by_provenance[provenance] = candidate
                if chosen is None:
                    chosen = candidate.copy()
                break
            if present_provenances.issubset(by_provenance):
                break
        if chosen is None:
            qid, qi = packages[package][0]
            chosen = dict(package_id=package, split=d['splits'][qid], group_id=d['groups'][package], eligible=False,
                          previously_encoded_identity=package in selected_packages)
        chosen['first_eligible_by_provenance'] = by_provenance
        write(output / (package + '.json'), chosen)
        eligibility.append(chosen)
        write(output / 'progress.json', dict(completed_identities=len(eligibility), total_identities=len(packages), last_identity=package))
        print('Eligibility', len(eligibility), '/', len(packages), package, chosen['eligible'], flush=True)
    csvout(output / 'all_identities.csv', eligibility)
    csvout(output / 'rejection_evidence.csv', failures)
    qualifying = [r for r in eligibility if r['eligible']]
    result = dict(status='complete', inspected_identities=len(eligibility), independent_quality_eligible_identities=len(qualifying),
        eligible_by_split={s: sum(r['split'] == s for r in qualifying) for s in ['train', 'val', 'test']},
        generated_cache_eligible_by_split={s: sum(r['split'] == s and 'generated' in r['first_eligible_by_provenance'] for r in qualifying) for s in ['train', 'val', 'test']},
        qualified_but_not_encoded=[r['package_id'] for r in qualifying if not r['previously_encoded_identity']],
        earlier_encoded_not_primary_quality_eligible=[r['package_id'] for r in eligibility if r['previously_encoded_identity'] and not r['eligible']],
        previous_train_and_validation_caps_do_not_cover_all_eligible=True,
        no_new_encoding_performed=True)
    result['previous_train_and_validation_caps_do_not_cover_all_eligible'] = bool(result['qualified_but_not_encoded'])
    write(output / 'complete.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
