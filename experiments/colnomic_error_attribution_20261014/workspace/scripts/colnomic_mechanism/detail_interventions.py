"""Precommitted numeric-region occlusion and equal-area blank-paper controls.

OCR only proposes a specification-bearing region. Its instance relevance and
box validity require a separate visual record. These are image edits, not token
deletion. Edited-image ranks are conditional responses, not identity accuracy.
"""
import argparse
from collections import Counter
from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from scripts.colnomic_mechanism.common import ROOT, data, read, write, csvout, sha
from scripts.colnomic_mechanism.display_pixels import display_image
from scripts.colnomic_mechanism.extended_layouts import to_raw

PATTERN = re.compile(r'\d[\d.,]*\s*(?:mg|mcg|ml|mL|grams?|tablets?|caplets?|capsules?|lbs|softgels?|pack)\b', re.I)
ARMS = ['sham', 'numeric_mask', 'blank_control', 'both_masks']


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    protocol = dict(version=1, purpose='specification-bearing-region image occlusion with matched-area controls',
        selection='first 3 independent packages per original B03/F03 joint outcome, original sorted query order; before edit features',
        strata=['both_correct', 'fusion_rescue', 'fusion_harm', 'both_wrong'],
        eligibility='OCR confidence>=0.95, numeric unit/count text in supplied subject box, matching blank-paper area inside box',
        visual_review='numeric text is an OCR proposal; semantic identity importance and subject box validity require direct image review',
        edit='replace one OCR line rectangle by median RGB of its surrounding ring; matched-size blank rectangle uses SAME replacement RGB',
        expected_relationship='occlusion does not assert new identity labels; score/gap response to removing visible information, not edited retrieval accuracy',
        arms=['base'] + ARMS, interaction='both-minus-numeric-minus-blank-plus-sham',
        selection_uses_baseline_correctness_for_predeclared_strata=True, selection_uses_no_edited_response=True,
        focus_browser_images_not_used=True, seed=20261013, historical_test_exposed=True)
    if (folder / 'protocol.json').exists():
        assert read(folder / 'protocol.json') == protocol
    else:
        write(folder / 'protocol.json', protocol)
    if (folder / 'preparation_complete.json').exists():
        for row in read(folder / 'encoding_manifest.json'):
            assert sha(row['path']) == row['image_SHA256']
        return
    d = data()
    ledger = {r['photo_id']: r for r in read(ROOT / 'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    focus = {'photo_41de3eb383934ec3', 'photo_21bd7df9ed6795e7', 'photo_1f48d45127db4129', 'photo_0e87e0717ab1c215', 'photo_2bdbf879ee1d7dd8'}
    counts = Counter(); chosen_packages = set(); sources = []; rejects = []
    for qi, qid in enumerate(d['qb']['ids']):
        case = ledger[qid]
        outcome = ('both_correct' if case['image_positive_rank'] == case['fusion_positive_rank'] == 1
                   else 'fusion_rescue' if case['fusion_positive_rank'] == 1
                   else 'fusion_harm' if case['image_positive_rank'] == 1 else 'both_wrong')
        package = d['photos'][qid]['package_id']
        if counts[outcome] >= 3 or package in chosen_packages or qid in focus:
            continue
        path = ROOT / d['qb']['paths'][qi]
        assert sha(path) == d['qb']['sha256'][qi]
        with Image.open(path) as im:
            pixels = np.asarray(display_image(im)).copy()
        height, width = pixels.shape[:2]
        subject = np.asarray(d['boxes'][qid]['bbox'], int)
        lines = d['qo'][qid]['lines']
        selected = None
        for li, line in enumerate(lines):
            if line.get('confidence', 0) < .95 or not PATTERN.search(line['text']):
                continue
            box = np.asarray(line['bbox'], float)
            box = np.r_[np.floor(box[:2] - 2), np.ceil(box[2:] + 2)].astype(int)
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                    and (box[:2] >= subject[:2]).all() and (box[2:] <= subject[2:]).all()):
                continue
            w, h = x2 - x1, y2 - y1
            if w < 8 or h < 6 or w * h > .12 * max(1, np.prod(subject[2:] - subject[:2])):
                continue
            blank = None
            for cy in np.linspace(subject[1], subject[3] - h, 21).astype(int):
                for cx in np.linspace(subject[0], subject[2] - w, 21).astype(int):
                    candidate = np.array([cx, cy, cx + w, cy + h])
                    if np.maximum(np.minimum(candidate[2:], box[2:]) - np.maximum(candidate[:2], box[:2]), 0).prod() > 0:
                        continue
                    if any(np.maximum(np.minimum(candidate[2:], np.asarray(l['bbox'])[2:]) -
                                      np.maximum(candidate[:2], np.asarray(l['bbox'])[:2]), 0).prod() > 0 for l in lines):
                        continue
                    block = pixels[cy:cy + h, cx:cx + w].reshape(-1, 3)
                    if np.max(np.quantile(block, .99, axis=0) - np.quantile(block, .01, axis=0)) <= 20:
                        blank = candidate
                        break
                if blank is not None:
                    break
            if blank is None:
                continue
            ring = pixels[max(0,y1-3):min(height,y2+3), max(0,x1-3):min(width,x2+3)]
            replacement = np.rint(np.median(ring.reshape(-1,3), axis=0)).astype(np.uint8)
            selected = (li, line['text'], box, blank, replacement)
            break
        if selected is None:
            rejects.append(dict(photo_id=qid, package_id=package, outcome=outcome, reason='no eligible numeric rectangle and matched blank-paper control'))
            continue
        li, text, box, blank, replacement = selected
        spec = d['metadata']['photos'][qi]
        source = dict(photo_id=qid, package_id=package, split=d['splits'][qid], group_id=d['groups'][package],
            outcome=outcome, cache_row=qi, cache_provenance=d['qb']['provenance'][qi]['kind'], original_path=str(path),
            image_SHA256=d['qb']['sha256'][qi], source_SHA256=d['qb']['source_sha256'][qi],
            proposed_text=text, OCR_line=li, numeric_bbox=box.tolist(), blank_bbox=blank.tolist(),
            replacement_RGB=replacement.tolist(), review_status='awaiting_direct_visual_semantic_verification', arms=[])
        source['arms'].append(dict(id=qid+'_fresh_base', source_id=qid, path=str(path), arm='base',
            split=source['split'], package_id=package, label=None, image_SHA256=sha(path)))
        for arm in ARMS:
            value = pixels.copy(); allowed = np.zeros((height,width), bool)
            for b in ([box] if arm == 'numeric_mask' else [blank] if arm == 'blank_control' else [box,blank] if arm == 'both_masks' else []):
                bx1,by1,bx2,by2 = b
                value[by1:by2,bx1:bx2] = replacement
                allowed[by1:by2,bx1:bx2] = True
            assert np.array_equal(value[~allowed], pixels[~allowed])
            output = folder / 'images' / (qid+'_'+arm+'.png'); output.parent.mkdir(exist_ok=True)
            exif = Image.Exif(); exif[274] = spec['exif_orientation']
            Image.fromarray(to_raw(value, spec['exif_orientation'])).save(output, exif=exif)
            source['arms'].append(dict(id=qid+'_'+arm, source_id=qid, path=str(output), arm=arm,
                split=source['split'], package_id=package, label=None, image_SHA256=sha(output)))
        panel = Image.fromarray(pixels)
        draw = ImageDraw.Draw(panel)
        draw.rectangle(tuple(box), outline=(255,0,0), width=max(2, width//400))
        draw.rectangle(tuple(blank), outline=(0,180,255), width=max(2,width//400))
        panel.thumbnail((1200,1200)); preview = folder / 'previews' / (qid+'.jpg'); preview.parent.mkdir(exist_ok=True)
        panel.save(preview, quality=95)
        source['preview'] = str(preview.relative_to(ROOT))
        sources.append(source); chosen_packages.add(package); counts[outcome] += 1
    manifest = [arm for source in sources for arm in source['arms']]
    write(folder / 'sources.json', sources); write(folder / 'encoding_manifest.json', manifest)
    csvout(folder / 'eligibility_rejections.csv', rejects)
    write(folder / 'preparation_complete.json', dict(status='prepared', source_count=len(sources),
        outcome_counts=dict(counts), encoding_records=len(manifest), manual_review_is_separate=True))
    print('Prepared', len(sources), 'numeric-region controls', dict(counts), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(2); prepare(args.output.resolve())


if __name__ == '__main__':
    main()
