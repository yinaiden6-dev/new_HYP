"""Prepare and append human-visible review evidence without changing source labels.

Material creation never counts as a visual review. Record payloads must contain
direct observations made by a reviewer after looking at the referenced image.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path('/mnt/lsdf_zyin')
BASE = ROOT / 'results/colnomic_full_audit/20261013_resume'
OUT = BASE / 'visual_reviews/visual_audit'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare_occlusion', 'scene_material', 'scene_supplement', 'record', 'record_compact', 'regular', 'summary'])
    p.add_argument('--payload', type=Path)
    p.add_argument('--start', type=int)
    p.add_argument('--end', type=int)
    a = p.parse_args()
    labels = {r['photo_id']: r for r in csv.DictReader((ROOT / 'results/scene_statistics/photo_labels.csv').open())}
    # The all-dossier index is incrementally rewritten by another worker; the
    # immutable complete ledger is safe to consume while that worker resumes.
    dossiers = {r['photo_id']: r for r in read(ROOT / 'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    materials = read(BASE / 'visual_queue/index.json')
    loc = {qid: (r, j) for r in materials for j, qid in enumerate(r['queries'])}
    if a.action == 'prepare_occlusion':
        qids = [q for q in dossiers if labels[q]['occlusion'] != 'none']
        qids.sort(key=lambda q: (not (dossiers[q]['image_top1_correct'] and not dossiers[q]['fusion_top1_correct']), q))
        for start in range(0, len(qids), 4):
            chosen = qids[start:start + 4]
            canvas = Image.new('RGB', (2000, 530 * len(chosen)), 'white')
            for j, qid in enumerate(chosen):
                mat, row = loc[qid]
                source = BASE / 'visual_queue' / mat['path']
                assert sha(source) == mat['sha256']
                with Image.open(source) as im:
                    canvas.paste(im.crop((0, row * 530, 2000, (row + 1) * 530)), (0, j * 530))
            path = OUT / 'priority_materials' / f'occlusion_{start // 4 + 1:03d}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(path, quality=97, subsampling=0)
            write(path.with_suffix('.json'), {'queries': chosen, 'path': str(path), 'sha256': sha(path), 'selection': 'historical_occlusion_label; fusion harms first; not random'})
        print(json.dumps({'occlusion_count': len(qids), 'priority_ids': qids}))
    elif a.action == 'scene_material':
        chosen = [(m, j, q) for m in materials if a.start <= m['batch'] <= a.end for j, q in enumerate(m['queries'])]
        for begin in range(0, len(chosen), 24):
            chunk = chosen[begin:begin + 24]
            canvas = Image.new('RGB', (2400, 530 * ((len(chunk) + 5) // 6)), 'white')
            for n, (mat, row, qid) in enumerate(chunk):
                with Image.open(BASE / 'visual_queue' / mat['path']) as im:
                    crop = im.crop((0, row * 530, 400, (row + 1) * 530))
                ImageDraw.Draw(crop).text((5, 15), qid, fill='black')
                canvas.paste(crop, ((n % 6) * 400, (n // 6) * 530))
            path = OUT / 'scene_materials' / f'scene_{a.start:03d}_{a.end:03d}_{begin // 24 + 1:02d}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True);canvas.save(path, quality=97, subsampling=0)
            write(path.with_suffix('.json'), dict(queries=[q for _, _, q in chunk], path=str(path), sha256=sha(path), scope='original-query scene supplement only; candidates in historical viewed sheets'))
    elif a.action in ['record', 'record_compact', 'regular', 'scene_supplement']:
        payload = read(a.payload)
        if a.action == 'scene_supplement':
            meta = read(Path(payload['evidence_image']).with_suffix('.json'))
            assert len(meta['queries']) == len(payload['notes'])
            records = []
            for qid, note in zip(meta['queries'], payload['notes']):
                if note is None:continue
                if len(note) == 5:
                    obj, capture, occ, key, observation = note
                    note = [obj, capture, occ, key, 'see existing historical actual-vision record',
                            'see historical identity-visibility observation; not newly enlarged', observation,
                            ['original-query scene overview; detail not enlarged'],
                            ['本条补充独立场景字段；原候选和包装面观察需与保留的历史review关联']]
                obj, capture, occ, key, face, identity, observation, quality, unknown = note
                records.append(dict(photo_id=qid, review_method='actual_view_image', review_status='scene_supplement_reviewed',
                    scene=dict(object_count=obj, capture=capture, foreground_occlusion=occ),
                    identity_key_region_occlusion=dict(status=key, position=observation if occ=='yes' else 'no visible foreground covering', degree='see direct observation'),
                    visible_face=face, identity_information_visible=identity, direct_observation=observation,
                    candidate_differences='See historical actual-vision review; candidate columns were not newly inspected in this scene-only sheet',
                    observable_difficulties=[], quality_observations=quality,
                    fine_detail_status='scene_overview_only; historical candidate review kept separately', unresolved=unknown,
                    review_scope='scene/query supplement; not a new candidate-detail verification'))
            payload['records'] = records
        if a.action == 'regular':
            all_records = []
            for batch, notes in payload['batches'].items():
                mat = next(m for m in materials if m['batch'] == int(batch))
                evidence = BASE / 'visual_queue' / mat['path']
                assert len(notes) == len(mat['queries'])
                for qid, row in zip(mat['queries'], notes):
                    if row is None:
                        continue
                    obj, capture, occ, key, face, identity, observation, differences, quality, unknown = row
                    r = dict(photo_id=qid, review_method='actual_view_image', review_status='reviewed_overview',
                        scene=dict(object_count=obj, capture=capture, foreground_occlusion=occ),
                        identity_key_region_occlusion=dict(status=key, position=observation if occ=='yes' else 'no visible foreground covering', degree='see direct observation'),
                        visible_face=face, identity_information_visible=identity, direct_observation=observation,
                        candidate_differences=differences, observable_difficulties=[], quality_observations=quality,
                        fine_detail_status='major_visible_features_observed_minor_text_unverified', unresolved=unknown,
                        evidence_image=str(evidence))
                    all_records.append(r)
            payload['records'] = all_records
            payload['evidence_image'] = all_records[0]['evidence_image']
        if a.action == 'record_compact':
            expanded = []
            for row in payload['notes']:
                qid, object_count, capture, key_status, key_position, face, identity, observation, differences, difficulties, quality, uncertainty = row
                expanded.append(dict(photo_id=qid, review_method='actual_view_image', review_status='reviewed_overview',
                    visible_face=face, identity_information_visible=identity,
                    scene=dict(object_count=object_count, capture=capture, foreground_occlusion='yes'),
                    identity_key_region_occlusion=dict(status=key_status, position=key_position, degree='局部；遮挡范围详见直接观察'),
                    direct_observation=observation, candidate_differences=differences,
                    observable_difficulties=difficulties, quality_observations=quality,
                    fine_detail_status='major_identity_details_visible_minor_text_unverified', unresolved=uncertainty))
            payload['records'] = expanded
        for r in payload['records']:
            evidence = Path(r.get('evidence_image', payload['evidence_image']))
            assert evidence.exists()
            q = r['photo_id']
            assert q in dossiers and r['review_method'] == 'actual_view_image'
            assert r['scene']['object_count'] in ['single', 'multi', 'unclear']
            assert r['scene']['capture'] in ['handheld', 'tabletop', 'unclear']
            assert r['scene']['foreground_occlusion'] in ['yes', 'none', 'unclear']
            assert r['identity_key_region_occlusion']['status'] in ['yes', 'none', 'unclear']
            r['scene_labels'] = {
                'background': {'single': 'single_object', 'multi': 'multi_object', 'unclear': 'unclear'}[r['scene']['object_count']],
                'placement': r['scene']['capture'],
                'occlusion': {'yes': 'foreground', 'none': 'none', 'unclear': 'unclear'}[r['scene']['foreground_occlusion']]}
            r['scene_labels_reviewed'] = all(v != 'unclear' for v in r['scene_labels'].values())
            r['identity_region_obscured'] = r['identity_key_region_occlusion']
            r.update(reviewer='visual_audit_agent', evidence_image=str(evidence), evidence_sha256=sha(evidence),
                     original_scene_labels=labels[q], historical_labels_unchanged=True,
                     evidence_level='direct_visual_observation_only',
                     mechanism_inference='observed appearance differences are not established failure mechanisms')
            target = OUT / 'records' / f'{q}.json'
            assert not target.exists(), f'Review already exists: {q}; use a separately named revision'
            write(target, r)
        print(f'Wrote {len(payload["records"])} observed records')
    records = [read(p) for p in sorted((OUT / 'records').glob('*.json'))] if (OUT / 'records').exists() else []
    write(OUT / 'coverage.json', {'independent_records': len(records), 'unique_queries': len({r['photo_id'] for r in records}),
          'total_queries': len(dossiers), 'priority_occlusion_records': sum(labels[r['photo_id']]['occlusion'] != 'none' for r in records),
          'fine_detail_verified': sum(r.get('fine_detail_status') == 'verified' for r in records),
          'scope': 'actual view_image evidence; overview review and fine-detail verification reported separately'})


if __name__ == '__main__':
    main()
