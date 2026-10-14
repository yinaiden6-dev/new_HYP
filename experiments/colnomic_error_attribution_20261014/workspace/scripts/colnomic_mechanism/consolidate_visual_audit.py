"""Consolidate actual visual evidence without promoting preparation to review.

Every retained observation has a source record and an image SHA. Scene
dimensions have independent provenance; token-agent package-only counts never
become scene background. Uncertainty never falls back to a historical label.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REL = Path('results/colnomic_full_audit/20261013_resume')
DIMS = ('background', 'placement', 'occlusion')
ALLOWED = {'background': {'single_object', 'multi_object', 'unclear'},
           'placement': {'handheld', 'tabletop', 'other', 'unclear'},
           'occlusion': {'foreground', 'none', 'unclear'}}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def csv_write(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in values for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def seq(value):
    return [] if value is None else value if isinstance(value, list) else [value]


def first(record, *keys, default=None):
    return next((record[k] for k in keys if record.get(k) is not None), default)


def record_files(base):
    review = base / 'visual_reviews'
    for path in sorted(review.glob('*.json')):
        if path.name.startswith(('batches_', 'photo_', 'zoom_photo_')):
            yield path
    for owner in ('root_resume', 'geometry_agent', 'full_tokens_agent', 'tokens_resume', 'scene35_agent'):
        for path in sorted((review / owner).glob('*.json')):
            if path.name.startswith(('batch_', 'batches_', 'photo_', 'scene_rechecks_')):
                yield path
        if owner in ('geometry_agent', 'full_tokens_agent'):
            yield from sorted((review / owner / 'detail_zooms').glob('*.json'))
    for sub in ('records', 'revisions', 'detail_zooms'):
        yield from sorted((review / 'visual_audit' / sub).glob('*.json'))
    for sub in ('new_reviews', 'scene_supplements', 'selected_details'):
        yield from sorted((base / 'visual_consolidated' / sub).glob('*.json'))
    for relative in ('reviewed_regions/manual_rois.json', 'reviewed_boxes/visual_review_notes.json',
                     'natural_shared_phrase_layout/manual_reviews.json', 'foreground_harms_review/join_records.json'):
        path = base / relative
        if path.exists():
            yield path


def unpack(value):
    if isinstance(value, list):
        return value, {}
    if isinstance(value, dict) and isinstance(value.get('records'), (list, dict)):
        records = value['records']
        return (list(records.values()) if isinstance(records, dict) else records,
                {k: v for k, v in value.items() if k != 'records'})
    if isinstance(value, dict) and value.get('photo_id'):
        return [value], {}
    if isinstance(value, dict) and isinstance(value.get('observations'), list):
        return value['observations'], {k: v for k, v in value.items() if k != 'observations'}
    return [], {}


def source_priority(path, record=None):
    value = str(path)
    revision = re.search(r'\.r(\d+)\.json$', value)
    if '/selected_details/' in value:
        return 1150
    if '/foreground_harms_review/' in value:
        return 1100
    if 'scene_supplements/' in value:
        return 1000
    if 'new_reviews/' in value:
        return 900
    if 'scene_rechecks_' in value:
        return 850
    if '/revisions/' in value:
        return 800 + (int(revision.group(1)) if revision else 0)
    for marker, priority in [('/visual_audit/records/', 700), ('/scene35_agent/', 600),
                             ('/root_resume/', 500), ('/tokens_resume/', 400),
                             ('/geometry_agent/', 300), ('/full_tokens_agent/', 200)]:
        if marker in value:
            return priority
    return 100


def scene_values(path, record):
    """Preserve reviewed uncertainty, separately for each observed dimension."""
    value = str(path)
    result = {}
    if 'scene_rechecks_' in value:
        for dimension in DIMS:
            if record.get(dimension) in ALLOWED[dimension]:
                result[dimension] = record[dimension]
        return result
    if isinstance(record.get('scene_labels'), dict):
        for dimension in DIMS:
            label = record['scene_labels'].get(dimension)
            if label in ALLOWED[dimension]:
                result[dimension] = label
        return result
    if '/tokens_resume/' not in value and '/root_resume/' not in value:
        return result
    if '/root_resume/' in value:
        background = {'single': 'single_object', 'multi': 'multi_object', 'multiple': 'multi_object', 'unclear': 'unclear'}.get(record.get('object_count'))
        if background:
            result['background'] = background
    placement = first(record, 'placement', 'capture')
    if placement in ALLOWED['placement']:
        result['placement'] = placement
    occlusion = first(record, 'occlusion', 'foreground_occlusion')
    occlusion = {'absent': 'none', 'present': 'foreground', 'yes': 'foreground', 'no': 'none', 'uncertain': 'unclear'}.get(occlusion, occlusion)
    if occlusion in ALLOWED['occlusion']:
        result['occlusion'] = occlusion
    return result


def resolve_path(raw, root, base, source_parent=None):
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [root / path, base / path, base / 'visual_queue' / path,
                  base / 'full_tokens/visual_material' / path]
    if source_parent:
        candidates.insert(0, source_parent / path)
    return next((p for p in candidates if p.exists()), candidates[0])


def evidence_for(path, record, envelope, root, base, loc, hashes):
    pairs = []
    if 'foreground_harms_review' in path.parts:
        declarations={}
        def collect(value):
            if isinstance(value,dict):
                if value.get('path') and value.get('sha256'):
                    declarations[value['path']]=value['sha256']
                for item in value.values():collect(item)
            elif isinstance(value,list):
                for item in value:collect(item)
        for name in ('materials_all8.json','detail_rois.json','candidate_detail_rois.json'):
            declaration_path=path.parent/name
            if declaration_path.exists():collect(read(declaration_path))
        for name in record.get('actually_viewed_evidence_paths',[]):
            if name not in declarations:
                raise ValueError('Actual foreground evidence lacks prepared SHA declaration: '+name)
            pairs.append((name,declarations[name]))
    for key, hashkey in [('viewed_image_path', 'viewed_image_sha256'),
                         ('evidence_image', 'evidence_sha256'), ('sheet_path', 'sheet_sha256'),
                         ('panel_path', 'panel_sha256')]:
        if record.get(key):
            declared = record.get(hashkey)
            if key == 'viewed_image_path':
                declared = declared or record.get('viewed_image_SHA256')
            pairs.append((record[key], declared))
    for name in seq(record.get('evidence')):
        if isinstance(name, str):
            pairs.append((name, record.get('evidence_hashes',{}).get(name) or record.get('evidence_sha256') or record.get('evidence_SHA256')))
    if record.get('additional_evidence'):
        pairs.append((record['additional_evidence'],record.get('additional_evidence_SHA256')))
    if not pairs and isinstance(envelope.get('evidence'), str):
        pairs.append((envelope['evidence'], envelope.get('evidence_sha256')))
    if not pairs and isinstance(envelope.get('evidence'), dict):
        material=loc.get(record.get('batch')) or next((m for m in loc.values() if record['photo_id'] in m['queries']),None)
        if material:
            prefix=f"batch_{material['batch']:03d}"
            pairs.extend((name,digest) for name,digest in envelope['evidence'].items() if name.startswith(prefix))
    if not pairs:
        material = loc.get(record.get('batch'))
        if material is None:
            material = next((m for m in loc.values() if record['photo_id'] in m['queries']), None)
        if material:
            pairs.append((str(base / 'visual_queue' / material['path']), material['sha256']))
    result = []
    for raw, declared in pairs:
        evidence = resolve_path(raw, root, base, path.parent)
        if evidence not in hashes:
            hashes[evidence] = sha(evidence) if evidence.exists() else None
        actual = hashes[evidence]
        result.append(dict(path=str(evidence.relative_to(root)) if evidence.is_relative_to(root) else str(evidence),
                           declared_sha256=declared, actual_sha256=actual, exists=evidence.exists(),
                           hash_matching=actual is not None and (declared is None or declared == actual),
                           declaration_scope='declared_in_record' if declared else 'associated_historical_material; hash computed now'))
    return result


def normalize(path, record, envelope, root, base, loc, hashes):
    observation = first(record, 'direct_observation', 'direct_observations', 'observations', 'direct_detail_observation')
    external_detail = ('reviewed_regions' in path.parts or 'reviewed_boxes' in path.parts
                       or 'natural_shared_phrase_layout' in path.parts or 'foreground_harms_review' in path.parts)
    selected_detail = external_detail or 'selected_details' in path.parts
    if external_detail:
        observation = record.get('actual_visual_observations') or record.get('observation') or [r['observation'] for r in record.get('rois', []) if r.get('observation')]
    if observation is None and '/tokens_resume/' in str(path) and path.name.startswith('batches_'):
        # These manual rows combine query observation and candidate comparison;
        # their enclosing method explicitly records actual contact-sheet viewing.
        observation = first(record, 'identity_information_visible', 'observable_differences')
    scene_only = ('scene_rechecks_' in path.name or 'scene_supplements' in path.parts
                  or record.get('review_status') == 'scene_supplement_reviewed')
    if observation is None and not scene_only:
        return None  # A material-only manifest is not a review.
    evidence = evidence_for(path, record, envelope, root, base, loc, hashes)
    if not evidence or not all(e['hash_matching'] for e in evidence):
        raise ValueError('Missing/mismatched actual-review evidence: ' + str(path))
    source = str(path.relative_to(root))
    face = first(record, 'visible_face', 'packaging_face', 'face', default='unclear_not_separately_annotated')
    identity = first(record, 'identity_information_visible', 'identity_information_visibility',
                     'identity_visibility', 'identity_visible', default='unclear_not_separately_annotated; see direct observation')
    differences = first(record, 'candidate_differences', 'observable_differences',
                        default=observation if not scene_only else 'historical candidate overview retained separately')
    return dict(photo_id=record['photo_id'], source_path=source, source_sha256=sha(path),
                priority=source_priority(path),
                review_status=first(record, 'review_status', default=envelope.get('review_status', 'historical_actual_visual_observations')),
                scope='scene_dimensions_only' if scene_only else 'selected_fine_details_only' if 'detail_zooms' in path.parts or selected_detail else 'candidate_overview_and_query',
                visible_face=face, identity_information_visible=identity,
                direct_observations=seq(observation), candidate_differences=differences,
                quality_observations=seq(first(record, 'quality_observations', 'image_quality_observation',
                                               'image_quality', 'quality', default='unclear_not_separately_annotated; see direct observation')),
                observable_difficulties=seq(first(record, 'observable_difficulties', 'visible_difficulties', 'difficulties', default=[])),
                uncertainty=seq(first(record, 'unresolved', 'unknown', 'unknowns', 'uncertainty', 'remaining_uncertainty',
                                      default='fine-print equality, identity validity and unique cause not established')),
                identity_ambiguity=record.get('identity_ambiguity', 'not_independently_adjudicated'),
                reference_label_adjudication_status=record.get('reference_label_adjudication_status'),
                evidence_level=first(record, 'evidence_level', 'evidence_grade', default=envelope.get('evidence_level', 'direct_visual_overview_only')),
                fine_detail_verified=first(record, 'fine_detail_verified', default=envelope.get('fine_detail_verified', False)) is True,
                fine_detail_scope=first(record, 'fine_detail_status', 'fine_detail_scope', 'fine_detail_verification', 'fine_details',
                                        'review_scope', 'direct_detail_observation', default='overview_only; fine detail not fully checked'),
                scene_values=scene_values(path, record), evidence=evidence,
                raw_record=record, enclosing_metadata=envelope)


def enrich_detail_fields(record):
    """Copy explicit, sourced observations; never derive scene absence from severity."""
    history=record['review_history']
    adjudications=[h for h in history if h.get('reference_label_adjudication_status')]
    record['reference_label_adjudication_status']=adjudications[-1]['reference_label_adjudication_status'] if adjudications else 'not_independently_adjudicated'
    record['adjudicated_detail_observations']=[dict(source_path=h['source_path'],source_sha256=h['source_sha256'],observations=h['direct_observations'],evidence=h['evidence']) for h in history if h['scope']=='selected_fine_details_only']
    corrections=[h for h in history if h['raw_record'].get('current_candidate_differences')]
    focused=[h for h in history if 'foreground_harms_review' in h['source_path']]
    if focused:
        record['candidate_differences']=' '.join(focused[-1]['direct_observations'])
        record['current_detail_observations']=focused[-1]['direct_observations']
        record['current_detail_uncertainty']=focused[-1]['uncertainty']
        record['current_detail_source']=focused[-1]['source_path']
    if corrections:
        record['candidate_differences']=corrections[-1]['raw_record']['current_candidate_differences']
        record['candidate_difference_revision_source']=corrections[-1]['source_path']
    detail=[]
    for h in history:
        raw=h['raw_record'];key=raw.get('identity_region_obscured',raw.get('identity_key_region_occlusion'))
        oc=raw.get('occlusion_detail',{})
        if key is None:key=oc.get('identity_region_obscured')
        if key is None and not raw.get('occlusion_location') and not oc.get('location'):continue
        detail.append(dict(source_path=h['source_path'],source_sha256=h['source_sha256'],evidence=h['evidence'],
            value=key,location=raw.get('occlusion_location',key.get('position') if isinstance(key,dict) else oc.get('location')),
            extent=raw.get('occlusion_extent',key.get('degree') if isinstance(key,dict) else oc.get('extent')),
            pair_discriminative_spec_occlusion=raw.get('pair_discriminative_spec_occlusion'),occluder=raw.get('occluder')))
    record['occlusion_detail_history']=detail
    if detail:
        current=detail[-1];key=current['value']
        record['identity_key_region_occlusion']=key
        record['identity_region_obscured']=key.get('status','unknown') if isinstance(key,dict) else key if key is not None else 'not_visually_verified'
        record['occlusion_location']=current['location'];record['occlusion_extent']=current['extent']
        record['occlusion_detail_source']=current['source_path'];record['occlusion_detail_source_sha256']=current['source_sha256']
        if current['pair_discriminative_spec_occlusion'] is not None:
            record['pair_discriminative_spec_occlusion']=current['pair_discriminative_spec_occlusion']
        if current['occluder'] is not None:record['occluder']=current['occluder']


def refresh_details_only(root,out):
    """Append focused detail evidence while preserving the already replayed scene snapshot."""
    base=root/REL;record_path=out/'records.json';before=sha(record_path)
    snapshot=out/'records_before_foreground_detail_join.json'
    if not snapshot.exists():snapshot.write_bytes(record_path.read_bytes())
    records=read(record_path);lookup={r['photo_id']:r for r in records};old_scene=sha(out/'scene_revisions.json')
    hashes={};sources={};new=[]
    for path in record_files(base):
        if not ('selected_details' in path.parts or 'foreground_harms_review' in path.parts):continue
        source_records,envelope=unpack(read(path))
        for raw in source_records:
            qid=raw.get('photo_id')
            if qid not in lookup:raise ValueError('Detail query outside inventory')
            h=normalize(path,raw,envelope,root,base,{},hashes)
            if h is None:continue
            if h['scope']!='selected_fine_details_only':raise ValueError('Focused join would alter overview scope')
            key=(h['source_path'],h['source_sha256'])
            old=lookup[qid]['review_history']
            if not any((v['source_path'],v['source_sha256'])==key for v in old):
                old.append(h);new.append(dict(photo_id=qid,source_path=h['source_path'],source_sha256=h['source_sha256']))
            sources[h['source_path']]=dict(sha256=h['source_sha256'],accepted_records=sum(r.get('photo_id') in lookup for r in source_records))
    for r in records:
        r['review_history'].sort(key=lambda h:(h['priority'],h['source_path']))
        enrich_detail_fields(r)
        r['direct_observations']=[o for h in r['review_history'] for o in h['direct_observations']]
        r['fine_detail_verified']=any(h['fine_detail_verified'] for h in r['review_history'])
        r['fine_detail_evidence']=[h for h in r['review_history'] if h['scope']=='selected_fine_details_only' or h['fine_detail_verified']]
    write(record_path,records)
    source_manifest=read(out/'source_manifest.json');source_manifest['sources'].update(sources)
    source_manifest['evidence_images'].update({str(p.relative_to(root)):h for p,h in hashes.items()})
    write(out/'source_manifest.json',source_manifest)
    coverage=read(out/'coverage.json');coverage['explicit_fine_detail_verified']=sum(r['fine_detail_verified'] for r in records)
    coverage['selected_detail_observation_records']=sum(bool(r['fine_detail_evidence']) for r in records);write(out/'coverage.json',coverage)
    write(out/'reference_label_adjudication.json',dict(exclusion_policy='only actual confirmed conflict; independent of feature response',
        conflict_needs_adjudication_ids=[r['photo_id'] for r in records if r['reference_label_adjudication_status']=='conflict_needs_adjudication'],
        records=[dict(photo_id=r['photo_id'],status=r['reference_label_adjudication_status'],details=r['adjudicated_detail_observations']) for r in records if r['reference_label_adjudication_status']!='not_independently_adjudicated']))
    checks=dict(scene_revisions_byte_unchanged=sha(out/'scene_revisions.json')==old_scene,
        every_source_hash_matches=all(sha(root/h['source_path'])==h['source_sha256'] for r in records for h in r['review_history']),
        every_evidence_hash_matches=all(sha(root/e['path'])==e['actual_sha256'] and (e['declared_sha256'] is None or sha(root/e['path'])==e['declared_sha256']) for r in records for h in r['review_history'] for e in h['evidence']),
        every_original_hash_matches=all(sha(root/r['original_path'])==r['original_sha256'] for r in records),
        candidate_overview_coverage_unchanged=sum(r['candidate_overview_reviewed'] for r in records)==coverage['candidate_overview_reviewed'])
    if not all(checks.values()):raise ValueError('Focused detail join failed actual validation')
    write(out/'detail_join_manifest.json',dict(schema='colnomic.focused_detail_join.v1',records_before_sha256=before,
        preserved_before_snapshot=str(snapshot.relative_to(root)),preserved_snapshot_sha256=sha(snapshot),
        records_after_sha256=sha(record_path),scene_revisions_sha256=old_scene,added_records=new,checks=checks,
        scope='focused physical occlusion/candidate observations only; scene labels, original metadata and score matrix unchanged'))
    validation=read(out/'validation.json');validation['focused_detail_join']=checks;write(out/'validation.json',validation)
    (out/'REPORT.md').write_text((out/'REPORT.md').read_text()+'\n8例原前景遮挡融合损害的实际局部复核及CAPSTAR候选修订追加于review_history；顶层关键区域与区分规格遮挡字段有独立来源。scene_revisions.json与35评分快照保持原执行字节；详见detail_join_manifest.json。\n')
    print(json.dumps(dict(added_detail_records=len(new),records_sha256=sha(record_path),checks=checks),ensure_ascii=False))


def refresh_scene_definitions(root,out):
    """Apply actual scene-only corrections and version the score-replay input."""
    base=root/REL;path=out/'scene_supplements/foreground_object_definition_recheck.json'
    records=read(out/'records.json');lookup={r['photo_id']:r for r in records}
    before=sha(out/'records.json');old_revisions=sha(out/'scene_revisions.json');hashes={}
    note_rows,envelope=unpack(read(path));changes=[]
    for raw in note_rows:
        h=normalize(path,raw,envelope,root,base,{},hashes);r=lookup[raw['photo_id']]
        if h['scope']!='scene_dimensions_only':raise ValueError('Unexpected scene correction scope')
        if not any(x['source_path']==h['source_path'] and x['source_sha256']==h['source_sha256'] for x in r['review_history']):r['review_history'].append(h)
        r['review_history'].sort(key=lambda x:(x['priority'],x['source_path']))
        for d,value in h['scene_values'].items():
            previous=r['scene_dimensions'][d]
            changes.append(dict(photo_id=r['photo_id'],dimension=d,previous=previous['value'],verified=value,
                previous_source=previous['source_path'],source_path=h['source_path'],source_sha256=h['source_sha256']))
            r['scene_dimensions'][d]=dict(value=value,status='reviewed_known' if value!='unclear' else 'reviewed_uncertain',
                source_path=h['source_path'],source_sha256=h['source_sha256'],evidence=h['evidence'],
                original_label=previous['original_label'],original_scene_reviewed=previous['original_scene_reviewed'])
            r['scene_labels'][d]=value
        r['direct_observations']=[o for x in r['review_history'] for o in x['direct_observations']]
    tag_changes=[]
    for r in records:
        old=r['difficulty_labels'][:];assign_difficulty_labels(r)
        if old!=r['difficulty_labels']:tag_changes.append(dict(photo_id=r['photo_id'],before=old,after=r['difficulty_labels']))
        enrich_detail_fields(r)
    write(out/'records.json',records)
    keys=('photo_id','review_status','scene_labels','scene_labels_reviewed','scene_dimensions_reviewed','scene_labels_audited',
        'scene_dimensions','visible_face','identity_information_visible','observable_difficulties','primary_review_path','evidence_level','uncertainty',
        'identity_region_obscured','occlusion_location','occlusion_extent','occlusion_detail_source','occlusion_detail_source_sha256',
        'pair_discriminative_spec_occlusion','occluder')
    write(out/'scene_revisions_final.json',dict(schema='colnomic.scene_revisions.dimension_provenance.v1',records=[{k:r[k] for k in keys if k in r} for r in records]))
    source_manifest=read(out/'source_manifest.json');source_manifest['sources'][str(path.relative_to(root))]=dict(sha256=sha(path),accepted_records=len(note_rows))
    source_manifest['evidence_images'].update({str(p.relative_to(root)):v for p,v in hashes.items()});write(out/'source_manifest.json',source_manifest)
    controls=export_controls(out,records,root);export_multilabel(out,records)
    coverage=read(out/'coverage.json');coverage['scene_counts']={d:dict(Counter(r['scene_labels'][d] for r in records)) for d in DIMS}
    coverage['same_face_success_controls']=dict(Counter(r['status'] for r in controls));write(out/'coverage.json',coverage)
    csv_write(out/'coverage.csv',[dict(photo_id=r['photo_id'],candidate_overview_reviewed=r['candidate_overview_reviewed'],
        scene_dimensions_audited=r['scene_labels_audited'],scene_all_known=r['scene_labels_reviewed'],
        **{d+'_status':r['scene_dimensions'][d]['status'] for d in DIMS},**r['scene_labels'],
        primary_review_path=r['primary_review_path'],fine_detail_verified=r['fine_detail_verified']) for r in records])
    checks=dict(replay_scene_input_preserved=sha(out/'scene_revisions.json')==old_revisions,
        every_scene_source_and_image_sha_valid=all(sha(root/v['source_path'])==v['source_sha256'] and all(sha(root/e['path'])==e['actual_sha256'] for e in v['evidence']) for r in records for v in r['scene_dimensions'].values()),
        full1117_unique=len(records)==len(lookup)==1117,scene_partitions_exact=all(sum(coverage['scene_counts'][d].values())==1117 for d in DIMS))
    if not all(checks.values()):raise ValueError('Final scene definition refresh validation failed')
    write(out/'scene_definition_corrections_manifest.json',dict(records_before_sha256=before,records_after_sha256=sha(out/'records.json'),
        preserved_replay_scene_input_sha256=old_revisions,final_scene_input='scene_revisions_final.json',final_scene_input_sha256=sha(out/'scene_revisions_final.json'),
        actual_scene_changes=changes,quality_value_parser_changes=tag_changes,checks=checks,
        classification_scope='Raw human labels/history preserved; uncalibrated descriptive mappings only; dict negative/unknown/zero values do not promote keys'))
    validation=read(out/'validation.json');validation['final_scene_definition_refresh']=checks;write(out/'validation.json',validation)
    refresh=read(out/'control_refresh.json');refresh['records_sha256']=sha(out/'records.json');refresh['controls_sha256']=sha(out/'same_face_success_controls.json');write(out/'control_refresh.json',refresh)
    lines=['# 最终逐图视觉证据整合','',f"稳定ledger1117例；原图SHA逐例验证；原图与关键候选overview实际审核1117/1117。",
        f"场景计数：{coverage['scene_counts']}；每维1117已实际审看，6例背景/1例前景仍不确定。",
        '鼠标、白色设备与独立白纸统一按额外独立实体计入multi_object；实际原图补查8例修订保留于scene_supplements/foreground_object_definition_recheck.json。原标注及原审核history均保留。',
        'scene_revisions.json保存首次完整35评分快照的原字节；scene_revisions_final.json供最终审后场景统计读取。',
        '8例前景遮挡融合损害追加实际局部与正负候选规格审核：区分规格被挡yes3、none4、unknown1。foreground与关键区域/区分规格遮挡独立，关键区域未挡不能转成无前景。',
        f"明确细节核验{coverage['explicit_fine_detail_verified']}例、选定细节观察{coverage['selected_detail_observation_records']}例；只认可每条细节声明的实际范围，overview不等于全文验证。",
        '同面控制逐失败配置保存B03/F03列表，37失败Query有经实际配对确认控制；质量、角度、尺度及复用混杂保留。未确认同面不代表同面成功图不存在。',
        'quality字典先解析值的否定/未知/零值再映射类别；原粗标签/quality原字段不删改，归一化仍是未校准的描述性多标签。',
        'records.json保留全部实际观察、修订来源SHA/图片SHA及候选ledger。机制结论须另接评分、Token与干预证据。']
    (out/'REPORT.md').write_text('\n\n'.join(lines)+'\n')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>最终视觉审核</h1><p><a href="records.json">逐图证据</a> | <a href="coverage.csv">覆盖表</a> | <a href="scene_revisions_final.json">最终场景来源</a> | <a href="same_face_success_controls.json">同面控制</a> | <a href="validation.json">验证</a></p><pre>'+html.escape('\n\n'.join(lines))+'</pre>')
    print(json.dumps(dict(changes=len(changes),scene_counts=coverage['scene_counts'],quality_mapping_changes=len(tag_changes),records_sha256=sha(out/'records.json')),ensure_ascii=False))


def face_key(face):
    """Manual conservative face equivalences; unknown descriptions stay unknown."""
    aliases={name:'primary_front' for name in ('front','主正面','正面','正面大面','front_rotated','front_upside_down',
        'upside_down_front','front_angled','front_closeup','partially_covered_front','partially_cropped_front',
        'front_long_face','flower_graphic_front','brand_front','logo_front','title_front_rotated',
        '主正面旋转','主正面倒置','产品正面','白盒正面','Increxxa正面','Vetrimycin正面','Docetaxel文字正面',
        'Silivex白蓝正面','CanineDewormer正面','滴眼液正面','深色护肤品正面','黑棕BBCream正面',
        '蓝白药名正面','黑色BBCream正面','驱虫药白色正面','蓝色感冒药正面','蓝绿抗过敏正面',
        '深绿止痛药正面','红白婴儿止痛药正面','绿黄兽药正面','药名剂量正面','主面旋转','主面')}
    aliases.update({name:'primary_front_and_top' for name in ('front_and_top','front and top','top_and_front',
        'front_top','front_and_top_upside_down','主正面及顶面','主正面及顶部','主面+顶部','倒置主面/顶部',
        'MilbeGuard正面兼顶面','蓝色感冒药正面兼顶面','黑色正面兼顶面')})
    aliases.update({'front_and_top': 'primary_front_and_top',
            'front and top': 'primary_front_and_top', 'front_and_side': 'primary_front_and_brand_side',
            'front+side': 'primary_front_and_brand_side', 'barcode_end': 'barcode_end',
            'drug_facts_side': 'drug_facts_side', 'manufacturer_end': 'manufacturer_end'})
    groups={
        'complete_drug_facts_back': ('Drug_Facts_back','DrugFacts_back','back_drug_facts','drug_facts_back',
            'Drug Facts back','DrugFacts背面','DrugFacts背面旋转','Drug_Facts_back_rotated','DrugFacts_back_rotated','DrugFacts_back_upside_down'),
        'drug_facts_side': ('drug_facts_side','Drug_Facts_side','side_drug_facts','Drug Facts side','DrugFacts_side',
            'DrugFacts side','Drug_Facts_side_rotated','狭窄Drug Facts面','白底DrugFacts主表窄面'),
        'continued_drug_facts_side': ('drug_facts_continued_side','Drug_Facts_continued_side','DrugFacts_continued_side',
            'Drug Facts continued side','Drug Facts continued侧面','红边Drug Facts continued窄面','DrugFacts续行窄面',
            '黄色DrugFactscontinued侧面','yellow DrugFacts continued side','red_DrugFacts_continued_side'),
        'generic_back_instructions': ('instructions_back','back_instructions','背部说明面','倒置背部说明面','dense_instructions_back'),
        'ingredients_storage_side': ('ingredients_storage_side','ingredients_storage_side_rotated','窄侧成分/储存面',
            '白底成分/储存说明侧面','成分贮存说明侧面','白底药品成分储存侧面'),
        'manufacturer_barcode_side': ('manufacturer_barcode_side','barcode manufacturer side','厂商条码侧面','厂商/条码侧面',
            '厂商侧面及条码面','黄色厂商条码侧面','white_distribution_barcode_side','instructions_distribution_barcode_side'),
        'manufacturer_barcode_end': ('manufacturer_barcode_end','barcode manufacturer end','厂商条码端面','manufacturer_barcode_end_upside_down'),
        'bottom_barcode': ('bottom_barcode','底部条码面','条码底面'),
        'narrow_brand_name_side': ('narrow_brand_side','brand_side','narrow branded side','narrow name side',
            'narrow_side_brand','narrow_logo_side','blue_title_weight_side','品牌规格长侧面','narrow branded barcode side'),
        'product_name_end': ('name end panel','title_end','title_end_rotated','title_end_upside_down','end_brand_panel',
            'brand_end_panel','dose_title_end','orange_title_end','yellow_title_end','green_title_end','black_title_end'),
    }
    for key,names in groups.items():
        aliases.update({name:key for name in names})
    return {name.lower():key for name,key in aliases.items()}.get(str(face).strip().lower())


DIFFICULTY_TERMS = {
    'shared_text_or_brand': ('共享', '公共', '同厂商', '同成分', '同名', '同品牌', '同系列', 'shared', 'common', 'same_brand', 'same_family', 'samebrand', 'sameactive', 'same-distributor', 'generic warning', 'generic medicine', 'genericinstructions'),
    'layout_or_linewrap': ('布局', '排版', '版式', '换行', 'layout', 'linewrap', 'line_wrap'),
    'numbers_specs_or_quantity': ('剂量', '浓度', '规格', '数量', '容量', '重量', '体重', '色号', 'strength', 'dose', 'spec', 'capacity', 'count', 'quantity', 'number', 'size_variant'),
    'graphics_or_color': ('图形', '图案', '图标', '颜色', '配色', '色带', '动物图', '犬图', '猫图', 'color', 'colour', 'graphic', 'illustration', 'animal'),
    'different_or_hidden_face': ('异面', '不同面', '不同包装面', '另一面', '主面不可见', 'side', 'end_panel', 'non_front', 'different_face', 'differentface', 'different face', 'face_mismatch', 'unseen', 'narrowface', 'narrow face', 'sparse face'),
    'occlusion': ('遮挡', '遮住', '被挡', 'occlu', 'obstruct', 'covered'),
    'blur_low_resolution_or_small': ('模糊', '小字', '尺度', '主体小', '主体较小', '细字', '小细节', '低分辨', 'blur', 'small', 'resolution', 'fine_print', 'fine print', 'tiny', 'scale'),
    'reflection_or_low_contrast': ('反光', '高光', '低对比', 'reflection', 'glare', 'contrast'),
    'perspective_rotation_or_deformation': ('透视', '旋转', '折叠', '变形', 'perspective', 'rotation', 'rotated', 'fold', 'deform'),
    'crop_or_partial_view': ('裁切', '截断', '不完整', '局部可见', 'crop', 'truncat', 'partial_view'),
    'extra_background_entities': ('背景', '多物体', 'background', 'clutter'),
    'identity_or_gallery_ambiguity': ('歧义', '标签', '图库', 'ambigu', 'gallery', 'label_issue'),
    'insufficient_identity_information': ('信息不足', '身份信息有限', '可见信息部分', '不可见', '不可确定', '无法辨', 'invisible', 'insufficient', 'unreadable', 'missing_identity', 'limited identity'),
    'product_name_ingredient_or_use_difference': ('品名差异', '名称', '成分', '用途', '剂型', 'brand_diff', 'ingredient', 'product_name', 'indication', 'formulation', 'composition', 'productform'),
    'dense_or_sparse_print': ('dense text', 'denseprint', 'denseDrugFacts', 'sparseprint', 'DrugFacts'),
    'barcode_or_code_detail': ('条码', '二维码', 'barcode', 'QR'),
    'multilingual_text': ('多语言', '文字语言差异', 'multilingual', '韩文'),
}


def difficulty_labels(tags):
    """Classify already human-written tags, never pixels or OCR as visibility."""
    text = ' '.join(map(str, tags)).lower()
    found = sorted(category for category, terms in DIFFICULTY_TERMS.items() if any(term in text for term in terms))
    return found or ['unclear_or_no_explicit_difficulty_tag']


def affirmative_quality_tags(values):
    """Parse values before keys, so reflection:none never becomes reflection."""
    tags=[]
    def visit(value,key=None):
        if isinstance(value,dict):
            for k,v in value.items():visit(v,k)
            return
        if isinstance(value,list):
            for item in value:visit(item,key)
            return
        if value is None or value is False:return
        text=str(value).lower().replace('_',' ').strip()
        if not text or re.search(r'\b(?:none|absent|unknown|unclear|false|not observed|not present|not separately|no\b|not\b)\b',text) or any(v in text for v in ('未见','不明显','没有','未确定','未观察','无反光','无模糊','无明显')):
            return
        if key:
            name=str(key).lower().replace('_',' ')
            if value is True or text in ('present','yes','observed','mild','moderate','strong','severe'):
                tags.append(name+' '+text)
            elif re.search('rotation|perspective|反光|reflection|blur|contrast|变形|透视|旋转',name):
                if not re.fullmatch(r'0(?:\.0+)?(?:\s*(?:deg|degrees|°))?',text):tags.append(name+' '+text)
            else:tags.append(text)
        else:tags.append(text)
    for value in values:visit(value)
    return tags


def assign_difficulty_labels(record):
    record['difficulty_labels']=difficulty_labels(record['observable_difficulties'])
    quality_tags=affirmative_quality_tags(record['quality_observations'])
    # "crop sharper" compares displayed panels; it does not show truncation.
    record['quality_difficulty_labels']=[c for c in difficulty_labels(quality_tags) if c in
        ('blur_low_resolution_or_small','reflection_or_low_contrast','perspective_rotation_or_deformation')]
    record['difficulty_labels']=sorted(set(record['difficulty_labels']+record['quality_difficulty_labels']))
    if len(record['difficulty_labels'])>1:
        record['difficulty_labels']=[c for c in record['difficulty_labels'] if c!='unclear_or_no_explicit_difficulty_tag']


def export_multilabel(out, records):
    all_categories = sorted(set(c for r in records for c in r['difficulty_labels']))
    populations = [('all_reviewed_queries', records), ('B03_or_F03_failure', [r for r in records if r['objective_outcome'] != 'both_correct'])]
    counts, overlap, identity = [], [], []
    for population, population_rows in populations:
        for category in all_categories:
            selected = [r for r in population_rows if category in r['difficulty_labels']]
            counts.append(dict(population=population, category=category, photos=len(selected), denominator=len(population_rows),
                fraction=len(selected)/len(population_rows) if population_rows else None,
                identities=len({r['package_id'] for r in selected}),
                source_groups=len({r['group_id'] for r in selected}), annotation_scope='multi-label human-observation tag normalization; not causal category'))
        for a in all_categories:
            for b in all_categories:
                overlap.append(dict(population=population, category_a=a, category_b=b,
                    intersection_photos=sum(a in r['difficulty_labels'] and b in r['difficulty_labels'] for r in population_rows),
                    denominator=len(population_rows)))
        for package in sorted({r['package_id'] for r in population_rows}):
            group = [r for r in population_rows if r['package_id'] == package]
            for category in all_categories:
                n = sum(category in r['difficulty_labels'] for r in group)
                if n:
                    identity.append(dict(population=population, package_id=package, category=category, photos=n,
                        identity_population_photos=len(group), fraction=n/len(group)))
    csv_write(out/'difficulty_multilabel_counts.csv', counts)
    csv_write(out/'difficulty_multilabel_overlap.csv', overlap)
    csv_write(out/'difficulty_multilabel_by_identity.csv', identity)
    write(out/'difficulty_taxonomy.json', dict(source='human-written observable_difficulties; explicitly affirmative human quality phrases separately identified', category_terms=DIFFICULTY_TERMS,
        empty_policy='unclear_or_no_explicit_difficulty_tag', overlap_policy='multiple labels per query; counts are not exclusive',
        mechanism_limit='observable cooccurrence cannot establish why retrieval failed',
        raw_tag_counts=dict(Counter(t for r in records for t in r['observable_difficulties']))))
    csv_write(out/'visible_face_identity_distribution.csv', [dict(visible_face=face, identity_information_visible=info,
        photos=n) for (face, info), n in Counter((str(r['visible_face']), str(r['identity_information_visible'])) for r in records).items()])


def export_controls(out, records, root):
    """Use actual paired-face confirmation; select success per failed method."""
    lookup={r['photo_id']:r for r in records}
    path=out/'same_face_pair_manual_reviews.json'
    reviews=read(path) if path.exists() else []
    confirmed=defaultdict(list);rejected=defaultdict(list)
    for observation in reviews:
        failure=lookup[observation['photo_id']];success=lookup[observation['success_id']]
        if failure['package_id']!=success['package_id']:
            raise ValueError('Control identity mismatch')
        if sha(root/observation['evidence_image'])!=observation['evidence_sha256']:
            raise ValueError('Actual control comparison image changed')
        for cfg in observation['configurations']:
            field='image_top1_correct' if cfg=='B03' else 'fusion_top1_correct'
            if failure['candidate_ledger'][field] or not success['candidate_ledger'][field]:
                raise ValueError('Control correctness does not match failing method')
        (confirmed if observation['same_identity_same_face_confirmed'] else rejected)[failure['photo_id']].append(observation)
    rows=[]
    for record in records:
        if record['objective_outcome']=='both_correct':continue
        qid=record['photo_id'];byconfig={}
        for cfg,field in [('B03','image_top1_correct'),('F03','fusion_top1_correct')]:
            failing=not record['candidate_ledger'][field]
            selected=[r for r in confirmed[qid] if cfg in r['configurations']] if failing else []
            candidates=[s['photo_id'] for s in records if failing and s['photo_id']!=qid
                and s['package_id']==record['package_id'] and s['candidate_ledger'][field]
                and record['face_control_key'] is not None and s['face_control_key']==record['face_control_key']]
            byconfig[cfg]=dict(failing_configuration=failing,confirmed_successes=[r['success_id'] for r in selected],
                alias_candidate_successes=candidates,status='actual_same_face_success_quality_not_matched' if selected else
                    'missing_confirmed_same_face_success; not evidence that no such photo exists' if failing else 'query_correct_not_a_failure',
                actual_visual_pair_evidence=selected)
        union=sorted({r['success_id'] for r in confirmed[qid]})
        rows.append(dict(photo_id=qid,package_id=record['package_id'],visible_face=record['visible_face'],
            face_control_key=record['face_control_key'],confirmed_same_identity_same_face_successes=union,
            controls_by_configuration=byconfig,B03=byconfig['B03']['confirmed_successes'],F03=byconfig['F03']['confirmed_successes'],
            configurations_with_confirmed_controls=[cfg for cfg in byconfig if byconfig[cfg]['confirmed_successes']],
            status='actual_same_face_success_quality_not_matched' if union else 'missing_confirmed_same_face_success',
            rejected_or_uncertain_pair_observations=rejected[qid],
            manual_review_path=str(path.relative_to(root)) if reviews else None,manual_review_sha256=sha(path) if reviews else None,
            residual_confounding=['angle, clarity, co-visible auxiliary faces and subject size are not pairwise matched',
                'control reuse allowed; per-method success kept distinct', 'unconfirmed does not imply nonexistence']))
    write(out/'same_face_success_controls.json',rows)
    csv_write(out/'same_face_success_controls_by_configuration.csv',[dict(photo_id=r['photo_id'],package_id=r['package_id'],
        configuration=cfg,confirmed_successes='|'.join(v['confirmed_successes']),status=v['status'])
        for r in rows for cfg,v in r['controls_by_configuration'].items() if v['failing_configuration']])
    coverage=[]
    for cfg,field in [('B03','image_top1_correct'),('F03','fusion_top1_correct')]:
        failing=[r for r in records if not r['candidate_ledger'][field]]
        mapping={r['photo_id']:r['controls_by_configuration'][cfg] for r in rows}
        for category in ['all']+sorted({c for r in failing for c in r['difficulty_labels']}):
            group=[r for r in failing if category=='all' or category in r['difficulty_labels']]
            selected=[r for r in group if mapping[r['photo_id']]['confirmed_successes']]
            coverage.append(dict(configuration=cfg,category=category,failure_queries=len(group),
                queries_with_actual_same_face_success=len(selected),control_missing_or_unconfirmed=len(group)-len(selected),
                identities_with_control=len({r['package_id'] for r in selected}),
                note='manual same physical face; quality/co-visible face differences remain; missing is not nonexistence'))
    csv_write(out/'same_face_control_coverage.csv',coverage)
    return rows


def run(root, out):
    root = Path(root).resolve()
    base = root / REL
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = root / 'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json'
    ledger = read(ledger_path)
    qids = [r['photo_id'] for r in ledger]
    if len(qids) != len(set(qids)):
        raise ValueError('Duplicate stable ledger queries')
    photos = {r['photo_id']: r for r in rows(root / 'dataset/lists/photos.csv')}
    if set(photos) != set(qids):
        raise ValueError('Inventory/ledger disagreement')
    labels = {r['photo_id']: r for r in rows(root / 'results/scene_statistics/photo_labels.csv')}
    index = read(base / 'visual_queue/index.json')
    loc = {r['batch']: r for r in index}
    by_query = {q: r for r in index for q in r['queries']}
    material_queries = [q for r in index for q in r['queries']]
    if len(material_queries) != len(set(material_queries)) or set(material_queries) != set(qids):
        raise ValueError('Material index is not exact query partition')
    hashes, sources, histories, ignored = {}, {}, defaultdict(list), []
    for path in record_files(base):
        values, envelope = unpack(read(path))
        accepted = 0
        for record in values:
            if not isinstance(record, dict) or record.get('photo_id') not in photos:
                continue
            normalized = normalize(path, record, envelope, root, base, loc, hashes)
            if normalized:
                histories[record['photo_id']].append(normalized)
                accepted += 1
        sources[str(path.relative_to(root))] = dict(sha256=sha(path), accepted_records=accepted)
        if not accepted:
            ignored.append(str(path.relative_to(root)))
    records, coverage, revisions, source_groups = [], [], [], Counter()
    for item in ledger:
        qid, photo = item['photo_id'], photos[item['photo_id']]
        history = sorted(histories[qid], key=lambda r: (r['priority'], r['source_path']))
        candidates = [r for r in history if r['scope'] == 'candidate_overview_and_query']
        primary = candidates[-1] if candidates else None
        if primary:
            source_groups[primary['source_path'].split('/visual_reviews/')[-1].split('/')[0]] += 1
        original = root / 'dataset' / photo['path']
        actual = sha(original) if original.exists() else None
        if actual != photo['sha256']:
            raise ValueError('Original image missing/changed: ' + qid)
        dimensions = {}
        for dimension in DIMS:
            observed = [r for r in history if dimension in r['scene_values']]
            selected = observed[-1] if observed else None
            value = selected['scene_values'][dimension] if selected else 'unclear'
            dimensions[dimension] = dict(value=value,
                status='reviewed_known' if selected and value != 'unclear' else 'reviewed_uncertain' if selected else 'unreviewed',
                source_path=selected['source_path'] if selected else None,
                source_sha256=selected['source_sha256'] if selected else None,
                evidence=selected['evidence'] if selected else [], original_label=labels[qid][dimension],
                original_scene_reviewed=labels[qid]['scene_reviewed'] == 'True')
        record = dict(photo_id=qid, package_id=item['package_id'], group_id=item['group_id'], split=item['split'],
            original_path=str(original.relative_to(root)), original_sha256=actual, original_image_verified=actual == photo['sha256'],
            review_status='query_and_key_candidates_overview_reviewed' if primary else 'missing_candidate_overview',
            candidate_overview_reviewed=primary is not None, visible_face=primary['visible_face'] if primary else 'unreviewed',
            identity_information_visible=primary['identity_information_visible'] if primary else 'unreviewed',
            direct_observations=[o for r in history for o in r['direct_observations']],
            candidate_differences=primary['candidate_differences'] if primary else None,
            quality_observations=[o for r in history for o in r['quality_observations']],
            observable_difficulties=sorted(set(str(t) for r in history for t in r['observable_difficulties'])),
            uncertainty=[o for r in history for o in r['uncertainty']],
            identity_ambiguity=primary['identity_ambiguity'] if primary else 'not_reviewed', scene_dimensions=dimensions,
            scene_labels={d: dimensions[d]['value'] for d in DIMS},
            scene_dimensions_reviewed=[d for d in DIMS if dimensions[d]['status'] != 'unreviewed'],
            scene_labels_reviewed=all(dimensions[d]['status'] == 'reviewed_known' for d in DIMS),
            scene_labels_audited=all(dimensions[d]['status'] != 'unreviewed' for d in DIMS),
            fine_detail_verified=any(r['fine_detail_verified'] for r in history),
            fine_detail_evidence=[r for r in history if r['scope'] == 'selected_fine_details_only' or r['fine_detail_verified']],
            primary_review_path=primary['source_path'] if primary else None,
            primary_review_sha256=primary['source_sha256'] if primary else None,
            evidence_level='direct_visual_observation; detailed scope retained separately',
            mechanism_claim='visual observation only; score/token/intervention evidence joined separately', review_history=history,
            objective_outcome='both_correct' if item['image_top1_correct'] and item['fusion_top1_correct'] else
                'image_wrong_fusion_correct' if item['fusion_top1_correct'] else 'image_correct_fusion_wrong' if item['image_top1_correct'] else 'both_wrong',
            candidate_ledger=item, face_control_key=face_key(primary['visible_face']) if primary else None)
        records.append(record)
        enrich_detail_fields(record)
        if record['reference_label_adjudication_status']=='no_capacity_conflict_on_actual_detail_recheck':
            record['observable_difficulties']=[t for t in record['observable_difficulties'] if t not in ('图库或标签歧义',)]
        assign_difficulty_labels(record)
        coverage.append(dict(photo_id=qid, batch=by_query[qid]['batch'], candidate_overview_reviewed=primary is not None,
            scene_dimensions_audited=record['scene_labels_audited'], scene_all_known=record['scene_labels_reviewed'],
            **{d + '_status': dimensions[d]['status'] for d in DIMS}, **record['scene_labels'],
            primary_review_path=record['primary_review_path'], fine_detail_verified=record['fine_detail_verified']))
        revisions.append({k: record[k] for k in ('photo_id', 'review_status', 'scene_labels', 'scene_labels_reviewed',
            'scene_dimensions_reviewed', 'scene_labels_audited', 'scene_dimensions', 'visible_face', 'identity_information_visible',
            'observable_difficulties', 'primary_review_path', 'evidence_level', 'uncertainty')})
    lookup = {r['photo_id']: r for r in records}
    controls = []
    for record in records:
        if record['objective_outcome'] == 'both_correct':
            continue
        options = [s['photo_id'] for s in records if s['package_id'] == record['package_id']
            and s['objective_outcome'] == 'both_correct' and record['face_control_key'] is not None
            and s['face_control_key'] == record['face_control_key']]
        controls.append(dict(photo_id=record['photo_id'], package_id=record['package_id'], visible_face=record['visible_face'],
            face_control_key=record['face_control_key'], confirmed_same_identity_same_face_successes=options,
            status='same_face_candidate_available_quality_not_matched' if options else 'missing_confirmed_same_face_success',
            face_mapping_source='manual conservative face equivalence table in consolidate_visual_audit.py',
            residual_confounding=['angle, clarity and subject size not all pairwise matched', 'unknown faces stay unmatched', 'control reuse allowed']))
    missing = [r['photo_id'] for r in records if not r['candidate_overview_reviewed']]
    missing_scene = {d: [r['photo_id'] for r in records if r['scene_dimensions'][d]['status'] == 'unreviewed'] for d in DIMS}
    uncertain_scene = {d: [r['photo_id'] for r in records if r['scene_dimensions'][d]['status'] == 'reviewed_uncertain'] for d in DIMS}
    summary = dict(schema='colnomic.consolidated_actual_vision.v1', queries=len(records), unique_queries=len(lookup),
        candidate_overview_reviewed=len(records) - len(missing), candidate_overview_missing=missing,
        scene_audited_by_dimension={d: len(records) - len(missing_scene[d]) for d in DIMS},
        scene_known_by_dimension={d: len(records) - len(missing_scene[d]) - len(uncertain_scene[d]) for d in DIMS},
        scene_unreviewed_ids=missing_scene, scene_uncertain_ids=uncertain_scene,
        scene_counts={d: dict(Counter(r['scene_labels'][d] for r in records)) for d in DIMS},
        all_three_scene_audited=sum(r['scene_labels_audited'] for r in records), all_three_scene_known=sum(r['scene_labels_reviewed'] for r in records),
        explicit_fine_detail_verified=sum(r['fine_detail_verified'] for r in records),
        selected_detail_observation_records=sum(bool(r['fine_detail_evidence']) for r in records),
        original_images_sha_verified=sum(r['original_image_verified'] for r in records),
        objective_outcomes=dict(Counter(r['objective_outcome'] for r in records)), primary_source_groups=dict(source_groups),
        same_face_success_controls=dict(Counter(r['status'] for r in controls)),
        mechanism_limit='1117 observed images does not imply 1117 established mechanisms',
        narrow_package_count_policy='token-agent package counts retained in raw records; actual scene rechecks supply background')
    write(out / 'records.json', records)
    write(out / 'scene_revisions.json', {'schema': 'colnomic.scene_revisions.dimension_provenance.v1', 'records': revisions})
    controls=export_controls(out,records,root)
    summary['same_face_success_controls']=dict(Counter(r['status'] for r in controls))
    summary['same_face_pair_comparisons_actually_viewed']=len(read(out/'same_face_pair_manual_reviews.json')) if (out/'same_face_pair_manual_reviews.json').exists() else 0
    export_multilabel(out, records)
    write(out / 'coverage.json', summary)
    write(out/'reference_label_adjudication.json',dict(exclusion_policy='only actual confirmed conflict; independent of feature response',
        conflict_needs_adjudication_ids=[r['photo_id'] for r in records if r['reference_label_adjudication_status']=='conflict_needs_adjudication'],
        records=[dict(photo_id=r['photo_id'],status=r['reference_label_adjudication_status'],details=r['adjudicated_detail_observations']) for r in records if r['reference_label_adjudication_status']!='not_independently_adjudicated']))
    csv_write(out / 'coverage.csv', coverage)
    write(out / 'source_manifest.json', dict(ledger_sha256=sha(ledger_path), photo_inventory_sha256=sha(root / 'dataset/lists/photos.csv'),
        definition_sha256=sha(root / 'configs/photo_labeling.json'), sources=sources, ignored_material_only_files=ignored,
        evidence_images={str(p.relative_to(root)): h for p, h in hashes.items() if p.is_relative_to(root)}))
    checks = dict(exact_query_inventory=len(records) == len(lookup) and [r['photo_id'] for r in records] == qids,
        all_original_sha_verified=all(r['original_image_verified'] for r in records),
        evidence_links_and_hashes=all(e['hash_matching'] for r in records for h in r['review_history'] for e in h['evidence']),
        source_records_sha_verified=all(sha(root / p) == v['sha256'] for p, v in sources.items()),
        scene_dimension_partition={d: sum(summary['scene_counts'][d].values()) == len(records) for d in DIMS},
        candidate_overview_complete=not missing, scene_audit_complete=all(not v for v in missing_scene.values()),
        controls_same_identity_same_face=all(all(lookup[q]['face_control_key'] == r['face_control_key']
            and lookup[q]['package_id'] == r['package_id'] for q in r['confirmed_same_identity_same_face_successes']) for r in controls))
    write(out / 'validation.json', checks)
    lines = ['# 最终逐图视觉证据整合', '',
        f"稳定ledger {len(records)}例；原图SHA验证{summary['original_images_sha_verified']}例；实际原图与关键候选overview审核{summary['candidate_overview_reviewed']}例。", '',
        f"三个场景维度已实际审核 {summary['scene_audited_by_dimension']}，已知 {summary['scene_known_by_dimension']}，未知单独保留。", '',
        'records.json逐例保留可见面、身份可见性、质量、差异、多标签困难、歧义/不确定性及完整历史记录与修订路径/SHA。材料生成不计为审核。', '',
        f"明确fine_detail_verified为true：{summary['explicit_fine_detail_verified']}；另有选定细节观察记录：{summary['selected_detail_observation_records']}。overview不等于全文细节核验。", '',
        f"同面成功控制状态：{summary['same_face_success_controls']}。只使用明确同身份同面；缺失不以另面替代，视角/质量/尺度残留混杂保留。", '',
        'scene_revisions.json供scene35_reviewed独立维度读取。原标签不改写；tokens_resume狭义包装数量保留原字段，仅真实scene_rechecks补背景定义。', '', summary['mechanism_limit']]
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>最终视觉审核</h1><p><a href="records.json">逐图证据</a> | <a href="coverage.csv">覆盖表</a> | <a href="same_face_success_controls.json">同面控制</a> | <a href="validation.json">验证</a></p><pre>' + html.escape('\n'.join(lines)) + '</pre>')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--refresh-details-only',action='store_true')
    parser.add_argument('--refresh-scene-definitions',action='store_true')
    args = parser.parse_args()
    function=refresh_scene_definitions if args.refresh_scene_definitions else refresh_details_only if args.refresh_details_only else run
    function(args.root, args.output or args.root / REL / 'visual_consolidated')
