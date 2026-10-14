"""Explicit old/new human-box sensitivity from frozen ColNomic features, CPU only.

The reviewed boxes are human assistance, including on the historically observed
test split.  No new encoder, automatic box-free claim, or blind test is implied.
Unchanged queries are reused byte-for-byte from the validated geometry audit.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT, data, read, write, csvout, sha
from scripts.colnomic_mechanism.geometry import anchors, estimate, constrained, grid, metrics
from scripts.colnomic_mechanism.split_geometry import (
    atomic_npz, select_parameters, shuffled_pair, describe_ranks, group_bootstrap)
from evaluation.scoring import fuse

DEFAULT = ROOT / 'results/colnomic_full_audit/20261013_resume/reviewed_boxes'
BOX_DIR = ROOT / 'results/colnomic_full_audit/20261013_resume/manual_box_review'
OLD_GEOMETRY = ROOT / 'results/token_mechanism_experiments/20261013_v1/stage2_geometry'
OLD_SPLIT = ROOT / 'results/colnomic_full_audit/20261013_resume/split_geometry'
OLD_BOX_SHA256 = '3a5bad5014d36ceb61bd7a599874aa3119772d34287456c9f98ff52049cfd480'


def normalized_box(record):
    assert record['status'] == 'confirmed'
    return np.asarray(record['bbox'], np.float32) / np.tile([record['width'], record['height']], 2)


def box_changes(before, after):
    assert before.keys() == after.keys(), 'box snapshot axes changed'
    fields = ['bbox', 'width', 'height', 'image_sha256', 'image_path', 'status']
    return [qid for qid in before if any(before[qid].get(k) != after[qid].get(k) for k in fields)]


def recompute_query(d, qi, record, reference_anchors, postings, lambdas, shuffle=True):
    """Repropose every gallery row using the supplied box; never reuse a prior H."""
    qid = d['qb']['ids'][qi]
    box = normalized_box(record)
    qa = anchors(d['qo'][qid], box)
    counts = Counter(ri for key in qa for ri in postings.get(key, []))
    proposed = sorted(ri for ri, n in counts.items() if n >= 4)
    base = d['image']['scores'][qi].numpy()
    penalties = np.repeat(base[None], len(lambdas), 0)
    shuffled = penalties.copy()
    support = np.zeros(len(base), np.float32)
    coordinate = support.copy()
    overlap = support.copy()
    available = support.copy()
    accepted = np.zeros(len(base), bool)
    fits, audits = [], []
    qxy = grid(d['metadata']['photos'][qi])
    query = d['qb']['embeddings'][qi]
    valid = query.float().norm(dim=1).numpy() > 1e-8
    subject = np.isfinite(qxy[valid]).all(1) & ((qxy[valid] >= box[:2]) & (qxy[valid] < box[2:])).all(1)
    for ri, count in counts.items():
        overlap[ri] = count / max(1, len(qa))
    maximum = 0.
    for ri in proposed:
        ra = reference_anchors[ri]
        fit = estimate(qa, ra, qa.keys() & ra.keys(), box)
        if fit is None:
            continue
        reference = d['rb']['embeddings'][ri]
        rxy = grid(d['metadata']['references'][ri])
        _, ps, cs, ns, original = constrained(query, reference, qxy, rxy, box,
                                             fit['homography'], lambdas, 'cpu')
        error = abs(original - float(base[ri]))
        assert error < 2e-6, (qid, ri, error)
        maximum = max(maximum, error)
        penalties[:, ri] = base[ri] + np.asarray(ps, np.float64) - original
        penalties[0, ri] = base[ri]
        support[ri], coordinate[ri] = fit['geometry_support'], cs
        available[ri] = ns / int(valid.sum())
        accepted[ri] = True
        row = dict(photo_id=qid, reference_id=d['rb']['ids'][ri], **fit, subject_tokens=ns)
        fits.append(row)
        if shuffle:
            change, replay, audit = shuffled_pair(query, reference, qxy, rxy, box, row,
                                                  lambdas, qid, row['reference_id'])
            assert abs(replay - float(base[ri])) < 2e-6
            shuffled[:, ri] = base[ri] + change
            shuffled[0, ri] = base[ri]
            audits.append(dict(photo_id=qid, reference_id=row['reference_id'], **audit,
                               original_score_replay_error=abs(replay-float(base[ri]))))
    assert np.array_equal(penalties[0], base)
    assert np.all(penalties[:, ~accepted] == base[~accepted])
    return dict(penalty=penalties, shuffle=shuffled, geometry_only=support,
                coordinate_only=coordinate, anchor_overlap=overlap, subject_fraction=available,
                accepted_alignment=accepted, fits=fits, shuffle_audit=audits,
                anchor_keys=sorted(qa), proposed_indices=proposed,
                subject_tokens=int(subject.sum()), valid_query_tokens=int(valid.sum()),
                max_original_score_replay_error=maximum)


def visual_evidence(d, before, after, changed, folder):
    """Scientific before/after annotation panels, using the exact original bytes."""
    qindex = {qid: i for i, qid in enumerate(d['qb']['ids'])}
    rindex = {rid: i for i, rid in enumerate(d['rb']['ids'])}
    panels, sources = [], []
    for qid in changed:
        qi = qindex[qid]
        path = ROOT / d['qb']['paths'][qi]
        assert sha(path) == d['qb']['sha256'][qi] == after[qid]['image_sha256']
        positive = d['ledger_by_id'][qid]['positive_reference_ids'][0]
        ri = rindex[positive]
        refpath = ROOT / d['rb']['paths'][ri]
        assert sha(refpath) == d['rb']['sha256'][ri]
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert('RGB')
        assert image.size == (after[qid]['width'], after[qid]['height'])
        panel = Image.new('RGB', (1400, 620), 'white')
        draw = ImageDraw.Draw(panel)
        draw.text((10, 8), qid+' / '+d['photos'][qid]['package_id']+' / '+d['splits'][qid], fill='black')
        scale = min(640/image.width, 560/image.height)
        preview = image.resize((round(image.width*scale), round(image.height*scale)))
        panel.paste(preview, (10, 42))
        for records, color in [(before, '#e03131'), (after, '#087f23')]:
            draw.rectangle(tuple((value*scale + (10 if i%2==0 else 42)) for i,value in enumerate(records[qid]['bbox'])), outline=color, width=4)
        draw.text((665, 42), 'RED: original human box / GREEN: reviewed human box', fill='black')
        with Image.open(refpath) as raw:
            reference = ImageOps.exif_transpose(raw).convert('RGB')
        reference.thumbnail((700, 490))
        panel.paste(reference, (665, 82))
        draw.text((665, 590), 'GT reference: '+positive+' (evaluation/visual check only)', fill='black')
        target = folder / 'visuals' / (qid+'.jpg')
        target.parent.mkdir(exist_ok=True)
        panel.save(target, quality=93)
        panels.append(panel)
        sources.append(dict(photo_id=qid, original=str(path), original_sha256=sha(path),
                            gt_reference=positive, reference_original=str(refpath),
                            reference_sha256=sha(refpath), evidence=str(target.relative_to(folder))))
    sheet = Image.new('RGB', (1400, 620*len(panels)), 'white')
    for i, panel in enumerate(panels):
        sheet.paste(panel, (0, i*620))
    sheet.save(folder/'reviewed_boxes_contact_sheet.jpg', quality=91)
    write(folder/'visual_sources.json', sources)


def build(folder, old_boxes_path, new_boxes_path):
    d = data()
    old_snapshot, new_snapshot = read(old_boxes_path), read(new_boxes_path)
    before, after = old_snapshot['records'], new_snapshot['records']
    assert sha(old_boxes_path) == OLD_BOX_SHA256
    changed = box_changes(before, after)
    assert len(changed) == 5
    d['boxes'] = after
    d['ledger_by_id'] = {r['photo_id']: r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    oldvalidation = read(OLD_SPLIT/'validation.json')
    assert oldvalidation['status'] == 'passed' and oldvalidation['validated_query_ids'] == 1117
    axes = read(OLD_GEOMETRY/'score_axes.json')
    assert axes['query_ids'] == d['qb']['ids'] and axes['reference_ids'] == d['rb']['ids']
    assert axes['reference_sha256'] == d['rb']['sha256']
    oldcache = np.load(OLD_GEOMETRY/'full_gallery_scores.npz', allow_pickle=False)
    base, lambdas = oldcache['original'], oldcache['lambdas']
    assert base.shape == (1117, 5405) and np.array_equal(base, d['image']['scores'].numpy())
    penalty = oldcache['geo_penalty'].copy()
    support, coordinate = oldcache['geometry_only'].copy(), oldcache['coordinate_only'].copy()
    oldselected = np.load(OLD_SPLIT/'selected_full_gallery_scores.npz', allow_pickle=False)
    available, accepted = oldselected['subject_fraction'].copy(), oldselected['accepted_alignment'].copy()
    overlap = oldselected['B03__ocr_anchor_overlap_control__raw'].copy()
    oldshuffle = np.load(OLD_SPLIT/'shuffled_geo_penalty.npy', mmap_mode='r')
    shuffled = np.lib.format.open_memmap(folder/'shuffled_geo_penalty.npy', mode='w+',
                                         dtype=np.float32, shape=oldshuffle.shape)
    shuffled[:] = oldshuffle
    shuffled.flush()
    fits = read(OLD_GEOMETRY/'alignments.json')
    qindex = {qid: i for i,qid in enumerate(d['qb']['ids'])}
    rindex = {rid: i for i,rid in enumerate(d['rb']['ids'])}
    priorfits = defaultdict(list)
    for fit in fits:
        priorfits[fit['photo_id']].append(fit)
    reference_anchors = [anchors(d['ro'][rid]) for rid in d['rb']['ids']]
    postings = defaultdict(list)
    for ri, record in enumerate(reference_anchors):
        for key in record:
            postings[key].append(ri)
    paths = [old_boxes_path, new_boxes_path, OLD_GEOMETRY/'full_gallery_scores.npz',
             OLD_GEOMETRY/'alignments.json', OLD_GEOMETRY/'score_axes.json',
             OLD_SPLIT/'shuffled_geo_penalty.npy', OLD_SPLIT/'selected_full_gallery_scores.npz',
             OLD_SPLIT/'validation.json', OLD_SPLIT/'selection.json',
             ROOT/'cache/image_embeddings/colnomic7b/photos.pt', ROOT/'cache/image_embeddings/colnomic7b/references.pt',
             ROOT/'cache/scores/colnomic7b/image.pt', ROOT/'cache/scores/bm25/bm25.pt',
             ROOT/'cache/scores/bm25_image_fusion/colnomic7b/alpha_0.5.pt',
             ROOT/'cache/ocr/photos.json', ROOT/'cache/ocr/references.json',
             ROOT/'results/token_mechanism_analysis/F03/reproduction/token_metadata.json',
             ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json']
    for name in ['photos','groups','positives','train','val','test']:
        paths.append(ROOT/'dataset/lists'/f'{name}.csv')
    protocol = dict(schema='colnomic.reviewed_box_sensitivity.v1', source_code_sha256=sha(Path(__file__)),
        old_boxes=str(old_boxes_path.relative_to(ROOT)), new_boxes=str(new_boxes_path.relative_to(ROOT)),
        changed_queries=changed, full_gallery_shape=list(base.shape), cpu_threads=2, device='cpu',
        gamma_grid=lambdas.tolist(), lambda_grid=lambdas.tolist(),
        recomputation='For every changed query, rebuild exact OCR-anchor proposals over all 5405 reference rows, refit homographies, recheck acceptance, remap subject tokens and score each accepted pair; rejected pairs receive exact cached MaxSim.',
        reuse='1112 unchanged queries reuse SHA-validated frozen original geometry and deterministic shuffle scores exactly.',
        shuffle='Permute coordinate-to-feature association only; same newly fitted accepted pairs, H, boxes, subject masks, OCR and embeddings as real-coordinate scoring; copy real-coordinate val-selected parameters.',
        selection='all 21 validation packages only; package_macro_r1, package_macro_mrr, smaller gamma+lambda, lambda, gamma',
        human_box_assistance=True, human_box_test_assistance=True, box_gt_free_system_claimed=False,
        historical_test_already_observed=True, encoder_rerun=False, original_inputs_modified=False,
        inputs={str(path.relative_to(ROOT)):dict(bytes=path.stat().st_size, sha256=sha(path)) for path in paths})
    write(folder/'protocol.json', protocol)
    write(folder/'score_axes.json', axes)
    write(folder/'split_verification.json', d['split_verification'])
    updates, audits, fit_changes = [], [], []
    started = time.monotonic()
    for qid in changed:
        qi = qindex[qid]
        old = recompute_query(d, qi, before[qid], reference_anchors, postings, lambdas)
        observed = {r['reference_id']: r for r in old['fits']}
        expected = {r['reference_id']: r for r in priorfits[qid]}
        assert observed.keys() == expected.keys(), ('old accepted H replay mismatch',qid)
        old_penalty_error = float(np.max(np.abs(old['penalty']-penalty[:,qi])))
        old_shuffle_error = float(np.max(np.abs(old['shuffle']-oldshuffle[:,qi])))
        assert old_penalty_error < 2e-6 and old_shuffle_error < 2e-6
        assert np.array_equal(old['accepted_alignment'],accepted[qi])
        assert np.max(np.abs(old['subject_fraction']-available[qi])) < 1e-7
        old_h_error = max([float(np.max(np.abs(np.asarray(row['homography'])-np.asarray(expected[rid]['homography'])))) for rid,row in observed.items()] or [0.])
        assert old_h_error < 1e-6
        new = recompute_query(d, qi, after[qid], reference_anchors, postings, lambdas)
        newfits = {r['reference_id']: r for r in new['fits']}
        for rid in sorted(expected.keys() | newfits.keys()):
            ri = rindex[rid]
            previous, current = expected.get(rid), newfits.get(rid)
            fit_changes.append(dict(photo_id=qid, reference_id=rid,
                change='retained' if previous and current else 'removed' if previous else 'added',
                old_fit=previous,new_fit=current,
                old_subject_fraction=float(available[qi,ri]),new_subject_fraction=float(new['subject_fraction'][ri]),
                old_penalty=penalty[:,qi,ri].tolist(),new_penalty=new['penalty'][:,ri].tolist(),
                old_shuffled_penalty=np.asarray(oldshuffle[:,qi,ri]).tolist(),new_shuffled_penalty=new['shuffle'][:,ri].tolist()))
        penalty[:,qi] = new['penalty']
        shuffled[:,qi] = new['shuffle']
        support[qi], coordinate[qi] = new['geometry_only'], new['coordinate_only']
        overlap[qi], available[qi], accepted[qi] = new['anchor_overlap'], new['subject_fraction'], new['accepted_alignment']
        audits.extend(new['shuffle_audit'])
        fits = [fit for fit in fits if fit['photo_id'] != qid] + new['fits']
        updates.append(dict(photo_id=qid, package_id=d['photos'][qid]['package_id'],split=d['splits'][qid],
            before=before[qid],after=after[qid],old_anchor_keys=old['anchor_keys'],new_anchor_keys=new['anchor_keys'],
            removed_anchor_keys=sorted(set(old['anchor_keys'])-set(new['anchor_keys'])),added_anchor_keys=sorted(set(new['anchor_keys'])-set(old['anchor_keys'])),
            old_proposals=len(old['proposed_indices']),new_proposals=len(new['proposed_indices']),
            old_accepted_pairs=len(old['fits']),new_accepted_pairs=len(new['fits']),
            alignments_added=sorted(newfits.keys()-expected.keys()),alignments_removed=sorted(expected.keys()-newfits.keys()),
            old_subject_tokens=old['subject_tokens'],new_subject_tokens=new['subject_tokens'],valid_query_tokens=new['valid_query_tokens'],
            old_homography_replay_max_error=old_h_error,old_penalty_replay_max_error=old_penalty_error,old_shuffle_replay_max_error=old_shuffle_error,
            original_MaxSim_replay_max_error=new['max_original_score_replay_error'],
            scoring_rows_changed=int(np.any(new['penalty']!=old['penalty'],axis=0).sum()),
            elapsed_seconds=time.monotonic()-started))
        print('Reviewed geometry',qid,'old/new fits',len(old['fits']),len(new['fits']),'elapsed',time.monotonic()-started,flush=True)
        write(folder/'progress.json',dict(stage='geometry',completed_changed_queries=len(updates),total_changed_queries=len(changed)))
    shuffled.flush()
    unchanged = np.asarray([qid not in changed for qid in d['qb']['ids']])
    assert np.array_equal(penalty[:,unchanged],oldcache['geo_penalty'][:,unchanged])
    assert np.array_equal(shuffled[:,unchanged],oldshuffle[:,unchanged])
    assert np.array_equal(accepted[unchanged],oldselected['accepted_alignment'][unchanged])
    assert np.array_equal(available[unchanged],oldselected['subject_fraction'][unchanged])
    assert np.all(penalty[:,~accepted] == base[~accepted])
    fits.sort(key=lambda r:(qindex[r['photo_id']],rindex[r['reference_id']]))
    write(folder/'alignments.json',fits)
    write(folder/'box_effects.json',updates)
    write(folder/'alignment_changes.json',fit_changes)
    csvout(folder/'changed_query_coordinate_permutations.csv',audits)
    atomic_npz(folder/'full_gallery_scores.npz',lambdas=lambdas,original=base,geo_penalty=penalty,
        geo_bonus=penalty+lambdas[:,None,None]*available[None],geometry_only=support,coordinate_only=coordinate,
        anchor_overlap=overlap,subject_fraction=available,accepted_alignment=accepted,
        query_ids=np.asarray(d['qb']['ids']),reference_ids=np.asarray(d['rb']['ids']),reference_sha256=np.asarray(d['rb']['sha256']))
    has_text=[bool(d['qo'][qid]['text'].strip()) for qid in d['qb']['ids']]
    selection=select_parameters(d,base,penalty,lambdas,available,has_text,folder)
    write(folder/'spatial_scoring_manifest.json',dict(schema='colnomic.explicit_spatial_scoring.v1',
        scores=str((folder/'full_gallery_scores.npz').relative_to(ROOT)),scores_sha256=sha(folder/'full_gallery_scores.npz'),
        alignments=str((folder/'alignments.json').relative_to(ROOT)),alignments_sha256=sha(folder/'alignments.json'),
        boxes=str(new_boxes_path.relative_to(ROOT)),boxes_sha256=sha(new_boxes_path),
        selection=str((folder/'selection.json').relative_to(ROOT)),selection_sha256=sha(folder/'selection.json'),
        score_axes=str((folder/'score_axes.json').relative_to(ROOT)),score_axes_sha256=sha(folder/'score_axes.json'),
        parameters=selection['selections']['B03']['combined'],
        loader='scripts.colnomic_mechanism.reviewed_box_sensitivity.load_spatial_inputs',
        loader_return='means, alignments keyed by (query_index, reference_index), box records keyed by photo_id, selected parameter dict',
        human_box_assistance=True,full1117_by5405_scalar_scores=True,encoder_rerun=False))
    visual_evidence(d,before,after,changed,folder)
    return d,selection,updates


def load_spatial_inputs(manifest_path, d, selection_key='combined'):
    """Weighted interface with the exact historical per-fit scalar rounding.

    The saved split-geometry means use a float32 subject_fraction array.  The
    historical weighted scorer instead formed gamma*integer_count/integer_count
    as a Python scalar and added it to each float32 penalty.  Keeping that order
    makes unchanged weighted inputs bitwise identical; both realize the same
    mathematical rule, but can differ by one float32 ULP. An explicit
    selection_key='spatial_only' (or 'B03.spatial_only') reads the separately
    validation-selected gamma=0/lambda parameters. The historical default stays
    'combined'; overriding the choice does not mutate either stored score bank.
    """
    manifest=read(manifest_path)
    for name in ['scores','alignments','boxes','selection','score_axes']:
        assert sha(ROOT/manifest[name]) == manifest[name+'_sha256'], name+' checksum'
    cached=np.load(ROOT/manifest['scores'],allow_pickle=False)
    assert cached['query_ids'].tolist()==d['qb']['ids'] and cached['reference_ids'].tolist()==d['rb']['ids']
    assert cached['reference_sha256'].tolist()==d['rb']['sha256']
    selection_key=selection_key.removeprefix('B03.')
    assert selection_key in ['combined','spatial_only','availability_only']
    parameters=read(ROOT/manifest['selection'])['selections']['B03'][selection_key]
    declared_key=manifest.get('selection_key','combined').removeprefix('B03.')
    if selection_key==declared_key:assert parameters == manifest['parameters']
    if selection_key=='spatial_only':assert parameters['gamma']==0
    qindex={qid:i for i,qid in enumerate(d['qb']['ids'])};rindex={rid:i for i,rid in enumerate(d['rb']['ids'])}
    alignments={(qindex[row['photo_id']],rindex[row['reference_id']]):row for row in read(ROOT/manifest['alignments'])}
    means=cached['geo_penalty'][parameters['lambda_index']].copy()
    valid_counts=[int((query.float().norm(dim=1)>1e-8).sum()) for query in d['qb']['embeddings']]
    if parameters['gamma']:
        for (qi,ri),fit in alignments.items():
            means[qi,ri] += parameters['gamma']*fit['subject_tokens']/valid_counts[qi]
    boxes=read(ROOT/manifest['boxes'])['records']
    return means,alignments,boxes,parameters


def evaluate(folder,d,selection,updates):
    cache=np.load(folder/'full_gallery_scores.npz',allow_pickle=False)
    base,penalty,available=cache['original'],cache['geo_penalty'],cache['subject_fraction']
    shuffle=np.load(folder/'shuffled_geo_penalty.npy',mmap_mode='r')
    has_text=[bool(d['qo'][qid]['text'].strip()) for qid in d['qb']['ids']]
    oldcache=np.load(OLD_SPLIT/'selected_full_gallery_scores.npz',allow_pickle=False)
    oldselection=read(OLD_SPLIT/'selection.json')
    oldpred=read(OLD_SPLIT/'predictions.json')
    arrays=dict(query_ids=cache['query_ids'],reference_ids=cache['reference_ids'],reference_sha256=cache['reference_sha256'])
    predictions,tables,changes,boxchanges,boots={},{},[],[],[]
    tables=[]
    changed={row['photo_id'] for row in updates}
    for config in ['B03','F03']:
        chosen=selection['selections'][config]
        raw_methods=dict(baseline=base,
            availability_only=base+chosen['availability_only']['gamma']*available,
            spatial_only=penalty[chosen['spatial_only']['lambda_index']],
            combined=penalty[chosen['combined']['lambda_index']]+chosen['combined']['gamma']*available,
            shuffled_spatial_only=shuffle[chosen['spatial_only']['lambda_index']],
            shuffled_combined=shuffle[chosen['combined']['lambda_index']]+chosen['combined']['gamma']*available,
            ocr_alignment_support=cache['geometry_only'],coordinate_only=cache['coordinate_only'],
            ocr_anchor_overlap_control=cache['anchor_overlap'],bm25_text_control=d['bm25']['scores'].numpy())
        baseline={r['photo_id']:r for r in describe_ranks(base if config=='B03' else d['fusion']['scores'].numpy(),d)}
        for method,raw in raw_methods.items():
            standalone=method in ['ocr_alignment_support','coordinate_only','ocr_anchor_overlap_control','bm25_text_control']
            scores=raw if config=='B03' or standalone else fuse(torch.from_numpy(np.asarray(raw)),d['bm25']['scores'],has_text,.5).numpy()
            if config=='F03' and method=='baseline':scores=d['fusion']['scores'].numpy()
            key=config+'__'+method
            arrays[key+'__raw'],arrays[key+'__scored']=np.asarray(raw),np.asarray(scores)
            pred=describe_ranks(scores,d);predictions[key]=pred
            previous={r['photo_id']:r for r in oldpred[key]}
            for split in ['train','val','test','all']:
                subset=[r for r in pred if split=='all' or r['split']==split]
                repaired=sum(r['correct'] and not baseline[r['photo_id']]['correct'] for r in subset)
                harmed=sum(not r['correct'] and baseline[r['photo_id']]['correct'] for r in subset)
                tables.append(dict(configuration=config,method=method,split=split,repaired=repaired,harmed=harmed,net=repaired-harmed,
                    reviewed_box_rank_changes=sum(r['rank']!=previous[r['photo_id']]['rank'] for r in subset),
                    reviewed_box_repaired=sum(r['correct'] and not previous[r['photo_id']]['correct'] for r in subset),
                    reviewed_box_harmed=sum(not r['correct'] and previous[r['photo_id']]['correct'] for r in subset),**metrics(subset)))
                if method in ['availability_only','spatial_only','combined','shuffled_spatial_only','shuffled_combined']:
                    for unit in ['package_id','group_id']:
                        boots.extend(group_bootstrap(subset,baseline,config,method,split,unit))
            for row in pred:
                old=baseline[row['photo_id']];prior=previous[row['photo_id']]
                changes.append(dict(configuration=config,method=method,**row,
                    baseline_rank=old['rank'],baseline_top1=old['top1'],rank_delta=row['rank']-old['rank'],
                    change='repaired' if row['correct'] and not old['correct'] else 'harmed' if not row['correct'] and old['correct'] else 'unchanged',
                    previous_box_rank=prior['rank'],previous_box_top1=prior['top1'],box_rank_delta=row['rank']-prior['rank']))
                if row['photo_id'] in changed:
                    qi=d['qb']['ids'].index(row['photo_id'])
                    prior_scores=oldcache[key+'__scored'][qi]
                    boxchanges.append(dict(configuration=config,method=method,photo_id=row['photo_id'],split=row['split'],
                        old_parameters=oldselection['selections'][config].get(method.removeprefix('shuffled_')),
                        new_parameters=chosen.get(method.removeprefix('shuffled_')),
                        old_rank=prior['rank'],new_rank=row['rank'],old_top1=prior['top1'],new_top1=row['top1'],
                        actual_scores_changed=int(np.count_nonzero(scores[qi]!=prior_scores)),
                        max_abs_scored_change=float(np.max(np.abs(scores[qi]-prior_scores))),
                        corrected_by_review=bool(row['correct'] and not prior['correct']),harmed_by_review=bool(not row['correct'] and prior['correct'])))
            print('Evaluated reviewed',key,metrics(pred)['r1'],flush=True)
    atomic_npz(folder/'selected_full_gallery_scores.npz',**arrays)
    write(folder/'predictions.json',predictions)
    csvout(folder/'metrics.csv',tables)
    csvout(folder/'per_photo_rank_changes.csv',changes)
    csvout(folder/'changed_photo_rank_and_score_changes.csv',boxchanges)
    csvout(folder/'paired_group_bootstrap.csv',boots)
    write(folder/'parameter_sensitivity.json',dict(old=oldselection['selections'],new=selection['selections'],
        parameter_changes=[dict(configuration=c,method=m,old={k:oldselection['selections'][c][m][k] for k in ['gamma','lambda_value','lambda_index']},
            new={k:selection['selections'][c][m][k] for k in ['gamma','lambda_value','lambda_index']})
            for c in ['B03','F03'] for m in ['availability_only','spatial_only','combined']
            if any(oldselection['selections'][c][m][k]!=selection['selections'][c][m][k] for k in ['gamma','lambda_value','lambda_index'])]))


def verify(folder):
    d=data();d['boxes']=read(ROOT/read(folder/'protocol.json')['new_boxes'])['records']
    d['ledger_by_id']={r['photo_id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    cache=np.load(folder/'full_gallery_scores.npz',allow_pickle=False)
    selected=np.load(folder/'selected_full_gallery_scores.npz',allow_pickle=False)
    predictions=read(folder/'predictions.json')
    selection=read(folder/'selection.json')
    metrics_rows=list(csv.DictReader((folder/'metrics.csv').open()))
    count=0
    for key,saved in predictions.items():
        values=selected[key+'__scored'];assert values.shape==(1117,5405) and np.isfinite(values).all()
        replay=describe_ranks(values,d);assert replay==saved
        config,method=key.split('__');baseline={r['photo_id']:r for r in predictions[config+'__baseline']}
        if method in ['availability_only','spatial_only','combined','shuffled_spatial_only','shuffled_combined']:
            par=selection['selections'][config][method.removeprefix('shuffled_')]
            penalty=np.load(folder/'shuffled_geo_penalty.npy',mmap_mode='r') if method.startswith('shuffled_') else cache['geo_penalty']
            expected=penalty[par['lambda_index']]+par['gamma']*cache['subject_fraction']
            assert np.array_equal(selected[key+'__raw'],expected)
        for split in ['train','val','test','all']:
            rows=[r for r in replay if split=='all' or r['split']==split]
            table=next(r for r in metrics_rows if r['configuration']==config and r['method']==method and r['split']==split)
            for name,value in metrics(rows).items():assert abs(float(table[name])-value)<1e-12
            assert int(table['repaired'])==sum(r['correct'] and not baseline[r['photo_id']]['correct'] for r in rows)
            assert int(table['harmed'])==sum(not r['correct'] and baseline[r['photo_id']]['correct'] for r in rows)
        count+=len(saved)
    protocol=read(folder/'protocol.json')
    assert all(sha(ROOT/path)==entry['sha256'] for path,entry in protocol['inputs'].items())
    manifest=read(folder/'spatial_scoring_manifest.json')
    means,alignments,boxes,parameters=load_spatial_inputs(folder/'spatial_scoring_manifest.json',d)
    weighted_mean_rounding_error=float(np.max(np.abs(means-selected['B03__combined__raw'])))
    assert weighted_mean_rounding_error <= float(np.finfo(np.float32).eps)
    assert np.array_equal(selected['B03__baseline__scored'],d['image']['scores'].numpy())
    assert np.array_equal(selected['F03__baseline__scored'],d['fusion']['scores'].numpy())
    report=dict(status='passed',changed_box_queries=5,unchanged_queries_deterministically_reused=1112,
        query_ids=1117,reference_rows=5405,unique_reference_sha256=5330,all_pairs_have_actual_scalar_scores=True,
        score_pairs_per_method=1117*5405,methods_replayed=len(predictions),query_rank_records_replayed=count,
        accepted_pairs=int(cache['accepted_alignment'].sum()),queries_with_alignment=int(cache['accepted_alignment'].any(1).sum()),
        old_affected_homography_and_penalty_replay=True,affected_homographies_refit=True,
        alignment_acceptance_and_subject_tokens_refit=True,shuffle_same_new_alignment_support=True,
        shuffle_parameters_copied_from_real_val_selection=True,validation_only_parameter_selection=True,
        original_B03_F03_baselines_exactly_unchanged=True,human_box_assistance_including_observed_test=True,
        box_GT_free_automatic_method_claimed=False,encoder_rerun=False,source_hashes_rechecked=len(protocol['inputs']),
        spatial_scoring_manifest=manifest,split_verification=d['split_verification'],
        weighted_mean_rounding='historical per-fit float32 penalty += Python scalar gamma*integer_subject_tokens/integer_valid_tokens',
        weighted_mean_vs_saved_split_geometry_combined_max_error=weighted_mean_rounding_error,
        limitations=['Human-reviewed test box is auxiliary annotation assistance; historical test already inspected.',
            'Frozen-feature scoring is executable; historical gallery encoder reproduction remains unresolved.',
            'The prior weighted_spatial results use old boxes/H/scores and are not reviewed-box results.'])
    write(folder/'validation.json',report)
    write(folder/'artifacts_sha256.json',{str(p.relative_to(folder)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in ['artifacts_sha256.json','run.log','runtime.json','progress.json']})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--old-boxes',type=Path,default=BOX_DIR/'subject_boxes_before_user_review.json')
    parser.add_argument('--new-boxes',type=Path,default=BOX_DIR/'subject_boxes_after_user_review.json')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--evaluate-only',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2);cv2.setNumThreads(2)
    folder=args.output.resolve();assert folder.is_relative_to(DEFAULT)
    folder.mkdir(parents=True,exist_ok=True)
    if args.verify_only:
        report=verify(folder)
    else:
        write(folder/'runtime.json',dict(pid=os.getpid(),python=sys.executable,argv=sys.argv,device='cpu',threads=2,status='running',source_code_sha256=sha(Path(__file__))))
        if args.evaluate_only:
            d=data();d['boxes']=read(args.new_boxes)['records']
            d['ledger_by_id']={r['photo_id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
            selection,updates=read(folder/'selection.json'),read(folder/'box_effects.json')
        else:
            d,selection,updates=build(folder,args.old_boxes.resolve(),args.new_boxes.resolve())
        evaluate(folder,d,selection,updates)
        del d
        report=verify(folder)
        runtime=read(folder/'runtime.json');runtime['status']='completed';write(folder/'runtime.json',runtime)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
