"""Query-only weights, with exhaustive candidate bounds and exact top10/GT rank.

The scoring rule never uses labels. Labels only identify evaluation ranks; all
gallery rows receive conservative bounds, and every potentially outranking row
is evaluated. Uncomputed low-scoring values remain explicitly uncomputed.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import torch

from scripts.colnomic_mechanism.common import ROOT, data, read, write, csvout, sha
from scripts.colnomic_mechanism.geometry import grid, metrics
from scripts.colnomic_mechanism.full_fusion import bootstrap
from evaluation.scoring import valid_tokens

ETAS = np.array([0., .02, .05, .1, .2], dtype=np.float64)


def weight_mask(query):
    # Uses frozen query vectors alone: no OCR, boxes, positions, IDs or labels.
    distances = (query - query.mean(0)).norm(dim=1)
    return (distances > torch.quantile(distances, .75)).numpy()


def score_bounds(query, mask, reference_max_norms, means, etas=ETAS, tolerance=2e-6, spatial_lambda=0., alignment_gamma=0.):
    n = len(query)
    k = int(mask.sum())
    norms = query.double().norm(dim=1).numpy()
    high = norms[mask].sum() * reference_max_norms + abs(alignment_gamma) * k
    low = -norms[mask].sum() * reference_max_norms - (abs(spatial_lambda) + abs(alignment_gamma)) * k
    other_high = norms[~mask].sum() * reference_max_norms + abs(alignment_gamma) * (n-k)
    other_low = -norms[~mask].sum() * reference_max_norms - (abs(spatial_lambda)+abs(alignment_gamma)) * (n-k)
    hi = n * (means.astype(np.float64) + tolerance)
    lo = n * (means.astype(np.float64) - tolerance)
    subset_hi = np.minimum(high, hi - other_low)
    subset_lo = np.maximum(low, lo - other_high)
    upper = (hi[None] + etas[:, None] * subset_hi[None]) / (n + etas[:, None] * k)
    lower = (lo[None] + etas[:, None] * subset_lo[None]) / (n + etas[:, None] * k)
    return lower, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spatial-selection', type=Path)
    parser.add_argument('--spatial-manifest', type=Path)
    parser.add_argument('--subject-boxes', type=Path)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    d = data()
    assert not (args.spatial_selection and args.spatial_manifest), 'Choose one spatial source'
    if args.subject_boxes:
        d['boxes'] = read(args.subject_boxes)['records']
    qb, rb = d['qb'], d['rb']
    ledger = {r['photo_id']: r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    references = [valid_tokens(z) for z in rb['embeddings']]
    reference_norms = np.array([float(z.double().norm(dim=1).max()) for z in references])
    hashes = rb['sha256']
    spec = None
    spatial_lambda, alignment_gamma = 0., 0.
    means = d['image']['scores'].numpy()
    alignments = {}
    if args.spatial_manifest:
        from scripts.colnomic_mechanism.reviewed_box_sensitivity import load_spatial_inputs
        means, alignments, boxes, parameters = load_spatial_inputs(args.spatial_manifest.resolve(), d)
        d['boxes'] = boxes
        spatial_lambda, alignment_gamma = parameters['lambda_value'], parameters['gamma']
        spec = read(args.spatial_manifest)
    if args.spatial_selection:
        if not args.subject_boxes:
            historical_boxes = ROOT/'results/colnomic_full_audit/20261013_resume/full_tokens/frozen_inputs/subject_boxes.json'
            assert historical_boxes.exists(), 'Pass immutable historical subject boxes with --subject-boxes'
            d['boxes'] = read(historical_boxes)['records']
        spec = read(args.spatial_selection)
        chosen = spec['selections']['B03']['combined']
        spatial_lambda, alignment_gamma = chosen['lambda_value'], chosen['gamma']
        cached = np.load(ROOT/'results/token_mechanism_experiments/20261013_v1/stage2_geometry/full_gallery_scores.npz')
        k = int(np.argmin(abs(cached['lambdas']-spatial_lambda)))
        assert abs(cached['lambdas'][k]-spatial_lambda) < 1e-10
        means = cached['geo_penalty'][k].copy()
        qi_by = {qid: i for i, qid in enumerate(qb['ids'])}
        ri_by = {rid: i for i, rid in enumerate(rb['ids'])}
        for fit in read(ROOT/'results/token_mechanism_experiments/20261013_v1/stage2_geometry/alignments.json'):
            qi, ri = qi_by[fit['photo_id']], ri_by[fit['reference_id']]
            alignments[qi, ri] = fit
            means[qi, ri] += alignment_gamma * fit['subject_tokens']/len(valid_tokens(qb['embeddings'][qi]))
    protocol = dict(etas=ETAS.tolist(), weights='1+eta for query tokens above within-query 75th-percentile feature distance from query mean; 1 for others; divide by sum of weights',
        inputs='frozen query vectors only; no coordinates, OCR, identity or truth', spatial_lambda=spatial_lambda, alignment_gamma=alignment_gamma,
        bound='sum-subset upper=min(sum token norm upper, N*mean minus outside lower); lower analogous',
        safety_tolerance=2e-6, selection='validation identity macro R@1 then MRR then smaller eta',
        evaluated_scope='all gallery rows bounded; exact scores for every row that may enter top10 or outrank any positive; stable SHA dedup',
        full_scalar_scores_for_low_bound_rows='not computed; explicitly NaN and accompanied by rigorous bounds', historical_test_exposure=True,
        input_scores_SHA256=sha(ROOT/'cache/scores/colnomic7b/image.pt'))
    if args.spatial_manifest:
        protocol['spatial_manifest_SHA256'] = sha(args.spatial_manifest)
        protocol['spatial_manifest_path'] = str(args.spatial_manifest.resolve())
        protocol['reviewed_box_snapshot'] = spec['boxes']
        protocol['reviewed_box_snapshot_SHA256'] = spec['boxes_sha256']
    if (folder/'protocol.json').exists():
        assert read(folder/'protocol.json') == protocol
    else:
        write(folder/'protocol.json', protocol)
    records = []
    for qi, qid in enumerate(qb['ids']):
        completed = folder/'queries'/qid/'complete.json'
        if completed.exists():
            previous = read(completed)
            assert previous['query_tensor_SHA256'] == d['metadata']['photos'][qi]['all_tokens_sha256']
            assert previous['scores_SHA256'] == sha(completed.parent/'scores_and_bounds.npz')
            assert previous['proof_min_slack'] > 0
            records.extend(previous['rankings'])
            continue
        target = completed.parent
        target.mkdir(parents=True, exist_ok=True)
        q = valid_tokens(qb['embeddings'][qi])
        mask = weight_mask(q)
        aligned = np.array([(qi,j) in alignments for j in range(len(references))],dtype=float)
        lower, upper = score_bounds(q, mask, reference_norms, means[qi], spatial_lambda=spatial_lambda*aligned, alignment_gamma=alignment_gamma*aligned)
        exact = np.full(upper.shape, np.nan)
        positives = set(ledger[qid]['positive_reference_sha256'])
        positive_rows = {j for j, h in enumerate(hashes) if h in positives}
        evaluated = set()
        qxy = grid(d['metadata']['photos'][qi])[qb['embeddings'][qi].float().norm(dim=1).numpy() > 1e-8]
        order = sorted(range(len(references)), key=lambda j: (-float(upper[:, j].max()), j))
        def current():
            rankings = []
            thresholds = []
            for ei, eta in enumerate(ETAS):
                sorted_rows = sorted(evaluated, key=lambda j: (-exact[ei, j], j))
                unique = []
                seen = set()
                for j in sorted_rows:
                    if hashes[j] not in seen:
                        unique.append(j)
                        seen.add(hashes[j])
                positive_rank = next((k+1 for k,j in enumerate(unique) if hashes[j] in positives), None)
                all_positive_evaluated = positive_rows.issubset(evaluated)
                if len(unique) >= 10 and positive_rank is not None and all_positive_evaluated:
                    thresholds.append(min(exact[ei, unique[9]], max(exact[ei,j] for j in positive_rows)))
                else:
                    thresholds.append(-np.inf)
                rankings.append(dict(eta=float(eta), photo_id=qid, package_id=d['photos'][qid]['package_id'], split=d['splits'][qid],
                    rank=positive_rank, correct=positive_rank==1, top1=rb['ids'][unique[0]] if unique else None,
                    top10=[rb['ids'][j] for j in unique[:10]], scoring='spatial_weighted' if spec else 'original_matching_query_weighted'))
            return rankings, np.array(thresholds)
        rankings, thresholds = current()
        for ri in order:
            if not (upper[:, ri] >= thresholds).any():
                continue
            dot = q @ references[ri].T
            maxima = dot.max(1).values.numpy()
            fit = alignments.get((qi,ri))
            if fit:
                import cv2
                box_record = d['boxes'][qid]
                box = np.array(box_record['bbox']) / np.tile([box_record['width'],box_record['height']], 2)
                subject = np.isfinite(qxy).all(1) & ((qxy>=box[:2]) & (qxy<box[2:])).all(1)
                rxy = grid(d['metadata']['references'][ri])[rb['embeddings'][ri].float().norm(dim=1).numpy()>1e-8]
                mapped = cv2.perspectiveTransform(qxy[subject][None].astype(np.float32),np.array(fit['homography'],np.float32))[0]
                distance = np.ones((subject.sum(),len(references[ri])),np.float32)
                rvalid = np.isfinite(rxy).all(1)
                distance[:,rvalid] = np.clip(np.linalg.norm(mapped[:,None]-rxy[rvalid][None],axis=2),0,1)
                distance[~np.isfinite(distance)] = 1.
                maxima[subject] = (dot[torch.from_numpy(subject)] - spatial_lambda*torch.from_numpy(distance)).max(1).values.numpy() + alignment_gamma
            assert abs(float(maxima.mean()) - means[qi,ri]) < 3e-6
            exact[:,ri] = (float(maxima.astype(np.float64).sum()) + ETAS * float(maxima[mask].astype(np.float64).sum())) / (len(q)+ETAS*mask.sum())
            exact[0,ri] = means[qi,ri]
            assert np.all(exact[:,ri] <= upper[:,ri]+1e-8) and np.all(exact[:,ri] >= lower[:,ri]-1e-8)
            evaluated.add(ri)
            rankings, thresholds = current()
        rankings, thresholds = current()
        remaining = np.array([j for j in range(len(references)) if j not in evaluated])
        slack = float(np.min(thresholds[:,None]-upper[:,remaining])) if len(remaining) else 1.
        assert slack > 0 and all(r['rank'] is not None for r in rankings)
        assert rankings[0]['rank'] == ledger[qid]['image_positive_rank'] if spec is None else True
        np.savez_compressed(target/'scores_and_bounds.npz', exact=exact, lower=lower, upper=upper, evaluated=np.array(sorted(evaluated)), query_weight_mask=mask)
        write(completed, dict(query_tensor_SHA256=d['metadata']['photos'][qi]['all_tokens_sha256'], gallery_rows=len(references),
              evaluated_rows=len(evaluated), bounded_not_scored_rows=len(remaining), proof_min_slack=slack,
              scores_SHA256=sha(target/'scores_and_bounds.npz'), rankings=rankings))
        records.extend(rankings)
        write(folder/'progress.json', dict(completed_queries=qi+1,total_queries=len(qb['ids']),last=qid))
        print('Weighted',qi+1,'/',len(qb['ids']),'exact',len(evaluated),'of',len(references),'proof slack',slack,flush=True)
    grid_rows = []
    for eta in ETAS:
        subset = [r for r in records if r['eta']==eta and r['split']=='val']
        grid_rows.append(dict(eta=float(eta), **metrics(subset)))
    selected = max(grid_rows, key=lambda r:(r['package_macro_r1'],r['package_macro_mrr'],-r['eta']))['eta']
    write(folder/'selection.json',dict(eta=selected,criterion=protocol['selection'],validation_only=True))
    csvout(folder/'validation_grid.csv',grid_rows)
    tables = []
    original = [r for r in records if r['eta']==0]
    old = {r['photo_id']:r for r in original}
    for eta in [0.,selected]:
        pred = [r for r in records if r['eta']==eta]
        for split in ['train','val','test','all']:
            subset = [r for r in pred if split=='all' or r['split']==split]
            rescued = sum(r['correct'] and not old[r['photo_id']]['correct'] for r in subset)
            harmed = sum(not r['correct'] and old[r['photo_id']]['correct'] for r in subset)
            tables.append(dict(eta=eta,split=split,**metrics(subset),rescued=rescued,harmed=harmed,net=rescued-harmed,
                               **bootstrap(original,pred,d,split)))
    csvout(folder/'metrics.csv',tables)
    csvout(folder/'all_parameter_rankings.csv',records)
    write(folder/'verification.json',dict(status='passed',coverage=len({r['photo_id'] for r in records}),
        reference_rows=len(references),exact_top10_and_positive_rank_certified=True,all_reference_rows_have_bounds=True,
        full_scalar_values_for_every_reference=False,selected_eta=selected,spatial_selection=spec,split_checks=d['split_verification']))


if __name__ == '__main__':
    main()
