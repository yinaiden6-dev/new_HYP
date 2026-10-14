"""Read-only requirement inventory and independent CPU signed-layout replay.

Writes exclusively into completion_audit/. Passing numerical replays does not
claim that the nine-section full task or natural identity-layout experiment is
complete. Original images, caches, labels and experiment artifacts are inputs.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scripts.colnomic_mechanism.common import workspace_path

csv.field_size_limit(1 << 30)
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/colnomic_full_audit/20261013_resume'
OUT = BASE / 'completion_audit'
KINDS = ['kernel_set_only', 'maxsim_only', 'pooled_only', 'pooled_plus_maxsim', 'word_ROI_assisted']
SEEDS = [17, 31, 53]


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(path).with_suffix(Path(path).suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def table(path, records):
    keys = list(dict.fromkeys(k for row in records for k in row))
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in records)


def displayed(path):
    # Direct, independently declared EXIF transform; do not serialize EXIF.
    transforms = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                  4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                  6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
                  8: Image.Transpose.ROTATE_90}
    with Image.open(path) as im:
        orientation = int(im.getexif().get(274, 1))
        assert 1 <= orientation <= 8
        return np.asarray((im.transpose(transforms[orientation]) if orientation in transforms else im).convert('RGB')).copy()


def predict(model, x):
    return ((np.asarray(x) - model['mean']) / model['scale']) @ model['coefficient'][:-1] + model['coefficient'][-1]


def model_at(path):
    stored = torch.load(path, weights_only=True, map_location='cpu')
    return {k: stored[k].numpy() for k in ['mean', 'scale', 'coefficient']}, stored


def fit(x, y, c, w):
    # Independently construct the declared objective and refit from zero.
    mean = np.average(x, axis=0, weights=w)
    scale = np.maximum(np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w)), 1e-5)
    design = torch.tensor(np.concatenate([(x - mean) / scale, np.ones((len(x), 1))], axis=1), dtype=torch.float64)
    target = torch.tensor(y, dtype=torch.float64)
    weight = torch.tensor(w, dtype=torch.float64)
    weight /= weight.sum()
    coefficient = torch.zeros(design.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([coefficient], max_iter=100, lr=1., line_search_fn='strong_wolfe', tolerance_grad=1e-8)
    def objective():
        optimizer.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(design @ coefficient, target, reduction='none') * weight).sum()
        loss = loss + coefficient[:-1].square().sum() / (2 * c * len(x))
        loss.backward()
        return loss
    optimizer.step(objective)
    return dict(mean=mean, scale=scale, coefficient=coefficient.detach().numpy())


def package_mean(values, packages):
    return float(np.mean([np.mean(np.asarray(values)[packages == p]) for p in sorted(set(packages))]))


def classification(y, scores, packages, threshold):
    correct = (scores >= threshold) == y
    comparisons = scores[y == 1, None] - scores[y == 0][None, :]
    return dict(accuracy=float(correct.mean()), package_macro_accuracy=package_mean(correct, packages),
                auc=float(np.mean((comparisons > 0) + .5 * (comparisons == 0))))


def pairs(y, scores, qids, packages):
    gap, package = [], []
    for qid in sorted(set(qids)):
        mask = qids == qid
        p, n = np.flatnonzero(mask & (y == 1)), np.flatnonzero(mask & (y == 0))
        assert len(p) == len(n) == 1
        gap.append(float(scores[p[0]] - scores[n[0]])); package.append(packages[p[0]])
    gap, package = np.asarray(gap), np.asarray(package)
    correct = (gap > 0).astype(float) + .5 * (gap == 0)
    return dict(pair_accuracy=float(correct.mean()), package_macro_pair_accuracy=package_mean(correct, package),
                gap_mean=float(gap.mean()), gap_median=float(np.median(gap)),
                gap_p05=float(np.quantile(gap, .05)), gap_p95=float(np.quantile(gap, .95)))


def threshold_from_validation(y, scores, packages):
    unique = np.unique(scores)
    candidates = np.r_[unique[0] - 1, (unique[:-1] + unique[1:]) / 2, unique[-1] + 1, 0.]
    return max((classification(y, scores, packages, t)['package_macro_accuracy'], -abs(float(t)), -float(t), float(t)) for t in candidates)[-1]


def kernel_mean(z, kernel):
    landmarks = kernel['landmarks']
    # Same raw FP32 distances, independent expression and row reduction.
    distance = np.maximum((z * z).sum(1)[:, None] + (landmarks * landmarks).sum(1)[None, :] - 2 * z @ landmarks.T, 0)
    return (np.exp(-distance / (2 * float(kernel['sigma2']))).sum(0, dtype=np.float64) / len(z)).astype(np.float32)


def visual(z, spec):
    l = spec['layout']
    return z[l['start']:l['start'] + l['count']].float().numpy()


def maxsim(q, r):
    q = q[torch.linalg.vector_norm(q.float(), dim=1) > 1e-8].float()
    r = r[torch.linalg.vector_norm(r.float(), dim=1) > 1e-8].float()
    return float(torch.amax(q @ r.T, dim=1).mean())


def roi(z, spec, bbox):
    l = spec['layout']; positions = np.arange(l['count'])
    lower = np.c_[positions % l['width'] / l['width'] * spec['raw_width'], positions // l['width'] / l['height'] * spec['raw_height']]
    upper = lower + np.array([spec['raw_width'] / l['width'], spec['raw_height'] / l['height']])
    area = np.maximum(np.minimum(upper, bbox[2:]) - np.maximum(lower, bbox[:2]), 0).prod(1)
    keep = positions[area / (spec['raw_width'] * spec['raw_height'] / l['count']) >= .01] + l['start']
    assert len(keep)
    v = z[keep].float().numpy()
    return np.r_[v.mean(0), v.std(0)]


def regression_fit(x, y, alpha, weight):
    mean = np.average(x, axis=0, weights=weight)
    scale = np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=weight)),1e-5)
    design = np.c_[(x-mean)/scale, np.ones(len(x))].astype(np.float64)
    regularizer = np.eye(design.shape[1]) * alpha; regularizer[-1,-1] = 0
    coefficient = np.linalg.solve(design.T @ (design * weight[:,None]) + regularizer, design.T @ (y * weight[:,None]))
    return dict(mean=mean,scale=scale,coefficient=coefficient)


def regression_predict(model, x):
    return np.c_[(x-model['mean'])/model['scale'], np.ones(len(x))] @ model['coefficient']


def regression_metrics(y, pred, packages):
    per_sample = np.abs(y-pred).mean(1)
    result = dict(mae=float(per_sample.mean()),package_macro_mae=package_mean(per_sample,packages),
                  r2=float(1-np.square(y-pred).sum()/max(1e-12,np.square(y-y.mean(0)).sum())))
    for dim in range(y.shape[1]):
        labels=np.minimum((y[:,dim]*4).astype(int),3); guesses=np.clip((pred[:,dim]*4).astype(int),0,3)
        result[f'dim{dim}_four_bin_accuracy']=float((labels==guesses).mean())
        result[f'dim{dim}_package_macro_four_bin_accuracy']=package_mean(labels==guesses,packages)
    return result


def recover_position_controls(output):
    """Recover missing null artifacts from immutable historical input indices.

    No current subject boxes are used: the historical saved target arrays retain
    the scope under which the original main/null statistics were computed.
    """
    prior=ROOT/'results/token_mechanism_experiments/20261013_v2/stage3_probes'
    destination=output/'position_null_recovery';destination.mkdir(exist_ok=True)
    qb=torch.load(ROOT/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',weights_only=True,mmap=True)
    qindex={qid:i for i,qid in enumerate(qb['ids'])}
    photos=rows(ROOT/'dataset/lists/photos.csv');photo_count=Counter(r['package_id'] for r in photos)
    saved_metrics=rows(prior/'position_metrics.csv');checks={};records=[];selected=[];inputs=[]
    for kind in ['absolute','subject_relative','ocr_vertical_order']:
        index=rows(prior/(kind+'_sample_index.csv'));qids=np.array([r['photo_id'] for r in index]);tokens=np.array([int(r['token_index']) for r in index])
        packages=np.array([r['package_id'] for r in index]);splits=np.array([r['split'] for r in index]);y=np.array([json.loads(r['target']) for r in index])
        sample_counts=Counter(qids);weight=np.array([1./(sample_counts[qid]*photo_count[package]) for qid,package in zip(qids,packages)])
        x=np.stack([qb['embeddings'][qindex[qid]][token].float().numpy() for qid,token in zip(qids,tokens)])
        train=splits=='train';val=splits=='val';test=splits=='test';model,stored=model_at(prior/(kind+'_head.pt'))
        alpha=float(stored['alpha']);selection=read(prior/(kind+'_selection.json'))
        mean=np.average(x[train],axis=0,weights=weight[train]);scale=np.maximum(np.sqrt(np.average((x[train]-mean)**2,axis=0,weights=weight[train])),1e-5)
        checks[kind+'_historical_train_standardization']=np.array_equal(mean,model['mean']) and np.array_equal(scale,model['scale'])
        old_predictions=np.array([json.loads(r['predicted']) for r in index]);pred=regression_predict(model,x)
        error=float(np.max(np.abs(old_predictions-pred)));checks[kind+'_main_predictions_replayed']=error<1e-10
        choices=[]
        for penalty in [.001,.01,.1,1.,10.,100.]:
            candidate=regression_fit(x[train],y[train],penalty,weight[train]);metrics=regression_metrics(y[val],regression_predict(candidate,x[val]),packages[val])
            choices.append((metrics['package_macro_mae'],penalty,candidate,metrics))
        choice=min(choices,key=lambda c:(c[0],-c[1]));coefficient_error=float(np.max(np.abs(choice[2]['coefficient']-model['coefficient'])))
        checks[kind+'_validation_alpha_refit']=choice[1]==alpha==selection['alpha'] and coefficient_error<1e-8
        selected.append(dict(probe=kind,alpha=alpha,main_prediction_max_error=error,main_refit_coefficient_max_error=coefficient_error,
                             validation_candidates=[dict(alpha=c[1],**c[3]) for c in choices]))
        np.savez_compressed(destination/(kind+'_historical_input_index.npz'),photo_ids=qids,token_indices=tokens,package_ids=packages,splits=splits,
                            targets=y,sample_weights=weight,main_predictions=pred)
        inputs.append(dict(probe=kind,samples=len(y),index_path=str((prior/(kind+'_sample_index.csv')).relative_to(ROOT)),index_sha256=digest(prior/(kind+'_sample_index.csv'))))
        for seed in SEEDS:
            labels=y[train].copy();rng=np.random.default_rng(seed)
            for package in sorted(set(packages[train])):
                ids=np.flatnonzero(packages[train]==package);labels[ids]=labels[rng.permutation(ids)]
            null=regression_fit(x[train],labels,alpha,weight[train]);prediction=regression_predict(null,x)
            metric=regression_metrics(y[test],prediction[test],packages[test]);old=next(r for r in saved_metrics if r['probe']==kind and r['control']=='label_shuffle' and int(r['seed'])==seed)
            metricerror=max(abs(float(old[k])-v) for k,v in metric.items());checks[kind+f'_shuffle_{seed}_historical_statistics_replayed']=metricerror<1e-8
            headpath=destination/f'{kind}_shuffle_{seed}_head.pt';predictionpath=destination/f'{kind}_shuffle_{seed}_predictions.npz'
            torch.save({**{k:torch.from_numpy(v) for k,v in null.items()},'alpha':alpha,'seed':seed,'scope':'historical frozen source-index recovery, no current box supervision'},headpath)
            np.savez_compressed(predictionpath,predictions=prediction,shuffled_training_targets=labels,training_sample_indices=np.flatnonzero(train))
            reopened,_=model_at(headpath);replayerror=float(np.max(np.abs(regression_predict(reopened,x)-np.load(predictionpath)['predictions'])))
            checks[kind+f'_shuffle_{seed}_saved_artifacts_replayed']=replayerror==0.
            records.append(dict(probe=kind,seed=seed,test_samples=int(test.sum()),test_identities=len(set(packages[test])),
                                max_historical_metric_error=metricerror,saved_prediction_replay_error=replayerror,
                                head_path=str(headpath.relative_to(ROOT)),prediction_path=str(predictionpath.relative_to(ROOT)),**metric))
        print('Historical position null artifacts recovered',kind,len(y),'samples',flush=True)
    protocol=dict(purpose='append-only recovery of previously unsaved null heads/predictions; independent historical metrics replay',
                  feature_source='frozen original query cache; token IDs and targets from original sample_index CSVs',
                  subject_boxes='current manual revisions are NOT consumed; historical targets preserved',
                  train_only_standardization=True,selection='replay alpha grid on historical validation identities; historical tests never select parameters',
                  shuffle_seeds=SEEDS,shuffle_scope='within training package, targets shuffled jointly across dimensions',
                  historical_test_observed=True,inputs=inputs,cache_path='cache/image_embeddings/colnomic7b/photos.pt',cache_sha256=digest(ROOT/'cache/image_embeddings/colnomic7b/photos.pt'),
                  limitation='OCR vertical ordering remains a geometric proxy; natural reading-row/identity-layout claims not established')
    result=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,recovered_heads=len(records),recovered_prediction_archives=len(records),
                position_probes=len(inputs),selection_replay=selected,scope='historical numerical position controls; no new encoder execution or revised-box result')
    write(destination/'protocol.json',protocol);write(destination/'independent_validation.json',result);table(destination/'replayed_null_metrics.csv',records)
    print('Position null recovery status',result['status'],'failed',[k for k,v in checks.items() if not v],flush=True)
    return result


def verify_layout(output):
    layout = BASE / 'symmetric_layouts'; probes = BASE / 'set_probes_symmetric'; folder = probes / 'layout'
    sources = read(layout / 'sources.json')
    metadata = {r['id']: r for r in read(ROOT / 'results/token_mechanism_analysis/F03/reproduction/token_metadata.json')['photos']}
    qb = torch.load(ROOT / 'cache/image_embeddings/colnomic7b/photos.pt', weights_only=True, map_location='cpu', mmap=True)
    qindex = {qid: i for i, qid in enumerate(qb['ids'])}
    a = dict(np.load(folder / 'new_population_baseline_inputs.npz'))
    b = dict(np.load(folder / 'layout_set_inputs.npz'))
    kernel = dict(np.load(folder / 'kernel.npz'))
    y, qids, packages, splits = [a[k] for k in ['labels', 'photo_ids', 'package_ids', 'splits']]
    checks = {}; maximum = {}; pixelrecords = []; cache_replays = []; records = {}; features = {}; committed = []
    checks['82_samples_order_matches_41_sources'] = (len(sources) == 41 and list(qids) == [s['photo_id'] for s in sources for _ in range(2)] and list(y) == [0, 1] * len(sources))
    checks['kernel_input_axes_equal_baseline_inputs'] = all(np.array_equal(a[k], b[k]) for k in ['labels', 'photo_ids', 'package_ids', 'splits'])
    checks['source_ids_and_package_ids_unique'] = len({r['photo_id'] for r in sources}) == len(sources) == len({r['package_id'] for r in sources})
    original_packages={r['package_id'] for r in rows(ROOT/'dataset/lists/photos.csv')}
    checkpoints=list((layout/'source_checkpoints').glob('package_*.json'))
    checks['all130_source_identity_checkpoints_present'] = {p.stem for p in checkpoints} == original_packages and len(original_packages)==130
    selected_sources=[read(p)['source'] for p in checkpoints if read(p).get('source')]
    checks['quality_selected_checkpoint_sources_equal_population'] = {r['photo_id'] for r in selected_sources} == {r['photo_id'] for r in sources}
    for field in ['package_id', 'group_id', 'image_SHA256', 'source_SHA256']:
        scope = {part: {r[field] for r in sources if r['split'] == part} for part in ['train', 'val', 'test']}
        checks[field + '_split_disjoint'] = all(not scope[x] & scope[z] for x, z in [('train','val'), ('train','test'), ('val','test')])
    for source in sources:
        qid = source['photo_id']; qi = qindex[qid]; spec = metadata[qid]
        original = displayed(workspace_path(source['original_path'])); x1, y1, x2, y2 = source['bbox_display']; step = source['shift_pixels']
        band = np.zeros(original.shape[:2], bool); band[y1:y2, x1-step:x2+step] = True
        block = original[y1:y2, x1:x2]
        checks[qid + '_original_SHA'] = digest(workspace_path(source['original_path'])) == source['image_SHA256'] == qb['sha256'][qi]
        images = {r['arm']: displayed(workspace_path(r['path'])) for r in source['arms'] if r['arm'] != 'base'}
        pixels = dict(photo_id=qid, package_id=source['package_id'], split=source['split'], step_pixels=int(step),
                      dimensions_preserved=all(v.shape == original.shape for v in images.values()),
                      outside_band_exact=all(np.array_equal(v[~band], original[~band]) for v in images.values()),
                      glyph_block_exact=all(np.array_equal(images[arm][y1:y2,x1+offset:x2+offset], block) for arm, offset in [('left',-step),('right',step),('sham',0)]),
                      editor_controls_leave_word_exact=all(np.array_equal(images[arm][y1:y2,x1:x2], block) for arm in ['editor_left','editor_right']))
        colors, counts = np.unique(images['sham'][band], axis=0, return_counts=True)
        pixels['RGB_histogram_conserved'] = all(np.array_equal(colors, np.unique(images[arm][band],axis=0,return_counts=True)[0]) and np.array_equal(counts,np.unique(images[arm][band],axis=0,return_counts=True)[1]) for arm in ['left','right'])
        # Reconstruct every deterministic edit from original RGB and declared boxes.
        bg = np.rint(np.median(np.r_[original[y1:y2,x1-step:x1].reshape(-1,3),original[y1:y2,x2:x2+step].reshape(-1,3)],axis=0)).astype(np.uint8)
        canvas = original.copy(); canvas[y1:y2,x1-step:x2+step] = bg
        expected = {}
        for arm, offset in [('sham',0), ('left',-step), ('right',step)]:
            image = canvas.copy(); image[y1:y2,x1+offset:x2+offset] = block; expected[arm] = image
        shade = np.rint(np.median(block[:,:2],axis=1)).astype(np.uint8)
        for arm, lo, hi in [('editor_left',x1-step,x1), ('editor_right',x2,x2+step)]:
            image = expected['sham'].copy(); image[y1:y2,lo:hi] = shade[:,None,:]; expected[arm] = image
        pixels['all_declared_edits_reconstructed_exactly'] = all(np.array_equal(images[arm], expected[arm]) for arm in images)
        pixelrecords.append(pixels)
        checks[qid + '_pixel_controls'] = all(v for k,v in pixels.items() if isinstance(v, bool))
        records[qid] = {}
        for arm in ['fresh_base', 'sham', 'left', 'right', 'editor_left', 'editor_right']:
            path = layout / 'embeddings' / (qid + '_' + arm + '.pt')
            record = torch.load(path, weights_only=True, map_location='cpu'); records[qid][arm] = record
            checks[qid + '_' + arm + '_encoded_input_SHA'] = record['image_sha256'] == digest(workspace_path(record['path']))
            checks[qid + '_' + arm + '_finite_features'] = bool(torch.isfinite(record['embedding']).all())
            committed.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path), bytes=path.stat().st_size))
        fresh = records[qid]['fresh_base']['embedding']; cached = qb['embeddings'][qi]
        error = float((fresh.float() - cached.float()).abs().max())
        cache_replays.append(dict(photo_id=qid, exact=torch.equal(fresh,cached), maximum_error=error))
        checks[qid + '_cached_original_exact'] = torch.equal(fresh, cached)
        features[qid] = {}
        for arm, record in records[qid].items():
            v = visual(record['embedding'], spec); moments = np.r_[v.mean(0), v.std(0)]
            scalar = np.array([maxsim(fresh,record['embedding'])])
            box = record.get('word_bbox') or records[qid]['sham']['word_bbox']
            features[qid][arm] = dict(kernel_set_only=kernel_mean(v,kernel), maxsim_only=scalar, pooled_only=moments,
                                      pooled_plus_maxsim=np.r_[moments,scalar], word_ROI_assisted=roi(record['embedding'],spec,box))
        print('Independent signed-layout input replay', len(pixelrecords), '/', len(sources), flush=True)
    checks['246_encodes_manifest_complete'] = len(read(layout/'encoding_manifest.json')) == len(committed) == 246
    xmap = dict(kernel_set_only=b['kernel_set'], maxsim_only=a['maxsim'], pooled_only=a['moments'],
                pooled_plus_maxsim=np.c_[a['moments'], a['maxsim']], word_ROI_assisted=a['word_ROI'])
    for kind, x in xmap.items():
        reconstructed = np.array([features[qid]['left' if label == 0 else 'right'][kind] for qid,label in zip(qids,y)])
        error = float(np.max(np.abs(x - reconstructed))); maximum[kind + '_input_error'] = error
        checks[kind + '_inputs_replayed_from_embedded_tokens'] = error < 2e-6
    # Replay unlabelled train-only landmarks and sigma selection.
    named = [(s['photo_id']+'_'+arm, visual(records[s['photo_id']][arm]['embedding'],metadata[s['photo_id']])) for s in sources if s['split']=='train' for arm in ['left','right']]
    rng = np.random.default_rng(20261015); samples = []; sample_index = []
    for name, v in named:
        take = np.sort(rng.choice(len(v),min(16,len(v)),replace=False)); samples.append(v[take]); sample_index.extend((name,int(i)) for i in take)
    sample = np.concatenate(samples).astype(np.float32); selected = rng.choice(len(sample),64,replace=False)
    landmarks = sample[selected]
    one = sample[rng.integers(len(sample),size=2048)]; two = sample[rng.integers(len(sample),size=2048)]
    sigma = np.median(np.square(one-two).sum(1))
    landmarks_index = rows(folder/'kernel_landmark_index.csv')
    checks['train_only_kernel_landmarks_exact_replay'] = np.array_equal(kernel['landmarks'],landmarks) and all((r['source_id'],int(r['visual_offset']))==sample_index[int(j)] for r,j in zip(landmarks_index,selected))
    checks['kernel_sigma_exact_replay'] = float(kernel['sigma2']) == float(sigma)
    predrows = rows(folder/'layout_predictions.csv'); nullrows = rows(folder/'shuffle_predictions.csv'); editorrows = rows(folder/'editing_sham_responses.csv')
    metricrows = rows(folder/'layout_metrics.csv'); train = splits=='train'; val = splits=='val'
    count = Counter(packages); w = np.array([1./count[p] for p in packages]); replay = []; testmetrics=[]
    for kind, x in xmap.items():
        model, stored = model_at(folder/(kind+'_head.pt')); scores = predict(model,x)
        pr = [r for r in predrows if r['probe']==kind]
        error = float(np.max(np.abs(scores - np.array([float(r['logit']) for r in pr])))); maximum[kind+'_logit_error']=error
        checks[kind+'_saved_logits_replayed'] = len(pr)==len(y) and error<1e-10 and all(r['photo_id']==qid and int(r['label'])==label for r,qid,label in zip(pr,qids,y))
        mean=np.average(x[train],axis=0,weights=w[train]); scale=np.maximum(np.sqrt(np.average((x[train]-mean)**2,axis=0,weights=w[train])),1e-5)
        checks[kind+'_standardization_train_only_replayed'] = np.array_equal(mean,model['mean']) and np.array_equal(scale,model['scale'])
        selection = read(folder/(kind+'_selection.json')); candidates=[]
        max_candidate_error = 0.
        for c in [.01,.1,1.,10.]:
            fitted=fit(x[train],y[train],c,w[train]); vs=predict(fitted,x[val]); pm=pairs(y[val],vs,qids[val],packages[val]); cm=classification(y[val],vs,packages[val],0.)
            candidate = next(r for r in selection['candidates'] if r['C']==c)
            max_candidate_error=max(max_candidate_error,max(abs(float(candidate[k])-v) for k,v in pm.items()),abs(float(candidate['auc'])-cm['auc']))
            candidates.append((pm['package_macro_pair_accuracy'],cm['auc'],-c,c,fitted))
        winner=max(candidates,key=lambda r:r[:3]); selected_c=winner[3]
        cutoff=threshold_from_validation(y[val],predict(model,x[val]),packages[val])
        maximum[kind+'_refit_candidate_metric_error']=max_candidate_error
        checks[kind+'_C_selection_replayed_on_validation_only'] = float(stored['C'])==selected_c==selection['C'] and max_candidate_error<1e-7
        checks[kind+'_threshold_validation_only_replayed'] = abs(float(stored['threshold'])-cutoff)<1e-12 and abs(selection['threshold']-cutoff)<1e-12
        coefficient_error=float(np.max(np.abs(winner[4]['coefficient']-model['coefficient'])))
        maximum[kind+'_refit_selected_coefficient_error']=coefficient_error
        checks[kind+'_selected_head_refit'] = coefficient_error<1e-7
        for seed in SEEDS:
            null, ns=model_at(folder/f'{kind}_shuffle_{seed}_head.pt'); shuffled=y[train].copy(); rng=np.random.default_rng(seed)
            for package in sorted(set(packages[train])):
                indices=np.flatnonzero(packages[train]==package); shuffled[indices]=shuffled[rng.permutation(indices)]
            nr=[r for r in nullrows if r['probe']==kind and int(r['seed'])==seed]; nullscores=predict(null,x)
            error=float(np.max(np.abs(nullscores-np.array([float(r['logit']) for r in nr]))))
            observed=np.array([int(r['shuffled_training_label']) for r in nr if r['split']=='train'])
            fitted=fit(x[train],shuffled,float(stored['C']),w[train]); coef=float(np.max(np.abs(fitted['coefficient']-null['coefficient'])))
            cutoff=threshold_from_validation(y[val],nullscores[val],packages[val])
            passed=(len(nr)==len(y) and np.array_equal(shuffled,observed) and error<1e-10 and coef<1e-7 and np.array_equal(null['mean'],mean) and np.array_equal(null['scale'],scale) and abs(float(ns['threshold'])-cutoff)<1e-12)
            checks[kind+f'_shuffle_{seed}_replay_and_refit']=bool(passed)
            replay.append(dict(probe=kind,seed=seed,logit_error=error,refit_coefficient_error=coef,passed=bool(passed)))
        for part in ['train','val','test']:
            keep=splits==part; pm=pairs(y[keep],scores[keep],qids[keep],packages[keep])
            for strategy,t in [('fixed_zero',0.),('validation_calibrated',float(stored['threshold']))]:
                expected={**classification(y[keep],scores[keep],packages[keep],t),**pm}
                mr=next(r for r in metricrows if r['probe']==kind and r['split']==part and r['threshold_strategy']==strategy and not r.get('control'))
                error=max(abs(float(mr[k])-v) for k,v in expected.items())
                checks[kind+'_'+part+'_'+strategy+'_metrics_replay']=error<1e-10
                if part=='test': testmetrics.append(dict(probe=kind,threshold_strategy=strategy,**expected,identities=int(len(set(packages[keep]))),scope='signed synthetic word position; not natural identity accuracy'))
        for source in sources:
            qid=source['photo_id']; response=next(r for r in editorrows if r['probe']==kind and r['photo_id']==qid)
            arm={arm:float(predict(model,np.array([values[kind]]))[0]) for arm,values in features[qid].items()}
            expected=dict(editor1_logit=arm['editor_left'],editor2_logit=arm['editor_right'],editor_gap=arm['editor_right']-arm['editor_left'],
                          base_logit=arm['fresh_base'],sham_logit=arm['sham'],sham_minus_base=arm['sham']-arm['fresh_base'],
                          class0_minus_sham=arm['left']-arm['sham'],class1_minus_sham=arm['right']-arm['sham'],layout_gap=arm['right']-arm['left'],
                          layout_gap_minus_editor_gap=(arm['right']-arm['left'])-(arm['editor_right']-arm['editor_left']))
            error=max(abs(float(response[k])-v) for k,v in expected.items()); maximum[kind+'_editor_response_error']=max(maximum.get(kind+'_editor_response_error',0.),error)
        checks[kind+'_205_editing_control_responses_replayed']=maximum[kind+'_editor_response_error']<2e-5
        print('Independent head selection/logit/shuffle/control replay',kind,flush=True)
    completion=read(probes/'completion.json'); snapshots=list(probes.glob('source_executed_*.py'))
    checks['executed_probe_source_snapshot_SHA']=any(digest(p)==completion['source_code_sha256'] for p in snapshots)
    result=dict(status='passed' if all(checks.values()) else 'failed',verified_utc=datetime.now(timezone.utc).isoformat(),checks=checks,maximum_errors=maximum,
                coverage=dict(source_identities=len(sources),samples=len(y),encodes=len(committed),heads=len(KINDS),shuffle_heads=len(replay),pixel_arms=5*len(sources),
                              sources_by_split=dict(Counter(s['split'] for s in sources)),cached_originals_exact=sum(r['exact'] for r in cache_replays)),
                refit_implementation='independent declared regularized logistic objective and independent scoring/selection/metrics; same documented torch LBFGS optimizer',
                scope='all symmetric synthetic position inputs, heads, selections, controls and pixels; no new encoding, no natural same-text identity Probe and no causal attribution to encoder layers')
    table(output/'signed_layout_pixel_replay.csv',pixelrecords);table(output/'signed_layout_cache_replay.csv',cache_replays)
    table(output/'signed_layout_shuffle_replay.csv',replay);table(output/'signed_layout_test_metrics.csv',testmetrics)
    write(output/'signed_layout_input_SHA256.json',committed);write(output/'signed_layout_independent_validation.json',result)
    print('Signed-layout independent status',result['status'],'failed',[k for k,v in checks.items() if not v],flush=True)
    return result


def inventory(output):
    """Inspect current artifacts; reserve unresolved task requirements explicitly."""
    dossiers=read(BASE/'full_tokens/dossiers_all.json'); qids=[r['photo_id'] for r in dossiers]
    ledger=read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')
    expected={r['photo_id'] for r in ledger}; checks={}; evidence=[]
    checks['1117_dossier_IDs_no_missing_or_duplicate']=len(qids)==len(set(qids))==len(expected)==1117 and set(qids)==expected
    matrices=read(BASE/'full_tokens/matrix_index_all.json')
    comparisons=rows_gzip(BASE/'full_tokens/comparisons_all.csv.gz')
    checks['1117_actual_query_case_cards']=sum((BASE/'full_tokens/queries'/qid/'index.html').exists() for qid in qids)==1117
    checks['1648_actual_contrast_records']=len(comparisons)==1648
    checks['2765_matrix_records_and_files']=len(matrices)==2765 and len(list((BASE/'full_tokens/matrices').rglob('*.npz')))==2765
    packages={r['package_id'] for r in rows(ROOT/'dataset/lists/photos.csv')}
    source_checkpoints=list((BASE/'symmetric_layouts/source_checkpoints').glob('package_*.json'))
    checks['130_symmetric_population_identity_checkpoints']=len(packages)==130 and {p.stem for p in source_checkpoints}==packages
    checkpoint_sources=[read(p)['source']['photo_id'] for p in source_checkpoints if read(p).get('source')]
    checks['41_symmetric_sources_equal_selected_population']=set(checkpoint_sources)=={r['photo_id'] for r in read(BASE/'symmetric_layouts/sources.json')} and len(checkpoint_sources)==41
    if (output/'existing_tests.xml').exists():
        import xml.etree.ElementTree as ET
        cases=ET.parse(output/'existing_tests.xml').findall('.//testcase')
        checks['4_existing_related_tests_passed']=len(cases)==4 and all(c.find('failure') is None and c.find('error') is None for c in cases)
    verifiers=['full_tokens/independent_validation.json','checkpoint_2108/fusion_independent_validation.json','checkpoint_2108/weighted_independent_validation.json',
               'checkpoint_2108/set_probes_independent_validation.json','split_geometry/validation.json','detail_interventions/analysis/independent_validation.json','scene35/validation.json']
    for relative in verifiers:
        path=BASE/relative; value=read(path)
        evidence.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=digest(path),status=value.get('status',value.get('passed')),keys=list(value)))
    review_files=list((BASE/'visual_reviews').rglob('*.json')); unique_reviews={}; review_schema=Counter()
    for path in review_files:
        value=read(path); review_schema[type(value).__name__]+=1
        values=value if isinstance(value,list) else value.get('records',value.get('reviews',value.get('cases',[]))) if isinstance(value,dict) else []
        if isinstance(value,dict) and value.get('photo_id'): values=[value]
        if isinstance(values,dict): values=[{**v,'photo_id':v.get('photo_id',k)} for k,v in values.items() if isinstance(v,dict)]
        for r in values:
            if isinstance(r,dict) and r.get('photo_id') in expected:unique_reviews.setdefault(r['photo_id'],[]).append(str(path.relative_to(ROOT)))
    # A record's existence is evidence availability, not proof that the complete
    # image/candidate set and details were actually inspected.
    scene_interface=read(BASE/'scene35/photo_evidence_interface.json')
    scenes=read(BASE/'scene35/manifest.json'); natural=read(BASE/'scene35/natural_layout_OCR_bag_inventory.json')
    symmetric=read(BASE/'symmetric_layouts/sources.json'); details=read(BASE/'detail_interventions/sources.json')
    layout_validation=read(output/'signed_layout_independent_validation.json') if (output/'signed_layout_independent_validation.json').exists() else {}
    totals=dict(query_dossiers=len(dossiers),contrast_records=len(comparisons),matrix_index_records=len(matrices),review_artifact_files=len(review_files),
                query_ids_with_parsed_visual_records=len(unique_reviews),review_schema=dict(review_schema),source_identities_symmetric=len(symmetric),
                source_identities_detail=len(details),symmetric_and_detail_identity_union=len({r['package_id'] for r in symmetric+details}),
                scene_interfaces=len(scene_interface),natural_same_text_distinct_SHA_OCR_proxy_groups=natural['exact_normalized_bag_groups_distinct_SHA'])
    requirements=[]
    def item(section,requirement,status,paths,proof,limitations=''):
        requirements.append(dict(section=section,requirement=requirement,status=status,authoritative_paths=paths,proof=proof,limitations=limitations))
    item('I','1117查询/5405记录/5330 SHA设计与轴、原始排名、精度有效Token','verified_existing',['full_tokens/independent_validation.json','full_tokens/baseline_verification.json','full_tokens/original_images_verified.json'], '独立验证检查3351完整排名、2765逐点FP32矩阵、1225578 Token；原图1974均有SHA', '完整图库冻结scalar replay不等于重新编码全部图像')
    item('I','训练/验证/测试身份、来源组、设计及SHA隔离','verified_existing',['full_tokens/split_verification.json','split_geometry/split_verification.json'],'783/173/161查询；91/21/18身份；实际交集为空','主体框有用户授权修改，原冻结分析继续用不可变旧快照')
    item('I','冻结重放/重新编码/旧错误排序分别验证','partial_with_recorded_limit',['symmetric_layouts/embeddings/complete.json','detail_interventions/embeddings/complete.json','scene35/historical_gallery_source_audit.json','../../token_mechanism_experiments/20261013_v2/cache_diagnosis'], '41 synthetic源原图均exact；近期PIL/backend漂移诊断已保存','旧图库所有5405条reused；原source/processor/intermediate缺失，唯一历史原因未确定；自然全量重新编码未执行')
    item('II','1117唯一主档案、候选、完整排名与对照卡','verified_existing',['full_tokens/dossiers_all.json','full_tokens/comparisons_all.csv.gz','full_tokens/full_gallery_rankings.npz','full_tokens/index.html'],f'实际档案{len(dossiers)}，独立候选对{len(comparisons)}；数据轴来自ledger','主档案的面/质量/观察字段仍可能保持未审核，应连接新增视觉审核')
    item('II','1117张逐图可追溯视觉审核、细节放大','pending_consolidation',['visual_reviews','visual_queue','full_tokens/visual_material','scene35/photo_evidence_interface.json'],f'实际存在{len(review_files)}审核JSON；本审计解析到{len(unique_reviews)}独立查询有记录','记录存在不能代替完整视觉验收；需统一审后候选、放大证据、审核程度和缺失字段')
    item('II','互斥4客观结果/多标签困难/机制证据分开统计','partial_with_recorded_limit',['scene35/original_labels','scene35/verified_labels','scene35/photo_evidence_interface.json','full_tokens/dossiers_all.json'],'4结果独立重算829/143/41/104；35配置39095预测无缺失','E标签为历史自动规则；可观察多标签频数/重叠/身份分布与机制证据需最终视觉合并')
    item('II','各失败类同身份同面成功控制及复用/残留混杂','pending_consolidation',['visual_reviews','../../token_mechanism_experiments/20261013_v1/stage1_same_face_controls.csv'],'历史重点家族已有直接面核验；不存在成功同面的例子明确保存','尚未发现全1117可追溯的控制关系汇总；不得另一面代替同面')
    item('III','全1117图像/融合最强错误对、多个正例、逐Token top1/top2/Delta','verified_existing',['full_tokens/queries','full_tokens/matrix_index_all.json','full_tokens/independent_validation.json'],'1648去重contrasts、2765压缩matrix、1225578 Token；矩阵逐点误差0','部分几何/语义关系依赖OCR/旧主体框，不是人工最终区域')
    item('III','Delta与原差额守恒、互斥区域守恒','verified_existing',['full_tokens/independent_validation.json','detail_interventions/analysis/independent_validation.json'],'Token/区域误差最大1.7959e-7；detail switch分区最大1.9847e-7','只证明确定性数学分解；不证明语义区域标注正确')
    item('III','布局/图案颜色/关键规格/公共文字/主体/背景/模板语义区域','partial_with_recorded_limit',['full_tokens/validation.json','full_tokens/queries','manual_box_review'],'规格/OCR/主体/背景/模板采用互斥代理分区并守恒','difference_layout、pattern_color未被独立人工语义分配；正在重算人工改框敏感性')
    item('III','跨行/跨区域/跨面/几何/多对一/覆盖/近并列与成功/Token数控制','partial_with_recorded_limit',['full_tokens/queries','visual_reviews','split_geometry'],'数字诊断均存在，不把集中/跨行直接认定机制','跨包装面和语义区域需视觉标签；成功控制/Token数分层关联需总报告整合')
    item('IV','gamma/lambda拆开、空间shuffle保持可用性、OCR/文本控制、三历史例','verified_existing',['split_geometry/validation.json','split_geometry/historical_and_focal_case_checks.json','split_geometry/selection.json'],'28方法×1117 rank replay；20326 accepted pairs/shuffles；所有selected公式误差0','仅第三历史例对coordinate shuffle敏感；前两例不能主张真正空间对应修复；OCR锚点有文字信息')
    item('IV','匹配×均匀/Query自权重完整图库 Recall/MRR/修复损害/区间','verified_existing_with_scalar_scope',['weighted_original/verification.json','weighted_spatial/verification.json','checkpoint_2108/weighted_independent_validation.json','weighted_original/metrics.csv','weighted_spatial/metrics.csv'],'两方法各1117全5405 bounds；exact top10和最佳正例rank证明通过','未计算所有低界候选scalar，NaN显式保留；不能称完整排序或所有scalar已输出')
    item('IV','全部改分方法按错误类型/身份/来源bootstrap','partial_with_recorded_limit',['split_geometry','weighted_original','weighted_spatial'],'split/身份宏与来源或身份bootstrap实际产物存在','最终可观察错误类型需合并视觉多标签；historical test已见过')
    item('V','全部查询融合图像/文字/归一化分解与OCR词贡献','verified_existing',['full_fusion/all_query_decomposition.csv','full_fusion/OCR_term_contributions.csv','checkpoint_2108/fusion_independent_validation.json'],'1117×5405五方法全部排序重放；5种methods；差额守恒3.3379e-6','BM25词解析与FP32原分数存在0.001285舍入误差，已记录')
    item('V','新融合权重/门控train-val选择且检查救回损害','verified_existing',['full_fusion/selection.json','full_fusion/validation_grid.csv','full_fusion/per_query_rank_changes.csv','full_fusion/group_bootstrap.json'],'constant/query_confidence_gate全部完整图库；选参grid独立重放','融合失败不可据此归因编码器细节损失')
    position_recovery=read(output/'position_null_recovery/independent_validation.json') if (output/'position_null_recovery/independent_validation.json').exists() else {}
    item('VI','冻结位置/垂直顺序 MAE R2 baseline与文字位置相关控制','verified_controls_with_target_limit' if position_recovery.get('status')=='passed' else 'partial_with_recorded_limit',['../../token_mechanism_experiments/20261013_v2/stage3_probes/position_metrics.csv','../../token_mechanism_experiments/20261013_v2/stage3_probes/ocr_vertical_order_y_only_test_predictions.csv','completion_audit/position_null_recovery'],'位置和OCR垂直序号、训练均值/多数类/坐标基线实际存在；本轮恢复9 null头/全样本预测并重放旧指标' if position_recovery.get('status')=='passed' else '原位置和OCR垂直序号基线存在，但null头/预测尚未完整保存','OCR垂直序号不是真阅读行；位置预测不证明同文字身份布局可分；恢复用旧targets不消费新人工框')
    item('VI','MaxSim/轻量MaxSim/pooled/pooled+MaxSim/Token集合头','verified_existing_with_representation_limit',['set_probes/natural','set_probes/natural/full_gallery','checkpoint_2108/set_probes_independent_validation.json'],'5头完整1117×5405 rank重放；2234受限训练样本、真实困难对、3shuffle/头','kernel mean 64维有限分布summary，不保留所有Token集合关系；Probe失败不证明信息不存在')
    item('VI','自然同文字身份相关词位置/换行Probe','unresolved_missing_eligible_proxy',['scene35/natural_layout_OCR_bag_inventory.json','symmetric_layouts/protocol.json','set_probes_symmetric/protocol.json'],'完整图库规范化OCR bag相同且不同SHA组=0，只是自动proxy','真实自然samewords是否存在仍未判定；synthetic signed-position与amplitude各stage都不能替代自然身份/换行Probe')
    item('VI','所有质量合格独立源signed-position、local/pooled/set对比/3null/权重预测种子','verified_this_audit' if layout_validation.get('status')=='passed' else 'verification_pending',['completion_audit/signed_layout_independent_validation.json','symmetric_layouts/source_checkpoints','set_probes_symmetric/layout'], '独立核验41身份27/9/5；82 samples；5头、15shuffle；参数/阈值仅val；kernel train-only','ROI辅助头有显式空间监督；测试只有5身份且历史已观察；v2单向幅度38身份20/8/10保留pilot标签')
    item('VII','编码首次漂移位置/环境/像素/processor/模板/投影前后','partial_with_recorded_limit',['../../token_mechanism_experiments/20261013_v2/cache_diagnosis','scene35/historical_gallery_source_audit.json','symmetric_layouts/embeddings'],'近期query backend原因已定位；模型/环境/精度多recipe记录；41原图exact','历史gallery processor/intermediate/source不可获得，不能唯一归因GPU或CPU卸载；不可合并新旧gallery成新准确率')
    item('VII','确定性glyph移动/原图sham等面积编辑像素守恒','verified_this_audit' if layout_validation.get('status')=='passed' else 'verification_pending',['completion_audit/signed_layout_pixel_replay.csv','detail_interventions/protocol.json','detail_interventions/analysis/independent_validation.json'], '41布局×5像素arm全reconstruct，60 detail encodes/12身份/4客观结果各3；145 detail matrices','像素守恒为signed布局；detail遮挡守恒指评分/contribution，不是遮挡后所有文字/像素不变')
    item('VII','朝更符合/更不符合Query自然身份布局方向性响应','partial_with_recorded_limit',['set_probes_symmetric/layout/editing_sham_responses.csv','../../token_mechanism_experiments/20261013_v2/stage4_extended_photos_v3/analysis'], 'signed left/right equal幅度均执行并有edit-minus-sham/control responses','left/right类不是自然identity更匹配/更不匹配方向；原focal精选对不能冒充覆盖多个自然身份的方向实验')
    item('VII','逐Token特征变化/match切换/分区抵消/组合交互','verified_existing_for_detail_scope',['detail_interventions/analysis','../../token_mechanism_experiments/20261013_v2/stage4_extended_photos_v3/analysis'],'12 detail源保存matrix、gap、switch贡献、regions、both-minus-single交互并通过独立守恒','conditional新旧混合gallery rank不是新运行时全gallery accuracy；synthetic-layout完整Token switch未普遍分析')
    item('VIII','每图直接观察/支持/竞争/反证/未知/等级，多因素机制且区分证据','pending_consolidation',['full_tokens/dossiers_all.json','visual_reviews','scene35/photo_evidence_interface.json'],'主档案schema具备字段；视觉追加记录正在统一','数值coverage不等于机制均查明；最终Token不能归因某编码层')
    item('IX','中文总报告+HTML首页五问题+可浏览全档案证据','final_delivery_pending',['full_tokens/index.html','scene35/REPORT.md','detail_interventions/analysis/REPORT.md'],'子模块报告与1117卡存在','尚未发现resume根总REPORT.md/index.html；root需最终合并报告和有效链接')
    item('IX','代码环境协议日志恢复/一键复现+只验证入口','partial_with_recorded_limit',['checkpoint_2108/TASK_SPEC_latest.txt','RESUME.md','full_tokens/REPRODUCE.md','scripts/colnomic_mechanism/verify_full_tokens.py','scripts/colnomic_mechanism/verify_full_fusion.py','scripts/colnomic_mechanism/verify_weighted.py','scripts/colnomic_mechanism/verify_set_probes.py','scripts/colnomic_mechanism/completion_audit.py'],'各模块运行与验证入口/历史test实际存在','尚未发现覆盖所有resume模块的一键执行/只验统一入口')
    item('IX','完整交付归档/SHA256/有效链接/原始不改与缺失清单','final_delivery_pending',['full_tokens/input_manifest.json','full_tokens/supplemental_input_manifest.json','completion_audit/requirement_matrix.csv'],'各模块committed SHA检查；用户框修改被单独快照标记','尚未发现包含最新resume所有交付的归档；归档必须排除claim完整实现未完成自然Probe和semantic区域')
    item('ADD35','历史35配置全统计/E-C含义/裁切监督/crisp/互补/场景','verified_existing',['scene35/validation.json','scene35/REPORT.md','scene35/configuration_semantics.json','scene35/preprocessing_inputs.csv','scene35/prediction_matrix.npz','scene35/all_config_pair_intersections.csv'],'35×1117预测；8历史表逐字段match；35配置语义、D01/D03输入SHA/裁切/OCR代理与交集','118 formal reviewed subset有抽样偏；scene corrections追加快照；场景关联不能升级因果')
    filemanifest=[]
    for p in sorted((ROOT/'scripts/colnomic_mechanism').glob('*.py')):
        filemanifest.append(dict(path=str(p.relative_to(ROOT)),sha256=digest(p),bytes=p.stat().st_size))
    for p in sorted((ROOT/'tests').glob('test_colnomic*.py')):
        filemanifest.append(dict(path=str(p.relative_to(ROOT)),sha256=digest(p),bytes=p.stat().st_size))
    write(output/'artifact_inventory.json',dict(status='inventory_only_full_goal_not_accepted',generated_utc=datetime.now(timezone.utc).isoformat(),checks=checks,totals=totals,existing_verifier_evidence=evidence,
                                               limitations=['Visual record existence is not manual-audit completion.', 'Existing verifier reports are reused evidence, not new replays by this inventory.', 'Open requirements remain open; a numerically passed subset is not overall full-goal acceptance.']))
    for r in requirements:
        r['actual_path_presence']={relative:(ROOT/relative if relative.startswith('scripts/') else BASE/relative).exists() for relative in r['authoritative_paths']}
    write(output/'requirement_matrix.json',requirements);table(output/'requirement_matrix.csv',requirements);write(output/'inspected_code_test_SHA256.json',filemanifest)
    lines=['# 全目标逐项验收清单','','本文件是实际产物与范围审计，不是全目标验收通过。signed-layout新增独立验证见 `signed_layout_independent_validation.json`。','',f'数据档案{totals["query_dossiers"]}/1117；Token对{totals["contrast_records"]}；矩阵{totals["matrix_index_records"]}；对称布局身份{totals["source_identities_symmetric"]}；detail身份{totals["source_identities_detail"]}。视觉记录只统计存在和可解析，实际审核覆盖以统一视觉验收为准。','','|节|要求|状态|权威证据|范围/限制|','|---|---|---|---|---|']
    for r in requirements:
        paths='、'.join('`'+p+'`' for p in r['authoritative_paths'])
        lines.append('|'+ '|'.join([r['section'],r['requirement'],r['status'],paths,r['proof']+'；'+r['limitations']]).replace('\n',' ')+'|')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Requirement inventory written:', len(requirements), 'entries. Full goal not accepted.', flush=True)
    return checks


def rows_gzip(path):
    import gzip
    with gzip.open(path,'rt',newline='') as f:
        return list(csv.DictReader(f))


def self_tests():
    model=dict(mean=np.array([1.,2.]),scale=np.array([2.,4.]),coefficient=np.array([3.,-2.,.5]))
    assert np.array_equal(predict(model,np.array([[3.,6.],[1.,2.]])),np.array([1.5,.5]))
    y=np.array([0,1,0,1]); scores=np.array([1.,2.,3.,2.]); q=np.array(['a','a','b','b']); p=np.array(['x','x','y','y'])
    assert pairs(y,scores,q,p)['pair_accuracy']==.5
    assert classification(y,scores,p,2.)['auc']==.5
    assert threshold_from_validation(y,scores,p)==1.5
    # A perturbed RGB multiset must fail even if its count remains unchanged.
    color=np.array([[1,2,3],[1,2,3],[4,5,6]],np.uint8)
    perm=color[[2,0,1]]; changed=perm.copy();changed[0,0]+=1
    def multiset(x):return np.unique(x,axis=0,return_counts=True)
    assert all(np.array_equal(a,b) for a,b in zip(multiset(color),multiset(perm)))
    assert not all(np.array_equal(a,b) for a,b in zip(multiset(color),multiset(changed)))
    return dict(status='passed',checks=['independent_logit_formula','within_source_pair_metric','AUC_tie_credit','validation_threshold_rule','pixel_multiset_negative_control'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--inventory-only',action='store_true');parser.add_argument('--recover-position',action='store_true')
    args=parser.parse_args();output=args.output.resolve()
    assert output==OUT or OUT in output.parents, 'Only dedicated completion_audit output is writable'
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=2)
    except ImportError:pass
    source=Path(__file__);source_sha=digest(source);snapshot=output/('source_executed_'+source_sha[:12]+'.py')
    if not snapshot.exists():snapshot.write_bytes(source.read_bytes())
    write(output/'runtime.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                                    torch_threads=torch.get_num_threads(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                    source_sha256=source_sha,executed_source_snapshot=str(snapshot.relative_to(ROOT))))
    (output/'REPRODUCE.md').write_text('''# 独立补充审计复现

从项目根目录运行。以下命令只消费冻结缓存、历史输入索引和已存在编辑图，不加载编码器、不启动 GPU、不修改原结果。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /home/jwei/miniconda3/envs/rebut/bin/python -B -m scripts.colnomic_mechanism.completion_audit --recover-position
```

它重放全部 signed-layout 像素/输入/权重/选参/null/编辑控制，恢复旧位置 Probe 的 9 组缺失 null 权重与全样本预测，并生成逐要求库存。

统一视觉/报告完成后可仅刷新库存：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /home/jwei/miniconda3/envs/rebut/bin/python -B -m scripts.colnomic_mechanism.completion_audit --inventory-only
```

`signed_layout_independent_validation.json` 和 `position_null_recovery/independent_validation.json` 的 passed 只适用其明确数值范围。`artifact_inventory.json` 与 `requirement_matrix` 不宣称全 goal 验收完成。自然同文字身份布局、真实阅读行和旧图库来源缺失的限制继续保留。
''')
    write(output/'self_test_results.json',self_tests())
    numerical=verify_layout(output) if not args.inventory_only else None
    recovery=recover_position_controls(output) if args.recover_position else None
    checks=inventory(output)
    assert all(checks.values()), [k for k,v in checks.items() if not v]
    if numerical is not None:assert numerical['status']=='passed', 'Signed-layout independent replay failed'
    if recovery is not None:assert recovery['status']=='passed', 'Historical position null recovery failed'


if __name__=='__main__':main()
