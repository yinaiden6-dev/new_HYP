"""Full-gallery, versioned sensitivity to the actual five user box revisions.

All original caches remain read-only. A fresh-before encoding audit separates
runtime drift from the effect of changing the crop. Reference vectors remain
frozen historical vectors; this is not a fresh-encoder whole-gallery experiment.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from evaluation.scoring import valid_tokens
from scripts.analyze_colnomic_tokens import token_hash
from scripts.colnomic_mechanism.common import ROOT, data, read, write, csvout, sha
from scripts.colnomic_mechanism.geometry import rank_rows, metrics


@torch.inference_mode()
def forward_scores(query, references, device, block=512, batch=4):
    """Exact blocked FP32 mean MaxSim, including all valid template tokens."""
    q = valid_tokens(query).to(device)
    assert len(q)
    out = np.empty(len(references), np.float32)
    for start in range(0, len(references), batch):
        rs = [valid_tokens(r) for r in references[start:start+batch]]
        assert all(len(r) for r in rs)
        lens = torch.tensor([len(r) for r in rs], device=device)
        padded = pad_sequence(rs, batch_first=True).to(device)
        total = torch.zeros(len(rs), device=device, dtype=torch.float32)
        for offset in range(0, len(q), block):
            qq = q[offset:offset+block]
            best = torch.full((len(rs), len(qq)), -torch.inf, device=device)
            for ro in range(0, padded.shape[1], block):
                rr = padded[:, ro:ro+block]
                sim = (qq @ rr.flatten(0, 1).T).reshape(len(qq), len(rs), rr.shape[1]).permute(1, 0, 2)
                mask = torch.arange(ro, ro+rr.shape[1], device=device)[None] < lens[:, None]
                best = torch.maximum(best, sim.masked_fill(~mask[:, None], -torch.inf).amax(-1))
            total += best.sum(-1)
        out[start:start+len(rs)] = (total/len(q)).cpu().numpy()
    assert np.isfinite(out).all()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    source, folder = args.input.resolve(), args.output.resolve(); folder.mkdir(parents=True, exist_ok=True)
    d = data(); manifest = read(source/'encoding_manifest.json'); controls = read(source/'crop_controls.json')
    complete = read(source/'embeddings/complete.json')
    assert complete['records'] == len(manifest) == 2*len(controls)
    assert complete['manifest_sha256'] == sha(source/'encoding_manifest.json')
    oldbankpath = ROOT/'cache/foreground/human_boxes/0a7a1ed078cd491c/photo_embeddings.pt'
    oldscorepath = oldbankpath.with_name('scores.pt')
    oldbank = torch.load(oldbankpath, map_location='cpu', mmap=True, weights_only=True)
    oldscore = torch.load(oldscorepath, map_location='cpu', mmap=True, weights_only=True)['scores'].numpy()
    assert oldbank['ids'] == d['qb']['ids'] and oldscore.shape == d['image']['scores'].shape
    qi = {qid:i for i,qid in enumerate(d['qb']['ids'])}; records = {}; parity = []; score_replay = []
    fingerprints = {str(p.relative_to(ROOT)):sha(p) for p in [oldbankpath, oldscorepath,
        ROOT/'cache/image_embeddings/colnomic7b/references.pt',ROOT/'cache/image_embeddings/colnomic7b/photos.pt',
        ROOT/'cache/scores/colnomic7b/image.pt',ROOT/'cache/scores/bm25_image_fusion/colnomic7b/alpha_0.5.pt']}
    score_rows = []
    for row in manifest:
        p = source/'embeddings'/(row['id']+'.pt'); record = torch.load(p, map_location='cpu', weights_only=True)
        assert all(record.get(k) == v for k,v in row.items())
        assert record['image_sha256'] == sha(row['path'])
        assert torch.isfinite(record['embedding']).all() and token_hash(record['embedding']) == record['tensor_sha256']
        records[row['id']] = record
        if row['arm'] == 'crop_before':
            cached = oldbank['embeddings'][qi[row['source_id']]]; fresh = record['embedding']
            same_shape = cached.shape == fresh.shape
            parity.append(dict(photo_id=row['source_id'], package_id=row['package_id'], split=row['split'],
                shape_equal=same_shape, bit_exact=same_shape and torch.equal(cached, fresh),
                max_error=float((cached.float()-fresh.float()).abs().max()) if same_shape else None,
                mean_cosine=float(torch.nn.functional.cosine_similarity(cached.float(),fresh.float(),dim=1).mean()) if same_shape else None))
        target = folder/(row['id']+'_scores.npy')
        if target.exists():
            receipt = read(target.with_suffix('.json')); assert receipt['embedding_SHA256'] == sha(p)
            assert receipt['reference_bank_SHA256'] == fingerprints['cache/image_embeddings/colnomic7b/references.pt']
            assert receipt['scores_SHA256'] == sha(target); scores = np.load(target)
        else:
            scores = forward_scores(record['embedding'], d['rb']['embeddings'], args.device)
            np.save(target, scores)
            write(target.with_suffix('.json'),dict(embedding_SHA256=sha(p), scores_SHA256=sha(target),
                reference_bank_SHA256=fingerprints['cache/image_embeddings/colnomic7b/references.pt'], device=args.device))
        assert scores.shape == (len(d['rb']['ids']),) and np.isfinite(scores).all()
        score_rows.append(scores)
        if row['arm'] == 'crop_before':
            error = float(np.max(np.abs(scores-oldscore[qi[row['source_id']]])))
            score_replay.append(dict(photo_id=row['source_id'],maximum_full_gallery_score_error=error))
        print('Scored',row['id'],'all',len(scores),'references',flush=True)
    matrices = {'D01_historical':oldscore.copy(), 'D01_fresh_before_fixed_gallery':oldscore.copy(),
        'D01_reviewed_after_fixed_gallery':oldscore.copy()}
    for row,scores in zip(manifest,score_rows):
        key = 'D01_fresh_before_fixed_gallery' if row['arm']=='crop_before' else 'D01_reviewed_after_fixed_gallery'
        matrices[key][qi[row['source_id']]] = scores
    np.savez_compressed(folder/'full_gallery_scores.npz',query_ids=np.array(d['qb']['ids']),
        reference_ids=np.array(d['rb']['ids']),**matrices)
    predictions = {name:rank_rows(scores,d) for name,scores in matrices.items()}
    tables = []; perphoto = []; oldpred = predictions['D01_historical']
    for name,pred in predictions.items():
        for split in ['all','train','val','test']:
            idx = [i for i,qid in enumerate(d['qb']['ids']) if split=='all' or d['splits'][qid]==split]
            subset=[pred[i] for i in idx]
            tables.append(dict(method=name,split=split,**metrics(subset),
                repairs=sum(not oldpred[i]['correct'] and pred[i]['correct'] for i in idx),
                harms=sum(oldpred[i]['correct'] and not pred[i]['correct'] for i in idx),
                unchanged_queries=len(idx)-sum(d['qb']['ids'][i] in {r['photo_id'] for r in controls} for i in idx)))
        perphoto.extend(dict(method=name,**r) for r in pred)
    csvout(folder/'metrics.csv',tables); csvout(folder/'per_photo_ranks.csv',perphoto)
    csvout(folder/'fresh_before_embedding_parity.csv',parity); csvout(folder/'fresh_before_score_replay.csv',score_replay)
    changed = {r['photo_id'] for r in controls}; details=[]
    base_image=rank_rows(d['image']['scores'].numpy(),d); base_fusion=rank_rows(d['fusion']['scores'].numpy(),d)
    for qid in sorted(changed):
        i=qi[qid]; row=next(r for r in controls if r['photo_id']==qid)
        details.append(dict(**row,B03_rank=base_image[i]['rank'],F03_rank=base_fusion[i]['rank'],
            before_D01_rank=oldpred[i]['rank'],fresh_before_D01_rank=predictions['D01_fresh_before_fixed_gallery'][i]['rank'],
            after_D01_rank=predictions['D01_reviewed_after_fixed_gallery'][i]['rank'],
            before_D01_top1=oldpred[i]['top1'],after_D01_top1=predictions['D01_reviewed_after_fixed_gallery'][i]['top1']))
    write(folder/'changed_query_results.json',details)
    for path,digest in fingerprints.items(): assert sha(ROOT/path)==digest
    assert all(np.array_equal(matrices[name][[i for i,q in enumerate(d['qb']['ids']) if q not in changed]],
        oldscore[[i for i,q in enumerate(d['qb']['ids']) if q not in changed]]) for name in matrices)
    if all(r['bit_exact'] for r in parity):
        assert max(r['maximum_full_gallery_score_error'] for r in score_replay) < 2e-6
        assert all(predictions['D01_fresh_before_fixed_gallery'][qi[q]]['rank'] ==
            oldpred[qi[q]]['rank'] for q in changed)
    write(folder/'validation.json',dict(status='passed',photos=len(d['qb']['ids']),changed_queries=len(changed),
        changed_package_identities=len({r['package_id'] for r in controls}),encodings=len(manifest),
        fresh_before_exact=sum(r['bit_exact'] for r in parity),score_replay=score_replay,
        unchanged_rows_verified=len(d['qb']['ids'])-len(changed),source_SHA256=fingerprints,
        scope='same frozen historical gallery; five changed queries; other1112 historicalD01 rows unchanged',
        extra_supervision='manual target localization revised after viewing historical results',
        interpretation='target selection/background/scale/content change together; no isolated background causal claim'))


if __name__ == '__main__': main()
