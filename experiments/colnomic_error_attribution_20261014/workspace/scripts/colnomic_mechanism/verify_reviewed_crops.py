"""Independent frozen-gallery crop input, scalar-axis and rank verification."""
from collections import defaultdict
import numpy as np
import torch
from scripts.colnomic_mechanism.common import ROOT,read,write,rows,sha,data,workspace_path
from scripts.analyze_colnomic_tokens import token_hash

BASE=ROOT/'results/colnomic_full_audit/20261013_resume'


def ranking(scores,hashes,positive):
    order=np.argsort(-scores,kind='stable');unique=[];seen=set()
    for i in order:
        h=hashes[int(i)]
        if h not in seen:seen.add(h);unique.append(int(i))
    hit=next(k for k,i in enumerate(unique,1) if hashes[i] in positive)
    return hit,unique


def main():
    torch.set_num_threads(2);d=data();source=BASE/'reviewed_crops';folder=source/'analysis'
    manifest=read(source/'encoding_manifest.json');control=read(source/'crop_controls.json')
    arrays=np.load(folder/'full_gallery_scores.npz');assert list(arrays['query_ids'])==d['qb']['ids'] and list(arrays['reference_ids'])==d['rb']['ids']
    historical=ROOT/'cache/foreground/human_boxes/0a7a1ed078cd491c/photo_embeddings.pt'
    oldbank=torch.load(historical,weights_only=True,map_location='cpu',mmap=True)
    oldscores=torch.load(historical.with_name('scores.pt'),weights_only=True,map_location='cpu',mmap=True)['scores'].numpy()
    assert np.array_equal(arrays['D01_historical'],oldscores)
    qi={q:i for i,q in enumerate(d['qb']['ids'])};ri={r:i for i,r in enumerate(d['rb']['ids'])}
    ledger={r['photo_id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    positive={r['photo_id']:set(r['positive_reference_sha256']) for r in ledger.values()};hashes=d['rb']['sha256']
    saved=defaultdict(dict)
    for row in rows(folder/'per_photo_ranks.csv'):saved[row['method']][row['photo_id']]=row
    rankchecks=0;mathchecks=0;maxerr=0;parity=[]
    for method in ['D01_historical','D01_fresh_before_fixed_gallery','D01_reviewed_after_fixed_gallery']:
        matrix=arrays[method];assert matrix.shape==(1117,5405) and np.isfinite(matrix).all()
        for i,qid in enumerate(d['qb']['ids']):
            hit,order=ranking(matrix[i],hashes,positive[qid]);record=saved[method][qid]
            assert int(record['rank'])==hit and record['top1']==d['rb']['ids'][order[0]]
            assert (record['correct']=='True')==(hit==1);rankchecks+=1
        unchanged=[i for i,q in enumerate(d['qb']['ids']) if q not in {r['photo_id'] for r in control}]
        assert np.array_equal(matrix[unchanged],oldscores[unchanged])
    for row in manifest:
        path=source/'embeddings'/(row['id']+'.pt');record=torch.load(path,weights_only=True,map_location='cpu')
        assert all(record.get(k)==v for k,v in row.items()) and record['image_sha256']==sha(workspace_path(row['path']))
        assert record['tensor_sha256']==token_hash(record['embedding']) and torch.isfinite(record['embedding']).all()
        raw=record['embedding'].float();query=raw[raw.norm(dim=1)>1e-8]
        i=qi[row['source_id']];method='D01_fresh_before_fixed_gallery' if row['arm']=='crop_before' else 'D01_reviewed_after_fixed_gallery'
        vector=np.load(folder/(row['id']+'_scores.npy'));assert np.array_equal(vector,arrays[method][i])
        receipt=read(folder/(row['id']+'_scores.json'));assert receipt['scores_SHA256']==sha(folder/(row['id']+'_scores.npy')) and receipt['embedding_SHA256']==sha(path)
        hit,order=ranking(vector,hashes,positive[row['source_id']])
        selected=set(order[:10])|{ri[rid] for rid in ledger[row['source_id']]['positive_reference_ids']}
        selected|={ri[ledger[row['source_id']][role]] for role in ['image_strongest_wrong','fusion_strongest_wrong']}
        rng=np.random.default_rng(20261014+i);selected.update(map(int,rng.choice(5405,8,replace=False)))
        for j in sorted(selected):
            reference=d['rb']['embeddings'][j].float();reference=reference[reference.norm(dim=1)>1e-8]
            expected=float((query@reference.T).amax(1).mean());error=abs(expected-float(vector[j]));maxerr=max(error,maxerr);assert error<2e-6;mathchecks+=1
        if row['arm']=='crop_before':
            assert torch.equal(record['embedding'],oldbank['embeddings'][i])
            assert np.max(np.abs(vector-oldscores[i]))<2e-6
            assert ranking(vector,hashes,positive[row['source_id']])[0]==ranking(oldscores[i],hashes,positive[row['source_id']])[0]
            parity.append(row['source_id'])
    assert len(parity)==len(control)==5 and rankchecks==3351
    checks=read(folder/'validation.json')['source_SHA256']
    for path,digest in checks.items():assert sha(ROOT/path)==digest
    write(folder/'independent_validation.json',dict(status='passed',queries=1117,refs=5405,
        rankings_replayed=rankchecks,changed_crops=len(control),fresh_before_bit_exact=len(parity),unchanged_rows=1112,
        selected_candidate_pointwise_score_checks=mathchecks,selected_math_max_error=maxerr,
        scalar_score_scope='all finite 1117x5405 axes and every rank; direct FP32 dot replay on all positives, top10, historical hard negatives and 8 seeded random refs for each fresh crop arm',
        source_SHA256=checks,verifier_SHA256=sha(__file__),
        limitation='manual localization is extra supervision; only query crops reencoded and frozen historical gallery reused; background/content/scale change together'))
    print('Independent reviewed crop verification passed',rankchecks,mathchecks,flush=True)


if __name__=='__main__':main()
