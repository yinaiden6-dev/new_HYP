#!/usr/bin/env python3
"""Independent arithmetic checks for GroZi's frozen retrieval results."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.data import digest, file_hash, read_json, write_json
from evaluation.grozi import DATA, IDS, MODELS, inventory, load_ocr


def plain_and_crisp(query, reference):
    q = query.double()
    r = reference.double()
    q = q[q.norm(dim=-1) > 1e-8]
    r = r[r.norm(dim=-1) > 1e-8]
    if not len(r):
        return -2., -2.
    if not len(q):
        return 0., 0.
    similarities = q @ r.T
    mean_maxsim = similarities.max(dim=1).values.mean()
    deviations = (r-r.mean(dim=0)).norm(dim=-1)
    weights = deviations/deviations.mean() if deviations.sum() > 1e-8 else torch.ones(len(r), dtype=torch.float64)
    adjusted = similarities * weights[None,:]
    crisp = adjusted.max(dim=1).values.mean() - adjusted.mean() + similarities.max(dim=0).values.mean()
    return float(mean_maxsim), float(crisp)


def tokens(text):
    return re.findall(r'[^\W_]+', text.lower(), flags=re.UNICODE)


def bm25_row(query, references):
    documents = [Counter(tokens(t)) for t in references]
    length = [sum(d.values()) for d in documents]
    avg = max(sum(length)/len(length), 1e-12)
    query = Counter(tokens(query))
    frequencies = {word:sum(word in d for d in documents) for word in query}
    scores = []
    for doc, size in zip(documents,length):
        score = 0.
        for word, qf in query.items():
            tf = doc.get(word,0)
            if tf:
                df = frequencies[word]
                idf = math.log1p((len(documents)-df+.5)/(df+.5))
                score += qf*idf*tf*2.2/(tf+1.2*(.25+.75*size/avg))
        scores.append(score)
    return np.array(scores)


def run(data, partial=False):
    torch.set_num_threads(4)
    metadata = inventory(data)
    config = read_json(data/'baseline_config.json')
    assert config['dataset_sha256'] == file_hash(data/'retrieval_subset/metadata.json')
    for path, expected in config['source_files'].items():
        assert file_hash(ROOT/path) == expected, f'Source changed: {path}'
        assert file_hash(data/'cache/intermediate/source_snapshot'/path) == expected
    query_ocr = load_ocr(data,'photo',metadata)[1]
    reference_ocr = load_ocr(data,'ref',metadata)[1]
    banks = {}
    for model in MODELS:
        for view,folder in [('image','image_embeddings'),('text','ocr_embeddings')]:
            if view == 'text' and model not in MODELS[:4]:
                continue
            for side,population,key in [('photo','photos','photo_id'),('ref','references','reference_id')]:
                path = data/'cache'/folder/model/f'{side}.pt'
                if not path.exists() and partial:
                    continue
                bank = torch.load(path,map_location='cpu',weights_only=True)
                rows = metadata[population]
                assert bank['ids'] == [r[key] for r in rows]
                assert bank['image_sha256'] == [r['sha256'] for r in rows]
                assert bank['class_ids'] == [r['class_id'] for r in rows]
                assert len(bank['embeddings']) == len(rows)
                recipe = dict(model=config['models'][model],view=view,
                    source={p:config['source_files'][p] for p in ['evaluation/encoders.py','evaluation/dense.py','evaluation/milk10k.py']},
                    inputs=[(r[key],r['sha256']) for r in rows],
                    text_sha256=digest(query_ocr if side=='photo' else reference_ocr) if view=='text' else None)
                assert bank['signature'] == digest(recipe) == digest(bank['recipe'])
                expected_has_text = [bool(t.strip()) for t in (query_ocr if side=='photo' else reference_ocr)] if view=='text' else [True]*len(rows)
                assert bank['has_text'] == expected_has_text
                for i,embedding in enumerate(bank['embeddings']):
                    assert torch.isfinite(embedding).all()
                    if not expected_has_text[i]:
                        assert not embedding.count_nonzero()
                    else:
                        norms=embedding.double().norm(dim=-1).reshape(-1)
                        norms=norms[norms>0]
                        assert len(norms) and (norms-1).abs().max()<.03
                banks[model,view,side]=bank
    results=[]
    for (model,mode),identifier in sorted(IDS.items(),key=lambda item:item[1]):
        directory=data/'results'/model/mode
        if not (directory/'metrics.json').exists() and partial:
            continue
        report=read_json(directory/'metrics.json')
        recorded=read_json(directory/'config.json')
        predictions=[json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
        value=torch.load(data/'cache/scores'/model/f'{mode}.pt',map_location='cpu',weights_only=True)
        scores=value['scores'].double().numpy()
        assert scores.shape==(468,1063) and np.isfinite(scores).all()
        assert value['query_ids']==[r['photo_id'] for r in metadata['photos']]
        assert value['reference_ids']==[r['reference_id'] for r in metadata['references']]
        assert value['signature']==recorded['score_signature']==digest(value['recipe'])
        assert recorded['dataset_sha256']==config['dataset_sha256']
        query_valid=[True]*468; reference_valid=[True]*1063
        if mode=='bm25':
            query_valid=[bool(tokens(t)) for t in query_ocr]
            reference_valid=[bool(tokens(t)) for t in reference_ocr]
        elif mode.startswith('ocr_to_'):
            query_valid=[bool(t.strip()) for t in query_ocr]
            if mode=='ocr_to_ocr': reference_valid=[bool(t.strip()) for t in reference_ocr]
        assert recorded['query_valid']==query_valid and recorded['reference_valid']==reference_valid
        ranks=[]
        for index,(photo,prediction) in enumerate(zip(metadata['photos'],predictions,strict=True)):
            correct=photo['positive_reference_index']
            valid=query_valid[index] and reference_valid[correct]
            # Count outranking scores directly, independently of the evaluator's argsort.
            rank=int(1+(scores[index]>scores[index,correct]).sum()+(scores[index,:correct]==scores[index,correct]).sum()) if valid else None
            assert prediction['photo_id']==photo['photo_id']
            assert prediction['positive_reference_id']==photo['positive_reference_id']
            assert prediction['rank']==rank
            failure=None if valid else ('empty_query_ocr' if not query_valid[index] else 'empty_positive_reference_ocr')
            assert prediction['failure']==failure
            top=np.lexsort((np.arange(1063),-scores[index]))[:10]
            expected_ids=[metadata['references'][j]['reference_id'] for j in top] if query_valid[index] else []
            assert prediction['top10_ids']==expected_ids
            ranks.append(rank or np.inf)
        ranks=np.array(ranks)
        cells=np.column_stack([ranks<=1,ranks<=5,ranks<=10,1/ranks])
        for i,key in enumerate(['R@1','R@5','R@10','MRR']):
            assert abs(cells[:,i].mean()-report[key])<1e-12
            classes=np.array([r['class_id'] for r in metadata['photos']])
            macro=np.mean([cells[classes==c,i].mean() for c in np.unique(classes)])
            assert abs(macro-report['class_average'][key])<1e-12
        assert report['queries']==468 and report['references']==1063
        assert report['failures']==int((~np.isfinite(ranks)).sum())
        error=0.; checked_pairs=0; check_kind=''; precision={}
        if 'fusion' in mode:
            image=torch.load(data/'cache/scores'/model/'image.pt',map_location='cpu',weights_only=True)['scores'].double().numpy()
            text_mode='bm25' if mode=='bm25_image_fusion' else ('ocr_to_ocr' if mode=='ocr_fusion' else 'ocr_to_image')
            text_model='bm25' if text_mode=='bm25' else model
            text=torch.load(data/'cache/scores'/text_model/f'{text_mode}.pt',map_location='cpu',weights_only=True)['scores'].double().numpy()
            def z(a):
                std=a.std(axis=1,keepdims=True)
                return np.divide(a-a.mean(axis=1,keepdims=True),std,out=np.zeros_like(a),where=std>0)
            expected=.5*z(image)+.5*z(text)
            has_text=[bool(tokens(t)) for t in query_ocr] if text_mode=='bm25' else [bool(t.strip()) for t in query_ocr]
            expected[~np.array(has_text)]=z(image)[~np.array(has_text)]
            error=float(np.max(np.abs(expected-scores)))
            # FP32 mean subtraction can add a query-wise constant, especially
            # when OCR-to-image scores have very small variance. Attribute that
            # rounding explicitly; keep FP64 variances and full independent ranks.
            def mean_rounding_offset(a):
                mean32=torch.from_numpy(a.astype(np.float32)).mean(dim=1).double().numpy()[:,None]
                std64=a.std(axis=1,keepdims=True)
                return np.divide(a.mean(axis=1,keepdims=True)-mean32,std64,
                    out=np.zeros_like(std64),where=std64>0)
            offset=.5*mean_rounding_offset(image)+.5*mean_rounding_offset(text)
            offset[~np.array(has_text)]=mean_rounding_offset(image)[~np.array(has_text)]
            adjusted_error=float(np.max(np.abs(expected+offset-scores)))
            difference=scores-expected
            centered_error=float(np.max(np.abs(difference-difference.mean(axis=1,keepdims=True))))
            order32=np.argsort(-scores,axis=1,kind='stable')
            order64=np.argsort(-expected,axis=1,kind='stable')
            positives=np.array([r['positive_reference_index'] for r in metadata['photos']])[:,None]
            rank32=np.argmax(order32==positives,axis=1)
            rank64=np.argmax(order64==positives,axis=1)
            assert np.array_equal(order32[:,0],order64[:,0]),'FP32/FP64 fusion top-1 changed'
            assert np.array_equal(rank32,rank64),'FP32/FP64 positive ranks changed'
            assert centered_error<5e-6 and adjusted_error<5e-5,(identifier,centered_error,adjusted_error)
            precision=dict(max_error_after_accounting_for_fp32_mean_rounding=adjusted_error,
                max_error_after_removing_query_constant=centered_error,
                fp32_fp64_top1_changes=0,fp32_fp64_positive_rank_changes=0)
            check_kind='all fusion scores independently recomputed in FP64'
        elif model=='bm25':
            for index in [0,1,57,117,234,350,467]:
                error=max(error,float(np.max(np.abs(bm25_row(query_ocr[index],reference_ocr)-scores[index]))))
            check_kind='7 complete BM25 query rows independently recomputed'
        else:
            view='image' if mode.startswith('image') else 'text'
            reference_view='text' if mode=='ocr_to_ocr' else 'image'
            q=banks[model,view,'photo']['embeddings']
            r=banks[model,reference_view,'ref']['embeddings']
            if isinstance(q,torch.Tensor):
                expected=(q.double()@r.double().T).numpy()
                expected[:,~np.array(banks[model,reference_view,'ref']['has_text'])]=-2.
                error=float(np.max(np.abs(expected-scores)))
                check_kind='full dense score matrix independently recomputed in FP64'
            else:
                indices={0,1,57,117,234,350,467}
                empty_query=[i for i,v in enumerate(q) if not v.shape[0]]
                if empty_query: indices.add(empty_query[0])
                empty_ref=[i for i,v in enumerate(r) if not v.shape[0]]
                for qi in sorted(indices):
                    refs={metadata['photos'][qi]['positive_reference_index'],(qi*173+11)%1063}
                    if empty_ref: refs.add(empty_ref[0])
                    for ri in sorted(refs):
                        pair=plain_and_crisp(q[qi],r[ri])
                        expected=pair[1] if mode.endswith('_crisp') else pair[0]
                        error=max(error,abs(expected-scores[qi,ri]));checked_pairs+=1
                check_kind='sampled MaxSim/CRISP pairs independently recomputed in FP64'
        if 'fusion' not in mode:
            assert error<5e-5,(identifier,model,mode,error)
        results.append(dict(id=identifier,model=model,mode=mode,ranked_queries_verified=468,
            arithmetic_check=check_kind,checked_pairs=checked_pairs,max_absolute_error=error,**precision))
        print(f'Verified {identifier}: {error:.3g} max error',flush=True)
    complete=len(results)==len(IDS) and len(banks)==20
    assert partial or complete
    output=dict(passed=True,all_planned_results_verified=complete,results_count=len(results),expected_results=len(IDS),
        embedding_banks_verified=len(banks),expected_embedding_banks=20,verified_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_sha256=config['dataset_sha256'],results=results,verifier_sha256=file_hash(Path(__file__)))
    write_json(data/'results'/('verification.json' if complete else 'verification.partial.json'),output)
    if complete: (data/'results/verification.partial.json').unlink(missing_ok=True)
    print(json.dumps({k:v for k,v in output.items() if k!='results'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    run(args.data,args.partial)
