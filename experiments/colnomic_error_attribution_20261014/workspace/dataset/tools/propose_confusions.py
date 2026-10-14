"""Generate review-only design candidates from reference OCR and image vectors.

Does not read photo retrieval scores, correctness, or user confusion decisions.
Run with the base Python (numpy + sklearn); torch is loaded in --embedding-python.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from photo_metadata import ROOT, PhotoStore, sha, write_json
from confusion_review import ConfusionReview, CATEGORIES


def strength_terms(text):
    return sorted(set(re.findall(r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|oz|%|tablets?|capsules?|count|ct)\b', text.lower())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--embedding-python', default=sys.executable)
    args = p.parse_args()
    root = args.root.resolve(); store = PhotoStore(root); review = ConfusionReview(store)
    refs = list(review.refs); index = {rid: i for i, rid in enumerate(refs)}
    ocr_path = root.parent / 'cache/ocr/references.json'
    bank_path = root.parent / 'cache/image_embeddings/Qwen3-VL-Embedding-2B/references.pt'
    print('Reading verified reference OCR and image bank…', flush=True)
    raw = ocr_path.read_bytes(); ocr_hash = sha(raw); ocr = json.loads(raw); del raw
    text = []
    for rid in refs:
        row = ocr[rid]
        if row['image_sha256'] != review.refs[rid]['sha256']: raise ValueError('Stale reference OCR: ' + rid)
        text.append(row['text'])
    del ocr
    with tempfile.TemporaryDirectory(prefix='confusion-proposal-') as tmp:
        npy = str(Path(tmp) / 'vectors.npy')
        code = """import csv,sys,torch,numpy as np
bank=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
rows=list(csv.DictReader(open(sys.argv[2])))
assert bank['ids']==[r['reference_id'] for r in rows]
assert bank['sha256']==[r['sha256'] for r in rows]
x=bank['embeddings'].float().numpy();assert np.isfinite(x).all()
np.save(sys.argv[3],x)
"""
        subprocess.run([args.embedding_python, '-B', '-c', code, str(bank_path), str(root/'lists/references.csv'), npy], check=True)
        vectors = np.load(npy)[[int(review.refs[r]['row']) for r in refs]]
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    print('Computing text and reference-image neighbours…', flush=True)
    words = TfidfVectorizer(lowercase=True, strip_accents='unicode', stop_words='english',
                            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b', max_df=.65, sublinear_tf=True)
    word_vectors = words.fit_transform(text)
    chars = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_features=70000,
                            strip_accents='unicode', sublinear_tf=True)
    char_vectors = chars.fit_transform(text)
    anchors = [index[review.positive[pid]] for pid in store.packages]
    word_scores = (word_vectors[anchors] @ word_vectors.T).toarray()
    char_scores = (char_vectors[anchors] @ char_vectors.T).toarray()
    image_scores = vectors[anchors] @ vectors.T
    sources = json.loads((root/'references/metadata.json').read_text())
    source_by_id = {Path(r['image']).stem: r.get('dailymed', {}) for r in sources}
    feature_names = words.get_feature_names_out()
    catalog = {}
    for rid, content in zip(refs, text):
        owners = review.owners[rid]
        lines = [x.strip() for x in content.splitlines() if len(x.strip()) > 2]
        title = review.names[owners[0]] if owners else ' · '.join(lines[:5])[:180]
        catalog[rid] = dict(title=title or review.refs[rid].get('identity', rid),
                            ocr_preview=content[:1800], search_text=content[:14000])
    by_package = {}; category_counts = Counter()
    for n, pid in enumerate(store.packages):
        anchor = review.positive[pid]; ai = index[anchor]
        channels = [('reference_text', word_scores[n], .08), ('text_spelling', char_scores[n], .10),
                    ('reference_image', image_scores[n], .45)]
        chosen = {}
        for name, scores, minimum in channels:
            count = 0
            for i in np.argsort(-scores, kind='stable'):
                if i == ai or scores[i] < minimum: continue
                chosen.setdefault(refs[i], []).append(name)
                count += 1
                if count >= 30: break
        # Same source-set designs may be packaging variants, not necessarily new versions.
        source_url = source_by_id.get(anchor, {}).get('url')
        siblings = [rid for rid in refs if rid != anchor and source_url and source_by_id.get(rid, {}).get('url') == source_url]
        siblings.sort(key=lambda rid: -(word_scores[n,index[rid]] + image_scores[n,index[rid]]))
        for rid in siblings[:12]: chosen.setdefault(rid, []).append('same_source_set')
        name_tokens = {v for v in re.findall(r'[a-z][a-z-]{3,}', review.names[pid].lower())
                       if v not in ENGLISH_STOP_WORDS and v not in {'hydrochloride','sodium','cream','tablets','injection','relief'}}
        proposals = []
        for rid, methods in chosen.items():
            i = index[rid]; cats, evidence = [], []
            joint = word_vectors[ai].multiply(word_vectors[i]).tocsr()
            order = np.argsort(-joint.data, kind='stable')[:6]
            shared = [str(feature_names[joint.indices[j]]) for j in order]
            if name_tokens.intersection(set(re.findall(r'[a-z][a-z-]{3,}', text[i].lower()))):
                cats.append('name_series')
            if word_scores[n,i] >= .12 or char_scores[n,i] >= .25:
                cats.append('text_similar')
            strengths_a, strengths_b = strength_terms(text[ai]), strength_terms(text[i])
            if word_scores[n,i] >= .25 and strengths_a and strengths_b and strengths_a != strengths_b:
                cats.append('strength_count'); evidence.append('规格文字有差异（需看图核实）')
            if 'reference_image' in methods: cats.append('visual_layout')
            if 'same_source_set' in methods: cats.append('source_variant'); evidence.append('DailyMed 来源 Set ID 相同')
            if shared: evidence.append('共有词：' + ' · '.join(shared))
            if 'reference_image' in methods: evidence.append('参考图视觉近邻（可能同时包含文字线索）')
            score = .45*word_scores[n,i] + .20*char_scores[n,i] + .35*max(0,image_scores[n,i])
            proposals.append(dict(reference_id=rid, proposed_categories=list(dict.fromkeys(cats)) or ['other'],
                evidence=evidence, methods=methods, scores=dict(reference_text=round(float(word_scores[n,i]),4),
                text_spelling=round(float(char_scores[n,i]),4), reference_image=round(float(image_scores[n,i]),4)),
                priority=round(float(score),6)))
            category_counts.update(set(cats))
        proposals.sort(key=lambda p: (-p['priority'], p['reference_id']))
        by_package[pid] = proposals
    payload = dict(schema='release.confusion_candidates.v1', inventory=review.inventory,
        generated_at=datetime.now(timezone.utc).isoformat(), categories=CATEGORIES, references=catalog, packages=by_package,
        provenance=dict(ocr_sha256=ocr_hash, image_bank_sha256=sha(bank_path.read_bytes()),
            script_sha256=sha(Path(__file__).read_bytes()), photo_scores_used=False,
            method='union of word TF-IDF, character TF-IDF, Qwen3 reference image neighbours and source-set siblings; human review required',
            candidate_scope='entire reference gallery; no quota or preference for designs with photos',
            full_gallery_top_per_channel=30, same_source_max=12,
            category_counts=dict(category_counts)))
    write_json(review.candidate_path, payload)
    print(json.dumps(dict(path=str(review.candidate_path), packages=len(by_package), unique_references=len(refs),
        proposed_pairs=sum(map(len,by_package.values())), categories=dict(category_counts)), ensure_ascii=False), flush=True)
    for pid in ['package_011','package_071','package_105']:
        print(pid, [(v['reference_id'], review.owners[v['reference_id']],v['proposed_categories']) for v in by_package.get(pid,[])[:5]], flush=True)


if __name__ == '__main__': main()
