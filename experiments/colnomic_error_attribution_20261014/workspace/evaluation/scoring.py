"""Exact full-gallery scoring; padding never participates in token maxima."""
from collections import Counter
import math
import re
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

SCORING_VERSION = 'full_gallery_fp32_masked_v1'
CRISP_SOURCE = 'https://arxiv.org/html/2608.26414v1#S3.SS3'


def valid_tokens(embedding):
    if embedding.ndim != 2:
        raise ValueError('Expected token matrix')
    return embedding[embedding.float().norm(dim=-1) > 1e-8].float()


@torch.inference_mode()
def multivector_scores(queries, references, device='cuda:0', query_batch=2, reference_batch=16,
                       token_block=2048, progress=None):
    """Compute MaxSim and paper equations (1)-(3) from the same similarities.

    FP32 products, reductions and accumulators. Both token axes are blocked.
    MaxSim is the mean over query tokens (sum gives identical query rankings).
    """
    queries = [valid_tokens(x) for x in queries]
    references = [valid_tokens(x) for x in references]
    if any(not len(x) for x in references):
        raise ValueError('An image reference contains no valid tokens')
    weights = []
    for r in references:
        deviation = (r-r.mean(0)).norm(dim=-1)
        # Identical tokens have no distinctiveness; use uniform weights.
        weights.append(deviation * (len(r)/deviation.sum()) if deviation.sum() > 1e-8
                       else torch.ones(len(r)))
    plain = torch.zeros(len(queries), len(references), dtype=torch.float32)
    crisp = torch.zeros_like(plain)
    started = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = False
    for qi in range(0, len(queries), query_batch):
        qlist = queries[qi:qi+query_batch]
        qlength = torch.tensor([len(q) for q in qlist], device=device)
        if not qlength.any():
            continue
        q = pad_sequence(qlist, batch_first=True).to(device)
        qmask = torch.arange(q.shape[1], device=device)[None, :] < qlength[:, None]
        for ri in range(0, len(references), reference_batch):
            rlist = references[ri:ri+reference_batch]
            rlength = torch.tensor([len(r) for r in rlist], device=device)
            r = pad_sequence(rlist, batch_first=True).to(device)
            w = pad_sequence(weights[ri:ri+reference_batch], batch_first=True).to(device)
            rmask = torch.arange(r.shape[1], device=device)[None, :] < rlength[:, None]
            shape = (len(q), len(r), q.shape[1])
            forward = torch.full(shape, -torch.inf, device=device)
            adjusted = torch.full_like(forward, -torch.inf)
            reverse = torch.full((len(q), len(r), r.shape[1]), -torch.inf, device=device)
            weighted_sum = torch.zeros((len(q), len(r)), device=device)
            for qt in range(0, q.shape[1], token_block):
                qs = q[:, qt:qt+token_block]
                qm = qmask[:, qt:qt+token_block]
                for rt in range(0, r.shape[1], token_block):
                    rs = r[:, rt:rt+token_block]
                    rm = rmask[:, rt:rt+token_block]
                    ws = w[:, rt:rt+token_block]
                    sim = (qs.flatten(0,1) @ rs.flatten(0,1).T).reshape(
                        len(q), qs.shape[1], len(r), rs.shape[1]).permute(0,2,1,3)
                    weighted = sim * ws[None,:,None,:]
                    weighted_sum += (weighted * qm[:,None,:,None]).sum((2,3))
                    forward[:,:,qt:qt+len(qs[0])] = torch.maximum(
                        forward[:,:,qt:qt+len(qs[0])],
                        sim.masked_fill(~rm[None,:,None,:], -torch.inf).amax(-1))
                    adjusted[:,:,qt:qt+len(qs[0])] = torch.maximum(
                        adjusted[:,:,qt:qt+len(qs[0])],
                        weighted.masked_fill(~rm[None,:,None,:], -torch.inf).amax(-1))
                    reverse[:,:,rt:rt+len(rs[0])] = torch.maximum(
                        reverse[:,:,rt:rt+len(rs[0])],
                        sim.masked_fill(~qm[:,None,:,None], -torch.inf).amax(-2))
            qn = qlength.clamp_min(1)[:,None]
            rn = rlength[None,:]
            f = forward.masked_fill(~qmask[:,None,:], 0).sum(-1) / qn
            a = adjusted.masked_fill(~qmask[:,None,:], 0).sum(-1) / qn - weighted_sum/(qn*rn)
            b = reverse.masked_fill(~rmask[None,:,:], 0).sum(-1)/rn
            result = a+b
            f[qlength == 0] = 0
            result[qlength == 0] = 0
            plain[qi:qi+len(q),ri:ri+len(r)] = f.cpu()
            crisp[qi:qi+len(q),ri:ri+len(r)] = result.cpu()
        if progress and (qi+len(q) == len(queries) or (qi//query_batch) % 10 == 0):
            progress(min(qi+query_batch,len(queries)), len(queries), time.monotonic()-started)
    return plain, crisp


def dot_scores(queries, references):
    return torch.as_tensor(queries).float() @ torch.as_tensor(references).float().T


def tokenize(text):
    return re.findall(r'[^\W_]+', text.lower(), flags=re.UNICODE)


def bm25_scores(query_texts, reference_texts, k1=1.2, b=0.75):
    documents = [Counter(tokenize(t)) for t in reference_texts]
    lengths = np.array([sum(d.values()) for d in documents], dtype=np.float64)
    avg = max(float(lengths.mean()), 1e-12)
    postings = {}
    for i, doc in enumerate(documents):
        for token, tf in doc.items():
            postings.setdefault(token, []).append((i, tf))
    scores = np.zeros((len(query_texts),len(documents)), dtype=np.float32)
    for qi, text in enumerate(query_texts):
        # Standard BM25 sums each query token occurrence; no learned query weights.
        for token, qf in Counter(tokenize(text)).items():
            hits = postings.get(token, [])
            if not hits:
                continue
            indices, tf = map(np.array, zip(*hits))
            idf = math.log(1 + (len(documents)-len(hits)+0.5)/(len(hits)+0.5))
            scores[qi,indices] += qf*idf*tf*(k1+1)/(tf+k1*(1-b+b*lengths[indices]/avg))
    return torch.from_numpy(scores)


def zscore(scores):
    std = scores.std(-1, unbiased=False, keepdim=True)
    return torch.where(std > 0, (scores-scores.mean(-1,keepdim=True))/std.clamp_min(1e-12), 0)


def fuse(image, text, has_text, alpha=0.5):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must lie in [0,1]')
    result = (1-alpha)*zscore(image) + alpha*zscore(text)
    # Explicit image fallback for unavailable OCR; never score an empty prompt.
    result[~torch.as_tensor(has_text,dtype=torch.bool)] = zscore(image)[~torch.as_tensor(has_text,dtype=torch.bool)]
    return result
