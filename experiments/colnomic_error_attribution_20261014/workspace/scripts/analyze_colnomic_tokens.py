"""Audit B03 or F03 ranking, then trace frozen MaxSim token contrasts.

No encoder inference or baseline cache mutation. Geometry is estimated with SIFT,
independently of ColNomic. Coordinates always name their EXIF display frame.
"""
import argparse
from collections import Counter, OrderedDict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import shutil

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.scoring import fuse, zscore

CASES = {
    'photo_41de3eb383934ec3': 'line_break',
    'photo_21bd7df9ed6795e7': 'color',
    'photo_1f48d45127db4129': 'text_structure',
    'photo_0e87e0717ab1c215': 'small_text',
}


def primary_predictions(configuration, image, fusion):
    if configuration=='B03':return image
    if configuration=='F03':return fusion
    raise ValueError('Unknown retrieval configuration: '+configuration)


def successful_control(configuration, image_prediction, fusion_prediction):
    return bool(image_prediction['correct'] and
                (configuration=='B03' or fusion_prediction['correct']))


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_hash(value):
    x = value.detach().cpu().contiguous()
    return hashlib.sha256(str(x.dtype).encode() +
        json.dumps(list(x.shape), separators=(',', ':')).encode() +
        x.view(torch.uint8).numpy().tobytes()).hexdigest()


def display_xy(points, orientation):
    """Normalized edge coordinates: raw decoded frame -> EXIF display frame."""
    x, y = np.asarray(points).T
    mappings = {1: (x, y), 2: (1-x, y), 3: (1-x, 1-y), 4: (x, 1-y),
                5: (y, x), 6: (1-y, x), 7: (1-y, 1-x), 8: (y, 1-x)}
    return np.stack(mappings[orientation], axis=1)


def trace_maxsim(query, reference):
    """Keep original tensor indices; zero/masked vectors never enter maxima."""
    qi = torch.where(query.float().norm(dim=1) > 1e-8)[0]
    ri = torch.where(reference.float().norm(dim=1) > 1e-8)[0]
    if not len(qi) or not len(ri):
        raise ValueError('Empty effective token sequence')
    similarity = query[qi].float() @ reference[ri].float().T
    maximum, winner = similarity.max(1)
    return qi.numpy(), ri[winner].numpy(), maximum.numpy()


def words(text):
    return set(re.findall(r'\w+', text.lower()))


def jaccard(a, b):
    a, b = words(a), words(b)
    return len(a & b) / max(1, len(a | b))


class Record:
    def __init__(self, root, bank, spec, ocr, box=None):
        self.spec, self.ocr = spec, ocr
        self.id, self.layout = spec['id'], spec['layout']
        self.z = bank['embeddings'][spec['bank_index']]
        assert bank['ids'][spec['bank_index']] == self.id
        assert bank['sha256'][spec['bank_index']] == spec['sha256'] == ocr['image_sha256']
        self.path = root / bank['paths'][spec['bank_index']]
        assert file_hash(self.path) == spec['sha256'], self.id
        assert tuple(self.z.shape) == (self.layout['total'], 128)
        assert token_hash(self.z) == spec['all_tokens_sha256'], self.id
        with Image.open(self.path) as im:
            assert im.size == (spec['raw_width'], spec['raw_height'])
            assert int(im.getexif().get(274, 1)) == spec['exif_orientation']
            self.image = ImageOps.exif_transpose(im).convert('RGB')
        self.size = np.array(self.image.size, dtype=float)
        assert list(self.image.size) == [ocr['width'], ocr['height']]
        self.box = np.array(box['bbox']) / np.tile(self.size, 2) if box else None
        if box:
            assert box['image_sha256'] == spec['sha256'] and box['status'] == 'confirmed'
        n, w, h = self.layout['count'], self.layout['width'], self.layout['height']
        k = np.arange(n)
        centers = np.stack(((k % w + .5) / w, (k // w + .5) / h), 1)
        self.centers = display_xy(centers, spec['exif_orientation'])
        self.polygons = []
        for row, col in zip(k // w, k % w):
            p = np.array([[col/w, row/h], [(col+1)/w, row/h],
                          [(col+1)/w, (row+1)/h], [col/w, (row+1)/h]])
            self.polygons.append(display_xy(p, spec['exif_orientation']).astype(np.float32))
        self.line_polygons = [np.asarray(l['polygon'], dtype=np.float32) / self.size.astype(np.float32)
                              for l in ocr['lines']]
        self.line_ids, self.line_overlap, self.line_candidates = [], [], []
        for polygon in self.polygons:
            area = abs(cv2.contourArea(polygon))
            overlaps = np.array([cv2.intersectConvexConvex(polygon, p)[0] / area
                                 for p in self.line_polygons])
            ids = np.flatnonzero(overlaps >= .05)
            best = int(overlaps.argmax()) if len(overlaps) and overlaps.max() >= .05 else -1
            self.line_ids.append(best)
            self.line_overlap.append(float(overlaps[best]) if best >= 0 else 0.)
            self.line_candidates.append(ids.tolist())

    def visual(self, indices):
        start = self.layout['start']
        return (indices >= start) & (indices < start + self.layout['count'])

    def info(self, tensor_index):
        k = int(tensor_index) - self.layout['start']
        if not 0 <= k < self.layout['count']:
            return dict(token_index=int(tensor_index), row_raw=None, col_raw=None,
                        x_display=None, y_display=None, ocr_line=-1, ocr_text='',
                        ocr_overlap=0., ocr_candidate_count=0)
        line = self.line_ids[k]
        return dict(token_index=int(tensor_index), row_raw=k // self.layout['width'],
                    col_raw=k % self.layout['width'], x_display=float(self.centers[k, 0]),
                    y_display=float(self.centers[k, 1]), ocr_line=line,
                    ocr_text=self.ocr['lines'][line]['text'] if line >= 0 else '',
                    ocr_overlap=self.line_overlap[k], ocr_candidate_count=len(self.line_candidates[k]))

    def inside(self, indices):
        result = np.zeros(len(indices), dtype=bool)
        visual = self.visual(indices)
        centers = self.centers[indices[visual] - self.layout['start']]
        if self.box is not None:
            result[visual] = ((centers >= self.box[:2]) & (centers < self.box[2:])).all(1)
        return result


def geometry(query, reference):
    """Image keypoint homography; do not interpret rejected fits as misalignment."""
    if reference.id in getattr(query, '_known_geometry', {}):
        return query._known_geometry[reference.id]
    def keypoints(record, masked):
        if hasattr(record, '_sift'):
            return record._sift
        scale = min(1., 1400 / max(record.image.size))
        size = tuple(int(round(x * scale)) for x in record.image.size)
        gray = cv2.cvtColor(np.asarray(record.image.resize(size)), cv2.COLOR_RGB2GRAY)
        mask = None
        if masked:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            x1, y1, x2, y2 = np.rint(record.box * np.tile(size, 2)).astype(int)
            mask[y1:y2, x1:x2] = 255
        kp, desc = cv2.SIFT_create(nfeatures=5000).detectAndCompute(gray, mask)
        record._sift = (kp, desc, np.array(size, dtype=float))
        return record._sift
    qk, qd, qs = keypoints(query, True)
    rk, rd, rs = keypoints(reference, False)
    result = dict(method='independent SIFT + ratio .75 + RANSAC 4px at max dimension 1400',
                  accepted=False, matches=0, inliers=0, coverage=0., homography=None)
    if qd is None or rd is None or len(rd) < 2:
        return result
    matches = [a for pair in cv2.BFMatcher().knnMatch(qd, rd, k=2)
               if len(pair) == 2 for a, b in [pair] if a.distance < .75*b.distance]
    result['matches'] = len(matches)
    if len(matches) < 8:
        return result
    qpts = np.float32([qk[m.queryIdx].pt for m in matches])
    rpts = np.float32([rk[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(20261013)
    matrix, mask = cv2.findHomography(qpts, rpts, cv2.RANSAC, 4., maxIters=10000, confidence=.999)
    if matrix is None or mask is None:
        return result
    inliers = mask.ravel().astype(bool)
    n = int(inliers.sum())
    result['inliers'] = n
    if n < 4:
        return result
    qnormalized = qpts[inliers] / qs
    boxarea = np.prod(query.box[2:] - query.box[:2])
    coverage = float(cv2.contourArea(cv2.convexHull(qnormalized.astype(np.float32))) / boxarea)
    matrix = np.diag([1/rs[0], 1/rs[1], 1.]) @ matrix @ np.diag([qs[0], qs[1], 1.])
    result.update(coverage=coverage, homography=matrix.tolist(),
                  query_inliers=qnormalized.tolist(), reference_inliers=(rpts[inliers]/rs).tolist(),
                  accepted=bool(n >= 8 and coverage >= .15))
    return result


def residuals(query, reference, qi, winners, fit):
    out = [dict(distance_patches=None, row_offset=None, different_ocr_line=None) for _ in qi]
    if not fit['accepted']:
        return out
    h = np.array(fit['homography'])
    # Display axes switch with EXIF transposition.
    width, height = reference.layout['width'], reference.layout['height']
    if reference.spec['exif_orientation'] >= 5:
        width, height = height, width
    for i, (a, b) in enumerate(zip(qi, winners)):
        qa, rb = query.info(a), reference.info(b)
        if qa['x_display'] is None or rb['x_display'] is None:
            continue
        p = h @ [qa['x_display'], qa['y_display'], 1.]
        if abs(p[2]) < 1e-10:
            continue
        expected = p[:2]/p[2]
        displacement = (np.array([rb['x_display'], rb['y_display']])-expected) * [width, height]
        expected_lines = [j for j, poly in enumerate(reference.line_polygons)
                          if cv2.pointPolygonTest(poly, tuple(map(float, expected)), False) >= 0]
        different = None
        if len(expected_lines) == 1 and rb['ocr_line'] >= 0 and rb['ocr_candidate_count'] == 1:
            different = bool(expected_lines[0] != rb['ocr_line'])
        out[i] = dict(distance_patches=float(np.linalg.norm(displacement)),
                      row_offset=float(displacement[1]), different_ocr_line=different)
    return out


def figures(folder, q, positive, negative, qi, pm, nm, delta, inside, fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch, Rectangle
    selected = np.flatnonzero(inside)
    # Equal tails show both directions; this is an illustration, not a random sample.
    order = selected[np.argsort(delta[selected], kind='stable')]
    selected = np.unique(np.r_[order[:8], order[-8:]])
    fig, axes = plt.subplots(2, 2, figsize=(12, 13))
    for row, (r, match, fit, title) in enumerate([(positive, pm, fits[0], 'Correct reference'),
                                                (negative, nm, fits[1], 'Wrong reference')]):
        for ax, record in zip(axes[row], (q, r)):
            im = record.image.copy(); im.thumbnail((1100, 1100))
            ax.imshow(im, extent=(0, 1, 1, 0)); ax.set_axis_off()
        axes[row, 0].set_title(q.id)
        axes[row, 1].set_title(f'{title}: {r.id}\nSIFT geometry accepted={fit["accepted"]}, inliers={fit["inliers"]}')
        for i in selected:
            a, b = q.info(qi[i]), r.info(match[i])
            if b['x_display'] is None:
                continue
            xyq, xyr = (a['x_display'], a['y_display']), (b['x_display'], b['y_display'])
            color = '#15803d' if delta[i] > 0 else '#b91c1c'
            line = ConnectionPatch(xyq, xyr, 'data', 'data', axesA=axes[row, 0],
                                   axesB=axes[row, 1], color=color, alpha=.5, linewidth=.8)
            fig.add_artist(line)
            axes[row, 0].plot(*xyq, 'o', color=color, markersize=3)
            axes[row, 1].plot(*xyr, 'o', color=color, markersize=3)
        x1, y1, x2, y2 = q.box
        axes[row, 0].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color='#2563eb'))
    fig.suptitle('MaxSim correspondence: 8 largest + / 8 largest - foreground deltas\nGreen favors correct; red favors wrong. Link displacement alone is not an error.')
    fig.tight_layout(); fig.savefig(folder/'correspondence.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im=q.image.copy(); im.thumbnail((900, 900)); axes[0].imshow(im,extent=(0,1,1,0))
    visual=q.visual(qi); xy=q.centers[qi[visual]-q.layout['start']]
    limit=max(float(np.quantile(abs(delta[visual]),.95)),1e-4)
    points=axes[0].scatter(xy[:,0],xy[:,1],c=delta[visual],cmap='RdYlGn',vmin=-limit,vmax=limit,s=12,alpha=.7)
    axes[0].set_axis_off(); axes[0].set_title('Query token signed evidence'); fig.colorbar(points,ax=axes[0],shrink=.6)
    order=np.argsort(delta,kind='stable'); axes[1].plot(np.arange(len(delta)),delta[order],linewidth=1)
    axes[1].axhline(0,color='black',linewidth=.5);axes[1].set_title('All valid token deltas, sorted');axes[1].set_xlabel('Token order by delta')
    cumulative=np.cumsum(delta[order])/len(delta);axes[2].plot(cumulative,linewidth=1)
    axes[2].axhline(0,color='black',linewidth=.5);axes[2].set_title(f'Cumulative mean gap = {delta.mean():+.6f}')
    fig.tight_layout();fig.savefig(folder/'contribution.png',dpi=150);plt.close(fig)
    if not any('details' in line['text'].lower() for line in q.ocr['lines']):
        return
    # Native focal crops preserve words; no token cell is treated as an OCR word.
    fig,axes=plt.subplots(1,3,figsize=(15,6))
    for ax,record in zip(axes,(q,positive,negative)):
        lines=[l for l in record.ocr['lines'] if 'details' in l['text'].lower()]
        if record is q:
            lines=[l for l in lines if (np.array(l['polygon']).mean(0)/record.size >= record.box[:2]).all()
                   and (np.array(l['polygon']).mean(0)/record.size < record.box[2:]).all()]
        if lines:
            # Reference "details" occurs in two paragraphs; show all occurrences.
            polys=np.concatenate([np.array(l['polygon']) for l in lines]);lo=polys.min(0);hi=polys.max(0)
            margin=np.array([80.,100.]);lo=np.maximum(0,lo-margin);hi=np.minimum(record.size,hi+margin)
            crop=record.image.crop(tuple(np.r_[lo,hi]));ax.imshow(crop)
        else:
            ax.imshow(record.image)
        ax.set_title(record.id);ax.set_axis_off()
    fig.suptitle('Details-line crops (OCR detections; reference repeats included)')
    fig.tight_layout();fig.savefig(folder/'focal_text.png',dpi=150);plt.close(fig)


def analyze(folder, q, positive, negative, image_scores, bm_scores, fusion_scores, indices, plot):
    qi, pm, pc = trace_maxsim(q.z, positive.z)
    qi2, nm, nc = trace_maxsim(q.z, negative.z)
    assert np.array_equal(qi, qi2)
    delta = pc-nc; inside=q.inside(qi)
    qidx,pidx,nidx=indices
    replay=[float(pc.mean()),float(nc.mean())]
    stored=[float(image_scores[qidx,pidx]),float(image_scores[qidx,nidx])]
    error=max(abs(a-b) for a,b in zip(replay,stored))
    assert error < 2e-6, (q.id,error)
    assert abs(float(delta.mean())-(stored[0]-stored[1])) < 2e-6
    fits=[geometry(q,positive),geometry(q,negative)]
    pr=residuals(q,positive,qi,pm,fits[0]);nr=residuals(q,negative,qi,nm,fits[1])
    rows=[]; regions=[]
    for i in range(len(qi)):
        a,b,c=q.info(qi[i]),positive.info(pm[i]),negative.info(nm[i])
        region='background' if not inside[i] else 'subject_text' if a['ocr_line']>=0 else 'subject_nontext'
        if a['x_display'] is None:region='nonvisual'
        regions.append(region)
        row=dict(q=a['token_index'],region=region,c_positive=float(pc[i]),c_negative=float(nc[i]),
                 delta=float(delta[i]),contribution_to_mean_gap=float(delta[i]/len(delta)))
        for prefix,info in [('q_',a),('pos_',b),('neg_',c)]:
            row.update({prefix+k:v for k,v in info.items()})
        for prefix,res in [('pos_',pr[i]),('neg_',nr[i])]:row.update({prefix+k:v for k,v in res.items()})
        row['pos_text_overlap']=jaccard(a['ocr_text'],b['ocr_text']) if a['ocr_text'] and b['ocr_text'] else None
        row['neg_text_overlap']=jaccard(a['ocr_text'],c['ocr_text']) if a['ocr_text'] and c['ocr_text'] else None
        rows.append(row)
    groups=[]
    for region in sorted(set(regions)):
        mask=np.array(regions)==region
        groups.append(dict(region=region,n=int(mask.sum()),delta_sum=float(delta[mask].sum()),
                           contribution_to_mean_gap=float(delta[mask].sum()/len(delta)),
                           within_region_mean=float(delta[mask].mean())))
    lines=[]
    for line in sorted({r['q_ocr_line'] for r in rows if r['q_ocr_line']>=0}):
        ix=np.array([r['q_ocr_line']==line for r in rows]);lines.append(dict(line=line,text=q.ocr['lines'][line]['text'],
            n=int(ix.sum()),delta_sum=float(delta[ix].sum()),contribution_to_mean_gap=float(delta[ix].sum()/len(delta)),
            in_subject_tokens=int(inside[ix].sum())))
    summary=dict(photo_id=q.id,positive=positive.id,negative=negative.id,valid_tokens=len(qi),
        subject_tokens=int(inside.sum()),image_positive=stored[0],image_negative=stored[1],
        image_gap=stored[0]-stored[1],bm25_gap=float(bm_scores[qidx,pidx]-bm_scores[qidx,nidx]),
        fusion_gap=float(fusion_scores[qidx,pidx]-fusion_scores[qidx,nidx]),
        positive_evidence=float(np.maximum(delta,0).sum()/len(delta)),
        negative_evidence=float(np.minimum(delta,0).sum()/len(delta)),
        subject_only_gap=float(delta[inside].mean()) if inside.any() else None,
        replay_max_error=error,groups=groups,
        query_raw_grid=q.layout,query_exif_orientation=q.spec['exif_orientation'],geometry=fits)
    summary.update(positive_delta_tokens=int((delta>0).sum()),negative_delta_tokens=int((delta<0).sum()),
                   foreground_unique_positive_matches=int(len(np.unique(pm[inside]))),
                   foreground_unique_negative_matches=int(len(np.unique(nm[inside]))),
                   multiple_OCR_line_subject_tokens=sum(r['q_ocr_candidate_count']>1 for r,keep in zip(rows,inside) if keep),
                   foreground_OCR_tokens=sum(r['q_ocr_line']>=0 for r,keep in zip(rows,inside) if keep))
    for label,res in [('positive',pr),('negative',nr)]:
        values=[r['distance_patches'] for r,keep in zip(res,inside) if keep and r['distance_patches'] is not None]
        summary[label+'_geometry_distance_median_patches']=float(np.median(values)) if values else None
        summary[label+'_geometry_over_3_patches_fraction']=float(np.mean(np.array(values)>3)) if values else None
        summary[label+'_geometry_evaluable_tokens']=len(values)
    folder.mkdir(parents=True,exist_ok=True)
    write_csv(folder/'tokens.csv',rows);write_csv(folder/'regions.csv',groups);write_csv(folder/'ocr_lines.csv',lines)
    write_json(folder/'summary.json',summary)
    if plot:figures(folder,q,positive,negative,qi,pm,nm,delta,inside,fits)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--configuration',choices=('B03','F03'),default='F03')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--query',action='append',choices=list(CASES))
    parser.add_argument('--all-errors',action='store_true',help='Analyze every failure and one available success control per contrast')
    parser.add_argument('--resume',action='store_true',help='Reuse completed diagnostics after checking their scores and image/token hashes')
    parser.add_argument('--reuse-from',type=Path,help='Reuse compatible prior token contrasts and image geometry, with provenance checks')
    args=parser.parse_args();root=args.root;configuration=args.configuration
    out=args.output or root/'results/token_mechanism_analysis'/configuration
    torch.set_num_threads(2);cv2.setNumThreads(2)
    selected=args.query or list(CASES)
    load=lambda p:torch.load(root/p,map_location='cpu',weights_only=True,mmap=True)
    qb=load('cache/image_embeddings/colnomic7b/photos.pt');rb=load('cache/image_embeddings/colnomic7b/references.pt')
    image=load('cache/scores/colnomic7b/image.pt');bm=load('cache/scores/bm25/bm25.pt')
    fusion=load('cache/scores/bm25_image_fusion/colnomic7b/alpha_0.5.pt')
    for bank in (image,bm,fusion):
        assert bank['query_ids']==qb['ids'] and bank['reference_ids']==rb['ids']
    pred=lambda p:{x['photo_id']:x for x in [json.loads(l) for l in (root/p).read_text().splitlines()]}
    ip=pred('results/colnomic7b/image/predictions.jsonl')
    fp=pred('results/bm25_image_fusion/colnomic7b/alpha_0.5/predictions.jsonl')
    primary=primary_predictions(configuration,ip,fp)
    bp=pred('results/bm25/bm25/predictions.jsonl')
    qo=read(root/'cache/ocr/photos.json');ro=read(root/'cache/ocr/references.json')
    boxes=read(root/'dataset/annotations/subject_boxes.json')['records']
    manifest_path=Path(read(root/'cache/content_box_ce_polish18/deployment/active.json')['path'])/'manifest.json'
    manifest=read(manifest_path);qs={x['id']:x for x in manifest['queries']}
    rs={x['id']:x for x in manifest['references'].values()}
    reference_by_sha={x['sha256']:x for x in rs.values()}
    qindex={x:i for i,x in enumerate(qb['ids'])};rindex={x:i for i,x in enumerate(rb['ids'])}
    # Confirm F03 arithmetic using the whole gallery, not candidate-specific scaling.
    expected=fuse(image['scores'],bm['scores'],[bool(words(qo[x]['text'])) for x in qb['ids']],.5)
    fusion_error=float((expected-fusion['scores']).abs().max());assert fusion_error<2e-6
    ambiguous_rows=list(csv.DictReader((root/'wrong/身份歧义/cases.csv').open()))
    ambiguous_photos={r['photo_id'] for r in ambiguous_rows}
    ambiguous_pairs={(r['photo_id'],r['correct_reference_sha256'],r['selected_reference_sha256']) for r in ambiguous_rows}
    def ambiguity(id,p):
        pair=(id,rb['sha256'][rindex[p['positive_reference_ids'][0]]],rb['sha256'][rindex[p['selected_reference_id']]])
        return pair in ambiguous_pairs
    ambiguous={id for id,p in primary.items() if not p['correct'] and ambiguity(id,p)}
    split_by_id={r['photo_id']:split for split in ('train','val','test')
                 for r in csv.DictReader((root/f'dataset/lists/{split}.csv').open())}
    inventory=[]
    for id,p in primary.items():
        if p['correct']:continue
        qi=qindex[id];pi=rindex[p['positive_reference_ids'][0]];ni=rindex[p['selected_reference_id']]
        ig=float(image['scores'][qi,pi]-image['scores'][qi,ni]);bg=float(bm['scores'][qi,pi]-bm['scores'][qi,ni])
        image_z=zscore(image['scores'][qi]);bm_z=zscore(bm['scores'][qi])
        iz=float(.5*(image_z[pi]-image_z[ni]));bz=float(.5*(bm_z[pi]-bm_z[ni]))
        inventory.append(dict(photo_id=id,package_id=p['package_id'],positive=p['positive_reference_ids'][0],
            negative=p['selected_reference_id'],image_rank=ip[id]['rank'],bm25_rank=bp[id]['rank'],fusion_rank=fp[id]['rank'],
            primary_rank=p['rank'],configuration=configuration,split=split_by_id[id],
            image_top1_correct=ip[id]['correct'],image_gap=ig,bm25_gap=bg,image_z_contribution=iz,bm25_z_contribution=bz,
            fusion_gap=float(fusion['scores'][qi,pi]-fusion['scores'][qi,ni]),
            failure_stage='fusion_overrode_correct_image_top1' if ip[id]['correct'] else 'image_already_wrong',
            identity_ambiguity=id in ambiguous,prior_other_candidate_ambiguity=id in ambiguous_photos and id not in ambiguous,
            user_case_type=CASES.get(id,'unclassified')))
        assert abs(iz+bz-inventory[-1]['fusion_gap']) < 1e-5, id
        if configuration=='B03':
            assert rb['ids'][int(image['scores'][qi].argmax())]==p['selected_reference_id'],id
            assert ig<=0,id
    source_mode='image' if configuration=='B03' else 'bm25_image_fusion__alpha_0.5'
    source_error_path=root/'wrong'/f'{configuration}__colnomic7b__{source_mode}'/'errors.csv'
    source_errors=list(csv.DictReader(source_error_path.open()))
    assert {r['photo_id'] for r in source_errors}=={r['photo_id'] for r in inventory}
    for r in source_errors:
        p=primary[r['photo_id']]
        assert r['selected_reference_id']==p['selected_reference_id']
        assert r['correct_reference_id']==p['positive_reference_ids'][0] and int(r['correct_rank'])==p['rank']
    write_csv(out/f'{configuration}_error_inventory.csv',inventory)
    if args.all_errors:
        selected=[x['photo_id'] for x in inventory]
    summaries=[];targets=[];record_cache=OrderedDict()
    completed_contrasts={};known_geometry={};prior_hash=None
    if args.reuse_from:
        prior=read(args.reuse_from/'summary.json')
        assert prior['sources']['image_score_signature']==image['signature']
        assert prior['sources']['manifest_sha256']==file_hash(manifest_path)
        assert prior['geometry_settings']==dict(sift_ratio=.75,max_dimension=1400,ransac_threshold_pixels=4,min_inliers=8,min_coverage=.15)
        prior_hash=file_hash(args.reuse_from/'summary.json')
        for s in prior['cases']:
            folder=args.reuse_from/s['target']/s['photo_id']
            completed_contrasts[(s['photo_id'],s['positive'],s['negative'])]=folder
            known_geometry.setdefault(s['photo_id'],{})[s['positive']]=s['geometry'][0]
            known_geometry[s['photo_id']][s['negative']]=s['geometry'][1]
    processor=None
    def record(id,reference=False):
        nonlocal processor
        if reference and id not in rs:
            index=rindex[id];sha=rb['sha256'][index]
            if sha in reference_by_sha:
                spec=dict(reference_by_sha[sha],id=id,bank_index=index)
            else:
                if processor is None:
                    from colpali_engine.models import ColQwen2_5_Processor
                    processor=ColQwen2_5_Processor.from_pretrained(root/'weights/colnomic7b/adapter',local_files_only=True)
                from drugpack.method.token_layout import token_layout
                with Image.open(root/rb['paths'][index]) as im:
                    width,height=im.size;orientation=int(im.getexif().get(274,1))
                spec=dict(id=id,bank_index=index,sha256=sha,raw_width=width,raw_height=height,
                          exif_orientation=orientation,layout=token_layout(processor,'colnomic7b',width,height),
                          all_tokens_sha256=token_hash(rb['embeddings'][index]),
                          layout_provenance='reconstructed with frozen local processor; score parity checked independently')
            rs[id]=spec
        if id not in record_cache:
            record_cache[id]=Record(root,rb if reference else qb,rs[id] if reference else qs[id],
                                    ro[id] if reference else qo[id],None if reference else boxes[id])
        record_cache.move_to_end(id)
        result=record_cache[id]
        result._known_geometry=known_geometry.get(id,{})
        while len(record_cache)>8:
            record_cache.popitem(last=False)
        return result
    for target in selected:
        p=primary[target];positive=p['positive_reference_ids'][0];negative=p['selected_reference_id']
        assert positive!=negative
        family=[x for x in qb['ids'] if fp[x]['package_id']==p['package_id']]
        controls=[x for x in family if successful_control(configuration,ip[x],fp[x]) and x!=target]
        # Prefer comparable photo conditions and foreground size; still report face confounding.
        def control_key(id):
            type_penalty=sum(ip[id]['type'][k]!=ip[target]['type'][k] for k in ip[target]['type'])
            b=boxes[id];fraction=np.prod(np.array(b['bbox'][2:])-b['bbox'][:2])/(b['width']*b['height'])
            t=boxes[target];tf=np.prod(np.array(t['bbox'][2:])-t['bbox'][:2])/(t['width']*t['height'])
            return type_penalty,abs(fraction-tf),id
        control=min(controls,key=control_key) if controls else None
        targets.append(dict(photo_id=target,case_type=CASES.get(target,'unclassified'),package_id=p['package_id'],
                            positive=positive,negative=negative,control=control,cohort_size=len(family),
                            identity_ambiguity=target in ambiguous,
                            prior_other_candidate_ambiguity=target in ambiguous_photos and target not in ambiguous,
                            split=split_by_id[target]))
        analyze_ids=[target]+([control] if control else []) if args.all_errors else family
        for id in analyze_ids:
            folder=out/target/id
            qrec,prec,nrec=record(id),record(positive,True),record(negative,True)
            key=(id,positive,negative)
            reused_from=None
            if not (folder/'summary.json').exists() and key in completed_contrasts:
                source=completed_contrasts[key];old=read(source/'summary.json')
                assert old['photo_id']==id and old['positive']==positive and old['negative']==negative
                assert abs(old['image_positive']-float(image['scores'][qindex[id],rindex[positive]]))<2e-6
                assert abs(old['image_negative']-float(image['scores'][qindex[id],rindex[negative]]))<2e-6
                folder.mkdir(parents=True,exist_ok=True)
                for name in ('summary.json','tokens.csv','regions.csv','ocr_lines.csv'):
                    shutil.copyfile(source/name,folder/name)
                reused_from=str(source)
                if id==target or target in CASES:
                    with (folder/'tokens.csv').open() as stream:old_rows=list(csv.DictReader(stream))
                    qi=np.array([int(r['q_token_index']) for r in old_rows]);pm=np.array([int(r['pos_token_index']) for r in old_rows]);nm=np.array([int(r['neg_token_index']) for r in old_rows])
                    delta=np.array([float(r['delta']) for r in old_rows],dtype=np.float32)
                    figures(folder,qrec,prec,nrec,qi,pm,nm,delta,qrec.inside(qi),old['geometry'])
            reuse=args.resume and all((folder/name).exists() for name in ('summary.json','tokens.csv','regions.csv','ocr_lines.csv'))
            reuse=reuse or reused_from is not None
            reuse=reuse and (id!=target or all((folder/name).exists() for name in ('correspondence.png','contribution.png')))
            if reuse:
                s=read(folder/'summary.json')
                assert s['positive']==positive and s['negative']==negative and s['photo_id']==id
                assert abs(s['image_positive']-float(image['scores'][qindex[id],rindex[positive]]))<2e-6
                assert abs(s['image_negative']-float(image['scores'][qindex[id],rindex[negative]]))<2e-6
                with (folder/'tokens.csv').open() as stream:
                    tokens=list(csv.DictReader(stream))
                assert len(tokens)==s['valid_tokens']
                assert abs(sum(float(x['contribution_to_mean_gap']) for x in tokens)-s['image_gap'])<2e-6
            else:
                s=analyze(folder,qrec,prec,nrec,image['scores'],bm['scores'],fusion['scores'],
                          (qindex[id],rindex[positive],rindex[negative]),id==target or target in CASES)
            s['resumed']=bool(reuse)
            if reused_from is not None:s['reused_from']=reused_from
            s.update(case_type=CASES.get(target,'unclassified'),target=target,role='target' if id==target else 'success_control' if id==control else 'cohort',
                     image_rank=ip[id]['rank'],fusion_rank=fp[id]['rank'],configuration=configuration,split=split_by_id[id])
            write_json(folder/'summary.json',s);summaries.append(s)
            completed_contrasts[key]=folder
            known_geometry.setdefault(id,{})[positive]=s['geometry'][0];known_geometry[id][negative]=s['geometry'][1]
            print(id,s['role'],'gap',round(s['image_gap'],6),'subject',None if s['subject_only_gap'] is None else round(s['subject_only_gap'],6),
                  'geometry',[(g['inliers'],round(g['coverage'],2),g['accepted']) for g in s['geometry']],flush=True)
        write_json(out/'progress.json',dict(completed_targets=len(targets),total_targets=len(selected),
                                          completed_comparisons=len(summaries),last_target=target))
    write_json(out/'summary.json',dict(schema='colnomic.token_mechanism.v1',targets=targets,cases=summaries,
        configuration=configuration,expected_errors=len(inventory),primary_score='image MaxSim' if configuration=='B03' else 'BM25 image fusion',
        identity_ambiguity=sum(x['identity_ambiguity'] for x in inventory),
        prior_other_candidate_ambiguity=sum(x['prior_other_candidate_ambiguity'] for x in inventory),
        success_control_definition='image Top-1 correct' if configuration=='B03' else 'image and fusion Top-1 correct',
        stage_counts=dict(Counter(x['failure_stage'] for x in inventory)),
        confirmed_fusion_replay_max_error=fusion_error,
        manual_error_annotation_count=len(read(root/'dataset/review/error_analysis/annotations.json')['labels']),
        coordinate_frame='normalized EXIF-transposed display; row_raw and col_raw refer to cached raw grid',
        geometry_settings=dict(sift_ratio=.75,max_dimension=1400,ransac_threshold_pixels=4,min_inliers=8,min_coverage=.15),
        OCR_assignment='maximum polygon overlap >=5% of patch; candidate count exposes ambiguous multi-line cells',
        sources=dict(script_sha256=file_hash(__file__),manifest_sha256=file_hash(manifest_path),
                     image_score_signature=image['signature'],bm25_score_signature=bm['signature'],fusion_score_signature=fusion['signature'],
                     source_errors_sha256=file_hash(source_error_path),
                     prior_summary_sha256=prior_hash)))
    flat=[]
    for s in summaries:
        flat.append({k:v for k,v in s.items() if not isinstance(v,(dict,list))})
    write_csv(out/'cohort_summary.csv',flat)


if __name__=='__main__':main()
