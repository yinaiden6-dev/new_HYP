"""Local shared-OCR-phrase identity probe with explicit ROI supervision.

This is a natural restricted identity task, not proof of layout-only causation:
frozen tokens retain whole-image context and cells may span unshared print.
All eligibility and phrase choices precede examining feature responses.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from scripts.colnomic_mechanism.common import ROOT, data, read, write, csvout, sha
from scripts.colnomic_mechanism.geometry import anchors
from scripts.colnomic_mechanism.full_tokens import TokenRecord
from scripts.colnomic_mechanism.display_pixels import display_image
from scripts.colnomic_mechanism.set_probes import train_selected, metrics_for_samples, pair_features, weights_for
from scripts.colnomic_mechanism.probes import logistic_fit, logistic_predict, shuffled_labels, save_head, validation_threshold, paired_metrics


def textkey(text): return ' '.join(re.findall(r'[^\W_]+',text.lower()))


def prepare(source,folder):
    d=data(); protocol=dict(scope='natural local shared phrases, not full-pack same-content and not a layout-only causal task',
        population='all1117 queries/all130 identities; positive vs unique image/fusion strongest negative',
        quality='both OCR homographies accepted;>=4 shared exact normalized phrases;every selected phrase confidence>=.95 andbboxheight>=10px',
        selection='four longest shared exact phrases, lexical tie;first eligible query peridentity for manual inspection only;all qualified pairs used',
        inputs='ROI-selected frozen visual features;explicit extra OCR/coordinate supervision for Token selection; no coordinates/index/ID in classifier',
        labels='original natural package positives vs negatives; candidate full content, dose, graphics and contextual encoding may differ',
        train_negatives='exclude any reference SHA that is a groundtruth design of validation/test packages',
        fitting='training-only standardization,validation identity-macro paired ranking thenAUC then smallerC;3 within-package training label shuffles',
        interpretation='ability to distinguish natural candidates from common-phrase selected features, not proof that layout alone is encoded or used',
        historical_test_exposure=True,seed=20261014,manual_review_required_before_confirmed_results=True)
    if (folder/'protocol.json').exists(): assert read(folder/'protocol.json')==protocol
    else:write(folder/'protocol.json',protocol)
    ledger=read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')
    fits={(r['photo_id'],r['reference_id']):r for r in read(source/'reviewed_boxes/alignments.json')}
    held={digest for r in ledger if d['splits'][r['photo_id']]!='train' for digest in r['positive_reference_sha256']}
    qi={x:i for i,x in enumerate(d['qb']['ids'])};ri={x:i for i,x in enumerate(d['rb']['ids'])}
    cases=[];rejected=[]
    for row in ledger:
        qid=row['photo_id'];boxrow=d['boxes'][qid];box=np.array(boxrow['bbox'],np.float32)/np.tile([boxrow['width'],boxrow['height']],2)
        qa=anchors(d['qo'][qid],box)
        for pos in row['positive_reference_ids']:
            for neg in sorted(set([row['image_strongest_wrong'],row['fusion_strongest_wrong']])):
                reason=None
                if (qid,pos) not in fits or (qid,neg) not in fits:reason='not_both_aligned'
                elif d['splits'][qid]=='train' and d['rb']['sha256'][ri[neg]] in held:reason='held_identity_negative_excluded_from_training'
                else:
                    common=sorted(set(qa)&set(anchors(d['ro'][pos]))&set(anchors(d['ro'][neg])))
                    if len(common)<4:reason='less_than4_shared_exact_phrases'
                    else:
                        common=sorted(common,key=lambda t:(-len(t),t))[:4]
                        selected_lines=[]
                        for rec in [d['qo'][qid],d['ro'][pos],d['ro'][neg]]:
                            lines=[l for l in rec['lines'] if textkey(l['text']) in common]
                            selected_lines.append(lines)
                        if any(len(ls)!=4 for ls in selected_lines):reason='non_unique_phrase_regions'
                        elif min(float(l['confidence']) for ls in selected_lines for l in ls)<.95:reason='low_phrase_confidence'
                        elif min(float(l['bbox'][3]-l['bbox'][1]) for ls in selected_lines for l in ls)<10:reason='phrase_height_less_than10px'
                if reason:
                    rejected.append(dict(photo_id=qid,package_id=row['package_id'],positive=pos,negative=neg,reason=reason));continue
                cases.append(dict(case_id=qid+'__'+pos+'__'+neg,photo_id=qid,positive=pos,negative=neg,
                    package_id=row['package_id'],group_id=d['groups'][row['package_id']],split=d['splits'][qid],
                    common_exact_phrases=common,manual_review_status='pending',automatic_eligibility=True))
    write(folder/'eligible_cases.json',cases);csvout(folder/'rejections.csv',rejected)
    write(folder/'eligibility.json',dict(all_queries=len(ledger),all_identities=len({r['package_id'] for r in ledger}),
        pairs=len(cases),queries=len({r['photo_id'] for r in cases}),identities=len({r['package_id'] for r in cases}),
        identities_by_split={s:len({r['package_id'] for r in cases if r['split']==s}) for s in ['train','val','test']},
        exclusion_counts=dict(Counter(r['reason'] for r in rejected)),protocol_SHA256=sha(folder/'protocol.json')))
    # Read pixels only once per identity for inspection cards; no feature response is used.
    selected={}
    for c in cases:selected.setdefault(c['package_id'],c)
    (folder/'manual_material').mkdir(exist_ok=True)
    manual=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for package,c in sorted(selected.items()):
        canvas=Image.new('RGB',(1800,650),'white');draw=ImageDraw.Draw(canvas)
        for col,(name,ident,bank,ocr) in enumerate([('QUERY',c['photo_id'],d['qb'],d['qo']),('POSITIVE',c['positive'],d['rb'],d['ro']),('NEGATIVE',c['negative'],d['rb'],d['ro'])]):
            index=qi[ident] if col==0 else ri[ident];path=ROOT/bank['paths'][index]
            assert sha(path)==bank['sha256'][index]
            with Image.open(path) as im:image=display_image(im)
            draw.text((col*600+6,5),name+' '+ident,fill='black',font=font)
            for lineindex,key in enumerate(c['common_exact_phrases']):
                line=next(l for l in ocr[ident]['lines'] if textkey(l['text'])==key)
                x1,y1,x2,y2=line['bbox'];patch=image.crop((max(0,int(x1)-3),max(0,int(y1)-3),min(image.width,int(x2)+3),min(image.height,int(y2)+3)))
                patch.thumbnail((590,105));canvas.paste(patch,(col*600+4,42+lineindex*150+(105-patch.height)//2))
                draw.text((col*600+4,150+lineindex*150),key[:73],fill='black',font=font)
        path=folder/'manual_material'/(package+'.jpg');canvas.save(path,quality=96)
        manual.append(dict(package_id=package,case_id=c['case_id'],photo_id=c['photo_id'],positive=c['positive'],negative=c['negative'],
            phrases=c['common_exact_phrases'],material=str(path.relative_to(folder)),material_SHA256=sha(path),manual_result=None))
    write(folder/'manual_review_queue.json',manual)
    print('Eligibility',len(cases),'pairs',len(selected),'identities',flush=True)


def analyze(source,folder):
    d=data();cases=read(folder/'eligible_cases.json');reviews=read(folder/'manual_reviews.json')
    confirmed={r['package_id'] for r in reviews if r['shared_phrase_content_confirmed']}
    assert len(reviews)==len({r['package_id'] for r in read(folder/'manual_review_queue.json')})
    # One visual word/face check per identity is not claimed as every source photograph's transcription.
    cases=[c for c in cases if c['package_id'] in confirmed]
    qi={x:i for i,x in enumerate(d['qb']['ids'])};ri={x:i for i,x in enumerate(d['rb']['ids'])};record_cache={}
    def record(ident):
        if ident not in record_cache:
            query=ident in qi;i=qi[ident] if query else ri[ident];bank=d['qb'] if query else d['rb']
            spec=d['metadata']['photos' if query else 'references'][i];ocr=(d['qo'] if query else d['ro'])[ident]
            record_cache[ident]=TokenRecord(bank,spec,ocr,ROOT,d['boxes'][ident] if query else None)
        return record_cache[ident]
    from scripts.colnomic_mechanism.cross_token_probe import cpu_relation, NAMES
    pools=[];scalars=[];relations=[];counts=[];indices=[];samples=[];rejected=[]
    for c in cases:
        selected=[]
        for ident in [c['photo_id'],c['positive'],c['negative']]:
            r=record(ident);lines={i for i,l in enumerate(r.ocr['lines']) if textkey(l['text']) in c['common_exact_phrases']}
            raw=np.array([i for i in r.valid if np.isfinite(r.xy[i]).all() and set(r.line_ids[i])&lines],dtype=int)
            if len(raw)<4:break
            selected.append((r,raw))
        if len(selected)!=3:
            rejected.append(dict(case_id=c['case_id'],reason='less_than4 ROI tokens'));continue
        q=selected[0][0].tensor[selected[0][1]].float().numpy();qm=np.r_[q.mean(0),q.std(0)]
        for y,(r,raw) in enumerate(selected[1:]):
            arr=r.tensor[raw].float().numpy();rm=np.r_[arr.mean(0),arr.std(0)];dot=q@arr.T
            pools.append(pair_features(qm,rm));scalars.append([float(dot.max(1).mean())])
            relations.append(cpu_relation(dot));counts.append([np.log1p(len(q)),np.log1p(len(arr)),len(q)/len(arr)])
            label=1-y
            samples.append(dict(**c,query_id=c['case_id'],reference_id=r.id,label=label,
                extra_supervision='OCR shared phrase ROI selection',whole_image_context_retained=True,
                query_tokens=len(q),candidate_tokens=len(arr)))
            indices.append(dict(case_id=c['case_id'],reference_id=r.id,query_indices=selected[0][1].tolist(),reference_indices=raw.tolist(),
                query_multi_line_tokens=sum(selected[0][0].multiple[selected[0][1]]>1),reference_multi_line_tokens=sum(r.multiple[raw]>1)))
    csvout(folder/'token_input_indices.csv',indices);csvout(folder/'sample_index.csv',samples);csvout(folder/'token_quality_rejections.csv',rejected)
    pools=np.array(pools);scalars=np.array(scalars);relations=np.array(relations);counts=np.array(counts)
    labels=np.array([r['label'] for r in samples]);packages=np.array([r['package_id'] for r in samples]);splits=np.array([r['split'] for r in samples]);qids=np.array([r['query_id'] for r in samples])
    np.savez_compressed(folder/'inputs.npz',pooled=pools,maxsim=scalars,relations=relations,token_counts=counts,labels=labels,package_ids=packages,splits=splits,query_ids=qids)
    write(folder/'relation_feature_names.json',NAMES)
    metricrows=[];predrows=[];pairrows=[];nullrows=[]
    candidates={'local_maxsim_only':scalars,'shared_phrase_pooled_only':pools,'shared_phrase_pooled_plus_maxsim':np.column_stack([pools,scalars]),
        'shared_phrase_relation_only':relations,'ROI_token_count_control':counts}
    for kind,x in candidates.items():
        model,C,threshold=train_selected(x,labels,qids,packages,splits,folder,kind);logits=logistic_predict(model,x)
        metricrows.extend(dict(probe=kind,**r) for r in metrics_for_samples(labels,logits,qids,packages,splits,threshold))
        predrows.extend(dict(**r,probe=kind,score=float(s),validation_threshold=threshold) for r,s in zip(samples,logits))
        _,pairs=paired_metrics(labels,logits,qids,packages);pairrows.extend(dict(probe=kind,**r) for r in pairs)
        train=splits=='train';val=splits=='val'
        for seed in [17,31,53]:
            sy=shuffled_labels(labels[train],packages[train],seed);null=logistic_fit(x[train],sy,C,weights_for(packages)[train]);cut=validation_threshold(labels[val],logistic_predict(null,x[val]),packages[val])
            save_head(folder/(kind+f'_shuffle_{seed}_head.pt'),null,dict(seed=seed,C=C,threshold=cut))
            guess=logistic_predict(null,x);metricrows.extend(dict(probe=kind,control='label_shuffle',seed=seed,**r) for r in metrics_for_samples(labels,guess,qids,packages,splits,cut))
            nullrows.extend(dict(**r,probe=kind,seed=seed,score=float(s)) for r,s in zip(samples,guess))
    csvout(folder/'metrics.csv',metricrows);csvout(folder/'predictions.csv',predrows);csvout(folder/'pair_gaps.csv',pairrows);csvout(folder/'null_predictions.csv',nullrows)
    assert all(not set(packages[splits==a])&set(packages[splits==b]) for a,b in [('train','val'),('train','test'),('val','test')])
    write(folder/'validation.json',dict(status='passed_for_shared_phrase_ROI_restricted_identity_probe',
        pairs=len(samples)//2,identities=len(set(packages)),identity_counts_by_split={s:len(set(packages[splits==s])) for s in ['train','val','test']},
        heads=len(candidates),null_heads=3*len(candidates),explicit_ROI_supervision=True,full_gallery_evaluation=False,
        natural_full_pack_same_words_not_asserted=True,layout_only_cause_not_asserted=True,
        all_pair_text_transcription='four phrase content checked in first eligible case per identity; other source cases OCR-confirmed only',
        original_input_cache_not_modified=True))
    print('Probe complete',len(samples)//2,'pairs',len(set(packages)),'identities',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--analyze',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(2);a.output.mkdir(parents=True,exist_ok=True)
    (analyze if a.analyze else prepare)(a.source.resolve(),a.output.resolve())


if __name__=='__main__':main()
