"""CPU replay of numeric pixel occlusion against the original frozen gallery.

These are conditional candidate responses, not edited-image identity accuracy.
Numeric OCR proposals and supplied boxes do not establish identity importance.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from scripts.colnomic_mechanism.common import ROOT, data, read, write, csvout, sha
from scripts.colnomic_mechanism.display_pixels import display_image
from scripts.colnomic_mechanism.geometry import grid
from scripts.encode_colnomic_images import token_hash
from scripts.analyze_colnomic_tokens import display_xy

ARMS = ['base', 'sham', 'numeric_mask', 'blank_control', 'both_masks']
REGIONS = ['numeric_proposal', 'blank_control', 'other_supplied_subject', 'background_outside_supplied_box', 'template']


def intersection_fraction(rectangles, box):
    rect = np.asarray(rectangles, np.float64); box = np.asarray(box, np.float64)
    area = np.prod(rect[:, 2:]-rect[:, :2], axis=1)
    intersection = np.prod(np.maximum(np.minimum(rect[:, 2:],box[2:])-np.maximum(rect[:, :2],box[:2]),0),axis=1)
    return intersection/area


def region_weights(spec, source, subject, display_size, qi):
    """Exclusive normalized soft assignment by actual patch/rectangle overlap."""
    layout=spec['layout'];start=layout['start'];width=layout['width'];height=layout['height']
    visual=(qi>=start)&(qi<start+layout['count']);weights=np.zeros((len(qi),5),np.float64)
    weights[~visual,4]=1
    k=qi[visual]-start;x=k%width;y=k//width
    points=np.stack([np.stack([x/width,y/height],1),np.stack([(x+1)/width,(y+1)/height],1)],1)
    mapped=display_xy(points.reshape(-1,2),spec['exif_orientation']).reshape(-1,2,2)
    rectangles=np.c_[mapped.min(1),mapped.max(1)]
    scale=np.tile(display_size,2)
    numeric=intersection_fraction(rectangles,np.asarray(source['numeric_bbox'])/scale)
    blank=intersection_fraction(rectangles,np.asarray(source['blank_bbox'])/scale)
    subj=intersection_fraction(rectangles,np.asarray(subject)/scale)
    if np.any(numeric+blank>subj+1e-6):
        raise ValueError('Edit ROI is outside supplied region; redefine partition explicitly')
    weights[visual,0]=numeric;weights[visual,1]=blank
    weights[visual,2]=np.maximum(0,subj-numeric-blank);weights[visual,3]=1-subj
    if not np.allclose(weights.sum(1),1,atol=1e-6) or np.min(weights)<-1e-7:
        raise ValueError('Region weights must conserve tokens')
    return weights,visual


def traced_similarity(q,r):
    qi=torch.nonzero(q.float().norm(dim=1)>1e-8).flatten().numpy()
    ri=torch.nonzero(r.float().norm(dim=1)>1e-8).flatten().numpy()
    if not len(qi) or len(ri)<2:
        raise ValueError('Need nonempty query and >=2 valid reference tokens')
    dot=(q[qi].float()@r[ri].float().T).numpy()
    order=np.argsort(-dot,axis=1,kind='stable')[:,:2]
    maximum=dot[np.arange(len(qi)),order[:,0]]
    return dict(qi=qi,ri=ri,similarity=dot,winner=ri[order[:,0]],second=ri[order[:,1]],
        maximum=maximum,second_maximum=dot[np.arange(len(qi)),order[:,1]],score=float(maximum.mean()))


def rgba_and_display(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB')).copy(),np.asarray(display_image(im)).copy()


def run(folder, subject_box_snapshot):
    output=folder/'analysis';output.mkdir(parents=True,exist_ok=True)
    sources=read(folder/'sources.json');manifest=read(folder/'encoding_manifest.json')
    completion=read(folder/'embeddings/complete.json')
    if completion['status']!='complete' or completion['records']!=len(manifest):
        raise ValueError('Encoding must finish before numerical analysis')
    d=data()
    frozen_boxes=read(subject_box_snapshot)
    d['boxes']=frozen_boxes.get('records',frozen_boxes)
    if not all(s['photo_id'] in d['boxes'] for s in sources):
        raise ValueError('Committed subject-box snapshot misses intervention sources')
    rindex={rid:i for i,rid in enumerate(d['rb']['ids'])}
    ledger={row['photo_id']:row for row in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    semantic_path=folder/'visual_numeric_verification.json'
    semantic_record=read(semantic_path) if semantic_path.exists() else {}
    semantics={r['photo_id']:r for r in semantic_record.get('records',[])}
    source_rows=[];pair_rows=[];region_rows=[];pixel_rows=[];matrix_index=[];conservation=[]
    processed=set();max_cache_score_error=0.;matrix_dir=output/'matrices';matrix_dir.mkdir(exist_ok=True)
    for source in sources:
        qid=source['photo_id'];qi_bank=source['cache_row'];cached=d['qb']['embeddings'][qi_bank]
        if d['qb']['ids'][qi_bank]!=qid or sha(source['original_path'])!=d['qb']['sha256'][qi_bank]:
            raise ValueError('Original query bank/source mismatch')
        arms={row['arm']:row for row in source['arms']}
        if set(arms)!=set(ARMS):raise ValueError('Incomplete intervention arms')
        vectors={};records={}
        for arm in ARMS:
            row=arms[arm];path=folder/'embeddings'/(row['id']+'.pt')
            record=torch.load(path,map_location='cpu',weights_only=True)
            if record['image_sha256']!=sha(row['path']) or record['tensor_sha256']!=token_hash(record['embedding']):
                raise ValueError('Intervention output hash mismatch')
            if not all(record.get(k)==v for k,v in row.items()):
                raise ValueError('Encoder record differs from committed manifest')
            embedding=record['embedding']
            if embedding.shape!=cached.shape or not torch.isfinite(embedding).all():
                raise ValueError('Invalid intervention embedding')
            vectors[arm]=embedding;records[arm]=record
        raw,display=rgba_and_display(arms['base']['path']);size=np.array([display.shape[1],display.shape[0]],float)
        raw_sham,display_sham=rgba_and_display(arms['sham']['path'])
        raw_equal=np.array_equal(raw,raw_sham);display_equal=np.array_equal(display,display_sham)
        if not raw_equal or not display_equal:raise ValueError('Sham does not preserve decoded base pixels')
        numeric=np.asarray(source['numeric_bbox']);blank=np.asarray(source['blank_bbox'])
        if not np.array_equal(numeric[2:]-numeric[:2],blank[2:]-blank[:2]):
            raise ValueError('Numeric/blank controls have unequal area')
        for arm in ARMS:
            _,pixels=rgba_and_display(arms[arm]['path'])
            changed=np.any(pixels!=display,axis=2);allowed=np.zeros(changed.shape,bool)
            boxes=[numeric] if arm=='numeric_mask' else [blank] if arm=='blank_control' else [numeric,blank] if arm=='both_masks' else []
            for b in boxes:allowed[b[1]:b[3],b[0]:b[2]]=True
            outside=int(np.sum(changed & ~allowed))
            if outside:raise ValueError('Pixel changes escaped committed rectangle')
            pixel_rows.append(dict(photo_id=qid,arm=arm,pixel_changes=int(changed.sum()),allowed_pixels=int(allowed.sum()),
                outside_allowed_pixels=outside,sham_raw_RGB_equal=raw_equal,sham_display_RGB_equal=display_equal,
                image_SHA256=sha(arms[arm]['path'])))
        exact=torch.equal(cached,vectors['base']);sham_exact=torch.equal(vectors['base'],vectors['sham'])
        cosine=torch.nn.functional.cosine_similarity(cached.float(),vectors['base'].float(),dim=1)
        scope='cached_query_reproduced_fixed_historical_gallery' if exact else 'new_runtime_query_fixed_historical_gallery_conditional'
        semantics_row=semantics.get(qid,{})
        source_rows.append(dict(photo_id=qid,package_id=source['package_id'],split=source['split'],outcome=source['outcome'],
            cache_provenance=source['cache_provenance'],base_cache_bitwise_equal=exact,
            base_cache_max_absolute_error=float((cached.float()-vectors['base'].float()).abs().max()),
            base_cache_cosine_mean=float(cosine.mean()),sham_base_bitwise_equal=sham_exact,
            sham_base_max_absolute_error=float((vectors['sham'].float()-vectors['base'].float()).abs().max()),
            conditional_scope=scope,proposed_text=source['proposed_text'],
            semantic_type=semantics_row.get('semantic_type','unreviewed_OCR_proposal'),
            critical_identity_relevance=semantics_row.get('critical_identity_relevance','not_verified'),
            blank_control_location=semantics_row.get('blank_control_location','not_verified'),
            semantic_review_evidence=semantics_row.get('evidence_path')))
        case=ledger[qid];positives=case['positive_reference_ids']
        wrong={}
        for kind in ['image','fusion']:
            rid=case[kind+'_strongest_wrong'];design=d['rb']['sha256'][rindex[rid]]
            wrong.setdefault(design,dict(reference_id=rid,roles=[]))['roles'].append(kind)
        all_rids=list(dict.fromkeys(positives+[v['reference_id'] for v in wrong.values()]))
        traces={};q_coordinates=grid(d['metadata']['photos'][qi_bank])
        for arm in ARMS:
            traces[arm]={}
            for rid in all_rids:
                ri_bank=rindex[rid];trace=traced_similarity(vectors[arm],d['rb']['embeddings'][ri_bank]);traces[arm][rid]=trace
                target=matrix_dir/(qid+'__'+arm+'__'+rid+'.npz')
                np.savez_compressed(target,similarity=trace['similarity'],query_valid_indices=trace['qi'],reference_valid_indices=trace['ri'])
                matrix_index.append(dict(photo_id=qid,arm=arm,reference_id=rid,path=str(target.relative_to(output)),SHA256=sha(target),
                    shape=list(trace['similarity'].shape),dtype='float32',query_tensor_sha256=records[arm]['tensor_sha256'],
                    reference_tensor_sha256=token_hash(d['rb']['embeddings'][ri_bank]),score=trace['score']))
        main_qi=traces['sham'][all_rids[0]]['qi']
        for arm in ARMS:
            for rid in all_rids:
                if not np.array_equal(main_qi,traces[arm][rid]['qi']):
                    raise ValueError('Intervention changed effective query indices; cannot use fixed-index interaction')
        weights,visual=region_weights(d['metadata']['photos'][qi_bank],source,d['boxes'][qid]['bbox'],size,main_qi)
        for positive in positives:
            for negative_info in wrong.values():
                negative=negative_info['reference_id'];pair_id=positive+'__'+negative;token_rows=[];deltas={};gaps={}
                for arm in ARMS:
                    a,b=traces[arm][positive],traces[arm][negative]
                    delta=a['maximum'].astype(np.float64)-b['maximum'].astype(np.float64);deltas[arm]=delta
                    gap=a['score']-b['score'];gaps[arm]=gap
                    error=abs(float(delta.mean())-gap)
                    if error>2e-6:raise ValueError('Mean Delta does not reproduce pair gap')
                    conservation.append(dict(photo_id=qid,pair_id=pair_id,arm=arm,mean_Delta=float(delta.mean()),pair_gap=gap,absolute_error=error))
                cached_pos=traced_similarity(cached,d['rb']['embeddings'][rindex[positive]])['score']
                cached_neg=traced_similarity(cached,d['rb']['embeddings'][rindex[negative]])['score']
                original_expected=float(d['image']['scores'][qi_bank,rindex[positive]]-d['image']['scores'][qi_bank,rindex[negative]])
                old_error=abs(cached_pos-cached_neg-original_expected);max_cache_score_error=max(max_cache_score_error,old_error)
                if old_error>2e-6:raise ValueError('Cached query does not reproduce historical candidate difference')
                for arm in ARMS:
                    a,b=traces[arm][positive],traces[arm][negative]
                    sham_a,sham_b=traces['sham'][positive],traces['sham'][negative]
                    switches=(a['winner']!=sham_a['winner'])|(b['winner']!=sham_b['winner'])
                    change=deltas[arm]-deltas['sham']
                    pair_rows.append(dict(photo_id=qid,package_id=source['package_id'],split=source['split'],outcome=source['outcome'],
                        positive=positive,negative=negative,negative_roles=negative_info['roles'],arm=arm,
                        positive_score=a['score'],negative_score=b['score'],gap=gaps[arm],edit_minus_sham_gap=gaps[arm]-gaps['sham'],
                        switched_tokens=int(switches.sum()),unchanged_match_tokens=int((~switches).sum()),valid_tokens=len(main_qi),
                        switched_token_contribution=float(change[switches].sum()/len(main_qi)),
                        unchanged_match_contribution=float(change[~switches].sum()/len(main_qi)),
                        cached_baseline_gap=original_expected,fresh_base_minus_cached_gap=gaps['base']-original_expected,
                        interaction_gap=gaps['both_masks']-gaps['numeric_mask']-gaps['blank_control']+gaps['sham'],
                        conditional_scope=scope,edited_identity_accuracy_not_computed=True,
                        semantic_type=semantics_row.get('semantic_type','unreviewed'),critical_identity_relevance=semantics_row.get('critical_identity_relevance','not_verified'),
                        blank_control_location=semantics_row.get('blank_control_location','not_verified')))
                    for region_i,region in enumerate(REGIONS):
                        w=weights[:,region_i];weighted=w*deltas[arm]
                        region_rows.append(dict(photo_id=qid,pair_id=pair_id,arm=arm,region=region,
                            soft_token_mass=float(w.sum()),overlapping_tokens=int(np.sum(w>0)),
                            positive_contribution=float(weighted.clip(min=0).sum()/len(main_qi)),
                            negative_contribution=float(weighted.clip(max=0).sum()/len(main_qi)),
                            net_contribution=float(weighted.sum()/len(main_qi)),
                            edit_minus_sham_region_contribution=float((w*change).sum()/len(main_qi)),
                            contribution_abs_mass=float(np.abs(weighted).sum()/len(main_qi)),
                            region_definition='exclusive soft patch-overlap partition; supplied subject box is additional annotation'))
                    if abs(sum(float((weights[:,j]*deltas[arm]).sum()/len(main_qi)) for j in range(5))-float(deltas[arm].mean()))>2e-6:
                        raise ValueError('Region partition does not conserve contribution')
                    feature_l2=(vectors[arm][main_qi].float()-vectors['sham'][main_qi].float()).norm(dim=1).numpy()
                    feature_cosine=torch.nn.functional.cosine_similarity(vectors[arm][main_qi].float(),vectors['sham'][main_qi].float(),dim=1).numpy()
                    positive_xy=grid(d['metadata']['references'][rindex[positive]])
                    negative_xy=grid(d['metadata']['references'][rindex[negative]])
                    for k,index in enumerate(main_qi):
                        coord=q_coordinates[index];xy=coord.tolist() if np.isfinite(coord).all() else [None,None]
                        pcoord=positive_xy[a['winner'][k]];p_xy=pcoord.tolist() if np.isfinite(pcoord).all() else [None,None]
                        ncoord=negative_xy[b['winner'][k]];n_xy=ncoord.tolist() if np.isfinite(ncoord).all() else [None,None]
                        token_rows.append(dict(arm=arm,query_index=int(index),token_type='visual' if visual[k] else 'template',x_display=xy[0],y_display=xy[1],
                            **{region+'_weight':float(weights[k,j]) for j,region in enumerate(REGIONS)},
                            positive_winner=int(a['winner'][k]),positive_x_display=p_xy[0],positive_y_display=p_xy[1],
                            positive_second=int(a['second'][k]),positive_similarity=float(a['maximum'][k]),
                            positive_top1_top2=float(a['maximum'][k]-a['second_maximum'][k]),
                            negative_winner=int(b['winner'][k]),negative_x_display=n_xy[0],negative_y_display=n_xy[1],
                            negative_second=int(b['second'][k]),negative_similarity=float(b['maximum'][k]),
                            negative_top1_top2=float(b['maximum'][k]-b['second_maximum'][k]),Delta=float(deltas[arm][k]),
                            edit_minus_sham_Delta=float(change[k]),match_switched_from_sham=bool(switches[k]),
                            positive_match_switched_from_sham=bool(a['winner'][k]!=sham_a['winner'][k]),
                            negative_match_switched_from_sham=bool(b['winner'][k]!=sham_b['winner'][k]),
                            feature_L2_from_sham=float(feature_l2[k]),feature_cosine_from_sham=float(feature_cosine[k]),
                            interaction_Delta=float(deltas['both_masks'][k]-deltas['numeric_mask'][k]-deltas['blank_control'][k]+deltas['sham'][k])))
                query_dir=output/'queries'/qid;query_dir.mkdir(parents=True,exist_ok=True)
                csvout(query_dir/(pair_id+'.tokens.csv'),token_rows)
        processed.add(qid)
        write(output/'progress.json',dict(completed_queries=len(processed),total_queries=len(sources),last=qid))
        print('Detail controls',len(processed),'/',len(sources),qid,flush=True)
    if processed!={s['photo_id'] for s in sources} or len(processed)!=len(sources):
        raise ValueError('Missing or duplicated source analysis')
    csvout(output/'sources_and_cache_parity.csv',source_rows);csvout(output/'pair_effects.csv',pair_rows)
    csvout(output/'region_contributions.csv',region_rows);csvout(output/'pixel_validation.csv',pixel_rows)
    csvout(output/'contribution_conservation.csv',conservation);write(output/'matrix_index.json',matrix_index)
    validation=dict(queries=len(processed),independent_packages=len({s['package_id'] for s in sources}),encoding_records=len(manifest),
        cache_exact_queries=sum(s['base_cache_bitwise_equal'] for s in source_rows),
        sham_base_exact_queries=sum(s['sham_base_bitwise_equal'] for s in source_rows),
        numeric_semantic_review_queries=len(semantics),matrices=len(matrix_index),
        max_cached_score_gap_replay_error=max_cache_score_error,
        max_Delta_conservation_error=max(r['absolute_error'] for r in conservation),
        source_coverage_exact=processed=={s['photo_id'] for s in sources},
        all_pixels_within_declared_rectangles=all(r['outside_allowed_pixels']==0 for r in pixel_rows),
        all_decoded_shams_equal_base=all(r['sham_raw_RGB_equal'] and r['sham_display_RGB_equal'] for r in pixel_rows),
        scope='original frozen gallery candidate pairs; no fresh-gallery accuracy or edited identity labels',
        subject_box_audit_state='historical committed boxes; user confirmation pending, not newly verified truth',
        selection_scope='12 identities, first 3 per original four-cell outcome; no edited response selection')
    write(output/'validation.json',validation)
    write(output/'input_manifest.json',dict(code_sha256=sha(Path(__file__)),sources_sha256=sha(folder/'sources.json'),
        encoding_manifest_sha256=sha(folder/'encoding_manifest.json'),semantic_verification_sha256=sha(semantic_path) if semantic_path.exists() else None,
        original_query_bank_sha256=sha(ROOT/'cache/image_embeddings/colnomic7b/photos.pt'),
        original_reference_bank_sha256=sha(ROOT/'cache/image_embeddings/colnomic7b/references.pt'),
        subject_box_snapshot=str(subject_box_snapshot),subject_box_snapshot_sha256=sha(subject_box_snapshot),
        current_subject_boxes_not_used_for_historical_partition=True))
    report=['# 数字区域遮挡与等面积编辑对照','',
        f"预先按四类原结果分层的 {len(processed)} 个独立身份；所有五臂均分析，未依据编辑后响应筛选。",
        f"原Query缓存逐位复现 {validation['cache_exact_queries']}/{len(processed)}；sham与fresh base逐位一致 {validation['sham_base_exact_queries']}/{len(processed)}。",
        '','分析只比较重新编码Query与原冻结图库候选的条件响应。非精确复现的原图独立标注new-runtime，未报告新运行时完整图库准确率或编辑后身份准确率。',
        '数字区域由OCR提议，独立视觉审核区分包装规格、公共用药说明与未确认的身份重要性。等面积空白区位置、颜色纹理仍是竞争混杂；桌面背景对照单独标记。',
        '','pair_effects.csv 保存正负差额、edit-minus-sham、匹配切换与未切换的贡献以及both-minus-numeric-minus-blank-plus-sham交互；queries/保存所有逐Token记录；region_contributions.csv采用归一化互斥软区域并校验守恒。',
        '','图像遮挡后重新编码、冻结Token删除和更换匹配规则是不同实验。本结果支持在该编辑与固定图库条件下的响应，不能直接断言数字是原始错误的唯一原因。']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(validation,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--subject-boxes',type=Path,required=True,help='immutable snapshot used during image preparation')
    args=parser.parse_args()
    torch.set_num_threads(2)
    run(args.input.resolve(),args.subject_boxes.resolve())


if __name__=='__main__':main()
