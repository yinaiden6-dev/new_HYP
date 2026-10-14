"""CPU-only cross-source layout matrices and conserved MaxSim gap decomposition.

This is a small manually edited same-content pilot with zero test identities.
It does not train a probe or assign natural identity accuracy to edited images.
"""
from __future__ import annotations
import argparse
import html
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from scripts.colnomic_mechanism.common import ROOT, read, write, rows, csvout, sha, workspace_path
from scripts.colnomic_mechanism.controlled_crossview_layouts import read_photo, ARMS
from scripts.colnomic_mechanism.symmetric_layouts import variants
from scripts.colnomic_mechanism.geometry import grid
from scripts.encode_colnomic_images import token_hash

BASE=ROOT/'results/colnomic_full_audit/20261013_resume/controlled_crossview_layouts'


def write_report(folder, summary, contrasts):
    """Render stored contrasts without recomputing or changing their scores."""
    out=folder/'analysis';groups=[]
    for pair in sorted({r['pair_id'] for r in contrasts}):
        subset=[r for r in contrasts if r['pair_id']==pair]
        groups.append(dict(pair_id=pair,relations=len(subset),positive_gaps=sum(float(r['predefined_layout_gap'])>0 for r in subset),
            mean_layout_gap=float(np.mean([float(r['predefined_layout_gap']) for r in subset])),
            mean_edit_minus_sham_gap=float(np.mean([float(r['edit_minus_sham_layout_gap']) for r in subset])),
            mean_layout_minus_editor_gap=float(np.mean([float(r['layout_minus_editor_gap']) for r in subset])),
            mean_fixed_winner_feature_gap_change=float(np.mean([float(r['fixed_winner_feature_gap_change']) for r in subset])),
            mean_argmax_switch_gap_change=float(np.mean([float(r['argmax_switch_gap_change']) for r in subset]))))
    csvout(out/'identity_pilot_summary.csv',groups)
    lines=['# 跨原图同内容布局匹配 pilot','',
        f"全130身份按预注册质量规则扫描，4对进入实际视觉核验；3对保留，1对因OCR框包含两个词而排除。保留{summary['qualified_identities']}身份、{summary['source_photos']}张不同原照片、{summary['encoded_arms']}个编码材料（2 train、1 val、0 test）。",
        '材料沿用固定OCR、均匀纸面、±0.35词宽和像素守恒条件。两原图的物理方向由共同GT的已接受单应矩阵映射；package_030 的B相机方向反转，左右标签据此交换。跨原图物理位移大小差异如实记录，未按模型响应选择来源或放宽规则。','',
        f"12个预定义跨原图比较中，正布局差额为{summary['positive_layout_gap_relations']}/12。平均 edit-minus-sham 布局差额 {summary['mean_edit_minus_sham_gap']:.8g}，平均 layout-minus-editor 差额 {summary['mean_layout_minus_editor_gap']:.8g}。差额正负混合；这批材料不能支持稳定的条件布局匹配结论。",
        '这些数是3身份内的重复描述性比较，不能当作12个独立样本、自然身份检索精度或test性能；未训练Probe。','',
        '|身份|比较数|正差额数|平均布局差额|平均edit−sham|平均layout−editor|',
        '|---|---|---|---|---|---|']
    lines += [f"|{r['pair_id']}|{r['relations']}|{r['positive_gaps']}|{r['mean_layout_gap']:.8g}|{r['mean_edit_minus_sham_gap']:.8g}|{r['mean_layout_minus_editor_gap']:.8g}|" for r in groups]
    lines += ['',
        '完整36×36 forward mean MaxSim矩阵保留所有原始、±、sham和editor-only跨来源组合。条件比较只使用预注册的不同原图A/B对应方向与相反方向。self/同原图矩阵格仅供诊断，不进入上述差额。','',
        'edit−sham 分解为固定sham赢家的feature变化和argmax切换，二者逐token守恒。词框/编辑带汇总仅使用token中心位置代理，不能解读成精确感受野或语义定位。',
        f"最大布局差额分解误差 {summary['maximum_gap_decomposition_error']:.3g}，最大逐token误差 {summary['maximum_token_decomposition_error']:.3g}。独立重放{summary['independent_pixel_arms_verified']}个像素材料（包含被排除身份），逐像素、字形RGB、编辑带外和±/sham色直方图断言均通过。编码complete清单SHA及每个tensor/image SHA已核对。",'',
        '[完整矩阵](complete_similarity_matrix.csv) · [12个预定义关系](predefined_layout_contrasts.csv) · [feature/argmax分解](feature_argmax_decomposition.csv) · [身份汇总](identity_pilot_summary.csv) · [验收](validation.json) · [像素重放](pixel_replay_validation.json)',
        '', '[预注册方案](../protocol.json) · [全130身份盘点](../identity_inventory.json) · [实际视觉核验](../manual_reviews.json) · [编码清单](../encoding_manifest.json)']
    report='\n'.join(lines)+'\n';(out/'REPORT.md').write_text(report)
    links=''.join(f"<li><a href='{html.escape(p.name)}'>{html.escape(p.name)}</a></li>" for p in sorted(out.iterdir()) if p.suffix in ('.csv','.json','.npz','.md'))
    (out/'index.html').write_text("<!doctype html><html lang='zh'><meta charset='utf-8'><title>跨原图布局pilot</title><style>body{font:16px sans-serif;margin:2em;max-width:1200px}pre{white-space:pre-wrap}li{margin:7px 0}</style><h1>跨原图布局pilot</h1><ul>"+links+'</ul><pre>'+html.escape(report)+'</pre></html>')
    write(out/'report_provenance.json',dict(matrix_computation_script_sha256=summary['script_sha256'],
        matrix_computation_script_snapshot='source_code_at_matrix_replay.py' if (out/'source_code_at_matrix_replay.py').exists() else None,
        report_script_sha256=sha(Path(__file__)),summary_sha256=sha(out/'summary.json'),
        contrasts_sha256=sha(out/'predefined_layout_contrasts.csv'),reported_values_loaded_from_stored_results=True))


def valid_tokens(embedding):
    value=embedding.detach().cpu().float()
    if value.ndim!=2 or value.shape[1]!=128 or not torch.isfinite(value).all():
        raise ValueError('Invalid encoder token matrix')
    indices=torch.nonzero(value.norm(dim=1)>1e-8).flatten()
    if not len(indices):raise ValueError('Empty valid tokens')
    return value[indices],indices.numpy()


def maxsim(query,reference):
    return (query@reference.T).amax(1).double().mean().item()


def decompose_query_change(sham,edited,reference):
    if sham.shape!=edited.shape:raise ValueError('Sham/edit query token shape mismatch')
    base_dot=sham@reference.T;edit_dot=edited@reference.T
    w0=base_dot.argmax(1);w1=edit_dot.argmax(1)
    indices=torch.arange(len(sham))
    base=base_dot[indices,w0].double();new=edit_dot[indices,w1].double()
    fixed_feature=(edit_dot[indices,w0]-base_dot[indices,w0]).double()
    switch=(edit_dot[indices,w1]-edit_dot[indices,w0]).double()
    total=new-base;error=total-fixed_feature-switch
    return dict(base_score=base.mean().item(),edited_score=new.mean().item(),delta=total.mean().item(),
        fixed_sham_winner_feature_delta=fixed_feature.mean().item(),argmax_switch_delta=switch.mean().item(),
        switch_tokens=int((w0!=w1).sum()),query_tokens=len(sham),
        conservation_error=abs((total-fixed_feature-switch).mean().item()),max_token_conservation_error=error.abs().max().item(),
        per_token=dict(base=base.numpy(),edited=new.numpy(),delta=total.numpy(),fixed_feature=fixed_feature.numpy(),
            switch=switch.numpy(),base_winner=w0.numpy(),edited_winner=w1.numpy()))


def validate_pixels(folder):
    reviewed=read(folder/'sources_reviewed.json')
    metadata={r['id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/token_metadata.json')['photos']}
    boxes=read(ROOT/'results/colnomic_full_audit/20261013_resume/manual_box_review/subject_boxes_after_user_review.json')['records']
    ocr=read(ROOT/'cache/ocr/photos.json');out=[];references={r['reference_id']:r for r in rows(ROOT/'dataset/lists/references.csv')}
    reference_evidence=[]
    for pair in reviewed:
        rid=pair['common_positive_reference'];ref=references[rid];reference_path=ROOT/'dataset'/ref['path']
        actual=sha(reference_path)
        if actual!=ref['sha256']:raise ValueError('Common GT/reference image SHA changed')
        reference_evidence.append(dict(pair_id=pair['pair_id'],reference_id=rid,path=str(reference_path.relative_to(ROOT)),
            declared_sha256=ref['sha256'],actual_sha256=actual,hash_matching=actual==ref['sha256'],
            scope='same identity positive reference and OCR geometry assistance; not edited identity label'))
        for role in ('A','B'):
            source=pair[role];qid=source['photo_id'];spec=metadata[qid]
            original=ROOT/source['original_path']
            if sha(original)!=source['image_sha256']:raise ValueError('Original source changed')
            array,raw=read_photo(original,spec);bb=tuple(source['bbox_display'])
            expected,step,allowed,glyph,paper=variants(array,bb,np.asarray(boxes[qid]['bbox']),ocr[qid]['lines'])
            if step!=source['shift_pixels']:raise ValueError('Recorded source step changed')
            sham_colors,sham_counts=np.unique(expected['sham'][allowed],axis=0,return_counts=True)
            for arm in [r for r in pair['arms'] if r['role']==role]:
                path=workspace_path(arm['path']);actual_sha=sha(path)
                if actual_sha!=arm['image_sha256']:raise ValueError('Arm image SHA changed')
                saved,_=read_photo(path,spec);target=array if arm['arm']=='base' else expected[arm['arm']]
                exact=bool(np.array_equal(saved,target));outside=bool(np.array_equal(saved[~allowed],array[~allowed]))
                offset=-step if arm['arm']=='left' else step if arm['arm']=='right' else 0
                glyph_exact=bool(np.array_equal(saved[bb[1]:bb[3],bb[0]+offset:bb[2]+offset],glyph))
                histogram_equal=None
                if arm['arm'] in ('left','right'):
                    colors,counts=np.unique(saved[allowed],axis=0,return_counts=True)
                    histogram_equal=bool(np.array_equal(colors,sham_colors) and np.array_equal(counts,sham_counts))
                if not exact or not outside or not glyph_exact or histogram_equal is False:
                    raise ValueError('Independent saved-pixel replay failed')
                out.append(dict(pair_id=pair['pair_id'],role=role,arm=arm['arm'],source_photo=qid,image_sha256=actual_sha,
                    saved_pixel_replay_exact=exact,glyph_rectangle_RGB_conserved=glyph_exact,outside_band_exact=outside,
                    sham_shift_RGB_histogram_equal=histogram_equal,source_exif_orientation=spec['exif_orientation'],
                    editor_only_original_word_position_unchanged=glyph_exact if arm['arm'].startswith('editor_') else None,
                    visually_accepted=pair['visual_review_status']=='confirmed_same_word_same_physical_face'))
    output=folder/'analysis';output.mkdir(exist_ok=True);write(output/'pixel_replay_validation.json',out)
    write(output/'reference_input_evidence.json',reference_evidence)
    return out


def analyze(folder,embedding_dir):
    manifest=read(folder/'encoding_manifest.json');expected_relations=read(folder/'expected_relations.json')
    complete=read(embedding_dir/'complete.json')
    if complete.get('status')!='complete' or complete.get('records')!=len(manifest) or complete.get('manifest_sha256')!=sha(folder/'encoding_manifest.json'):
        raise ValueError('Encoder completion/manifest SHA disagreement')
    pixel=validate_pixels(folder);out=folder/'analysis'
    sources={r['pair_id']:r for r in read(folder/'sources_reviewed.json') if r['visual_review_status'].startswith('confirmed')}
    metadata={r['id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/token_metadata.json')['photos']}
    vectors={};indices={};embedding_records={};inputs={};byid={r['id']:r for r in manifest};lookup={(r['pair_id'],r['role'],r['arm']):r['id'] for r in manifest}
    for row in manifest:
        path=embedding_dir/(row['id']+'.pt');record=torch.load(path,map_location='cpu',weights_only=True)
        if any(record.get(k)!=v for k,v in row.items()):raise ValueError('Encoder output/manifest row disagreement')
        if record['image_sha256']!=sha(workspace_path(row['path'])) or token_hash(record['embedding'])!=record['tensor_sha256']:
            raise ValueError('Encoder image/tensor SHA mismatch')
        vectors[row['id']],indices[row['id']]=valid_tokens(record['embedding']);embedding_records[row['id']]=record
        inputs[str(path.relative_to(ROOT))]=sha(path)
    ids=[r['id'] for r in manifest];scores=np.empty((len(ids),len(ids)),np.float64);matrix=[]
    for i,q in enumerate(ids):
        for j,r in enumerate(ids):
            score=maxsim(vectors[q],vectors[r]);scores[i,j]=score
            qr,rr=byid[q],byid[r]
            matrix.append(dict(query_id=q,candidate_id=r,forward_mean_MaxSim=score,query_pair=qr['pair_id'],candidate_pair=rr['pair_id'],
                query_role=qr['role'],candidate_role=rr['role'],query_arm=qr['arm'],candidate_arm=rr['arm'],
                same_source_image=qr['source_id']==rr['source_id'],same_identity=qr['package_id']==rr['package_id'],
                predefined_paired_cross_source=qr['pair_id']==rr['pair_id'] and qr['role']!=rr['role'],
                evaluation_use='only predefined cross-source pairs enter layout contrasts; self/within-view cells diagnostic only'))
    csvout(out/'complete_similarity_matrix.csv',matrix);np.savez_compressed(out/'complete_similarity_matrix.npz',score=scores,ids=np.asarray(ids))
    score_lookup={(r['query_id'],r['candidate_id']):r['forward_mean_MaxSim'] for r in matrix}
    contrasts=[];decompositions=[];token_rows=[];region_rows=[]
    for relation in expected_relations:
        q=relation['query_id'];pos=relation['consistent_id'];neg=relation['inconsistent_id'];row=byid[q]
        sham=lookup[row['pair_id'],row['role'],'sham'];base=lookup[row['pair_id'],row['role'],'base']
        editor=lookup[row['pair_id'],row['role'],'editor_'+row['arm']]
        if byid[pos]['source_id']==row['source_id'] or byid[neg]['source_id']==row['source_id']:
            raise ValueError('Same-view self shortcut entered layout relation')
        gaps={arm:score_lookup[query,pos]-score_lookup[query,neg] for arm,query in [('edit',q),('sham',sham),('base',base),('editor_only',editor)]}
        contrast=dict(**relation,query_role=row['role'],query_image_arm=row['arm'],query_physical_direction=row['physical_direction'],
            consistent_score=score_lookup[q,pos],inconsistent_score=score_lookup[q,neg],
            predefined_layout_gap=gaps['edit'],sham_layout_gap=gaps['sham'],fresh_base_layout_gap=gaps['base'],
            editor_only_layout_gap=gaps['editor_only'],edit_minus_sham_layout_gap=gaps['edit']-gaps['sham'],
            editor_minus_sham_layout_gap=gaps['editor_only']-gaps['sham'],layout_minus_editor_gap=gaps['edit']-gaps['editor_only'],
            evidence_scope='deterministic same-content cross-photo pilot; zero test identities; not natural retrieval accuracy')
        feature_gap=switch_gap=0.
        for change,changed_id in [('glyph_edit',q),('editor_only',editor)]:
            if not np.array_equal(indices[sham],indices[changed_id]):raise ValueError('Token axes differ between same-source arms')
            for candidate_role,rid in [('consistent',pos),('inconsistent',neg)]:
                d=decompose_query_change(vectors[sham],vectors[changed_id],vectors[rid]);detail=d.pop('per_token')
                if d['conservation_error']>2e-6 or d['max_token_conservation_error']>2e-6:raise ValueError('Feature/switch decomposition failed')
                if abs(d['edited_score']-score_lookup[changed_id,rid])>2e-6 or abs(d['base_score']-score_lookup[sham,rid])>2e-6:
                    raise ValueError('Decomposed score differs from full matrix')
                decompositions.append(dict(pair_id=row['pair_id'],query_id=changed_id,sham_id=sham,candidate_id=rid,
                    query_change=change,candidate_relation=candidate_role,**d))
                if change=='glyph_edit':
                    sign=1 if candidate_role=='consistent' else -1
                    feature_gap+=sign*d['fixed_sham_winner_feature_delta'];switch_gap+=sign*d['argmax_switch_delta']
                source=sources[row['pair_id']][row['role']];spec=metadata[row['source_id']]
                xy=grid(spec)[indices[sham]]*np.array([source['display_width'],source['display_height']])
                bb=source['bbox_display'];step=source['shift_pixels'];offset=-step if row['arm']=='left' and change=='glyph_edit' else step if row['arm']=='right' and change=='glyph_edit' else 0
                regions=[]
                for k,tok in enumerate(indices[sham]):
                    x,y=xy[k]
                    region='nonvisual' if not np.isfinite([x,y]).all() else 'original_or_shifted_word_box_center_proxy' if bb[1]<=y<bb[3] and (bb[0]<=x<bb[2] or bb[0]+offset<=x<bb[2]+offset) else 'other_edit_band_center_proxy' if bb[1]<=y<bb[3] and bb[0]-step<=x<bb[2]+step else 'outside_edit_band_center_proxy'
                    regions.append(region)
                    token_rows.append(dict(pair_id=row['pair_id'],query_id=changed_id,sham_id=sham,candidate_id=rid,query_change=change,candidate_relation=candidate_role,
                        query_token=int(tok),sham_candidate_winner_token=int(indices[rid][detail['base_winner'][k]]),
                        changed_candidate_winner_token=int(indices[rid][detail['edited_winner'][k]]),region=region,
                        score_delta=float(detail['delta'][k]),fixed_winner_feature_delta=float(detail['fixed_feature'][k]),
                        argmax_switch_delta=float(detail['switch'][k]),mean_score_contribution=float(detail['delta'][k]/len(indices[sham]))))
                for region in sorted(set(regions)):
                    mask=np.asarray(regions)==region
                    region_rows.append(dict(pair_id=row['pair_id'],query_id=changed_id,candidate_id=rid,query_change=change,candidate_relation=candidate_role,region=region,
                        tokens=int(mask.sum()),score_delta=float(detail['delta'][mask].sum()/len(mask)),
                        fixed_feature_delta=float(detail['fixed_feature'][mask].sum()/len(mask)),argmax_switch_delta=float(detail['switch'][mask].sum()/len(mask)),
                        limitation='token center proxy only; not exact receptive field or semantic localization'))
        contrast['fixed_winner_feature_gap_change']=feature_gap;contrast['argmax_switch_gap_change']=switch_gap
        contrast['gap_decomposition_error']=abs(contrast['edit_minus_sham_layout_gap']-feature_gap-switch_gap)
        if contrast['gap_decomposition_error']>2e-6:raise ValueError('Contrast gap decomposition fails')
        contrasts.append(contrast)
    csvout(out/'predefined_layout_contrasts.csv',contrasts);csvout(out/'feature_argmax_decomposition.csv',decompositions)
    csvout(out/'per_query_token_changes.csv',token_rows);csvout(out/'word_band_context_aggregation.csv',region_rows)
    summary=dict(schema='colnomic.crossview_layout_analysis.v1',qualified_identities=len(sources),source_photos=len({r['source_id'] for r in manifest}),encoded_arms=len(manifest),
        source_identities_by_split=dict(Counter(r['split'] for r in sources.values())),test_identities=sum(r['split']=='test' for r in sources.values()),
        full_matrix_shape=list(scores.shape),predefined_cross_source_relations=len(contrasts),
        positive_layout_gap_relations=sum(r['predefined_layout_gap']>0 for r in contrasts),
        mean_edit_minus_sham_gap=float(np.mean([r['edit_minus_sham_layout_gap'] for r in contrasts])),
        mean_layout_minus_editor_gap=float(np.mean([r['layout_minus_editor_gap'] for r in contrasts])),
        maximum_gap_decomposition_error=max(r['gap_decomposition_error'] for r in contrasts),
        maximum_token_decomposition_error=max(r['max_token_conservation_error'] for r in decompositions),
        independent_pixel_arms_verified=len(pixel),encoder_complete_sha256=sha(embedding_dir/'complete.json'),
        encoding_manifest_sha256=sha(folder/'encoding_manifest.json'),script_sha256=sha(Path(__file__)),embedding_inputs_sha256=inputs,
        statistical_scope='3 identity pilot/2 train+1 val/0test; descriptive paired traces only; no trained probe/no held-test performance/no natural identity accuracy')
    write(out/'summary.json',summary)
    write(out/'validation.json',dict(matrix_finite=bool(np.isfinite(scores).all()),matrix_shape_exact=scores.shape==(len(manifest),len(manifest)),
        all_relations_cross_source=all(byid[r['query_id']]['source_id']!=byid[r['consistent_id']]['source_id'] and byid[r['query_id']]['source_id']!=byid[r['inconsistent_id']]['source_id'] for r in expected_relations),
        all_feature_switch_conservation=all(r['conservation_error']<=2e-6 and r['max_token_conservation_error']<=2e-6 for r in decompositions),
        all_gap_conservation=all(r['gap_decomposition_error']<=2e-6 for r in contrasts),
        embedding_input_hashes_unchanged=all(sha(ROOT/path)==digest for path,digest in inputs.items()),
        encoder_manifest_hash_matches=complete['manifest_sha256']==sha(folder/'encoding_manifest.json')))
    write_report(folder,summary,contrasts)
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',type=Path,default=BASE);parser.add_argument('--embeddings',type=Path);parser.add_argument('--validate-pixels-only',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    if args.validate_pixels_only:print('Independent pixel arms verified',len(validate_pixels(args.folder.resolve())))
    else:analyze(args.folder.resolve(),(args.embeddings or args.folder/'embeddings').resolve())
