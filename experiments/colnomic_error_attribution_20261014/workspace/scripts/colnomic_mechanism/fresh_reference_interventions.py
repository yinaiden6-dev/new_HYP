"""Fresh same-runtime candidate replay for the frozen twelve detail interventions.

Candidate membership is the union of the already frozen 145 matrix records.
This branch does not estimate full-gallery or edited-image identity accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from scripts.colnomic_mechanism.common import ROOT, read, write, rows, csvout, sha, workspace_path
from scripts.colnomic_mechanism.analyze_detail_controls import ARMS, REGIONS, region_weights
from scripts.colnomic_mechanism.display_pixels import display_image
from scripts.encode_colnomic_images import token_hash

BASE = ROOT / 'results/colnomic_full_audit/20261013_resume'
OLD = BASE / 'detail_interventions'
FOLDER = BASE / 'fresh_reference_interventions'
SCOPE = 'All preselected detail candidates with fresh same-runtime references; restricted conditional responses, no full-gallery or edited-label accuracy'


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def trace(q, r):
    qi = torch.where(q.float().norm(dim=1) > 1e-8)[0].numpy()
    ri = torch.where(r.float().norm(dim=1) > 1e-8)[0].numpy()
    dot = (q[qi].float() @ r[ri].float().T).numpy()
    assert len(qi) and len(ri) >= 2 and np.isfinite(dot).all()
    order = np.argsort(-dot, axis=1, kind='stable')[:, :2]
    top = dot[np.arange(len(qi)), order[:, 0]]
    second = dot[np.arange(len(qi)), order[:, 1]]
    return dict(qi=qi, ri=ri, dot=dot, winner=ri[order[:, 0]], runner_up=ri[order[:, 1]],
                top=top, second=second, score=float(top.mean()))


def load_inputs():
    """Check frozen selection, original bytes, all tensor receipts and runtime."""
    selection = read(FOLDER / 'candidate_selection.json')
    old_index = read(OLD / 'analysis/matrix_index.json')
    assert sha(OLD / 'analysis/matrix_index.json') == selection['old_matrix_index_sha256']
    assert sha(OLD / 'analysis/pair_effects.csv') == selection['old_pair_effects_sha256']
    assert selection['reference_ids'] == sorted({x['reference_id'] for x in old_index})
    sources = read(OLD / 'sources.json')
    assert len(sources) == 12 and sum(len(s['arms']) for s in sources) == 60
    assert selection['per_query_reference_ids'] == {
        s['photo_id']: sorted({x['reference_id'] for x in old_index if x['photo_id'] == s['photo_id']}) for s in sources}
    manifest = read(FOLDER / 'encoding_manifest.json')
    complete = read(FOLDER / 'embeddings/complete.json')
    assert complete['status'] == 'complete' and complete['records'] == len(manifest) == 40
    assert complete['manifest_sha256'] == sha(FOLDER / 'encoding_manifest.json')
    a = read(OLD / 'embeddings/runtime.json'); b = read(FOLDER / 'embeddings/runtime.json')
    keys = ['packages', 'model_config', 'processor_backend',
            'optimized_encoder_sha256', 'numeric_encoder_sha256', 'GPU_memory_caps_fraction']
    runtime_checks = {k: a[k] == b[k] for k in keys}
    runtime_checks['residency_arithmetic'] = a['residency']['arithmetic'] == b['residency']['arithmetic']
    runtime_checks['optimized_encoder_actual_SHA'] = sha(Path(__file__).with_name('optimized_encoder.py')) == b['optimized_encoder_sha256']
    runtime_checks['numeric_encoder_actual_SHA'] = sha(Path(__file__).with_name('offloaded_encoder.py')) == b['numeric_encoder_sha256']
    assert all(runtime_checks.values()), runtime_checks
    expected_runtime = a['packages']; hashes = {}; new = {}; vectors = {}; record_hashes = {}
    for row in manifest:
        p = workspace_path(row['path'], must_exist=True)
        assert sha(p) == row['image_sha256'] == row['image_SHA256']
        file = FOLDER / 'embeddings' / (row['id'] + '.pt')
        record = torch.load(file, map_location='cpu', weights_only=True)
        assert all(record.get(k) == v for k, v in row.items())
        assert record['runtime'] == expected_runtime and record['processor_backend'] == a['processor_backend']
        assert record['image_sha256'] == sha(p) and record['tensor_sha256'] == token_hash(record['embedding'])
        assert record['embedding'].ndim == 2 and record['embedding'].shape[1] == 128 and torch.isfinite(record['embedding']).all()
        new[row['id']] = record; hashes[str(file.relative_to(ROOT))] = sha(file)
    parity = []
    for source in sources:
        qid = source['photo_id']; vectors[qid] = {}
        for row in source['arms']:
            p = workspace_path(row['path'], must_exist=True); assert sha(p) == row['image_SHA256']
            file = OLD / 'embeddings' / (row['id'] + '.pt')
            record = torch.load(file, map_location='cpu', weights_only=True)
            assert all(record.get(k) == v for k, v in row.items())
            assert record['runtime'] == expected_runtime and record['processor_backend'] == a['processor_backend']
            assert record['image_sha256'] == sha(p) and record['tensor_sha256'] == token_hash(record['embedding'])
            vectors[qid][row['arm']] = record['embedding']; record_hashes[(qid, row['arm'])] = record['tensor_sha256']
            hashes[str(file.relative_to(ROOT))] = sha(file)
        rerun = new[qid + '_rerun_base']['embedding']
        exact = torch.equal(rerun, vectors[qid]['base'])
        assert exact, 'New Query base differs: all edited arms must be reencoded before same-runtime interpretation'
        assert torch.equal(vectors[qid]['base'], vectors[qid]['sham'])
        parity.append(dict(photo_id=qid,rerun_base_bitwise_exact=exact,
                           tensor_SHA256=token_hash(rerun),max_absolute_error=0.0))
    rb = torch.load(ROOT / 'cache/image_embeddings/colnomic7b/references.pt', weights_only=True, mmap=True, map_location='cpu')
    rindex = {rid: i for i, rid in enumerate(rb['ids'])}
    metadata = read(ROOT / 'results/token_mechanism_analysis/F03/reproduction/token_metadata.json')
    boxes = read(BASE / 'manual_box_review/subject_boxes_before_user_review.json')['records']
    reference_parity = []
    for rid in selection['reference_ids']:
        i = rindex[rid]; old = rb['embeddings'][i]; fresh = new[rid]['embedding']
        assert new[rid]['image_sha256'] == rb['sha256'][i]
        same = old.shape == fresh.shape
        reference_parity.append(dict(reference_id=rid,same_shape=same,old_shape=list(old.shape),fresh_shape=list(fresh.shape),
            frozen_tensor_SHA256=token_hash(old),fresh_tensor_SHA256=token_hash(fresh),
            bitwise_equal=bool(same and torch.equal(old,fresh)),
            max_absolute_error=float((old.float()-fresh.float()).abs().max()) if same else None,
            mean_token_cosine=float(torch.nn.functional.cosine_similarity(old.float(),fresh.float(),dim=1).mean()) if same else None,
            index_comparison_scope='same ordinal Token index axis; physical-grid equivalence unverified' if same else 'ordinal indices incomparable across reference layouts',
            physical_lattice_equivalence_verified=False))
    for file in [FOLDER / 'encoding_manifest.json', FOLDER / 'candidate_selection.json', FOLDER / 'embeddings/complete.json',
                 FOLDER / 'embeddings/runtime.json', OLD / 'sources.json', OLD / 'encoding_manifest.json', OLD / 'embeddings/runtime.json',
                 OLD / 'analysis/matrix_index.json', OLD / 'analysis/pair_effects.csv',
                 BASE / 'manual_box_review/subject_boxes_before_user_review.json',
                 ROOT/'results/token_mechanism_analysis/F03/reproduction/token_metadata.json',
                 ROOT/'cache/image_embeddings/colnomic7b/references.pt']:
        hashes[str(file.relative_to(ROOT))] = sha(file)
    return dict(selection=selection,sources=sources,manifest=manifest,old_index=old_index,new=new,vectors=vectors,
                tensor_hashes=record_hashes,hashes=hashes,rb=rb,rindex=rindex,metadata=metadata,boxes=boxes,
                runtime_checks=runtime_checks,query_parity=parity,reference_parity=reference_parity,
                worker_provenance=dict(old_worker_SHA256=a['worker_sha256'],new_worker_SHA256=b['worker_sha256'],
                    same_worker_source=a['worker_sha256']==b['worker_sha256'],
                    source_difference_reason='Historical worker source snapshot unavailable; cause of difference not established',
                    compatibility_basis='Equal runtime/config/backend/numeric encoder source plus twelve actual bitwise-equal Query reruns'))


def analyze(output):
    assert not output.exists() or not any(output.iterdir()), 'Use a new empty output directory'
    output.mkdir(parents=True, exist_ok=True)
    d = load_inputs(); matrix_index=[]; pair_rows=[]; region_rows=[]; comparisons=[]; interactions=[]
    matrix_lookup={(x['photo_id'],x['arm'],x['reference_id']):x for x in d['old_index']}
    frozen_pairs=rows(OLD/'analysis/pair_effects.csv')
    pair_lookup={(x['photo_id'],x['positive'],x['negative'],x['arm']):x for x in frozen_pairs}
    maximum_original_error=0.;maximum_conservation_error=0.
    for source in d['sources']:
        qid=source['photo_id'];candidates=d['selection']['per_query_reference_ids'][qid]
        traces={bank:{arm:{} for arm in ARMS} for bank in ['frozen','fresh']}
        for bank in traces:
            for arm in ARMS:
                for rid in candidates:
                    r=(d['rb']['embeddings'][d['rindex'][rid]] if bank=='frozen' else d['new'][rid]['embedding'])
                    t=trace(d['vectors'][qid][arm],r);traces[bank][arm][rid]=t
                    if bank=='frozen':
                        old=matrix_lookup[(qid,arm,rid)];file=workspace_path(OLD/'analysis'/old['path'],must_exist=True)
                        assert old['query_tensor_sha256']==d['tensor_hashes'][(qid,arm)] and old['reference_tensor_sha256']==token_hash(r)
                        assert sha(file)==old['SHA256']
                        with np.load(file) as saved:
                            err=float(np.abs(t['dot']-saved['similarity']).max())
                            assert np.array_equal(t['qi'],saved['query_valid_indices']) and np.array_equal(t['ri'],saved['reference_valid_indices'])
                        maximum_original_error=max(maximum_original_error,err);assert err<=2e-6
                    else:
                        file=output/'matrices'/(qid+'__'+arm+'__'+rid+'.npz');file.parent.mkdir(exist_ok=True)
                        np.savez_compressed(file,similarity=t['dot'],query_valid_indices=t['qi'],reference_valid_indices=t['ri'],
                            winner_indices=t['winner'],runner_up_indices=t['runner_up'],top1=t['top'],top2=t['second'])
                    matrix_index.append(dict(photo_id=qid,arm=arm,reference_id=rid,bank=bank,
                        path=str(file.relative_to(ROOT)) if bank=='frozen' else str(file.relative_to(output)),
                        path_scope='workspace' if bank=='frozen' else 'analysis_output',SHA256=sha(file),
                        similarity_SHA256=array_hash(t['dot']),query_indices_SHA256=array_hash(t['qi']),reference_indices_SHA256=array_hash(t['ri']),
                        shape=list(t['dot'].shape),dtype=str(t['dot'].dtype),query_tensor_SHA256=d['tensor_hashes'][(qid,arm)],
                        reference_tensor_SHA256=token_hash(r),score=t['score']))
        qi=traces['fresh']['sham'][candidates[0]]['qi']
        assert all(np.array_equal(qi,t['qi']) for b in traces.values() for a in b.values() for t in a.values())
        spec=d['metadata']['photos'][source['cache_row']]
        assert d['vectors'][qid]['base'].shape[0]==spec['layout']['total']
        with Image.open(workspace_path(source['original_path'],must_exist=True)) as image:
            size=np.array(display_image(image).size,float)
        weights,visual=region_weights(spec,source,d['boxes'][qid]['bbox'],size,qi)
        pairs=sorted({(x['positive'],x['negative']) for x in frozen_pairs if x['photo_id']==qid})
        for positive,negative in pairs:
            pair=positive+'__'+negative
            deltas={bank:{arm:traces[bank][arm][positive]['top'].astype(float)-traces[bank][arm][negative]['top'].astype(float) for arm in ARMS} for bank in traces}
            gaps={bank:{arm:traces[bank][arm][positive]['score']-traces[bank][arm][negative]['score'] for arm in ARMS} for bank in traces}
            for bank in traces:
                tokens=[];interaction=deltas[bank]['both_masks']-deltas[bank]['numeric_mask']-deltas[bank]['blank_control']+deltas[bank]['sham']
                for arm in ARMS:
                    a=traces[bank][arm][positive];b=traces[bank][arm][negative];sa=traces[bank]['sham'][positive];sb=traces[bank]['sham'][negative]
                    delta=deltas[bank][arm];change=delta-deltas[bank]['sham']
                    switch=(a['winner']!=sa['winner'])|(b['winner']!=sb['winner'])
                    err=abs(float(delta.mean())-gaps[bank][arm]);maximum_conservation_error=max(maximum_conservation_error,err);assert err<2e-6
                    row=dict(photo_id=qid,package_id=source['package_id'],split=source['split'],outcome=source['outcome'],positive=positive,negative=negative,arm=arm,bank=bank,
                        positive_score=a['score'],negative_score=b['score'],gap=gaps[bank][arm],edit_minus_sham_gap=gaps[bank][arm]-gaps[bank]['sham'],
                        switched_tokens=int(switch.sum()),valid_tokens=len(qi),switched_token_contribution=float(change[switch].sum()/len(qi)),unchanged_match_contribution=float(change[~switch].sum()/len(qi)),
                        interaction_gap=float(interaction.mean()),full_gallery_accuracy_computed=False,scope=SCOPE)
                    pair_rows.append(row)
                    for j,region in enumerate(REGIONS):
                        region_rows.append(dict(photo_id=qid,pair_id=pair,bank=bank,arm=arm,region=region,soft_token_mass=float(weights[:,j].sum()),
                            net_contribution=float((weights[:,j]*delta).sum()/len(qi)),edit_minus_sham_contribution=float((weights[:,j]*change).sum()/len(qi)),
                            positive_contribution=float((weights[:,j]*delta).clip(min=0).sum()/len(qi)),negative_contribution=float((weights[:,j]*delta).clip(max=0).sum()/len(qi))))
                    for k,index in enumerate(qi):
                        tokens.append(dict(arm=arm,query_index=int(index),token_type='visual' if visual[k] else 'template',
                            **{region+'_weight':float(weights[k,j]) for j,region in enumerate(REGIONS)},
                            positive_winner=int(a['winner'][k]),positive_runner_up=int(a['runner_up'][k]),positive_top1=float(a['top'][k]),positive_top2=float(a['second'][k]),
                            negative_winner=int(b['winner'][k]),negative_runner_up=int(b['runner_up'][k]),negative_top1=float(b['top'][k]),negative_top2=float(b['second'][k]),
                            Delta=float(delta[k]),edit_minus_sham_Delta=float(change[k]),interaction_Delta=float(interaction[k]),match_switched_from_sham=bool(switch[k]),
                            fresh_minus_frozen_Delta=float(deltas['fresh'][arm][k]-deltas['frozen'][arm][k])))
                for j,region in enumerate(REGIONS):
                    interactions.append(dict(photo_id=qid,pair_id=pair,bank=bank,region=region,interaction_contribution=float((weights[:,j]*interaction).sum()/len(qi))))
                csvout(output/'queries'/qid/(pair+'__'+bank+'.tokens.csv'),tokens)
            for arm in ARMS:
                p_same=np.array_equal(traces['fresh'][arm][positive]['ri'],traces['frozen'][arm][positive]['ri'])
                n_same=np.array_equal(traces['fresh'][arm][negative]['ri'],traces['frozen'][arm][negative]['ri'])
                before=pair_lookup[(qid,positive,negative,arm)];old_error=abs(float(before['gap'])-gaps['frozen'][arm]);assert old_error<2e-6
                comparisons.append(dict(photo_id=qid,positive=positive,negative=negative,arm=arm,frozen_gap=gaps['frozen'][arm],fresh_gap=gaps['fresh'][arm],fresh_minus_frozen_gap=gaps['fresh'][arm]-gaps['frozen'][arm],
                    frozen_edit_minus_sham_gap=gaps['frozen'][arm]-gaps['frozen']['sham'],fresh_edit_minus_sham_gap=gaps['fresh'][arm]-gaps['fresh']['sham'],
                    fresh_minus_frozen_edit_effect=(gaps['fresh'][arm]-gaps['fresh']['sham'])-(gaps['frozen'][arm]-gaps['frozen']['sham']),
                    positive_argmax_changed_from_frozen=int(np.sum(traces['fresh'][arm][positive]['winner']!=traces['frozen'][arm][positive]['winner'])) if p_same else None,
                    negative_argmax_changed_from_frozen=int(np.sum(traces['fresh'][arm][negative]['winner']!=traces['frozen'][arm][negative]['winner'])) if n_same else None,
                    comparable_positive_lattice=p_same,comparable_negative_lattice=n_same,
                    lattice_fields_legacy_scope='Boolean fields refer only to equal valid ordinal Token index axes, not physical-grid equivalence',
                    positive_same_ordinal_token_axis=p_same,negative_same_ordinal_token_axis=n_same,
                    physical_lattice_equivalence_verified=False,old_pair_row_error=old_error))
        print('Fresh-reference candidate replay',qid,flush=True)
    csvout(output/'query_runtime_parity.csv',d['query_parity']);csvout(output/'reference_cache_parity.csv',d['reference_parity'])
    csvout(output/'pair_effects.csv',pair_rows);csvout(output/'fresh_vs_frozen.csv',comparisons);csvout(output/'region_contributions.csv',region_rows);csvout(output/'region_interactions.csv',interactions)
    write(output/'matrix_index.json',matrix_index)
    validation=dict(status='passed',scope=SCOPE,queries=12,query_arms=60,fresh_reference_records=28,query_base_reruns=12,query_base_reruns_bitwise_exact=12,
        runtime_parameter_checks=d['runtime_checks'],matrices=len(matrix_index),fresh_matrices=sum(x['bank']=='fresh' for x in matrix_index),frozen_matrices=sum(x['bank']=='frozen' for x in matrix_index),
        worker_provenance=d['worker_provenance'],
        pair_arm_rows=len(pair_rows),fresh_frozen_comparisons=len(comparisons),max_original_matrix_error=maximum_original_error,max_Delta_gap_conservation_error=maximum_conservation_error,
        reference_bitwise_exact=sum(x['bitwise_equal'] for x in d['reference_parity']),source_SHA256=d['hashes'],code_SHA256=sha(Path(__file__)),code_path=str(Path(__file__).relative_to(ROOT)),
        historical_query_subject_box_scope='Original immutable box snapshot for historical interventions; boxes/ROI are extra supervision',
        full_gallery_accuracy_computed=False,edited_identity_accuracy_computed=False,causal_complete=False)
    write(output/'validation.json',validation)
    report=(f'# 新运行时干预候选对照\n\n全部12个独立身份、60个Query arms与预先冻结的28个Reference重放。'
        f'40张新编码中12个Query base与原干预base全部逐位相等；运行时版本、处理后端、模型配置与数值encoder源码一致。'
        f'历史/新Reference分别保留145个完整矩阵，17个候选对共170个pair-arm统计。\n\n'
        f'28个新Reference中{validation["reference_bitwise_exact"]}个与历史cache逐位相等。'
        'fresh_vs_frozen.csv保留差值、edit-minus-sham效应及同索引轴上的argmax切换诊断；Token数量或索引相同不证明旧新处理器产生同一物理格点。'
        'token trace与软ROI分区保留完整Delta、top2、切换、interaction与守恒分解。'
        '旧/新manifest worker源码SHA不同，旧worker源码快照缺失，不能确定差异原因；兼容性由数值源码/参数相等及12个Query原图逐位复现桥接。'
        '这些是预选候选的条件响应，不能视为全图库准确率、编辑后身份标签或因果机制完整证明。\n')
    (output/'REPORT.md').write_text(report)
    return validation


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=FOLDER/'analysis')
    args=parser.parse_args();torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    result=analyze(args.output.resolve());print(result['status'],result['scope'],flush=True)


if __name__=='__main__':main()
