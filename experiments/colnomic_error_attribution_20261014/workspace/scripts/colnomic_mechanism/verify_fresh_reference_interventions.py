"""Independent full pointwise/trace replay of fresh-reference detail controls."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from scripts.colnomic_mechanism.common import ROOT, read, rows, write, sha, workspace_path, csvout
from scripts.colnomic_mechanism.display_pixels import display_image
from scripts.analyze_colnomic_tokens import display_xy
from scripts.encode_colnomic_images import token_hash

BASE=ROOT/'results/colnomic_full_audit/20261013_resume'
OLD=BASE/'detail_interventions'
FOLDER=BASE/'fresh_reference_interventions'
ARMS=['base','sham','numeric_mask','blank_control','both_masks']
REGIONS=['numeric_proposal','blank_control','other_supplied_subject','background_outside_supplied_box','template']


def ahash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def reduce(q,r):
    qi=torch.where(q.float().norm(dim=1)>1e-8)[0].numpy();ri=torch.where(r.float().norm(dim=1)>1e-8)[0].numpy()
    dot=(q[qi].float()@r[ri].float().T).numpy()
    first=np.argmax(dot,axis=1);copy=dot.copy();copy[np.arange(len(qi)),first]=-np.inf
    second=np.argmax(copy,axis=1)
    return dict(qi=qi,ri=ri,dot=dot,winner=ri[first],runner=ri[second],
                top=dot[np.arange(len(qi)),first],second=dot[np.arange(len(qi)),second])


def weights_from_rectangles(spec,source,box,qi):
    with Image.open(workspace_path(source['original_path'],must_exist=True)) as im:size=display_image(im).size
    l=spec['layout'];visual=(qi>=l['start'])&(qi<l['start']+l['count']);weights=np.zeros((len(qi),5),float)
    weights[~visual,4]=1
    for j,index in enumerate(qi):
        if not visual[j]:continue
        k=index-l['start'];x=k%l['width'];y=k//l['width']
        corners=np.array([[x/l['width'],y/l['height']],[(x+1)/l['width'],(y+1)/l['height']]])
        mapped=display_xy(corners,spec['exif_orientation']);rect=np.r_[mapped.min(0),mapped.max(0)]*np.tile(size,2)
        area=(rect[2]-rect[0])*(rect[3]-rect[1])
        def overlap(b):return max(0,min(rect[2],b[2])-max(rect[0],b[0]))*max(0,min(rect[3],b[3])-max(rect[1],b[1]))/area
        n=overlap(source['numeric_bbox']);c=overlap(source['blank_bbox']);s=overlap(box)
        assert n+c<=s+1e-6
        weights[j]=[n,c,max(0,s-n-c),1-s,0]
    assert np.allclose(weights.sum(1),1) and weights.min()>-1e-7
    return weights,visual


def verify(analysis,output):
    assert not output.exists() or not any(output.iterdir()),'Independent receipts require a new empty output'
    output.mkdir(parents=True,exist_ok=True)
    result=read(analysis/'validation.json');assert result['status']=='passed'
    executed=workspace_path(result.get('code_path',str(Path(__file__).with_name('fresh_reference_interventions.py'))),must_exist=True)
    assert result['code_SHA256']==sha(executed)
    assert not result['full_gallery_accuracy_computed'] and not result['edited_identity_accuracy_computed']
    source_checks=0
    for path,digest in result['source_SHA256'].items():
        assert sha(workspace_path(path,must_exist=True))==digest,path;source_checks+=1
    selection=read(FOLDER/'candidate_selection.json');old_index=read(OLD/'analysis/matrix_index.json')
    assert selection['reference_ids']==sorted({m['reference_id'] for m in old_index})
    assert sha(OLD/'analysis/matrix_index.json')==selection['old_matrix_index_sha256']
    assert sha(OLD/'analysis/pair_effects.csv')==selection['old_pair_effects_sha256']
    manifest=read(FOLDER/'encoding_manifest.json');new={};runtime=read(FOLDER/'embeddings/runtime.json');prior=read(OLD/'embeddings/runtime.json')
    completion=read(FOLDER/'embeddings/complete.json')
    assert completion['status']=='complete' and completion['records']==len(manifest)==40
    assert completion['manifest_sha256']==sha(FOLDER/'encoding_manifest.json')
    assert len({r['id'] for r in manifest})==40
    runtime_keys=['packages','model_config','processor_backend','optimized_encoder_sha256','numeric_encoder_sha256','GPU_memory_caps_fraction']
    assert all(runtime[k]==prior[k] for k in runtime_keys)
    assert runtime['residency']['arithmetic']==prior['residency']['arithmetic']
    assert sha(Path(__file__).with_name('optimized_encoder.py'))==runtime['optimized_encoder_sha256']
    assert sha(Path(__file__).with_name('offloaded_encoder.py'))==runtime['numeric_encoder_sha256']
    assert result['worker_provenance']['old_worker_SHA256']==prior['worker_sha256'] and result['worker_provenance']['new_worker_SHA256']==runtime['worker_sha256']
    for row in manifest:
        record=torch.load(FOLDER/'embeddings'/(row['id']+'.pt'),map_location='cpu',weights_only=True)
        assert all(record.get(k)==v for k,v in row.items())
        assert record['runtime']==runtime['packages'] and record['processor_backend']==runtime['processor_backend']
        assert sha(workspace_path(row['path'],must_exist=True))==record['image_sha256']==row['image_sha256']
        assert token_hash(record['embedding'])==record['tensor_sha256']
        assert record['embedding'].ndim==2 and record['embedding'].shape[1]==128 and torch.isfinite(record['embedding']).all()
        new[row['id']]=record['embedding']
    assert len(new)==40
    sources=read(OLD/'sources.json');queries={};base_exact=0;pixel_checks=0
    assert len(sources)==len({s['photo_id'] for s in sources})==len({s['package_id'] for s in sources})==12
    qb=torch.load(ROOT/'cache/image_embeddings/colnomic7b/photos.pt',weights_only=True,mmap=True,map_location='cpu')
    assert set(new)==set(selection['reference_ids'])|{s['photo_id']+'_rerun_base' for s in sources}
    assert selection['per_query_reference_ids']=={s['photo_id']:sorted({m['reference_id'] for m in old_index if m['photo_id']==s['photo_id']}) for s in sources}
    parity_rows=rows(analysis/'query_runtime_parity.csv');assert len(parity_rows)==12
    parity_by_id={r['photo_id']:r for r in parity_rows};assert len(parity_by_id)==12
    for s in sources:
        qid=s['photo_id'];queries[qid]={}
        assert qb['ids'][s['cache_row']]==qid
        assert sha(workspace_path(s['original_path'],must_exist=True))==qb['sha256'][s['cache_row']]
        assert {r['arm'] for r in s['arms']}==set(ARMS) and len(s['arms'])==5
        for row in s['arms']:
            record=torch.load(OLD/'embeddings'/(row['id']+'.pt'),map_location='cpu',weights_only=True)
            assert record['runtime']==runtime['packages'] and record['processor_backend']==runtime['processor_backend']
            assert sha(workspace_path(row['path'],must_exist=True))==record['image_sha256']==row['image_SHA256']
            assert token_hash(record['embedding'])==record['tensor_sha256']
            assert record['embedding'].ndim==2 and record['embedding'].shape==qb['embeddings'][s['cache_row']].shape and torch.isfinite(record['embedding']).all()
            queries[qid][row['arm']]=record['embedding']
        assert torch.equal(new[qid+'_rerun_base'],queries[qid]['base']);base_exact+=1
        assert torch.equal(qb['embeddings'][s['cache_row']],queries[qid]['base'])
        assert torch.equal(queries[qid]['sham'],queries[qid]['base'])
        parity=parity_by_id[qid]
        assert parity['rerun_base_bitwise_exact']=='True' and parity['tensor_SHA256']==token_hash(new[qid+'_rerun_base']) and float(parity['max_absolute_error'])==0
        arms={x['arm']:x for x in s['arms']}
        with Image.open(workspace_path(arms['base']['path'],must_exist=True)) as im:original=np.asarray(display_image(im)).copy()
        nb=s['numeric_bbox'];bb=s['blank_bbox'];assert nb[2]-nb[0]==bb[2]-bb[0] and nb[3]-nb[1]==bb[3]-bb[1]
        for arm in ARMS:
            with Image.open(workspace_path(arms[arm]['path'],must_exist=True)) as im:edited=np.asarray(display_image(im)).copy()
            allowed=np.zeros(original.shape[:2],bool)
            for b in ([nb] if arm=='numeric_mask' else [bb] if arm=='blank_control' else [nb,bb] if arm=='both_masks' else []):allowed[b[1]:b[3],b[0]:b[2]]=True
            assert np.array_equal(original[~allowed],edited[~allowed]);pixel_checks+=1
    rb=torch.load(ROOT/'cache/image_embeddings/colnomic7b/references.pt',weights_only=True,mmap=True,map_location='cpu');rindex={rid:i for i,rid in enumerate(rb['ids'])}
    reference_parity=rows(analysis/'reference_cache_parity.csv');assert len(reference_parity)==28
    reference_by_id={r['reference_id']:r for r in reference_parity};assert set(reference_by_id)==set(selection['reference_ids'])
    for rid,parity in reference_by_id.items():
        i=rindex[rid];old=rb['embeddings'][i];fresh=new[rid];same=old.shape==fresh.shape
        assert next(r for r in manifest if r['id']==rid)['image_sha256']==rb['sha256'][i]
        assert (parity['same_shape']=='True')==same
        import json
        assert json.loads(parity['old_shape'])==list(old.shape) and json.loads(parity['fresh_shape'])==list(fresh.shape)
        assert parity['frozen_tensor_SHA256']==token_hash(old) and parity['fresh_tensor_SHA256']==token_hash(fresh)
        assert (parity['bitwise_equal']=='True')==bool(same and torch.equal(old,fresh))
        if same:
            assert abs(float(parity['max_absolute_error'])-float((old.float()-fresh.float()).abs().max()))<1e-9
            assert abs(float(parity['mean_token_cosine'])-float(torch.nn.functional.cosine_similarity(old.float(),fresh.float(),dim=1).mean()))<1e-9
        else:assert parity['max_absolute_error']==parity['mean_token_cosine']==''
    matrices=read(analysis/'matrix_index.json');expected={(bank,m['photo_id'],m['arm'],m['reference_id']) for bank in ['fresh','frozen'] for m in old_index}
    assert {(m['bank'],m['photo_id'],m['arm'],m['reference_id']) for m in matrices}==expected and len(matrices)==290
    traces={};matrix_error=0.;matrix_files=0;numpy_error=0.
    for m in matrices:
        q=queries[m['photo_id']][m['arm']];r=rb['embeddings'][rindex[m['reference_id']]] if m['bank']=='frozen' else new[m['reference_id']]
        t=reduce(q,r);file=workspace_path(m['path'],must_exist=True) if m['path_scope']=='workspace' else analysis/m['path']
        assert file.resolve().is_relative_to(ROOT) or file.resolve().is_relative_to(analysis.resolve())
        assert sha(file)==m['SHA256'] and token_hash(q)==m['query_tensor_SHA256'] and token_hash(r)==m['reference_tensor_SHA256']
        with np.load(file) as saved:
            assert saved['similarity'].dtype==np.float32 and saved['similarity'].shape==tuple(m['shape']) and np.isfinite(saved['similarity']).all()
            assert np.array_equal(saved['query_valid_indices'],t['qi']) and np.array_equal(saved['reference_valid_indices'],t['ri'])
            error=float(np.abs(saved['similarity']-t['dot']).max());matrix_error=max(matrix_error,error);assert error<2e-6
            independently_multiplied=q[t['qi']].float().numpy()@r[t['ri']].float().numpy().T
            error=float(np.abs(saved['similarity']-independently_multiplied).max());numpy_error=max(numpy_error,error);assert error<2e-6
            if m['bank']=='fresh':
                for field,key in [('winner_indices','winner'),('runner_up_indices','runner'),('top1','top'),('top2','second')]:assert np.array_equal(saved[field],t[key])
        assert ahash(t['dot'])==m['similarity_SHA256'] and ahash(t['qi'])==m['query_indices_SHA256'] and ahash(t['ri'])==m['reference_indices_SHA256']
        assert abs(float(t['top'].mean())-m['score'])<2e-6
        traces[(m['bank'],m['photo_id'],m['arm'],m['reference_id'])]=t;matrix_files+=1
    metadata=read(ROOT/'results/token_mechanism_analysis/F03/reproduction/token_metadata.json')
    boxes=read(BASE/'manual_box_review/subject_boxes_before_user_review.json')['records'];source_by_id={s['photo_id']:s for s in sources}
    region_weights={};visuals={}
    for qid,s in source_by_id.items():
        qi=traces[('fresh',qid,'base',selection['per_query_reference_ids'][qid][0])]['qi']
        region_weights[qid],visuals[qid]=weights_from_rectangles(metadata['photos'][s['cache_row']],s,boxes[qid]['bbox'],qi)
    pair_rows=rows(analysis/'pair_effects.csv');pair_values={};max_scalar_error=0.;max_roi_error=0.;token_checks=0
    def close(actual,want):
        nonlocal max_scalar_error
        error=abs(float(actual)-float(want));max_scalar_error=max(max_scalar_error,error);assert error<2e-6,(actual,want)
    keys={(r['photo_id'],r['positive'],r['negative']) for r in rows(OLD/'analysis/pair_effects.csv')}
    assert len(keys)==17 and len(pair_rows)==170
    for bank in ['fresh','frozen']:
        for qid,p,n in sorted(keys):
            delta={};gap={};switches={}
            for arm in ARMS:
                a=traces[(bank,qid,arm,p)];b=traces[(bank,qid,arm,n)]
                delta[arm]=a['top'].astype(float)-b['top'].astype(float);gap[arm]=float(a['top'].mean())-float(b['top'].mean())
                assert np.array_equal(a['qi'],b['qi'])
                switches[arm]=(a['winner']!=traces[(bank,qid,'sham',p)]['winner'])|(b['winner']!=traces[(bank,qid,'sham',n)]['winner'])
            interaction=delta['both_masks']-delta['numeric_mask']-delta['blank_control']+delta['sham'];w=region_weights[qid]
            pair_values[(bank,qid,p,n)]=(delta,gap,switches,interaction)
            token_rows=rows(analysis/'queries'/qid/(p+'__'+n+'__'+bank+'.tokens.csv'))
            expected_count=len(delta['base'])*5;assert len(token_rows)==expected_count
            seen=set()
            for row in token_rows:
                arm=row['arm'];index=int(row['query_index']);a=traces[(bank,qid,arm,p)];b=traces[(bank,qid,arm,n)]
                k=int(np.searchsorted(a['qi'],index));assert k<len(a['qi']) and a['qi'][k]==index and (arm,index) not in seen;seen.add((arm,index))
                for field,tkey,t in [('positive_winner','winner',a),('negative_winner','winner',b),('positive_runner_up','runner',a),('negative_runner_up','runner',b)]:assert int(row[field])==int(t[tkey][k])
                for field,tkey,t in [('positive_top1','top',a),('positive_top2','second',a),('negative_top1','top',b),('negative_top2','second',b)]:close(row[field],t[tkey][k])
                close(row['Delta'],delta[arm][k]);close(row['edit_minus_sham_Delta'],delta[arm][k]-delta['sham'][k]);close(row['interaction_Delta'],interaction[k])
                fresh=traces[('fresh',qid,arm,p)]['top'].astype(float)-traces[('fresh',qid,arm,n)]['top'].astype(float)
                frozen=traces[('frozen',qid,arm,p)]['top'].astype(float)-traces[('frozen',qid,arm,n)]['top'].astype(float)
                close(row['fresh_minus_frozen_Delta'],fresh[k]-frozen[k]);assert (row['match_switched_from_sham']=='True')==bool(switches[arm][k])
                assert row['token_type']==('visual' if visuals[qid][k] else 'template')
                for j,region in enumerate(REGIONS):
                    error=abs(float(row[region+'_weight'])-w[k,j]);max_roi_error=max(max_roi_error,error);assert error<1e-9
                token_checks+=1
    for row in pair_rows:
        bank=row['bank'];qid=row['photo_id'];p=row['positive'];n=row['negative'];arm=row['arm'];delta,gap,switch,interaction=pair_values[(bank,qid,p,n)]
        close(row['gap'],gap[arm]);close(row['positive_score'],traces[(bank,qid,arm,p)]['top'].mean());close(row['negative_score'],traces[(bank,qid,arm,n)]['top'].mean())
        close(row['edit_minus_sham_gap'],gap[arm]-gap['sham']);close(row['interaction_gap'],interaction.mean())
        change=delta[arm]-delta['sham'];close(row['switched_token_contribution'],change[switch[arm]].sum()/len(change));close(row['unchanged_match_contribution'],change[~switch[arm]].sum()/len(change))
        assert int(row['switched_tokens'])==switch[arm].sum() and int(row['valid_tokens'])==len(change);close(delta[arm].mean(),gap[arm])
    region_rows=rows(analysis/'region_contributions.csv');assert len(region_rows)==850
    assert len({(r['bank'],r['photo_id'],r['pair_id'],r['arm'],r['region']) for r in region_rows})==850
    for row in region_rows:
        p,n=row['pair_id'].split('__');bank=row['bank'];qid=row['photo_id'];arm=row['arm'];j=REGIONS.index(row['region'])
        delta,gap,switch,interaction=pair_values[(bank,qid,p,n)];w=region_weights[qid][:,j];weighted=w*delta[arm]
        close(row['soft_token_mass'],w.sum());close(row['net_contribution'],weighted.mean());close(row['edit_minus_sham_contribution'],(w*(delta[arm]-delta['sham'])).mean())
        close(row['positive_contribution'],weighted.clip(min=0).mean());close(row['negative_contribution'],weighted.clip(max=0).mean())
    interaction_rows=rows(analysis/'region_interactions.csv');assert len(interaction_rows)==170
    assert len({(r['bank'],r['photo_id'],r['pair_id'],r['region']) for r in interaction_rows})==170
    for row in interaction_rows:
        p,n=row['pair_id'].split('__');delta,gap,switch,interaction=pair_values[(row['bank'],row['photo_id'],p,n)]
        close(row['interaction_contribution'],(region_weights[row['photo_id']][:,REGIONS.index(row['region'])]*interaction).mean())
    comparisons=rows(analysis/'fresh_vs_frozen.csv');assert len(comparisons)==85
    assert {(r['photo_id'],r['positive'],r['negative'],r['arm']) for r in comparisons}=={(q,p,n,a) for q,p,n in keys for a in ARMS}
    for row in comparisons:
        qid=row['photo_id'];p=row['positive'];n=row['negative'];arm=row['arm'];df,gf,sf,inf=pair_values[('fresh',qid,p,n)];do,go,so,ino=pair_values[('frozen',qid,p,n)]
        for field,want in [('fresh_gap',gf[arm]),('frozen_gap',go[arm]),('fresh_minus_frozen_gap',gf[arm]-go[arm]),('fresh_edit_minus_sham_gap',gf[arm]-gf['sham']),('frozen_edit_minus_sham_gap',go[arm]-go['sham']),('fresh_minus_frozen_edit_effect',(gf[arm]-gf['sham'])-(go[arm]-go['sham']))]:close(row[field],want)
        for prefix,rid in [('positive',p),('negative',n)]:
            a=traces[('fresh',qid,arm,rid)];b=traces[('frozen',qid,arm,rid)];same=np.array_equal(a['ri'],b['ri'])
            assert (row['comparable_'+prefix+'_lattice']=='True')==same
            if same:assert int(row[prefix+'_argmax_changed_from_frozen'])==np.sum(a['winner']!=b['winner'])
            else:assert row[prefix+'_argmax_changed_from_frozen']==''
    for bank in ['fresh','frozen']:
        assert {(r['photo_id'],r['positive'],r['negative'],r['arm']) for r in pair_rows if r['bank']==bank}=={(q,p,n,a) for q,p,n in keys for a in ARMS}
    verification=dict(status='passed',scope='Independent complete raw pointwise matrices (NumPy and original FP32 arithmetic), stable top2/argmax, source/runtime/base rerun, all pair/ROI/interaction/token traces',
        source_SHA_checks=source_checks,new_encoding_records=len(new),query_base_bitwise_exact=base_exact,runtime_parameter_checks=len(runtime_keys)+1,pixel_checks=pixel_checks,
        full_pointwise_matrices=matrix_files,fresh_matrices=145,frozen_matrices=145,pair_arm_rows=len(pair_rows),pair_token_rows=token_checks,region_rows=len(region_rows),fresh_frozen_comparisons=len(comparisons),
        max_matrix_error=matrix_error,max_independent_NumPy_matrix_error=numpy_error,max_scalar_conservation_error=max_scalar_error,max_ROI_weight_error=max_roi_error,
        reference_parity_rows=len(reference_parity),query_parity_rows=len(parity_rows),region_interaction_rows=len(interaction_rows),
        analysis_validation_sha256=sha(analysis/'validation.json'),verifier_SHA256=sha(Path(__file__)),full_gallery_accuracy_computed=False,edited_identity_accuracy_computed=False,causal_complete=False)
    write(output/'validation.json',verification);return verification


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--analysis',type=Path,default=FOLDER/'analysis');parser.add_argument('--output',type=Path,default=FOLDER/'independent')
    args=parser.parse_args();torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    result=verify(args.analysis.resolve(),args.output.resolve());print(result['status'],result['full_pointwise_matrices'],result['pair_token_rows'],flush=True)


if __name__=='__main__':main()
