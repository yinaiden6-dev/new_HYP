"""Independent NumPy score/trace replay, with selected full Token matrices."""
from collections import defaultdict
import numpy as np
import torch
from scripts.colnomic_mechanism.common import ROOT,read,write,rows,csvout,sha,workspace_path
from scripts.analyze_colnomic_tokens import token_hash
BASE=ROOT/'results/colnomic_full_audit/20261013_resume/controlled_crossview_layouts'


def main():
    torch.set_num_threads(2);manifest=read(BASE/'encoding_manifest.json');out=BASE/'analysis';vectors={};indices={};byid={r['id']:r for r in manifest}
    complete=read(BASE/'embeddings/complete.json');assert complete['manifest_sha256']==sha(BASE/'encoding_manifest.json') and complete['records']==len(manifest)==36
    for row in manifest:
        stored=torch.load(BASE/'embeddings'/(row['id']+'.pt'),map_location='cpu',weights_only=True)
        assert all(stored[k]==v for k,v in row.items()) and stored['image_sha256']==sha(workspace_path(row['path'])) and stored['tensor_sha256']==token_hash(stored['embedding'])
        value=stored['embedding'].float().numpy();ix=np.flatnonzero(np.linalg.norm(value,axis=1)>1e-8);vectors[row['id']]=value[ix];indices[row['id']]=ix
    matrix=np.load(out/'complete_similarity_matrix.npz');ids=[r['id'] for r in manifest];assert matrix['ids'].tolist()==ids
    details=rows(out/'feature_argmax_decomposition.csv');needed={(r[k],r['candidate_id']) for r in details for k in ['query_id','sham_id']}
    matrixdir=out/'selected_token_matrices';matrixdir.mkdir(exist_ok=True);export=[];dots={};score_error=0
    for i,qid in enumerate(ids):
        for j,rid in enumerate(ids):
            dot=vectors[qid]@vectors[rid].T;score=float(dot.max(1).mean(dtype=np.float64));score_error=max(score_error,abs(score-float(matrix['score'][i,j])))
            assert abs(score-float(matrix['score'][i,j]))<2e-6
            if (qid,rid) in needed:
                target=matrixdir/(qid+'__'+rid+'.npz');np.savez_compressed(target,similarity=dot,query_indices=indices[qid],reference_indices=indices[rid])
                dots[qid,rid]=dot;export.append(dict(query_id=qid,candidate_id=rid,path=str(target.relative_to(out)),SHA256=sha(target)))
    grouped=defaultdict(list)
    for r in rows(out/'per_query_token_changes.csv'):grouped[r['query_id'],r['sham_id'],r['candidate_id'],r['query_change']].append(r)
    maximum=0.;winner_error=0.;checks=0
    for key,rs in grouped.items():
        qid,sid,rid,change=key;edited=dots[qid,rid];sham=dots[sid,rid]
        assert len(rs)==len(indices[qid]) and [int(r['query_token']) for r in rs]==list(indices[qid])
        index={int(t):i for i,t in enumerate(indices[rid])};w0=np.array([index[int(r['sham_candidate_winner_token'])] for r in rs]);w1=np.array([index[int(r['changed_candidate_winner_token'])] for r in rs]);k=np.arange(len(rs))
        before=sham[k,w0].astype(np.float64);after=edited[k,w1].astype(np.float64)
        winner_error=max(winner_error,float(np.max(np.abs(sham.max(1)-before))),float(np.max(np.abs(edited.max(1)-after))))
        delta=after-before;fixed=(edited[k,w0]-sham[k,w0]).astype(np.float64);switch=(edited[k,w1]-edited[k,w0]).astype(np.float64)
        for name,wanted in [('score_delta',delta),('fixed_winner_feature_delta',fixed),('argmax_switch_delta',switch),('mean_score_contribution',delta/len(rs))]:
            error=float(np.max(np.abs(np.array([float(r[name]) for r in rs])-wanted)));maximum=max(maximum,error);assert error<2e-6
        assert np.max(np.abs(delta-fixed-switch))<2e-6;checks+=len(rs)
    assert winner_error<2e-6
    score={(q,r):float(matrix['score'][i,j]) for i,q in enumerate(ids) for j,r in enumerate(ids)}
    contrasts=rows(out/'predefined_layout_contrasts.csv');relations=read(BASE/'expected_relations.json');assert len(contrasts)==len(relations)==12
    for relation,row in zip(relations,contrasts):
        q=relation['query_id'];p=relation['consistent_id'];n=relation['inconsistent_id']
        assert byid[q]['source_id']!=byid[p]['source_id'] and byid[q]['source_id']!=byid[n]['source_id']
        assert abs(score[q,p]-score[q,n]-float(row['predefined_layout_gap']))<1e-12
        assert abs(float(row['predefined_layout_gap'])-float(row['sham_layout_gap'])-float(row['edit_minus_sham_layout_gap']))<1e-12
        assert abs(float(row['predefined_layout_gap'])-float(row['editor_only_layout_gap'])-float(row['layout_minus_editor_gap']))<1e-12
    sourcechecks=read(out/'validation.json');assert all(sourcechecks.values())
    write(out/'selected_token_matrix_index.json',export)
    write(out/'independent_validation.json',dict(status='passed',scores_replayed=36*36,selected_full_token_matrices=len(export),token_trace_rows_replayed=checks,
        NumPy_vs_Torch_score_max_error=score_error,token_trace_max_error=maximum,recorded_winner_maximum_value_error=winner_error,
        relations=len(contrasts),source_validation_checks=sourcechecks,verifier_SHA256=sha(__file__),
        source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in [BASE/'encoding_manifest.json',BASE/'embeddings/complete.json',out/'complete_similarity_matrix.npz',out/'per_query_token_changes.csv',out/'predefined_layout_contrasts.csv']},
        scope='cross-view3 identity pilot; no held-test group or fitted classifier; all scalar pairs and selected complete Token matrices verified independently'))
    print('Independent crossview layout',len(export),'matrices',checks,'traces passed',flush=True)


if __name__=='__main__':main()
