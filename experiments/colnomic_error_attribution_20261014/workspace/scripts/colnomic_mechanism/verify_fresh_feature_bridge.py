"""Independent NumPy replay of reused Query edit feature changes.

All twelve repeated Query bases are bitwise equal. Feature deltas can therefore
be reused across the fresh/frozen candidate banks, while their MaxSim matches
and score contributions are independently checked in the main branch verifier.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.colnomic_mechanism.common import ROOT,read,rows,write,csvout,sha,workspace_path
from scripts.encode_colnomic_images import token_hash

BASE=ROOT/'results/colnomic_full_audit/20261013_resume'
OLD=BASE/'detail_interventions'
FOLDER=BASE/'fresh_reference_interventions'


def verify(output):
    assert not output.exists() or not any(output.iterdir()),'Use a fresh feature replay output'
    output.mkdir(parents=True,exist_ok=True)
    sources=read(OLD/'sources.json');pairs=rows(OLD/'analysis/pair_effects.csv');hashes={};inputs={};stats=[]
    source_by_id={r['photo_id']:r for r in sources}
    keys={(r['photo_id'],r['positive'],r['negative']) for r in pairs}
    assert len(keys)==17 and len(sources)==12
    for source in sources:
        qid=source['photo_id'];arms={}
        for row in source['arms']:
            path=OLD/'embeddings'/(row['id']+'.pt');record=torch.load(path,map_location='cpu',weights_only=True)
            assert token_hash(record['embedding'])==record['tensor_sha256']
            assert sha(workspace_path(row['path'],must_exist=True))==record['image_sha256']
            arms[row['arm']]=record['embedding'];hashes[str(path.relative_to(ROOT))]=sha(path)
        repeated=FOLDER/'embeddings'/(qid+'_rerun_base.pt');record=torch.load(repeated,map_location='cpu',weights_only=True)
        assert torch.equal(record['embedding'],arms['base']) and torch.equal(arms['sham'],arms['base'])
        hashes[str(repeated.relative_to(ROOT))]=sha(repeated)
        indices=np.flatnonzero(np.linalg.norm(arms['sham'].float().numpy().astype(float),axis=1)>1e-8)
        sham=arms['sham'][indices].float().numpy().astype(float);sham_norm=np.linalg.norm(sham,axis=1)
        tokens=[]
        for arm,vector in arms.items():
            observed=vector[indices].float().numpy().astype(float);diff=observed-sham
            norm=np.linalg.norm(diff,axis=1)
            cosine=(observed*sham).sum(1)/(np.linalg.norm(observed,axis=1)*sham_norm)
            assert np.isfinite(norm).all() and np.isfinite(cosine).all()
            inputs[(qid,arm)]=(indices,norm,cosine)
            stats.append(dict(photo_id=qid,package_id=source['package_id'],split=source['split'],arm=arm,
                valid_tokens=len(indices),changed_feature_tokens=int(np.any(diff!=0,axis=1).sum()),
                mean_feature_L2_from_sham=float(norm.mean()),max_feature_L2_from_sham=float(norm.max()),
                mean_feature_cosine_from_sham=float(cosine.mean()),min_feature_cosine_from_sham=float(cosine.min()),
                candidate_scope='Identical reused Query vectors for both fresh and frozen candidate banks; feature changes alone are not score attribution'))
            for index,l2,cos in zip(indices,norm,cosine):
                tokens.append(dict(arm=arm,query_index=int(index),feature_L2_from_sham=float(l2),feature_cosine_from_sham=float(cos)))
        csvout(output/'feature_tokens'/(qid+'.csv'),tokens)
    checked=0;max_l2=0.;max_cos=0.
    for qid,p,n in sorted(keys):
        file=OLD/'analysis/queries'/qid/(p+'__'+n+'.tokens.csv');records=rows(file);hashes[str(file.relative_to(ROOT))]=sha(file)
        seen=set()
        for row in records:
            arm=row['arm'];index=int(row['query_index']);indices,l2,cos=inputs[(qid,arm)]
            k=int(np.searchsorted(indices,index));assert k<len(indices) and indices[k]==index and (arm,index) not in seen;seen.add((arm,index))
            error=abs(float(row['feature_L2_from_sham'])-l2[k]);max_l2=max(max_l2,error);assert error<2e-6
            error=abs(float(row['feature_cosine_from_sham'])-cos[k]);max_cos=max(max_cos,error);assert error<2e-6
            checked+=1
        assert seen=={(arm,int(i)) for q,a in inputs if q==qid for arm in [a] for i in inputs[(q,arm)][0]}
    assert checked==63575 and len(stats)==60
    csvout(output/'feature_changes_per_query.csv',stats)
    receipt=dict(status='passed',scope='Independent NumPy float64 recomputation of every original Query edit feature L2/cosine trace; bridged by twelve actual bitwise repeated bases',
        identities=12,query_arms=60,preselected_pairs=17,replayed_feature_token_rows=checked,
        max_L2_error=max_l2,max_cosine_error=max_cos,source_SHA256=hashes,code_SHA256=sha(Path(__file__)),
        full_gallery_accuracy_computed=False,edited_identity_accuracy_computed=False,causal_complete=False)
    write(output/'validation.json',receipt);return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=FOLDER/'feature_bridge')
    args=parser.parse_args();torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    result=verify(args.output.resolve());print('Feature bridge passed',result['replayed_feature_token_rows'],flush=True)


if __name__=='__main__':main()
