#!/usr/bin/env python3
"""Check interpretation of actual source formulas, without model inference.

Numerical examples establish algebraic identities, not their importance for
identity recognition or causality in the pretrained head.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rc_pair_relation_localization_v1/embedding_identities.json'


def bind(p):
    return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    rng=np.random.default_rng(20260924)
    grid=np.stack(np.meshgrid(np.linspace(-.875,.875,8),np.linspace(-.875,.875,8)),axis=-1).reshape(-1,2)
    omega=rng.normal(size=(512,2))*2*np.pi
    phase=grid@omega.T
    phi=np.concatenate([np.sin(phase),np.cos(phase)],axis=1)
    logits=rng.normal(size=(19,len(grid)))
    a=np.exp(logits-logits.max(1,keepdims=True));a/=a.sum(1,keepdims=True)
    p=a@phi
    lhs=np.sum(p*p,axis=1)/512
    kernel=np.cos((grid[:,None,:]-grid[None,:,:])@omega.T).mean(-1)
    rhs=np.einsum('ij,jk,ik->i',a,kernel,a)
    err=float(np.max(abs(lhs-rhs)));assert err<1e-12
    # Jointly translating all positions rotates each sine/cosine pair and
    # preserves its norm: concentration cannot identify absolute position.
    moved=(grid+np.array([.213,-.137]))@omega.T
    shifted=a@np.concatenate([np.sin(moved),np.cos(moved)],axis=1)
    shift_err=float(np.max(abs(np.sum(shifted*shifted,axis=1)/512-lhs)))
    assert shift_err<1e-12
    rms=np.sqrt(np.mean(p*p))
    assert abs(2*rms*rms-np.mean(lhs))<1e-12
    # For SINGLE, both sym(M) and sym(M*L) lose the common query factor.
    q=np.exp(rng.normal(size=13));r=np.exp(rng.normal(size=128));content=np.exp(rng.normal(size=128))
    contrasts=[];products=[]
    for u in q:
        m=np.sqrt(u*r)
        contrasts.append((m-m[0])/(m+m[0]))
        s=m*content;products.append((s-s[0])/(s+s[0]))
    cancel_error=max(float(np.max(np.ptp(np.asarray(v),axis=0))) for v in [contrasts,products])
    assert cancel_error<1e-12
    result=dict(status='SOURCE_FORMULA_IDENTITIES_NUMPY_PASS',
        sources=[bind(Path(__file__)),bind(ROOT.parents[2]/'third_party/RoMaV2/src/romav2/matcher.py'),
                 bind(ROOT/'programs/rc_pair_quality_core_v1.py')],
        positional_distribution=dict(expression='P_i=sum_j a_ij [sin(Omega x_j),cos(Omega x_j)]',
            norm_identity='||P_i||^2/K=sum_jk a_ij a_ik mean_l cos(omega_l dot (x_j-x_k))',
            K=512,D=1024,kernel_identity_error=err,translation_norm_invariance_error=shift_err,
            summary_identity='mean_i ||P_i||^2/K = 2 * global_matching_position_rms^2',
            meaning='Kernel-smoothed concentration of the soft correspondence distribution; vector also carries location-dependent phase.',
            not_proven=['Entropy or unique correspondence cannot be recovered from its norm alone.',
                        'These identities do not establish which part the trained head actually uses.',
                        'No new geometry or identity theorem is claimed.']),
        single_branch=dict(symmetry_cancellation_error=cancel_error,
            assumptions='Positive values, ignoring existing epsilon; cancellation is exact for sym(M) and sym(M*L).',
            implication='At fixed query SINGLE M contributes a candidate reference-only scalar preference; it is not a direct test of rich query-reference content comparison.'),
        actual_dataset_predictions=False,new_training=False,new_GPU=False)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
