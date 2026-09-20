"""Verification-null V2 math: fixed-denominator unary evidence and metrics."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import torch
from .cw1_sr0_structure_v1 import (
    STATUS_H0,
    STATUS_MULTI_REGION_SAME_CANDIDATE,
    STATUS_MULTI_TARGET,
    STATUS_SINGLE,
)

CANONICAL_ARM="CW1_QUERY_MULTITILE_LOCAL_COMPONENT_SET"
UNVERIFIED_H0="VERIFICATION_NULL_UNVERIFIED"
VERIFIED_H1="REFERENCE_BINDING_VERIFIED"
HOLD="HOLD_NO_VERIFIED_CANDIDATE"
HOLD_STRUCTURAL="HOLD_STRUCTURAL_PROPOSAL_UNAVAILABLE"
SINGLE_VERIFIED="SINGLE_VERIFIED_REQUIRES_DEPLOYMENT_GATE"
MULTI_VERIFIED="MULTI_VERIFIED_REQUIRES_T1_COMPETITION"
PAIR_COMPETITION_REQUIRED="PAIR_COMPETITION_REQUIRED"
PRIMARY_UNDETERMINED_HOLD="PRIMARY_UNDETERMINED_HOLD"
class VerificationNullError(RuntimeError):pass
def req(c:object,m:str):
    if not c:raise VerificationNullError(m)

def fixed_denominator_unique_patch_unary(*,signed_head:torch.nn.Module,relational_by_root:torch.Tensor,root_query_masks:torch.Tensor,sealed_union_mask:torch.Tensor,root_ready:torch.Tensor,root_ordinals:tuple[int,...],arm_name:str,structural_available:bool=True)->torch.Tensor:
    req(arm_name==CANONICAL_ARM,"verification-null arm is not canonical")
    req(isinstance(signed_head,torch.nn.Linear) and signed_head.out_features==1 and signed_head.bias is None,"signed head")
    value=torch.as_tensor(relational_by_root);masks=torch.as_tensor(root_query_masks,dtype=torch.bool,device=value.device);union=torch.as_tensor(sealed_union_mask,dtype=torch.bool,device=value.device).flatten();ready=torch.as_tensor(root_ready,dtype=torch.bool,device=value.device).flatten()
    req(value.ndim==3 and value.shape[:2]==masks.shape and value.shape[2]==signed_head.in_features and masks.shape[1]==union.numel() and ready.shape==(value.shape[0],) and root_ordinals==tuple(range(value.shape[0])) and value.is_floating_point() and value.dtype==signed_head.weight.dtype and value.device==signed_head.weight.device and bool(torch.isfinite(value).all()),"unary population")
    coverage=masks.sum(dim=0);req(bool(union.any()) and torch.equal(coverage.gt(0),union),"sealed union/coverage")
    req(bool(value[~masks].eq(0).all()),"relational evidence escaped root mask")
    req(bool(value[~ready].eq(0).all()),"missing-root relational evidence is nonzero")
    if not structural_available:
        req(not bool(ready.any()) and bool(value.eq(0).all()),"structural-unavailable population is not exact zero")
        return signed_head.weight.square().sum()*0.0
    req(int(ready.sum())>=2,"structural-ready multitile needs at least two ready roots")
    patch_scores=signed_head(value).squeeze(-1);active=masks.to(patch_scores.dtype)*ready[:,None].to(patch_scores.dtype);numerator=(patch_scores*active).sum(dim=0);denominator=coverage.clamp_min(1).to(patch_scores.dtype);per_patch=torch.where(union,numerator/denominator,torch.zeros_like(numerator));score=per_patch[union].sum()/union.sum().to(per_patch.dtype)
    if bool(value.detach().eq(0).all()):return signed_head.weight.square().sum()*0.0
    return score

def verification_evidence(a_to_b:torch.Tensor,b_to_a:torch.Tensor)->torch.Tensor:
    a=torch.as_tensor(a_to_b);b=torch.as_tensor(b_to_a);req(a.ndim==b.ndim==0 and a.is_floating_point() and b.is_floating_point() and bool(torch.isfinite(a)) and bool(torch.isfinite(b)),"direction evidence");return torch.minimum(a,b.to(a.dtype))
def verification_probability(a_to_b:torch.Tensor,b_to_a:torch.Tensor)->torch.Tensor:return torch.sigmoid(verification_evidence(a_to_b,b_to_a))

@dataclass(frozen=True)
class QueryVerificationDecision:
    status:str
    verified_candidates:tuple[str,...]

def decide_query(candidate_directions:Mapping[str,tuple[torch.Tensor,torch.Tensor]],*,ambiguity_status:str=STATUS_SINGLE,structural_available:Mapping[str,bool]|None=None)->QueryVerificationDecision:
    req(bool(candidate_directions),"empty candidate population");verified=[]
    availability={key:True for key in candidate_directions} if structural_available is None else dict(structural_available);req(set(availability)==set(candidate_directions),"structural availability population")
    if not any(availability.values()):return QueryVerificationDecision(HOLD_STRUCTURAL,())
    for key,directions in candidate_directions.items():
        req(isinstance(key,str) and key and isinstance(directions,tuple) and len(directions)==2,"candidate directions")
        if availability[key] and float(verification_evidence(*directions).detach())>0:verified.append(key)
    ordered=tuple(sorted(verified))
    if not ordered:return QueryVerificationDecision(HOLD,ordered)
    req(ambiguity_status in (STATUS_SINGLE,STATUS_MULTI_TARGET,STATUS_MULTI_REGION_SAME_CANDIDATE),"unqualified ambiguity status")
    if ambiguity_status in (STATUS_MULTI_TARGET,STATUS_MULTI_REGION_SAME_CANDIDATE):return QueryVerificationDecision(PRIMARY_UNDETERMINED_HOLD,ordered)
    return QueryVerificationDecision(SINGLE_VERIFIED if len(ordered)==1 else MULTI_VERIFIED,ordered)

def group_balanced_weights(supergroups:tuple[str,...],*,dtype:torch.dtype=torch.float64)->torch.Tensor:
    req(bool(supergroups) and all(isinstance(x,str) and x for x in supergroups),"supergroups");counts={g:supergroups.count(g) for g in set(supergroups)};groups=len(counts);return torch.tensor([1.0/(groups*counts[g]) for g in supergroups],dtype=dtype)
def verification_evaluation_weights(query_ids:tuple[str,...],supergroups:tuple[str,...],labels:torch.Tensor)->torch.Tensor:
    y=torch.as_tensor(labels,dtype=torch.float64);req(len(query_ids)==len(supergroups)==y.numel() and bool(((y==0)|(y==1)).all()),"evaluation population");queries=sorted(set(query_ids));groups=sorted(set(supergroups));req(queries and groups,"evaluation groups");query_group={}
    for query,group in zip(query_ids,supergroups,strict=True):
        prior=query_group.setdefault(query,group);req(prior==group,"query crosses supergroups")
    group_queries={group:sorted(query for query in queries if query_group[query]==group) for group in groups};weights=torch.zeros(y.numel(),dtype=torch.float64)
    for group in groups:
        qrows=group_queries[group];req(qrows,"empty group")
        for query in qrows:
            indices=[i for i,value in enumerate(query_ids) if value==query];positive=[i for i in indices if float(y[i])==1.0];negative=[i for i in indices if float(y[i])==0.0];req(len(positive)==1 and negative,"each query needs one positive and >=1 null")
            query_mass=1.0/(len(groups)*len(qrows));weights[positive[0]]=0.5*query_mass
            for index in negative:weights[index]=0.5*query_mass/len(negative)
    req(abs(float(weights.sum())-1.0)<1e-12,"evaluation weight sum");return weights
def group_balanced_log_loss(probabilities:torch.Tensor,labels:torch.Tensor,query_ids:tuple[str,...],supergroups:tuple[str,...])->torch.Tensor:
    p=torch.as_tensor(probabilities,dtype=torch.float64);y=torch.as_tensor(labels,dtype=torch.float64);req(p.ndim==y.ndim==1 and p.shape==y.shape==(len(supergroups),) and len(query_ids)==p.numel() and bool((p>0).all()) and bool((p<1).all()) and bool(((y==0)|(y==1)).all()),"logloss population");w=verification_evaluation_weights(query_ids,supergroups,y);return -(w*(y*torch.log(p)+(1-y)*torch.log1p(-p))).sum()
def verification_log_loss(logits:torch.Tensor,labels:torch.Tensor,query_ids:tuple[str,...],supergroups:tuple[str,...])->torch.Tensor:
    z=torch.as_tensor(logits,dtype=torch.float64);y=torch.as_tensor(labels,dtype=torch.float64);req(z.ndim==y.ndim==1 and z.shape==y.shape==(len(supergroups),) and len(query_ids)==z.numel() and bool(torch.isfinite(z).all()) and bool(((y==0)|(y==1)).all()),"verification logloss population");w=verification_evaluation_weights(query_ids,supergroups,y);return (w*(torch.nn.functional.softplus(z)-y*z)).sum()
def fixed_bin_ece(probabilities:torch.Tensor,labels:torch.Tensor,query_ids:tuple[str,...],supergroups:tuple[str,...])->torch.Tensor:
    p=torch.as_tensor(probabilities,dtype=torch.float64);y=torch.as_tensor(labels,dtype=torch.float64);req(p.shape==y.shape==(len(supergroups),) and len(query_ids)==p.numel() and bool((p>=0).all()) and bool((p<=1).all()),"ECE population");w=verification_evaluation_weights(query_ids,supergroups,y);total=p.new_zeros(())
    for index in range(10):
        low=index/10;high=(index+1)/10;mask=(p>=low)&((p<high) if index<9 else (p<=high))
        if bool(mask.any()):
            mass=w[mask].sum();confidence=(w[mask]*p[mask]).sum()/mass;accuracy=(w[mask]*y[mask]).sum()/mass;total=total+mass*torch.abs(confidence-accuracy)
    return total

def calibration_intercept_slope(logits:torch.Tensor,labels:torch.Tensor,query_ids:tuple[str,...],supergroups:tuple[str,...],*,max_iterations:int=100,tolerance:float=1e-10)->tuple[float,float]:
    z=torch.as_tensor(logits,dtype=torch.float64);y=torch.as_tensor(labels,dtype=torch.float64);req(z.shape==y.shape==(len(supergroups),) and len(query_ids)==z.numel() and bool(torch.isfinite(z).all()) and bool(((y==0)|(y==1)).all()),"calibration population");w=verification_evaluation_weights(query_ids,supergroups,y);x=torch.stack((torch.ones_like(z),z),dim=1);beta=torch.tensor([0.0,1.0],dtype=torch.float64);converged=False
    for _ in range(max_iterations):
        p=torch.sigmoid(x@beta);gradient=x.T@(w*(p-y));curvature=w*p*(1-p);hessian=x.T@(curvature[:,None]*x);det=torch.linalg.det(hessian);req(bool(torch.isfinite(hessian).all()) and float(torch.abs(det))>1e-14,"calibration Hessian singular");step=torch.linalg.solve(hessian,gradient);beta=beta-step
        req(bool(torch.isfinite(beta).all()),"calibration nonfinite")
        if float(torch.max(torch.abs(step)))<=tolerance:converged=True;break
    req(converged,"calibration did not converge");return float(beta[0]),float(beta[1])

__all__=["CANONICAL_ARM","UNVERIFIED_H0","VERIFIED_H1","HOLD","HOLD_STRUCTURAL","SINGLE_VERIFIED","MULTI_VERIFIED","PAIR_COMPETITION_REQUIRED","PRIMARY_UNDETERMINED_HOLD","VerificationNullError","fixed_denominator_unique_patch_unary","verification_evidence","verification_probability","QueryVerificationDecision","decide_query","group_balanced_weights","verification_evaluation_weights","group_balanced_log_loss","verification_log_loss","fixed_bin_ece","calibration_intercept_slope"]
