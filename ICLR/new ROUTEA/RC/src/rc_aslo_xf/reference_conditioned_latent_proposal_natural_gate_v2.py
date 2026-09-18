"""Post-hoc TRAIN32 gate using one sign-independent legal-family P score."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Sequence

import torch
from torch.nn import functional as F


BASE_PATH=Path(__file__).with_name("reference_conditioned_latent_proposal_natural_gate_v1.py")
BASE_SHA256="30c75700018dd7086ef9f949bd92cc3ac5a3246dab8e121ef5a89f04cee0d282"


def _load_base()->ModuleType:
    if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()!=BASE_SHA256:raise RuntimeError("natural gate V1 hash drift")
    name="rc_lth_p_only_natural_gate_base_v1";spec=importlib.util.spec_from_file_location(name,BASE_PATH)
    if spec is None or spec.loader is None:raise RuntimeError("natural gate V1 import failed")
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


_base=_load_base()
TRAIN_QUERY_COUNT=_base.TRAIN_QUERY_COUNT;CANDIDATE_COUNT=_base.CANDIDATE_COUNT;TRAIN_UPDATES=_base.TRAIN_UPDATES;TRAIN_SEED=_base.TRAIN_SEED;LEARNING_RATE=_base.LEARNING_RATE;ADAM_BETAS=_base.ADAM_BETAS;ADAM_EPS=_base.ADAM_EPS
ARM_REAL=_base.ARM_REAL;ARM_ALLPATCH=_base.ARM_ALLPATCH;ARM_QUERY_ONLY=_base.ARM_QUERY_ONLY;ARM_NAMES=_base.ARM_NAMES;C_BIND_NAMESPACE=_base.C_BIND_NAMESPACE;P_COORD_NAMESPACE_PREFIX=_base.P_COORD_NAMESPACE_PREFIX
AnonymousNaturalEpisode=_base.AnonymousNaturalEpisode;PostjoinProposalRole=_base.PostjoinProposalRole;ThreeArmProposalHeads=_base.ThreeArmProposalHeads;ArmForward=_base.ArmForward;EpisodeForward=_base.EpisodeForward;EpisodeStatistic=_base.EpisodeStatistic;PairedComparison=_base.PairedComparison;GateReduction=_base.GateReduction
make_optimizers=_base.make_optimizers;deterministic_training_order=_base.deterministic_training_order;score_episode=_base.score_episode;detach_episode_statistic=_base.detach_episode_statistic
arm_forward_from_scores=_base.arm_forward_from_scores
FAILURE_PRECEDENCE=("P_GENERATION_COVERAGE_NO_GO","P_NONSELECTIVE_RANK_NO_GO","P_RERANKER_SHORTCUT_NO_GO","P_CANDIDATE_BINDING_NO_GO","P_CANDIDATE_BOUND_NONSPATIAL")
_validated_episode_bindings:set[str]=set()
_original_validate_control_axes=_base._validate_control_axes


def _validate_control_axes_once(episode:Any,p_core:ModuleType)->None:
    key=hashlib.sha256((episode.query_resource_key+episode.c_bind_plan_sha256+episode.p_coord_derivation_sha256).encode()).hexdigest()
    if key not in _validated_episode_bindings:_original_validate_control_axes(episode,p_core);_validated_episode_bindings.add(key)


_base._validate_control_axes=_validate_control_axes_once


@dataclass(frozen=True)
class RelativePOnlyLoss:
    total:torch.Tensor
    full_c128_ce:torch.Tensor
    target_vs_strongest_wrong:torch.Tensor
    c_bind_margin:torch.Tensor
    p_coord_margin:torch.Tensor
    missing_witness:bool=False


def five_term_loss(arm:ArmForward,target_indices:Sequence[int])->RelativePOnlyLoss:
    target=torch.tensor(tuple(target_indices),dtype=torch.long,device=arm.scores.device)
    if target.numel()==0:raise ValueError("relative P loss requires a target destination")
    ce=torch.logsumexp(arm.scores,dim=0)-torch.logsumexp(arm.scores[target],dim=0)
    rank=F.softplus(-arm.margin)
    bind=F.softplus(-(arm.margin-arm.c_bind_margin))
    coord=F.softplus(arm.margin-arm.margin) if arm.name==ARM_ALLPATCH else F.softplus(-(arm.margin-arm.p_coord_margin))
    total=ce+rank+bind+coord
    if not bool(torch.isfinite(total)):raise FloatingPointError("relative P-only loss nonfinite")
    return RelativePOnlyLoss(total,ce,rank,bind,coord,False)


def _mean(values:Sequence[float])->float:
    if not values:raise ValueError("empty mean")
    return math.fsum(float(x) for x in values)/len(values)


def _group(rows:Sequence[EpisodeStatistic],fn)->float:
    groups:dict[str,list[float]]={}
    for row in rows:groups.setdefault(row.supergroup,[]).append(float(fn(row)))
    return _mean([_mean(v) for _,v in sorted(groups.items())])


def _paired(rows:Sequence[EpisodeStatistic],name:str)->PairedComparison:
    fn=(lambda r:r.allpatch_margin) if name==ARM_ALLPATCH else (lambda r:r.query_only_margin)
    rescue=sum(r.real_margin>0 and fn(r)<=0 for r in rows);breaks=sum(r.real_margin<=0 and fn(r)>0 for r in rows)
    return PairedComparison(rescue,breaks,rescue-breaks,_mean([r.real_margin-fn(r) for r in rows]))


def reduce_natural_gate(statistics:Sequence[EpisodeStatistic],*,stage:str)->GateReduction:
    if stage!="optimization":raise ValueError("V3 authorizes post-hoc optimization only; formal panel requires a new authority")
    rows=tuple(statistics)
    if len(rows)!=32 or len({r.query_resource_key for r in rows})!=32 or any(r.fold!=1 or r.candidate_count!=128 for r in rows):raise ValueError("TRAIN32 full-C128 axis invalid")
    if any(not r.target_present for r in rows):raise ValueError("TRAIN32 target absence invalid")
    numeric=[x for r in rows for x in (r.real_margin,r.allpatch_margin,r.query_only_margin,r.c_bind_margin,r.p_coord_margin)]
    if not all(math.isfinite(x) for x in numeric):raise FloatingPointError("gate statistic nonfinite")
    raw=sum(r.raw_target_atom_present for r in rows);connected=sum(r.selected_target_connected_h1_present for r in rows)
    if any(r.selected_target_positive_h1_present!=r.selected_target_connected_h1_present for r in rows):raise ValueError("score sign affected structural H1 state")
    success=sum(r.real_margin>0 for r in rows);failure=32-success;group_real=_group(rows,lambda r:r.real_margin)
    c_drops=[r.real_margin-r.c_bind_margin for r in rows];p_drops=[r.real_margin-r.p_coord_margin for r in rows];c_count=sum(x>0 for x in c_drops);p_count=sum(x>0 for x in p_drops);group_c=_group(rows,lambda r:r.real_margin-r.c_bind_margin);group_p=_group(rows,lambda r:r.real_margin-r.p_coord_margin)
    allpatch=_paired(rows,ARM_ALLPATCH);query=_paired(rows,ARM_QUERY_ONLY);binding=c_count>=21 and group_c>0;coordinate=p_count>=21 and group_p>0
    flags={"P_GENERATION_COVERAGE_NO_GO":connected<26,"P_NONSELECTIVE_RANK_NO_GO":success<21 or group_real<=0 or success<=failure,"P_RERANKER_SHORTCUT_NO_GO":any(x.paired_net<4 or x.rescue<=x.break_count for x in (allpatch,query)),"P_CANDIDATE_BINDING_NO_GO":not binding,"P_CANDIDATE_BOUND_NONSPATIAL":binding and not coordinate}
    failures=tuple(x for x in FAILURE_PRECEDENCE if flags[x]);fraction=sum(r.h1_candidate_count for r in rows)/(32.0*128.0)
    return GateReduction(stage,not failures,failures,(),32,raw,connected,connected,success,failure,group_real,c_count,group_c,p_count,group_p,fraction,allpatch,query,{})


__all__=["TRAIN_QUERY_COUNT","CANDIDATE_COUNT","TRAIN_UPDATES","TRAIN_SEED","LEARNING_RATE","ADAM_BETAS","ADAM_EPS","ARM_REAL","ARM_ALLPATCH","ARM_QUERY_ONLY","ARM_NAMES","C_BIND_NAMESPACE","P_COORD_NAMESPACE_PREFIX","FAILURE_PRECEDENCE","AnonymousNaturalEpisode","PostjoinProposalRole","ThreeArmProposalHeads","ArmForward","EpisodeForward","EpisodeStatistic","PairedComparison","GateReduction","make_optimizers","deterministic_training_order","score_episode","detach_episode_statistic","arm_forward_from_scores","RelativePOnlyLoss","five_term_loss","reduce_natural_gate"]
