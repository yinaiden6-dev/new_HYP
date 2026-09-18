"""P-only legal-family selector with identical train and deploy semantics."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

import torch
from torch import nn


BASE_PATH = Path(__file__).with_name("reference_conditioned_latent_proposal_p_only_v1.py")
BASE_SHA256 = "98ea07177779433ef42df454f787a153592ef5f3004fd894ef1a11dfcc37947e"
FEATURE_DIM = 9
PARAMETER_COUNT = 10
MIN_QUERY_PATCHES = 4
MIN_REFERENCE_CELLS = 2
REFERENCE_CHEBYSHEV = 2
STRUCTURAL_H0_REASON = "LEGAL_HYPOTHESIS_FAMILY_EMPTY"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_base() -> ModuleType:
    if _sha_file(BASE_PATH) != BASE_SHA256:
        raise RuntimeError("frozen P base hash drift")
    name = "rc_lth_p_only_legal_family_base_v1"
    spec = importlib.util.spec_from_file_location(name, BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen P base import failed")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base()
ProposalAtomBank = _base.ProposalAtomBank
validate_atom_bank = _base.validate_atom_bank
seal_atom_bank = _base.seal_atom_bank
coordinate_destroy_atom_bank = _base.coordinate_destroy_atom_bank
query_only_features = _base.query_only_features
tensor_sha256 = _base.tensor_sha256


def logical_sha256(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _fixed_tetrominoes() -> tuple[tuple[tuple[int, int], ...], ...]:
    shapes: set[frozenset[tuple[int, int]]] = {frozenset(((0, 0),))}
    for _ in range(MIN_QUERY_PATCHES - 1):
        grown: set[frozenset[tuple[int, int]]] = set()
        for shape in shapes:
            for row, col in shape:
                for cell in ((row-1,col),(row+1,col),(row,col-1),(row,col+1)):
                    if cell not in shape:
                        candidate=set(shape);candidate.add(cell);mr=min(x[0] for x in candidate);mc=min(x[1] for x in candidate)
                        grown.add(frozenset((x[0]-mr,x[1]-mc) for x in candidate))
        shapes=grown
    output=tuple(sorted(tuple(sorted(shape)) for shape in shapes))
    if len(output)!=19: raise RuntimeError("tetromino family drift")
    return output


FIXED_TETROMINOES = _fixed_tetrominoes()


@dataclass(frozen=True)
class LegalHypothesisSeed:
    canonical_seed_ordinal: int
    atom_indices: torch.Tensor
    query_indices: torch.Tensor
    reference_indices: torch.Tensor
    query_coordinates: torch.Tensor
    reference_coordinates: torch.Tensor
    logical_sha256: str


@dataclass(frozen=True)
class ScoredLegalHypothesis:
    slot: int
    seed: LegalHypothesisSeed
    score: torch.Tensor

    @property
    def logical_sha256(self) -> str:
        return self.seed.logical_sha256


@dataclass(frozen=True)
class LegalFamilyDecision:
    query_resource_key: str
    candidate_resource_key: str
    reference_resource_key: str
    hypotheses: tuple[ScoredLegalHypothesis, ...]
    candidate_score: torch.Tensor
    h1_decision: bool
    h0_reason: str | None
    legal_seed_count: int
    legal_seed_atom_indices: torch.Tensor
    legal_seed_axis_sha256: str
    complete_seed_scores: torch.Tensor
    complete_seed_scores_sha256: str
    map_seed_ordinal: int | None
    map_seed_sha256: str | None
    map_hypothesis: ScoredLegalHypothesis | None
    assignment_sha256: str
    source_assignment_sha256: str
    coordinate_binding_sha256: str


@dataclass(frozen=True)
class LegalFamilyCache:
    assignment_sha256: str
    legal_seed_axis_sha256: str
    legal_seed_atom_indices: torch.Tensor
    seed_mean_features: torch.Tensor


_FAMILY_CACHE: dict[str, LegalFamilyCache] = {}


def _paired_connected(q: torch.Tensor, r: torch.Tensor) -> bool:
    neighbours=[set() for _ in range(4)]
    for left in range(4):
        for right in range(left+1,4):
            if int((q[left]-q[right]).abs().sum())==1 and int((r[left]-r[right]).abs().max())<=REFERENCE_CHEBYSHEV:
                neighbours[left].add(right);neighbours[right].add(left)
    seen={0};stack=[0]
    while stack:
        current=stack.pop()
        for nxt in neighbours[current]:
            if nxt not in seen:seen.add(nxt);stack.append(nxt)
    return len(seen)==4


def _legal_atom_axis(bank: ProposalAtomBank) -> torch.Tensor:
    valid=torch.as_tensor(bank.valid_mask,dtype=torch.bool); qrc=torch.as_tensor(bank.query_rc,dtype=torch.long); rrc=torch.as_tensor(bank.reference_rc,dtype=torch.long)
    rows_by_cell={tuple(int(v) for v in qrc[row].tolist()):row for row in torch.nonzero(valid,as_tuple=False).flatten().tolist()}
    height,width=bank.query_grid_shape; candidates=[]
    for shape in FIXED_TETROMINOES:
        sh=max(x[0] for x in shape)+1;sw=max(x[1] for x in shape)+1
        for row in range(height-sh+1):
            for col in range(width-sw+1):
                cells=tuple(sorted((row+dr,col+dc) for dr,dc in shape))
                if not all(cell in rows_by_cell for cell in cells):continue
                candidates.append(tuple(rows_by_cell[cell] for cell in cells))
    if candidates:
        candidate_axis=torch.tensor(candidates,dtype=torch.long);refs=torch.as_tensor(bank.reference_indices,dtype=torch.long)[candidate_axis];ordered_refs=torch.sort(refs,dim=1).values;distinct=(ordered_refs[:,1:]!=ordered_refs[:,:-1]).any(dim=1);sq=qrc[candidate_axis];sr=rrc[candidate_axis];q_near=(sq[:,:,None,:]-sq[:,None,:,:]).abs().sum(dim=-1)<=1;r_near=(sr[:,:,None,:]-sr[:,None,:,:]).abs().amax(dim=-1)<=REFERENCE_CHEBYSHEV;reach=q_near&r_near
        for _ in range(3):reach=reach|(torch.bmm(reach.to(torch.int8),reach.to(torch.int8))>0)
        keep=distinct&reach[:,0,:].all(dim=1);candidate_axis=candidate_axis[keep];qindices=torch.as_tensor(bank.query_indices,dtype=torch.long)[candidate_axis];order=sorted(range(candidate_axis.shape[0]),key=lambda i:tuple(int(x) for x in qindices[i].tolist()));candidates=[tuple(int(x) for x in candidate_axis[i].tolist()) for i in order]
    else:candidates=[]
    return torch.tensor(candidates,dtype=torch.long).reshape(-1,4).contiguous()


def _seed_hash(assignment_sha256:str,axis:torch.Tensor)->str:
    return hashlib.sha256(bytes.fromhex(assignment_sha256)+torch.as_tensor(axis,dtype=torch.long).cpu().contiguous().numpy().tobytes()).hexdigest()


def enumerate_legal_hypotheses(bank: ProposalAtomBank) -> tuple[LegalHypothesisSeed, ...]:
    """The sole legal-family enumerator used by training and deployment."""
    bank=validate_atom_bank(bank);atom_axis=_legal_atom_axis(bank);seeds=[]
    for ordinal,axis in enumerate(atom_axis):
        seeds.append(LegalHypothesisSeed(ordinal,axis,bank.query_indices[axis],bank.reference_indices[axis],bank.query_rc[axis],bank.reference_rc[axis],_seed_hash(bank.assignment_sha256,axis)))
    return tuple(seeds)


def _validate_head(head: nn.Module) -> None:
    parameters=tuple(head.parameters())
    if sum(x.numel() for x in parameters)!=PARAMETER_COUNT or any(x.dtype!=torch.float64 for x in parameters):raise ValueError("legal-family head must be ten FP64 parameters")


def build_legal_family_cache(bank: ProposalAtomBank) -> LegalFamilyCache:
    bank=validate_atom_bank(bank);atom_axis=_legal_atom_axis(bank)
    means=bank.features[atom_axis].to(torch.float64).mean(dim=1).contiguous() if atom_axis.numel() else torch.empty((0,FEATURE_DIM),dtype=torch.float64)
    axis_sha=logical_sha256({"assignment_sha256":bank.assignment_sha256,"atom_axis_sha256":tensor_sha256(atom_axis)})
    return LegalFamilyCache(bank.assignment_sha256,axis_sha,atom_axis,means)


def install_legal_family_cache(values: Sequence[LegalFamilyCache]) -> None:
    for value in values:
        expected=logical_sha256({"assignment_sha256":value.assignment_sha256,"atom_axis_sha256":tensor_sha256(value.legal_seed_atom_indices)})
        if value.legal_seed_axis_sha256!=expected or value.legal_seed_atom_indices.ndim!=2 or value.legal_seed_atom_indices.shape[1]!=4 or value.seed_mean_features.shape!=(value.legal_seed_atom_indices.shape[0],FEATURE_DIM):raise ValueError("legal-family cache invalid")
        prior=_FAMILY_CACHE.get(value.assignment_sha256)
        if prior is not None and (prior.legal_seed_axis_sha256!=value.legal_seed_axis_sha256 or not torch.equal(prior.seed_mean_features,value.seed_mean_features)):raise ValueError("legal-family cache collision")
        _FAMILY_CACHE[value.assignment_sha256]=value


def _cache(bank: ProposalAtomBank) -> LegalFamilyCache:
    value=_FAMILY_CACHE.get(bank.assignment_sha256)
    if value is None:value=build_legal_family_cache(bank);_FAMILY_CACHE[bank.assignment_sha256]=value
    return value


def score_legal_hypothesis_family(bank: ProposalAtomBank, head: nn.Module) -> LegalFamilyDecision:
    _validate_head(head);cached=_FAMILY_CACHE.get(bank.assignment_sha256)
    if cached is None:bank=validate_atom_bank(bank);cached=_cache(bank)
    device=next(head.parameters()).device
    if cached.legal_seed_atom_indices.shape[0]:
        scores=head(cached.seed_mean_features.to(device)).flatten();family=torch.logsumexp(scores,dim=0)-math.log(scores.numel());maximum=scores.detach().max();tied=[i for i,x in enumerate(scores.detach()) if bool(x==maximum)];chosen=min(tied,key=lambda i:tuple(int(x) for x in cached.legal_seed_atom_indices[i].tolist()));axis=cached.legal_seed_atom_indices[chosen];seed_hash=_seed_hash(bank.assignment_sha256,axis);seed=LegalHypothesisSeed(chosen,axis,bank.query_indices[axis],bank.reference_indices[axis],bank.query_rc[axis],bank.reference_rc[axis],seed_hash);mapped=ScoredLegalHypothesis(chosen+1,seed,scores[chosen]);hypotheses=(mapped,);reason=None
    else:
        zero=sum((p.sum()*0.0 for p in head.parameters()),torch.zeros((),dtype=torch.float64,device=device));scores=torch.empty(0,dtype=torch.float64,device=device);family=zero;hypotheses=();mapped=None;chosen=None;reason=STRUCTURAL_H0_REASON
    return LegalFamilyDecision(bank.query_resource_key,bank.candidate_resource_key,bank.reference_resource_key,hypotheses,family,bool(cached.legal_seed_atom_indices.shape[0]),reason,int(cached.legal_seed_atom_indices.shape[0]),cached.legal_seed_atom_indices,cached.legal_seed_axis_sha256,scores,tensor_sha256(scores.detach().cpu().contiguous()),chosen,None if mapped is None else mapped.seed.logical_sha256,mapped,bank.assignment_sha256,bank.source_assignment_sha256,bank.coordinate_binding_sha256)


def deploy_legal_hypothesis_family(bank: ProposalAtomBank, head: nn.Module) -> LegalFamilyDecision:
    return score_legal_hypothesis_family(bank, head)


class ConnectedProposalSelector(nn.Module):
    def __init__(self)->None:
        super().__init__();self.head=nn.Linear(FEATURE_DIM,1,bias=True,dtype=torch.float64)
        with torch.no_grad():self.head.weight.zero_();self.head.bias.zero_()
    def forward(self,bank:ProposalAtomBank)->LegalFamilyDecision:return score_legal_hypothesis_family(bank,self.head)


class AllPatchNoHypComparator(nn.Module):
    def __init__(self)->None:
        super().__init__();self.head=nn.Linear(FEATURE_DIM,1,bias=True,dtype=torch.float64)
        with torch.no_grad():self.head.weight.zero_();self.head.bias.zero_()
    def forward(self,bank:ProposalAtomBank)->torch.Tensor:
        device=self.head.weight.device;valid=torch.as_tensor(bank.valid_mask,dtype=torch.bool,device=device);zero=sum((p.sum()*0.0 for p in self.parameters()),torch.zeros((),dtype=torch.float64,device=device))
        if not bool(valid.any()):return zero
        features=torch.as_tensor(bank.features,dtype=torch.float64,device=device)[valid];evidence=torch.as_tensor(bank.signed_evidence,dtype=torch.float64,device=device)[valid];weights=torch.softmax(self.head(features).flatten(),dim=0);return (weights*evidence).sum()


def score_reference_axis(selector:ConnectedProposalSelector,banks:Sequence[ProposalAtomBank]):
    query={x.query_resource_key for x in banks};keys=[x.candidate_resource_key for x in banks]
    if len(query)!=1 or len(keys)!=len(set(keys)):raise ValueError("candidate axis invalid")
    proposals=tuple(selector(x) for x in banks);return torch.stack([x.candidate_score for x in proposals]),proposals


@dataclass(frozen=True)
class QueryOnlyLegalSelection:
    groups: tuple[tuple[int,...],...]
    weights: torch.Tensor
    features: torch.Tensor
    group_axis: torch.Tensor
    axis_sha256: str


class QueryOnlyRegionComparator(nn.Module):
    def __init__(self)->None:
        super().__init__();self.head=nn.Linear(FEATURE_DIM,1,bias=True,dtype=torch.float64)
        with torch.no_grad():self.head.weight.zero_();self.head.bias.zero_()
    def select(self,query_resource_key:str,features:torch.Tensor,query_rc:torch.Tensor,valid_mask:torch.Tensor)->QueryOnlyLegalSelection:
        f=torch.as_tensor(features,dtype=torch.float64);rc=torch.as_tensor(query_rc,dtype=torch.long);valid=torch.as_tensor(valid_mask,dtype=torch.bool);by_cell={tuple(int(v) for v in rc[i].tolist()):i for i in torch.nonzero(valid,as_tuple=False).flatten().tolist()};h=int(rc[:,0].max())+1;w=int(rc[:,1].max())+1;groups=[]
        for shape in FIXED_TETROMINOES:
            sh=max(x[0] for x in shape)+1;sw=max(x[1] for x in shape)+1
            for row in range(h-sh+1):
                for col in range(w-sw+1):
                    cells=tuple(sorted((row+dr,col+dc) for dr,dc in shape))
                    if all(x in by_cell for x in cells):groups.append(tuple(by_cell[x] for x in cells))
        groups=sorted(set(groups));axis=torch.tensor(groups,dtype=torch.long).reshape(-1,4);weights=torch.ones(f.shape[0],dtype=torch.float64,device=self.head.weight.device);return QueryOnlyLegalSelection((),weights,f,axis,logical_sha256([list(x) for x in groups]))
    def score(self,bank:ProposalAtomBank,selection:QueryOnlyLegalSelection)->torch.Tensor:
        device=self.head.weight.device
        if selection.group_axis.numel()==0:return sum((p.sum()*0.0 for p in self.parameters()),torch.zeros((),dtype=torch.float64,device=device))
        inverse=torch.empty(bank.query_indices.numel(),dtype=torch.long);inverse[torch.as_tensor(bank.query_indices,dtype=torch.long)]=torch.arange(bank.query_indices.numel());qlogits=self.head(selection.features.to(device)).flatten();e=torch.as_tensor(bank.signed_evidence,dtype=torch.float64,device=device);cells=selection.group_axis.to(device);rows=inverse[cells.cpu()].to(device);scores=(qlogits[cells]+e[rows]).mean(dim=1);return torch.logsumexp(scores,dim=0)-math.log(scores.numel())


def serialize_family_decision(value:LegalFamilyDecision)->dict[str,Any]:
    return {"candidate_resource_key":value.candidate_resource_key,"legal_seed_count":value.legal_seed_count,"legal_seed_axis_sha256":value.legal_seed_axis_sha256,"legal_seed_atom_indices_sha256":tensor_sha256(value.legal_seed_atom_indices),"complete_seed_scores_binary64":[float(x).hex() for x in value.complete_seed_scores.detach().cpu()],"complete_seed_scores_sha256":value.complete_seed_scores_sha256,"family_score_binary64":float(value.candidate_score.detach().cpu()).hex(),"map_seed_ordinal":value.map_seed_ordinal,"map_seed_sha256":value.map_seed_sha256,"state":"H1" if value.h1_decision else "STRUCTURAL_H0","h0_reason":value.h0_reason,"assignment_sha256":value.assignment_sha256}


def validate_serialized_family_decision(value:Mapping[str,Any],bank:ProposalAtomBank,head:nn.Module)->dict[str,Any]:
    replay=deploy_legal_hypothesis_family(bank,head);expected=serialize_family_decision(replay)
    if dict(value)!=expected:raise ValueError("serialized legal-family parity mismatch")
    return expected


__all__=["ProposalAtomBank","validate_atom_bank","seal_atom_bank","coordinate_destroy_atom_bank","query_only_features","AllPatchNoHypComparator","tensor_sha256","LegalHypothesisSeed","ScoredLegalHypothesis","LegalFamilyDecision","LegalFamilyCache","enumerate_legal_hypotheses","build_legal_family_cache","install_legal_family_cache","score_legal_hypothesis_family","deploy_legal_hypothesis_family","serialize_family_decision","validate_serialized_family_decision","ConnectedProposalSelector","score_reference_axis","QueryOnlyRegionComparator","STRUCTURAL_H0_REASON"]
