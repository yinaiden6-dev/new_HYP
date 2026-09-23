"""Controlled paths into the overlap predictor, with native geometry held fixed."""
import itertools
import torch
from torch.nn import functional as F
from rc_roma_visual_origin_v1 import Observer as NativeObserver

FACTORS = {f'A{a}J{j}P{p}':(a,j,p) for a,j,p in itertools.product((0,1),repeat=3)}
EXTRA = ('NO_COARSE_LOGIT','NO_LR_DELTA','NO_HR_DELTA','NO_REFINER_DELTA')
BRANCHES = tuple(FACTORS)+EXTRA


class Observer(NativeObserver):
    def __init__(self,model,core,input_sink,output_sink):
        super().__init__(model,core,input_sink)
        import romav2.matcher as matcher
        from romav2.geometry import bhwc_interpolate
        self.matcher_module=matcher;self.interpolate=bhwc_interpolate
        self.original_head=matcher._compute_head_preds
        self.output_sink=output_sink;self.head_calls=0;self.states={};self.inside={};self.raw_delta={}
        matcher._compute_head_preds=self.head
        for stride,module in model.refiners.items():
            self.handles.append(module.confidence_head.register_forward_hook(self.delta(stride)))

    def select(self,*args,**kwargs):
        super().select(*args,**kwargs)
        self.head_calls=0;self.states={};self.inside={};self.raw_delta={}

    @torch.inference_mode()
    def head(self,**kwargs):
        side=self.head_calls;self.head_calls+=1;assert side in (0,1)
        originals=list(kwargs['f_list_A'])
        native=self.original_head(**kwargs)
        states={};coarse={}
        for name,(a,j,p) in FACTORS.items():
            inputs=dict(kwargs,f_list_A=[v.clone() if a else torch.zeros_like(v) for v in originals],
                f_mv_A=kwargs['f_mv_A'] if j else torch.zeros_like(kwargs['f_mv_A']),
                match_emb_AB=kwargs['match_emb_AB'] if p else torch.zeros_like(kwargs['match_emb_AB']))
            warp,confidence=self.original_head(**inputs)
            if name=='A1J1P1':
                assert torch.equal(warp,native[0]) and torch.equal(confidence,native[1]), 'HEAD_FACTOR_NATIVE_PARITY'
            states[name]=confidence.detach().clone()
        states.update({name:torch.zeros_like(native[1]) if name=='NO_COARSE_LOGIT' else native[1].detach().clone() for name in EXTRA})
        for name,confidence in states.items():coarse[name]=self.record(confidence,side,True)
        self.states[side]=states
        components=dict(appearance=originals[-1],joint_context=kwargs['f_mv_A'],matching_position=kwargs['match_emb_AB'])
        # Scalar/space summaries for every pair; first query/reference embeddings
        # are additionally retained by the outer snapshot policy.
        summaries={k:dict(shape=list(v.shape),rms=float(v.double().square().mean().sqrt()),
            spatial_rms16=F.interpolate(v.float().square().mean(-1)[None],size=(16,16),mode='area')[0].sqrt().cpu()) for k,v in components.items()}
        self.inside[side]=dict(coarse=coarse,stages={},components=summaries,
            first_pair_components={k:v.detach().cpu().clone() for k,v in components.items()} if self.output_sink.capture_full else None)
        return native

    def delta(self,stride):
        def callback(module,args,output):self.raw_delta[stride]=output[:,0:1].permute(0,2,3,1).detach().clone()
        return callback

    def record(self,confidence,side,retain):
        overlap=confidence[0,...,0].sigmoid().detach().cpu()
        weights=self.core.cell_means(overlap,self.geometry[side])
        out=dict(mean_weight=float(weights.mean()),logit_mean=float(confidence[...,0].double().mean()))
        if retain:out.update(weights=weights,overlap_grid16=F.interpolate(overlap[None,None],size=(16,16),mode='area')[0,0])
        return out

    def refiner(self,stride):
        native_callback=super().refiner(stride)
        def callback(module,args,output):
            count=self.calls['refiner'+stride];side=count%2;stage=('LR' if count<2 else 'HR')+stride
            delta=self.raw_delta[stride]
            native_callback(module,args,output)
            saved={}
            for name,state in self.states[side].items():
                previous=self.interpolate(state,size=output['confidence'].shape[1:3],mode='bilinear',align_corners=False)
                use_delta=not (name=='NO_REFINER_DELTA' or name=='NO_LR_DELTA' and stage.startswith('LR') or name=='NO_HR_DELTA' and stage.startswith('HR'))
                logit=previous[...,:1]+delta if use_delta else previous[...,:1]
                confidence=output['confidence'].detach().clone();confidence[...,:1]=logit
                if name=='A1J1P1':assert torch.equal(confidence[...,:1],output['confidence'][...,:1]), ('NATIVE_CONFIDENCE_RECONSTRUCTION',stage)
                self.states[side][name]=confidence
                saved[name]=self.record(confidence,side,stage=='HR1')
            self.inside[side]['stages'][stage]=dict(branches=saved,
                native_delta_mean=float(delta.double().mean()),native_delta_rms=float(delta.double().square().mean().sqrt()),
                native_delta_grid16=F.interpolate(delta.permute(0,3,1,2),size=(16,16),mode='area')[0,0].cpu())
        return callback

    @torch.inference_mode()
    def forward(self,*args,**kwargs):
        result=super().forward(*args,**kwargs)
        assert self.head_calls==2 and set(self.inside)=={0,1}
        self.output_sink(self.identity,self.inside)
        return result

    def close(self):
        self.matcher_module._compute_head_preds=self.original_head
        super().close()


def self_test():
    assert len(FACTORS)==8 and len(BRANCHES)==12
    # Validate algebra of independent confidence increments before GPU parity.
    x=torch.tensor([-2.,0.,3.]);d1=torch.tensor([.5,-1.,.2]);d2=torch.tensor([.2,.3,-.1])
    native=(x+d1)+d2
    assert torch.equal(native,(x+d1)+d2)
    assert torch.equal(torch.zeros_like(x)+d1+d2,d1+d2)
    assert torch.equal((x+d1)+torch.zeros_like(d2),x+d1)
    return dict(status='VISUAL_ORIGIN_TRANSFORMS_PASS',branches=list(BRANCHES),
                meaning='Direct predictor paths and additive confidence branches; actual model equality requires GPU pilot')
