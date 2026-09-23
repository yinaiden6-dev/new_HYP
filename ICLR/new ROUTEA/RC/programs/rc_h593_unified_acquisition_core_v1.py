"""One native RoMa forward feeds the three registered acquisition protocols."""
from collections import Counter
import time
import torch
import rc_roma_visual_origin_v1 as V
import rc_roma_m_inside_v1 as I
import run_rc_h593_roma_coordinate_precision_v2 as C
from rc_roma_exact_feature_cache_v2 import clone_tree, equal_tree

class MemorySink:
    def __init__(self): self.capture_full=False; self.value=None
    def __call__(self,identity,value): self.value=clone_tree(value)

class JointObserver(I.Observer):
    def __init__(self,model,core,input_sink):
        self.want_inside=False;self.sink=MemorySink();self.coarse_value=None
        self.pair_cache={};self.cache_identity=None;self.match_counts=Counter()
        super().__init__(model,core,input_sink,self.sink)
        self.matcher_forward=model.matcher.forward
        model.matcher.forward=self.memo_matcher
        self.handles.append(model.matcher.register_forward_hook(self.capture_matcher))
    def head(self,**kwargs):
        if self.want_inside:return super().head(**kwargs)
        return self.original_head(**kwargs)
    def delta(self,stride):
        original=super().delta(stride)
        def hook(module,args,output):
            if self.want_inside:original(module,args,output)
        return hook
    def refiner(self,stride):
        inside=I.Observer.refiner(self,stride);native=V.Observer.refiner(self,stride)
        def hook(module,args,output):
            (inside if self.want_inside else native)(module,args,output)
        return hook
    def memo_matcher(self,*args,**kwargs):
        if self.cache_identity!=self.identity:
            self.pair_cache.clear();self.cache_identity=self.identity
        key=V.tags(self.arm)
        if key in self.pair_cache:
            old=self.pair_cache[key]
            assert equal_tree((args,kwargs),old['inputs']), 'JOINT_MATCHER_INPUT_DRIFT'
            assert not self.want_inside,'INTERNAL_PATHS_MUST_BE_COLLECTED_ON_NATIVE_MISS'
            self.match_counts['matcher_hits']+=1
            return clone_tree(old['outputs'])
        inputs=clone_tree((args,kwargs));output=self.matcher_forward(*args,**kwargs)
        self.pair_cache[key]=dict(inputs=inputs,outputs=clone_tree(output))
        self.match_counts['matcher_misses']+=1
        return output
    def capture_matcher(self,module,args,output):
        self.coarse_value={k:output[k].detach().cpu().clone() for k in ('warp_AB','warp_BA','confidence_AB','confidence_BA')}
    @torch.inference_mode()
    def forward(self,*args,**kwargs):
        result=V.Observer.forward(self,*args,**kwargs)
        if self.want_inside:
            assert self.head_calls==2 and set(self.inside)=={0,1}
            self.output_sink(self.identity,self.inside)
        return result
    def close(self):
        self.model.matcher.forward=self.matcher_forward
        self.pair_cache.clear()
        super().close()

def stages(observer):
    value=clone_tree(observer.stages)
    for stage,sides in value.items():
        sides['M']=float(torch.sqrt(sides['AB']['weights'].mean()*sides['BA']['weights'].mean()))
    return value

@torch.inference_mode()
def acquire_pair(model,core,ob,hook,qi,ri,q,ref,qgeom,rgeom,qmeta,rmeta,pos,free_content,need,previous,index):
    physical=q['candidate_physical_rows'][pos]
    ob.sink.capture_full=index==0 and pos==0
    ob.sink.value=None
    identity=(q['query_source_sha256'],ref['source_image_sha256'])
    native_stages=None
    for family in ('inside','visual'):
        if previous.get(family):native_stages=previous[family]['arms']['NATIVE']['stages'];break
    counts=Counter();times={};native=None
    # A new internal intervention needs head inputs; coordinate RMS needs the
    # native dense warp. Those were not retained in historical compact caches.
    fresh_native=need['inside'] or need['coordinate'] or native_stages is None
    if fresh_native:
        hook.select('NATIVE');ob.want_inside=need['inside']
        ob.select('NATIVE',qgeom,rgeom,*identity)
        torch.cuda.synchronize();t=time.monotonic();native=model.match(qi,ri);torch.cuda.synchronize()
        times['NATIVE']=time.monotonic()-t;counts['native_forwards']+=1
        fresh_stages=stages(ob)
        if native_stages is not None:assert equal_tree(fresh_stages,native_stages),'PREVIOUS_NATIVE_STAGE_DRIFT'
        native_stages=fresh_stages
    else:counts['native_reused']+=1
    native_arm=dict(stages=native_stages,seconds=times.get('NATIVE',0.))
    out={};base=dict(candidate_position=pos,physical_row=physical,
        reference_image=dict(path=ref['source_path'],sha256=ref['source_image_sha256']),reference_geometry=rmeta,free_content=free_content)
    if need['inside']:
        assert ob.sink.value is not None
        out['inside']=dict(**base,arms={'NATIVE':native_arm})
        out['inside_paths']=dict(index=index,query_sha=identity[0],reference_sha=identity[1],branches=list(I.BRANCHES),
             sides=ob.sink.value,geometry='Native warp and native refiner evidence held fixed',full_first_pair_components=ob.sink.capture_full)
    if need['coordinate']:
        assert native is not None
        coarse=clone_tree(ob.coarse_value);arms={}
        for name in C.LEVELS:
            if name=='NATIVE':pred=native
            else:
                ob.want_inside=False;ob.select('NATIVE',qgeom,rgeom,*identity);hook.select(name)
                pred=model.match(qi,ri);counts['coordinate_changed_forwards']+=1
                assert equal_tree(ob.coarse_value,coarse),'COARSE_MATCHER_CHANGED'
            u=core.cell_means(pred['overlap_AB'][0,...,0].cpu(),qgeom)
            v=core.cell_means(pred['overlap_BA'][0,...,0].cpu(),rgeom)
            scores=C.c4(core,q['query_tokens'],ref['tokens'],u,v)
            if name=='NATIVE':
                changed=sum(int(torch.count_nonzero(C.quantize(pred[k],16)-pred[k])) for k in ('warp_AB','warp_BA'))
                assert changed>0
                diagnostics=dict(terminal_coordinate_elements_changed=changed,terminal_score_dependency=False)
            else:
                assert len(hook.calls)==12 and sum(c['changed'] for c in hook.calls)>0
                diagnostics=dict(calls=list(hook.calls),query_map_l1=float((u-arms['NATIVE']['query_visibility']).abs().mean()),
                    reference_map_l1=float((v-arms['NATIVE']['reference_visibility']).abs().mean()),
                    final_warp_RMS_normalized={k:float(((pred[k]-native[k]).double()**2).mean().sqrt()) for k in ('warp_AB','warp_BA')})
            intermediate=dict(query_coordinates=C.I.coordinate_record(pred['warp_AB'],qgeom),
                reference_coordinates=C.I.coordinate_record(pred['warp_BA'],rgeom),content=C.I.content_record(q['query_tokens'],ref['tokens'],u,v))
            assert abs(float(intermediate['content']['query_token_score_contribution'].sum())-scores[C.M.SCORE_KEYS[0]])<2e-12
            arms[name]=dict(query_visibility=u,reference_visibility=v,scores=scores,diagnostics=diagnostics,intermediate=intermediate)
            if name!='NATIVE':del pred
        out['coordinate']=dict(candidate_position=pos,physical_row=int(physical),reference_tokens_sha256=ref['tokens_sha256'],arms=arms,coarse_matcher=coarse)
    del native
    hook.select('NATIVE');ob.want_inside=False
    if need['visual']:
        arms={'NATIVE':native_arm}
        for arm in V.ARMS[1:]:
            ob.select(arm,qgeom,rgeom,*identity);torch.cuda.synchronize();t=time.monotonic()
            pred=model.match(qi,ri);torch.cuda.synchronize()
            arms[arm]=dict(stages=stages(ob),seconds=time.monotonic()-t)
            counts['visual_changed_forwards']+=1;del pred
        out['visual']=dict(**base,arms=arms)
    return out,dict(counts)
