#!/usr/bin/env python3
"""CPU-only closure of existing pair-quality traces and H71 fusion outputs.

No model training or new image/model forward. Learned M, fixed soft-relation
diagnostics, and RoMa's actual matcher output are deliberately kept separate.
"""
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rc_pair_relation_localization_v1/appearance_controls'


def read(p): return json.loads(Path(p).read_text())


def bind(p):
    p=Path(p).resolve(); h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return dict(path=str(p),sha256=h.hexdigest())


def checked(b):
    assert bind(b['path']) == b
    return Path(b['path'])


def write(name, result):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def describe(v):
    v=np.asarray(v,dtype=float)
    return dict(n=len(v),mean=float(v.mean()),minimum=float(v.min()),median=float(np.median(v)),maximum=float(v.max()))


def pair_traces():
    base=ROOT/'results/rc_h593_pair_quality_cpu_v1'
    resultp=ROOT/'results/rc_pair_quality_fold0_analysis_v1/result.json'
    original=read(resultp); byqid={r['query_id']:r for r in original['rows']}
    rows={q:dict(query_id=q,target_position=r['target_position'],winner=r['winner'],component=r['component'],
        original_rescue=q in original['original_rescues'],scores={
        'NATIVE_ROMA_M':r['models']['ROMA']['M'], 'COLNOMIC_FREE_L':r['models']['ROMA']['L0'],
        **{arm+'_LEARNED_M':r['models'][arm]['M'] for arm in ['COL_ONLY_SINGLE','COL_ONLY_PAIR','COARSE_SINGLE','COARSE_PAIR']}})
        for q,r in byqid.items()}
    sources=[]
    for idx,arm in [(1,'COL_ONLY_PAIR'),(3,'COARSE_PAIR')]:
        sealp=base/f'fit{idx:02d}/validation.json'; seal=read(sealp)
        payloadp=checked(seal['payload']); payload=read(payloadp)
        assert payload['config']['arm']==arm and len(payload['predictions'])==119
        sources += [bind(sealp),seal['payload']]
        for binding in payload['predictions']:
            v=torch.load(checked(binding),map_location='cpu',weights_only=True,mmap=True)
            row=byqid[v['query_id']]; assert v['candidate_physical_rows']==row['candidate_physical_rows']
            assert np.max(abs(v['M'].numpy()-np.asarray(row['models'][arm]['M'])))<1e-14
            assert len(v['quality_traces'])==128
            s=np.array([[*t['query_relation_summary'][-2:].tolist(),*t['reference_relation_summary'][-2:].tolist()]
                        for t in v['quality_traces']])
            # Cached summaries are [mean entropy_q,mean return_q,mean entropy_r,mean return_r].
            # Same fixed temperature .1 relation, BEFORE the learned quality MLP.
            concentration=1-(s[:,0]+s[:,2])/2
            soft_return=np.sqrt(np.maximum(0,s[:,1]*s[:,3]))
            rows[v['query_id']]['scores'][arm+'_FIXED_SOFT_CONCENTRATION']=concentration.tolist()
            rows[v['query_id']]['scores'][arm+'_FIXED_SOFT_RETURN']=soft_return.tolist()
            sources.append(binding)
    summaries={}; valid=[r for r in rows.values() if r['target_position'] is not None]
    assert len(valid)==113
    for name in next(iter(rows.values()))['scores']:
        first=[];meanrank=[];margins=[];win=[];corr=[];rescue_first=[]
        for r in valid:
            x=np.asarray(r['scores'][name]);t=r['target_position'];w=r['winner']
            wrong=np.delete(x,t); first.append(bool(x[t]>wrong.max()))
            meanrank.append(1+int((x>x[t]).sum()));margins.append(float(x[t]-wrong.max()))
            if t!=w:win.append(bool(x[t]>x[w]))
            y=np.asarray(r['scores']['NATIVE_ROMA_M'])
            rankx=np.argsort(np.argsort(x,kind='stable'),kind='stable');ranky=np.argsort(np.argsort(y,kind='stable'),kind='stable')
            corr.append(float(np.corrcoef(rankx,ranky)[0,1]))
            if r['original_rescue']:rescue_first.append(bool(x[t]>wrong.max()))
        summaries[name]=dict(strict_target_first=sum(first),inpool=113,total_queries=119,
            mean_optimistic_rank=float(np.mean(meanrank)),mean_target_minus_strongestwrong=float(np.mean(margins)),
            target_beats_RAWwinner_among_RAW_errors=sum(win),inpool_RAW_errors=len(win),
            original_rescues_target_strict_first=sum(rescue_first),original_rescue_queries=len(rescue_first),
            mean_within_query_ordinal_rank_correlation_with_M=float(np.mean(corr)))
    out=dict(status='EXISTING_FOLD0_SOFT_RELATION_TRACES_RECOUNTED',sources=[bind(__file__),bind(resultp)]+sources,
        scope='Original fold0 held119, 113 target in natural ColNomic C128; no new fit; original target labels already opened in sealed analysis',
        interpretation='Ranking-only candidate discrimination; these direct statistics are not trained HOLD/SWITCH heads.',
        definitions=dict(concentration='1 - (mean normalized softmax entropy q+r)/2',
            soft_return='sqrt(mean_i sum_j A_ij B_ji * mean_j sum_i B_ji A_ij)',
            relation='L2 cosine of saved single-image descriptors; row/column softmax temperature0.1; not RoMa warp or RoMa cross-image Transformer',
            coarse='Two DINOv3 layers pooled on ColNomic grid, layer-normalized and fixed 2048->128 Rademacher projection'),
        fixed_source_training_historical=original['train_fit'],summary=summaries,rows=list(rows.values()),
        no_new_training=True,no_GPU=True,
        limits=['Pooled projected coarse descriptors differ from native full matcher feature grid.',
            'No learned-MLP negative result proves absence of identity information.',
            'Different scale raw margins cannot compare effect sizes between statistics.',
            'Rank correlation is descriptive, deterministic tie order may affect tied ranks.'])
    write('pair_trace_recount.json',out)
    print(json.dumps(summaries,indent=2))


def selected(x,theta,pred):
    z=x@theta[:-1]+theta[-1]
    k=int(z.argmax());pos=pred['challenger_positions'][k] if z[k]>0 else pred['winner']
    return pred['candidate_physical_rows'][pos],z


def fusion():
    base=ROOT/'results/rc_h71_feature_fusion_pilot_v1'; snap=read(base/'snapshot.json')
    published=read(base/'result.json'); meta={r['query_id']:r for r in snap['records']}
    published_rows={r['query_id']:r for r in published['rows']}
    summaries={}; fitrows=[]; sourcebindings=[bind(__file__),bind(base/'result.json'),bind(base/'snapshot.json')]
    baselines={}
    for fold in range(5):
        for offset,mode in [(8,'FREE'),(9,'ROMA_WEIGHTED')]:
            folder=base/f'fit{fold*10+offset:03d}'; seal=read(folder/'validation.json');p=read(checked(seal['payload']))
            head=torch.load(checked(p['checkpoint']),map_location='cpu',weights_only=True)['theta'].numpy()
            for b in p['predictions']:
                pp=torch.load(checked(b),map_location='cpu',weights_only=True,mmap=True)
                baselines[(fold,mode,pp['query_id'])]=(pp,head,b)
    for idx in range(50):
        folder=base/f'fit{idx:03d}';seal=read(folder/'validation.json');payload=read(checked(seal['payload']))
        model=torch.load(checked(payload['checkpoint']),map_location='cpu',weights_only=True)
        cfg=payload['config'];name=cfg['source']+'/'+cfg['mode'];theta=model['theta'].numpy()
        assert model['step']==payload['steps']==240
        cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert torch.equal(cp['theta'],model['theta'])
        initial=np.array([float.fromhex(x) for x in model['warm']['theta_hex']])
        st=dict(config=cfg,train_queries=len(payload['train_query_ids']),held_queries=len(payload['held_query_ids']),
            steps=240,passes_equivalent=240/len(payload['train_query_ids']),head_change_l2=float(np.linalg.norm(theta-initial)),
            training_active_vjp_candidates=describe([x['active_vjp_candidates'] for x in cp['history']]),
            up_weight_norm=None if model['adapter'] is None else float(model['adapter']['up.weight'].norm()),
            optimizer_last_exp_avg_sq_norms=[float(v['exp_avg_sq'].norm()) for v in cp['optimizer']['state'].values()])
        changes=[];logit_changes=[];gate_mean=[];gate_max=[];token_cos=[];swaps=[]
        sourcebindings += [bind(folder/'validation.json'),seal['payload'],payload['checkpoint'],bind(folder/'checkpoint.pt')]
        for binding in payload['predictions']:
            v=torch.load(checked(binding),map_location='cpu',weights_only=True,mmap=True)
            q=v['query_id'];basev,baseth,basebinding=baselines[(cfg['fold'],cfg['mode'],q)]
            assert v['candidate_physical_rows']==basev['candidate_physical_rows']
            pick,z=selected(v['X'].numpy(),theta,v)
            assert pick==v['selected']==published_rows[q]['selected'][name]
            assert np.max(abs(z-v['logits'].numpy()))<1e-10
            C=np.stack([a['c4'].numpy() for a in v['pairs']]);B=np.stack([a['c4'].numpy() for a in basev['pairs']])
            changes.extend(abs(C[:,0]-B[:,0]).tolist());logit_changes.extend(abs(z-basev['logits'].numpy()).tolist())
            score_swap,_=selected(v['X'].numpy(),baseth,v);head_swap,_=selected(basev['X'].numpy(),theta,v)
            swaps.append(dict(query_id=q,base_selected=basev['selected'],native_selected=pick,
                              fused_features_base_head=score_swap,base_features_fused_head=head_swap))
            if model['adapter'] is not None:
                ep=folder/'encoded'/(meta[q]['query_image_key']+'.pt')
                e=torch.load(ep,map_location='cpu',weights_only=True,mmap=True)
                assert e['model_checkpoint']==payload['checkpoint']
                g=e['gate'].double();z=e['adapted_tokens'].double();orig=z/(1+g)
                gate_mean.append(float(g.abs().mean()));gate_max.append(float(g.abs().max()))
                token_cos.append(float(torch.nn.functional.cosine_similarity(z,orig,dim=1).mean()))
                sourcebindings.append(bind(ep))
            sourcebindings.append(binding)
        st.update(content_S_abs_change=describe(changes),logit_abs_change=describe(logit_changes),
            query_gate_abs_mean=None if not gate_mean else describe(gate_mean),
            query_gate_abs_max=None if not gate_max else describe(gate_max),
            query_token_cosine_original_analysis_copy=None if not token_cos else describe(token_cos),
            native_decision_changes=sum(x['native_selected']!=x['base_selected'] for x in swaps),
            features_only_swap_changes=sum(x['fused_features_base_head']!=x['base_selected'] for x in swaps),
            head_only_swap_changes=sum(x['base_features_fused_head']!=x['base_selected'] for x in swaps),swaps=swaps)
        fitrows.append(st)
        print(json.dumps(dict(stage='fusion_recount',index=idx,source=name,max_S_change=st['content_S_abs_change']['maximum'],
                              choice_changes=st['native_decision_changes'])),flush=True)
    for name in published['counts']:
        matches=[r for r in fitrows if r['config']['source']+'/'+r['config']['mode']==name]
        if not matches:continue
        summaries[name]=dict(correct=published['counts'][name],n=71,
            max_abs_content_S_change=max(r['content_S_abs_change']['maximum'] for r in matches),
            max_abs_logit_change=max(r['logit_abs_change']['maximum'] for r in matches),
            native_choice_changes=sum(r['native_decision_changes'] for r in matches),
            features_only_swap_changes=sum(r['features_only_swap_changes'] for r in matches),
            head_only_swap_changes=sum(r['head_only_swap_changes'] for r in matches),
            minimum_up_weight_norm=None if name.startswith('NO_ADAPTER') else min(r['up_weight_norm'] for r in matches),
            maximum_query_gate_abs=None if name.startswith('NO_ADAPTER') else max(r['query_gate_abs_max']['maximum'] for r in matches))
    write('fusion_training_effect_recount.json',dict(status='H71_ALL50_FITS_REAL_ADAPTER_CHANGE_AND_HEADSWAP_RECOUNT_PASS',
        sources=sourcebindings,summary=summaries,fold_configs=fitrows,
        scope='Opened71 queries/37 groups, natural ColNomic C128, original fivefold split; 240-step seed17 COST1 fusion and 7-parameter joint head',
        content_model='RoMa backbone single-image coarse/fine pools randomly projected to128D drive bounded channel multiplier (1+0.1*tanh(up(down(x)))) on existing ColNomic final tokens; no candidate conditioning inside adapter',
        precision='Original scoring and logits FP64; query gate/token comparisons use stored FP32 analysis copies',
        no_new_training=True,no_GPU=True,
        limits=['Same final decisions does not mean unchanged features, zero gradients or absence of useful backbone information.',
            'Only bounded multiplicative single-image fusion, compressed inputs and short240 updates evaluated.',
            'The headswap is algebraic readout replay on previously opened outputs, not a new trained model.',
            'Near-identity output gate cannot add a completely new descriptor direction or recover discarded coordinates.']))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pair-traces','fusion']);args=parser.parse_args()
    pair_traces() if args.stage=='pair-traces' else fusion()
