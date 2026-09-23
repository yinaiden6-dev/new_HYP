#!/usr/bin/env python3
"""Posthoc, zero-fit investigation of the sealed 492/593 competition result."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/rc_h593_s_bias_competition_v1'
OUT = ROOT / 'results/rc_h593_competition_mechanism_v1'
FIG = ROOT / 'reports/figures/h593_competition_mechanism_20260921_v1'
AUTH = ROOT / 'registry/rc_h593_s_bias_competition_authority_v1_20260921.json'


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def checked(source):
    assert bind(source['path']) == source
    return source['path']


def distribution(values):
    v = np.array(values)
    return dict(n=len(v), mean=float(v.mean()), min=int(v.min()), max=int(v.max()),
                quantiles={str(q): float(np.quantile(v, q)) for q in (.025,.25,.5,.75,.975)})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = read(AUTH)
    result = read(SOURCE/'result.json')
    validation = read(SOURCE/'validation.json')
    assert validation['result'] == bind(SOURCE/'result.json')
    assert validation['status'] == 'S_BIAS_COMPETITION_ALL_COUNTS_PASS'
    assert validation['authority'] == result['authority'] == bind(AUTH)
    gallery = {r['physical_row']:r for r in read(checked(a['public_sources']['gallery']))['records']}
    roles = {r['query_id']:r for r in read(checked(a['join_sources']['curator']))['records']}
    joined = {r['query_id']:r for r in result['rows']}
    payloads, records, sources, objectives = [], [], [], []
    for f in range(5):
        folder = SOURCE/f'fold{f}'
        v, iv = read(folder/'validation.json'), read(folder/'independent_validation.json')
        assert v['payload'] == iv['payload'] == bind(folder/'payload.json')
        assert iv['passed'] and iv['authority'] == v['authority'] == bind(AUTH)
        p = read(folder/'payload.json')
        payloads.append(p)
        sources.append(bind(folder/'payload.json'))
        gap, both = p['parameters']['GAP_BIAS2'], p['parameters']['S_GAP3']
        objectives.append(dict(fold=f,
            gap_surrogate=gap['optimization']['fitted_surrogate_loss'],
            both_surrogate=both['optimization']['fitted_surrogate_loss'],
            gap_inner_net=gap['training_net_gain'], both_inner_net=both['training_net_gain'],
            gap_inner_rescues=gap['training_rescues'],gap_inner_breaks=gap['training_breaks'],
            both_inner_rescues=both['training_rescues'],both_inner_breaks=both['training_breaks'],
            gap_outer_correct=result['summary']['GAP_BIAS2']['fold_correct'][str(f)],
            both_outer_correct=result['summary']['S_GAP3']['fold_correct'][str(f)],
            delta_alpha=both['alpha'], delta_beta=both['beta']-gap['beta'],
            delta_bias=both['bias']-gap['bias']))
        for row in p['predictions']:
            q=row['query_id']; role=roles[q]; truth=joined[q]
            z=np.array([float.fromhex(v) for v in row['models']['COST1_FULL']['logits_hex']])
            order=sorted(range(127),key=lambda k:(-z[k],k));j,k=order[:2]
            top=row['candidate_physical_rows'][row['challenger_positions'][j]]
            second=row['candidate_physical_rows'][row['challenger_positions'][k]]
            raw=row['candidate_physical_rows'][row['winner']]
            correct_top=gallery[top]['identity']==role['identity']
            correct_raw=gallery[raw]['identity']==role['identity']
            m=float(z[j]);d=float(z[j]-z[k]);h=row['h_s']
            g=((m+0.0*h)+gap['beta']*d)+gap['bias']
            assert d.hex()==float(row['d']).hex()
            action=m>0 or (m<=0 and not gap['disabled'] and g>0)
            chosen=top if action else raw
            assert chosen==row['models']['GAP_BIAS2']['selected']
            both_g=((m+both['alpha']*h)+both['beta']*d)+both['bias']
            rec=dict(query_id=q,original_query_id=role['original_query_id'],fold=f,
                     component=role['component'],identity=role['identity'],original_path=role['original_path'],
                     m=m,d=d,h_s=h,runnerup_score=float(z[k]),gap_gate=g,both_gate=both_g,
                     top_physical_row=top,runnerup_physical_row=second,raw_physical_row=raw,
                     top_identity=gallery[top]['identity'],runnerup_identity=gallery[second]['identity'],
                     raw_identity=gallery[raw]['identity'],
                     top_image=gallery[top]['image_path'],runnerup_image=gallery[second]['image_path'],
                     raw_image=gallery[raw]['image_path'],
                     correct_top=correct_top,correct_raw=correct_raw,delta=int(correct_top)-int(correct_raw),
                     target_in_C128=truth['target_in_C128'],correct=truth['correct'],selected=truth['selected'],
                     delta_gate_extra_S=both['alpha']*h,
                     delta_gate_beta=(both['beta']-gap['beta'])*d,
                     delta_gate_bias=both['bias']-gap['bias'])
            records.append(rec)
    assert len(records)==593
    records.sort(key=lambda r:r['query_id'])
    # Frozen architecture capacity, using labels ONLY to describe an oracle ceiling.
    rescusable=[r for r in records if r['m']<=0 and not r['correct_raw'] and r['correct_top']]
    remaining=Counter()
    for r in records:
        if r['correct']['GAP_BIAS2']:continue
        if not r['target_in_C128']: key='target_absent_C128'
        elif r['m']>0: key='original_wrong_SWITCH_locked'
        elif r['correct_raw']: key='new_false_SWITCH'
        elif r['correct_top']: key='target_top_but_still_HOLD'
        else: key='original_HOLD_top_challenger_wrong'
        remaining[key]+=1
    assert sum(remaining.values())==101
    ceiling=481+len(rescusable)
    assert ceiling==sum(r['correct']['COST1_FULL'] or (r['m']<=0 and r['correct_top']) for r in records)
    # Same m/score scale; shuffle the donor competition coordinate within fold.
    # Conditional bins come exclusively from the sealed inner HOLD scores.
    # These are descriptive posthoc controls, not randomization-test p-values.
    folds=[]
    for f,p in enumerate(payloads):
        held=[r for r in records if r['fold']==f and r['m']<=0]
        m=np.array([r['m'] for r in held]);d=np.array([r['d'] for r in held])
        delta=np.array([r['delta'] for r in held])
        train_m=np.array([r['m'] for r in p['calibration'] if r['m']<=0])
        edges=np.unique(np.quantile(train_m,[.2,.4,.6,.8]))
        bins=np.searchsorted(edges,m,side='right')
        gap=p['parameters']['GAP_BIAS2']
        idx=[np.flatnonzero(bins==b) for b in range(len(edges)+1)]
        folds.append(dict(fold=f,m=m,d=d,delta=delta,beta=gap['beta'],bias=gap['bias'],bins=idx,
                          edges=edges.tolist(),n=len(held)))
    actual=481+sum(int(np.sum(f['delta'][(f['m']+f['beta']*f['d'])+f['bias']>0])) for f in folds)
    assert actual==492
    rng=np.random.default_rng(20260921);repeated={k:[] for k in ('within_fold','within_fold_and_inner_m_quintile')}
    for _ in range(1000):
        counts=dict.fromkeys(repeated,481)
        for f in folds:
            d=f['d'];perm=rng.permutation(len(d));conditioned=np.arange(len(d))
            for idx in f['bins']:conditioned[idx]=rng.permutation(idx)
            for name,ix in (('within_fold',perm),('within_fold_and_inner_m_quintile',conditioned)):
                g=(f['m']+f['beta']*d[ix])+f['bias']
                counts[name]+=int(np.sum(f['delta'][g>0]))
        for name,count in counts.items():repeated[name].append(count)
    shuffles={name:dict(**distribution(v),fraction_at_least_actual=float(np.mean(np.array(v)>=492)),
                       counts=v) for name,v in repeated.items()}
    # Exact fixed-coordinate differences on all changed correct decisions.
    disagreements=[r for r in records if r['correct']['GAP_BIAS2']!=r['correct']['S_GAP3']]
    breaks=[r for r in records if r['correct']['COST1_FULL'] and not r['correct']['GAP_BIAS2']]
    rescues=[r for r in records if not r['correct']['COST1_FULL'] and r['correct']['GAP_BIAS2']]
    # Group composition is descriptive, never a filter for model fitting.
    component_outcomes=defaultdict(lambda:dict(n=0,rescues=0,breaks=0,identities=set()))
    for r in records:
        c=component_outcomes[r['component']];c['n']+=1;c['identities'].add(r['identity'])
        c['rescues']+=int(not r['correct']['COST1_FULL'] and r['correct']['GAP_BIAS2'])
        c['breaks']+=int(r['correct']['COST1_FULL'] and not r['correct']['GAP_BIAS2'])
    components=[dict(component=k,**{kk:(sorted(vv) if isinstance(vv,set) else vv) for kk,vv in v.items()})
                for k,v in sorted(component_outcomes.items())]
    historical_path=ROOT/'results/rc_h593_endpoint_competition_v1/result.json'
    historical=read(historical_path)
    output=dict(status='H593_COMPETITION_MECHANISM_POSTHOC_COMPLETE',code=bind(__file__),
                authority=bind(AUTH),source_result=bind(SOURCE/'result.json'),source_payloads=sources,
                population=593,parameter_updates=0,encoder_forwards=0,
                method_selected_from_opened_development=True,
                objective_comparison=objectives,
                capacity=dict(original_correct=481,original_HOLD_target_top_count=len(rescusable),
                              oracle_ceiling_under_locked_top_and_original_SWITCH=ceiling,
                              actual_correct=492,remaining_error_partition=dict(remaining),
                              oracle_is_deployable_result=False),
                donor_shuffles=shuffles,
                shuffle_bins=[dict(fold=f['fold'],inner_m_edges=f['edges'],outer_hold_count=f['n'],
                                   bin_sizes=[len(i) for i in f['bins']]) for f in folds],
                gap_rescues=rescues,gap_breaks=breaks,extra_S_correctness_disagreements=disagreements,
                components=components,records=records,
                historical_endpoint=dict(result=bind(historical_path),scores=historical['scores'],
                                         different_base_and_feature=True),
                limits=['All new analyses are posthoc, no refit or new heldout performance claim.',
                        'Donor shuffle tail fractions are descriptive; not confirmatory p-values.',
                        'S remains in the original COST1 logits.',
                        'Training calibration rows overlap across outer folds; their net counts cannot be summed as new queries.',
                        'Oracle ceiling uses evaluation truth and is not a trainable performance estimate.'])
    (OUT/'analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    keys=['original_query_id','fold','m','d','h_s','gap_gate','both_gate','top_identity','runnerup_identity',
          'raw_identity','correct_raw','correct_top','delta_gate_extra_S','delta_gate_beta','delta_gate_bias']
    with (OUT/'changed_cases.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['case_type']+keys);writer.writeheader()
        for kind,rs in [('rescue',rescues),('break',breaks),('gap_vs_extra_S',disagreements)]:
            for r in rs:writer.writerow(dict(case_type=kind,**{k:r[k] for k in keys}))
    print(json.dumps(dict(status=output['status'],objective_comparison=objectives,capacity=output['capacity'],
                          donor_shuffles={k:{kk:vv for kk,vv in v.items() if kk!='counts'} for k,v in shuffles.items()},
                          break_candidates=[{k:r[k] for k in keys} for r in breaks],
                          extra_S_disagreements=[{k:r[k] for k in keys} for r in disagreements]),ensure_ascii=False,indent=2))
    make_figures(output)


def make_figures(output):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/h593-competition-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True,exist_ok=True)
    records=output['records'];objectives=output['objective_comparison']
    fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for f,ax in enumerate(axs.flat[:5]):
        rows=[r for r in records if r['fold']==f and r['m']<=0]
        for delta,color,label in ((0,'#b1bac4','Both answers wrong'),(-1,'#3c78a5','RAW correct'),(1,'#d43f3a','Challenger correct')):
            rs=[r for r in rows if r['delta']==delta]
            ax.scatter([r['m'] for r in rs],[r['d'] for r in rs],c=color,s=18 if delta==0 else 28,
                       alpha=.5 if delta==0 else .8,label=label)
        breaks=[r for r in rows if r['correct']['COST1_FULL'] and not r['correct']['GAP_BIAS2']]
        ax.scatter([r['m'] for r in breaks],[r['d'] for r in breaks],s=95,facecolors='none',edgecolors='black',linewidths=1.2,label='New break')
        p=read(SOURCE/f'fold{f}/payload.json')['parameters']['GAP_BIAS2']
        ymax=max(r['d'] for r in rows)*1.05;ys=np.linspace(0,ymax,200)
        xs=-p['bias']-p['beta']*ys
        ax.plot(xs,ys,color='#15191d',linewidth=1.5,label='Learned SWITCH boundary')
        ax.set_xlim(min(r['m'] for r in rows)*1.05,.05);ax.set_ylim(-.02,ymax)
        ax.set_title(f'Fold {f}: +{[1,5,-1,3,3][f]} vs COST1'.replace('+-','-'))
        ax.set_xlabel('Original top challenger logit m');ax.set_ylabel('Top-minus-runner-up gap d')
        ax.grid(alpha=.15)
    axs.flat[5].axis('off')
    handles,labels=axs.flat[0].get_legend_handles_labels()
    axs.flat[5].legend(handles,labels,loc='upper left',frameon=False)
    axs.flat[5].text(.02,.35,'Original SWITCH decisions are locked.\nPoints show original HOLDs only.\nNew SWITCH lies right of each boundary.\nFive independently fitted fold-specific gates.\nOpened H593 development analysis.',fontsize=12,va='top')
    fig.suptitle('Competition-calibrated HOLD release: 481 to 492 / 593',fontsize=19)
    fig.savefig(FIG/'01_competition_plane.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');x=np.arange(5);w=.35
    for ax,fields,title,ylabel in ((axs[0],('gap_surrogate','both_surrogate'),'Extra S lowers the training objective','Regularized logistic loss'),
                                  (axs[1],('gap_inner_net','both_inner_net'),'But does not improve inner net gain','Inner OOF rescues minus breaks')):
        ax.bar(x-w/2,[r[fields[0]] for r in objectives],w,color='#167a90',label='Gap + bias')
        ax.bar(x+w/2,[r[fields[1]] for r in objectives],w,color='#c96549',label='Extra S + gap + bias')
        ax.set_xticks(x,[f'Fold {f}' for f in x]);ax.set_title(title);ax.set_ylabel(ylabel);ax.legend(frameon=False)
    fig.suptitle('Objective improvement and correction utility diverge on the calibration data',fontsize=14)
    fig.savefig(FIG/'02_loss_vs_net_gain.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
