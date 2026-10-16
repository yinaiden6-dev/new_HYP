"""Matched candidate controls and frozen image-then-OCR factorial evaluation."""
from pathlib import Path
import sys,json,time,argparse,hashlib
import numpy as np
import torch
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;R=Path('/mnt/lsdf_zyin');OUT=D/'roma_coordinates'
sys.path.insert(0,str(P));from experiment_utils import read,write,csvwrite,Ranking,metrics,zscore

def run(partial=False,wait=False):
    torch.set_num_threads(2);plan=read(OUT/'PLAN.json');c=read(P/'cohort.json');qids=[q['query'] for q in c]
    expected={(t['query'],t['row']) for t in plan['tasks']};start=time.time()
    while True:
        files=list((OUT/'pairs').glob('*/*.json'))
        if len(files)==len(expected):break
        if partial:break
        if not wait:raise RuntimeError(f'Incomplete coordinates: {len(files)}/{len(expected)}')
        if time.time()-start>16*3600:raise TimeoutError('Coordinate completion exceeded16h')
        time.sleep(30)
    records={}
    for f in files:
        x=read(f);assert x['signature']==plan['signature'];key=(x['query'],x['row']);assert key in expected and key not in records;records[key]=x
    rb=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',mmap=True,weights_only=True)
    ids=rb['ids'];sha=rb['sha256'];rk=Ranking(ids,sha,c);excluded=set(read(V/'identifier_layout_v1/protocol.json')['excluded_queries'])
    scopes={'all':np.arange(len(c)),'validation':np.array([i for i,x in enumerate(c) if x['split']=='val']),'test':np.array([i for i,x in enumerate(c) if x['split']=='test']),'retained':np.array([i for i,q in enumerate(qids) if q not in excluded]),'target54':np.array([i for i,q in enumerate(plan['queries']) if q['target54']])}
    provenance=read(OUT/'QUERY_PROVENANCE.json');assert [x['query'] for x in provenance]==qids
    for kind in ['generated','reused']:scopes['query_cache_'+kind]=np.array([i for i,x in enumerate(provenance) if x['kind']==kind])
    diagnostics=[]
    for q in plan['queries']:
        for family,pair in q['diagnostic_pairs'].items():
            keyp=(q['query'],pair['positive']);keyn=(q['query'],pair['negative'])
            if keyp not in records or keyn not in records:continue
            a=records[keyp];b=records[keyn];diagnostics.append(dict(query=q['query'],family=family,positive=ids[pair['positive']],negative=ids[pair['negative']],positive_features=a['features'],negative_features=b['features'],gaps={k:a['features'][k]-b['features'][k] for k in ['MaxSim','M','S','G','D','H','Droll','Hroll']},positive_evidence=f"pairs/{q['query']}/{pair['positive']:06}.npz",negative_evidence=f"pairs/{q['query']}/{pair['negative']:06}.npz"))
    write(OUT/'DIAGNOSTICS.json',dict(complete_pairs=len(records),expected_pairs=len(expected),diagnostic_comparisons=diagnostics,selection_uses_truth=False,diagnostic_pair_uses_truth=True))
    if partial:print('PARTIAL',len(records),len(diagnostics));return
    assert set(records)==expected
    while not (OUT/'COLOR.json').exists() or not read(OUT/'COLOR.json')['complete']:
        if not wait:raise RuntimeError('Color correspondence controls incomplete')
        if time.time()-start>16*3600:raise TimeoutError('Color completion exceeded16h')
        time.sleep(30)
    colors=read(OUT/'COLOR.json')
    for (q,j),rec in records.items():
        cv=colors['pairs'][q+'/'+str(j)];rec['features'].update({k:cv[k] for k in ['C','Croll']})
    features=['M','S','G','D','H','Droll','Hroll','MaxSim','C','Croll'];betas=plan['protocol']['rank_weights'];gates=plan['protocol']['gates']
    allcurve=[];summaries={};selected_orders={};selected_score_arrays={};case_rows=[]
    for family,path in [('image',V/'selected_image_scores.npy'),('joint',D/'fixed1051_scores.npy')]:
        scores=np.load(path);base,basewin,_=rk.all(scores);top=np.array([q['top'][family] for q in plan['queries']]);bs=zscore(np.take_along_axis(scores,top,axis=1))
        fs={k:np.array([[records[(q,j)]['features'][k] for j in tt] for q,tt in zip(qids,top)]) for k in features+['subject_fraction']}
        zs={k:zscore(v) for k,v in fs.items() if k!='subject_fraction'}
        def rerank(key,beta,gate):
            rr=base.copy();ww=basewin.copy();order=top.copy()
            if beta==0:return rr,ww,order
            for i in range(len(c)):
                vals=bs[i]+beta*zs[key][i];idx=np.lexsort((top[i],-vals))
                if idx[0]!=0 and fs['subject_fraction'][i,idx[0]]<gate:continue
                order[i]=top[i,idx];ww[i]=ids[order[i,0]]
                if base[i]<=10:rr[i]=next(j+1 for j,k in enumerate(order[i]) if sha[k] in set(c[i]['positive_sha256']))
            return rr,ww,order
        choices=[]
        for feature in features:
            for beta in betas:
                for gate in gates:
                    rr,ww,order=rerank(feature,beta,gate);vm=metrics(rr,c,scopes['validation'],base)
                    item=dict(feature=feature,beta=beta,gate=gate,validation=vm);choices.append(item)
                    for scope,ix in scopes.items():allcurve.append(dict(family=family,feature=feature,beta=beta,gate=gate,scope=scope,**metrics(rr,c,ix,base)))
        # Rolled controls never compete for the selected model; their role is falsification.
        select=lambda xs:max(xs,key=lambda x:(x['validation']['macro_accuracy'],x['validation']['macro_mrr'],-x['beta'],-x['gate'],-features.index(x['feature'])))
        best=select([x for x in choices if x['feature'] not in ['Droll','Hroll','Croll']])
        nongeo=select([x for x in choices if x['feature'] in ['M','S','MaxSim']]);geom=select([x for x in choices if x['feature'] in ['D','H']])
        # Matched hyperparameters for geometry, visibility and shuffled-coordinate explanations.
        arms={'selected':best,'selected_nongeometric':nongeo,'selected_geometric':geom}
        for feature in features:arms['matched_'+feature]=dict(feature=feature,beta=geom['beta'],gate=geom['gate'])
        colorbest=select([x for x in choices if x['feature']=='C']);arms['selected_color']=colorbest
        arms['color_roll_control']=dict(feature='Croll',beta=colorbest['beta'],gate=colorbest['gate'])
        result={}
        for name,params in arms.items():
            rr,ww,order=rerank(params['feature'],params['beta'],params['gate'])
            result[name]=dict(parameters=params,metrics={scope:metrics(rr,c,ix,base) for scope,ix in scopes.items()},repairs=[qids[i] for i in range(len(c)) if base[i]>1 and rr[i]==1],harms=[qids[i] for i in range(len(c)) if base[i]==1 and rr[i]>1])
            np.savez_compressed(OUT/f'{family}_{name}.npz',query_ids=qids,ranks=rr,top1=ww,top10=order)
            for i,q in enumerate(qids):case_rows.append(dict(query=q,family=family,arm=name,baseline_rank=int(base[i]),rank=int(rr[i]),winner=str(ww[i])))
            if name=='selected':
                selected_orders[family]=order;new=scores.copy()
                # Preserve every group's original within-group offsets; permute only
                # the ten original group-max score slots, never create a candidate.
                for i in range(len(c)):
                    oldvals=scores[i,top[i]]
                    for k,j in enumerate(order[i]):
                        ix=np.flatnonzero(np.array(sha)==sha[j]);delta=oldvals[k]-scores[i,j];new[i,ix]+=delta
                nr,nw,_=rk.all(new);assert np.array_equal(nr,rr) and np.array_equal(nw,ww)
                np.save(OUT/f'{family}_selected_scores.npy',new);selected_score_arrays[family]=new
        # Independent scalar Python ordering of the selected branch.
        rr,ww,order=rerank(best['feature'],best['beta'],best['gate'])
        for i in range(len(c)):
            if best['beta']==0:expected_order=top[i].tolist()
            else:
                vals=bs[i]+best['beta']*zs[best['feature']][i];ss=sorted(range(10),key=lambda k:(-float(vals[k]),int(top[i,k])))
                if ss[0]!=0 and fs['subject_fraction'][i,ss[0]]<best['gate']:ss=list(range(10))
                expected_order=top[i,ss].tolist()
            assert expected_order==order[i].tolist()
        summaries[family]=dict(selected=best,results=result,checks=dict(candidate_sets_identical=True,full_gallery_score_permutation_replay=True,independent_order_replay=True))
        np.savez_compressed(OUT/f'{family}_inputs.npz',top=top,base_z=bs,**fs)
    summary=dict(status='complete',population=1117,pairs=len(records),protocol=plan['protocol'],color_protocol=colors['protocol'],families=summaries,unavailable_pairs=0)
    write(OUT/'SUMMARY.json',summary);csvwrite(OUT/'curve.csv',allcurve);csvwrite(OUT/'case_ranks.csv',case_rows)
    print('COORDINATE EVALUATION DONE',json.dumps({k:v['results']['selected']['metrics']['all'] for k,v in summaries.items()}),flush=True)
    # Image repair is now fixed. Only after this stage do we combine with OCR.
    while not (D/'SELECTIVE_OCR_SUMMARY.json').exists() or not (D/'OCR_EVIDENCE_RETENTION.json').exists():
        if not wait:return
        if time.time()-start>16*3600:raise TimeoutError('OCR completion exceeded16h')
        time.sleep(30)
    ocr=read(D/'SELECTIVE_OCR_SUMMARY.json');assert ocr['status']=='complete'
    retention=read(D/'OCR_EVIDENCE_RETENTION.json');assert retention['status']=='complete'
    fixed=np.load(D/'fixed1051_scores.npy');oscores=np.load(D/'selected_ocr_scores.npy');ocr_delta=oscores-fixed
    oldimg=np.load(V/'selected_image_scores.npy');inc=np.load(V/'incremental_image_scores.npy')
    newinc=inc+.4*(selected_score_arrays['image']-oldimg)
    imdelta=.5*(zscore(newinc)-zscore(inc))
    configs={'prior_image_prior_ocr':fixed,'repaired_image_prior_ocr':fixed+imdelta,'prior_image_repaired_ocr':fixed+ocr_delta,'repaired_image_repaired_ocr':fixed+imdelta+ocr_delta,
        'joint_rerank_prior_ocr':selected_score_arrays['joint'],'joint_rerank_repaired_ocr':selected_score_arrays['joint']+ocr_delta}
    retained=np.load(D/'retained_ocr_scores.npy')
    configs.update(prior_image_retained_ocr=retained,repaired_image_retained_ocr=retained+imdelta)
    base,_,_=rk.all(fixed);factorial={};fr=[]
    for name,s in configs.items():
        rr,ww,_=rk.all(s);factorial[name]=dict(metrics={scope:metrics(rr,c,ix,base) for scope,ix in scopes.items()},repairs=[qids[i] for i in range(len(c)) if base[i]>1 and rr[i]==1],harms=[qids[i] for i in range(len(c)) if base[i]==1 and rr[i]>1])
        for i in range(len(c)):
            seen=set();position=0;allowed=set(c[i]['positive_sha256'])
            for j in np.argsort(-s[i],kind='stable'):
                if sha[j] in seen:continue
                seen.add(sha[j]);position+=1
                if sha[j] in allowed:break
            assert position==rr[i],(name,i,position,int(rr[i]))
        np.save(OUT/(name+'_scores.npy'),s)
        for i,q in enumerate(qids):fr.append(dict(query=q,method=name,baseline_rank=int(base[i]),rank=int(rr[i]),winner=str(ww[i])))
    write(OUT/'FINAL_FACTORIAL.json',dict(status='complete',image_selection=summaries['image']['selected'],ocr_selection=ocr['selected'],results=factorial,independent_full_gallery_rank_checks=len(c)*len(configs),supplementary_ocr_retention=retention['protocol'],protocol='Image parameters fixed before OCR; primary2x2 follows prior incremental formula by replacing only its0.4 new-image branch, re-normalizing, and adding the already validation-selected OCR delta. No per-case truth choice. Two joint reranker arms and two post-hoc old/new code-retention arms are separate comparisons and do not replace the primary2x2.',limits=['A score-slot permutation tests a conservative interface, not every possible fusion calibration.','Historical exploratory data, not a new blind test.','Code-retention control was added after observing an OCR replacement failure; its parameters are frozen but the control itself is post-hoc.']))
    csvwrite(OUT/'final_factorial_cases.csv',fr)
    print('FINAL FACTORIAL DONE',json.dumps({k:v['metrics']['all'] for k,v in factorial.items()}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');ap.add_argument('--partial',action='store_true');a=ap.parse_args();run(a.partial,a.wait)
