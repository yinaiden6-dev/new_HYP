"""Read-only decomposition of sealed probe scores; no fit or threshold search."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
ROOT = SOURCE.parents[1]
bound = {}

def load(p):
    p = Path(p)
    b = p.read_bytes()
    bound[str(p.resolve())] = hashlib.sha256(b).hexdigest()
    return json.loads(b)

def verify(b):
    p = Path(b['path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == b['sha256']
    return p

def main():
    validation = load(SOURCE/'validation.json')
    result = load(verify(validation['result']))
    authority = load(verify(validation['authority']))
    manifest = load(verify(authority['manifest']))
    seal = load(verify(result['prelabel_seal']))
    for b in seal['predictions'] + seal['decisions']:
        verify(b)
    source_rows = {r['query_id']:r for r in manifest['probe_rows']}
    output = []
    max_error = 0.
    head_values = {}
    for row in result['rows']:
        q = row['query_id']; r = source_rows[q]
        raw = np.asarray(r['raw_scores'],dtype=np.float64)
        mass = np.asarray(r['M'],dtype=np.float64)
        w = r['winner_index']; ii = np.asarray(r['challenger_positions'])
        target = [i for i,x in enumerate(r['candidate_identities']) if x==row['identity']]
        assert len(target) in (0,1)
        audited = load(SOURCE/'audited_decisions'/f'{q}.json')['predictions']
        native = load(SOURCE/'predictions/REAL'/f'{q}.json')
        cv = load(ROOT/'results/rc_prellm_m_adapter_v2/encoder_cache'/q/'validation.json')
        parity = load(verify(cv['parity']))
        records = {}
        for mode in ('REAL','REAL_REFIT','ADDITIVE4'):
            content = np.asarray(parity['fresh_L0'] if mode=='ADDITIVE4' else native['L'],dtype=np.float64)
            theta = np.asarray(audited[mode]['theta'],dtype=np.float64)
            names = audited[mode]['features']; head_values[mode] = dict(zip(names,theta.tolist()))
            def sym(x):
                return (x[ii]-x[w])/(np.abs(x[ii])+abs(x[w])+1e-12)
            columns = [(raw[ii]-raw[w])/raw.std()]
            if mode=='ADDITIVE4': columns.append(sym(mass))
            columns += [sym(content), np.ones(127)]
            x = np.stack(columns,axis=1)
            terms = x*theta[None,:]
            z = terms.sum(axis=1)
            expected = np.asarray(audited[mode]['logits'])
            error = float(np.max(np.abs(z-expected)))
            max_error = max(error,max_error)
            assert error<1e-10
            scores=np.zeros(128);scores[ii]=z
            chosen = int(ii[np.argmax(z)]) if z.max()>0 else w
            assert chosen==audited[mode]['prediction_position']
            wrongs = [i for i in ii if i not in target]
            strongest_wrong = max(wrongs,key=lambda i:scores[i])
            def entry(pos):
                j = list(ii).index(pos) if pos!=w else None
                return dict(position=int(pos),candidate_id=r['candidate_ids'][pos],identity=r['candidate_identities'][pos],
                    raw_rank=1+int(np.sum(raw>raw[pos])),content_rank=1+int(np.sum(content>content[pos])),
                    raw=float(raw[pos]),M=float(mass[pos]),L=float(content[pos]),
                    action=float(scores[pos]),action_rank=1+int(np.sum(scores>scores[pos])),
                    features={k:float(v) for k,v in zip(names,x[j])} if j is not None else {'HOLD':0.},
                    terms={k:float(v) for k,v in zip(names,terms[j])} if j is not None else {'HOLD':0.})
            selections={'raw_winner':w,'content_best':int(content.argmax()),'selected':chosen,'strongest_wrong_challenger':strongest_wrong}
            if target: selections['target']=target[0]
            records[mode] = {k:entry(v) for k,v in selections.items()}
            if target and target[0]!=w:
                t=entry(target[0]); wrong=entry(strongest_wrong)
                records[mode]['target_minus_strongest_wrong']={k:t['terms'][k]-wrong['terms'][k] for k in names}
                records[mode]['target_minus_strongest_wrong']['total']=t['action']-wrong['action']
        output.append(dict(query_id=q,identity=row['identity'],raw_correct=row['correct']['RAW'],target_in_C128=bool(target),correct=row['correct'],paths=records))
    out=dict(status='SEALED_READOUT_DECOMPOSITION_PASS',feature_formula='dRAW=(RAWc-RAWw)/std(RAW); sym_X=(Xc-Xw)/(|Xc|+|Xw|+1e-12); HOLD=0',heads=head_values,max_logit_reconstruction_error=max_error,
        source_sha256=bound,rows=output,fitting_updates=0,new_inference=0,threshold_searches=0,
        scope='Post-hoc explanation on previously opened group-disjoint PROBE8; no selected new model')
    (HERE/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'heads':head_values,'max_error':max_error}))
    for r in output:
        if r['raw_correct']: continue
        print(r['query_id'],r['identity'])
        for mode,p in r['paths'].items():
            print(mode,json.dumps({k:p[k] for k in ('target','strongest_wrong_challenger','content_best','target_minus_strongest_wrong') if k in p}))

if __name__=='__main__': main()
