#!/usr/bin/env python3
"""Read-only TRAIN/PAIR algebra check; no model fit, EVAL role join, or action."""
from pathlib import Path
import hashlib,json,math,sys
sys.dont_write_bytecode=True
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_new_hyp_product_contrast_training_identity_v1/result.json'
CLOSURE=ROOT/'results/rc_absolute_evidence_scale_calibration_v1/input_closure.json'
PAIR=ROOT/'results/routea_matched_three_arm_pair64_training_features_v2/payload.pt'
PAIR_SHA='d7be701ac4629059d22301c17b0f0d44b69b4b67666072b8fab7c3c0cb7716e3'
EPS=1e-12

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(x,message):
    if not x:raise RuntimeError(message)
def symmetric(a,b):return (a-b)/(abs(a)+abs(b)+EPS)

def main():
    rows=[];closure=json.loads(CLOSURE.read_text());bindings=[]
    # Source shards contain both split roles. EVAL records are filtered by the
    # pre-existing split-role flag before any evidence or feature is examined.
    for binding in closure['source_shards']:
        p=ROOT/binding['path'];need(sha(p)==binding['sha256'],'FULL_SOURCE_SHA')
        rows.extend((r,'FULL_TRAIN') for r in torch.load(p,map_location='cpu',mmap=True,weights_only=True)['records'] if r['data_split_role']=='TRAIN')
        bindings.append(binding)
    need(sha(PAIR)==PAIR_SHA,'PAIR_SOURCE_SHA')
    rows.extend((r,'PAIR') for r in torch.load(PAIR,map_location='cpu',mmap=True,weights_only=True)['records'])
    out={};digest=hashlib.sha256()
    for row,role in rows:
        d=out.setdefault(role,{'rows':0,'comparisons':0,'candidate_occurrences':0,'M_floor_active':0,'M_nonpositive':0,'L_nonpositive':0,'S_nonpositive':0,'positive_comparisons':0,'min_M':math.inf,'min_L':math.inf,'min_S':math.inf,'max_product_reconstruction_residual':0.,'max_epsilon_free_formula_residual':0.,'max_epsilon_aware_formula_residual':0.,'max_nonlinear_vs_additive_difference':0.})
        d['rows']+=1;ev=row['evidence']['C_PAIRED'];w=int(row['base_winner_position'])
        for i,candidate in enumerate(row['challenger_positions']):
            d['comparisons']+=1;values=[]
            for pos in (int(candidate),w):
                v=ev[pos];s=float(v['real_score']);m=float(v['visibility_mass']);l=s/max(m,EPS)
                need(all(math.isfinite(x) for x in (s,m,l)),'FINITE_SCALARS')
                values.append((s,m,l));d['candidate_occurrences']+=1
                d['M_floor_active']+=int(m<EPS);d['M_nonpositive']+=int(m<=0);d['L_nonpositive']+=int(l<=0);d['S_nonpositive']+=int(s<=0)
                for key,value in [('M',m),('L',l),('S',s)]:d['min_'+key]=min(d['min_'+key],value)
                d['max_product_reconstruction_residual']=max(d['max_product_reconstruction_residual'],abs(s-m*l))
            (sc,mc,lc),(sw,mw,lw)=values
            native=[float(v) for v in row['real_native_features']['C_PAIRED'][i]]
            need([v.hex() for v in native[1:4]]==[symmetric(a,b).hex() for a,b in ((sc,sw),(mc,mw),(lc,lw))],'STORED_NATIVE_CONTRAST_BITS')
            digest.update((role+':'+row['query_id']+':'+str(candidate)+':'+';'.join(float(v).hex() for vs in values for v in vs)+';'+';'.join(v.hex() for v in native)).encode())
            if min(sc,mc,lc,sw,mw,lw)<=0:continue
            d['positive_comparisons']+=1
            ds,dm,dl=native[1:4];a=mc+mw;b=lc+lw
            free=(dm+dl)/(1+dm*dl)
            aware=(a*dl*(b+EPS)+b*dm*(a+EPS))/(a*b+dm*dl*(a+EPS)*(b+EPS)+2*EPS)
            d['max_epsilon_free_formula_residual']=max(d['max_epsilon_free_formula_residual'],abs(ds-free))
            d['max_epsilon_aware_formula_residual']=max(d['max_epsilon_aware_formula_residual'],abs(ds-aware))
            d['max_nonlinear_vs_additive_difference']=max(d['max_nonlinear_vs_additive_difference'],abs(ds-(dm+dl)))
    need(out['FULL_TRAIN']['rows']==32 and out['FULL_TRAIN']['comparisons']==4064,'FULL_TRAIN_AXIS')
    need(out['PAIR']['rows']==64 and out['PAIR']['comparisons']==64,'PAIR_AXIS')
    result={'status':'NEW_HYP_PRODUCT_CONTRAST_TRAINING_ALGEBRA_CHECK_COMPLETE','statistics':out,
        'program_sha256':sha(Path(__file__)),'original_input_closure_sha256':sha(CLOSURE),'source_shards':bindings,
        'PAIR_source':{'path':str(PAIR.relative_to(ROOT)),'sha256':PAIR_SHA},
        'training_source_scalar_feature_sequence_sha256':digest.hexdigest(),
        'EVAL_evidence_or_feature_rows_examined':0,'EVAL_target_role_reads':0,'new_predictions':0,'new_training_runs':0,
        'original_feature_values_changed':False,'task':'Check positive-domain applicability and numerical size of product-contrast identity on original training inputs.',
        'limits':['Original shard containers include EVAL records, excluded by existing split-role flag before their scalars/features are examined.',
                  'TRAIN-only numerical property, not EVAL gain, population generalization, causal contribution, or statistical interaction proof.',
                  'All residuals are diagnostics; neither algebraic form is substituted into frozen FP64 features.']}
    text=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if OUT.exists():need(OUT.read_text()==text,'APPEND_ONLY_OUTPUT_MISMATCH')
    else:OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(text)
    print(json.dumps({'status':result['status'],'result_sha256':sha(OUT),'statistics':out},indent=2))
if __name__=='__main__':main()
