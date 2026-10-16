"""Separate code-reading, reference representation, and score-saturation controls."""
from pathlib import Path
import runpy,json
import numpy as np
D=Path(__file__).resolve().parent
# Reuse the audited complete-gallery scorer and fixed partitions, not a second implementation.
g=runpy.run_path(str(D/'run_identifier.py'))
globals().update({k:v for k,v in g.items() if k not in ['__name__','__file__','__builtins__']})
write(D/'IDENTIFIER_FOLLOWUP_PROTOCOL.json',dict(reason='Binary any-code reward can cancel a unique code when a shared postal/contact code matches both candidates.',score='Sum per-query-code reference IDF/log(N+1), with max credit once per query code per candidate; preserve original score.',modes=['exact','one_edit_min8'],weights=protocol['automatic_code_match']['weights'],selection='Same retained validation package-macro Top1, MRR and smaller weight; test not used.',scope='Exploratory follow-up prompted by score structure; not a fresh untouched test evaluation.'))
def weighted(cc,mode):
 result=np.zeros(len(ids))
 for code in cc:
  js,pairs=match({code},mode);contribution=np.zeros(len(ids))
  for pair in pairs:
   indexes=list(posts[pair['reference_code']]);idf=math.log(1+(len(ids)-len(indexes)+.5)/(len(indexes)+.5))/math.log(len(ids)+1)
   contribution[indexes]=np.maximum(contribution[indexes],idf)
  result+=contribution
 return result
newchoices=[];weighted_bonus={};diagnostic_curves=[]
for mode in ['exact','one_edit_min8']:
 bonus=np.stack([weighted(cc,mode) for cc in qcodes]);weighted_bonus[mode]=bonus
 for w in protocol['automatic_code_match']['weights']:
  rr,tt,mm=rk.all(joint+w*bonus);v=metrics(rr,c,scopes['validation_query_face_sufficient'],baseline)
  newchoices.append(dict(mode=mode,weight=w,ranks=rr,validation=v))
  for scope,ix in scopes.items():diagnostic_curves.append(dict(mode=mode,weight=w,scope=scope,**metrics(rr,c,ix,baseline)))
best=max(newchoices,key=lambda x:(x['validation']['macro_accuracy'],x['validation']['macro_mrr'],-x['weight'],x['mode']=='exact'))
write(D/'WEIGHTED_AUTOMATIC.json',dict(selection={k:v for k,v in best.items() if k!='ranks'},scopes={scope:metrics(best['ranks'],c,ix,baseline) for scope,ix in scopes.items()},curves=diagnostic_curves))
csvwrite(D/'weighted_automatic_curve.csv',diagnostic_curves)
allrows=[]
for arm in newchoices:
 for i,r in enumerate(c):allrows.append(dict(query=r['query'],mode=arm['mode'],weight=arm['weight'],baseline_rank=int(baseline[i]),rank=int(arm['ranks'][i])))
csvwrite(D/'weighted_automatic_cases.csv',allrows)
newocr=read(D/'rerecognized.json');reread=[]
for q,ocrentry in newocr.items():
 i=qi[q];x,y,X,Y=boxes[q]['bbox'];bx,by,_,_=ocrentry['bounds'];lines=[l for l in ocrentry['ocr']['lines'] if x<=(l['bbox'][0]+l['bbox'][2])/2+bx<=X and y<=(l['bbox'][1]+l['bbox'][3])/2+by<=Y]
 text='\n'.join(l['text'] for l in lines);cc=codes(text);counts=Counter(tok(text));raw=np.zeros(len(ids))
 for word,f in counts.items():
  if word in weights:ix,w=weights[word];raw[ix]+=f*w
 variants={'replace_subject_ocr':rk.one(.5*(zi[i]+zscore(raw)),i)}
 for mode in ['exact','one_edit_min8']:
  b=weighted(cc,mode)
  for w in protocol['automatic_code_match']['weights']:variants[mode+'_bonus_'+str(w)]=rk.one(joint[i]+w*b,i)
 reread.append(dict(query=q,old_rank=int(baseline[i]),new_ocr_codes=sorted(cc),new_ocr_text=text,interventions=variants,scope='Selected5 human-box crop OCR diagnostic; no manually supplied reading used.'))
write(D/'RERECOGNITION_RESULTS.json',reread)

# Annotated full-code diagnostic on BOTH query and the displayed reference pair.
# Only these reference readings are verified; no claim that the entire gallery's codebook is clean.
negcodes={'photo_85e10feb92319a7b':'17033-160-30','photo_8c54b7644f2e05b2':'372603700107','photo_c8920e14924534af':'372603700107','photo_f8f533ac85b05854':'87946','photo_fbc7b6d852b1125b':'B83261U-4'}
both=[]
for m in manual:
 q=m['query'];i=qi[q];p,n=ri[m['baseline']['positive']],ri[m['baseline']['negative']];truecode=next(iter(codes(m['code'])));negativecode=next(iter(codes(negcodes[q])));known={j for j,cc in enumerate(rcodes) if truecode in cc}
 # Image-SHA-equivalent references share the same corrected visible identifier.
 known.update(j for j in range(len(ids)) if sha[j]==sha[p]);known.difference_update(j for j in range(len(ids)) if sha[j]==sha[n] and truecode!=negativecode)
 bonus=np.zeros(len(ids));bonus[list(known)]=1
 both.append(dict(query=q,query_code=truecode,positive_reference_code=truecode,negative_reference_code=negativecode,baseline_rank=int(baseline[i]),ranks={str(w):rk.one(joint[i]+w*bonus,i)['rank'] for w in protocol['automatic_code_match']['weights']},scope='Manual full-code alignment diagnostic; verified reference-pair readings, not deployable retrieval.'))
write(D/'BOTH_SIDES_READING_DIAGNOSTICS.json',both)
print(json.dumps(dict(weighted_selection={k:v for k,v in best.items() if k!='ranks'},reread=[dict(query=r['query'],ranks={k:v['rank'] for k,v in r['interventions'].items()}) for r in reread],both_sides=both)),flush=True)
