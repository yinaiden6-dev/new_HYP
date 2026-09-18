#!/usr/bin/env python3
"""Render all-query outcomes from four independently validated, frozen artifacts."""
from __future__ import annotations
import csv,hashlib,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR','/tmp/rc_verified_outcome_matrices_matplotlib_v1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch,Rectangle
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/figures/new_hyp_verified_outcome_matrices_v1'
PINNED={
 'sufficiency':('rc_retrained_evidence_sufficiency_v1','20102901242c734c2b5c0404e0e5571999ab1dc25ceefa5cee2f37f884a12601','ebac4a7d704e121d972f8999dbcbe767e588681e3f9a7352974f78d9e809e195'),
 'factorial':('rc_product_response_factorial_v1','e9dae1e5837e72d3f38ba3e4473d0ff555b6ee90fadfbe3d3dbf64376759c0bf','f91c01caac1824ceb8cde358a9c22f95b94fe0d846965b35815c66cc778ed3bb'),
 'native_fixed':('rc_frozen_group_effect_native64_v1','eebbd6e9b15be6181ed375f04f6c8cbc7a985f5f871073c41b29c1f03a5975fe','c0d4fd6e729d3e23e36fc3b2533cb92da823c33931569a740c88539f1e2445cd'),
 'frozenc90':('rc_frozen_group_effect_difficult90_v1','c27f7c3aa543ba73130be059e1047415d86a35dc2fb17ffe5c00febfbbdb3356','13ce0b1d7d0bce7a0264f03f010466230b6def55236198bee5375166f7349b76')}
GREEN='#449775';RED='#d76862';COLORMAP=ListedColormap([RED,GREEN]);NORM=BoundaryNorm([-.5,.5,1.5],2)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(x,m):
 if not bool(x):raise RuntimeError(m)

def load():
 sources={};bindings={}
 for name,(directory,result_sha,validation_sha) in PINNED.items():
  p=ROOT/'results'/directory/'result.json';v=p.with_name('independent_validation.json')
  need(sha(p)==result_sha and sha(v)==validation_sha,'FROZEN_SOURCE_HASH:'+name)
  outcome=json.loads(p.read_text());validation=json.loads(v.read_text())
  need(validation['result_sha256']==result_sha and validation['checks'] and all(x is True for x in validation['checks'].values()),'INDEPENDENT_SOURCE_VALIDATION:'+name)
  need('PASS' in validation['status'],'VALIDATION_STATUS:'+name)
  sources[name]=outcome;bindings[name]={'result':{'path':str(p.relative_to(ROOT)),'sha256':result_sha},
   'independent_validation':{'path':str(v.relative_to(ROOT)),'sha256':validation_sha,'status':validation['status']}}
 return sources,bindings

def index(rows,ordinal):
 ordered=sorted(rows,key=lambda r:int(r[ordinal]));need(len({r['query_id'] for r in ordered})==len(ordered),'UNIQUE_QUERY_IDS')
 need(len({r[ordinal] for r in ordered})==len(ordered),'UNIQUE_ORDINALS')
 return ordered,{r['query_id']:r for r in ordered}

def native(source):
 s=source['sufficiency']['actions']['EVAL'];f=source['factorial']['actions']['EVAL'];d=source['native_fixed']['actions']['EVAL']['REAL']
 source_columns={'M3':s['RAW_PLUS_M3']['REAL'],'L3':s['RAW_PLUS_L3']['REAL'],
  'JOINT4':f['JOINT4']['REAL'],'PRODUCT5':f['PRODUCT5']['REAL'],'RESPONSE6':f['RESPONSE6']['REAL'],
  'ORIGINAL7':f['ORIGINAL7']['REAL'],'DROP_S':d['DROP_S'],'DROP_QR':d['DROP_QR'],'DROP_S_QR':d['DROP_S_QR']}
 original,original_by=index(f['ORIGINAL7']['REAL'],'execution_ordinal');need(len(original)==32,'FULL_EVAL32')
 # Anchor duplicate results across the three result lineages before joining them.
 for a,b in ((s['JOINT4']['REAL'],f['JOINT4']['REAL']),(s['ORIGINAL7']['REAL'],f['ORIGINAL7']['REAL'])):
  need(a==b,'SAME_REFITTED_HEAD_ACTIONS_EXACT')
 fields=('base_winner','target_position','final_position','final_correct','base_correct','decision','switch_logit')
 for condition,head in (('ORIGINAL7','ORIGINAL7'),('REFIT_JOINT4','JOINT4'),('REFIT_PRODUCT5','PRODUCT5'),('REFIT_RESPONSE6','RESPONSE6')):
  _,dd=index(d[condition],'execution_ordinal');_,ff=index(f[head]['REAL'],'execution_ordinal')
  need(set(dd)==set(ff),'NATIVE_FIXED_SOURCE_QUERY_AXIS')
  for q in dd:need(all(dd[q][k]==ff[q][k] for k in fields),'FIXED_REFIT_PARENT_FIELDS_EXACT')
 columns=['RAW',*source_columns];maps={k:index(v,'execution_ordinal')[1] for k,v in source_columns.items()}
 for m in maps.values():need(set(m)==set(original_by),'FULL_EVAL32_MODEL_QUERY_AXIS')
 _,group_rows=index(d['ORIGINAL7'],'execution_ordinal');rows=[]
 for baseline in original:
  q=baseline['query_id'];row={'execution_ordinal':baseline['execution_ordinal'],'query_id':q,
   'supergroup':group_rows[q]['supergroup'],'target_position':baseline['target_position'],'base_winner_position':baseline['base_winner'],'outcomes':{}}
  row['outcomes']['RAW']={'correct':bool(baseline['base_correct']),'action':'HOLD','raw_action':'HOLD','final_position':baseline['base_winner'],'switch_logit':''}
  for name in source_columns:
   a=maps[name][q]
   need(a['execution_ordinal']==baseline['execution_ordinal'] and a['target_position']==baseline['target_position'] and a['base_winner']==baseline['base_winner'],'SAME_NATIVE_QUERY_CANDIDATE_ANCHORS')
   row['outcomes'][name]={'correct':bool(a['final_correct']),'action':a['decision'],'raw_action':a['decision'],'final_position':a['final_position'],'switch_logit':float(a['switch_logit']).hex()}
  rows.append(row)
 totals={name:sum(r['outcomes'][name]['correct'] for r in rows) for name in columns}
 need(list(totals.values())==[25,26,26,26,27,26,28,26,27,26],'EXPECTED_VALIDATED_EVAL_COUNTS')
 return rows,columns,totals

def frozenc(source):
 outcome=source['frozenc90'];original,by=index(outcome['actions']['ORIGINAL'],'query_ordinal');need(len(original)==90,'FULL_DIFFICULT90')
 need([r['query_ordinal'] for r in original]==list(range(90)),'EXHAUSTIVE_90_ORDINALS')
 columns=['RAW','ORIGINAL','DROP_S','DROP_QR','DROP_S_QR'];maps={name:index(outcome['actions'][name],'query_ordinal')[1] for name in columns[1:]}
 for m in maps.values():need(set(m)==set(by),'FULL90_MODEL_QUERY_AXIS')
 rows=[]
 for baseline in original:
  q=baseline['query_id'];row={'query_ordinal':baseline['query_ordinal'],'query_id':q,'opened_split':baseline['opened_split'],
   'target_present_c128':baseline['target_present_c128'],'base_winner_position':baseline['base_winner_position'],'outcomes':{}}
  row['outcomes']['RAW']={'correct':bool(baseline['base_correct']),'action':'HOLD','raw_action':'HOLD','final_position':baseline['base_winner_position'],'switch_logit':''}
  for name in columns[1:]:
   a=maps[name][q];need(a['query_ordinal']==baseline['query_ordinal'] and a['base_correct']==baseline['base_correct'] and a['target_present_c128']==baseline['target_present_c128'],'SAME90_QUERY_BASE')
   row['outcomes'][name]={'correct':bool(a['final_correct']),'action':a['effective_action'],'raw_action':a['raw_action'],
    'final_position':a['final_prediction_position'],'switch_logit':a['switch_logit_binary64']}
  rows.append(row)
 totals={name:sum(r['outcomes'][name]['correct'] for r in rows) for name in columns}
 need(list(totals.values())==[61,69,68,70,65],'EXPECTED_VALIDATED_FROZENC90_COUNTS')
 for name in columns[1:]:need(totals[name]==outcome['summaries'][name]['final_top1'],'SOURCE90_SUMMARY_MATCH')
 return rows,columns,totals

def csv_export(name,rows,columns):
 path=OUT/(name+'.csv');metadata=[k for k in rows[0] if k!='outcomes'];fields=metadata+[c+'__'+k for c in columns for k in ('correct','action','raw_action','final_position','switch_logit')]
 records=[]
 for row in rows:
  flat={k:row[k] for k in metadata}
  for c in columns:
   for k,v in row['outcomes'][c].items():flat[c+'__'+k]=int(v) if k=='correct' else v
  records.append(flat)
 with path.open('w',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(records)
 with path.open(newline='') as stream:loaded=list(csv.DictReader(stream))
 need(len(loaded)==len(rows),'CSV_COMPLETE_ROWS')
 for exported,row in zip(loaded,rows,strict=True):
  need(exported['query_id']==row['query_id'],'CSV_QUERY_ORDER')
  for c in columns:need(int(exported[c+'__correct'])==int(row['outcomes'][c]['correct']) and exported[c+'__action']==row['outcomes'][c]['action'],'CSV_CELL_AND_ACTION_MATCH')
 return path

def matrix(ax,rows,columns,totals,labels,denominator,fontsize=7):
 values=np.asarray([[int(r['outcomes'][c]['correct']) for c in columns] for r in rows]);n,m=values.shape
 ax.pcolormesh(np.arange(m+1),np.arange(n+1),values,cmap=COLORMAP,norm=NORM,edgecolors='white',linewidth=.38,antialiased=True)
 ax.set_xlim(0,m);ax.set_ylim(n,0)
 ax.set_xticks(np.arange(m)+.5,labels=[c+'\n'+str(totals[c])+'/'+str(denominator) for c in columns])
 ax.xaxis.tick_top();ax.tick_params(axis='x',length=0,pad=5,labelsize=fontsize)
 plt.setp(ax.get_xticklabels(),rotation=43,ha='left',rotation_mode='anchor')
 ax.set_yticks(np.arange(n)+.5,labels=labels);ax.tick_params(axis='y',length=0,pad=4,labelsize=fontsize)
 for ri,row in enumerate(rows):
  for ci,c in enumerate(columns):
   if row['outcomes'][c]['action']=='SWITCH':ax.plot(ci+.5,ri+.5,'o',ms=2.2,mec='none',color='#111111')
 for spine in ax.spines.values():spine.set_linewidth(.6);spine.set_color('#444444')
 return values

def legend(fig,y):
 handles=[Patch(facecolor=GREEN,label='Correct'),Patch(facecolor=RED,label='Wrong'),Line2D([],[],marker='o',ls='none',markersize=3,color='#111111',label='SWITCH')]
 fig.legend(handles=handles,loc='center',bbox_to_anchor=(.5,y),ncol=3,frameon=False,fontsize=8,handlelength=1.1,columnspacing=2)

def save(fig,name):
 paths=[]
 for suffix in ('svg','pdf','png'):
  path=OUT/(name+'.'+suffix)
  if suffix=='svg':meta={'Date':None,'Creator':Path(__file__).name,'Description':'Existing validated outcomes only; all original queries; no new training or predictions.'}
  elif suffix=='pdf':meta={'CreationDate':None,'ModDate':None,'Creator':Path(__file__).name,'Title':name,'Subject':'Existing validated outcomes; source lineages remain separate.'}
  else:meta={'Software':Path(__file__).name}
  fig.savefig(path,dpi=300,metadata=meta);paths.append(path)
 plt.close(fig);return paths

def plot_native(rows,columns,totals):
 fig=plt.figure(figsize=(7.7,8.6));ax=fig.add_axes([.21,.11,.76,.705])
 labels=[str(r['execution_ordinal']).zfill(3)+'  '+r['query_id'] for r in rows]
 matrix(ax,rows,columns,totals,labels,32,fontsize=6.9)
 ax.axvline(1,color='#222222',lw=1);ax.axvline(7,color='#222222',lw=1.3)
 ax.add_patch(Rectangle((6,0),1,32,fill=False,lw=1.1,ec='#111111'))
 for text,row in zip(ax.get_yticklabels(),rows,strict=True):
  if not row['outcomes']['ORIGINAL7']['correct']:text.set_fontweight('bold')
 fig.text(.5,.985,'NATIVE7 lineage · original RAW C128 · opened EVAL32',ha='center',va='top',fontsize=10,fontweight='bold')
 fig.text(.5,.961,'All 32 queries in original execution order; no new predictions',ha='center',va='top',fontsize=8)
 fig.text(.21+.76*.40,.925,'Refitted subsets and original head',ha='center',fontsize=8)
 fig.text(.21+.76*.85,.925,'Fixed ORIGINAL7\nterms removed',ha='center',va='center',fontsize=7.4)
 legend(fig,.065)
 fig.text(.5,.025,'Bold query labels: four ORIGINAL7 errors.  Removal columns retain the other ORIGINAL7 coefficients.',ha='center',fontsize=7)
 return save(fig,'native_eval32_action_matrix')

def plot_90(rows,columns,totals):
 fig=plt.figure(figsize=(7.6,8.4));axes=[fig.add_axes([.08,.12,.36,.69]),fig.add_axes([.57,.12,.36,.69])]
 for ax,block in zip(axes,(rows[:45],rows[45:]),strict=True):
  labels=[str(r['query_ordinal']).zfill(2)+(' †' if not r['target_present_c128'] else '') for r in block]
  matrix(ax,block,columns,totals,labels,90,fontsize=6.5)
  ax.axvline(1,color='#222222',lw=1);ax.axvline(2,color='#222222',lw=1.3)
 fig.text(.5,.985,'FROZEN_C lineage · original RAW C128 · opened difficult90',ha='center',va='top',fontsize=9.6,fontweight='bold')
 fig.text(.5,.959,'Separate historical head; all 90 ordinals, fixed coefficients and zero-threshold action',ha='center',va='top',fontsize=8)
 fig.text(.5,.932,'Column totals cover all 90 queries; intervention outcomes are not adopted model improvements.',ha='center',fontsize=7.1)
 legend(fig,.070)
 fig.text(.5,.025,'† Target absent from natural C128 (6 queries).  Both panels share the same five conditions.',ha='center',fontsize=7)
 return save(fig,'frozenc_difficult90_action_matrix')

def comparison_sets(rows,columns):
 correct={c:{r['query_id'] for r in rows if r['outcomes'][c]['correct']} for c in columns}
 return {a+'_vs_'+b:{'new_only_correct':sorted(correct[a]-correct[b]),'base_only_correct':sorted(correct[b]-correct[a])}
         for i,a in enumerate(columns) for b in columns[:i]}

def main():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
  'svg.hashsalt':'new_hyp_verified_outcome_matrices_v1','savefig.facecolor':'white'})
 source,bindings=load();native_rows,native_columns,native_totals=native(source);old_rows,old_columns,old_totals=frozenc(source)
 OUT.mkdir(parents=True,exist_ok=True)
 paths=[csv_export('native_eval32_action_matrix',native_rows,native_columns),csv_export('frozenc_difficult90_action_matrix',old_rows,old_columns)]
 paths+=plot_native(native_rows,native_columns,native_totals)+plot_90(old_rows,old_columns,old_totals)
 result={'status':'VERIFIED_OUTCOME_MATRICES_RENDERED','source_program':{'path':str(Path(__file__).resolve().relative_to(ROOT)),'sha256':sha(Path(__file__))},
  'validated_sources':bindings,'artifacts':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths],
  'native_eval32':{'head_lineage':'NATIVE7 / matched original RAW C128 / all127 zero-threshold action','query_count':32,
   'columns':native_columns,'correct_counts':native_totals,'query_ids_in_order':[r['query_id'] for r in native_rows],
   'execution_ordinals':[r['execution_ordinal'] for r in native_rows],'paired_correctness_sets':comparison_sets(native_rows,native_columns),
   'column_sources':{'RAW':'factorial ORIGINAL7 base_correct','M3':'sufficiency RAW_PLUS_M3 REAL','L3':'sufficiency RAW_PLUS_L3 REAL',
    'JOINT4':'factorial JOINT4 REAL','PRODUCT5':'factorial PRODUCT5 REAL','RESPONSE6':'factorial RESPONSE6 REAL','ORIGINAL7':'factorial ORIGINAL7 REAL',
    'DROP_S':'native_fixed EVAL REAL DROP_S','DROP_QR':'native_fixed EVAL REAL DROP_QR','DROP_S_QR':'native_fixed EVAL REAL DROP_S_QR'}},
  'frozenc_difficult90':{'head_lineage':source['frozenc90']['head'],'query_count':90,'columns':old_columns,'correct_counts':old_totals,
   'query_ids_in_order':[r['query_id'] for r in old_rows],'query_ordinals':[r['query_ordinal'] for r in old_rows],
   'target_absent_ordinals':[r['query_ordinal'] for r in old_rows if not r['target_present_c128']],
   'paired_correctness_sets':comparison_sets(old_rows,old_columns)},
  'software':{'numpy':np.__version__,'matplotlib':matplotlib.__version__},
  'checks':{'source_result_and_validation_hashes_verified':True,'all_validation_checks_true':True,
   'overlapping_native_heads_action_fields_exact':True,'complete_original_query_axes':True,'CSV_cells_roundtrip_exact':True,
   'new_model_fits':0,'new_predictions':0,'scheduler_queries':0},
  'scope':'Visualization of previously opened, independently validated outcomes; not new data, model adoption, external confirmation or HYP GO.'}
 p=OUT/'manifest.json';p.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'manifest_sha256':sha(p),'native_counts':native_totals,'frozenc90_counts':old_totals,'artifact_count':len(paths)},indent=2))
if __name__=='__main__':main()
