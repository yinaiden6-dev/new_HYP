#!/usr/bin/env python3
"""Render already validated EVAL128 group counts; never score, fit, or read curator."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROGRAM=Path(__file__).resolve()
SOURCE=ROOT/'results/rc_original7_eval128_full_evidence_v1'
OUT=ROOT/'results/rc_original7_eval128_group_plot_v1'
HEAD_SHA='ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263'
STEM='original7_eval128_all21_group_results_v1'

def need(value,message):
 if not bool(value):raise RuntimeError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def bind(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text())
def check(binding,path):
 target=Path(binding['path']);target=target if target.is_absolute() else ROOT/target
 need(target.resolve()==Path(path).resolve() and binding['sha256']==sha(path),'SOURCE_BINDING:'+str(path))
def ready_rows():
 paths={name:SOURCE/name for name in ('result.json','validation.json','independent_review.json')}
 need(all(p.is_file() for p in paths.values()),'COMPLETE_RESULT_VALIDATION_AND_REVIEW_REQUIRED')
 validation=read(paths['validation.json']);review=read(paths['independent_review.json'])
 need(validation['status']=='ORIGINAL7_EVAL128_POSTJOIN_LITERAL_REPLAY_PASS' and all(v is True for v in validation['checks'].values()),'CPU_VALIDATION_PASS_REQUIRED')
 need(review['status']=='ORIGINAL7_EVAL128_POST_RESULT_INDEPENDENT_REVIEW_PASS','INDEPENDENT_REVIEW_PASS_REQUIRED')
 check(validation['result'],paths['result.json']);check(review['result'],paths['result.json']);check(review['source_validation'],paths['validation.json'])
 result=read(paths['result.json'])
 need(result['query_count']==review['query_count']==128 and result['identity_count']==review['identity_count']==24 and result['source_group_count']==review['source_group_count']==21,'FROZEN128_24_21_POPULATION')
 need(result['parameter_sha256']==HEAD_SHA and result['new_training_updates']==review['new_training_updates']==0,'FROZEN_ORIGINAL7_NO_NEW_TRAINING')
 need(result['target_insertion_count']==result['target_absent_queries_dropped']==0 and not result['technical_or_metadata_errors'],'ALL128_NATURAL_C128_RETAINED')
 need(result['authority']==review['source_authority'] and result['program']==review['source_cpu_program'] and result['prejoin_seal']==review['prejoin_seal'] and result['prejoin_validation']==review['prejoin_validation'],'REVIEW_AND_RESULT_SAME_PROVENANCE')
 for key in ('authority','program','prejoin_seal','prejoin_validation'):
  p=Path(result[key]['path']);p=p if p.is_absolute() else ROOT/p;check(result[key],p)
 metric=result['metrics_all128']['REAL']
 for key in ('query_count','RAW_top1','final_top1','rescue','break','net'):
  need(metric[key]==review['metrics_all128']['REAL'][key],'REVIEWED_PRIMARY_COUNTS:'+key)
 source_groups=result['primary_group_statistics']['groups']
 groups={g['group']:g for g in source_groups};review_groups={g['group']:g for g in review['primary_group_statistics']['groups']}
 need(len(source_groups)==len(groups)==21 and set(groups)==set(review_groups),'ALL21_GROUPS_REQUIRED')
 # Only map the already sealed rescue/break IDs to their already sealed group.
 membership={a['query_id']:a['group'] for a in result['actions']['REAL']}
 rescues=result['primary_rescue_query_ids'];losses=result['primary_break_query_ids']
 need(rescues==review['primary_rescue_query_ids'] and losses==review['primary_break_query_ids'],'REVIEWED_PAIRED_ID_SETS')
 need(len(membership)==128 and set(membership.values())==set(groups),'EXISTING_ALL128_GROUP_MEMBERSHIP')
 rows=[]
 for i,name in enumerate(sorted(groups)):
  g=groups[name];v=review_groups[name]
  for key in ('query_count','RAW_correct','REAL_correct','net'):need(g[key]==v[key],'REVIEWED_GROUP_COUNTS:'+key)
  rows.append({'display_group':f'G{i+1:02d}','source_group':name,'image_count':g['query_count'],
   'RAW_correct':g['RAW_correct'],'ORIGINAL7_correct':g['REAL_correct'],
   'paired_rescues':sum(membership[q]==name for q in rescues),
   'paired_losses':sum(membership[q]==name for q in losses),'net_correct':g['net']})
 need(sum(r['image_count'] for r in rows)==128 and sum(r['paired_rescues'] for r in rows)==metric['rescue'] and sum(r['paired_losses'] for r in rows)==metric['break'],'DISPLAY_TOTALS_MATCH_SEALED_COUNTS')
 return result,rows,{name:bind(path) for name,path in paths.items()}

def render():
 need(not OUT.exists(),'APPEND_ONLY_PLOT_EXISTS')
 result,rows,sources=ready_rows()
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Patch
 from matplotlib.ticker import MaxNLocator
 matplotlib.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'original7-eval128-groups-v1','pdf.fonttype':42})
 fig,(left,right)=plt.subplots(1,2,figsize=(14,11),sharey=True,gridspec_kw={'width_ratios':[1.2,1]})
 fig.subplots_adjust(left=.12,right=.97,top=.865,bottom=.14,wspace=.28)
 y=list(range(21));raw=[r['RAW_correct'] for r in rows];original=[r['ORIGINAL7_correct'] for r in rows]
 rescue=[r['paired_rescues'] for r in rows];loss=[r['paired_losses'] for r in rows]
 left.barh([v+.16 for v in y],raw,height=.28,color='#9ca3af')
 left.barh([v-.16 for v in y],original,height=.28,color='#2563eb')
 for i,r in enumerate(rows):
  left.text(raw[i]+.07,i+.16,str(raw[i]),va='center',fontsize=8)
  left.text(original[i]+.07,i-.16,str(original[i]),va='center',fontsize=8,color='#1d4ed8')
 left.set_yticks(y,[f"{r['display_group']}  (n={r['image_count']})" for r in rows],fontsize=9)
 left.invert_yaxis();left.set_xlim(0,max(r['image_count'] for r in rows)+1)
 left.set_title('Correct images: RAW -> ORIGINAL7',fontsize=12)
 left.set_xlabel('Correct image count');left.set_ylabel('All source groups, sorted by fixed group name')
 right.barh(y,rescue,height=.58,color='#169b62');right.barh(y,[-x for x in loss],height=.58,color='#d65245')
 for i in y:
  if rescue[i]:right.text(rescue[i]+.07,i,f'+{rescue[i]}',va='center',fontsize=8,color='#087443')
  if loss[i]:right.text(-loss[i]-.07,i,f'-{loss[i]}',va='center',ha='right',fontsize=8,color='#a92b25')
  if not rescue[i] and not loss[i]:right.text(0,i,'0 / 0',va='center',ha='center',fontsize=7,color='#6b7280')
 extent=max(1,max(rescue+loss))+1;right.set_xlim(-extent,extent);right.axvline(0,color='#64748b',linewidth=.7)
 right.tick_params(axis='y',labelleft=False);right.set_title('Paired rescues and losses',fontsize=12)
 right.set_xlabel('Losses < 0                 Rescues > 0')
 for ax in (left,right):
  ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(axis='x',color='#e5e7eb',linewidth=.5);ax.set_axisbelow(True)
  ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
 m=result['metrics_all128']['REAL']
 fig.suptitle('Frozen ORIGINAL7 vs RAW: 128 images | 24 identities | 21 source groups',fontsize=15,y=.978)
 fig.text(.5,.945,f"RAW {m['RAW_top1']}/128 -> ORIGINAL7 {m['final_top1']}/128 | paired rescues {m['rescue']}, losses {m['break']}",ha='center',fontsize=12)
 fig.text(.5,.919,'Natural RAW C128 | frozen seven-parameter head | no new training',ha='center',fontsize=10)
 fig.legend(handles=[Patch(color='#9ca3af',label='RAW'),Patch(color='#2563eb',label='ORIGINAL7'),
  Patch(color='#169b62',label='Rescue: wrong -> correct'),Patch(color='#d65245',label='Loss: correct -> wrong')],
  loc='lower center',bbox_to_anchor=(.53,.070),ncol=4,frameon=False,fontsize=9)
 fig.text(.5,.043,'G01-G21 follow lexicographic source-group names; full name mapping is in CSV. All 128 images are retained.',ha='center',fontsize=9)
 fig.text(.5,.022,'Historically opened sources; groups, not individual images, are the statistical units.',ha='center',fontsize=9,color='#4b5563')
 OUT.mkdir()
 csv_path=OUT/'groups.csv'
 with csv_path.open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 csv_path.chmod(0o444)
 fig.savefig(OUT/(STEM+'.png'),dpi=220)
 fig.savefig(OUT/(STEM+'.svg'),metadata={'Date':None})
 fig.savefig(OUT/(STEM+'.pdf'),metadata={'CreationDate':None,'ModDate':None})
 plt.close(fig)
 for ext in ('png','svg','pdf'):(OUT/(STEM+'.'+ext)).chmod(0o444)
 caption=f'''# 新128图：冻结 ORIGINAL7 的来源组结果

本图覆盖全部128张图片、24个identity和21个来源组，使用自然RAW C128及
原冻结七参数ORIGINAL7/NATIVE7，不含任何新训练。参数SHA为`{HEAD_SHA}`。

左图为每组RAW与ORIGINAL7的正确图片数；右图为已封存的paired救回和
损失，绿色向右、红色向左。G01–G21严格按原group名称排序，未按效果
选择或重排；CSV保留完整组名映射和每组图片数。零变化的组也全部保留。

总体RAW为{m['RAW_top1']}/128，ORIGINAL7为{m['final_top1']}/128，救回{m['rescue']}，
损失{m['break']}。包含target缺席C128的图片，全128分母没有改成条件子集。

图形仅汇总CPU结果及独立复核已核验的计数，没有重算模型、修改预测、
读取curator或新增评价。来源历史上已打开；21个来源组是相关性与统计
推断的单位，128张图不等于128个独立身份。结果、CPU验证和独立复核
的SHA均由manifest绑定。
'''
 cap=OUT/'CAPTION.md';cap.write_text(caption);cap.chmod(0o444)
 manifest={'status':'ORIGINAL7_EVAL128_ALL21_GROUP_FIGURE_READY','sources':sources,'program':bind(PROGRAM),
  'population':{'images':128,'identities':24,'source_groups':21},'parameter_sha256':HEAD_SHA,
  'group_order':'lexicographic_original_group_name_no_effect_sorting','source_group_order':[r['source_group'] for r in rows],
  'plot':'one_figure_two_panels','candidate_source':'natural_RAW_C128','new_training_or_scoring_calls':0,'curator_reads':0,
  'artifacts':{ext:bind(OUT/(STEM+'.'+ext)) for ext in ('png','svg','pdf')},'csv':bind(csv_path),'caption':bind(cap),
  'matplotlib_version':matplotlib.__version__}
 mp=OUT/'manifest.json';mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n');mp.chmod(0o444)
 print(json.dumps({'status':manifest['status'],'manifest':bind(mp),'artifacts':manifest['artifacts']}),flush=True)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute-reviewed-result',action='store_true',required=True);parser.parse_args();render()
if __name__=='__main__':main()
