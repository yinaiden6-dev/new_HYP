from pathlib import Path
import sys,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(sys.argv[1]);a=json.loads((P/'REVIEW49.json').read_text());s=a['summary']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
labels={'版式/空间结构':'Layout / spatial structure','颜色':'Color','小码/条码编号':'Small code / barcode','剂量/规格文字':'Dose / quantity / specification','文字内容差异':'Other text difference','弱线索/可见面受限':'Weak visible evidence'}
fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');d=sorted(s['primary_cue_counts'].items(),key=lambda x:-x[1]);y=np.arange(len(d));bars=ax.barh(y,[v for k,v in d],color=['#267c8e','#277da8','#4e9a80','#d19635','#716195','#969eaa']);ax.set_yticks(y,[labels[k]for k,v in d]);ax.invert_yaxis();ax.bar_label(bars,padding=5);ax.set_xlim(0,18);ax.set_xlabel('Number of queries (mutually exclusive primary cue)');ax.set_title('49 retained failures: visual review of the actual visible face',loc='left',fontweight='bold');ax.text(0,-.27,'Visual cue categories are not proof of encoder causality. The 13 exclusions are unchanged.',transform=ax.transAxes,fontsize=9,color='#555');fig.savefig(P/'classification.png',bbox_inches='tight');plt.close(fig)
names={'image_mix_no_ocr':'Drop all OCR','direct_repaired_image_current_ocr':'Repaired image + OCR only','remove_legacy_full_renorm':'Drop legacy full-image branch','remove_subject_renorm':'Drop subject-token branch','keep_old_image_scale':'Keep old normalization scale','remove_text_keep_code':'Drop text, keep codes','remove_code_keep_text':'Drop codes, keep text'}
fig,ax=plt.subplots(figsize=(11,5),layout='constrained');ys=np.arange(len(names));rep=[s['controls'][k]['all']['repairs']for k in names];harm=[s['controls'][k]['all']['harms']for k in names];p=ax.barh(ys-.18,rep,height=.34,color='#238c75',label='Repaired');n=ax.barh(ys+.18,harm,height=.34,color='#c45c58',label='New errors');ax.bar_label(p,padding=3);ax.bar_label(n,padding=3);ax.set_yticks(ys,list(names.values()));ax.invert_yaxis();ax.set_xlim(0,122);ax.set_xlabel('Queries compared with the fixed 1058/1117 primary result');ax.set_title('All 1,117 queries: fixed diagnostic interventions',loc='left',fontweight='bold');ax.legend(loc='lower right');ax.text(0,-.22,'These controls were not selected as a new method. Per-query best-arm selection is an oracle diagnostic.',transform=ax.transAxes,fontsize=9,color='#555');fig.savefig(P/'controls.png',bbox_inches='tight');plt.close(fig)
print('PLOTS_COMPLETE')
