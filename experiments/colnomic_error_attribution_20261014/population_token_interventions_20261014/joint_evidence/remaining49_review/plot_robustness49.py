from pathlib import Path
import sys,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=Path(sys.argv[1]);read=lambda n:json.loads((p/n).read_text());r=read('ROBUST_RESULTS.json');b=read('BARCODE_RESULTS.json')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained',gridspec_kw={'width_ratios':[1.15,1]})
ms=[r['grid']['cap_0.5_direct_1.0_code_current']['metrics'],b['grid']['decoded_query_weight_1.0']['metrics'],b['grid']['decoded_or_ocr_query_weight_4.0']['metrics']];ys=np.arange(3)
for offset,key,col,label in [(-.18,'repairs','#28866f','Recovered'),(.18,'harms','#c3615e','New errors')]:
 bars=axs[0].barh(ys+offset,[m['all'][key]for m in ms],height=.33,color=col,label=label);axs[0].bar_label(bars,padding=3)
axs[0].set_yticks(ys,['OCR term cap + repaired image','Decoded barcode, weight 1','Decoded / OCR GTIN, weight 4']);axs[0].invert_yaxis();axs[0].set_xlim(0,23);axs[0].set_xlabel('Queries versus the fixed 1058 primary result');axs[0].legend(loc='lower right');axs[0].set_title('Repair and harm must be counted together',loc='left',weight='bold')
ws=[0.,.5,1.,2.,4.]
for scope,col,marker,label in [('all','#28866f','o','All 1117'),('validation','#346a9d','s','Validation 173'),('test','#a66a28','^','Test 161')]:
 vals=[b['grid'][f'decoded_or_ocr_query_weight_{w}']['metrics'][scope]for w in ws];axs[1].plot(ws,[100*v['accuracy']for v in vals],marker=marker,color=col,label=label)
axs[1].set_ylim(94.3,96.2);axs[1].set_xticks(ws);axs[1].set_xlabel('Uniform exact-GTIN bonus weight');axs[1].set_ylabel('Top-1 accuracy (%)');axs[1].set_title('Four recoveries; validation and test unchanged',loc='left',weight='bold');axs[1].legend(loc='lower right');axs[1].grid(axis='y',alpha=.2)
fig.supxlabel('Historical exploratory controls. Validation tie-break retains weight 0; 1062 and 1065 do not replace the primary result.',fontsize=10)
fig.savefig(p/'robustness.png',bbox_inches='tight');plt.close(fig);print('ROBUST_PLOT_COMPLETE')
