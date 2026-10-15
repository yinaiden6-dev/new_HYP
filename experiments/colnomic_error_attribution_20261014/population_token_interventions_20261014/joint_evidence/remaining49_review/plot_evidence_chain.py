from pathlib import Path
import json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=Path(sys.argv[1]);d=json.loads((p/'EVIDENCE_CHAIN.json').read_text())
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
fig,axs=plt.subplots(1,2,figsize=(16,6),layout='constrained',gridspec_kw={'width_ratios':[1,1.4]})
labels=[['Original image','Subject tokens','Input / reference repair','RoMa-supported image'],['Original fusion','Subject OCR + scoring','Numeric tokenization','+ Subject tokens','+ Image repair','+ Identifier matching','Final image + OCR','Old-code retention (post hoc)']]
for ax,key,lab in zip(axs,['image','joint'],labels):
 rows=d[key];counts=[r['all_correct']for r in rows];y=np.arange(len(rows));cols=['#277c95']*len(rows)if key=='image'else['#25816c']*(len(rows)-1)+['#a5aab2']
 bars=ax.barh(y,[c/1117*100 for c in counts],color=cols);ax.set_yticks(y,lab);ax.invert_yaxis();ax.set_xlim(0,101);ax.set_xlabel('Top-1 accuracy on all 1,117 queries (%)');ax.set_title('Image only'if key=='image'else'Image + OCR',loc='left',weight='bold')
 for b,c in zip(bars,counts):ax.text(b.get_width()+.6,b.get_y()+b.get_height()/2,str(c),va='center',fontsize=10)
fig.suptitle('Evidence-backed milestones from image retrieval to the final audit',x=.01,ha='left',weight='bold')
fig.supxlabel('Named configurations, not a cumulative ablation. Post-hoc controls are separate; historical splits were repeatedly inspected.',fontsize=10)
fig.savefig(p/'milestones.png',bbox_inches='tight');plt.close(fig)
print('MILESTONES_COMPLETE')
