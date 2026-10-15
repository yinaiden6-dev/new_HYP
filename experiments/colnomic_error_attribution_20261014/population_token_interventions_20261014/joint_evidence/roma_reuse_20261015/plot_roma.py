from pathlib import Path
import sys,json,csv
D=Path(__file__).resolve().parent;V=D.parent;O=V.parent;P=O.parent;B=P.parent;A=D/'roma_reuse'
sys.path.insert(0,str(O/'crop_factorial_v1/plot_runtime'));sys.path.insert(0,str(B))
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from roi_utils import image_source,read
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
def cr(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
fig,axs=plt.subplots(2,2,figsize=(15,9),layout='constrained')
for row,(folder,label) in enumerate([(A/'pure_selected_image','纯图像944分支'),(A,'数字串联合1051分支')]):
 curve=cr(folder/'curve.csv');s=read(folder/'SUMMARY.json');sel=s['selection']
 for col,scope in enumerate(['all','validation']):
  ax=axs[row,col]
  for key,name,color in [('M','只加重叠置信度','#157f78'),('S','重叠置信度 × Token加权','#2455a4'),('L','仅Token加权（除去置信度乘数）','#ba5e39')]:
   pts=[x for x in curve if x['feature']==key and x['scope']==scope];xx=[float(x['beta']) for x in pts];yy=[int(x['correct']) for x in pts];ax.plot(xx,yy,'o-',label=name,color=color)
   if key==sel['feature']:
    idx=xx.index(sel['beta']);ax.scatter([xx[idx]],[yy[idx]],marker='*',s=240,c='black',zorder=5)
  ax.set_xscale('symlog',linthresh=.05);ax.set_xticks([0,.05,.1,.2,.5,1,2,5],['0','.05','.1','.2','.5','1','2','5']);ax.grid(alpha=.18);ax.set_xlabel('RoMa 权重 β');ax.set_ylabel('Top-1 正确照片数');ax.set_title(label+' · '+('全部1117张' if scope=='all' else '验证集173张'))
  ax.axhline(s['results']['baseline']['metrics'][scope]['correct'],color='#666666',ls='--',lw=1)
axs[0,0].legend(fontsize=9);fig.suptitle('历史 RoMa 缓存复用：同一Top10内重排；完整缓存不足时保留原结果\n黑星为验证集选择；历史测试已被反复观察，本轮属于探索性实验',fontsize=15)
fig.savefig(A/'18_roma_curves.png',dpi=160);fig.savefig(A/'18_roma_curves.svg');plt.close(fig)

g=read(A/'SUBJECT_GATE_SUMMARY.json');fig,axs=plt.subplots(1,2,figsize=(15,5.2),layout='constrained')
for ax,family,label in zip(axs,['image','fusion'],['纯图像944分支','数字串联合1051分支']):
 s=read((A/'pure_selected_image' if family=='image' else A)/'SUMMARY.json');prop=s['results']['selected' if family=='image' else 'S_previous_fixed_0.2'];res=g['families'][family]['results'];names=['直接加入RoMa','主体重叠门控≥50%','只按框面积门控','打乱重叠位置门控'];stats=[prop]+[res[k] for k in ['subject_fixed0.5','area_fixed0.5','roll_fixed0.5']];x=np.arange(4);rep=[a['metrics']['all']['repairs'] for a in stats];harm=[a['metrics']['all']['harms'] for a in stats]
 ax.bar(x-.17,rep,width=.34,label='修复',color='#157f78');ax.bar(x+.17,harm,width=.34,label='新增错误',color='#b94b48')
 for i,z in enumerate(stats):ax.text(i,max(rep[i],harm[i])+1,str(z['metrics']['all']['correct'])+'/1117',ha='center',fontsize=10)
 ax.set_xticks(x,names,rotation=12,ha='right',fontsize=9);ax.set_ylim(0,max(rep+harm)*1.3+3);ax.set_title(label);ax.set_ylabel('照片数');ax.legend()
fig.suptitle('人工主体框用于检查RoMa提议的新候选；没有使用测试真值选择候选\n门控设计源自本轮误伤检查，仍是探索性修复；位置打乱与框面积为竞争解释对照',fontsize=14)
fig.savefig(A/'19_roma_subject_gate.png',dpi=160);fig.savefig(A/'19_roma_subject_gate.svg');plt.close(fig)

audit=read(A/'REGION_AUDIT.json');preview=np.load(A/'overlap_previews.npz');inv=read(B/'image_inventory.json');meta=read('C:/Users/shazer/Desktop/B03_token_mechanism_analysis_20261013/token_mechanism_analysis/B03/reproduction/token_metadata.json')
for r in meta['references']:inv['references'].setdefault(r['id'],dict(image_path=r['original_path'],image_sha256=r['image_sha256']))
chosen=[]
for q in ['photo_813fb072c0f3b380','photo_0e87e0717ab1c215','photo_0a0e75a3b7797609']:
 chosen.append(next(r for r in audit['rows'] if r['query']==q and r['family']=='fusion_fixed_0.2'))
fig,axs=plt.subplots(len(chosen),4,figsize=(17,12),layout='constrained')
for row,r in enumerate(chosen):
 q=r['query'];photo=image_source(q,inv)
 for offset,role,label in [(0,'positive','正确参考'),(2,'negative','错误参考')]:
  info=r[role];rid=info['reference'];key=q+'__'+rid
  for col,im,which,title in [(offset,photo,'query',f'{q[6:12]}… 照片对{label}\n主体内置信度占比 {info["query_overlap_inside_fraction"]:.1%}'),(offset+1,image_source(rid,inv),'reference',label+' '+rid)]:
   ax=axs[row,col];ax.imshow(im);a=preview[key+'__'+which];ax.imshow(a,cmap='inferno',vmin=0,vmax=1,alpha=np.clip(a*.95,0,.75),extent=(0,im.width,im.height,0));ax.set_xlim(0,im.width);ax.set_ylim(im.height,0);ax.set_title(title,fontsize=11);ax.axis('off')
   if which=='query':
    x,y,X,Y=info['box'];ax.add_patch(Rectangle((x,y),X-x,Y-y,fill=False,edgecolor='#00eaff',lw=1.8))
fig.suptitle('RoMa重叠置信度：亮区表示可能匹配的区域，青框为人工主体框\n依次：修复示例、误伤示例、用户颜色案例。图中没有Token连线，不能据此证明布局对齐。',fontsize=15)
fig.savefig(A/'20_roma_overlap_examples.png',dpi=135);plt.close(fig)
print('ROMA FIGURES COMPLETE')
