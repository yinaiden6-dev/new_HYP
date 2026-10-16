"""Scientific figures for the existing report, with no separate HTML report."""
from pathlib import Path
import sys,json,csv
D=Path(__file__).resolve().parent;V=D.parent;O=V.parent
sys.path.insert(0,str(O/'crop_factorial_v1/plot_runtime'))
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from PIL import Image,ImageDraw,ImageFont
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
def read(name):return json.loads((D/name).read_text(encoding='utf-8'))
def cr(name):
 with (D/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def save(fig,name):
 for ext in ['png','svg']:fig.savefig(D/(name+'.'+ext),dpi=180,bbox_inches='tight')
 plt.close(fig)
comparison=cr('query_exclusion_results.csv');idx={(r['method'],r['scope']):r for r in comparison}
nret=int(idx['final_incremental','query_face_sufficient']['n']);nexcluded=1117-nret
methods=[('image/historical_original','原图像'),('image/validation_selected_image','图像修复'),('prior/original_rank','原融合'),('prior/center_presence_rank','主体OCR＋计分修正'),('prior/numeric_rank','数值分词'),('prior/selected_numeric_fresh_rank','数值＋主体Token'),('final_incremental','再追加图像修复')]
fig,ax=plt.subplots(figsize=(11,5.6));yy=np.arange(len(methods))
for shift,scope,label,color in [(-.18,'all','原始1117张','#8798a0'),(.18,'query_face_sufficient',f'剔除用户确认{nexcluded}张后的{nret}张','#287387')]:
 values=[100*float(idx[k,scope]['accuracy']) for k,l in methods];bars=ax.barh(yy+shift,values,height=.33,label=label,color=color)
 for bar,(k,l) in zip(bars,methods):
  r=idx[k,scope];ax.text(bar.get_width()+.5,bar.get_y()+bar.get_height()/2,f"{r['correct']}/{r['n']}",va='center',fontsize=9)
ax.set_yticks(yy,[l for k,l in methods]);ax.invert_yaxis();ax.set_xlim(0,108);ax.set_xlabel('Top-1（%）');ax.set_title('只剔除 Query，不改变图库、标签或模型参数');ax.legend(loc='upper left',bbox_to_anchor=(0,-.12),ncol=2,frameon=False);ax.grid(axis='x',alpha=.15);fig.tight_layout();save(fig,'14_query_face_exclusion')
rows=read('COLOR_SUMMARY.json')['results'];idx={(r['view'],r['query_color'],r['reference_arm']):r for r in rows}
arms=[('original__visible','原始'),('blank_sham','空白区编辑对照'),('original__hidden','隐藏编号'),('gray__visible','参考图转灰度'),('swap_side_colors__visible','交换侧面颜色')]
fig,ax=plt.subplots(figsize=(10,4.8));xx=np.arange(len(arms))
for shift,view,label,color in [(-.18,'full','原照片','#8798a0'),(.18,'human_crop','人工框裁剪','#287387')]:
 vals=[idx[view,'original',a]['margin'] for a,l in arms];ax.bar(xx+shift,vals,width=.34,label=label,color=color)
 for x,v in zip(xx+shift,vals):ax.text(x,v+(0.00025 if v>=0 else -.00025),f'{v:+.4f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=8)
ax.axhline(0,color='black',lw=.8);ax.set_xticks(xx,[l for a,l in arms]);ax.set_ylim(-.007,.012);ax.set_ylabel('正确参考 − 错误参考：mean MaxSim');ax.set_title('颜色案例：固定003333／003331，正值表示正确候选胜出');ax.legend();ax.grid(axis='y',alpha=.15);fig.tight_layout();save(fig,'15_color_interventions')
binary=cr('automatic_curve.csv');weighted=cr('weighted_automatic_curve.csv');fig,axes=plt.subplots(1,3,figsize=(14,4.5))
nv=int(next(r for r in weighted if r['scope']=='validation_query_face_sufficient')['n'])
for ax,scope,title in zip(axes,['all','test','validation_query_face_sufficient'],['全量1117张','历史测试161张',f'更正名单后的验证{nv}张（敏感性）']):
 for data,mode,label,color,style in [(binary,'exact','任一编号·精确','#89979c','--'),(binary,'one_edit_min8','任一编号·容许1字符差异','#bd9071','--'),(weighted,'exact','逐编号累加·精确','#287387','-'),(weighted,'one_edit_min8','逐编号累加·容许1字符差异','#b6533d','-')]:
  rr=sorted([r for r in data if r['scope']==scope and r['mode']==mode],key=lambda r:float(r['weight']))
  ax.plot([float(r['weight']) for r in rr],[int(r['correct']) for r in rr],marker='o',ls=style,color=color,label=label)
 ax.axvline(0,ls=':',color='black');ax.set(title=title,xlabel='编号奖励权重',ylabel='正确张数');ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(alpha=.15)
 if scope=='validation_query_face_sufficient':
  values=[int(r['correct']) for r in binary+weighted if r['scope']==scope];ax.set_ylim(min(values)-.5,max(values)+.5)
axes[0].legend(fontsize=8);fig.suptitle('自动数字串对照：修正Query名单后重算；锁定主方案保持1041',fontsize=13);fig.tight_layout();save(fig,'16_identifier_curves')
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
canvas=Image.new('RGB',(1100,750),'white');draw=ImageDraw.Draw(canvas)
for j,(name,label) in enumerate([('query_face_color','照片：人工指定侧面并校正几何'),('reference_003333_left_color','正确参考003333：同侧面'),('reference_003331_left_color','错误参考003331：同侧面')]):
 im=Image.open(D/'color_previews'/('face_'+name+'.png')).rotate(-90,expand=True);im.thumbnail((1040,200));draw.text((30,j*250+10),label,font=font,fill='black');canvas.paste(im,((1100-im.width)//2,j*250+48))
canvas.save(D/'color_face_comparison.jpg',quality=95)
print('FIGURES DONE')
