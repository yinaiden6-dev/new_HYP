"""Plot verified cached evidence using only previously published thumbnails.

Usage: python build_heatmap_gallery.py /path/to/release
No model inference, score changes, new labels, or original-image publication.
"""
from pathlib import Path
import sys,json,hashlib
from functools import lru_cache
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,LogNorm,Normalize,ListedColormap,to_rgba
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox,OffsetImage

R=Path(sys.argv[1]);E=R/'joint_evidence';B=E/'remaining49_review'
O=B/'heatmap_gallery';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
review=read(B/'REVIEW49.json');cases=review['cases'];viewer=read(E/'CASE_VIEWER_DATA.json');index=viewer['imageIndex']
old={x['query']:x for x in read(E/'roma_coordinates_20261015/MECHANISM54.json')}
ledger={x['query']:x for x in read(E/'roma_coordinates_20261015/MECHANISM54_COORDINATES.json')}
types={'版式/空间结构':'Layout','颜色':'Color','小码/条码编号':'Small code / barcode','剂量/规格文字':'Dose / specification','文字内容差异':'Other text','弱线索/可见面受限':'Limited visible cues'}
keys=['legacy_full','legacy_subject','repaired_image','text','code']
labels=['Old image','Subject tokens','Repaired image','Ordinary OCR','Identifier']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':130})
red='#be4946';blue='#276d9c';green='#278265'
rank_cmap=ListedColormap([to_rgba(green)]+list(plt.get_cmap('YlOrRd')(np.linspace(.12,1,255))))

@lru_cache(maxsize=40)
def sheet(name):
    with Image.open(R/name) as im:return im.convert('RGB').copy()
def tile(q):
    z=index[q];x=z['col']*320;y=z['row']*320
    return sheet(z['sheet']).crop((x,y,x+320,y+320))
def photo(q):
    z=index[q];w,h=z['width'],z['height'];scale=min(312/w,312/h,1)
    ww,hh=round(w*scale),round(h*scale);x,y=(320-ww)//2,(320-hh)//2
    return tile(q).crop((x,y,x+ww,y+hh)),h/w
def picture(ax,q,title):
    ax.imshow(tile(q));ax.axis('off');ax.set_title(title,loc='left',fontsize=11,weight='bold')
def bbox(ax,c):
    x1,y1,x2,y2=c['locator']['subject_bbox'];w=c['locator']['width'];h=c['locator']['height']
    ax.add_patch(Rectangle((x1/w,y1/h),(x2-x1)/w,(y2-y1)/h,fill=False,edgecolor='#00bdd7',linewidth=1.2))
def query_map(ax,c,xy,vals,title,cmap,norm):
    im,ratio=photo(c['query']);ax.imshow(im,extent=(0,1,1,0))
    xx=np.unique(xy[:,0]);yy=np.unique(xy[:,1]);assert len(xx)*len(yy)==len(xy)
    grid=np.full((len(yy),len(xx)),np.nan);grid[np.searchsorted(yy,xy[:,1]),np.searchsorted(xx,xy[:,0])]=vals
    assert np.isfinite(grid).all()
    edges=lambda a:np.r_[max(0,a[0]-(a[1]-a[0])/2),(a[:-1]+a[1:])/2,min(1,a[-1]+(a[-1]-a[-2])/2)]
    m=ax.pcolormesh(edges(xx),edges(yy),grid,cmap=cmap,norm=norm,alpha=.62,rasterized=True,shading='flat')
    bbox(ax,c);ax.set_xlim(0,1);ax.set_ylim(1,0);ax.set_aspect(ratio);ax.axis('off');ax.set_title(title,fontsize=10,loc='left')
    return m
def ascii_term(x):return x.encode('ascii','backslashreplace').decode('ascii')
def save(fig,name):
    fig.savefig(O/name,facecolor='white',bbox_inches='tight');plt.close(fig)

# Independently check every plotted quantity and pair before any output.
arrays={};checks=[];missing=[]
for c in cases:
    assert abs(sum(c['parts'].values())-c['margin'])<1e-7
    q=c['query'];p=E/'roma_coordinates_20261015/token_deltas54'/f'{q}.npz'
    if not p.exists():missing.append(q);continue
    assert old[q]['positive']==c['positive'] and old[q]['negative']==c['negative']
    with np.load(p) as a:arrays[q]={k:a[k].copy() for k in a.files}
    a=arrays[q];n=len(a['q_xy']);delta=a['MaxSim_positive_contribution']-a['MaxSim_negative_contribution']
    assert np.max(np.abs(delta-a['MaxSim_delta']))<1e-12
    assert abs(float(delta.sum())-ledger[q]['patch_contributions']['MaxSim']['total'])<2e-6
    for sign in ['positive','negative']:
        assert abs(float(a[f'MaxSim_{sign}_contribution'].sum())-ledger[q][f'{sign}_features']['MaxSim'])<2e-6
    loc=c['locator'];box=np.array(loc['subject_bbox'])/np.array([loc['width'],loc['height']]*2)
    measured=((a['q_xy']>=box[:2])&(a['q_xy']<=box[2:])).all(1)
    assert np.array_equal(measured,a['subject_mask']),q
    checks.append({'query':q,'positive':c['positive'],'negative':c['negative'],'tokens':n,'patch_maxsim_gap':float(delta.sum()),'inside_gap':float(delta[a['subject_mask']].sum()),'outside_gap':float(delta[~a['subject_mask']].sum()),'pair_match':True,'human_box_mask_verified':True})
assert len(cases)==49 and len(arrays)==47 and len(missing)==2
limit=float(np.quantile(np.concatenate([np.abs(a['MaxSim_delta']*len(a['q_xy']))for a in arrays.values()]),.98))
overview_only='--overview-only' in sys.argv
meta=read(O/'GALLERY_DATA.json')['cases'] if overview_only else []
for i,c in enumerate([] if overview_only else cases):
    q=c['query'];a=arrays.get(q);fig,axs=plt.subplots(2,4,figsize=(15.5,8.5),layout='constrained',gridspec_kw={'width_ratios':[1,1,1,1.05]})
    picture(axs[0,0],q,'Query | human subject box')
    loc=c['locator'];w,h=loc['width'],loc['height'];scale=312/max(w,h);ox=(320-round(w*scale))//2;oy=(320-round(h*scale))//2
    x1,y1,x2,y2=loc['subject_bbox'];axs[0,0].add_patch(Rectangle((ox+x1*scale,oy+y1*scale),(x2-x1)*scale,(y2-y1)*scale,fill=False,edgecolor='#00bdd7',linewidth=1.5))
    picture(axs[0,1],c['positive'],'Correct | '+c['positive'])
    picture(axs[0,2],c['negative'],'Wrong in 1058 | '+c['negative'])
    vals=[c['parts'][k]for k in keys];ax=axs[0,3];ax.barh(labels,vals,color=[blue if v>=0 else red for v in vals]);ax.axvline(0,color='#555',lw=.7);ax.invert_yaxis();ax.margins(x=.32)
    for j,v in enumerate(vals):ax.text(v,j,f' {v:+.3f} ',va='center',ha='left' if v>=0 else 'right',fontsize=9)
    ax.set_title(f'Final score gap = {c["margin"]:+.4f}',loc='left',fontsize=11,weight='bold');ax.set_xlabel('Correct minus wrong; actual score units')
    if a is not None:
        n=len(a['q_xy']);pos=a['MaxSim_positive_contribution']*n;neg=a['MaxSim_negative_contribution']*n
        m=query_map(axs[1,0],c,a['q_xy'],pos,'Frozen MaxSim to correct','viridis',Normalize(0,1))
        query_map(axs[1,1],c,a['q_xy'],neg,'Frozen MaxSim to wrong','viridis',Normalize(0,1))
        fig.colorbar(m,ax=[axs[1,0],axs[1,1]],shrink=.55,pad=.02,label='Per-patch maximum similarity (0 to 1)')
        m=query_map(axs[1,2],c,a['q_xy'],pos-neg,'Token difference: correct minus wrong','RdBu',TwoSlopeNorm(0,-limit,limit))
        fig.colorbar(m,ax=axs[1,2],shrink=.55,pad=.02,label='Blue: correct; red: wrong')
    else:
        for ax in axs[1,:3]:ax.axis('off');ax.text(.5,.5,'No verified token map\nfor this candidate pair.\nScore evidence remains available.',ha='center',va='center',transform=ax.transAxes,color='#69747d')
    terms=c['text_trace']['negative_terms'][:4]+c['text_trace']['positive_terms'][:2];ax=axs[1,3]
    if terms:
        values=[t['total'] for t in terms];names=[ascii_term(t['word'])[:23] for t in terms]
        ax.barh(range(len(terms)),values,color=[blue if v>=0 else red for v in values]);ax.set_yticks(range(len(terms)),names);ax.invert_yaxis();ax.axvline(0,color='#555',lw=.7);ax.margins(x=.35)
        for j,v in enumerate(values):ax.text(v,j,f' {v:+.3f} ',va='center',ha='left' if v>=0 else 'right',fontsize=8)
        shown=sum(values);ax.set_title(f'OCR terms (subset); all = {c["parts"]["text"]:+.3f}',loc='left',fontsize=10);ax.set_xlabel('Remainder = '+f'{c["parts"]["text"]-shown:+.3f}')
    else:ax.axis('off');ax.text(.5,.5,'No discriminative ordinary-OCR terms',ha='center',va='center',wrap=True,transform=ax.transAxes)
    cr=c['composed_controls']['retained_old_codes_plus_union_weight4'];fig.suptitle(f'{q} | {types[c["visual"]["primary"]]}\n1058 primary rank {c["rank"]}  /  fixed 1063 control rank {cr}',fontsize=14,weight='bold')
    fig.supxlabel('Same fixed candidate pair. Frozen patch maps are NOT the final fused score or causal pixel attribution.\nAll token-difference maps share a 98th-percentile color limit; no smoothing. Published thumbnails only.',fontsize=9)
    name=q+'_evidence.png';save(fig,name)
    meta.append({'query':q,'figure':name,'type':c['visual']['primary'],'tags':c['visual']['tags'],'positive':c['positive'],'negative':c['negative'],'rank1058':c['rank'],'rank1063':cr,'has_token_map':a is not None,'finding':c['finding'],'conclusion':c['bounded_conclusion'],'direction':c['direction']})
    if (i+1)%10==0:print('CASE_FIGURES',i+1,flush=True)

ordered=sorted(cases,key=lambda c:(list(types).index(c['visual']['primary']),c['query']))
def overview(matrix,cols,name,title,kind):
    fig,ax=plt.subplots(figsize=(12.5,17),layout='constrained')
    if kind=='score':norm=TwoSlopeNorm(0,-3,3);cmap='RdBu';display=matrix
    else:norm=LogNorm(1,max(2,float(matrix.max())));cmap=rank_cmap;display=matrix
    im=ax.imshow(display,cmap=cmap,norm=norm,aspect='auto');ax.set_xticks(range(len(cols)),cols);ax.xaxis.tick_top();ax.tick_params(axis='x',labelsize=9,length=0,pad=10)
    ax.set_yticks(range(len(ordered)),[c['query'][6:14]+'  '+types[c['visual']['primary']]for c in ordered],fontsize=8)
    for row,c in enumerate(ordered):
        thumb=tile(c['query']);thumb.thumbnail((27,27));ab=AnnotationBbox(OffsetImage(thumb,zoom=1),(len(cols)-.05,row),frameon=False,box_alignment=(0,.5),annotation_clip=False);ax.add_artist(ab)
        for col,val in enumerate(matrix[row]):
            ax.text(col,row,f'{val:+.2f}'if kind=='score' else str(int(val)),ha='center',va='center',fontsize=8,color='white'if(kind=='score'and abs(val)>1.7)or(kind!='score'and (val==1 or norm(val)>.7))else '#172b3a')
    ax.set_title(title+'\nQuery thumbnails at right; all 49 fixed primary failures',loc='left',fontsize=14,weight='bold',pad=42)
    fig.colorbar(im,ax=ax,pad=.07,shrink=.35,label='Correct minus wrong score (colors clip at +/-3)'if kind=='score'else 'Full-gallery rank (log color scale; 1 is best)')
    save(fig,name)
matrix=np.array([[c['parts'][k]for k in keys]+[c['margin']]for c in ordered]);overview(matrix,labels+['Total'],'49_branch_heatmap.png','Which branch supports the wrong candidate?','score')
rankcols=['Primary\n1058','Repaired\nimage only','Mixed\nimage only','Repaired\nimage + OCR','Remove\nordinary OCR','Remove\nidentifiers','Fixed\n1063']
matrix=np.array([[c['rank'],c['pure_repaired_image_rank'],c['controls']['image_mix_no_ocr'],c['controls']['direct_repaired_image_current_ocr'],c['controls']['remove_text_keep_code'],c['controls']['remove_code_keep_text'],c['composed_controls']['retained_old_codes_plus_union_weight4']]for c in ordered]);overview(matrix,rankcols,'49_intervention_heatmap.png','Which fixed intervention changes full-gallery rank?','rank')
targets=[c for c in cases if c['target18']];methods=['repaired_image_only','mixed_image_only','repaired_image_ocr','mixed_image_ocr'];matrix=np.array([[c['interface_factorial']['pairs'][m]['rank']for m in methods]for c in targets])
fig,ax=plt.subplots(figsize=(12,8),layout='constrained');rank_norm=LogNorm(1,max(matrix.max(),2));ax.imshow(matrix,cmap=rank_cmap,norm=rank_norm,aspect='auto')
ax.set_xticks(range(4),['Repaired image','+ old image mix','+ OCR only','+ old mix + OCR']);ax.xaxis.tick_top();ax.set_yticks(range(18),[c['query'][6:14]for c in targets])
for i,c in enumerate(targets):
    for j,val in enumerate(matrix[i]):ax.text(j,i,str(val),ha='center',va='center',weight='bold',color='white'if val==1 or rank_norm(val)>.7 else '#172b3a')
    ax.text(3.6,i,c['interface_factorial']['mechanism_when_repaired_image_correct'].replace('_',' '),va='center',fontsize=9)
ax.set_title('18 image-correct cases: a complete 2 x 2 score intervention\nEach cell is the correct reference rank in the full gallery',loc='left',fontsize=14,weight='bold',pad=40);save(fig,'18_interface_heatmap.png')
protocol={'status':'verified','case_figures':49,'token_spatial_maps':47,'token_panels':141,'overview_heatmaps':3,'missing_token_map_queries':missing,'primary':1058,'exploratory_composition':1063,'exclusions':13,'token_pair_checks':checks,'token_delta_common_color_limit':limit,'token_color_rule':'98th percentile of abs(per-patch positive maximum minus negative maximum), pooled across 47 cases; only display colors clipped, not numeric values','token_values':'NPZ contributions are mean-score contributions (max similarity / N); multiply by N for patch heatmap. No interpolation. Blue favors correct, red favors wrong. Stored patch means are separate from final normalized mixture.','image_source':'Only already-public 320px sprite tiles; original-resolution files are not read.','causal_limit':'Score decomposition is exact; token maps are associational, not proof of a unique causal pixel mechanism.','cases':meta}
(O/'GALLERY_DATA.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
files=[O/x['figure']for x in meta]+[O/n for n in ['49_branch_heatmap.png','49_intervention_heatmap.png','18_interface_heatmap.png','GALLERY_DATA.json']]
(O/'FILES.json').write_text(json.dumps([{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in files],indent=2)+'\n')
print('GALLERY_COMPLETE',json.dumps({'cases':49,'spatial':47,'missing':missing,'figures':52}),flush=True)
