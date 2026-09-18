#!/usr/bin/env python3
"""Publication figure of an algebra identity; no model or experimental inputs."""
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR','/tmp/rc_new_hyp_product_contrast_matplotlib_v1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/figures/new_hyp_product_contrast_v1'
BASENAME='new_hyp_product_contrast_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
        'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':BASENAME,
        'mathtext.fontset':'dejavusans','savefig.facecolor':'white'})
    coordinate=np.linspace(-.98,.98,401,dtype=np.float64)
    x,y=np.meshgrid(coordinate,coordinate,indexing='xy')
    additive=x+y;product=(x+y)/(1+x*y);residual=product-additive
    assert np.all(1+x*y>0) and np.isfinite(product).all()
    assert np.max(np.abs(product))<1
    # Positive M/L representatives make the displayed identity directly checkable.
    mc,mw=(1+x)/2,(1-x)/2;lc,lw=(1+y)/2,(1-y)/2
    direct=(mc*lc-mw*lw)/(mc*lc+mw*lw)
    identity_residual=float(np.max(np.abs(direct-product)))
    assert identity_residual<1e-14
    fig,axes=plt.subplots(1,3,figsize=(7.5,3.05))
    fig.subplots_adjust(left=.065,right=.985,top=.805,bottom=.24,wspace=.36)
    definitions=[('a  Additive contrast',r'$x+y$',additive,2.,[-2,-1,0,1,2]),
                 ('b  Product contrast',r'$(x+y)/(1+xy)$',product,2.,[-2,-1,0,1,2]),
                 ('c  Nonlinear correction',r'$(x+y)/(1+xy)-(x+y)$',residual,1.,[-1,-.5,0,.5,1])]
    for ax,(label,formula,values,limit,ticks) in zip(axes,definitions):
        image=ax.imshow(values,origin='lower',extent=[-.98,.98,-.98,.98],
            interpolation='nearest',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),rasterized=True)
        ax.set_title(label+'\n'+formula,pad=7)
        ax.set_xlabel(r'Mass contrast $x=dM$',labelpad=3)
        ax.set_ylabel(r'Content contrast $y=dL$',labelpad=3)
        ax.set_xticks([-.98,0,.98],labels=['−0.98','0','0.98'])
        ax.set_yticks([-.98,0,.98],labels=['−0.98','0','0.98'])
        ax.tick_params(length=2.5,pad=2)
        ax.contour(x,y,values,levels=[0],colors='#555555',linewidths=.55,linestyles='dashed')
        cb=fig.colorbar(image,ax=ax,orientation='horizontal',pad=.29,fraction=.06,aspect=25,ticks=ticks)
        cb.ax.tick_params(length=2,pad=2,labelsize=7)
        cb.outline.set_linewidth(.5)
        cb.set_label('Contrast value' if limit==2 else 'Difference in contrast',fontsize=7,labelpad=3)
    fig.text(.5,.975,'A linear head can read a nonlinear quality–content relation',ha='center',va='top',fontsize=10)
    fig.text(.5,.035,'Ideal positive M, L; ε = 0.  Panels a–b share the same color scale.  Algebra illustration; no experimental data.',
        ha='center',va='bottom',fontsize=7,color='#333333')
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[]
    for suffix in ('svg','pdf','png'):
        p=OUT/(BASENAME+'.'+suffix)
        if suffix=='svg':metadata={'Date':None,'Creator':BASENAME,'Description':'Algebra only; positive mass/content, epsilon zero; no experimental data.'}
        elif suffix=='pdf':metadata={'CreationDate':None,'ModDate':None,'Creator':BASENAME,'Title':'Product contrast as a nonlinear joint calibration term','Subject':'Algebra only, no empirical gain claim.'}
        else:metadata={'Software':BASENAME}
        fig.savefig(p,format=suffix,dpi=300,metadata=metadata)
        paths.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size})
    plt.close(fig)
    manifest={'status':'NEW_HYP_PRODUCT_CONTRAST_ALGEBRA_FIGURE_CREATED','source_program':{'path':str(Path(__file__).resolve().relative_to(ROOT)),'sha256':sha(Path(__file__))},
        'artifacts':paths,'numpy_version':str(np.__version__),'matplotlib_version':str(matplotlib.__version__),
        'domain':{'x':[-.98,.98],'y':[-.98,.98],'samples_per_axis':401,'dtype':'float64'},
        'color_scales':{'additive':[-2,2],'product':[-2,2],'residual':[-1,1]},
        'extrema':{name:{'min':float(v.min()),'max':float(v.max())} for name,v in [('additive',additive),('product',product),('residual',residual)]},
        'direct_positive_product_identity_max_abs_residual':identity_residual,
        'data_source':'Analytic rectangular mesh only; no experimental scores, models, labels or predictions.',
        'assumptions':['M and L are strictly positive.','S=M*L in real arithmetic.','epsilon=0 and no denominator floor.'],
        'scientific_scope':'Shows the functional difference from an additive readout; does not demonstrate empirical gain, causal necessity, novelty of the identity, or population generalization.'}
    p=OUT/'manifest.json';p.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'manifest_sha256':sha(p),'artifacts':paths,'identity_residual':identity_residual},indent=2))
if __name__=='__main__':main()
