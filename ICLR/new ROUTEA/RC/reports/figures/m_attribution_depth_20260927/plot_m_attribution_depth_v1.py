#!/usr/bin/env python3
"""Three-panel scientific summary of completed frozen-POST M interventions.

Reads accepted statistics only; no model or new experimental computation.
Produces editable-source provenance, exact plotted values, PNG and vector PDF.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import numpy as np

RC=Path(__file__).resolve().parents[1]
DEFAULT_SOURCE=RC/'results/rc_post_m_level_gap_v1'
DEFAULT_OUTPUT=RC/'reports/figures/m_attribution_depth_20260927'
PANELS=(
    dict(scope='phase9/log_M/native_amplitude',title='LOCAL → GLOBAL phase\nnative amplitude',queries=9,groups=9,letter='a',ylim=(-.00155,.00118)),
    dict(scope='jp128/log_M/P_with_J',title='Add distribution input P\nwith joint input J',queries=120,groups=45,letter='b',ylim=(-.005,.072)),
    dict(scope='jp128/log_M/J_without_P',title='Add joint input J\nwithout distribution input P',queries=120,groups=45,letter='c',ylim=(-.041,.043)),
)
EFFECTS=('symmetric_mu','symmetric_delta','total')
LABELS=('Common level\n$\\mu$','Relative gap\n$\\delta$','Total')
COLORS=('#B95746','#187F87','#253F5A')


def read(path):return json.loads(Path(path).read_text())

def binding(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return dict(path=str(path.resolve()),sha256=h.hexdigest())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=DEFAULT_SOURCE);ap.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);args=ap.parse_args()
    source=args.source;out=args.output;out.mkdir(parents=True,exist_ok=True)
    validation=read(source/'validation.json');second=read(source/'second_arithmetic_validation.json')
    assert validation['status']=='POST_M_LEVEL_GAP_INDEPENDENT_PASS'
    assert second['status']=='POST_M_LEVEL_GAP_SECOND_ARITHMETIC_PASS'
    assert binding(source/'result.json')==validation['result']
    result=read(source/'result.json');rows=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
        'xtick.labelsize':9.5,'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'})
    fig,axes=plt.subplots(1,3,figsize=(11.6,5.3),sharey=False)
    fig.subplots_adjust(left=.095,right=.985,bottom=.295,top=.705,wspace=.38)
    fig.suptitle('Common M level can reinforce or offset relative separation',x=.095,y=.97,ha='left',fontsize=15,fontweight='semibold',color='#20354B')
    fig.text(.095,.908,'Frozen POST readout  |  log M coordinates  |  signed target − fixed-wrong content gap',ha='left',fontsize=10,color='#4C6073')
    for ax,panel in zip(axes,PANELS):
        stats=result['summary'][panel['scope']]['POST_content_gap'];vals=[]
        for i,effect in enumerate(EFFECTS):
            r=stats[effect];assert r['groups']==panel['groups']
            lo,hi=r['exploratory_bootstrap95'];value=r['mean'];vals.append(value)
            assert lo<=value<=hi
            ax.errorbar(i,value,yerr=np.asarray([[value-lo],[hi-value]]),fmt='o',markersize=7,
                color=COLORS[i],ecolor=COLORS[i],elinewidth=1.8,capsize=5,capthick=1.6,zorder=4)
            rows.append(dict(panel=panel['letter'],scope=panel['scope'],queries=panel['queries'],groups=panel['groups'],
                endpoint='POST_content_gap',coordinate='log_M',effect=effect,group_mean=value,
                exploratory_group_bootstrap95_lower=lo,exploratory_group_bootstrap95_upper=hi))
        assert abs(vals[0]+vals[1]-vals[2])<1e-12
        ax.set_ylim(*panel['ylim']);ax.set_xlim(-.45,2.45)
        ax.axvspan(1.62,2.38,color='#F1F4F7',zorder=0)
        ax.axhline(0,color='#485868',linewidth=.85,zorder=2)
        ax.yaxis.grid(True,color='#DCE3E9',linewidth=.6,alpha=.8,zorder=1)
        ax.set_axisbelow(True)
        ax.set_xticks(range(3),LABELS)
        ax.tick_params(axis='x',length=0,pad=9)
        ax.tick_params(axis='y',length=3,color='#718292')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        fmt=ScalarFormatter(useMathText=True);fmt.set_scientific(True);fmt.set_powerlimits((0,0));ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.get_offset_text().set_fontsize(9)
        for name in ('left','bottom'):ax.spines[name].set_color('#81909D');ax.spines[name].set_linewidth(.75)
        ax.set_title(f"({panel['letter']}) {panel['title']}",loc='left',pad=37,fontweight='semibold',color='#20354B')
        ax.text(0,1.075,f"{panel['queries']} queries · {panel['groups']} groups",transform=ax.transAxes,ha='left',va='bottom',fontsize=9,color='#526678')
    axes[0].set_ylabel('Change in POST content gap\n(not accuracy)',labelpad=11)
    fig.text(.095,.167,'Points: group means. Whiskers: exploratory 95% group-bootstrap intervals. Each panel has its own y-scale.',fontsize=9.2,color='#415569')
    fig.text(.095,.117,r'Shown: symmetric contributions of $\mu$ and $\delta$; their sum equals total. Four phase directions are averaged within groups.',fontsize=9.2,color='#415569')
    fig.text(.095,.066,'Post-hoc, coordinate-dependent mechanism diagnosis; not unique causal attribution. Panels overlap and are not pooled.',fontsize=9.2,color='#415569')
    png=out/'post_M_level_gap_three_panels.png';pdf=out/'post_M_level_gap_three_panels.pdf'
    fig.savefig(png,dpi=300)
    fig.savefig(pdf,metadata={'Title':'Frozen POST: M common level and relative gap','Subject':'Post-hoc log-M finite-change decomposition; group-bootstrap confidence intervals','Author':'new HYP research'})
    plt.close(fig)
    with (out/'plotted_values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    sourcecopy=out/Path(__file__).name
    if sourcecopy.resolve()!=Path(__file__).resolve():shutil.copy2(__file__,sourcecopy)
    manifest=dict(status='M_ATTRIBUTION_DEPTH_FIGURE_COMPLETE',source_result=binding(source/'result.json'),
        source_validation=binding(source/'validation.json'),second_validation=binding(source/'second_arithmetic_validation.json'),
        program=binding(Path(__file__)),program_copy=binding(sourcecopy),plotted_values=rows,
        artifacts=[binding(png),binding(pdf),binding(out/'plotted_values.csv')],
        no_new_experiments=True,no_model_forwards=True,separate_y_axes=True,unit='signed POST target-minus-fixed-wrong content gap change',
        scope='Phase9 uses9queries/9groups; full120 target-present of the existing128panel uses45groups. The panels overlap.',
        uncertainty='Exploratory group-bootstrap intervals copied from accepted result; post-hoc coordinate-dependent finite-difference attribution, not independent confirmation.')
    (out/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (out/'README.md').write_text('''# M attribution depth: three-panel scientific figure

`post_M_level_gap_three_panels.png` is the 300-dpi raster figure; the matching PDF is vector output with embedded fonts. `plotted_values.csv` contains the exact nine points and confidence limits. `plot_m_attribution_depth_v1.py` is the plotting source; `figure_manifest.json` records source and output SHA256 hashes.

The endpoint is the **change in frozen POST target-minus-fixed-wrong content gap**, not retrieval accuracy. Means and exploratory95% group-bootstrap intervals are copied from accepted results. Each panel has its own clearly labelled y-scale. The phase panel has9queries/9groups; the two J/P panels use all120 target-present queries of the existing128panel and45groups. They overlap and are not pooled.

The log-M common-level and relative-gap terms are symmetric finite-change contributions whose sum equals total. The figure is a post-hoc, coordinate-dependent mechanism diagnosis and does not establish unique causal attribution. Full details, log-odds sensitivity, domain limits and negative results are in the companion report `reports/REPORT_POST_M_LEVEL_GAP_INTERPRETATION_20260927.md`.

Rerun with explicit paths after downloading:

```sh
python plot_m_attribution_depth_v1.py --source /path/to/rc_post_m_level_gap_v1 --output /path/to/figure_output
```
''')
    print(json.dumps(dict(status=manifest['status'],png=str(png),pdf=str(pdf),points=len(rows),new_forwards=0)))

if __name__=='__main__':main()
