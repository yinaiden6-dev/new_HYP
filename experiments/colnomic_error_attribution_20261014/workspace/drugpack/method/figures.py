import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import RESULTS, read, save, sha


def metrics(selection, dataset, name, k=None):
    split = 'test' if dataset.startswith('medicine') else 'transfer'
    k = 32 if dataset == 'forb32' else k or selection['runs'][name]['candidate_k']
    return read(RESULTS/'evaluation'/dataset/split/f'{name}_k{k}'/'complete.json')['metrics']


def render(selection):
    assert read(RESULTS/'verification_final.json')['status']=='PASS'
    folder = RESULTS/'figures'
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    files = []
    def export(fig, name):
        for suffix in ('pdf','png'):
            path = folder/f'{name}.{suffix}'
            fig.savefig(path,dpi=200,bbox_inches='tight')
            files.append(dict(path=str(path),sha256=sha(path)))
        plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    datasets = ('medicine','grozi','milk','forb32')
    labels = ('Medicine test','GroZi','MILK','FORB')
    colors = ('#1764ab','#dd7625','#25845b','#8155a6')
    for axis, regime in zip(axes,('strict','catalog'),strict=True):
        methods = (('Selected adapter',selection['primary'][regime]),
                   ('External head',selection['selected'][regime+'/external']),
                   ('RoMa-free Reader',selection['selected'][regime+'/reader']),
                   ('Fixed weighted matching','weighted_L'))
        for offset,((label,name),color) in enumerate(zip(methods,colors,strict=True)):
            estimates, lows, highs = [], [], []
            for ds in datasets:
                ds = 'medicine_catalog' if ds=='medicine' and regime=='catalog' and name!='weighted_L' else ds
                result = metrics(selection,ds,name,k=128 if name=='weighted_L' else None)
                estimates.append(100*(result['micro']['R@1']-result['raw']['R@1']))
                interval = result['paired_intervals']['micro']['R@1']
                lows.append(100*interval['low']); highs.append(100*interval['high'])
            x = np.arange(len(datasets))+(offset-1.5)*.16
            axis.vlines(x,lows,highs,color=color,linewidth=1.5)
            axis.scatter(x,estimates,color=color,s=27,label=label,zorder=3)
        axis.axhline(0,color='#555555',linestyle='--',linewidth=.8)
        axis.set(xticks=range(4),xticklabels=labels,title=regime.capitalize()+' training',
                 ylabel='R@1 change vs frozen ColNomic 7B (pp)')
        axis.grid(axis='y',alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.06),
               frameon=False,fontsize=8,ncol=4)
    export(fig,'paired_r1_changes')

    fig, axes = plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    ks = (8,16,32,64,128)
    families = (('hidden','Hidden adapter'),('patch_hidden','Patch adapter'),
                ('embedding','Embedding adapter'),('external','External head'),('reader','Reader'))
    for axis, regime in zip(axes,('strict','catalog'),strict=True):
        ds = 'medicine_catalog' if regime=='catalog' else 'medicine'
        for family,label in families:
            name = selection['selected'][regime+'/'+family]
            rows = [metrics(selection,ds,name,k) for k in ks]
            axis.plot(ks,[100*r['micro']['R@1'] for r in rows],marker='o',markersize=3,label=label)
        axis.axhline(100*rows[0]['raw']['R@1'],color='#555555',linestyle='--',label='Frozen ColNomic 7B')
        axis.set_xscale('log',base=2)
        axis.set(xticks=ks,xticklabels=[str(k) for k in ks],xlabel='Reranking candidates K',
                 ylabel='Medicine test R@1 (%)',title=regime.capitalize()+' training')
        axis.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=7,ncol=2)
    export(fig,'candidate_sensitivity')
    save(folder/'manifest.json',dict(selection_sha256=sha(RESULTS/'selection.json'),
        verification_sha256=sha(RESULTS/'verification_final.json'),files=files,
        intervals='95% paired group bootstrap; bars compare each method with frozen ColNomic 7B.',
        candidate_sensitivity='Frozen validation-selected checkpoints. Test K sweeps are diagnostics, not test-selected operating points.'))


if __name__=='__main__':
    render(read(RESULTS/'selection.json'))
