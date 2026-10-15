"""After experiments and verification finish, update ONLY the existing report."""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
import argparse,sys,time,json,csv,shutil,re,hashlib,subprocess,tarfile
import numpy as np
from PIL import Image,ImageOps
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;A=D/'roma_coordinates';OUT=P/'release';E=OUT/'joint_evidence';DEST=E/'roma_coordinates_20261015'
sys.path.insert(0,str(P));from experiment_utils import read,write,csvwrite

def cr(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def js(x):return json.dumps(x,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')

def oriented_rgb(path):
    with Image.open(path) as im:
        orientation=int(im.getexif().get(274,1));rgb=im.copy()
    op={2:Image.Transpose.FLIP_LEFT_RIGHT,3:Image.Transpose.ROTATE_180,4:Image.Transpose.FLIP_TOP_BOTTOM,5:Image.Transpose.TRANSPOSE,6:Image.Transpose.ROTATE_270,7:Image.Transpose.TRANSVERSE,8:Image.Transpose.ROTATE_90}.get(orientation)
    if op is not None:rgb=rgb.transpose(op)
    rgb=rgb.convert('RGB');rgb.info.clear();return rgb

def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def figure(summary,factorial):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=cr(A/'curve.csv');fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    labels={'M':'Overlap only','S':'Visibility + MaxSim','G':'Geometry + confidence','D':'Dense constraint','H':'Homography constraint','Droll':'Shuffled dense','Hroll':'Shuffled homography','C':'Matched color'}
    for row,family in enumerate(['image','joint']):
        for col,scope in enumerate(['all','validation']):
            ax=axes[row,col]
            for feat in labels:
                rr=[r for r in rows if r['family']==family and r['scope']==scope and r['feature']==feat and float(r['gate'])==.5]
                ax.plot([float(r['beta']) for r in rr],[100*float(r['macro_accuracy'] if scope=='validation' else r['accuracy']) for r in rr],marker='.',label=labels[feat],linestyle='--' if 'roll' in feat else '-')
            ax.set_xscale('symlog',linthresh=.05);ax.set_xlabel('Added feature weight');ax.set_ylabel('Macro Top1 (%)' if scope=='validation' else 'Top1 (%)');ax.set_title(f'{family} / {scope} / subject gate0.5');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2);fig.savefig(DEST/'21_coordinate_controls.png',dpi=170);fig.savefig(DEST/'21_coordinate_controls.svg');plt.close(fig)
    names=['prior_image_prior_ocr','repaired_image_prior_ocr','prior_image_repaired_ocr','repaired_image_repaired_ocr']
    labels=['Prior image\nPrior OCR','Repaired image\nPrior OCR','Prior image\nRepaired OCR','Repaired image\nRepaired OCR']
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained');counts=[factorial['results'][k]['metrics']['all']['correct'] for k in names]
    axes[0].bar(labels,counts,color=['#8a94a6','#28658d','#c28b2c','#268174']);axes[0].set_ylabel('Correct / 1117');axes[0].set_ylim(0,1117);axes[0].bar_label(axes[0].containers[0]);axes[0].set_title('Image and OCR: fixed 2 x 2 comparison')
    x=np.arange(4);rep=[factorial['results'][k]['metrics']['all']['repairs'] for k in names];harm=[factorial['results'][k]['metrics']['all']['harms'] for k in names]
    axes[1].bar(x-.18,rep,.36,label='Repairs',color='#268174');axes[1].bar(x+.18,harm,.36,label='New errors',color='#b6534e');axes[1].set_xticks(x,labels);axes[1].set_title('Compared with the 1051 baseline');axes[1].legend();fig.savefig(DEST/'22_final_factorial.png',dpi=170);fig.savefig(DEST/'22_final_factorial.svg');plt.close(fig)

def run(wait):
    start=time.time();required=[A/'SUMMARY.json',A/'FINAL_FACTORIAL.json',A/'INDEPENDENT_VERIFICATION.json',A/'HOREKA_PROVENANCE.json',D/'SELECTIVE_OCR_SUMMARY.json',D/'SELECTIVE_OCR_INDEPENDENT_VERIFICATION.json',A/'MECHANISM54_COORDINATES.json']
    while not all(p.exists() for p in required):
        if not wait:raise RuntimeError('Required experiments/verification incomplete')
        if time.time()-start>18*3600:raise TimeoutError('Report wait exceeded18h')
        time.sleep(45)
    summ=read(A/'SUMMARY.json');fact=read(A/'FINAL_FACTORIAL.json');check=read(A/'INDEPENDENT_VERIFICATION.json');prov=read(A/'HOREKA_PROVENANCE.json');ocr=read(D/'SELECTIVE_OCR_SUMMARY.json');ledger=read(A/'MECHANISM54_COORDINATES.json');plan=read(A/'PLAN.json')
    assert summ['status']==fact['status']==ocr['status']=='complete' and check['status']==prov['status']=='passed'
    assert read(D/'SELECTIVE_OCR_INDEPENDENT_VERIFICATION.json')['status']=='passed'
    assert check['files_sha_verified']==len(plan['tasks']) and check['native_diagnostic_pairs_verified']==133 and len(ledger)==54
    report=OUT/'REPORT.html';assert report.exists();backup=OUT/'REPORT.before_coordinates_final_20261015.html'
    if not backup.exists():shutil.copy2(report,backup)
    DEST.mkdir(parents=True,exist_ok=True)
    # Keep large native caches private on the server; publish summaries/delta arrays,
    # reproducible protocols, figures and scripts in the existing evidence folder.
    for f in A.glob('*'):
        if f.is_file() and (f.suffix in ['.json','.csv','.png'] or (f.suffix=='.npz' and 'inputs' not in f.name)) and not any(x in f.name for x in ['JOB','worker','PLAN','COLOR','PARTIAL','CONTINUATION','DELIVERY','REPORT_UPDATE_RECEIPT']):shutil.copy2(f,DEST/f.name)
    shutil.copytree(A/'token_deltas54',DEST/'token_deltas54',dirs_exist_ok=True)
    for name in ['SELECTIVE_OCR_SUMMARY.json','SELECTIVE_OCR_EVAL_PROTOCOL.json','OCR_BASELINE_REPLAY.json','SELECTIVE_OCR_INDEPENDENT_VERIFICATION.json','OCR_EVIDENCE_RETENTION.json','ocr_retention_cases1117.csv','selective_ocr_curve.csv','selective_ocr_cases1117.csv','REPAIR_SUMMARY.json','MECHANISM54.json','TARGETED_SUMMARY.json','QUERY_GALLERY_current_base.json','QUERY_GALLERY_historical.json','fusion_curve.csv','color_curve.csv','routing_curve.csv','ablation_curve.csv']:
        if (D/name).exists():shutil.copy2(D/name,DEST/name)
    for name in ['roma_coordinates.py','evaluate_coordinates.py','coordinate_color.py','verify_coordinates.py','verify_horeka_provenance.py','diagnose_priority.py','plot_correspondences.py','complete_coordinate_report.py','selective_ocr.py','evaluate_selective_ocr.py','verify_selective_ocr.py','test_ocr_retention.py']:
        shutil.copy2(D/name,DEST/name)
    write(DEST/'PROTOCOL.json',plan['protocol']);figure(summ,fact)
    viewer=read(E/'CASE_VIEWER_DATA.json');rows=viewer['rows'];oldranks=[r['incremental_rank'] for r in rows]
    preds={}
    for family in ['image','joint']:
        z=np.load(A/f'{family}_selected.npz');preds[family]={q:dict(rank=int(z['ranks'][i]),winner=str(z['top1'][i])) for i,q in enumerate(z['query_ids'])}
    fr={}
    for x in cr(A/'final_factorial_cases.csv'):fr.setdefault(x['query'],{})[x['method']]=dict(rank=int(x['rank']),winner=x['winner'])
    om={r['query']:r for r in cr(D/'selective_ocr_cases1117.csv')};lm={r['query']:r for r in ledger}
    for r in rows:
        q=r['query'];r['coordinate_evidence']=dict(image=preds['image'][q],joint=preds['joint'][q],factorial=fr[q],ocr=om[q],diagnosis=lm.get(q))
    assert oldranks==[r['incremental_rank'] for r in rows]
    needed={z['winner'] for r in rows for z in [r['coordinate_evidence']['image'],r['coordinate_evidence']['joint'],*r['coordinate_evidence']['factorial'].values()]}
    missing=sorted(needed-set(viewer['imageIndex']));refmap={r['id']:r for r in plan['references'].values()}
    # Full-gallery OCR may introduce a winner outside the RoMa candidate union.
    reference_ocr=read(Path('/mnt/lsdf_zyin/cache/ocr/references.json'))
    for rid in missing:
        if rid not in refmap:refmap[rid]=dict(path=str(Path('/mnt/lsdf_zyin')/reference_ocr[rid]['image_path']))
    for offset in range(0,len(missing),64):
        batch=missing[offset:offset+64];sheet=Image.new('RGB',(2560,2560),'white');name=f'new_reference_images_{offset//64:02}.webp'
        for k,rid in enumerate(batch):
            im=oriented_rgb(refmap[rid]['path'])
            im.thumbnail((310,310));x=k%8*320+(320-im.width)//2;y=k//8*320+(320-im.height)//2;sheet.paste(im,(x,y));viewer['imageIndex'][rid]=dict(sheet='joint_evidence/roma_coordinates_20261015/'+name,col=k%8,row=k//8)
        sheet.save(DEST/name,quality=92)
    write(E/'CASE_VIEWER_DATA.json',viewer)
    prefix='joint_evidence/roma_coordinates_20261015/'
    def metricrow(label,result,base):
        a,v,t,k=[result['metrics'][scope] for scope in ['all','validation','test','retained']]
        return [label,f"{a['correct']}/1117",f"{a['accuracy']:.2%}",f"{v['correct']}/173",f"{t['correct']}/161",f"{k['correct']}/1104",f"{a['repairs']}／{a['harms']}（对{base}）"]
    sr=[]
    for family,label,base in [('image','纯图像',944),('joint','1051联合',1051)]:
        for arm,desc in [('selected','验证集选择'),('selected_nongeometric','仅特征或置信度选择'),('selected_geometric','几何约束选择'),('selected_color','对应区域颜色选择')]:sr.append(metricrow(label+'／'+desc,summ['families'][family]['results'][arm],base))
    sr.append(metricrow('自动主体OCR重新识别',ocr,1051))
    labels={'prior_image_prior_ocr':'原图像＋原OCR','repaired_image_prior_ocr':'修复图像＋原OCR','prior_image_repaired_ocr':'原图像＋修复OCR','repaired_image_repaired_ocr':'修复图像＋修复OCR','joint_rerank_prior_ocr':'联合分支重排＋原OCR','joint_rerank_repaired_ocr':'联合分支重排＋修复OCR','prior_image_retained_ocr':'追加对照：原图像＋保留旧编号证据','repaired_image_retained_ocr':'追加对照：修复图像＋保留旧编号证据'}
    sr.extend(metricrow(labels[k],v,1051) for k,v in fact['results'].items())
    final=fact['results']['repaired_image_repaired_ocr']['metrics']['all'];image=summ['families']['image']['results']['selected']['metrics']['all']
    ic=summ['families']['image']['selected'];jc=summ['families']['joint']['selected']
    section=f'''<!-- COORDINATES_FINAL_BEGIN -->
<h3 id="coordinates-final">补齐对应坐标后的全量验证与最终图像 OCR 联合实验</h3>
<p><strong>本轮14,902个图片对已完成，并通过独立核验。</strong>覆盖全部1117张照片，两条既定分支各自自然Top10的并集；133个诊断对另保留完整原生坐标场。人工指定的正确参考只参与诊断，未加入任何方法的推理候选。其余图库分数和排序仍保留，因此这是全1117张上的Top10重排检验，不是重新计算1117×5405个RoMa匹配。</p>
<p>纯图像验证集选择后为<strong>{image['correct']}/1117</strong>；按先固定图像修复、再加入已固定OCR修复的顺序，最终2×2组合为<strong>{final['correct']}/1117（{final['accuracy']:.2%}）</strong>，相对1051修复{final['repairs']}张、新增{final['harms']}个错误。此处没有逐例按真值选择最优配置。历史数据已经反复分析，本轮全部结果属于探索性验证，原锁定1041主方案保留。</p>
{table(['方案','全部','Top1','验证','历史测试','剔除13张后','修复／新增错误'],sr)}
<p>纯图像选择：{ic['feature']}，权重{ic['beta']}，主体门槛{ic['gate']}。联合重排选择：{jc['feature']}，权重{jc['beta']}，主体门槛{jc['gate']}。参数只按173张原验证集的包装宏平均Top1、宏平均MRR和预设平局规则选择。所有未选中配置仍完整导出；测试分数不能用于改选参数。</p>
<h4>空间匹配到底增加了什么</h4>
<p>M是RoMa重叠置信度；S是区域加权但仍无空间约束的ColNomic MaxSim；G不用ColNomic相似度，以RoMa几何一致性和匹配置信度评分；D和H分别加入密集对应坐标、单一平面变换的距离惩罚。Droll、Hroll打乱对应坐标；C和Croll比较匹配位置的颜色及打乱位置的颜色。比较真实坐标和打乱坐标时使用相同权重、相同主体门槛。RoMa本身是额外视觉模型，几何和颜色也增加了原MaxSim没有使用的信息；提升不能单独证明ColNomic已经完整编码布局。</p>
<img src="{prefix}21_coordinate_controls.png" alt="同候选同门槛的几何、置信度和坐标打乱曲线">
<h4>最终先图像后 OCR 的组合检验</h4>
<p>先固定纯图像选择，将其对原Top10的排序变化映射回同一批原分数位置；随后仅替换既有融合公式中0.4权重的新图像支路并重新归一化。OCR使用已预设自动触发规则覆盖的379张，含原本成功照片；没有人工读数。最后比较旧图像／新图像×旧OCR／新OCR四组。联合分支直接重排另列，避免与“先修图像再融文字”混为一谈。</p>
<img src="{prefix}22_final_factorial.png" alt="图像修复和OCR修复四组对照与误伤">
<p>OCR重新识别的评估先按历史单精度累加规则复算全部1117张旧文字和编号分数，再独立核验新分数及全图库排名。初次双精度复算产生的微小差额已定位为数值累加方式不一致，未放宽校验阈值。详见<a href="{prefix}OCR_BASELINE_REPLAY.json">旧分数精确复算</a>和<a href="{prefix}SELECTIVE_OCR_INDEPENDENT_VERIFICATION.json">OCR独立排名核验与修复／误伤分解</a>。</p>
<p>另增两组编号证据保留对照：对每个候选统一取新旧编号加分的较大值，OCR阈值和权重保持原验证选择。该规则是在观察到重新识别漏掉编号开头、丢失旧匹配奖励后提出的追加实验，不能视为独立验证；它仍可能保留旧的错误匹配。两组结果单列，不替换上面的原定四组。见<a href="{prefix}OCR_EVIDENCE_RETENTION.json">全量证据保留对照</a>。</p>
<h4>原54张错误的逐例证据和修复方向</h4>
<p>54张均已保存固定正负参考的逐Token差额、主体内外贡献、双向匹配和各支路贡献；可在下方逐例检索查看。本轮记录多个可共存的问题，不把一个案例强行归入单一原因。计分代数、匹配干预和像素根因是不同证据层级；不能把尚未成功的修复假设标为因果确认。</p>
<p><button class="jump-filter" data-filter="coordinates_repairs">本轮最终组合修复</button> <button class="jump-filter" data-filter="coordinates_harms">本轮最终组合新增错误</button> <button class="jump-filter" data-filter="coordinates_remaining54">原54张逐Token证据</button></p>
<p><a href="{prefix}mechanism54_coordinates.csv">54张证据与方向</a> · <a href="{prefix}MECHANISM54_COORDINATES.json">逐例完整证据</a> · <a href="{prefix}curve.csv">全部坐标对照曲线数据</a> · <a href="{prefix}FINAL_FACTORIAL.json">最终联合实验</a> · <a href="{prefix}final_factorial_cases.csv">全1117逐例结果</a> · <a href="{prefix}INDEPENDENT_VERIFICATION.json">独立核验</a></p>
<h4>HoreKa 缓存来源与本轮运行环境</h4>
<p>重新核验全部5405参考张量哈希，与找回的HoreKa原始5413行缓存逐位对应；原任务4076613运行于A100。5390行身份名同源一致，其余15行已有标签修订记录，未在本轮修改图库标签。当前Query缓存是439张复用历史特征、678张按h593_encoder_v1生成的混合来源；结果按来源分别统计。RoMa坐标在i14s45运行，ColNomic历史特征保持冻结。历史参考特征与后来的重编码不相等，不能仅凭硬件或缓存差异归因于编码器不稳定。</p>
<p><a href="{prefix}HOREKA_PROVENANCE.json">本轮逐张哈希核验及来源证据</a> · <a href="{prefix}HOREKA_LABEL_REVISIONS.json">15行历史标签对应记录</a>。源码、坐标约定与输入校验一并归档；原生大型坐标缓存保存在服务器实验目录。</p>
<h4>已有修复路线的完整负结果</h4>
<p>本轮另完成统一融合权重、全图色度统计和置信度切换对照：相应验证集选择结果为1046、1046、1049，均未超过原1051固定数字串对照。三个重点案例的裁剪、方向和灰度输入也已在5405行完整图库评分；不能把人工指定正负对上的改善直接当成全图库修复。详见<a href="{prefix}REPAIR_SUMMARY.json">全量结果</a>、<a href="{prefix}QUERY_GALLERY_current_base.json">当前图库输入干预</a>和<a href="{prefix}TARGETED_SUMMARY.json">受控输入实验</a>。</p>
'''
    for q in ['photo_02fb3c06e1fb70e5','photo_0913bcc35e9beb5a','photo_0a0e75a3b7797609','photo_fbc7b6d852b1125b']:
        section+=f'<details><summary>{q}：MaxSim 与 RoMa 对应</summary><img src="{prefix}{q}_correspondences.png" alt="固定正负参考的真实坐标匹配对照"></details>'
    section+='<!-- COORDINATES_FINAL_END -->\n'
    s=report.read_text(encoding='utf-8');s=re.sub(r'<!-- COORDINATES_FINAL_BEGIN -->.*?<!-- COORDINATES_FINAL_END -->\s*','',s,flags=re.S);anchor='<h3 id="composition">';assert anchor in s;s=s.replace(anchor,section+anchor,1)
    s=re.sub(r'const rows=.*?;const imageIndex=',lambda _: 'const rows='+js(rows)+';const imageIndex=',s,count=1,flags=re.S)
    s,n=re.subn(r'const imageIndex=.*?;const tokenEvidence=',lambda _:'const imageIndex='+js(viewer['imageIndex'])+';const tokenEvidence=',s,count=1,flags=re.S);assert n==1
    s=s.replace('showRomaReview(e,r);e.append','showRomaReview(e,r);showCoordinateReview(e,r);e.append');s=s.replace('||romaFilter(r,f))','||romaFilter(r,f)||coordinateFilter(r,f))')
    script='''
// COORDINATE_SCRIPT_BEGIN
function coordinateFilter(r,f){const d=r.coordinate_evidence;if(!d)return false;const a=d.factorial.prior_image_prior_ocr.rank,b=d.factorial.repaired_image_repaired_ocr.rank;return(f==='coordinates_repairs'&&a>1&&b===1)||(f==='coordinates_harms'&&a===1&&b>1)||(f==='coordinates_remaining54'&&!!d.diagnosis)}
function showCoordinateReview(e,r){const d=r.coordinate_evidence;if(!d)return;e.append(textNode('h3','新增：补齐坐标与最终联合对照'));addTable(e,['方法','全图库排名'],[['纯图像验证选择',d.image.rank],['原图像＋原OCR',d.factorial.prior_image_prior_ocr.rank],['修复图像＋原OCR',d.factorial.repaired_image_prior_ocr.rank],['原图像＋修复OCR',d.factorial.prior_image_repaired_ocr.rank],['修复图像＋修复OCR',d.factorial.repaired_image_repaired_ocr.rank]]);for(const [label,z] of [['本轮纯图像首位',d.image],['最终组合首位',d.factorial.repaired_image_repaired_ocr]]){e.append(textNode('p',label));const b=document.createElement('button');b.className='photo-button';b.type='button';setPicture(b,z.winner,116);b.setAttribute('aria-label',label);e.append(b)}if(d.diagnosis){const t=d.diagnosis;e.append(textNode('p','Query缓存来源：'+t.cache_source));for(const s of t.evidence)e.append(textNode('p','证据：'+s));for(const s of t.repair_directions)e.append(textNode('p','修复方向：'+s));addTable(e,['固定候选对','整体差额','主体内','主体外'],Object.entries(t.patch_contributions).map(([k,x])=>[k,fmt(x.total),fmt(x.inside),fmt(x.outside)]));const a=document.createElement('a');a.href='joint_evidence/roma_coordinates_20261015/token_deltas54/'+r.query+'.npz';a.textContent='逐Token差额与坐标';e.append(a)}}
// COORDINATE_SCRIPT_END
'''
    s=re.sub(r'\n// COORDINATE_SCRIPT_BEGIN.*?// COORDINATE_SCRIPT_END\n','',s,flags=re.S);s=s.replace('</script>',script+'</script>',1)
    options=''.join(f'<option value="{k}">{v}</option>' for k,v in [('coordinates_repairs','最终组合新增修复'),('coordinates_harms','最终组合新增错误'),('coordinates_remaining54','原54张Token证据')]);s=re.sub(r'<option value="coordinates_[^"]*">.*?</option>','',s);s=re.sub(r'(<select id="filter"[^>]*>.*?)(</select>)',lambda m:m.group(1)+options+m.group(2),s,count=1,flags=re.S)
    if '<a href="#coordinates-final">坐标与最终联合</a>' not in s:s=s.replace('<a href="#roma-reuse">RoMa缓存</a>','<a href="#roma-reuse">RoMa缓存</a><a href="#coordinates-final">坐标与最终联合</a>',1)
    s=s.replace('现有379张自动触发主体OCR重识别仍在运行，不计入以上结果；本页不是剩余错误全部结案声明。','以上缓存复用结果不包含后续OCR重识别；379张自动触发重识别及坐标补全现已完成，见下方“坐标与最终联合”。')
    # Add each tested configuration to the existing complete-results export.
    complete=read(E/'COMPLETE_RESULTS.json');complete['records']=[r for r in complete['records'] if not r['group'].startswith('coordinates_') and r['group']!='selective_ocr_rerun']
    curve=cr(A/'curve.csv');groups={}
    for r in curve:groups.setdefault((r['family'],r['feature'],r['beta'],r['gate']),{})[r['scope']]=r
    for key,z in groups.items():
        a,v,t,k=[z[s] for s in ['all','validation','test','retained']];family,feature,beta,gate=key
        complete['records'].append(dict(group='coordinates_grid',method='/'.join(key),label=f'坐标/{family}/{feature}/权重{beta}/门槛{gate}',all_correct=int(a['correct']),all_n=1117,all_top1=float(a['accuracy']),all_errors=1117-int(a['correct']),validation_correct=int(v['correct']),validation_n=173,test_correct=int(t['correct']),test_n=161,test_top1=float(t['accuracy']),comparison_baseline='纯图像944' if family=='image' else '数字串1051',repairs=int(a['repairs']),harms=int(a['harms']),retained_correct=int(k['correct']),retained_n=1104,note='同候选全量；未选中配置也保留；历史探索性'))
    for name,result in fact['results'].items():
        a,v,t,k=[result['metrics'][s] for s in ['all','validation','test','retained']];complete['records'].append(dict(group='coordinates_factorial',method=name,label=labels[name],all_correct=a['correct'],all_n=1117,all_top1=a['accuracy'],all_errors=1117-a['correct'],validation_correct=v['correct'],validation_n=173,test_correct=t['correct'],test_n=161,test_top1=t['accuracy'],comparison_baseline='数字串1051',repairs=a['repairs'],harms=a['harms'],retained_correct=k['correct'],retained_n=1104,note='图像和OCR参数各自固定后联合；非逐例真值选择'))
    for filename,group,columns,title in [('fusion_curve.csv','coordinates_previous_fusion',['method'],'统一融合'),('color_curve.csv','coordinates_previous_color',['family','weight'],'全图色度'),('routing_curve.csv','coordinates_previous_routing',['method'],'置信度切换'),('ablation_curve.csv','coordinates_previous_ablation',['method'],'支路移除'),('selective_ocr_curve.csv','selective_ocr_rerun',['method'],'自动OCR重识别')]:
        groups={}
        for r in cr(D/filename):groups.setdefault(tuple(r[k] for k in columns),{})[r['scope']]=r
        for key,z in groups.items():
            a,v,t,k=[z[s] for s in ['all','validation','test','retained']]
            complete['records'].append(dict(group=group,method='/'.join(key),label=title+'/'+'/'.join(key),all_correct=int(a['correct']),all_n=1117,all_top1=float(a['accuracy']),all_errors=1117-int(a['correct']),validation_correct=int(v['correct']),validation_n=173,test_correct=int(t['correct']),test_n=161,test_top1=float(t['accuracy']),comparison_baseline='数字串1051',repairs=int(a['repairs']),harms=int(a['harms']),retained_correct=int(k['correct']),retained_n=1104,note='完整历史探索性网格，包含负结果和未选中配置'))
    write(E/'COMPLETE_RESULTS.json',complete);fields=list(dict.fromkeys(k for r in complete['records'] for k in r))
    with (E/'COMPLETE_RESULTS.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(complete['records'])
    s=s.replace('完整导出共194条。',f'完整导出共{len(complete["records"])}条。').replace('不代表194个互不重复的新模型。','不代表互不重复的新模型。')
    # Validate links and syntax before committing the HTML replacement.
    refs=[];anchors=[]
    class Parser(HTMLParser):
        def handle_starttag(self,tag,attrs):
            d=dict(attrs)
            if 'id' in d:anchors.append(d['id'])
            refs.extend(d[k] for k in ['src','href'] if k in d)
    Parser().feed(s);assert len(anchors)==len(set(anchors))
    from urllib.parse import urlparse,unquote
    assert all((OUT/unquote(r.split('#')[0])).is_file() for r in refs if not r.startswith('#') and not urlparse(r).scheme)
    assert all(r[1:] in anchors for r in refs if r.startswith('#'))
    node=shutil.which('node')
    if node:
        source=A/'viewer_syntax_check.js';source.write_text('\n'.join(re.findall(r'<script>(.*?)</script>',s,re.S)));subprocess.run([node,'--check',str(source)],check=True)
    tmp=report.with_suffix('.tmp');tmp.write_text(s,encoding='utf-8');tmp.replace(report)
    facts=read(OUT/'REPORT_FACTS.json');facts['coordinate_completion_20261015']=dict(summary=summ,factorial=fact,ocr=ocr,verification=check,provenance=prov,backup=backup.name,original_locked_main_unchanged=1041);facts['complete_results_section']=complete;write(OUT/'REPORT_FACTS.json',facts)
    assert all((OUT/x['sheet']).is_file() for x in viewer['imageIndex'].values())
    validation=read(E/'REPORT_VALIDATION.json');validation.update(coordinate_verification=check,coordinate_source_provenance=prov['status'],complete_result_rows=len(complete['records']),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),coordinate_figure_visual_review='pending local review',coordinate_javascript_syntax='passed' if node else 'pending local node check');write(E/'REPORT_VALIDATION.json',validation)
    manifest={str(f.relative_to(OUT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in E.rglob('*') if f.is_file() and f.name!='MANIFEST.json'};write(E/'MANIFEST.json',manifest)
    write(A/'REPORT_UPDATE_RECEIPT.json',dict(status='complete',report=str(report),backup=str(backup),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),links_checked=len(refs),javascript_syntax='passed' if node else 'pending_local_node_check',figure_visual_review='pending local review',new_reference_thumbnails=len(missing),complete_result_rows=len(complete['records']),primary_unchanged=1041,final_correct=final['correct']))
    deliver=[report,backup,OUT/'REPORT_FACTS.json',E/'CASE_VIEWER_DATA.json',E/'COMPLETE_RESULTS.json',E/'COMPLETE_RESULTS.csv',E/'REPORT_VALIDATION.json',E/'MANIFEST.json',*[f for f in DEST.rglob('*') if f.is_file()]]
    archive=A/'REPORT_DELIVERY.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for f in deliver:tf.add(f,arcname=str(f.relative_to(OUT)).replace('\\','/'))
    write(A/'DELIVERY_RECEIPT.json',dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(deliver),bytes=archive.stat().st_size,report_sha256=hashlib.sha256(report.read_bytes()).hexdigest()))
    print('REPORT UPDATED',json.dumps(read(A/'REPORT_UPDATE_RECEIPT.json')),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args();run(args.wait)
