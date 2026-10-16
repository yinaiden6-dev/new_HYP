"""Build only after all planned image experiments and validations complete."""
import sys,csv,json,html,shutil,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent;P=O.parent;D=P/'release'/'pure_image_mechanisms';sys.path[:0]=[str(P),str(P/'plot_dependencies')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_utils import read,write,csvwrite
def cr(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def main():
 for name in ['CAUSAL_VALIDATION_309.json','CAUSAL_VALIDATION_62.json','REFERENCE_CONTRAST_VALIDATION.json','PURE_MATCHING_VALIDATION.json']:assert read(O/name)['status']=='passed'
 D.mkdir(exist_ok=True);ledger=cr(O/'all1117_image_evidence.csv');facts=read(O/'IMAGE_EVIDENCE_SUMMARY.json');c309=read(O/'causal_summary_309.json');c62=read(O/'causal_summary_62.json');metrics309=cr(O/'causal_metrics_309.csv');metrics62=cr(O/'causal_metrics_62.csv');matching=read(O/'pure_matching_summary.json');contrast=read(O/'reference_contrast_summary.json');cs=read(O/'REFERENCE_CONTRAST_STATISTICS.json');bridge=read(O/'image_ocr_bridge_summary.json');tie=read(O/'ocr_only_tie_summary.json');pair=read(O/'context_pair_summary_309.json');index=read(P/'release/case_image_index.json');cohort={r['query']:r for r in read(P/'cohort.json')}
 def metric(rr,role,arm,scope):return next(r for r in rr if r['role']==role and r['arm']==arm and r['scope']==scope)
 failure_role=next(k for k,v in c309['groups'].items() if v['n']==247);cf=c309['groups'][failure_role];ch=c62['groups']['harm']
 def save(name,fig):
  fig.savefig(D/(name+'.png'),bbox_inches='tight',dpi=170);fig.savefig(D/(name+'.svg'),bbox_inches='tight');plt.close(fig)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 methods=[('Original',870,0,0),('Subject readout',906,67,31),('Prior crop',881,62,51)]
 for k in ['visual_forward','visual_balanced']:a=matching[k]['all'];methods.append((k.replace('_',' '),a['correct'],a['repairs'],a['harms']))
 for k in ['global_std','hard20_std']:a=contrast['selected'][k]['all'];methods.append((k.replace('_',' '),a['correct'],a['repairs'],a['harms']))
 fig,ax=plt.subplots(figsize=(10,4.8));y=np.arange(len(methods));ax.barh(y,[m[2] for m in methods],color='#267b76',label='Repairs');ax.barh(y,[-m[3] for m in methods],color='#b65357',label='New errors');ax.axvline(0,color='#888',lw=.8);ax.set_yticks(y,[f'{m[0]}  ({m[1]}/1117)' for m in methods]);ax.invert_yaxis();ax.set_xlabel('Counts relative to original image retrieval');ax.legend();ax.set_title('Image-only full-gallery results / supervision differs by method');save('01_image_repairs_harms',fig)
 fig,axs=plt.subplots(1,2,figsize=(11,4.3));arms=['far_background_gray','far_background_blur'];scopes=['all','subject_template']
 for ax,scope in zip(axs,scopes):
  rows=[metric(metrics309,failure_role,a,scope) for a in ['original']+arms];ax.bar(['Original','Far gray','Far blur'],[int(r['correct']) for r in rows],color=['#aeb8c1','#527fa6','#267b76']);ax.set_title(scope.replace('_',' ')+' / 247 original image failures');ax.set_ylabel('Correct in full gallery / 247');ax.set_ylim(0,max(int(r['correct']) for r in rows)+12)
  for i,r in enumerate(rows):ax.text(i,int(r['correct'])+.8,r['correct'],ha='center')
 save('02_context_original_failures',fig)
 fig,ax=plt.subplots(figsize=(7,4));matrix=np.array([[61,6],[109,71]]);im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=115);ax.set_xticks([0,1],['Corrected fusion fixes','Corrected fusion still wrong']);ax.set_yticks([0,1],['Subject image fixes','Subject image still wrong']);ax.set_title('Same 247 original image failures / existing OCR reused')
 for (y,x),v in np.ndenumerate(matrix):ax.text(x,y,str(v),ha='center',va='center',fontsize=21,color='white' if v>65 else '#152235')
 save('03_image_ocr_bridge',fig)
 curves=cr(O/'pure_matching_curves.csv')+cr(O/'reference_contrast_curves.csv');fig,axs=plt.subplots(1,2,figsize=(11,4.5))
 for ax,scope in zip(axs,['val','test']):
  for method in ['reverse','balanced','visual_forward','visual_balanced','global_std','hard20_std']:
   rs=sorted([r for r in curves if r['scope']==scope and r['method']==method],key=lambda r:float(r['weight']));ax.plot([float(r['weight']) for r in rs],[100*float(r['macro_accuracy']) for r in rs],marker='.',label=method.replace('_',' '))
  ax.set_title(scope+' identities / macro Top-1');ax.set_xlabel('Mixture weight');ax.set_ylabel('Accuracy (%)');ax.grid(alpha=.15)
 axs[1].legend(fontsize=8);save('04_image_matching_curves',fig)
 cp=cr(O/'context_pair_mechanisms_62.csv');strong=[r for r in cp if r['both_fill_subject_repair']=='True'];fig,ax=plt.subplots(figsize=(10,4.2));xx=np.arange(len(strong));ax.bar(xx-.24,[float(r['original_margin']) for r in strong],.24,label='Original');ax.bar(xx,[float(r['locked_margin']) for r in strong],.24,label='Edited / old matches');ax.bar(xx+.24,[float(r['free_margin']) for r in strong],.24,label='Edited / rematched');ax.axhline(0,color='#333',lw=.7);ax.set_xticks(xx,[r['package_id'].replace('package_','')+' '+('gray' if r['arm'].endswith('gray') else 'blur') for r in strong],rotation=30);ax.set_ylabel('Correct minus fixed wrong reference');ax.set_title('Four dual-fill rescues: representations and correspondence interact');ax.legend(fontsize=8);save('05_locked_correspondence',fig)
 fig,ax=plt.subplots(figsize=(10,4.2));aa=['far_background_gray','far_background_blur','all_background_gray','boundary_ring_gray','matched_background_gray','matched_subject_gray'];x=np.arange(len(aa));a=[int(metric(metrics62,'harm',arm,'subject_template')['correct']) for arm in aa];b=[int(metric(metrics62,'success_control',arm,'subject_template')['harms']) for arm in aa];ax.bar(x-.18,a,.36,label='Repairs among31 readout harms',color='#267b76');ax.bar(x+.18,b,.36,label='Harms among31 success controls',color='#b65357');ax.set_xticks(x,[v.replace('_','\n') for v in aa]);ax.set_ylabel('Queries');ax.legend(fontsize=8);ax.set_title('Pixel controls / subject readout / selected62, not deployment accuracy');save('06_pixel_controls',fig)
 # Only public aggregate tables, protocols, code and existing approved case ledger.
 copies=['IMAGE_EVIDENCE_SUMMARY.json','all1117_image_evidence.csv','causal_metrics_309.csv','causal_metrics_62.csv','causal_summary_309.json','causal_summary_62.json','pure_matching_curves.csv','pure_matching_summary.json','reference_contrast_curves.csv','reference_contrast_summary.json','image_ocr_bridge_summary.json','ocr_only_tie_summary.json','line_token_bridge_summary.json','prior_crop_link_summary.json','context_pair_summary_309.json','context_pair_summary_62.json','protocol.json','expanded_pixel_protocol.json','bridge_protocol.json','pure_matching_protocol.json','reference_contrast_protocol.json','CAUSAL_VALIDATION_309.json','CAUSAL_VALIDATION_62.json','REFERENCE_CONTRAST_VALIDATION.json','PURE_MATCHING_VALIDATION.json','run_causal_encoding.py','score_causal.py','run_image_matching.py','run_reference_contrast.py','verify_causal.py','verify_reference_contrast.py','aggregate_evidence.py','summarize_causal.py','analyze_context_pairs.py','build_image_report.py']
 copies+=['REFERENCE_CONTRAST_STATISTICS.json','contrast_statistics.py','REGION_SCORE_SUMMARY.json','decompose_regions.py']
 fig,ax=plt.subplots(figsize=(9,3.8));vals=[r['correct'] for r in contrast['nulls']];ax.scatter(np.arange(10),vals,s=55,color='#667e92',label='Within-group shuffled weights');ax.axhline(870,color='#666',ls='--',label='Original 870');ax.axhline(888,color='#267b76',lw=2,label='Candidate-derived weights 888');ax.set_ylim(859,893);ax.set_xticks(np.arange(10),np.arange(1,11));ax.set_xlabel('Deterministic shuffle');ax.set_ylabel('Correct /1117');ax.set_title('Same weight distribution and visual/template mass; token assignment matters');ax.legend(fontsize=8);save('07_contrast_nulls',fig)
 for name in copies:shutil.copy2(O/name,D/name)
 def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
 main_table=table(['纯图像方法','正确 / 1117','修复','新增错误'],methods)
 pixrows=[]
 for arm in ['original']+aa:
  h=metric(metrics62,'harm',arm,'subject_template');s=metric(metrics62,'success_control',arm,'subject_template');pixrows.append([arm,h['correct']+'/31',s['correct']+'/31'])
 control_table=table(['输入干预','31 个主体筛选错误：正确数','31 个成功对照：正确数'],pixrows)
 directions=[['主体证据被其他 Token 抵消','冻结特征，主体计分修复67，但新增31错','软主体权重＋全图回退；自动定位验证','已证实可修复的读出瓶颈，非所有样本根因'],['上下文改变主体表征','309张主体输入图块不变；背景编辑改变特征与部分排名','训练背景稳定性，按验证集确定门控；不能全量置灰','像素因果响应；背景本身含身份信息未被证明'],['局部配对缺乏面/行/成分约束','既有Mucus案例：跨面6/7，跨成分60→600；新锁定匹配对照','视觉对应面／成分区域绑定，多对多保留合理视角变化','局部案例已验证；通用反向／均衡匹配未普遍有效'],['关键区域与全图库身份不唯一','Bravecto布局读出第4→2，新第一名也有同布局','联合容量、剂量、颜色等多线索；每次查完整图库','旧错误被击败不代表身份已经判对'],['输入、尺度与朝向','旧裁剪实验修复62、新增51；与主体修复重叠38','复用原裁剪结果；针对新样本单因素控制朝向与尺度','裁剪是多因素，不称纯分辨率问题'],['图像与文字的关系丢失','20mL三照片图像可救回；OCR容量词与20lb等共享','视觉局部证据＋OCR数字/单位/字段绑定；冲突门控','OCR识别正确也可能因计分归属错而无效']]
 directions.insert(0,['竞争候选下的Token权重','前20候选差异加权870→888，26修复/8新错；打乱对照864–871','冻结编码器；优先扩展无OCR、无人工框的竞争候选加权','全量有增益；历史测试+2，泛化区间仍含0'])
 facts.update(causal309=c309,causal62=c62,full_image_methods=methods,report_scope='Experiments completed; unsupported semantic roots remain explicitly separated from verified observations.');write(D/'REPORT_FACTS.json',facts)
 prose=f'''# ColNomic 纯图像机制与修复实验报告

实验日期：2026-10-14。原评测共1,117张照片、130个包装身份，图库5,405行、5,330个SHA组。纯图像原始Top-1为870/1117（77.89%），247张失败。本报告先分析图像编码、匹配和汇总，再连接已完成的OCR联合实验。没有重新执行26例OCR诊断。

## 已经成立的结论

**新增的无OCR、无人工框修复：候选差异加权。** 冻结原始embedding，以原始前20个不同SHA候选中每个Query Token的MaxSim标准差作权重，归一化后与原分数按验证集选出的0.5混合，再重排完整图库。正确数870→888（77.89%→79.50%），修复26张、涉及22个身份，新增8错。修复中8张不在主体计分修复集合内。权重在视觉/模板组内打乱、保持权重分布及组总权重的10组对照仅864–871正确。按全部图库估计差异反而867，说明竞争候选的选择很关键。

历史测试集只有18个身份，125/161→127/161，4修复、2新增错；按身份重采样的Top-1差值95%区间为约−1.23至+3.60个百分点，包含0。全量改善支持继续研究这一方向，但尚不能称跨身份泛化已确立。Token标准差是区分性启发式，不是已经证明每个高权重Token都表示身份细节。

**OCR不是唯一修复方向，纯图像也不是单一根因。** 保持原embedding不变，主体Token计分修复67张、新增31错；完整图库正确数为906。旧裁剪实验修复62、新增51，其中24张不在主体计分的67张内。这是两种不同干预的互补响应，不能把修复数直接相加。

**主体Token并不是只表示主体像素。** 本轮309张原图重编码均与历史query缓存逐位一致；无损编辑对照也逐位一致。远背景编辑保留主体实际processor图块，但改变视觉编码阶段和最终主体特征。原247张失败中，两种背景编辑均修复全图检索的有{cf['full_repaired_both_fills']}张，均修复主体读出的有{cf['subject_repaired_both_fills']}张；二者可能重叠。此前31张主体筛选新增错误中，4张在两种远背景编辑下均恢复全图库正确。背景位置的Token可能含上下文形成的主体信号，不能等同于无用背景语义。

**配对阶段仍然重要。** 对上述4张、两种填充，共8个固定候选对照，特征改变后若锁定原匹配，部分仍为负分差，重新匹配才转正。这个分解说明表征和对应会交互，不是独立、顺序无关的因果贡献。通用反向MaxSim、压平多对一重复计分没有形成稳定整体改善；视觉Token计分混合仅870→872。不能因此排除对应面、成分行约束，但不应直接推广双向匹配。

**编辑能修复，并不自动指向背景特有根因。** 31张主体筛选新增错误中，等面积主体编辑救回12张，同时损害31个成功对照中的4张；等面积背景编辑救回3张、损害0张。远背景编辑证明了上下文因果依赖，但一般扰动、特征变化和重匹配也会产生救回。直接全部去背景仍不能作为通用方案。

30张双背景编辑全图救回中，26张原本就能用主体计分救回，只有4张在该集合外。在固定原正负候选上，仅替换主体外区域的分数变化，置灰时已足以翻正29/30、模糊时27/30；仅替换主体与模板变化分别为9/30、11/30。这些集合可重叠，且只是分数分解，不是新的真实图像或完整图库干预。30张大多支持主体外计分干扰，不能全部计成新的编码缺陷。

## 纯图像完整图库结果

参数只按历史验证身份的宏平均Top-1、MRR、较小权重依次选取，测试身份不参与选择。历史测试集已多轮分析，结果不等于新独立留出集。主体框为人工审核，裁剪为既有多因素实验；这些成绩不是自动部署系统成绩。

{chr(10).join('| '+str(m[0])+' | '+str(m[1])+'/1117 | '+str(m[2])+' | '+str(m[3])+' |' for m in methods)}

## 图像到OCR的证据链

在同一247张原图失败上：主体图像与既有OCR修正融合都救回61张；图像独有6张；OCR融合独有109张；两者均未救回71张。已按相同正负候选分解图像与文字分差，3,351个融合排名重放一致。

6张图像独有修复来自4个身份，不能称6个独立身份。其中3张20mL照片，OCR已认对20mL；但50mL参考还包含20lb、20–25°C及其他位置的mL，存在词项而缺少字段绑定。仅加OCR或数字拆分没有救回。另一个案例的文字负优势会把图像正证据压过。

原247张失败中，修正OCR单独Top-1有193张，但24张只是平分后的顺序胜出，严格超过所有负参考的只有169张；其中18张又被融合拉错。因此不能把193都称为OCR已区分身份。7,402条既有OCR行与图像区域建立关联；156张存在文字正贡献而重叠图像Token负贡献。区域重叠不证明该Token只编码那一个词，也不直接证明编码器丢失文字。

## 修复覆盖与边界

本轮所列纯图像修复族，按逐例取任一成功的诊断并集覆盖{facts['repair_coverage']['any_image_intervention']}/247；仍有{facts['not_repaired_by_this_family']}张没有被这些干预修好。该并集用到了逐例真值选择，只说明响应覆盖，不是一个可部署方法的准确率，也不是全部机制已查明。

已证实：部分身份判别信号可从冻结特征读出；背景可通过编码上下文改变主体表示；配对重选可参与排名变化；局部线索不一定唯一确定全图库身份。仍不能对每一张未救回照片断言信息丢失或匹配错误。每图表把已验证证据与后续区分实验分开。

## 可复核性

309张×4核心输入，62张×8输入，均完整图库复算；全1117张另有匹配和参考差异加权实验。独立CPU双精度分数复核、原缓存逐位校验、实际processor输入重建、无损对照均通过。主体框近似而非像素分割；247失败加31读出损害加31成功组成的309诊断队列，不代表完整1117部署准确率。旧方向/数字/布局单例和旧裁剪结果均明确标为复用。完整embedding、矩阵和逐Token记录保留在本地/服务器，本公开版仅提供报告、聚合统计、代码和已授权的图片索引。
'''
 # Markdown is the established repository report format.
 mdtable='| 方法 | 正确数 | 修复 | 新错 |\n|---|---:|---:|---:|\n'+'\n'.join('| '+' | '.join(map(str,m))+' |' for m in methods)
 prose=prose.replace(chr(10).join('| '+str(m[0])+' | '+str(m[1])+'/1117 | '+str(m[2])+' | '+str(m[3])+' |' for m in methods),mdtable)
 prose+='\n## 修复方向\n\n| 环节 | 证据 | 方向 | 结论强度 |\n|---|---|---|---|\n'+'\n'.join('| '+' | '.join(r)+' |' for r in directions)+'\n\n[逐图浏览与曲线](REPORT.html) · [每图证据](all1117_image_evidence.csv) · [前期完整报告](../README.md)\n'
 (D/'README.md').write_text(prose,encoding='utf-8')
 import markdown
 body=markdown.markdown(prose,extensions=['tables']);body+= '<h2>统计与曲线</h2>'+''.join(f'<figure><img src="{n}.png" alt="{n}"></figure>' for n in ['01_image_repairs_harms','02_context_original_failures','03_image_ocr_bridge','04_image_matching_curves','05_locked_correspondence','06_pixel_controls','07_contrast_nulls'])+'<h2>62张像素对照</h2>'+control_table
 body+='''<h2>逐图证据：照片与正确参考</h2><p>点击图片展开证据。筛选不会把诊断修复并集当成部署结果；未进行像素实验的照片显示“未选入”。</p><label>筛选 <select id="filter"><option value="all">全部1117张</option><option value="fail">原图247张失败</option><option value="harm">主体筛选31张新增错误</option><option value="contrast">候选加权26张修复</option><option value="contrast_harm">候选加权8张新增错误</option><option value="context">双背景编辑救回</option><option value="only">图像救回、OCR融合仍错</option><option value="remaining">本轮图像修复族未救回</option></select></label><span id="count"></span><div id="cases"></div><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button><dialog id="detail"><button id="close">关闭</button><div id="detailbody"></div></dialog>'''
 data=json.dumps(ledger,ensure_ascii=False).replace('</','<\\/');idx=json.dumps(index,ensure_ascii=False).replace('</','<\\/')
 script='''const rows=ROWS,idx=INDEX;let page=0;const yes=x=>x==='True';function image(id,size=128){let a=idx.items[id];if(!a)return '<span>图片缺失</span>';let scale=size/idx.tile;return `<span class="thumb" role="img" aria-label="药盒图片" style="width:${size}px;height:${size}px;background-image:url('../${a.sheet}');background-size:${idx.columns*size}px auto;background-position:-${a.col*size}px -${a.row*size}px"></span>`}function selected(){let f=document.getElementById('filter').value;return rows.filter(r=>f==='all'||(f==='fail'&&yes(r.image_failed))||(f==='contrast'&&yes(r.image_failed)&&Number(r.contrast_hard20_std_rank)===1)||(f==='contrast_harm'&&!yes(r.image_failed)&&Number(r.contrast_hard20_std_rank)>1)||(f==='harm'&&yes(r.subject_harmed))||(f==='context'&&(yes(r.context_both_full_repair)||yes(r.context_both_subject_repair)))||(f==='only'&&yes(r.subject_repaired)&&Number(r.corrected_fusion_rank)>1)||(f==='remaining'&&yes(r.image_failed)&&!yes(r.any_tested_image_repair)))}function show(r){document.getElementById('detailbody').innerHTML=`<div class="pair">${image(r.query,240)}${image(r.positive_reference,240)}</div><p>左：照片；右：正确参考。历史原图排名 ${r.image_rank}；候选差异加权 ${r.contrast_hard20_std_rank}；主体 ${r.subject_rank}；旧裁剪 ${r.crop_rank}；修正OCR融合 ${r.corrected_fusion_rank}。</p><p><b>已验证证据：</b>${r.evidence}</p><p><b>修复方向：</b>${r.repair_direction}</p><p>远背景置灰：全图 ${r.far_background_gray_all_rank||'未选入'}／主体 ${r.far_background_gray_subject_template_rank||'未选入'}；模糊：全图 ${r.far_background_blur_all_rank||'未选入'}／主体 ${r.far_background_blur_subject_template_rank||'未选入'}。</p><p>${r.causal_limit}</p>`;document.getElementById('detail').showModal()}function render(){let a=selected(),pages=Math.max(1,Math.ceil(a.length/40));page=Math.max(0,Math.min(page,pages-1));document.getElementById('count').textContent=` ${a.length} 张`;document.getElementById('page').textContent=` ${page+1}/${pages} `;let host=document.getElementById('cases');host.innerHTML='';for(let r of a.slice(page*40,(page+1)*40)){let b=document.createElement('button');b.className='case';b.innerHTML=`<span class="pair">${image(r.query)}${image(r.positive_reference)}</span><span>图像 ${r.image_rank} → 主体 ${r.subject_rank}　融合 ${r.corrected_fusion_rank}</span><small>${r.evidence}</small>`;b.onclick=()=>show(r);host.appendChild(b)}}document.getElementById('filter').onchange=()=>{page=0;render()};document.getElementById('prev').onclick=()=>{page--;render()};document.getElementById('next').onclick=()=>{page++;render()};document.getElementById('close').onclick=()=>document.getElementById('detail').close();render();'''.replace('ROWS',data).replace('INDEX',idx)
 css='''body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#1a2a3a;background:#f5f7f9;margin:0}main{max-width:1120px;margin:auto;padding:36px}h1{line-height:1.25}h2{margin-top:38px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{border:1px solid #dce3e8;padding:9px;text-align:left}th{background:#e5edf2}figure{margin:22px 0;background:white;padding:10px}figure img{max-width:100%;height:auto}a{color:#17617d}.thumb{display:inline-block;background-repeat:no-repeat;background-color:#fff;flex-shrink:0}.pair{display:flex;justify-content:center;gap:8px}#cases{display:grid;grid-template-columns:repeat(auto-fill,minmax(282px,1fr));gap:12px;margin:20px 0}.case{background:white;border:1px solid #cad6df;border-radius:8px;padding:10px;text-align:left;cursor:pointer}.case>span,.case small{display:block}.case .pair{display:flex}.case small{line-height:1.45;color:#52677a;margin-top:7px}select,button{font:inherit}dialog{max-width:900px;width:85%;border:1px solid #a6b8c6;border-radius:10px}dialog::backdrop{background:#142435aa}#close{float:right}#detailbody{clear:both;padding:12px}code{font-size:.9em}@media(max-width:620px){main{padding:15px}.pair{flex-wrap:wrap}table{font-size:12px}}'''
 (D/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ColNomic 纯图像机制实验</title><style>'+css+'</style><main>'+body+'</main><script>'+script+'</script></html>',encoding='utf-8')
 # Static validation: every photo and positive reference resolves to an existing authorized sprite.
 for r in ledger:
  for key in ['query','positive_reference']:assert r[key] in index['items'] and (P/'release'/index['items'][r[key]]['sheet']).is_file()
 assert len(ledger)==1117 and sum(r['image_failed']=='True' for r in ledger)==247
 write(D/'REPORT_VALIDATION.json',dict(status='passed',all1117_photo_and_reference_images_resolve=True,pixel_experiments_complete=True,independent_checks_passed=True,browser_interaction_validation='Not performed; prior local-file browser security restriction retained.',files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.iterdir() if p.is_file() and p.name!='REPORT_VALIDATION.json'}));print('REPORT READY',D)
if __name__=='__main__':main()
