"""Seven conserved descriptive regions with explicit human ROI/unknown coverage.

Only region annotations change. Frozen signed deltas, candidate pairs and
MaxSim scores remain the original complete-photo scores. Manual ROIs are
annotation assistance, not an automatic retrieval method or encoder causality.
"""
import argparse,csv,gzip,hashlib,json,re,sys,time,shutil
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT,read,write,sha,csvout,load
from scripts.analyze_colnomic_tokens import display_xy

REGIONS=('key_numbers_specs','difference_layout','common_text','pattern_color','subject_other','background','template')
PRIORITY=('key_numbers_specs','difference_layout','pattern_color','common_text','subject_other')
OLD=ROOT/'results/colnomic_full_audit/20261013_resume/full_tokens'
OUT=ROOT/'results/colnomic_full_audit/20261013_resume/reviewed_regions'
BOXDIR=ROOT/'results/colnomic_full_audit/20261013_resume/manual_box_review'
FIELDS=['query','positive','negative','q_index','q_type','q_x','q_y','q_ocr_text','delta','pos_sim','neg_sim',
        'contribution_to_mean_gap','old_region','new_region','region_status','region_reason',
        'old_subject','new_subject','box_membership_changed','manual_roi_ids','semantic_overlap_labels']


def normalized_box(record):
    return np.asarray(record['bbox'],np.float32)/np.tile([record['width'],record['height']],2)


def subject(row,record):
    if row['q_type']!='visual':return False
    point=np.asarray([float(row['q_x']),float(row['q_y'])]);box=normalized_box(record)
    return bool(((point>=box[:2])&(point<box[2:])).all())


def patch_rectangle(raw_index,spec):
    layout=spec['layout'];i=int(raw_index)-layout['start'];assert 0<=i<layout['count']
    col,row=i%layout['width'],i//layout['width']
    corners=np.array([[col/layout['width'],row/layout['height']],[(col+1)/layout['width'],row/layout['height']],
                      [(col+1)/layout['width'],(row+1)/layout['height']],[col/layout['width'],(row+1)/layout['height']]])
    points=display_xy(corners,spec['exif_orientation'])*[spec['display_width'],spec['display_height']]
    return np.r_[points.min(0),points.max(0)]


def intersection_fraction(cell,roi):
    lo=np.maximum(cell[:2],roi[:2]);hi=np.minimum(cell[2:],roi[2:])
    return float(np.maximum(hi-lo,0).prod()/max(1e-12,np.prod(cell[2:]-cell[:2])))


def classify(row,oldbox,newbox,manual,spec):
    labels=json.loads(row.get('semantic_overlap_labels') or '[]')
    oldsubject,newsubject=subject(row,oldbox),subject(row,newbox)
    if row['q_type']=='template':return 'template','attended_nonvisual_token','no spatial coordinate',labels,[],False,False
    if not newsubject:return 'background','human_box_center_proxy','outside reviewed human subject box; boundary cells can straddle it',labels,[],oldsubject,False
    digit=bool(re.search(r'\d',row.get('q_ocr_text','')))
    if digit:labels.append('OCR_digits_present_identity_relevance_not_automatically_assumed')
    hits=[]
    if manual:
        cell=patch_rectangle(row['q_index'],spec)
        hits=[roi for roi in manual['rois'] if intersection_fraction(cell,np.asarray(roi['bbox_xyxy_display_pixels']))>=.05]
        for roi in hits:
            labels.extend(['manual_ROI:'+roi['roi_id'],'overlap_semantic:'+roi['primary_region']]+roi.get('semantic_labels',[]))
    if hits:
        chosen=min(hits,key=lambda roi:PRIORITY.index(roi['primary_region']))
        return chosen['primary_region'],'manual_actual_image_ROI','manual patch intersection; semantic overlap recorded separately; no encoder-causal claim',list(dict.fromkeys(labels)),[roi['roi_id'] for roi in hits],oldsubject,True
    if row['region']=='common_text':
        return 'common_text','automatic_shared_OCR_proxy','legacy exact shared-candidate OCR-word proxy, not human verified semantic identity evidence',list(dict.fromkeys(labels)),[],oldsubject,True
    status='unlocated_semantics_with_OCR_digits' if digit else 'unlocated_subject_semantics'
    reason='unreviewed digit-containing OCR row is not promoted to identity-key numbers' if digit else 'no justified manual semantic ROI; retained as subject_other'
    return 'subject_other',status,reason,list(dict.fromkeys(labels)),[],oldsubject,True


def summaries(rows,n):
    result=[]
    for region in REGIONS:
        selected=[row for row in rows if row['new_region']==region]
        values=np.array([float(row['delta']) for row in selected],np.float64)
        absolute=float(np.abs(values).sum())
        result.append(dict(region=region,tokens=len(values),positive_contribution=float(np.maximum(values,0).sum()/n),
            negative_contribution=float(np.minimum(values,0).sum()/n),net_contribution=float(values.sum()/n),
            within_region_mean=float(values.mean()) if len(values) else None,
            largest_absolute_share=float(np.abs(values).max()/absolute) if absolute else None,
            hhi_absolute=float(np.square(np.abs(values)/absolute).sum()) if absolute else None,
            annotation_status_counts=dict(Counter(row['region_status'] for row in selected))))
    assert sum(row['tokens'] for row in result)==n
    return result


def coverage_status(qid,region,manual,tokens):
    rois=[r for r in manual.get(qid,{}).get('rois',[]) if r['primary_region']==region]
    if rois:return 'manual_localized' if tokens else 'manual_ROI_localized_no_primary_patch'
    if region=='key_numbers_specs':
        if qid=='photo_21bd7df9ed6795e7':return 'count_identity_ROI_obscured_shared_dose_not_key'
        if qid=='photo_41de3eb383934ec3':return '20_vs50mL_capacity_face_not_visible_shared_dose_not_key'
        return 'identity_relevance_not_located_or_reviewed'
    if region in ['difference_layout','pattern_color']:return 'semantic_ROI_not_located'
    if region=='common_text':return 'automatic_shared_OCR_proxy' if tokens else 'no_shared_OCR_proxy_assigned'
    if region=='subject_other':return 'retained_unlocated_subject_semantics'
    if region=='background':return 'human_subject_box_center_proxy'
    return 'attended_nonvisual_tokens'


def build(out):
    csv.field_size_limit(sys.maxsize)
    before_path=BOXDIR/'subject_boxes_before_user_review.json';after_path=BOXDIR/'subject_boxes_after_user_review.json'
    before,after=read(before_path)['records'],read(after_path)['records']
    manual_doc=read(out/'manual_rois.json');manual={r['photo_id']:r for r in manual_doc['records']}
    for record in manual.values():
        for source in record['sources']:assert sha(source['path'])==source['sha256']
        assert sha(out/record['evidence'])==record['evidence_sha256']
        assert record['actual_view_image_completed']
    metadata_path=ROOT/'results/token_mechanism_analysis/F03/reproduction/token_metadata.json'
    qspec={r['id']:r for r in read(metadata_path)['photos']}
    image=load('cache/scores/colnomic7b/image.pt')
    qi={qid:i for i,qid in enumerate(image['query_ids'])};ri={rid:i for i,rid in enumerate(image['reference_ids'])}
    with gzip.open(OLD/'comparisons_all.csv.gz','rt') as stream:comparisons=list(csv.DictReader(stream))
    assert len(comparisons)==1648 and len({r['query'] for r in comparisons})==1117
    query_commits={qid:{r['path']:r for r in read(OLD/'queries'/qid/'complete.json')['files']} for qid in qi}
    paths=[before_path,after_path,out/'manual_rois.json',OLD/'comparisons_all.csv.gz',OLD/'validation.json',
           metadata_path,ROOT/'cache/scores/colnomic7b/image.pt']
    protocol=dict(schema='colnomic.reviewed_seven_regions.v1',source_sha256=sha(Path(__file__)),device='cpu',threads=2,
        queries=1117,pairs=1648,regions=list(REGIONS),subject_boxes='explicit before/after snapshots; current reviewed snapshot drives classification',
        delta='reuse every original frozen signed delta exactly; no feature/model scoring rerun',
        key_numbers='only explicitly viewed dose/capacity manual ROI; unverified OCR digits remain subject_other with separate overlap label',
        layout_and_pattern='only justified manual original-image ROI; all other cases explicitly unlocated',
        common_text='manual public/shared text ROIs or unchanged legacy exact shared-OCR proxy; not verified identity relevance',
        priority=list(PRIORITY),manual_ROI_mapping='displayed token patch intersection / patcharea>=.05; reviewed-box token-center rule first',
        overlap='nonexclusive semantic labels saved separately; primary seven-region partition is exclusive',
        manual_queries=list(manual),manual_annotation_assistance=True,observed_test_and_GT_assisted_annotation=True,
        encoder_causality_claimed=False,automatic_full_population_semantic_ROIs_claimed=False,
        inputs={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths})
    write(out/'protocol.json',protocol)
    (out/'source_executed.py').write_text(Path(__file__).read_text())
    pairrows,regionrows,coverage,transitions,source_manifest=[],[],[],[],[]
    firstquery=set();global_counts=Counter();manual_hit_counts=Counter();offset=0;maxerror=0.;started=time.monotonic()
    temp=out/'tokens_all.csv.gz.tmp'
    with gzip.open(temp,'wt',compresslevel=5,newline='') as target:
        writer=csv.DictWriter(target,fieldnames=FIELDS);writer.writeheader()
        for index,pair in enumerate(comparisons):
            qid=pair['query'];source=OLD/'queries'/qid/pair['token_file']
            relative=str(source.relative_to(OLD));digest=sha(source)
            assert digest==query_commits[qid][relative]['sha256']
            with gzip.open(source,'rt') as stream:oldrows=list(csv.DictReader(stream))
            n=len(oldrows);assert n==int(pair['valid_query_tokens'])
            assert len({r['q_index'] for r in oldrows})==n
            newrows=[];m=manual.get(qid)
            # ROIs were observed against these exact paired template IDs.
            if m and (pair['positive']!=m['positive_reference'] or pair['negative']!=m['negative_reference']):m=None
            for oldrow in oldrows:
                region,status,reason,labels,ids,oldsubject,newsubject=classify(oldrow,before[qid],after[qid],m,qspec[qid])
                row={key:oldrow.get(key,'') for key in FIELDS if key in oldrow}
                row.update(query=qid,positive=pair['positive'],negative=pair['negative'],old_region=oldrow['region'],new_region=region,
                    region_status=status,region_reason=reason,old_subject=oldsubject,new_subject=newsubject,
                    box_membership_changed=oldsubject!=newsubject,manual_roi_ids=ids,semantic_overlap_labels=labels,
                    contribution_to_mean_gap=float(oldrow['delta'])/n)
                newrows.append(row)
                writer.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,list) else v for k,v in row.items() if k in FIELDS})
                global_counts[region]+=1
                manual_hit_counts.update(ids)
            values=np.array([float(row['delta']) for row in newrows],np.float64)
            expected=float(image['scores'][qi[qid],ri[pair['positive']]]-image['scores'][qi[qid],ri[pair['negative']]])
            assert expected==float(pair['image_gap'])
            summary=summaries(newrows,n);conserved=sum(r['net_contribution'] for r in summary)
            error=abs(conserved-expected);maxerror=max(maxerror,error);assert error<=2e-7,(qid,error)
            assert [r['delta'] for r in newrows]==[r['delta'] for r in oldrows]
            for r in summary:regionrows.append(dict(query=qid,positive=pair['positive'],negative=pair['negative'],N=n,image_gap=expected,**r))
            pairrows.append(dict(query=qid,positive=pair['positive'],negative=pair['negative'],negative_roles=pair['negative_roles'],N=n,
                image_gap=expected,delta_mean=float(values.mean()),region_net_sum=conserved,conservation_error=error,
                copied_delta_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                aggregate_row_start=offset,aggregate_row_count=n,manual_ROI_used=m is not None,
                tokens_changing_subject_membership=sum(r['box_membership_changed'] for r in newrows),
                unverified_numeric_old_region_demoted=sum(o['region']=='key_numbers_specs' and r['new_region']!='key_numbers_specs' for o,r in zip(oldrows,newrows))))
            if qid not in firstquery:
                firstquery.add(qid)
                for r in summary:coverage.append(dict(query=qid,region=r['region'],unique_query_tokens=r['tokens'],
                    localization_status=coverage_status(qid,r['region'],manual,r['tokens']),
                    original_pixels_actually_viewed_for_semantic_ROI=qid in manual,manual_roi_ids=[roi['roi_id'] for roi in manual.get(qid,{}).get('rois',[]) if roi['primary_region']==r['region']]))
            counts=Counter((o['region'],r['new_region'],r['box_membership_changed'],r['region_status']) for o,r in zip(oldrows,newrows))
            transitions.extend(dict(query=qid,positive=pair['positive'],negative=pair['negative'],old_region=a,new_region=b,
                box_membership_changed=c,region_status=s,tokens=count) for (a,b,c,s),count in counts.items())
            source_manifest.append(dict(query=qid,positive=pair['positive'],negative=pair['negative'],path=str(source.relative_to(ROOT)),sha256=digest,bytes=source.stat().st_size))
            offset+=n
            if (index+1)%100==0 or index+1==len(comparisons):
                print('Refined regions',index+1,'/',len(comparisons),'tokens',offset,'maxerr',maxerror,'elapsed',time.monotonic()-started,flush=True)
                write(out/'progress.json',dict(pairs_completed=index+1,total_pairs=1648,token_pair_records=offset,max_conservation_error=maxerror))
    temp.replace(out/'tokens_all.csv.gz')
    csvout(out/'pair_summaries.csv',pairrows);csvout(out/'region_contributions.csv',regionrows)
    csvout(out/'query_region_coverage.csv',coverage);csvout(out/'old_new_region_transitions.csv',transitions)
    write(out/'source_token_files.json',source_manifest)
    write(out/'localization_coverage.json',dict(total_queries=1117,total_pairs=1648,token_pair_records=offset,
        manual_semantic_queries=4,manual_ROIs=sum(len(r['rois']) for r in manual.values()),manual_roi_hit_token_pair_counts=dict(manual_hit_counts),
        primary_token_pair_counts=dict(global_counts),
        status_counts={region:dict(Counter(row['localization_status'] for row in coverage if row['region']==region)) for region in REGIONS},
        unknowns_retained=True,full_population_manual_ROI_coverage_claimed=False))
    write(out/'box_revision_explanation.json',dict(before_sha256=sha(before_path),after_sha256=sha(after_path),
        changed_queries=[q for q in before if before[q]['bbox']!=after[q]['bbox']],
        changed_pair_records=[r for r in pairrows if r['tokens_changing_subject_membership']],
        interpretation='Only query subject/background assignment and semantic annotation changed. Delta and full-photo image margins are preserved. OCR numeric rows were separately demoted from identity-key claims; those semantic changes are not caused by box revisions.'))


def verify(out):
    csv.field_size_limit(sys.maxsize)
    pairlist=list(csv.DictReader((out/'pair_summaries.csv').open()))
    expected={(r['query'],r['positive'],r['negative']):r for r in pairlist}
    regionlist=list(csv.DictReader((out/'region_contributions.csv').open()))
    regions={(r['query'],r['positive'],r['negative'],r['region']):r for r in regionlist}
    observed=defaultdict(list)
    with gzip.open(out/'tokens_all.csv.gz','rt') as stream:
        for row in csv.DictReader(stream):
            assert row['new_region'] in REGIONS
            observed[row['query'],row['positive'],row['negative']].append(row)
    assert set(observed)==set(expected) and len(observed)==1648
    maxerror=0.;total=0
    for key,rows in observed.items():
        n=len(rows);assert n==int(expected[key]['N']) and len({r['q_index'] for r in rows})==n
        values=np.array([float(row['delta']) for row in rows],np.float64)
        assert hashlib.sha256(values.tobytes()).hexdigest()==expected[key]['copied_delta_sha256']
        replay=summaries(rows,n)
        for summary in replay:
            saved=regions[*key,summary['region']]
            assert int(saved['tokens'])==summary['tokens']
            for field in ['positive_contribution','negative_contribution','net_contribution']:
                assert float(saved[field])==summary[field]
        error=abs(sum(r['net_contribution'] for r in replay)-float(expected[key]['image_gap']))
        assert error<=2e-7;maxerror=max(maxerror,error);total+=n
    protocol=read(out/'protocol.json')
    assert all(sha(ROOT/path)==entry['sha256'] for path,entry in protocol['inputs'].items())
    sources=read(out/'source_token_files.json');assert all(sha(ROOT/r['path'])==r['sha256'] for r in sources)
    coverage=list(csv.DictReader((out/'query_region_coverage.csv').open()))
    assert len(coverage)==1117*7
    report=dict(status='passed',queries=1117,pairs=1648,token_pair_records=total,seven_region_rows_replayed=len(regions),
        exactly_one_primary_region_per_valid_token=True,all_signed_deltas_preserved_by_pair_digest=True,
        sum_positive_negative_net_over_N_matches_original_image_gap=True,max_image_gap_conservation_error=maxerror,
        conservation_limit=2e-7,manual_ROI_queries=4,manual_ROI_annotations=13,
        explicit_query_region_status_rows=len(coverage),
        unlocated_key_layout_pattern_query_region_rows=sum(r['region'] in ['key_numbers_specs','difference_layout','pattern_color'] and not r['localization_status'].startswith('manual') for r in coverage),
        coverage_status_clarification='Every query has7 status records; manual_localized means at leastone justified manual ROI, not exhaustive semantic coverage of that category.',
        source_token_files_SHA_rechecked=len(sources),
        old_full_tokens_modified=False,automatic_identity_key_numbers_from_all_OCR_digits=False,
        thresholded_color_identity_patterns_claimed=False,semantic_overlap_labels_separate_from_primary_partition=True,
        manual_ROI_is_annotation_assistance=True,encoder_causality_claimed=False,
        limitations=['Only four requested focal queries have manually localized semantic ROIs.',
            'Other cases retain automatic shared-OCR proxies or explicitly unlocated subject semantics.',
            'Manual patch ROIs can straddle neighbouring words, lines and colors; a token is not a word.',
            'Observed candidate layout, dose and visual design differences remain confounded; this is descriptive signed-score decomposition.'])
    write(out/'validation.json',report)
    write(out/'artifacts_sha256.json',{str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['artifacts_sha256.json','run.log','progress.json']})
    return report


def write_per_query_outputs(out):
    """Small per-photo delivery links and descriptive signed-region frequencies."""
    pairs=list(csv.DictReader((out/'pair_summaries.csv').open()))
    regional=list(csv.DictReader((out/'region_contributions.csv').open()))
    coverage=list(csv.DictReader((out/'query_region_coverage.csv').open()))
    oldboxes=read(BOXDIR/'subject_boxes_before_user_review.json')['records']
    newboxes=read(BOXDIR/'subject_boxes_after_user_review.json')['records']
    manual={r['photo_id']:r for r in read(out/'manual_rois.json')['records']}
    bypair=defaultdict(list);byquery=defaultdict(list);qc=defaultdict(list);frequencies=Counter();seen=defaultdict(set)
    for r in regional:
        item={**r}
        for field in ['N','tokens']:item[field]=int(r[field])
        for field in ['image_gap','positive_contribution','negative_contribution','net_contribution']:
            item[field]=float(r[field])
        for field in ['within_region_mean','largest_absolute_share','hhi_absolute']:
            item[field]=float(r[field]) if r[field] else None
        item['annotation_status_counts']=json.loads(r['annotation_status_counts'])
        bypair[r['query'],r['positive'],r['negative']].append(item)
        net,gap=item['net_contribution'],item['image_gap']
        pattern=('no_primary_tokens' if not item['tokens'] else 'exact_zero_net' if net==0 else
                 'region_supports_GT_but_total_prefers_wrong' if net>0 and gap<0 else
                 'region_and_total_support_GT' if net>0 and gap>0 else
                 'region_prefers_wrong_but_total_supports_GT' if net<0 and gap>0 else
                 'region_and_total_prefer_wrong' if net<0 and gap<0 else 'total_exact_tie')
        status='contains_manual_actual_image_ROI' if item['annotation_status_counts'].get('manual_actual_image_ROI',0) else 'proxy_or_unlocated'
        key=r['region'],status,pattern;frequencies[key]+=1;seen[key].add(r['query'])
    for row in coverage:
        row={**row,'unique_query_tokens':int(row['unique_query_tokens']),
             'original_pixels_actually_viewed_for_semantic_ROI':row['original_pixels_actually_viewed_for_semantic_ROI']=='True',
             'manual_roi_ids':json.loads(row['manual_roi_ids'])}
        qc[row['query']].append(row)
    for pair in pairs:
        qid=pair['query'];item={**pair}
        for field in ['N','aggregate_row_start','aggregate_row_count','tokens_changing_subject_membership','unverified_numeric_old_region_demoted']:
            item[field]=int(pair[field])
        for field in ['image_gap','delta_mean','region_net_sum','conservation_error']:item[field]=float(pair[field])
        item['manual_ROI_used']=pair['manual_ROI_used']=='True'
        item['negative_roles']=json.loads(pair['negative_roles'])
        item['regions']=bypair[qid,pair['positive'],pair['negative']]
        byquery[qid].append(item)
    for qid,contrasts in byquery.items():
        record=manual.get(qid)
        write(out/'queries'/qid/'summary.json',dict(schema='colnomic.reviewed_regions.query.v1',photo_id=qid,
            contrasts=contrasts,localization_coverage=qc[qid],manual_roi_record=record,
            manual_roi_evidence='../../'+record['evidence'] if record else None,
            manual_ROI_query=record is not None,box_geometry_changed=oldboxes[qid]['bbox']!=newboxes[qid]['bbox'],
            before_box=oldboxes[qid],reviewed_box=newboxes[qid],
            token_csv='../../tokens_all.csv.gz',source_delta_preserved=True,
            full_photo_B03_F03_baselines_unchanged=True,
            interpretation='Signed frozen-score decomposition. Region overlap labels are separate from one exclusive primary class. Unlocated semantics remain unknown; manual_ROI annotation is assistance, not encoder-causal evidence.'))
    table=[dict(region=region,annotation_scope=status,contribution_pattern=pattern,pairs=count,
                distinct_queries=len(seen[region,status,pattern])) for (region,status,pattern),count in sorted(frequencies.items())]
    csvout(out/'region_contribution_pattern_frequencies.csv',table)
    report=dict(status='passed',query_summary_files=len(byquery),contrast_records=sum(len(v) for v in byquery.values()),
        source_sha256=sha(Path(__file__)),frequency_table='region_contribution_pattern_frequencies.csv',
        sign_rule='exact float64 sign of preserved signed delta/N; empty and exactzero classes separate',
        frequency_scope='1648 selected original positive/wrong contrasts; repeated queries may appear in several pairs. Does not estimate automatic identity-region accuracy or causal mechanism frequency.',
        manual_ROI_scope='Four focal queries/13patch ROIs; contains_manual_actual_image_ROI means at leastone justified ROI, not complete localization of that category.',
        maximum_conservation_error=max(float(r['conservation_error']) for r in pairs))
    assert len(byquery)==1117 and report['contrast_records']==1648
    write(out/'query_summary_generation.json',report)
    (out/'source_after_delivery_refinement.py').write_text(Path(__file__).read_text())
    print('Per-query region summaries',report['query_summary_files'],report['contrast_records'],flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--verify-only',action='store_true');parser.add_argument('--summaries-only',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    out=args.output.resolve();assert out.is_relative_to(OUT);out.mkdir(parents=True,exist_ok=True)
    if args.summaries_only:write_per_query_outputs(out);return
    if not args.verify_only:build(out)
    write_per_query_outputs(out)
    print(json.dumps(verify(out),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
