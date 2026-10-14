"""Equal-magnitude left/right word relocation; all eligible source identities.

This tests signed word position, not displacement magnitude or natural identity.
Every selection and pixel artifact is committed before any feature response.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import math
import numpy as np
import torch
import cv2
from PIL import Image, ImageOps

from scripts.colnomic_mechanism.common import ROOT, data, read, write, csvout, sha
from scripts.colnomic_mechanism.extended_layouts import prepare, to_raw, raw_bbox
from scripts.encode_colnomic_images import token_hash
from scripts.colnomic_mechanism.display_pixels import display_image


def variants(array, box, subject, other_lines):
    x1,y1,x2,y2=box
    step=max(2,int(round((x2-x1)*.35)))
    left=x1-step
    right=x2+step
    if left<max(0,subject[0]) or right>min(array.shape[1],subject[2]):
        raise ValueError('symmetric shift outside subject')
    paper=np.r_[array[y1:y2,left:x1].reshape(-1,3),array[y1:y2,x2:right].reshape(-1,3)]
    background=np.rint(np.median(paper,axis=0)).astype(np.uint8)
    if np.any(np.quantile(paper,.99,axis=0)-np.quantile(paper,.01,axis=0)>24) or np.any(np.abs(paper.astype(float)-background).max(1)>35):
        raise ValueError('left/right destinations not uniform unmarked paper')
    for line in other_lines:
        b=np.array(line['bbox'])
        # Source line itself overlaps the original glyph rectangle.
        if min(b[2],x2)>max(b[0],x1) and min(b[3],y2)>max(b[1],y1):
            continue
        if min(b[2],right)>max(b[0],left) and min(b[3],y2)>max(b[1],y1):
            raise ValueError('symmetric band overlaps another OCR line')
    block=array[y1:y2,x1:x2].copy()
    canvas=array.copy()
    canvas[y1:y2,left:right]=background
    result={}
    for arm,offset in [('sham',0),('left',-step),('right',step)]:
        value=canvas.copy()
        value[y1:y2,x1+offset:x2+offset]=block
        assert np.array_equal(value[y1:y2,x1+offset:x2+offset],block)
        result[arm]=value
    # Non-letter paper controls: same displacement sign, original word retained.
    shade=np.rint(np.median(block[:,:2],axis=1)).astype(np.uint8)
    for arm,start,end in [('editor_left',left,x1),('editor_right',x2,right)]:
        value=result['sham'].copy()
        value[y1:y2,start:end]=shade[:,None,:]
        assert np.array_equal(value[y1:y2,x1:x2],block)
        result[arm]=value
    allowed=np.zeros(array.shape[:2],bool)
    allowed[y1:y2,left:right]=True
    sham_colors,sham_counts=np.unique(result['sham'][allowed],axis=0,return_counts=True)
    for arm,value in result.items():
        assert np.array_equal(value[~allowed],array[~allowed])
        if arm in ['left','right']:
            colors,counts=np.unique(value[allowed],axis=0,return_counts=True)
            assert np.array_equal(colors,sham_colors) and np.array_equal(counts,sham_counts)
    return result,step,allowed,block,background


def prepare_all(folder):
    d=data()
    protocol=dict(version=1,source_population='all130 original query identities; no provenance or train/val count cap',
        selection='first quality-eligible source/isolated word per identity, deterministic photo and OCR order; no model response or correctness',
        quality='existing .08 tilt/.95 OCR/foreground/uniform-paper rules, plus symmetric blank destination within subject',
        classes='left=0/right=1, SAME absolute pixel displacement for both; Query features absent from position-only head',
        scope='signed isolated-word position in same-content images; not natural packaging identity or reading-order discrimination',
        historical_test_exposure=True,prior_right_only_magnitude_results='remain separately labeled as pilot',
        expected_relation='opposite signed word positions, equal shift magnitude; edit-minus-sham measures response, no original identity accuracy assigned to edited images',
        encoder='reuse compatible prior exact image tensors if available; otherwise single bounded-memory torchvision FP32-LoRA worker',
        cached_originals='generated originals may replay exactly; historical original differences remain separate new-runtime baseline')
    if (folder/'protocol.json').exists():assert read(folder/'protocol.json')==protocol
    else:write(folder/'protocol.json',protocol)
    groups=defaultdict(list)
    for qi,qid in enumerate(d['qb']['ids']):groups[d['photos'][qid]['package_id']].append((qid,qi))
    sources=[]
    rejects=[]
    for package in sorted(groups):
        checkpoint=folder/'source_checkpoints'/(package+'.json')
        if checkpoint.exists():
            record=read(checkpoint)
            for filename,digest in record.get('output_SHA256',{}).items():assert sha(folder/filename)==digest
            if record.get('source'):sources.append(record['source'])
            rejects.extend(record['rejections'])
            continue
        chosen=None
        rejected=[]
        for qid,qi in sorted(groups[package]):
            path=ROOT/d['qb']['paths'][qi]
            assert sha(path)==d['qb']['sha256'][qi]
            with Image.open(path) as image:
                raw=np.asarray(image.convert('RGB'))
                array=np.asarray(display_image(image)).copy()
            subject=np.array(d['boxes'][qid]['bbox'])
            spec=d['metadata']['photos'][qi]
            for li,line in enumerate(d['qo'][qid]['lines']):
                try:
                    box,_,_=prepare(array,line,spec,subject,d['qo'][qid]['lines'],.08)
                    edits,step,allowed,block,background=variants(array,box,subject,d['qo'][qid]['lines'])
                except ValueError as error:
                    rejected.append(dict(package_id=package,photo_id=qid,OCR_line=li,reason=str(error)))
                    continue
                assert np.array_equal(to_raw(array,spec['exif_orientation']),raw)
                chosen=dict(photo_id=qid,cache_row=qi,package_id=package,split=d['splits'][qid],group_id=d['groups'][package],word=line['text'],OCR_line=li,
                    image_SHA256=d['qb']['sha256'][qi],source_SHA256=d['qb']['source_sha256'][qi],original_path=str(path),
                    bbox_display=list(map(int,box)),shift_pixels=step,exif_orientation=spec['exif_orientation'],cache_provenance=d['qb']['provenance'][qi]['kind'],
                    label_scope='signed word position, magnitude held equal',validation=dict(equal_shift_magnitude=abs(-step)==abs(step),
                    foreground_rectangle_RGB_conserved=all(np.array_equal(v[box[1]:box[3],box[0]+offset:box[2]+offset],block) for arm,offset in [('sham',0),('left',-step),('right',step)] for v in [edits[arm]]),
                    outside_band_exact=all(np.array_equal(v[~allowed],array[~allowed]) for v in edits.values()),
                    sham_edit_histograms_identical=True,original_display_inverse_exact=np.array_equal(to_raw(array,spec['exif_orientation']),raw)),
                    background_RGB=background.tolist(),arms=[])
                for arm,value in edits.items():
                    output=folder/'images'/(qid+'_'+arm+'.png')
                    output.parent.mkdir(parents=True,exist_ok=True)
                    exif=Image.Exif();exif[274]=spec['exif_orientation']
                    Image.fromarray(to_raw(value,spec['exif_orientation'])).save(output,exif=exif)
                    offset=-step if arm=='left' else step if arm=='right' else 0
                    chosen['arms'].append(dict(id=qid+'_'+arm,source_id=qid,path=str(output),arm=arm,split=chosen['split'],package_id=package,
                        label=0 if arm=='left' else 1 if arm=='right' else None,
                        word_bbox=raw_bbox((box[0]+offset,box[1],box[2]+offset,box[3]),array.shape,spec['exif_orientation'])))
                chosen['arms'].insert(0,dict(id=qid+'_fresh_base',source_id=qid,path=str(path),arm='base',split=chosen['split'],package_id=package,label=None))
                break
            if chosen:break
        if chosen:sources.append(chosen)
        rejects.extend(rejected)
        hashes={str(Path(r['path']).relative_to(folder)):sha(r['path']) for r in chosen['arms'] if r['arm']!='base'} if chosen else {}
        write(checkpoint,dict(source=chosen,rejections=rejected,output_SHA256=hashes))
        write(folder/'preparation_progress.json',dict(identities_completed=len(list((folder/'source_checkpoints').glob('*.json'))),total_identities=len(groups),qualified_sources=len(sources)))
        print('Symmetric source',package,bool(chosen),len(sources),flush=True)
    write(folder/'sources.json',sources)
    write(folder/'encoding_manifest.json',[arm for source in sources for arm in source['arms']])
    csvout(folder/'rejections.csv',rejects)
    write(folder/'preparation_complete.json',dict(status='complete',source_identities=len(sources),all_population_identities_inspected=len(groups),
        source_identities_by_split={s:sum(r['split']==s for r in sources) for s in ['train','val','test']},
        no_identity_caps=True,quality_rule_only_selection=True,equal_shift_magnitude=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    folder=args.output.resolve();folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    prepare_all(folder)


if __name__=='__main__':main()
