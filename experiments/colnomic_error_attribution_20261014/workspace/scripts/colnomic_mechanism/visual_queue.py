"""Prepare all visual audit materials ahead of numerical token computation."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
from scripts.colnomic_mechanism.common import ROOT,data,read,write,sha
from scripts.colnomic_mechanism.display_pixels import display_image


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    folder=args.output.resolve();folder.mkdir(parents=True,exist_ok=True)
    d=data();ledger={r['photo_id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
    ri={rid:i for i,rid in enumerate(d['rb']['ids'])}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    names=['Query original','Query subject crop','Correct primary','Image strongest wrong','Fusion strongest wrong']
    index=[]
    old=ROOT/'results/colnomic_full_audit/20261013_resume/full_tokens'
    old_index={r['batch']:r for r in read(old/'contact_sheet_index.json')} if (old/'contact_sheet_index.json').exists() else {}
    for start in range(0,len(d['qb']['ids']),4):
        number=start//4+1;qids=d['qb']['ids'][start:start+4]
        image_path=folder/('batch_%03d.jpg'%number);record_path=folder/('batch_%03d.json'%number)
        if record_path.exists():
            record=read(record_path)
            assert record['queries']==qids and sha(image_path)==record['sha256']
            index.append(record);continue
        if number in old_index and old_index[number]['queries']==qids:
            source=old/old_index[number]['path']
            import shutil
            assert sha(source)==old_index[number]['sha256']
            shutil.copyfile(source,image_path)
            record=dict(batch=number,queries=qids,path=image_path.name,sha256=sha(image_path),material_status='unreviewed',reused_from=str(source))
        else:
            canvas=Image.new('RGB',(2000,530*len(qids)),(233,235,238));draw=ImageDraw.Draw(canvas)
            missing=[]
            for row,qid in enumerate(qids):
                qi=start+row;case=ledger[qid];y0=row*530
                label=f'{qi+1}/{len(d["qb"]["ids"])} {qid} {case["package_id"]} {case["split"]} image rank={case["image_positive_rank"]} fusion rank={case["fusion_positive_rank"]} UNREVIEWED'
                draw.text((8,y0+4),label,font=font,fill='black')
                selected=[('q',qi),('crop',qi),('r',ri[case['positive_reference_ids'][0]]),('r',ri[case['image_strongest_wrong']]),('r',ri[case['fusion_strongest_wrong']])]
                for column,(kind,bank_index) in enumerate(selected):
                    x0=column*400
                    draw.rectangle((x0+4,y0+43,x0+396,y0+525),fill='white')
                    draw.text((x0+8,y0+25),names[column],font=font,fill='black')
                    bank=d['qb'] if kind in ['q','crop'] else d['rb']
                    source=ROOT/bank['paths'][bank_index]
                    if not source.exists():
                        missing.append(str(source));draw.text((x0+10,y0+80),'MISSING ORIGINAL',font=font,fill='red');continue
                    assert sha(source)==bank['sha256'][bank_index]
                    with Image.open(source) as picture:image=display_image(picture)
                    if kind=='crop':image=image.crop(tuple(d['boxes'][qid]['bbox']))
                    image.thumbnail((390,475))
                    canvas.paste(image,(x0+5+(390-image.width)//2,y0+47+(475-image.height)//2))
            canvas.save(image_path,quality=95,subsampling=0)
            record=dict(batch=number,queries=qids,path=image_path.name,sha256=sha(image_path),material_status='unreviewed',missing_originals=missing)
        write(record_path,record);index.append(record)
        write(folder/'index.json',index)
        write(folder/'progress.json',dict(material_queries=sum(len(r['queries']) for r in index),total_queries=len(d['qb']['ids']),
              actual_visual_reviews=0,material_generation_does_not_count_as_review=True))
        if number%10==0:print('Visual material',number,'/',(len(d['qb']['ids'])+3)//4,flush=True)
    assert sum(len(r['queries']) for r in index)==len(d['qb']['ids'])
    write(folder/'complete.json',dict(status='materials_ready',queries=len(d['qb']['ids']),batches=len(index),all_unique_queries=len({q for r in index for q in r['queries']}),
          visual_review_not_yet_completed=True))


if __name__=='__main__':main()
