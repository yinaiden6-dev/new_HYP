"""Rotate only inspection snippets before display scaling; no encoder inputs change."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.colnomic_mechanism.common import ROOT,data,read,write,sha
from scripts.colnomic_mechanism.display_pixels import display_image
from scripts.colnomic_mechanism.natural_shared_phrase_probe import textkey

def main():
    folder=ROOT/'results/colnomic_full_audit/20261013_resume/natural_shared_phrase_layout'
    d=data();qi={q:i for i,q in enumerate(d['qb']['ids'])};ri={r:i for i,r in enumerate(d['rb']['ids'])}
    result=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for row in read(folder/'manual_review_queue.json'):
        canvas=Image.new('RGB',(1800,650),'white');draw=ImageDraw.Draw(canvas);rotations=[]
        for col,ident in enumerate([row['photo_id'],row['positive'],row['negative']]):
            bank=d['qb'] if col==0 else d['rb'];index=qi[ident] if col==0 else ri[ident]
            path=ROOT/bank['paths'][index];assert sha(path)==bank['sha256'][index]
            with Image.open(path) as im:image=display_image(im)
            ocr=(d['qo'] if col==0 else d['ro'])[ident]
            draw.text((col*600+6,5),ident+' (inspection rotations only)',fill='black',font=font)
            for lineindex,key in enumerate(row['phrases']):
                line=next(l for l in ocr['lines'] if textkey(l['text'])==key)
                x1,y1,x2,y2=line['bbox'];patch=image.crop((max(0,int(x1)-3),max(0,int(y1)-3),min(image.width,int(x2)+3),min(image.height,int(y2)+3)))
                rotation=90 if patch.height>2*patch.width else 0
                if rotation:patch=patch.rotate(rotation,expand=True)
                patch.thumbnail((590,105));canvas.paste(patch,(col*600+4,42+lineindex*150+(105-patch.height)//2))
                draw.text((col*600+4,150+lineindex*150),key[:73],fill='black',font=font)
                rotations.append(dict(image=ident,phrase=key,inspection_rotation=rotation))
        path=folder/'readability_material'/(row['package_id']+'.jpg');path.parent.mkdir(exist_ok=True);canvas.save(path,quality=96)
        result.append(dict(**row,readability_material=str(path.relative_to(folder)),readability_SHA256=sha(path),inspection_transformations=rotations))
    write(folder/'readability_queue.json',result);print('Readable cards',len(result),flush=True)

if __name__=='__main__':main()
