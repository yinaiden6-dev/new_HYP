"""Versioned before/after D01 inputs for the five user-reviewed box changes."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from scripts.colnomic_mechanism.common import ROOT, data, read, write, rows, csvout, sha
from scripts.colnomic_mechanism.display_pixels import display_image
from evaluation.regions import crop_region


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    revision = read(ROOT / 'results/colnomic_full_audit/20261013_resume/manual_box_review/box_revisions_final.json')
    changed = [r for r in revision['changed_records'] if r['before'].get('bbox') != r['after'].get('bbox')]
    d = data()
    previous = {r['photo_id']: r for r in rows(ROOT / 'results/colnomic_full_audit/20261013_resume/scene35/preprocessing_inputs.csv') if r['config_id'] == 'D01'}
    manifest, controls = [], []
    for change in changed:
        qid = change['photo_id']; row = previous[qid]
        source = ROOT / row['source_path']; assert sha(source) == change['after']['image_sha256']
        with Image.open(source) as im:
            image = display_image(im)
        old_crop, old_bounds = crop_region(image, change['before']['bbox'], margin=.05)
        after_crop, after_bounds = crop_region(image, change['after']['bbox'], margin=.05)
        old_path = ROOT / row['processed_input']
        assert sha(old_path) == row['input_sha256']
        with Image.open(old_path) as im:
            assert np.array_equal(np.asarray(im.convert('RGB')), np.asarray(old_crop)), qid
        new_path = folder / 'images' / (qid + '_crop_after.png'); new_path.parent.mkdir(exist_ok=True)
        after_crop.save(new_path, format='PNG', exif=Image.Exif())
        with Image.open(new_path) as im:
            assert np.array_equal(np.asarray(im.convert('RGB')), np.asarray(after_crop))
        for arm, path in [('crop_before', old_path), ('crop_after', new_path)]:
            manifest.append(dict(id=qid + '_' + arm, source_id=qid, path=str(path), arm=arm,
                split=d['splits'][qid], package_id=d['photos'][qid]['package_id'], label=None,
                image_SHA256=sha(path), full_image_SHA256=sha(source),
                box_snapshot_SHA256=revision['before_SHA256'] if arm == 'crop_before' else revision['after_SHA256']))
        controls.append(dict(photo_id=qid, package_id=d['photos'][qid]['package_id'], split=d['splits'][qid],
            original_path=str(source), original_size=list(image.size), before_path=str(old_path), after_path=str(new_path),
            before_bbox=change['before']['bbox'], after_bbox=change['after']['bbox'], before_bounds=old_bounds, after_bounds=after_bounds,
            before_crop_size=list(old_crop.size), after_crop_size=list(after_crop.size), old_crop_pixels_match_recorded_input=True,
            extra_supervision='user-reviewed target localization; displayed reference supplied in editor',
            confounds='target selection, background, relative scale and retained content change together; no isolated background claim'))
        preview = Image.new('RGB', (1500, 520), 'white'); draw = ImageDraw.Draw(preview)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
        for i, (label, part) in enumerate([('FULL ORIGINAL', image.copy()), ('BEFORE HUMAN CROP', old_crop), ('AFTER USER REVIEW', after_crop)]):
            part = part.copy(); part.thumbnail((495,475)); preview.paste(part,(i*500+(500-part.width)//2,40+(475-part.height)//2))
            draw.text((i*500+5,5),label+' '+qid,fill='black',font=font)
        target = folder / 'previews' / (qid+'.jpg'); target.parent.mkdir(exist_ok=True); preview.save(target,quality=95)
    write(folder / 'encoding_manifest.json', manifest); write(folder / 'crop_controls.json', controls)
    write(folder / 'protocol.json', dict(purpose='corrected human-box D01 sensitivity, fixed original gallery',
        source='all actual user changes of bbox coordinates, without retrieval-response selection',
        identities=len({r['package_id'] for r in controls}), photos=len(controls),
        baseline_before='old D01 processed input exact decoded pixels; must separately compare fresh old-crop encoding with historical D01 bank',
        after='same .05 padding and image processor; changed target box only upstream',
        no_original_data_cache_or_history_rewrites=True, matching_labels='natural original package labels; edited pixels are a new crop of same original',
        old_image_baseline='B03 full original and F03 unchanged; this sensitivity is separately named and is not an extra member of the historical35',
        prior_historical_gallery_encoding_reproducibility='not resolved; frozen gallery conditional scope remains explicit'))
    print('Prepared',len(manifest),'old/new crop encodings for',len(controls),'changed queries',flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); torch.set_num_threads(2); prepare(args.output.resolve())


if __name__ == '__main__': main()
