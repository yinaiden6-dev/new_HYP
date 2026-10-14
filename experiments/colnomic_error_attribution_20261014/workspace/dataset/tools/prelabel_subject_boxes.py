"""Create resumable SAM3 box suggestions without changing human annotations."""
import argparse
import fcntl
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from photo_metadata import ROOT, PhotoStore, sha, write_json
from subject_boxes import SubjectBoxes

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def select_best_mask(outputs, threshold):
    """Select before full-size mask expansion; preserve SAM3's stable top-1 choice."""
    import torch
    scores = outputs.pred_logits.sigmoid()
    if outputs.presence_logits is not None:
        scores = scores * outputs.presence_logits.sigmoid()
    if scores.shape[0] != 1 or not torch.isfinite(scores).all():
        raise ValueError('Expected one image with finite SAM3 scores')
    index = int(scores[0].argmax())
    keep = scores[0] > threshold
    selected = SimpleNamespace(
        pred_logits=outputs.pred_logits[:, index:index+1],
        pred_boxes=outputs.pred_boxes[:, index:index+1],
        pred_masks=outputs.pred_masks[:, index:index+1],
        presence_logits=outputs.presence_logits)
    return selected, dict(candidates=int(keep.sum()), selected_index=int(keep[:index].sum()))


def limit_mask_memory(segmenter):
    """The editor needs one box, so do not expand every detected instance to 4K."""
    selection = {}
    if not segmenter.official and not getattr(segmenter, 'top1_before_postprocessing', False):
        original = segmenter.processor.post_process_instance_segmentation

        def post_process(outputs, **kwargs):
            best, details = select_best_mask(outputs, kwargs['threshold'])
            selection.update(details)
            return original(best, **kwargs)

        segmenter.processor.post_process_instance_segmentation = post_process
    return selection


def cached_regions(project, checkpoint, settings):
    """Reuse matching foreground masks' unpadded boxes, never padded crop bounds."""
    rows = {}
    for path in sorted((project / 'cache/foreground/sam3_auto').glob('*/config.json')):
        config = json.loads(path.read_text())
        if config.get('sam') != checkpoint or any(config['settings'].get(k) != v for k, v in settings.items()):
            continue
        manifest = path.parent / 'manifest.json'
        if manifest.exists():
            for row in json.loads(manifest.read_text())['photos']:
                rows[row['photo_id']] = dict(row, cache_path=str(manifest.relative_to(project)))
            continue
        for item in (path.parent / 'intermediate/regions').glob('*.json'):
            row = json.loads(item.read_text())
            rows[row['photo_id']] = dict(row, cache_path=str(item.relative_to(project)))
    return rows


def run(args):
    # Keep the lightweight review server independent of GPU packages.
    import torch
    from evaluation.foreground import Sam3Segmenter, checkpoint_spec
    from evaluation.regions import open_photo

    torch.set_num_threads(4)
    store = PhotoStore(args.root)
    boxes = SubjectBoxes(store)
    project = store.root.parent
    directory = project / 'cache/subject_boxes'
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or project / 'weights/sam3'
    settings = json.loads((project / 'configs/foreground.json').read_text())
    settings = {k: v for k, v in settings.items() if k.startswith('sam_')}
    started = time.monotonic()
    errors = []
    completed = 0

    def progress(stage, **extra):
        data = dict(stage=stage, done=completed, total=len(store.photos), pid=os.getpid(),
                    elapsed_seconds=round(time.monotonic() - started, 1), errors=errors, **extra)
        write_json(directory / 'progress.json', data)
        if completed % 25 == 0 or stage != 'running':
            print(json.dumps(data, ensure_ascii=False), flush=True)

    with (directory / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            progress('loading')
            checkpoint_info = checkpoint_spec(checkpoint)
            recipe = dict(model='sam3', checkpoint=checkpoint_info, settings=settings,
                          selection='highest confidence mask; unpadded mask bounding box',
                          orientation='EXIF-transposed original pixels',
                          code_sha256=sha(inspect.getsource(Sam3Segmenter).encode()))
            recipe_hash = sha(json.dumps(recipe, sort_keys=True).encode())
            write_json(directory / 'recipe.json', recipe)
            existing = boxes.read_prelabels()['records']
            pending = [row for pid, row in store.photos.items()
                       if existing.get(pid, {}).get('image_sha256') != row['sha256']
                       or existing.get(pid, {}).get('recipe_sha256') != recipe_hash]
            completed = len(store.photos) - len(pending)
            if args.limit is not None:
                pending = pending[:args.limit]
            cached = cached_regions(project, checkpoint_info, settings)
            segmenter = None
            for row in pending:
                pid = row['photo_id']
                info = boxes.image_info(pid)
                previous = cached.get(pid, {})
                if (previous.get('source_sha256') == info['image_sha256'] and
                    (previous.get('width'), previous.get('height')) == (info['width'], info['height'])):
                    bbox = previous.get('bbox')
                    details = {k: previous[k] for k in ('confidence', 'candidates', 'fallback_reason', 'cache_path') if k in previous}
                else:
                    if segmenter is None:
                        segmenter = Sam3Segmenter(checkpoint, settings, args.device)
                        selection = limit_mask_memory(segmenter)
                    image = open_photo(store.root / row['path'])
                    mask, details = segmenter(image)
                    details.update(selection)
                    bbox = list(mask.getbbox()) if mask is not None else None
                # A failed proposal stays pending; never substitute a full-image box.
                if bbox and (bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2):
                    bbox = None
                    details['fallback_reason'] = 'tiny_mask'
                boxes.save_prelabel(pid, bbox, recipe_sha256=recipe_hash, **details)
                completed += 1
                progress('running')
            progress('complete' if completed == len(store.photos) else 'partial')
        except Exception as error:
            errors.append(dict(type=type(error).__name__, message=str(error)))
            progress('failed')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, help='Process at most this many missing proposals')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    run(args)


if __name__ == '__main__':
    main()
