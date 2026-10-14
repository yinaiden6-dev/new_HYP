"""Create query / wrong top-1 / correct-reference contact sheets for all errors."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import time

from PIL import Image, ImageDraw, ImageFont
from export_wrong_photos import ROOT, read_csv, write_csv, sha256

WIDTH, HEIGHT = 2100, 1100
PANEL_WIDTH, IMAGE_WIDTH, IMAGE_HEIGHT = 666, 630, 730
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
FONTS = {size: ImageFont.truetype(str(FONT_DIR/'DejaVuSans.ttf'), size) for size in (19, 21, 25, 28)}
BOLD = ImageFont.truetype(str(FONT_DIR/'DejaVuSans-Bold.ttf'), 25)


def thumbnail(path, expected_sha256):
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected_sha256, str(path)
    with Image.open(io.BytesIO(payload)) as original:
        orientation = original.getexif().get(274, 1)
        image = original.convert('RGB')
        method = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                  4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                  6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
                  8: Image.Transpose.ROTATE_90}.get(orientation)
        if method is not None:
            image = image.transpose(method)
        image.thumbnail((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
    return image


@lru_cache(maxsize=512)
def reference_thumbnail(path, expected_sha256):
    return thumbnail(path, expected_sha256)


def wrap(draw, value, font, width, max_lines=2):
    lines = []
    current = ''
    for word in value.split():
        candidate = (current+' '+word).strip()
        if draw.textlength(candidate, font=font) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and draw.textlength(lines[-1]+'...', font=font) > width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += '...'
    for i, line in enumerate(lines):
        if draw.textlength(line, font=font) > width:
            while line and draw.textlength(line+'...', font=font) > width:
                line = line[:-1]
            lines[i] = line+'...'
    return lines


def contact_sheet(config, prediction, photo, query, references, unique_reference_images):
    selected = references[prediction['selected_reference_id']]
    assert len(prediction['positive_reference_ids']) == 1
    correct = references[prediction['positive_reference_ids'][0]]
    assert prediction['correct'] is False and valid_error_rank(prediction)
    decision_only=prediction.get('ranking_scope')=='decision_only'
    rank_label='N/A (decision-only head)' if decision_only else f"{prediction['rank']} / {unique_reference_images}"
    assert selected['sha256'] != correct['sha256']
    canvas = Image.new('RGB', (WIDTH, HEIGHT), '#f1f5f9')
    draw = ImageDraw.Draw(canvas)
    title = f"{config['id']}  |  {config['model']}  |  {config['mode']}"
    if config['mode'] in ('fusion', 'ocr_fusion', 'bm25_image_fusion'):
        title += '  |  alpha=0.5'
    draw.text((28, 22), title, font=FONTS[28], fill='#0f172a')
    draw.text((28, 66), f"{photo['photo_id']}  |  {photo['package_id']}  |  Correct reference rank: {rank_label}",
              font=FONTS[25], fill='#334155')
    panels = [
        ('QUERY PHOTO', '#2563eb', query, photo['photo_id'], photo['identity']),
        ('WRONG REFERENCE (TOP-1)', '#b91c1c', reference_thumbnail(ROOT/'dataset'/selected['path'], selected['sha256']),
         selected['reference_id'], selected['identity']),
        ('CORRECT REFERENCE (RANK N/A)' if decision_only else f"CORRECT REFERENCE (RANK {prediction['rank']})", '#15803d',
         reference_thumbnail(ROOT/'dataset'/correct['path'], correct['sha256']), correct['reference_id'], correct['identity'])]
    for i, (label, color, image, identifier, identity) in enumerate(panels):
        x = 28+i*(PANEL_WIDTH+22)
        draw.rounded_rectangle((x, 118, x+PANEL_WIDTH, 1052), radius=12, fill='white', outline='#cbd5e1', width=2)
        draw.text((x+18, 137), label, font=BOLD, fill=color)
        draw.text((x+18, 177), identifier, font=FONTS[21], fill='#475569')
        draw.rectangle((x+18, 217, x+18+IMAGE_WIDTH, 217+IMAGE_HEIGHT), fill='#f8fafc')
        canvas.paste(image, (x+18+(IMAGE_WIDTH-image.width)//2, 217+(IMAGE_HEIGHT-image.height)//2))
        for line, text in enumerate(wrap(draw, identity, FONTS[19], IMAGE_WIDTH)):
            draw.text((x+18, 970+line*26), text, font=FONTS[19], fill='#334155')
    footer=('Left: query photo   |   Middle: incorrect final answer   |   Right: ground truth; full-gallery rank unavailable'
            if decision_only else 'Left: query photo   |   Middle: incorrect top-1 reference   |   Right: ground-truth reference with its retrieval rank')
    draw.text((28, 1070), footer,
              font=FONTS[19], fill='#475569')
    return canvas


def valid_error_rank(prediction):
    if prediction.get('ranking_scope')=='decision_only':
        return prediction.get('rank') is None
    return isinstance(prediction.get('rank'),int) and prediction['rank']>=2


def folder_name(config):
    name = '__'.join([config['id'], config['model'], config['mode']])
    if config['mode'] in ('fusion', 'ocr_fusion', 'bm25_image_fusion'):
        name += '__alpha_0.5'
    assert re.fullmatch(r'[A-Za-z0-9._-]+', name)
    return name


def identity_ambiguity_keys(output):
    """Preserve user-curated destinations when regenerating contact sheets."""
    registry = output/'身份歧义/cases.csv'
    rows = read_csv(registry) if registry.exists() else []
    keys = {(r['config_id'], r['photo_id']) for r in rows}
    if len(keys) != len(rows):
        raise ValueError('Duplicate identity-ambiguity cases')
    return keys


def case_destination(output, config, photo_id, ambiguity):
    folder = folder_name(config)
    return output/('身份歧义' if (config['id'], photo_id) in ambiguity else '')/folder/(photo_id+'.jpg')


def verify_group_paths(output, summary):
    folder = output/summary['folder']
    expected = {(folder/r['case_image']).resolve() for r in read_csv(folder/'errors.csv')}
    actual = {p.resolve() for base in [folder, output/'身份歧义'/summary['folder']] if base.exists()
              for p in base.iterdir() if p.suffix.lower() in ('.jpg', '.png', '.jpeg', '.webp')}
    assert actual == expected and len(actual) == summary['errors']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'wrong')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--preview', type=Path)
    parser.add_argument('--config', action='append', help='Render only these configuration IDs; retain and verify other groups')
    args = parser.parse_args()
    verification = json.loads((ROOT/'results/verification.json').read_text())
    assert verification['status'] == 'passed'
    configurations = read_csv(ROOT/'results/all_results.csv')
    if args.config and not set(args.config) <= {r['id'] for r in configurations}:
        parser.error('Unknown configuration ID')
    photos = {r['photo_id']: r for r in read_csv(ROOT/'dataset/lists/photos.csv')}
    references = {r['reference_id']: r for r in read_csv(ROOT/'dataset/lists/references.csv')}
    unique_references = verification['unique_reference_images']
    assert len(configurations) == verification['configurations']
    assert len(photos) == verification['queries_per_configuration']
    cases = defaultdict(list)
    summaries = []
    inputs = {name: sha256(ROOT/name) for name in ('results/all_results.csv', 'results/verification.json',
              'dataset/lists/photos.csv', 'dataset/lists/references.csv')}
    fields = ['photo_id', 'case_image', 'package_id', 'correct_rank', 'selected_reference_id',
              'correct_reference_id', 'query_path', 'selected_reference_path', 'correct_reference_path']
    output = args.output.resolve()
    ambiguity = identity_ambiguity_keys(output)
    ambiguity_registry = output/'身份歧义/cases.csv'
    ambiguity_records = {(r['config_id'], r['photo_id']): r for r in read_csv(ambiguity_registry)} if ambiguity_registry.exists() else {}
    if ambiguity_registry.exists():
        inputs[str(ambiguity_registry)] = sha256(ambiguity_registry)
    for config in configurations:
        result = ROOT/config['result']
        predictions = [json.loads(s) for s in (result/'predictions.jsonl').read_text().splitlines()]
        assert len(predictions) == len(photos) and {p['photo_id'] for p in predictions} == set(photos)
        errors = [p for p in predictions if p['correct'] is False]
        assert len(errors) == round(len(photos)*(1-float(config['recall_at_1'])))
        name = folder_name(config)
        rows = []
        for prediction in errors:
            photo = photos[prediction['photo_id']]
            assert prediction['selected_reference_id'] is not None and valid_error_rank(prediction)
            assert len(prediction['positive_reference_ids']) == 1
            if args.preview:
                image = contact_sheet(config, prediction, photo,
                    thumbnail(ROOT/'dataset'/photo['path'], photo['sha256']), references, unique_references)
                image.save(args.preview, quality=92, subsampling=0)
                print('Preview:', args.preview, flush=True)
                return
            correct_id = prediction['positive_reference_ids'][0]
            selected_id = prediction['selected_reference_id']
            if (config['id'], photo['photo_id']) in ambiguity:
                curated = ambiguity_records[config['id'], photo['photo_id']]
                if (curated['selected_reference_sha256'] != references[selected_id]['sha256'] or
                        curated['correct_reference_sha256'] != references[correct_id]['sha256']):
                    raise ValueError('References changed for a curated identity-ambiguity case: ' + photo['photo_id'])
            destination = case_destination(output, config, photo['photo_id'], ambiguity)
            destination.parent.mkdir(parents=True, exist_ok=True)
            cases[photo['photo_id']].append((config, prediction, destination))
            image_path = ('../身份歧义/'+name+'/'+destination.name
                          if (config['id'], photo['photo_id']) in ambiguity else destination.name)
            rows.append(dict(photo_id=photo['photo_id'], case_image=image_path,
                package_id=photo['package_id'], correct_rank=prediction['rank'],
                selected_reference_id=selected_id, correct_reference_id=correct_id,
                query_path='dataset/'+photo['path'], selected_reference_path='dataset/'+references[selected_id]['path'],
                correct_reference_path='dataset/'+references[correct_id]['path']))
        folder = output/name
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(folder/'errors.csv', rows, fields)
        (folder/'config.json').write_bytes((result/'config.json').read_bytes())
        inputs[str((result/'predictions.jsonl').relative_to(ROOT))] = sha256(result/'predictions.jsonl')
        summaries.append(dict(id=config['id'], model=config['model'], mode=config['mode'], folder=name,
            queries=len(photos), errors=len(errors), recall_at_1=config['recall_at_1'], result=config['result']))
    assert {p.name for p in output.iterdir() if p.is_dir() and p.name != '身份歧义'} == {s['folder'] for s in summaries}
    started = time.monotonic()
    completed = generated = written_bytes = 0

    bad_names = {r for r, reference in references.items()
                 if any(FONTS[19].getlength(word) > IMAGE_WIDTH for word in reference['identity'].split())}

    def render_photo(photo_id, items):
        pending = []
        total_bytes = 0
        for item in items:
            config, prediction, destination = item
            if args.config and config['id'] not in args.config:
                assert destination.is_file() and not destination.is_symlink(), str(destination)
                total_bytes += destination.stat().st_size
                continue
            refresh = prediction['selected_reference_id'] in bad_names or any(
                r in bad_names for r in prediction['positive_reference_ids'])
            if destination.exists() and not refresh:
                assert not destination.is_symlink()
                with Image.open(destination) as stored:
                    assert stored.format == 'JPEG' and stored.size == (WIDTH, HEIGHT)
                    stored.verify()
                total_bytes += destination.stat().st_size
            else:
                pending.append(item)
        if not pending:
            return len(items), total_bytes
        photo = photos[photo_id]
        query = thumbnail(ROOT/'dataset'/photo['path'], photo['sha256'])
        for config, prediction, destination in pending:
            image = contact_sheet(config, prediction, photo, query, references, unique_references)
            temporary = destination.with_name(destination.name+'.partial')
            image.save(temporary, format='JPEG', quality=92, subsampling=0)
            temporary.replace(destination)
            total_bytes += destination.stat().st_size
        return len(items), total_bytes

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(render_photo, photo_id, items) for photo_id, items in cases.items()]
        for future in as_completed(futures):
            count, size = future.result()
            completed += 1; generated += count; written_bytes += size
            if completed % 50 == 0 or completed == len(cases):
                print(json.dumps(dict(unique_photos_done=completed, unique_photos=len(cases),
                    cases_done=generated, seconds=round(time.monotonic()-started, 1))), flush=True)
    for summary in summaries:
        verify_group_paths(output, summary)
    assert generated == sum(s['errors'] for s in summaries)
    write_csv(output/'summary.csv', summaries, list(summaries[0]))
    report = dict(status='complete', created_at=datetime.now(timezone.utc).isoformat(),
        configurations=len(summaries), case_images=generated, unique_error_photos=len(cases),
        layout=['query photo', 'incorrect top-1 reference', 'correct reference with its retrieval rank'],
        dimensions=[WIDTH, HEIGHT], rank_population=unique_references, layout_version=2,
        rank_definition='1-based rank among unique reference images; copied from independently verified predictions',
        source_sha256_verified=True, bytes=written_bytes, inputs_sha256=inputs,
        elapsed_seconds=time.monotonic()-started, groups=summaries)
    if args.config:
        report['rendered_configurations'] = args.config
    (output/'export.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    lines = [f'# {len(summaries)} 组实验的错误案例对照图', '',
        '每张 JPG 按顺序合成：左侧查询照片、中间误判的 Top-1 reference、右侧正确 reference。',
        '图中标注实验配置、照片与参考图 ID，以及正确图在本组检索中的排名。',
        '正确图的排名取自已独立核验的逐图结果，是 5,330 张去重参考图中的 1 起始排名。',
        '英文标题 QUERY PHOTO / WRONG REFERENCE / CORRECT REFERENCE 分别对应上述三个位置；图片保持比例。',
        '按 correct=false 导出；每个配置文件夹附 errors.csv 与 config.json。清单中的原图路径相对仓库根目录。',
        f'共 {len(summaries)} 组、{generated:,} 张对照图，涉及 {len(cases):,} 张不同查询照片。', '',
        f'用户归入身份歧义的 {len(ambiguity)} 个案例保留在身份歧义子目录；路径与两张参考图记录见 [身份歧义/cases.csv](身份歧义/cases.csv)。', '',
        '| ID | 文件夹 | 错误案例数 |', '|---|---|---:|']
    lines += [f"| {s['id']} | [{s['folder']}]({s['folder']}/) | {s['errors']} |" for s in summaries]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs_sha256', 'groups')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
