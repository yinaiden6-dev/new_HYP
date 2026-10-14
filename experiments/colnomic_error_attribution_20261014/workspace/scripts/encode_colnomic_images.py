"""Encode new/edited images with the frozen ColNomic 7B baseline recipe.

This writes a new bank, never changes the released banks. --prepare-only validates
raw-image decoding, processor inputs and token layout without loading model weights.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch
from PIL import Image

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from drugpack.method.token_layout import token_layout


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def token_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str(value.dtype).encode() +
        json.dumps(list(value.shape), separators=(',', ':')).encode() +
        value.view(torch.uint8).numpy().tobytes()).hexdigest()


def prepare_record(processor, path, identifier):
    with Image.open(path) as source:
        width, height = source.size
        orientation = int(source.getexif().get(274, 1))
        image = source.convert('RGB')  # Same raw decoding as evaluation/encoders.py.
    inputs = processor.process_images([image])
    layout = token_layout(processor, 'colnomic7b', width, height)
    attended = inputs['input_ids'][0][inputs['attention_mask'][0].bool()]
    image_token = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    positions = torch.where(attended == image_token)[0].tolist()
    if len(attended) != layout['total'] or positions != list(range(layout['start'], layout['start'] + layout['count'])):
        raise ValueError('Processor inputs differ from reconstructed layout')
    return dict(id=identifier, path=str(path), image_sha256=file_hash(path),
        raw_width=width, raw_height=height, exif_orientation=orientation, layout=layout,
        attended_input_ids=attended.tolist(), image_grid_thw=inputs['image_grid_thw'].tolist(),
        decode='PIL convert RGB in raw stored frame; no EXIF transpose',
        input_shapes={key:list(value.shape) for key, value in inputs.items() if isinstance(value, torch.Tensor)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=CODE_ROOT)
    parser.add_argument('--image', action='append', type=Path, default=[])
    parser.add_argument('--manifest', type=Path, help='CSV with id,path; paths relative to --root')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, help='Complete local adapter checkpoint (including weights)')
    parser.add_argument('--base', type=Path, help='Complete local base checkpoint (including weights)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.output.exists() and not args.overwrite:
        parser.error('Output already exists; choose a new path or use --overwrite')
    records = [dict(id=path.stem, path=str(path)) for path in args.image]
    if args.manifest:
        with args.manifest.open(newline='') as stream:
            records.extend(csv.DictReader(stream))
    if not records or len({row['id'] for row in records}) != len(records):
        parser.error('Supply at least one image and unique IDs')
    config = json.loads((root/'configs/baselines.json').read_text())['models']['colnomic7b']
    if args.adapter:
        config['checkpoint'] = str(args.adapter.resolve())
    if args.base:
        config['base_checkpoint'] = str(args.base.resolve())
    from colpali_engine.models import ColQwen2_5_Processor
    processor = ColQwen2_5_Processor.from_pretrained(root/config['checkpoint'], local_files_only=True)
    paths = [root/row['path'] for row in records]
    metadata = [prepare_record(processor, path, row['id']) for row, path in zip(records, paths, strict=True)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        args.output.write_text(json.dumps(dict(schema='colnomic.prepared_images.v1', records=metadata,
            model_config=config, inference_performed=False), ensure_ascii=False, indent=2)+'\n')
        print(f'Prepared {len(metadata)} images; no model inference', flush=True)
        return
    from evaluation.encoders import Encoder
    encoder = Encoder('colnomic7b', config, root, args.device)
    embeddings = []
    try:
        for i, (path, spec) in enumerate(zip(paths, metadata, strict=True)):
            value = encoder.encode_image(path)
            if list(value.shape) != [spec['layout']['total'], 128] or value.dtype != torch.float16:
                raise ValueError('Unexpected encoded shape or storage dtype')
            spec['all_tokens_sha256'] = token_hash(value)
            embeddings.append(value)
            print(f'Encoded {i+1}/{len(paths)} {spec["id"]}', flush=True)
    finally:
        encoder.release()
    result = dict(schema='colnomic.new_image_embeddings.v1', ids=[row['id'] for row in records],
        paths=[str(path) for path in paths], sha256=[row['image_sha256'] for row in metadata],
        embeddings=embeddings, records=metadata, model_config=config,
        compute_dtype='torch.bfloat16', storage_dtype='torch.float16',
        provenance='evaluation.encoders.Encoder.encode_image, frozen checkpoint, eval mode')
    temporary = args.output.with_name(args.output.name+'.tmp')
    torch.save(result, temporary)
    temporary.replace(args.output)
    print(f'Saved {len(embeddings)} image embeddings to {args.output}', flush=True)


if __name__ == '__main__':
    main()
