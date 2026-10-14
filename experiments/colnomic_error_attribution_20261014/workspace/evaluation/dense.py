"""Frozen global image encoders with pinned local checkpoints."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import torch
from torch.nn import functional as F

DENSE_DIMENSIONS = {'Qwen3-VL-Embedding-2B': 2048, 'dinov3': 1024, 'siglip2': 1024}
IMAGE_ENCODERS = ('dinov3', 'siglip2')


class DenseEncoder:
    def __init__(self, name, config, root, device='cuda:0'):
        from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel
        self.name = name
        self.device = device
        self.contract = config['image_contract']
        self.dtype = getattr(torch, self.contract['dtype'].removeprefix('torch.'))
        path = Path(root) / config['checkpoint']
        if not (path / 'model.safetensors').is_file():
            raise FileNotFoundError(f'Missing {name} weights: {path}. Run scripts/download_weights.py --model {name}')
        backend = self.contract['processor_backend']
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True, backend=backend)
        if self.processor.backend != backend:
            raise ValueError(f'{name} processor backend changed: {backend} -> {self.processor.backend}')
        model_class = SiglipVisionModel if name == 'siglip2' else AutoModel
        self.model, loading = model_class.from_pretrained(path, local_files_only=True, dtype=self.dtype,
            attn_implementation='sdpa', output_loading_info=True)
        unexpected = [key for key in loading.get('unexpected_keys', []) if not (
            name == 'siglip2' and (key.startswith('text_model.') or key in ('logit_scale', 'logit_bias')))]
        if unexpected or loading.get('missing_keys') or loading.get('mismatched_keys'):
            raise ValueError(f'Incomplete {name} vision checkpoint: {loading}')
        self.model = self.model.to(device).eval()
        self.workers = ThreadPoolExecutor(max_workers=4)

    def prepare_image(self, path):
        with Image.open(path) as source:
            image = source.convert('RGB')
        return self.processor(images=image, return_tensors='pt',
                              **self.contract['processor_kwargs'])['pixel_values'][0]

    @torch.inference_mode()
    def encode_images(self, paths):
        pixels = torch.stack(list(self.workers.map(self.prepare_image, paths)))
        output = self.model(pixel_values=pixels.to(device=self.device, dtype=self.dtype))
        vectors = output.last_hidden_state[:, 0] if self.name == 'dinov3' else output.pooler_output
        vectors = F.normalize(vectors.float(), dim=-1).cpu()
        if vectors.shape != (len(paths), DENSE_DIMENSIONS[self.name]) or not torch.isfinite(vectors).all():
            raise ValueError(f'Invalid {self.name} image vectors')
        return list(vectors.unbind())

    def encode_image(self, path):
        return self.encode_images([path])[0]

    def release(self):
        self.workers.shutdown(wait=True)
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ParallelDenseEncoder:
    """Run independent frozen replicas; preserve input order when joining batches."""
    def __init__(self, name, config, root, device='cuda:0', *, devices):
        if not devices or len(set(devices)) != len(devices):
            raise ValueError('Image encoding devices must be nonempty and distinct')
        self.encoders = []
        try:
            for target in devices:
                self.encoders.append(DenseEncoder(name, config, root, target))
        except Exception:
            for encoder in self.encoders:
                encoder.release()
            raise
        self.executor = ThreadPoolExecutor(max_workers=len(self.encoders))

    def encode_images(self, paths):
        if not paths:
            return []
        size = (len(paths) + len(self.encoders) - 1) // len(self.encoders)
        futures = [self.executor.submit(encoder.encode_images, paths[start:start + size])
                   for encoder, start in zip(self.encoders, range(0, len(paths), size))]
        return [value for future in futures for value in future.result()]

    def release(self):
        self.executor.shutdown(wait=True)
        for encoder in self.encoders:
            encoder.release()
        self.encoders.clear()
