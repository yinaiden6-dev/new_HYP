"""Pinned retrieval encoders; full OCR text and explicit image recipes."""
import gc
import importlib.util
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from PIL import Image

class SinglePatchProjection(torch.nn.Module):
    def __init__(self, convolution):
        super().__init__()
        if (
            convolution.stride != convolution.kernel_size
            or convolution.padding != (0, 0, 0)
            or convolution.dilation != (1, 1, 1)
            or convolution.groups != 1
        ):
            raise ValueError("The projection must contain exactly one unpadded patch")
        self.weight = convolution.weight
        self.bias = convolution.bias
        self.patch_shape = tuple(convolution.weight.shape[1:])

    def forward(self, inputs):
        if tuple(inputs.shape[1:]) != self.patch_shape:
            raise ValueError("Unexpected patch shape")
        # One output position makes Conv3d exactly a dense projection.
        output = F.linear(inputs.flatten(1), self.weight.flatten(1), self.bias)
        return output[:, :, None, None, None]


def stable_sdpa(module, query, key, value, attention_mask, dropout=0.0,
                scaling=None, is_causal=None, **kwargs):
    dtype = query.dtype
    if query.is_cuda and dtype == torch.bfloat16 and torch.cuda.get_device_capability(query.device)[0] < 8:
        query, key, value = query.float(), key.float(), value.float()
        if attention_mask is not None and attention_mask.is_floating_point():
            attention_mask = attention_mask.float()
    groups = query.shape[1] // key.shape[1]
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    causal = getattr(module, 'is_causal', True) if is_causal is None else is_causal
    causal = bool(query.shape[2] > 1 and attention_mask is None and causal)
    output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
        dropout_p=dropout, scale=scaling, is_causal=causal)
    return output.to(dtype).transpose(1, 2).contiguous(), None


class Encoder:
    def __init__(self, name, config, root, device="cuda:0"):
        from transformers import AutoConfig
        self.name = name
        self.config = config
        self.root = Path(root)
        self.device = str(device)
        self.spec = dict(config['text_model'])
        self.family = config['family']
        self.dtype = getattr(torch, config['text_contract']['compute_dtype'].removeprefix('torch.'))
        self.path = str(self.root / config['checkpoint'])
        if not Path(self.path).is_dir():
            raise FileNotFoundError(f'Pinned checkpoint is unavailable: {self.path}')
        kwargs = dict(torch_dtype=self.dtype, attn_implementation='sdpa', local_files_only=True)
        torch.cuda.set_device(self.device)
        loading = {}
        if name == 'colpali':
            from transformers import ColPaliForRetrieval, ColPaliProcessor
            self.processor = ColPaliProcessor.from_pretrained(self.path, local_files_only=True)
            self.model, loading = ColPaliForRetrieval.from_pretrained(self.path,
                key_mapping={r'^vlm\.language_model\.model\.': 'vlm.language_model.'},
                output_loading_info=True, **kwargs)
            self.model.to(self.device).eval()
            self.context = self.model.config.vlm_config.text_config.max_position_embeddings
        elif name == 'Qwen3-VL-Embedding-2B':
            source = Path(self.path) / 'scripts/qwen3_vl_embedding.py'
            spec = importlib.util.spec_from_file_location('release_qwen3_embedding', source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.embedder = module.Qwen3VLEmbedder(model_name_or_path=self.path, **kwargs)
            self.processor = self.embedder.processor
            self.model = self.embedder.model.to(self.device).eval()
            self.model.visual.patch_embed.proj = SinglePatchProjection(self.model.visual.patch_embed.proj)
            self.context = 32768
        else:
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            from transformers.models.qwen2_5_vl import Qwen2_5_VLModel
            ALL_ATTENTION_FUNCTIONS.register('sdpa', stable_sdpa)

            class ShardedColQwen(ColQwen2_5):
                def forward(self, *args, **kw):
                    if 'pixel_values' in kw:
                        offsets = kw['image_grid_thw'][:,1] * kw['image_grid_thw'][:,2]
                        kw['pixel_values'] = torch.cat([v[:n] for v,n in zip(kw['pixel_values'],offsets)],dim=0)
                    kw.pop('return_dict', None)
                    kw.pop('output_hidden_states', None)
                    kw.pop('use_cache', None)
                    hidden = Qwen2_5_VLModel.forward(self, *args, **kw,
                        use_cache=False, output_hidden_states=True, return_dict=True).last_hidden_state
                    output = self.custom_text_proj(hidden.to(next(self.custom_text_proj.parameters()).device))
                    output = output / output.norm(dim=-1, keepdim=True)
                    output = output * kw['attention_mask'].to(output.device).unsqueeze(-1)
                    if 'pixel_values' in kw and self.mask_non_image_embeddings:
                        output *= kw['input_ids'].eq(self.config.image_token_id).to(output.device).unsqueeze(-1)
                    return output

            self.processor = ColQwen2_5_Processor.from_pretrained(self.path, local_files_only=True)
            # Pin the trained query format independently of engine defaults.
            self.processor.query_prefix = config['text_contract']['query_prefix']
            base = self.spec['base_model']
            self.spec['base_revision'] = config['base_revision']
            base_path = str(self.root / config['base_checkpoint'])
            model_class = ColQwen2_5
            if name == 'colnomic7b':
                from accelerate import infer_auto_device_map, init_empty_weights
                if torch.cuda.device_count() < 2 or torch.device(self.device).index not in (None,0):
                    raise ValueError('ColNomic 7B requires two visible GPUs and --device cuda:0')
                model_class = ShardedColQwen
                with init_empty_weights():
                    empty = model_class(AutoConfig.from_pretrained(base_path, local_files_only=True))
                placement = infer_auto_device_map(empty, max_memory={0:'7680MiB',1:'9GiB'},
                    dtype=self.dtype, no_split_module_classes=empty._no_split_modules)
                placement['custom_text_proj'] = 0
                assert not any(v in ('cpu','disk') for v in placement.values())
                del empty
                kwargs['device_map'] = placement
            self.model, loading = model_class.from_pretrained(base_path, output_loading_info=True, **kwargs)
            if name == 'colnomic7b':
                loading['unexpected_keys'] = [k for k in loading.get('unexpected_keys',[]) if k != 'lm_head.weight']
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.path, local_files_only=True)
            if name != 'colnomic7b':
                self.model.to(self.device)
            self.model.eval()
            self.context = self.model.config.text_config.max_position_embeddings
        assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys')), loading
        self.contract = dict(recipe='release.ocr_text_embeddings.v1', view='text', source='PP-OCRv6 text',
            field='text', preprocessing='verbatim UTF-8; preserve case and newlines', truncation=False,
            chunking=False, images_supplied=False, same_recipe_for_photos_and_references=True,
            compute_dtype=str(self.dtype), context_limit=self.context,
            empty_text='has_text=false; zero dense vector or empty multi-vector; no model call',
            output_dtype='torch.float32' if name == 'Qwen3-VL-Embedding-2B' else 'torch.float16')
        if name == 'Qwen3-VL-Embedding-2B':
            self.contract.update(instruction=self.embedder.default_instruction,
                template='official chat template with add_generation_prompt=True',
                pooling='official last attended token; L2 normalize in model dtype, then cast to float32')
        else:
            self.contract.update(query_prefix=self.processor.query_prefix,
                query_augmentation_token=self.processor.query_augmentation_token, augmentation_repeats=10,
                pooling='all attended token vectors; model L2 normalization; remove batch padding')
        if self.family in ('colnomic_3b', 'colnomic_7b', 'colqwen2_5'):
            self.contract['attention'] = 'fp32_sdpa_on_pre_ampere_v1'

        if self.contract != config['text_contract']:
            raise ValueError('Processor or text encoding contract changed')

    def prepare(self, texts):
        if self.name == 'Qwen3-VL-Embedding-2B':
            conv = [self.embedder.format_model_input(text=t) for t in texts]
            rendered = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=rendered, padding=True, truncation=False, return_tensors='pt')
        elif self.name == 'colpali':
            inputs = self.processor(text=texts, padding=True, truncation=False,
                                    max_length=self.context, return_tensors='pt')
        else:
            inputs = self.processor.process_queries(texts)
        assert not any('pixel' in key or 'image' in key or 'video' in key for key in inputs)
        assert inputs['input_ids'].shape[1] <= self.context
        return inputs

    @torch.inference_mode()
    def encode_texts(self, texts):
        return self.encode_prepared(self.prepare(texts))

    @torch.inference_mode()
    def encode_prepared(self, inputs):
        inputs = inputs.to(self.device)
        counts = inputs['attention_mask'].sum(dim=1).tolist()
        output = self.model(**inputs, use_cache=False)
        if self.name == 'Qwen3-VL-Embedding-2B':
            result = F.normalize(self.embedder._pooling_last(output.last_hidden_state, inputs['attention_mask']),
                                 p=2, dim=-1).detach().float().cpu()
            values = list(result.unbind())
        else:
            vectors = output.embeddings if self.name == 'colpali' else output
            mask = inputs['attention_mask'].bool().to(vectors.device)
            values = [v[m].detach().half().cpu() for v,m in zip(vectors,mask,strict=True)]
        for value in values:
            assert torch.isfinite(value).all(), 'Non-finite model output'
            norms = value.float().norm(dim=-1)
            norms = norms[norms > 0] if norms.ndim else norms.reshape(1)
            assert norms.numel(), 'Model output contains no effective tokens'
            assert ((norms - 1).abs() < 0.03).all(), f'Bad norm {norms.min()} {norms.max()}'
        return values, counts

    @torch.inference_mode()
    def encode_image(self, path):
        with Image.open(path) as source:
            image = source.convert('RGB')
        if self.name == 'Qwen3-VL-Embedding-2B':
            # Reuse the official processor but fail on vision errors instead of NULL fallback.
            from qwen_vl_utils import process_vision_info
            conversation = [self.embedder.format_model_input(image=image,
                instruction='Retrieve the matching product package image for the query.')]
            prompt = self.processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=False)
            images, videos, kwargs = process_vision_info(conversation,image_patch_size=16,
                return_video_metadata=True,return_video_kwargs=True)
            if images is None or videos is not None:
                raise ValueError('Invalid image-only model input')
            inputs = self.processor(text=prompt,images=images,truncation=True,
                max_length=self.embedder.max_length,padding=True,do_resize=False,return_tensors='pt',**kwargs)
        elif self.name == 'colpali':
            inputs = self.processor(images=[image],return_tensors='pt')
        else:
            inputs = self.processor.process_images([image])
        return self.encode_prepared(inputs)[0][0]

    def encode_text(self, text):
        if not text.strip():
            return (torch.zeros(2048,dtype=torch.float32) if self.name=='Qwen3-VL-Embedding-2B'
                    else torch.empty(0,128,dtype=torch.float16)), 0
        embeddings,counts = self.encode_texts([text])
        return embeddings[0],counts[0]

    def release(self):
        self.model = None
        if hasattr(self,'embedder'):
            self.embedder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
