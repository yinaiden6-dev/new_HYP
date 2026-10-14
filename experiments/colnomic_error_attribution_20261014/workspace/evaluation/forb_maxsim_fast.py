"""Exact FP32 MaxSim with reusable gallery storage and asynchronous GPU work.

Cached FP16 vectors retain their original bits; every dot product, maximum,
sum and division is FP32. Full-gallery columns and valid-token counts remain
unchanged. This module does not alter the frozen original scoring reference.
"""
import torch
from torch.nn.utils.rnn import pad_sequence

from .forb_scoring import _positive_batch_sizes, _tokens


BACKEND = 'resident_query_cached_gallery_fp32_v1'


class MaxSimEngine:
    @torch.inference_mode()
    def __init__(self, references, device='cuda:0', query_batch=4,
                 reference_batch=32, token_block=1024, cache_gallery=True,
                 gpu_cache_reserve_bytes=2 * 1024 ** 3):
        _positive_batch_sizes(query_batch=query_batch, reference_batch=reference_batch,
                              token_block=token_block)
        self.device = torch.device(device)
        self.query_batch, self.reference_batch, self.token_block = query_batch, reference_batch, token_block
        self.count = len(references)
        self.references = references
        self.cache_gallery = cache_gallery
        self.prepared = []
        self.dimension = None
        self.gpu_cache_bytes = 0
        self.cpu_cache_bytes = 0
        if self.device.type == 'cuda' and cache_gallery:
            free, _ = torch.cuda.mem_get_info(self.device)
            self.gpu_cache_limit = max(0, free - gpu_cache_reserve_bytes)
        else:
            self.gpu_cache_limit = 0
        torch.backends.cuda.matmul.allow_tf32 = False
        if cache_gallery:
            for first in range(0, self.count, reference_batch):
                self.prepared.append(self._prepare(references[first:first + reference_batch]))

    def _prepare(self, values):
        original_half = all(v.dtype == torch.float16 for v in values)
        tokens = [_tokens(v) for v in values]
        if self.dimension is None:
            self.dimension = tokens[0].shape[1]
        if any(v.shape[1] != self.dimension for v in tokens):
            raise ValueError('Reference embedding dimensions differ')
        # Conversion back to the original storage dtype is lossless: _tokens
        # merely selects rows and promotes the input to FP32.
        storage = [v.half() for v in tokens] if original_half else tokens
        reference = pad_sequence(storage, batch_first=True)
        lengths = torch.tensor([len(v) for v in tokens], dtype=torch.long)
        size = reference.numel() * reference.element_size()
        if self.cache_gallery and self.gpu_cache_bytes + size <= self.gpu_cache_limit:
            reference = reference.to(self.device)
            lengths = lengths.to(self.device)
            self.gpu_cache_bytes += size
        else:
            self.cpu_cache_bytes += size if self.cache_gallery else 0
        return reference, lengths

    @torch.inference_mode()
    def score(self, queries, progress=None):
        torch.backends.cuda.matmul.allow_tf32 = False
        qs = [_tokens(v) for v in queries]
        output = torch.empty((len(qs), self.count), device=self.device, dtype=torch.float32)
        if not qs or not self.count:
            return output.cpu()
        dimension = qs[0].shape[1]
        if any(v.shape[1] != dimension for v in qs):
            raise ValueError('Query embedding dimensions differ')
        if self.dimension is not None and dimension != self.dimension:
            raise ValueError('Query and reference embedding dimensions differ')
        query_batches = []
        for first in range(0, len(qs), self.query_batch):
            values = qs[first:first + self.query_batch]
            lengths = torch.tensor([len(v) for v in values], device=self.device)
            query = pad_sequence(values, batch_first=True).to(self.device)
            blocks = []
            for start in range(0, query.shape[1], self.token_block):
                block = query[:, start:start + self.token_block]
                valid = torch.arange(start, start + block.shape[1], device=self.device)[None] < lengths[:, None]
                blocks.append((block.flatten(0, 1).contiguous(), block.shape[1], valid))
            query_batches.append((first, len(query), lengths, blocks))
        for number, first in enumerate(range(0, self.count, self.reference_batch)):
            stored, lengths = (self.prepared[number] if self.cache_gallery else
                               self._prepare(self.references[first:first + self.reference_batch]))
            if self.dimension != dimension:
                raise ValueError('Query and reference embedding dimensions differ')
            reference = stored.to(device=self.device, dtype=torch.float32)
            lengths = lengths.to(self.device)
            reference_blocks = []
            for start in range(0, reference.shape[1], self.token_block):
                block = reference[:, start:start + self.token_block]
                valid = torch.arange(start, start + block.shape[1], device=self.device)[None] < lengths[:, None]
                reference_blocks.append((block.flatten(0, 1).T.contiguous(), block.shape[1], valid))
            for qfirst, qcount, qlengths, query_blocks in query_batches:
                total = torch.zeros((qcount, len(reference)), device=self.device, dtype=torch.float32)
                for query, qtokens, qvalid in query_blocks:
                    maxima = None
                    for ref, rtokens, rvalid in reference_blocks:
                        similarity = (query @ ref).view(qcount, qtokens, len(reference), rtokens)
                        similarity.masked_fill_(~rvalid[None, None, :, :], -torch.inf)
                        current = similarity.amax(-1).transpose(1, 2)
                        maxima = current if maxima is None else torch.maximum(maxima, current)
                        del similarity
                    maxima.masked_fill_(~qvalid[:, None, :], 0)
                    total.add_(maxima.sum(-1))
                output[qfirst:qfirst + qcount, first:first + len(reference)] = total / qlengths[:, None]
            if progress:
                progress(min(first + self.reference_batch, self.count), self.count)
        # Exactly one device synchronization / score transfer per query chunk.
        return output.cpu()


def maxsim_scores(queries, references, device='cuda:0', query_batch=4,
                  reference_batch=32, token_block=1024, progress=None):
    return MaxSimEngine(references, device, query_batch, reference_batch, token_block,
                        cache_gallery=False).score(queries, progress)
