"""Visual token grid used by the frozen rebut feature and RoMa caches."""

def token_layout(processor, model, width, height):
    if model == 'colpali':
        # The official processor prepends 1024 image tokens before its prompt.
        return dict(height=32, width=32, start=0, count=1024, total=1030)
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    ip = processor.image_processor
    factor = ip.patch_size * ip.merge_size
    h, w = smart_resize(height=height, width=width, factor=factor,
                       min_pixels=ip.size['shortest_edge'], max_pixels=ip.size['longest_edge'])
    count = (h//factor)*(w//factor)
    text = processor.visual_prompt_prefix.replace(processor.image_token, processor.image_token*count)
    ids = processor.tokenizer(text, add_special_tokens=True)['input_ids']
    token = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    positions = [i for i, value in enumerate(ids) if value == token]
    if len(positions) != count or positions != list(range(positions[0], positions[0]+count)):
        raise ValueError('Unexpected visual token order')
    return dict(height=h//factor, width=w//factor, start=positions[0], count=count, total=len(ids))
