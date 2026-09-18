---
base_model:
- Qwen/Qwen2.5-VL-7B-Instruct
library_name: peft
datasets:
- llamaindex/vdr-multilingual-train
- nomic-ai/colpali_train_set_split_by_source
language:
- en
- it
- fr
- de
- es
pipeline_tag: visual-document-retrieval
tags:
- vidore
- colpali
- multimodal_embedding
- multilingual_embedding
- Text-to-Visual Document (T→VD) retrieval
license: apache-2.0
---

# ColNomic Embed Multimodal 7B: State-of-the-Art Visual Document Retrieval

`colnomic-embed-multimodal-7b` is a multi-vector state-of-the-art multimodal embedding model that excels at visual document retrieval tasks:

- **High Performance**: Achieves 62.7 NDCG@5 on Vidore-v2, outperforming all other models
- **Unified Text-Image Encoding**: Directly encodes interleaved text and images without complex preprocessing
- **Advanced Architecture**: 7B parameter multimodal embedding model
- **Fully Open-Source**: Model weights, training data, and code available

## Performance

| Model | Avg. | ESG Restaurant Human | Econ Macro Multi. | AXA Multi. | MIT Bio | ESG Restaurant Synth. | ESG Restaurant Synth. Multi. | MIT Bio Multi. | AXA | Econ. Macro |
|-------|------|----------------------|-------------------|------------|---------|----------------------|----------------------------|---------------|-----|------------|
| **ColNomic Embed Multimodal 7B** | 62.7 | 73.9 | 54.7 | 61.3 | 66.1 | 57.3 | 56.7 | 64.2 | 68.3 | 61.6 |
| [ColNomic Embed Multimodal 3B](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-3b) | 61.2 | 65.8 | 55.4 | 61.0 | 63.5 | 56.6 | 57.2 | 62.5 | 68.8 | 60.2 |
| T-Systems ColQwen2.5-3B | 59.9 | 72.1 | 51.2 | 60.0 | 65.3 | 51.7 | 53.3 | 61.7 | 69.3 | 54.8 |
| [Nomic Embed Multimodal 7B](https://huggingface.co/nomic-ai/nomic-embed-multimodal-7b) | 59.7 | 65.7 | 57.7 | 59.3 | 64.0 | 49.2 | 51.9 | 61.2 | 66.3 | 63.1 |
| GME Qwen2 7B | 59.0 | 65.8 | 56.2 | 55.4 | 64.0 | 54.3 | 56.7 | 55.1 | 60.7 | 62.9 |
| [Nomic Embed Multimodal 3B](https://huggingface.co/nomic-ai/nomic-embed-multimodal-3b) | 58.8 | 59.8 | 57.5 | 58.8 | 62.5 | 49.4 | 49.4 | 58.6 | 69.6 | 63.5 |
| Llama Index vdr-2b-multi-v1 | 58.4 | 63.1 | 52.8 | 61.0 | 60.6 | 50.3 | 51.2 | 56.9 | 68.8 | 61.2 |
| Voyage Multimodal 3 | 55.0 | 56.1 | 55.0 | 59.5 | 56.4 | 47.2 | 46.2 | 51.5 | 64.1 | 58.8 |

## Getting Started

To use `colnomic-embed-multimodal-7b`, please install `colpali` from source

```bash
pip install git+https://github.com/illuin-tech/colpali.git
```


```python
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

model_name = "nomic-ai/colnomic-embed-multimodal-7b"

model = ColQwen2_5.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = ColQwen2_5_Processor.from_pretrained(model_name)

# Your inputs
images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (64, 32), color="black"),
]
queries = [
    "What is the organizational structure for our R&D department?",
    "Can you provide a breakdown of last year’s financial performance?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
```

## Model Architecture

- **Total Parameters**: 7B
- **Training Approach**: Fine-tuned from Qwen2.5-VL 7B Instruct
- **Architecture Type**: Vision-Language Model with unified text and image input processing
- **Key Innovations**:
  - Same-source sampling to create harder in-batch negatives
  - Multi-vector output option for enhanced performance

## Integration with RAG Workflows

Nomic Embed Multimodal 7B seamlessly integrates with Retrieval Augmented Generation (RAG) workflows:

1. **Direct Document Embedding**: Skip OCR and complex processing by directly embedding document page images
2. **Faster Processing**: Eliminate preprocessing steps for quicker indexing
3. **More Complete Information**: Capture both textual and visual cues in a single embedding
4. **Simple Implementation**: Use the same API for both text and images

## Recommended Use Cases

The model excels at handling real-world document retrieval scenarios that challenge traditional text-only systems:

- **Research Papers**: Capture equations, diagrams, and tables
- **Technical Documentation**: Encode code blocks, flowcharts, and screenshots
- **Product Catalogs**: Represent images, specifications, and pricing tables
- **Financial Reports**: Embed charts, graphs, and numerical data
- **Visually Rich Content**: Where layout and visual information are important
- **Multilingual Documents**: Where visual context provides important cues

## Training Details

ColNomic Embed Multimodal 7B was developed through several key innovations:

1. **Sampling From the Same Source**: Forcing sampling from the same dataset source creates harder in-batch negatives, preventing the model from learning dataset artifacts.

2. **Multi-Vector Configuration**: Providing a multi-vector variant that achieves higher performance than the dense variant.

## Limitations

- Performance may vary when processing documents with unconventional layouts or unusual visual elements
- While it handles multiple languages, performance is strongest on English content
- Processing very large or complex documents may require dividing them into smaller chunks
- Performance on documents with handwriting or heavily stylized fonts may be reduced

## Join the Nomic Community

- Nomic Embed Ecosystem: [https://www.nomic.ai/embed](https://www.nomic.ai/embed)
- Website: [https://nomic.ai](https://nomic.ai)
- Twitter: [https://twitter.com/nomic_ai](https://twitter.com/nomic_ai)
- Discord: [https://discord.gg/myY5YDR8z8](https://discord.gg/myY5YDR8z8)

## Citation

If you find this model useful in your research or applications, please consider citing:

```bibtex
@misc{faysse2024colpaliefficientdocumentretrieval,
  title={ColPali: Efficient Document Retrieval with Vision Language Models}, 
  author={Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and Céline Hudelot and Pierre Colombo},
  year={2024},
  eprint={2407.01449},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2407.01449}, 
}
@misc{ma2024unifyingmultimodalretrievaldocument,
      title={Unifying Multimodal Retrieval via Document Screenshot Embedding}, 
      author={Xueguang Ma and Sheng-Chieh Lin and Minghan Li and Wenhu Chen and Jimmy Lin},
      year={2024},
      eprint={2406.11251},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2406.11251}, 
}
@misc{nomicembedmultimodal2025,
  title={Nomic Embed Multimodal: Interleaved Text, Image, and Screenshots for Visual Document Retrieval},
  author={Nomic Team},
  year={2025},
  publisher={Nomic AI},
  url={https://nomic.ai/blog/posts/nomic-embed-multimodal},
}
```