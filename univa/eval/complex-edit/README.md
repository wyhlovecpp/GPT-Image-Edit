# Complex-Edit with Flux Kontext

This directory contains scripts to evaluate Flux Kontext models on the Complex-Edit benchmark for instruction-based image editing.

## Quick Start

### 1. Generate Edited Images

Run the generation script with complexity level 8 (most challenging):

```bash
cd univa/eval/complex-edit

# Single GPU
python step1_gen_samples.py \
  --config complex_edit.yaml 

# Multi-GPU (recommended)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node 8 \
  step1_gen_samples.py \
  --config complex_edit.yaml 
```

### 2. Run Evaluation

After generation is complete, evaluate the results:

```bash
python eval.py \
  -p /path/to/results/complex_edit_kontext/complexity_8_real \
  -c 8 \
  --image-type real \
  -n 2 \
  -m 5 \
  --num-processes 16
```

## Configuration

Edit `complex_edit.yaml` to customize:

- `complexity`: Editing complexity level (1-8, default: 8)
- `image_type`: Dataset split - "real" or "syn" (default: "real")
- `height`/`width`: Output image dimensions (default: 1024x1024)
- `num_inference_steps`: Denoising steps (default: 28)
- `guidance_scale`: CFG guidance strength (default: 3.5)

## Dataset

The script automatically downloads the Complex-Edit dataset from HuggingFace:
- **Dataset**: `UCSC-VLAA/Complex-Edit`
- **Splits**: `test_real` (real images), `test_syn` (synthetic images)
- **Size**: ~1K samples per split

## Key Features

1. **Automatic Data Loading**: Uses HuggingFace Datasets to load Complex-Edit
2. **Multi-GPU Support**: Efficient distribution across multiple GPUs
3. **Flexible Complexity**: Test different editing difficulty levels (1-8)
4. **Image Resizing**: Automatic resizing to match pipeline requirements
5. **Resume Support**: Skips already processed samples

## Workflow Example

```python
# The script follows this workflow:
from datasets import load_dataset

# 1. Load dataset
dataset = load_dataset("UCSC-VLAA/Complex-Edit")
test_data = dataset["test_real"]

# 2. Extract complexity-8 prompts
prompts = [
    edit["compound"][7]["compound_instruction"]  # complexity 8 = index 7
    for edit in test_data["edit"]
]

# 3. Process each sample
for idx, (image, prompt) in enumerate(zip(test_data["image"], prompts)):
    edited_image = flux_kontext_pipeline(image=image, prompt=prompt)
    edited_image.save(f"output/{idx:05d}.png")

# 4. Evaluate with original eval.py
```

## Output Structure

```
results/complex_edit_kontext/
└── complexity_8_real/
    ├── 00000.png
    ├── 00001.png
    ├── ...
    ├── alignment_evaluator_results/
    ├── quality_evaluator_results/
    └── overall/
```

## Evaluation Metrics

The evaluation measures:
- **Alignment**: How well edits follow instructions
- **Quality**: Visual quality of edited images  
- **Overall**: Combined score

Results are saved in JSON format for detailed analysis.

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size or use smaller image dimensions
2. **Dataset download issues**: Ensure internet connection and HuggingFace access
3. **Evaluation errors**: Check that output directory contains numbered PNG files

## References

- [Complex-Edit Paper](https://arxiv.org/abs/2504.13143)
- [Complex-Edit Dataset](https://huggingface.co/datasets/UCSC-VLAA/Complex-Edit)
- [Project Page](https://ucsc-vlaa.github.io/Complex-Edit/) 