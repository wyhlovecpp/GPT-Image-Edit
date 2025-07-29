# 📣 News

* **[2025.07.27]** 🤗 We release GPT-Image-Edit, a state-of-the-art image editing model with 1.5M high-quality editing samples. All [data](https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M), [models](https://huggingface.co/UCSC-VLAA/gpt-image-edit-training), training code and evaluation code are open-sourced. Our code is based on [UniWorld-V1](https://github.com/PKU-YuanGroup/UniWorld-V1), Thanks to the authors of UniWorld-V1. Checking our [report](https://arxiv.org/abs/2507.21033) for more details. Welcome to **watch** 👀 this repository for the latest updates.



<!-- # 😮 Highlights

### 1. All Resources Fully Open-Sourced
- We fully open-source the models, data, training and evaluation code to facilitate rapid community exploration of unified architectures. 

### 2. Image Priors via VLM Encoding Without Learnable Tokens
- We find that multimodal features encoded by VLMs can interpret instructions while retaining image priors. Due to causal attention, the format `<image> <instruction>` is particularly important. -->



# 🔥 Quick Start
1.Set up environment

```bash
git clone https://github.com/wyhlovecpp/GPT-Image-Edit.git
cd GPT-Image-Edit
conda create -n univa python=3.10 -y
conda activate univa
pip install -r requirements.txt
pip install flash_attn --no-build-isolation
```

2.Download pretrained checkpoint
```bash
MODEL_PATH="model"
huggingface-cli download --resume-download UCSC-VLAA/gpt-image-edit-training --local-dir ${MODEL_PATH}
huggingface-cli download --resume-download black-forest-labs/FLUX.1-Kontext-dev --local-dir ${FLUX_PATH}
```

3.Run with CLI
```bash
MODEL_PATH="path/to/model/final_checkpoint"
FLUX_PATH="path/to/flux"
CUDA_VISIBLE_DEVICES=0 python -m univa.serve.cli \
    --model_path ${MODEL_PATH} \
    --flux_path ${FLUX_PATH}
```

4.Run with Gradio Web Interface
Highly recommend trying out our web demo by the following command.
```bash
python app.py --model_path ${MODEL_PATH} --flux_path ${FLUX_PATH}
```

5.Run with Gradio Serve
```bash
CUDA_VISIBLE_DEVICES=0 python -m univa.serve.gradio_web_server \
    --model_path ${MODEL_PATH} \
    --flux_path ${FLUX_PATH}
```


# 🗝️ Training

### Data preparation

Download the data from [UCSC-VLAA/GPT-Image-Edit-1.5M](https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M). The dataset consists of two parts: source images and annotation metadata. The json files is under [UCSC-VLAA/gpt-image-edit-training/training_json](https://huggingface.co/UCSC-VLAA/gpt-image-edit-training/tree/main/training_json)

Prepare a `data.txt` file in the following format:

1. The first column is the root path to the image.

2. The second column is the corresponding annotation JSON file.

3. The third column indicates whether to enable the region-weighting strategy. We use False in our training setting.

We have prepared a `data.txt` file about gpt-edit for your reference.
<details><summary>`data.txt` for gpt-edit</summary><p>
    
```
data/gpt-edit/hqedit/edit,training_json/hqedit_gpt_edit.json,false
data/gpt-edit/hqedit/generate,training_json/hqedit_gpt_generate.json,false
data/gpt-edit/omniedit,training_json/omniedit_gpt.json,false
data/gpt-edit/omniedit,training_json/omniedit_gpt_rewrite.json,false
data/gpt-edit/omniedit/complex-edit,training_json/complexedit_gpt.json,false
data/gpt-edit/ultraedit,training_json/ultraedit_gpt.json,false
```

</p></details>



### Data details

<details><summary>Image Editing</summary><p>
    
- [UCSC-VLAA/GPT-Image-Edit-1.5M](https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M) [5T storage usage.]

</p></details>


### Training

#### Prepare pretrained weights
Download [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) to `$FLUX_PATH`.
Download [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) to `$QWENVL_PATH`. We also support other sizes of Qwen2.5-VL.

```bash
SAVE_PATH="path/to/save/UniWorld-Qwen2.5-VL-7B-Instruct-FLUX.1-dev-fp32"
python scripts/make_univa_qwen2p5vl_weight.py \
    --origin_flux_ckpt_path $FLUX_PATH \
    --origin_qwenvl_ckpt_path $QWENVL_PATH \
    --save_path ${SAVE_PATH}
```

#### Stage 1

You need to specify `pretrained_lvlm_name_or_path` to `${SAVE_PATH}` in `flux_qwen2p5vl_7b_vlm_stage1_512.yaml`.

We recommend using pretrained weight from [LanguageBind/UniWorld-V1/stage1](https://huggingface.co/LanguageBind/UniWorld-V1/tree/main/stage1).

```bash
# stage 1
# if use prodigy, pip install prodigy
bash scripts/denoiser/flux_qwen2p5vl_7b_vlm_stage1_512.sh
```

#### Stage 2

You need to specify `pretrained_mlp2_path`, which is trained by stage 1 or use the pretrained weight from [LanguageBind/UniWorld-V1/stage1](https://huggingface.co/LanguageBind/UniWorld-V1/tree/main/stage1).
 
For training with 512×512 scale images (batch size 1), it consume about **78G** in 1 node (8 GPUs). 

Setting `ema_pretrained_lvlm_name_or_path: null` can saving memory if you want to train the higher resolution (e.g, 1024×1024 scale) or larger batch size. Using more nodes also can save memory because we use zero2 for main model in stage 2.

```bash
# stage 2
bash scripts/denoiser/flux_qwen2p5vl_7b_vlm_stage2_1024.sh
```

# ⚡️ Evaluation

### Image Editing

<details><summary>ImgEdit</summary><p>

```bash
cd univa/eval/imgedit
# follow the instruction in univa/eval/imgedit/README.md
```

</p></details>

<details><summary>GEdit</summary><p>


```bash
cd univa/eval/gdit
# follow the instruction in univa/eval/gdit/README.md
```

</p></details>

<details><summary>complex-edit</summary><p>


```bash
cd univa/eval/complex-edit
# follow the instruction in univa/eval/complex-edit/README.md
```

</p></details>


<details><summary>OmniContext</summary><p>


```bash
cd univa/eval/omnicontext
# follow the instruction in univa/eval/omnicontext/README.md
```

</p></details>



# 📊 Benchmarks

### GEdit-EN-full
| Model | BG<br>Change | Color<br>Alt. | Mat.<br>Mod. | Motion | Portrait | Style | Add | Remove | Replace | Text | Tone | Avg |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| *Open-Sourced Models* |||||||||||||
| AnyEdit | 4.31 | 4.25 | 2.64 | 0.67 | 1.90 | 1.95 | 3.72 | 3.75 | 3.23 | 0.77 | 4.21 | 2.85 |
| MagicBrush | 6.17 | 5.41 | 4.75 | 1.55 | 2.90 | 4.10 | 5.53 | 4.13 | 5.10 | 1.33 | 5.07 | 4.19 |
| Instruct-Pix2Pix | 3.94 | 5.40 | 3.52 | 1.27 | 2.62 | 4.39 | 3.07 | 1.50 | 3.48 | 1.13 | 5.10 | 3.22 |
| OmniGen | 5.23 | 5.93 | 5.44 | 3.12 | 3.17 | 4.88 | 6.33 | 6.35 | 5.34 | 4.31 | 4.96 | 5.01 |
| Step1X-Edit | 7.03 | 6.26 | 6.46 | 3.66 | 5.23 | 7.24 | 7.17 | 6.42 | 7.39 | 7.40 | 6.62 | 6.44 |
| Bagel | 7.44 | 6.99 | 6.26 | 5.09 | 4.82 | 6.04 | 7.94 | 7.37 | 7.31 | 7.16 | 6.17 | 6.60 |
| Bagel-thinking | 7.22 | 7.24 | 6.69 | 7.12 | 6.03 | 6.17 | 7.93 | 7.44 | 7.45 | 3.61 | 6.36 | 6.66 |
| Ovis-U1 | 7.49 | 6.88 | 6.21 | 4.79 | 5.98 | 6.46 | 7.49 | 7.25 | 7.27 | 4.48 | 6.31 | 6.42 |
| OmniGen2 | - | - | - | - | - | - | - | - | - | - | - | 6.42 |
| Step1X-Edit (v1.1) | 7.45 | 7.38 | 6.95 | 4.73 | 4.70 | 7.11 | 8.20 | 7.59 | 7.80 | 7.91 | 6.85 | 6.97 |
| FluxKontext dev | 7.06 | 7.03 | 5.52 | 5.62 | 4.68 | 5.55 | 6.95 | 6.76 | 6.13 | 6.10 | 7.48 | 6.26 |
| *Proprietary Models* |||||||||||||
| Gemini | 7.11 | 7.14 | 6.47 | 5.67 | 3.99 | 4.95 | 8.12 | 6.89 | 7.41 | 6.85 | 7.01 | 6.51 |
| Doubao | 8.07 | 7.36 | 7.20 | 5.38 | 6.28 | 7.20 | 8.05 | 7.71 | 7.87 | 4.01 | 7.67 | 6.98 |
| GPT-4o | 6.96 | 6.85 | 7.10 | 5.41 | 6.74 | 7.44 | 7.51 | 8.73 | 8.55 | 8.45 | 8.69 | 7.49 |
| **Ours** | **7.80** | **7.54** | **7.12** | **7.75** | **7.09** | **6.74** | **8.04** | **7.95** | **7.17** | **5.45** | **6.95** | **7.24** |

### Complex-Edit
| Method | IF | IP | PQ | Overall |
|:--|:--:|:--:|:--:|:--:|
| AnyEdit | 1.60 | 8.15 | 7.25 | 5.67 |
| UltraEdit | 6.56 | 5.93 | 7.29 | 6.59 |
| OmniGen | 6.25 | 6.42 | 7.54 | 6.74 |
| FluxKontext Dev | 8.56 | 8.39 | 8.51 | 8.49 |
| Imagen3 | 7.56 | 6.55 | 7.67 | 7.26 |
| SeedEdit | 8.49 | 6.91 | 8.74 | 8.04 |
| GPT-4o | 9.29 | 7.51 | 9.47 | 8.76 |
| **Ours** | **8.99** | **8.41** | **8.93** | **8.78** |

### ImgEdit-Full
| Model | Add | Adjust | Extract | Replace | Remove | Background | Style | Hybrid | Action | Overall |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| MagicBrush | 2.84 | 1.58 | 1.51 | 1.97 | 1.58 | 1.75 | 2.38 | 1.62 | 1.22 | 1.90 |
| Instruct-Pix2Pix | 2.45 | 1.83 | 1.44 | 2.01 | 1.50 | 1.44 | 3.55 | 1.20 | 1.46 | 1.88 |
| AnyEdit | 3.18 | 2.95 | 1.88 | 2.47 | 2.23 | 2.24 | 2.85 | 1.56 | 2.65 | 2.45 |
| UltraEdit | 3.44 | 2.81 | 2.13 | 2.96 | 1.45 | 2.83 | 3.76 | 1.91 | 2.98 | 2.70 |
| OmniGen | 3.47 | 3.04 | 1.71 | 2.94 | 2.43 | 3.21 | 4.19 | 2.24 | 3.38 | 2.96 |
| Step1X-Edit | 3.88 | 3.14 | 1.76 | 3.40 | 2.41 | 3.16 | 4.63 | 2.64 | 2.52 | 3.06 |
| ICEdit | 3.58 | 3.39 | 1.73 | 3.15 | 2.93 | 3.08 | 3.84 | 2.04 | 3.68 | 3.05 |
| BAGEL | 3.56 | 3.31 | 1.70 | 3.30 | 2.62 | 3.24 | 4.49 | 2.38 | 4.17 | 3.20 |
| UniWorld-V1 | 3.82 | 3.64 | 2.27 | 3.47 | 3.24 | 2.99 | 4.21 | 2.96 | 2.74 | 3.26 |
| OmniGen2 | 3.57 | 3.06 | 1.77 | 3.74 | 3.20 | 3.57 | 4.81 | 2.52 | 4.68 | 3.44 |
| Ovis-U1 | 4.13 | 3.62 | 2.98 | 4.45 | 4.06 | 4.22 | 4.69 | 3.45 | 4.61 | 4.00 |
| FluxKontext dev | 3.76 | 3.45 | 2.15 | 3.98 | 2.94 | 3.78 | 4.38 | 2.96 | 4.26 | 3.52 |
| GPT-4o | 4.61 | 4.33 | 2.90 | 4.35 | 3.66 | 4.57 | 4.93 | 3.96 | 4.89 | 4.20 |
| **Ours** | **4.07** | **3.79** | **2.04** | **4.13** | **3.89** | **3.90** | **4.84** | **3.04** | **4.52** | **3.80** |

# 👍 Acknowledgement and Related Work
* [UniWorld-V1](https://github.com/PKU-YuanGroup/UniWorld-V1): UniWorld-V1 is a unified framework for understanding, generation, and editing.
* [ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit): ImgEdit is a large-scale, high-quality image-editing dataset comprising 1.2 million carefully curated edit pairs and a comprehensive benchmark for image editing.
* [Complex-edit](https://github.com/UCSC-VLAA/Complex-Edit): Complex-edit is benchmark for complex image editing.
* [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct): The new flagship vision-language model of Qwen.
* [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev): A state-of-the-art image editing model.
* [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit): A state-of-the-art image editing model and a comprehensive benchmark for image editing.
* [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2): A state-of-the-art image editing model and a comprehensive benchmark for image editing.



# 🔒 License
* See [LICENSE](LICENSE) for details. The FLUX Kontext weights fall under the [FLUX.1 Kontext [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/blob/main/LICENSE.md).

# ✏️ Citing
```bibtex
@misc{wang2025gptimageedit15mmillionscalegptgeneratedimage,
      title={GPT-IMAGE-EDIT-1.5M: A Million-Scale, GPT-Generated Image Dataset}, 
      author={Yuhan Wang and Siwei Yang and Bingchen Zhao and Letian Zhang and Qing Liu and Yuyin Zhou and Cihang Xie},
      year={2025},
      eprint={2507.21033},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.21033}, 
}
```
```bibtex
@article{lin2025uniworld,
  title={UniWorld: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation},
  author={Lin, Bin and Li, Zongjian and Cheng, Xinhua and Niu, Yuwei and Ye, Yang and He, Xianyi and Yuan, Shenghai and Yu, Wangbo and Wang, Shaodong and Ge, Yunyang and others},
  journal={arXiv preprint arXiv:2506.03147},
  year={2025}
}
```


