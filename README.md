# 📣 News

* **[2025.07.27]** 🤗 We release GPT-Image-Edit, a state-of-the-art image editing model with 1.5M high-quality editing samples. All [data](https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M), [models](https://huggingface.co/UCSC-VLAA/gpt-image-edit-training), training code and evaluation code are open-sourced. Our code is based on [UniWorld-V1](https://github.com/PKU-YuanGroup/UniWorld-V1), Thanks to the authors of UniWorld-V1.



<!-- # 😮 Highlights

### 1. All Resources Fully Open-Sourced
- We fully open-source the models, data, training and evaluation code to facilitate rapid community exploration of unified architectures. 

### 2. Image Priors via VLM Encoding Without Learnable Tokens
- We find that multimodal features encoded by VLMs can interpret instructions while retaining image priors. Due to causal attention, the format `<image> <instruction>` is particularly important. -->



# 🔥 Quick Start
1.Set up environment

```bash
# git clone https://github.com/PKU-YuanGroup/UniWorld-V1
cd GPT-Image-Edit
conda create -n univa python=3.10 -y
conda activate univa
pip install -r requirements.txt
pip install flash_attn --no-build-isolation
```

2.Download pretrained checkpoint
```bash
huggingface-cli download --resume-download UCSC-VLAA/gpt-image-edit-training --local-dir ${MODEL_PATH}
huggingface-cli download --resume-download black-forest-labs/FLUX.1-Kontext-dev --local-dir ${FLUX_PATH}
```

3.Run with CLI
```bash
MODEL_PATH="path/to/model"
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
    
- [UCSC-VLAA/GPT-Image-Edit-1.5M](https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M)[5T storage usage.]

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



<!-- # 📊 Benchmarks -->




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
@article{lin2025uniworld,
  title={UniWorld: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation},
  author={Lin, Bin and Li, Zongjian and Cheng, Xinhua and Niu, Yuwei and Ye, Yang and He, Xianyi and Yuan, Shenghai and Yu, Wangbo and Wang, Shaodong and Ge, Yunyang and others},
  journal={arXiv preprint arXiv:2506.03147},
  year={2025}
}
```


