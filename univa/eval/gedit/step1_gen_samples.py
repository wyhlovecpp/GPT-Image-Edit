
import sys
import os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(root)
import json
import torch
import random
import subprocess
import numpy as np
import torch.distributed as dist
import pandas as pd
import argparse
import torch
import os
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
from qwen_vl_utils import process_vision_info
from torchvision import transforms
from transformers import AutoProcessor
from univa.utils.flux_pipeline import FluxKontextPipeline
from univa.eval.configuration_eval import EvalConfig
from univa.utils.get_ocr import get_ocr_result
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration
from univa.utils.anyres_util import dynamic_resize
from univa.utils.anyres_util import pick_ratio, compute_size
# Import Step1X tokenizer and dataset helpers
from univa.dataset.qwen2vl_dataset import Step1XTokenizer, Qwen2VLDataset

# adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/random.py#L31
def set_seed(seed, rank, device_specific=True):
    if device_specific:
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_models(args, device):

    # Load main model and task head
    model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        args.pretrained_lvlm_name_or_path,
        torch_dtype=torch.bfloat16
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        args.pretrained_lvlm_name_or_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    # Load FluxKontext pipeline
    pipe = FluxKontextPipeline.from_pretrained(
        args.pretrained_denoiser_name_or_path,
        transformer=model.denoise_tower.denoiser,
        torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]

    # Initialize Step1X tokenizer
    image_token = '<|image_pad|>'  # Adjust based on your model's image token
    step1x_tokenizer = Step1XTokenizer(processor.tokenizer, image_token=image_token)

    return {
        'model': model,
        'processor': processor,
        'pipe': pipe,
        'tokenizers': tokenizers,
        'text_encoders': text_encoders,
        'device': device,
        'step1x_tokenizer': step1x_tokenizer,
    }


def init_gpu_env(args):
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    args.local_rank = local_rank
    args.world_size = world_size
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl', init_method='env://', 
        world_size=world_size, rank=local_rank
        )
    return args


def run_model_and_return_samples(args, state, prompt_text, image1=None, image2=None):
    # Build conversation content expected by Qwen-VL processor
    content = []
    image_paths = []
    orig = Image.open(image1)
    ow, oh = orig.size
    
    rw, rh = pick_ratio(oh, ow, anyres='any_17ratio')
    vis_h , vis_w = 448, 448
    # vis_h, vis_w  = compute_size(
    #     rw, rh,
    #     stride=28,
    #     min_pixels=args.min_pixels,
    #     max_pixels=args.max_pixels
    # )
    # gen_h, gen_w = oh, ow
    gen_h, gen_w = compute_size(
        rw, rh,
        stride=16,
        anchor_pixels=args.height * args.width
    )

    print(f"vis_h: {vis_h}, vis_w: {vis_w}, gen_h: {gen_h}, gen_w: {gen_w}")
    for img in (image1, image2):
        if img:         
            content.append(
                {
                    "type": "image",
                    "image": img,
                    "resized_height": vis_h,
                    "resized_width":  vis_w,
                }
            )
            image_paths.append(img)
    if prompt_text:
        content.append({"type": "text", "text": prompt_text})

    convo = [{"role": "user", "content": content}]

    # Prepare text tokens
    chat_text = state["processor"].apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )
    chat_text = "<|im_end|>\n".join(chat_text.split("<|im_end|>\n")[1:])

    # Extract vision features
    image_inputs, video_inputs = process_vision_info(convo)

    inputs = state["processor"](
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(state["device"])

    # ===== LVLM forward =====
    with torch.no_grad():
        lvlm_embeds = state["model"](
            inputs.input_ids,
            pixel_values=getattr(inputs, "pixel_values", None),
            attention_mask=inputs.attention_mask,
            image_grid_thw=getattr(inputs, "image_grid_thw", None),
            output_type="denoise_embeds",
        )

        prm_embeds, pooled = encode_prompt(
            state["text_encoders"],
            state["tokenizers"],
            prompt_text if args.joint_with_t5 else "",
            256,
            state["device"],
            1,
        )

    # Assemble final prompt embeddings
    if args.only_use_t5:
        prompt_embeds = prm_embeds
    else:
        prompt_embeds = (
            torch.cat([lvlm_embeds, prm_embeds], dim=1)
            if args.joint_with_t5
            else lvlm_embeds
        )

    # ===== Build conditioning image batch =====
    condition_pixel_values = None
    if image_paths:
        cond_imgs = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            img_t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
            img_t = img_t.permute(2, 0, 1)  # C H W
            img_t = (img_t - 0.5) / 0.5  # [-1,1]
            cond_imgs.append(img_t)
        condition_pixel_values = torch.stack(cond_imgs).to(state["device"], dtype=torch.float32)

    # ===== Diffusion generation =====
    with torch.no_grad():
        images = state["pipe"](
            image=condition_pixel_values,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled,
            height=gen_h,
            width=gen_w,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt
        ).images

    return images
    

def main(args):

    args = init_gpu_env(args)

    torch.backends.cuda.matmul.allow_tf32 = False 
    torch.backends.cudnn.allow_tf32 = False
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(args.seed, rank=args.local_rank, device_specific=True)
    device = torch.cuda.current_device()
    state = initialize_models(args, device)

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the evaluation prompts
    with open(args.gedit_prompt_path, "r") as f:
        data = json.load(f)

    inference_list = []
    
    for key, value in tqdm(data.items()):
        outpath = args.output_dir
        os.makedirs(outpath, exist_ok=True)

        prompt = value["prompt"]
        image_path = value['id']
        inference_list.append([prompt, outpath, key, image_path])
            
    inference_list = inference_list[args.local_rank::args.world_size]
    
    for prompt, output_path, key, image_path in tqdm(inference_list):

        output_path = os.path.join(output_path, image_path)
        real_image_path = os.path.join(args.gedit_image_dir, image_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path):
            continue
        image = run_model_and_return_samples(args, state, prompt, image1=real_image_path, image2=None)
        image = image[0]
        image.save(
            output_path
        )


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--pretrained_lvlm_name_or_path", type=str, default=None, required=False)
    parser.add_argument("--output_dir", type=str, default=None, required=False)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    schema = OmegaConf.structured(EvalConfig)
    conf = OmegaConf.merge(schema, config)
    if args.pretrained_lvlm_name_or_path is not None:
        assert args.output_dir is not None
        conf.pretrained_lvlm_name_or_path = args.pretrained_lvlm_name_or_path
        conf.output_dir = args.output_dir
    main(conf)
