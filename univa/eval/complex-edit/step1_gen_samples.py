import os
import sys
import json
import argparse
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

import torch
import torch.distributed as dist

# Project root
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(root)

# === Univa imports ===
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from univa.utils.flux_pipeline import FluxKontextPipeline
from univa.eval.configuration_eval import EvalConfig
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration
from univa.utils.anyres_util import dynamic_resize, pick_ratio, compute_size

# Step1X helpers
from univa.dataset.qwen2vl_dataset import Step1XTokenizer

# -------------------- utils --------------------

def set_seed(seed: int, rank: int = 0):
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_gpu_env(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    args.local_rank = local_rank
    args.world_size = world_size
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=local_rank
        )
    return args


# -------------------- model init --------------------

def initialize_models(args, device):
    # Main LVLM
    model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        args.pretrained_lvlm_name_or_path, torch_dtype=torch.bfloat16
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        args.pretrained_lvlm_name_or_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        padding_side=getattr(args, "padding_side", "right"),
    )

    # FluxKontext diffusion pipeline (share denoiser weights)
    pipe = FluxKontextPipeline.from_pretrained(
        args.pretrained_denoiser_name_or_path,
        transformer=model.denoise_tower.denoiser,
        torch_dtype=torch.bfloat16,
    ).to(device)

    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]

    image_token = "<|image_pad|>"
    step1x_tokenizer = Step1XTokenizer(processor.tokenizer, image_token=image_token)

    return {
        "model": model,
        "processor": processor,
        "pipe": pipe,
        "tokenizers": tokenizers,
        "text_encoders": text_encoders,
        "device": device,
        "step1x_tokenizer": step1x_tokenizer,
    }


# -------------------- core inference --------------------

def run_model_and_return_samples(args, state, prompt_text, pil_image, idx):
    """Perform conditional generation given an editing prompt and source image."""

    # 
    # Vision preprocessing sizes
    ow, oh = pil_image.size
    rw, rh = pick_ratio(oh, ow, anyres="any_17ratio")
    vis_h, vis_w = 448, 448
    # vis_h, vis_w = compute_size(
    #     rw, rh, stride=28, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    # )
    gen_h, gen_w = compute_size(rw, rh, stride=16, anchor_pixels=args.height * args.width)

    # Create a unique temporary path per sample to avoid cross-process clobbering
    tmp_dir = os.path.join(args.output_dir, "_tmp_imgs")
    os.makedirs(tmp_dir, exist_ok=True)
    image_path_placeholder = os.path.join(tmp_dir, f"rank{args.local_rank}_{idx}.png")
    pil_image.save(image_path_placeholder)

    content = [
        {
            "type": "image",
            "image": image_path_placeholder,
            "resized_height": vis_h,
            "resized_width": vis_w,
        },
        {"type": "text", "text": prompt_text},
    ]
    convo = [{"role": "user", "content": content}]

    # Prepare inputs
    chat_text = state["processor"].apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )
    chat_text = "<|im_end|>\n".join(chat_text.split("<|im_end|>\n")[1:])

    image_inputs, video_inputs = process_vision_info(convo)

    inputs = state["processor"](
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(state["device"])

    # LVLM forward to obtain prompt embeds
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

    prompt_embeds = (
        torch.cat([lvlm_embeds, prm_embeds], dim=1)
        if (not args.only_use_t5 and args.joint_with_t5)
        else lvlm_embeds
        if not args.only_use_t5
        else prm_embeds
    )

    # Build conditioning tensor (normalize to [-1,1])
    img_np = np.array(pil_image.convert("RGB"), dtype=np.float32) / 255.0
    img_t = torch.tensor(img_np).permute(2, 0, 1)  # C H W
    img_t = (img_t - 0.5) / 0.5
    cond_pixel_values = img_t.unsqueeze(0).to(state["device"], dtype=torch.float32)

    # Diffusion generation
    with torch.no_grad():
        images = state["pipe"](
            image=cond_pixel_values,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled,
            height=gen_h,
            width=gen_w,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt,
        ).images

    try:
        os.remove(image_path_placeholder)
    except OSError:
        pass

    return images


# -------------------- main --------------------

def main(args):
    args = init_gpu_env(args)

    # TF32 handling
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(args.seed, rank=args.local_rank)
    device = torch.cuda.current_device()
    state = initialize_models(args, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Complex-Edit dataset
    dataset = load_dataset("UCSC-VLAA/Complex-Edit")
    data_split = f"test_{args.image_type}"
    if data_split not in dataset:
        raise ValueError(f"Split {data_split} not in dataset")
    data = dataset[data_split]

    # Build task list
    prompts = [
        edit["compound"][args.complexity - 1]["compound_instruction"] for edit in data["edit"]
    ]
    images = data["image"]
    task_list = list(zip(images, prompts, range(len(images))))

    task_list = task_list[args.local_rank :: args.world_size]

    for input_img, prompt_text, idx in tqdm(task_list, desc=f"Rank {args.local_rank}"):
        out_path = os.path.join(args.output_dir, f"{idx:05}.png")
        if os.path.exists(out_path):
            continue
        try:
            # Resize to anchor resolution for processing convenience
            if input_img.size != (args.width, args.height):
                input_img = input_img.resize((args.width, args.height), Image.Resampling.LANCZOS)
            gen_imgs = run_model_and_return_samples(args, state, prompt_text, input_img, idx)
            gen_imgs[0].save(out_path)
        except Exception as e:
            print(f"Error on {idx}: {e}")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Complex-Edit adaptive sampling")
    parser.add_argument("config", type=str)
    parser.add_argument("--pretrained_lvlm_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    cli_args = parser.parse_args()

    cfg = OmegaConf.load(cli_args.config)

    # Allow additional keys (e.g., complexity, image_type) not defined in EvalConfig
    schema = OmegaConf.structured(EvalConfig)
    from omegaconf import OmegaConf as _OC
    _OC.set_struct(schema, False)

    conf = _OC.merge(schema, cfg)

    if cli_args.pretrained_lvlm_name_or_path is not None:
        assert cli_args.output_dir is not None
        conf.pretrained_lvlm_name_or_path = cli_args.pretrained_lvlm_name_or_path
        conf.output_dir = cli_args.output_dir

    main(conf) 