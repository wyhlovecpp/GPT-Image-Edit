
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
from transformers import AutoProcessor, CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
from transformers import SiglipImageProcessor, SiglipVisionModel
from univa.utils.flux_pipeline import FluxPipeline
from univa.eval.configuration_eval import EvalConfig
from univa.utils.get_ocr import get_ocr_result
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration
from univa.utils.anyres_util import dynamic_resize

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

    # Load main model and task head - always needed for the transformer
    model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        args.pretrained_lvlm_name_or_path,
        torch_dtype=torch.bfloat16
    ).to(device)

    # Load processor only if we need qwenvl processing  
    processor = None
    if not args.only_use_t5:
        processor = AutoProcessor.from_pretrained(
            args.pretrained_lvlm_name_or_path,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )

    # Load FLUX pipeline with external text encoders
    pipe = FluxPipeline.from_pretrained(
        args.pretrained_denoiser_name_or_path,
        transformer=model.denoise_tower.denoiser,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    # Load text encoders separately
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_denoiser_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_denoiser_name_or_path,
        subfolder="tokenizer_2",
    )
    
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_denoiser_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    ).to(device)
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_denoiser_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    siglip_processor = None
    siglip_model = None
    if args.pretrained_siglip_name_or_path:
        siglip_processor = SiglipImageProcessor.from_pretrained(args.pretrained_siglip_name_or_path)
        siglip_model = SiglipVisionModel.from_pretrained(
            args.pretrained_siglip_name_or_path,
            torch_dtype=torch.bfloat16,
        ).to(device)

    return {
        'model': model,
        'processor': processor,
        'pipe': pipe,
        'tokenizers': tokenizers,
        'text_encoders': text_encoders,
        'device': device,
        'siglip_model': siglip_model,
        'siglip_processor': siglip_processor,
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

def update_size(i1, i2, anyres='any_11ratio', anchor_pixels=1024*1024):
    shapes = []
    for p in (i1, i2):
        if p:
            im = Image.open(p)
            w, h = im.size
            shapes.append((w, h))
    if not shapes:
        return int(anchor_pixels**0.5), int(anchor_pixels**0.5)
    if len(shapes) == 1:
        w, h = shapes[0]
    else:
        w = sum(s[0] for s in shapes) / len(shapes)
        h = sum(s[1] for s in shapes) / len(shapes)
    new_h, new_w = dynamic_resize(int(h), int(w), anyres, anchor_pixels=anchor_pixels)
    return new_h, new_w
    
def run_model_and_return_samples(args, state, text, image1=None, image2=None):
    
    new_h, new_w = update_size(image1, image2, 'any_11ratio', anchor_pixels=args.height * args.width)

    # Process condition image for VAE encoding (like training validation)
    pipeline_image = None
    if image1:
        # Load and preprocess condition image to pixel space [-1,1]
        from torchvision.transforms import ToTensor, Normalize, Compose, Resize
        cond_img = Image.open(image1).convert('RGB')
        transform = Compose([
            Resize((new_h, new_w)),
            ToTensor(),                    # [0,1]
            Normalize([0.5], [0.5])        # [-1,1]
        ])
        pipeline_image = transform(cond_img).unsqueeze(0).to(
            device=state['device'], dtype=state['pipe'].vae.dtype
        )

    # Generate T5 and CLIP embeddings
    with torch.no_grad():
        t5_prompt_embeds, pooled_prompt_embeds = encode_prompt(
            state['text_encoders'], 
            state['tokenizers'],
            text, 
            256, 
            state['device'], 
            1
        )

    prompt_embeds = None
    prompt_embeds = t5_prompt_embeds
    with torch.no_grad():
        img = state['pipe'](
            image=pipeline_image,  # Pass condition image
            prompt_embeds=prompt_embeds, 
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=new_h, 
            width=new_w,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt, 
        ).images
    return img
    

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
    with open(args.imgedit_prompt_path, "r") as f:
        data = json.load(f)

    inference_list = []
    
    for key, value in tqdm(data.items()):
        outpath = args.output_dir
        os.makedirs(outpath, exist_ok=True)

        prompt = value["prompt"]
        image_path = os.path.join(args.imgedit_image_dir, value["id"])
        inference_list.append([prompt, outpath, key, image_path])
            
    inference_list = inference_list[args.local_rank::args.world_size]
    
    for prompt, output_path, key, image_path in tqdm(inference_list):
        if os.path.exists(os.path.join(output_path, f"{key}.png")):
            continue
        image = run_model_and_return_samples(args, state, prompt, image1=image_path, image2=None)
        image = image[0]
        # image = image.resize((args.resized_width, args.resized_height))
        image.save(
            os.path.join(output_path, f"{key}.png")
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