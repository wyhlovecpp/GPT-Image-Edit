import dotenv

dotenv.load_dotenv(override=True)

import sys
import os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(root)

import argparse
import json
import torch
import random
import numpy as np
import torch.distributed as dist
import datasets
from tqdm import tqdm
from typing import List, Tuple
from torch.utils.data import DataLoader
from PIL import Image, ImageOps

from torchvision.transforms.functional import to_pil_image, to_tensor
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from univa.utils.flux_pipeline import FluxKontextPipeline
from univa.eval.configuration_eval import EvalConfig
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration
from univa.utils.anyres_util import dynamic_resize, pick_ratio, compute_size
from univa.dataset.qwen2vl_dataset import Step1XTokenizer, Qwen2VLDataset
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Univa image generation script.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--pretrained_lvlm_name_or_path",
        type=str,
        default=None,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="univa_flux_omnicontext",
        help="Model name for output directory.",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Path to save the generated images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    return parser.parse_args()


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


def init_gpu_env():
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend='nccl', init_method='env://', 
            world_size=world_size, rank=local_rank
        )
    return local_rank, world_size


def initialize_models(args, device):
    """Initialize the models and processors."""
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


class Collator:
    def __call__(self, features):
        return features


def run_model_and_return_samples(args, state, prompt_text, input_images):
    """Run the model and return generated samples."""
    # Build conversation content expected by Qwen-VL processor
    content = []
    image_paths = []
    
    # Get first image for size calculation
    first_image = input_images[0] if input_images else None
    if first_image:
        if isinstance(first_image, str):
            orig = Image.open(first_image)
        else:
            orig = first_image
        ow, oh = orig.size
        
        rw, rh = 448, 448
        vis_h, vis_w = compute_size(
            rw, rh,
            stride=28,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels
        )
        gen_h, gen_w = compute_size(
            rw, rh,
            stride=16,
            anchor_pixels=args.height * args.width
        )
    else:
        vis_h, vis_w = args.height, args.width
        gen_h, gen_w = args.height, args.width

    # Process input images
    for img in input_images:
        if isinstance(img, str):
            content.append(
                {
                    "type": "image",
                    "image": img,
                    "resized_height": vis_h,
                    "resized_width":  vis_w,
                }
            )
            image_paths.append(img)
        else:
            # Save PIL image to temporary path
            temp_path = f"/tmp/temp_image_{random.randint(1000, 9999)}.png"
            img.save(temp_path)
            content.append({
                "type": "image",
                "image": temp_path,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
            })
            image_paths.append(temp_path)

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

    # Clean up temporary files
    for p in image_paths:
        if p.startswith("/tmp/temp_image_"):
            try:
                os.remove(p)
            except:
                pass

    return images


def create_collage(images: List[torch.Tensor]) -> Image.Image:
    """Create a horizontal collage from a list of images."""
    max_height = max(img.shape[-2] for img in images)
    total_width = sum(img.shape[-1] for img in images)
    canvas = torch.zeros((3, max_height, total_width), device=images[0].device)
    
    current_x = 0
    for img in images:
        h, w = img.shape[-2:]
        canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
        current_x += w
    
    return to_pil_image(canvas)


def main(args: argparse.Namespace, root_dir: str) -> None:
    """Main function to run the image generation process."""
    # Load configuration
    config = OmegaConf.load(args.config)
    schema = OmegaConf.structured(EvalConfig)
    conf = OmegaConf.merge(schema, config)
    
    # Override with command line arguments
    if args.pretrained_lvlm_name_or_path is not None:
        conf.pretrained_lvlm_name_or_path = args.pretrained_lvlm_name_or_path
    if args.output_dir is not None:
        conf.output_dir = args.output_dir
    else:
        conf.output_dir = args.result_dir

    # Initialize GPU environment
    local_rank, world_size = init_gpu_env()
    
    # Configure backends
    torch.backends.cuda.matmul.allow_tf32 = False 
    torch.backends.cudnn.allow_tf32 = False
    if conf.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(conf.seed, rank=local_rank, device_specific=True)
    device = torch.cuda.current_device()

    # Initialize models
    state = initialize_models(conf, device)

    # Load test dataset
    test_dataset = datasets.load_dataset(args.test_data, split="train")
    print(f'test_dataset size: {len(test_dataset)}')

    # Filter dataset for only single_character and single_object tasks
    filtered_data = []
    for item in test_dataset:
        if item['task_type'] in ['single_character', 'single_object']:
            filtered_data.append(item)
    
    print(f'Filtered dataset size: {len(filtered_data)} (only single_character and single_object)')

    # Distribute work across processes
    filtered_data = filtered_data[local_rank::world_size]
    print(f'Process {local_rank} handling {len(filtered_data)} items')

    # Create output directory
    os.makedirs(conf.output_dir, exist_ok=True)

    # Process each item
    for data in tqdm(filtered_data, desc=f"Rank {local_rank} generating images", disable=local_rank != 0):
        key = data['key']
        task_type = data['task_type']
        instruction = data['instruction']
        input_images = data['input_images']
        
        # Preprocess images
        input_images = [ImageOps.exif_transpose(img) for img in input_images]

        # Create output directory for this task type
        sub_dir = os.path.join(conf.output_dir, args.model_name, "fullset", task_type)
        os.makedirs(sub_dir, exist_ok=True)
        output_image_path = os.path.join(sub_dir, f"{key}.png")
        
        # Skip if output already exists
        if os.path.exists(output_image_path):
            continue

        # Generate images
        try:
            results = run_model_and_return_samples(conf, state, instruction, input_images)
            
            # Save results
            if len(results) > 1:
                for i, image in enumerate(results):
                    image_name, ext = os.path.splitext(output_image_path)
                    image.save(f"{image_name}_{i}{ext}")
            
            # Create collage for visualization
            if len(results) > 1:
                vis_images = [to_tensor(image) * 2 - 1 for image in results]
                output_image = create_collage(vis_images)
                output_image.save(output_image_path)
            else:
                results[0].save(output_image_path)
                
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            continue

    print(f"Process {local_rank} completed!")


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args()
    main(args, root_dir)