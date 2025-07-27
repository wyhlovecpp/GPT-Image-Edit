import sys
sys.path.append("..")
from transformers import AutoTokenizer, AutoProcessor 
from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration
# Remove SigLIP imports since it's no longer used
# from transformers import SiglipImageProcessor, SiglipVisionModel
from univa.utils.flux_pipeline import FluxKontextPipeline  # Changed to FluxKontextPipeline
from univa.utils.get_ocr import get_ocr_result
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
from qwen_vl_utils import process_vision_info
from univa.utils.anyres_util import dynamic_resize
import torch
from PIL import Image
from transformers import set_seed
from torch import nn
import os
import argparse
import numpy as np  # Added for image processing

seed = 42
set_seed(seed) 

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

generate_image_temp = './generate_image_{}.png'

def load_main_model_and_processor(
    model_path, 
    device, 
    min_pixels=448*448,
    max_pixels=448*448
):
    # Load model and processor
    model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    task_head = nn.Sequential(
        nn.Linear(3584, 10240),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(10240, 2)
    ).to(device)
    task_head.load_state_dict(torch.load(os.path.join(model_path, 'task_head_final.pt')))
    task_head.eval()

    processor = AutoProcessor.from_pretrained(
        model_path, 
        min_pixels=min_pixels, max_pixels=max_pixels
        )
    return model, task_head, processor


def load_pipe(
    denoiser, 
    flux_path, 
    device, 
):
    # Changed to FluxKontextPipeline
    pipe = FluxKontextPipeline.from_pretrained(
        flux_path,
        transformer=denoiser,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [
        pipe.text_encoder,
        pipe.text_encoder_2,
    ]

    return pipe, tokenizers, text_encoders


# Remove VAE conditioning functions since they're integrated into the model
# def load_vae_cond_encoder and preprocess_vae_conditioning functions removed

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

def prepare_condition_images(image_paths, device):
    """Prepare conditioning images for the pipeline"""
    if not image_paths:
        return None
        
    cond_imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img_t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        img_t = img_t.permute(2, 0, 1)  # C H W
        img_t = (img_t - 0.5) / 0.5  # [-1,1]
        cond_imgs.append(img_t)
    
    if cond_imgs:
        condition_pixel_values = torch.stack(cond_imgs).to(device, dtype=torch.float32)
        return condition_pixel_values
    
    return None
    
def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, task_head, processor = load_main_model_and_processor(
        args.model_path, 
        device,  
    )

    pipe, tokenizers, text_encoders = load_pipe(
        model.denoise_tower.denoiser, args.flux_path, device
        )
    
    # VAE conditioning is now integrated into the model - no separate encoder needed

    # Conversation history
    cur_ocr_i = 0
    cur_genimg_i = 0
    history_image_paths = []
    conversation = [
        # {"role": "system", "content": "You are a helpful assistant."},
        ]  # list of message dicts: {"role": "system"/"user"/"assistant", "content": [{...}]}

    print("Interactive UniWorld-V1 Chat (Exit if input is empty)")
    while True:
        # Prompt for optional text input
        txt = input("Text prompt (or press Enter to skip): ").strip()
        # Prompt for multiple image URLs (comma-separated)
        img_input = input("Image URLs (comma-separated, or press Enter to skip): ").strip()

        # Exit if no input provided
        if not img_input and not txt:
            print("Exit.")
            break

        # Build message content list
        content = []
        if txt:
            ocr_sentences = ''
            if args.ocr_enhancer and img_input:  # Fixed condition
                urls = [u.strip() for u in img_input.split(',') if u.strip()]
                num_img = len(urls)
                ocr_sentences = []
                for i in range(num_img):
                    ocr_sentences.append(get_ocr_result(urls[i], cur_ocr_i))
                    cur_ocr_i += 1
                ocr_sentences = '\n'.join(ocr_sentences)
            txt = txt + ocr_sentences
            content.append({"type": "text", "text": txt})


        new_h, new_w = args.height, args.width
        if img_input:
            urls = [u.strip() for u in img_input.split(',') if u.strip()]
            for url in urls:
                content.append({"type": "image", "image": url, "min_pixels": 448*448, "max_pixels": 448*448})
                history_image_paths.append(url)
        
            new_h, new_w = update_size(
                urls[0] if len(urls) > 0 else None, urls[1] if len(urls) > 1 else None, 
                'any_11ratio', anchor_pixels=args.height * args.width
                )


        conversation.append({"role": "user", "content": content})
        print('conversation:\n', conversation)

        # Prepare inputs for model
        chat_text = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        chat_text = '<|im_end|>\n'.join(chat_text.split('<|im_end|>\n')[1:])  # drop system
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        # Generate response
        with torch.inference_mode():
            outputs = model(**inputs, return_dict=True, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # B L D
        assistant_mask = inputs.input_ids == 77091
        assistant_vectors = hidden_states[assistant_mask][-1:]
        task_result = task_head(assistant_vectors.float())[0]

        if task_result[0] < task_result[1]:
        # if task_result > 0.5:
            # gen
            # VAE conditioning is now handled internally by the model
            with torch.no_grad():
                lvlm_embeds = model(
                    inputs.input_ids,
                    pixel_values=getattr(inputs, 'pixel_values', None),
                    attention_mask=inputs.attention_mask, 
                    image_grid_thw=getattr(inputs, 'image_grid_thw', None),
                    # VAE conditioning happens internally in the model
                    output_type="denoise_embeds",
                )
            assert lvlm_embeds.shape[0] == 1
            
            t5_prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders,
                tokenizers,
                txt if not args.no_joint_with_t5 else '',
                256,
                device,
                1,
            )
            
            # Assemble final prompt embeddings
            if not args.no_joint_with_t5:
                prompt_embeds = torch.concat([lvlm_embeds, t5_prompt_embeds], dim=1)
            else:
                prompt_embeds = lvlm_embeds

            # Prepare conditioning images
            condition_pixel_values = prepare_condition_images(history_image_paths, device)

            output_image = pipe(
                image=condition_pixel_values,  # Changed parameter name
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                height=new_h,
                width=new_w,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, 
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).images[0]
            img_url = generate_image_temp.format(cur_genimg_i)
            cur_genimg_i += 1
            output_image.save(img_url)
            conversation.append({"role": "assistant", "content": [{"type": "image", "image": img_url}]})
            history_image_paths.append(img_url)
            print(f"Assistant: generate image at {img_url}\n")

        else:
            # und
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            # Decode only newly generated tokens
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            reply = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            print(f"Assistant: {reply}\n")

            # Append assistant response to history
            conversation.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Model and component paths")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--flux_path", type=str, required=True)
    # Removed SigLIP parameter since it's no longer used
    # parser.add_argument("--siglip_path", type=str, required=True)
    parser.add_argument("--no_auto_hw", action="store_true")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--ocr_enhancer", action='store_true')
    parser.add_argument("--no_joint_with_t5", action="store_true")

    args = parser.parse_args()
    main(args)
