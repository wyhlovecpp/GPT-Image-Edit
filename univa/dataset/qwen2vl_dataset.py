from typing import Any, Callable, Optional, List

import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import os
from PIL import Image
from univa.utils.prompter import Prompter
import numpy as np
from einops import rearrange
import random
# from qwen_vl_utils.vision_process import fetch_image, fetch_video
from qwen_vl_utils.vision_process import to_rgb, smart_resize, fetch_video
from univa.utils.constant import SPACIAL_TOKEN, GENERATE_TOKEN
from univa.utils.get_mask import get_weight_mask
from univa.utils.get_ocr import get_ocr_result
from fractions import Fraction
from torchvision.transforms import functional
from torchvision import transforms
from io import BytesIO
import base64
import requests
import torch
from PIL import Image
from torchvision import io, transforms
from typing import Optional
import re


class Step1XTokenizer:
    """
    Enhanced Step1X tokenization for better quote protection and image token handling.
    
    核心功能：
    1. 文本分段：识别引号内容和普通文本
    2. 字面量保护：对引号内容进行字符级保护
    3. 分段tokenize：避免截断，保护图像token
    """
    
    def __init__(self, tokenizer, image_token: str):
        self.tokenizer = tokenizer
        self.image_token = image_token
        # 使用一个在词汇表中存在但极少在普通文本中出现的token作为临时占位符
        self.placeholder_token = "<|endoftext|>"
        self.placeholder_token_id = self.tokenizer.convert_tokens_to_ids(self.placeholder_token)
        self.is_checkpoint_tokenizer = self._detect_checkpoint_tokenizer()
        self.failure_count = 0
        self.max_failures = 10
        
    def _detect_checkpoint_tokenizer(self) -> bool:
        """检测是否为checkpoint加载的tokenizer"""
        try:
            if hasattr(self.tokenizer, 'name_or_path'):
                path = str(self.tokenizer.name_or_path)
                return 'checkpoint' in path.lower()  # 只检测checkpoint，不硬编码UniWorld
            return False
        except:
            return False
    
    def _normalize_quotes(self, text: str) -> str:
        """标准化引号类型"""
        # 修复：正确处理中文/弯引号
        text = text.replace('"', '"').replace('"', '"')  # 左右弯双引号
        text = text.replace(''', "'").replace(''', "'")  # 左右弯单引号
        return text
    
    def _extract_literal_segments(self, text: str) -> List[tuple]:
        """
        Step1X核心：提取字面量段落
        
        返回: List[(segment_text, is_literal, quote_type)]
        """
        text = self._normalize_quotes(text)
        segments = []
        current_segment = ""
        in_literal = False
        quote_char = None
        
        i = 0
        while i < len(text):
            char = text[i]
            
            # 处理引号字符 (", ', `)
            if char in ['"', "'", '`'] and (quote_char is None or char == quote_char):
                if current_segment:
                    segments.append((current_segment, in_literal, quote_char))
                    current_segment = ""
                
                if not in_literal:
                    # 开始字面量
                    in_literal = True
                    quote_char = char
                    current_segment = char
                else:
                    # 结束字面量
                    current_segment += char
                    segments.append((current_segment, True, quote_char))
                    current_segment = ""
                    in_literal = False
                    quote_char = None
            else:
                current_segment += char
            
            i += 1
        
        # 添加剩余段落
        if current_segment:
            segments.append((current_segment, in_literal, quote_char))
        
        return segments
    
    def _protect_literal_content(self, text: str, quote_type: str) -> str:
        """
        Step1X字面量保护：给引号内每个字符加空格
        
        例如: "step1x" → " s t e p 1 x "
        """
        if len(text) <= 2:  # 只有引号
            return text
        
        # 提取引号内内容
        if text.startswith(quote_type) and text.endswith(quote_type):
            inner_text = text[1:-1]
        else:
            inner_text = text
        
        # 优化：单侧空格即可，避免过多空格
        protected = quote_type
        for i, char in enumerate(inner_text):
            if char.isspace():
                protected += char
            else:
                if i > 0:  # 非第一个字符前加空格
                    protected += " "
                protected += char
        protected += quote_type
        
        return protected
    

    
    def tokenize_with_protection(self, text: str, **kwargs) -> dict:
        """
        Step1X主函数：带保护的tokenization
        
        核心策略：
        1. 检测checkpoint tokenizer → 直接回退
        2. 无引号内容 → 标准tokenization  
        3. 有引号内容 → Step1X分段处理
        4. 失败 → 计数并回退
        """
        
        # 策略1: Checkpoint tokenizer直接回退
        if self.is_checkpoint_tokenizer:
            return self.tokenizer(text, **kwargs)
        
        # 策略2: 失败次数过多，自动禁用
        if self.failure_count >= self.max_failures:
            return self.tokenizer(text, **kwargs)
        
        # 策略3: 无引号内容，使用标准tokenization
        if '"' not in text and "'" not in text and '`' not in text:
            return self.tokenizer(text, **kwargs)

        try:
            # 修复：只传text，避免参数冲突
            return self._step1x_process(text, **kwargs)
        except Exception as e:
            self.failure_count += 1
            print(f"Warning: Step1X failed ({self.failure_count}/{self.max_failures}): {e}")
            if self.failure_count >= self.max_failures:
                print("Warning: Step1X disabled due to repeated failures")
            
            # 回退到标准tokenization
            return self.tokenizer(text, **kwargs)
    
    def _step1x_process(self, text: str, **kwargs) -> dict:
        """
        Step1X核心处理逻辑
        """
        # Step 0: 保护 image_token
        has_image_token = self.image_token in text
        if has_image_token:
            text = text.replace(self.image_token, self.placeholder_token)

        # Step 1: 文本分段
        segments = self._extract_literal_segments(text)
        
        # Step 2: 分段tokenize
        token_segments = []
        
        for segment_text, is_literal, quote_type in segments:
            if not segment_text.strip():
                continue

            # Step 2.1: 字面量保护
            if is_literal and quote_type:
                segment_text = self._protect_literal_content(segment_text, quote_type)
            
            # Step 2.2: 分段tokenize（修复：内部硬编码参数，避免冲突）
            try:
                # 使用 self.tokenizer 而不是直接调用 __call__
                segment_result = self.tokenizer(
                    text=segment_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                    truncation=False,
                )
                
                if segment_result.input_ids.shape[1] > 0:
                    token_segments.append(segment_result.input_ids)
                    
            except Exception as seg_e:
                print(f"Warning: Segment tokenization failed: {seg_e}")
                # 段落级回退：对这个段落使用标准tokenization
                try:
                    fallback_result = self.tokenizer(
                        segment_text,
                        add_special_tokens=False,
                        return_tensors="pt",
                        truncation=False,
                    )
                    if fallback_result.input_ids.shape[1] > 0:
                        token_segments.append(fallback_result.input_ids)
                except:
                    # 段落完全失败，跳过
                    continue
        
        # Step 3: 合并所有段落
        if not token_segments:
            # 如果没有有效的token，返回一个空的tokenization结果
             return self.tokenizer(
                "", return_tensors="pt", add_special_tokens=False
            )

        # 拼接所有token段落
        combined_tokens = torch.cat(token_segments, dim=1)

        # Step 4: 恢复 image_token
        if has_image_token:
            image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
            combined_tokens[combined_tokens == self.placeholder_token_id] = image_token_id
        
        # 返回标准格式
        # 使用 **kwargs 来传递额外的参数给最终的 BatchEncoding
        final_result = self.tokenizer.prepare_for_model(
            combined_tokens.squeeze(0).tolist(), 
            add_special_tokens=False, 
            return_tensors='pt', 
            **kwargs
        )

        return final_result

def get_aspect_ratio(img):
    width, height = img.size
    return Fraction(width, height).limit_denominator()

def has_same_aspect_ratio(img1, img2):
    if not isinstance(img1, Image.Image):
        img1 = Image.open(img1).convert('RGB')
    if not isinstance(img2, Image.Image):
        img2 = Image.open(img2).convert('RGB')
    ratio1 = get_aspect_ratio(img1)
    ratio2 = get_aspect_ratio(img2)
    return ratio1 == ratio2

def has_same_resolution(img1, img2):
    if not isinstance(img1, Image.Image):
        img1 = Image.open(img1).convert('RGB')
    if not isinstance(img2, Image.Image):
        img2 = Image.open(img2).convert('RGB')
    return img1.size == img2.size

class Qwen2VLDataset(Dataset):
    def __init__(
        self,
        dataset_type: str,
        data_txt: str,
        transform: Callable, 
        tokenizer: PreTrainedTokenizer,
        prompter: Prompter,
        image_processor: Callable,
        processor: Callable = None,
        min_pixels: int = 384*384, 
        max_pixels: int = 384*384, 
        image_token_length: int = 729,
        only_generated_task: bool = False,
        drop_prompt_rate: float = 0.0,
        joint_ref_feature: bool = False,
        anyres: bool = False, 
        mask_weight_type: str = 'log', 
        siglip_processor: Callable = None,
        ocr_enhancer: bool = False, 
        random_data: bool = False, 
        maxnum_per_data: int = -1, 
        notry: bool = False, 
        use_step1x_preprocessing: bool = True,  # 临时禁用Step1X避免训练问题
    ):
        assert dataset_type == 'qwen2vl' or dataset_type == 'qwen2p5vl', "dataset_type == 'qwen2vl' or dataset_type == 'qwen2p5vl'"
        with open(data_txt, "r") as f:
            self.datasets = [line.strip() for line in f.readlines()]

        self.data = []
        self._load_data(maxnum_per_data)
        
        self.transform = transform
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.prompter = prompter
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_token = SPACIAL_TOKEN[dataset_type]['image_token']
        self.image_begin_token = SPACIAL_TOKEN[dataset_type]['image_begin_token']
        self.image_end_token = SPACIAL_TOKEN[dataset_type]['image_end_token']
        self.generated_image_token = GENERATE_TOKEN
        self.image_processor = processor.image_processor
        # self.factor = 4 if joint_ref_feature else 1
        self.factor = 2

        self.only_generated_task = only_generated_task  # For denoiser training
        self.drop_prompt_rate = drop_prompt_rate
        if self.drop_prompt_rate > 0:
            assert self.only_generated_task, (
                "Only generated task is supported when drop_prompt_rate > 0"
            )
        self.mask_weight_type = mask_weight_type
        self.siglip_processor = siglip_processor
        self.ocr_enhancer = ocr_enhancer
        self.random_data = random_data
        self.notry = notry
        self.use_step1x_preprocessing = use_step1x_preprocessing

        # Initialize Step1X tokenizer if enabled
        if self.use_step1x_preprocessing:
            self.step1x_tokenizer = Step1XTokenizer(self.tokenizer, image_token=self.image_token)
        else:
            self.step1x_tokenizer = None

        # Add image token if not exists.
        assert self.image_token in self.tokenizer.get_vocab()
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        self.image_begin_token_id = self.tokenizer.convert_tokens_to_ids(
            self.image_begin_token
        )
        assert isinstance(self.image_begin_token_id, int), (
            f"tokenizer miss image begin token `{self.image_begin_token}`"
        )
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids(
            self.image_end_token
        )
        assert isinstance(self.image_end_token_id, int), (
            f"tokenizer miss image end token `{self.image_end_token}`"
        )

    def _load_data(self, maxnum_per_data=-1):
        for dataset in self.datasets:
            image_root, json_file, need_weight = dataset.split(",")

            # Load json file
            with open(json_file, "r") as f:
                data = json.load(f)
            if maxnum_per_data > 0 and maxnum_per_data < len(data):
                print(f'original data: {len(data)}, sample: {maxnum_per_data}')
                data = random.sample(data, maxnum_per_data)
            dataset_data = []
            for line in tqdm(data):
                if "image" not in line:
                    line["image"] = []
                # Ensure `image` is a list
                if isinstance(line["image"], str):
                    line["image"] = [line["image"]]
                assert isinstance(line["image"], list), (
                    "`image` must be a str or a list."
                )

                # Convert image path to absolute path
                line["need_weight"] = need_weight
                line["image"] = [
                    os.path.join(image_root, image_path) for image_path in line["image"]
                ]
                dataset_data.append(line)

            print(f"Load {len(dataset_data)} data from {json_file}.")
            self.data.extend(dataset_data)

    def __len__(self):
        return len(self.data)

    def _get_random_data(self, ):
        
        prompt = self.prompter(
            [
                {"from": "system", "value": "You are a helpful assistant."},
                {
                    "from": "user",
                    "value": f"test an image {self.image_token}",
                },
            ]
        )
        input_ids = self.tokenizer.batch_encode_plus(
            [prompt], return_tensors="pt", truncation=False,
        ).input_ids
        labels = input_ids

        width, height = 448, 448
        random_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(random_data, 'RGB')

        image_slice = [image]
        image_dict = self._load_image(
            image_slice, self.max_pixels, self.min_pixels, 
            processor=self.processor, image_token=self.image_token, 
            factor=self.factor, 
            last_image=image,
            vae_image_transform=self.transform, 
            drop_prompt=False, 
            prompt=prompt, 
            mask_weight_type=self.mask_weight_type, 
            siglip_processor=self.siglip_processor, 
            need_weight='true',
            )
        
        image_token_lengths = image_dict['image_token_lengths']
        pixel_values = image_dict['pixel_values']
        image_grid_thw = image_dict['image_grid_thw']
        ref_pixel_values = image_dict['ref_pixel_values']
        pil_pixel_values = image_dict['pil_pixel_values']
        siglip_pixel_values = image_dict['siglip_pixel_values']
        weights = image_dict['weights']

        input_ids, labels, image_position = self._process_image_token(
                input_ids,
                labels=labels,
                image_token_id=self.image_token_id,
                image_begin_token_id=self.image_begin_token_id,
                image_end_token_id=self.image_end_token_id,
                image_token_lengths=image_token_lengths, 
            )
        
        generated_image = torch.randn(3, 512, 512)
        
        return_data = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_position": image_position,
            "image_grid_thw": image_grid_thw, 
            "prompt": prompt,
            "ref_pixel_values": ref_pixel_values,
            "pil_pixel_values": pil_pixel_values, 
            "siglip_pixel_values": siglip_pixel_values, 
            "weights": weights, 
            "generated_image": generated_image, 
        }
        return return_data



    def getitem(self, data):
        # Reformat the conversation to the format of prompter
        conversations = []
        prompt = ""
        for item in data["conversations"]:
            if item["from"] == "human":
                role = self.prompter.user_role
                prompt = item["value"]
            elif item["from"] == "gpt":
                role = self.prompter.assistant_role
            else:
                raise ValueError(f"Unknown role: {item['from']}")
            conversations.append({"from": role, "value": item["value"]})
        assert prompt != "", "prompt != ''"
        # The last turn instruction will be used for t5_embed
        prompt = prompt.replace('<image>', '').replace('\n', '')

        # Make prompt
        drop_prompt = False
        if self.only_generated_task:
            if self.drop_prompt_rate < random.random():  # Randomly drop the prompt
                prompt_list = self.prompter.get_train_prompt(conversations)
            else:
                drop_prompt = True
                num_images = (''.join([i['value'] for i in conversations])).count('<image>')
                # Drop the prompt
                prompt_list = [
                    {
                        "from": self.prompter.system_role,
                        "value": "You are a helpful assistant.",
                    },
                    {
                        "from": self.prompter.user_role,
                        # "value": f"{num_images * '<image>'} Generate an image.",
                        "value": "Generate an image.",
                    },
                    {
                        "from": self.prompter.assistant_role,
                        "value": self.generated_image_token,
                    },
                ]
                prompt_list = self.prompter.get_train_prompt(prompt_list)
        else:
            prompt_list = self.prompter.get_train_prompt(conversations)
            
        input_ids = []
        labels = []
        has_generated_image = False
        cur_i = 0
        for item in prompt_list:
            item["prompt"] = item["prompt"].replace('<image>', self.image_token)
            
            if self.generated_image_token in item["prompt"]:  # Check if self.generated_image_token in prompt
                assert item["from"] == self.prompter.assistant_role, (
                    "Generated image token must be in assistant role"
                )
                assert (
                    f"{self.generated_image_token}{self.prompter.eos_token}"
                    in item["prompt"]
                ), "Generated image token must in end of prompt"

                # Replace the generated image token with image begin token and without eos token
                item["prompt"] = item["prompt"].replace(
                    f"{self.generated_image_token}{self.prompter.eos_token}",
                    self.image_begin_token,
                )
                has_generated_image = True

            if self.ocr_enhancer and (self.image_token in item["prompt"]):
                # print('item["prompt"]', item["prompt"])
                if not has_generated_image:
                    num_img = item["prompt"].count(self.image_token)
                    ocr_sentences = []
                    for i in range(num_img):
                        ocr_sentences.append(get_ocr_result(data["image"][cur_i], cur_i))
                        cur_i += 1
                    ocr_sentences = '\n'.join(ocr_sentences)
                    if len(ocr_sentences.split()) > 256:
                        print(f'ocr_sentences too long, total len {len(ocr_sentences.split())} trunk first 256')
                        ocr_sentences = ' '.join(ocr_sentences.split()[:256])
                    # ocr_sentences = ''
                    assert item['prompt'][-len(self.prompter.eos_token):] == self.prompter.eos_token, \
                        "item['prompt'][-len(self.prompter.eos_token):] == self.prompter.eos_token"
                    assert item['prompt'].count(self.prompter.eos_token) == 1, \
                        "item['prompt'].count(self.prompter.eos_token) == 1"
                    item["prompt"] = item["prompt"].replace(self.prompter.eos_token, f'{ocr_sentences} {self.prompter.eos_token}')

            # Apply Step1X preprocessing if enabled
            if (self.use_step1x_preprocessing and 
                self.step1x_tokenizer):
                
                try:
                    # 检查是否包含图像token，如果有则传递图像标识
                    has_images = '<image>' in item["prompt"] or self.image_token in item["prompt"]
                    
                    # Step1X tokenization：保护引号内容，不截断
                    tokenized_item = self.step1x_tokenizer.tokenize_with_protection(
                        item["prompt"],
                        return_tensors="pt", # 确保返回tensor
                        truncation=False,
                    )
                    
                    # 验证结果有效性
                    if not hasattr(tokenized_item, 'input_ids') or tokenized_item.input_ids.shape[1] == 0:
                        # 如果是空prompt，tokenizer可能会返回空结果，这是合法的
                        if item["prompt"].strip() == "":
                             pass
                        else:
                            raise ValueError("Step1X returned empty or invalid result for non-empty prompt")
                        
                except Exception as e:
                    print(f"Warning: Step1X failed for prompt: {e}")
                    
            else:
                # 使用标准tokenization
                tokenized_item = self.tokenizer(
                    item["prompt"],
                    return_tensors="pt",
                    truncation=False,
                    max_length=None,
                )
                
            if item["is_labels"]:  # If this prompt is labels
                labels.append(tokenized_item.input_ids)
            else:
                labels.append(torch.full_like(tokenized_item.input_ids, -100))
            input_ids.append(tokenized_item.input_ids)

        if (
            self.only_generated_task and not has_generated_image
        ):  # For denoiser training
            raise ValueError(
                f"Only generated task is not supported. But this prompt not contains generated image token: {prompt_list[0]['prompt']}"
            )

        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)

        # Load images
        if has_generated_image:
            # generate task
            # process images but exclude the last image, which need to generate
            image_slice = data["image"][:-1]
        else:
            # understanding task
            image_slice = data["image"]

        image_dict = self._load_image(
            image_slice, self.max_pixels, self.min_pixels, 
            processor=self.processor, image_token=self.image_token, 
            factor=self.factor, 
            last_image=data["image"][-1] if has_generated_image else None,
            vae_image_transform=self.transform, 
            drop_prompt=drop_prompt, 
            prompt=prompt, 
            mask_weight_type=self.mask_weight_type, 
            siglip_processor=self.siglip_processor, 
            need_weight=data['need_weight'], 
            )
        
        image_token_lengths = image_dict['image_token_lengths']
        pixel_values = image_dict['pixel_values']
        image_grid_thw = image_dict['image_grid_thw']
        ref_pixel_values = image_dict['ref_pixel_values']
        pil_pixel_values = image_dict['pil_pixel_values']
        siglip_pixel_values = image_dict['siglip_pixel_values']
        weights = image_dict['weights']

        # Process image tokens only if there are any
        if len(image_token_lengths) > 0:
            input_ids, labels, image_position = self._process_image_token(
                input_ids,
                labels=labels,
                image_token_id=self.image_token_id,
                image_begin_token_id=self.image_begin_token_id,
                image_end_token_id=self.image_end_token_id,
                image_token_lengths=image_token_lengths, 
            )
        else:
            # No images, no processing needed
            image_position = []

        # 修复：添加超长prompt保护，防止显存爆炸
        max_sequence_length = 32768  # 设置合理的最大长度
        if input_ids.shape[1] > max_sequence_length:
            print(f"Warning: Sequence too long ({input_ids.shape[1]} > {max_sequence_length}), truncating...")
            input_ids = input_ids[:, -max_sequence_length:]
            labels = labels[:, -max_sequence_length:]
            # 重新计算image_position（如果有的话）
            if image_position:
                offset = input_ids.shape[1] - max_sequence_length
                image_position = [pos - offset for pos in image_position if pos - offset >= 0]

        return_data = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_position": image_position,
            "image_grid_thw": image_grid_thw, 
            "prompt": prompt,
            "ref_pixel_values": ref_pixel_values,
            "pil_pixel_values": pil_pixel_values, 
            "siglip_pixel_values": siglip_pixel_values, 
            "weights": weights, 
        }

        if has_generated_image: # If this item is a generation task
            image = Image.open(data["image"][-1]).convert("RGB")
            # if self.anyres:
            #     image = image.resize(pil_pixel_values[-1].size)
            image_tensor = torch.tensor(np.array(image)) / 255.0  # scale to 0-1
            image_tensor = rearrange(image_tensor, "h w c -> c h w")
            return_data["generated_image"] = self.transform(image_tensor)
        else:
            return_data["generated_image"] = []
        return return_data
    
    def __getitem__(self, idx):
        if self.random_data:
            return self._get_random_data()
        
        data: Any = self.data[idx]
        if self.notry:
            return self.getitem(data)
        try:
            return self.getitem(data)
        except Exception as e:
            print(f'Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__()-1))

    @staticmethod
    def _load_image(
        image_slice: List[str],
        max_pixels: int = 448*448,  
        min_pixels: int = 448*448, 
        processor: Callable = None, 
        image_processor: Callable = None, 
        image_token_lengths: int = 729, 
        image_token: str = '<|image_pad|>', 
        factor: int = 1, 
        last_image: Optional[str] = None, 
        vae_image_transform: Callable = None,
        drop_prompt: bool = False, 
        prompt: str = '', 
        mask_weight_type: str = None, 
        siglip_processor: Callable = None, 
        need_weight: str = 'true', 
    ):
        resize_ref_image = False
        pil_pixel_values_last = []
        if last_image is not None:
            last_vision_infos = dict(
                image=last_image, min_pixels=min_pixels, max_pixels=max_pixels
                )
            last_image_inputs, last_video_inputs = process_vision_info([last_vision_infos], factor=factor)

            pil_pixel_values_last.append(last_image_inputs[0])
            
            if len(image_slice) > 0 and not all([has_same_resolution(image_path, last_image) for image_path in image_slice]):
                resize_ref_image = True
                resize_w, resize_h = last_image_inputs[0].size

        image_token_lengths = []
        pixel_values = []
        image_grid_thw = []
        ref_pixel_values = []  # Formerly condition_pixel_values, renamed to fix collator
        pil_pixel_values = []
        siglip_pixel_values = []
        # Ignore the last image (generated image)
        for image_path in image_slice: 
            # --- 1. Processing for QwenVL Encoder ---
            # This part remains the same, it prepares images for the language model's vision encoder.
            vision_infos = dict(image=image_path, min_pixels=min_pixels, max_pixels=max_pixels)
            if resize_ref_image:
                vision_infos.update(
                    dict(resized_height=resize_h, resized_width=resize_w)
                    )
            image_inputs, _ = process_vision_info([vision_infos], factor=factor)
            inputs = processor(text=[f'dummy {image_token}'], images=image_inputs, videos=None, padding=True, return_tensors="pt")
            
            if not drop_prompt:
                pixel_values.append(inputs.pixel_values)
                image_grid_thw.append(inputs.image_grid_thw)
                image_token_length = (inputs.input_ids[0] == processor.tokenizer.convert_tokens_to_ids(image_token)).sum()
                image_token_lengths.append(image_token_length)
            
            # This is for logging/masking and should use the Qwen-resized image
            pil_pixel_values.append(image_inputs[0])

            # --- 2. Processing for FLUX VAE Conditioning ---
            # This logic is now aligned with how `generated_image` is processed in `getitem`.
            raw_cond_image = Image.open(image_path).convert("RGB")
            # Convert to tensor C, H, W for the transform pipeline
            raw_cond_tensor = torch.tensor(np.array(raw_cond_image), dtype=torch.float32) / 255.0
            raw_cond_tensor = rearrange(raw_cond_tensor, "h w c -> c h w")

            if vae_image_transform:
                # Apply the full transform (resize + norm) passed from getitem
                transformed_cond_tensor = vae_image_transform(raw_cond_tensor)
            else:
                # Fallback if no transform is provided
                transformed_cond_tensor = (raw_cond_tensor - 0.5) / 0.5

            # Add batch dimension for collation
            transformed_cond_tensor = transformed_cond_tensor.unsqueeze(0)

            if drop_prompt:
                ref_pixel_values.append(torch.zeros_like(transformed_cond_tensor))
            else:
                ref_pixel_values.append(transformed_cond_tensor)

            # --- 3. Optional SigLIP processing ---
            if siglip_processor is not None:
                siglip_pixel_value = siglip_processor.preprocess(
                            images=raw_cond_image, # Use raw image
                            do_resize=True, return_tensors="pt", do_convert_rgb=True
                        ).pixel_values  # 1 c h w
                if drop_prompt:
                    siglip_pixel_values.append(torch.zeros_like(siglip_pixel_value))
                else:
                    siglip_pixel_values.append(siglip_pixel_value)
            
        # if multi-image in a sample, concat them
        # assume pixel_values[0] (n1, 1176), pixel_values[1] (n2, 1176), pixel_values will be (n1+n2, 1176)
        if len(pixel_values) > 0:
            pixel_values = torch.concat(pixel_values)
            image_grid_thw = torch.concat(image_grid_thw)  # (b, 3), 3 mean the grid of t, h, w
        if len(ref_pixel_values) > 0:
            ref_pixel_values = torch.cat(ref_pixel_values, dim=0)

        if len(siglip_pixel_values) > 0: 
            siglip_pixel_values = torch.concat(siglip_pixel_values)  # b c h w

        pil_pixel_values = pil_pixel_values + pil_pixel_values_last
        
        if mask_weight_type is not None:
            _, weights = get_weight_mask(pil_pixel_values, prompt, mask_weight_type, need_weight)
            if need_weight.lower() == 'false':
                assert torch.all(weights == 1)
        else:
            weights = []
        return {
            'pixel_values': pixel_values, 
            'image_grid_thw': image_grid_thw, 
            'image_token_lengths': image_token_lengths, 
            'ref_pixel_values': ref_pixel_values,
            'pil_pixel_values': pil_pixel_values, 
            'siglip_pixel_values': siglip_pixel_values, 
            'weights': weights, 
            }

    @staticmethod
    def _process_image_token(
        input_ids: torch.Tensor,
        image_token_id: int,
        image_begin_token_id: int,
        image_end_token_id: int,
        image_token_lengths: List[int],
        labels: Optional[torch.Tensor] = None,
    ):
        # Find the indices of the image token
        image_token_indices = (input_ids == image_token_id).nonzero(as_tuple=True)
        # assert len(image_token_lengths) == image_token_indices[1].numel()
        image_position = []
        offset = 0
        cur_i = 0
        if isinstance(image_token_lengths, int):
            image_token_lengths = [image_token_lengths] * len(image_token_indices[1])
        for idx in image_token_indices[1]:
            image_token_length = image_token_lengths[cur_i]
            adjusted_idx = idx + offset
            assert input_ids[0, adjusted_idx] == image_token_id, "assert input_ids[0, adjusted_idx] == image_token_id"

            # Add image begin and end token
            input_ids = torch.cat(
                [
                    input_ids[:, :adjusted_idx],
                    input_ids.new_full(
                        (1, 1), image_begin_token_id
                    ),  # image begin token
                    input_ids.new_full(
                        (1, image_token_length), image_token_id
                    ),  # Repeat the image token to the length of image_token_length
                    input_ids.new_full((1, 1), image_end_token_id),  # image end token
                    input_ids[:, adjusted_idx + 1 :],
                ],
                dim=1,
            )
            if labels is not None:
                labels = torch.cat(
                    [
                        labels[:, :adjusted_idx],
                        labels.new_full(
                            (1, 1), image_begin_token_id
                        ),  # Make begin token as label
                        labels.new_full((1, image_token_length), -100),
                        labels.new_full((1, 1), -100),
                        labels[:, adjusted_idx + 1 :],
                    ],
                    dim=1,
                )

            adjusted_idx += 1  # skip the image begin token
            image_position.append(adjusted_idx.item())
            offset += image_token_length - 1
            offset += 2  # begin and end token

            cur_i += 1

        return input_ids, labels, image_position
    

def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = 28) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels")
        max_pixels = ele.get("max_pixels")
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)

    return image

def process_vision_info(
    vision_infos: list,
    return_video_kwargs: bool = False,
    factor: int = 1, 
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, size_factor=28*factor))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs