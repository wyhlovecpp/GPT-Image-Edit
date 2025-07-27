"""
Text preprocessing utilities for UniWorld-V1.
Includes Step1X-style prompt preprocessing to preserve quoted text during tokenization.
"""

import re
import torch
from typing import List, Union, Optional, Tuple, Any
from transformers import AutoProcessor


def split_string(s: str) -> List[str]:
    """
    Split string while preserving quoted text for better tokenization.
    Based on Step1X implementation.
    
    This function protects quoted text by wrapping each character in quotes
    individually, preventing the tokenizer from merging them inappropriately.
    
    Args:
        s: Input string to split
        
    Returns:
        List of string segments for separate tokenization
    """
    # Convert Chinese quotes to English quotes
    s = s.replace(""", '"').replace(""", '"').replace("'", '''"''')
    result = []
    in_quotes = False
    temp = ""

    for idx, char in enumerate(s):
        # Only process quotes after index 155 (to avoid interfering with system prompts)
        if char == '"' and idx > 155:
            temp += char
            if not in_quotes:
                result.append(temp)
                temp = ""
            in_quotes = not in_quotes
            continue
            
        if in_quotes:
            # For quoted content, wrap each character individually
            if char.isspace():
                pass  # Skip spaces in quotes
            result.append(""" + char + """)
        else:
            temp += char

    if temp:
        result.append(temp)

    return result


def split_literal(prompt: str) -> Tuple[str, List[str]]:
    """
    Extract literal strings from prompts and replace with placeholders.
    
    Args:
        prompt: Input prompt string
        
    Returns:
        Tuple of (processed_text, list_of_literals)
    """
    literals = []
    
    # Find strings in quotes or backticks
    quote_pattern = r'["\'\`]([^"\'`]*)["\'\`]'
    
    def replace_literal(match):
        literal = match.group(1)
        literals.append(literal)
        return f"▁<lit{len(literals)-1}>"
    
    processed_text = re.sub(quote_pattern, replace_literal, prompt)
    
    return processed_text, literals


def restore_literals(text: str, literals: List[str]) -> str:
    """
    Restore literal strings from placeholders.
    
    Args:
        text: Text with placeholders
        literals: List of literal strings
        
    Returns:
        Text with restored literals
    """
    for i, literal in enumerate(literals):
        placeholder = f"▁<lit{i}>"
        text = text.replace(placeholder, literal)
    
    return text


class Step1XTextPreprocessor:
    """Step1X-style text preprocessor for preserving quoted literals."""
    
    def __init__(self):
        pass
        
    def preprocess(self, text: str) -> Tuple[str, List[str]]:
        """
        Preprocess text to preserve quoted literals.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (processed_text, literals_list)
        """
        return split_literal(text)
        
    def postprocess(self, text: str, literals: List[str]) -> str:
        """
        Restore literals after tokenization.
        
        Args:
            text: Processed text
            literals: List of preserved literals
            
        Returns:
            Text with restored literals
        """
        return restore_literals(text, literals)


def preprocess_prompt_step1x_style(
    processor: AutoProcessor,
    text: str,
    ref_image: Optional[Any] = None,
    prefix: str = "",
    device: str = "cuda"
) -> dict:
    """
    Preprocess prompt using Step1X style tokenization.
    
    Args:
        processor: Qwen2.5-VL processor
        text: Input text prompt
        ref_image: Reference image (optional)
        prefix: System prefix (optional)
        device: Target device
        
    Returns:
        Dictionary with processed inputs
    """
    # Prepare messages for chat template
    messages = [{"role": "user", "content": []}]
    
    if prefix:
        messages[0]["content"].append({"type": "text", "text": prefix})
    
    if ref_image is not None:
        messages[0]["content"].append({"type": "image", "image": ref_image})
    
    messages[0]["content"].append({"type": "text", "text": text})
    
    # Apply chat template
    formatted_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
    )
    
    # Process vision info if image is provided
    if ref_image is not None:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        image_inputs = None
    
    # Initial tokenization
    inputs = processor(
        text=[formatted_text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Apply Step1X split_string preprocessing
    old_input_ids = inputs.input_ids
    text_split_list = split_string(formatted_text)
    
    # Tokenize each segment separately
    token_list = []
    for text_segment in text_split_list:
        segment_inputs = processor(
            text=text_segment,
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        token_segment = segment_inputs.input_ids
        
        # Remove BOS and EOS tokens if present
        if token_segment[0][0] == 2073 and token_segment[0][-1] == 854:
            token_segment = token_segment[:, 1:-1]
        
        token_list.append(token_segment)
    
    # Concatenate all segments
    new_input_ids = torch.cat(token_list, dim=1).to(device)
    new_input_ids = new_input_ids.to(old_input_ids.device)
    
    # Find the split point and reconstruct
    try:
        idx1 = (old_input_ids == 151653).nonzero(as_tuple=True)[1][0]
        idx2 = (new_input_ids == 151653).nonzero(as_tuple=True)[1][0]
        
        inputs.input_ids = torch.cat([
            old_input_ids[0, :idx1], 
            new_input_ids[0, idx2:]
        ], dim=0).unsqueeze(0).to(device)
        
        inputs.attention_mask = (inputs.input_ids > 0).long().to(device)
    except IndexError:
        # Fallback if special tokens not found
        inputs.input_ids = new_input_ids
        inputs.attention_mask = (inputs.input_ids > 0).long().to(device)
    
    return inputs


def prepare_text_embeddings_step1x(
    model: Any,
    processor: AutoProcessor,
    captions: List[str],
    ref_images: Optional[List[Any]] = None,
    max_length: int = 640,
    prefix: str = "",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare text embeddings using Step1X preprocessing.
    
    Args:
        model: Qwen2.5-VL model
        processor: Qwen2.5-VL processor
        captions: List of text prompts
        ref_images: List of reference images (optional)
        max_length: Maximum sequence length
        prefix: System prefix
        device: Target device
        dtype: Target dtype
        
    Returns:
        Tuple of (embeddings, masks)
    """
    batch_size = len(captions)
    hidden_size = model.config.hidden_size
    
    # Initialize output tensors
    embeddings = torch.zeros(
        batch_size, max_length, hidden_size,
        dtype=dtype, device=device
    )
    masks = torch.zeros(
        batch_size, max_length,
        dtype=torch.long, device=device
    )
    
    # Process each caption
    for idx, caption in enumerate(captions):
        ref_image = ref_images[idx] if ref_images else None
        
        # Preprocess with Step1X style
        inputs = preprocess_prompt_step1x_style(
            processor=processor,
            text=caption,
            ref_image=ref_image,
            prefix=prefix,
            device=device
        )
        
        # Move inputs to correct device and dtype
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.float32:
                    inputs[key] = value.to(dtype)
                inputs[key] = inputs[key].to(device)
        
        # Get embeddings from model
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values.to(device) if hasattr(inputs, 'pixel_values') else None,
                image_grid_thw=inputs.image_grid_thw.to(device) if hasattr(inputs, 'image_grid_thw') else None,
                output_hidden_states=True,
            )
        
        # Extract embeddings (skip first 217 tokens which are system/image tokens)
        emb = outputs['hidden_states'][-1]
        seq_len = min(max_length, emb.shape[1] - 217)
        
        if seq_len > 0:
            embeddings[idx, :seq_len] = emb[0, 217:217+seq_len]
            masks[idx, :seq_len] = 1
    
    return embeddings, masks


# Legacy function for backward compatibility
def preprocess_text_with_quotes(text: str) -> str:
    """
    Legacy function for simple quote preprocessing.
    For full Step1X compatibility, use preprocess_prompt_step1x_style instead.
    """
    # Simple quote normalization
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")
    return text 