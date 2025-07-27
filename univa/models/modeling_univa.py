from typing import Optional, List, Tuple, Union, Literal, Dict
import torch
import torch.nn as nn
from transformers import (
    Qwen2Model,
    Qwen2PreTrainedModel,
    GenerationMixin,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from univa.models.configuration_univa import UnivaConfig
from univa.models.modeling_univa_denoise_tower import UnivaDenoiseTower
import numpy as np


class UnivaQwen2Model(Qwen2Model):
    def __init__(self, config: UnivaConfig):
        super().__init__(config)
        self.config = config


class UnivaQwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    config_class = UnivaConfig

    def __init__(self, config: UnivaConfig):
        super().__init__(config)
        self.model = UnivaQwen2Model(config)
        self.denoise_tower = UnivaDenoiseTower(config.denoise_tower)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.forward_denoiser = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_denoise_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[List[torch.FloatTensor]] = None,
        image_position: Optional[torch.LongTensor] = None,
    ):
        input_embeds = self(input_ids, pixel_values, image_position)[0]
        input_embeds = self.denoise_tower(input_embeds)
        return input_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[List[torch.FloatTensor]] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        image_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_type: Literal["lvlm", "denoise_model_pred", "denoise_embeds"] = "lvlm",
        only_use_t5: bool = False, 
        denoiser_kwargs: Optional[Dict] = {},
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if not only_use_t5:
            if (
                self.forward_denoiser
            ):  # Force forward denoiser, which is used in FSDP training
                return self.denoise_tower.denoiser(**kwargs)

            if "hidden_states" in kwargs:
                print(
                    "You are using this model as a denoiser, please use the forward_denoiser_context to forward the model."
                )
                print("For example:")
                print("with self.forward_denoiser_context():")
                print("    ... # Your code ...")

            inputs_embeds, shortcut_image_embeds = self.prepare_inputs_for_multimodal(
                input_ids,
                pixel_values,
                image_position,
                past_key_values,
                output_image_embeds=True,
            )

            if output_type == "denoise_model_pred":
                assert len(denoiser_kwargs) > 0, (
                    "denoiser_kwargs should not be empty when output_type is denoise_model_pred"
                )
                return_dict = False

            outputs = self.inner_forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                output_denoise_embeds=output_type.startswith("denoise"),
                **kwargs,
            )
        else:
            outputs = None

        if output_type.startswith("denoise"):
            if output_type == "denoise_embeds":
                # LVLM outputs -> MLP2 -> prompt_embeds
                # with prompt_embeds, we can directly forward the denoiser.
                return self.denoise_tower.denoise_projector(outputs)
            elif output_type == "denoise_model_pred":
                # LM outputs -> MLP2 -> Denoiser -> model_pred
                return self.denoise_tower(
                    encoder_hidden_states=outputs, **denoiser_kwargs
                )
            else:
                raise ValueError(f"Unknown output_type: {output_type}.")

        return outputs

    def prepare_inputs_for_multimodal(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[List[torch.FloatTensor]] = None,
        image_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_image_embeds: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[int, int, int, torch.Tensor]]]]:
        batch_size, _ = input_ids.shape
        input_embeds = self.model.embed_tokens(input_ids)
        if (
            past_key_values is not None and len(past_key_values.key_cache) > 0
        ):  # Skip if using cache
            return input_embeds, None

        if pixel_values is None:  # No image input
            return input_embeds, None

        # Since we removed vision_tower, we no longer process images here
        # QwenVL models handle vision processing internally
        # This method now mainly handles text embeddings
        
        # For compatibility, return empty shortcut_image_embeds
        shortcut_image_embeds = []
        
        if output_image_embeds:
            return input_embeds, shortcut_image_embeds
        else:
            return input_embeds, None

    def inner_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_denoise_embeds: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        if output_denoise_embeds:
            return hidden_states

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward_denoiser_context(self):
        class ForwardDenoiserContext:
            def __init__(self, model):
                self.model = model
                self.backup_config = None

            def __enter__(self):
                self.backup_config = self.model.config
                self.model.config = self.model.denoise_tower.denoiser.config
                self.model.forward_denoiser = True
                return self.model

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.model.forward_denoiser = False
                self.model.config = self.backup_config
                return False

        return ForwardDenoiserContext(self)

    def find_true_blocks(self, mask):
        """Helper function to find consecutive True blocks in a boolean mask"""
        # Convert to numpy for easier processing
        mask_np = mask.cpu().numpy()
        
        # Find where True blocks start and end
        diff = np.diff(np.concatenate(([False], mask_np, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Calculate lengths and create start_end_index
        lengths = ends - starts
        start_end_index = starts.tolist()
        
        return list(range(len(starts))), start_end_index, lengths.tolist()
