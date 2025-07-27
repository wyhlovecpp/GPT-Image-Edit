from univa.models.configuration_univa_denoise_tower import UnivaDenoiseTowerConfig
from transformers.modeling_utils import PreTrainedModel

from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn
import numpy as np
from diffusers import FluxTransformer2DModel, SD3Transformer2DModel
from diffusers.utils import is_torch_version
from diffusers.models.modeling_outputs import Transformer2DModelOutput


class UnivaDenoiseTower(PreTrainedModel):
    config_class = UnivaDenoiseTowerConfig
    base_model_prefix = "model"

    def __init__(self, config: UnivaDenoiseTowerConfig):
        super().__init__(config)
        self.config = config
        if config.denoiser_type == "flux":
            self.denoiser = FluxTransformer2DModel.from_config(config.denoiser_config)
        elif config.denoiser_type == "sd3":
            self.denoiser = SD3Transformer2DModel.from_config(config.denoiser_config)
        else:
            raise ValueError(f"Unknown denoiser type: {config.denoiser_type}")

        # Only keep denoise projector for QwenVL -> FLUX dimension mapping
        if hasattr(config, 'denoise_projector_type') and config.denoise_projector_type:
            self._init_denoise_projector()

    def _init_denoise_projector(self):
        """Initialize the denoise_projector for QwenVL -> FLUX dimension mapping."""
        if self.config.denoise_projector_type == "mlp2x_gelu":
            self.denoise_projector = nn.Sequential(
                nn.Linear(
                    self.config.input_hidden_size,
                    self.config.output_hidden_size * 3,
                ),
                nn.SiLU(),
                nn.Linear(
                    self.config.output_hidden_size * 3, self.config.output_hidden_size
                ),
            )
        else:
            raise ValueError(
                f"Unknown denoise_projector_type: {self.config.denoise_projector_type}"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Simplified forward pass - VAE latents are directly concatenated with hidden_states.
        No projectors needed.
        """
        if self.config.denoiser_type == "flux":
            prefix_prompt_embeds = kwargs.pop("prefix_prompt_embeds", None)
            
            if encoder_hidden_states is not None:
                if prefix_prompt_embeds is not None:
                    encoder_hidden_states = torch.concat(
                        [encoder_hidden_states, prefix_prompt_embeds], dim=1
                    )
            else:
                assert prefix_prompt_embeds is not None
                encoder_hidden_states = prefix_prompt_embeds
                
            txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(
                hidden_states.device, dtype=hidden_states.dtype
            )

            joint_attention_kwargs = kwargs.pop('joint_attention_kwargs', None)
            # if joint_attention_kwargs is not None and 'attention_mask' in joint_attention_kwargs:
            #     attention_mask = joint_attention_kwargs['attention_mask']
            # else:
            #     attention_mask = torch.full(
            #         (hidden_states.shape[0], 1, hidden_states.shape[1]), 
            #         True, dtype=torch.bool, device=hidden_states.device
            #         )
                
            enc_attention_mask = kwargs.pop('enc_attention_mask', None)
            # if enc_attention_mask is None:
            #     enc_attention_mask = torch.full(
            #         (encoder_hidden_states.shape[0], 1, encoder_hidden_states.shape[1]), 
            #         True, dtype=torch.bool, device=encoder_hidden_states.device
            #         )
            # else:
            #     enc_attention_mask = enc_attention_mask.unsqueeze(1)
                    
            # attention_mask = torch.concat([enc_attention_mask, attention_mask], dim=-1)
            # attention_mask = attention_mask.unsqueeze(1)

            # joint_attention_kwargs['attention_mask'] = attention_mask
            # kwargs['joint_attention_kwargs'] = joint_attention_kwargs

            # print(f'hidden_states.shape, {hidden_states.shape}, encoder_hidden_states.shape, {encoder_hidden_states.shape}')
            # return self.fixed_flux_forward(
            return self.denoiser(
                hidden_states=hidden_states,
                timestep=timestep, # Note: timestep is in [0, 1]. It has been scaled by 1000 in the training script.
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                txt_ids=txt_ids,
                **kwargs,
            )[0]

        elif self.config.denoiser_type == "sd3":
            prefix_prompt_embeds = kwargs.pop("prefix_prompt_embeds", None)
            if prefix_prompt_embeds is not None:
                encoder_hidden_states = torch.concat(
                    [prefix_prompt_embeds, encoder_hidden_states], dim=1
                )

            return self.denoiser(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                **kwargs,
            )[0]





