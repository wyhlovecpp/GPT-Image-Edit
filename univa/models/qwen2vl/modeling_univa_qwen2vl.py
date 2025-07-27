from typing import Optional, List, Tuple, Union, Literal, Dict
import torch._dynamo
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers import GenerationMixin
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLModel,
    Qwen2VLPreTrainedModel,
    Qwen2VisionTransformerPretrainedModel, 
    Qwen2VLCausalLMOutputWithPast
)
# from univa.models.modeling_univa_vision_tower import UnivaVisionTower
# from univa.models.configuration_univa import UnivaConfig
from univa.models.qwen2vl.configuration_univa_qwen2vl import UnivaQwen2VLConfig
from univa.models.modeling_univa_denoise_tower import UnivaDenoiseTower

class UnivaQwen2VLModel(Qwen2VLModel):
    def __init__(self, config: UnivaQwen2VLConfig):
        super().__init__(config)
        self.config = config

class UnivaQwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = UnivaQwen2VLConfig

    def __init__(self, config: UnivaQwen2VLConfig):
        super().__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
        print("visual init done")
        self.model = UnivaQwen2VLModel(config)
        print("model init done")
        self.denoise_tower = UnivaDenoiseTower(config.denoise_tower)
        print("denoise tower init done")

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        self.forward_denoiser = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_denoise_embeds(
        self,
        input_ids: torch.LongTensor,
        images: Optional[List[torch.FloatTensor]] = None,
        image_position: Optional[torch.LongTensor] = None,
    ):
        input_embeds = self(input_ids, images, image_position)[0]
        input_embeds = self.denoise_tower(input_embeds)
        return input_embeds


    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    # @torch._dynamo.disable
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i].to(input_ids.device) == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                #################
                # skip last boi, because last boi do NOT have true image_token
                vision_start_indices = vision_start_indices[vision_start_indices + 1 <len(input_ids)]
                ##############
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas
        

    # @torch.compile
    def forward_visual(self, pixel_values, grid_thw):
        return self.visual(pixel_values, grid_thw=grid_thw)

    # @torch.compile
    def forward(
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
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_type: Literal["lvlm", "denoise_model_pred", "denoise_embeds"] = "lvlm",
        denoiser_kwargs: Optional[Dict] = {},
        only_use_t5: bool = False,
        vlm_residual_image_factor: float = 0.0, 
        **kwargs,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
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
            # if isinstance(pixel_values, list):
                # print('pixel_values is list:', *[i.shape for i in pixel_values])
                # pixel_values = torch.cat(pixel_values)
                # print('pixel_values convert to tensor:', pixel_values.shape)
            # if isinstance(image_grid_thw, list):
                # print('image_grid_thw is list:', *[i.shape for i in image_grid_thw])
                # image_grid_thw = torch.cat(image_grid_thw)
                # print('image_grid_thw convert to tensor:', image_grid_thw.shape)
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.visual.get_dtype())
                    #################################
                    # add these line
                    image_embeds = self.forward_visual(pixel_values, grid_thw=image_grid_thw)
                    
                    if self.config.shortcut_projector_type is not None:
                        shortcut_image_embeds_batch = image_embeds
                    else:
                        shortcut_image_embeds_batch = None
                    #################################
                    n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                    n_image_features = image_embeds.shape[0]
                    if n_image_tokens != n_image_features:
                        raise ValueError(
                            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                        )
                    image_mask = (
                        (input_ids == self.config.image_token_id)
                        .unsqueeze(-1)
                        .expand_as(inputs_embeds)
                        .to(inputs_embeds.device)
                    )
                    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                    video_embeds = self.forward_visual(pixel_values_videos, grid_thw=video_grid_thw)
                    n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                    n_video_features = video_embeds.shape[0]
                    if n_video_tokens != n_video_features:
                        raise ValueError(
                            f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                        )
                    video_mask = (
                        (input_ids == self.config.video_token_id)
                        .unsqueeze(-1)
                        .expand_as(inputs_embeds)
                        .to(inputs_embeds.device)
                    )
                    video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(inputs_embeds.device)

                shortcut_image_embeds = []
                if pixel_values is not None and shortcut_image_embeds_batch is not None:
                    cum_image_len = 0
                    for batch_idx in range(input_ids.shape[0]):
                        cur_input_ids = input_ids[batch_idx]
                        num_blocks, start_end_index, lengths = self.find_true_blocks((cur_input_ids == self.config.image_token_id))
                        for i in range(len(num_blocks)):
                            shortcut_image_embeds.append(
                                (
                                    # batch_idx,
                                    # pos,
                                    # lengths,
                                    # shortcut_image_embeds_batch, 
                                    batch_idx,
                                    start_end_index[i],
                                    lengths[i],
                                    shortcut_image_embeds_batch[cum_image_len: cum_image_len+lengths[i]], 
                                )
                            )
                            cum_image_len = cum_image_len + lengths[i]

            if output_type == "denoise_model_pred":
                assert len(denoiser_kwargs) > 0, (
                    "denoiser_kwargs should not be empty when output_type is denoise_model_pred"
                )
                return_dict = False

            # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
            if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
                # calculate RoPE index once per generation in the pre-fill stage only
                if (
                    (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
                ):
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids, image_grid_thw, video_grid_thw, attention_mask
                    )
                    self.rope_deltas = rope_deltas
                # then use the prev pre-calculated rope-deltas to get the correct position ids
                else:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    if cache_position is not None:  # otherwise `deltas` is an int `0`
                        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                        delta = delta.to(position_ids.device)
                    position_ids = position_ids.add(delta)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
        
            hidden_states = outputs[0]
            
            
            if output_type.startswith("denoise"):
                outputs = outputs[0]
        else:
            outputs = None

        if output_type.startswith("denoise"):
            if outputs is not None and vlm_residual_image_factor > 0.0 and pixel_values is not None:
                old = outputs[image_mask[:, :, 0]]            # shape [N, D]
                blended = old * (1 - vlm_residual_image_factor) + image_embeds * vlm_residual_image_factor  # shape [N, D]
                outputs = outputs.masked_scatter(image_mask, blended)
            if outputs is not None and shortcut_image_embeds is not None and self.config.shortcut_image_embeds:
                for (
                    batch_idx,
                    pos,
                    image_seq_length,
                    image_embeds_item,
                ) in shortcut_image_embeds:
                    outputs[batch_idx, pos : pos + image_seq_length, :] = (
                        self.config.shortcut_image_embeds_scale * image_embeds_item
                        + (1 - self.config.shortcut_image_embeds_scale)
                        * outputs[batch_idx, pos : pos + image_seq_length, :]
                    )

            ref_features_for_vlm = kwargs.pop('ref_features_for_vlm', None)
            siglip_hidden_states = kwargs.pop('siglip_hidden_states', None)
            if outputs is not None:
                outputs = self.denoise_tower.denoise_projector(outputs)
                if ref_features_for_vlm is not None:
                    outputs_ref_features = self.denoise_tower.vae_projector(ref_features_for_vlm)
                    outputs = torch.cat([outputs, outputs_ref_features], dim=1)
                if siglip_hidden_states is not None:
                    siglip_hidden_states = self.denoise_tower.siglip_projector(siglip_hidden_states)
                    indices_list = self.find_all_token_positions(input_ids, self.config.image_end_token_id)
                    # import ipdb;ipdb.set_trace()
                    outputs = self._insert_img_to_vlm(outputs, siglip_hidden_states, indices_list)
                    # print(outputs.shape)
                

            if output_type == "denoise_embeds":
                # LVLM outputs -> MLP2 -> prompt_embeds
                # with prompt_embeds, we can directly forward the denoiser.
                return outputs
            elif output_type == "denoise_model_pred":
                # LM outputs -> MLP2 -> Denoiser -> model_pred
                return self.forward_denoise_tower(
                    outputs, **denoiser_kwargs
                )
            else:
                raise ValueError(f"Unknown output_type: {output_type}.")

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        outputs = Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return outputs
    
    def forward_denoise_tower(self, outputs, **denoiser_kwargs):
        return self.denoise_tower(
            encoder_hidden_states=outputs, **denoiser_kwargs
        )

    def find_all_token_positions(self, input_ids, token_id):
        """
        返回一个列表，列表中每个元素是该 batch 中对应样本中 token_id 出现的位置索引（1D Tensor）
        """
        match = (input_ids == token_id)  # [B, L] 的 bool 矩阵
        batch_indices, seq_indices = torch.where(match)  # 都是 1D，长度为匹配总数

        # 构建一个列表：每个样本一个 Tensor，记录匹配位置
        batch_size = input_ids.size(0)
        result = [[] for _ in range(batch_size)]
        for b, s in zip(batch_indices.tolist(), seq_indices.tolist()):
            result[b].append(s)
        return result

    def find_true_blocks(self, tensor):
        tensor = tensor.bool()
        # pad左右两边，方便处理边界
        padded = torch.nn.functional.pad(tensor[None].float(), (1, 1))  # 1D tensor -> shape (1, L+2)
        diff = padded[:, 1:] - padded[:, :-1]  # shape (1, L+1)

        # +1 表示从 False -> True（块开始），-1 表示从 True -> False（块结束）
        starts = (diff == 1).nonzero(as_tuple=True)[1]
        ends = (diff == -1).nonzero(as_tuple=True)[1] - 1  # 结束 index 是最后一个 True 的位置

        lengths = ends - starts + 1
        num_blocks = starts.numel()
        return num_blocks, list(zip(starts, ends)), lengths

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen2-VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if model_inputs["cache_position"][0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs
    
    def _insert_img_to_vlm(self, vlm_feature, img_feature, indices_list):
        B, L, D = vlm_feature.shape
        assert img_feature.ndim == 3
        img_L = img_feature.shape[1]
        max_new_len = max([L+img_L*len(inds) for inds in indices_list])

        new_vlm_feature = torch.zeros(B, max_new_len, D, dtype=vlm_feature.dtype, device=vlm_feature.device)

        img_mask = torch.zeros((B, max_new_len, 1), dtype=torch.bool, device=vlm_feature.device)
        for i, inds in enumerate(indices_list):
            for j, pos in enumerate(inds):
                # print(i, f'{j*img_L + pos} -> {(j+1)*img_L + pos}')
                img_mask[i, j*img_L + pos: (j+1)*img_L + pos] = True

        vlm_mask = ~img_mask
        for i, inds in enumerate(indices_list):
            # print(i, f'{L+img_L*len(inds)}')
            vlm_mask[i, L+img_L*len(inds): ] = False

        img_mask = img_mask.repeat(1, 1, D)
        assert torch.sum(img_mask) == img_feature.numel()
        new_vlm_feature.masked_scatter_(img_mask, img_feature)

        vlm_mask = vlm_mask.repeat(1, 1, D)
        assert torch.sum(vlm_mask) == vlm_feature.numel()
        new_vlm_feature.masked_scatter_(vlm_mask, vlm_feature.view(-1, D))
        return new_vlm_feature

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        vision_start_mask = input_ids == vision_start_token_id
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(input_ids)

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    if not isinstance(dict_to_expand[key], list):
                        raise TypeError(
                            f"Expected value for key '{key}' to be a list, but got {type(dict_to_expand[key])} instead."
                        )
                    tensor = torch.tensor(dict_to_expand[key])
                    lengths = list(video_nums)
                    tensor = _repeat_interleave_samples(tensor, lengths=lengths, repeat_times=expand_size)
                    dict_to_expand[key] = tensor.tolist()
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # input_ids is required for expanding visual inputs
        # If input_ids is unavailable, visual inputs will not be used; therefore, there is no need to expand visual inputs.
        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


# __all__ = ["Qwen2VLForConditionalGeneration", "Qwen2VLModel", "Qwen2VLPreTrainedModel"]
