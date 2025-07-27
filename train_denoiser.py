import torch._dynamo
torch._dynamo.config.optimize_ddp = False
from univa.training.configuration_denoise import UnivaTrainingDenoiseConfig
from pathlib import Path
import os
from typing import List, Dict, Callable, Optional
import math
import random
import shutil
from einops import rearrange, repeat
import copy
import json
import deepspeed
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
from accelerate.utils import ProjectConfiguration, set_seed
from univa.models import MODEL_TYPE, UnivaQwen2ForCausalLM
from univa.models.modeling_univa_denoise_tower import UnivaDenoiseTower
from univa.dataset import DATASET_TYPE
from univa.dataset.data_collator import DataCollator, pad_list_of_tensors
from univa.utils.prompter import PROMPT_TYPE, Qwen2Prompter
from univa.utils.constant import SPACIAL_TOKEN, GENERATE_TOKEN
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt, _encode_prompt_with_t5
from univa.utils.get_ocr import ocr_with_paddle, draw_boxes, get_ocr_result
from univa.utils.flux_pipeline import FluxKontextPipeline
from univa.utils.create_ema import EMAModel, _z3_params_to_fetch
from univa.utils.anyres_util import dynamic_resize
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers.integrations import HfDeepSpeedConfig
from transformers import (
    CLIPTextModel,
    T5EncoderModel,
    CLIPTokenizer,
    T5TokenizerFast,
    AutoImageProcessor,
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoProcessor, 
)
from torchvision import transforms
import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from PIL import Image
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    # FluxKontextPipeline,
)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
import torch.serialization
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.utils.tensor_fragment import fragment_address

# Add all required DeepSpeed classes/functions to the list of safe globals
torch.serialization.add_safe_globals([
    LossScaler, 
    ZeroStageEnum, 
    fragment_address
])
from contextlib import nullcontext
import wandb

GB = 1024 * 1024 * 1024
def get_trainable_params(
    layers_to_train: int = list(range(57)), num_transformer_blocks: int = 19, only_img_branch: bool = True
):
    components = [
        # "x_embedder"
        ]
    transformer_components = [
        "attn.norm_q", 
        "attn.norm_k", 
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out",
        "norm1.linear",
    ]
    single_transformer_components = [
        "attn.norm_q", 
        "attn.norm_k", 
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "norm.linear",
    ]
    if not only_img_branch:
        components.extend(
            [
                # "context_embedder"
            ]
        )
        transformer_components.extend(
            [
                "norm1_context.linear", "attn.norm_added_q", "attn.norm_added_k", "ff.net", "ff_context.net"
            ]
        )
        single_transformer_components.extend(
            [
                "proj_mlp", "proj_out"
            ]
        )
    for layer in layers_to_train:
        if layer < num_transformer_blocks:
            prefix = f"denoise_tower.denoiser.transformer_blocks.{layer}"
            base_components = transformer_components
        else:
            prefix = f"denoise_tower.denoiser.single_transformer_blocks.{layer - num_transformer_blocks}"
            base_components = single_transformer_components
        components.extend([f"{prefix}.{comp}" for comp in base_components])

    return components


def check_param_is_in_components(name: str, components: List[str]) -> bool:
    return any(component in name for component in components)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
                raise NotImplementedError
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def gather_zero3ema(accelerator, ema_model):
    model_to_save = ema_model.model.module if hasattr(ema_model.model, "module") else ema_model.model
    model_state_dict = {}
    for k, v in model_to_save.named_parameters():
        # only gather z3 params
        params_to_fetch = _z3_params_to_fetch([v])
        with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
            vv = v.data.cpu()
            # if accelerator.process_index == 0:
            model_state_dict[k] = vv
    return model_state_dict


def pad_x_and_mask(model_input, attention_mask=None, max_h=None, max_w=None):
    if attention_mask is None:
        attention_mask = [None] * len(model_input)
    batch_attention_mask = None
    max_h = max(t.shape[2] for t in model_input) if max_h is None else max_h
    max_w = max(t.shape[3] for t in model_input) if max_w is None else max_w

    padded_list = []
    padded_mask_list = []
    for t, m in zip(model_input, attention_mask):
        _, _, h, w = t.shape
        pad_h = max_h - h
        pad_w = max_w - w
        # pad 的顺序是 (left, right, top, bottom)
        # 这里只在右边和下边 pad
        padded = F.pad(t, pad=(0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_list.append(padded)
        # import ipdb;ipdb.set_trace()
        if m is not None:
            m = m[:, :1]
            m = F.pad(m, pad=(0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_mask_list.append(m)
    batch_model_input = torch.cat(padded_list, dim=0) 
    if padded_mask_list[0] is not None:
        batch_attention_mask = torch.cat(padded_mask_list, dim=0) 
    return batch_model_input, batch_attention_mask
    
def build_validation_info(args):
    base_eval_prompts = []
    base_eval_image_paths = []
    base_phase_names = []
    if args.dataset_config.validation_t2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_t2i_prompt)
        base_eval_image_paths.append(None)
        base_phase_names.append('vlm->generate image')
    if args.dataset_config.validation_it2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_it2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_image_path)
        base_phase_names.append('vlm->reconstruct image')
    if args.dataset_config.validation_iit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_iit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_iit2i_path)
        base_phase_names.append('vlm->fusion 2 images')
    if args.dataset_config.validation_cannyt2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_cannyt2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_cannyt2i_path)
        base_phase_names.append('vlm->generate image based on canny')
    if args.dataset_config.validation_it2canny_prompt:
        base_eval_prompts.append(args.dataset_config.validation_it2canny_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_it2canny_path)
        base_phase_names.append('vlm->generate canny')
    if args.dataset_config.validation_poset2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_poset2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_poset2i_path)
        base_phase_names.append('vlm->generate image based on pose')
    if args.dataset_config.validation_it2pose_prompt:
        base_eval_prompts.append(args.dataset_config.validation_it2pose_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_it2pose_path)
        base_phase_names.append('vlm->generate pose')
    if args.dataset_config.validation_NIKEit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_NIKEit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_NIKEit2i_path)
        base_phase_names.append('vlm->edit nike')
    
    if args.dataset_config.validation_TRANSFERit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_TRANSFERit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_TRANSFERit2i_path)
        base_phase_names.append('vlm->transfer')

    if args.dataset_config.validation_EXTRACTit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_EXTRACTit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_EXTRACTit2i_path)
        base_phase_names.append('vlm->extract')
    if args.dataset_config.validation_TRYONit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_TRYONit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_TRYONit2i_path)
        base_phase_names.append('vlm->try on')

    if args.dataset_config.validation_REPLACEit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_REPLACEit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_REPLACEit2i_path)
        base_phase_names.append('vlm->replace')

    if args.dataset_config.validation_DETit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_DETit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_DETit2i_path)
        base_phase_names.append('vlm->detect')

    if args.dataset_config.validation_SEGit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_SEGit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_SEGit2i_path)
        base_phase_names.append('vlm->segment')
    
    if args.dataset_config.validation_REFiit2i_prompt:
        base_eval_prompts.append(args.dataset_config.validation_REFiit2i_prompt)
        base_eval_image_paths.append(args.dataset_config.validation_REFiit2i_path)
        base_phase_names.append('vlm->transfer based on ref-style ')
    return base_eval_prompts, base_eval_image_paths, base_phase_names

# deepspeed.init_distributed()
def create_ema_model(
        accelerator, 
        args, 
        resume_checkpoint_path, 
        model_cls,
        model_config,
        ema_model_state_dict,
        ds_config=None, 
        ):
    # model_config = AutoConfig.from_pretrained(model_name_or_path)
    ds_config["train_micro_batch_size_per_gpu"] = args.dataset_config.batch_size
    ds_config["fp16"]["enabled"] = False
    ds_config["bf16"]["enabled"] = False
    ds_config["gradient_accumulation_steps"] = args.training_config.gradient_accumulation_steps
    ds_config["train_batch_size"] = args.dataset_config.batch_size * args.training_config.gradient_accumulation_steps * accelerator.num_processes

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    accelerator.print(f'EMA deepspeed config {ds_config}')
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
            
    if resume_checkpoint_path:
        ema_model_path = os.path.join(resume_checkpoint_path, "model_ema")
        if os.path.exists(ema_model_path):
            ema_model = EMAModel.from_pretrained(ema_model_path, model_cls=model_cls)
            accelerator.print(f'Successully resume EMAModel from {ema_model_path}')
    else:
        # we load weights from original model instead of deepcopy
        # model = model_cls.from_config(model_config)
        # accelerator.print('init model', model)
        # for k, v in model.state_dict().items():
        #     accelerator.print(k, v.shape)
        # model.load_state_dict(ema_model_state_dict, strict=True)
        model = model_cls.from_pretrained(
            args.model_config.ema_pretrained_lvlm_name_or_path,
            # config=lvlm_model.config,
            # deepspeed=dschf.to_dict(),    
            torch_dtype=torch.float32,           # fp32
        )
        accelerator.print(f"model_cls.from_pretrained finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")
        model.eval().requires_grad_(False)
        model.to(accelerator.device)
        # model.config.hidden_size = 4096
        ema_model = EMAModel(
            model, decay=args.training_config.ema_decay,
            model_cls=model_cls, model_config=model_config
            )
        accelerator.print(f"EMAModel finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")
        accelerator.print(f'Successully deepcopy EMAModel from model')
    # from deepspeed.runtime.zero import Init as DSZeroInit
    # with DSZeroInit(config=ds_config):
    ema_model.model, _, _, _ = deepspeed.initialize(model=ema_model.model, config_params=ds_config)
    return ema_model

def main(args: UnivaTrainingDenoiseConfig, attn_implementation='sdpa'):
    # Prepare accelerator
    logging_dir = Path(
        args.training_config.output_dir, args.training_config.logging_dir
    )
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.training_config.output_dir, logging_dir=logging_dir
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.training_config.gradient_accumulation_steps,
        mixed_precision=args.training_config.mixed_precision,
        log_with=args.training_config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    # Set seed
    set_seed(args.training_config.seed, device_specific=True)

    # Create output directory
    if accelerator.is_main_process:
        if args.training_config.output_dir is not None:
            os.makedirs(args.training_config.output_dir, exist_ok=True)

    # Set weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Resume from checkpoint
    resume_checkpoint_path = None
    if args.training_config.resume_from_checkpoint:
        if args.training_config.resume_from_checkpoint != "latest":
            path = os.path.basename(args.training_config.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.training_config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.training_config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.training_config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(os.path.join(args.training_config.output_dir, path))
            resume_checkpoint_path = os.path.join(args.training_config.output_dir, path)
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            # first_epoch = global_step // num

    else:
        initial_global_step = 0

    dataset_type = args.dataset_config.dataset_type
    model_class = MODEL_TYPE[dataset_type]
    dataset_class = DATASET_TYPE[dataset_type]


    # Load models
    lvlm_model = model_class.from_pretrained(
        args.model_config.pretrained_lvlm_name_or_path,
        attn_implementation=attn_implementation,
    )
    accelerator.print(f'{lvlm_model}')
    lvlm_tokenizer, image_processor, processor = None, None, None
    if dataset_type == 'qwen2vl' or dataset_type == 'qwen2p5vl':
        processor = AutoProcessor.from_pretrained(
            args.model_config.pretrained_lvlm_name_or_path,
        )
        lvlm_tokenizer = processor.tokenizer
        image_processor = processor.image_processor

    elif dataset_type == 'llava':
        lvlm_tokenizer = AutoTokenizer.from_pretrained(
            args.model_config.pretrained_lvlm_name_or_path,
        )
        image_processor = AutoImageProcessor.from_pretrained(
            args.model_config.pretrained_lvlm_name_or_path,
        )
    else:
         raise NotImplementedError(f"Only support dataset_type in ['qwen2vl', 'llava'], but found {dataset_type}")

    # No SigLIP models needed
    siglip_processor, siglip_model = None, None

    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.model_config.pretrained_denoiser_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.model_config.pretrained_denoiser_name_or_path,
        subfolder="tokenizer_2",
    )
    # Load models
    text_encoder_cls_one = CLIPTextModel.from_pretrained(
        args.model_config.pretrained_denoiser_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )
    text_encoder_cls_two = T5EncoderModel.from_pretrained(
        args.model_config.pretrained_denoiser_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=weight_dtype,
    )

    vae = AutoencoderKL.from_pretrained(
        args.model_config.pretrained_denoiser_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
    )

    accelerator.print(f'{text_encoder_cls_two}')
    accelerator.print(f'{vae}')
    # Load scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_config.pretrained_denoiser_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # ------------------------------------------------------------------
    # Instantiate a *minimal* FluxKontextPipeline **after** VAE is ready.
    # We only need its utility methods (pack/unpack/prepare_latents). No
    # scheduler or text encoders are required during training.
    flux_pipeline = FluxKontextPipeline(
        scheduler=None,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        transformer=None,
    )
    # ------------------------------------------------------------------

    # Move models to device and set grad
    vae_dtype = torch.float32 if args.model_config.vae_fp32 else weight_dtype
    vae.to(accelerator.device, dtype=vae_dtype)
    accelerator.print(f"Load vae model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")

    text_encoder_cls_one.to(accelerator.device, dtype=weight_dtype)
    accelerator.print(f"Load text_encoder_cls_one model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")

    text_encoder_cls_two.to(accelerator.device, dtype=weight_dtype)
    accelerator.print(f"Load text_encoder_cls_two model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")

    lvlm_model.to(accelerator.device, dtype=weight_dtype)
    accelerator.print(f"Load main model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")

    if siglip_model is not None:
        siglip_model.to(accelerator.device, dtype=weight_dtype)
        accelerator.print(f"Load siglip model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")

    vae.requires_grad_(False)
    text_encoder_cls_one.requires_grad_(False)
    text_encoder_cls_two.requires_grad_(False)
    lvlm_model.requires_grad_(False)
    lvlm_model.denoise_tower.requires_grad_(False)
    if siglip_model is not None:
        siglip_model.requires_grad_(False)


    if args.training_config.gradient_checkpointing:
        # lvlm_model._set_gradient_checkpointing()
        lvlm_model.denoise_tower.denoiser.enable_gradient_checkpointing()

    # Setup model saving and loading
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(accelerator.unwrap_model(model), model_class):
                    accelerator.unwrap_model(model).save_pretrained(
                        os.path.join(output_dir, "univa"),
                    )
                    processor.save_pretrained(
                        os.path.join(output_dir, "univa"),
                    )
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
                if weights:
                    weights.pop()
        if args.training_config.ema_deepspeed_config_file is not None:
            ema_model.save_pretrained(os.path.join(output_dir, "model_ema"))
            if accelerator.is_main_process:
                processor.save_pretrained(os.path.join(output_dir, "model_ema"))


    accelerator.register_save_state_pre_hook(save_model_hook)
    # accelerator.register_load_state_pre_hook(load_model_hook)

 
    for name, param in lvlm_model.named_parameters():
        if 'denoise_tower.denoise_projector' in name:
            param.requires_grad_(False)



    if args.model_config.pretrained_mlp2_path is not None:
        pretrained_mlp2 = torch.load(args.model_config.pretrained_mlp2_path)
        if accelerator.is_main_process:
            accelerator.print(f'Load {[k for k in pretrained_mlp2.keys()]} from {args.model_config.pretrained_mlp2_path}')
        msg = lvlm_model.load_state_dict(pretrained_mlp2, strict=False)
        assert len(msg[1]) == 0, msg


    if args.model_config.only_tune_mlp2:
        lvlm_model.requires_grad_(False)
        if args.model_config.only_tune_mlp2:
            for name, param in lvlm_model.named_parameters():
                if 'denoise_tower.denoise_projector' in name:
                    param.requires_grad_(True)
    else:
        if args.model_config.flux_train_layer_idx is not None:
            trainable_components = get_trainable_params(
                layers_to_train=args.model_config.flux_train_layer_idx, 
                only_img_branch=args.model_config.only_tune_image_branch, 
            )
            for name, module in lvlm_model.named_modules():
                if check_param_is_in_components(
                    name, trainable_components
                ):
                    module.requires_grad_(True)

    if args.model_config.with_tune_mlp2:
        for name, param in lvlm_model.named_parameters():
            if 'denoise_tower.denoise_projector' in name:
                param.requires_grad_(True)


    # =======================================================================================================
    # STEP 6: Create EMAModel
    if args.training_config.ema_deepspeed_config_file is not None:
        ema_model_state_dict = lvlm_model.state_dict()
        with open(args.training_config.ema_deepspeed_config_file, 'r') as f:
            ds_config = json.load(f)
        ema_model = create_ema_model(
            accelerator, args, resume_checkpoint_path, model_cls=model_class, model_config=lvlm_model.config, 
            ema_model_state_dict=ema_model_state_dict, ds_config=ds_config
            )

        accelerator.print(f"Load ema model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB")
    # =======================================================================================================
    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    trainable_names = []
    trainable_params = []
    for name, param in lvlm_model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            trainable_names.append(name)
    if accelerator.is_main_process:
        with open("trainable_params.txt", "w") as f:
            for name in trainable_names:
                f.write(f"{name}\n")
                print(f"{name}.requires_grad: True")
        accelerator.print("Trainable params:", len(trainable_params))
    if use_deepspeed_optimizer:
        from accelerate.utils import DummyOptim

        optimizer = DummyOptim(
            trainable_params,
            lr=args.training_config.learning_rate,
            betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
            eps=args.training_config.adam_epsilon,
        )
    else:
        if args.training_config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=args.training_config.learning_rate,
                betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
                eps=args.training_config.adam_epsilon,
                weight_decay=args.training_config.adam_weight_decay,
            )
        elif args.training_config.optimizer.lower() == "prodigy":
            try:
                import prodigyopt
            except ImportError:
                raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

            if args.training_config.learning_rate <= 0.1:
                raise ValueError(
                    "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
                )

            optimizer = prodigyopt.Prodigy(
                trainable_params,
                betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
                beta3=args.training_config.prodigy_beta3,
                weight_decay=args.training_config.adam_weight_decay,
                eps=args.training_config.adam_epsilon,
                decouple=args.training_config.prodigy_decouple,
                use_bias_correction=args.training_config.prodigy_use_bias_correction,
                safeguard_warmup=args.training_config.prodigy_safeguard_warmup,
                d_coef=args.training_config.prodigy_d_coef,
            )


    # Load dataset
    prompter = PROMPT_TYPE[dataset_type]()

    anchor_pixels = args.dataset_config.height * args.dataset_config.width
    resize_lambda = transforms.Lambda(
        lambda img: transforms.Resize(
            dynamic_resize(
                img.shape[1], img.shape[2], args.dataset_config.anyres, anchor_pixels), 
                # img.shape[1], img.shape[2], 'any_1ratio', anchor_pixels), 
                interpolation=transforms.InterpolationMode.BICUBIC
            )(img)
    )
    transform = transforms.Compose([
            resize_lambda,
            transforms.Normalize([0.5], [0.5]),
        ]
    )


    data_collator = DataCollator(tokenizer=lvlm_tokenizer, padding_side=args.dataset_config.padding_side)
    dataset = dataset_class(
        dataset_type=dataset_type, 
        data_txt=args.dataset_config.data_txt,
        transform=transform, 
        tokenizer=lvlm_tokenizer,
        prompter=prompter,
        image_processor=image_processor,
        processor=processor,
        min_pixels=args.dataset_config.min_pixels,
        max_pixels=args.dataset_config.max_pixels,
        image_token_length=lvlm_model.config.image_token_length,
        only_generated_task=True,
        drop_prompt_rate=args.training_config.drop_condition_rate,
        joint_ref_feature=args.model_config.joint_ref_feature, 
        anyres=args.dataset_config.anyres, 
        mask_weight_type=args.training_config.mask_weight_type, 
        siglip_processor=siglip_processor, 
        ocr_enhancer=args.dataset_config.ocr_enhancer, 
        random_data=args.dataset_config.random_data,
        use_step1x_preprocessing=True,  # Disable Step1X temporarily due to tokenizer issues
    )
    lvlm_model.config.image_token_id = dataset.image_token_id
    lvlm_model.config.image_begin_token_id = dataset.image_begin_token_id
    lvlm_model.config.image_end_token_id = dataset.image_end_token_id

    # Create dataloader
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.dataset_config.batch_size,
        shuffle=True,
        pin_memory=args.dataset_config.pin_memory,
        num_workers=args.dataset_config.num_workers,
        collate_fn=data_collator,
        prefetch_factor=None if args.dataset_config.num_workers == 0 else 4, 
        # prefetch_factor=None, 
        # persistent_workers=True, 
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.training_config.gradient_accumulation_steps
    )
    if args.training_config.max_train_steps is None:
        args.training_config.max_train_steps = (
            args.training_config.num_train_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.training_config.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.training_config.max_train_steps
            * accelerator.num_processes,
            num_warmup_steps=args.training_config.lr_warmup_steps
            * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.training_config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.training_config.lr_warmup_steps
            * accelerator.num_processes,
            num_training_steps=args.training_config.max_train_steps
            * accelerator.num_processes,
            num_cycles=args.training_config.lr_num_cycles,
            power=args.training_config.lr_power,
        )

    # Prepare training
    device_placement = None
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        device_placement = [True, True, True, False]
    lvlm_model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        lvlm_model, optimizer, lr_scheduler, train_dataloader, 
        device_placement=device_placement
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.training_config.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.training_config.max_train_steps = (
            args.training_config.num_train_epochs * num_update_steps_per_epoch
        )
    # Afterwards we recalculate our number of training epochs
    args.training_config.num_train_epochs = math.ceil(
        args.training_config.max_train_steps / num_update_steps_per_epoch
    )
    print('OmegaConf.to_container(args, resolve=True)', OmegaConf.to_container(args, resolve=True))
    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.training_config.wandb_project,
            init_kwargs={"wandb": {"name": args.training_config.wandb_name}},
            config=OmegaConf.to_container(args, resolve=True)
        )

    total_batch_size = (
        args.dataset_config.batch_size
        * accelerator.num_processes
        * args.training_config.gradient_accumulation_steps
    )

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num examples = {len(train_dataloader)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num Epochs = {args.training_config.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.dataset_config.batch_size}")
    accelerator.print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    accelerator.print(
        f"  Gradient Accumulation steps = {args.training_config.gradient_accumulation_steps}"
    )
    accelerator.print(f"  Total optimization steps = {args.training_config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Resume from checkpoint
    if resume_checkpoint_path is not None:
        accelerator.load_state(resume_checkpoint_path)
        first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.training_config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [
        text_encoder_cls_one,
        text_encoder_cls_two,
    ]
    empty_t5_prompt_embeds, empty_pooled_prompt_embeds = encode_prompt(
        text_encoders,
        tokenizers,
        prompt="",
        max_sequence_length=256,
        device=accelerator.device,
        num_images_per_prompt=1,
    )
    empty_pooled_prompt_embeds = empty_pooled_prompt_embeds.repeat(
        args.dataset_config.batch_size, 1
    )

    if args.training_config.drop_t5_rate == 1.0:
        del text_encoders, text_encoder_cls_one, text_encoder_cls_two
        free_memory()

    prof = None
    if args.training_config.profile_out_dir is not None:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
                ], 
            schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.training_config.profile_out_dir),
            profile_memory=True,
            with_stack=True,
            record_shapes=True
            )
    

    latent_image_ids_dict = {}
    for epoch in range(first_epoch, args.training_config.num_train_epochs):
        lvlm_model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(lvlm_model):
                generated_image = batch["generated_image"]
                if isinstance(generated_image, list):
                    assert args.dataset_config.batch_size != 1
                    generated_image = [gen_img.to(
                            accelerator.device, dtype=vae.dtype, non_blocking=True
                        ) for gen_img in generated_image]
                else:
                    generated_image = generated_image.to(
                        accelerator.device, dtype=vae.dtype, non_blocking=True
                    )

                input_ids = batch["input_ids"].to(
                    accelerator.device, non_blocking=True
                )
                attention_mask = batch["attention_mask"].to(
                    accelerator.device, non_blocking=True
                )
                pixel_values = batch["pixel_values"].to(
                    accelerator.device, dtype=weight_dtype, non_blocking=True
                ) if batch["pixel_values"] is not None else None
                image_position = batch["image_position"]
                image_grid_thw = batch["image_grid_thw"].to(
                    accelerator.device, non_blocking=True
                ) if batch["image_grid_thw"] is not None else None
                prompts = batch["prompts"]  # the value of last turn , which is instruction
                ref_pixel_values = batch["ref_pixel_values"]
                area_mask_weights = batch["weights"]
                pil_pixel_values = batch["pil_pixel_values"]

                if args.training_config.drop_t5_rate <= random.random():
                    with torch.no_grad():
                        t5_prompt_embeds, _ = encode_prompt(
                            [None, text_encoder_cls_two],
                            tokenizers,
                            prompt=prompts,  # the value of last turn, which is instruction
                            max_sequence_length=256,
                            device=accelerator.device,
                            num_images_per_prompt=1,
                        )
                else:
                    t5_prompt_embeds = None
                
                # Process reference images for FluxKontext style VAE conditioning
                # `ref_pixel_values` from the dataloader now contains the correctly transformed
                # images for VAE conditioning. We'll load them into `condition_pixel_values`
                # which is the variable used by the downstream pipeline logic.
                condition_pixel_values = batch.get("ref_pixel_values")
                
                if condition_pixel_values is not None and len(condition_pixel_values) > 0:

                    if condition_pixel_values.ndim == 5:
                        b, n, c, h, w = condition_pixel_values.shape
                        condition_pixel_values = condition_pixel_values.view(b * n, c, h, w)
                    condition_pixel_values = condition_pixel_values.to(accelerator.device, dtype=torch.float32)
                    
                    with torch.no_grad():
                        latents = vae.encode(condition_pixel_values).latent_dist.sample()

                else:
                    condition_pixel_values = None
                # print('generated_image.shape', generated_image.shape)
                # VAE encode
                def vae_encode(x):
                    with torch.no_grad():
                        model_input = vae.encode(x).latent_dist.sample()
                    model_input = (
                        model_input - vae.config.shift_factor
                    ) * vae.config.scaling_factor
                    model_input = model_input
                    return model_input



                denoiser_attention_mask = None
                weight_mask = None
                unpad_model_input = None
                if isinstance(generated_image, list):
                    assert args.dataset_config.batch_size != 1
                    # import ipdb;ipdb.set_trace()
                    unpad_model_input = [vae_encode(x) for x in generated_image]
                    denoiser_attention_mask = [torch.ones_like(x, device=x.device, dtype=x.dtype) for x in unpad_model_input]
                    model_input, denoiser_attention_mask = pad_x_and_mask(unpad_model_input, denoiser_attention_mask)
                    weight_mask = denoiser_attention_mask.detach().clone()
                    denoiser_attention_mask = F.max_pool2d(denoiser_attention_mask, kernel_size=2, stride=2).bool()
                    # import ipdb;ipdb.set_trace()
                    denoiser_attention_mask = denoiser_attention_mask.flatten(-2)
                else:
                    model_input = vae_encode(generated_image)

                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

                if model_input.shape in latent_image_ids_dict:
                    latent_image_ids = latent_image_ids_dict[model_input.shape]
                else:
                    latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                        model_input.shape[0],
                        model_input.shape[2] // 2,
                        model_input.shape[3] // 2,
                        accelerator.device,
                        weight_dtype,
                    )
                    latent_image_ids_dict[model_input.shape] = latent_image_ids

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]


                if args.training_config.discrete_timestep:
                    # Sample a random timestep for each image
                    # for weighting schemes where we sample timesteps non-uniformly
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.training_config.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.training_config.logit_mean,
                        logit_std=args.training_config.logit_std,
                        mode_scale=args.training_config.mode_scale,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(
                        device=model_input.device, non_blocking=True
                    )

                    # Add noise according to flow matching.
                    # zt = (1 - texp) * x + texp * z1
                    sigmas = get_sigmas(
                        timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
                    )
                else:
                    def calculate_shift(
                        image_seq_len,
                        base_seq_len: int = 256,
                        max_seq_len: int = 4096,
                        base_shift: float = 0.5,
                        max_shift: float = 1.16,
                    ):
                        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
                        b = base_shift - m * base_seq_len
                        mu = image_seq_len * m + b
                        return mu

                    def apply_flux_schedule_shift(sigmas, noise):
                        # Resolution-dependent shifting of timestep schedules as per section 5.3.2 of SD3 paper
                        # Resolution-dependent shift value calculation used by official Flux inference implementation
                        image_seq_len = (noise.shape[-1] * noise.shape[-2]) // 4
                        mu = calculate_shift(
                            image_seq_len,
                            noise_scheduler_copy.config.base_image_seq_len,
                            noise_scheduler_copy.config.max_image_seq_len,
                            noise_scheduler_copy.config.base_shift,
                            noise_scheduler_copy.config.max_shift,
                        )
                        shift = math.exp(mu)
                        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
                        return sigmas
    
                    sigmas = torch.sigmoid(
                        1.0 * torch.randn((bsz,), device=model_input.device, dtype=torch.float32)
                    )
                    sigmas = apply_flux_schedule_shift(sigmas, noise)
                    timesteps = sigmas * 1000.0  # rescale to [0, 1000.0)
                    while sigmas.ndim < model_input.ndim:
                        sigmas = sigmas.unsqueeze(-1)
                # print(f'STEP[{global_step}]-RANK[{accelerator.process_index}], sigmas={sigmas}, noise: max {noise.max()}, min {noise.min()}, mean {noise.mean()}, std {noise.std()}')
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # Use FluxKontextPipeline for VAE latent processing
                batch_size = model_input.shape[0]
                num_channels_latents = model_input.shape[1]
                height = model_input.shape[2] 
                width = model_input.shape[3]
                # `flux_pipeline` is already instantiated; do **not** overwrite
                # it here. Just keep a reference for clarity.
                # flux_pipeline = FluxKontextPipeline
                # Convert to latents and prepare with FluxKontext pipeline method
                if flux_pipeline is not None and condition_pixel_values is not None:
                    # Use FluxKontext prepare_latents to handle VAE latent concatenation
                    # NOTE: It returns UNPACKED target latents but PACKED image_latents. We must pack the target latents manually.
                    unpacked_latents, image_latents, latent_ids_for_target, image_ids_for_cond = flux_pipeline.prepare_latents(
                        image=condition_pixel_values,
                        batch_size=batch_size,
                        num_channels_latents=num_channels_latents,
                        height=height * vae_scale_factor,  # Scale back to pixel space
                        width=width * vae_scale_factor,
                        dtype=weight_dtype,
                        device=accelerator.device,
                        generator=None,
                        latents=noisy_model_input,
                    )
                    
                    packed_target_latents = flux_pipeline._pack_latents(
                        unpacked_latents,
                        batch_size=batch_size,
                        num_channels_latents=num_channels_latents,
                        height=height,
                        width=width,
                    )

                    # Combine latents and image latents as in FluxKontext
                    if image_latents is not None:
                        packed_noisy_model_input = torch.cat([packed_target_latents, image_latents], dim=1)
                        latent_image_ids = torch.cat([latent_ids_for_target, image_ids_for_cond], dim=0)
                    else:
                        packed_noisy_model_input = packed_target_latents
                        latent_image_ids = latent_ids_for_target
                else:
                    # Fall back to original packing without conditioning
                    packed_noisy_model_input = flux_pipeline._pack_latents(
                        noisy_model_input,
                        batch_size=batch_size,
                        num_channels_latents=num_channels_latents,
                        height=height,
                        width=width,
                    ) if flux_pipeline else FluxKontextPipeline._pack_latents(
                        noisy_model_input,
                        batch_size=batch_size,
                        num_channels_latents=num_channels_latents,
                        height=height,
                        width=width,
                    )
                    latent_image_ids = flux_pipeline._prepare_latent_image_ids(
                        batch_size, height // 2, width // 2, accelerator.device, weight_dtype
                    ) if flux_pipeline else FluxKontextPipeline._prepare_latent_image_ids(
                        batch_size, height // 2, width // 2, accelerator.device, weight_dtype
                    )
                # print(f'STEP[{global_step}]-RANK[{accelerator.process_index}], noisy_model_input={noisy_model_input.dtype}, packed_noisy_model_input={packed_noisy_model_input.dtype}')

                # guidance = torch.tensor(
                #     [args.model_config.guidance_scale], device=accelerator.device
                # )
                # guidance = guidance.expand(model_input.shape[0])


                guidance = torch.full(
                    (model_input.shape[0],),  
                    fill_value=args.model_config.guidance_scale,
                    device=accelerator.device,
                )


                ref_features_for_vlm = None

                model_pred = lvlm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    pixel_values=pixel_values,
                    image_position=image_position,
                    image_grid_thw=image_grid_thw, 
                    output_type="denoise_model_pred",
                    only_use_t5=args.model_config.only_use_t5, 
                    ref_features_for_vlm=ref_features_for_vlm,
                    vlm_residual_image_factor=args.model_config.vlm_residual_image_factor, 
                    # Simplified: no VAE conditioning through LVLM
                    denoiser_kwargs={
                        "prefix_prompt_embeds": t5_prompt_embeds,
                        "hidden_states": packed_noisy_model_input.to(weight_dtype),
                        "timestep": (timesteps / 1000).to(weight_dtype),
                        "guidance": guidance,
                        "pooled_projections": empty_pooled_prompt_embeds,
                        "img_ids": latent_image_ids,
                        "joint_attention_kwargs": dict(attention_mask=denoiser_attention_mask) if denoiser_attention_mask is not None else {}
                    },
                )
                # Extract only target portion for loss computation if we concatenated reference features
                if args.model_config.joint_ref_feature and condition_pixel_values is not None:
                    noisy_model_input_len = (model_input.shape[2] // 2) * (model_input.shape[3] // 2)
                    model_pred = model_pred[:, :noisy_model_input_len]
                model_pred = FluxKontextPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                target = noise - model_input
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.training_config.weighting_scheme,
                    sigmas=sigmas,
                ) if not args.training_config.sigmas_as_weight else sigmas

                def save_fig(matrix, name):
                    import numpy as np
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(4, 4), dpi=100)
                    plt.imshow(matrix, interpolation='nearest', aspect='auto')
                    plt.colorbar()    
                    plt.savefig(f'{name}.png',
                                dpi=300,                   
                                bbox_inches='tight'        
                            )
                
                area_mask_weights_bak = None
                if args.training_config.mask_weight_type is not None:
                    if isinstance(area_mask_weights, list):
                        assert args.dataset_config.batch_size != 1
                        area_mask_weights_bak = area_mask_weights
                        area_mask_weights = [
                            F.interpolate(
                                w.to(
                                    device=accelerator.device, non_blocking=True
                                ), 
                                size=unpad_model_input[unpad_i].shape[-2:] if unpad_model_input is not None else model_pred.shape[-2:], 
                                mode='nearest'
                                ) for unpad_i, w in enumerate(area_mask_weights)
                            ]
                        max_h, max_w = model_pred.shape[-2:]
                        area_mask_weights, _ = pad_x_and_mask(area_mask_weights, max_h=max_h, max_w=max_w)
                        
                        # save_fig(area_mask_weights[0][0].detach().float().cpu().numpy(), 'pad_x_and_mask_area_mask_weights[0][0]')
                        # save_fig(area_mask_weights[1][0].detach().float().cpu().numpy(), 'pad_x_and_mask_area_mask_weights[1][0]')
                        assert area_mask_weights.shape[-2:] == model_pred.shape[-2:]
                    else:
                        area_mask_weights = area_mask_weights.to(
                            device=accelerator.device, non_blocking=True
                            )
                        assert weighting.ndim == area_mask_weights.ndim
                        if not area_mask_weights.shape[-2:] == model_pred.shape[-2:]:
                            area_mask_weights = F.interpolate(area_mask_weights, size=model_pred.shape[-2:], mode='nearest')
                    weighting = weighting.float() * area_mask_weights.float()

                if weight_mask is not None:
                    # save_fig(weight_mask[0][0].detach().float().cpu().numpy(), 'weight_mask[0][0]')
                    # save_fig(weight_mask[1][0].detach().float().cpu().numpy(), 'weight_mask[1][0]')
                    weighting = weighting.float() * weight_mask.float()
                # save_fig(weighting[0][0].detach().float().cpu().numpy(), 'weighting[0][0]')
                # save_fig(weighting[1][0].detach().float().cpu().numpy(), 'weighting[1][0]')
                loss = (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1)
                
                # if area_mask_weights_bak is not None:
                    # import ipdb;ipdb.set_trace()
                if weight_mask is not None:
                    assert args.dataset_config.batch_size != 1
                    loss = loss.sum() / weight_mask.sum() / model_pred.shape[1]
                else:
                    loss = loss.mean()
                avg_loss_list = accelerator.gather(loss)
                # if loss > 1.6:
                #     print(f'HIGH LOSS!!! STEP[{global_step}]-RANK[{accelerator.process_index}] ERROR: loss={loss.detach().item()}, sigmas={sigmas}, prompts={prompts}, model_pred: {model_pred.shape}, max {model_pred.max()} min {model_pred.max()} mean {model_pred.max()} std {model_pred.max()}')
                #     loss = loss * 0.0
                accelerator.backward(loss)
                # import ipdb;ipdb.set_trace()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params, args.training_config.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:        
                if args.training_config.ema_deepspeed_config_file is not None and global_step % args.training_config.ema_update_freq == 0:
                    ema_model.step(lvlm_model.parameters())

                progress_bar.update(1)
                global_step += 1

                if (
                    accelerator.is_main_process
                    or accelerator.distributed_type == DistributedType.DEEPSPEED
                ) and global_step % args.training_config.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.training_config.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.training_config.output_dir)
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")
                        ]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1])
                        )
                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if (
                            len(checkpoints)
                            >= args.training_config.checkpoints_total_limit
                        ):
                            num_to_remove = (
                                len(checkpoints)
                                - args.training_config.checkpoints_total_limit
                                + 1
                            )
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            accelerator.print(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            accelerator.print(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}"
                            )
                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    args.training_config.output_dir,
                                    removing_checkpoint,
                                )
                                shutil.rmtree(removing_checkpoint)
                    save_path = os.path.join(
                        args.training_config.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)
                    # Save denoise projector if trained
                    if args.model_config.only_tune_mlp2 or args.model_config.with_tune_mlp2:
                        keys_to_match = ['denoise_tower.denoise_projector']
                        weight_mlp2 = get_mm_adapter_state_maybe_zero_3(lvlm_model.named_parameters(), keys_to_match)
                        weight_mlp2 = {k.replace('module.', ''): v for k, v in weight_mlp2.items()}
                        torch.save(weight_mlp2, os.path.join(save_path, 'denoise_projector.bin'))
                        accelerator.print("Saved denoise_projector")
                    accelerator.print(f"Saved state to {save_path}")

                # num_machines = accelerator.state.num_processes // 8
                # node_rank = accelerator.process_index // 8
                # print(f'node_rank: {node_rank}, num_machines: {num_machines}')
                # num_run_per_node = 8 // num_run_per_node
                if global_step % args.training_config.validation_steps == 0:
                    base_eval_prompts, base_eval_image_paths, base_phase_names = build_validation_info(args)
                    # if len(base_eval_prompts) > 0:
                        # base_eval_prompts = base_eval_prompts[node_rank::num_machines]
                        # base_eval_image_paths = base_eval_image_paths[node_rank::num_machines]
                        # base_phase_names = base_phase_names[node_rank::num_machines]
                        # if args.training_config.ema_deepspeed_config_file is not None:
                        #     ema_state_dict = gather_zero3ema(accelerator, ema_model)
                    
                    
                # if accelerator.process_index % 8 == 0 and global_step % args.training_config.validation_steps == 0:
                # if accelerator.is_local_main_process and global_step % args.training_config.validation_steps == 0:
                if accelerator.is_main_process and global_step % args.training_config.validation_steps == 0:
                    # print(f'validation rank: {accelerator.process_index}', *[i+'\n------------\n' for i in base_eval_prompts])
                    if len(base_eval_prompts) > 0:
                        # if args.training_config.ema_deepspeed_config_file is not None:
                        #     # ema_state_dict = gather_zero3ema(accelerator, ema_model)
                        #     ema_model.store(lvlm_model.parameters())
                        #     # ema_model.copy_to(lvlm_model.parameters())
                        #     # print('ema_state_dict.keys()', list(ema_state_dict.keys())[0])
                        #     # print('lvlm_model.state_dict().keys()', list(lvlm_model.state_dict().keys())[0])
                        #     lvlm_model.load_state_dict({'module.'+k: v for k, v in ema_state_dict.items()})

                        unwrapped_lvlm_model = accelerator.unwrap_model(lvlm_model)
                        pipe = FluxKontextPipeline.from_pretrained(
                            args.model_config.pretrained_denoiser_name_or_path,
                            transformer=None,
                            vae=vae,
                            text_encoder=None,
                            text_encoder_2=None,
                            torch_dtype=weight_dtype,
                        )
                        pipe.to(accelerator.device)
                        pipe.transformer = unwrapped_lvlm_model

                    def warpped_log_validation(
                            prompt, 
                            image_path, 
                            text_encoders, 
                            phase_name, 
                            only_use_t5, 
                            joint_ref_feature, 
                            joint_ref_feature_as_condition, 
                            ):
                        
                        log_validation(
                            accelerator=accelerator,
                            prompt=prompt,
                            image_path=image_path,
                            image_processor=image_processor,
                            args=args,
                            vae=vae,
                            lvlm_model=lvlm_model,
                            tokenizer=lvlm_tokenizer,
                            prompter=prompter,
                            pooled_prompt_embeds=empty_pooled_prompt_embeds,
                            negative_pooled_prompt_embeds=empty_pooled_prompt_embeds,
                            weight_dtype=weight_dtype,
                            processor=processor,
                            min_pixels=args.dataset_config.min_pixels,
                            max_pixels=args.dataset_config.max_pixels,
                            dataset_type=dataset_type, 
                            _process_image_token=dataset_class._process_image_token, 
                            _load_image=dataset_class._load_image, 
                            text_encoders=text_encoders, 
                            tokenizers=tokenizers,
                            negative_t5_prompt_embeds=empty_t5_prompt_embeds,
                            vae_image_transform=transform,
                            anyres=args.dataset_config.anyres, 
                            phase_name=phase_name, 
                            only_use_t5=only_use_t5, 
                            joint_ref_feature=joint_ref_feature, 
                            joint_ref_feature_as_condition=joint_ref_feature_as_condition, 
                            siglip_processor=siglip_processor, 
                            siglip_model=siglip_model, 
                            pipe=pipe, 
                            unwrapped_lvlm_model=unwrapped_lvlm_model, 
                        )
                    
                    if len(base_eval_prompts) > 0:
                        for i, j, k in zip(base_eval_prompts, base_eval_image_paths, base_phase_names):
                            if args.model_config.only_use_t5:
                                warpped_log_validation(
                                    prompt=i, 
                                    image_path=j, 
                                    text_encoders=[None, text_encoders[1]],  # we do not need clip
                                    phase_name=k.replace('vlm', 't5'), 
                                    only_use_t5=True, 
                                    joint_ref_feature=False, 
                                    joint_ref_feature_as_condition=False, 
                                )
                            else:
                                warpped_log_validation(
                                    prompt=i, 
                                    image_path=j, 
                                    text_encoders=[None, text_encoders[1]] if args.training_config.drop_t5_rate < 1.0 else None,  # we do not need clip
                                    phase_name=('t5-'+k) if args.training_config.drop_t5_rate < 1.0 else k, 
                                    only_use_t5=False, 
                                    joint_ref_feature=False, 
                                    joint_ref_feature_as_condition=False, 
                                )



                    if len(base_eval_prompts) > 0:
                        
                        # if args.training_config.ema_deepspeed_config_file is not None:
                        #     ema_model.restore(lvlm_model.parameters())
                        del pipe
                        free_memory()

            if prof is not None:
                prof.step()

            log_interval = 1
            if global_step % log_interval == 0:
                logs = {
                    # "loss": loss.detach().item(),
                    "loss": avg_loss_list.mean().detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if args.training_config.optimizer.lower() == "prodigy":
                    d = optimizer.param_groups[0]['d']
                    beta1, beta2 = optimizer.param_groups[0]['betas']
                    k = optimizer.param_groups[0]['k']
                    lr = max(group['lr'] for group in optimizer.param_groups)
                    d_lr = d * lr
                    bias_correction = ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))
                    d_lr_bias_corr = d_lr * bias_correction
                    prodigy_log = {"d*lr": d_lr, "d*lr*bias_corr": d_lr_bias_corr}
                    logs.update(prodigy_log)
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= args.training_config.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()

    
@torch.no_grad()
def log_validation(
    accelerator: Accelerator,
    prompt: str,
    args: UnivaTrainingDenoiseConfig,
    vae: AutoencoderKL,
    lvlm_model: UnivaQwen2ForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompter: Qwen2Prompter,
    weight_dtype: torch.dtype,
    negative_t5_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    image_path: Optional[str] = None,
    image_processor: Optional[Callable] = None,
    processor: Optional[Callable] = None,
    max_pixels: int = 384*384,
    min_pixels: int = 384*384,
    dataset_type: str = 'llava',
    _process_image_token: Optional[Callable] = None,
    _load_image: Optional[Callable] = None,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    text_encoders = None,
    tokenizers = None,
    joint_ref_feature: bool = False, 
    joint_ref_feature_as_condition: bool = False, 
    vae_image_transform: Optional[Callable] = None,
    ref_cfg: bool = True, 
    anyres: bool = False, 
    phase_name: Optional[str] = None, 
    only_use_t5: bool = False, 
    siglip_model: Optional[Callable] = None,
    siglip_processor: Optional[Callable] = None,
    pipe: Optional[Callable] = None,
    unwrapped_lvlm_model: Optional[Callable] = None,
):
    # unwrapped_lvlm_model = accelerator.unwrap_model(lvlm_model)

    image_token = SPACIAL_TOKEN[dataset_type]['image_token']
    image_begin_token = SPACIAL_TOKEN[dataset_type]['image_begin_token']
    image_end_token = SPACIAL_TOKEN[dataset_type]['image_end_token']

    prompt = prompt.replace('<image>', image_token)
    num_images = prompt.count(image_token)
    
    if image_path:
        assert image_processor is not None or processor is not None, (
            "image_processor or processor must be provided if image_path is provided"
        )
        assert image_token in prompt, f"prompt must have {image_token} if image_path is provided"

    if text_encoders is not None and tokenizers is not None:
        t5_prompt = prompt.replace(image_token, '').replace('\n', '')  # the value of last turn, which is instruction
        t5_prompt_embeds, _ = encode_prompt(
            text_encoders,
            tokenizers,
            prompt=t5_prompt, 
            max_sequence_length=256,
            device=accelerator.device,
            num_images_per_prompt=1,
        )
    else:
        assert not only_use_t5
        t5_prompt_embeds = None

    ocr_sentences = ''
    cur_i = 0
    if args.dataset_config.ocr_enhancer:
        image_path = [image_path] if isinstance(image_path, str) else image_path
        num_img = prompt.count(image_token)
        ocr_sentences = []
        for i in range(num_img):
            ocr_sentences.append(get_ocr_result(image_path[cur_i], cur_i))
            cur_i += 1
        ocr_sentences = '\n'.join(ocr_sentences)
    test_prompt = [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "user", "value": prompt + ocr_sentences},
    ]
    negative_prompt = [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "user", "value": "Generate an image."},
    ]  # ATTENTION: the negative prompt in LlavaDataset is hardcode.

    prompt = prompter(test_prompt)
    negative_prompt = prompter(negative_prompt)
    input_ids = tokenizer.batch_encode_plus(
        [prompt, negative_prompt], 
        padding="longest", return_tensors="pt", 
        padding_side=args.dataset_config.padding_side, 
    ).input_ids.to(accelerator.device)

    width, height = None, None
    vae_cond_hidden_states = None
    if image_path:
        image_path = [image_path] if isinstance(image_path, str) else image_path
        image_dict = _load_image(
            image_path, 
            max_pixels=max_pixels,  
            min_pixels=min_pixels, 
            processor=processor, 
            image_processor=image_processor, 
            image_token=image_token, 
            factor=1, 
            last_image=image_path[-1], 
            vae_image_transform=vae_image_transform,
            siglip_processor=None,  # No longer using SigLIP
        )
        image_token_lengths = image_dict['image_token_lengths']
        pixel_values = image_dict['pixel_values'].cuda() if image_dict['pixel_values'] is not None else None
        image_grid_thw = image_dict['image_grid_thw'].cuda() if image_dict['image_grid_thw'] is not None else None
        
        condition_pixel_values = image_dict.get('ref_pixel_values')
        pipeline_image = None
        if condition_pixel_values is not None and len(condition_pixel_values) > 0:

            if condition_pixel_values.ndim == 5:
                b, n, c, h, w = condition_pixel_values.shape
                condition_pixel_values = condition_pixel_values.view(b * n, c, h, w)
            condition_pixel_values = condition_pixel_values.to(accelerator.device, dtype=torch.float32)
            
            with torch.no_grad():
                latents = vae.encode(condition_pixel_values).latent_dist.sample()
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
            pipeline_image = latents.to(dtype=weight_dtype)          

        pil_pixel_values = image_dict['pil_pixel_values']
        
        # Process VAE conditioning for validation (if vae_cond_encoder is available)
        # Note: For validation, we might not have vae_cond_encoder accessible here
        # This would need to be passed as a parameter or handled differently
        # For now, we'll set it to None and handle it in the calling code if needed

        prompt_input_ids = input_ids[0].unsqueeze(0)
        prompt_input_ids, _, image_position = _process_image_token(
            prompt_input_ids,
            image_token_id=tokenizer.convert_tokens_to_ids(image_token),
            image_begin_token_id=tokenizer.convert_tokens_to_ids(image_begin_token),
            image_end_token_id=tokenizer.convert_tokens_to_ids(image_end_token),
            image_token_lengths=image_token_lengths,
        )
        image_position = [image_position]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [prompt_input_ids[0], input_ids[1]],
            padding_value=tokenizer.pad_token_id,
            batch_first=True,
            padding_side=args.dataset_config.padding_side, 
        )
        
        if pipeline_image is not None:

            latent_h, latent_w = pipeline_image.shape[-2:]
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)  
            height = latent_h * vae_scale_factor        # = pixel-space H
            width = latent_w * vae_scale_factor
        else:
            width, height = Image.open(image_path[-1]).size
            anchor_pixels = args.dataset_config.width * args.dataset_config.height
            height, width = dynamic_resize(height, width, anyres, anchor_pixels)
    else:
        pixel_values = None
        image_position = None
        image_grid_thw = None
        ref_pixel_values = None
        pipeline_image = None
        height, width = args.dataset_config.height, args.dataset_config.width


    generator = (
        torch.Generator(device=accelerator.device).manual_seed(
            args.training_config.seed,
        )
        if args.training_config.seed is not None
        else None
    )
    
    ref_features_for_vlm = None
    if not only_use_t5:
        # TODO: use attention_mask
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        lvlm_embeds = unwrapped_lvlm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,  # image degrade
            pixel_values=pixel_values,
            image_position=image_position,
            image_grid_thw=image_grid_thw, 
            ref_features_for_vlm=ref_features_for_vlm,
            vlm_residual_image_factor=args.model_config.vlm_residual_image_factor, 
            output_type="denoise_embeds",
        )
        prompt_embeds = lvlm_embeds[0].unsqueeze(0)
        negative_prompt_embeds = lvlm_embeds[1].unsqueeze(0)

    if not only_use_t5:
        if t5_prompt_embeds is not None:
            prompt_embeds = torch.concat([prompt_embeds, t5_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.concat([negative_prompt_embeds, negative_t5_prompt_embeds], dim=1)
    else:
        prompt_embeds = t5_prompt_embeds
        negative_prompt_embeds = negative_t5_prompt_embeds
        prompt = t5_prompt
        
    autocast_ctx = nullcontext()
    with autocast_ctx and unwrapped_lvlm_model.forward_denoiser_context():

        images = [
            pipe(
                image=pipeline_image,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds[0].unsqueeze(0),
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                height=height or args.dataset_config.height,
                width=width or args.dataset_config.width,
                _auto_resize=False,                        
                generator=generator,
                num_inference_steps=28,
                guidance_scale=3.5,
            ).images[0]
            for _ in range(args.training_config.num_validation_images)
        ]

    # if accelerator.is_local_main_process:
    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            phase_name = phase_name or "validation"
            if tracker.name == "wandb":
                tracker.log(
                    {
                        phase_name: [
                            wandb.Image(image, caption=f"{i}: {prompt}")
                            for i, image in enumerate(images)
                        ]
                    }
                )


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    schema = OmegaConf.structured(UnivaTrainingDenoiseConfig)
    conf = OmegaConf.merge(schema, config)
    # main(conf, attn_implementation='sdpa')
    main(conf, attn_implementation='flash_attention_2')
