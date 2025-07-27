#!/bin/bash
export WANDB_MODE="online"
export WANDB_API_KEY=""

export TOKENIZERS_PARALLELISM=true

# 强制 NCCL 只用 lo 回环口，禁用 InfiniBand
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_PROCESSES=$((8 * WORLD_SIZE))

# NEED MODIFY in YAML:
  # data_txt
  # pretrained_lvlm_name_or_path: recommend use ema weight in stage1
  # ema_pretrained_lvlm_name_or_path: recommend use ema weight in stage1
  # pretrained_denoiser_name_or_path
  # pretrained_mlp2_path: recomment use ema weight in stage1
  # pretrained_siglip_mlp_path

accelerate launch \
  --config_file scripts/accelerate_configs/multi_node_example_zero2.yaml \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${RANK} \
  --num_machines ${WORLD_SIZE} \
  --num_processes ${NUM_PROCESSES} \
  train_denoiser.py \
  scripts/denoiser/flux_qwen2p5vl_7b_vlm_stage2_1024.yaml
