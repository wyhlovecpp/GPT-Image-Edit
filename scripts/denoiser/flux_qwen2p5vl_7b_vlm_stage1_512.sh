#!/bin/bash
export WANDB_MODE="online"
export WANDB_API_KEY=""

export TOKENIZERS_PARALLELISM=true

export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_PROCESSES=$((8 * WORLD_SIZE))


accelerate launch \
  --config_file scripts/accelerate_configs/multi_node_example_zero1.yaml \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${RANK} \
  --num_machines ${WORLD_SIZE} \
  --num_processes ${NUM_PROCESSES} \
  train_denoiser.py \
  scripts/denoiser/flux_qwen2p5vl_7b_vlm_stage1_512.yaml
