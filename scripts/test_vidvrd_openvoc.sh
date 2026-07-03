#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"
gpu_id=${GPU_ID:-0}
batch_size_eval=${BATCH_SIZE_EVAL:-32}
ckpt=${CKPT_PATH:-../dataset/vidvrd/model/stage2_tcp.pth}
CUDA_VISIBLE_DEVICES=$gpu_id python test_openvoc.py \
    --batch_size_eval ${batch_size_eval} \
    --ckpt_path ${ckpt}
