#!/usr/bin/env bash
set -x
CFG=$1
CUDA_VISIBLE_DEVICES=$2
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NGPU=${#GPUS[@]}
echo "use gpu ids: $CUDA_VISIBLE_DEVICES num gpus: $NGPU"
NCCL_DEBUG=INFO
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=$2 python core/gdrn_modeling/main_gdrn.py \
    --config-file $CFG --num-gpus $NGPU  ${@:3}
