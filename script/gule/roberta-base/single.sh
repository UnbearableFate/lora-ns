#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -m abe

set -euo pipefail

WORKSPACE="/home/yu/workspace/lora-ns"
PYTHON_PATH="/home/yu/peft_playground/.venv/bin/python"
TRAIN_CONFIG="configs/gule/roberta-base/fb-sst2.yaml"

cd "${WORKSPACE}"

export ACCELERATE_CONFIG_FILE="${WORKSPACE}/accelerate_config/local_config.yaml"

#export HF_HOME="/work/xg24i002/x10041/hf_home"
#export HF_DATASETS_CACHE="/work/xg24i002/x10041/data"

init_methods=("True" "eva" "corda" "lora_ga" "gaussian" "true" "olora" "pissa" "orthogonal")
init_lora_weights_for_sr=("True" "gaussian" "olora" "orthogonal")
seeds=(11 23 37 43 57)

for init_method in "${init_methods[@]}"; do
    for seed in "${seeds[@]}"; do
        "${PYTHON_PATH}" train.py --config "${TRAIN_CONFIG}" --init_lora_weights "${init_method}" --seed "${seed}"
    done
done

for init_method in "${init_lora_weights_for_sr[@]}"; do
    for seed in "${seeds[@]}"; do
        "${PYTHON_PATH}" train.py --config "${TRAIN_CONFIG}" --init_lora_weights "${init_method}" --use_sr_trainer --seed "${seed}"
    done
done