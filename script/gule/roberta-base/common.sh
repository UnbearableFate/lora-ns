#!/bin/bash

set -euo pipefail

: "${WORKSPACE:=/home/yu/workspace/lora-ns}"
: "${PYTHON_PATH:=/home/yu/peft_playground/.venv/bin/python}"
: "${TRAIN_CONFIG:=configs/gule/roberta-base/fb-sst2.yaml}"

cd "${WORKSPACE}"

export ACCELERATE_CONFIG_FILE="${WORKSPACE}/accelerate_config/local_config.yaml"

#export HF_HOME="/work/xg24i002/x10041/hf_home"
#export HF_DATASETS_CACHE="/work/xg24i002/x10041/data"

seeds=(11 23 37 43 57)

run_train() {
    local init_lora_weights="$1"
    shift

    local extra_args=()
    if [[ "${1:-}" == "--use_sr_trainer" ]]; then
        extra_args+=(--use_sr_trainer)
        shift || true
    fi

    for seed in "${seeds[@]}"; do
        "${PYTHON_PATH}" train.py \
            --config "${TRAIN_CONFIG}" \
            --init_lora_weights "${init_lora_weights}" \
            "${extra_args[@]}" \
            --seed "${seed}"
    done
}

