#!/bin/bash
#PBS -q short-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=05:00:00
#PBS -j oe
#PBS -m abe

set -euo pipefail

: "${WORKSPACE:=/work/xg24i002/x10041/lora-ns}"
: "${PYTHON_PATH:=/work/xg24i002/x10041/lora-ns/.venv/bin/python}"
: "${TRAIN_CONFIG:=configs/gule/roberta-base/fb-sst2.yaml}"
: "${init_lora_weights:=True}"
: "${use_sr_trainer:=0}"

cd "${WORKSPACE}"

export ACCELERATE_CONFIG_FILE="${WORKSPACE}/accelerate_config/local_config.yaml"
export HF_HOME="/work/xg24i002/x10041/hf_home"
export HF_DATASETS_CACHE="/work/xg24i002/x10041/data"

seeds=(11 23 37 43 57)

is_true() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

run_train() {
    for seed in "${seeds[@]}"; do
        local -a extra_args=()
        if [[ -n "${init_lora_weights}" ]]; then
            extra_args+=(--init_lora_weights "${init_lora_weights}")
        fi
        if is_true "${use_sr_trainer}"; then
            extra_args+=(--use_sr_trainer)
        fi

        "${PYTHON_PATH}" train.py \
            --config "${TRAIN_CONFIG}" \
            "${extra_args[@]}" \
            --seed "${seed}"
    done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    run_train
fi