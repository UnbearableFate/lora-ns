#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -m abe

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"

TRAIN_CONFIG="configs/gule/roberta-base/fb-sst2.yaml"

export ACCELERATE_CONFIG_FILE="/work/xg24i002/x10041/lora-ns/accelerate_config/local_config.yaml"

PYTHON_PATH="/work/xg24i002/x10041/lora-ns/.venv/bin/python"

export HF_HOME="/work/xg24i002/x10041/hf_home"
export HF_DATASETS_CACHE="/work/xg24i002/x10041/data"

init_methods=()

"${PYTHON_PATH}" train.py --config "${TRAIN_CONFIG}" --seed 11
"${PYTHON_PATH}" train.py --config "${TRAIN_CONFIG}" --seed 23
"${PYTHON_PATH}" train.py --config "${TRAIN_CONFIG}" --seed 37