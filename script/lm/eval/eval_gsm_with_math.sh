#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -m abe

set -euo pipefail
cd /work/xg24i002/x10041/lora-ns

PYTHON_PATH="/work/xg24i002/x10041/lora-ns/.venv/bin/python"

HF_HOME="/work/xg24i002/x10041/hf_home"
HF_DATASETS_CACHE="/work/xg24i002/x10041/data"

export HF_HOME HF_DATASETS_CACHE
export CUDA_VISIBLE_DEVICES=0

export CC=/bin/gcc
export CXX=/bin/g++
export TRITON_CXX=/bin/g++

adapter_path= "/work/xg24i002/x10041/lora-ns/outputs/Qwen3-1.7B/Qwen3-1.7B_MetaMathQA_r16_a1_True_lora_s42_20251225_211730"
${PYTHON_PATH} eval_gsm8k.py \
  --model "Qwen/Qwen3-1.7B" \
  --adapter_path "$adapter_path" \
  --data_file "data/GSM8K_test.jsonl" \
  --batch_size 8 \
  --tensor_parallel_size 1 \
  --filepath_output results

${PYTHON_PATH} eval_math.py \
  --model "Qwen/Qwen3-1.7B" \
  --adapter_path "$adapter_path" \
  --data_file "data/MATH_test.jsonl" \
  --batch_size 8 \
  --tensor_parallel_size 1 \
  --filepath_output results