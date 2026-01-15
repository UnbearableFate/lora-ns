#!/bin/bash

python eval_glue_lora.py \
  --config configs/gule/qwen3_1.7b_lora_sst2.yaml \
  --adapter_path outputs/Qwen3-1.7B/Qwen3-1.7B_glue_sst2_r8_a16_True_lora_s42_20260113_173157 \
  --split test