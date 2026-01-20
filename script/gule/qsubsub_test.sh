#!/bin/bash

run_script="/work/xg24i002/x10041/lora-ns/script/gule/test.sh"

init_lora_weights_list=("eva" "corda" "lora_ga" "gaussian" "true" "olora" "pissa" "orthogonal" )
fit_init_lora_weights_list=("gaussian" "true" "olora" "orthogonal" )

TRAIN_CONFIG="/work/xg24i002/x10041/lora-ns/configs/gule/roberta-base/mnli.yaml"

qsub -v TRAIN_CONFIG="${TRAIN_CONFIG}",init_lora_weights="true",use_sr_trainer=0 \
 "${run_script}"

qsub -v TRAIN_CONFIG="${TRAIN_CONFIG}",init_lora_weights="true",use_sr_trainer=1 \
 "${run_script}"