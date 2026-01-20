#!/bin/bash

run_script="/work/xg24i002/x10041/lora-ns/script/gule/run.sh"

init_lora_weights_list=("eva" "corda" "lora_ga" "gaussian" "true" "olora" "pissa" "orthogonal" )
fit_init_lora_weights_list=("gaussian" "true" "olora" "orthogonal" )

TRAIN_CONFIG="/work/xg24i002/x10041/lora-ns/configs/gule/roberta-base/fb-stsb.yaml"

qsub -v TRAIN_CONFIG="${TRAIN_CONFIG}",init_lora_weights="eva" \
 "${run_script}"

qsub -v TRAIN_CONFIG="${TRAIN_CONFIG}",init_lora_weights="lora_ga" \
 "${run_script}"