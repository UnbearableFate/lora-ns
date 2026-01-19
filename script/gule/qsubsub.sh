#!/bin/bash

run_script="/work/xg24i002/x10041/lora-ns/script/gule/run.sh"

init_lora_weights_list=("eva" "corda" "lora_ga" "gaussian" "true" "olora" "pissa" "orthogonal" )
fit_init_lora_weights_list=("gaussian" "true" "olora" "orthogonal" )

# qsub -v TRAIN_CONFIG="configs/gule/roberta-large-cola/fb-cola.yaml",init_lora_weights="eva" \
#  "${run_script}"

qsub -v TRAIN_CONFIG="configs/gule/roberta-large-cola/fb-cola.yaml",init_lora_weights="orthogonal",use_sr_trainer=1 \
 "${run_script}"