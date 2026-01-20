#!/bin/bash

TRAIN_CONFIG="/work/xg24i002/x10041/lora-ns/configs/gule/roberta-base/rte.yaml"

run_script="/work/xg24i002/x10041/lora-ns/script/gule/run.sh"

init_lora_weights_list=("eva" "corda" "lora_ga" "gaussian" "true" "olora" "pissa" "orthogonal" )
fit_init_lora_weights_list=("gaussian" "true" "olora" "orthogonal" )

for init_lora_weights in "${init_lora_weights_list[@]}"; do
    qsub_output="$(qsub -v TRAIN_CONFIG="${TRAIN_CONFIG}",init_lora_weights="${init_lora_weights}" \
     "${run_script}")"
    qsub_outputs+=("${qsub_output}")
done

for init_lora_weights in "${fit_init_lora_weights_list[@]}"; do
    qsub_output="$(qsub -v TRAIN_CONFIG="${TRAIN_CONFIG}",init_lora_weights="${init_lora_weights}",use_sr_trainer=1 \
     "${run_script}")"
    qsub_outputs+=("${qsub_output}")
done

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="qsub_history_${timestamp}.log"

{
    echo "TRAIN_CONFIG: ${TRAIN_CONFIG}"
    echo "===== TRAIN_CONFIG CONTENT ====="
    cat "${TRAIN_CONFIG}"
    echo "===== QSUB OUTPUTS ====="
    printf '%s\n' "${qsub_outputs[@]}"
} > "${log_file}"
