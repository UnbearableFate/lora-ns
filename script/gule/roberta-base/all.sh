#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -m abe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
    "kaiming.sh"
    "eva.sh"
    "corda.sh"
    "lora_ga.sh"
    "gaussian.sh"
    "olora.sh"
    "pissa.sh"
    "orthogonal.sh"
    "kaiming_sr.sh"
    "gaussian_sr.sh"
    "olora_sr.sh"
    "orthogonal_sr.sh"
)

submit_or_run() {
    local script_path="$1"
    if [[ "${SUBMIT:-0}" == "1" ]] && command -v qsub >/dev/null 2>&1; then
        qsub "${script_path}"
    else
        bash "${script_path}"
    fi
}

for script_name in "${scripts[@]}"; do
    submit_or_run "${SCRIPT_DIR}/${script_name}"
done
