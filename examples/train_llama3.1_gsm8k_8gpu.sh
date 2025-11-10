#!/bin/bash
# Training script for Llama-3.1-8B on GSM8K
# Distributed training on 8×H200 GPUs

set -e  # Exit on error

echo "================================================"
echo "Llama-3.1-8B GSM8K Training"
echo "================================================"
echo ""
echo "Configuration:"
echo "  - Model: meta-llama/Llama-3.1-8B"
echo "  - Dataset: GSM8K (~7.5K samples)"
echo "  - GPUs: 8×H200"
echo "  - Training Mode: DDP (DistributedDataParallel)"
echo "  - LoRA Rank: 32"
echo "  - Init Method: PiSSA + RSLoRA"
echo "  - Trainer: SpectralRefactorTrainer"
echo ""
echo "Expected Training Time: ~1-2 hours"
echo "================================================"
echo ""

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. CUDA required."
    exit 1
fi

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Check number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -lt 8 ]; then
    echo "⚠️  Warning: Expected 8 GPUs, found $NUM_GPUS"
    echo "   Training will proceed with $NUM_GPUS GPUs"
    echo ""
fi

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO  # Enable NCCL debug logging
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=eth0  # Adjust based on your network interface
export OMP_NUM_THREADS=8  # OpenMP threads per process

echo "Environment Variables:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  NCCL_DEBUG: $NCCL_DEBUG"
echo ""

# Check for HuggingFace token (required for Llama-3.1)
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Warning: HF_TOKEN not set"
    echo "   Llama-3.1 is a gated model and requires authentication"
    echo "   Set your token with: export HF_TOKEN=your_token_here"
    echo ""
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output directory
mkdir -p ./outputs/llama3.1_8b_gsm8k

# Start training with accelerate
echo "Starting training..."
echo "================================================"
echo ""

accelerate launch \
    --config_file accelerate_config/accelerate_config.yaml \
    --num_processes 8 \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_port 29501 \
    --mixed_precision bf16 \
    train.py --config configs/Llama-3.1/llama3.1_8b_gsm8k.yaml

echo ""
echo "================================================"
echo "Training Complete!"
echo "================================================"
echo ""
echo "Output location: ./outputs/llama3.1_8b_gsm8k/"
echo "Logs: ./outputs/llama3.1_8b_gsm8k/logs/"
echo ""
echo "To resume training from checkpoint:"
echo "  python train.py --config configs/Llama-3.1/llama3.1_8b_gsm8k.yaml \\"
echo "    --resume_from_checkpoint ./outputs/llama3.1_8b_gsm8k/checkpoint-XXXX"
echo ""
