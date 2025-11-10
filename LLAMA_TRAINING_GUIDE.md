# Llama-3.1-8B Training Guide

## Overview

This guide covers distributed training of Llama-3.1-8B on MetaMathQA and GSM8K datasets using 8×H200 GPUs.

## Hardware Requirements

- **GPUs**: 8×NVIDIA H200 (or A100/H100 with 80GB+ VRAM)
- **CPU**: High-core count recommended (e.g., 64+ cores)
- **RAM**: 256GB+ system memory
- **Storage**: 500GB+ for model, datasets, and checkpoints
- **Network**: High-bandwidth interconnect (InfiniBand recommended for multi-node)

## Memory Estimates

### MetaMathQA (395K samples)

- **Model**: Llama-3.1-8B base (~16GB in bf16)
- **LoRA adapters**: ~512MB (rank=32)
- **Activation memory**: ~12GB per GPU (batch_size=4, max_length=1024)
- **Optimizer states**: ~1GB per GPU (AdamW)
- **Gradient checkpointing**: Reduces activation memory by ~50%
- **Total per GPU**: ~22-26GB (well within H200 141GB capacity)

### GSM8K (7.5K samples)

- Similar memory profile to MetaMathQA
- Smaller dataset allows higher learning rate and more epochs

## Pre-requisites

### 1. HuggingFace Token

Llama-3.1 is a gated model requiring authentication:

```bash
# Request access at: https://huggingface.co/meta-llama/Llama-3.1-8B
# Then set your token:
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Or login via CLI:
huggingface-cli login
```

### 2. Accelerate Configuration

Verify your accelerate config at `accelerate_config/accelerate_config.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU  # Use DDP
num_processes: 8
gpu_ids: all
mixed_precision: bf16
```

### 3. Python Environment

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Training Commands

### MetaMathQA Training (~8-12 hours)

```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here

# Run training
bash examples/train_llama3.1_metamath_8gpu.sh
```

**Expected outputs**:
- Training steps: ~6,172 (395K samples / 64 effective batch size)
- Checkpoints: Every 617 steps (~10% of training)
- Evaluations: Every 617 steps
- Final model: `outputs/llama3.1_8b_metamath/`

### GSM8K Training (~1-2 hours)

```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here

# Run training
bash examples/train_llama3.1_gsm8k_8gpu.sh
```

**Expected outputs**:
- Training steps: ~586 (7.5K samples / 64 effective batch size × 5 epochs)
- Checkpoints: Every ~12 steps (frequent saving)
- Evaluations: Every ~12 steps (50 times total)
- Final model: `outputs/llama3.1_8b_gsm8k/`

## Manual Training (Advanced)

For more control, run with `accelerate launch` directly:

```bash
accelerate launch \
    --config_file accelerate_config/accelerate_config.yaml \
    --num_processes 8 \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_port 29500 \
    --mixed_precision bf16 \
    train.py --config configs/Llama-3.1/llama3.1_8b_metamath.yaml
```

### Multi-Node Training

For training across multiple nodes:

```bash
# Node 0 (master):
accelerate launch \
    --config_file accelerate_config/accelerate_config.yaml \
    --num_processes 16 \
    --num_machines 2 \
    --machine_rank 0 \
    --main_process_ip 192.168.1.100 \
    --main_process_port 29500 \
    --mixed_precision bf16 \
    train.py --config configs/Llama-3.1/llama3.1_8b_metamath.yaml

# Node 1:
accelerate launch \
    --config_file accelerate_config/accelerate_config.yaml \
    --num_processes 16 \
    --num_machines 2 \
    --machine_rank 1 \
    --main_process_ip 192.168.1.100 \
    --main_process_port 29500 \
    --mixed_precision bf16 \
    train.py --config configs/Llama-3.1/llama3.1_8b_metamath.yaml
```

## Configuration Details

### Hyperparameters

| Parameter | MetaMathQA | GSM8K | Rationale |
|-----------|------------|-------|-----------|
| Learning rate | 1e-4 | 3e-4 | Larger dataset → lower LR |
| Batch size (effective) | 128 | 64 | 4×8×4 vs 4×8×2 |
| Epochs | 2 | 5 | 395K vs 7.5K samples |
| LoRA rank | 32 | 32 | Higher capacity for 8B model |
| Max length | 1024 | 1024 | Math problems can be long |
| Warmup ratio | 0.05 | 0.1 | More warmup for small dataset |

### LoRA Configuration

Both configs use:
- **Init method**: PiSSA (better than gaussian)
- **RSLoRA**: Enabled (stabilizes training)
- **Target modules**: All attention + MLP layers
- **Rank**: 32 (α=64, dropout=0.05)

### SpectralRefactorTrainer

Custom trainer with spectral normalization:
- **Refactor frequency**: Every 200 steps (MetaMathQA), 100 steps (GSM8K)
- **Spectral norm**: Stabilizes gradient flow
- **Compatible with**: DDP, gradient checkpointing, mixed precision

## Monitoring

### WandB Integration

Both configs automatically log to WandB:

```bash
# Check your runs at:
https://wandb.ai/YOUR_USERNAME/Llama-3.1-8B-MetaMath
https://wandb.ai/YOUR_USERNAME/Llama-3.1-8B-GSM8K
```

**Metrics tracked**:
- `train/loss`: Training loss
- `eval/loss`: Validation loss
- `eval/token_accuracy`: Token-level accuracy
- `eval/answer_accuracy`: Extracted answer accuracy
- `train/learning_rate`: LR schedule
- `train/grad_norm`: Gradient norms
- GPU utilization, memory usage, throughput

### TensorBoard (Alternative)

If not using WandB:

```bash
tensorboard --logdir outputs/llama3.1_8b_metamath/logs
```

## Expected Performance

### MetaMathQA

- **Baseline (no training)**: ~5-10% answer accuracy
- **After 1 epoch**: ~40-50% answer accuracy
- **After 2 epochs**: ~55-65% answer accuracy
- **Token accuracy**: ~70-80% (less meaningful for generation)

### GSM8K

- **Baseline (no training)**: ~10-15% answer accuracy
- **After 3 epochs**: ~60-70% answer accuracy
- **After 5 epochs**: ~70-80% answer accuracy
- **State-of-the-art**: ~85%+ (with larger models)

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: CUDA OOM errors during training

**Solutions**:
1. Reduce `per_device_train_batch_size` from 4 to 2
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_length` from 1024 to 768
4. Enable more aggressive gradient checkpointing

**Example fix**:
```yaml
per_device_train_batch_size: 2  # was 4
gradient_accumulation_steps: 8  # was 4 (maintains effective batch=128)
```

### Slow Training

**Symptoms**: <100 samples/sec throughput

**Possible causes**:
1. Disk I/O bottleneck → Use faster SSD, increase `dataloader_num_workers`
2. CPU preprocessing → Increase `preprocessing_num_workers`
3. Network latency → Check InfiniBand, adjust `NCCL_SOCKET_IFNAME`
4. Slow gradient synchronization → Verify all GPUs are same type/speed

**Optimizations**:
```yaml
dataloader_num_workers: 8  # Increase for faster data loading
dataloader_prefetch_factor: 4  # Prefetch batches
```

### NCCL Errors

**Symptoms**: "NCCL error", "Communication timeout"

**Solutions**:
1. Check all GPUs are visible: `nvidia-smi`
2. Verify network interface: `export NCCL_SOCKET_IFNAME=<your_interface>`
3. Disable InfiniBand if not available: `export NCCL_IB_DISABLE=1`
4. Increase timeout: `export NCCL_TIMEOUT=1800`

### HuggingFace Token Issues

**Symptoms**: "Repository not found", "Access denied"

**Solutions**:
1. Request access at https://huggingface.co/meta-llama/Llama-3.1-8B
2. Wait for approval (usually instant if licensed)
3. Set token: `export HF_TOKEN=...` or `huggingface-cli login`
4. Verify: `huggingface-cli whoami`

### Checkpoint Recovery

**If training crashes**:

```bash
# List available checkpoints
ls outputs/llama3.1_8b_metamath/checkpoint-*

# Resume from latest
python train.py \
    --config configs/Llama-3.1/llama3.1_8b_metamath.yaml \
    --resume_from_checkpoint outputs/llama3.1_8b_metamath/checkpoint-3086
```

## Performance Benchmarks

### H200 8-GPU Training

| Dataset | Samples/sec | GPU Util | Memory/GPU | Time to Complete |
|---------|-------------|----------|------------|------------------|
| MetaMathQA | 120-150 | 90-95% | 24-28GB | 8-10 hours |
| GSM8K | 150-180 | 85-90% | 22-26GB | 1-1.5 hours |

### A100 80GB 8-GPU Training

| Dataset | Samples/sec | GPU Util | Memory/GPU | Time to Complete |
|---------|-------------|----------|------------|------------------|
| MetaMathQA | 100-120 | 85-90% | 26-32GB | 10-12 hours |
| GSM8K | 120-150 | 80-85% | 24-28GB | 1.5-2 hours |

## Next Steps

After training:

1. **Merge adapter weights**:
   ```bash
   python merge_adapter.py \
       --base_model meta-llama/Llama-3.1-8B \
       --adapter_path outputs/llama3.1_8b_metamath \
       --output_path models/llama3.1_8b_metamath_merged
   ```

2. **Run inference**:
   ```bash
   python inference.py \
       --model_path models/llama3.1_8b_metamath_merged \
       --prompt "Solve: If a train travels 120 miles in 2 hours, what is its average speed?"
   ```

3. **Evaluate on test set**:
   ```bash
   python validate_config.py \
       --config configs/Llama-3.1/llama3.1_8b_gsm8k.yaml \
       --checkpoint outputs/llama3.1_8b_gsm8k/checkpoint-best
   ```

## Comparison: SmolLM vs Llama-3.1

| Aspect | SmolLM2-135M | Llama-3.1-8B |
|--------|--------------|--------------|
| **Purpose** | Prototyping/Testing | Production |
| **Parameters** | 135M | 8B |
| **GPUs required** | 1 (any GPU) | 8×H200 recommended |
| **Training time (MetaMathQA)** | 20-30 min (10K subset) | 8-12 hours (full dataset) |
| **LoRA rank** | 8 | 32 |
| **Max length** | 512 | 1024 |
| **Batch size** | 2-8 | 128 (distributed) |
| **Expected accuracy** | 30-40% | 60-70% |
| **Use case** | Quick experiments | Final model deployment |

## References

- Llama-3.1 Paper: https://arxiv.org/abs/2407.21783
- LoRA: https://arxiv.org/abs/2106.09685
- RSLoRA: https://arxiv.org/abs/2312.03732
- PiSSA: https://arxiv.org/abs/2404.02948
- MetaMathQA: https://huggingface.co/datasets/meta-math/MetaMathQA
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
