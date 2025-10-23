# Installation and Setup Guide

## Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- 20GB+ GPU VRAM (for 7B models)

## Step-by-Step Installation

### 1. Create Virtual Environment (Recommended)

```bash
cd /home/yu/workspace/lora-ns

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention for faster training (requires CUDA)
pip install flash-attn --no-build-isolation
```

### 3. Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check PEFT
python -c "import peft; print(f'PEFT: {peft.__version__}')"

# Validate configurations
python validate_config.py --all
```

### 4. Configure HuggingFace Token (for Gated Models)

For models like Llama-2, you need a HuggingFace token:

```bash
# Login to HuggingFace
huggingface-cli login

# Or set environment variable
export HF_TOKEN="your_token_here"
```

## Quick Test

### Test 1: Validate Configurations

```bash
python validate_config.py --all
```

Expected output: All configurations should pass validation.

### Test 2: Quick Training Test

Run a quick training test with minimal data:

```bash
python examples/quick_start.py
```

This will train on a small subset of GLUE MRPC for 1 epoch.

### Test 3: Check GPU Usage

```bash
# Monitor GPU during training
nvidia-smi -l 1
```

## Configuration for Your Environment

### Limited GPU Memory (<16GB)

Edit your config files to use smaller batches:

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
  gradient_checkpointing: true
  optim: "paged_adamw_8bit"  # Use 8-bit optimizer
```

### CPU Only Training (Not Recommended for Large Models)

```yaml
accelerate:
  use_cpu: true
  
training:
  fp16: false
  bf16: false
```

### Multiple GPUs

```bash
# Configure Accelerate
accelerate config

# Launch training
accelerate launch train.py --config configs/gsm8k.yaml
```

## Common Installation Issues

### Issue 1: CUDA Version Mismatch

```bash
# Check CUDA version
nvcc --version

# Install PyTorch with matching CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
```

### Issue 2: bitsandbytes Installation Fails

```bash
# Install from source
pip install bitsandbytes --prefer-binary
```

### Issue 3: Flash Attention Build Fails

Flash Attention is optional. If it fails:
- Skip it and continue without Flash Attention
- Or install pre-built wheels: https://github.com/Dao-AILab/flash-attention/releases

### Issue 4: Out of Memory During Installation

```bash
# Install packages one by one
pip install torch
pip install transformers
pip install datasets
pip install peft
# ... etc
```

## Environment Variables

Create a `.env` file (optional):

```bash
# HuggingFace
HF_TOKEN=your_huggingface_token
HF_HOME=/path/to/huggingface/cache

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=peft-training

# Training
CUDA_VISIBLE_DEVICES=0,1  # Specify GPUs to use
OMP_NUM_THREADS=8
```

## Directory Structure After Installation

```
lora-ns/
├── .venv/                    # Virtual environment (created)
├── configs/                  # Configuration files
├── utils/                    # Utility modules
├── examples/                 # Example scripts
├── outputs/                  # Training outputs (created during training)
├── train.py                  # Main scripts
├── inference.py
└── requirements.txt
```

## Next Steps

After installation:

1. ✅ Read the [README.md](README.md) for usage instructions
2. ✅ Review [OVERVIEW.md](OVERVIEW.md) for project structure
3. ✅ Check example scripts in `examples/`
4. ✅ Start with GLUE task for quick validation
5. ✅ Customize configs for your use case

## Getting Help

If you encounter issues:

1. Check configuration files are valid: `python validate_config.py --all`
2. Verify GPU availability: `nvidia-smi`
3. Check package versions match requirements
4. Review error messages carefully
5. Consult HuggingFace documentation

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf .venv

# Remove outputs (optional)
rm -rf outputs/
```

Happy Training! 🚀
