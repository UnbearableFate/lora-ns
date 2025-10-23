# PEFT Training Framework - Project Overview

## 📁 Project Structure

```
lora-ns/
├── configs/                      # Configuration files
│   ├── glue_mrpc.yaml           # GLUE benchmark (classification)
│   ├── metamath_qa.yaml         # MetaMathQA (math reasoning)
│   ├── gsm8k.yaml               # GSM8K (grade school math)
│   ├── code_feedback.yaml       # Code generation tasks
│   ├── template.yaml            # Template for custom tasks
│   ├── accelerate_config.yaml   # Multi-GPU training config
│   └── deepspeed_config.json    # DeepSpeed ZeRO config
│
├── utils/                        # Utility modules
│   ├── __init__.py
│   ├── config_utils.py          # YAML config loading/validation
│   ├── model_utils.py           # Model loading with PEFT
│   ├── dataset_loader.py        # Dataset loading/preprocessing
│   └── metrics.py               # Evaluation metrics
│
├── examples/                     # Example scripts
│   ├── quick_start.py           # Quick training example
│   ├── train_glue.sh            # GLUE training script
│   ├── train_metamath.sh        # MetaMathQA training script
│   ├── train_multi_gpu.sh       # Multi-GPU training script
│   └── run_inference.sh         # Inference script
│
├── train.py                      # Main training script
├── inference.py                  # Inference script
├── merge_adapter.py              # Merge LoRA with base model
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation
└── .gitignore                    # Git ignore patterns
```

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Train a Model
```bash
# Classification task (GLUE MRPC)
python train.py --config configs/glue_mrpc.yaml

# Math reasoning (GSM8K)
python train.py --config configs/gsm8k.yaml

# Code generation
python train.py --config configs/code_feedback.yaml
```

### 3. Run Inference
```bash
python inference.py \
  --config configs/gsm8k.yaml \
  --model_path ./outputs/gsm8k \
  --input_text "What is 25 * 4?"
```

## 📊 Supported Tasks

| Task | Dataset | Model Example | PEFT Method | Training Time* |
|------|---------|---------------|-------------|----------------|
| **Text Classification** | GLUE | BERT, RoBERTa | LoRA | ~1 hour |
| **Math Reasoning** | MetaMathQA | Llama-2, Mistral | LoRA | ~10 hours |
| **Grade School Math** | GSM8K | Llama-2, Mistral | LoRA | ~5 hours |
| **Code Generation** | Code-Feedback | CodeLlama | LoRA | ~8 hours |

*Approximate times on a single A100 GPU

## 🔧 Configuration System

All tasks are configured via YAML files with the following structure:

```yaml
task_name: "your_task"
task_type: "causal_lm"  # or "classification"

model:
  name_or_path: "model-name"
  
peft:
  method: "lora"
  lora_r: 16
  lora_alpha: 32
  
dataset:
  name: "dataset-name"
  prompt_template: |
    Your template here
    
training:
  num_train_epochs: 3
  learning_rate: 2e-4
  # ... more options
```

## 💡 Key Features

### 1. PEFT Methods
- **LoRA**: Low-Rank Adaptation (most popular)
- **Prefix Tuning**: Add trainable prefix tokens
- **Prompt Tuning**: Soft prompt optimization

### 2. Quantization Support
- 4-bit quantization (QLoRA)
- 8-bit quantization
- BFloat16 training

### 3. Multi-GPU Training
```bash
# Using Accelerate
accelerate launch train.py --config configs/metamath_qa.yaml

# Using DeepSpeed
accelerate launch --config_file configs/accelerate_config.yaml \
  train.py --config configs/metamath_qa.yaml
```

### 4. Model Merging
```bash
python merge_adapter.py \
  --config configs/gsm8k.yaml \
  --adapter_path ./outputs/gsm8k \
  --output_path ./merged_model
```

## 📝 Creating Custom Tasks

1. Copy template configuration:
```bash
cp configs/template.yaml configs/my_task.yaml
```

2. Edit configuration:
   - Set model name
   - Configure dataset
   - Adjust prompt template
   - Tune hyperparameters

3. Train:
```bash
python train.py --config configs/my_task.yaml
```

## 🎯 Best Practices

### Memory Optimization
- Use gradient checkpointing
- Enable quantization (4-bit/8-bit)
- Reduce batch size, increase gradient accumulation
- Use bf16 instead of fp16

### LoRA Configuration
- Small datasets: r=8, alpha=16
- Large datasets: r=16-32, alpha=32-64
- Target all linear layers for best performance

### Learning Rate
- Classification: 3e-4 to 5e-4
- Causal LM: 1e-4 to 2e-4
- Use cosine scheduler with warmup

## 📚 Additional Resources

- HuggingFace Transformers: https://huggingface.co/docs/transformers
- PEFT Library: https://huggingface.co/docs/peft
- TRL (Transformer Reinforcement Learning): https://huggingface.co/docs/trl
- Accelerate: https://huggingface.co/docs/accelerate

## 🐛 Troubleshooting

### CUDA Out of Memory
```yaml
# Reduce batch size
per_device_train_batch_size: 2
gradient_accumulation_steps: 16

# Enable optimizations
gradient_checkpointing: true
optim: "paged_adamw_8bit"
```

### Slow Training
```yaml
# Use mixed precision
bf16: true

# Optimize batch size
per_device_train_batch_size: 8  # Increase if memory allows
```

### Poor Results
- Increase LoRA rank (r)
- Train for more epochs
- Check data preprocessing
- Validate prompt template

## 📧 Support

For issues and questions:
1. Check the README.md
2. Review example scripts in `examples/`
3. Examine configuration templates
4. Consult HuggingFace documentation

## 🙏 Acknowledgments

Built with:
- HuggingFace Transformers & PEFT
- PyTorch
- Accelerate & DeepSpeed
- Dataset providers (GLUE, MetaMathQA, GSM8K, Code-Feedback)
