# PEFT Training Framework

A comprehensive Parameter-Efficient Fine-Tuning (PEFT) framework for training Large Language Models using HuggingFace's transformers, PEFT, TRL, and Accelerate libraries.

## Features

- 🚀 **Multiple PEFT Methods**: LoRA, Prefix Tuning, Prompt Tuning
- 📊 **Multiple Tasks**: GLUE benchmark, MetaMathQA, GSM8K, Code-Feedback
- ⚙️ **YAML Configuration**: Easy-to-manage configuration files
- 🔧 **Quantization Support**: 4-bit and 8-bit quantization with bitsandbytes
- 📈 **SFT Trainer**: Supervised Fine-Tuning with TRL
- 🏃 **Accelerate Integration**: Multi-GPU and distributed training support
- 💾 **Model Merging**: Merge LoRA weights with base models

## Project Structure

```
lora-ns/
├── configs/              # Configuration files for different tasks
│   ├── glue_mrpc.yaml   # GLUE MRPC task config
│   ├── metamath_qa.yaml # MetaMathQA task config
│   ├── gsm8k.yaml       # GSM8K task config
│   └── code_feedback.yaml # Code-Feedback task config
├── utils/               # Utility modules
│   ├── __init__.py
│   ├── config_utils.py  # Configuration loading/validation
│   ├── model_utils.py   # Model loading with PEFT
│   ├── dataset_loader.py # Dataset loading and preprocessing
│   └── metrics.py       # Evaluation metrics
├── train.py            # Main training script
├── inference.py        # Inference script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation

1. Clone the repository:
```bash
cd /home/yu/workspace/lora-ns
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For Flash Attention (optional, requires CUDA):
```bash
pip install flash-attn --no-build-isolation
```

## Quick Start

### 1. Training

Train on GLUE MRPC (classification):
```bash
python train.py --config configs/glue_mrpc.yaml
```

Train on MetaMathQA (math reasoning):
```bash
python train.py --config configs/metamath_qa.yaml
```

Train on GSM8K (math word problems):
```bash
python train.py --config configs/gsm8k.yaml
```

Train on Code-Feedback (code generation):
```bash
python train.py --config configs/code_feedback.yaml
```

### 2. Resume Training

Resume from a checkpoint:
```bash
python train.py --config configs/gsm8k.yaml --resume_from_checkpoint ./outputs/gsm8k/checkpoint-1000
```

### 3. Inference

Generate text with trained model:
```bash
python inference.py \
  --config configs/metamath_qa.yaml \
  --model_path ./outputs/metamath_qa \
  --input_text "What is 2 + 2?" \
  --max_new_tokens 256
```

Batch inference from file:
```bash
python inference.py \
  --config configs/gsm8k.yaml \
  --model_path ./outputs/gsm8k \
  --input_file inputs.txt \
  --output_file outputs.json
```

## Configuration

All configurations are managed via YAML files. Here's an overview of key configuration sections:

### Model Configuration
```yaml
model:
  name_or_path: "meta-llama/Llama-2-7b-hf"
  trust_remote_code: true
  use_auth_token: true  # For gated models
```

### PEFT Configuration
```yaml
peft:
  method: "lora"  # lora, prefix-tuning, prompt-tuning
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  bias: "none"
  task_type: "CAUSAL_LM"
```

### Dataset Configuration
```yaml
dataset:
  name: "gsm8k"
  subset: "main"
  train_split: "train"
  eval_split: "test"
  max_length: 1024
  prompt_template: |
    Question: {question}
    Answer: {answer}
```

### Training Configuration
```yaml
training:
  output_dir: "./outputs/gsm8k"
  num_train_epochs: 5
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  bf16: true
  gradient_checkpointing: true
  optim: "paged_adamw_8bit"
```

## Supported Tasks

### 1. GLUE Benchmark (Classification)
- Multiple subtasks: MRPC, QQP, QNLI, SST-2, etc.
- Models: BERT, RoBERTa, ALBERT, etc.
- PEFT: LoRA on attention layers

### 2. MetaMathQA (Math Reasoning)
- Dataset: 395K math problems with solutions
- Models: Llama-2, Mistral, etc.
- PEFT: LoRA on all linear layers

### 3. GSM8K (Grade School Math)
- Dataset: 8.5K grade school math word problems
- Models: Llama-2, Mistral, etc.
- PEFT: LoRA with higher rank for better performance

### 4. Code-Feedback (Code Generation)
- Dataset: Code generation with feedback
- Models: CodeLlama, DeepSeek-Coder, etc.
- PEFT: LoRA on attention layers

## Advanced Features

### Multi-GPU Training

Using Accelerate:
```bash
accelerate config  # Configure once

accelerate launch train.py --config configs/metamath_qa.yaml
```

### DeepSpeed Integration

Create a DeepSpeed config file `ds_config.json` and run:
```bash
accelerate launch --config_file deepspeed_config.yaml train.py --config configs/metamath_qa.yaml
```

### Quantization

4-bit quantization (QLoRA):
```yaml
training:
  optim: "paged_adamw_8bit"  # or paged_adamw_32bit
  bf16: true
```

### Custom Datasets

Create a custom config file:
```yaml
task_name: "my_custom_task"
task_type: "causal_lm"  # or "classification"

dataset:
  name: "username/dataset-name"
  subset: null
  train_split: "train"
  eval_split: "validation"
  prompt_template: |
    Your custom template here: {input_field}
    Response: {output_field}
```

## Model Support

### Classification Models
- BERT (bert-base-uncased, bert-large-uncased)
- RoBERTa (roberta-base, roberta-large)
- ALBERT (albert-base-v2, albert-large-v2)
- DistilBERT (distilbert-base-uncased)

### Causal LM Models
- Llama-2 (meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-13b-hf)
- Mistral (mistralai/Mistral-7B-v0.1)
- CodeLlama (codellama/CodeLlama-7b-hf)
- DeepSeek-Coder (deepseek-ai/deepseek-coder-6.7b-base)
- GPT-2 (gpt2, gpt2-medium, gpt2-large)

## Monitoring

### TensorBoard
```bash
tensorboard --logdir ./outputs/[task_name]/logs
```

### Weights & Biases
Add to config:
```yaml
training:
  report_to: ["wandb"]
```

And set environment variables:
```bash
export WANDB_API_KEY=your_key_here
export WANDB_PROJECT=peft-training
```

## Tips and Best Practices

1. **Memory Optimization**:
   - Use gradient checkpointing for large models
   - Use quantization (4-bit/8-bit)
   - Reduce batch size and increase gradient accumulation
   - Use `bf16` instead of `fp16` for better stability

2. **LoRA Configuration**:
   - For small datasets: r=8, alpha=16
   - For large datasets: r=16-32, alpha=32-64
   - Target all linear layers for best performance

3. **Learning Rate**:
   - Classification: 3e-4 to 5e-4
   - Causal LM: 1e-4 to 2e-4
   - Use cosine scheduler with warmup

4. **Evaluation**:
   - For causal LM: evaluate on perplexity and generation quality
   - For classification: use accuracy, F1, precision, recall

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`
- Use 4-bit quantization

### Slow Training
- Increase batch size
- Use multiple GPUs with `accelerate`
- Enable `bf16` training
- Use Flash Attention (if available)

### Poor Performance
- Increase LoRA rank (r)
- Train for more epochs
- Adjust learning rate
- Check data preprocessing

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- HuggingFace for transformers, PEFT, TRL, and Accelerate libraries
- Meta for Llama models
- MistralAI for Mistral models
- Dataset creators for GLUE, MetaMathQA, GSM8K, and Code-Feedback

## Citation

If you use this framework in your research, please cite the relevant papers:
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
