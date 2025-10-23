"""
Model loading utilities with PEFT support.
"""

import os
import torch
from typing import Dict, Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import logging

logger = logging.getLogger(__name__)


def load_tokenizer(model_name: str, config: Dict) -> AutoTokenizer:
    """Load tokenizer from pretrained model."""
    model_config = config.get("model", {})
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config.get("trust_remote_code", True),
        use_auth_token=model_config.get("use_auth_token", False),
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Loaded tokenizer: {model_name}")
    return tokenizer


def load_base_model(model_name: str, config: Dict):
    """Load base model based on task type."""
    task_type = config.get("task_type", "causal_lm")
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    
    # Determine if we need quantization
    use_quantization = False
    load_in_8bit = False
    load_in_4bit = False
    
    optim = training_config.get("optim", "adamw_torch")
    if "8bit" in optim:
        use_quantization = True
        load_in_8bit = True
    elif "4bit" in optim or "qlora" in optim.lower():
        use_quantization = True
        load_in_4bit = True
    
    # Configure quantization if needed
    bnb_config = None
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if training_config.get("bf16", False) else torch.float16,
            bnb_4bit_quant_type="nf4" if load_in_4bit else None,
            bnb_4bit_use_double_quant=True if load_in_4bit else False,
        )
        logger.info(f"Using quantization: 8-bit={load_in_8bit}, 4-bit={load_in_4bit}")
    
    # Load model based on task type
    common_kwargs = {
        "trust_remote_code": model_config.get("trust_remote_code", True),
        "use_auth_token": model_config.get("use_auth_token", False),
        "device_map": "auto",
    }
    
    if bnb_config:
        common_kwargs["quantization_config"] = bnb_config
    
    if task_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if training_config.get("bf16", False) else torch.float16,
            **common_kwargs,
        )
    elif task_type == "classification":
        # For classification, we need to know the number of labels
        dataset_config = config.get("dataset", {})
        num_labels = dataset_config.get("num_labels", 2)  # Default to binary classification
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    logger.info(f"Loaded base model: {model_name}")
    
    # Prepare model for k-bit training if quantized
    if use_quantization:
        model = prepare_model_for_kbit_training(model)
        logger.info("Prepared model for k-bit training")
    
    return model


def get_peft_config(config: Dict) -> LoraConfig:
    """Get PEFT configuration based on config."""
    peft_config = config.get("peft", {})
    method = peft_config.get("method", "lora").lower()
    task_type_str = peft_config.get("task_type", "CAUSAL_LM")
    
    # Map task type string to TaskType enum
    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
        "QUESTION_ANS": TaskType.QUESTION_ANS,
    }
    task_type = task_type_map.get(task_type_str, TaskType.CAUSAL_LM)
    
    if method == "lora":
        peft_cfg = LoraConfig(
            r=peft_config.get("lora_r", 8),
            lora_alpha=peft_config.get("lora_alpha", 16),
            lora_dropout=peft_config.get("lora_dropout", 0.1),
            target_modules=peft_config.get("target_modules", ["q_proj", "v_proj"]),
            bias=peft_config.get("bias", "none"),
            task_type=task_type,
            inference_mode=peft_config.get("inference_mode", False),
        )
        logger.info(f"Created LoRA config: r={peft_cfg.r}, alpha={peft_cfg.lora_alpha}")
    elif method == "prefix-tuning":
        peft_cfg = PrefixTuningConfig(
            num_virtual_tokens=peft_config.get("num_virtual_tokens", 20),
            task_type=task_type,
        )
        logger.info(f"Created Prefix Tuning config")
    elif method == "prompt-tuning":
        peft_cfg = PromptTuningConfig(
            num_virtual_tokens=peft_config.get("num_virtual_tokens", 20),
            task_type=task_type,
        )
        logger.info(f"Created Prompt Tuning config")
    else:
        raise ValueError(f"Unknown PEFT method: {method}")
    
    return peft_cfg


def setup_model_and_tokenizer(config: Dict) -> Tuple:
    """Setup model, tokenizer, and PEFT config."""
    model_name = config["model"]["name_or_path"]
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_name, config)
    
    # Load base model
    model = load_base_model(model_name, config)
    
    # Get PEFT config
    peft_config = get_peft_config(config)
    
    # Apply PEFT
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer, peft_config


def save_model(model, tokenizer, output_dir: str):
    """Save model and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save PEFT model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Model saved to {output_dir}")


def load_peft_model(model_name: str, peft_model_path: str, config: Dict):
    """Load a PEFT model for inference."""
    from peft import PeftModel
    
    # Load base model
    base_model = load_base_model(model_name, config)
    
    # Load PEFT weights
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    
    logger.info(f"Loaded PEFT model from {peft_model_path}")
    return model
