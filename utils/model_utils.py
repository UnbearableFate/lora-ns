"""
Model loading utilities with PEFT support.
"""

import os
from pathlib import Path
import torch
from typing import Dict, Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    prepare_model_for_kbit_training,
)

import logging

logger = logging.getLogger(__name__)

def load_tokenizer(model_name: str, config: Dict) -> AutoTokenizer:
    """Load and configure tokenizer from pretrained model."""
    tokenizer_config = config.get("tokenizer", {})

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=True,
        padding_side="right",
        truncation_side="right",
    )

    # Ensure special tokens exist for padding/eos when training causal models.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Allow overriding padding/truncation behaviour from config.
    """
    padding_side = tokenizer_config.get("padding_side") or dataset_config.get("padding_side")
    if padding_side:
        tokenizer.padding_side = padding_side

    truncation_side = tokenizer_config.get("truncation_side") or dataset_config.get("truncation_side")
    if truncation_side:
        tokenizer.truncation_side = truncation_side
    """

    # Optionally override chat template when working with instruct/chat checkpoints.
    chat_template = tokenizer_config.get("chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template

    logger.info(
        "Loaded tokenizer %s (padding_side=%s, truncation_side=%s, max_length=%s)",
        model_name,
        tokenizer.padding_side,
        tokenizer.truncation_side,
        tokenizer.model_max_length,
    )
    return tokenizer


def load_base_model(model_name: str, config: Dict):
    """Load base model based on task type."""
    task_type = config.get("task_type", "CAUSAL_LM")
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    
    # Determine if we need quantization (explicit config takes precedence over heuristic)
    use_quantization = False
    load_in_8bit = False
    load_in_4bit = False

    quantization_overrides = model_config.get("quantization")
    if quantization_overrides:
        use_quantization = True
        load_in_8bit = quantization_overrides.get("load_in_8bit", False)
        load_in_4bit = quantization_overrides.get("load_in_4bit", False)
    else:
        optim = training_config.get("optim", "adamw_torch")
        if isinstance(optim, str) and "8bit" in optim.lower():
            use_quantization = True
            load_in_8bit = True
        elif isinstance(optim, str) and ("4bit" in optim.lower() or "qlora" in optim.lower()):
            use_quantization = True
            load_in_4bit = True
    
    # Configure quantization if needed
    bnb_config = None
    if use_quantization:
        if quantization_overrides:
            quant_cfg = dict(quantization_overrides)
            compute_dtype = quant_cfg.get("bnb_4bit_compute_dtype")
            if isinstance(compute_dtype, str):
                quant_cfg["bnb_4bit_compute_dtype"] = getattr(torch, compute_dtype)
            bnb_config = BitsAndBytesConfig(**quant_cfg)
        else:
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
        "token": model_config.get("token", False),
        "device_map": model_config.get("device_map", "auto"),
        "revision": model_config.get("revision"),
        "low_cpu_mem_usage": model_config.get("low_cpu_mem_usage", True),
    }
    common_kwargs = {k: v for k, v in common_kwargs.items() if v is not None}
    
    if bnb_config:
        common_kwargs["quantization_config"] = bnb_config
    
    dtype_override = model_config.get("torch_dtype")
    if isinstance(dtype_override, str):
        dtype_override = getattr(torch, dtype_override)
    elif dtype_override is None:
        if training_config.get("bf16", False):
            dtype_override = torch.bfloat16
        elif training_config.get("fp16", False):
            dtype_override = torch.float16
        else:
            # Default to float16 for parity with previous behaviour unless explicitly overridden.
            dtype_override = torch.float16

    if task_type == "CAUSAL_LM":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True, 
            attn_implementation="eager",
            #**common_kwargs,
        )
    elif task_type == "SEQ_CLS":
        # For classification, we need to know the number of labels
        dataset_config = config.get("dataset", {})
        num_labels = dataset_config.get("num_labels", 2)  # Default to binary classification
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=dtype_override,
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

def save_model(model, tokenizer, output_dir: str):
    """Save model and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save PEFT model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Model saved to {output_dir}")
