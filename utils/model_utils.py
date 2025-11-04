"""
Model loading utilities with PEFT support.
"""

import os
from pathlib import Path
import time
import torch
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader
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

def _ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)

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
    task_type = config.get("task_type", "CAUSAL_LM")
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
    
    if task_type == "CAUSAL_LM":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if training_config.get("bf16", False) else torch.float16,
            **common_kwargs,
        )
    elif task_type == "SEQ_CLS":
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
            target_modules=peft_config.get("target_modules", None),
            bias=peft_config.get("bias", "none"),
            task_type=task_type,
            inference_mode=peft_config.get("inference_mode", False),
            init_lora_weights=peft_config.get("init_lora_weights", "gaussian"),
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

def setup_model_and_peft(config: Dict) -> Tuple:
    """Setup model, tokenizer, and PEFT config."""
    model_name = config["model"]["name_or_path"]
    
    # Load base model
    model = load_base_model(model_name, config)
    
    # Get PEFT config
    peft_config = get_peft_config(config)
    
    # Apply PEFT
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, peft_config

def setup_model_and_init_peft(config: Dict, dataset, tokenizer, accelerator) -> Tuple:
    """Setup model, tokenizer, and PEFT config."""
    from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling
    
    model_name = config["model"]["name_or_path"]
    dataset_name = config["dataset"].get("name", "custom_dataset")
    loraga_config_dict = config.get("loraga")
    assert loraga_config_dict is not None, "loraga config must be provided for LoRA-GA initialization"

    if config["dataset"].get("subset"):
        dataset_name += f"_{config['dataset'].get('subset')}"
    
    # Load base model
    model = load_base_model(model_name, config)
    
    # Set model's pad_token_id to match tokenizer
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        logger.info(f"Set model pad_token_id to {tokenizer.pad_token_id}")

    # Create appropriate data collator based on task type
    task_type = config.get("task_type", "CAUSAL_LM")
    train_dataset = dataset["train"]
    if task_type == "SEQ_CLS":
        print(f"Train dataset columns: {train_dataset.column_names}")
        columns_to_keep = ["input_ids", "attention_mask", "labels"]
        columns_to_remove = [col for col in train_dataset.column_names if col not in columns_to_keep]
        gradient_dataset = train_dataset.remove_columns(columns_to_remove) if columns_to_remove else train_dataset
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    elif task_type == "CAUSAL_LM":
        gradient_dataset = train_dataset
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        gradient_dataset = train_dataset
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    num_gradient_samples = min(loraga_config_dict["num_samples"], len(gradient_dataset))
    gradient_subset = gradient_dataset.select(range(num_gradient_samples))
    # Create DataLoader for gradient estimation (no num_workers to avoid serialization issues)
    gradient_loader = DataLoader(
        dataset=gradient_subset,  # use a small subset for gradient estimation
        batch_size=loraga_config_dict.get("batch_size", 8),
        collate_fn=data_collator,
    )
    # Get PEFT config
    #peft_config = get_peft_config(config)

    peft_config_dict = config.get("peft")
    from my_peft import LoraGAConfig
    pack_loraga_config = LoraGAConfig(
        r=peft_config_dict.get("lora_r", 8),
        lora_alpha=peft_config_dict.get("lora_alpha", 16),
        lora_dropout=peft_config_dict.get("lora_dropout", 0.1),
        target_modules=peft_config_dict.get("target_modules", None),
        bias=peft_config_dict.get("bias", "none"),
        task_type=task_type, # TaskType.CAUSAL_LM,
        inference_mode=peft_config_dict.get("inference_mode", False),
        init_lora_weights=peft_config_dict.get("init_lora_weights", "lora_ga"),
        bsz=loraga_config_dict.get("batch_size", 8),
        direction=loraga_config_dict.get("direction", "ArB2r"),
        dtype=loraga_config_dict.get("dtype", "float32"),
    )

    print(f"PEFT config: {peft_config_dict}")
    print(f"loraga config: {loraga_config_dict}")

    from my_peft.utils.lora_ga_utils import (
                LoraGAContext,
                estimate_gradient,
            )
    from my_peft import get_peft_model as my_get_peft_model

    grad_save_path = loraga_config_dict.get("gradient").get("save_path")
    if grad_save_path:
        grad_save_path = Path(grad_save_path , f"grad_save_{model_name.replace('/', '-')}_{dataset_name}.pt")
        _ensure_directory(grad_save_path.parent)
    else:
        grad_save_path = Path("data_cache") / f"grad_save_{model_name.replace('/', '-')}_{dataset_name}.pt"
        _ensure_directory(grad_save_path.parent)
    named_grad = estimate_gradient(
        model=model,
        dataloader=gradient_loader,
        accelerator=accelerator,
        quant_flag=False,
        origin_type=None,
        quant_type=None,
        no_split_module_classes=None,
        grad_save_path=grad_save_path,
    )
    start_time = time.time()
    with LoraGAContext(model=model, named_grad=named_grad):
        model = my_get_peft_model(model=model, peft_config=pack_loraga_config)
    logger.info(f"LoRA-GA initialization took {time.time() - start_time:.2f} seconds")

    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, peft_config_dict


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
