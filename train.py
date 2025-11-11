"""
Main training script for PEFT fine-tuning.
Supports multiple tasks: GLUE, MetaMathQA, GSM8K, Code-Feedback, etc.
"""

import os
import logging
import argparse
from typing import Optional

from click import Path
from accelerate import Accelerator
import wandb

# Import utilities
from trainer.trainer_preparation import setup_training_args, train_causal_lm_task, train_classification_task
from utils import (
    load_config,
    validate_config,
    print_config,
    setup_model_and_peft,
    prepare_dataset,
)
from utils.model_utils import load_tokenizer, setup_model_and_init_peft

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def extract_experiment_tags(config):
    """
    Extract key experiment information from config to use as WandB tags.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of tag strings describing the experiment
    """
    tags = []
    
    # PEFT configuration tags
    peft_config = config.get("peft", {})
    if peft_config.get("method"):
        tags.append(f"method:{peft_config['method']}")
    
    # LoRA rank and alpha
    if peft_config.get("lora_r"):
        tags.append(f"rank:{peft_config['lora_r']}")
    if peft_config.get("lora_alpha"):
        tags.append(f"alpha:{peft_config['lora_alpha']}")
    
    # LoRA initialization method
    if peft_config.get("init_lora_weights"):
        init_method = peft_config["init_lora_weights"]
        if init_method != "gaussian":  # Only tag if not default
            tags.append(f"init:{init_method}")
    
    # Advanced LoRA variants
    if peft_config.get("use_dora"):
        tags.append("dora")
    if peft_config.get("use_rslora"):
        tags.append("rslora")
    if peft_config.get("use_qalora"):
        tags.append("qalora")
    
    # Training configuration tags
    training_config = config.get("training", {})
    
    # Learning rate
    if training_config.get("learning_rate"):
        lr = training_config["learning_rate"]
        # Convert to float if it's a string (e.g., "1e-4")
        if isinstance(lr, str):
            lr = float(lr)
        tags.append(f"lr:{lr:.0e}")  # Scientific notation, e.g., "lr:1e-04"
    
    # Optimizer
    if training_config.get("optim"):
        optim = training_config["optim"]
        # Simplify optimizer names
        optim_short = optim.replace("adamw_torch_fused", "adamw-fused").replace("adamw_torch", "adamw")
        tags.append(f"optim:{optim_short}")
    
    # Batch size (effective batch size)
    per_device_bs = training_config.get("per_device_train_batch_size", 1)
    grad_accum = training_config.get("gradient_accumulation_steps", 1)
    # Note: num_gpus will be added dynamically if needed
    tags.append(f"bs:{per_device_bs}x{grad_accum}")
    
    # Mixed precision
    if training_config.get("bf16"):
        tags.append("bf16")
    elif training_config.get("fp16"):
        tags.append("fp16")
    
    # Gradient checkpointing
    if training_config.get("gradient_checkpointing"):
        tags.append("grad-ckpt")
    
    # Trainer type
    trainer_config = config.get("trainer", {})
    if trainer_config.get("name") and trainer_config["name"] != "Trainer":
        trainer_name = trainer_config["name"].replace("Trainer", "").replace("Spectral", "Spec")
        tags.append(f"trainer:{trainer_name}")
    
    # Dataset information
    dataset_config = config.get("dataset", {})
    if dataset_config.get("name"):
        dataset_name = dataset_config["name"]
        tags.append(f"data:{dataset_name}")
    
    # Model information (extract short name)
    model_config = config.get("model", {})
    if model_config.get("name_or_path"):
        model_path = model_config["name_or_path"]
        # Extract model name (e.g., "SmolLM2-135M" from "HuggingFaceTB/SmolLM2-135M")
        model_name = model_path.split("/")[-1]
        tags.append(f"model:{model_name}")
    
    # Task type
    task_type = config.get("task_type")
    if task_type:
        tags.append(f"task:{task_type}")
    
    return tags

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PEFT Training Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    accelerator = Accelerator()
    log_level = logging.INFO if accelerator.is_main_process else logging.WARNING
    logger.setLevel(log_level)
    # Load and validate config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    validate_config(config)
    print_config(config)
    
    # Initialize accelerator
    

    # Seed everything
    seed = config["training"].get("seed", 42)
    logger.info(f"Setting random seed to {seed}")
    seed_everything(seed)
    
    logger.info(f"Accelerator state: {accelerator.state}")
    
    # Setup tokenizer
    logger.info("Setting up tokenizer")
    model_name = config["model"]["name_or_path"]
    tokenizer = load_tokenizer(model_name, config)

    # Prepare dataset
    logger.info("Preparing dataset")
    dataset = prepare_dataset(config, tokenizer)

    if config.get("peft").get("init_lora_weights") in ["lora_ga","lora_ns"]:
        model, peft_config = setup_model_and_init_peft(config, dataset, tokenizer, accelerator)
    else:
        model, peft_config = setup_model_and_peft(config)
    
    if accelerator.is_main_process:
        print(model)

    logger.info(f"Train dataset size: {len(dataset['train'])}")
    if "validation" in dataset:
        logger.info(f"Validation dataset size: {len(dataset['validation'])}")
    
    # Setup training arguments
    training_args = setup_training_args(config,len(dataset['train'])// accelerator.num_processes)
    if accelerator.is_main_process:
        logger.info(f"Training arguments: {training_args}")
    
    # Create trainer based on task type
    task_type = config.get("task_type", "CAUSAL_LM")

    wandb_config = config.get("wandb")
    if wandb_config and accelerator.is_main_process:
        run_name = f"{config['dataset']['name']}_{config['dataset'].get('subset', '')}_{config['trainer'].get('name', '')}_{config['peft'].get('method', '')}_{config['peft'].get('init_lora_weights', '')}_seed{seed}"
        if wandb_config.get("online"):
            os.environ["WANDB_MODE"] = "online"
        else:
            os.environ["WANDB_MODE"] = "offline"
        
        # Get user-defined tags from config
        user_tags = config.get("wandb", {}).get("tags", [])
        
        # Extract experiment tags from config
        experiment_tags = extract_experiment_tags(config)
        
        # Combine user tags and auto-extracted tags
        all_tags = user_tags + experiment_tags
        
        logger.info(f"WandB tags: {all_tags}")
        
        wandb.init(
            project=wandb_config.get("project", "peft-finetuning"),
            name=run_name,
            tags=all_tags,
            config=config)
    
    if task_type == "SEQ_CLS":
        trainer = train_classification_task(config, model, tokenizer, dataset, training_args)
    elif task_type == "CAUSAL_LM":
        trainer = train_causal_lm_task(config, model, tokenizer, dataset, training_args)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Train
    logger.info("Starting training")
    
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        train_result = trainer.train()
    
    # Save model
    logger.info("Saving model")
    trainer.save_model()
    
    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Evaluate
    if "validation" in dataset:
        logger.info("Running evaluation")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    logger.info("Training complete!")
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
