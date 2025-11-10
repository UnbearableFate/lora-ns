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
        run_name = f"{config['dataset']['name']}_{config['dataset'].get('subset', '')}_{config['trainer'].get('name', '')}_{config['peft'].get('method', '')}_{config['peft'].get('init_lora_weights', '')}_seed{seed}_{wandb_config.get('run_name_suffix', '')}"
        if wandb_config.get("online"):
            os.environ["WANDB_MODE"] = "online"
        else:
            os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project=wandb_config.get("project", "peft-finetuning"),
            name=run_name,
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
