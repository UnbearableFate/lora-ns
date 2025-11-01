"""
Main training script for PEFT fine-tuning.
Supports multiple tasks: GLUE, MetaMathQA, GSM8K, Code-Feedback, etc.
"""

from gc import callbacks
import os
import sys
import logging
import argparse
from typing import Optional

from click import Path
import torch
from transformers import (
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer
from accelerate import Accelerator
import wandb

# Import utilities
from utils import (
    load_config,
    validate_config,
    print_config,
    setup_model_and_peft,
    save_model,
    prepare_dataset,
    get_metrics_function,
)
from utils.MomentumPolarizedTrainer import MomentumPolarizedTrainer
from utils.MuonLoraTrainer import MuonLoRATrainer
from utils.SpectralRefactorTrainer import SpectralRefactorTrainer
from utils.model_utils import load_tokenizer, setup_model_and_init_peft

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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


def setup_training_args(config: dict, train_data_num_per_process:int) -> TrainingArguments:
    """Setup training arguments from config."""
    training_config = config["training"]
    
    # Convert learning_rate to float if it's a string (YAML might parse 3e-4 as string)
    learning_rate = training_config.get("learning_rate", 2e-4)
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    
    max_steps = 1000
    if "max_steps" in training_config:
        max_steps = training_config["max_steps"]
    elif "num_train_epochs" in training_config:
        max_steps = training_config.get("num_train_epochs") * train_data_num_per_process // (training_config.get("per_device_train_batch_size", 8) * training_config.get("gradient_accumulation_steps", 1) ) 
    eval_steps = max_steps // training_config.get("total_eval_times", 50)
    save_steps = eval_steps
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        max_steps=max_steps,
        #num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        learning_rate=learning_rate,
        weight_decay=training_config.get("weight_decay", 0.0),
        warmup_ratio=training_config.get("warmup_ratio", 0.03),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        
        # Evaluation
        eval_strategy=training_config.get("eval_strategy", "steps"),
        eval_steps=eval_steps,
        save_strategy=training_config.get("save_strategy", "steps"),
        save_steps=save_steps,
        load_best_model_at_end=training_config.get("load_best_model_at_end", False),
        metric_for_best_model=training_config.get("metric_for_best_model", "loss"),
        greater_is_better=training_config.get("greater_is_better", False),
        
        # Optimization
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", False),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        optim=training_config.get("optim", "adamw_torch"),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        
        # Logging
        logging_steps=training_config.get("logging_steps", 10),
        logging_dir=training_config.get("logging_dir", None),
        report_to=training_config.get("report_to", ["tensorboard"]),
        
        # Misc
        seed=training_config.get("seed", 42),
        data_seed=training_config.get("data_seed", 42),
        remove_unused_columns=training_config.get("remove_unused_columns", True),
        save_total_limit=training_config.get("save_total_limit", 3),
        
        # Distributed training
        ddp_find_unused_parameters=False,
    )
    
    return training_args


def train_classification_task(config: dict, model, tokenizer, dataset, training_args):
    """Train classification tasks (e.g., GLUE)."""
    logger.info("Training classification task")
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Metrics
    task_name = config.get("task_name", "")
    compute_metrics = get_metrics_function(task_name)
    
    # Trainer
    callbacks = []
    early_stopping_patience = config.get("early_stopping_patience", None) 
    if early_stopping_patience is not None and early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    if config.get("trainer", {}).get("name") == "SpectralRefactorTrainer":
        trainer = SpectralRefactorTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        ) 
    elif config.get("trainer", {}).get("name") == "MomentumPolarizedTrainer":
        trainer = MomentumPolarizedTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
    elif config.get("trainer", {}).get("name") == "MuonLoRATrainer":
        trainer = MuonLoRATrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
    
    return trainer


def train_causal_lm_task(config: dict, model, tokenizer, dataset, training_args):
    """Train causal LM tasks (e.g., MetaMathQA, GSM8K, Code-Feedback)."""
    logger.info("Training causal LM task")
    
    # SFT specific config
    sft_config = config.get("sft", {})
    max_seq_length = sft_config.get("max_seq_length", 2048)
    packing = sft_config.get("packing", False)
    dataset_text_field = sft_config.get("dataset_text_field", "text")
    
    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation")
    )

    return trainer


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load and validate config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    validate_config(config)
    print_config(config)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
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
        run_name = f"{config['model']['name_or_path'].replace('/', '_')}_{config['dataset']['name']}_{config['dataset'].get('subset', '')}_{config['trainer'].get('name', '')}_{config['peft'].get('method', '')}_{config['peft'].get('init_lora_weights', '')}_{wandb_config.get('run_name_suffix', '')}"
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
