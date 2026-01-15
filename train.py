"""
Main training script for PEFT fine-tuning.
Supports multiple tasks: GLUE, MetaMathQA, GSM8K, Code-Feedback, etc.
"""

from copy import deepcopy
import datetime
import json
import os
import logging
import argparse
import time
from typing import Optional

from click import Path
from accelerate import Accelerator
import torch
import wandb

# Import utilities
from trainer.sr_init_trainer import restart_init_train
from trainer.trainer_preparation import setup_training_args, train_causal_lm_task, train_classification_task, get_collator
from utils import (
    load_config,
    validate_config,
    print_config,
    prepare_dataset,
)
from utils.model_utils import load_tokenizer,load_base_model
from utils.lora_loader import build_LoraHyperparameters_from_yaml_dict, get_lora_config, attach_lora_adapter , freeze_lora_A_weights

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
    #if peft_config.get("method"):
    #    tags.append(f"method:{peft_config['method']}")
    if peft_config.get("variant"):
        tags.append(f"variant:{peft_config['variant']}")
    
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
    task_type = config['peft'].get("task_type")
    if task_type:
        tags.append(f"task:{task_type}")
    
    target_modules = peft_config.get("target_modules")
    if target_modules:
        modules_str = "&".join(target_modules)
        tags.append(f"tgt-mods:{modules_str}")
    
    return tags

def get_run_name(config, timestamp: Optional[str] = None) -> str:
    lora_config = config.get("peft", {})
    dataset_cfg = config.get("dataset", {})
    model_name = config["model"]["name_or_path"].split("/")[-1]
    dataset_name = dataset_cfg.get("name").split("/")[-1]
    dataset_subset = ("_" + dataset_cfg.get("subset")) if dataset_cfg.get("subset") else ""
    init_weights = lora_config.get("init_lora_weights")
    wandb_run_name = (
        f"{model_name}_{dataset_name}{dataset_subset}"
        f"_r{lora_config.get('lora_r')}_a{lora_config.get('lora_alpha')}_{init_weights}_{lora_config.get('variant')}"
    )
    if config.get("trainer", {}).get("name") == "SpectralTrainer":
        wandb_run_name += "_sr-init"
        if config.get("trainer", {}).get("warmup_steps") < 10000:
            wandb_run_name += "&train"
    wandb_run_name += f"_s{config['training']['seed']}_{timestamp}"
    return wandb_run_name

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
        "--timestamp",
        type=str,
        default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Local rank for distributed training",
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()

def main(accelerator, args=None):
    """Main training function."""
    # Parse arguments
    if args is None:
        args = parse_args()
    log_level = logging.INFO if accelerator.is_main_process else logging.WARNING
    logger.setLevel(log_level)
    # Load and validate config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    validate_config(config)

    wandb_config = config.get("wandb")
    wandb_run = None
    run_name = get_run_name(config, timestamp=args.timestamp)
   
    if wandb_config and accelerator.is_main_process:
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
        wandb_project = f"{wandb_config.get("project", "nlp")}_{config["model"].get("name_or_path","").replace("/","-")}_{config.get("dataset",{}).get("name","")}"
        if config.get("dataset",{}).get("subset"):
            wandb_project += "_"+config.get("dataset",{}).get("subset")
        
        wandb_run = wandb.init(
            project=wandb_project,
            name=run_name,
            tags=all_tags,
            config=config)

    start_time = time.time()
    # Seed everything
    seed = args.seed
    config["training"]["seed"] = seed
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

    model = load_base_model(model_name, config)
    tokenizer = load_tokenizer(model_name, config)
    lora_hyperparams = build_LoraHyperparameters_from_yaml_dict(config)
    peft_config = get_lora_config(lora_hyperparams)
    model = attach_lora_adapter(
        model,
        peft_config,
        dataset["train"],
        tokenizer,
        init_num_samples= lora_hyperparams.init_num_samples,
        batch_size= lora_hyperparams.init_batch_size,
        seed=lora_hyperparams.init_seed,
        accelerator= accelerator,
    )

    if accelerator.is_main_process:
        model.print_trainable_parameters()
        print(f"peft_config: {peft_config}")

    logger.info(f"Train dataset size: {len(dataset['train'])}")
    if "validation" in dataset:
        logger.info(f"Validation dataset size: {len(dataset['validation'])}")
    
    # Setup training arguments
    training_args = setup_training_args(config,len(dataset['train']) ,accelerator.num_processes,run_name)
    if accelerator.is_main_process:
        logger.info(f"Training arguments: {training_args}")
    
    lora_init_kwargs = config["peft"].get("lora_init_kwargs", {})
    if lora_init_kwargs.get("method") == "sr-init":
        model = restart_init_train(
            trainning_args = training_args,
            init_steps = lora_init_kwargs.get("init_steps",1000),
            model = model,
            data_collator= get_collator(task_type=config["peft"]["task_type"], tokenizer= tokenizer,dataset= dataset),
            train_dataset = dataset['train'], 
        )

    # Create trainer based on task type
    task_type = config['peft'].get("task_type")
    
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

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Save model
    logger.info("Saving model")
    trainer.save_model()
    
    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Evaluate
    run_validation_eval = config.get("training", {}).get("run_validation_eval", True)
    if (not getattr(args, "skip_validation_eval", False)) and run_validation_eval and "validation" in dataset:
        logger.info("Running evaluation")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
    
    logger.info("Training complete!")
    logger.info(f"Model saved to {training_args.output_dir}")

    if accelerator.is_main_process and wandb_run is not None:
        logger.info(f"Training time (min): {elapsed_time/60.0:.2f}")
        wandb_run.summary["total_training_time_min"] = elapsed_time/60.0
        wandb_run.summary["training_time_per_step_sec"] = elapsed_time/trainer.state.global_step
        wandb_run.summary["max_cuda_allocate_GB"] = torch.cuda.max_memory_allocated()/1024**3
    
    config['total_training_time'] = elapsed_time/60.0
    config['training_time_per_step_sec'] = elapsed_time/trainer.state.global_step
    config['max_cuda_allocate_GB'] = torch.cuda.max_memory_allocated()/1024**3

    json.dump(config, open(os.path.join(training_args.output_dir,f"{run_name}.json"), "w"), indent=4)
    print("Trainer Done.")

if __name__ == "__main__":
    accelerator = Accelerator()
    args = parse_args()
    main(accelerator, args)
    accelerator.wait_for_everyone() 
    accelerator.end_training()
    exit(0)
