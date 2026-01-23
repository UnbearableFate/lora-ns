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
from utils.common import extract_experiment_tags, seed_everything, get_run_name, write_eval_results_csv

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
        "--init_lora_weights",
        type=str,
        default=None,
        help="Initialize LoRA weights with a specific method",
    )
    parser.add_argument(
        "--use_sr_trainer",
        action='store_true',
        help="Use SR-init trainer",
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=8,
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
    seed = args.seed
    config["training"]["seed"] = seed
    config["training"]["data_seed"] = seed
    if args.init_lora_weights is not None:
        if str(args.init_lora_weights).lower() == "true":
            args.init_lora_weights = True
        config["peft"]["init_lora_weights"] = args.init_lora_weights
        logger.info(f"Overriding init_lora_weights to {args.init_lora_weights} from command line argument")
    if args.use_sr_trainer:
        config["trainer"]["name"] = "CleanedSvdRefactorTrainer"
        logger.info(f"Using SR-init trainer as specified in command line argument")
    if seed < 20 :
        config["wandb"]["enabled"] = True
        config["wandb"]["online"] = True
        logger.info(f"Enabling WandB online logging for seed {seed} < 20")
    validate_config(config)

    wandb_config = config.get("wandb")
    wandb_run = None
    run_name = get_run_name(config, timestamp=args.timestamp)
   
    if wandb_config and wandb_config["enabled"] and accelerator.is_main_process:
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
        wandb_project = f"{wandb_config.get("project", "nlp")}_{config["model"].get("name_or_path","").replace("/","-")}_{config.get("dataset",{}).get("name","").split('/')[-1]}"
        if config.get("dataset",{}).get("subset"):
            wandb_project += "_"+config.get("dataset",{}).get("subset")
        
        wandb_run = wandb.init(
            project=wandb_project,
            name=run_name,
            tags=all_tags,
            config=config)

    start_time = time.time()
    # Seed everything
    
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

        if accelerator.is_main_process:
            extra_info = ""
            if config.get("trainer", {}).get("name") == "CleanedSvdRefactorTrainer":
                extra_info = f"sr&rp{config['trainer'].get('repeat_n',0)}&rwr{config['trainer'].get('repeat_warmup_ratio',0)}"
            peft_cfg = config.get("peft", {})
            path_parts = [config.get("generation", {}).get("output_dir","./eval_outputs"), config["model"]["name_or_path"].split("/")[-1], config.get("dataset", {}).get("name")]
            if config.get("dataset", {}).get("subset"):
                path_parts.append(config.get("dataset", {}).get("subset"))
            csv_path = os.path.join(*path_parts, "eval_results.csv")
            write_eval_results_csv(
                csv_path,
                metrics,
                base_model_name=config["model"]["name_or_path"],
                dataset_name=config.get("dataset", {}).get("name", ""),
                subset=config.get("dataset", {}).get("subset", ""),
                timestamp=args.timestamp,
                init_lora_weights=peft_cfg.get("init_lora_weights"),
                extra=extra_info,
                seed=config.get("training", {}).get("seed"),
                run_time=elapsed_time/60.0,
                peft_config=peft_cfg,
            )
            logger.info("Wrote eval CSV results to %s", csv_path) 
    
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
