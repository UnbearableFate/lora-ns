
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments

from trainer.SpectralRefactorTrainer import SpectralRefactorTrainer

import logging

from utils.metrics import get_metrics_function

logger = logging.getLogger(__name__)

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
        do_train=True,
        do_eval=True,
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
        report_to=training_config.get("report_to", ["wandb"]),
        
        # Misc
        seed=training_config.get("seed", 42),
        data_seed=training_config.get("data_seed", 42),
        remove_unused_columns=training_config.get("remove_unused_columns", True),
        save_total_limit=training_config.get("save_total_limit", 3),
        
        # Distributed training
        ddp_find_unused_parameters=False,

        # DataLoader settings
        dataloader_pin_memory =training_config.get("dataloader_pin_memory", True),
        dataloader_num_workers =training_config.get("dataloader_num_workers", 2),
        dataloader_persistent_workers=training_config.get("dataloader_persistent_workers", False),
        dataloader_prefetch_factor= training_config.get("dataloader_prefetch_factor", 2),
    )
    
    return training_args


def train_classification_task(config: dict, model, tokenizer, dataset, training_args):
    """Train classification tasks (e.g., GLUE)."""
    logger.info("Training classification task")
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Metrics
    task_name = config.get("task_name", "mrpc")
    compute_metrics = get_metrics_function(task_name)
    
    # Preprocess logits for metrics (especially for T5 models)
    def preprocess_logits_for_metrics(logits, labels):
        """
        Preprocess logits before computing metrics.
        This is especially needed for T5 models which may output sequence-level logits.
        """
        if isinstance(logits, tuple):
            # Take the first element if it's a tuple
            logits = logits[0]
        
        # For T5ForSequenceClassification, logits might be 3D: (batch, seq_len, num_labels)
        # We need to take the last token's logits for classification
        if len(logits.shape) == 3:
            logits = logits[:, -1, :]
        
        return logits
    
    # Trainer
    callbacks = []
    early_stopping_patience = config.get("early_stopping_patience", None) 
    if early_stopping_patience is not None and early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))


    common_trainer_params = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks
    )
    if config.get("trainer", {}).get("name") == "SpectralRefactorTrainer":
        trainer = SpectralRefactorTrainer(
            **common_trainer_params,
            refactor_every=config["trainer"].get("refactor_every", 100),
            warmup_steps=config["trainer"].get("warmup_steps", 0),
            refactor_mode=config["trainer"].get("refactor_mode", "balanced"),
            balance_lambda=config["trainer"].get("balance_lambda", 1.0),
            preserve_momentum=config["trainer"].get("preserve_momentum", False),
            clear_momentum=config["trainer"].get("clear_momentum", True),
            damping_eps=config["trainer"].get("damping_eps", 0.0),
            clip_min_sigma=config["trainer"].get("clip_min_sigma", 0.0),
        ) 
    else:
        trainer = Trainer(
            **common_trainer_params
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation")
    )

    return trainer
