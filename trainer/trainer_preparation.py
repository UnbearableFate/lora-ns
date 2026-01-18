import inspect
import os
from transformers import (
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from utils.metrics import get_glue_metrics_function
from optimizer.muon import SingleDeviceMuonWithAuxAdam
from accelerate import Accelerator
from .DistributedSvdRefactorTrainer import DistributedSvdRefactorTrainer
logger = logging.getLogger(__name__)


TRAINER_REGISTRY: Dict[str, Type[Trainer]] = {
    "Trainer": Trainer,
    "CleanedSvdRefTrainer": DistributedSvdRefactorTrainer,
}

class CompletionDataCollator:
    """Pad input/label pairs for completion-style causal LM supervision."""

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id or 0
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = -100

    def _pad_sequences(self, sequences, pad_value):
        if not sequences:
            return torch.empty(0)

        max_length = max(len(seq) for seq in sequences)
        if self.pad_to_multiple_of:
            remainder = max_length % self.pad_to_multiple_of
            if remainder:
                max_length += self.pad_to_multiple_of - remainder

        padded = []
        for seq in sequences:
            pad_len = max_length - len(seq)
            padded.append(seq + [pad_value] * pad_len)
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, features):
        input_ids = [feature["input_ids"] for feature in features]
        attention_masks = [feature.get("attention_mask") for feature in features]
        labels = [feature.get("labels") for feature in features]

        batch_input_ids = self._pad_sequences(input_ids, self.pad_token_id)

        if all(mask is not None for mask in attention_masks):
            batch_attention_mask = self._pad_sequences(attention_masks, 0)
        else:
            batch_attention_mask = torch.zeros_like(batch_input_ids)
            for idx, ids in enumerate(input_ids):
                batch_attention_mask[idx, : len(ids)] = 1

        if any(label is not None for label in labels):
            batch_labels = self._pad_sequences(
                [label if label is not None else [] for label in labels],
                self.label_pad_token_id,
            )
        else:
            batch_labels = batch_input_ids.clone()

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }


def _split_lora_and_non_lora_parameters(model) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Return LoRA parameters (identified via name) and the remaining trainable params."""
    lora_params: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        else:
            other_params.append(param)
    return lora_params, other_params


def _maybe_build_muon_optimizer(config: dict, model, training_args: TrainingArguments):
    trainer_config = _get_trainer_config(config)
    muon_cfg = trainer_config.get("muon_optimizer") or {}
    if not muon_cfg.get("enabled"):
        return None

    lora_params, other_params = _split_lora_and_non_lora_parameters(model)
    if not lora_params:
        raise ValueError("Muon optimizer enabled but no LoRA parameters were found.")

    def _get_float(key: str, default: float) -> float:
        value = muon_cfg.get(key, default)
        return float(value)

    def _get_tuple(key: str, default: Tuple[float, float]) -> Tuple[float, float]:
        value = muon_cfg.get(key, default)
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(f"{key} must have two values (beta1, beta2)")
            return float(value[0]), float(value[1])
        if value is None:
            return float(default[0]), float(default[1])
        raise ValueError(f"Unsupported value for {key}: {value}")

    muon_group = dict(
        params=lora_params,
        lr=_get_float("muon_lr", _get_float("lr", training_args.learning_rate)),
        momentum=_get_float("muon_momentum", _get_float("momentum", 0.95)),
        weight_decay=_get_float("muon_weight_decay", _get_float("weight_decay", training_args.weight_decay)),
        use_muon=True,
    )

    param_groups: List[Dict[str, Any]] = []
    if other_params:
        param_groups.append(
            dict(
                params=other_params,
                lr=_get_float("adam_lr", training_args.learning_rate),
                betas=_get_tuple("adam_betas", (training_args.adam_beta1, training_args.adam_beta2)),
                eps=_get_float("adam_eps", training_args.adam_epsilon),
                weight_decay=_get_float("adam_weight_decay", training_args.weight_decay),
                use_muon=False,
            )
        )

    param_groups.append(muon_group)
    logger.info(
        "Using SingleDeviceMuonWithAuxAdam optimizer for %d LoRA parameters (muon_lr=%s, momentum=%s)",
        len(lora_params),
        muon_group["lr"],
        muon_group["momentum"],
    )
    return SingleDeviceMuonWithAuxAdam(param_groups)


def _get_trainer_config(config: dict) -> Dict[str, Any]:
    return config.get("trainer", {}) or {}

def _parse_int_list(value) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        return [int(p) for p in parts if p]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]

def _get_global_batch_size(config: dict, training_args: TrainingArguments) -> int:
    training_cfg = config.get("training", {}) or {}
    if "global_batch_size" in training_cfg:
        return int(training_cfg["global_batch_size"])
    world_size = getattr(training_args, "world_size", 1) or 1
    return int(training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size)


def _resolve_trainer_class(config: dict) -> Type[Trainer]:
    trainer_name = _get_trainer_config(config).get("name", "Trainer")
    if trainer_name not in TRAINER_REGISTRY:
        raise ValueError(f"Unsupported trainer: {trainer_name}")
    return TRAINER_REGISTRY[trainer_name]


def _build_trainer_kwargs_from_config(
    config: dict, trainer_cls: Type[Trainer], existing_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Pick out trainer-specific kwargs declared in config["trainer"] that
    are explicitly accepted by trainer_cls.__init__. This prevents passing
    unexpected keys while still allowing custom trainers to receive their
    config.
    """
    trainer_config = _get_trainer_config(config)
    if not trainer_config:
        return {}

    init_sig = inspect.signature(trainer_cls.__init__)
    accepted_keys = {
        name
        for name, param in init_sig.parameters.items()
        if name != "self" and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }

    return {
        key: trainer_config[key]
        for key in accepted_keys
        if key in trainer_config and key not in existing_kwargs
    }


def _spectral_trainer_kwargs(config: dict, total_steps: Optional[int]) -> Dict[str, Any]:
    trainer_config = _get_trainer_config(config)
    if trainer_config.get("name", "Trainer") != "SpectralRefactorTrainer":
        return {}

    keys = [
        "refactor_every",
        "balance_lambda",
        "warmup_steps"
        "cooldown_steps",
        #"preserve_momentum",
        #"clear_momentum",
        "damping_eps",
        "clip_min_sigma",
    ]
    spectral_kwargs = {key: trainer_config[key] for key in keys if key in trainer_config}
    return spectral_kwargs


def _build_callbacks(config: dict) -> List[Any]:
    callbacks: List[Any] = []
    early_stopping_patience = config.get("early_stopping_patience")
    if early_stopping_patience and early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
    return callbacks


def _build_trainer(
    *,
    config: dict,
    model,
    tokenizer,
    dataset,
    training_args: TrainingArguments,
    data_collator,
    compute_metrics=None,
    preprocess_logits_for_metrics=None,
    callbacks: Optional[List[Any]] = None,
):
    trainer_name = _get_trainer_config(config).get("name", "Trainer")
    if trainer_name in {"CleanedSvdRefTrainer", "CleanedSvdRefactorTrainer"}:
        from .cleaned_svd_ref_trainer import get_cleaned_svd_ref_trainer

        trainer_kwargs: Dict[str, Any] = {
            "model": model,
            "args": training_args,
            "train_dataset": dataset["train"],
            "eval_dataset": dataset.get("validation"),
            "tokenizer": tokenizer,
            "data_collator": data_collator,
        }

        if compute_metrics is not None:
            trainer_kwargs["compute_metrics"] = compute_metrics
        if preprocess_logits_for_metrics is not None:
            trainer_kwargs["preprocess_logits_for_metrics"] = preprocess_logits_for_metrics
        if callbacks:
            trainer_kwargs["callbacks"] = callbacks

        trainer_config = _get_trainer_config(config)
        adjust_lora_alpha_at = _parse_int_list(trainer_config.get("adjust_lora_alpha_at"))
        basic_alpha = trainer_config.get("basic_alpha")
        if basic_alpha is None:
            basic_alpha = config.get("peft", {}).get("lora_alpha", 2.0)
        basic_alpha = float(basic_alpha)

        cleaned_kwargs = {
            "global_batch_size": _get_global_batch_size(config, training_args),
            "basic_alpha": basic_alpha,
        }
        if adjust_lora_alpha_at is not None:
            cleaned_kwargs["adjust_lora_alpha_at"] = adjust_lora_alpha_at

        for key in [
            "min_alpha_ratio",
            "max_alpha_ratio",
            "repeat_n",
            "repeat_warmup_ratio",
            "repeat_decay_ratio",
            "repeat_end_lr_rate",
            "final_warmup_ratio",
            "min_lr_rate",
            "repeat_decay_type",
            "final_decay_type",
            "warmup_start_lr_rate",
            "first_warmup_start_lr_rate",
            "last_epoch",
        ]:
            if key in trainer_config:
                cleaned_kwargs[key] = trainer_config[key]

        return get_cleaned_svd_ref_trainer(**trainer_kwargs, **cleaned_kwargs)

    trainer_cls = _resolve_trainer_class(config)

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset.get("validation"),
        "tokenizer": tokenizer,
        "data_collator": data_collator,
    }

    trainer_kwargs.update(_build_trainer_kwargs_from_config(config, trainer_cls, trainer_kwargs))

    if compute_metrics is not None:
        trainer_kwargs["compute_metrics"] = compute_metrics
    if preprocess_logits_for_metrics is not None:
        trainer_kwargs["preprocess_logits_for_metrics"] = preprocess_logits_for_metrics
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    #trainer_kwargs.update(_spectral_trainer_kwargs(config, total_steps=training_args.max_steps))

    muon_optimizer = _maybe_build_muon_optimizer(config, model, training_args)
    if muon_optimizer is not None:
        trainer_kwargs["optimizers"] = (muon_optimizer, None)
    trainer = trainer_cls(**trainer_kwargs)
    if str(config.get("model").get("name_or_path")).find("roberta") != -1:
        trainer.model_accepts_loss_kwargs = False
    return trainer

def setup_training_args(config: dict,train_dataset_length:int, num_processes, run_name) -> TrainingArguments:
    """Setup training arguments from config."""
    training_config = config["training"]
    
    # Convert learning_rate to float if it's a string (YAML might parse 3e-4 as string)
    learning_rate = training_config.get("learning_rate", 2e-4)
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    
    per_device_train_batch_size = training_config.get("per_device_train_batch_size", 8) 
    global_batch_size = training_config.get("global_batch_size", 8)
    gradient_accumulation_steps = global_batch_size // (per_device_train_batch_size * num_processes)
    assert global_batch_size == per_device_train_batch_size * gradient_accumulation_steps * num_processes, f"global_batch_size {global_batch_size} != per_device_train_batch_size {per_device_train_batch_size} * gradient_accumulation_steps {gradient_accumulation_steps} * num_processes {num_processes}"

    num_train_epochs=training_config.get("num_train_epochs", 3)
    max_steps = num_train_epochs * train_dataset_length // global_batch_size
    
    eval_steps = max_steps // training_config.get("total_eval_times", 50)
    save_steps = eval_steps

    model_name = config["model"]["name_or_path"].split('/')[-1]
    output_dir =  os.path.join(training_config.get("output_dir", "outputs") ,model_name, run_name) 
    log_dir = os.path.join(output_dir, "logs")
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir=output_dir,  # This will be overridden in train.py
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=training_config.get("weight_decay", 0.0),
        warmup_ratio=training_config.get("warmup_ratio", 0.03),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        
        # Evaluation
        eval_strategy= training_config.get("eval_strategy", "steps"),
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
        logging_dir=log_dir,
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
    task_name = config["dataset"].get("subset", "")
    compute_metrics = get_glue_metrics_function(task_name)
    
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
    callbacks = _build_callbacks(config)

    return _build_trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )

def get_collator(task_type,tokenizer, dataset,pad_multiple:int =8):
    if str(task_type).lower() == "causal_lm":
        train_columns = set(dataset["train"].column_names)
        if "labels" in train_columns:
            data_collator = CompletionDataCollator(
                tokenizer=tokenizer,
                pad_to_multiple_of=pad_multiple,
            )
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=pad_multiple,
            )
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_multiple)
    return data_collator

def train_causal_lm_task(config: dict, model, tokenizer, dataset, training_args):
    """Train causal LM tasks (e.g., MetaMathQA, GSM8K, Code-Feedback)."""
    logger.info("Training causal LM task")
    train_columns = set(dataset["train"].column_names)
    pad_multiple = config.get("training", {}).get("pad_to_multiple_of", 8)

    if "labels" in train_columns:
        data_collator = CompletionDataCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_multiple,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=pad_multiple,
        )
    
    # Get metrics function for this task
    task_name = config.get("task_name", "")
    compute_metrics = None #get_metrics_function(task_name, tokenizer=tokenizer)
    
    if compute_metrics:
        logger.info(f"Using metrics for task: {task_name}")

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)
    else:
        logger.info(f"No metrics defined for task: {task_name}, using loss only")
        preprocess_logits_for_metrics = None
    
    return _build_trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
