import torch
import typing as tp

def find_all_linear_modules(model) -> tp.List[str]:
    r"""
    Finds all available modules to apply lora.
    """
    linear_cls = torch.nn.Linear

    output_layer_names = ["lm_head", "embed_tokens"]

    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and not any(
            [output_layer in name for output_layer in output_layer_names]
        ):
            module_names.add(name.split(".")[-1])
    return list(module_names)

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

def get_run_name(config, timestamp: tp.Optional[str] = None) -> str:
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
    if config.get("trainer", {}).get("name") == "CleanedSvdRefactorTrainer":
        wandb_run_name += f"_sr-init#rp{config.get('trainer', {}).get('repeat_n',0)}&rwr{config.get('trainer', {}).get('repeat_warmup_ratio',0)}"
    wandb_run_name += f"_s{config['training']['seed']}_{timestamp}"
    return wandb_run_name


def _rewrite_csv_with_extended_header(csv_path, fieldnames):
    import csv
    import os

    if not os.path.exists(csv_path):
        return

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)


def append_row_to_csv(csv_path, row):
    import csv
    import os

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            header_reader = csv.reader(f)
            existing_header = next(header_reader, [])
        fieldnames = list(dict.fromkeys(existing_header + [k for k in row.keys() if k not in existing_header]))
        if fieldnames != existing_header:
            _rewrite_csv_with_extended_header(csv_path, fieldnames)
    else:
        fieldnames = list(row.keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(row)

def get_info_from_model_path(model_path):
    import os

    normalized = os.path.normpath(model_path)
    base_name = os.path.basename(normalized)
    if "checkpoint" in base_name:
        base_name = os.path.basename(os.path.dirname(normalized))
    parts = base_name.split("_")
    info = {"run_name": base_name}
    if parts:
        info["timestamp"] = parts[-1]
    if len(parts) >= 2 and parts[-2].startswith("s"):
        try:
            info["seed"] = int(parts[-2][1:])
        except ValueError:
            pass
    if len(parts) >= 3 and parts[-3].startswith("sr"):
        info["extra"] = parts[-3]
    else:
        info["extra"] = "none"
    return info


def write_acc_to_csv(filepath, model_name, eval_dataset_name, acc, extra_fields=None):
    import os

    csv_dir = filepath
    if filepath:
        base, ext = os.path.splitext(filepath)
        if ext:
            csv_dir = os.path.dirname(filepath)
    if not csv_dir:
        csv_dir = "."
    csv_path = os.path.join(csv_dir, "eval_results.csv")
    row = {
        "model": model_name,
        "dataset": eval_dataset_name,
        "accuracy": f"{acc:.5f}",
    }
    if extra_fields:
        row.update(extra_fields)
    append_row_to_csv(csv_path, row)

def get_lora_rank(adapter_path):
    import json
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path, "r", encoding="utf8") as f:
        config = json.load(f)
    lora_rank = config.get("r")
    if lora_rank is None:
        raise ValueError(f"LoRA rank 'r' not found in {config_path}")
    return int(lora_rank)