from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import time
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader
import tqdm
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding

from peft import LoraConfig, get_peft_model, initialize_lora_eva_weights, prepare_model_for_kbit_training
from peft.tuners.lora.corda import preprocess_corda
from peft.tuners.lora.config import CordaConfig, EvaConfig

from my_peft import LoraGAConfig
from accelerate import Accelerator

from trainer.trainer_preparation import CompletionDataCollator

import logging
logger = logging.getLogger(__name__)

@dataclass
class LoraHyperparameters:
    """Hyperparameters for the LoRA-family adapters."""

    variant: str = "lora"  # lora, dora, qalora, rslora
    task_type: str = "CAUSAL_LM"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    )
    init_lora_weights: Union[bool, str, None] = True # ["gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]", "corda", "loftq", "orthogonal"]
    init_num_samples: int = 512
    init_batch_size: int = 8
    corda_method: str = "kpm"  # kpm or ipm

    loraga_direction : str = "ArB2r"  # ArB2r, A2rB, BrA2r
    loraga_dtype : torch.dtype = torch.float32
    exclude_modules: Optional[Union[List[str], str]] = None

    cache_dir: Optional[str] = "data_cache"
    unique_cache_filename : Optional[str] = None
    model_name_or_path: Optional[str] = None
    dataset_name: Optional[str] = None
    subdataset_name: Optional[str] = None
    init_seed: int = 1337

    def __post_init__(self):
        unique_cache_filename = f"{self.model_name_or_path.replace('/', '-')}_{self.dataset_name}"
        if self.subdataset_name:
            unique_cache_filename += f"_{self.subdataset_name}"
        self.unique_cache_filename = f"{unique_cache_filename}_r{self.r}_dp{self.init_num_samples}_bs{self.init_batch_size}_{self.init_seed}.pt"
    
    def get_unique_cache_path(self,path_mid_name) -> str:
        parent_path = Path(self.cache_dir, path_mid_name)
        if not parent_path.exists():
            parent_path.mkdir(parents=True, exist_ok=True)
        return parent_path.joinpath(self.unique_cache_filename).as_posix()

_VARIANT_TO_FLAGS = {
    "lora": {"use_dora": False, "use_rslora": False, "use_qalora": False},
    "dora": {"use_dora": True, "use_rslora": False, "use_qalora": False},
    "rslora": {"use_dora": False, "use_rslora": True, "use_qalora": False},
    "qalora": {"use_dora": False, "use_rslora": False, "use_qalora": True},
}


def build_LoraHyperparameters_from_yaml_dict(cfg_dict) -> LoraHyperparameters:
    peft_config = cfg_dict.get("peft", {})
    loraga_config = cfg_dict.get("loraga", {})
    lora_init_kwargs = peft_config.get("lora_init_kwargs", {})
    return LoraHyperparameters(
        variant= peft_config['variant'],
        task_type= peft_config.get("task_type", "CAUSAL_LM"),
        r= peft_config['lora_r'],
        alpha= peft_config['lora_alpha'],
        dropout= peft_config['lora_dropout'],
        bias= peft_config['bias'],
        target_modules= peft_config['target_modules'],
        init_lora_weights= peft_config['init_lora_weights'],
        init_num_samples= lora_init_kwargs.get('init_num_samples', 512),
        init_batch_size= lora_init_kwargs.get('init_batch_size', 8),

        corda_method= lora_init_kwargs.get('corda_method', "kpm"),
        loraga_direction= lora_init_kwargs.get('loraga_direction', "ArB2r") if loraga_config else "ArB2r",
        loraga_dtype= torch.float32,
        
        cache_dir= peft_config.get('cache_dir', "data_cache"),
        exclude_modules= peft_config.get("exclude_modules", None),
        model_name_or_path= cfg_dict["model"]["name_or_path"],
        dataset_name= cfg_dict["dataset"]["name"],
        subdataset_name= cfg_dict["dataset"].get("subset", None),
        init_seed= lora_init_kwargs.get('init_seed', cfg_dict['training'].get("seed", 42) *2 +1),
    )

def get_lora_config(lora_cfg: LoraHyperparameters) -> LoraConfig | LoraGAConfig:
    variant = lora_cfg.variant.lower()
    if variant not in _VARIANT_TO_FLAGS:
        raise ValueError(f"Unsupported LoRA variant: {variant}")
    peft_config = None
    if lora_cfg.init_lora_weights != "lora_ga":
        corda_config = None
        eva_config = None
        if lora_cfg.init_lora_weights == "corda":
                corda_config = CordaConfig(
                    corda_method=lora_cfg.corda_method, # kpm or ipm
                    cache_file=lora_cfg.get_unique_cache_path("corda_cache"),
                    covariance_file=lora_cfg.get_unique_cache_path("covariance_file"),
                )
        elif lora_cfg.init_lora_weights == "eva":
            eva_config = EvaConfig()

        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            bias=lora_cfg.bias,
            target_modules=list(lora_cfg.target_modules),
            exclude_modules=lora_cfg.exclude_modules,
            task_type=lora_cfg.task_type,
            init_lora_weights=lora_cfg.init_lora_weights,
            corda_config=corda_config,
            eva_config=eva_config,
            **_VARIANT_TO_FLAGS[variant],
        )
        
    else:
        peft_config = LoraGAConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            bias=lora_cfg.bias,
            target_modules=list(lora_cfg.target_modules),
            task_type=lora_cfg.task_type,
            bsz=lora_cfg.init_batch_size,
            direction=lora_cfg.loraga_direction,
            dtype= lora_cfg.loraga_dtype,
            gradient_save_path=lora_cfg.get_unique_cache_path("loraga_gradient"),
            **_VARIANT_TO_FLAGS[variant],
        )
    
    print(f"lora config: {peft_config}")
    return peft_config

def attach_lora_adapter(base_model,lora_cfg: LoraConfig|LoraGAConfig, train_dataset,tokenizer, init_num_samples:int, batch_size:int,seed: int, accelerator: Accelerator, save_dir: Path = None):
    if lora_cfg.init_lora_weights not in ["corda", "eva", "lora_ga"]:
        return get_peft_model(base_model, lora_cfg)
    sub_dataset = train_dataset.shuffle(seed=seed).select(range(init_num_samples))
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    columns_to_remove = [col for col in sub_dataset.column_names if col not in columns_to_keep]
    sub_dataset = sub_dataset.remove_columns(columns_to_remove) if columns_to_remove else sub_dataset
    if str(lora_cfg.task_type).lower() == "causal_lm":
        if "labels" in sub_dataset.column_names:
            data_collator = CompletionDataCollator(
                tokenizer=tokenizer,
                pad_to_multiple_of=8,
            )
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8,
            )
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        )

    if lora_cfg.init_lora_weights == "corda":
        return get_peft_model_with_corda(base_model, lora_cfg, sub_dataset,data_collator,accelerator=accelerator)
    elif lora_cfg.init_lora_weights == "eva":
        return get_peft_model_with_eva(base_model, lora_cfg, sub_dataset,data_collator ,batch_size ,accelerator=accelerator)
    elif lora_cfg.init_lora_weights == "lora_ga":
        # Some decoder-only checkpoints (e.g. Qwen3) ship without a padding token in the config.
        # Transformers sequence-classification heads will error on batch_size > 1 unless
        # `model.config.pad_token_id` is set.
        if getattr(getattr(base_model, "config", None), "pad_token_id", None) is None:
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    if hasattr(base_model, "resize_token_embeddings"):
                        base_model.resize_token_embeddings(len(tokenizer))
            if getattr(getattr(base_model, "config", None), "pad_token_id", None) is None:
                base_model.config.pad_token_id = tokenizer.pad_token_id
            generation_config = getattr(base_model, "generation_config", None)
            if generation_config is not None and getattr(generation_config, "pad_token_id", None) is None:
                generation_config.pad_token_id = tokenizer.pad_token_id
        return get_peft_model_with_lora_ga(base_model, lora_cfg, sub_dataset,data_collator ,batch_size,accelerator=accelerator)

def freeze_lora_A_weights(peft_model):
    for name, param in peft_model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

def get_peft_model_with_corda(base_model,lora_cfg: LoraConfig,sub_dataset,data_collator,accelerator: Accelerator):
    calib_loader = DataLoader(
        sub_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator,
    )
    base_model.to(accelerator.device)
    device = base_model.device
    print(f"Running Corda preprocessing on device: {device}")
    #calib_loader = accelerator.prepare(calib_loader)

    @torch.no_grad()
    def _run_model():
        was_training = base_model.training
        base_model.eval()
        # for batch in calib_loader:
        #     batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        #     base_model(**batch)
        for batch in tqdm.tqdm(calib_loader, desc="corda preprocessing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            base_model(**batch)
        if was_training:
            base_model.train()

    print(f"Starting Corda preprocessing... with sub-dataset of size {len(sub_dataset)}")
    preprocess_corda(
        base_model,
        lora_cfg,
        run_model=_run_model,
    )
    return get_peft_model(base_model, lora_cfg)

def get_peft_model_with_eva(
        base_model,
        lora_cfg: LoraConfig,
        sub_dataset,
        data_collator,
        batch_size: int,
        accelerator: Accelerator,
    ):
    
    def get_input(examples):
        batch = data_collator(examples)
        print(batch.__class__)
        return {k: v.to(base_model.device) for k, v in batch.items()}
    
    dataloader = DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        collate_fn=get_input,
    )
    dataloader = accelerator.prepare(dataloader)
    base_model.to(accelerator.device)

    peft_model = get_peft_model(base_model, lora_cfg, low_cpu_mem_usage=True)
    print(f"Initializing Eva LoRA weights... with sub-dataset of size {len(sub_dataset)}")
    initialize_lora_eva_weights(peft_model, dataloader)
    return peft_model

__all__ = [
    "load_base_model",
    "load_tokenizer",
    "get_lora_config",
]

def get_peft_model_with_lora_ga(
        model,
        lora_ga_cfg: LoraGAConfig,
        sub_dataset,
        data_collator,
        batch_size: int,
        accelerator,
    ):

    from my_peft.utils.lora_ga_utils import (
                LoraGAContext,
                estimate_gradient,
            )
    from my_peft import get_peft_model as my_get_peft_model

    gradient_loader = DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    named_grad = estimate_gradient(
        model=model,
        dataloader=gradient_loader,
        accelerator=accelerator,
        quant_flag=False,
        origin_type=None,
        quant_type=None,
        no_split_module_classes=None,
        grad_save_path=lora_ga_cfg.gradient_save_path,
    )
    start_time = time.time()
    with LoraGAContext(model=model, named_grad=named_grad):
        model = my_get_peft_model(model=model, peft_config=lora_ga_cfg)
    logger.info(f"LoRA-GA initialization took {time.time() - start_time:.2f} seconds")
    
    return model
