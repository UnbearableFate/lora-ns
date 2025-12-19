from copy import deepcopy
import os
from typing import Optional, Set
from peft import LoraModel, PeftModel
import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import Trainer, TrainingArguments

def iter_lora_factors_with_names(model: nn.Module,
                                 target_adapter_keys: Optional[Set[str]] = None):
    for module_name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for name in module.lora_A.keys():
                if target_adapter_keys and name not in target_adapter_keys:
                    continue
                yield module_name, name, module.lora_B[name].weight, module.lora_A[name].weight


class DistributedSvdRefactorTrainer(Trainer):
    """
    使用分布式低秩 SVD 重构 LoRA 因子，保持 mB A + B mA 的方向不变。
    """

    def __init__(
        self,
        *args,
        refactor_every: int = 100,
        cooldown_steps: int = 0,
        target_adapter_keys: Optional[Set[str]] = None,
        adjust_lora_alpha: bool = True,
        do_refactor: bool = True,
        keep_s: bool = False,
        balance_lambda: float = 0.5,
        alpha_beta: float = 0.25,
        alpha_clip_ratio: float = 1.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.refactor_every = max(1, int(refactor_every))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.target_adapter_keys = set(target_adapter_keys) if target_adapter_keys else None
        self.adjust_lora_alpha = bool(adjust_lora_alpha)
        self.do_refactor = bool(do_refactor)
        self.keep_s = bool(keep_s)
        
        self.balance_lambda = float(balance_lambda)
        self.alpha_beta = float(alpha_beta)
        self.alpha_clip_ratio = float(alpha_clip_ratio)
        
        self._last_lr_values = None
        self._lr_restart_last_checked_step = -1
        self.alpha_log = {}

    def get_exp_avg(self, param: Tensor) -> Optional[Tensor]:
        if not hasattr(self, "optimizer") or self.optimizer is None:
            return None
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p is param:
                    return self.optimizer.state.get(p, {}).get("exp_avg", None)
        return None

    @torch.no_grad()
    def distributed_low_rank_refactor(self, do_refactor: bool = True ,adjust_lora_alpha: bool = True, keep_s :bool = False):
        is_dist = self.accelerator.num_processes > 1 and dist.is_initialized()
        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index

        model = self.model
        was_training = model.training
        model.eval()

        variance_of_layers = {}
        device_for_broadcast = None
        broadcast_works = []

        for idx, (module_name, name, B, A) in enumerate(
            iter_lora_factors_with_names(model, self.target_adapter_keys)
        ):
            lora_r = A.shape[0]
            compute_here = (not is_dist) or (rank == idx % world_size)
            if device_for_broadcast is None:
                device_for_broadcast = B.device

            mB_state = self.get_exp_avg(B)
            mA_state = self.get_exp_avg(A)

            if compute_here:
                # 始终对 B @ A 做 SVD，作为正交基
                base = B @ A
                U, S, Vh = torch.svd_lowrank(base.float(), q=lora_r)
                variance_of_layers[f"{module_name}.{name}"] = float((S ** 2).sum().item())
                if keep_s:
                    # maybe worse result
                    S_bar = S.mean()
                    S_tilde = (1.0 - float(self.balance_lambda)) * S + float(self.balance_lambda) * S_bar
                    S_half = torch.diag(torch.sqrt(torch.clamp(S_tilde, min=0.0)))
                    B_new = (U @ S_half).to(B.dtype)
                    A_new = (S_half @ Vh.t()).to(A.dtype)
                else:
                    B_new = U.to(B.dtype)
                    A_new = Vh.t().to(A.dtype)

                # 计算 T_B^{-1}, T_A^{-1} 以保持 mB A + B mA 的方向
                if mA_state is not None and mB_state is not None and do_refactor:
                    B_pinv = torch.linalg.pinv(B.float())
                    A_pinv = torch.linalg.pinv(A.float())
                    T_B = B_pinv @ B_new.float()  # B_new = B T_B
                    T_A = A_new.float() @ A_pinv  # A_new = T_A A
                    T_B_inv = torch.linalg.pinv(T_B)
                    T_A_inv = torch.linalg.pinv(T_A)
                    mB_new = (
                        mB_state @ T_A_inv.to(mB_state.dtype)
                        if mB_state is not None
                        else torch.zeros_like(B)
                    )
                    mA_new = (
                        T_B_inv.to(mA_state.dtype) @ mA_state
                        if mA_state is not None
                        else torch.zeros_like(A)
                    )
                    mB_state.copy_(mB_new)
                    mA_state.copy_(mA_new)
                
                if do_refactor:
                    B.copy_(B_new)
                    A.copy_(A_new)

            if is_dist and do_refactor:
                broadcast_works.append(
                    dist.broadcast(B, src=idx % world_size, async_op=True).get_future()
                )
                broadcast_works.append(
                    dist.broadcast(A, src=idx % world_size, async_op=True).get_future()
                )
                if mB_state is not None:
                    broadcast_works.append(
                        dist.broadcast(mB_state, src=idx % world_size, async_op=True).get_future()
                    )
                if mA_state is not None:
                    broadcast_works.append(
                        dist.broadcast(mA_state, src=idx % world_size, async_op=True).get_future()
                    )

        if is_dist and broadcast_works and do_refactor:
            torch.futures.wait_all(broadcast_works)

        if is_dist:
            gathered_variances = [None] * world_size
            dist.all_gather_object(gathered_variances, variance_of_layers)
            variance_of_layers = {}
            for part in gathered_variances:
                variance_of_layers.update(part)

        if adjust_lora_alpha and variance_of_layers:
            if rank == 0:
                avg_of_global_variance = sum(variance_of_layers.values()) / len(variance_of_layers)
                clip_ratio = self.alpha_clip_ratio
                beta = self.alpha_beta
                for module_name, sub_module in model.named_modules():
                    if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B") and hasattr(sub_module, "lora_alpha"):
                        for adapter_name in sub_module.lora_A.keys():
                            if self.target_adapter_keys and adapter_name not in self.target_adapter_keys:
                                continue
                            if adapter_name not in sub_module.lora_alpha:
                                continue
                            layer_key = f"{module_name}.{adapter_name}"
                            if layer_key not in variance_of_layers:
                                continue
                            layer_var = variance_of_layers[layer_key]
                            ratio = layer_var / avg_of_global_variance
                            ratio_new = ratio ** beta
                            if clip_ratio is not None and clip_ratio > 0:
                                ratio_new = max(1.0 / clip_ratio, min(ratio_new, clip_ratio))
                            sub_module.lora_alpha[adapter_name] = ratio_new

            alpha_values = []
            alpha_indices = []
            for module_name, sub_module in model.named_modules():
                if hasattr(sub_module, "lora_A") and hasattr(sub_module, "lora_B") and hasattr(sub_module, "lora_alpha"):
                    for adapter_name in sub_module.lora_A.keys():
                        if self.target_adapter_keys and adapter_name not in self.target_adapter_keys:
                            continue
                        if adapter_name not in sub_module.lora_alpha:
                            continue
                        alpha_indices.append((module_name, adapter_name))
                        if rank == 0:
                            alpha_values.append(float(sub_module.lora_alpha[adapter_name]))
                        else:
                            alpha_values.append(0.0)

            if alpha_values:
                if device_for_broadcast is None:
                    device_for_broadcast = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                alpha_tensor = torch.tensor(alpha_values, device=device_for_broadcast, dtype=torch.float32)
                if is_dist:
                    dist.broadcast(alpha_tensor, src=0)

                modules_dict = dict(model.named_modules())
                for idx, (module_name, adapter_name) in enumerate(alpha_indices):
                    sub_module = modules_dict[module_name]
                    if f"{module_name}.{adapter_name}" not in self.alpha_log:
                        self.alpha_log[f"{module_name}.{adapter_name}"] = [sub_module.lora_alpha[adapter_name]]
                    sub_module.lora_alpha[adapter_name] = float(alpha_tensor[idx].item())    
                    self.alpha_log[f"{module_name}.{adapter_name}"].append(sub_module.lora_alpha[adapter_name])
                    print(f"Adjusted alpha of {module_name}.{adapter_name} to {sub_module.lora_alpha[adapter_name]:.4f}")

        if was_training:
            model.train()

    def _is_lr_restart(self):
        if not hasattr(self, "lr_scheduler") or self.lr_scheduler is None:
            return False

        step = self.state.global_step
        # Avoid double-processing the same optimizer step when using gradient accumulation
        if self._lr_restart_last_checked_step == step:
            return False
        self._lr_restart_last_checked_step = step

        if not hasattr(self.lr_scheduler, "get_last_lr"):
            return False

        try:
            warmup_steps = self.args.get_warmup_steps(self.state.max_steps)
        except Exception:
            warmup_steps = getattr(self.args, "warmup_steps", 0) or 0

        current_lrs = list(self.lr_scheduler.get_last_lr())
        if self._last_lr_values is None:
            self._last_lr_values = current_lrs
            return False

        # After warmup, cosine with hard restarts is strictly decreasing inside a cycle.
        # Any lr increase indicates a restart just happened at the previous step.
        is_restart = step > warmup_steps and any(
            cur_lr > prev_lr * (1.0 + 1e-12) for cur_lr, prev_lr in zip(current_lrs, self._last_lr_values)
        )
        self._last_lr_values = current_lrs
        return is_restart

    def training_step(self, model: nn.Module, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        step = self.state.global_step

        if self.optimizer is None:
            return loss
        
        if step > self.state.max_steps - self.cooldown_steps:
            return loss
        if step < self.refactor_every or (step % self.refactor_every) != 0:
            return loss
        
        #if not self._is_lr_restart():
        #    return loss

        self.distributed_low_rank_refactor(
            do_refactor = self.do_refactor,
            adjust_lora_alpha = self.adjust_lora_alpha,
            keep_s = self.keep_s)
        
        return loss


def restart_init_train(trainning_args:TrainingArguments,init_steps,model:PeftModel| LoraModel, data_collator ,train_dataset):
    training_arguments0 = deepcopy(trainning_args)
    training_arguments0.num_train_epochs = 0
    training_arguments0.max_steps = init_steps
    training_arguments0.output_dir = os.path.join(training_arguments0.output_dir,"initial_phase")
    training_arguments0.report_to = "none"
    training_arguments0.eval_strategy = "no"
    training_arguments0.save_strategy = "no"
    training_arguments0.load_best_model_at_end = False
    training_arguments0.data_seed = training_arguments0.data_seed* 2 + 1  # to avoid mixing data orders
    trainer0 = DistributedSvdRefactorTrainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = None,
        args = training_arguments0,
        data_collator = data_collator, 
        refactor_every = 1000000,
    )
    trainer0.train()
    trainer0.distributed_low_rank_refactor(adjust_lora_alpha=True, do_refactor=True, keep_s=False)
    return model