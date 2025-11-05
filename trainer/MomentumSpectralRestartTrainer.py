# -*- coding: utf-8 -*-
from typing import Optional, Set

import torch
from torch import nn
from transformers import Trainer

from .SpectralRefactorTrainer import iter_lora_factors

import logging

logger = logging.getLogger(__name__)


class MomentumSpectralRestartTrainer(Trainer):
    """
    Trainer that periodically recomputes LoRA factors from the optimizer momentum
    combination mB @ A + B @ mA and resets the associated Adam states. This implements
    the exploratory idea of restarting LoRA factors from the dominant subspace indicated
    by the accumulated momentum.
    """

    def __init__(self,
                 *args,
                 refactor_every: int = 200,
                 target_adapter_keys: Optional[Set[str]] = None,
                 min_svd_norm: float = 1e-8,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.refactor_every = max(1, int(refactor_every))
        self.target_adapter_keys = set(target_adapter_keys) if target_adapter_keys else None
        self.min_svd_norm = float(min_svd_norm)

    def _get_param_state(self, param: torch.Tensor) -> Optional[dict]:
        if not hasattr(self, "optimizer") or self.optimizer is None:
            return None
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p is param:
                    return self.optimizer.state.setdefault(p, {})
        return None

    def _get_state_tensor(self, param: torch.Tensor, key: str) -> Optional[torch.Tensor]:
        state = self._get_param_state(param)
        if not state:
            return None
        return state.get(key, None)

    def _zero_state_tensor(self, param: torch.Tensor, key: str) -> None:
        state = self._get_param_state(param)
        if not state:
            return
        tens = state.get(key, None)
        if tens is None:
            return
        tens.zero_()

    def _reset_adam_state(self, param: torch.Tensor) -> None:
        self._zero_state_tensor(param, "exp_avg")
        self._zero_state_tensor(param, "exp_avg_sq")

    @torch.no_grad()
    def _refactor_once(self) -> None:
        if self.state.global_step < self.refactor_every:
            return
        if self.state.global_step % self.refactor_every != 0:
            return
        if not hasattr(self, "optimizer") or self.optimizer is None:
            return

        logger.info(
            "[MomentumSpectralRestartTrainer] Step %s: momentum-guided spectral restart",
            self.state.global_step,
        )

        model = self.model
        was_training = model.training
        model.eval()

        for (B, A) in iter_lora_factors(model, self.target_adapter_keys):
            self._refactor_lora_pair(B, A)

        if was_training:
            model.train()

    @torch.no_grad()
    def _refactor_lora_pair(self, B: torch.Tensor, A: torch.Tensor) -> None:
        rank = B.shape[1]
        device = B.device

        mB = self._get_state_tensor(B, "exp_avg")
        mA = self._get_state_tensor(A, "exp_avg")
        B_float = B.float()
        A_float = A.float()

        momentum_term = torch.zeros(
            (B.shape[0], A.shape[1]),
            device=device,
            dtype=torch.float32,
        )

        if mB is not None:
            momentum_term = momentum_term + mB.to(device=device, dtype=torch.float32) @ A_float
        if mA is not None:
            momentum_term = momentum_term + B_float @ mA.to(device=device, dtype=torch.float32)

        norm_val = torch.linalg.norm(momentum_term)
        if torch.isnan(norm_val) or norm_val < self.min_svd_norm:
            return

        U, S, Vh = torch.linalg.svd(momentum_term, full_matrices=False)
        if U.shape[1] < rank or Vh.shape[0] < rank:
            rank = min(U.shape[1], Vh.shape[0])
        if rank == 0:
            return

        U_r = U[:, :rank].to(device=device, dtype=B.dtype)
        Vh_r = Vh[:rank, :].to(device=device, dtype=A.dtype)

        B.copy_(U_r)
        A.copy_(Vh_r)

        self._reset_adam_state(B)
        self._reset_adam_state(A)

    def training_step(self, model: nn.Module, inputs, num_items_in_batch: int):
        output = super().training_step(model, inputs, num_items_in_batch)
        self._refactor_once()
        return output
