from transformers import Trainer
import torch

class MomentumPolarizedTrainer(Trainer):
    def __init__(self, *args, svd_every=10, rank_mode="r", svd_niter=2, target_keys=None, **kw):
        super().__init__(*args, **kw)
        self.svd_every = max(1, int(svd_every))
        self.rank_mode = rank_mode  # "r" or "2r" or int
        self.svd_niter = svd_niter
        self.target_keys = set(target_keys) if target_keys else None

    @torch.no_grad()
    def _iter_lora_pairs(self):
        for m in self.model.modules():
            if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
                for key in m.lora_A.keys():
                    if self.target_keys and key not in self.target_keys:
                        continue
                    A = m.lora_A[key].weight  # [r, d_in]
                    B = m.lora_B[key].weight  # [d_out, r]
                    yield (A, B)

    @torch.no_grad()
    def _polarize_momentum_once(self, optimizer):
        # 低频触发
        if self.state.global_step % self.svd_every != 0:
            return
        for group in optimizer.param_groups:
            for p in group["params"]:
                # 先确保 state 初始化
                _ = optimizer.state[p]

        # 快速索引 AdamW 动量
        def get_exp_avg(t):
            st = optimizer.state[t]
            return st.get("exp_avg", None)

        for A, B in self._iter_lora_pairs():
            mA = get_exp_avg(A)
            mB = get_exp_avg(B)
            if mA is None or mB is None:
                continue

            # 1) 组装等效矩阵动量 M_W = mB A + B mA
            M = mB @ A + B @ mA   # [d_out, d_in]
            d_out, d_in = M.shape
            r = A.shape[0]
            if self.rank_mode == "2r":
                k = min(2*r, min(d_out, d_in))
            elif self.rank_mode == "r":
                k = min(r, min(d_out, d_in))
            else:
                k = max(1, min(int(self.rank_mode), min(d_out, d_in)))

            # 2) 低秩 SVD，取 UV^T 作为正交方向（谱均衡）
            U, S, V = torch.svd_lowrank(M.float(), q=k, niter=self.svd_niter)
            Q = (U @ V.t()).to(dtype=M.dtype, device=M.device)

            # 3) 回写到动量缓冲：mB <- Q A^T, mA <- B^T Q
            mB.copy_(Q @ A.t())
            mA.copy_(B.t() @ Q)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx=0, **kw):
        # 在真正 step 前极分解动量
        self._polarize_momentum_once(optimizer)
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, **kw)