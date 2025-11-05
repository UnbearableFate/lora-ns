from transformers import Trainer

from typing import Iterator, Optional, Tuple
import torch
from torch import nn
from transformers import Trainer

# ====== 工具函数：找到 PEFT/LoRA 层，并以 (lora_B, lora_A) 形式返回 ======
def iter_lora_factors(model: nn.Module) -> Iterator[Tuple[nn.Parameter, nn.Parameter]]:
    """
    遍历模型中所有含 LoRA 因子的层，返回 (B, A) 参数对。
    兼容 peft 的常见实现：每个 LoRA 包装层拥有 .lora_A[adapter].weight 与 .lora_B[adapter].weight
    如果你的项目有自定义命名，请在这里按需调整匹配逻辑。
    """
    for module in model.modules():
        # 典型 peft.lora 层：存在字典 lora_A / lora_B（多 adapter 场景）
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            # module.lora_A / lora_B: Dict[str, nn.Linear]
            for name in getattr(module, "lora_A").keys():
                A_lin = module.lora_A[name]
                B_lin = module.lora_B[name]
                if hasattr(A_lin, "weight") and hasattr(B_lin, "weight"):
                    yield (B_lin.weight, A_lin.weight)  # B: [d_out, r], A: [r, d_in]


# ====== 核心：LoRA 因子正交再牵引 ======
@torch.no_grad()
def lora_qr_retraction(B: torch.Tensor,
                       A: torch.Tensor,
                       mode: str = "both",
                       orth_on: str = "B_first") -> None:
    """
    对 (B, A) 执行薄 QR 正交化，并把 R 的缩放吸收到对侧因子里，严格保持 ΔW = B @ A 不变。

    参数：
      B: [d_out, r]   LoRA 左因子（列空间）
      A: [r, d_in]    LoRA 右因子（行空间）
      mode:
        "both"   : 先对 B 列做 QR，再对 A 行做 QR（通过对 A^T 做 QR 实现）
        "B_only" : 仅对 B 做 QR（吸收 R 到 A）
        "A_only" : 仅对 A 做 QR（吸收 R 到 B）
      orth_on:
        "B_first" 或 "A_first"：当 mode="both" 时，先正交化哪一侧（交替可减少互相“打架”）

    数学细节：
      - 对 B 做 thin-QR：  B = Q_B @ R_B，令 B <- Q_B，A <- R_B @ A   ⇒ ΔW 新 = Q_B (R_B A) = B A（不变）
      - 对 A 行做正交：对 A^T 做 thin-QR：A^T = Q_AT @ R_AT ⇒ A = R_AT^T @ Q_AT^T
        令 A <- Q_AT^T，B <- B @ R_AT^T   ⇒ ΔW 新 = (B R_AT^T) Q_AT^T = B A（不变）
    """
    # 保持 dtype / device 一致
    dev = B.device
    dtype = B.dtype

    def _qr_B():
        # B: [d_out, r] = Q_B [d_out, r] @ R_B [r, r]
        Q_B, R_B = torch.linalg.qr(B.to(torch.float32), mode="reduced")
        Q_B = Q_B.to(dtype).to(dev)
        R_B = R_B.to(dtype).to(dev)
        # 吸收 R_B 到 A 侧：A <- R_B @ A
        A.matmul_(A.new_zeros(R_B.shape).copy_(R_B))  # avoid aliasing: make inplace-friendly
        A.copy_(R_B @ A)
        # 替换 B <- Q_B
        B.copy_(Q_B)

    def _qr_A_rows():
        # 对 A^T 做 QR，使 A 的“行”正交：A^T = Q_AT @ R_AT
        # A: [r, d_in] -> A_T: [d_in, r] ; thin-QR 给 Q_AT: [d_in, r], R_AT: [r, r]
        A_T = A.t().to(torch.float32)  # [d_in, r]
        Q_AT, R_AT = torch.linalg.qr(A_T, mode="reduced")
        Q_AT = Q_AT.to(dtype).to(dev)
        R_AT = R_AT.to(dtype).to(dev)
        # A <- Q_AT^T
        A.copy_(Q_AT.t())
        # 吸收 R_AT^T 到 B：B <- B @ R_AT^T
        B.copy_(B @ R_AT.t())

    if mode == "B_only":
        _qr_B()
    elif mode == "A_only":
        _qr_A_rows()
    elif mode == "both":
        if orth_on == "B_first":
            _qr_B()
            _qr_A_rows()
        else:
            _qr_A_rows()
            _qr_B()
    else:
        raise ValueError(f"Unknown mode={mode}")


# ====== 自定义 Trainer：在 optimizer_step 之后做再牵引 ======
class MuonLoRATrainer(Trainer):
    def __init__(self,
                 *args,
                 retraction_every: int = 3,
                 retraction_mode: str = "both",
                 retraction_order: str = "B_first",
                 **kwargs):
        """
        参数：
          retraction_every : 每多少个优化步执行一次再牵引（例如 50~200）
          retraction_mode  : "both" / "B_only" / "A_only"
          retraction_order : 当 mode="both" 时先正交哪一侧
        """
        super().__init__(*args, **kwargs)
        self.retraction_every = max(1, int(retraction_every))
        self.retraction_mode = retraction_mode
        self.retraction_order = retraction_order

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int = 0,
        **kwargs,
    ):
        """
        先按常规定义执行优化器 step（更新参数），再根据步数周期执行 LoRA 再牵引。
        这样可以保证“几何约束”作用在“已更新的参数”上，不会被同一步的优化器覆盖。
        """
        # 1) 正常优化一步
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, **kwargs)

        # 2) 条件触发：每 retraction_every 步做一次
        # 注：global_step 在 optimizer_step 调用后已自增，这里用 self.state.global_step
        if (self.state.global_step % self.retraction_every) != 0:
            return

        # 3) 对所有 LoRA 因子做薄 QR 再牵引（严格保持 ΔW 不变）
        model = self.model
        was_training = model.training
        model.eval()  # 防止某些层在 train() 下有噪声/随机性；我们只做代数重排

        # AMP/FSDP/DP 下也安全：我们只对 .weight.data 做 no_grad 赋值
        with torch.no_grad():
            for (B, A) in iter_lora_factors(model):
                lora_qr_retraction(
                    B=B.data,
                    A=A.data,
                    mode=self.retraction_mode,
                    orth_on=self.retraction_order,
                )

        if was_training:
            model.train()