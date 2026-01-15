import argparse
import math
from pathlib import Path

import torch

from trainer.cleaned_svd_ref_trainer import (
    get_warmup_restart_then_final_decay_scheduler_ratio,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate LR schedule and plot the curve."
    )
    parser.add_argument("--num-training-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--repeat-n", type=int, default=2)
    parser.add_argument("--repeat-warmup-ratio", type=float, default=0.03)
    parser.add_argument("--repeat-decay-ratio", type=float, default=0.03)
    parser.add_argument("--repeat-end-lr-rate", type=float, default=0.99)
    parser.add_argument("--final-warmup-ratio", type=float, default=0.03)
    parser.add_argument("--min-lr-rate", type=float, default=1e-4)
    parser.add_argument("--repeat-decay-type", type=str, default="cosine")
    parser.add_argument("--final-decay-type", type=str, default="cosine")
    parser.add_argument("--warmup-start-lr-rate", type=float, default=0.1)
    parser.add_argument("--first-warmup-start-lr-rate", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="lr_curve.png")
    return parser.parse_args()


def main():
    args = parse_args()

    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([param], lr=args.learning_rate)

    scheduler = get_warmup_restart_then_final_decay_scheduler_ratio(
        optimizer=optimizer,
        num_training_steps=args.num_training_steps,
        repeat_n=args.repeat_n,
        repeat_warmup_ratio=args.repeat_warmup_ratio,
        repeat_decay_ratio=args.repeat_decay_ratio,
        repeat_end_lr_rate=args.repeat_end_lr_rate,
        final_warmup_ratio=args.final_warmup_ratio,
        min_lr_rate=args.min_lr_rate,
        repeat_decay_type=args.repeat_decay_type,
        final_decay_type=args.final_decay_type,
        warmup_start_lr_rate=args.warmup_start_lr_rate,
        first_warmup_start_lr_rate=args.first_warmup_start_lr_rate,
        last_epoch=-1,
    )

    lrs = []
    for _ in range(args.num_training_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    out_path = Path(args.output)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(range(1, args.num_training_steps + 1), lrs, linewidth=1.5)
        plt.title("LR Schedule")
        plt.xlabel("Step")
        plt.ylabel("Learning rate")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"Saved plot to {out_path.resolve()}")
    except Exception as exc:
        txt_path = out_path.with_suffix(".txt")
        with open(txt_path, "w") as f:
            for i, lr in enumerate(lrs, 1):
                f.write(f"{i}\t{lr}\n")
        print(
            f"matplotlib unavailable ({exc}); saved raw LR values to {txt_path.resolve()}"
        )


if __name__ == "__main__":
    main()
