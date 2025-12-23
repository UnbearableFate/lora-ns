"""Standalone CLI for evaluating a base model + LoRA adapter on MetaMathQA tests.

Example:
  accelerate launch run_meta_math_eval.py \
    --config configs/meta_math_qa/qwen.yaml \
    --adapter_path outputs/my_run \
    --splits gsm8k,math
"""

import argparse
import os
from typing import List, Optional

from accelerate import Accelerator

from utils import load_config, validate_config
from lora_evaluate import run_post_training_evaluation


def _parse_splits(value: str) -> List[str]:
    splits = [s.strip() for s in value.split(",") if s.strip()]
    if not splits:
        raise argparse.ArgumentTypeError("splits must be a comma-separated list, e.g. gsm8k,math")
    return splits


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapter on GSM8K/MATH (MetaMathQA tests)")
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML config")
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to saved LoRA adapter dir (e.g., training output_dir)",
    )
    parser.add_argument(
        "--splits",
        type=_parse_splits,
        default=["gsm8k", "math"],
        help="Comma-separated splits to run: gsm8k,math (default: gsm8k,math)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override generation max_new_tokens")
    parser.add_argument("--temperature", type=float, default=None, help="Override generation temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Override generation top_p")
    parser.add_argument(
        "--dump_debug_jsonl",
        action="store_true",
        help="Also write per-split debug.jsonl with answer extraction details (for debugging correctness).",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    accelerator = Accelerator()

    config = load_config(args.config)
    validate_config(config)

    generation_cfg = config.get("generation", {})
    output_dir = generation_cfg.get("output_dir", "results")

    run_post_training_evaluation(
        accelerator=accelerator,
        config=config,
        output_dir=output_dir,
        adapter_path=args.adapter_path,
        splits=args.splits,
        max_new_tokens=generation_cfg.get("max_new_tokens", 512),
        temperature=generation_cfg.get("temperature", 0.2),
        top_p=generation_cfg.get("top_p", 0.7),
        dump_debug_jsonl=args.dump_debug_jsonl,
    )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
