"""Simple inference CLI for a base model + optional LoRA adapter.

For MetaMath-style math prompting, this defaults to the official MetaMath eval prompt:
  "### Response: Let's think step by step."
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_config, validate_config


_OFFICIAL_METHAMATH_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
)

_OFFICIAL_STOP_STRINGS = [
    "Question:",
    "Question",
    "USER:",
    "USER",
    "ASSISTANT:",
    "ASSISTANT",
    "Instruction:",
    "Instruction",
    "Response:",
    "Response",
]


def _apply_stop_strings(text: str, stop_strings: List[str]) -> str:
    if not text:
        return ""
    cut = None
    for stop in stop_strings:
        idx = text.find(stop)
        if idx >= 0:
            cut = idx if cut is None else min(cut, idx)
    return text if cut is None else text[:cut]


def _format_prompt(instruction: str, config: Dict) -> str:
    dataset_cfg = config.get("dataset", {}) or {}
    prompt_style = (dataset_cfg.get("prompt_style") or "").lower().strip()
    template = dataset_cfg.get("infer_prompt_template") or dataset_cfg.get("eval_prompt_template")

    if template:
        return template.format(instruction=instruction, query=instruction)

    if prompt_style == "metamath" or dataset_cfg.get("name") == "meta-math/MetaMathQA":
        return _OFFICIAL_METHAMATH_PROMPT.format(instruction=instruction)

    prompt_template = dataset_cfg.get("prompt_template")
    if prompt_template:
        return prompt_template.format(instruction=instruction, query=instruction)

    return instruction


def _load_model_and_tokenizer(config: Dict, adapter_path: Optional[str], device: torch.device):
    model_name = config["model"]["name_or_path"]
    torch_dtype = torch.bfloat16 if config.get("training", {}).get("bf16") else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config.get("model", {}).get("trust_remote_code", True),
        token=config.get("model", {}).get("token", False),
        padding_side="left",
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=config.get("model", {}).get("trust_remote_code", True),
        token=config.get("model", {}).get("token", False),
        torch_dtype=torch_dtype,
        device_map=None,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def _iter_inputs(input_text: Optional[str], input_file: Optional[str]) -> Iterable[str]:
    if input_text is not None:
        yield input_text
        return
    if input_file is None:
        raise ValueError("One of --input_text or --input_file must be provided.")
    with Path(input_file).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Inference for base model + LoRA adapter")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (Trainer output_dir). If omitted, runs the base model.",
    )
    parser.add_argument("--input_text", type=str, default=None, help="Single input instruction")
    parser.add_argument("--input_file", type=str, default=None, help="Text file with 1 instruction per line")
    parser.add_argument("--output_file", type=str, default=None, help="Write JSONL outputs to this path")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_model_and_tokenizer(config, args.adapter_path, device)

    outputs = []
    for instruction in _iter_inputs(args.input_text, args.input_file):
        prompt = _format_prompt(instruction, config)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        text = _apply_stop_strings(text, _OFFICIAL_STOP_STRINGS)
        record = {"instruction": instruction, "prediction": text}
        outputs.append(record)

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for record in outputs:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        for record in outputs:
            print(record["prediction"])


if __name__ == "__main__":
    main()

