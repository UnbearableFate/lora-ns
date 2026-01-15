"""
Evaluate a LoRA adapter on GLUE-style text classification tasks and log results to CSV.
"""

import argparse
import csv
import json
import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

import evaluate
import numpy as np
from peft import PeftModel, PeftConfig
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from utils import load_config, prepare_dataset, validate_config, get_glue_metrics_function
from utils.model_utils import load_base_model, load_tokenizer
from src.utils import append_row_to_csv, get_info_from_model_path

logger = logging.getLogger(__name__)


def _metrics_fallback() -> callable:
    metric = evaluate.load("accuracy")

    def _compute(eval_preds):
        logits, labels = eval_preds
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if hasattr(logits, "shape") and len(getattr(logits, "shape", [])) == 3:
            logits = logits[:, -1, :]
        if hasattr(logits, "argmax"):
            predictions = logits.argmax(axis=-1)
        else:
            predictions = [int(row.index(max(row))) for row in logits]
        return metric.compute(predictions=predictions, references=labels)

    return _compute


def _resolve_split(config: Dict, split_override: Optional[str], dataset_keys: Iterable[str]) -> str:
    if split_override:
        return split_override
    dataset_cfg = config.get("dataset", {})
    if dataset_cfg.get("eval_split"):
        return dataset_cfg["eval_split"]
    if "validation" in dataset_keys:
        return "validation"
    if dataset_cfg.get("test_split"):
        return dataset_cfg["test_split"]
    return "test"


def _labels_are_valid(eval_dataset, num_labels: Optional[int]) -> Tuple[bool, Optional[int], Optional[int]]:
    if num_labels is None or num_labels <= 0:
        return True, None, None
    if "labels" not in eval_dataset.column_names:
        return False, None, None
    labels = list(eval_dataset["labels"])
    if not labels:
        return False, None, None
    min_label = min(labels)
    max_label = max(labels)
    return (min_label >= 0 and max_label < num_labels), min_label, max_label


def _save_glue_submission(predictions: List[int], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t")
        writer.writerow(["index", "prediction"])
        for idx, pred in enumerate(predictions):
            writer.writerow([idx, pred])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapters on GLUE tasks")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--split", type=str, default=None, help="Dataset split override")
    parser.add_argument("--output_csv", type=str, default=None, help="CSV path for results")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for evaluation samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    config = load_config(args.config)
    validate_config(config)

    tokenizer = load_tokenizer(config["model"]["name_or_path"], config)
    base_model = load_base_model(config["model"]["name_or_path"], config)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(base_model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=False)
    model.eval()

    dataset = prepare_dataset(config, tokenizer)
    split = _resolve_split(config, args.split, dataset.keys())
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found. Available: {list(dataset.keys())}")
    eval_dataset = dataset[split]
    if args.max_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_samples, len(eval_dataset))))

    task_name = config.get("dataset", {}).get("subset", "")
    compute_metrics = get_glue_metrics_function(task_name) or _metrics_fallback()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_args = TrainingArguments(
        output_dir=os.path.join(args.adapter_path, "eval_outputs"),
        per_device_eval_batch_size=config.get("training", {}).get("per_device_eval_batch_size", 32),
        report_to=[],
        do_train=False,
        do_eval=True,
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    num_labels = config.get("dataset", {}).get("num_labels") or getattr(base_model.config, "num_labels", None)
    labels_valid, min_label, max_label = _labels_are_valid(eval_dataset, num_labels)
    if labels_valid and split != "test":
        metrics = trainer.evaluate()
        logger.info("Evaluation metrics: %s", metrics)
    else:
        logger.warning(
            "Prediction-only run (split=%s, num_labels=%s, min=%s, max=%s).",
            split,
            num_labels,
            min_label,
            max_label,
        )
        if "labels" in eval_dataset.column_names:
            eval_dataset = eval_dataset.remove_columns("labels")
            trainer.eval_dataset = eval_dataset
        preds_output = trainer.predict(eval_dataset)
        logits = preds_output.predictions
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        logits = np.asarray(logits)
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        predictions = logits.argmax(axis=-1).tolist()
        metrics = {"eval_prediction_only": True}
        if split == "test" and config.get("dataset", {}).get("name") == "glue":
            output_dir = os.path.join(args.adapter_path, "submissions")
            subset = config.get("dataset", {}).get("subset", "glue")
            submit_path = os.path.join(output_dir, f"glue_{subset}_test.tsv")
            _save_glue_submission(predictions, submit_path)
            metrics["submission_file"] = submit_path

    output_csv = args.output_csv or os.path.join(args.adapter_path, "eval_results.csv")
    run_info = get_info_from_model_path(args.adapter_path)
    peft_config = None
    try:
        peft_config = PeftConfig.from_pretrained(args.adapter_path)
    except Exception:
        peft_config = None

    base_model_name = config["model"]["name_or_path"]
    if peft_config is not None and getattr(peft_config, "base_model_name_or_path", None):
        base_model_name = peft_config.base_model_name_or_path

    row = {
        "timestamp": run_info.get("timestamp"),
        "base_model": base_model_name.split("/")[-1],
        "dataset_name": config.get("dataset", {}).get("name", ""),
        "subset": config.get("dataset", {}).get("subset", ""),
        "split": split,
        "init_lora_weights": getattr(peft_config, "init_lora_weights", None),
        "extra": run_info.get("extra"),
        "seed": run_info.get("seed") or config.get("training", {}).get("seed"),
        "adapter_path": args.adapter_path,
        "model_path": os.path.basename(args.adapter_path.rstrip("/")),
    }
    if str(row["init_lora_weights"]).lower() == "true":
        row["init_lora_weights"] = "kaiming"

    for key, value in metrics.items():
        row[f"metric_{key}"] = value

    if peft_config is not None:
        try:
            peft_dict = peft_config.to_dict()
        except Exception:
            peft_dict = {}
        for key in (
            "r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "bias",
            "use_dora",
            "use_rslora",
        ):
            if key in peft_dict:
                value = peft_dict[key]
                row[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value

    append_row_to_csv(output_csv, row)
    logger.info("Wrote CSV results to %s", output_csv)


if __name__ == "__main__":
    main()
