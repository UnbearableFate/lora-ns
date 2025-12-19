"""Distributed evaluation on MetaMathQA test sets.

Implements the same prompt format + answer extraction logic as the official
MetaMath repo (eval_gsm8k.py / eval_math.py), but runs generation with
Transformers + PEFT and shards work across processes with Accelerate.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LOCAL_TEST_FILES = {
    "gsm8k": _REPO_ROOT / "data" / "GSM8K_test.jsonl",
    "math": _REPO_ROOT / "data" / "MATH_test.jsonl",
}
_DEFAULT_TEST_URLS = {
    "gsm8k": "https://raw.githubusercontent.com/meta-math/MetaMath/main/data/test/GSM8K_test.jsonl",
    "math": "https://raw.githubusercontent.com/meta-math/MetaMath/main/data/test/MATH_test.jsonl",
}

_OFFICIAL_EVAL_PROMPT = (
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


@dataclass
class EvalSample:
    query: str
    reference: str
    meta: Dict


def _iter_jsonl(path: Path):
    import json

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _count_nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _shard_count(total: int, num_shards: int, shard_index: int) -> int:
    """Count of indices i in [0, total) with i % num_shards == shard_index."""
    if total <= 0:
        return 0
    if shard_index < 0 or shard_index >= num_shards:
        return 0
    if shard_index >= total:
        return 0
    return (total - shard_index + num_shards - 1) // num_shards


def _apply_stop_strings(text: str, stop_strings: List[str]) -> str:
    if not text:
        return ""
    cut = None
    for stop in stop_strings:
        idx = text.find(stop)
        if idx >= 0:
            cut = idx if cut is None else min(cut, idx)
    return text if cut is None else text[:cut]


def _format_eval_prompt(instruction: str) -> str:
    return _OFFICIAL_EVAL_PROMPT.format(instruction=instruction)


def _gsm8k_extract_answer_number_debug(completion: str) -> Tuple[Optional[int], Dict[str, Any]]:
    import re
    from fractions import Fraction

    def _is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata

            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    parts = (completion or "").split("The answer is: ")
    if len(parts) <= 1:
        return None, {"found_answer_prefix": False}

    extract_ans = parts[-1].strip()
    match = re.search(r"[\-+]?\d*[\.,/]?\d+", extract_ans)
    if not match:
        return None, {"found_answer_prefix": True, "answer_field": extract_ans, "matched_token": None}
    token = match.group()
    if "/" in token:
        numerator, denominator = token.split("/", 1)
        if not (_is_number(numerator) and _is_number(denominator)):
            return None, {
                "found_answer_prefix": True,
                "answer_field": extract_ans,
                "matched_token": token,
                "parse_error": "non_numeric_fraction",
            }
        if denominator == "0":
            pred_num = round(float(numerator.replace(",", "")))
            return pred_num, {"found_answer_prefix": True, "answer_field": extract_ans, "matched_token": token}
        frac = Fraction(token.replace(",", ""))
        pred_num = round(float(frac.numerator / frac.denominator))
        return pred_num, {"found_answer_prefix": True, "answer_field": extract_ans, "matched_token": token}

    val = float(token.replace(",", ""))
    if val == float("inf"):
        return None, {
            "found_answer_prefix": True,
            "answer_field": extract_ans,
            "matched_token": token,
            "parse_error": "inf",
        }
    pred_num = round(val)
    return pred_num, {"found_answer_prefix": True, "answer_field": extract_ans, "matched_token": token}


def _gsm8k_extract_answer_number(completion: str) -> Optional[int]:
    pred_num, _debug = _gsm8k_extract_answer_number_debug(completion)
    return pred_num


def _last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def _remove_boxed(s: str):
    left = "\\boxed{"
    try:
        if not s.startswith(left):
            return None
        if not s.endswith("}"):
            return None
        return s[len(left) : -1]
    except Exception:
        return None


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            string = splits[0]
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if string.startswith("."):
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    if "\\sqrt" in string:
        parts = string.split("\\sqrt")
        new_string = parts[0]
        for part in parts[1:]:
            if part and part[0] != "{":
                new_string += "\\sqrt{" + part[0] + "}" + part[1:]
            else:
                new_string += "\\sqrt" + part
        string = new_string
    string = string.replace(" ", "")
    if "\\frac" in string:
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr and substr[0] == "{":
                    new_str += substr
                else:
                    if len(substr) < 2:
                        return string
                    a, b = substr[0], substr[1]
                    if b != "{":
                        post = substr[2:] if len(substr) > 2 else ""
                        new_str += "{" + a + "}{" + b + "}" + post
                    else:
                        post = substr[2:] if len(substr) > 2 else ""
                        new_str += "{" + a + "}" + b + post
        string = new_str
    if string == "0.5":
        string = "\\frac{1}{2}"
    if len(string.split("/")) == 2:
        a, b = string.split("/")
        try:
            ai = int(a)
            bi = int(b)
            if string == f"{ai}/{bi}":
                string = f"\\frac{{{ai}}}{{{bi}}}"
        except Exception:
            pass
    return string


def _is_equiv(str1: str, str2: str) -> bool:
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:
        return str1 == str2


def _math_expected_answer(reference_solution: str) -> Optional[str]:
    boxed = _last_boxed_only_string(reference_solution or "")
    if boxed is None:
        return None
    return _remove_boxed(boxed)


def _math_extract_pred_answer_debug(completion: str) -> Tuple[Optional[str], Dict[str, Any]]:
    parts = (completion or "").split("The answer is: ")
    if len(parts) <= 1:
        return None, {"found_answer_prefix": False}
    ans = parts[-1]
    extract_ans_temp = ans.split(".\n")[0].strip()
    extract_ans = extract_ans_temp[:-1] if extract_ans_temp.endswith(".") else extract_ans_temp
    pred = extract_ans.strip() or None
    return pred, {
        "found_answer_prefix": True,
        "answer_field": ans,
        "extracted_temp": extract_ans_temp,
        "extracted": extract_ans,
    }


def _math_extract_pred_answer(completion: str) -> Optional[str]:
    pred, _debug = _math_extract_pred_answer_debug(completion)
    return pred


def _math_pred_matches(completion: str, expected_answer: Optional[str]) -> bool:
    if expected_answer is None:
        return False
    pred = _math_extract_pred_answer(completion)
    return pred is not None and _is_equiv(pred, expected_answer)


def _gsm8k_expected_answer(reference_solution: str) -> Optional[int]:
    if not reference_solution:
        return None
    if "####" not in reference_solution:
        return None
    try:
        ans = reference_solution.split("####", 1)[1].strip()
        return int(ans.replace(",", ""))
    except Exception:
        return None


def _prepare_model(model_name: str, adapter_path: Optional[str], device: torch.device, torch_dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    return model


def _prepare_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def run_post_training_evaluation(
    accelerator: Accelerator,
    config: Dict,
    output_dir: str,
    adapter_path: Optional[str] = None,
    splits: Optional[List[str]] = None,
    batch_size: Optional[int] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    dump_debug_jsonl: bool = False,
):
    """Run evaluation on MetaMathQA GSM8K and MATH test sets.

    Args:
        accelerator: accelerate.Accelerator used for DDP.
        config: training config dict; must contain model path.
        output_dir: where to store predictions/metrics.
        adapter_path: path to LoRA adapter (usually training_args.output_dir).
        splits: which test splits to run (subset of {"gsm8k", "math"}); defaults to both.
        batch_size: per-process batch size; defaults to config["generation"]["batch_size"] or 1.
        max_new_tokens: generation length.
        temperature/top_p: sampling params.
        dump_debug_jsonl: write per-split debug.jsonl with extraction details.
    """

    generation_cfg = config.get("generation", {})
    if batch_size is None:
        batch_size = int(generation_cfg.get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    model_name = config["model"]["name_or_path"]
    torch_dtype = torch.bfloat16 if config.get("training", {}).get("bf16") else torch.float16

    tokenizer = _prepare_tokenizer(model_name)
    model = _prepare_model(model_name, adapter_path, accelerator.device, torch_dtype)

    # Ensure all processes have model ready
    accelerator.wait_for_everyone()

    results = {}
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    selected_splits = splits or list(_DEFAULT_TEST_URLS.keys())
    unknown = sorted(set(selected_splits) - set(_DEFAULT_TEST_URLS.keys()))
    if unknown:
        raise ValueError(f"Unknown split(s) {unknown}; supported: {sorted(_DEFAULT_TEST_URLS.keys())}")

    for split_key in selected_splits:
        local_preds: List[str] = []
        local_correct: List[bool] = []
        local_records: List[Dict[str, Any]] = []
        local_debug_records: List[Dict[str, Any]] = []

        local_file = _DEFAULT_LOCAL_TEST_FILES.get(split_key)
        data_source = local_file if local_file and local_file.exists() else None
        if data_source is None:
            url = _DEFAULT_TEST_URLS[split_key]
            raise RuntimeError(
                f"Missing local test file for split '{split_key}' ({local_file}); "
                f"download it from {url} or place it under data/."
            )

        data_path = Path(data_source)
        iterator = _iter_jsonl(data_path)
        num_total = _count_nonempty_lines(data_path)
        num_local = _shard_count(num_total, accelerator.num_processes, accelerator.process_index)
        pbar = tqdm(
            total=num_local,
            disable=not accelerator.is_local_main_process,
            desc=f"eval[{split_key}] shard {accelerator.process_index}/{accelerator.num_processes}",
            unit="ex",
        )

        pending: List[Dict[str, Any]] = []

        def _flush_pending():
            if not pending:
                return

            prompts = [item["prompt"] for item in pending]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(accelerator.device)
            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                generated = accelerator.unwrap_model(model).generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for batch_idx, item in enumerate(pending):
                # Take generated portion
                gen_text_raw = tokenizer.decode(
                    generated[batch_idx][input_len:],
                    skip_special_tokens=True,
                )
                gen_text = _apply_stop_strings(gen_text_raw, _OFFICIAL_STOP_STRINGS)

                local_preds.append(gen_text)

                expected = item["expected"]
                reference = item["reference"]
                if split_key == "gsm8k":
                    pred_num = None
                    pred_debug: Optional[Dict[str, Any]] = None
                    if dump_debug_jsonl:
                        pred_num, pred_debug = _gsm8k_extract_answer_number_debug(gen_text)
                    else:
                        pred_num = _gsm8k_extract_answer_number(gen_text)
                    correct = expected is not None and pred_num is not None and float(pred_num) == float(expected)
                else:
                    pred_answer = None
                    pred_answer_debug: Optional[Dict[str, Any]] = None
                    if dump_debug_jsonl:
                        pred_answer, pred_answer_debug = _math_extract_pred_answer_debug(gen_text)
                        correct = (
                            expected is not None
                            and pred_answer is not None
                            and _is_equiv(pred_answer, expected)
                        )
                    else:
                        correct = _math_pred_matches(gen_text, expected)

                local_correct.append(bool(correct))
                local_records.append(
                    {
                        "idx": item["idx"],
                        "split": split_key,
                        "instruction": item["instruction"],
                        "reference": reference,
                        "prediction": gen_text,
                        "correct": bool(correct),
                        "expected_answer": expected,
                    }
                )

                if dump_debug_jsonl:
                    if split_key == "gsm8k":
                        expected_raw = None
                        if reference and "####" in reference:
                            expected_raw = reference.split("####", 1)[1].strip()
                        judge = {
                            "task": "gsm8k",
                            "expected_raw": expected_raw,
                            "expected_number": expected,
                            "pred_number": pred_num,
                            "extraction": pred_debug,
                        }
                    else:
                        pred_debug = pred_answer_debug or {}
                        expected_boxed = _last_boxed_only_string(reference or "")
                        expected_norm = _strip_string(expected) if expected is not None else None
                        pred_norm = _strip_string(pred_answer) if pred_answer is not None else None
                        judge = {
                            "task": "math",
                            "expected_boxed": expected_boxed,
                            "expected_answer": expected,
                            "expected_normalized": expected_norm,
                            "pred_answer": pred_answer,
                            "pred_normalized": pred_norm,
                            "extraction": pred_debug,
                        }
                    local_debug_records.append(
                        {
                            "idx": item["idx"],
                            "split": split_key,
                            "prompt": item["prompt"],
                            "instruction": item["instruction"],
                            "reference": reference,
                            "prediction_raw": gen_text_raw,
                            "prediction": gen_text,
                            "judge": judge,
                            "correct": bool(correct),
                        }
                    )

            pending.clear()

        for idx, example in enumerate(iterator):
            if (idx % accelerator.num_processes) != accelerator.process_index:
                continue

            if split_key == "gsm8k":
                instruction = example.get("query") or ""
                reference = example.get("response") or ""
                expected = _gsm8k_expected_answer(reference)
            else:
                instruction = example.get("instruction") or ""
                reference = example.get("output") or ""
                expected = _math_expected_answer(reference)

            prompt = _format_eval_prompt(instruction)
            pending.append(
                {
                    "idx": idx,
                    "instruction": instruction,
                    "reference": reference,
                    "expected": expected,
                    "prompt": prompt,
                }
            )
            pbar.update(1)
            if len(pending) >= batch_size:
                _flush_pending()

        _flush_pending()
        pbar.close()

        # Gather across processes
        all_preds = accelerator.gather_object(local_preds)
        all_correct = accelerator.gather_object(local_correct)
        all_records = accelerator.gather_object(local_records)
        all_debug_records = accelerator.gather_object(local_debug_records) if dump_debug_jsonl else None

        if accelerator.is_main_process:
            flat_preds = [p for sub in all_preds for p in sub]
            flat_correct = [c for sub in all_correct for c in sub]
            flat_records = [rec for sub in all_records for rec in sub]
            flat_debug_records = (
                [rec for sub in all_debug_records for rec in sub] if all_debug_records is not None else None
            )
            flat_records.sort(key=lambda r: r.get("idx", -1))
            if flat_debug_records is not None:
                flat_debug_records.sort(key=lambda r: r.get("idx", -1))

            accuracy = (sum(flat_correct) / len(flat_correct)) if flat_correct else 0.0
            split_dir = Path(output_dir) / split_key
            split_dir.mkdir(parents=True, exist_ok=True)

            with open(split_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
                for rec in flat_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if dump_debug_jsonl and flat_debug_records is not None:
                with open(split_dir / "debug.jsonl", "w", encoding="utf-8") as f:
                    for rec in flat_debug_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            metrics = {"accuracy": accuracy, "total": len(flat_preds)}
            with open(split_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            logger.info("[%s] accuracy=%.4f (%d examples)", split_key, accuracy, len(flat_preds))
            results[split_key] = metrics

    accelerator.wait_for_everyone()
    return results
