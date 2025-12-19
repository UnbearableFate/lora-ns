"""
Dataset loading / preprocessing entrypoints.

Design:
- Keep the *minimum* shared machinery in a small base class (HF load + split parsing).
- Implement dataset-specific preprocessing in dedicated subclasses (no task_type if/else).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


PROMPT_TEMPLATES: Dict[str, str] = {
    # ---- AnsAug (forward solving) ----
    "GSM_AnsAug": (
        "You are a careful math solver.\n"
        "Solve the following word problem step by step.\n"
        "Write the final numeric answer on the last line in the format:\n"
        "#### <answer>\n\n"
        "Problem:\n{query}\n"
    ),
    "MATH_AnsAug": (
        "You are a careful math solver.\n"
        "Solve the following math problem step by step.\n"
        "If the answer should be in terms of symbols (e.g., pi), keep exact form.\n"
        "Write the final answer on the last line.\n\n"
        "Problem:\n{query}\n"
    ),
    # ---- Rephrased (equivalent phrasing) ----
    "GSM_Rephrased": (
        "You are a careful math solver.\n"
        "The problem may be rephrased but is mathematically equivalent to a standard question.\n"
        "Solve step by step and output the final answer as:\n"
        "#### <answer>\n\n"
        "Problem:\n{query}\n"
    ),
    "MATH_Rephrased": (
        "You are a careful math solver.\n"
        "The problem may be rephrased but is mathematically equivalent to a standard question.\n"
        "Solve step by step and give the final answer on the last line.\n\n"
        "Problem:\n{query}\n"
    ),
    # ---- SV (solve for variable) ----
    "GSM_SV": (
        "You are a careful math solver.\n"
        "Solve for the unknown variable specified in the problem.\n"
        "Show algebraic steps briefly and output the value as:\n"
        "#### <answer>\n\n"
        "Problem:\n{query}\n"
    ),
    "MATH_SV": (
        "You are a careful math solver.\n"
        "Solve for the unknown variable specified in the problem.\n"
        "Show algebraic steps briefly and write the final value on the last line.\n\n"
        "Problem:\n{query}\n"
    ),
    # ---- FOBAR (backward inference) ----
    "GSM_FOBAR": (
        "You are a careful math solver.\n"
        "This is a backward reasoning problem: the final answer to an original version is given.\n"
        "Use that given answer as a constraint to infer the unknown variable.\n"
        "Show key steps and output the inferred value as:\n"
        "#### <answer>\n\n"
        "Problem:\n{query}\n"
    ),
    "MATH_FOBAR": (
        "You are a careful math solver.\n"
        "This is a backward reasoning problem: the final answer to an original version is given.\n"
        "Use that given answer as a constraint to infer the unknown variable.\n"
        "Show key steps and write the inferred value on the last line.\n\n"
        "Problem:\n{query}\n"
    ),
}


def build_metamathqa_prompt(data_point: Dict[str, Any]) -> str:
    """
    Build a type-aware prompt for a MetaMathQA datapoint.

    Expected keys in data_point:
      - "type": str
      - "query": str
    Optional:
      - "original_question", "response"
    """
    dp_type = str(data_point.get("type", "") or "")
    query = str(data_point.get("query", "") or "")

    template = PROMPT_TEMPLATES.get(
        dp_type,
        (
            "You are a careful math solver.\n"
            "Solve the problem step by step and provide the final answer on the last line.\n\n"
            "Problem:\n{query}\n"
        ),
    )
    return template.format(query=query)

def build_classical_metamathqa_prompt(data_point: Dict[str, Any]) -> str:
    query = str(data_point.get("query", "") or "")
    template = """
        Below is an instruction that describes a task, paired with an input that provides further context. 
        Write a response that appropriately completes the request.\n\n
        ### Instruction:\n{query}\n\n### Response:
    """
    return template.format(query=query)

class BaseDatasetProcessor(ABC):
    """Shared HF dataset loading + split helpers. Subclasses implement preprocess()."""

    def __init__(self, config: Dict):
        self.config = config
        self.dataset_cfg = config.get("dataset", {}) or {}

    def load(self) -> DatasetDict:
        dataset_name = self.dataset_cfg.get("name")
        subset = self.dataset_cfg.get("subset")
        if not dataset_name:
            raise ValueError("dataset.name is required")

        load_args = [dataset_name]
        if subset:
            load_args.append(subset)

        load_kwargs = {
            "data_files": self.dataset_cfg.get("data_files"),
            "data_dir": self.dataset_cfg.get("data_dir"),
            "cache_dir": self.dataset_cfg.get("cache_dir"),
            "streaming": self.dataset_cfg.get("streaming", False),
            "keep_in_memory": self.dataset_cfg.get("keep_in_memory", False),
            "verification_mode": self.dataset_cfg.get("verification_mode"),
        }
        load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}

        logger.info("Loading dataset: %s", dataset_name)
        raw = load_dataset(*load_args, **load_kwargs)

        train_split = self.dataset_cfg.get("train_split", "train")
        eval_split = self.dataset_cfg.get("eval_split")
        test_split = self.dataset_cfg.get("test_split")

        dataset_dict: Dict[str, Dataset] = {"train": self._get_split(raw, train_split)}
        if eval_split:
            dataset_dict["validation"] = self._get_split(raw, eval_split)
        if test_split:
            dataset_dict["test"] = self._get_split(raw, test_split)

        return DatasetDict(dataset_dict)

    def _get_split(self, dataset: Union[Dataset, DatasetDict], split_expr: str) -> Dataset:
        """Supports split expressions like 'train[:1000]'."""
        if split_expr is None:
            raise ValueError("split expression must be provided")

        if "[" not in split_expr:
            return dataset[split_expr] if isinstance(dataset, DatasetDict) else dataset

        split_name = split_expr.split("[", 1)[0]
        split_range = split_expr.split("[", 1)[1].rstrip("]")

        if ":" in split_range:
            left, right = split_range.split(":", 1)
            start = int(left) if left else 0
            source = dataset[split_name] if isinstance(dataset, DatasetDict) else dataset
            end = int(right) if right else len(source)
            return source.select(range(start, end))

        idx = int(split_range)
        source = dataset[split_name] if isinstance(dataset, DatasetDict) else dataset
        return source.select([idx])

    @abstractmethod
    def preprocess(self, dataset: DatasetDict, tokenizer) -> DatasetDict:
        raise NotImplementedError


class MetaMathQADatasetProcessor(BaseDatasetProcessor):
    """MetaMathQA SFT preprocessing using raw columns: type/query/original_question/response.

    This intentionally *does not* convert `query` into `instruction`/`input`.
    Instead it selects a prompt template based on the example `type` (see PROMPT_TEMPLATES).
    """

    def preprocess(self, dataset: DatasetDict, tokenizer) -> DatasetDict:
        cfg = self.dataset_cfg
        num_workers = cfg.get("preprocessing_num_workers", 8)
        max_length = cfg.get("max_length") or tokenizer.model_max_length

        append_eos = cfg.get("append_eos_token", True)
        add_bos = cfg.get("add_bos_token", False)
        truncate_prompt_only = cfg.get("truncate_prompt_only", True)
        truncate_from_end = cfg.get("truncate_from_end", True)
        strip_whitespace = cfg.get("strip_prompt", True)

        query_field = "query"
        original_question_field = "original_question"
        type_field = "type"
        target_field = cfg.get("target_field", "response") or "response"

        def _render_prompt(example_type: str, query: str, original_question: str, response: str) -> str:
            rendered = build_metamathqa_prompt(
                {
                    "type": example_type,
                    "query": query,
                    "original_question": original_question,
                    "response": response,
                }
            )
            return rendered.strip() if strip_whitespace else rendered

        def _render_target(target: str) -> str:
            return (target or "").strip() if strip_whitespace else (target or "")

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id

        def _encode_pair(prompt: str, target: str) -> Tuple[List[int], List[int]]:
            rendered_prompt = prompt
            if add_bos and bos_id is not None and tokenizer.bos_token:
                rendered_prompt = tokenizer.bos_token + rendered_prompt
            prompt_ids = tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]

            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            if append_eos and eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
                target_ids = target_ids + [eos_id]
            return prompt_ids, target_ids

        def tokenize_batch(examples: Dict[str, List]) -> Dict[str, List[List[int]]]:
            input_ids_list: List[List[int]] = []
            labels_list: List[List[int]] = []
            attention_list: List[List[int]] = []

            types = examples.get(type_field, [])
            queries = examples.get(query_field, [])
            originals = examples.get(original_question_field, [])
            targets = examples.get(target_field, [])

            for example_type, query, original_question, target_text in zip(types, queries, originals, targets):
                prompt = _render_prompt(
                    example_type=str(example_type or ""),
                    query=str(query or ""),
                    original_question=str(original_question or ""),
                    response=str(target_text or ""),
                )
                target = _render_target(str(target_text or ""))

                prompt_ids, target_ids = _encode_pair(prompt, target)

                total_len = len(prompt_ids) + len(target_ids)
                if max_length and total_len > max_length:
                    overflow = total_len - max_length
                    if truncate_prompt_only:
                        trim = min(len(prompt_ids), overflow)
                        prompt_ids = prompt_ids[trim:]
                        overflow -= trim
                    if overflow > 0:
                        if truncate_from_end:
                            target_ids = target_ids[:-overflow] if overflow < len(target_ids) else []
                        else:
                            target_ids = target_ids[overflow:]

                if not target_ids:
                    continue

                input_ids = prompt_ids + target_ids
                labels = [-100] * len(prompt_ids) + target_ids
                attention = [1] * len(input_ids)

                input_ids_list.append(input_ids)
                labels_list.append(labels)
                attention_list.append(attention)

            return {
                "input_ids": input_ids_list,
                "labels": labels_list,
                "attention_mask": attention_list,
            }

        original_columns = dataset["train"].column_names
        tokenized = dataset.map(
            tokenize_batch,
            batched=True,
            num_proc=num_workers,
            remove_columns=original_columns,
            desc="MetaMathQA: tokenize",
        )
        return tokenized


class GlueDatasetProcessor(BaseDatasetProcessor):
    """GLUE-style text classification preprocessing (kept for backward compatibility)."""

    def preprocess(self, dataset: DatasetDict, tokenizer) -> DatasetDict:
        cfg = self.dataset_cfg
        text_column = cfg.get("text_column", [])
        label_column = cfg.get("label_column", "label")
        max_length = cfg.get("max_length", 512)
        num_workers = cfg.get("preprocessing_num_workers", 4)

        def tokenize_function(examples):
            if isinstance(text_column, list) and len(text_column) == 2:
                result = tokenizer(
                    examples[text_column[0]],
                    examples[text_column[1]],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                )
            else:
                col = text_column[0] if isinstance(text_column, list) else text_column
                result = tokenizer(
                    examples[col],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                )

            if label_column in examples and label_column != "labels":
                result["labels"] = examples[label_column]
            elif "labels" in examples:
                result["labels"] = examples["labels"]
            return result

        return dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_workers,
            desc="GLUE: tokenize",
        )


_PROCESSOR_REGISTRY: Dict[str, Type[BaseDatasetProcessor]] = {
    "meta-math/MetaMathQA": MetaMathQADatasetProcessor,
    "glue": GlueDatasetProcessor,
}


def _get_processor(config: Dict) -> BaseDatasetProcessor:
    dataset_name = (config.get("dataset", {}) or {}).get("name")
    processor_cls = _PROCESSOR_REGISTRY.get(dataset_name)
    if processor_cls is None:
        supported = ", ".join(sorted(_PROCESSOR_REGISTRY.keys()))
        raise ValueError(f"Unsupported dataset.name={dataset_name!r}. Supported: {supported}")
    return processor_cls(config)


class DatasetLoader:
    """Compatibility wrapper (older code imports DatasetLoader)."""

    def __init__(self, config: Dict):
        self.config = config
        self._processor = _get_processor(config)

    def load(self) -> DatasetDict:
        return self._processor.load()

    def preprocess(self, dataset: DatasetDict, tokenizer) -> DatasetDict:
        return self._processor.preprocess(dataset, tokenizer)


def prepare_dataset(config: Dict, tokenizer) -> DatasetDict:
    processor = _get_processor(config)
    dataset = processor.load()
    return processor.preprocess(dataset, tokenizer)


def prepare_glue_dataset(config: Dict, tokenizer) -> DatasetDict:
    return prepare_dataset(config, tokenizer)


def prepare_math_dataset(config: Dict, tokenizer) -> DatasetDict:
    return prepare_dataset(config, tokenizer)


def prepare_code_dataset(config: Dict, tokenizer) -> DatasetDict:
    return prepare_dataset(config, tokenizer)
