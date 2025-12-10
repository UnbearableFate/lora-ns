"""
Dataset loading utilities for various tasks.
Supports GLUE, MetaMathQA, GSM8K, Code-Feedback, and custom datasets.
"""

from typing import Dict, List, Optional, Tuple, Union
from datasets import load_dataset, Dataset, DatasetDict
import logging


class _SafeFormatDict(dict):
    """Default to empty string when formatting templates with missing keys."""

    def __missing__(self, key):  # pragma: no cover - tiny helper
        return ""

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and preprocess datasets based on task type."""
    
    def __init__(self, config: Dict):
        self.config = config
        peft_cfg = config.get("peft", {}) or {}
        self.task_type = peft_cfg.get("task_type", "CAUSAL_LM")
        self.dataset_config = config.get("dataset", {})
        self.tokenizer_config = config.get("tokenizer", {})
        
    def load(self) -> DatasetDict:
        """Load dataset based on configuration."""
        dataset_name = self.dataset_config.get("name")
        subset = self.dataset_config.get("subset")
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        load_args = [dataset_name]
        if subset:
            load_args.append(subset)

        load_kwargs = {
            "data_files": self.dataset_config.get("data_files"),
            "data_dir": self.dataset_config.get("data_dir"),
            "cache_dir": self.dataset_config.get("cache_dir"),
            "streaming": self.dataset_config.get("streaming", False),
            "keep_in_memory": self.dataset_config.get("keep_in_memory", False),
            "verification_mode": self.dataset_config.get("verification_mode"),
        }
        load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}

        dataset = load_dataset(*load_args, **load_kwargs)
        
        # Get splits
        train_split = self.dataset_config.get("train_split", "train")
        eval_split = self.dataset_config.get("eval_split")
        test_split = self.dataset_config.get("test_split")
        
        # Handle split expressions like "train[:5000]"
        train_data = self._get_split(dataset, train_split)
        eval_data = self._get_split(dataset, eval_split) if eval_split else None
        
        # Create dataset dict
        dataset_dict = {"train": train_data}
        if eval_data:
            dataset_dict["validation"] = eval_data
        if test_split:
            dataset_dict["test"] = self._get_split(dataset, test_split)
        
        return DatasetDict(dataset_dict)
    
    def _get_split(self, dataset: Union[Dataset, DatasetDict], split_expr: str) -> Dataset:
        """Get dataset split, handling expressions like 'train[:1000]'."""
        if split_expr is None:
            raise ValueError("split expression must be provided")
        if "[" in split_expr:
            # Parse split expression
            split_name = split_expr.split("[")[0]
            split_range = split_expr.split("[")[1].rstrip("]")
            
            if ":" in split_range:
                # Handle slice notation
                parts = split_range.split(":")
                start = int(parts[0]) if parts[0] else None
                end = int(parts[1]) if parts[1] else None
                
                if isinstance(dataset, DatasetDict):
                    return dataset[split_name].select(range(start or 0, end or len(dataset[split_name])))
                else:
                    return dataset.select(range(start or 0, end or len(dataset)))
            else:
                # Handle single index
                idx = int(split_range)
                if isinstance(dataset, DatasetDict):
                    return dataset[split_name].select([idx])
                else:
                    return dataset.select([idx])
        else:
            # Simple split name
            if isinstance(dataset, DatasetDict):
                return dataset[split_expr]
            else:
                return dataset
    
    def preprocess(self, dataset: DatasetDict, tokenizer) -> DatasetDict:
        """Preprocess dataset based on task type."""
        if self.task_type == "SEQ_CLS":
            return self._preprocess_classification(dataset, tokenizer)
        elif self.task_type == "CAUSAL_LM":
            return self._preprocess_causal_lm(dataset, tokenizer)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _preprocess_classification(self, dataset: DatasetDict, tokenizer) -> DatasetDict:
        """Preprocess classification datasets (e.g., GLUE)."""
        text_column = self.dataset_config.get("text_column", [])
        label_column = self.dataset_config.get("label_column", "label")
        max_length = self.dataset_config.get("max_length", 512)
        num_workers = self.dataset_config.get("preprocessing_num_workers", 4)
        
        def tokenize_function(examples):
            if isinstance(text_column, list) and len(text_column) == 2:
                # Sentence pair task
                result = tokenizer(
                    examples[text_column[0]],
                    examples[text_column[1]],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                )
            else:
                # Single sentence task
                col = text_column[0] if isinstance(text_column, list) else text_column
                result = tokenizer(
                    examples[col],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                )
            
            # Rename label column to 'labels' if needed (transformers expects 'labels')
            if label_column in examples and label_column != "labels":
                result["labels"] = examples[label_column]
            elif "labels" in examples:
                result["labels"] = examples["labels"]
            
            return result
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_workers,
            desc="Tokenizing dataset",
        )
        
        return tokenized_dataset
    
    def _preprocess_causal_lm(self, dataset: DatasetDict, tokenizer) -> DatasetDict:
        """Preprocess causal LM datasets with flexible prompt/completion templates."""

        dataset_cfg = self.dataset_config
        num_workers = dataset_cfg.get("preprocessing_num_workers", 8)
        max_length = dataset_cfg.get("max_length") or tokenizer.model_max_length
        append_eos = dataset_cfg.get("append_eos_token", True)
        add_bos = dataset_cfg.get("add_bos_token", False)
        completion_prefix = dataset_cfg.get("completion_prefix", "")
        completion_suffix = dataset_cfg.get("completion_suffix", "")
        truncate_prompt_only = dataset_cfg.get("truncate_prompt_only", True)
        truncate_from_end = dataset_cfg.get("truncate_from_end", True)
        apply_chat_template = dataset_cfg.get("apply_chat_template", False)
        system_prompt = dataset_cfg.get("system_prompt")
        assistant_prefix = dataset_cfg.get("assistant_prefix", "")

        candidate_columns = dataset["train"].column_names if isinstance(dataset, DatasetDict) else dataset.column_names
        target_field = dataset_cfg.get("target_field")
        if not target_field:
            for candidate in ["response", "answer", "output", "target", "completion", "text", "label"]:
                if candidate in candidate_columns:
                    target_field = candidate
                    break
        if not target_field:
            raise ValueError("Unable to infer target field for causal LM task. Set dataset.target_field in config.")
        if target_field not in candidate_columns:
            raise ValueError(f"Target field '{target_field}' not found in dataset columns: {candidate_columns}")

        input_fields = dataset_cfg.get("input_fields")
        if isinstance(input_fields, str):
            input_fields = [input_fields]
        if not input_fields:
            preferred_inputs = [
                "instruction",
                "input",
                "question",
                "query",
                "context",
                "prompt",
                "text",
            ]
            input_fields = [col for col in preferred_inputs if col in candidate_columns and col != target_field]
        if not input_fields:
            input_fields = [col for col in candidate_columns if col != target_field]
        if not input_fields:
            raise ValueError("Unable to infer prompt fields. Set dataset.input_fields in the config.")

        prompt_template = dataset_cfg.get("prompt_template")
        response_template = dataset_cfg.get("response_template", "{target}")
        strip_whitespace = dataset_cfg.get("strip_prompt", True)

        def _render_prompt(example_values: Dict[str, str]) -> str:
            if prompt_template:
                formatted = prompt_template.format_map(_SafeFormatDict(**example_values))
            else:
                formatted = "\n\n".join(
                    f"{field}: {example_values.get(field, '')}" for field in input_fields if example_values.get(field)
                )
            return formatted.strip() if strip_whitespace else formatted

        def _render_response(target_text: str, example_values: Dict[str, str]) -> str:
            values = {**example_values, "target": target_text}
            formatted = response_template.format_map(_SafeFormatDict(**values))
            formatted = formatted.strip() if strip_whitespace else formatted
            return formatted

        def format_function(examples):
            prompts: List[str] = []
            responses: List[str] = []

            target_values = examples.get(target_field, [])
            example_count = len(target_values)
            for idx in range(example_count):
                values = {}
                for field in input_fields:
                    column = examples.get(field)
                    value = column[idx] if column is not None else ""
                    if value is None:
                        value = ""
                    values[field] = str(value)

                target_text = target_values[idx]
                if target_text is None:
                    continue
                target_text = str(target_text)

                prompt_text = _render_prompt(values)
                response_text = _render_response(target_text, values)

                prompts.append(prompt_text)
                responses.append(response_text)

            return {"prompt": prompts, "response": responses}

        original_columns = dataset["train"].column_names if isinstance(dataset, DatasetDict) else dataset.column_names
        formatted_dataset = dataset.map(
            format_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=original_columns,
            desc="Formatting dataset",
        )

        eos_id = tokenizer.eos_token_id
        bos_id = tokenizer.bos_token_id

        def _encode_chat(prompt_text: str, response_text: str) -> Tuple[List[int], List[int]]:
            if not hasattr(tokenizer, "apply_chat_template"):
                raise ValueError("Tokenizer does not define a chat template but 'apply_chat_template' was requested.")

            conversation = []
            if system_prompt:
                conversation.append({"role": "system", "content": system_prompt})
            conversation.append({"role": "user", "content": prompt_text})

            prompt_render = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

            full_conversation = conversation + [
                {"role": "assistant", "content": f"{assistant_prefix}{response_text}"}
            ]
            full_render = tokenizer.apply_chat_template(
                full_conversation,
                add_generation_prompt=False,
                tokenize=False,
            )

            prompt_ids = tokenizer(prompt_render, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_render, add_special_tokens=False)["input_ids"]
            response_ids = full_ids[len(prompt_ids) :]
            return prompt_ids, response_ids

        def _encode_standard(prompt_text: str, response_text: str) -> Tuple[List[int], List[int]]:
            rendered_prompt = prompt_text
            if add_bos and bos_id is not None:
                rendered_prompt = tokenizer.bos_token + rendered_prompt if tokenizer.bos_token else rendered_prompt

            prompt_ids = tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
            completion_text = f"{assistant_prefix}{completion_prefix}{response_text}{completion_suffix}"
            response_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
            return prompt_ids, response_ids

        def tokenize_function(examples):
            input_ids_list: List[List[int]] = []
            label_list: List[List[int]] = []
            attention_masks: List[List[int]] = []

            for prompt_text, response_text in zip(examples["prompt"], examples["response"]):
                prompt_text = prompt_text or ""
                response_text = response_text or ""

                if apply_chat_template:
                    prompt_ids, response_ids = _encode_chat(prompt_text, response_text)
                else:
                    prompt_ids, response_ids = _encode_standard(prompt_text, response_text)

                if append_eos and eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
                    response_ids = response_ids + [eos_id]

                total_length = len(prompt_ids) + len(response_ids)
                if max_length and total_length > max_length:
                    overflow = total_length - max_length
                    if truncate_prompt_only:
                        trim = min(len(prompt_ids), overflow)
                        prompt_ids = prompt_ids[trim:]
                        overflow -= trim
                    if overflow > 0:
                        if truncate_from_end:
                            response_ids = response_ids[:-overflow] if overflow < len(response_ids) else []
                        else:
                            response_ids = response_ids[overflow:]
                    total_length = len(prompt_ids) + len(response_ids)

                if not response_ids:
                    continue  # Skip examples with empty completions after truncation

                input_ids = prompt_ids + response_ids
                labels = [-100] * len(prompt_ids) + response_ids
                attention_mask = [1] * len(input_ids)

                input_ids_list.append(input_ids)
                label_list.append(labels)
                attention_masks.append(attention_mask)

            return {
                "input_ids": input_ids_list,
                "labels": label_list,
                "attention_mask": attention_masks,
            }

        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_workers,
            desc="Tokenizing dataset",
        )

        keep_columns = {"input_ids", "labels", "attention_mask"}
        removable_columns = [
            col for col in tokenized_dataset["train"].column_names if col not in keep_columns
        ]
        if removable_columns:
            tokenized_dataset = tokenized_dataset.remove_columns(removable_columns)

        return tokenized_dataset


def prepare_glue_dataset(config: Dict, tokenizer) -> DatasetDict:
    """Prepare GLUE dataset for training."""
    loader = DatasetLoader(config)
    dataset = loader.load()
    dataset = loader.preprocess(dataset, tokenizer)
    return dataset


def prepare_math_dataset(config: Dict, tokenizer) -> DatasetDict:
    """Prepare math reasoning datasets (MetaMathQA, GSM8K) for training."""
    loader = DatasetLoader(config)
    dataset = loader.load()
    dataset = loader.preprocess(dataset, tokenizer)
    return dataset


def prepare_code_dataset(config: Dict, tokenizer) -> DatasetDict:
    """Prepare code datasets for training."""
    loader = DatasetLoader(config)
    dataset = loader.load()
    dataset = loader.preprocess(dataset, tokenizer)
    return dataset


def prepare_dataset(config: Dict, tokenizer) -> DatasetDict:
    """Universal dataset preparation function."""
    loader = DatasetLoader(config)
    dataset = loader.load()
    
    # For causal LM tasks, preprocess to format text
    if config.get("task_type") == "CAUSAL_LM":
        dataset = loader.preprocess(dataset, tokenizer)
    else:
        # For classification tasks, tokenize
        dataset = loader.preprocess(dataset, tokenizer)
    
    return dataset
