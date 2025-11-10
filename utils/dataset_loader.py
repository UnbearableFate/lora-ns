"""
Dataset loading utilities for various tasks.
Supports GLUE, MetaMathQA, GSM8K, Code-Feedback, and custom datasets.
"""

import os
from typing import Dict, List, Optional, Union
from datasets import load_dataset, Dataset, DatasetDict
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and preprocess datasets based on task type."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.task_type = config.get("peft").get("task_type", "CAUSAL_LM")
        self.dataset_config = config.get("dataset", {})
        
    def load(self) -> DatasetDict:
        """Load dataset based on configuration."""
        dataset_name = self.dataset_config.get("name")
        subset = self.dataset_config.get("subset")
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Load dataset from HuggingFace
        if subset:
            dataset = load_dataset(dataset_name, subset)
        else:
            dataset = load_dataset(dataset_name)
        
        # Get splits
        train_split = self.dataset_config.get("train_split", "train")
        eval_split = self.dataset_config.get("eval_split")
        
        # Handle split expressions like "train[:5000]"
        train_data = self._get_split(dataset, train_split)
        eval_data = self._get_split(dataset, eval_split) if eval_split else None
        
        # Create dataset dict
        dataset_dict = {"train": train_data}
        if eval_data:
            dataset_dict["validation"] = eval_data
            
        return DatasetDict(dataset_dict)
    
    def _get_split(self, dataset: Union[Dataset, DatasetDict], split_expr: str) -> Dataset:
        """Get dataset split, handling expressions like 'train[:1000]'."""
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
        """Preprocess causal LM datasets (e.g., MetaMathQA, GSM8K, Code-Feedback)."""
        task_name = self.config.get("task_name", "")
        prompt_template = self.dataset_config.get("prompt_template", "")
        max_length = self.dataset_config.get("max_length", 2048)
        num_workers = self.dataset_config.get("preprocessing_num_workers", 8)
        
        def format_function(examples):
            """Format examples based on task type."""
            formatted_texts = []
            
            if "metamath" in task_name.lower():
                # MetaMathQA format
                for query, response in zip(examples.get("query", []), examples.get("response", [])):
                    text = prompt_template.format(query=query, response=response)
                    formatted_texts.append(text)
                    
            elif "gsm8k" in task_name.lower():
                # GSM8K format
                for question, answer in zip(examples.get("question", []), examples.get("answer", [])):
                    text = prompt_template.format(question=question, answer=answer)
                    formatted_texts.append(text)
                    
            elif "code" in task_name.lower():
                # Code-Feedback format
                for inst, inp, out in zip(
                    examples.get("instruction", []),
                    examples.get("input", examples.get("query", [""] * len(examples.get("instruction", [])))),
                    examples.get("output", examples.get("response", []))
                ):
                    text = prompt_template.format(instruction=inst, input=inp, output=out)
                    formatted_texts.append(text)
            else:
                # Generic format - try to use 'text' field or first text field
                if "text" in examples:
                    formatted_texts = examples["text"]
                else:
                    # Fallback: concatenate all string fields
                    for i in range(len(list(examples.values())[0])):
                        text_parts = []
                        for key, values in examples.items():
                            if isinstance(values[i], str):
                                text_parts.append(f"{key}: {values[i]}")
                        formatted_texts.append("\n".join(text_parts))
            
            return {"text": formatted_texts}
        
        # Format dataset
        formatted_dataset = dataset.map(
            format_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=dataset["train"].column_names,
            desc="Formatting dataset",
        )
        
        return formatted_dataset


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
