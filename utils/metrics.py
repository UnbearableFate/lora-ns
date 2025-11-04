"""
Evaluation and metrics utilities.
"""

import numpy as np
from typing import Dict, List
import evaluate
import logging

logger = logging.getLogger(__name__)

def compute_metrics(metric, eval_preds):
    logits, labels = eval_preds
    
    # Handle T5 models which may return sequence outputs
    # For T5ForSequenceClassification, logits shape can be (batch, seq_len, num_labels)
    # We need to extract the final classification logits
    if isinstance(logits, (list, tuple)):
        # If logits is a tuple/list (e.g., from some models), take the first element
        logits = logits[0]
    
    # Convert to numpy array if needed and handle irregular shapes
    if not isinstance(logits, np.ndarray):
        try:
            logits = np.array(logits)
        except:
            # If conversion fails, try to extract from nested structure
            logits = np.array([np.array(x) for x in logits], dtype=object)
    
    # Handle different logit shapes
    if len(logits.shape) == 3:
        # For T5: (batch, seq_len, num_labels) -> take the last token's logits
        logits = logits[:, -1, :]
    elif len(logits.shape) == 1:
        # If 1D, assume binary classification with single logit per sample
        predictions = (logits > 0).astype(int)
        return metric.compute(predictions=predictions, references=labels)
    
    # Standard argmax for 2D logits (batch, num_labels)
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def get_metrics_function(task_name: str) :
    split_task = task_name.split("_") if task_name else []
    if split_task[0] == "glue":
        if split_task[1] in ["sst2", "cola", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"]:
            metric = evaluate.load("glue", split_task[1])
            return lambda eval_preds: compute_metrics(metric, eval_preds)
        else:
            raise ValueError(f"Unsupported GLUE subtask: {split_task[1]}")

