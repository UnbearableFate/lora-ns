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

