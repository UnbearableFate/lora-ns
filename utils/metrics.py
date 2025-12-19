"""
Evaluation and metrics utilities.
"""

import re
import numpy as np
from typing import Dict, List, Callable, Optional
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


def get_glue_metrics_function(dataset_name: str) -> Optional[Callable]:
    """
    Get the appropriate GLUE metrics function based on dataset name.
    
    Args:
        dataset_name: Name of the GLUE dataset (e.g., "sst2", "mrpc")
    
    Returns:
        A compute_metrics function or None
    """
    supported_datasets = [
        "sst2", "cola", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"
    ]
    
    if dataset_name in supported_datasets:
        metric = evaluate.load("glue", dataset_name)
        return lambda eval_preds: compute_metrics(metric, eval_preds)
    
    logger.warning(f"Unsupported GLUE dataset: {dataset_name}")
    return None