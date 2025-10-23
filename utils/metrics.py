"""
Evaluation and metrics utilities.
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
import logging

logger = logging.getLogger(__name__)


def compute_classification_metrics(eval_pred):
    """Compute metrics for classification tasks."""
    predictions, labels = eval_pred
    
    # Get predicted class
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Compute Matthews Correlation Coefficient
    mcc = matthews_corrcoef(labels, predictions)
    
    # For binary classification
    if len(np.unique(labels)) == 2:
        f1 = f1_score(labels, predictions, average='binary')
        precision = precision_score(labels, predictions, average='binary')
        recall = recall_score(labels, predictions, average='binary')
    else:
        # For multi-class classification
        f1 = f1_score(labels, predictions, average='weighted')
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'matthews_correlation': mcc,
    }


def compute_glue_metrics(eval_pred, task_name: str = "mrpc"):
    """Compute metrics for GLUE tasks."""
    return compute_classification_metrics(eval_pred)


def compute_causal_lm_metrics(eval_pred):
    """Compute metrics for causal LM tasks."""
    predictions, labels = eval_pred
    
    # For language modeling, we typically look at perplexity
    # which is computed from the loss during training
    # Here we can return a dummy metric or compute accuracy on tokens
    
    if len(predictions.shape) > 2:
        predictions = np.argmax(predictions, axis=-1)
    
    # Flatten predictions and labels
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Remove padding tokens (usually -100)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    # Compute token-level accuracy
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
    }


def get_metrics_function(task_type: str, task_name: str = None):
    """Get appropriate metrics function based on task type."""
    if task_type == "classification":
        if task_name and "glue" in task_name.lower():
            return lambda x: compute_glue_metrics(x, task_name)
        return compute_classification_metrics
    elif task_type == "causal_lm":
        return compute_causal_lm_metrics
    else:
        logger.warning(f"Unknown task type: {task_type}, using default metrics")
        return compute_classification_metrics
