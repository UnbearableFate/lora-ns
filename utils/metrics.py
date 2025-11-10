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


def extract_math_answer(text: str) -> str:
    """
    Extract the final answer from math problem solutions.
    Supports multiple formats:
    - "#### 42" (MetaMathQA/GSM8K format)
    - "The answer is 42"
    - "Final answer: 42"
    - Last number in the text
    """
    if not text:
        return ""
    
    # Try #### format first (most reliable for MetaMathQA)
    if "####" in text:
        answer = text.split("####")[-1].strip()
        # Extract the first number from the answer
        numbers = re.findall(r'-?\d+\.?\d*', answer)
        if numbers:
            return numbers[0]
        return answer
    
    # Try "the answer is" pattern (case insensitive)
    answer_patterns = [
        r"the answer is[:\s]+(-?\d+\.?\d*)",
        r"final answer[:\s]+(-?\d+\.?\d*)",
        r"answer[:\s]+(-?\d+\.?\d*)",
    ]
    
    text_lower = text.lower()
    for pattern in answer_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Fallback: extract the last number in the text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    
    return text.strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove whitespace
    answer = answer.strip()
    
    # Try to convert to number for numeric comparison
    try:
        # Handle fractions and decimals
        if '/' in answer:
            parts = answer.split('/')
            if len(parts) == 2:
                num = float(parts[0]) / float(parts[1])
                return str(num)
        else:
            num = float(answer)
            # Remove trailing zeros for decimal numbers
            if '.' in answer:
                return str(num).rstrip('0').rstrip('.')
            return str(int(num))
    except (ValueError, ZeroDivisionError):
        pass
    
    # If not a number, return lowercase stripped version
    return answer.lower().strip()


def compute_causal_lm_metrics(eval_preds) -> Dict[str, float]:
    """
    Compute basic metrics for causal language modeling tasks.
    
    Metrics:
    - Token accuracy: Percentage of correctly predicted tokens
    """
    predictions, labels = eval_preds
    
    # predictions can be logits or token IDs
    # If predictions are logits (3D), take argmax
    if len(predictions.shape) == 3:
        pred_ids = np.argmax(predictions, axis=-1)
    else:
        pred_ids = predictions
    
    # Flatten and mask out padding tokens (label = -100)
    mask = labels != -100
    
    # Token accuracy
    correct = (pred_ids == labels) & mask
    token_accuracy = correct.sum() / mask.sum() if mask.sum() > 0 else 0.0
    
    return {
        "token_accuracy": float(token_accuracy),
    }


def compute_math_generation_metrics(tokenizer) -> Callable:
    """
    Create a compute_metrics function for math generation tasks.
    
    Metrics:
    - Token accuracy: Token-level accuracy
    - Answer accuracy: Exact match accuracy on extracted answers
    
    Args:
        tokenizer: The tokenizer used for decoding
    
    Returns:
        A compute_metrics function
    """
    def compute_metrics(eval_preds) -> Dict[str, float]:
        predictions, labels = eval_preds
        
        # Get predicted token IDs
        if len(predictions.shape) == 3:
            pred_ids = np.argmax(predictions, axis=-1)
        else:
            pred_ids = predictions
        
        # Basic token accuracy
        mask = labels != -100
        correct = (pred_ids == labels) & mask
        token_accuracy = correct.sum() / mask.sum() if mask.sum() > 0 else 0.0
        
        # Decode predictions and labels for answer extraction
        decoded_preds = []
        decoded_labels = []
        
        for pred, label in zip(pred_ids, labels):
            # Remove padding tokens
            valid_indices = label != -100
            pred_valid = pred[valid_indices]
            label_valid = label[valid_indices]
            
            try:
                pred_text = tokenizer.decode(pred_valid, skip_special_tokens=True)
                label_text = tokenizer.decode(label_valid, skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"Decoding error: {e}")
                pred_text = ""
                label_text = ""
            
            decoded_preds.append(pred_text)
            decoded_labels.append(label_text)
        
        # Extract answers
        pred_answers = [extract_math_answer(p) for p in decoded_preds]
        label_answers = [extract_math_answer(l) for l in decoded_labels]
        
        # Normalize answers for comparison
        pred_answers_norm = [normalize_answer(a) for a in pred_answers]
        label_answers_norm = [normalize_answer(a) for a in label_answers]
        
        # Answer accuracy (exact match after normalization)
        answer_correct = sum(
            p == l for p, l in zip(pred_answers_norm, label_answers_norm)
        )
        answer_accuracy = answer_correct / len(pred_answers) if pred_answers else 0.0
        
        # Log some examples for debugging (first 3)
        if len(decoded_preds) > 0:
            logger.info("\n" + "="*60)
            logger.info("Sample predictions (for debugging):")
            for i in range(min(3, len(decoded_preds))):
                logger.info(f"\nExample {i+1}:")
                logger.info(f"  Prediction: {decoded_preds[i][:100]}...")
                logger.info(f"  Label: {decoded_labels[i][:100]}...")
                logger.info(f"  Extracted pred answer: {pred_answers[i]}")
                logger.info(f"  Extracted label answer: {label_answers[i]}")
                logger.info(f"  Match: {pred_answers_norm[i] == label_answers_norm[i]}")
            logger.info("="*60 + "\n")
        
        return {
            "token_accuracy": float(token_accuracy),
            "answer_accuracy": float(answer_accuracy),
        }
    
    return compute_metrics


def get_metrics_function(task_name: str, tokenizer=None) -> Optional[Callable]:
    """
    Get the appropriate metrics function based on task name.
    
    Args:
        task_name: Name of the task (e.g., "glue_sst2", "metamath_qa")
        tokenizer: Tokenizer for generative tasks (optional)
    
    Returns:
        A compute_metrics function or None
    """
    split_task = task_name.split("_") if task_name else []
    
    # GLUE tasks
    if split_task[0] == "glue":
        if split_task[1] in ["sst2", "cola", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"]:
            metric = evaluate.load("glue", split_task[1])
            return lambda eval_preds: compute_metrics(metric, eval_preds)
        else:
            raise ValueError(f"Unsupported GLUE subtask: {split_task[1]}")
    
    # Math tasks
    elif any(keyword in task_name.lower() for keyword in ["math", "metamath", "gsm8k"]):
        if tokenizer is None:
            logger.warning("Tokenizer not provided for math task, using basic metrics")
            return compute_causal_lm_metrics
        return compute_math_generation_metrics(tokenizer)
    
    # Generic causal LM tasks
    elif "causal" in task_name.lower() or "lm" in task_name.lower():
        return compute_causal_lm_metrics
    
    # Code generation tasks
    elif "code" in task_name.lower():
        if tokenizer is None:
            logger.warning("Tokenizer not provided for code task, using basic metrics")
            return compute_causal_lm_metrics
        # For now, use basic metrics; can be extended with code-specific metrics
        return compute_math_generation_metrics(tokenizer)
    
    # Default: return None (trainer will only use loss)
    logger.info(f"No specific metrics defined for task '{task_name}', will use loss only")
    return None

