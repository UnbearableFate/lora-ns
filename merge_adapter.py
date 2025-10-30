"""
Merge LoRA weights with base model and save the full model.
Useful for deployment and inference without PEFT dependencies.
"""

import os
import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel

from utils import load_config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapter weights",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save merged model",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID (username/model-name)",
    )
    return parser.parse_args()


def merge_and_save(config: dict, adapter_path: str, output_path: str):
    """Merge LoRA weights with base model and save."""
    model_name = config["model"]["name_or_path"]
    task_type = config.get("task_type", "CAUSAL_LM")
    
    logger.info(f"Loading base model from {model_name}")
    
    # Load base model
    if task_type == "CAUSAL_LM":
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=config["model"].get("trust_remote_code", True),
        )
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=config["model"].get("trust_remote_code", True),
        )
    
    logger.info(f"Loading LoRA adapter from {adapter_path}")
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("Merging LoRA weights with base model...")
    
    # Merge weights
    merged_model = model.merge_and_unload()
    
    logger.info(f"Saving merged model to {output_path}")
    
    # Save merged model
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    
    # Save tokenizer
    logger.info("Saving tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)
    
    logger.info("Model merge complete!")
    
    return merged_model, tokenizer


def main():
    """Main function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Merge and save
    merged_model, tokenizer = merge_and_save(config, args.adapter_path, args.output_path)
    
    # Push to hub if requested
    if args.push_to_hub:
        if not args.hub_model_id:
            logger.error("--hub_model_id is required when using --push_to_hub")
            return
        
        logger.info(f"Pushing model to HuggingFace Hub: {args.hub_model_id}")
        merged_model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        logger.info("Model pushed to Hub successfully!")


if __name__ == "__main__":
    main()
