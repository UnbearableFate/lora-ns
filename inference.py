"""
Inference script for PEFT models.
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
    parser = argparse.ArgumentParser(description="PEFT Inference Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained PEFT model",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default=None,
        help="Input text for inference",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input file with texts (one per line)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file to save results",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    return parser.parse_args()


def load_model_for_inference(config: dict, model_path: str):
    """Load model and tokenizer for inference."""
    model_name = config["model"]["name_or_path"]
    task_type = config.get("task_type", "CAUSAL_LM")
    
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading base model from {model_name}")
    if task_type == "CAUSAL_LM":
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
        )
    
    logger.info(f"Loading PEFT weights from {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 512):
    """Generate text from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def classify_text(model, tokenizer, text: str):
    """Classify text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    
    return predicted_class, probabilities


def main():
    """Main inference function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    task_type = config.get("task_type", "CAUSAL_LM")
    
    # Load model
    model, tokenizer = load_model_for_inference(config, args.model_path)
    
    # Prepare inputs
    inputs = []
    if args.input_text:
        inputs = [args.input_text]
    elif args.input_file:
        with open(args.input_file, 'r') as f:
            inputs = [line.strip() for line in f if line.strip()]
    else:
        logger.error("Please provide either --input_text or --input_file")
        return
    
    # Run inference
    results = []
    for i, input_text in enumerate(inputs):
        logger.info(f"Processing input {i+1}/{len(inputs)}")
        
        if task_type == "CAUSAL_LM":
            output = generate_text(model, tokenizer, input_text, args.max_new_tokens)
            results.append({
                "input": input_text,
                "output": output,
            })
            logger.info(f"Input: {input_text}")
            logger.info(f"Output: {output}")
            logger.info("-" * 50)
        else:
            predicted_class, probabilities = classify_text(model, tokenizer, input_text)
            results.append({
                "input": input_text,
                "predicted_class": predicted_class,
                "probabilities": probabilities.tolist(),
            })
            logger.info(f"Input: {input_text}")
            logger.info(f"Predicted class: {predicted_class}")
            logger.info(f"Probabilities: {probabilities}")
            logger.info("-" * 50)
    
    # Save results
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
