"""
Quick start example for PEFT training.
This script demonstrates how to quickly train a model on a small dataset.
"""

from utils import load_config, setup_model_and_tokenizer, prepare_dataset
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_train():
    """Quick training example."""
    
    # Load config
    config = load_config("configs/glue_mrpc.yaml")
    
    # Modify for quick training (small subset)
    config["dataset"]["train_split"] = "train[:100]"
    config["dataset"]["eval_split"] = "validation[:50]"
    config["training"]["num_train_epochs"] = 1
    config["training"]["per_device_train_batch_size"] = 8
    config["training"]["save_steps"] = 50
    config["training"]["eval_steps"] = 50
    
    logger.info("Loading model and tokenizer...")
    model, tokenizer, peft_config = setup_model_and_tokenizer(config)
    
    logger.info("Preparing dataset...")
    dataset = prepare_dataset(config, tokenizer)
    
    logger.info(f"Training on {len(dataset['train'])} examples")
    logger.info(f"Evaluating on {len(dataset['validation'])} examples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/quick_test",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=10,
        learning_rate=3e-4,
        remove_unused_columns=True,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save
    trainer.save_model()
    logger.info("Model saved to ./outputs/quick_test")
    
    # Evaluate
    logger.info("Evaluating...")
    metrics = trainer.evaluate()
    logger.info(f"Metrics: {metrics}")

if __name__ == "__main__":
    quick_train()
