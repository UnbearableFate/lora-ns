#!/bin/bash
# Example multi-GPU training script using Accelerate

echo "Configuring Accelerate (run once)..."
# accelerate config

echo "Training with Accelerate on GSM8K task..."
accelerate launch train.py --config configs/gsm8k.yaml

echo "Training complete!"
