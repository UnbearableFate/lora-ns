#!/bin/bash
# Example training script for MetaMathQA task

echo "Training on MetaMathQA task..."
python train.py --config configs/metamath_qa.yaml

echo "Training complete! Model saved to ./outputs/metamath_qa"
