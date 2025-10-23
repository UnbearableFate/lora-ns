#!/bin/bash
# Example training script for GLUE MRPC task

echo "Training on GLUE MRPC task..."
python train.py --config configs/glue_mrpc.yaml

echo "Training complete! Model saved to ./outputs/glue_mrpc"
