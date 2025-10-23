#!/bin/bash
# Training script for GLUE CoLA task with Qwen2.5-1.5B

echo "Training GLUE CoLA task with Qwen2.5-1.5B..."
echo "Task: Corpus of Linguistic Acceptability (Binary Classification)"
echo "Model: Qwen/Qwen2.5-1.5B"
echo "PEFT Method: LoRA (r=8, alpha=16)"
echo ""

python train.py --config configs/glue_cola_qwen.yaml

echo ""
echo "Training complete! Model saved to ./outputs/glue_cola_qwen"
echo ""
echo "To run inference:"
echo "python inference.py \\"
echo "  --config configs/glue_cola_qwen.yaml \\"
echo "  --model_path ./outputs/glue_cola_qwen \\"
echo "  --input_text 'The book was written by John.'"
