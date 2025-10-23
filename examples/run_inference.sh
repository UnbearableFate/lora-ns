#!/bin/bash
# Example inference script

MODEL_PATH="./outputs/gsm8k"
CONFIG="configs/gsm8k.yaml"

echo "Running inference on GSM8K model..."

python inference.py \
  --config $CONFIG \
  --model_path $MODEL_PATH \
  --input_text "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?" \
  --max_new_tokens 256

echo "Inference complete!"
