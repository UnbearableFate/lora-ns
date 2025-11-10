#!/bin/bash
# Training script for GSM8K with SmolLM2-135M

echo "================================================"
echo "Training SmolLM2-135M on GSM8K"
echo "================================================"
echo ""
echo "Configuration: configs/smol/135m_gsm8k.yaml"
echo "Model: HuggingFaceTB/SmolLM2-135M"
echo "Dataset: GSM8K (Grade School Math 8K)"
echo "Trainer: SpectralRefactorTrainer"
echo ""
echo "Dataset Info:"
echo "  - Train: ~7,500 problems"
echo "  - Test: ~1,300 problems"
echo ""
echo "================================================"
echo ""

# Run training
python train.py --config configs/smol/135m_gsm8k.yaml

echo ""
echo "================================================"
echo "Training complete!"
echo "Check outputs at: ./outputs/smol_135m_gsm8k/"
echo "================================================"
