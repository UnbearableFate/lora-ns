#!/bin/bash
# Training script for MetaMathQA with SmolLM2-135M

echo "================================================"
echo "Training SmolLM2-135M on MetaMathQA"
echo "================================================"
echo ""
echo "Configuration: configs/smol/135m_metamath.yaml"
echo "Model: HuggingFaceTB/SmolLM2-135M"
echo "Dataset: meta-math/MetaMathQA"
echo "Trainer: SpectralRefactorTrainer"
echo ""
echo "================================================"
echo ""

# Run training
python train.py --config configs/smol/135m_metamath.yaml

echo ""
echo "================================================"
echo "Training complete!"
echo "Check outputs at: ./outputs/smol_135m_metamath/"
echo "================================================"
