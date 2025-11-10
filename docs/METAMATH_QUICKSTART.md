# MetaMathQA 训练指南

## 🎯 快速开始

### 1. 测试配置（推荐）
```bash
./test_metamath_setup.sh
```

### 2. 开始训练
```bash
# 单GPU
python train.py --config configs/smol/135m_metamath.yaml

# 或使用脚本
./examples/train_smol_metamath.sh

# 多GPU (使用 accelerate)
accelerate launch train.py --config configs/smol/135m_metamath.yaml
```

## 📋 配置说明

- **配置文件**: `configs/smol/135m_metamath.yaml`
- **模型**: SmolLM2-135M (小型高效)
- **数据集**: MetaMathQA (数学推理)
- **训练方法**: LoRA + PiSSA + SpectralRefactorTrainer
- **默认设置**: 使用10,000样本子集进行快速训练

## 🔧 主要改进

✅ 添加了 `DataCollatorForLanguageModeling`  
✅ 完善了数据 tokenization 流程  
✅ 支持 SpectralRefactorTrainer 用于 CAUSAL_LM  
✅ 自动创建 labels 用于语言模型训练

## 📊 查看详细信息

- 完整总结: [METAMATH_SETUP_SUMMARY.md](METAMATH_SETUP_SUMMARY.md)
- 技术分析: [METAMATH_SUPPORT_ANALYSIS.md](METAMATH_SUPPORT_ANALYSIS.md)

## 📈 监控训练

训练输出位置:
- 模型: `./outputs/smol_135m_metamath/SpectralRefactor/`
- 日志: `./outputs/smol_135m_metamath/logs/`
- WandB: 项目 `SmolLM2-135M-MetaMath`

---
创建日期: 2025-11-10
