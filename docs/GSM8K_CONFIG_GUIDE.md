# GSM8K 配置说明

## 📊 数据集概述

### GSM8K (Grade School Math 8K)
- **来源**: OpenAI
- **规模**: 
  - 训练集: ~7,500 个问题
  - 测试集: ~1,319 个问题
- **难度**: 小学数学水平
- **特点**: 
  - 多步骤推理问题
  - 每个问题都有详细的解答步骤
  - 答案格式: `#### [数字]`

### 数据格式
GSM8K 使用以下字段：
- `question`: 数学问题
- `answer`: 完整的解答过程和最终答案

示例：
```
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Answer: Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72
```

## 🆚 与 MetaMathQA 的对比

| 特性 | GSM8K | MetaMathQA |
|------|-------|------------|
| **数据规模** | ~7.5K 训练 | ~395K 训练 |
| **问题来源** | 原创 | 从其他数据集增强 |
| **难度** | 小学数学 | 混合难度 |
| **字段名称** | `question`, `answer` | `query`, `response` |
| **答案格式** | `#### number` | `#### number` |
| **训练时间** | 较短 | 较长 |
| **适合场景** | 快速原型、基准测试 | 全面数学推理训练 |

## ⚙️ 配置差异

### 1. 数据集配置

**GSM8K**:
```yaml
dataset:
  name: "gsm8k"
  subset: "main"  # GSM8K 有 'main' subset
  train_split: "train"  # 完整训练集
  eval_split: "test"  # 官方测试集
```

**MetaMathQA**:
```yaml
dataset:
  name: "meta-math/MetaMathQA"
  subset: null  # 无 subset
  train_split: "train[:10000]"  # 使用子集
  eval_split: "train[10000:12000]"  # 从训练集划分
```

### 2. Prompt 模板

**GSM8K** (`configs/smol/135m_gsm8k.yaml`):
```yaml
prompt_template: |
  Below is a math problem. Solve it step by step and provide the final answer.
  
  ### Question:
  {question}
  
  ### Answer:
  {answer}
```

**MetaMathQA** (`configs/smol/135m_metamath.yaml`):
```yaml
prompt_template: |
  Below is an instruction that describes a task. Write a response that appropriately completes the request.
  
  ### Instruction:
  {query}
  
  ### Response:
  {response}
```

### 3. 训练超参数

| 参数 | GSM8K | MetaMathQA |
|------|-------|------------|
| `num_train_epochs` | 3 | 1 |
| `per_device_train_batch_size` | 4 | 2 |
| `gradient_accumulation_steps` | 2 | 1 |
| `total_eval_times` | 30 | 50 |
| `warmup_ratio` | 0.06 | 0.05 |

**原因**:
- GSM8K 数据量较小，可以训练更多 epoch
- MetaMathQA 数据量大，1 epoch 就足够
- GSM8K 使用稍大的 batch size 以提高稳定性

## 🚀 使用方法

### 方法1: 直接运行
```bash
python train.py --config configs/smol/135m_gsm8k.yaml
```

### 方法2: 使用脚本
```bash
./examples/train_smol_gsm8k.sh
```

### 方法3: 测试配置
```bash
python test_gsm8k_config.py
```

## 📈 评估指标

两个配置都使用相同的评估指标：

1. **token_accuracy**: Token 级别准确率
2. **answer_accuracy**: 提取答案后的精确匹配率

```yaml
training:
  metric_for_best_model: "answer_accuracy"
  greater_is_better: true
```

## 🎯 推荐使用场景

### 使用 GSM8K 当：
- ✅ 快速原型开发
- ✅ 基准测试和对比
- ✅ 资源有限（小数据集）
- ✅ 需要标准化评估（官方测试集）
- ✅ 研究小学数学推理

### 使用 MetaMathQA 当：
- ✅ 需要更强的数学能力
- ✅ 有充足的计算资源
- ✅ 想要更全面的数学训练
- ✅ 需要处理多样化的数学问题
- ✅ 追求最佳性能

## 📝 注意事项

### GSM8K 特定注意事项

1. **数据集下载**: 首次运行会下载 GSM8K（~2MB）
2. **测试集**: 使用官方测试集，不要在测试集上训练
3. **答案格式**: 确保模型输出包含 `#### number` 格式
4. **评估**: 可以直接与论文中的结果对比

### 通用注意事项

1. **Metrics 自动检测**: `task_name: "gsm8k"` 会自动使用数学 metrics
2. **Data Collator**: 已配置 `pad_to_multiple_of=8`
3. **Labels**: 由 `DataCollatorForLanguageModeling` 自动创建
4. **WandB**: 默认 `online: false`，可根据需要修改

## 🔧 自定义配置

### 增加训练数据
```yaml
dataset:
  train_split: "train"  # 使用完整训练集
```

### 调整 batch size
```yaml
training:
  per_device_train_batch_size: 8  # 如果 GPU 内存充足
  gradient_accumulation_steps: 1
```

### 使用标准 Trainer（不使用 SpectralRefactor）
```yaml
trainer:
  name: "Trainer"
```

### 启用在线 WandB 记录
```yaml
wandb:
  online: true
```

## 📊 预期结果

### 训练时间（单 GPU, RTX 3090）
- **GSM8K**: ~30-45 分钟（3 epochs）
- **MetaMathQA**: ~1-2 小时（1 epoch, 10K samples）

### 性能指标（SmolLM2-135M）
由于模型较小，不要期望达到 SOTA 水平，但应该能看到：
- Token accuracy: 60-75%
- Answer accuracy: 20-40%（基线模型）

*注意: 这是参考范围，实际结果取决于具体配置和训练过程*

## 📚 相关文件

### 配置文件
- GSM8K: `configs/smol/135m_gsm8k.yaml`
- MetaMathQA: `configs/smol/135m_metamath.yaml`

### 训练脚本
- GSM8K: `examples/train_smol_gsm8k.sh`
- MetaMathQA: `examples/train_smol_metamath.sh`

### 测试脚本
- GSM8K: `test_gsm8k_config.py`
- MetaMathQA: `test_full_pipeline.py`

### 文档
- Metrics 指南: `docs/METRICS_GUIDE.md`
- 快速参考: `docs/METRICS_QUICKREF.md`

## ✅ 检查清单

开始训练前：
- [ ] 配置文件加载成功
- [ ] Tokenizer 设置正确（pad_token）
- [ ] 数据集下载完成
- [ ] Data collator 配置正确
- [ ] Metrics 自动检测工作
- [ ] WandB 设置符合需求

## 🎉 总结

GSM8K 配置已经完全就绪！

**快速开始**:
```bash
# 测试配置
python test_gsm8k_config.py

# 开始训练
python train.py --config configs/smol/135m_gsm8k.yaml
```

所有必要的组件都已配置：
- ✅ 数据加载和预处理
- ✅ Prompt 模板
- ✅ 评估指标（token + answer accuracy）
- ✅ Data collator with padding
- ✅ SpectralRefactorTrainer 支持
- ✅ WandB 集成

准备好训练了！🚀
