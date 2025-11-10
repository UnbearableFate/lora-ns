# 数学推理任务配置 - 快速参考

## 📊 可用配置

### 1. MetaMathQA
```bash
# 配置文件
configs/smol/135m_metamath.yaml

# 训练
python train.py --config configs/smol/135m_metamath.yaml
./examples/train_smol_metamath.sh

# 测试
python test_full_pipeline.py
```

**特点**:
- 数据量: 395K（配置中使用 10K 子集）
- 字段: `query`, `response`
- 适合: 全面的数学推理训练

### 2. GSM8K
```bash
# 配置文件
configs/smol/135m_gsm8k.yaml

# 训练
python train.py --config configs/smol/135m_gsm8k.yaml
./examples/train_smol_gsm8k.sh

# 测试
python test_gsm8k_config.py
```

**特点**:
- 数据量: 7.5K
- 字段: `question`, `answer`
- 适合: 快速原型、基准测试

## 🔑 关键差异对比

| 配置项 | MetaMathQA | GSM8K |
|--------|------------|-------|
| **数据集** | `meta-math/MetaMathQA` | `gsm8k` |
| **Subset** | `null` | `main` |
| **规模** | 395K | 7.5K |
| **Epochs** | 1 | 3 |
| **Batch Size** | 2 | 4 |
| **Eval Times** | 50 | 30 |
| **WandB Project** | `SmolLM2-135M-MetaMath` | `SmolLM2-135M-GSM8K` |

## 📝 Prompt 模板

### MetaMathQA
```
### Instruction:
{query}

### Response:
{response}
```

### GSM8K
```
### Question:
{question}

### Answer:
{answer}
```

## 📈 评估指标（两者相同）

```yaml
metric_for_best_model: "answer_accuracy"
greater_is_better: true
```

自动计算：
- `token_accuracy` - Token 级准确率
- `answer_accuracy` - 答案精确匹配率

## 🚀 快速开始

### 选择数据集
```bash
# 小数据集、快速测试 → GSM8K
python train.py --config configs/smol/135m_gsm8k.yaml

# 大数据集、完整训练 → MetaMathQA
python train.py --config configs/smol/135m_metamath.yaml
```

### 自定义配置
修改 YAML 文件中的：
- `trainer.name`: `"Trainer"` 或 `"SpectralRefactorTrainer"`
- `peft.init_lora_weights`: `pissa`, `lora_ga`, `lora_ns`, `gaussian`
- `training.per_device_train_batch_size`: 根据 GPU 内存调整
- `wandb.online`: `true` 启用在线 WandB

## 📁 相关文件

```
configs/smol/
├── 135m_metamath.yaml   # MetaMathQA 配置
└── 135m_gsm8k.yaml      # GSM8K 配置

examples/
├── train_smol_metamath.sh   # MetaMathQA 训练脚本
└── train_smol_gsm8k.sh      # GSM8K 训练脚本

tests/
├── test_full_pipeline.py    # MetaMathQA 测试
└── test_gsm8k_config.py     # GSM8K 测试

docs/
├── METRICS_GUIDE.md         # 详细 metrics 文档
├── METRICS_QUICKREF.md      # Metrics 快速参考
└── GSM8K_CONFIG_GUIDE.md    # GSM8K 配置指南
```

## ✅ 检查清单

训练前确认：
- [ ] Python 环境已激活
- [ ] 依赖已安装（transformers, datasets, peft, etc.）
- [ ] GPU 可用（`nvidia-smi`）
- [ ] 配置文件路径正确
- [ ] WandB 设置符合需求

## 🎯 推荐工作流

```bash
# 1. 测试配置
python test_gsm8k_config.py  # 或 test_full_pipeline.py

# 2. 开始训练
python train.py --config configs/smol/135m_gsm8k.yaml

# 3. 监控训练
# 查看 WandB 或终端输出

# 4. 评估结果
# 检查 outputs/ 目录下的指标
```

---
✨ 两个配置都已完全测试并可用！
