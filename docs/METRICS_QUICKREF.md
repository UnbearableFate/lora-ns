# MetaMathQA Metrics - 快速参考

## 📊 新增的评估指标

### Token Accuracy
- **定义**: Token 级别的预测准确率
- **计算**: `正确预测的tokens / 总tokens（不含padding）`
- **用途**: 评估整体生成质量

### Answer Accuracy  
- **定义**: 提取答案后的精确匹配率
- **计算**: `答案完全匹配的样本数 / 总样本数`
- **用途**: 评估数学推理能力（MetaMathQA的核心指标）

## 🎯 支持的答案格式

```python
"#### 42"                      # MetaMathQA 标准格式 ✓
"The answer is 42"             # 自然语言格式 ✓
"Final answer: 256"            # 明确标注格式 ✓
"Therefore x = -5.5"           # 推理结果格式 ✓
"Calculate: 1/2 = 0.5"         # 分数自动转换 ✓
```

## ⚙️ 配置选项

### 使用答案准确率（推荐用于数学任务）
```yaml
training:
  metric_for_best_model: "answer_accuracy"
  greater_is_better: true
```

### 使用 Token 准确率
```yaml
training:
  metric_for_best_model: "token_accuracy"
  greater_is_better: true
```

### 只使用 Loss
```yaml
training:
  metric_for_best_model: "loss"
  greater_is_better: false
```

## 🚀 快速开始

```bash
# 1. 使用配置文件训练（自动启用 metrics）
python train.py --config configs/smol/135m_metamath.yaml

# 2. 查看评估结果
# 训练日志会显示:
#   eval_token_accuracy: 0.8567
#   eval_answer_accuracy: 0.7234
```

## 📈 评估输出示例

```
***** Evaluation *****
  eval_loss             = 1.234
  eval_token_accuracy   = 85.67%   ← Token级准确率
  eval_answer_accuracy  = 72.34%   ← 答案准确率（关键指标）
  eval_runtime          = 12.3s
```

## 🔍 支持的任务

| Task Name Pattern | Metrics Used |
|------------------|--------------|
| `metamath*` | token_accuracy + answer_accuracy |
| `gsm8k` | token_accuracy + answer_accuracy |
| `*math*` | token_accuracy + answer_accuracy |
| `glue_*` | GLUE标准指标 (accuracy, f1, etc.) |
| `*causal*` | token_accuracy |
| 其他 | loss only |

## 💡 最佳实践

1. **数学任务**: 使用 `answer_accuracy` 作为主要指标
2. **通用生成**: 使用 `token_accuracy` 或 `loss`
3. **分类任务**: 使用任务特定指标（如 `accuracy`）
4. **调试**: 查看训练日志中的样例预测

## 📝 关键文件

- 配置: `configs/smol/135m_metamath.yaml`
- Metrics: `utils/metrics.py`
- Trainer: `trainer/trainer_preparation.py`
- 文档: `docs/METRICS_GUIDE.md`
- 报告: `METRICS_INTEGRATION_REPORT.md`

---
✅ 完全支持 MetaMathQA 评估！
