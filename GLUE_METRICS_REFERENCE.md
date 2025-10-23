# GLUE任务指标参考

## 常见GLUE任务的最佳评估指标

| 任务代码 | 任务名称 | 任务类型 | 推荐指标 | 配置示例 |
|---------|---------|---------|---------|---------|
| **CoLA** | Corpus of Linguistic Acceptability | 二分类 | `matthews_correlation` | `metric_for_best_model: "matthews_correlation"` |
| **SST-2** | Stanford Sentiment Treebank | 二分类 | `accuracy` | `metric_for_best_model: "accuracy"` |
| **MRPC** | Microsoft Research Paraphrase Corpus | 二分类 | `f1` 或 `accuracy` | `metric_for_best_model: "f1"` |
| **QQP** | Quora Question Pairs | 二分类 | `f1` 或 `accuracy` | `metric_for_best_model: "f1"` |
| **STS-B** | Semantic Textual Similarity | 回归 | `spearman_correlation` | `metric_for_best_model: "spearman"` |
| **MNLI** | Multi-Genre NLI | 三分类 | `accuracy` | `metric_for_best_model: "accuracy"` |
| **QNLI** | Question NLI | 二分类 | `accuracy` | `metric_for_best_model: "accuracy"` |
| **RTE** | Recognizing Textual Entailment | 二分类 | `accuracy` | `metric_for_best_model: "accuracy"` |
| **WNLI** | Winograd NLI | 二分类 | `accuracy` | `metric_for_best_model: "accuracy"` |

## 当前支持的指标

我们的框架在分类任务中自动计算以下指标：

```python
{
    'accuracy': 准确率,
    'f1': F1分数,
    'precision': 精确率,
    'recall': 召回率,
    'matthews_correlation': Matthews相关系数
}
```

所有这些指标都可以在 `metric_for_best_model` 中使用。

## 使用示例

### CoLA任务
```yaml
training:
  metric_for_best_model: "matthews_correlation"
  greater_is_better: true
```

### MRPC任务
```yaml
training:
  metric_for_best_model: "f1"
  greater_is_better: true
```

### SST-2任务
```yaml
training:
  metric_for_best_model: "accuracy"
  greater_is_better: true
```

## 注意事项

1. **greater_is_better**: 对于所有这些指标，都应该设置为 `true`
2. **eval_strategy**: 设置为 `"epoch"` 或 `"steps"`
3. **load_best_model_at_end**: 设置为 `true` 以加载最佳模型

## 添加自定义指标

如果需要添加其他指标，可以修改 `utils/metrics.py` 中的 `compute_classification_metrics` 函数：

```python
from sklearn.metrics import your_metric

def compute_classification_metrics(eval_pred):
    predictions, labels = eval_pred
    # ... existing code ...
    
    # 添加自定义指标
    custom_metric = your_metric(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'matthews_correlation': mcc,
        'your_metric': custom_metric,  # 添加这里
    }
```

然后在配置文件中使用：
```yaml
training:
  metric_for_best_model: "your_metric"
```
