# MetaMathQA Metrics 功能说明

## 📊 新增功能概述

为 MetaMathQA 和其他生成式任务添加了完整的评估指标支持。

## 🎯 新增的 Metrics 函数

### 1. `extract_math_answer(text: str) -> str`
**功能**: 从数学问题的解答中提取最终答案

**支持的格式**:
- `"#### 42"` - MetaMathQA/GSM8K 标准格式
- `"The answer is 42"` - 自然语言格式
- `"Final answer: 42"` - 明确标注格式
- 提取文本中最后一个数字（回退方案）

**示例**:
```python
from utils.metrics import extract_math_answer

text1 = "We calculate: 2 + 2 = 4 #### 4"
answer1 = extract_math_answer(text1)  # Returns: "4"

text2 = "The answer is 42"
answer2 = extract_math_answer(text2)  # Returns: "42"

text3 = "Therefore x = -5.5"
answer3 = extract_math_answer(text3)  # Returns: "-5.5"
```

### 2. `normalize_answer(answer: str) -> str`
**功能**: 标准化答案以便比较

**特性**:
- 数值标准化（去除多余的零）
- 分数转小数（如 "1/2" → "0.5"）
- 大小写不敏感

**示例**:
```python
from utils.metrics import normalize_answer

normalize_answer("42")      # "42"
normalize_answer("42.0")    # "42"
normalize_answer("42.00")   # "42"
normalize_answer("3.140")   # "3.14"
normalize_answer("1/2")     # "0.5"
```

### 3. `compute_causal_lm_metrics(eval_preds) -> Dict[str, float]`
**功能**: 计算基础因果语言模型指标

**返回指标**:
- `token_accuracy`: Token 级别的准确率

**适用场景**: 通用的语言生成任务

**示例**:
```python
# 在训练过程中自动调用
# 返回格式: {"token_accuracy": 0.85}
```

### 4. `compute_math_generation_metrics(tokenizer) -> Callable`
**功能**: 创建数学生成任务的评估函数

**返回指标**:
- `token_accuracy`: Token 级别的准确率
- `answer_accuracy`: 提取答案的精确匹配率

**特性**:
- 自动解码生成的文本
- 智能提取答案
- 标准化后比较
- 记录样例供调试

**适用场景**: MetaMathQA, GSM8K 等数学推理任务

**示例**:
```python
from transformers import AutoTokenizer
from utils.metrics import compute_math_generation_metrics

tokenizer = AutoTokenizer.from_pretrained("model_name")
compute_fn = compute_math_generation_metrics(tokenizer)

# 在训练时使用
# 返回格式: {
#     "token_accuracy": 0.85,
#     "answer_accuracy": 0.72
# }
```

### 5. `get_metrics_function(task_name: str, tokenizer=None) -> Optional[Callable]`
**功能**: 根据任务名称自动选择合适的评估函数

**支持的任务**:
- **GLUE 任务**: `glue_sst2`, `glue_cola`, `glue_mrpc`, 等
- **数学任务**: `metamath_qa`, `gsm8k`, 包含 "math" 的任务
- **代码任务**: 包含 "code" 的任务
- **通用 LM**: 包含 "causal" 或 "lm" 的任务

**示例**:
```python
from utils.metrics import get_metrics_function

# 数学任务
metrics_fn = get_metrics_function("metamath_qa", tokenizer=tokenizer)

# GLUE 任务
metrics_fn = get_metrics_function("glue_sst2")

# 未知任务
metrics_fn = get_metrics_function("unknown_task")  # Returns: None
```

## 🔧 在 Trainer 中的集成

### 自动集成
在 `trainer/trainer_preparation.py` 的 `train_causal_lm_task` 函数中，metrics 会自动根据任务名称选择：

```python
# 自动检测任务类型并选择 metrics
task_name = config.get("task_name", "")
compute_metrics = get_metrics_function(task_name, tokenizer=tokenizer)

if compute_metrics:
    logger.info(f"Using metrics for task: {task_name}")
    # 添加到 trainer
    common_trainer_params["compute_metrics"] = compute_metrics
else:
    logger.info(f"No metrics defined, using loss only")
```

### 配置文件示例

对于 MetaMathQA (`configs/smol/135m_metamath.yaml`):
```yaml
task_name: "metamath_qa"  # 自动使用 compute_math_generation_metrics
task_type: "CAUSAL_LM"

training:
  metric_for_best_model: "answer_accuracy"  # 可以选择保存最佳答案准确率的模型
  greater_is_better: true
```

## 📈 训练时的输出

### 评估日志示例
```
Evaluation metrics:
  - eval_loss: 1.234
  - eval_token_accuracy: 0.8567
  - eval_answer_accuracy: 0.7234
  - eval_runtime: 12.34s
  - eval_samples_per_second: 162.3
```

### 调试信息
在评估时会记录前3个样例供调试：
```
============================================================
Sample predictions (for debugging):

Example 1:
  Prediction: Below is an instruction...#### 42...
  Label: Below is an instruction...#### 42...
  Extracted pred answer: 42
  Extracted label answer: 42
  Match: True

Example 2:
  Prediction: ...
  ...
============================================================
```

## 🎯 使用场景

### 1. MetaMathQA 训练
```bash
python train.py --config configs/smol/135m_metamath.yaml
```

自动获得：
- ✅ Token 准确率 - 衡量整体生成质量
- ✅ Answer 准确率 - 衡量数学推理能力

### 2. GSM8K 训练
```yaml
task_name: "gsm8k"  # 同样使用数学 metrics
```

### 3. 通用 Causal LM
```yaml
task_name: "causal_lm"  # 使用基础 token accuracy
```

### 4. 只使用 Loss（不需要 metrics）
```yaml
task_name: "some_custom_task"  # 如果不匹配任何规则，只使用 loss
```

## 📊 Metrics 比较

| Metric | 计算方式 | 适用任务 | 优点 | 缺点 |
|--------|---------|---------|------|------|
| **Loss** | 交叉熵 | 所有任务 | 直接反映训练目标 | 不够直观 |
| **Token Accuracy** | Token 级别匹配 | 生成任务 | 细粒度评估 | 对整体质量不敏感 |
| **Answer Accuracy** | 提取答案后匹配 | 数学/QA | 直接评估任务目标 | 需要答案提取逻辑 |

## 🔍 高级配置

### 1. 自定义 metric_for_best_model

在配置文件中指定用哪个指标保存最佳模型：

```yaml
training:
  load_best_model_at_end: true
  metric_for_best_model: "answer_accuracy"  # 或 "token_accuracy" 或 "loss"
  greater_is_better: true  # answer_accuracy 越大越好
```

### 2. 调整评估频率

```yaml
training:
  total_eval_times: 20  # 总共评估20次
  # 或直接设置
  eval_steps: 100       # 每100步评估一次
```

### 3. 禁用某些 metrics

如果不想使用自动 metrics，可以：
- 使用不匹配的 task_name
- 或修改代码在 `get_metrics_function` 中返回 None

## 📝 注意事项

### 1. 内存使用
- 使用 `preprocess_logits_for_metrics` 减少内存
- 只保存 argmax 预测，不保存完整 logits

### 2. 答案提取准确性
- 依赖于文本格式
- 建议在训练数据中使用统一的答案格式（如 "#### answer"）
- 可以根据需要修改 `extract_math_answer` 函数

### 3. 评估速度
- Answer accuracy 需要解码，会稍慢
- 可以通过减少 `eval_batch_size` 来节省内存
- 在大规模评估时考虑使用子集

## 🛠️ 扩展方法

如果需要添加新的 metrics：

1. 在 `utils/metrics.py` 中创建新函数：
```python
def compute_my_custom_metrics(tokenizer):
    def compute_metrics(eval_preds):
        # 你的逻辑
        return {"my_metric": value}
    return compute_metrics
```

2. 在 `get_metrics_function` 中添加条件：
```python
elif "my_task" in task_name.lower():
    return compute_my_custom_metrics(tokenizer)
```

## ✅ 总结

现在项目完全支持 MetaMathQA 的评估！

**关键改进**:
- ✅ 智能答案提取
- ✅ 标准化比较
- ✅ Token 和 Answer 双重准确率
- ✅ 自动任务检测
- ✅ 调试友好的日志
- ✅ 内存优化

**开始使用**:
```bash
python train.py --config configs/smol/135m_metamath.yaml
```

训练时会自动计算和记录 `token_accuracy` 和 `answer_accuracy`！
