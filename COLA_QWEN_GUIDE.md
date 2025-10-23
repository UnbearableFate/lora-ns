# GLUE CoLA with Qwen2.5-1.5B - 使用说明

## 任务描述

**CoLA (Corpus of Linguistic Acceptability)** 是GLUE基准测试中的一个二分类任务，用于判断英语句子在语言学上是否可接受。

- **任务类型**: 二分类（可接受/不可接受）
- **训练样本**: 8,551条
- **验证样本**: 1,043条
- **评估指标**: Matthews Correlation Coefficient (MCC)

## 模型配置

### 基础模型
- **模型**: Qwen/Qwen2.5-1.5B
- **参数量**: 1.5B
- **架构**: Qwen系列，类似于Llama架构

### PEFT配置
- **方法**: LoRA
- **Rank (r)**: 8
- **Alpha**: 16
- **Dropout**: 0.1
- **目标模块**: q_proj, k_proj, v_proj, o_proj
- **可训练参数**: ~1M (仅0.067%的原始参数)

### 训练参数
- **训练轮数**: 10 epochs
- **批次大小**: 16 (每设备)
- **梯度累积**: 2步
- **有效批次**: 32
- **学习率**: 3e-4
- **优化器**: AdamW
- **混合精度**: BF16

## 快速开始

### 1. 训练模型
```bash
# 使用训练脚本
bash examples/train_cola_qwen.sh

# 或直接运行
python train.py --config configs/glue_cola_qwen.yaml
```

### 2. 监控训练
```bash
# 启动TensorBoard
tensorboard --logdir ./outputs/glue_cola_qwen/logs
```

### 3. 推理测试
```bash
python inference.py \
  --config configs/glue_cola_qwen.yaml \
  --model_path ./outputs/glue_cola_qwen \
  --input_text "The book was written by John."
```

## 示例句子

### 可接受的句子 (Label=1)
```
"The book was written by John."
"She gave him a present."
"I think that he is smart."
```

### 不可接受的句子 (Label=0)
```
"The book was written John."  # 缺少介词
"She gave he a present."       # 错误的代词格
"I think that is he smart."    # 错误的语序
```

## 预期结果

在CoLA验证集上的预期性能：
- **Matthews Correlation**: ~0.50-0.60
- **Accuracy**: ~80-85%

训练时间（单GPU A100）：
- **预计时间**: 15-20分钟
- **GPU内存**: ~8-10GB

## 配置文件说明

配置文件位置: `configs/glue_cola_qwen.yaml`

关键配置项：
```yaml
# 数据集特定配置
dataset:
  subset: "cola"              # CoLA子任务
  text_column: ["sentence"]   # 单句输入
  num_labels: 2               # 二分类

# 评估配置
training:
  metric_for_best_model: "matthews_correlation"  # CoLA使用MCC作为主要指标
  greater_is_better: true
```

## 微调建议

### 提高性能
1. **增加LoRA rank**: 将`lora_r`从8增加到16
2. **增加训练轮数**: 从10增加到15-20
3. **调整学习率**: 尝试2e-4或5e-4
4. **目标更多层**: 添加`gate_proj`, `up_proj`, `down_proj`

### 内存优化
如果遇到OOM问题：
```yaml
training:
  per_device_train_batch_size: 8   # 减小批次
  gradient_accumulation_steps: 4    # 增加累积
  gradient_checkpointing: true      # 启用检查点
```

### 使用量化
对于低显存GPU（<8GB）：
```yaml
training:
  optim: "paged_adamw_8bit"  # 8-bit优化器
```

## 评估指标

CoLA任务使用Matthews相关系数(MCC)作为主要评估指标。

我们的训练框架会计算以下指标：
- **matthews_correlation**: Matthews相关系数（CoLA主要指标）
- **accuracy**: 准确率
- **f1**: F1分数
- **precision**: 精确率
- **recall**: 召回率

MCC的优点：
- 平衡考虑所有混淆矩阵元素
- 适用于类别不平衡的数据
- 取值范围: -1到+1（0表示随机预测）

## 多GPU训练

使用Accelerate进行多GPU训练：

```bash
# 配置Accelerate（首次运行）
accelerate config

# 启动多GPU训练
accelerate launch train.py --config configs/glue_cola_qwen.yaml
```

## 故障排除

### 1. 模型下载问题
如果无法下载Qwen模型：
```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后使用本地路径
# 修改配置文件中的 model.name_or_path
```

### 2. 内存不足
- 减小批次大小: `per_device_train_batch_size: 8`
- 启用梯度检查点: `gradient_checkpointing: true`
- 使用8-bit量化: `optim: "paged_adamw_8bit"`

### 3. 训练不收敛
- 降低学习率: `learning_rate: 2e-4`
- 增加warmup: `warmup_ratio: 0.2`
- 检查数据预处理是否正确

## 参考资料

- **CoLA论文**: https://nyu-mll.github.io/CoLA/
- **GLUE基准**: https://gluebenchmark.com/
- **Qwen模型**: https://huggingface.co/Qwen
- **LoRA论文**: https://arxiv.org/abs/2106.09685

祝训练顺利！🚀
