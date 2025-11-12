# 使用 nnU-Net 进行预训练

## 简介

迄今为止，nnU-Net 仅支持监督式预训练，这意味着你需要在某个预训练数据集上训练一个常规的 nnU-Net，然后将最终的网络权重作为目标数据集的初始化。

请记住，由于 nnU-Net 以自动化数据集分析和实验规划而闻名，许多训练超参数（如 patch 大小和网络拓扑）会随着数据集而改变。因此，在默认情况下，你无法直接将一个数据集的网络权重用于另一个数据集。

因此，两个任务之间的计划需要保持一致。本文档将说明如何完成这一操作，以及如何使用得到的权重进行初始化。

### 术语

在本文档中，我们使用以下术语：

- `pretraining dataset`：你希望用于预训练的数据集
- `finetuning dataset`：你感兴趣的数据集，即希望在其上进行微调的数据集

## 在预训练数据集上训练

为了获得匹配的网络拓扑，我们需要将计划从一个数据集转移到另一个数据集。因为我们最终只关心微调数据集，所以首先需要为它运行实验规划（以及预处理）：

```bash
nnUNetv2_plan_and_preprocess -d FINETUNING_DATASET
```

然后，我们需要提取预训练数据集的数据集指纹（如果尚未生成）：

```bash
nnUNetv2_extract_fingerprint -d PRETRAINING_DATASET
```

现在，我们可以将微调数据集的计划转移到预训练数据集上：

```bash
nnUNetv2_move_plans_between_datasets -s FINETUNING_DATASET -t PRETRAINING_DATASET -sp FINETUNING_PLANS_IDENTIFIER -tp PRETRAINING_PLANS_IDENTIFIER
```

通常情况下，如果你没有在 `nnUNetv2_plan_and_preprocess` 中更改实验规划器，`FINETUNING_PLANS_IDENTIFIER` 很可能是 nnUNetPlans。对于 `PRETRAINING_PLANS_IDENTIFIER`，建议使用自定义名称，以免覆盖默认计划。

请注意，数据集之间会传输所有内容。不仅包括网络拓扑、batch 大小和 patch 大小，还包括归一化方案！因此，在归一化方案不同的数据集之间进行迁移可能效果欠佳（当然也可能有效，这取决于具体方案！）。

有关 CT 归一化的说明：是的，裁剪范围、均值和标准差也会被转移！

现在，你可以对预训练数据集运行预处理：

```bash
nnUNetv2_preprocess -d PRETRAINING_DATASET -plans_name PRETRAINING_PLANS_IDENTIFIER
```

然后像往常一样运行训练：

```bash
nnUNetv2_train PRETRAINING_DATASET CONFIG all -p PRETRAINING_PLANS_IDENTIFIER
```

请注意，这里使用 `all` 折来利用所有可用数据。对于预训练，没有必要对数据进行划分。

## 使用预训练权重

完成预训练（或者你通过其他方式获得了兼容的权重）后，可以用它们来初始化你的模型：

```bash
nnUNetv2_train FINETUNING_DATASET CONFIG FOLD -pretrained_weights PATH_TO_CHECKPOINT
```

在 `PATH_TO_CHECKPOINT` 中指定 checkpoint。

加载预训练权重时，除分割层之外的所有层都会被使用！

目前 nnU-Net 尚未提供专门用于微调的训练器，因此建议继续使用 nnUNetTrainer。不过，你也可以轻松编写自己的训练器，以实现学习率预热、分割头微调或缩短训练时间等功能。