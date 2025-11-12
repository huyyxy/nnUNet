# 忽略标签

_忽略标签_ 可用于标记 nnU-Net 在训练时应当忽略的区域。这在仅有稀疏标注的图像（例如涂鸦标注或少量切片标注）中尤其有用。其内部机制是通过部分损失实现的，即只在被标注的像素上计算损失并忽略其余像素。可以查看我们的 [`DC_and_BCE_loss`](../nnunetv2/training/loss/compound_losses.py) 了解实现细节。
在推理阶段（包括验证与预测）中，nnU-Net 始终会输出密集分割；验证时的指标计算当然也只会在已标注像素上进行。

使用稀疏标注可以训练模型以应用于新的、未见过的图像，或在给定稀疏标签的前提下自动补全训练样本。

（更多信息请参阅我们的[论文](https://arxiv.org/abs/2403.12834)）

忽略标签的典型使用场景包括：
- 通过稀疏标注方案节省标注时间
  - 对全部或部分切片进行涂鸦式标注（Scribble Supervision）
  - 对部分切片进行密集标注
  - 对图像中选定的补丁/立方块进行密集标注
- 粗略地遮挡参考分割中的错误区域
- 因其他原因遮挡指定区域

如果你使用了 nnU-Net 的忽略标签，请在引用原始 nnU-Net 论文的基础上，另外引用以下论文：

```
Gotkowski, K., Lüth, C., Jäger, P. F., Ziegler, S., Krämer, L., Denner, S., Xiao, S., Disch, N., H., K., & Isensee, F. 
(2024). Embarrassingly Simple Scribble Supervision for 3D Medical Segmentation. ArXiv. /abs/2403.12834
```

## 使用场景

### 涂鸦监督

涂鸦是自由形状的笔画，用于粗略标注图像。正如我们在近期[论文](https://arxiv.org/abs/2403.12834)中所展示的，nnU-Net 的部分损失实现可以从部分标注数据中达到最先进的学习效果，甚至超越许多专门为涂鸦学习设计的方法。作为起点，对于每个图像切片和每个类别（包括背景），应生成内部涂鸦和边界涂鸦：

- 内部涂鸦：随机放置于类别实例内部的涂鸦
- 边界涂鸦：大致勾勒类别实例边界一小段的涂鸦

此类涂鸦标注示例见图 1，动画演示见动画 1。
根据数据的可用性及其多样性，也可以只标注所选切片的子集。

<p align="center">
    <img src="assets/scribble_example.png" width="1024px" />
    <figcaption>图 1：分割类型示例，其中 (A) 为密集分割，(B) 为涂鸦分割。</figcaption>
</figure>
</p>

<p align="center">
    <img width="512px" src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbmdndHQwMG96M3FqZWtwbHR2enUwZXhwNHVsbndzNmNpZnVlbHJ6OSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/KRJ48evmroDlIgcqcO/giphy.gif">
    <img width="512px" src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExem10Z3ZqZHQ2MWNsMjdibG1zc3M2NzNqbG9mazdudG5raTk4d3h4MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/ifVxQQfco5ro1gH6bQ/giphy.gif">
    <figcaption>动画 1：密集分割与涂鸦标注的演示。为了更清晰地展示，图中未显示背景涂鸦。</figcaption>
</p>

### 部分切片的密集标注

稀疏标注的另一种形式是对部分切片进行密集标注。这些切片可以由用户随机选择、依据切片间的类别变化挑选，或在主动学习设置中确定。仅标注 10% 切片的示例见图 2。

<p align="center">
    <img src="assets/amos2022_sparseseg10_2d.png" width="512px" />
    <img src="assets/amos2022_sparseseg10.png" width="512px" />
    <figcaption>图 2：部分切片密集标注的示例。红色区域表示被忽略的区域。</figcaption>
</figure>
</p>


## 在 nnU-Net 中的使用方式

在 nnU-Net 中使用忽略标签非常简单，只需在 `_dataset.json_` 中定义一个名为 _ignore_ 的标签。
该忽略标签的数值必须是分割中最大的整数标签。例如，若存在背景和两个前景类别，则忽略标签的数值必须为 3。忽略标签在 `_dataset.json_` 中的名称必须是 _ignore_。以 BraTS 数据集为例，`_dataset.json_` 中的 `labels` 字典应如下所示：

```python
...
"labels": {
    "background": 0,
    "edema": 1,
    "non_enhancing_and_necrosis": 2,
    "enhancing_tumor": 3,
    "ignore": 4
},
...
```

当然，忽略标签也兼容[基于区域的训练](region_based_training.md)：

```python
...
"labels": {
    "background": 0,
    "whole_tumor": (1, 2, 3),
    "tumor_core": (2, 3),
    "enhancing_tumor": 3,  # 或 (3, )
    "ignore": 4
},
"regions_class_order": (1, 2, 3),  # 不要在此处声明忽略标签！它不会被预测
...
```

随后即可像使用其他数据集那样使用该数据集。

请记住，nnU-Net 会运行交叉验证，因此也会在你的部分标注数据上进行评估。这当然是可行的！如果你希望比较不同的稀疏标注策略（例如通过仿真），建议在密集标注的图像上运行推理，并使用 `nnUNetv2_evaluate_folder` 或 `nnUNetv2_evaluate_simple` 进行评估。