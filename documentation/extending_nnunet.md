# 扩展 nnU-Net
我们希望 nnU-Net v2 的全新结构能够让修改它变得更加直观！我们无法为其中的每一个细节提供详尽的修改教程。建议你先在代码仓库中定位到你想要调整的实现位置，然后循着代码逻辑逐步深入。设置断点并调试 nnU-Net 对理解其机制很有帮助，也能让你更顺利地完成必要的改动。

在开始之前，你可能想先阅读以下内容：
- 现在通过计划文件（plans files）编辑 nnU-Net 配置非常强大，你可以调整大量与预处理、重采样、网络拓扑等相关的内容。请阅读[这篇文档](explanation_plans_files.md)！
- [图像归一化](explanation_normalization.md)和[输入/输出格式](dataset_format.md#supported-file-formats)都很容易扩展！
- 可以按照[这里](manual_data_splits.md)的说明定义手动数据划分。
- 你可以将任意配置串联成级联，详见[同一篇文档](explanation_plans_files.md)。
- 了解我们对[基于区域的训练](region_based_training.md)的支持。
- 如果你打算修改训练流程（损失函数、采样、数据增强、学习率调度器等），就需要实现你自己的 trainer 类。最佳实践是创建一个继承自 nnUNetTrainer 的类，然后实现所需的改动。前往我们的[trainer 类文件夹](../nnunetv2/training/nnUNetTrainer)寻找灵感！其中会有与你想要调整内容相似的 trainer，可以作为参考。nnUNetTrainer 的结构与 PyTorch Lightning 的 trainer 类似，这也会让事情更简单！
- 集成新的网络结构有两种方式：
  - 快速但粗糙：实现一个新的 nnUNetTrainer 类并重写其 `build_network_architecture` 函数。确保你的结构兼容深度监督（如果不兼容，请以 `nnUNetTrainerNoDeepSupervision` 为基类！），并能处理传入的各种 patch 大小！你的结构不应在末尾应用任何非线性（softmax、sigmoid 等），这些由 nnU-Net 来完成！
  - “正统”（但更困难）的方式：构建一个可动态配置的架构，例如默认使用的 `PlainConvUNet` 类。它需要具备某种 GPU 内存估计方法，用于评估给定 patch 大小和拓扑是否能满足特定的 GPU 内存目标。构建一个新的 `ExperimentPlanner` 来配置你的新类，并与其内存预算估计进行通信。运行 `nnUNetv2_plan_and_preprocess` 时，指定你的自定义 `ExperimentPlanner` 和自定义 `plans_name`。实现一个能够使用你的 `ExperimentPlanner` 生成的计划来实例化网络结构的 nnUNetTrainer。在运行 `nnUNetv2_train` 时指定你的计划和 trainer。务必先阅读并理解对应的 nnU-Net 代码，并把它作为实现模板，这总是物有所值的！
- 记住，现在多 GPU 训练、基于区域的训练、忽略标签以及级联训练已经统一整合到一个 nnUNetTrainer 类中。不再需要创建单独的类（在实现自定义 trainer 时务必记得支持这些特性！否则请抛出 `NotImplementedError`）。

[//]: # (- Read about our support for [ignore label]&#40;ignore_label.md&#41; and [region-based training]&#40;region_based_training.md&#41;)
