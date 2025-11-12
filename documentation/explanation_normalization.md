# nnU-Net 中的强度归一化

在 nnU-Net 中，强度归一化的方式可以通过 dataset.json 里的 `channel_names`（旧版称为 `modalities`）字段来控制。与旧版 nnU-Net 相同，我们支持按通道的 z-score 归一化，以及基于前景强度的全数据集 z-score 归一化。此外，我们还新增了一些选项。

提醒：`channel_names` 字段通常如下所示：

    "channel_names": {
        "0": "T2",
        "1": "ADC"
    },

该字段的条目数与数据集的输入通道数相同。

偷偷告诉你：nnU-Net 实际上并不关心通道名称本身，我们只是用它来决定这个数据集采用哪种归一化策略。nnU-Net 要求你为每个输入通道指定一个归一化策略！如果你填写的通道名称不在下述列表中，将自动采用默认策略（`zscore`）。

目前可用的归一化策略如下：

- `CT`：执行 CT 归一化。具体流程是：从所有训练病例的前景类别（除背景和忽略标签外）收集强度值，计算其均值、标准差以及 0.5 和 99.5 分位数。先将强度裁剪到这两个分位数之间，再减去均值并除以标准差。对该输入通道而言，每个训练病例都会使用相同的归一化参数。nnU-Net 会将这些参数存储在对应计划文件的 `foreground_intensity_properties_per_channel` 字段中。此归一化适用于表示物理量的模态，如 CT 图像和 ADC 图像。
- `noNorm`：不执行任何归一化。
- `rescale_to_0_1`：将强度重缩放到 [0, 1]。
- `rgb_to_0_1`：假设输入为 uint8，通过除以 255 将其缩放至 [0, 1]。
- `zscore`/其他任意名称：对每个训练病例分别执行 z-score 归一化（减去均值并除以标准差）。

**重要提示：** nnU-Net 默认对 CT 图像使用 `CT` 归一化，对其他模态使用 `zscore`！如果你想调整默认策略，请务必验证这样做是否真的带来提升！

# 如何实现自定义归一化策略？
- 打开 `nnunetv2/preprocessing/normalization` 目录。
- 通过继承 `ImageNormalization` 来实现一个新的图像归一化类。
- 在 `nnunetv2/preprocessing/normalization/map_channel_name_to_normalization.py` 的 `channel_name_to_normalization_mapping` 中注册它，也就是在这里指定该归一化策略对应的通道名称。
- 在 dataset.json 中填写对应的通道名称以启用它。

归一化目前只能一次作用于单个通道，暂不支持同时处理多个通道的归一化方案！