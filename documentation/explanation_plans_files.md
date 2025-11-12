# 修改 nnU-Net 配置

nnU-Net 为我们评估过的几乎所有数据集都提供了前所未有的开箱即用分割性能。但仍然有提升空间。一个稳妥的办法是先使用默认的 nnU-Net，然后针对手头的具体数据集进行手动微调。**本指南介绍如何通过 plans 文件调整 nnU-Net 配置，不涉及 nnU-Net 的代码扩展。如果需要扩展代码，请参阅[这里](extending_nnunet.md)。**

在 nnU-Net V2 中，plans 文件比 v1 时强大太多了。你可以调整的选项更多，无需依赖 hack 式方案，甚至无需直接修改 nnU-Net 代码！额外的好处是：plans 文件现在是 `.json` 格式，用户无需再与 pickle 打交道，直接用你喜欢的文本编辑器打开即可！

如果感到无从下手，可以先看看我们的[示例](#examples)！

# `plans.json` 结构

plans 包含全局设置和局部设置。全局设置会应用于该 plans 文件中的所有配置，而局部设置附着在特定的配置上。

## 全局设置

- `foreground_intensity_properties_by_modality`：前景区域（除背景和忽略标签外的所有标签）的强度统计信息，基于所有训练病例计算。用于 [CT 归一化方案](explanation_normalization.md)。
- `image_reader_writer`：本数据集应该使用的图像读写类名称。如果想在推理时使用不同格式的文件，可以修改这里。指定的类必须位于 `nnunetv2.imageio`。
- `label_manager`：负责标签处理的类名。可查看 `nnunetv2.utilities.label_handling.LabelManager` 了解其作用。如果要替换它，请将自定义版本放在 `nnunetv2.utilities.label_handling`。
- `transpose_forward`：nnU-Net 会转置输入数据，使得分辨率最高（间距最小）的轴排在最后。这是因为 2D U-Net 在尾部维度上运行（数组内部内存布局使切片更高效）。未来可能会把该设置改成只影响单个配置。
- `transpose_backward`：`numpy.transpose` 用于还原 `transpose_forward` 的轴顺序。
- [`original_median_shape_after_transp`]：仅供参考。
- [`original_median_spacing_after_transp`]：仅供参考。
- [`plans_name`]：请勿修改。供内部使用。
- [`experiment_planner_used`]：仅作为元数据，用来记录最初生成该文件的 planner。
- [`dataset_name`]：请勿修改。该 plans 文件适用的数据集。

## 局部设置

plans 还包含 `configurations` 键，实际的配置就存放在这里。`configurations` 本身是一个字典，键为配置名称，值为对应的局部设置。

为了更好地理解 plans 文件中描述网络拓扑的各个组件，请阅读我们论文的[补充材料](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf)第 6.2 节（第 13 页）！

局部设置包括：
- `spacing`：该配置使用的目标体素间距。
- `patch_size`：训练该配置时使用的 patch 大小。
- `data_identifier`：该配置的预处理数据将保存到 `nnUNet_preprocessed/DATASET_NAME/_data_identifier_`。若添加新配置，请务必设置唯一的 `data_identifier`，以免与其他配置冲突（除非计划复用其他配置的数据，例如级联中的做法）。
- `batch_size`：训练使用的批大小。
- `batch_dice`：是否使用批级 Dice（将批中的所有样本视作一张图像，整体计算 Dice 损失），或不使用（每个样本单独计算 Dice 损失，再对样本求平均）。
- `preprocessor_name`：运行预处理时使用的预处理类名称。类必须位于 `nnunetv2.preprocessing.preprocessors`。
- `use_mask_for_norm`：归一化时是否使用非零掩码（对 BraTS 等数据集相关，其他数据集多为 `False`）。与 `ImageNormalization` 类交互。
- `normalization_schemes`：通道标识到 `ImageNormalization` 类名的映射。`ImageNormalization` 类必须位于 `nnunetv2.preprocessing.normalization`。详见[这里](explanation_normalization.md)。
- `resampling_fn_data`：用于重新采样图像数据的函数名称。该函数需满足 `callable(data, current_spacing, new_spacing, **kwargs)`，并位于 `nnunetv2.preprocessing.resampling`。
- `resampling_fn_data_kwargs`：传递给 `resampling_fn_data` 的关键字参数。
- `resampling_fn_probabilities`：用于重新采样预测类别概率/对数值的函数名称。需满足 `callable(data: Union[np.ndarray, torch.Tensor], current_spacing, new_spacing, **kwargs)`，并位于 `nnunetv2.preprocessing.resampling`。
- `resampling_fn_probabilities_kwargs`：传递给 `resampling_fn_probabilities` 的关键字参数。
- `resampling_fn_seg`：用于重新采样分割图（取值为 0、1、2、3 等整数）的函数名称。需满足 `callable(data, current_spacing, new_spacing, **kwargs)`，并位于 `nnunetv2.preprocessing.resampling`。
- `resampling_fn_seg_kwargs`：传递给 `resampling_fn_seg` 的关键字参数。
- `network_arch_class_name`：UNet 类名，可用于集成自定义动态架构。
- `UNet_base_num_features`：UNet 架构的起始特征通道数，默认 32。默认情况下，每次下采样特征翻倍。
- `unet_max_num_features`：特征通道数的上限（默认 3D 为 320，2D 为 512），用于防止参数量过大。
- `conv_kernel_sizes`：nnU-Net 在编码器各阶段使用的卷积核大小。解码器与编码器对称，因此未在此显式列出。该列表长度与 `n_conv_per_stage_encoder` 一致。
- `n_conv_per_stage_encoder`：编码器每个阶段（即每个特征图分辨率）使用的卷积层数量。默认值 2，列表长度等于编码器阶段数。
- `n_conv_per_stage_decoder`：解码器每个阶段使用的卷积层数量。参见 `n_conv_per_stage_encoder`。
- `num_pool_per_axis`：网络在每个空间轴上的池化次数。用于推理时确定如何填充图像尺寸（如 `num_pool = 5` 表示输入尺寸必须能被 `2**5 = 32` 整除）。
- `pool_op_kernel_sizes`：编码器各阶段使用的池化核大小（同时也是步长）。
- [`median_image_size_in_voxels`]：训练集在当前目标间距下的图像体素数中位值。仅供参考，请勿修改。

特殊的局部设置：
- `inherits_from`：配置之间可以继承，方便只修改少量局部设置时新增配置。若使用继承，记得设置新的 `data_identifier`（如有需要）。
- `previous_stage`：如果该配置属于级联，需要在此指定前一阶段（例如低分辨率配置）的名称。
- `next_stage`：如果该配置属于级联，需要指定可能的后续阶段。因为在验证时必须以正确的体素间距导出预测结果。`next_stage` 可以是字符串或字符串列表。

# 示例

## 为大规模数据集增大批大小

如果数据集很大，训练会从更大的 `batch_size` 中获益。只需在 `configurations` 字典中创建一个新配置：

    "configurations": {
      "3d_fullres_bs40": {
        "inherits_from": "3d_fullres",
        "batch_size": 40
      }
    }

无需修改 `data_identifier`。`3d_fullres_bs40` 会直接使用 `3d_fullres` 的预处理数据。
也无需重新运行 `nnUNetv2_preprocess`，因为我们可以复用 `3d_fullres` 的已有数据（如果可用）。

## 使用自定义预处理器

想使用不同的预处理类时，可以这样指定：

    "configurations": {
      "3d_fullres_my_preprocesor": {
        "inherits_from": "3d_fullres",
        "preprocessor_name": MY_PREPROCESSOR,
        "data_identifier": "3d_fullres_my_preprocesor"
      }
    }

需要为这个新配置重新运行预处理：
`nnUNetv2_preprocess -d DATASET_ID -c 3d_fullres_my_preprocesor`，因为预处理发生了变化。每次修改预处理数据，都要记得设置唯一的 `data_identifier`！

## 修改目标间距

    "configurations": {
      "3d_fullres_my_spacing": {
        "inherits_from": "3d_fullres",
        "spacing": [X, Y, Z],
        "data_identifier": "3d_fullres_my_spacing"
      }
    }

需要为这个新配置重新运行预处理：
`nnUNetv2_preprocess -d DATASET_ID -c 3d_fullres_my_spacing`，因为预处理发生了变化。每次修改预处理数据，都要记得设置唯一的 `data_identifier`！

## 为原本没有级联的数据集添加级联

海马体（Hippocampus）数据集很小，没有级联。虽然在这里添加级联不一定合理，但为了演示，我们可以这么做。
我们需要调整以下内容：

- `spacing`：低分辨率阶段应使用更低的分辨率。
- `median_image_size_in_voxels`：更新该条目，以反映原始图像尺寸的参考值。
- `patch_size`：设定一个基于 `median_image_size_in_voxels` 的 patch 大小。
- 记得 patch 大小在每个轴上都必须能被 `2**num_pool` 整除。
- 相应调整卷积核大小、池化操作等网络参数。
- 指定下一阶段的名称。
- 添加高分辨率阶段。

示例配置如下（以 `3d_fullres` 为参考）：

    "configurations": {
      "3d_lowres": {
        "inherits_from": "3d_fullres",
        "data_identifier": "3d_lowres"
        "spacing": [2.0, 2.0, 2.0], # 由 3d_fullres 的 [1.0, 1.0, 1.0] 调整而来
        "median_image_size_in_voxels": [18, 25, 18], # 由 [36, 50, 35] 调整而来
        "patch_size": [20, 28, 20], # 由 [40, 56, 40] 调整而来
        "n_conv_per_stage_encoder": [2, 2, 2], # 比 3d_fullres 少一个条目（原为 [2, 2, 2, 2]）
        "n_conv_per_stage_decoder": [2, 2], # 比 3d_fullres 少一个条目
        "num_pool_per_axis": [2, 2, 2], # 每个维度比 3d_fullres 少一次池化（原为 [3, 3, 3]）
        "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2], [2, 2, 2]], # 少一个 [2, 2, 2]
        "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3]], # 少一个 [3, 3, 3]
        "next_stage": "3d_cascade_fullres" # 级联中的下一阶段名称
      },
      "3d_cascade_fullres": { # 不需要 data_identifier，因为可以复用 3d_fullres 的数据
        "inherits_from": "3d_fullres",
        "previous_stage": "3d_lowres" # 上一阶段名称
      }
    }

为了更好地理解 plans 文件中描述网络拓扑的组件，请阅读我们论文的[补充材料](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf)第 6.2 节（第 13 页）！
