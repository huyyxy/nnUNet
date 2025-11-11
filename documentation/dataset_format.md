# nnU-Net 数据集格式
将数据导入 nnU-Net 的唯一方式，就是按照特定格式存储。由于 nnU-Net 源自 [Medical Segmentation Decathlon](http://medicaldecathlon.com/)（MSD），因此其数据集格式深受 MSD 启发，但后来有所演变（另见[这里](#如何使用-decathlon-数据集)）。

一个数据集由三部分组成：原始图像、对应的分割图，以及一个描述元数据的 dataset.json 文件。

如果你正从 nnU-Net v1 迁移，请阅读[这里](#如何使用-nnu-net-v1-任务)转换已有的任务。


## 训练样本长什么样？
每个训练样本都对应一个唯一的标识符（identifier），即该样本的唯一名称。nnU-Net 通过这个标识符将图像与正确的分割结果关联起来。

一个训练样本包含图像及其对应的分割。

之所以用 **Images** 的复数，是因为 nnU-Net 支持任意数量的输入通道。为了尽可能灵活，nnU-Net 要求除了 RGB 自然图像外，每个输入通道都单独存储在一个图像文件中。这些图像可以是 T1、T2 MRI 等任何你需要的模态。不同通道必须拥有相同的几何信息（例如尺寸、像素间距等，如适用），并且需要（在适用时）完成配准。nnU-Net 通过文件名的 FILE_ENDING（四位整数）识别输入通道。图像文件必须遵循以下命名规则：{CASE_IDENTIFIER}_{XXXX}.{FILE_ENDING}。其中 XXXX 是 4 位通道/模态编号（每个模态/通道都应该唯一，例如 T1 用 “0000”、T2 MRI 用 “0001” 等），FILE_ENDING 则是图像格式的扩展名（如 .png、.nii.gz 等）。具体示例见下文。dataset.json 通过 `channel_names` 字段将通道名称与这些编号关联起来（见下文细节）。

补充说明：通常每个通道/模态都需要单独的文件，并通过通道编号 XXXX 访问。唯一例外是自然图像（RGB；.png），三个颜色通道可以共存在一个文件中（例子见 [road segmentation](../nnunetv2/dataset_conversion/Dataset120_RoadSegmentation.py) 数据集）。

**Segmentations（分割图）** 必须与对应图像拥有相同的几何信息（例如相同的尺寸）。分割图是整数映射，每个像素值代表一个语义类别。背景必须为 0。如果没有背景，也不要把 0 用于其他类别！语义类别的整数值必须连续（0、1、2、3 ...）。当然，每个训练样本不必包含所有类别。分割图的命名格式为 {CASE_IDENTIFIER}.{FILE_ENDING}。

在同一个训练样本内，所有图像的几何信息（输入通道、对应分割）必须一致；在不同训练样本之间，它们可以不同，nnU-Net 会处理这些差异。

重要提示：输入通道必须保持一致！具体来说，**所有图像都需要包含相同的输入通道，顺序一致，并且每次都必须齐全**。推理阶段同样适用！


## 支持的文件格式
nnU-Net 要求图像与分割图使用相同的文件格式！推理阶段也会使用同样的格式。因此，目前无法使用 .png 训练后再用 .jpg 推理。

nnU-Net V2 的一大变化是支持多种输入文件类型。从此再也不用把所有内容都转换成 .nii.gz！这是通过 `BaseReaderWriter` 抽象图像与分割的输入输出实现的。nnU-Net 内置了丰富的读写器组合，你也可以编写自己的 Reader+Writer 来支持自定义格式！详见[这里](../nnunetv2/imageio/readme.md)。

更棒的是，nnU-Net 现在原生支持 2D 输入图像，不再需要把 2D 数据硬凑成伪 3D nifti —— 真是太棒了。

需要注意的是，无论原始数据是什么格式，nnU-Net 在内部（用于存储和访问预处理数据）都会使用自己的文件格式，这是为了性能考虑。


默认支持的文件格式如下：

- NaturalImage2DIO：.png、.bmp、.tif
- NibabelIO：.nii.gz、.nrrd、.mha
- NibabelIOWithReorient：.nii.gz、.nrrd、.mha（该读取器会把图像重采样到 RAS 朝向）
- SimpleITKIO：.nii.gz、.nrrd、.mha
- Tiff3DIO：.tif、.tiff（3D TIF 图像！由于 TIF 没有标准化的像素间距存储方式，nnU-Net 期望每个 TIF 文件配套一个同名 .json 文件用来存储这些信息，详见[这里](#datasetjson)）

上面的扩展名列表并不穷尽，具体取决于底层库的支持情况。例如 nibabel 和 SimpleITK 支持的格式远不止列出的三个。这里给出的只是我们验证过的扩展名！

重要提示：nnU-Net 只能使用无损（或不压缩）格式！由于文件格式是在整个数据集一级定义的（而不是图像和分割分别定义，也许未来会支持），我们必须保证没有压缩伪影破坏分割结果。所以禁止使用 .jpg 等有损格式！

## 数据集文件夹结构
数据集必须位于 `nnUNet_raw` 文件夹中（你可以在安装 nnU-Net 时指定，或在运行 nnU-Net 命令前导出/设置该路径）。
每个分割数据集作为一个独立的 “Dataset” 存放。每个数据集都与一个三位整数 ID 和一个可自定义的名称关联。例如 Dataset005_Prostate 的数据集名称是 “Prostate”，ID 为 5。`nnUNet_raw` 文件夹中的结构如下：

```
nnUNet_raw/
├── Dataset001_BrainTumour
├── Dataset002_Heart
├── Dataset003_Liver
├── Dataset004_Hippocampus
├── Dataset005_Prostate
├── ...
```

在每个数据集文件夹内部，期望的结构如下：

```
Dataset001_BrainTumour/
├── dataset.json
├── imagesTr
├── imagesTs  # 可选
└── labelsTr
```

当你添加自定义数据集时，请查看 [dataset_conversion](../nnunetv2/dataset_conversion) 文件夹，并选择一个尚未使用的 ID。ID 001-010 保留给 Medical Segmentation Decathlon。

- **imagesTr**：训练样本的图像。nnU-Net 会使用这些数据进行流程配置、交叉验证训练、寻找后处理方法以及最优集成。
- **imagesTs**（可选）：测试样本的图像。nnU-Net 不会使用它们！但这可以是你存放测试图像的一个便捷位置，延续自 MSD 的目录结构。
- **labelsTr**：训练样本的真值分割图像。
- **dataset.json**：数据集的元数据。

上文介绍的命名方案（见[训练样本长什么样？](#训练样本长什么样)）会形成如下目录结构。以下以 MSD 的第一个数据集 BrainTumour 为例。该数据集有四个输入通道：FLAIR（0000）、T1w（0001）、T1gd（0002）、T2w（0003）。请注意 imagesTs 文件夹是可选的，可以不存在。

```
nnUNet_raw/Dataset001_BrainTumour/
├── dataset.json
├── imagesTr
│   ├── BRATS_001_0000.nii.gz
│   ├── BRATS_001_0001.nii.gz
│   ├── BRATS_001_0002.nii.gz
│   ├── BRATS_001_0003.nii.gz
│   ├── BRATS_002_0000.nii.gz
│   ├── BRATS_002_0001.nii.gz
│   ├── BRATS_002_0002.nii.gz
│   ├── BRATS_002_0003.nii.gz
│   ├── ...
├── imagesTs
│   ├── BRATS_485_0000.nii.gz
│   ├── BRATS_485_0001.nii.gz
│   ├── BRATS_485_0002.nii.gz
│   ├── BRATS_485_0003.nii.gz
│   ├── BRATS_486_0000.nii.gz
│   ├── BRATS_486_0001.nii.gz
│   ├── BRATS_486_0002.nii.gz
│   ├── BRATS_486_0003.nii.gz
│   ├── ...
└── labelsTr
    ├── BRATS_001.nii.gz
    ├── BRATS_002.nii.gz
    ├── ...
```

再举一个 MSD 第二个数据集（Heart）的例子，它只有一个输入通道：

```
nnUNet_raw/Dataset002_Heart/
├── dataset.json
├── imagesTr
│   ├── la_003_0000.nii.gz
│   ├── la_004_0000.nii.gz
│   ├── ...
├── imagesTs
│   ├── la_001_0000.nii.gz
│   ├── la_002_0000.nii.gz
│   ├── ...
└── labelsTr
    ├── la_003.nii.gz
    ├── la_004.nii.gz
    ├── ...
```

请牢记：每个训练样本内，所有图像必须具有相同的几何信息，以确保像素阵列对齐。同时确保所有数据都完成配准！

另见 [dataset format inference](dataset_format_inference.md)！

## dataset.json
dataset.json 保存了 nnU-Net 训练所需的元数据。相比 v1，我们大幅减少了必填字段！

下面是 MSD 中 Dataset005_Prostate 数据集的示例：

```
{ 
 "channel_names": {  # 之前叫 modalities
   "0": "T2", 
   "1": "ADC"
 }, 
 "labels": {  # 这个现在不一样了！
   "background": 0,
   "PZ": 1,
   "TZ": 2
 }, 
 "numTraining": 32, 
 "file_ending": ".nii.gz"
 "overwrite_image_reader_writer": "SimpleITKIO"  # 可选！若不提供，nnU-Net 会自动决定 ReaderWriter
 }
```

`channel_names` 决定 nnU-Net 的归一化方式。如果某个通道标记为 “CT”，则使用基于前景像素强度的全局归一化。否则会对每个通道进行 z-score 归一化。更多细节请参考我们论文的[方法部分](https://www.nature.com/articles/s41592-020-01008-z)。nnU-Net v2 新增了更多可选的归一化方案，并允许你自定义，详见[这里](explanation_normalization.md)。

相较于 nnU-Net v1 的重要变化：
- “modality” 现改为 “channel_names”，以减少对医学图像的偏向
- labels 的结构发生变化（名称 -> 整数，而非整数 -> 名称），这是为了支持[基于区域的训练](region_based_training.md)
- 新增 “file_ending”，以支持不同的输入文件类型
- “overwrite_image_reader_writer” 为可选项！可用于指定特定（自定义）的 ReaderWriter 类；若不提供，nnU-Net 会自动选择
- “regions_class_order” 仅在[基于区域的训练](region_based_training.md)中使用

我们提供了一个工具，可以自动生成 dataset.json，位置在[这里](../nnunetv2/dataset_conversion/generate_dataset_json.py)。请参考 [dataset_conversion](../nnunetv2/dataset_conversion) 里的示例了解用法，并务必阅读文档！

如上所述，对于 TIFF 文件，需要一个包含像素间距信息的 json 文件。
例如，一个在 x、y 方向像素间距为 7.6、z 方向为 80 的 3D TIFF 栈，其 json 文件如下：

```
{
    "spacing": [7.6, 7.6, 80.0]
}
```

在数据集文件夹中，该文件（以 `cell6.json` 为例）需要放在如下位置：

```
nnUNet_raw/Dataset123_Foo/
├── dataset.json
├── imagesTr
│   ├── cell6.json
│   └── cell6_0000.tif
└── labelsTr
    ├── cell6.json
    └── cell6.tif
```


## 如何使用 nnU-Net v1 任务
如果你要从旧版 nnU-Net 迁移，请使用 `nnUNetv2_convert_old_nnUNet_dataset` 转换已有数据集！

迁移 nnU-Net v1 任务的示例：
```bash
nnUNetv2_convert_old_nnUNet_dataset /media/isensee/raw_data/nnUNet_raw_data_base/nnUNet_raw_data/Task027_ACDC Dataset027_ACDC 
```
使用 `nnUNetv2_convert_old_nnUNet_dataset -h` 查看详细说明。


## 如何使用 Decathlon 数据集
参见 [convert_msd_dataset.md](convert_msd_dataset.md)。

## 如何在 nnU-Net 中使用 2D 数据
nnU-Net 现已原生支持 2D 数据（欢呼！）。请查看[支持的文件格式](#支持的文件格式)以及示例数据集的[脚本](../nnunetv2/dataset_conversion/Dataset120_RoadSegmentation.py)。


## 如何更新现有数据集
更新数据集时，最佳做法是先删除 `nnUNet_preprocessed/DatasetXXX_NAME` 中的预处理数据，以便重新开始。然后替换 `nnUNet_raw` 中的数据，并重新运行 `nnUNetv2_plan_and_preprocess`。如有需要，也可以删除旧训练的结果。

# 数据集转换脚本示例
`dataset_conversion` 文件夹（见[这里](../nnunetv2/dataset_conversion)）中提供了多个将数据集转换为 nnU-Net 格式的示例脚本。这些脚本不能直接运行（需要你打开并修改路径），但它们是学习如何转换自有数据集的绝佳示例。挑一个与你的数据集最相似的脚本作为起点即可。
示例脚本列表会持续更新。如果你发现某个公开数据集缺失，欢迎提交 PR 添加！
