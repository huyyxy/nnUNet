# 推理数据格式
请先阅读整体的[数据格式](dataset_format.md)文档！

推理所需的数据格式必须与原始数据保持一致（**尤其是图像必须与 imagesTr 文件夹中的格式完全相同**）。与之前一样，文件名需要以唯一标识符开头，后接 4 位的模态编号。以下是两个不同数据集的示例：

1) Task005_Prostate：

    该任务包含 2 个模态，因此输入文件夹中的文件应如下所示：

        input_folder
        ├── prostate_03_0000.nii.gz
        ├── prostate_03_0001.nii.gz
        ├── prostate_05_0000.nii.gz
        ├── prostate_05_0001.nii.gz
        ├── prostate_08_0000.nii.gz
        ├── prostate_08_0001.nii.gz
        ├── ...

    _0000 必须对应 T2 图像，_0001 必须对应 ADC 图像（如 dataset.json 中的 `channel_names` 所指定），这与训练时完全一致。

2) Task002_Heart：

        imagesTs
        ├── la_001_0000.nii.gz
        ├── la_002_0000.nii.gz
        ├── la_006_0000.nii.gz
        ├── ...
    
    Task002 只有一个模态，因此每个病例仅包含一个 _0000.nii.gz 文件。
  

输出文件夹中的分割结果命名为 {CASE_IDENTIFIER}.nii.gz（不再包含模态编号）。

用于推理的文件格式（本例中为 .nii.gz）必须与训练时保持一致（并且与 dataset.json 中的 `file_ending` 设置一致）！
   