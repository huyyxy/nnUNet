# 来自 nnU-Net V1 的快速迁移指南

- nnU-Net V2 可以与 V1 同时安装，彼此不会互相影响。
- V2 所需的环境变量名称略有不同，请阅读[此文档](setting_up_paths.md)。
- nnU-Net V2 的数据集命名为 DatasetXXX_NAME，而不是 Task。
- 数据集结构保持不变（imagesTr、labelsTr、dataset.json），但现在我们支持更多[文件格式](dataset_format.md#supported-file-formats)。dataset.json 也已简化，请使用 `nnunetv2.dataset_conversion.generate_dataset_json.py` 中的 `generate_dataset_json` 脚本。
- 注意：标签的声明方式不再是 value:name，而是 name:value，这与[层级标签](region_based_training.md)有关。
- nnU-Net V2 的命令以 `nnUNetv2...` 开头，大部分与之前相同（但并非全部），可使用 `-h` 查看帮助。
- 你可以使用 `nnUNetv2_convert_old_nnUNet_dataset` 将 V1 的原始数据集迁移到 V2，但训练好的模型无法迁移，推理仍需在旧版 nnU-Net 中完成。
- 下面是你最可能按顺序使用的命令：
  - `nnUNetv2_plan_and_preprocess`，示例：`nnUNetv2_plan_and_preprocess -d 2`
  - `nnUNetv2_train`，示例：`nnUNetv2_train 2 3d_fullres 0`
  - `nnUNetv2_find_best_configuration`，示例：`nnUNetv2_find_best_configuration 2 -c 2d 3d_fullres`。该命令会在 `nnUNet_preprocessed/DatasetXXX_NAME/` 文件夹中生成 `inference_instructions.txt`，告诉你如何执行推理。
  - `nnUNetv2_predict`，示例：`nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -c 3d_fullres -d 2`
  - `nnUNetv2_apply_postprocessing`（详见 inference_instructions.txt）
