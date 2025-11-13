## **2024-04-18 更新：全新残差编码器 UNet 预设上线！**
推荐使用的 nnU-Net 预设已经更新！查看[这里](resenc_presets.md)了解如何启用它们。

## 如何在全新数据集上运行 nnU-Net

给定任意数据集，nnU-Net 会自动配置一整套与其特性匹配的分割流水线。
nnU-Net 覆盖了完整流程：从预处理、模型配置、模型训练、后处理一直到集成。
当 nnU-Net 运行完成后，训练好的模型即可用于测试集推理。

### 数据集格式
nnU-Net 期望数据集遵循结构化格式，其灵感来自
[Medical Segmentation Decathlon](http://medicaldecathlon.com/) 的数据组织方式。请阅读
[这份说明](dataset_format.md)，了解如何让数据集满足 nnU-Net 的要求。

**自 2.0 版本起我们支持多种图像格式（.nii.gz、.png、.tif 等）！参阅 dataset_format 文档了解更多细节！**

**nnU-Net v1 的数据集可以通过 `nnUNetv2_convert_old_nnUNet_dataset INPUT_FOLDER OUTPUT_DATASET_NAME` 转换为 v2。**
请记住：v2 使用 DatasetXXX_Name（而不是 Task）作为数据集命名，其中 XXX 是 3 位数字。
务必提供旧任务的**完整路径**，而不仅仅是任务名称。nnU-Net v2 不知道 v1 的任务存放在哪里！

### 实验规划与预处理
在新数据集上，nnU-Net 会提取数据集指纹（dataset fingerprint），即一组数据特定属性，例如
图像大小、体素间距、强度信息等。随后据此设计三种 U-Net 配置，
每套流水线都会在各自的预处理数据上运行。

运行指纹提取、实验规划和预处理最简单的方式是：

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

其中 `DATASET_ID` 即数据集编号。首次运行时建议加上 `--verify_dataset_integrity`，
它能帮助检查最常见的问题！

你也可以一次处理多个数据集，例如 `-d 1 2 3 [...]`。如果你已经知道需要哪种 U-Net 配置，
可以通过 `-c 3d_fullres` 指定（别忘了调整 -np！）。更多可用选项请运行 `nnUNetv2_plan_and_preprocess -h`。

`nnUNetv2_plan_and_preprocess` 会在你的 `nnUNet_preprocessed` 目录下创建一个以数据集命名的子目录。
命令执行完成后，该目录下会包含 `dataset_fingerprint.json` 与 `nnUNetPlans.json`（如果你感兴趣可以查看），
以及对应 U-Net 配置的预处理数据子目录。

[可选]
如果你更希望分步骤执行，也可以依次使用 `nnUNetv2_extract_fingerprint`、
`nnUNetv2_plan_experiment` 与 `nnUNetv2_preprocess`。

### 模型训练
#### 概览
你可以自行选择要训练的配置（2d、3d_fullres、3d_lowres、3d_cascade_fullres）。
如果不确定哪种效果最好，直接全部训练，交给 nnU-Net 帮你选最优即可。

nnU-Net 会在训练集上对所有配置执行 5 折交叉验证。这么做有两点原因：
1) 让 nnU-Net 评估各配置性能，从而告诉你适合的方案；
2) 自然获得性能更好的模型集成（将 5 个模型的输出求平均）。

你可以影响 5 折交叉验证的划分方式（见[此处](manual_data_splits.md)）。
如果你只想在所有训练样本上训练一个模型，也可以（见下文）。

**注意：并非所有数据集都会生成所有 U-Net 配置。
对于图像尺寸很小的数据集，U-Net 级联（包括 3d_lowres）会被省略，
因为全分辨率 U-Net 的 patch 已覆盖输入的大部分区域。**

训练使用 `nnUNetv2_train` 命令，基本结构如下：
```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [更多选项参见 -h]
```

`UNET_CONFIGURATION` 是指定 U-Net 配置的字符串（默认包括 2d、3d_fullres、3d_lowres、3d_cascade_fullres）。
`DATASET_NAME_OR_ID` 用于指定数据集，`FOLD` 则表示训练 5 折中的哪一折，只能取 "all" 或 0-4 之间的整数。

nnU-Net 每 50 个 epoch 保存一次检查点。如果需要继续之前的训练，在命令中加上 `--c` 即可。

重要：如果计划使用 `nnUNetv2_find_best_configuration`（见下文），请加上 `--npz`。
它会在最终验证阶段保存 softmax 输出，这是后续步骤所需。
由于 softmax 预测文件很大，默认不开启。如果此前未加 `--npz`，
现在又需要 softmax 预测，可通过以下命令重新运行验证：
```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --val --npz
```

你可以用 `-device DEVICE` 指定使用的设备，`DEVICE` 只能是 cpu、cuda 或 mps。
若有多块 GPU，可通过 `CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...]` 选择具体 GPU（此时 `-device` 必须为 cuda）。

更多选项请运行 `nnUNetv2_train -h`。

nnUNetv2_train -h
用法: nnUNetv2_train [-h] [-tr TR] [-p P] [-pretrained_weights PRETRAINED_WEIGHTS] [-num_gpus NUM_GPUS]
                     [--npz] [--c] [--val] [--val_best] [--disable_checkpointing] [-device DEVICE]
                     dataset_name_or_id configuration fold

位置参数:
  dataset_name_or_id    用于训练的数据集名称或ID
  configuration         要训练的配置方案
  fold                  5折交叉验证的折数。应为0到4之间的整数

选项:
  -h, --help            显示此帮助信息并退出
  -tr TR                [可选] 使用此标志指定自定义训练器。默认: nnUNetTrainer
  -p P                  [可选] 使用此标志指定自定义计划标识符。默认: nnUNetPlans
  -pretrained_weights PRETRAINED_WEIGHTS
                        [可选] 用作预训练模型的nnU-Net检查点文件路径。
                        仅在实际训练时使用。测试功能，请谨慎使用
  -num_gpus NUM_GPUS    指定用于训练的GPU数量
  --npz                 [可选] 将最终验证的softmax预测另存为npz文件（除预测分割结果外）。
                        用于寻找最佳模型集成
  --c                   [可选] 从最新检查点继续训练
  --val                 [可选] 设置此标志仅运行验证。要求训练已完成
  --val_best            [可选] 若设置，将使用checkpoint_best而非checkpoint_final执行验证。
                        与--disable_checkpointing不兼容！
                        注意：这将使用与常规验证相同的'validation'文件夹，
                        且无法区分两次验证结果！
  --disable_checkpointing
                        [可选] 设置此标志以禁用检查点保存。
                        适用于测试场景，避免硬盘被检查点文件填满
  -device DEVICE        用于设置训练运行的设备。
                        可用选项为'cuda'（GPU）、'cpu'（CPU）和'mps'（Apple M1/M2）。
                        请勿使用此参数设置GPU ID！
                        应使用 CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] 格式！

### 2D U-Net
对于 FOLD 属于 [0, 1, 2, 3, 4] 的情况，运行：
```bash
nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD [--npz]
```

### 3D 全分辨率 U-Net
对于 FOLD 属于 [0, 1, 2, 3, 4] 的情况，运行：
```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD [--npz]
```

### 3D U-Net 级联
#### 3D 低分辨率 U-Net
对于 FOLD 属于 [0, 1, 2, 3, 4] 的情况，运行：
```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres FOLD [--npz]
```

#### 3D 全分辨率 U-Net
对于 FOLD 属于 [0, 1, 2, 3, 4] 的情况，运行：
```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_cascade_fullres FOLD [--npz]
```
**注意：级联中的 3D 全分辨率 U-Net 需要先完成 3D 低分辨率 U-Net 的 5 个折。**

训练好的模型会写入 `nnUNet_results` 目录。每个训练都会自动生成如下结构的输出路径：

```
nnUNet_results/DatasetXXX_MYNAME/TRAINER_CLASS_NAME__PLANS_NAME__CONFIGURATION/FOLD
```

以 MSD 的 Dataset002_Heart 为例，目录结构如下：

```
nnUNet_results/
├── Dataset002_Heart
    │── nnUNetTrainer__nnUNetPlans__2d
    │    ├── fold_0
    │    ├── fold_1
    │    ├── fold_2
    │    ├── fold_3
    │    ├── fold_4
    │    ├── dataset.json
    │    ├── dataset_fingerprint.json
    │    └── plans.json
    └── nnUNetTrainer__nnUNetPlans__3d_fullres
         ├── fold_0
         ├── fold_1
         ├── fold_2
         ├── fold_3
         ├── fold_4
         ├── dataset.json
         ├── dataset_fingerprint.json
         └── plans.json
```

注意这里没有 3d_lowres 和 3d_cascade_fullres，说明该数据集没有触发级联。
每个模型训练输出目录（即各个 `fold_x`）都会包含以下文件：
- `debug.json`：总结训练所用的蓝图参数和推断参数，外加其他信息。虽然阅读不太轻松，但调试时非常有用 ;-)
- `checkpoint_best.pth`：训练过程中找到的最佳模型检查点。除非手动指定，否则当前不会使用。
- `checkpoint_final.pth`：最终模型检查点（训练结束后）。验证和推理都会使用它。
- `network_architecture.pdf`（需安装 hiddenlayer）：网络结构可视化图。
- `progress.png`：展示训练期间的损失、伪 Dice、学习率和 epoch 时长。顶部包含训练（蓝）与验证（红）损失，
  以及 Dice 近似（绿）和其移动平均（绿虚线）。该 Dice 近似基于验证集中随机抽取的 patch，
  并将所有 patch 当作来自同一个体积（“全局 Dice”）。**它只能大致反映模型是否在正常训练**，
  真正的完整验证仅在训练结束时执行。
- `validation` 目录：包含训练结束后的验证预测。`summary.json` 提供各类别的平均指标。
  如果设置了 `--npz`，压缩的 softmax 输出（.npz）也会存放在此。

训练过程中关注进度非常有帮助。我们建议在首次训练时查看生成的 `progress.png`，
该文件会在每个 epoch 后更新。

训练时间主要取决于 GPU。推荐的最低配置是 Nvidia RTX 2080 Ti，在此条件下所有网络训练都不会超过 2 天。
参考我们的[基准测试](benchmarking.md)来检查你的系统表现是否符合预期。

### 使用多 GPU 进行训练

如果手头有多块 GPU，最佳做法是在每块 GPU 上分别运行一个 nnU-Net 训练。
这是因为对于类似 nnU-Net 这样的小型网络，数据并行并不会线性扩展。

示例：

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_NAME_OR_ID 2d 0 [--npz] & # 在 GPU 0 训练
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train DATASET_NAME_OR_ID 2d 1 [--npz] & # 在 GPU 1 训练
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train DATASET_NAME_OR_ID 2d 2 [--npz] & # 在 GPU 2 训练
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train DATASET_NAME_OR_ID 2d 3 [--npz] & # 在 GPU 3 训练
CUDA_VISIBLE_DEVICES=4 nnUNetv2_train DATASET_NAME_OR_ID 2d 4 [--npz] & # 在 GPU 4 训练
...
wait
```

**重要：首次运行训练时，nnU-Net 会将预处理数据解压为未压缩的 numpy 数组以提升速度！
在同一配置下启动多个训练前，请等待该操作完成并确认首个训练已经开始使用 GPU。
依据数据集大小和系统性能，该过程通常只需几分钟。**

如果你执意采用 DDP 多 GPU 训练，我们也提供支持：

```
nnUNetv2_train DATASET_NAME_OR_ID 2d 0 [--npz] -num_gpus X
```

仍需注意：与在每块 GPU 上独立训练相比，这种方式通常更慢。
DDP 只有在你手动调整 nnU-Net 配置、训练更大的模型（更大 patch 和/或 batch）时才更有意义。

使用 `-num_gpus` 时请注意：
1) 如果你只想使用两块 GPU，但机器上有更多 GPU，需要通过 `CUDA_VISIBLE_DEVICES=0,1`
   指定具体编号。
2) GPU 数不能超过 minibatch 中的样本数。例如 batch size 为 2，则最多只能用 2 块 GPU。
3) 确保 batch size 能被 GPU 数整除，否则无法充分利用硬件。

与旧版 nnU-Net 相比，现在的 DDP 完全零痛点。尽情享用吧！

### 自动挑选最佳配置
在完成所需配置的整套交叉验证后，可以让 nnU-Net 自动为你找出最佳组合：

```commandline
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS 
```

`CONFIGURATIONS` 是你希望评估的配置列表。默认会开启集成，
即 nnU-Net 将枚举所有两两配置组合。这样需要验证集的概率输出 `.npz` 文件
（训练时需添加 `--npz`，见上文）。如果不想集成，请添加 `--disable_ensembling`。

更多选项请运行 `nnUNetv2_find_best_configuration -h`。

`nnUNetv2_find_best_configuration` 还会自动确定应使用的后处理方案。
nnU-Net 的后处理逻辑只考虑“保留最大连通域”这一操作（针对前景 vs 背景以及每个标签/区域分别执行一次）。

命令完成后，终端会打印出推理所需的具体命令。
同时会在 `nnUNet_results/DATASET_NAME` 目录下生成两个文件供你查看：
- `inference_instructions.txt`：再次列出推理命令；
- `inference_information.json`：可查看所有配置与集成的表现、后处理效果以及调试信息。

### 运行推理
请确保输入目录中的数据后缀与训练所用数据集一致，
并遵循 nnU-Net 的文件命名约定（参见 [dataset format](dataset_format.md) 与
[inference data format](dataset_format_inference.md)）。

`nnUNetv2_find_best_configuration`（见上）会直接输出一条推理命令字符串。
最简单的方式就是直接执行这些命令。

如果想手动指定用于推理的配置，可使用以下命令：

#### 运行预测
针对每个目标配置，执行：
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
```

只有在需要进行集成时才指定 `--save_probabilities`。
该选项会将预测的概率图与分割结果一起保存，占用额外磁盘空间。

请为每个配置选择独立的 `OUTPUT_FOLDER`！

默认情况下，推理会使用 5 折交叉验证得到的全部模型进行集成。
我们强烈建议使用全部 5 个折，因此推理前务必完成所有折的训练。

若希望仅用单个模型进行预测，请训练 `all` 折，并在 `nnUNetv2_predict` 中通过 `-f all` 指定。

#### 集成多个配置
若要对不同配置的预测结果进行集成，可使用：
```bash
nnUNetv2_ensemble -i FOLDER1 FOLDER2 ... -o OUTPUT_FOLDER -np NUM_PROCESSES
```

你可以指定任意数量的文件夹，但要确保每个文件夹都包含由 `nnUNetv2_predict` 生成的 `.npz` 文件。
更多选项请运行 `nnUNetv2_ensemble -h`。

#### 应用后处理
最后，将预先确定的后处理应用于（可能已集成的）预测结果：

```commandline
nnUNetv2_apply_postprocessing -i FOLDER_WITH_PREDICTIONS -o OUTPUT_FOLDER --pp_pkl_file POSTPROCESSING_FILE -plans_json PLANS_FILE -dataset_json DATASET_JSON_FILE
```

`nnUNetv2_find_best_configuration`（或其生成的 `inference_instructions.txt`）会告诉你后处理文件的位置。
如果找不到，请到结果目录中查找 `postprocessing.pkl`。
若源结果来自集成，还需要显式指定 `-plans_json` 和 `-dataset_json`，
这些文件可以从任意一个集成成员的训练目录中获取。

## 如何使用预训练模型进行推理
参见[这里](run_inference_with_pretrained_models.md)。

## 如何部署并在其它设备上运行你自己的预训练模型
若要在另一台计算机上方便地使用你的预训练模型进行推理，请按照以下精简流程操作：
1. 导出模型：使用 `nnUNetv2_export_model_to_zip` 将训练好的模型打包为 `.zip` 文件，其中包含所有必要文件。
2. 传输模型：将 `.zip` 文件拷贝至目标计算机。
3. 导入模型：在新机器上运行 `nnUNetv2_install_pretrained_model_from_zip`，从 `.zip` 中安装模型。

请注意，两台计算机上都必须正确安装 nnU-Net 及其依赖，以确保模型能够兼容且正常运行。

[//]: # (## Examples)

[//]: # ()

[//]: # (To get you started we compiled two simple to follow examples:)

[//]: # (- run a training with the 3d full resolution U-Net on the Hippocampus dataset. See [here](documentation/training_example_Hippocampus.md).)

[//]: # (- run inference with nnU-Net's pretrained models on the Prostate dataset. See [here](documentation/inference_example_Prostate.md).)

[//]: # ()

[//]: # (Usability not good enough? Let us know!)
