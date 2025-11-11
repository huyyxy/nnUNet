# 欢迎来到全新的 nnU-Net！

如果你在寻找旧版，请点击[这里](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1)。

来自 V1？请阅读 [TLDR 迁移指南](documentation/tldr_migration_guide_from_v1.md)。强烈建议继续阅读其余文档 ;-)

## **2025-10-23 提示：torch 2.9.0 搭配 3D 卷积（使用 AMP 时）存在[严重性能退化](https://github.com/pytorch/pytorch/issues/166122)。使用 nnU-Net 时请保持 torch 版本在 2.8.0 或以下！**


## **2024-04-18 更新：新增残差编码器 UNet 预设！**
残差编码器 UNet 预设显著提升分割表现，适配多种 GPU 显存限制，绝对物超所值！
详情请看 :point_right: [这里](documentation/resenc_presets.md) :point_left:

也别错过我们最新的[论文](https://arxiv.org/pdf/2404.09556.pdf)，系统性评估了医学图像分割的最新进展，可能会刷新你的认知！

# 什么是 nnU-Net？
图像数据集千差万别：维度（2D、3D）、模态/输入通道（RGB、CT、MRI、显微图像等）、图像尺寸、体素间距、类别比例、目标结构特性……在不同数据集之间变化极大。传统上，遇到新问题总得手工设计与调优一套定制方案——这不仅容易出错、不具备规模化能力，还高度依赖实验者的个人水平。即便是专家也会头疼：设计选择太多、数据属性错综复杂，而且彼此耦合紧密，几乎无法可靠地手动完成管线优化！

![nnU-Net overview](documentation/assets/nnU-Net_overview.png)

**nnU-Net 是一种会自动适配数据集的语义分割方法。它会分析你提供的训练样本，并自动配置一套匹配的基于 U-Net 的分割管线。无需任何专业背景！直接训练模型，然后用于你的应用即可。**

nnU-Net 发布时已在 23 个生物医学竞赛数据集上完成评估。即便面对每个数据集的定制方案，它的全自动管线仍在公开排行榜上多次夺冠！之后也经受住了时间考验：它持续被用作基线与方法开发框架（[MICCAI 2020 的冠军队伍中有 9/10 使用了 nnU-Net](https://arxiv.org/abs/2101.00232)，MICCAI 2021 为 5/7；[我们也凭借 nnU-Net 夺得 AMOS2022 冠军](https://amos22.grand-challenge.org/final-ranking/)）！

使用 nnU-Net 时请引用[以下论文](https://www.google.com/url?q=https://www.nature.com/articles/s41592-020-01008-z&sa=D&source=docs&ust=1677235958581755&usg=AOvVaw3dWL0SrITLhCJUBiNIHCQO)：

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

## nnU-Net 能为你做什么？
如果你是想分析自己图像的**领域科学家**（生物学家、放射科医生等），nnU-Net 提供几乎一定能在你专属数据集上取得优异结果的开箱即用方案。只需把数据集转换为 nnU-Net 格式，就能轻松享受 AI 的力量，无需任何专业知识！

如果你是开发分割方法的**AI 研究者**，nnU-Net：
- 提供强大的开箱即用基线供你对比；
- 可作为方法开发框架，让你无需为每个数据集单独调管线，就能在大量数据集上测试你的改进（例如新的损失函数）；
- 是进一步针对特定数据集优化的绝佳起点，尤其适用于分割竞赛；
- 带来关于分割方法设计的新视角：也许你能挖掘出数据集属性与最佳管线配置之间更优的联系？

## nnU-Net 的适用范围是什么？
nnU-Net 为语义分割而生，可处理 2D 与 3D 图像，接受任意输入模态/通道。它能理解体素间距与各向异性，对类别严重不平衡也保持稳健。

nnU-Net 基于监督学习，因此需要为你的应用提供训练样本。所需样本数量取决于分割任务的复杂度，没有放之四海而皆准的数字！它并不比其他方案需要更多样本——甚至可能更少，因为我们大量使用数据增强。

nnU-Net 预处理和后处理阶段假设能一次处理整幅图像，因此不适合处理极端巨大的图像。作为参考：我们测试过 3D 图像从 40×40×40 到 1500×1500×1500，2D 图像从 40×40 到约 30000×30000！只要内存允许，尺寸还可更大。

## nnU-Net 如何工作？
面对新数据集，nnU-Net 会系统分析训练样本，构建“数据集指纹”。然后为该数据集创建多个 U-Net 配置：
- `2d`：2D U-Net（适用 2D 与 3D 数据集）
- `3d_fullres`：高分辨率 3D U-Net（仅 3D 数据集）
- `3d_lowres` → `3d_cascade_fullres`：3D U-Net 级联，先在低分辨率图像上运行，再由高分辨率 3D U-Net 细化预测（仅针对图像尺寸较大的 3D 数据集）

**注意：并非所有数据集都会生成全部配置。对于图像尺寸较小的数据集，会省略级联（包括 3d_lowres），因为全分辨率 U-Net 的 patch 尺寸已经覆盖输入图像的大部分区域。**

nnU-Net 依据“三步法”配置其分割管线：
- **固定参数**保持不变。在开发过程中，我们识别出一套稳健配置（特定架构和训练属性），可以一直沿用，例如损失函数、大部分数据增强策略、学习率等。
- **规则驱动参数**借助数据指纹并按照硬编码规则调整。例如，网络拓扑（池化行为与网络深度）会随 patch 尺寸改变；patch 尺寸、网络拓扑与 batch 大小会在给定 GPU 显存约束下统一优化。
- **经验参数**依赖试验。例如为特定数据集挑选最佳 U-Net 配置（2D、3D 全分辨率、3D 低分辨率、3D 级联）以及优化后处理策略。

## 如何上手？
先读这些：
- [安装指南](documentation/installation_instructions.md)
- [数据集转换](documentation/dataset_format.md)
- [使用指南](documentation/how_to_use_nnunet.md)

更多资料：
- [从稀疏标注学习（涂鸦、切片）](documentation/ignore_label.md)
- [基于区域的训练](documentation/region_based_training.md)
- [手动划分数据](documentation/manual_data_splits.md)
- [预训练与微调](documentation/pretraining_and_finetuning.md)
- [nnU-Net 中的强度归一化](documentation/explanation_normalization.md)
- [手动编辑 nnU-Net 配置](documentation/explanation_plans_files.md)
- [扩展 nnU-Net](documentation/extending_nnunet.md)
- [V2 有哪些不同？](documentation/changelog.md)

竞赛：
- [AutoPET II](documentation/competitions/AutoPETII.md)

[//]: # (- [忽略标签](documentation/ignore_label.md))

## nnU-Net 在哪些情况下表现突出，哪些情况下不适用？
nnU-Net 擅长需要从零开始训练的分割任务，例如：拥有非标准图像模态与输入通道的研究应用、生物医学领域的分割挑战、绝大多数 3D 分割问题等。到目前为止，我们还没遇到过 nnU-Net 工作原理失效的数据集！

注意：对于标准分割任务，如 ADE20k、Cityscapes 等 2D RGB 图像，微调在大规模相似图像（如 Imagenet 22k、JFT-300M）上预训练的基础模型会比 nnU-Net 表现更好，因为这些模型具备更优秀的初始化。nnU-Net 不支持基础模型，原因在于：1) 它们不适用于偏离标准设定的分割任务（见上文数据集）；2) 通常只支持 2D 架构；3) 与我们为每个数据集精细调整网络拓扑的核心原则冲突（拓扑一旦改变，预训练权重就无法迁移！）。

## 旧版 nnU-Net 去哪里了？
旧版 nnU-Net 的核心是在 2018 年参加 Medical Segmentation Decathlon 竞赛时匆忙搭建的，代码结构和质量并不理想。后来又陆续加入了许多与原始设计理念不符的功能，整体十分混乱，也让人难以维护。

nnU-Net V2 是一次彻底重写，真正的“全删重来”。因此一切都焕然一新（作者语：哈哈）。虽然分割性能[保持不变](https://docs.google.com/spreadsheets/d/13gqjIKEMPFPyMMMwA1EML57IyoBjfC3-QCTn4zRN_Mg/edit?usp=sharing)，但新增了大量酷炫功能。把它用作开发框架或手动微调以适配新数据集也容易得多。重写的另一大驱动力是 [Helmholtz Imaging](http://helmholtz-imaging.de) 的出现，促使我们扩展 nnU-Net 支持更多图像格式与领域。更多亮点请看[这里](documentation/changelog.md)。

# 致谢
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net 由 [Helmholtz Imaging](http://helmholtz-imaging.de) 的 Applied Computer Vision Lab (ACVL) 与 [德国癌症研究中心 (DKFZ)](https://www.dkfz.de/en/index.html) 的 [医学图像计算部门](https://www.dkfz.de/en/mic/index.php) 共同开发与维护。

```
nnUNet_raw/                                 # 原始数据根目录（手动准备）
└── Dataset123_Ki67/                        # 自定义数据集（编号+名称）
    ├── dataset.json                        # 描述通道、标签、文件后缀等元信息
    ├── imagesTr/                           # 训练影像（按通道分文件，例如 0000=RGB 或灰度）
    │   ├── case001_0000.png               # 训练样本 case001，第0通道影像
    │   ├── case001_0001.png               # 若有第1通道（可选）
    │   └── ...                            # 更多训练样本/通道
    ├── labelsTr/                          # 训练标注（像素级真值，背景为0）
    │   ├── case001.png                    # case001 的标签掩膜
    │   └── ...
    └── imagesTs/                          # 可选：测试集影像，nnU-Net 不会在训练时使用

nnUNet_preprocessed/                        # 预处理缓存根目录（自动生成）
└── Dataset123_Ki67/                        # 与数据集同名文件夹
    ├── dataset_fingerprint.json            # 数据指纹（尺寸、spacing 等分析结果）
    ├── plans.json                          # 针对该数据集的预设/计划（patch、网络配置等）
    ├── splits_final.json                   # 最终划分的 cross-validation 折分（5-fold 等）
    └── nnUNetResEncUNetPlans__2d/          # 某个配置的预处理数据存放处（名称含 plans 与配置）
        ├── images/                         # 预处理后的训练影像（NumPy 缓存）
        ├── labels/                         # 预处理后的训练标签
        └── properties.json                 # 每个 case 的属性（原始 spacing、shape 等）

nnUNet_results/                             # 训练结果根目录（自动生成）
└── Dataset123_Ki67/                        # 数据集对应结果文件夹
    └── nnUNetTrainer__nnUNetResEncUNetPlans__2d/   # 具体训练器+plans+配置组合
        ├── dataset.json                    # 训练时使用的数据描述快照
        ├── dataset_fingerprint.json        # 指纹副本（与预处理时一致）
        ├── plans.json                      # 训练所用 plans 副本
        ├── inference_instructions.txt      # 自动生成的推理命令说明（若运行 find_best_configuration）
        ├── inference_information.json      # 推理/后处理配置总结（同上）
        ├── postprocessing.pkl              # 自动选定的后处理策略
        ├── fold_0/                         # 第0折训练输出（共5个折，0-4）
        │   ├── checkpoint_final.pth        # 最终模型权重（默认用于验证/推理）
        │   ├── checkpoint_best.pth         # 训练中最佳验证表现权重
        │   ├── debug.json                  # 蓝图与超参的详细记录（排错用）
        │   ├── progress.png                # 损失/学习率/伪 Dice 曲线
        │   ├── network_architecture.pdf    # 若安装 hiddenlayer，则记录网络结构图
        │   └── validation/                 # 该折的验证输出
        │       ├── summary.json            # 验证指标（Dice 等）
        │       └── *.npz                   # 若训练时加 --npz，保存验证 softmax 概率
        ├── fold_1/ ... fold_4/             # 其余折结构同上
        └── maybe fold_all/                 # 若训练 all fold（单模型全量数据），结构类似
```